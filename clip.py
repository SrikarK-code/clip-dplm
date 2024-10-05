import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional
from configuration_hybrid_clip import HybridCLIPConfig
from flax.core.frozen_dict import FrozenDict
from transformers.modeling_flax_utils import FlaxPreTrainedModel
from transformers.models.clip.modeling_flax_clip import FlaxCLIPOutput

class CLIPEncoder(nn.Module):
    config: Any
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [nn.Dense(self.config.hidden_size, dtype=self.dtype) for _ in range(self.config.num_hidden_layers)]
        self.layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

    def __call__(self, x, deterministic: bool = True):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.layernorm(x)

class FlaxRNAProteinCLIPModule(nn.Module):
    config: HybridCLIPConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.rna_model = CLIPEncoder(self.config.rna_config, dtype=self.dtype)
        self.protein_model = CLIPEncoder(self.config.protein_config, dtype=self.dtype)
        self.rna_projection = nn.Dense(self.config.projection_dim, dtype=self.dtype, use_bias=False)
        self.protein_projection = nn.Dense(self.config.projection_dim, dtype=self.dtype, use_bias=False)
        self.logit_scale = self.param("logit_scale", jax.nn.initializers.constant(self.config.logit_scale_init_value), [])

    def __call__(self, rna_values, protein_values, deterministic: bool = True):
        rna_outputs = self.rna_model(rna_values, deterministic=deterministic)
        protein_outputs = self.protein_model(protein_values, deterministic=deterministic)

        rna_embeds = self.rna_projection(rna_outputs)
        protein_embeds = self.protein_projection(protein_outputs)

        rna_embeds = rna_embeds / jnp.linalg.norm(rna_embeds, axis=-1, keepdims=True)
        protein_embeds = protein_embeds / jnp.linalg.norm(protein_embeds, axis=-1, keepdims=True)

        logit_scale = jnp.exp(self.logit_scale)
        logits_per_rna_protein = jnp.matmul(rna_embeds, protein_embeds.T) * logit_scale

        return FlaxCLIPOutput(
            logits_per_rna_protein=logits_per_rna_protein,
            rna_embeds=rna_embeds,
            protein_embeds=protein_embeds,
        )

class FlaxDiffMapProteinCLIPModule(nn.Module):
    config: HybridCLIPConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.diffmap_model = CLIPEncoder(self.config.diffmap_config, dtype=self.dtype)
        self.protein_model = CLIPEncoder(self.config.protein_config, dtype=self.dtype)
        self.diffmap_projection = nn.Dense(self.config.projection_dim, dtype=self.dtype, use_bias=False)
        self.protein_projection = nn.Dense(self.config.projection_dim, dtype=self.dtype, use_bias=False)
        self.logit_scale = self.param("logit_scale", jax.nn.initializers.constant(self.config.logit_scale_init_value), [])

    def __call__(self, diffmap_values, protein_values, deterministic: bool = True):
        diffmap_outputs = self.diffmap_model(diffmap_values, deterministic=deterministic)
        protein_outputs = self.protein_model(protein_values, deterministic=deterministic)

        diffmap_embeds = self.diffmap_projection(diffmap_outputs)
        protein_embeds = self.protein_projection(protein_outputs)

        diffmap_embeds = diffmap_embeds / jnp.linalg.norm(diffmap_embeds, axis=-1, keepdims=True)
        protein_embeds = protein_embeds / jnp.linalg.norm(protein_embeds, axis=-1, keepdims=True)

        logit_scale = jnp.exp(self.logit_scale)
        logits_per_diffmap_protein = jnp.matmul(diffmap_embeds, protein_embeds.T) * logit_scale

        return FlaxCLIPOutput(
            logits_per_diffmap_protein=logits_per_diffmap_protein,
            diffmap_embeds=diffmap_embeds,
            protein_embeds=protein_embeds,
        )

class FlaxRNAProteinCLIP(FlaxPreTrainedModel):
    config_class = HybridCLIPConfig
    module_class = FlaxRNAProteinCLIPModule

    def __init__(self, config: HybridCLIPConfig, input_shape: Optional[tuple] = None, seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs):
        if input_shape is None:
            input_shape = ((1, config.rna_config.max_position_embeddings), (1, config.protein_config.max_position_embeddings))
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def __call__(self, rna_values, protein_values, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train: bool = False):
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(rna_values, dtype=self.dtype),
            jnp.array(protein_values, dtype=self.dtype),
            not train,
            rngs={"dropout": dropout_rng} if dropout_rng is not None else None,
        )

class FlaxDiffMapProteinCLIP(FlaxPreTrainedModel):
    config_class = HybridCLIPConfig
    module_class = FlaxDiffMapProteinCLIPModule

    def __init__(self, config: HybridCLIPConfig, input_shape: Optional[tuple] = None, seed: int = 0, dtype: jnp.dtype = jnp.float32, **kwargs):
        if input_shape is None:
            input_shape = ((1, config.diffmap_config.max_position_embeddings), (1, config.protein_config.max_position_embeddings))
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype)

    def __call__(self, diffmap_values, protein_values, params: dict = None, dropout_rng: jax.random.PRNGKey = None, train: bool = False):
        return self.module.apply(
            {"params": params or self.params},
            jnp.array(diffmap_values, dtype=self.dtype),
            jnp.array(protein_values, dtype=self.dtype),
            not train,
            rngs={"dropout": dropout_rng} if dropout_rng is not None else None,
        )
