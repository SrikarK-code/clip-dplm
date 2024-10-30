import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from configuration_hybrid_clip import HybridCLIPConfig
from transformers import PreTrainedModel

class CLIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)])
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.layernorm(x)

## could experiment with more complex projection head
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.projection(x)

class RNAProteinCLIPModule(nn.Module):
    def __init__(self, config: HybridCLIPConfig):
        super().__init__()
        self.config = config
        self.rna_model = CLIPEncoder(config.rna_config)
        self.protein_model = CLIPEncoder(config.protein_config)
        self.rna_projection = ProjectionHead(
            input_dim=config.rna_config.hidden_size,
            output_dim=config.projection_dim,
            hidden_dim=config.projection_dim * 2
        )
        self.protein_projection = ProjectionHead(
            input_dim=config.protein_config.hidden_size,
            output_dim=config.projection_dim,
            hidden_dim=config.projection_dim * 2
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init_value)

    def forward(self, rna_values, protein_values):
        rna_outputs = self.rna_model(rna_values)
        protein_outputs = self.protein_model(protein_values)

        rna_embeds = self.rna_projection(rna_outputs)
        protein_embeds = self.protein_projection(protein_outputs)

        rna_embeds = F.normalize(rna_embeds, dim=-1)
        protein_embeds = F.normalize(protein_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_rna_protein = torch.matmul(rna_embeds, protein_embeds.t()) * logit_scale

        return {
            "logits_per_rna_protein": logits_per_rna_protein,
            "rna_embeds": rna_embeds,
            "protein_embeds": protein_embeds,
        }

class DiffMapProteinCLIPModule(nn.Module):
    def __init__(self, config: HybridCLIPConfig):
        super().__init__()
        self.config = config
        self.diffmap_model = CLIPEncoder(config.diffmap_config)
        self.protein_model = CLIPEncoder(config.protein_config)
        self.diffmap_projection = ProjectionHead(
            input_dim=config.diffmap_config.hidden_size,
            output_dim=config.projection_dim,
            hidden_dim=config.projection_dim * 2
        )
        self.protein_projection = ProjectionHead(
            input_dim=config.protein_config.hidden_size,
            output_dim=config.projection_dim,
            hidden_dim=config.projection_dim * 2
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * config.logit_scale_init_value)

    def forward(self, diffmap_values, protein_values):
        diffmap_outputs = self.diffmap_model(diffmap_values)
        protein_outputs = self.protein_model(protein_values)

        diffmap_embeds = self.diffmap_projection(diffmap_outputs)
        protein_embeds = self.protein_projection(protein_outputs)

        diffmap_embeds = F.normalize(diffmap_embeds, dim=-1)
        protein_embeds = F.normalize(protein_embeds, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_diffmap_protein = torch.matmul(diffmap_embeds, protein_embeds.t()) * logit_scale

        return {
            "logits_per_diffmap_protein": logits_per_diffmap_protein,
            "diffmap_embeds": diffmap_embeds,
            "protein_embeds": protein_embeds,
        }

class RNAProteinCLIP(PreTrainedModel):
    config_class = HybridCLIPConfig
    base_model_prefix = "rna_protein_clip"

    def __init__(self, config: HybridCLIPConfig):
        super().__init__(config)
        self.config = config
        self.rna_protein_clip = RNAProteinCLIPModule(config)

    def forward(self, rna_values, protein_values):
        return self.rna_protein_clip(rna_values, protein_values)

class DiffMapProteinCLIP(PreTrainedModel):
    config_class = HybridCLIPConfig
    base_model_prefix = "diffmap_protein_clip"

    def __init__(self, config: HybridCLIPConfig):
        super().__init__(config)
        self.config = config
        self.diffmap_protein_clip = DiffMapProteinCLIPModule(config)

    def forward(self, diffmap_values, protein_values):
        return self.diffmap_protein_clip(diffmap_values, protein_values)
