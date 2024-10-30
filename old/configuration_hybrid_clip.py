import copy
from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class HybridCLIPConfig(PretrainedConfig):
    model_type = "hybrid-clip"
    is_composition = True

    def __init__(self, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        super().__init__(**kwargs)

        if "rna_config" not in kwargs:
            raise ValueError("`rna_config` cannot be `None`.")
        if "protein_config" not in kwargs:
            raise ValueError("`protein_config` cannot be `None`.")
        if "diffmap_config" not in kwargs:
            raise ValueError("`diffmap_config` cannot be `None`.")

        rna_config = kwargs.pop("rna_config")
        protein_config = kwargs.pop("protein_config")
        diffmap_config = kwargs.pop("diffmap_config")

        from transformers import AutoConfig

        self.rna_config = AutoConfig.for_model("custom", **rna_config)
        self.protein_config = AutoConfig.for_model("custom", **protein_config)
        self.diffmap_config = AutoConfig.for_model("custom", **diffmap_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value

    @classmethod
    def from_configs(cls, rna_config: PretrainedConfig, protein_config: PretrainedConfig, diffmap_config: PretrainedConfig, **kwargs):
        return cls(
            rna_config=rna_config.to_dict(),
            protein_config=protein_config.to_dict(),
            diffmap_config=diffmap_config.to_dict(),
            **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["rna_config"] = self.rna_config.to_dict()
        output["protein_config"] = self.protein_config.to_dict()
        output["diffmap_config"] = self.diffmap_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output










import copy
from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

@dataclass
class ModelArchitectureConfig:
    type: str = "mlp"  # mlp, transformer, resnet
    num_layers: int = 2
    hidden_size: int = 512
    dropout: float = 0.1
    attention_heads: Optional[int] = 8
    intermediate_size: Optional[int] = 2048
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    initializer_range: float = 0.02
    use_cache: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 100
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    temperature: float = 0.07
    use_amp: bool = True
    
class HybridCLIPConfig(PretrainedConfig):
    model_type = "hybrid-clip"
    is_composition = True
    
    def __init__(
        self,
        projection_dim: int = 512,
        logit_scale_init_value: float = 2.6592,
        cache_size: int = 8192,
        max_position_embeddings: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        use_hard_negatives: bool = True,
        hard_negative_weight: float = 0.5,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-4,
        use_mean_pooling: bool = True,
        embedding_dim: int = 768,
        architectures: Optional[Dict[str, ModelArchitectureConfig]] = None,
        training: Optional[TrainingConfig] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if "rna_config" not in kwargs:
            raise ValueError("`rna_config` cannot be `None`.")
        if "protein_config" not in kwargs:
            raise ValueError("`protein_config` cannot be `None`.")
        if "diffmap_config" not in kwargs:
            raise ValueError("`diffmap_config` cannot be `None`.")
            
        rna_config = kwargs.pop("rna_config")
        protein_config = kwargs.pop("protein_config")
        diffmap_config = kwargs.pop("diffmap_config")
        
        from transformers import AutoConfig
        
        self.rna_config = AutoConfig.for_model("custom", **rna_config)
        self.protein_config = AutoConfig.for_model("custom", **protein_config)
        self.diffmap_config = AutoConfig.for_model("custom", **diffmap_config)
        
        # Model architecture
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.cache_size = cache_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.embedding_dim = embedding_dim
        
        # Training configurations
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_weight = hard_negative_weight
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.use_mean_pooling = use_mean_pooling
        
        # Default architectures if none provided
        self.architectures = architectures or {
            "mlp": ModelArchitectureConfig(),
            "transformer": ModelArchitectureConfig(
                type="transformer",
                num_layers=6,
                hidden_size=768
            ),
            "resnet": ModelArchitectureConfig(
                type="resnet",
                num_layers=4,
                hidden_size=512
            )
        }
        
        # Default training config if none provided
        self.training = training or TrainingConfig()
        
    @classmethod
    def from_configs(
        cls,
        rna_config: PretrainedConfig,
        protein_config: PretrainedConfig,
        diffmap_config: PretrainedConfig,
        **kwargs
    ):
        return cls(
            rna_config=rna_config.to_dict(),
            protein_config=protein_config.to_dict(),
            diffmap_config=diffmap_config.to_dict(),
            **kwargs
        )
        
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["rna_config"] = self.rna_config.to_dict()
        output["protein_config"] = self.protein_config.to_dict()
        output["diffmap_config"] = self.diffmap_config.to_dict()
        output["architectures"] = {
            k: vars(v) for k, v in self.architectures.items()
        }
        output["training"] = vars(self.training)
        output["model_type"] = self.__class__.model_type
        return output
        
    def create_experiment_config(
        self,
        experiment_type: str,
        **override_kwargs
    ) -> "HybridCLIPConfig":
        """Creates a new config for specific experiment types"""
        config = copy.deepcopy(self)
        
        if experiment_type == "embedding_sweep":
            config.projection_dim = override_kwargs.get("projection_dim", self.projection_dim)
            config.embedding_dim = override_kwargs.get("embedding_dim", self.embedding_dim)
            
        elif experiment_type == "architecture_search":
            arch_type = override_kwargs.get("architecture_type", "mlp")
            config.architectures[arch_type] = ModelArchitectureConfig(
                **override_kwargs.get("architecture_config", {})
            )
            
        elif experiment_type == "training_sweep":
            config.training = TrainingConfig(
                **{**vars(self.training), **override_kwargs}
            )
            
        return config
