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
