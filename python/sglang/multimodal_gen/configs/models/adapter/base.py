# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.base import ArchConfig, ModelConfig


@dataclass
class AdapterArchConfig(ArchConfig):
    # convert weights name from HF-format to SGLang-dit-format
    param_names_mapping: dict = field(default_factory=dict)

    # Reverse mapping for saving checkpoints: custom -> hf
    reverse_param_names_mapping: dict = field(default_factory=dict)

    hidden_size: int = 0
    num_attention_heads: int = 0
    num_channels_latents: int = 0
    exclude_lora_layers: list[str] = field(default_factory=list)
    boundary_ratio: float | None = None


@dataclass
class AdapterConfig(ModelConfig):
    arch_config: AdapterArchConfig = field(default_factory=AdapterArchConfig)

    # sglang-diffusion Adapter-specific parameters
    prefix: str = ""
