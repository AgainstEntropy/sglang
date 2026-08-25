# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


@dataclass
class LingbotVlaV2PipelineConfig(PipelineConfig):
    """Configuration for LingBot-VLA-V2 action policies (Qwen3-VL + MoE expert)."""

    task_type: ModelTaskType = ModelTaskType.VLA_ACTION
    should_use_guidance: bool = False
    enable_autocast: bool = False
    generator_device: str | None = None

    policy_family: str = "lingbot_vla_v2"

    # robbyant/lingbot-vla-v2-6b public checkpoint layout. The checkpoint's
    # config.json carries only {"vlm_family": "qwen3_vl"}; the architecture
    # constants live in LingbotVlaV2ArchConfig next to the model code.
    max_token_len: int = 72
    action_horizon: int = 50
    action_dim: int = 55
    state_dim: int = 55
    output_action_dim: int = 55
    n_action_steps: int = 50
    default_num_inference_steps: int = 10
    time_embedding_min_period: float = 4e-3
    time_embedding_max_period: float = 4.0

    image_keys: tuple[str, ...] = (
        "camera_top",
        "camera_wrist_left",
        "camera_wrist_right",
    )
    image_size: tuple[int, int] = (256, 256)

    enable_global_prefix_cache: bool = False
    prefix_cache_max_entries: int = 1
    prefix_cache_layout_version: str = "lingbot-vla-v2-prefix-v1"
    empty_cache_after_prefix: bool = False

    parallel_layout_version: str = "lingbot-vla-v2-monolithic-v1"
    materialize_dtype: str = "bf16"

    def supports_disaggregation(self) -> bool:
        return False

    def estimate_request_cost(self, batch) -> float:
        return float(
            self.action_horizon * self.action_dim * self.default_num_inference_steps
        )

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig()
