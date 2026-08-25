# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_vla_v2 import (
    LingbotVlaV2PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.lingbot_vla_v2 import (
    LingbotVlaV2SamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.vlas import LingbotVlaV2PolicyModel
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_vla_v2_preprocess import (
    LingbotVlaV2Preprocessor,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
    VLAActionDenoisingStage,
    VLAActionPostprocessStage,
    VLAObservationPreprocessStage,
    VLAPrefixEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.vla.prefix_cache import VLAPrefixCacheManager

logger = init_logger(__name__)


class LingbotVlaV2Pipeline(ComposedPipelineBase):
    pipeline_name = "LingbotVlaV2Pipeline"
    pipeline_config_cls = LingbotVlaV2PipelineConfig
    sampling_params_cls = LingbotVlaV2SamplingParams
    _required_config_modules: list[str] = []

    def validate_disagg_role(self, role: RoleType) -> None:
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "LingbotVlaV2Pipeline v1 supports same-process execution only."
            )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, torch.nn.Module]:
        if loaded_modules is not None:
            return loaded_modules
        pipeline_config: LingbotVlaV2PipelineConfig = server_args.pipeline_config
        policy_model = LingbotVlaV2PolicyModel.from_pretrained(
            self.model_path,
            pipeline_config,
        )
        return {"policy_model": policy_model}

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        pipeline_config: LingbotVlaV2PipelineConfig = server_args.pipeline_config
        self.preprocessor = LingbotVlaV2Preprocessor(
            pipeline_config,
            self.model_path,
        )
        self.prefix_cache = VLAPrefixCacheManager(
            max_entries=pipeline_config.prefix_cache_max_entries
        )

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            VLAObservationPreprocessStage(self.preprocessor),
            "lingbot_vla_v2_preprocess",
        )
        self.add_stage(
            VLAPrefixEncodingStage(
                self.get_module("policy_model"),
                self.prefix_cache,
            ),
            "lingbot_vla_v2_prefix",
        )
        self.add_stage(
            VLAActionDenoisingStage(self.get_module("policy_model")),
            "lingbot_vla_v2_action_denoise",
        )
        self.add_stage(
            VLAActionPostprocessStage(),
            "lingbot_vla_v2_postprocess",
        )


EntryClass = LingbotVlaV2Pipeline
