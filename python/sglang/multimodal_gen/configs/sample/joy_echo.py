# SPDX-License-Identifier: Apache-2.0
import dataclasses
from dataclasses import field
from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.joy_echo import (
    JOY_ECHO_DEFAULT_SIGMAS,
)
from sglang.multimodal_gen.configs.sample.ltx_2 import LTX2SamplingParams


@dataclasses.dataclass
class JoyEchoSamplingParams(LTX2SamplingParams):
    """Sampling parameters for JoyEcho DMD inference."""

    seed: int = 12345
    generator_device: str = "cuda"

    height: int = 480
    width: int = 832
    num_frames: int = 121
    fps: int = 25

    guidance_scale: float = 1.0
    num_inference_steps: int = 8

    sigmas: tuple[float, ...] = field(default_factory=lambda: JOY_ECHO_DEFAULT_SIGMAS)

    negative_prompt: str | None = None

    video_cfg_scale: float = 1.0
    audio_cfg_scale: float = 1.0

    enable_memory_bank: bool = True
    reset_memory_bank: bool = True

    def _validate_with_server_args(self, server_args: Any) -> None:
        super()._validate_with_server_args(server_args)
        self._reject_unscheduled_step_count(server_args)

    def _reject_unscheduled_step_count(self, server_args: Any) -> None:
        # JoyEchoSigmaPreparationStage runs one step per sigma interval.
        sigmas = (
            self.sigmas
            if self.sigmas is not None
            else server_args.pipeline_config.default_sigmas
        )
        scheduled_steps = max(len(sigmas) - 1, 1)
        if (
            "num_inference_steps" in self._explicit_fields
            and self.num_inference_steps != scheduled_steps
        ):
            raise ValueError(
                f"num_inference_steps={self.num_inference_steps} is not supported: "
                f"the distilled JoyEcho model runs a fixed {scheduled_steps}-step "
                f"sigma schedule. Omit num_inference_steps or set it to "
                f"{scheduled_steps}."
            )
