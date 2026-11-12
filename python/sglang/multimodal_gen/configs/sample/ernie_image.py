# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters for ErnieImage."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class ErnieImageSamplingParams(SamplingParams):
    negative_prompt: str = " "
    num_frames: int = 1
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    use_pe: bool = True

    @classmethod
    def image_request_extra_fields(cls) -> frozenset[str]:
        return frozenset({"use_pe"})


@dataclass
class ErnieImageTurboSamplingParams(ErnieImageSamplingParams):
    # The Turbo model card recipe; guidance 1.0 also skips the negative pass.
    guidance_scale: float = 1.0
    num_inference_steps: int = 8
