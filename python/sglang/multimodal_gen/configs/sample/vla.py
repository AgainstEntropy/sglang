# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams


@dataclass
class VLAObservationSamplingParams(ActionSamplingParams):
    """Shared sampling parameters for VLA policies driven by robot observations.

    Carries the observation payload (camera images, proprioceptive state,
    optional pre-sampled noise) into ``Req.extra["vla"]`` for the generic VLA
    pipeline stages. Model families subclass this with their own dimension
    defaults (see ``Pi05SamplingParams``, ``LingbotVlaV2SamplingParams``).
    """

    action_horizon: int = 50
    action_dim: int = 32
    output_format: str = "list"
    return_timing: bool = True
    enable_prefix_cache: bool = True
    enable_cuda_graph: bool = True

    state: Any = field(default=None, metadata={"batch_sig_exclude": True})
    images: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    image_masks: dict[str, bool] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    camera_order: list[str] | tuple[str, ...] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )
    noise: Any = field(default=None, metadata={"batch_sig_exclude": True})
    observation: dict[str, Any] | None = field(
        default=None, metadata={"batch_sig_exclude": True}
    )

    def build_request_extra(self) -> dict[str, Any]:
        extra = super().build_request_extra()
        observation = dict(self.observation or {})
        if self.images is not None:
            observation["images"] = self.images
        if self.image_masks is not None:
            observation["image_masks"] = self.image_masks
        if self.state is not None:
            observation["state"] = self.state
        if self.camera_order is not None:
            observation["camera_order"] = tuple(self.camera_order)
        if self.prompt is not None:
            observation["prompt"] = self.prompt
        if self.noise is not None:
            observation["noise"] = self.noise

        extra["vla"] = {
            "observation": observation,
            "options": {
                "output_format": self.output_format,
                "return_timing": self.return_timing,
                "enable_prefix_cache": self.enable_prefix_cache,
                "enable_cuda_graph": self.enable_cuda_graph,
            },
        }
        return extra

    def _validate(self):
        super()._validate()
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.output_format not in ("list", "numpy"):
            raise ValueError("output_format must be 'list' or 'numpy'")
