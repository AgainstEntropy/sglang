# SPDX-License-Identifier: Apache-2.0
"""The duck-typed contract between VLA policy models and the generic stages.

``VLAObservationPreprocessStage`` / ``VLAPrefixEncodingStage`` /
``VLAActionDenoisingStage`` / ``VLAActionPostprocessStage`` (see
``pipelines_core/stages/vla.py``) drive any policy model implementing this
protocol. ``Pi05PolicyModel`` and ``LingbotVlaV2PolicyModel`` are the current
implementations; new diffusion-VLA models should satisfy it rather than adding
model branches to the stages.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch

from sglang.multimodal_gen.runtime.vla.observation import VLAObservationBatch
from sglang.multimodal_gen.runtime.vla.prefix_cache import PrefixContext


class VLAPolicy(Protocol):
    device: torch.device

    def build_prefix_cache_key(self, observation: VLAObservationBatch) -> str:
        """Exact-match key for the server-level prefix K/V cache."""
        ...

    def encode_prefix(
        self,
        observation: VLAObservationBatch,
        *,
        use_cuda_graph: bool = True,
    ) -> PrefixContext:
        """Run the observation prefix once; K/V is reused by every denoise step."""
        ...

    def sample_noise(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor: ...

    def sample_actions(
        self,
        observation: VLAObservationBatch,
        prefix_context: PrefixContext,
        *,
        noise: torch.Tensor | None,
        num_steps: int,
        use_cuda_graph: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Flow-matching Euler loop returning [batch, horizon, action_dim] fp32."""
        ...

    def should_run_action_denoise(
        self,
        prefix_context: PrefixContext | None,
    ) -> bool:
        """Whether this rank participates in action denoising (split groups)."""
        ...

    def action_parallel_info(
        self,
        prefix_context: PrefixContext | None,
    ) -> dict[str, Any]: ...

    def warmup_actions(self, batch_size: int = 1) -> torch.Tensor: ...
