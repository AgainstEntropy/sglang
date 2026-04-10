"""Custom op implementations for the dummy OOT platform.

Demonstrates the two MultiPlatformOp dispatch mechanisms:
1. register_oot_forward() — replaces forward on existing SGLang ops
2. forward_<key>() method — auto-discovered by dispatch key name
"""

import torch
import torch.nn.functional as F

from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

# ── Replacement forwards for existing SGLang ops ────────────────────


def silu_and_mul_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Custom SiluAndMul forward for the dummy platform.

    Sets ``_dummy_dispatched`` so tests can verify this path was taken.
    """
    self._dummy_dispatched = True
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


# ── Custom op provided by this plugin ───────────────────────────────


class DummyActivation(MultiPlatformOp):
    """Example custom op shipped by the dummy platform plugin.

    Uses the ``forward_dummy()`` naming convention so dispatch_forward()
    auto-selects it when the platform's dispatch key is ``"dummy"``.
    """

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward_dummy(self, x: torch.Tensor) -> torch.Tensor:
        self._dummy_dispatched = True
        return x * 2
