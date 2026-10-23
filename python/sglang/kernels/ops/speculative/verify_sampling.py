"""Speculative verify sampling ops, resolved once per platform.

A name is ``None`` where the platform has no kernel for it.
``tree_speculative_sampling`` / ``chain_speculative_sampling`` run the resolved
kernel, and a pure-torch reference on devices without one and on out-of-tree
platforms.
"""

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.srt.utils import is_cuda, is_musa, is_npu

if is_npu():
    try:
        from sgl_kernel_npu.sample import (
            chain_speculative_sampling_triton,
            tree_speculative_sampling_target_only,
        )
    except ImportError:
        chain_speculative_sampling_triton = None
        tree_speculative_sampling_target_only = None
else:
    from sglang.kernels.ops.speculative.reject_sampling import (
        chain_speculative_sampling_triton,
    )

    if is_cuda():
        from sglang.kernels.ops.speculative.sampling import (
            tree_speculative_sampling_target_only,
        )
    elif is_musa():
        try:
            from sgl_kernel import tree_speculative_sampling_target_only
        except Exception:
            tree_speculative_sampling_target_only = None
    else:
        tree_speculative_sampling_target_only = None


class TreeSpeculativeSamplingOp(BaseFusedOp):
    """Sampled tree verification against the target distribution only; writes
    predicts / accept_index / accept_token_num (and rejected draft_probs) in place."""

    op = "speculative.tree_speculative_sampling_target_only"

    def forward_native(self, **kwargs) -> None:
        from sglang.kernels.ops.speculative.sampling_native import (
            tree_speculative_sampling_target_only_native,
        )

        kwargs.pop("deterministic", None)
        tree_speculative_sampling_target_only_native(**kwargs)

    def _forward_resolved(self, **kwargs) -> None:
        tree_speculative_sampling_target_only(**kwargs)

    forward_cuda = forward_npu = forward_musa = _forward_resolved
    # HIP has no tree sampling kernel; without this it would take forward_cuda.
    forward_hip = forward_native


class ChainSpeculativeSamplingOp(BaseFusedOp):
    """Rejection-sampling verification of a draft chain against target and draft
    distributions; writes predicts / accept_index / accept_token_num in place."""

    op = "speculative.chain_speculative_sampling"

    def forward_native(self, *, block_verification: bool = False, **kwargs) -> None:
        from sglang.kernels.ops.speculative.sampling_native import (
            chain_speculative_sampling_native,
        )

        for unused in (
            "retrive_next_token",
            "retrive_next_sibling",
            "threshold_single",
            "threshold_acc",
            "deterministic",
        ):
            kwargs.pop(unused, None)
        chain_speculative_sampling_native(
            block_verification=block_verification, **kwargs
        )

    def forward_triton(self, **kwargs) -> None:
        from sglang.kernels.ops.speculative.reject_sampling import (
            chain_speculative_sampling_triton as triton_kernel,
        )

        triton_kernel(**kwargs)

    def forward_npu(self, **kwargs) -> None:
        chain_speculative_sampling_triton(**kwargs)

    forward_cuda = forward_musa = forward_xpu = forward_triton


_TREE_SPECULATIVE_SAMPLING = register_fused_op(
    TreeSpeculativeSamplingOp(), __name__, "_TREE_SPECULATIVE_SAMPLING"
)
_CHAIN_SPECULATIVE_SAMPLING = register_fused_op(
    ChainSpeculativeSamplingOp(), __name__, "_CHAIN_SPECULATIVE_SAMPLING"
)


def tree_speculative_sampling(**kwargs) -> None:
    """Keyword arguments as ``tree_speculative_sampling_target_only``."""
    _TREE_SPECULATIVE_SAMPLING(**kwargs)


def chain_speculative_sampling(**kwargs) -> None:
    """Keyword arguments as ``chain_speculative_sampling_triton``."""
    _CHAIN_SPECULATIVE_SAMPLING(**kwargs)


__all__ = [
    "ChainSpeculativeSamplingOp",
    "TreeSpeculativeSamplingOp",
    "chain_speculative_sampling",
    "chain_speculative_sampling_triton",
    "tree_speculative_sampling",
    "tree_speculative_sampling_target_only",
]
