"""Top-k / top-p probability renormalization, resolved once per platform.

A name is ``None`` where the platform has no kernel for it. ``renorm_top_k`` /
``renorm_top_p`` run the resolved kernel, and a pure-torch reference on devices
without one and on out-of-tree platforms.
"""

import torch

from sglang.kernels.fused_op import BaseFusedOp, register_fused_op
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu
from sglang.utils import LazyImport

if is_cuda():
    # Resolved on first call: CUDA builds without FlashInfer must still import
    # this module.
    top_k_renorm_prob = LazyImport("flashinfer.sampling", "top_k_renorm_probs")
    top_p_renorm_prob = LazyImport("flashinfer.sampling", "top_p_renorm_probs")
elif is_hip():
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_k_renorm_probs_triton as top_k_renorm_prob,
    )
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_p_renorm_probs_triton as top_p_renorm_prob,
    )
elif is_npu():
    try:
        from sgl_kernel_npu.sample import top_k_renorm_prob, top_p_renorm_prob
    except ImportError:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
elif is_musa():
    try:
        from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob
    except Exception:
        top_k_renorm_prob = None
        top_p_renorm_prob = None
else:
    top_k_renorm_prob = None
    top_p_renorm_prob = None


class TopKRenormProbOp(BaseFusedOp):
    """Keep each row's ``top_k`` largest probabilities and renormalize."""

    op = "sampling.top_k_renorm_prob"

    def forward_native(self, probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
        top_ks = top_ks.reshape(-1).to(device=probs.device, dtype=torch.int64)
        top_ks = top_ks.clamp(min=1, max=probs.shape[-1])
        max_top_k = int(top_ks.max().item())
        topk_probs, topk_indices = torch.topk(probs, k=max_top_k, dim=-1)
        ranks = torch.arange(max_top_k, device=probs.device)[None, :]
        topk_probs = topk_probs.masked_fill(ranks >= top_ks[:, None], 0.0)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        return torch.zeros_like(probs).scatter_(1, topk_indices, topk_probs)

    def _forward_resolved(
        self, probs: torch.Tensor, top_ks: torch.Tensor
    ) -> torch.Tensor:
        return top_k_renorm_prob(probs, top_ks)

    forward_cuda = forward_hip = forward_npu = forward_musa = _forward_resolved


class TopPRenormProbOp(BaseFusedOp):
    """Keep each row's smallest prefix reaching ``top_p`` mass and renormalize."""

    op = "sampling.top_p_renorm_prob"

    def forward_native(self, probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps = top_ps.reshape(-1, 1).to(device=probs.device, dtype=probs.dtype)
        probs_sort = probs_sort.masked_fill((probs_sum - probs_sort) > top_ps, 0.0)
        probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
        return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)

    def _forward_resolved(
        self, probs: torch.Tensor, top_ps: torch.Tensor
    ) -> torch.Tensor:
        return top_p_renorm_prob(probs, top_ps)

    forward_cuda = forward_hip = forward_npu = forward_musa = _forward_resolved


_TOP_K_RENORM_PROB = register_fused_op(
    TopKRenormProbOp(), __name__, "_TOP_K_RENORM_PROB"
)
_TOP_P_RENORM_PROB = register_fused_op(
    TopPRenormProbOp(), __name__, "_TOP_P_RENORM_PROB"
)


def renorm_top_k(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    return _TOP_K_RENORM_PROB(probs, top_ks)


def renorm_top_p(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    return _TOP_P_RENORM_PROB(probs, top_ps)


__all__ = [
    "TopKRenormProbOp",
    "TopPRenormProbOp",
    "renorm_top_k",
    "renorm_top_p",
    "top_k_renorm_prob",
    "top_p_renorm_prob",
]
