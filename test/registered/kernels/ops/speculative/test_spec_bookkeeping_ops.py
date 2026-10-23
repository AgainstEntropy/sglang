import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative import cache_locs
from sglang.kernels.ops.speculative.cache_locs import (
    AssignDraftCacheLocsOp,
    AssignReqToTokenPoolOp,
    FillFrontOutCacheLocOp,
)
from sglang.kernels.ops.speculative.eagle import FillBonusTokensOp
from sglang.kernels.ops.speculative.gather_spec_extras import GatherSpecExtrasOp
from sglang.kernels.spec import KernelBackend
from sglang.srt.mem_cache import allocation
from sglang.srt.platforms.interface import PlatformCapabilities
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NATIVE = KernelBackend.TORCH


def _oot_platform(*, supports_triton: bool = False):
    return SimpleNamespace(
        is_out_of_tree=lambda: True,
        capabilities=PlatformCapabilities(supports_triton=supports_triton),
        get_speculative_cache_locs_fn=lambda: None,
    )


class TestSpecBookkeepingNative(CustomTestCase):
    """The pure-torch references that out-of-tree platforms fall back to."""

    def test_fill_bonus_tokens(self):
        accept_tokens = torch.tensor([10, 11, 12, 20, 21, 22, 30, 31, 32])
        bonus = torch.full((3,), -1, dtype=torch.int32)
        FillBonusTokensOp()(
            accept_tokens, torch.tensor([2, 3, 1]), bonus, 3, 3, backend=NATIVE
        )
        self.assertEqual(bonus.tolist(), [11, 22, 30])

    def test_assign_draft_cache_locs_with_positions(self):
        req_to_token = torch.arange(40).view(4, 10)
        out = torch.empty(4, dtype=torch.int64)
        positions = torch.empty(2, dtype=torch.int64)
        mrope = torch.empty((3, 2), dtype=torch.int64)
        AssignDraftCacheLocsOp()(
            torch.tensor([3, 1]),
            req_to_token,
            torch.tensor([5, 2]),
            out,
            1,
            2,
            positions,
            mrope,
            backend=NATIVE,
        )
        self.assertEqual(out.tolist(), [35, 36, 12, 13])
        self.assertEqual(positions.tolist(), [5, 2])
        self.assertEqual(mrope.tolist(), [[5, 2]] * 3)

    def test_assign_req_to_token_pool_uneven_spans(self):
        req_to_token = torch.zeros((3, 8), dtype=torch.int32)
        AssignReqToTokenPoolOp()(
            torch.tensor([2, 0]),
            req_to_token,
            torch.tensor([1, 4]),
            torch.tensor([4, 6]),
            torch.tensor([7, 8, 9, 5, 6]),
            2,
            backend=NATIVE,
        )
        self.assertEqual(req_to_token[2].tolist(), [0, 7, 8, 9, 0, 0, 0, 0])
        self.assertEqual(req_to_token[0].tolist(), [0, 0, 0, 0, 5, 6, 0, 0])

    def test_fill_front_out_cache_loc(self):
        # Each row's window holds 3 slots; keep the first num_front of each.
        front = torch.full((4,), -1, dtype=torch.int64)
        FillFrontOutCacheLocOp()(
            torch.tensor([2, 0, 1]),
            torch.tensor([10, 11, 12, 20, 21, 22, 30, 31, 32]),
            front,
            backend=NATIVE,
        )
        self.assertEqual(front.tolist(), [10, 11, 30, -1])

    def test_extend_cache_locs_without_platform_fn(self):
        # An out-of-tree platform without get_speculative_cache_locs_fn used
        # to fall through every branch and get None back.
        req_to_token = torch.arange(30).view(3, 10)
        flags = ("_is_cuda", "_is_hip", "_is_musa", "_is_xpu", "_is_npu", "_is_cpu")
        with contextlib.ExitStack() as stack:
            for flag in flags:
                stack.enter_context(mock.patch.object(cache_locs, flag, False))
            stack.enter_context(
                mock.patch("sglang.srt.platforms.current_platform", _oot_platform())
            )
            out = cache_locs.assign_extend_cache_locs_func(
                req_pool_indices=torch.tensor([2, 0]),
                req_to_token=req_to_token,
                start_offset=torch.tensor([3, 7]),
                end_offset=torch.tensor([5, 9]),
                batch_size=2,
                draft_token_num=2,
                device="cpu",
            )
        self.assertEqual(out.tolist(), [23, 24, 7, 8])

    def test_get_last_loc_without_triton(self):
        req_to_token = torch.arange(20).view(2, 10)
        triton_impl = mock.Mock(side_effect=AssertionError("Triton launched"))
        with (
            mock.patch.object(allocation, "current_platform", _oot_platform()),
            mock.patch.object(allocation, "get_last_loc_triton", triton_impl),
        ):
            last_loc = allocation.get_last_loc(
                req_to_token, torch.tensor([1, 0]), torch.tensor([3, 0])
            )
        self.assertEqual(last_loc.tolist(), [12, -1])


@unittest.skipUnless(torch.cuda.is_available(), "needs a Triton device")
class TestSpecBookkeepingNativeMatchesTriton(CustomTestCase):
    def _assert_same(self, op, make_args):
        outputs = []
        for backend in (KernelBackend.TRITON, NATIVE):
            args = make_args()
            result = op(*args, backend=backend)
            outputs.append((result, [a for a in args if isinstance(a, torch.Tensor)]))
        (triton_result, triton_args), (native_result, native_args) = outputs
        for a, b in zip(triton_args, native_args):
            self.assertTrue(torch.equal(a, b))
        if triton_result is not None:
            for a, b in zip(triton_result, native_result):
                self.assertTrue(a is None and b is None or torch.equal(a, b))

    def test_ops_match(self):
        g = torch.Generator().manual_seed(0)
        bs, pool_rows, pool_len = 5, 8, 64
        req_pool_indices = torch.randperm(pool_rows, generator=g)[:bs].cuda()
        seq_lens = torch.randint(1, 40, (bs,), generator=g).cuda()
        req_to_token = torch.randint(0, 10_000, (pool_rows, pool_len), generator=g)
        req_to_token = req_to_token.to(torch.int32).cuda()

        with self.subTest(op="gather_spec_extras"):
            bufs = [
                torch.randn(pool_rows, 4).cuda(),
                torch.randint(0, 9, (pool_rows, 4)).cuda(),
            ]
            bufs += [
                torch.randint(0, 9, (pool_rows,)).cuda(),
                torch.randn(pool_rows, 16).cuda(),
            ]
            self._assert_same(GatherSpecExtrasOp(), lambda: (req_pool_indices, *bufs))
        with self.subTest(op="fill_bonus_tokens"):
            accept_tokens = torch.randint(0, 9, (bs * 4,), generator=g).cuda()
            accept_lens = torch.randint(1, 5, (bs,), generator=g).cuda()
            self._assert_same(
                FillBonusTokensOp(),
                lambda: (
                    accept_tokens,
                    accept_lens,
                    torch.zeros(bs, dtype=torch.int32, device="cuda"),
                    4,
                    bs,
                ),
            )
        for topk, num_cols in ((1, 4), (2, 3)):
            with self.subTest(op="assign_draft_cache_locs", topk=topk):
                self._assert_same(
                    AssignDraftCacheLocsOp(),
                    lambda: (
                        req_pool_indices,
                        req_to_token,
                        seq_lens,
                        torch.empty(
                            bs * topk * num_cols, dtype=torch.int64, device="cuda"
                        ),
                        topk,
                        num_cols,
                        torch.empty(bs * topk, dtype=torch.int64, device="cuda"),
                        torch.empty((3, bs * topk), dtype=torch.int64, device="cuda"),
                    ),
                )
        with self.subTest(op="fill_front_out_cache_loc"):
            window = torch.randint(0, 10_000, (bs * 6,), generator=g).cuda()
            num_front = torch.randint(0, 7, (bs,), generator=g).cuda()
            self._assert_same(
                FillFrontOutCacheLocOp(),
                lambda: (
                    num_front,
                    window,
                    torch.full((bs * 6,), -1, dtype=torch.int64, device="cuda"),
                ),
            )
        with self.subTest(op="assign_req_to_token_pool"):
            ends = seq_lens + torch.randint(0, 6, (bs,), generator=g).cuda()
            out_cache_loc = torch.randint(
                0, 10_000, (int((ends - seq_lens).sum()),)
            ).cuda()
            self._assert_same(
                AssignReqToTokenPoolOp(),
                lambda: (
                    req_pool_indices,
                    req_to_token.clone(),
                    seq_lens,
                    ends,
                    out_cache_loc,
                    bs,
                ),
            )


if __name__ == "__main__":
    unittest.main()
