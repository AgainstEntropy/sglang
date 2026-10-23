import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.sampling.murmur_hash import _murmur_hash32_torch, murmur_hash32
from sglang.kernels.ops.sampling.renorm import TopKRenormProbOp, TopPRenormProbOp
from sglang.kernels.ops.speculative.row_argmax import row_argmax
from sglang.kernels.ops.speculative.spec_tree import BuildEagleTreeOp
from sglang.kernels.ops.speculative.verify_sampling import (
    ChainSpeculativeSamplingOp,
    TreeSpeculativeSamplingOp,
)
from sglang.kernels.spec import KernelBackend
from sglang.srt.platforms.interface import PlatformCapabilities
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).parent))
from test_eagle_tree import QLEN_ONLY, _build, _gen_draft_tree  # noqa: E402

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NATIVE = KernelBackend.TORCH


def _murmur_reference(seed: int, pos: int, col: int) -> int:
    """MurmurHash3 over (seed_lo, seed_hi, pos, col) on Python ints."""

    def mix(h, k):
        k = (k * 0xCC9E2D51) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * 0x1B873593) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        return (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    h = 0
    for k in (seed & 0xFFFFFFFF, seed >> 32, pos, col):
        h = mix(h, k)
    h ^= 16
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    return h ^ (h >> 16)


def _sampling_outputs(bs, slots, num_spec_tokens, device):
    return (
        torch.full((bs * slots,), -1, dtype=torch.int32, device=device),
        torch.full((bs, num_spec_tokens), -1, dtype=torch.int32, device=device),
        torch.zeros(bs, dtype=torch.int32, device=device),
    )


def _chain_retrieve(bs, slots, device):
    index = torch.arange(bs * slots, device=device).view(bs, slots)
    next_token = torch.where(
        torch.arange(slots, device=device) < slots - 1,
        torch.arange(1, slots + 1, device=device),
        -1,
    ).repeat(bs, 1)
    return index, next_token, torch.full_like(index, -1)


class TestSpecSamplingNative(CustomTestCase):
    """The pure-torch references that out-of-tree platforms fall back to."""

    def test_tree_sampling_accepts_then_samples_the_residual(self):
        # One chain of 3 drafts (tokens 1, 2, 3) over a 5-token vocab. The
        # target accepts drafts 1 and 2 outright and gives draft 3 no mass, so
        # it is rejected and the bonus comes from the residual without token 3.
        vocab, slots = 5, 4
        candidates = torch.tensor([[0, 1, 2, 3]])
        target = torch.zeros(1, slots, vocab)
        target[0, 0, 1] = target[0, 1, 2] = 1.0
        target[0, 2] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
        target[0, 3, 0] = 1.0
        draft = torch.zeros_like(target)
        predicts, accept_index, accept_num = _sampling_outputs(1, slots, slots, "cpu")
        TreeSpeculativeSamplingOp()(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_num,
            candidates=candidates,
            retrive_index=_chain_retrieve(1, slots, "cpu")[0],
            retrive_next_token=_chain_retrieve(1, slots, "cpu")[1],
            retrive_next_sibling=_chain_retrieve(1, slots, "cpu")[2],
            uniform_samples=torch.full((1, slots), 0.5),
            uniform_samples_for_final_sampling=torch.tensor([0.3]),
            target_probs=target,
            draft_probs=draft,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
            backend=NATIVE,
        )
        self.assertEqual(accept_num.tolist(), [2])
        self.assertEqual(accept_index.tolist(), [[0, 1, 2, -1]])
        self.assertEqual(predicts.tolist(), [1, 2, 4, -1])

    def test_chain_sampling_accepts_matching_draft(self):
        # Target equals draft, so every draft is accepted (coin * q < p for any
        # coin < 1) and the bonus is sampled from the last target row.
        vocab, slots = 4, 3
        target = torch.tensor(
            [[[0.1, 0.6, 0.2, 0.1], [0.1, 0.1, 0.7, 0.1], [0.0, 0.0, 0.0, 1.0]]]
        )
        predicts, accept_index, accept_num = _sampling_outputs(1, slots, slots, "cpu")
        for block in (False, True):
            with self.subTest(block_verification=block):
                ChainSpeculativeSamplingOp()(
                    predicts=predicts,
                    accept_index=accept_index,
                    accept_token_num=accept_num,
                    candidates=torch.tensor([[0, 1, 2]]),
                    retrive_index=torch.tensor([[0, 1, 2]]),
                    retrive_next_token=None,
                    retrive_next_sibling=None,
                    uniform_samples=torch.full((1, slots), 0.9),
                    uniform_samples_for_final_sampling=torch.tensor([0.5]),
                    target_probs=target,
                    draft_probs=target[:, :2].clone(),
                    threshold_single=1.0,
                    threshold_acc=1.0,
                    deterministic=True,
                    block_verification=block,
                    backend=NATIVE,
                )
                self.assertEqual(accept_num.tolist(), [2])
                self.assertEqual(accept_index.tolist(), [[0, 1, 2]])
                self.assertEqual(predicts.tolist(), [1, 2, 3])

    def test_murmur_hash_matches_scalar_reference(self):
        seeds = [0, 12345, 2**63 + 7, 2**64 - 1]
        positions = [0, 3, 1 << 20, 99]
        cols = torch.tensor([0, 1, 7, 151935])
        hashed = _murmur_hash32_torch(
            torch.tensor(seeds, dtype=torch.uint64),
            torch.tensor(positions, dtype=torch.uint64),
            cols,
        )
        expected = [
            [_murmur_reference(s, p, int(c)) for c in cols]
            for s, p in zip(seeds, positions)
        ]
        self.assertEqual(hashed.to(torch.int64).tolist(), expected)

    def test_out_of_tree_without_triton_uses_torch(self):
        # Neither the Triton kernels nor the sgl_kernel CPU hash can run here.
        platform = SimpleNamespace(
            is_out_of_tree=lambda: True,
            capabilities=PlatformCapabilities(supports_triton=False),
        )
        seed = torch.tensor([7, 2**40], dtype=torch.uint64)
        pos = torch.tensor([3, 5], dtype=torch.uint64)
        cols = torch.arange(4)
        scores = torch.tensor([[0.1, 0.9, 0.2], [0.7, 0.3, 0.5]])
        with mock.patch("sglang.srt.platforms.current_platform", platform):
            hashed = murmur_hash32(seed, pos, cols)
            argmax = row_argmax(scores)
        self.assertTrue(torch.equal(hashed, _murmur_hash32_torch(seed, pos, cols)))
        self.assertEqual(argmax.tolist(), [1, 0])

    def test_renorm(self):
        probs = torch.tensor([[0.1, 0.4, 0.2, 0.3], [0.2, 0.3, 0.4, 0.1]])
        top_k = TopKRenormProbOp()(probs, torch.tensor([2, 1]), backend=NATIVE)
        torch.testing.assert_close(
            top_k, torch.tensor([[0.0, 4 / 7, 0.0, 3 / 7], [0.0, 0.0, 1.0, 0.0]])
        )
        # A token is kept while the mass before it is at most top_p.
        top_p = TopPRenormProbOp()(probs, torch.tensor([0.6, 0.5]), backend=NATIVE)
        torch.testing.assert_close(
            top_p, torch.tensor([[0.0, 4 / 7, 0.0, 3 / 7], [0.0, 3 / 7, 4 / 7, 0.0]])
        )


@unittest.skipUnless(torch.cuda.is_available(), "needs the CUDA sampling kernels")
class TestSpecSamplingNativeMatchesDevice(CustomTestCase):
    def test_tree_sampling(self):
        g = torch.Generator().manual_seed(1)
        for trial in range(10):
            for topk, steps, dtn in ((1, 3, 4), (2, 3, 6), (4, 4, 8)):
                bs, vocab = 4, 50
                parent_list, selected = _gen_draft_tree(bs, topk, steps, dtn, g)
                _, _, retrieve = _build(
                    BuildEagleTreeOp(),
                    parent_list=parent_list.cuda(),
                    selected_index=selected.cuda(),
                    seq_lens=torch.randint(1, 9, (bs,), generator=g).cuda(),
                    topk=topk,
                    depth=steps,
                    dtn=dtn,
                    mode=QLEN_ONLY,
                    backend=None,
                )
                # Few candidate tokens with boosted mass make partial acceptance likely.
                candidates = torch.randint(0, 6, (bs, dtn), generator=g).cuda()
                target = torch.rand(bs, dtn, vocab, generator=g).pow(4)
                target[..., :6] *= 20
                target = (target / target.sum(-1, keepdim=True)).cuda()
                coins = torch.rand(bs, dtn, generator=g).cuda()
                final_coins = torch.rand(bs, generator=g).cuda()
                results = []
                for backend in (None, NATIVE):
                    outputs = _sampling_outputs(bs, dtn, steps + 1, "cuda")
                    draft = torch.zeros_like(target)
                    TreeSpeculativeSamplingOp()(
                        predicts=outputs[0],
                        accept_index=outputs[1],
                        accept_token_num=outputs[2],
                        candidates=candidates,
                        retrive_index=retrieve[0],
                        retrive_next_token=retrieve[1],
                        retrive_next_sibling=retrieve[2],
                        uniform_samples=coins,
                        uniform_samples_for_final_sampling=final_coins,
                        target_probs=target,
                        draft_probs=draft,
                        threshold_single=1.0,
                        threshold_acc=1.0,
                        deterministic=True,
                        backend=backend,
                    )
                    results.append((*outputs, draft))
                with self.subTest(trial=trial, topk=topk):
                    for device_out, native_out in zip(*results):
                        self.assertTrue(torch.equal(device_out, native_out))

    def test_chain_sampling(self):
        g = torch.Generator().manual_seed(2)
        for block in (False, True):
            for steps, vocab in ((1, 7), (3, 17), (5, 4099)):
                bs, slots = 7, steps + 1
                target = torch.rand(bs, slots, vocab, generator=g).softmax(-1)
                draft = torch.rand(bs, steps, vocab, generator=g).softmax(-1)
                candidates = torch.zeros(bs, slots, dtype=torch.long)
                candidates[:, 1:] = torch.multinomial(
                    draft.flatten(0, 1), 1, generator=g
                ).view(bs, steps)
                coins = torch.rand(bs, slots, generator=g).cuda()
                final_coins = torch.rand(bs, generator=g).cuda()
                results = []
                for backend in (KernelBackend.TRITON, NATIVE):
                    outputs = _sampling_outputs(bs, slots, slots, "cuda")
                    ChainSpeculativeSamplingOp()(
                        predicts=outputs[0],
                        accept_index=outputs[1],
                        accept_token_num=outputs[2],
                        candidates=candidates.cuda(),
                        retrive_index=torch.arange(bs * slots).view(bs, slots).cuda(),
                        retrive_next_token=None,
                        retrive_next_sibling=None,
                        uniform_samples=coins,
                        uniform_samples_for_final_sampling=final_coins,
                        target_probs=target.cuda(),
                        draft_probs=draft.cuda(),
                        threshold_single=1.0,
                        threshold_acc=1.0,
                        deterministic=True,
                        block_verification=block,
                        backend=backend,
                    )
                    results.append(outputs)
                with self.subTest(block=block, steps=steps):
                    for device_out, native_out in zip(*results):
                        self.assertTrue(torch.equal(device_out, native_out))

    def test_small_ops(self):
        g = torch.Generator().manual_seed(3)
        seed = torch.randint(0, 2**62, (6,), generator=g).to(torch.uint64).cuda()
        pos = torch.randint(0, 10_000, (6,), generator=g).to(torch.uint64).cuda()
        cols = torch.arange(300).cuda()
        self.assertTrue(
            torch.equal(
                murmur_hash32(seed, pos, cols), _murmur_hash32_torch(seed, pos, cols)
            )
        )
        probs = torch.rand(16, 1000, generator=g).softmax(-1).cuda()
        top_ks = torch.randint(1, 200, (16,), generator=g).to(torch.int32).cuda()
        top_ps = (torch.rand(16, generator=g) * 0.9 + 0.05).cuda()
        for op, arg in ((TopKRenormProbOp(), top_ks), (TopPRenormProbOp(), top_ps)):
            device_out, native_out = op(probs, arg), op(probs, arg, backend=NATIVE)
            self.assertTrue(torch.equal(device_out > 0, native_out > 0))
            torch.testing.assert_close(device_out, native_out, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
