import unittest

import torch

from sglang.kernels.ops.speculative.spec_tree import (
    BuildEagleTreeOp,
    VerifyEagleTreeGreedyOp,
)
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

FULL_MASK, QLEN_ONLY = 0, 1
NATIVE = KernelBackend.TORCH


def _gen_draft_tree(bs, topk, num_steps, draft_token_num, generator):
    """A random but valid (parent_list, selected_index), as the EAGLE draft loop builds."""
    scores = torch.rand(bs, topk, generator=generator)
    score_chunks = [scores]
    parents_chunks = [torch.arange(-1, topk, dtype=torch.int64).expand(bs, -1)]
    cum_scores = scores
    for i in range(1, num_steps):
        expand_scores = cum_scores.unsqueeze(2) * torch.rand(
            bs, topk, topk, generator=generator
        )
        cum_scores, topk_cs_index = torch.topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )
        score_chunks.append(expand_scores.flatten(start_dim=1))
        parents_chunks.append(topk_cs_index + (topk * topk * (i - 1) + topk))
    selected_index = torch.sort(
        torch.topk(torch.cat(score_chunks, dim=1), draft_token_num - 1).indices, dim=-1
    ).values
    return torch.cat(parents_chunks[:-1], dim=1).contiguous(), selected_index


def _build(
    op, *, parent_list, selected_index, seq_lens, topk, depth, dtn, mode, backend
):
    device = seq_lens.device
    bs = seq_lens.numel()
    if mode == FULL_MASK:
        mask_len = int(seq_lens.sum()) * dtn + bs * dtn * dtn
    else:
        mask_len = bs * dtn * dtn
    tree_mask = torch.ones(mask_len, dtype=torch.bool, device=device)
    positions = torch.empty(bs * dtn, dtype=torch.long, device=device)
    retrieve = torch.full((3, bs, dtn), -1, dtype=torch.long, device=device)
    op(
        parent_list,
        selected_index,
        seq_lens,
        tree_mask,
        positions,
        *retrieve,
        topk,
        depth,
        dtn,
        mode,
        backend=backend,
    )
    return tree_mask, positions, retrieve


def _verify(op, *, candidates, retrieve, target_predict, num_spec_step, backend):
    bs, dtn = candidates.shape
    device = candidates.device
    predicts = torch.full((bs * dtn,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, num_spec_step), -1, dtype=torch.int32, device=device)
    accept_token_num = torch.zeros(bs, dtype=torch.int32, device=device)
    op(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        *retrieve,
        target_predict,
        backend=backend,
    )
    return predicts, accept_index, accept_token_num


class TestEagleTreeNative(CustomTestCase):
    """The pure-torch reference that out-of-tree platforms fall back to."""

    def test_chain_tree_is_causal(self):
        # topk=1 chain (NEXTN / MTP): each draft attends to the root and every
        # earlier draft, and sits one position after its parent.
        bs, num_steps, seq_lens = 2, 3, torch.tensor([5, 3])
        dtn = num_steps + 1
        parent_list = torch.arange(-1, num_steps - 1).repeat(bs, 1)
        selected_index = torch.arange(num_steps).repeat(bs, 1)
        causal = torch.tril(torch.ones(dtn, dtn, dtype=torch.bool))
        for mode in (QLEN_ONLY, FULL_MASK):
            with self.subTest(mode=mode):
                tree_mask, positions, retrieve = _build(
                    BuildEagleTreeOp(),
                    parent_list=parent_list,
                    selected_index=selected_index,
                    seq_lens=seq_lens,
                    topk=1,
                    depth=num_steps,
                    dtn=dtn,
                    mode=mode,
                    backend=NATIVE,
                )
                self.assertEqual(positions.tolist(), [5, 6, 7, 8, 3, 4, 5, 6])
                self.assertEqual(retrieve[0].tolist(), [[0, 1, 2, 3], [4, 5, 6, 7]])
                self.assertEqual(retrieve[1].tolist(), [[1, 2, 3, -1]] * bs)
                self.assertEqual(retrieve[2].tolist(), [[-1] * dtn] * bs)
                if mode == QLEN_ONLY:
                    expected = causal.repeat(bs, 1).flatten()
                else:
                    # Prefix columns keep the caller's fill; the tree block is causal.
                    expected = torch.cat(
                        [
                            torch.cat(
                                [torch.ones(dtn, int(s), dtype=torch.bool), causal],
                                dim=1,
                            ).flatten()
                            for s in seq_lens
                        ]
                    )
                self.assertTrue(torch.equal(tree_mask, expected))

    def test_verify_matches_golden(self):
        # Same case as the CUDA kernel test in test_eagle_utils.py.
        candidates = torch.tensor([[0, 1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12]])
        retrieve = torch.tensor(
            [
                [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
                [[1, 2, -1, 4, 5, -1], [4, 2, 3, -1, 5, -1]],
                [[-1, 3, -1, -1, -1, -1], [-1, -1, -1, -1, 1, -1]],
            ]
        )
        target_predict = torch.full((2, 6), 18)
        target_predict[0, 0], target_predict[0, 3], target_predict[0, 4] = 3, 4, 5
        target_predict[1, 0], target_predict[1, 4] = 11, 12
        predicts, accept_index, accept_token_num = _verify(
            VerifyEagleTreeGreedyOp(),
            candidates=candidates,
            retrieve=retrieve,
            target_predict=target_predict,
            num_spec_step=4,
            backend=NATIVE,
        )
        self.assertEqual(
            predicts.tolist(), [3, -1, -1, 4, 5, 18, 11, -1, -1, -1, 12, 18]
        )
        self.assertEqual(accept_index.tolist(), [[0, 3, 4, 5], [6, 10, 11, -1]])
        self.assertEqual(accept_token_num.tolist(), [3, 2])


@unittest.skipUnless(torch.cuda.is_available(), "needs the CUDA tree kernels")
class TestEagleTreeNativeMatchesCuda(CustomTestCase):
    def test_random_trees(self):
        generator = torch.Generator().manual_seed(0)
        build_op, verify_op = BuildEagleTreeOp(), VerifyEagleTreeGreedyOp()
        for topk, num_steps, dtn in ((1, 3, 4), (2, 3, 6), (4, 4, 8), (4, 5, 12)):
            bs = 3
            parent_list, selected_index = _gen_draft_tree(
                bs, topk, num_steps, dtn, generator
            )
            seq_lens = torch.randint(1, 20, (bs,), generator=generator)
            for mode in (QLEN_ONLY, FULL_MASK):
                with self.subTest(topk=topk, num_steps=num_steps, mode=mode):
                    outputs = {}
                    for backend in (None, NATIVE):
                        outputs[backend] = _build(
                            build_op,
                            parent_list=parent_list.cuda(),
                            selected_index=selected_index.cuda(),
                            seq_lens=seq_lens.cuda(),
                            topk=topk,
                            depth=num_steps,
                            dtn=dtn,
                            mode=mode,
                            backend=backend,
                        )
                    for cuda_out, native_out in zip(outputs[None], outputs[NATIVE]):
                        self.assertTrue(torch.equal(cuda_out, native_out))

            retrieve = outputs[None][2]
            # A small vocab makes partial acceptance along each tree likely.
            candidates = torch.randint(0, 3, (bs, dtn), generator=generator).cuda()
            target_predict = torch.randint(0, 3, (bs, dtn), generator=generator).cuda()
            with self.subTest(topk=topk, num_steps=num_steps, op="verify"):
                results = [
                    _verify(
                        verify_op,
                        candidates=candidates,
                        retrieve=retrieve,
                        target_predict=target_predict,
                        num_spec_step=num_steps + 1,
                        backend=backend,
                    )
                    for backend in (None, NATIVE)
                ]
                for cuda_out, native_out in zip(*results):
                    self.assertTrue(torch.equal(cuda_out, native_out))


if __name__ == "__main__":
    unittest.main()
