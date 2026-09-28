# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 K/V-gather sequence parallelism: the gathered-K/V attention
matches full attention on every rank's row shard, and each layer resolves its
SP exchange the way the shared USPAttention rule does."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
    MiniMaxH3Attention,
    _kv_gather_attention_varlen,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_H3 = "sglang.multimodal_gen.runtime.models.dits.minimax_h3"


def _sdpa_thd(q, k, v, scale):
    return F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), scale=scale
    ).transpose(0, 1)


class _DenseImpl:
    def __init__(self, scale: float):
        self.scale = scale

    def forward(self, q, k, v, _metadata):
        return _sdpa_thd(q[0], k[0], v[0], self.scale)[None]


class TestKVGatherAttention(unittest.TestCase):
    def test_row_shards_match_full_attention_on_real_rows(self):
        torch.manual_seed(0)
        world, local_rows, heads, head_dim, used = 2, 8, 3, 4, 13
        seq = world * local_rows
        q, k, v = (torch.randn(seq, heads, head_dim) for _ in range(3))
        scale = head_dim**-0.5
        impl = _DenseImpl(scale)
        expected = _sdpa_thd(q[:used], k[:used], v[:used], scale)
        gathered = torch.stack((k, v))

        for rank in range(world):
            rows = slice(rank * local_rows, (rank + 1) * local_rows)
            with patch(
                f"{_H3}.sequence_model_parallel_all_gather",
                return_value=gathered,
            ) as all_gather:
                out = _kv_gather_attention_varlen(
                    q[rows], k[rows], v[rows], attn_impl=impl, real_seq_len=used
                )
            local_kv, dim = all_gather.call_args.args[0], all_gather.call_args.kwargs
            self.assertEqual(tuple(local_kv.shape), (2, local_rows, heads, head_dim))
            self.assertEqual(dim, {"dim": 1})
            self.assertEqual(tuple(out.shape), (local_rows, heads, head_dim))
            real = min(max(used - rank * local_rows, 0), local_rows)
            torch.testing.assert_close(out[:real], expected[rows][:real])


class TestKVGatherModeResolution(unittest.TestCase):
    def _uses(self, backend_enum, *, degree=2, auto=False):
        attn = MiniMaxH3Attention.__new__(MiniMaxH3Attention)
        attn._attention_backend_enum = backend_enum
        attn._kv_gather = None
        stub = SimpleNamespace(kv_gather_degree=degree, sp_split_auto=auto)
        with patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=stub,
        ):
            return attn._uses_kv_gather()

    def test_dense_backends_take_the_gather(self):
        for backend in (
            AttentionBackendEnum.FA,
            AttentionBackendEnum.DYNAMIC_CUDNN_SDPA,
            AttentionBackendEnum.TORCH_SDPA,
        ):
            for auto in (False, True):
                self.assertTrue(self._uses(backend, auto=auto))

    def test_degree_one_keeps_ulysses(self):
        self.assertFalse(self._uses(AttentionBackendEnum.FA, degree=1))

    def test_sparse_and_hybrid_backends(self):
        for backend in (
            AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,
            AttentionBackendEnum.SUBBLOCK_SPARSE_ATTN,
            AttentionBackendEnum.HYBRID_WINDOW_ATTN_H3,
        ):
            self.assertFalse(self._uses(backend, auto=True))
            with self.assertRaises(NotImplementedError):
                self._uses(backend, auto=False)


if __name__ == "__main__":
    unittest.main()
