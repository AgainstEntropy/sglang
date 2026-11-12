"""MLP must honor bias=False on both of its projections."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.mlp import MLP
from sglang.test.test_utils import CustomTestCase

# The parallel linears read only the TP group's size and rank.
_TP_GROUP = SimpleNamespace(world_size=1, rank_in_group=0)
_GET_TP_GROUP = "sglang.multimodal_gen.runtime.layers.linear.get_tp_group"


class TestMLPBias(CustomTestCase):
    def _mlp(self, *, bias: bool) -> MLP:
        with patch(_GET_TP_GROUP, return_value=_TP_GROUP):
            return MLP(input_dim=8, mlp_hidden_dim=16, bias=bias, dtype=torch.float32)

    def test_bias_false_has_no_bias(self):
        mlp = self._mlp(bias=False)
        self.assertIsNone(mlp.fc_in.bias)
        self.assertIsNone(mlp.fc_out.bias)

    def test_bias_true_has_bias(self):
        mlp = self._mlp(bias=True)
        self.assertIsNotNone(mlp.fc_in.bias)
        self.assertIsNotNone(mlp.fc_out.bias)


if __name__ == "__main__":
    unittest.main()
