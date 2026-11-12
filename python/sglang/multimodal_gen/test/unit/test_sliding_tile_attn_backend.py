# SPDX-License-Identifier: Apache-2.0
"""Sliding Tile Attention reads --attention-backend-config, an addict.Dict.

A missing key reads as an empty Dict rather than None, so these cases pin the
defaults and the errors for keys the user leaves out or misspells.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import addict
import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import STA_Mode
from sglang.multimodal_gen.runtime.layers.attention.backends import sliding_tile_attn
from sglang.multimodal_gen.runtime.pipelines_core.stages import denoising
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.test.test_utils import CustomTestCase

PIPELINE_SKIP_TIME_STEPS = 15


def _server_args(**config):
    return SimpleNamespace(
        attention_backend_config=addict.Dict(config),
        pipeline_config=SimpleNamespace(skip_time_steps=PIPELINE_SKIP_TIME_STEPS),
    )


class TestSlidingTileAttnImpl(CustomTestCase):
    def test_missing_mask_strategy_file_path_names_the_flag(self):
        """Leaving out mask_strategy_file_path raises the flag hint, not open({})'s TypeError."""
        with (
            patch.object(sliding_tile_attn, "st_attn_backend_available", True),
            patch.object(
                sliding_tile_attn,
                "get_global_server_args",
                return_value=_server_args(STA_mode="STA_INFERENCE"),
            ),
            self.assertRaisesRegex(ValueError, "mask_strategy_file_path="),
        ):
            sliding_tile_attn.SlidingTileAttentionImpl(
                num_heads=2, head_size=64, causal=False, softmax_scale=0.125
            )


class TestPrepareStaParam(CustomTestCase):
    def setUp(self):
        self.stage = DenoisingStage.__new__(DenoisingStage)
        self.stage.transformer = SimpleNamespace(
            config=SimpleNamespace(num_layers=2, num_attention_heads=3)
        )
        configure_sta = patch.object(denoising, "configure_sta", MagicMock())
        self.configure_sta = configure_sta.start()
        self.addCleanup(configure_sta.stop)

    def _prepare(self, server_args):
        # 69x768x1280 is the only resolution the search and tuning modes accept.
        batch = SimpleNamespace(
            timesteps=torch.zeros(4), num_frames=69, height=768, width=1280
        )
        self.stage.prepare_sta_param(batch, server_args)
        return self.configure_sta.call_args.kwargs

    def test_unknown_mode_lists_the_valid_names(self):
        """A bad STA_mode raises ValueError, not UnboundLocalError from the handler.

        STA_inference is the enum value, not its name; NONE has no STA_param branch.
        """
        for mode_name in ("STA_BOGUS", "STA_inference", "NONE"):
            with self.subTest(mode_name=mode_name):
                with self.assertRaisesRegex(ValueError, "STA_INFERENCE") as ctx:
                    self._prepare(_server_args(STA_mode=mode_name))
                message = str(ctx.exception)
                self.assertIn(repr(mode_name), message)
                self.assertNotIn("'NONE'", message.partition("expected one of")[2])

    def test_missing_mode_defaults_to_inference(self):
        with patch.dict(
            os.environ, {"SGLANG_DIFFUSION_ATTENTION_CONFIG": "/tmp/sta_config.json"}
        ):
            kwargs = self._prepare(_server_args(mask_strategy_file_path="/tmp/m.json"))
        self.assertIs(kwargs["mode"], STA_Mode.STA_INFERENCE)

    def test_missing_skip_time_steps_uses_the_pipeline_default(self):
        """Tuning without skip_time_steps gets an int, not the empty Dict addict returns."""
        kwargs = self._prepare(_server_args(STA_mode="STA_TUNING"))
        self.assertIs(kwargs["mode"], STA_Mode.STA_TUNING)
        self.assertEqual(kwargs["skip_time_steps"], PIPELINE_SKIP_TIME_STEPS)


if __name__ == "__main__":
    unittest.main()
