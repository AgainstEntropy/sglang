"""Explicit request values a pipeline would change are rejected, not rewritten."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from sglang.multimodal_gen.configs.pipeline_configs.joy_echo import (
    JoyEchoPipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
    LingBotWorldCausalDMDConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2_5 import (
    LTX25_DISTILLED_SIGMA_VALUES,
    LTX25PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_2_I2V_A14B_Config,
    Wan_Animate_2_14B_Config,
    WanI2V480PConfig,
    WanI2V720PConfig,
)
from sglang.multimodal_gen.configs.sample.joy_echo import JoyEchoSamplingParams
from sglang.multimodal_gen.configs.sample.lingbot_world import (
    LingBotWorldSamplingParams,
)
from sglang.multimodal_gen.configs.sample.ltx_2_5 import LTX25SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.wan import (
    Wan2_2_I2V_A14B_SamplingParam,
    Wan_Animate_2_14B_SamplingParam,
    WanI2V_14B_480P_SamplingParam,
    WanI2V_14B_720P_SamplingParam,
)
from sglang.multimodal_gen.configs.task_type import DataType
from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils

_IMAGE = "/nonexistent/condition.png"


def _server_args(pipeline_config, **overrides):
    fields = dict(
        model_path="test-model",
        model_id=None,
        model_variant=None,
        backend="sglang",
        pipeline_class_name=None,
        pipeline_config=pipeline_config,
        output_path=None,
        comfyui_mode=False,
        num_gpus=1,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _build(server_args, defaults, **kwargs):
    """The SamplingParams path shared by DiffGenerator, the CLI and the HTTP routes."""
    with patch.object(SamplingParams, "from_pretrained", return_value=defaults):
        return SamplingParams.from_user_sampling_params_args(
            server_args.model_path, server_args, prompt="a fox", **kwargs
        )


def _build_http(server_args, defaults, **kwargs):
    with (
        patch.object(openai_utils, "get_global_server_args", return_value=server_args),
        patch.object(SamplingParams, "from_pretrained", return_value=defaults),
    ):
        return openai_utils.build_sampling_params(
            "req", prompt="a fox", request_data_type=DataType.VIDEO, **kwargs
        )


class TestLTX25PinnedStepCount(unittest.TestCase):
    pinned_steps = len(LTX25_DISTILLED_SIGMA_VALUES)

    def test_explicit_step_count_off_the_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_inference_steps=20"):
            _build(
                _server_args(LTX25PipelineConfig()),
                LTX25SamplingParams(),
                num_inference_steps=20,
            )

    def test_http_returns_400(self):
        with self.assertRaises(HTTPException) as ctx:
            _build_http(
                _server_args(LTX25PipelineConfig()),
                LTX25SamplingParams(),
                num_inference_steps=30,
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("num_inference_steps=30", ctx.exception.detail)

    def test_unset_or_matching_step_count_is_accepted(self):
        server_args = _server_args(LTX25PipelineConfig())
        for kwargs in ({}, {"num_inference_steps": self.pinned_steps}):
            with self.subTest(kwargs=kwargs):
                params = _build(server_args, LTX25SamplingParams(), **kwargs)
                self.assertEqual(params.num_inference_steps, self.pinned_steps)

    def test_dev_variant_takes_any_step_count(self):
        # The dev pipeline drops the pinned schedule only in the worker process.
        for variant in ("dev", "SFT"):
            with self.subTest(variant=variant):
                params = _build(
                    _server_args(LTX25PipelineConfig(), model_variant=variant),
                    LTX25SamplingParams(),
                    num_inference_steps=30,
                )
                self.assertEqual(params.num_inference_steps, 30)

    def test_unpinned_schedule_takes_any_step_count(self):
        config = LTX25PipelineConfig()
        config.default_sigmas = None
        params = _build(
            _server_args(config), LTX25SamplingParams(), num_inference_steps=30
        )
        self.assertEqual(params.num_inference_steps, 30)

    def test_comfyui_mode_is_exempt(self):
        # ComfyUI drives one step per call and never runs the sigma stage.
        params = _build(
            _server_args(LTX25PipelineConfig(), comfyui_mode=True),
            LTX25SamplingParams(),
            num_inference_steps=1,
        )
        self.assertEqual(params.num_inference_steps, 1)


class TestJoyEchoScheduledStepCount(unittest.TestCase):
    def test_explicit_step_count_off_the_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_inference_steps=30"):
            _build(
                _server_args(JoyEchoPipelineConfig()),
                JoyEchoSamplingParams(),
                num_inference_steps=30,
            )

    def test_http_returns_400(self):
        with self.assertRaises(HTTPException) as ctx:
            _build_http(
                _server_args(JoyEchoPipelineConfig()),
                JoyEchoSamplingParams(),
                num_inference_steps=30,
            )
        self.assertEqual(ctx.exception.status_code, 400)

    def test_step_count_matching_the_sigmas_is_accepted(self):
        server_args = _server_args(JoyEchoPipelineConfig())
        for kwargs, steps in (
            ({}, 8),
            ({"num_inference_steps": 8}, 8),
            ({"num_inference_steps": 4, "sigmas": (1.0, 0.9, 0.7, 0.4, 0.0)}, 4),
        ):
            with self.subTest(kwargs=kwargs):
                params = _build(server_args, JoyEchoSamplingParams(), **kwargs)
                self.assertEqual(params.num_inference_steps, steps)


class TestWanI2VMaxArea(unittest.TestCase):
    def test_explicit_size_above_max_area_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "1280x720.*max_area of 399360"):
            _build(
                _server_args(WanI2V480PConfig()),
                WanI2V_14B_480P_SamplingParam(),
                image_path=_IMAGE,
                width=1280,
                height=720,
            )

    def test_one_explicit_side_counts_with_the_default_other_side(self):
        # The stage multiplies the explicit width by the default height of 480.
        with self.assertRaisesRegex(ValueError, "1280x480"):
            _build(
                _server_args(WanI2V480PConfig()),
                WanI2V_14B_480P_SamplingParam(),
                image_path=_IMAGE,
                width=1280,
            )

    def test_http_size_returns_400(self):
        with self.assertRaises(HTTPException) as ctx:
            _build_http(
                _server_args(WanI2V480PConfig()),
                WanI2V_14B_480P_SamplingParam(),
                image_path=_IMAGE,
                size="1280x720",
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("max_area", ctx.exception.detail)

    def test_sizes_within_max_area_are_accepted(self):
        for config, defaults, size in (
            (WanI2V480PConfig(), WanI2V_14B_480P_SamplingParam(), (832, 480)),
            (WanI2V480PConfig(), WanI2V_14B_480P_SamplingParam(), (480, 832)),
            (WanI2V720PConfig(), WanI2V_14B_720P_SamplingParam(), (1280, 720)),
            (Wan2_2_I2V_A14B_Config(), Wan2_2_I2V_A14B_SamplingParam(), (832, 480)),
        ):
            with self.subTest(config=type(config).__name__, size=size):
                params = _build(
                    _server_args(config),
                    defaults,
                    image_path=_IMAGE,
                    width=size[0],
                    height=size[1],
                )
                self.assertEqual((params.width, params.height), size)

    def test_default_size_is_not_checked(self):
        params = _build(
            _server_args(Wan2_2_I2V_A14B_Config()),
            Wan2_2_I2V_A14B_SamplingParam(),
            image_path=_IMAGE,
        )
        self.assertEqual((params.width, params.height), (1280, 720))

    def test_wan_animate_2_and_lingbot_world_realtime_are_exempt(self):
        # Neither runs InputValidationStage, so a larger size is honored.
        for config, defaults in (
            (Wan_Animate_2_14B_Config(), Wan_Animate_2_14B_SamplingParam()),
            (LingBotWorldCausalDMDConfig(), LingBotWorldSamplingParams()),
        ):
            with self.subTest(config=type(config).__name__):
                config.check_requested_area(
                    SimpleNamespace(
                        width=1920, height=1080, _explicit_fields={"width", "height"}
                    )
                )
                self.assertGreater(1920 * 1080, config.max_area)


if __name__ == "__main__":
    unittest.main()
