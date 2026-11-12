import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.base import TextConditioningOutput
from sglang.multimodal_gen.configs.pipeline_configs.ernie_image import (
    ErnieImagePipelineConfig,
    ernie_image_postprocess_text,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines.ernie_image import ErnieImagePipeline


class TestErnieImagePostprocessText(unittest.TestCase):
    def test_single_request_returns_full_length_conditioning(self) -> None:
        hidden_states = torch.randn(1, 5, 4)
        outputs = BaseEncoderOutput(hidden_states=(hidden_states, hidden_states))
        text_inputs = SimpleNamespace(attention_mask=torch.ones(1, 5, dtype=torch.long))

        result = ernie_image_postprocess_text(outputs, text_inputs)

        self.assertIsInstance(result, TextConditioningOutput)
        self.assertEqual(result.prompt_seq_lens, [5])
        self.assertEqual(result.prompt_embeds.shape, (1, 5, 4))
        self.assertTrue(bool(result.prompt_embeds_mask.all()))

    def test_ragged_batch_preserves_real_lengths(self) -> None:
        hidden_states = torch.randn(2, 6, 4)
        outputs = BaseEncoderOutput(hidden_states=(hidden_states, hidden_states))
        # Row 0 has 4 real tokens + 2 pad, row 1 has 6 real tokens (no pad).
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
        text_inputs = SimpleNamespace(attention_mask=mask)

        result = ernie_image_postprocess_text(outputs, text_inputs)

        self.assertIsInstance(result, TextConditioningOutput)
        self.assertEqual(result.prompt_seq_lens, [4, 6])
        self.assertEqual(result.prompt_embeds.shape, (2, 6, 4))
        self.assertEqual(
            result.prompt_embeds_mask.tolist(),
            [[True, True, True, True, False, False], [True] * 6],
        )


class TestErnieImagePrepareCondKwargs(unittest.TestCase):
    def test_sp_keeps_spatial_latents_for_dit_embedding(self) -> None:
        latents = torch.randn(2, 128, 8, 12)
        result, sharded = ErnieImagePipelineConfig().shard_latents_for_sp(None, latents)
        self.assertIs(result, latents)
        self.assertFalse(sharded)

    def _make_batch(self, prompt_embeds, prompt_seq_lens):
        return SimpleNamespace(
            height=256,
            width=256,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=prompt_embeds,
            prompt_seq_lens=prompt_seq_lens,
            negative_prompt_seq_lens=prompt_seq_lens,
        )

    def test_uniform_lengths_need_no_mask(self) -> None:
        config = ErnieImagePipelineConfig()
        prompt_embeds = [torch.randn(2, 30, 4)]
        batch = self._make_batch(prompt_embeds, [[30, 30]])

        cond_kwargs = config.prepare_pos_cond_kwargs(
            batch, device=torch.device("cpu"), rotary_emb=None, dtype=torch.float32
        )

        self.assertIsNone(cond_kwargs["encoder_hidden_states_mask"])
        self.assertEqual(cond_kwargs["txt_seq_lens"], [30, 30])

    def test_ragged_lengths_build_boundary_mask(self) -> None:
        config = ErnieImagePipelineConfig()
        prompt_embeds = [torch.randn(2, 30, 4)]
        batch = self._make_batch(prompt_embeds, [[18, 30]])

        cond_kwargs = config.prepare_pos_cond_kwargs(
            batch, device=torch.device("cpu"), rotary_emb=None, dtype=torch.float32
        )

        mask = cond_kwargs["encoder_hidden_states_mask"]
        self.assertEqual(mask.shape, (2, 30))
        self.assertTrue(bool(mask[0, 17]))
        self.assertFalse(bool(mask[0, 18]))
        self.assertTrue(bool(mask[1].all()))


class TestErnieImageSamplingDefaults(unittest.TestCase):
    """Turbo must not inherit the base model's 50-step, CFG 5.0 defaults."""

    # (model path, num_inference_steps, guidance_scale, uses CFG)
    CASES = [
        ("baidu/ERNIE-Image", 50, 5.0, True),
        ("/models/ERNIE-Image", 50, 5.0, True),
        ("baidu/ERNIE-Image-Turbo", 8, 1.0, False),
        ("/models/ERNIE-Image-Turbo", 8, 1.0, False),
        ("/hub/models--baidu--ERNIE-Image-Turbo/snapshots/4f2a", 8, 1.0, False),
    ]

    def setUp(self) -> None:
        registry.get_model_info.cache_clear()
        registry._get_config_info.cache_clear()
        self.addCleanup(registry.get_model_info.cache_clear)
        self.addCleanup(registry._get_config_info.cache_clear)
        model_index = {"_class_name": ErnieImagePipeline.pipeline_name}
        patcher = patch.object(
            registry, "maybe_download_model_index", lambda _: model_index
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_defaults_follow_the_checkpoint(self) -> None:
        for model_path, steps, guidance, uses_cfg in self.CASES:
            with self.subTest(model_path=model_path):
                params = SamplingParams.from_pretrained(model_path, backend="sglang")
                self.assertEqual(params.num_inference_steps, steps)
                self.assertEqual(params.guidance_scale, guidance)
                self.assertEqual(params.should_use_classifier_free_guidance(), uses_cfg)


if __name__ == "__main__":
    unittest.main()
