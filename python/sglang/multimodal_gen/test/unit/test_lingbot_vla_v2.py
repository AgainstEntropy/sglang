# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_vla_v2 import (
    LingbotVlaV2PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.action import ActionSamplingParams
from sglang.multimodal_gen.configs.sample.lingbot_vla_v2 import (
    LingbotVlaV2SamplingParams,
)
from sglang.multimodal_gen.configs.sample.vla import VLAObservationSamplingParams
from sglang.multimodal_gen.runtime.models.vlas.lingbot_vla_v2_core import (
    LingbotMoeMlp,
    LingbotVlaV2ArchConfig,
    LingbotVlaV2CoreModel,
    eager_joint_attention,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_vla_v2_preprocess import (
    _tensor_from_image,
    format_lingbot_prompt,
    pad_state_vector,
)


def _tiny_vlm_config() -> Qwen3VLConfig:
    return Qwen3VLConfig(
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=True,
        text_config={
            "attention_bias": False,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "head_dim": 16,
            "hidden_act": "silu",
            "hidden_size": 64,
            "intermediate_size": 128,
            "max_position_embeddings": 4096,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [4, 2, 2],
                "rope_type": "default",
            },
            "rope_theta": 5000000,
            "tie_word_embeddings": True,
            # must cover the vision boundary / eos special token ids
            "vocab_size": 151936,
        },
        vision_config={
            "deepstack_visual_indexes": [0, 1],
            "depth": 2,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 32,
            "in_channels": 3,
            "intermediate_size": 64,
            "num_heads": 2,
            "num_position_embeddings": 64,
            "out_hidden_size": 64,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    )


_TINY_ARCH = LingbotVlaV2ArchConfig(
    expert_num_layers=2,
    expert_hidden_size=32,
    expert_num_attention_heads=4,
    expert_num_key_value_heads=2,
    expert_head_dim=16,
    moe_num_experts=4,
    moe_top_k=2,
    moe_intermediate_size=16,
    moe_shared_expert_intermediate_size=16,
    max_state_dim=7,
    max_action_dim=7,
    action_horizon=5,
    num_task_tokens=4,
    align_num_backbone_tokens=8,
)


@pytest.fixture(scope="module")
def tiny_core_model() -> LingbotVlaV2CoreModel:
    torch.manual_seed(0)
    model = LingbotVlaV2CoreModel(_TINY_ARCH, vlm_config=_tiny_vlm_config())
    model.eval()
    return model


def test_sampling_params_is_action_params_without_visual_fields():
    params = LingbotVlaV2SamplingParams()
    assert isinstance(params, ActionSamplingParams)
    assert isinstance(params, VLAObservationSamplingParams)
    assert params.action_dim == 55
    assert params.action_horizon == 50
    assert params.num_inference_steps == 10
    for visual_field in ("height", "width", "fps", "negative_prompt"):
        assert not hasattr(params, visual_field)


def test_sampling_params_build_request_extra_carries_observation():
    params = LingbotVlaV2SamplingParams(
        prompt="pick up the cup",
        state=[0.1] * 14,
        images={"camera_top": [[0, 0, 0]]},
        camera_order=["camera_top"],
        output_format="numpy",
        enable_prefix_cache=False,
    )
    extra = params.build_request_extra()
    vla = extra["vla"]
    assert vla["observation"]["prompt"] == "pick up the cup"
    assert vla["observation"]["state"] == [0.1] * 14
    assert vla["observation"]["camera_order"] == ("camera_top",)
    assert vla["options"]["output_format"] == "numpy"
    assert vla["options"]["enable_prefix_cache"] is False


def test_pipeline_config_defaults():
    config = LingbotVlaV2PipelineConfig()
    assert config.task_type.is_action_gen()
    assert config.supports_action_endpoint()
    assert not config.supports_disaggregation()
    assert config.policy_family == "lingbot_vla_v2"
    assert config.action_dim == config.state_dim == config.output_action_dim == 55


def test_registry_resolves_lingbot_vla_v2():
    from sglang.multimodal_gen.registry import (
        KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS,
        get_model_info,
    )

    assert (
        KNOWN_NON_DIFFUSERS_DIFFUSION_MODEL_PATTERNS["lingbot-vla"]
        == "LingbotVlaV2Pipeline"
    )
    info = get_model_info("robbyant/lingbot-vla-v2-6b")
    assert info.pipeline_cls.pipeline_name == "LingbotVlaV2Pipeline"
    assert info.sampling_param_cls is LingbotVlaV2SamplingParams
    assert info.pipeline_config_cls is LingbotVlaV2PipelineConfig


def test_format_lingbot_prompt_matches_qwen3_chat_template():
    assert (
        format_lingbot_prompt("pick up the cup")
        == "<|im_start|>user\npick up the cup<|im_end|>\n"
    )


def test_pad_state_vector():
    state = torch.arange(14, dtype=torch.float32).unsqueeze(0)
    padded = pad_state_vector(state, 55)
    assert padded.shape == (1, 55)
    assert torch.equal(padded[0, :14], state[0])
    assert padded[0, 14:].abs().sum() == 0
    with pytest.raises(ValueError):
        pad_state_vector(torch.zeros(1, 56), 55)


def test_tensor_from_image_scales_to_byte_range():
    hwc_uint8 = torch.randint(0, 256, (16, 16, 3), dtype=torch.uint8)
    chw = _tensor_from_image(hwc_uint8)
    assert chw.shape == (3, 16, 16)
    assert torch.equal(chw, hwc_uint8.permute(2, 0, 1).to(torch.float32))

    unit_range = torch.rand(3, 16, 16)
    scaled = _tensor_from_image(unit_range)
    assert torch.allclose(scaled, unit_range * 255.0)


def test_moe_mlp_matches_naive_per_expert_loop():
    torch.manual_seed(1)
    mlp = LingbotMoeMlp(_TINY_ARCH)
    for param in mlp.parameters():
        torch.nn.init.normal_(param, std=0.1)
    mlp.e_score_correction_bias.normal_(std=0.5)

    x = torch.randn(1, 6, _TINY_ARCH.expert_hidden_size)
    out = mlp(x)

    flat = x.reshape(-1, x.shape[-1])
    logits = F.linear(flat.float(), mlp.gate.weight.float())
    scores = logits.sigmoid()
    _, selected = torch.topk(
        scores + mlp.e_score_correction_bias, _TINY_ARCH.moe_top_k, dim=-1
    )
    weights = scores.gather(1, selected)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
    weights = weights * _TINY_ARCH.moe_routed_scaling_factor

    expected = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for slot in range(_TINY_ARCH.moe_top_k):
            expert = int(selected[token, slot])
            h = flat[token]
            gate = F.silu(mlp.experts.gate_proj[expert] @ h)
            inter = gate * (mlp.experts.up_proj[expert] @ h)
            expected[token] += weights[token, slot].to(h.dtype) * (
                mlp.experts.down_proj[expert] @ inter
            )
        expected[token] += mlp.shared_expert(flat[token])
    assert torch.allclose(out.reshape(-1, x.shape[-1]), expected, atol=1e-5)


def test_eager_joint_attention_matches_sdpa():
    torch.manual_seed(2)
    q = torch.randn(1, 5, 4, 16)
    k = torch.randn(1, 2, 9, 16)
    v = torch.randn(1, 2, 9, 16)
    mask = torch.rand(1, 5, 9) > 0.3
    mask[..., 0] = True  # keep every query row attendable

    out = eager_joint_attention(q, k, v, mask)

    expected = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        attn_mask=mask[:, None],
    )
    expected = expected.permute(0, 2, 1, 3).reshape(1, 5, -1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_tiny_core_state_dict_matches_checkpoint_naming(tiny_core_model):
    keys = set(tiny_core_model.state_dict().keys())
    expected = [
        "state_proj.weight",
        "action_in_proj.bias",
        "action_out_proj.weight",
        "action_time_mlp_in.weight",
        "action_time_mlp_out.bias",
        "depth_align_embs",
        "current_video_align_embs",
        "future_depth_align_embs",
        "future_video_align_embs",
        "current_shared_task_proj.weight",
        "future_shared_task_proj.bias",
        "qwenvl_with_expert.qwen_expert.model.norm.weight",
        "qwenvl_with_expert.qwen_expert.model.layers.0.input_layernorm.weight",
        "qwenvl_with_expert.qwen_expert.model.layers.0.input_layernorm.gamma.weight",
        "qwenvl_with_expert.qwen_expert.model.layers.0.input_layernorm.beta.bias",
        "qwenvl_with_expert.qwen_expert.model.layers.0.post_attention_layernorm.weight",
        "qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.q_proj.bias",
        "qwenvl_with_expert.qwen_expert.model.layers.0.self_attn.o_proj.weight",
        "qwenvl_with_expert.qwen_expert.model.layers.0.mlp.gate.weight",
        "qwenvl_with_expert.qwen_expert.model.layers.0.mlp.e_score_correction_bias",
        "qwenvl_with_expert.qwen_expert.model.layers.0.mlp.experts.gate_proj",
        "qwenvl_with_expert.qwen_expert.model.layers.0.mlp.experts.up_proj",
        "qwenvl_with_expert.qwen_expert.model.layers.0.mlp.experts.down_proj",
        "qwenvl_with_expert.qwen_expert.model.layers.0.mlp.shared_expert.gate_proj.weight",
        "qwenvl_with_expert.qwenvl.model.visual.patch_embed.proj.weight",
        "qwenvl_with_expert.qwenvl.model.language_model.embed_tokens.weight",
        "qwenvl_with_expert.qwenvl.model.language_model.layers.0.self_attn.q_norm.weight",
    ]
    missing = [key for key in expected if key not in keys]
    assert not missing, f"missing keys: {missing}"
    assert not any(key.startswith("qwenvl_with_expert.qwenvl.lm_head") for key in keys)


def test_tiny_core_prefix_and_denoise_step(tiny_core_model):
    torch.manual_seed(3)
    num_cams = 2
    patch_dim = 3 * 2 * 4 * 4
    pixel = torch.randn(num_cams * 16, patch_dim)
    grid = torch.tensor([[1, 4, 4]] * num_cams)
    image_masks = torch.tensor([[True, False]])
    tokens = torch.randint(0, 1000, (1, 6))
    token_masks = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)

    with torch.no_grad():
        cache, pad_masks, layout = tiny_core_model.encode_prefix(
            pixel_values=pixel,
            image_grid_thw=grid,
            image_masks=image_masks,
            lang_tokens=tokens,
            lang_masks=token_masks,
        )
    # 2 cameras x (4 merged patches + 2 boundary tokens) + 6 language tokens
    # + 2 x num_task_tokens align queries.
    assert pad_masks.shape == (1, 26)
    assert len(cache) == _TINY_ARCH.expert_num_layers
    # masked camera and padded language positions are invalid
    assert not pad_masks[0, 6:12].any()
    assert not pad_masks[0, 16:18].any()
    assert layout["prefix_offsets"].shape == (1,)

    state = torch.randn(1, 7)
    x_t = torch.randn(1, 5, 7)
    with torch.no_grad():
        v1 = tiny_core_model.denoise_step(
            past_key_values=cache,
            prefix_pad_masks=pad_masks,
            prefix_offsets=layout["prefix_offsets"],
            state=state,
            x_t=x_t,
            timestep=torch.tensor([1.0]),
        )
        v2 = tiny_core_model.denoise_step(
            past_key_values=cache,
            prefix_pad_masks=pad_masks,
            prefix_offsets=layout["prefix_offsets"],
            state=state,
            x_t=x_t,
            timestep=torch.tensor([1.0]),
        )
    assert v1.shape == (1, _TINY_ARCH.action_horizon, _TINY_ARCH.max_action_dim)
    assert torch.isfinite(v1).all()
    assert torch.equal(v1, v2)
    # the cache must not grow across denoise steps
    assert cache.get_seq_length() == 26
