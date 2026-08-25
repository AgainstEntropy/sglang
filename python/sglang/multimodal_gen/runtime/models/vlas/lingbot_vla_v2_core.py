# SPDX-License-Identifier: Apache-2.0
"""LingBot-VLA-V2 core model: Qwen3-VL prefix + MoE action expert.

Ported from https://github.com/robbyant/lingbot-vla-v2 (Apache-2.0). The prefix
VLM and the 36-layer action expert run one joint attention per layer: each side
computes its own Q/K/V, the sequences are concatenated, Qwen3-VL mrope and a
single attention are applied, and each side finishes with its own o_proj/MLP.
Actions are produced by flow matching (Euler, t: 1 -> 0) with the prefix K/V
cached across denoise steps.

Module attribute names deliberately mirror the checkpoint weight names of
robbyant/lingbot-vla-v2-6b with the leading "model." stripped, so weight
loading needs no renaming.
"""

from __future__ import annotations

import math

import msgspec
import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    apply_rotary_pos_emb,
)

from sglang.multimodal_gen.runtime.vla.functional import make_att_2d_masks
from sglang.multimodal_gen.runtime.vla.prefix_cache import VLADensePrefixCache

# Matches the reference eager attention's masked fill value.
_ATTENTION_MASK_VALUE = -2.3819763e38


class LingbotVlaV2ArchConfig(msgspec.Struct, frozen=True):
    """Architecture constants of robbyant/lingbot-vla-v2-6b.

    The checkpoint's config.json carries only {"vlm_family": "qwen3_vl"}; these
    values were derived from the released training config and verified against
    every weight shape in the checkpoint.
    """

    expert_num_layers: int = 36
    expert_hidden_size: int = 768
    expert_num_attention_heads: int = 32
    expert_num_key_value_heads: int = 8
    expert_head_dim: int = 128
    expert_rms_norm_eps: float = 1e-6

    moe_num_experts: int = 32
    moe_top_k: int = 4
    moe_intermediate_size: int = 512
    moe_shared_expert_intermediate_size: int = 704
    moe_routed_scaling_factor: float = 4.0

    max_state_dim: int = 55
    max_action_dim: int = 55
    action_horizon: int = 50

    num_task_tokens: int = 8
    align_num_backbone_tokens: int = 256

    time_embedding_min_period: float = 4e-3
    time_embedding_max_period: float = 4.0


def build_qwen3_vl_4b_config() -> Qwen3VLConfig:
    """Qwen3-VL-4B-Instruct config, inlined because the VLA checkpoint ships
    no HF config of its own."""
    return Qwen3VLConfig(
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
        tie_word_embeddings=True,
        text_config={
            "attention_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "max_position_embeddings": 262144,
            "num_attention_heads": 32,
            "num_hidden_layers": 36,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_type": "default",
            },
            "rope_theta": 5000000,
            "tie_word_embeddings": True,
            "vocab_size": 151936,
        },
        vision_config={
            "deepstack_visual_indexes": [5, 11, 17],
            "depth": 24,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 1024,
            "in_channels": 3,
            "intermediate_size": 4096,
            "num_heads": 16,
            "num_position_embeddings": 2304,
            "out_hidden_size": 2560,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    )


def create_sinusoidal_time_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float,
    max_period: float,
) -> torch.Tensor:
    # fp32 on purpose: the reference computes this in fp32, unlike openpi/pi05
    # which uses fp64.
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must have shape [batch]")
    fraction = torch.linspace(
        0.0, 1.0, dimension // 2, dtype=torch.float32, device=time.device
    )
    period = min_period * (max_period / min_period) ** fraction
    scaling = 1.0 / period * 2 * math.pi
    sin_input = scaling[None, :] * time[:, None].to(torch.float32)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def eager_joint_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    att_2d_masks: torch.Tensor,
) -> torch.Tensor:
    """fp32 eager attention over [B, S, H, D] queries and [B, KVH, S, D] K/V."""
    bsize, q_len, num_heads, head_dim = query_states.shape
    num_kv_heads = key_states.shape[1]
    num_groups = num_heads // num_kv_heads

    query = query_states.permute(0, 2, 1, 3)
    key = key_states.repeat_interleave(num_groups, dim=1)
    value = value_states.repeat_interleave(num_groups, dim=1)

    att_weights = torch.matmul(query, key.transpose(-1, -2)) * head_dim**-0.5
    att_weights = torch.where(att_2d_masks[:, None], att_weights, _ATTENTION_MASK_VALUE)
    probs = F.softmax(att_weights, dim=-1).to(value.dtype)
    output = torch.matmul(probs, value)
    return output.permute(0, 2, 1, 3).reshape(bsize, q_len, num_heads * head_dim)


class LingbotAdaRMSNorm(nn.Module):
    """RMSNorm + FiLM conditioning on the flow-matching time embedding."""

    def __init__(self, hidden_size: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(
            torch.float32
        )
        return hidden_states.to(input_dtype)


class LingbotRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LingbotExpertAttention(nn.Module):
    def __init__(self, arch: LingbotVlaV2ArchConfig):
        super().__init__()
        hidden = arch.expert_hidden_size
        head_dim = arch.expert_head_dim
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden, arch.expert_num_attention_heads * head_dim)
        self.k_proj = nn.Linear(hidden, arch.expert_num_key_value_heads * head_dim)
        self.v_proj = nn.Linear(hidden, arch.expert_num_key_value_heads * head_dim)
        self.o_proj = nn.Linear(
            arch.expert_num_attention_heads * head_dim, hidden, bias=False
        )


class LingbotSharedExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LingbotFusedExperts(nn.Module):
    """Per-expert weights stored fused, matching the checkpoint layout
    [num_experts, out_features, in_features]."""

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size)
        )
        self.up_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )


class LingbotMoeMlp(nn.Module):
    """Sigmoid-routed top-k MoE with a loss-free-balance selection bias.

    The correction bias participates only in expert selection; the mixing
    weights use the raw sigmoid scores (DeepSeek-V3 style). At the action
    expert's tiny sequence lengths every expert is computed densely and mixed
    with one-hot weights, mirroring the reference eager path.
    """

    def __init__(self, arch: LingbotVlaV2ArchConfig):
        super().__init__()
        self.top_k = arch.moe_top_k
        self.num_experts = arch.moe_num_experts
        self.routed_scaling_factor = arch.moe_routed_scaling_factor
        self.gate = nn.Linear(arch.expert_hidden_size, arch.moe_num_experts, bias=False)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(arch.moe_num_experts),
            persistent=True,
        )
        self.experts = LingbotFusedExperts(
            num_experts=arch.moe_num_experts,
            hidden_size=arch.expert_hidden_size,
            intermediate_size=arch.moe_intermediate_size,
        )
        self.shared_expert = LingbotSharedExpertMLP(
            hidden_size=arch.expert_hidden_size,
            intermediate_size=arch.moe_shared_expert_intermediate_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsize, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim)

        # fp32 router: bf16 logits can flip top-k selection on near ties.
        router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())
        routing_scores = router_logits.sigmoid()
        scores_for_choice = routing_scores + self.e_score_correction_bias.unsqueeze(0)
        _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1)
        routing_weights = routing_scores.gather(1, selected_experts)
        routing_weights = routing_weights / (
            routing_weights.sum(dim=-1, keepdim=True) + 1e-20
        )
        routing_weights = routing_weights * self.routed_scaling_factor

        gate_out = torch.einsum("th,eih->eti", hidden_flat, self.experts.gate_proj)
        up_out = torch.einsum("th,eih->eti", hidden_flat, self.experts.up_proj)
        expert_out = torch.einsum(
            "eti,ehi->eth", F.silu(gate_out) * up_out, self.experts.down_proj
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        mix_weights = (
            (expert_mask.float() * routing_weights.unsqueeze(-1))
            .sum(dim=1)
            .to(hidden_states.dtype)
        )
        routed = torch.einsum("eth,te->th", expert_out, mix_weights)

        output = routed + self.shared_expert(hidden_flat)
        return output.reshape(bsize, seq_len, hidden_dim)


class LingbotExpertDecoderLayer(nn.Module):
    def __init__(self, arch: LingbotVlaV2ArchConfig):
        super().__init__()
        hidden = arch.expert_hidden_size
        self.self_attn = LingbotExpertAttention(arch)
        self.mlp = LingbotMoeMlp(arch)
        self.input_layernorm = LingbotAdaRMSNorm(
            hidden, hidden, eps=arch.expert_rms_norm_eps
        )
        self.post_attention_layernorm = LingbotAdaRMSNorm(
            hidden, hidden, eps=arch.expert_rms_norm_eps
        )

    def compute_qkv(
        self,
        hidden_states: torch.Tensor,
        ada_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.input_layernorm(hidden_states, ada_cond)
        hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)
        query = self.self_attn.q_proj(hidden_states).view(hidden_shape)
        key = self.self_attn.k_proj(hidden_states).view(hidden_shape)
        value = self.self_attn.v_proj(hidden_states).view(hidden_shape)
        return query, key, value

    def finish(
        self,
        hidden_states: torch.Tensor,
        att_output: torch.Tensor,
        ada_cond: torch.Tensor,
    ) -> torch.Tensor:
        att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
        out = self.self_attn.o_proj(att_output)
        out = out + hidden_states
        after_first_residual = out
        out = self.post_attention_layernorm(out, ada_cond)
        out = self.mlp(out)
        return out + after_first_residual


class LingbotExpertModel(nn.Module):
    def __init__(self, arch: LingbotVlaV2ArchConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            LingbotExpertDecoderLayer(arch) for _ in range(arch.expert_num_layers)
        )
        self.norm = LingbotRMSNorm(
            arch.expert_hidden_size, eps=arch.expert_rms_norm_eps
        )


class LingbotActionExpert(nn.Module):
    """Named ``qwen_expert`` in the checkpoint; wraps the layer stack under
    ``.model`` to mirror the weight tree."""

    def __init__(self, arch: LingbotVlaV2ArchConfig):
        super().__init__()
        self.model = LingbotExpertModel(arch)


class LingbotQwenvlWithExpert(nn.Module):
    """Qwen3-VL prefix plus the action expert, walked layer by layer."""

    def __init__(self, arch: LingbotVlaV2ArchConfig, vlm_config: Qwen3VLConfig):
        super().__init__()
        self.arch = arch
        self.qwenvl = Qwen3VLForConditionalGeneration._from_config(vlm_config)
        del self.qwenvl.lm_head
        self.qwen_expert = LingbotActionExpert(arch)
        if arch.expert_num_layers != vlm_config.text_config.num_hidden_layers:
            raise ValueError(
                "action expert and VLM must have the same number of layers"
            )

    @property
    def _language_model(self) -> nn.Module:
        return self.qwenvl.model.language_model

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._language_model.embed_tokens(tokens)

    def embed_images(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode flattened patches for all cameras in one vision-tower pass.

        Returns merged embeds stacked per image [num_images, tokens, hidden]
        and the deepstack features split the same way.
        """
        visual = self.qwenvl.model.visual
        output = visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // visual.spatial_merge_size**2).tolist()
        # HF >= 5.x: pooler_output is the merged patch stream; last_hidden_state
        # is pre-merger.
        image_embeds = torch.stack(
            list(torch.split(output.pooler_output, split_sizes)), dim=0
        )
        deepstack_embeds = [
            torch.stack(list(torch.split(level, split_sizes)), dim=0)
            for level in output.deepstack_features
        ]
        return image_embeds, deepstack_embeds

    def _vlm_layer_qkv(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = layer.input_layernorm(hidden_states)
        attn = layer.self_attn
        hidden_shape = (*hidden_states.shape[:-1], -1, attn.head_dim)
        query = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape))
        key = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape))
        value = attn.v_proj(hidden_states).view(hidden_shape)
        return query, key, value

    def _vlm_layer_finish(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        att_output: torch.Tensor,
    ) -> torch.Tensor:
        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out = layer.self_attn.o_proj(att_output)
        out = out + hidden_states
        after_first_residual = out
        out = layer.post_attention_layernorm(out)
        out = layer.mlp(out)
        return out + after_first_residual

    def apply_mrope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._language_model.rotary_emb(query_states, position_ids)
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

    @staticmethod
    def _apply_deepstack(
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states[visual_pos_masks] = hidden_states[visual_pos_masks] + (
            visual_embeds.to(hidden_states.dtype)
        )
        return hidden_states

    def forward_prefix(
        self,
        prefix_embs: torch.Tensor,
        att_2d_masks: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        deepstack_visual_embeds: list[torch.Tensor],
    ) -> VLADensePrefixCache:
        """Run the VLM prefix once, filling per-layer roped K/V for reuse."""
        cache = VLADensePrefixCache()
        hidden_states = prefix_embs
        layers = self._language_model.layers
        for layer_idx, layer in enumerate(layers):
            query, key, value = self._vlm_layer_qkv(layer, hidden_states)
            # fp32 joint-attention path, matching the reference numerics.
            query = query.float()
            key = key.float()
            value = value.float()
            query, key = self.apply_mrope(query, key, position_ids)
            cache.update(
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                layer_idx,
            )
            att_output = eager_joint_attention(
                query,
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                att_2d_masks,
            )
            hidden_states = self._vlm_layer_finish(layer, hidden_states, att_output)
            if layer_idx < len(deepstack_visual_embeds):
                hidden_states = self._apply_deepstack(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
        return cache

    def forward_suffix(
        self,
        suffix_embs: torch.Tensor,
        att_2d_masks: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: VLADensePrefixCache,
        ada_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Run the action expert over cached prefix K/V for one denoise step."""
        read_only_cache = VLADensePrefixCache(past_key_values.layers, read_only=True)
        hidden_states = suffix_embs
        for layer_idx, layer in enumerate(self.qwen_expert.model.layers):
            query, key, value = layer.compute_qkv(hidden_states, ada_cond)
            query = query.float()
            key = key.float()
            value = value.float()
            query, key = self.apply_mrope(query, key, position_ids)
            full_key, full_value = read_only_cache.update(
                key.permute(0, 2, 1, 3),
                value.permute(0, 2, 1, 3),
                layer_idx,
            )
            att_output = eager_joint_attention(
                query, full_key, full_value, att_2d_masks
            )
            hidden_states = layer.finish(hidden_states, att_output, ada_cond)
        return self.qwen_expert.model.norm(hidden_states)


class LingbotVlaV2CoreModel(nn.Module):
    """Flow-matching action policy over LingbotQwenvlWithExpert.

    Attribute names (state_proj, action_in_proj, ..., *_align_embs) mirror the
    checkpoint weight tree under its "model." prefix.
    """

    def __init__(
        self,
        arch: LingbotVlaV2ArchConfig,
        vlm_config: Qwen3VLConfig | None = None,
    ):
        super().__init__()
        self.arch = arch
        if vlm_config is None:
            vlm_config = build_qwen3_vl_4b_config()
        vlm_config._attn_implementation = "sdpa"
        vlm_config.text_config._attn_implementation = "sdpa"
        vlm_config.vision_config._attn_implementation = "sdpa"
        self.qwenvl_with_expert = LingbotQwenvlWithExpert(arch, vlm_config)

        hidden = arch.expert_hidden_size
        self.state_proj = nn.Linear(arch.max_state_dim, hidden)
        self.action_in_proj = nn.Linear(arch.max_action_dim, hidden)
        self.action_out_proj = nn.Linear(hidden, arch.max_action_dim)
        self.action_time_mlp_in = nn.Linear(hidden * 2, hidden)
        self.action_time_mlp_out = nn.Linear(hidden, hidden)

        vlm_hidden = vlm_config.text_config.hidden_size
        align_tokens = arch.align_num_backbone_tokens
        self.depth_align_embs = nn.Parameter(torch.zeros(align_tokens, vlm_hidden))
        self.current_video_align_embs = nn.Parameter(
            torch.zeros(align_tokens, vlm_hidden)
        )
        self.future_depth_align_embs = nn.Parameter(
            torch.zeros(align_tokens, vlm_hidden)
        )
        self.future_video_align_embs = nn.Parameter(
            torch.zeros(align_tokens, vlm_hidden)
        )
        self.current_shared_task_proj = nn.Linear(vlm_hidden * 2, vlm_hidden)
        self.future_shared_task_proj = nn.Linear(vlm_hidden * 2, vlm_hidden)

    @property
    def _vlm_config(self):
        return self.qwenvl_with_expert.qwenvl.config

    def _pool_align_tokens(self, embs: torch.Tensor) -> torch.Tensor:
        num_task_tokens = self.arch.num_task_tokens
        return embs.view(num_task_tokens, -1, embs.shape[-1]).mean(dim=1)

    def _align_query_embs(
        self, batch_size: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Current/future distillation query tokens appended after language.

        The released 6b checkpoint uses shared depth+video queries for both the
        current and the future segment (share_future_depth_query=True), so the
        prefix carries exactly 2 * num_task_tokens query tokens.
        """
        current = self.current_shared_task_proj(
            torch.cat(
                [
                    self._pool_align_tokens(self.depth_align_embs),
                    self._pool_align_tokens(self.current_video_align_embs),
                ],
                dim=-1,
            )
        )
        future = self.future_shared_task_proj(
            torch.cat(
                [
                    self._pool_align_tokens(self.future_depth_align_embs),
                    self._pool_align_tokens(self.future_video_align_embs),
                ],
                dim=-1,
            )
        )
        expand = (
            lambda embs: embs.unsqueeze(0).expand(batch_size, -1, -1).to(dtype)
        )  # noqa: E731
        return expand(current), expand(future)

    def embed_prefix(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Build prefix embeddings [images | language | align queries].

        pixel_values: [num_images * patches, patch_dim] flattened patches for
        all cameras (batch size 1). image_grid_thw: [num_images, 3].
        image_masks: [batch, num_images]. lang_tokens/masks: [batch, len].
        """
        bsize, num_images = image_masks.shape
        device = lang_tokens.device
        cfg = self._vlm_config

        img_embs, deepstack_embeds = self.qwenvl_with_expert.embed_images(
            pixel_values, image_grid_thw
        )
        embed_dtype = img_embs.dtype
        num_patch = img_embs.shape[1]
        img_embs = img_embs.reshape(bsize, num_images, num_patch, -1)
        deepstack_embeds = [
            level.reshape(bsize, num_images, num_patch, -1)
            for level in deepstack_embeds
        ]

        # Vision boundary tokens wrap every camera block (66 tokens per camera
        # at 256x256: <vision_start> + 64 patches + <vision_end>).
        embed_token = self.qwenvl_with_expert.embed_language_tokens
        boundary_ids = torch.tensor(
            [cfg.vision_start_token_id, cfg.vision_end_token_id], device=device
        )
        start_emb, end_emb = embed_token(boundary_ids).to(embed_dtype).unbind(0)
        expand_boundary = lambda emb: emb.view(1, 1, 1, -1).expand(  # noqa: E731
            bsize, num_images, 1, -1
        )
        img_blocks = torch.cat(
            [expand_boundary(start_emb), img_embs, expand_boundary(end_emb)], dim=2
        )
        image_token_len = num_patch + 2

        image_pad_masks = image_masks[:, :, None].expand(
            bsize, num_images, image_token_len
        )
        image_visual_masks = torch.zeros_like(image_pad_masks)
        image_visual_masks[:, :, 1 : 1 + num_patch] = image_masks[:, :, None].expand(
            bsize, num_images, num_patch
        )
        fake_image_ids = torch.full(
            (bsize, num_images, image_token_len),
            cfg.image_token_id,
            dtype=torch.long,
            device=device,
        )
        fake_image_ids[:, :, 0] = cfg.vision_start_token_id
        fake_image_ids[:, :, -1] = cfg.vision_end_token_id

        img_flat = img_blocks.reshape(bsize, num_images * image_token_len, -1)
        image_pad_flat = image_pad_masks.reshape(bsize, -1)
        visual_pos_flat = image_visual_masks.reshape(bsize, -1)
        fake_image_flat = fake_image_ids.reshape(bsize, -1)

        lang_embs = embed_token(lang_tokens).to(embed_dtype)

        current_query, future_query = self._align_query_embs(bsize, embed_dtype)
        num_query = current_query.shape[1]
        query_pad = torch.ones(bsize, num_query, dtype=lang_masks.dtype, device=device)
        fake_query_ids = torch.full(
            (bsize, num_query),
            cfg.text_config.eos_token_id,
            dtype=torch.long,
            device=device,
        )

        prefix_embs = torch.cat(
            [img_flat, lang_embs, current_query, future_query], dim=1
        )
        pad_masks = torch.cat(
            [image_pad_flat, lang_masks, query_pad, query_pad], dim=1
        ).to(torch.bool)
        fake_input_ids = torch.cat(
            [fake_image_flat, lang_tokens, fake_query_ids, fake_query_ids], dim=1
        )
        visual_pos_masks = torch.cat(
            [
                visual_pos_flat,
                torch.zeros(
                    bsize,
                    lang_tokens.shape[1] + 2 * num_query,
                    dtype=torch.bool,
                    device=device,
                ),
            ],
            dim=1,
        ).to(torch.bool)

        # Deepstack features exist only for present cameras; align them with
        # the True positions of visual_pos_masks.
        present = image_masks.reshape(-1).to(torch.bool)
        filtered_deepstack = [
            level.reshape(bsize * num_images, num_patch, -1)[present].reshape(
                -1, level.shape[-1]
            )
            for level in deepstack_embeds
        ]

        rope_grid = image_grid_thw[present]
        if rope_grid.numel() == 0:
            rope_grid = image_grid_thw[:1]
        mm_token_type_ids = (fake_input_ids == cfg.image_token_id).int()
        position_ids, _ = self.qwenvl_with_expert.qwenvl.model.get_rope_index(
            input_ids=fake_input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=rope_grid,
            video_grid_thw=None,
            attention_mask=pad_masks.long(),
        )

        return {
            "prefix_embs": prefix_embs,
            "pad_masks": pad_masks,
            "position_ids": position_ids,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": filtered_deepstack,
        }

    def encode_prefix(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[VLADensePrefixCache, torch.Tensor, dict]:
        prefix = self.embed_prefix(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_masks=image_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
        )
        pad_masks = prefix["pad_masks"]
        position_ids = prefix["position_ids"]
        # The checkpoint was trained with vlm_causal=True: every prefix token
        # starts its own attention block, i.e. causal prefix attention.
        att_masks = torch.ones_like(pad_masks)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        cache = self.qwenvl_with_expert.forward_prefix(
            prefix_embs=prefix["prefix_embs"],
            att_2d_masks=att_2d_masks,
            position_ids=position_ids,
            visual_pos_masks=prefix["visual_pos_masks"],
            deepstack_visual_embeds=prefix["deepstack_visual_embeds"],
        )
        valid_positions = position_ids.masked_fill(~pad_masks.unsqueeze(0), 0)
        prefix_offsets = valid_positions.amax(dim=(0, 2)) + 1
        layout = {"prefix_offsets": prefix_offsets}
        return cache, pad_masks, layout

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (time_emb, suffix_embs, suffix_att_masks) for one step."""
        dtype = self.state_proj.weight.dtype
        state_emb = self.state_proj(state.to(dtype))

        time_emb = create_sinusoidal_time_embedding(
            timestep,
            self.arch.expert_hidden_size,
            self.arch.time_embedding_min_period,
            self.arch.time_embedding_max_period,
        ).to(dtype)

        action_emb = self.action_in_proj(noisy_actions.to(dtype))
        time_expanded = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_expanded], dim=-1)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        suffix_embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        # State opens its own attention block; action tokens form one
        # bidirectional block after it.
        att_masks = torch.zeros(
            suffix_embs.shape[:2], dtype=torch.bool, device=suffix_embs.device
        )
        att_masks[:, :2] = True
        return time_emb, suffix_embs, att_masks

    def denoise_step(
        self,
        past_key_values: VLADensePrefixCache,
        prefix_pad_masks: torch.Tensor,
        prefix_offsets: torch.Tensor,
        state: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        time_emb, suffix_embs, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep
        )
        bsize, suffix_len = suffix_embs.shape[:2]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d = prefix_pad_masks[:, None, :].expand(
            bsize, suffix_len, prefix_len
        )
        suffix_pad_masks = torch.ones(
            bsize, suffix_len, dtype=torch.bool, device=suffix_embs.device
        )
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

        suffix_positions = prefix_offsets[:, None] + torch.arange(
            suffix_len, device=suffix_embs.device
        )
        position_ids = suffix_positions.unsqueeze(0).expand(3, -1, -1)

        suffix_out = self.qwenvl_with_expert.forward_suffix(
            suffix_embs=suffix_embs,
            att_2d_masks=full_att_2d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            ada_cond=time_emb,
        )
        suffix_out = suffix_out[:, -self.arch.action_horizon :]
        suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)
