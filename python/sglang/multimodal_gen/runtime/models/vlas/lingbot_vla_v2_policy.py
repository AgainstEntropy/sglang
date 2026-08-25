# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_vla_v2 import (
    LingbotVlaV2PipelineConfig,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.models.vlas.lingbot_vla_v2_core import (
    LingbotVlaV2ArchConfig,
    LingbotVlaV2CoreModel,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.vla.observation import (
    VLAObservationBatch,
    tensor_fingerprint,
)
from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLAPrefixCacheManager,
)

logger = init_logger(__name__)

# Checkpoint weights that are deliberately not loaded: the distillation
# projector heads only produce training-time alignment losses. The align query
# embeddings and shared task projections ARE loaded - they shape the prefix.
_IGNORED_SOURCE_KEY_MARKERS = (
    ".depth_align_head.",
    ".future_depth_align_head.",
    ".current_video_align_head.",
    ".future_video_align_head.",
)

_SOURCE_PREFIX = "model."


def _iter_checkpoint_files(model_path: str) -> list[str]:
    files = sorted(
        os.path.join(model_path, name)
        for name in os.listdir(model_path)
        if name.endswith(".safetensors")
    )
    if not files:
        raise FileNotFoundError(f"No .safetensors files under {model_path}")
    return files


class LingbotVlaV2PolicyModel(nn.Module):
    """LingBot-VLA-V2 policy exposing the generic VLA stage contract.

    v1 scope: monolithic single-device execution, batch size 1, no CUDA graph
    and no prefix/action split parallelism; the contract keyword arguments for
    those features are accepted and ignored.
    """

    def __init__(
        self,
        config: LingbotVlaV2PipelineConfig,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.config = config
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.arch = LingbotVlaV2ArchConfig(
            action_horizon=config.action_horizon,
            max_action_dim=config.action_dim,
            max_state_dim=config.state_dim,
            time_embedding_min_period=config.time_embedding_min_period,
            time_embedding_max_period=config.time_embedding_max_period,
        )
        with set_default_torch_dtype(dtype), skip_init_modules():
            self.core_model = LingbotVlaV2CoreModel(self.arch)
        self.core_model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: LingbotVlaV2PipelineConfig,
    ) -> LingbotVlaV2PolicyModel:
        local_path = maybe_download_model(
            model_path,
            force_diffusers_model=False,
            allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"],
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if config.materialize_dtype == "bf16" else torch.float32
        model = cls(
            config=config,
            model_path=local_path,
            device=device,
            dtype=dtype,
        )
        model._load_weights(local_path)
        model.core_model.to(device)
        logger.info(
            "LingbotVlaV2 policy loaded: dtype=%s, device=%s, horizon=%d, dim=%d",
            dtype,
            device,
            config.action_horizon,
            config.action_dim,
        )
        return model

    def _load_weights(self, local_path: str) -> None:
        """Stream checkpoint tensors into the CPU-resident module tree.

        Checkpoint keys equal target keys plus a leading "model." prefix; F32
        source tensors are cast to the target dtype during copy. Any missing or
        unmapped tensor is a hard error - running a robot policy with
        uninitialized weights is unsafe.
        """
        target_state = dict(self.core_model.state_dict())
        loaded: set[str] = set()
        ignored: list[str] = []
        unexpected: list[str] = []
        for file_path in _iter_checkpoint_files(local_path):
            with safe_open(file_path, framework="pt", device="cpu") as reader:
                for source_key in reader.keys():
                    if any(m in source_key for m in _IGNORED_SOURCE_KEY_MARKERS):
                        ignored.append(source_key)
                        continue
                    if not source_key.startswith(_SOURCE_PREFIX):
                        unexpected.append(source_key)
                        continue
                    target_key = source_key[len(_SOURCE_PREFIX) :]
                    target = target_state.get(target_key)
                    if target is None:
                        unexpected.append(source_key)
                        continue
                    tensor = reader.get_tensor(source_key)
                    if tensor.shape != target.shape:
                        raise ValueError(
                            f"Shape mismatch for {source_key}: checkpoint "
                            f"{tuple(tensor.shape)} vs model {tuple(target.shape)}"
                        )
                    with torch.no_grad():
                        target.copy_(tensor)
                    loaded.add(target_key)
        missing = sorted(set(target_state) - loaded)
        if missing:
            raise ValueError(
                f"LingbotVlaV2 checkpoint is missing {len(missing)} weights; "
                f"first missing: {missing[:5]}"
            )
        if unexpected:
            logger.warning(
                "LingbotVlaV2 checkpoint has %d unmapped tensors (first: %s)",
                len(unexpected),
                unexpected[:5],
            )
        logger.info(
            "LingbotVlaV2 weights loaded: %d tensors, %d distillation-head "
            "tensors skipped",
            len(loaded),
            len(ignored),
        )

    # --- generic VLA stage contract ---

    def build_prefix_cache_key(self, observation: VLAObservationBatch) -> str:
        camera_order = tuple(observation.metadata.get("camera_order", ()))
        image_hashes = {
            name: tensor_fingerprint(observation.images[name]) for name in camera_order
        }
        masks = {
            name: bool(mask.item()) for name, mask in observation.image_masks.items()
        }
        model_revision = os.path.basename(os.path.normpath(self.model_path))
        return VLAPrefixCacheManager.make_key(
            model_revision=model_revision,
            tokenizer_id=f"lingbot_vla_v2:{self.config.max_token_len}",
            camera_order=camera_order,
            image_hashes=image_hashes,
            token_digest=tensor_fingerprint(observation.tokens),
            token_mask_digest=tensor_fingerprint(observation.token_masks),
            masks=masks,
            positions_version=self.config.prefix_cache_layout_version,
            dtype=str(self.dtype).replace("torch.", ""),
            parallel_layout_version=self.config.parallel_layout_version,
            cache_namespace="lingbot_vla_v2",
        )

    @torch.no_grad()
    def encode_prefix(
        self,
        observation: VLAObservationBatch,
        *,
        use_cuda_graph: bool = False,
    ) -> PrefixContext:
        if observation.batch_size != 1:
            raise ValueError("LingbotVlaV2 v1 expects one observation per request")
        camera_order = tuple(observation.metadata.get("camera_order", ()))
        grid_by_camera = observation.metadata["image_grid_thw"]

        pixel_values = torch.cat(
            [
                observation.images[name][0].to(self.device, dtype=self.dtype)
                for name in camera_order
            ],
            dim=0,
        )
        image_grid_thw = torch.cat(
            [grid_by_camera[name].to(self.device) for name in camera_order],
            dim=0,
        )
        image_masks = torch.cat(
            [observation.image_masks[name].to(self.device) for name in camera_order],
        ).unsqueeze(0)
        tokens = observation.tokens.to(self.device)
        token_masks = observation.token_masks.to(self.device)

        past_key_values, prefix_pad_masks, layout = self.core_model.encode_prefix(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_masks=image_masks,
            lang_tokens=tokens,
            lang_masks=token_masks,
        )
        return PrefixContext(
            past_key_values=past_key_values,
            prefix_pad_masks=prefix_pad_masks,
            prefix_len=prefix_pad_masks.shape[1],
            layout=layout,
        )

    def sample_noise(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randn(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        observation: VLAObservationBatch,
        prefix_context: PrefixContext,
        *,
        noise: torch.Tensor | None,
        num_steps: int,
        use_cuda_graph: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if observation.state is None:
            raise ValueError("LingbotVlaV2 requires a proprioceptive state vector")
        state = observation.state.to(self.device, dtype=torch.float32)

        x_t = noise
        if x_t is None:
            x_t = self.sample_noise(observation.batch_size, generator=generator)
        else:
            x_t = x_t.to(device=self.device, dtype=torch.float32).clone()

        prefix_offsets = prefix_context.layout["prefix_offsets"]
        dt = -1.0 / num_steps
        timesteps = torch.linspace(
            1.0, 1.0 / num_steps, num_steps, dtype=torch.float32, device=self.device
        )
        for timestep_value in timesteps:
            timestep = timestep_value.expand(observation.batch_size)
            velocity = self.core_model.denoise_step(
                past_key_values=prefix_context.past_key_values,
                prefix_pad_masks=prefix_context.prefix_pad_masks,
                prefix_offsets=prefix_offsets,
                state=state,
                x_t=x_t,
                timestep=timestep,
            )
            x_t.add_(velocity.to(torch.float32), alpha=dt)
        return x_t

    def should_run_action_denoise(
        self,
        prefix_context: PrefixContext | None,
    ) -> bool:
        return True

    def action_parallel_info(
        self,
        prefix_context: PrefixContext | None,
    ) -> dict[str, Any]:
        return {
            "split_group": False,
            "runtime_role": "all",
            "action_sequence_parallel": False,
        }

    def warmup_actions(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            device=self.device,
            dtype=torch.float32,
        )
