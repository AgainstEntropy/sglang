# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as tv2
from transformers import AutoImageProcessor, AutoTokenizer

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_vla_v2 import (
    LingbotVlaV2PipelineConfig,
)
from sglang.multimodal_gen.runtime.vla.observation import VLAObservationBatch

# Qwen3 chat rendering of one user message with neither tools nor a system
# prompt; the checkpoint tokenizer ships no chat template, so it is fixed here.
_PROMPT_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n"


def format_lingbot_prompt(prompt: str) -> str:
    return _PROMPT_TEMPLATE.format(prompt=prompt)


def pad_state_vector(state: torch.Tensor, state_dim: int) -> torch.Tensor:
    """Right-pad a [batch, dim] state to the model's padded state width."""
    if state.shape[-1] > state_dim:
        raise ValueError(
            f"LingbotVlaV2 state dim must be <= {state_dim}, got {state.shape[-1]}"
        )
    if state.shape[-1] == state_dim:
        return state
    return torch.nn.functional.pad(state, (0, state_dim - state.shape[-1]))


def _tensor_from_image(value: Any) -> torch.Tensor:
    """Normalize an input camera image to CHW float32 in [0, 255]."""
    if isinstance(value, Image.Image):
        arr = np.asarray(value.convert("RGB"), dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1)

    if isinstance(value, (np.ndarray, list)):
        value = torch.from_numpy(np.ascontiguousarray(np.asarray(value)))
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Unsupported LingbotVlaV2 image type: {type(value)}")

    tensor = value.detach()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("LingbotVlaV2 v1 expects one observation per request")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got {tensor.shape}")
    if tensor.shape[0] in (1, 3, 4):
        tensor = tensor[:3]
    elif tensor.shape[-1] in (1, 3, 4):
        tensor = tensor[..., :3].permute(2, 0, 1)
    else:
        raise ValueError(f"Could not infer image channels from shape {tensor.shape}")

    is_float_unit_range = tensor.is_floating_point() and tensor.max() <= 2.0
    tensor = tensor.to(torch.float32)
    if is_float_unit_range:
        tensor = tensor * 255.0
    return tensor


class LingbotVlaV2Preprocessor:
    """Turn a raw action-request observation into a VLAObservationBatch.

    Cameras are resized to the training resolution and run through the
    checkpoint's Qwen image processor into flattened patches; the prompt is
    rendered with the fixed Qwen3 chat template and right-padded; the state is
    right-padded to the model's 55-dim action space.
    """

    def __init__(self, config: LingbotVlaV2PipelineConfig, model_path: str):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "right"
        self.image_processor = AutoImageProcessor.from_pretrained(model_path)
        # antialias=True matches the reference deployment's torchvision Resize.
        self._resize = tv2.Resize(config.image_size, antialias=True)

    def _process_camera(self, value: Any) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = self._resize(_tensor_from_image(value))
        processed = self.image_processor(images=tensor, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(torch.float32)
        image_grid_thw = processed["image_grid_thw"].reshape(-1, 3)[:1]
        return pixel_values.unsqueeze(0), image_grid_thw

    def _tokenize(self, prompt: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        rendered = [format_lingbot_prompt(item) for item in prompt]
        encoded = self.tokenizer(
            rendered,
            max_length=self.config.max_token_len,
            padding="max_length",
            padding_side="right",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(torch.long), encoded["attention_mask"].to(
            torch.bool
        )

    def __call__(self, raw_observation: dict[str, Any]) -> VLAObservationBatch:
        prompt_value = raw_observation.get("prompt", "")
        if isinstance(prompt_value, list):
            prompt = [str(x) for x in prompt_value]
        else:
            prompt = [str(prompt_value)]
        if len(prompt) != 1:
            raise ValueError("LingbotVlaV2 v1 expects one prompt per action request")

        raw_images = raw_observation.get("images") or {}
        image_masks_in = raw_observation.get("image_masks") or {}
        camera_order = tuple(
            raw_observation.get("camera_order") or self.config.image_keys
        )

        images: dict[str, torch.Tensor] = {}
        image_masks: dict[str, torch.Tensor] = {}
        grid_by_camera: dict[str, torch.Tensor] = {}
        for key in camera_order:
            value = raw_images.get(key)
            is_present = value is not None and bool(image_masks_in.get(key, True))
            if is_present:
                pixel_values, image_grid_thw = self._process_camera(value)
                images[key] = pixel_values
                grid_by_camera[key] = image_grid_thw
            images.setdefault(key, None)  # placeholder resolved below
            image_masks[key] = torch.tensor([is_present], dtype=torch.bool)

        present = [key for key in camera_order if images[key] is not None]
        if not present:
            raise ValueError("LingbotVlaV2 requires at least one present camera image")
        # Absent cameras still occupy prefix positions (fully masked); give
        # them zero patches with the same grid as a present camera.
        template_key = present[0]
        for key in camera_order:
            if images[key] is None:
                images[key] = torch.zeros_like(images[template_key])
                grid_by_camera[key] = grid_by_camera[template_key].clone()

        state = raw_observation.get("state")
        if state is None:
            raise ValueError("LingbotVlaV2 requires observation.state")
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if state_tensor.shape[0] != 1:
            raise ValueError("LingbotVlaV2 v1 expects one state vector per request")
        state_tensor = pad_state_vector(state_tensor, self.config.state_dim)

        noise = raw_observation.get("noise")
        noise_tensor = None
        if noise is not None:
            noise_tensor = torch.as_tensor(noise, dtype=torch.float32)
            if noise_tensor.ndim == 2:
                noise_tensor = noise_tensor.unsqueeze(0)
            expected = (1, self.config.action_horizon, self.config.action_dim)
            if tuple(noise_tensor.shape) != expected:
                raise ValueError(
                    f"LingbotVlaV2 noise must have shape {expected}, "
                    f"got {tuple(noise_tensor.shape)}"
                )

        tokens = raw_observation.get("tokens")
        token_masks = raw_observation.get("token_masks")
        if tokens is not None:
            tokens_tensor = torch.as_tensor(tokens, dtype=torch.long)
            if tokens_tensor.ndim == 1:
                tokens_tensor = tokens_tensor.unsqueeze(0)
            if token_masks is None:
                token_masks_tensor = tokens_tensor != self.tokenizer.pad_token_id
            else:
                token_masks_tensor = torch.as_tensor(token_masks, dtype=torch.bool)
                if token_masks_tensor.ndim == 1:
                    token_masks_tensor = token_masks_tensor.unsqueeze(0)
        else:
            tokens_tensor, token_masks_tensor = self._tokenize(prompt)

        return VLAObservationBatch(
            prompt=prompt,
            images=images,
            image_masks=image_masks,
            state=state_tensor,
            noise=noise_tensor,
            tokens=tokens_tensor,
            token_masks=token_masks_tensor,
            batch_size=1,
            metadata={
                "camera_order": camera_order,
                "image_grid_thw": grid_by_camera,
            },
        )
