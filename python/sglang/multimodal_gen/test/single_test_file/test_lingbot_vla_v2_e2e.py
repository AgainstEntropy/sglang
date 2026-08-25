# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import numpy as np
import pytest

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

pytestmark = pytest.mark.skipif(
    os.getenv("SGLANG_RUN_LINGBOT_VLA_V2_E2E") != "1",
    reason="set SGLANG_RUN_LINGBOT_VLA_V2_E2E=1 to run LingBot-VLA-V2 GPU e2e tests",
)

_MODEL_PATH = os.getenv("SGLANG_LINGBOT_VLA_V2_E2E_MODEL", "robbyant/lingbot-vla-v2-6b")
_CAMERA_ORDER = ("camera_top", "camera_wrist_left", "camera_wrist_right")


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _image(camera_index: int) -> np.ndarray:
    height = width = _env_int("SGLANG_LINGBOT_VLA_V2_E2E_IMAGE_SIZE", 256)
    y = np.arange(height, dtype=np.uint16)[:, None]
    x = np.arange(width, dtype=np.uint16)[None, :]
    image = np.stack(
        (
            (x + camera_index * 17) % 256 + np.zeros_like(y),
            (y + camera_index * 29) % 256 + np.zeros_like(x),
            (x + y + camera_index * 41) % 256,
        ),
        axis=-1,
    )
    return image.astype(np.uint8)


def _action_request_kwargs(tag: str) -> dict:
    action_horizon = _env_int("SGLANG_LINGBOT_VLA_V2_E2E_ACTION_HORIZON", 50)
    action_dim = _env_int("SGLANG_LINGBOT_VLA_V2_E2E_ACTION_DIM", 55)
    rng = np.random.default_rng(_env_int("SGLANG_LINGBOT_VLA_V2_E2E_NOISE_SEED", 0))
    prompt = os.getenv("SGLANG_LINGBOT_VLA_V2_E2E_PROMPT", "pick up the blue block")
    return {
        "prompt": f"{prompt} [{tag}]",
        "images": {name: _image(idx) for idx, name in enumerate(_CAMERA_ORDER)},
        "camera_order": list(_CAMERA_ORDER),
        "state": np.linspace(
            -0.5,
            0.5,
            _env_int("SGLANG_LINGBOT_VLA_V2_E2E_STATE_DIM", 55),
            dtype=np.float32,
        ),
        "noise": rng.standard_normal((action_horizon, action_dim)).astype(np.float32),
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "num_inference_steps": _env_int("SGLANG_LINGBOT_VLA_V2_E2E_NUM_STEPS", 10),
        "return_timing": True,
        "enable_prefix_cache": True,
        "enable_cuda_graph": False,
    }


@pytest.fixture(scope="module")
def lingbot_generator():
    generator = DiffGenerator.from_pretrained(
        local_mode=True,
        model_path=_MODEL_PATH,
        num_gpus=1,
        warmup_mode="off",
        trust_remote_code=False,
    )
    try:
        yield generator
    finally:
        generator.shutdown()


def _actions(output: dict) -> np.ndarray:
    return np.asarray(output["actions"], dtype=np.float32)


def _assert_action_output(
    output: dict, *, expect_cache_hit: bool | None = None
) -> None:
    actions = _actions(output)
    assert actions.shape == (
        _env_int("SGLANG_LINGBOT_VLA_V2_E2E_ACTION_HORIZON", 50),
        55,
    )
    assert np.isfinite(actions).all()
    timings = output.get("timings") or {}
    assert timings.get("preprocess_ms", 0.0) >= 0.0
    assert timings.get("prefix_ms", 0.0) >= 0.0
    assert timings.get("action_denoise_ms", 0.0) > 0.0
    cache = output.get("cache") or {}
    if expect_cache_hit is not None:
        assert bool(cache.get("hit")) is expect_cache_hit


def test_lingbot_vla_v2_python_action_e2e(lingbot_generator):
    output = lingbot_generator.generate_action(_action_request_kwargs("e2e"))
    _assert_action_output(output)


def test_lingbot_vla_v2_repeatability(lingbot_generator):
    first = lingbot_generator.generate_action(_action_request_kwargs("repeatability"))
    second = lingbot_generator.generate_action(_action_request_kwargs("repeatability"))
    _assert_action_output(first)
    _assert_action_output(second)
    np.testing.assert_allclose(_actions(first), _actions(second), atol=1e-3)
