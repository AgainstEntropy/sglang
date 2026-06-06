# SPDX-License-Identifier: Apache-2.0
"""S3 test — SanaWMChunkGenerator: incremental, state-carried streaming.

A tiny CPU model is stepped several times (one chunk per step). The generator must
carry the per-block KV cache + the growing latent across steps and produce finite
chunks — the engine the interactive WASD/IJKL UI drives.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.sana_wm import (
    SanaWMArchConfig,
    SanaWMConfig,
)
from sglang.multimodal_gen.runtime import server_args as _sa_mod
from sglang.multimodal_gen.runtime.models.dits.sana_wm import SanaWMTransformer3DModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.chunk_generator import (
    SanaWMChunkGenerator,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args

MC = 8


class _ZeroCross(torch.nn.Module):
    def forward(self, x, y, mask=None):
        return torch.zeros_like(x)


@pytest.fixture
def _global_args():
    prev = _sa_mod._global_server_args
    set_global_server_args(
        SimpleNamespace(
            comfyui_mode=False,
            enable_cfg_parallel=False,
            enable_torch_compile=False,
            attention_backend=None,
        )
    )
    try:
        yield
    finally:
        set_global_server_args(prev)


def _tiny_model():
    arch = SanaWMArchConfig(
        in_channels=MC, out_channels=MC, num_layers=2,  # GDN-only -> CPU-safe main path
        num_attention_heads=2, attention_head_dim=16, linear_head_dim=16,
        num_cross_attention_heads=2, cross_attention_head_dim=16, cross_attention_dim=32,
        caption_channels=32, model_max_length=8, softmax_every_n=4,
        update_rule="torch_recurrent", cam_update_rule="torch_recurrent", chunk_size=None,
    )
    m = SanaWMTransformer3DModel(SanaWMConfig(arch_config=arch)).double().eval()
    for b in m.blocks:
        b.cross_attn = _ZeroCross()
    return m


def test_chunk_generator_multi_step_carries_state(_global_args):
    m = _tiny_model()
    generator = SanaWMChunkGenerator(
        m,
        denoising_step_list=(1000, 700, 0),
        num_frame_per_block=3,
        cfg_scale=1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    torch.manual_seed(0)
    first_latent = torch.randn(1, MC, 1, 2, 2, dtype=torch.float64)  # VAE-encoded first frame
    prompt = torch.randn(1, 4, 32, dtype=torch.float64)
    generator.reset(first_latent, prompt)

    # Step 1 (chunk 0): condition frame + 2 new frames.
    f1 = generator.step(n_frames=2)
    assert f1.shape == (1, MC, 2, 2, 2)
    assert torch.isfinite(f1).all()
    assert generator.latents.shape[2] == 3  # 1 cond + 2
    assert generator.chunk_idx == 1
    assert generator.kv_cache[0][0][0] is not None  # GDN state stored

    # Step 2 (chunk 1): 3 new frames, carrying the kv-cache.
    f2 = generator.step(n_frames=3)
    assert f2.shape == (1, MC, 3, 2, 2)
    assert torch.isfinite(f2).all()
    assert generator.latents.shape[2] == 6
    assert generator.chunk_idx == 2
    assert len(generator.kv_cache) == 2

    # Step 3: another 3 frames.
    f3 = generator.step(n_frames=3)
    assert generator.latents.shape[2] == 9
    assert torch.isfinite(f3).all()
    # The condition (first) frame is held fixed across the whole generator.
    assert torch.allclose(generator.latents[:, :, 0], first_latent[:, :, 0])


def test_chunk_generator_open_ended_seeded_and_evicts(_global_args):
    """Open-ended mode: no noise_buffer / total_latent_frames.

    Uniform chunk grid, per-chunk noise from the seeded fallback generator
    (reproducible), and stale per-chunk KV entries evicted so an unbounded
    session does not leak the softmax blocks' concat K/V."""
    m = _tiny_model()

    def _run():
        g = SanaWMChunkGenerator(
            m, denoising_step_list=(1000, 700, 0), num_frame_per_block=2,
            num_cached_blocks=2, sink_token=True,
            cfg_scale=1.0, device=torch.device("cpu"), dtype=torch.float64,
        )
        fl = torch.ones(1, MC, 1, 2, 2, dtype=torch.float64)
        pe = torch.zeros(1, 4, 32, dtype=torch.float64)
        g.reset(fl, pe, seed=1234)
        for _ in range(5):
            g.step(n_frames=2)
        return g

    g = _run()
    # Uniform grid: chunk 0 = cond + 2, then +2 per step.
    assert g.latents.shape[2] == 11
    assert g.chunk_indices == [0, 3, 5, 7, 9, 11]
    assert torch.isfinite(g.latents).all()

    # Eviction: only the sink chunk + the last num_cached_blocks chunks keep
    # their cache entries (mirrors accumulate_kv_cache's read window).
    def _has_any(entry):
        return any(slot is not None for block in entry for slot in block)

    kept = [i for i, e in enumerate(g.kv_cache) if _has_any(e)]
    assert kept == [0, 3, 4]

    # Seeded fallback noise is reproducible run-to-run.
    g2 = _run()
    assert torch.equal(g.latents, g2.latents)


def test_chunk_generator_reset_clears_state(_global_args):
    m = _tiny_model()
    generator = SanaWMChunkGenerator(
        m, denoising_step_list=(1000, 0), num_frame_per_block=2,
        cfg_scale=1.0, device=torch.device("cpu"), dtype=torch.float64,
    )
    fl = torch.randn(1, MC, 1, 2, 2, dtype=torch.float64)
    pe = torch.randn(1, 4, 32, dtype=torch.float64)
    generator.reset(fl, pe)
    generator.step(n_frames=2)
    assert generator.chunk_idx == 1
    generator.reset(fl, pe)
    assert generator.chunk_idx == 0 and len(generator.kv_cache) == 0
    assert generator.latents.shape[2] == 1
