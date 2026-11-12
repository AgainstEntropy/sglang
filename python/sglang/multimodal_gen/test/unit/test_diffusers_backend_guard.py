"""The Diffusers backend rejects native-only options instead of ignoring them."""

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.diffusers_generic import (
    DiffusersGenericPipelineConfig,
    DiffusersT2VPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.diffusers_generic import (
    DiffusersGenericSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.pipelines.anima_pipeline import AnimaPipeline
from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import (
    DiffusersPipeline,
)
from sglang.multimodal_gen.runtime.server_args import Backend, ServerArgs
from sglang.multimodal_gen.test.unit.conftest import mock_cuda_platform

_MODEL = "/data/diffusers-model"


def _server_args(pipeline_config=None, **kwargs) -> ServerArgs:
    kwargs = {"model_path": _MODEL, "backend": "diffusers", **kwargs}
    with (
        patch.object(
            PipelineConfig,
            "from_kwargs",
            return_value=pipeline_config or DiffusersGenericPipelineConfig(),
        ),
        mock_cuda_platform(),
    ):
        return ServerArgs.from_kwargs(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "flag"),
    [
        ({"quantization": "fp8"}, "--quantization fp8"),
        ({"quantization_ignored_layers": ["proj"]}, "--quantization-ignored-layers"),
        (
            {"component_quantizations": {"text_encoder": "fp8"}},
            "--text-encoder-quantization",
        ),
        ({"transformer_weights_path": "/w.safetensors"}, "--transformer-weights-path"),
        ({"direct_gpu_weight_loading": True}, "--direct-gpu-weight-loading"),
        (
            {"component_direct_gpu_weight_loading": {"vae": True}},
            "--component-direct-gpu-weight-loading.vae",
        ),
        ({"component_paths": {"vae": "/missing/vae"}}, "--vae-path"),
        (
            {"component_weights_paths": {"transformer": "/w.safetensors"}},
            "--transformer-weights-path",
        ),
        ({"num_gpus": 2, "tp_size": 2}, "--tp-size 2"),
        ({"num_gpus": 2, "sp_degree": 2}, "--sp-degree 2"),
        ({"num_gpus": 2, "ulysses_degree": 2}, "--ulysses-degree 2"),
        ({"num_gpus": 2, "ring_degree": 2}, "--ring-degree 2"),
        ({"use_fsdp_inference": True}, "--use-fsdp-inference"),
        ({"dit_layerwise_offload": True}, "--dit-layerwise-offload"),
        (
            {"layerwise_offload_components": ["text_encoder"]},
            "--layerwise-offload-components",
        ),
        ({"attention_backend": "fa"}, "--attention-backend fa is an SGLang"),
        ({"attention_backend": "sage_attn"}, "--attention-backend sage_attn"),
        ({"attention_backend_config": "sparsity=0.5"}, "--attention-backend-config"),
        (
            {"enable_attention_backend_autotune": True},
            "--enable-attention-backend-autotune",
        ),
        ({"lora_path": "/missing/lora"}, "--lora-path"),
        ({"enable_breakable_cuda_graph": True}, "--enable-breakable-cuda-graph"),
    ],
)
def test_explicit_native_only_flag_is_rejected(kwargs, flag):
    with pytest.raises(ValueError, match="Diffusers backend") as exc_info:
        _server_args(**kwargs)
    assert flag in str(exc_info.value)


@pytest.mark.parametrize(
    ("kwargs", "flag"),
    [
        ({"lora_path": "/missing/lora"}, "--lora-path"),
        (
            {"component_residency": "all=layerwise-offload"},
            "layerwise-offload requires the native SGLang backend",
        ),
    ],
)
def test_auto_fallback_is_guarded_too(kwargs, flag):
    """--backend auto can resolve to diffusers, where backend == DIFFUSERS checks miss."""
    with pytest.raises(ValueError, match="Diffusers backend") as exc_info:
        _server_args(backend="auto", **kwargs)
    assert flag in str(exc_info.value)


def test_task_type_subclass_is_guarded():
    with pytest.raises(ValueError, match="--quantization fp8"):
        _server_args(DiffusersT2VPipelineConfig(), quantization="fp8")


def test_all_offending_flags_are_listed_in_one_error():
    with pytest.raises(ValueError) as exc_info:
        _server_args(quantization="fp8", lora_path="/missing/lora")
    assert "--quantization fp8" in str(exc_info.value)
    assert "--lora-path" in str(exc_info.value)


def test_cache_dit_env_is_rejected(monkeypatch):
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
    with pytest.raises(ValueError, match="use --cache-dit-config"):
        _server_args()


# docs/docs/sglang-diffusion/cache_dit.mdx, "Distributed inference": parallel.yaml.
_DOCS_PARALLEL_YAML = """\
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  ulysses_anything: true
"""
_DOCS_PATH = (
    Path(__file__).resolve().parents[5] / "docs/docs/sglang-diffusion/cache_dit.mdx"
)


def _cache_dit_yaml(tmp_path, text: str) -> str:
    path = tmp_path / "cache_dit.yaml"
    path.write_text(text)
    return str(path)


def test_num_gpus_without_cache_dit_parallelism_names_both_options():
    with pytest.raises(ValueError, match="--num-gpus 4") as exc_info:
        _server_args(num_gpus=4)
    message = str(exc_info.value)
    assert "set parallelism_config in --cache-dit-config" in message
    assert "use --dp-size N with --num-gpus N" in message


def test_num_gpus_with_cache_dit_parallelism_is_accepted(tmp_path):
    """cache-dit shards one diffusers pipeline over all --num-gpus ranks."""
    config = _cache_dit_yaml(tmp_path, _DOCS_PARALLEL_YAML)
    args = _server_args(num_gpus=4, cache_dit_config=config)
    # SGLang's sp group of all ranks fans each request out to every rank.
    assert args.sp_degree == 4


def test_documented_cache_dit_parallel_configs_are_accepted(tmp_path):
    """Every parallelism_config example in the Cache-DiT docs passes the guard."""
    if not _DOCS_PATH.exists():
        pytest.skip(f"{_DOCS_PATH} is not in this checkout")
    blocks = re.findall(r"```yaml[^\n]*\n(.*?)```", _DOCS_PATH.read_text(), re.S)
    parallel_blocks = [b for b in blocks if "parallelism_config:" in b]
    assert _DOCS_PARALLEL_YAML in parallel_blocks
    for block in parallel_blocks:
        assert yaml.safe_load(block)["parallelism_config"]
        # 8 GPUs fit the 3D example (2 x 2 x 2) and every "auto" one.
        _server_args(num_gpus=8, cache_dit_config=_cache_dit_yaml(tmp_path, block))


@pytest.mark.parametrize(
    ("parallelism_config", "kwargs", "error"),
    [
        (
            {"ulysses_size": "auto"},
            {"num_gpus": 4, "dp_size": 2},
            "--dp-size 2: a cache-dit parallelism_config splits one pipeline",
        ),
        (
            {"ulysses_size": 2, "ring_size": 2},
            {"num_gpus": 2},
            "needs a world size of 4",
        ),
        (
            {"ulysses_size": "auto", "tp_size": 2},
            {"num_gpus": 3},
            "needs a world size of a multiple of 2",
        ),
    ],
)
def test_cache_dit_parallelism_must_span_all_gpus(parallelism_config, kwargs, error):
    """cache-dit meshes the whole torch.distributed world, so DP replicas
    or a size product other than --num-gpus would fail inside the workers."""
    config = {"parallelism_config": parallelism_config}
    with pytest.raises(ValueError, match="Diffusers backend") as exc_info:
        _server_args(cache_dit_config=config, **kwargs)
    assert error in str(exc_info.value)


def test_unreadable_cache_dit_config_with_num_gpus_is_rejected(tmp_path):
    missing = str(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match=f"--cache-dit-config {missing}"):
        _server_args(num_gpus=4, cache_dit_config=missing)


def test_cfg_parallel_names_the_flag():
    with pytest.raises(ValueError, match="remove --enable-cfg-parallel"):
        _server_args(num_gpus=4, dp_size=2, enable_cfg_parallel=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"cache_dit_config": "/cache.yaml"},
        {"enable_torch_compile": True},
        {"attention_backend": "flash"},
        {"attention_backend": "aiter"},
        {"dit_cpu_offload": True},
        {"component_residency": "all=component-offload"},
        {"num_gpus": 2, "dp_size": 2},
        # config files mark every key explicit, so default values must pass
        {
            "quantization": None,
            "tp_size": 1,
            "sp_degree": 1,
            "use_fsdp_inference": False,
            "dit_layerwise_offload": False,
            "lora_path": None,
            "enable_breakable_cuda_graph": False,
            "enable_attention_backend_autotune": False,
            "direct_gpu_weight_loading": False,
        },
    ],
)
def test_honored_and_default_flags_are_accepted(kwargs):
    _server_args(**kwargs)


def test_native_backend_is_not_guarded():
    from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
        QwenImagePipelineConfig,
    )

    args = _server_args(
        QwenImagePipelineConfig(), backend="sglang", attention_backend="fa"
    )
    assert args.backend == Backend.SGLANG


def test_explicit_native_pipeline_class_gets_its_own_config():
    """A native --pipeline-class-name runs that pipeline, so a diffusers
    fallback config (any task type) must not stay attached to it."""
    from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
        MiniMaxH3PipelineConfig,
    )

    get_model_info.cache_clear()
    try:
        config = PipelineConfig.from_kwargs(
            {
                "model_path": "h3.safetensors",
                "pipeline_class_name": "MiniMaxH3Pipeline",
                "model_id": "MiniMax-H3",
            }
        )
    finally:
        get_model_info.cache_clear()
    assert isinstance(config, MiniMaxH3PipelineConfig)


def test_deployment_config_disables_native_auto_policies():
    deployment = DiffusersGenericPipelineConfig().get_model_deployment_config()
    assert not deployment.supports_cfg_parallel
    assert not deployment.auto_enable_cfg_parallel
    assert not deployment.auto_auxiliary_layerwise_offload


def _validate_request(params, pipeline_config):
    params._validate_with_server_args(SimpleNamespace(pipeline_config=pipeline_config))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enable_teacache", True),
        ("enable_spectrum", True),
        ("enable_cache_dit", True),
        ("enable_cache_dit", False),
        ("cache_dit_params", {"Fn_compute_blocks": 1}),
        ("attention_backend_override", "fa"),
        ("skip_softmax_params", {"threshold_scale_factor": 1.0}),
    ],
)
def test_native_only_request_field_is_rejected(field, value):
    params = DiffusersGenericSamplingParams(**{field: value})
    params._explicit_fields = {field}
    with pytest.raises(ValueError, match=field):
        _validate_request(params, DiffusersGenericPipelineConfig())


def test_request_defaults_and_unset_fields_are_accepted():
    params = DiffusersGenericSamplingParams(enable_teacache=False)
    params._explicit_fields = {"enable_teacache", "prompt"}
    _validate_request(params, DiffusersGenericPipelineConfig())

    # a field the user did not send never counts
    params = DiffusersGenericSamplingParams(enable_cache_dit=True)
    params._explicit_fields = {"prompt"}
    _validate_request(params, DiffusersGenericPipelineConfig())


def test_native_config_accepts_native_request_fields():
    from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
        ZImagePipelineConfig,
    )

    params = SamplingParams(enable_teacache=True)
    params._explicit_fields = {"enable_teacache"}
    _validate_request(params, ZImagePipelineConfig())


class _Component:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.backend = None

    def set_attention_backend(self, backend):
        if self.error is not None:
            raise self.error
        self.backend = backend


def _attention_args(backend: str, *, explicit: bool):
    return SimpleNamespace(
        attention_backend=backend,
        pipeline_config=DiffusersGenericPipelineConfig(),
        is_arg_explicitly_set=lambda name: explicit and name == "attention_backend",
    )


def _apply(pipe, server_args):
    DiffusersPipeline._apply_attention_backend(None, pipe, server_args)


def test_user_set_attention_backend_failure_raises():
    pipe = SimpleNamespace(transformer=_Component(ValueError("unknown backend")))
    with pytest.raises(ValueError, match="--attention-backend bogus failed"):
        _apply(pipe, _attention_args("bogus", explicit=True))


def test_user_set_attention_backend_without_target_raises():
    with pytest.raises(ValueError, match="no transformer or unet"):
        _apply(SimpleNamespace(), _attention_args("flash", explicit=True))


def test_user_set_shared_name_goes_to_diffusers():
    transformer = _Component()
    _apply(
        SimpleNamespace(transformer=transformer),
        _attention_args("aiter", explicit=True),
    )
    assert transformer.backend == "aiter"


def test_automatic_sglang_default_is_skipped():
    transformer = _Component()
    _apply(
        SimpleNamespace(transformer=transformer),
        _attention_args("aiter", explicit=False),
    )
    assert transformer.backend is None


@pytest.mark.parametrize(
    ("backend", "pipeline_cls"),
    [("auto", DiffusersPipeline), ("sglang", AnimaPipeline)],
)
def test_quantized_name_falls_back_to_diffusers_only_under_auto(
    monkeypatch, tmp_path, backend, pipeline_cls
):
    """A -4bit/-awq/-gptq path must not reroute an explicit --backend sglang."""
    monkeypatch.setattr(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        lambda _: {"_class_name": "AnimaModularPipeline"},
    )
    model_path = tmp_path / "custom-anima-4bit"
    model_path.mkdir()
    get_model_info.cache_clear()
    try:
        info = get_model_info(str(model_path), backend)
    finally:
        get_model_info.cache_clear()
    assert info.pipeline_cls is pipeline_cls


@pytest.mark.parametrize(
    "model_path",
    [
        "/nonexistent/dir/Some-Model",
        "/nonexistent/dir/Some-Model-4bit",
        "./nonexistent/Some-Model-AWQ",
        "/nonexistent/dir/Some-Model-gptq",
    ],
)
def test_missing_local_model_path_is_not_found_under_auto(model_path):
    """A mistyped local path that no registered model name matches, quantized-looking
    names included, must not fall back to diffusers, where the Diffusers-backend
    option check or a repo-id error would hide the real problem."""
    get_model_info.cache_clear()
    try:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            get_model_info(model_path, "auto")
    finally:
        get_model_info.cache_clear()
