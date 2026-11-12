# SPDX-License-Identifier: Apache-2.0
"""
Generic pipeline configuration for diffusers backend.

This module provides a minimal pipeline configuration that works with the diffusers backend.
Since diffusers handles its own model loading and configuration, this config is intentionally minimal.
"""

from dataclasses import dataclass, field
from typing import Any

import yaml

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.configs.task_type import ModelTaskType

# Also a diffusers attention backend name, so a user-set value goes to diffusers.
DIFFUSERS_SHARED_ATTENTION_BACKEND_NAMES = frozenset({"aiter"})

_DIFFUSERS_ATTENTION_HINT = (
    "use a diffusers attention backend name such as flash, _flash_3_hub, sage, "
    "xformers or native"
)
_PARALLELISM_HINT = (
    "to split one pipeline across --num-gpus N, set parallelism_config in "
    "--cache-dit-config; to serve N replicas, use --dp-size N with --num-gpus N"
)
# cache-dit's own spellings of the three sizes, with their short aliases.
_CACHE_DIT_PARALLEL_SIZE_KEYS = (
    ("ulysses_size", "ulysses"),
    ("ring_size", "ring"),
    ("tp_size", "tp"),
)
_QUANTIZATION_HINT = (
    "load a pre-quantized diffusers checkpoint or set quantization_config in the "
    "pipeline config, or use --backend sglang"
)
_COMPONENT_PATH_HINT = (
    "the Diffusers backend loads every component from --model-path; point it at "
    "a diffusers repo with the components you want, or use --backend sglang"
)
_LAYERWISE_HINT = (
    "use --component-residency all=component-offload for diffusers model CPU "
    "offload, or use --backend sglang"
)

# Request fields that only native SGLang pipelines read; the Diffusers
# backend would drop them. Values are the alternative named in the 400.
_NATIVE_ONLY_REQUEST_FIELDS = {
    "enable_teacache": "use --cache-dit-config at server start",
    "enable_spectrum": "use --cache-dit-config at server start",
    "enable_cache_dit": "cache-dit is configured server-wide with --cache-dit-config",
    "cache_dit_params": "cache-dit is configured server-wide with --cache-dit-config",
    "attention_backend_override": "set --attention-backend at server start",
    "skip_softmax_params": "use --backend sglang",
}


def sglang_attention_backend_names() -> frozenset[str]:
    """SGLang attention backend names, including the fa3/fa4 aliases."""
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    return frozenset({e.name.lower() for e in AttentionBackendEnum} | {"fa3", "fa4"})


def _is_request_field_set(name: str, value: Any) -> bool:
    if name in ("enable_teacache", "enable_spectrum"):
        return bool(value)
    return value is not None


@dataclass
class DiffusersGenericPipelineConfig(PipelineConfig):
    """
    Generic pipeline configuration for diffusers backend.

    This is a minimal configuration since the diffusers backend handles most
    configuration internally. It provides sensible defaults for the required fields.
    """

    # default to T2I since it's the most common
    task_type: ModelTaskType = ModelTaskType.T2I

    dit_precision: str = "bf16"
    vae_precision: str = "bf16"

    should_use_guidance: bool = True
    embedded_cfg_scale: float = 1.0
    flow_shift: float | None = None
    disable_autocast: bool = True  # let diffusers handle dtype

    # diffusers handles its own loading
    dit_config: DiTConfig = field(default_factory=DiTConfig)
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    image_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (EncoderConfig(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp16",))

    # VAE settings
    vae_tiling: bool = False  # diffusers handles this
    vae_slicing: bool = False  # slice VAE decode for lower memory usage
    vae_sp: bool = False

    # Quantization config for pipeline-level quantization
    # See: https://huggingface.co/docs/diffusers/main/en/quantization/overview
    # Use PipelineQuantizationConfig for component-level control:
    #   from diffusers.quantizers import PipelineQuantizationConfig
    #   quantization_config = PipelineQuantizationConfig(
    #       quant_backend="bitsandbytes_4bit",
    #       quant_kwargs={"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.bfloat16},
    #       components_to_quantize=["transformer", "text_encoder_2"],
    #   )
    quantization_config: Any = None

    def check_pipeline_config(self) -> None:
        """
        Override to skip most validation since diffusers handles its own config.
        """
        pass

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        # Diffusers runs one unsharded pipeline per GPU, and layerwise offload
        # is native-only, so auto-tuning must not pick either.
        return ModelDeploymentConfig(
            supports_cfg_parallel=False,
            auto_enable_cfg_parallel=False,
            auto_auxiliary_layerwise_offload=False,
        )

    # Runs for --backend diffusers and for the registry's auto fallback, where
    # server_args.backend stays auto.
    def validate_server_args(self, server_args: Any) -> None:
        super().validate_server_args(server_args)
        errors = _native_only_server_arg_errors(server_args)
        if errors:
            raise ValueError(
                f"Model {server_args.model_path!r} runs on the Diffusers backend, "
                "which does not support these options:\n  " + "\n  ".join(errors)
            )

    def validate_sampling_params(self, sampling_params: Any) -> None:
        super().validate_sampling_params(sampling_params)
        explicit = getattr(sampling_params, "_explicit_fields", None)
        rejected = [
            f"{name} ({hint})"
            for name, hint in _NATIVE_ONLY_REQUEST_FIELDS.items()
            if (explicit is None or name in explicit)
            and _is_request_field_set(name, getattr(sampling_params, name))
        ]
        if rejected:
            raise ValueError(
                "The Diffusers backend does not support these request fields: "
                + "; ".join(rejected)
            )


# Config files mark every key explicit, so callers also test the value.
def _explicit_value(server_args: Any, name: str) -> Any:
    if server_args.is_arg_explicitly_set(name):
        return getattr(server_args, name)
    return None


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _weight_errors(server_args: Any) -> list[str]:
    errors = []
    if quantization := _explicit_value(server_args, "quantization"):
        errors.append(f"--quantization {quantization}: {_QUANTIZATION_HINT}")
    if _explicit_value(server_args, "quantization_ignored_layers"):
        errors.append(f"--quantization-ignored-layers: {_QUANTIZATION_HINT}")
    if _explicit_value(server_args, "transformer_weights_path"):
        errors.append(f"--transformer-weights-path: {_COMPONENT_PATH_HINT}")
    if _explicit_value(server_args, "direct_gpu_weight_loading"):
        errors.append(
            "--direct-gpu-weight-loading: diffusers loads its own weights; "
            "remove it or use --backend sglang"
        )
    for component, enabled in sorted(
        (
            _explicit_value(server_args, "component_direct_gpu_weight_loading") or {}
        ).items()
    ):
        if enabled:
            errors.append(
                f"--component-direct-gpu-weight-loading.{component}: diffusers "
                "loads its own weights; remove it or use --backend sglang"
            )
    for name, suffix, hint in (
        ("component_quantizations", "quantization", _QUANTIZATION_HINT),
        ("component_paths", "path", _COMPONENT_PATH_HINT),
        ("component_weights_paths", "weights-path", _COMPONENT_PATH_HINT),
    ):
        for component in sorted(_explicit_value(server_args, name) or ()):
            errors.append(f"{_flag(component)}-{suffix}: {hint}")
    return errors


def _load_cache_dit_parallelism_config(cache_dit_config: Any) -> dict | None:
    """The parallelism_config section of --cache-dit-config, or None without one."""
    if isinstance(cache_dit_config, str):
        # cache-dit reads YAML and JSON files with the same yaml.safe_load.
        with open(cache_dit_config) as f:
            cache_dit_config = yaml.safe_load(f)
    if not isinstance(cache_dit_config, dict):
        return None
    section = cache_dit_config.get("parallelism_config")
    return section if isinstance(section, dict) else None


def _cache_dit_parallelism_errors(
    parallelism_config: dict, num_gpus: int, dp_size: int
) -> list[str]:
    # cache-dit builds its device mesh over the whole torch.distributed world,
    # i.e. all --num-gpus ranks, so it cannot share that world with DP replicas.
    if dp_size > 1:
        return [
            f"--dp-size {dp_size}: a cache-dit parallelism_config splits one "
            f"pipeline across all --num-gpus {num_gpus} GPUs; remove --dp-size"
        ]
    sizes = [
        parallelism_config.get(key, parallelism_config.get(alias))
        for key, alias in _CACHE_DIT_PARALLEL_SIZE_KEYS
    ]
    has_auto = False
    fixed = 1
    for size in sizes:
        if isinstance(size, str) and size.lower() == "auto":
            has_auto = True
        elif type(size) is int and size >= 1:
            fixed *= size
        elif size is not None:
            # Malformed values fail in cache_dit.load_configs with its own error.
            return []
    fits = num_gpus % fixed == 0 if has_auto else num_gpus == fixed
    if fits:
        return []
    expected = f"a multiple of {fixed}" if has_auto else str(fixed)
    return [
        f"--num-gpus {num_gpus}: the cache-dit parallelism_config "
        "(ulysses_size x ring_size x tp_size) needs a world size of "
        f"{expected}; cache-dit builds its mesh over all --num-gpus GPUs"
    ]


def _num_gpus_errors(server_args: Any) -> list[str]:
    num_gpus = server_args.num_gpus
    dp_size = server_args.dp_size or 1
    if num_gpus <= 1 or not server_args.cache_dit_config:
        parallelism_config = None
    else:
        try:
            parallelism_config = _load_cache_dit_parallelism_config(
                server_args.cache_dit_config
            )
        except (OSError, yaml.YAMLError) as e:
            return [f"--cache-dit-config {server_args.cache_dit_config}: {e}"]
    if parallelism_config is not None:
        return _cache_dit_parallelism_errors(
            parallelism_config, num_gpus=num_gpus, dp_size=dp_size
        )
    explicit_num_gpus = _explicit_value(server_args, "num_gpus")
    if explicit_num_gpus is not None and explicit_num_gpus > dp_size:
        return [f"--num-gpus {explicit_num_gpus}: {_PARALLELISM_HINT}"]
    return []


def _parallelism_errors(server_args: Any) -> list[str]:
    errors = _num_gpus_errors(server_args)
    for name in (
        "tp_size",
        "sp_degree",
        "ulysses_degree",
        "ring_degree",
        "kv_gather_degree",
    ):
        degree = _explicit_value(server_args, name)
        if degree is not None and degree > 1:
            errors.append(f"{_flag(name)} {degree}: {_PARALLELISM_HINT}")
    if _explicit_value(server_args, "use_fsdp_inference"):
        errors.append(f"--use-fsdp-inference: {_PARALLELISM_HINT}")
    return errors


def _offload_errors(server_args: Any) -> list[str]:
    from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
        resolve_diffusers_pipeline_offload,
    )

    errors = [
        f"{_flag(name)}: {_LAYERWISE_HINT}"
        for name in ("dit_layerwise_offload", "layerwise_offload_components")
        if _explicit_value(server_args, name)
    ]
    if residency := _explicit_value(server_args, "component_residency"):
        try:
            resolve_diffusers_pipeline_offload(residency)
        except ValueError as e:
            errors.append(str(e))
    return errors


def _attention_errors(server_args: Any) -> list[str]:
    errors = []
    backend = _explicit_value(server_args, "attention_backend")
    if (
        backend
        and backend.lower() in sglang_attention_backend_names()
        and backend.lower() not in DIFFUSERS_SHARED_ATTENTION_BACKEND_NAMES
    ):
        errors.append(
            f"--attention-backend {backend} is an SGLang backend name: "
            f"{_DIFFUSERS_ATTENTION_HINT}"
        )
    if _explicit_value(server_args, "attention_backend_config"):
        errors.append(
            "--attention-backend-config configures SGLang attention backends only: "
            f"{_DIFFUSERS_ATTENTION_HINT} with --attention-backend"
        )
    if _explicit_value(server_args, "enable_attention_backend_autotune"):
        errors.append(
            "--enable-attention-backend-autotune times SGLang backends only: "
            f"{_DIFFUSERS_ATTENTION_HINT} with --attention-backend"
        )
    return errors


def _runtime_errors(server_args: Any) -> list[str]:
    errors = []
    if _explicit_value(server_args, "lora_path"):
        errors.append("--lora-path: LoRA needs --backend sglang")
    if _explicit_value(server_args, "enable_breakable_cuda_graph"):
        errors.append(
            "--enable-breakable-cuda-graph captures native DiT forwards only: use "
            "--enable-torch-compile instead"
        )
    if envs.SGLANG_CACHE_DIT_ENABLED:
        errors.append(
            "SGLANG_CACHE_DIT_ENABLED=true drives native pipelines only: use "
            "--cache-dit-config instead"
        )
    return errors


def _native_only_server_arg_errors(server_args: Any) -> list[str]:
    return [
        *_weight_errors(server_args),
        *_parallelism_errors(server_args),
        *_offload_errors(server_args),
        *_attention_errors(server_args),
        *_runtime_errors(server_args),
    ]


# Static subclasses for each non-default task type.
# These exist so that _get_diffusers_model_info() can swap the task_type without
# dynamically creating a class via make_dataclass -- dynamic classes break pickle
# when multiprocessing uses the 'spawn' start method (see #21453).


@dataclass
class DiffusersT2VPipelineConfig(DiffusersGenericPipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2V


@dataclass
class DiffusersI2VPipelineConfig(DiffusersGenericPipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2V


@dataclass
class DiffusersTI2VPipelineConfig(DiffusersGenericPipelineConfig):
    task_type: ModelTaskType = ModelTaskType.TI2V


@dataclass
class DiffusersI2IPipelineConfig(DiffusersGenericPipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2I


@dataclass
class DiffusersTI2IPipelineConfig(DiffusersGenericPipelineConfig):
    task_type: ModelTaskType = ModelTaskType.TI2I


@dataclass
class DiffusersI2MPipelineConfig(DiffusersGenericPipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2M


DIFFUSERS_TASK_TYPE_TO_CONFIG: dict[
    ModelTaskType, type[DiffusersGenericPipelineConfig]
] = {
    ModelTaskType.T2I: DiffusersGenericPipelineConfig,
    ModelTaskType.T2V: DiffusersT2VPipelineConfig,
    ModelTaskType.I2V: DiffusersI2VPipelineConfig,
    ModelTaskType.TI2V: DiffusersTI2VPipelineConfig,
    ModelTaskType.I2I: DiffusersI2IPipelineConfig,
    ModelTaskType.TI2I: DiffusersTI2IPipelineConfig,
    ModelTaskType.I2M: DiffusersI2MPipelineConfig,
}
