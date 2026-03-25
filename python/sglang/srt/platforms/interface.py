"""
SGLang Hardware Platform Abstraction.

Defines the Platform base class and PlatformEnum. Each hardware backend
(CUDA, ROCm, NPU, XPU, etc.) implements a Platform subclass providing
device operations, configuration hooks, and subsystem factory methods.

Out-of-tree platforms register via setuptools entry_points under the
"sglang.platform_plugins" group.
"""

import enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


class PlatformEnum(enum.Enum):
    """Enumeration of known platform types."""

    CUDA = enum.auto()
    ROCM = enum.auto()
    CPU = enum.auto()
    XPU = enum.auto()
    MUSA = enum.auto()
    OOT = enum.auto()  # Out-of-tree (external plugin)
    UNSPECIFIED = enum.auto()


class Platform:
    """
    Abstract base class for hardware platform backends.

    Provides sensible defaults for all methods. Hardware plugins override
    the methods relevant to their platform.

    Class-level attributes:
        _enum: PlatformEnum identifying this platform.
        device_name: Short name (e.g. "cuda", "npu", "xpu").
        device_type: Torch device type string.
        dispatch_key: PyTorch dispatch key (e.g. "CUDA", "PrivateUse1").
        device_control_env_var: Env var controlling device visibility.
        supported_quantization: List of supported quantization method names.
        dist_backend: Distributed backend name (e.g. "nccl", "hccl", "gloo").
    """

    _enum: PlatformEnum = PlatformEnum.UNSPECIFIED
    device_name: str = "unknown"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"
    device_control_env_var: str = ""
    supported_quantization: list[str] = []
    dist_backend: str = "gloo"

    # ---- Identity queries ----

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    def is_musa(self) -> bool:
        return self._enum == PlatformEnum.MUSA

    def is_cuda_alike(self) -> bool:
        """Returns True for CUDA or ROCm (both support CUDA-like APIs)."""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    def is_out_of_tree(self) -> bool:
        """Returns True for externally-registered OOT platforms."""
        return self._enum == PlatformEnum.OOT

    # ---- Device operations ----

    @classmethod
    def set_device(cls, device: "torch.device") -> None:
        """Set the current device."""
        pass

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get human-readable device name."""
        return "unknown"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total device memory in bytes."""
        return 0

    @classmethod
    def get_current_memory_usage(cls, device: Optional["torch.device"] = None) -> float:
        """Get current peak memory usage in bytes."""
        return 0.0

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> Optional[tuple[int, int]]:
        """Get device compute capability (major, minor). None if N/A."""
        return None

    # ---- Configuration lifecycle ----

    @classmethod
    def pre_register_and_update(cls) -> None:
        """
        Phase 1: Called early at process startup.

        Use this to register quantization methods, attention backends,
        custom CLI arguments, and apply global monkey patches.
        """
        pass

    @classmethod
    def apply_server_args_defaults(cls, server_args) -> None:
        """
        Phase 2: Called after ServerArgs is parsed.

        Apply platform-specific default values to server arguments.
        """
        pass

    @classmethod
    def check_and_update_config(cls, server_args) -> None:
        """
        Phase 3: Validate and fix configuration.

        Raise on incompatible configurations.
        """
        pass

    # ---- Subsystem factory methods ----

    @classmethod
    def get_default_attention_backend(cls) -> str:
        """Return the default attention backend name for this platform."""
        return "flashinfer"

    @classmethod
    def get_graph_runner_cls(cls) -> type:
        """Return the graph runner class for this platform."""
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

        return CudaGraphRunner

    @classmethod
    def get_mha_kv_pool_cls(cls) -> type:
        """Return the MHA KV pool class for this platform."""
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool

    @classmethod
    def get_mla_kv_pool_cls(cls) -> type:
        """Return the MLA KV pool class for this platform."""
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        return MLATokenToKVPool

    @classmethod
    def get_nsa_kv_pool_cls(cls) -> type:
        """Return the NSA KV pool class for this platform (DeepSeek V3.2)."""
        from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

        return NSATokenToKVPool

    @classmethod
    def get_paged_allocator_cls(cls) -> type:
        """Return the paged allocator class for this platform."""
        from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator

        return PagedTokenToKVPoolAllocator

    @classmethod
    def get_default_lora_backend(cls) -> str:
        """Return the default LoRA backend name for this platform."""
        return "triton"

    @classmethod
    def get_compile_backend(cls, mode: str | None = None) -> str:
        """Return the compilation backend identifier (or callable).

        ``mode`` is an optional hint for the platform (e.g. "npugraph_ex").
        The default implementation returns ``"inductor"``.
        """
        return "inductor"

    @classmethod
    def get_piecewise_backend_cls(cls) -> type:
        """Return the piecewise compilation backend class for this platform."""
        from sglang.srt.compilation.cuda_piecewise_backend import CUDAPiecewiseBackend

        return CUDAPiecewiseBackend

    @classmethod
    def support_torch_compile(cls) -> bool:
        """Whether this platform supports torch.compile."""
        return True

    # ---- Capability flags ----

    @classmethod
    def supports_fp8(cls) -> bool:
        """Whether this platform supports FP8 quantization."""
        return False

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Whether pinned memory is available on this platform."""
        return True

    @classmethod
    def support_cuda_graph(cls) -> bool:
        """Whether this platform supports CUDA graph capture."""
        return True

    @classmethod
    def support_cublas(cls) -> bool:
        """Whether this platform supports cuBLAS initialization."""
        return False

    @classmethod
    def support_kernel_warmup(cls) -> bool:
        """Whether this platform should run kernel warmup."""
        return False

    # ---- Initialization and patching ----

    @classmethod
    def init_backend(cls) -> None:
        """One-time backend initialization. Called in each worker."""
        pass

    @classmethod
    def apply_global_patches(cls) -> None:
        """Apply monkey patches needed at process startup (phase 1)."""
        pass

    @classmethod
    def apply_worker_patches(cls) -> None:
        """Apply monkey patches needed in each worker (phase 2)."""
        pass

    # ---- MultiPlatformOp integration ----

    @classmethod
    def get_dispatch_key_name(cls) -> str:
        """
        Return the dispatch key name for MultiPlatformOp.

        This determines which forward_<key>() method is selected.
        E.g. "cuda", "npu", "hip", "xpu", "cpu".
        """
        return "native"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device_name})"
