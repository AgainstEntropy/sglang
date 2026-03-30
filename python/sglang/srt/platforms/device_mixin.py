"""
Shared device abstraction for SGLang platforms.

DeviceMixin provides the common device identity queries and operations
shared between the SRT (LLM inference) and Multimodal (diffusion)
platform hierarchies.  Concrete per-device mixins (e.g. MyDeviceMixin)
implement the abstract operations; subsystem-specific platforms
(SRTPlatform, MMPlatform) inherit DeviceMixin and add their own methods.

Hierarchy example (OOT plugin)::

    DeviceMixin
    ├── MyDeviceMixin(DeviceMixin)        # vendor-specific device operations
    ├── SRTPlatform(DeviceMixin)          # + graph runner, KV pool, …
    │   └── MySRTPlatform(SRTPlatform, MyDeviceMixin)
    └── MMPlatform(DeviceMixin)           # + attention backend, VAE, …
        └── MyMMPlatform(MMPlatform, MyDeviceMixin)
"""

import enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import torch


class PlatformEnum(enum.Enum):
    """Enumeration of known platform types.

    Superset of both SRT and MM enums so that a single PlatformEnum can
    be shared across subsystems.
    """

    CUDA = enum.auto()
    ROCM = enum.auto()
    CPU = enum.auto()
    XPU = enum.auto()
    MUSA = enum.auto()
    NPU = enum.auto()
    TPU = enum.auto()
    MPS = enum.auto()
    OOT = enum.auto()  # Out-of-tree (external plugin)
    UNSPECIFIED = enum.auto()


class DeviceMixin:
    """Mixin providing device identity queries and basic device operations.

    Class-level attributes (override in subclasses):
        _enum:       PlatformEnum identifying this platform.
        device_name: Human-readable short name (e.g. "cuda", "npu").
        device_type: ``torch.device`` type string (e.g. "cuda", "npu").
    """

    _enum: PlatformEnum = PlatformEnum.UNSPECIFIED
    device_name: str = "unknown"
    device_type: str = "cpu"

    # ------------------------------------------------------------------
    # Platform identity queries
    # ------------------------------------------------------------------

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

    def is_npu(self) -> bool:
        return self._enum == PlatformEnum.NPU

    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    def is_mps(self) -> bool:
        return self._enum == PlatformEnum.MPS

    def is_cuda_alike(self) -> bool:
        """True for CUDA, ROCm, or MUSA (all expose CUDA-like APIs)."""
        return self._enum in (
            PlatformEnum.CUDA,
            PlatformEnum.ROCM,
            PlatformEnum.MUSA,
        )

    def is_out_of_tree(self) -> bool:
        """True for externally-registered OOT platforms."""
        return self._enum == PlatformEnum.OOT

    # ------------------------------------------------------------------
    # Device operations (subclasses must implement)
    # ------------------------------------------------------------------

    def set_device(self, device: "torch.device") -> None:
        """Set the current device."""
        raise NotImplementedError

    def get_device_name(self, device_id: int = 0) -> str:
        """Get human-readable device name."""
        raise NotImplementedError

    def get_device_total_memory(self, device_id: int = 0) -> int:
        """Get total device memory in bytes."""
        raise NotImplementedError

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        """Get current peak memory usage in bytes."""
        raise NotImplementedError

    def get_device_capability(
        self, device_id: int = 0
    ) -> Optional[tuple[int, int]]:
        """Get device compute capability ``(major, minor)``.  None if N/A."""
        raise NotImplementedError

    def get_torch_distributed_backend_str(self) -> str:
        """Return the torch.distributed backend string (e.g. "nccl", "hccl")."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device_name})"
