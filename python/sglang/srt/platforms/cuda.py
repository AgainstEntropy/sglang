"""
CUDA platform implementation.
"""

import torch

from sglang.srt.platforms.interface import Platform, PlatformEnum


class CudaPlatform(Platform):
    """Platform implementation for NVIDIA CUDA GPUs."""

    _enum = PlatformEnum.CUDA
    device_name: str = "cuda"
    device_type: str = "cuda"
    dispatch_key: str = "CUDA"
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    dist_backend: str = "nccl"

    @classmethod
    def set_device(cls, device) -> None:
        torch.cuda.set_device(device)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).total_mem

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        return torch.cuda.max_memory_allocated(device)

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return torch.cuda.get_device_capability(device_id)

    @classmethod
    def get_default_attention_backend(cls) -> str:
        return "flashinfer"

    @classmethod
    def supports_fp8(cls) -> bool:
        cap = cls.get_device_capability()
        if cap is not None:
            return cap[0] >= 9  # Hopper and above
        return False

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return True

    @classmethod
    def support_cuda_graph(cls) -> bool:
        return True

    @classmethod
    def support_cublas(cls) -> bool:
        return True

    @classmethod
    def support_kernel_warmup(cls) -> bool:
        return True

    @classmethod
    def get_dispatch_key_name(cls) -> str:
        return "cuda"
