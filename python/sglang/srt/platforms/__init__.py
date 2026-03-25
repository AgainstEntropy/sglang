"""
SGLang Platform Discovery and Lazy Initialization.

Provides `current_platform` as a module-level lazy singleton. On first access,
it discovers platform plugins (OOT > builtin) and instantiates the appropriate
Platform subclass.

Usage:
    from sglang.srt.platforms import current_platform
    print(current_platform.device_name)
"""

import logging
from typing import Optional

from sglang.srt.platforms.interface import Platform

logger = logging.getLogger(__name__)

_current_platform: Optional[Platform] = None


def _resolve_platform() -> Platform:
    """
    Discover and instantiate the active platform.

    Priority: OOT plugins > builtin detection.
    At most one OOT plugin and one builtin may activate.
    OOT takes precedence over builtin.
    """
    from sglang.srt.plugins import load_plugins_by_group, PLATFORM_PLUGINS_GROUP

    # Phase 1: Check for OOT platform plugins
    oot_plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)
    oot_platform_cls = None
    oot_name = None

    for name, plugin_fn in oot_plugins.items():
        try:
            result = plugin_fn()
            if result is not None:
                if oot_platform_cls is not None:
                    logger.warning(
                        "Multiple OOT platform plugins activated: %s and %s. "
                        "Using the first one (%s).",
                        oot_name,
                        name,
                        oot_name,
                    )
                    continue
                oot_platform_cls = _load_platform_class(result)
                oot_name = name
                logger.info("OOT platform plugin activated: %s -> %s", name, result)
        except Exception:
            logger.exception("Failed to activate platform plugin: %s", name)

    if oot_platform_cls is not None:
        return oot_platform_cls()

    # Phase 2: Builtin platform detection
    for name, detect_fn in _builtin_platform_detectors().items():
        try:
            cls_path = detect_fn()
            if cls_path is not None:
                platform_cls = _load_platform_class(cls_path)
                logger.info("Builtin platform detected: %s -> %s", name, cls_path)
                return platform_cls()
        except Exception:
            logger.debug("Builtin platform detection failed for %s", name, exc_info=True)

    # Fallback: return base Platform
    logger.warning("No platform detected. Using base Platform with defaults.")
    return Platform()


def _load_platform_class(qualname: str) -> type:
    """Load a Platform subclass from its fully-qualified class name."""
    from sglang.srt.plugins.hook_registry import resolve_obj

    cls = resolve_obj(qualname)
    if not isinstance(cls, type) or not issubclass(cls, Platform):
        raise TypeError(
            f"Expected a Platform subclass, got {type(cls)}: {qualname}"
        )
    return cls


def _builtin_platform_detectors() -> dict:
    """
    Return ordered dict of builtin platform detection functions.

    Each function returns a Platform class qualname string if the platform
    is available, or None otherwise.
    """
    return {
        "cuda": _detect_cuda,
        "rocm": _detect_rocm,
        "cpu": _detect_cpu,
    }


def _detect_cuda() -> Optional[str]:
    """Detect NVIDIA CUDA platform."""
    try:
        import torch

        if torch.cuda.is_available() and not hasattr(torch.version, "hip"):
            return "sglang.srt.platforms.cuda.CudaPlatform"
    except ImportError:
        pass
    return None


def _detect_rocm() -> Optional[str]:
    """Detect AMD ROCm platform."""
    try:
        import torch

        if torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip:
            return "sglang.srt.platforms.rocm.RocmPlatform"
    except ImportError:
        pass
    return None


def _detect_cpu() -> Optional[str]:
    """Detect CPU-only platform."""
    import os

    if os.environ.get("SGLANG_USE_CPU_ENGINE", "0") == "1":
        return "sglang.srt.platforms.cpu.CpuPlatform"
    return None


def __getattr__(name: str):
    """Lazy initialization of current_platform on first access."""
    if name == "current_platform":
        global _current_platform
        if _current_platform is None:
            _current_platform = _resolve_platform()
        return _current_platform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
