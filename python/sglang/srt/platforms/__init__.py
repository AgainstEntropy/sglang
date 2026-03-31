"""
SGLang Platform Discovery and Lazy Initialization.

Provides `current_platform` as a module-level lazy singleton. On first access,
it discovers platform plugins via entry_points and instantiates the appropriate
SRTPlatform subclass.

Usage:
    from sglang.srt.platforms import current_platform
    print(current_platform.device_name)
"""

import logging
import os
from typing import Optional

from sglang.srt.platforms.interface import SRTPlatform

logger = logging.getLogger(__name__)

_current_platform: Optional[SRTPlatform] = None


def _resolve_platform() -> SRTPlatform:
    """
    Discover and instantiate the active platform.

    Discovery sources (in priority order):
    1. SGLANG_PLATFORM env var (force a specific platform class, for dev/testing)
    2. entry_points in group "sglang.platform_plugins" (OOT plugins)
    3. Fallback: base SRTPlatform with safe defaults
    """
    # 1. Optional: Force a specific platform via env var
    if forced := os.environ.get("SGLANG_PLATFORM"):
        logger.info("SGLANG_PLATFORM override: %s", forced)
        return _load_platform_class(forced)()

    # 2. Discover OOT platform plugins via entry_points
    from sglang.srt.plugins import PLATFORM_PLUGINS_GROUP, load_plugins_by_group

    oot_plugins = load_plugins_by_group(PLATFORM_PLUGINS_GROUP)
    oot_platform_cls = None
    oot_name = None

    for name, plugin_fn in oot_plugins.items():
        try:
            result = plugin_fn()
            if result is not None:
                if oot_platform_cls is not None:
                    raise RuntimeError(
                        f"Multiple OOT platform plugins activated: {oot_name!r} and {name!r}. "
                        f"Set SGLANG_PLUGINS to select exactly one."
                    )
                oot_platform_cls = _load_platform_class(result)
                oot_name = name
                logger.info(
                    "OOT platform plugin activated: %s -> %s", name, result
                )
        except Exception:
            logger.exception("Failed to activate platform plugin: %s", name)

    if oot_platform_cls is not None:
        return oot_platform_cls()

    # 3. Fallback: base SRTPlatform
    logger.warning("No platform detected. Using base SRTPlatform with defaults.")
    return SRTPlatform()


def _load_platform_class(qualname: str) -> type:
    """Load an SRTPlatform subclass from its fully-qualified class name."""
    import pkgutil

    cls = pkgutil.resolve_name(qualname)
    if not isinstance(cls, type) or not issubclass(cls, SRTPlatform):
        raise TypeError(
            f"Expected an SRTPlatform subclass, got {type(cls)}: {qualname}"
        )
    return cls


def __getattr__(name: str):
    """Lazy initialization of current_platform on first access."""
    if name == "current_platform":
        global _current_platform
        if _current_platform is None:
            _current_platform = _resolve_platform()
        return _current_platform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
