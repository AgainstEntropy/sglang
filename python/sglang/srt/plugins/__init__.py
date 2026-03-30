"""
SGLang Unified Plugin Framework.

Supports two types of plugins via setuptools entry_points:
1. Hardware Platform Plugins (sglang.platform_plugins) - register custom hardware platforms
2. General Plugins (sglang.plugins) - inject hooks into functions/methods, replace classes, etc.

Plugins are discovered automatically when installed via pip. Use the SGLANG_PLUGINS
environment variable (comma-separated) to restrict which plugins are loaded.
"""

import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Entry point group names
PLATFORM_PLUGINS_GROUP = "sglang.platform_plugins"
GENERAL_PLUGINS_GROUP = "sglang.plugins"

# Guard against multiple loads in the same process
_plugins_loaded = False


def load_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:
    """
    Discover and load plugins registered under the given entry point group.

    Args:
        group: The setuptools entry_point group name.

    Returns:
        Dictionary mapping plugin name to its loaded callable.
    """
    from importlib.metadata import entry_points

    allowed_plugins_str = os.environ.get("SGLANG_PLUGINS")
    allowed_set = (
        set(allowed_plugins_str.split(",")) if allowed_plugins_str else None
    )

    discovered = entry_points(group=group)
    if len(discovered) == 0:
        logger.debug("No plugins found for group %s.", group)
        return {}

    logger.info("Available plugins for group %s:", group)
    for ep in discovered:
        logger.info("  - %s -> %s", ep.name, ep.value)

    if allowed_set is None:
        logger.debug(
            "All plugins in group %s will be loaded. "
            "Set SGLANG_PLUGINS to control which plugins to load.",
            group,
        )

    plugins: dict[str, Callable[[], Any]] = {}
    for ep in discovered:
        if allowed_set is not None and ep.name not in allowed_set:
            logger.debug("Skipping plugin %s (not in SGLANG_PLUGINS)", ep.name)
            continue
        try:
            func = ep.load()
            plugins[ep.name] = func
            logger.info("Loaded plugin %s from group %s", ep.name, group)
        except Exception:
            logger.exception("Failed to load plugin %s from group %s", ep.name, group)

    return plugins


def load_plugins():
    """
    Load and execute all general plugins, then apply registered hooks.

    Idempotent - safe to call multiple times. General plugins are functions
    whose side effects (registering hooks, replacing classes, etc.) are the
    desired behavior. Return values are ignored.

    After all plugins execute, ``HookRegistry.apply_hooks()`` is called
    automatically so callers only need this single function call.

    This should be called early in every process (main, engine core, workers).
    """
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True

    plugins = load_plugins_by_group(GENERAL_PLUGINS_GROUP)
    for name, func in plugins.items():
        try:
            func()
            logger.info("Executed general plugin: %s", name)
        except Exception:
            logger.exception("Failed to execute general plugin: %s", name)

    # Apply all registered hooks (idempotent — already-patched targets are skipped).
    from sglang.srt.plugins.hook_registry import HookRegistry

    HookRegistry.apply_hooks()
