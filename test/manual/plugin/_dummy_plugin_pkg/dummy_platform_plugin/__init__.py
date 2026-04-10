"""
Dummy OOT platform plugin for SGLang integration tests.

Registers both a platform plugin (sglang.platform_plugins) and a general
plugin (sglang.plugins) to exercise the full plugin lifecycle:
  - Platform activation → DummySRTPlatform
  - General hooks → AFTER hook on assert_pkg_version
  - Custom op forwards → SiluAndMul replacement via register_oot_forward()
"""

import logging

logger = logging.getLogger(__name__)

hook_log: list[tuple[str, tuple]] = []


def activate():
    """Platform plugin entry point. Returns the qualified class name."""
    return "dummy_platform_plugin.platform.DummySRTPlatform"


def register():
    """General plugin entry point.

    Called by ``load_plugins()`` — registers hooks and custom op forwards.
    """
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.utils.common.assert_pkg_version",
        _after_assert_pkg_version,
        HookType.AFTER,
    )

    _register_custom_ops()


def _after_assert_pkg_version(result, *args, **kwargs):
    hook_log.append(("assert_pkg_version", args))
    return result


def _register_custom_ops():
    """Register OOT forward implementations for known SGLang ops."""
    from dummy_platform_plugin.ops import silu_and_mul_forward

    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    try:
        from sglang.srt.layers.activation import SiluAndMul

        MultiPlatformOp.register_oot_forward(SiluAndMul, silu_and_mul_forward, "dummy")
        logger.info("Registered custom SiluAndMul forward for dummy platform")
    except Exception:
        logger.debug("SiluAndMul not available, skipping registration", exc_info=True)
