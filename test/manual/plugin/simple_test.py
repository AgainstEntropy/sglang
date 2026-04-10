from sglang.srt.plugins.hook_registry import HookRegistry, HookType, plugin_hook

hook_called = False


@plugin_hook(
    "sglang.srt.utils.common.assert_pkg_version",
    type=HookType.AFTER,
)
def my_after_hook(result, *args, **kwargs):
    global hook_called
    hook_called = True
    print(f"[HOOK] AFTER called! result={result}")


HookRegistry.apply_hooks()

from sglang.srt.utils.common import assert_pkg_version

assert_pkg_version("setuptools", "0.0.1", "test")

print(f"Hook was called: {hook_called}")
