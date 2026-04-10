"""
Integration tests for the SGLang plugin system.

Uses a real pip-installed dummy platform plugin to test the full lifecycle:
  entry_points discovery → activate() → current_platform resolution
  → factory methods → general plugin hooks → custom op dispatch

Each test runs in a subprocess to guarantee a clean Python process
(no stale singletons or cached imports from the test runner).

Prerequisites:
    pip install -e test/manual/plugin/_dummy_plugin_pkg/
"""

import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.test_utils import CustomTestCase

_PLUGIN_PKG_DIR = os.path.join(os.path.dirname(__file__), "_dummy_plugin_pkg")


def _is_dummy_plugin_installed() -> bool:
    """Check if dummy-platform-plugin is importable."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import dummy_platform_plugin"],
            capture_output=True,
        )
        return r.returncode == 0
    except Exception:
        return False


_SKIP_MSG = (
    "dummy-platform-plugin not installed " f"(run: pip install -e {_PLUGIN_PKG_DIR})"
)


def _run(code, sglang_platform="dummy"):
    """Run a Python snippet in a clean subprocess."""
    env = dict(os.environ)
    if sglang_platform is not None:
        env["SGLANG_PLATFORM"] = sglang_platform
    else:
        env.pop("SGLANG_PLATFORM", None)
    env.pop("SGLANG_PLUGINS", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
    )


# ═════════════════════════════════════════════════════════════════════
#  Suite 1: Platform & general-plugin lifecycle
# ═════════════════════════════════════════════════════════════════════


@unittest.skipUnless(_is_dummy_plugin_installed(), _SKIP_MSG)
class TestPluginIntegration(CustomTestCase):
    """End-to-end tests using a real pip-installed dummy OOT platform plugin."""

    # -- Platform resolution --

    def test_platform_resolves_with_env(self):
        """SGLANG_PLATFORM=dummy selects the dummy platform."""
        r = _run("""
            from sglang.srt.platforms import current_platform
            print(type(current_platform).__name__)
            print(current_platform.device_name)
            print(current_platform.is_out_of_tree())
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.strip().splitlines()
        self.assertEqual(lines[0], "DummySRTPlatform")
        self.assertEqual(lines[1], "dummy")
        self.assertEqual(lines[2], "True")

    def test_auto_discover_finds_dummy(self):
        r = _run(
            """
            from sglang.srt.platforms import current_platform
            print(type(current_platform).__name__)
            """,
            sglang_platform=None,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "DummySRTPlatform")

    def test_wrong_platform_name_errors(self):
        r = _run(
            """
            from sglang.srt.platforms import current_platform
            _ = current_platform.device_name
            """,
            sglang_platform="nonexistent_device",
        )
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not found", r.stderr)

    # -- Identity queries --

    def test_identity_queries(self):
        r = _run("""
            from sglang.srt.platforms import current_platform
            results = [
                current_platform.is_out_of_tree(),
                current_platform.is_cuda(),
                current_platform.is_rocm(),
                current_platform.is_npu(),
                current_platform.is_xpu(),
                current_platform.is_cuda_alike(),
            ]
            for v in results:
                print(v)
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(
            r.stdout.strip().splitlines(),
            ["True", "False", "False", "False", "False", "False"],
        )

    # -- Factory methods & capability flags --

    def test_factory_methods(self):
        r = _run("""
            from sglang.srt.platforms import current_platform
            print(current_platform.get_default_attention_backend())
            print(current_platform.support_cuda_graph())
            print(current_platform.support_piecewise_cuda_graph())
            print(current_platform.supports_fp8())
            print(current_platform.get_compile_backend())
            print(current_platform.get_dispatch_key_name())
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.strip().splitlines()
        self.assertEqual(lines[0], "dummy_attn")
        self.assertEqual(lines[1], "False")
        self.assertEqual(lines[2], "False")
        self.assertEqual(lines[3], "True")
        self.assertEqual(lines[4], "dummy_inductor")
        self.assertEqual(lines[5], "dummy")

    def test_device_memory_queries(self):
        r = _run("""
            from sglang.srt.platforms import current_platform
            print(current_platform.get_device_total_memory())
            print(current_platform.get_current_memory_usage())
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.strip().splitlines()
        self.assertEqual(int(lines[0]), 16 * (1 << 30))
        self.assertEqual(float(lines[1]), 0.0)

    # -- General plugin hooks --

    def test_general_plugin_hook_fires(self):
        r = _run("""
            from sglang.srt.plugins import load_plugins
            load_plugins()

            from sglang.srt.utils.common import assert_pkg_version
            assert_pkg_version("setuptools", "0.0.1", "test")

            import dummy_platform_plugin
            print(len(dummy_platform_plugin.hook_log))
            print(dummy_platform_plugin.hook_log[0][0])
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.strip().splitlines()
        self.assertEqual(lines[0], "1")
        self.assertEqual(lines[1], "assert_pkg_version")

    def test_sglang_plugins_whitelist_blocks(self):
        r = _run("""
            import os
            os.environ["SGLANG_PLUGINS"] = "nonexistent_plugin"

            from sglang.srt.plugins import load_plugins
            load_plugins()

            from sglang.srt.utils.common import assert_pkg_version
            assert_pkg_version("setuptools", "0.0.1", "test")

            import dummy_platform_plugin
            print(len(dummy_platform_plugin.hook_log))
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "0")

    # -- Platform lifecycle hook --

    def test_apply_server_args_defaults_called(self):
        r = _run("""
            from sglang.srt.platforms import current_platform

            class FakeServerArgs:
                pass

            args = FakeServerArgs()
            current_platform.apply_server_args_defaults(args)
            print(getattr(args, "_dummy_plugin_applied", False))
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "True")

    # -- Excluded dists --

    def test_excluded_dists_skips_unselected_general_plugins(self):
        r = _run("""
            from sglang.srt.plugins import load_plugins, _get_excluded_dists
            excluded = _get_excluded_dists()
            print(f"excluded={excluded}")

            load_plugins()
            import dummy_platform_plugin
            print(f"hook_registered={len(dummy_platform_plugin.hook_log) == 0}")
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("hook_registered=True", r.stdout)


# ═════════════════════════════════════════════════════════════════════
#  Suite 2: Model-layer-level replacement (plugin-provided ops)
# ═════════════════════════════════════════════════════════════════════


@unittest.skipUnless(_is_dummy_plugin_installed(), _SKIP_MSG)
class TestModelLayerReplacement(CustomTestCase):
    """Tests for real model-layer replacement via the plugin system.

    The dummy plugin ships:
    - ``DummyActivation(MultiPlatformOp)`` with ``forward_dummy()``
      → auto-discovered by dispatch key name
    - ``silu_and_mul_forward`` registered on ``SiluAndMul``
      → replaces SGLang's real activation forward via register_oot_forward()
    """

    # -- Plugin-provided custom op (forward_<key>() convention) --

    def test_plugin_custom_op_dispatch(self):
        """Plugin's DummyActivation dispatches to forward_dummy() automatically."""
        r = _run("""
            import torch
            from dummy_platform_plugin.ops import DummyActivation

            op = DummyActivation()
            x = torch.tensor([1.0, 2.0, 3.0])
            result = op(x)
            print(f"result={result.tolist()}")
            print(f"dispatched={getattr(op, '_dummy_dispatched', False)}")
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.strip().splitlines()
        self.assertEqual(lines[0], "result=[2.0, 4.0, 6.0]")
        self.assertEqual(lines[1], "dispatched=True")

    # -- Plugin replaces real SGLang SiluAndMul (register_oot_forward) --

    def test_plugin_replaces_real_silu_and_mul(self):
        """load_plugins() registers the plugin's forward for the real SiluAndMul."""
        r = _run("""
            import torch
            from unittest.mock import MagicMock
            import sglang.srt.server_args as _sa
            _sa._global_server_args = MagicMock()
            _sa._global_server_args.rl_on_policy_target = None

            from sglang.srt.plugins import load_plugins
            load_plugins()

            from sglang.srt.layers.activation import SiluAndMul

            op = SiluAndMul()
            x = torch.randn(2, 8)
            result = op(x)
            print(f"shape={list(result.shape)}")
            print(f"dispatched={getattr(op, '_dummy_dispatched', False)}")
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.strip().splitlines()
        self.assertEqual(lines[0], "shape=[2, 4]")
        self.assertEqual(lines[1], "dispatched=True")

    # -- Dispatch fallback (no registration, no forward_<key>()) --

    def test_dispatch_fallback_to_native(self):
        """Without registered forward or forward_<key>(), falls back to forward_native."""
        r = _run("""
            import torch
            from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

            class Unregistered(MultiPlatformOp):
                def forward_native(self, x):
                    return x + 100

            op = Unregistered()
            result = op(torch.tensor([1.0]))
            print(result.item())
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip(), "101.0")

    # -- Class REPLACE on a real SGLang class --

    def test_class_replace_with_propagation(self):
        """REPLACE hook swaps a real SGLang class; propagation updates from-importers."""
        r = _run("""
            from sglang.srt.plugins.hook_registry import HookRegistry, HookType
            from sglang.srt.platforms.interface import SRTPlatform

            original_cls = SRTPlatform

            class CustomSRTPlatform(SRTPlatform):
                custom_marker = True
                def get_default_attention_backend(self):
                    return "custom_attn"

            HookRegistry.register(
                "sglang.srt.platforms.interface.SRTPlatform",
                CustomSRTPlatform,
                HookType.REPLACE,
            )
            HookRegistry.apply_hooks()

            import sglang.srt.platforms.interface as iface
            print(f"replaced={iface.SRTPlatform is CustomSRTPlatform}")
            print(f"has_marker={hasattr(iface.SRTPlatform, 'custom_marker')}")
            print(f"subclass={issubclass(CustomSRTPlatform, original_cls)}")
            print(f"factory={iface.SRTPlatform().get_default_attention_backend()}")
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.strip().splitlines()
        self.assertEqual(lines[0], "replaced=True")
        self.assertEqual(lines[1], "has_marker=True")
        self.assertEqual(lines[2], "subclass=True")
        self.assertEqual(lines[3], "factory=custom_attn")

    # -- Method AROUND hook for instrumentation --

    def test_method_around_hook_profiling(self):
        """AROUND hook instruments a real SGLang function (profiling use case)."""
        r = _run("""
            import time
            from sglang.srt.plugins.hook_registry import HookRegistry, HookType

            timings = []
            def profiler(original_fn, *args, **kwargs):
                start = time.perf_counter()
                result = original_fn(*args, **kwargs)
                timings.append(time.perf_counter() - start)
                return result

            HookRegistry.register(
                "sglang.srt.utils.common.assert_pkg_version",
                profiler,
                HookType.AROUND,
            )
            HookRegistry.apply_hooks()

            from sglang.srt.utils.common import assert_pkg_version
            assert_pkg_version("setuptools", "0.0.1", "test")

            print(f"calls={len(timings)}")
            print(f"timed={timings[0] >= 0}")
        """)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.strip().splitlines()
        self.assertEqual(lines[0], "calls=1")
        self.assertEqual(lines[1], "timed=True")


if __name__ == "__main__":
    unittest.main()
