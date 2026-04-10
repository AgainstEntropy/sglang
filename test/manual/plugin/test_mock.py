"""
Unit tests for the SGLang plugin system.

Tests platform discovery, hook registry mechanics, patch propagation,
and plugin loading with environment variable filtering.

All tests are mock-based (no pip install, no GPU) and can run in CPU CI.
"""

import os
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.plugins.hook_registry import (
    HookRegistry,
    HookSource,
    HookType,
    _propagate_patch,
    plugin_hook,
)
from sglang.test.test_utils import CustomTestCase

# ── Dummy platform classes ──────────────────────────────────────────
# Registered in sys.modules so pkgutil.resolve_name() can find them.

_TEST_MOD = "test_plugin_system_dummies"


class DummyDeviceMixin(DeviceMixin):
    _enum = PlatformEnum.OOT
    device_name = "dummy"
    device_type = "cpu"

    def get_device_total_memory(self, device_id=0):
        return 16 * (1 << 30)

    def get_current_memory_usage(self, device=None):
        return 0.0


class DummySRTPlatform(SRTPlatform, DummyDeviceMixin):
    def get_default_attention_backend(self):
        return "dummy_attn"

    def support_cuda_graph(self):
        return False

    def get_dispatch_key_name(self):
        return "dummy"


class SecondDummySRTPlatform(SRTPlatform, DummyDeviceMixin):
    device_name = "dummy2"

    def get_default_attention_backend(self):
        return "dummy2_attn"


sys.modules[_TEST_MOD] = sys.modules[__name__]


# ── Synthetic hook-target module ────────────────────────────────────
# A fake module with functions/classes we can hook without side effects.
# Originals are created via factories so _propagate_patch cannot
# corrupt cross-test state by replacing module-level references.

_TARGETS = "test_plugin_system_targets"
_target_mod = types.ModuleType(_TARGETS)
sys.modules[_TARGETS] = _target_mod


def _make_add():
    def add(a, b):
        return a + b

    return add


def _make_klass():
    class Klass:
        def greet(self, name):
            return f"hello {name}"

    return Klass


# ── Helpers ─────────────────────────────────────────────────────────


def _make_ep(name, load_returns, dist_name="dummy-dist"):
    """Create a mock setuptools entry_point."""
    ep = MagicMock()
    ep.name = name
    ep.value = f"<mock:{name}>"
    ep.dist = MagicMock()
    ep.dist.name = dist_name
    ep.load.return_value = load_returns
    return ep


@contextmanager
def _plugin_env(**kw):
    """Set/unset plugin-related env vars for a test, restoring afterwards."""
    keys = ("SGLANG_PLATFORM", "SGLANG_PLUGINS")
    saved = {k: os.environ.pop(k) for k in keys if k in os.environ}
    for k, v in kw.items():
        os.environ[k] = v
    try:
        yield
    finally:
        for k in kw:
            os.environ.pop(k, None)
        os.environ.update(saved)


def _activate_dummy():
    return f"{_TEST_MOD}.DummySRTPlatform"


def _activate_dummy2():
    return f"{_TEST_MOD}.SecondDummySRTPlatform"


def _activate_none():
    return None


# ═════════════════════════════════════════════════════════════════════
#  Suite 1: Platform Discovery (_resolve_platform)
# ═════════════════════════════════════════════════════════════════════


class TestPlatformDiscovery(CustomTestCase):

    def setUp(self):
        import sglang.srt.platforms as pm

        pm._current_platform = None

    def _resolve(self, eps, **env):
        with (
            patch("importlib.metadata.entry_points", return_value=eps),
            _plugin_env(**env),
        ):
            from sglang.srt.platforms import _resolve_platform

            return _resolve_platform()

    # -- auto-discover path (SGLANG_PLATFORM not set) --

    def test_auto_single_activates(self):
        p = self._resolve([_make_ep("d", _activate_dummy)])
        self.assertIsInstance(p, DummySRTPlatform)
        self.assertTrue(p.is_out_of_tree())
        self.assertEqual(p.device_name, "dummy")

    def test_auto_none_activated_falls_back(self):
        p = self._resolve([_make_ep("d", _activate_none)])
        self.assertIsInstance(p, SRTPlatform)
        self.assertFalse(p.is_out_of_tree())

    def test_auto_no_plugins_falls_back(self):
        p = self._resolve([])
        self.assertIsInstance(p, SRTPlatform)

    def test_auto_multiple_activated_errors(self):
        eps = [
            _make_ep("a", _activate_dummy, "da"),
            _make_ep("b", _activate_dummy2, "db"),
        ]
        with self.assertRaises(RuntimeError) as cm:
            self._resolve(eps)
        self.assertIn("Multiple", str(cm.exception))

    def test_auto_one_none_one_valid(self):
        eps = [
            _make_ep("skip", _activate_none, "d1"),
            _make_ep("ok", _activate_dummy, "d2"),
        ]
        p = self._resolve(eps)
        self.assertIsInstance(p, DummySRTPlatform)

    def test_auto_activate_exception_skipped(self):
        def bad():
            raise RuntimeError("kaboom")

        p = self._resolve([_make_ep("bad", bad)])
        self.assertIsInstance(p, SRTPlatform)

    # -- explicit SGLANG_PLATFORM path --

    def test_explicit_selects_by_name(self):
        p = self._resolve([_make_ep("d", _activate_dummy)], SGLANG_PLATFORM="d")
        self.assertIsInstance(p, DummySRTPlatform)

    def test_explicit_not_found_errors(self):
        with self.assertRaises(RuntimeError) as cm:
            self._resolve(
                [_make_ep("d", _activate_dummy)], SGLANG_PLATFORM="nonexistent"
            )
        self.assertIn("not found", str(cm.exception))

    def test_explicit_activate_none_errors(self):
        with self.assertRaises(RuntimeError) as cm:
            self._resolve([_make_ep("d", _activate_none)], SGLANG_PLATFORM="d")
        self.assertIn("returned None", str(cm.exception))

    def test_explicit_only_loads_selected(self):
        ep_target = _make_ep("target", _activate_dummy, "dt")
        ep_other = _make_ep("other", _activate_dummy2, "do")
        self._resolve([ep_target, ep_other], SGLANG_PLATFORM="target")
        ep_other.load.assert_not_called()

    def test_explicit_activate_exception_propagates(self):
        def bad():
            raise ValueError("hw fail")

        with self.assertRaises(ValueError):
            self._resolve([_make_ep("d", bad)], SGLANG_PLATFORM="d")

    # -- platform class validation --

    def test_load_platform_class_rejects_non_subclass(self):
        from sglang.srt.platforms import _load_platform_class

        with self.assertRaises(TypeError):
            _load_platform_class("builtins.dict")

    # -- identity queries & factory defaults --

    def test_identity_queries(self):
        p = DummySRTPlatform()
        self.assertTrue(p.is_out_of_tree())
        self.assertFalse(p.is_cuda())
        self.assertFalse(p.is_rocm())
        self.assertFalse(p.is_npu())
        self.assertFalse(p.is_xpu())
        self.assertFalse(p.is_cuda_alike())

    def test_base_factory_raises_not_implemented(self):
        p = SRTPlatform()
        for method in (
            p.get_default_attention_backend,
            p.get_graph_runner_cls,
            p.get_mha_kv_pool_cls,
            p.get_mla_kv_pool_cls,
            p.get_nsa_kv_pool_cls,
            p.get_paged_allocator_cls,
            p.get_piecewise_backend_cls,
        ):
            with self.assertRaises(NotImplementedError, msg=method.__name__):
                method()

    def test_base_capability_defaults(self):
        p = SRTPlatform()
        self.assertEqual(p.get_compile_backend(), "inductor")
        self.assertFalse(p.support_cuda_graph())
        self.assertFalse(p.support_piecewise_cuda_graph())
        self.assertFalse(p.supports_fp8())
        self.assertTrue(p.is_pin_memory_available())
        self.assertEqual(p.get_dispatch_key_name(), "native")

    def test_dummy_factory_overrides(self):
        p = DummySRTPlatform()
        self.assertEqual(p.get_default_attention_backend(), "dummy_attn")
        self.assertFalse(p.support_cuda_graph())
        self.assertEqual(p.get_dispatch_key_name(), "dummy")
        self.assertEqual(p.get_device_total_memory(), 16 * (1 << 30))
        self.assertEqual(p.get_current_memory_usage(), 0.0)


# ═════════════════════════════════════════════════════════════════════
#  Suite 2: HookRegistry
# ═════════════════════════════════════════════════════════════════════


class TestHookRegistry(CustomTestCase):

    def setUp(self):
        HookRegistry.reset()
        self._orig_add = _make_add()
        self._orig_klass = _make_klass()
        _target_mod.add = self._orig_add
        _target_mod.Klass = self._orig_klass

    def _t(self, name="add"):
        return f"{_TARGETS}.{name}"

    # -- individual hook types --

    def test_after_modifies_result(self):
        HookRegistry.register(self._t(), lambda r, *a, **k: r * 2, HookType.AFTER)
        HookRegistry.apply_hooks()
        self.assertEqual(_target_mod.add(3, 4), 14)

    def test_after_none_preserves_result(self):
        seen = []
        HookRegistry.register(
            self._t(),
            lambda r, *a, **k: (seen.append(r), None)[-1],
            HookType.AFTER,
        )
        HookRegistry.apply_hooks()
        self.assertEqual(_target_mod.add(1, 2), 3)
        self.assertEqual(seen, [3])

    def test_before_modifies_args(self):
        HookRegistry.register(self._t(), lambda *a, **k: ((10, 20), k), HookType.BEFORE)
        HookRegistry.apply_hooks()
        self.assertEqual(_target_mod.add(0, 0), 30)

    def test_before_none_keeps_args(self):
        HookRegistry.register(self._t(), lambda *a, **k: None, HookType.BEFORE)
        HookRegistry.apply_hooks()
        self.assertEqual(_target_mod.add(5, 6), 11)

    def test_around_wraps(self):
        HookRegistry.register(
            self._t(), lambda orig, *a, **k: -orig(*a, **k), HookType.AROUND
        )
        HookRegistry.apply_hooks()
        self.assertEqual(_target_mod.add(3, 4), -7)

    def test_replace_function(self):
        HookRegistry.register(self._t(), lambda a, b: a * b, HookType.REPLACE)
        HookRegistry.apply_hooks()
        self.assertEqual(_target_mod.add(3, 4), 12)

    def test_replace_class(self):
        OrigKlass = self._orig_klass

        class Extended(OrigKlass):
            def greet(self, name):
                return f"hi {name}!"

        HookRegistry.register(self._t("Klass"), Extended, HookType.REPLACE)
        HookRegistry.apply_hooks()
        self.assertEqual(_target_mod.Klass().greet("w"), "hi w!")
        self.assertIsInstance(_target_mod.Klass(), OrigKlass)

    # -- error cases --

    def test_class_hook_with_non_replace_raises_at_register(self):
        with self.assertRaises(TypeError):
            HookRegistry.register(self._t(), type("X", (), {}), HookType.AROUND)

    def test_non_replace_on_class_target_fails_gracefully(self):
        """apply_hooks catches the TypeError; target is not marked as patched."""
        HookRegistry.register(self._t("Klass"), lambda *a, **k: None, HookType.AROUND)
        HookRegistry.apply_hooks()
        self.assertNotIn(self._t("Klass"), HookRegistry._patched)

    def test_invalid_target_path_handled_gracefully(self):
        HookRegistry.register("no_dots", lambda: None, HookType.AFTER)
        HookRegistry.apply_hooks()
        self.assertNotIn("no_dots", HookRegistry._patched)

    # -- composition: REPLACE applied first, then AROUND, BEFORE, AFTER --

    def test_all_hook_types_composed(self):
        """Verify execution order: BEFORE -> AROUND -> REPLACE -> AROUND -> AFTER."""
        order = []
        t = self._t()

        def replace_fn(a, b):
            order.append("R")
            return a + b + 100

        def around_hook(orig, *a, **k):
            order.append("Ar-pre")
            r = orig(*a, **k)
            order.append("Ar-post")
            return r

        def before_hook(*a, **k):
            order.append("B")
            return None

        def after_hook(result, *a, **k):
            order.append("Af")
            return result + 1

        HookRegistry.register(t, replace_fn, HookType.REPLACE)
        HookRegistry.register(t, around_hook, HookType.AROUND)
        HookRegistry.register(t, before_hook, HookType.BEFORE)
        HookRegistry.register(t, after_hook, HookType.AFTER)
        HookRegistry.apply_hooks()

        result = _target_mod.add(1, 2)
        self.assertEqual(result, 104)  # (1+2+100) + 1
        self.assertEqual(order, ["B", "Ar-pre", "R", "Ar-post", "Af"])

    def test_replace_applied_before_around_regardless_of_order(self):
        """Even if AROUND is registered before REPLACE, REPLACE is applied first."""
        t = self._t()
        HookRegistry.register(
            t, lambda orig, *a, **k: orig(*a, **k) * 10, HookType.AROUND
        )
        HookRegistry.register(t, lambda a, b: a * b, HookType.REPLACE)
        HookRegistry.apply_hooks()

        # REPLACE: 3*4=12, then AROUND: 12*10=120
        self.assertEqual(_target_mod.add(3, 4), 120)

    # -- idempotency & metadata --

    def test_apply_idempotent(self):
        count = []
        HookRegistry.register(
            self._t(), lambda r, *a, **k: (count.append(1), r)[-1], HookType.AFTER
        )
        HookRegistry.apply_hooks()
        HookRegistry.apply_hooks()
        _target_mod.add(1, 2)
        self.assertEqual(len(count), 1)

    def test_wrapped_attr_points_to_original(self):
        HookRegistry.register(self._t(), lambda r, *a, **k: r, HookType.AFTER)
        HookRegistry.apply_hooks()
        self.assertIs(_target_mod.add.__wrapped__, self._orig_add)

    def test_source_tracking(self):
        src = HookSource("myplugin", "mydist")
        HookRegistry.register(
            self._t(), lambda r, *a, **k: r, HookType.AFTER, source=src
        )
        hooks = HookRegistry._hooks[self._t()]
        self.assertEqual(hooks[0][2], src)

    def test_reset_clears_all(self):
        HookRegistry.register(self._t(), lambda r, *a, **k: r, HookType.AFTER)
        HookRegistry.apply_hooks()
        HookRegistry.reset()
        self.assertEqual(len(HookRegistry._hooks), 0)
        self.assertEqual(len(HookRegistry._patched), 0)

    def test_decorator_api(self):
        @plugin_hook(self._t(), type=HookType.AROUND)
        def hook(orig, *a, **k):
            return orig(*a, **k) * 10

        HookRegistry.apply_hooks()
        self.assertEqual(_target_mod.add(2, 3), 50)

    def test_decorator_returns_function_unchanged(self):
        @plugin_hook(self._t(), type=HookType.AFTER)
        def my_hook(result, *a, **k):
            return result

        self.assertTrue(callable(my_hook))
        self.assertEqual(my_hook.__name__, "my_hook")


# ═════════════════════════════════════════════════════════════════════
#  Suite 3: Patch Propagation
# ═════════════════════════════════════════════════════════════════════


class TestPropagatePatch(CustomTestCase):

    def test_replaces_stale_bindings(self):
        original = lambda: "old"
        wrapped = lambda: "new"

        consumer = types.ModuleType("_test_prop_consumer")
        consumer.fn = original
        source = types.ModuleType("_test_prop_source")
        source.fn = wrapped

        sys.modules["_test_prop_consumer"] = consumer
        sys.modules["_test_prop_source"] = source
        try:
            count = _propagate_patch(original, wrapped, source)
            self.assertGreaterEqual(count, 1)
            self.assertIs(consumer.fn, wrapped)
        finally:
            sys.modules.pop("_test_prop_consumer", None)
            sys.modules.pop("_test_prop_source", None)

    def test_skips_source_module(self):
        original = lambda: "old"
        wrapped = lambda: "new"
        source = types.ModuleType("_test_prop_self")
        source.fn = original

        sys.modules["_test_prop_self"] = source
        try:
            _propagate_patch(original, wrapped, source)
            self.assertIs(source.fn, original)
        finally:
            sys.modules.pop("_test_prop_self", None)

    def test_hook_propagates_to_from_importers(self):
        """After apply_hooks, a module that did 'from X import fn' sees the patched version."""
        HookRegistry.reset()
        orig_add = _make_add()
        _target_mod.add = orig_add

        importer = types.ModuleType("_test_importer")
        importer.add = orig_add
        sys.modules["_test_importer"] = importer

        try:
            HookRegistry.register(
                f"{_TARGETS}.add", lambda r, *a, **k: r * 5, HookType.AFTER
            )
            HookRegistry.apply_hooks()
            self.assertEqual(importer.add(2, 3), 25)
        finally:
            sys.modules.pop("_test_importer", None)


# ═════════════════════════════════════════════════════════════════════
#  Suite 4: load_plugins / load_plugins_by_group
# ═════════════════════════════════════════════════════════════════════


class TestLoadPlugins(CustomTestCase):

    def setUp(self):
        HookRegistry.reset()
        self._orig_add = _make_add()
        _target_mod.add = self._orig_add
        import sglang.srt.plugins as pm

        pm._plugins_loaded = False

    def test_by_group_discovers_plugins(self):
        fn = MagicMock()
        ep = _make_ep("p", fn)
        with (
            patch("importlib.metadata.entry_points", return_value=[ep]),
            _plugin_env(),
        ):
            from sglang.srt.plugins import load_plugins_by_group

            result = load_plugins_by_group("sglang.plugins")
        self.assertIn("p", result)
        self.assertIs(result["p"][0], fn)

    def test_sglang_plugins_whitelist(self):
        ep_yes = _make_ep("yes", MagicMock(), "d1")
        ep_no = _make_ep("no", MagicMock(), "d2")
        with (
            patch("importlib.metadata.entry_points", return_value=[ep_yes, ep_no]),
            _plugin_env(SGLANG_PLUGINS="yes"),
        ):
            from sglang.srt.plugins import load_plugins_by_group

            result = load_plugins_by_group("sglang.plugins")
        self.assertIn("yes", result)
        self.assertNotIn("no", result)

    def test_excluded_dists_skipped(self):
        ep = _make_ep("p", MagicMock(), "bad-dist")
        with (
            patch("importlib.metadata.entry_points", return_value=[ep]),
            _plugin_env(),
        ):
            from sglang.srt.plugins import load_plugins_by_group

            result = load_plugins_by_group(
                "sglang.plugins", excluded_dists={"bad-dist"}
            )
        self.assertNotIn("p", result)

    def test_load_plugins_idempotent(self):
        calls = []
        ep = _make_ep("p", lambda: calls.append(1))
        with (
            patch("importlib.metadata.entry_points", return_value=[ep]),
            _plugin_env(),
        ):
            from sglang.srt.plugins import load_plugins

            load_plugins()
            load_plugins()
        self.assertEqual(len(calls), 1)

    def test_load_plugins_applies_hooks(self):
        def register_hook():
            HookRegistry.register(
                f"{_TARGETS}.add", lambda r, *a, **k: r * 3, HookType.AFTER
            )

        ep = _make_ep("h", register_hook)
        with (
            patch("importlib.metadata.entry_points", return_value=[ep]),
            _plugin_env(),
        ):
            from sglang.srt.plugins import load_plugins

            load_plugins()
        self.assertEqual(_target_mod.add(2, 3), 15)

    def test_load_plugins_sets_source_context(self):
        captured = []

        def check():
            from sglang.srt.plugins.hook_registry import _current_plugin_source

            captured.append(_current_plugin_source.get())

        ep = _make_ep("src", check, "my-dist")
        with (
            patch("importlib.metadata.entry_points", return_value=[ep]),
            _plugin_env(),
        ):
            from sglang.srt.plugins import load_plugins

            load_plugins()

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].plugin_name, "src")
        self.assertEqual(captured[0].dist_name, "my-dist")

    def test_get_excluded_dists_with_platform(self):
        ep_sel = _make_ep("sel", MagicMock(), "sel-d")
        ep_oth = _make_ep("oth", MagicMock(), "oth-d")
        with (
            patch("importlib.metadata.entry_points", return_value=[ep_sel, ep_oth]),
            _plugin_env(SGLANG_PLATFORM="sel"),
        ):
            from sglang.srt.plugins import _get_excluded_dists

            self.assertEqual(_get_excluded_dists(), {"oth-d"})

    def test_get_excluded_dists_without_platform(self):
        with _plugin_env():
            from sglang.srt.plugins import _get_excluded_dists

            self.assertEqual(_get_excluded_dists(), set())

    def test_failed_plugin_does_not_block_others(self):
        calls = []

        def good():
            calls.append("good")

        def bad():
            raise RuntimeError("broken plugin")

        ep_bad = _make_ep("bad", bad, "d1")
        ep_good = _make_ep("good", good, "d2")
        with (
            patch("importlib.metadata.entry_points", return_value=[ep_bad, ep_good]),
            _plugin_env(),
        ):
            from sglang.srt.plugins import load_plugins

            load_plugins()
        self.assertEqual(calls, ["good"])


if __name__ == "__main__":
    unittest.main()
