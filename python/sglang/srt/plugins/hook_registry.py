"""
Hook registry for SGLang plugins.

Provides before/after/around/replace hooks that can be applied to any
function, method, or class in the sglang codebase. Hooks are registered
during plugin loading and applied before the engine starts.

Usage:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def my_timer(original_fn, *args, **kwargs):
        start = time.perf_counter()
        result = original_fn(*args, **kwargs)
        print(f"Elapsed: {time.perf_counter() - start:.3f}s")
        return result

    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.schedule",
        my_timer,
        HookType.AROUND,
    )
"""

import functools
import logging
import pkgutil
from collections import defaultdict
from collections.abc import Callable
from enum import Enum

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Types of hooks that can be applied to functions or classes."""

    BEFORE = "before"  # Execute before original; can modify args
    AFTER = "after"  # Execute after original; can modify return value
    AROUND = "around"  # Wrap original; full control over execution
    REPLACE = "replace"  # Replace the original function or class entirely


class HookRegistry:
    """
    Global registry for function/method/class hooks.

    Thread safety: All registration should happen during load_plugins()
    phase (single-threaded). apply_hooks() should be called once before the
    engine starts serving requests.
    """

    _hooks: dict[str, list[tuple[HookType, Callable]]] = defaultdict(list)
    _patched: set[str] = set()

    @classmethod
    def register(
        cls,
        target: str,
        hook: Callable,
        hook_type: HookType = HookType.AFTER,
    ):
        """
        Register a hook on a target function, method, or class.

        Args:
            target: Fully-qualified dotted path to the target.
                    e.g. "sglang.srt.managers.scheduler.Scheduler.schedule"
                    or   "sglang.srt.managers.scheduler.Scheduler" (class)
            hook: The hook callable (function or class). Signature depends on hook_type:
                - BEFORE:  fn(*args, **kwargs) -> (args, kwargs) or None
                - AFTER:   fn(result, *args, **kwargs) -> new_result or None
                - AROUND:  fn(original_fn, *args, **kwargs) -> result
                - REPLACE: fn(*args, **kwargs) -> result   (function replacement)
                           MyClass                         (class replacement)
            hook_type: Type of hook (default: AFTER).

        Raises:
            TypeError: If a class is passed with a hook_type other than REPLACE.
        """
        if isinstance(hook, type) and hook_type != HookType.REPLACE:
            raise TypeError(
                f"Class {hook.__name__} can only be used with HookType.REPLACE, "
                f"got HookType.{hook_type.name}. "
                f"Use a function for BEFORE/AFTER/AROUND hooks."
            )
        cls._hooks[target].append((hook_type, hook))
        logger.debug(
            "Registered %s hook on %s",
            hook_type.value,
            target,
        )

    @classmethod
    def apply_hooks(cls):
        """
        Apply all registered hooks to their target functions/classes.

        This performs the actual monkey-patching. Should be called once after
        all plugins have been loaded and before the engine starts.
        """
        for target, hooks in cls._hooks.items():
            if target in cls._patched:
                continue
            try:
                cls._apply_target(target, hooks)
                cls._patched.add(target)
            except Exception:
                logger.exception("Failed to apply hooks to %s", target)

    @classmethod
    def _apply_target(cls, target: str, hooks: list):
        """Resolve target, build wrapper chain, and replace the original."""
        parts = target.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid target path (need at least module.attr): {target}")

        obj_path, attr_name = parts
        obj = pkgutil.resolve_name(obj_path)
        original = getattr(obj, attr_name)

        # Guard: if the target is a class, only REPLACE is safe. Wrapping a
        # class in a function would break isinstance/issubclass/inheritance.
        if isinstance(original, type):
            bad = [ht for ht, _ in hooks if ht != HookType.REPLACE]
            if bad:
                raise TypeError(
                    f"Target '{target}' is a class. Only HookType.REPLACE is "
                    f"allowed for class targets (got {bad[0].value}). "
                    f"To hook a method, use '{target}.<method_name>' instead."
                )

        # Build the wrapper chain
        wrapped = original
        for hook_type, hook in hooks:
            if isinstance(hook, type) and hook_type == HookType.REPLACE:
                # Class replacement: direct substitution to preserve type identity.
                # This keeps isinstance(), issubclass(), and inheritance working.
                wrapped = hook
            else:
                wrapped = _wrap_fn(wrapped, hook, hook_type)

        setattr(obj, attr_name, wrapped)
        logger.info("Applied %d hook(s) to %s", len(hooks), target)

    @classmethod
    def reset(cls):
        """Reset all hooks and patches. Primarily for testing."""
        cls._hooks.clear()
        cls._patched.clear()


def _wrap_fn(
    original_fn: Callable, hook: Callable, hook_type: HookType
) -> Callable:
    """Create a wrapper function based on the hook type."""
    if hook_type == HookType.REPLACE:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            return hook(*args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.BEFORE:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            result = hook(*args, **kwargs)
            if result is not None:
                args, kwargs = result
            return original_fn(*args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.AFTER:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            modified = hook(result, *args, **kwargs)
            return modified if modified is not None else result

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.AROUND:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            return hook(original_fn, *args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    else:
        raise ValueError(f"Unknown hook type: {hook_type}")


def plugin_hook(
    target: str,
    type: HookType = HookType.AFTER,
) -> Callable:
    """Decorator that registers a function or class as a hook on *target*.

    Usage::

        # Function hook (AROUND)
        @plugin_hook("sglang.srt.managers.scheduler.Scheduler.schedule",
                      type=HookType.AROUND)
        def my_timer(original_fn, *args, **kwargs):
            start = time.perf_counter()
            result = original_fn(*args, **kwargs)
            print(f"Elapsed: {time.perf_counter() - start:.3f}s")
            return result

        # Class replacement (REPLACE)
        @plugin_hook("sglang.srt.managers.scheduler.Scheduler",
                      type=HookType.REPLACE)
        class MyScheduler(Scheduler):
            ...

    The decorated function/class is returned unchanged so it can still be
    used directly if needed.
    """

    def decorator(hook: Callable) -> Callable:
        HookRegistry.register(target, hook, type)
        return hook

    return decorator
