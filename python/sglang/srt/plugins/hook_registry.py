"""
Function-level hook registry for SGLang plugins.

Provides before/after/around/replace hooks that can be applied to any
function or method in the sglang codebase. Hooks are registered during
plugin loading and applied before the engine starts.

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
from collections import defaultdict
from collections.abc import Callable
from enum import Enum

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Types of hooks that can be applied to functions."""

    BEFORE = "before"  # Execute before original; can modify args
    AFTER = "after"  # Execute after original; can modify return value
    AROUND = "around"  # Wrap original; full control over execution
    REPLACE = "replace"  # Completely replace the original function


class HookRegistry:
    """
    Global registry for function/method hooks.

    Thread safety: All registration should happen during load_general_plugins()
    phase (single-threaded). apply_hooks() should be called once before the
    engine starts serving requests.
    """

    _hooks: dict[str, list[tuple[HookType, Callable, int]]] = defaultdict(list)
    _patched: set[str] = set()

    @classmethod
    def register(
        cls,
        target: str,
        hook_fn: Callable,
        hook_type: HookType = HookType.AFTER,
        priority: int = 100,
    ):
        """
        Register a hook on a target function.

        Args:
            target: Fully-qualified dotted path to the target function/method.
                    e.g. "sglang.srt.managers.scheduler.Scheduler.schedule"
            hook_fn: The hook function. Signature depends on hook_type:
                - BEFORE:  fn(*args, **kwargs) -> (args, kwargs) or None
                - AFTER:   fn(result, *args, **kwargs) -> new_result or None
                - AROUND:  fn(original_fn, *args, **kwargs) -> result
                - REPLACE: fn(*args, **kwargs) -> result
            hook_type: Type of hook (default: AFTER).
            priority: Lower numbers execute first (default: 100).
        """
        cls._hooks[target].append((hook_type, hook_fn, priority))
        cls._hooks[target].sort(key=lambda x: x[2])
        logger.debug(
            "Registered %s hook on %s (priority=%d)",
            hook_type.value,
            target,
            priority,
        )

    @classmethod
    def apply_hooks(cls):
        """
        Apply all registered hooks to their target functions.

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
        obj = resolve_obj(obj_path)
        original_fn = getattr(obj, attr_name)

        # Build the wrapper chain
        wrapped = original_fn
        for hook_type, hook_fn, _ in hooks:
            wrapped = _wrap_fn(wrapped, hook_fn, hook_type)

        setattr(obj, attr_name, wrapped)
        logger.info("Applied %d hook(s) to %s", len(hooks), target)

    @classmethod
    def reset(cls):
        """Reset all hooks and patches. Primarily for testing."""
        cls._hooks.clear()
        cls._patched.clear()


def resolve_obj(qualname: str):
    """
    Resolve a dotted qualname to a Python object.

    Supports nested classes: "sglang.srt.managers.scheduler.Scheduler"
    resolves to the Scheduler class in the scheduler module.
    """
    import importlib

    parts = qualname.split(".")
    # Try progressively shorter module paths
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            obj = importlib.import_module(module_path)
            for attr in parts[i:]:
                obj = getattr(obj, attr)
            return obj
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"Cannot resolve object: {qualname}")


def _wrap_fn(
    original_fn: Callable, hook_fn: Callable, hook_type: HookType
) -> Callable:
    """Create a wrapper function based on the hook type."""
    if hook_type == HookType.REPLACE:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            return hook_fn(*args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.BEFORE:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            result = hook_fn(*args, **kwargs)
            if result is not None:
                args, kwargs = result
            return original_fn(*args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.AFTER:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            result = original_fn(*args, **kwargs)
            modified = hook_fn(result, *args, **kwargs)
            return modified if modified is not None else result

        wrapper.__wrapped__ = original_fn
        return wrapper

    elif hook_type == HookType.AROUND:

        @functools.wraps(original_fn)
        def wrapper(*args, **kwargs):
            return hook_fn(original_fn, *args, **kwargs)

        wrapper.__wrapped__ = original_fn
        return wrapper

    else:
        raise ValueError(f"Unknown hook type: {hook_type}")
