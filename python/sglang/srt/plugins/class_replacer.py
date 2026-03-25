"""
Class-level OOT (Out-Of-Tree) replacement registry.

Allows plugins to transparently replace classes in the sglang engine
with custom implementations. Similar to vLLM's CustomOp.register_oot pattern.

Usage:
    from sglang.srt.plugins.class_replacer import ClassReplacer

    # In a plugin's register() function:
    ClassReplacer.register(
        "sglang.srt.managers.scheduler.Scheduler",
        "my_plugin.custom_scheduler.EnhancedScheduler"
    )

    # In engine code (factory methods):
    SchedulerCls = ClassReplacer.maybe_replace(Scheduler)
    scheduler = SchedulerCls(...)
"""

import logging

logger = logging.getLogger(__name__)


class ClassReplacer:
    """
    Global registry for class replacements.

    Plugins register replacements mapping original class paths to their
    custom implementation paths. The engine queries this registry at
    class instantiation points to transparently substitute implementations.
    """

    _registry: dict[str, str] = {}

    @classmethod
    def register(cls, original_cls_path: str, replacement_cls_path: str):
        """
        Register a class replacement.

        Args:
            original_cls_path: Fully-qualified path of the original class.
            replacement_cls_path: Fully-qualified path of the replacement class
                                  (lazy-loaded on first use).
        """
        if original_cls_path in cls._registry:
            logger.warning(
                "Overwriting class replacement for %s: %s -> %s",
                original_cls_path,
                cls._registry[original_cls_path],
                replacement_cls_path,
            )
        cls._registry[original_cls_path] = replacement_cls_path
        logger.info(
            "Registered class replacement: %s -> %s",
            original_cls_path,
            replacement_cls_path,
        )

    @classmethod
    def get_replacement(cls, original_cls_path: str) -> type | None:
        """
        Get the replacement class for the given original class path.

        Returns None if no replacement is registered.
        """
        replacement_path = cls._registry.get(original_cls_path)
        if replacement_path is None:
            return None
        from sglang.srt.plugins.hook_registry import resolve_obj

        return resolve_obj(replacement_path)

    @classmethod
    def maybe_replace(cls, original_cls: type) -> type:
        """
        Return the replacement class if registered, otherwise the original.

        Builds the qualname from the class's __module__ and __qualname__ attributes.
        Use this in factory methods / instantiation points.
        """
        qualname = f"{original_cls.__module__}.{original_cls.__qualname__}"
        replacement = cls.get_replacement(qualname)
        if replacement is not None:
            logger.info("Replacing %s with %s", qualname, replacement)
            return replacement
        return original_cls

    @classmethod
    def reset(cls):
        """Reset all registered replacements. Primarily for testing."""
        cls._registry.clear()
