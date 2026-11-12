"""SGLANG_DIFFUSION_SYNC_STAGE_PROFILING is parsed as a boolean env var."""

import os
import unittest
from unittest.mock import MagicMock, patch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    RequestMetrics,
    StageProfiler,
)
from sglang.test.test_utils import CustomTestCase

_ENV = "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING"


class TestSyncStageProfilingEnv(CustomTestCase):
    def _syncs(self, value: str | None) -> bool:
        device = MagicMock()
        device.is_available.return_value = True
        with (
            patch.dict(os.environ),
            patch("torch.get_device_module", return_value=device),
        ):
            os.environ.pop(_ENV, None)
            if value is not None:
                os.environ[_ENV] = value
            profiler = StageProfiler(
                "stage", init_logger(__name__), RequestMetrics("req")
            )
            profiler._maybe_sync_device()
        return device.synchronize.called

    def test_truthy_values_sync(self):
        for value in ("1", "true", "True", "yes", "on", " 1 "):
            with self.subTest(value=value):
                self.assertTrue(self._syncs(value))

    def test_unset_and_falsy_values_do_not_sync(self):
        for value in (None, "0", "false", "off", ""):
            with self.subTest(value=value):
                self.assertFalse(self._syncs(value))


if __name__ == "__main__":
    unittest.main()
