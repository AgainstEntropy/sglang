"""Nightly one-B200 cases for catalog families no other lane loads."""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.test.server.common.case_fixtures import (
    diffusion_case_fixture,
)
from sglang.multimodal_gen.test.server.common.synthetic_inputs import (
    write_synthetic_inputs,
)
from sglang.multimodal_gen.test.server.gpu_cases import NIGHTLY_ONE_GPU_B200_CASES
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)


@pytest.fixture(scope="module", autouse=True)
def _synthetic_inputs():
    write_synthetic_inputs()


class TestDiffusionServerNightlyOneGpuB200(DiffusionServerBase):
    case = diffusion_case_fixture(NIGHTLY_ONE_GPU_B200_CASES)
