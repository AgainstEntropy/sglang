# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image 2.1 full-checkpoint tests, run nightly on H100.

SGLANG_QWEN_IMAGE21_TEST_MODEL overrides the checkpoint and
SGLANG_QWEN_IMAGE21_TEST_IMAGE the editing reference (a local file or URL).
"""

import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    TI2I_QWEN_IMAGE_EDIT_INPUT,
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
)
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_QWEN_IMAGE_21_MODEL_NAME_FOR_TEST,
    is_image_url,
)

MODEL = os.environ.get(
    "SGLANG_QWEN_IMAGE21_TEST_MODEL", DEFAULT_QWEN_IMAGE_21_MODEL_NAME_FOR_TEST
)


@pytest.fixture(params=["generation", "edit", "alpha"])
def case(request):
    mode = request.param
    image = None
    prompt = "A red ceramic teapot on a wooden table beside a window."
    if mode == "edit":
        image = os.environ.get(
            "SGLANG_QWEN_IMAGE21_TEST_IMAGE", TI2I_QWEN_IMAGE_EDIT_INPUT
        )
        if not is_image_url(image):
            image = Path(image)
            assert image.is_file(), f"Reference image does not exist: {image}"
        prompt = "Change the teapot to blue, keeping its shape and the scene unchanged."
    elif mode == "alpha":
        prompt = (
            "A single fluffy orange cat sitting, full body, isolated on a transparent "
            "background. A clean cutout with an alpha channel, transparent outside "
            "the cat, no floor, no shadow, no background."
        )
    return DiffusionTestCase(
        f"qwen_image21_{mode}",
        DiffusionServerArgs(
            model_path=MODEL,
            modality="image",
            extras=[
                "--model-id Qwen-Image-2.1",
                "--performance-mode speed",
                "--attention-backend torch_sdpa",
            ],
        ),
        DiffusionSamplingParams(
            prompt=prompt,
            image_path=image,
            output_size="1024x1024",
            output_format="png",
            extras={"num_inference_steps": 40, "guidance_scale": 1, "seed": 42},
        ),
        perf_repeat_requests=2,
        run_perf_check=False,
        # No h100 scenario yet; drop this once one is recorded.
        require_perf_baseline=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
        expected_model_id="Qwen-Image-2.1",
        run_t2v_input_reference_check=False,
    )


class TestQwenImage21Server(DiffusionServerBase):
    def run_and_collect(self, ctx, case_id, generate_fn, collect_perf=True):
        record, content = super().run_and_collect(
            ctx, case_id, generate_fn, collect_perf
        )
        with Image.open(io.BytesIO(content)) as image:
            assert image.mode == "RGBA"
            assert image.size == (1024, 1024)
            if case_id.endswith("_alpha"):
                alpha = np.asarray(image.getchannel("A"))
                assert alpha.min() == 0 and alpha.max() == 255
                assert np.mean(alpha <= 5) > 0.4
        return record, content
