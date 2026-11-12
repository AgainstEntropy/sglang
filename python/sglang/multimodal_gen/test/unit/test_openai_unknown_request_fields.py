# SPDX-License-Identifier: Apache-2.0
"""Unknown fields on /v1/images and /v1/videos get a 400 instead of being dropped."""

import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.configs.sample.kandinsky6_sr import (
    Kandinsky6SRSamplingParams,
)
from sglang.multimodal_gen.configs.sample.ming_image import (
    MingImageLayerSamplingParams,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.openai import image_api, video_api
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    raise_on_unknown_request_fields,
)


@pytest.fixture
def client(tmp_path):
    args = SimpleNamespace(
        input_save_path=None,
        output_path=str(tmp_path / "outputs"),
        model_path="unused",
        served_model_name="test",
    )
    app = FastAPI()
    app.include_router(image_api.router)
    app.include_router(video_api.router)
    dispatch = AsyncMock()
    with (
        patch(
            "sglang.multimodal_gen.runtime.server_args.server_args._global_server_args",
            args,
        ),
        patch.object(
            image_api, "resolve_sampling_params_cls", return_value=SamplingParams
        ),
        patch.object(
            video_api, "resolve_sampling_params_cls", return_value=SamplingParams
        ),
        patch.object(image_api, "process_generation_batch", dispatch),
        patch.object(video_api, "_dispatch_job_async", dispatch),
        TestClient(app) as test_client,
    ):
        test_client.dispatch = dispatch
        yield test_client


def _png() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _assert_rejected(client, response, *unknown: str) -> None:
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail.startswith(f"Unknown request fields: {', '.join(unknown)}."), detail
    # the client learns which extension fields it may send instead
    assert "enable_cache_dit" in detail
    client.dispatch.assert_not_called()


@pytest.mark.parametrize(
    "body, unknown",
    [
        ({"mask": "m.png", "bogus": 1}, ("bogus", "mask")),
        ({"extra_body": {"num_layers": 2}}, ("num_layers",)),
        ({"extra_body": json.dumps({"bogus": 1})}, ("bogus",)),
    ],
    ids=["top_level", "extra_body", "extra_body_json_string"],
)
def test_image_generations_rejects_unknown_fields(client, body, unknown):
    response = client.post("/v1/images/generations", json={"prompt": "a cat", **body})
    _assert_rejected(client, response, *unknown)


@pytest.mark.parametrize(
    "fields, unknown",
    [
        ({"bogus": "1"}, ("bogus",)),
        # generations reads this extension, but /edits never does
        ({"cfg_gate_step": "3"}, ("cfg_gate_step",)),
        # /edits reads no transport container
        ({"extra_body": "{}"}, ("extra_body",)),
    ],
)
def test_image_edits_rejects_unknown_form_fields(fields, unknown):
    with pytest.raises(HTTPException) as exc_info:
        image_api._raise_on_unknown_edit_form_fields(
            {"prompt": "a", "image[]": b"", "url[]": "u", **fields}, SamplingParams
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        f"Unknown request fields: {', '.join(unknown)}. Extension fields accepted "
        "by this model: none"
    )


def test_image_edits_route_rejects_before_saving_inputs(client):
    form = {"prompt": "a cat", "bogus": "1"}
    files = {"image": ("in.png", _png(), "image/png")}
    response = client.post("/v1/images/edits", data=form, files=files)
    assert response.status_code == 400, response.text
    assert response.json()["detail"].startswith("Unknown request fields: bogus.")

    # the mask has its own message and is checked first
    files["mask"] = ("mask.png", _png(), "image/png")
    response = client.post("/v1/images/edits", data=form, files=files)
    assert response.status_code == 400
    assert "Field 'mask' is not supported" in response.json()["detail"]


@pytest.mark.parametrize(
    "body, unknown",
    [
        ({"response_format": "b64_json"}, ("response_format",)),
        ({"extra_body": {"enable_memory_bank": False}}, ("enable_memory_bank",)),
        ({"extra_args": {"bogus": 1}}, ("bogus",)),
    ],
    ids=["top_level", "extra_body", "extra_args"],
)
def test_video_json_rejects_unknown_fields(client, body, unknown):
    response = client.post("/v1/videos", json={"prompt": "a cat", **body})
    _assert_rejected(client, response, *unknown)


@pytest.mark.parametrize(
    "form, unknown",
    [
        ({"response_format": "b64_json"}, ("response_format",)),
        ({"extra_body": json.dumps({"bogus": 1})}, ("bogus",)),
        (
            {"extra_params": json.dumps({"sr_resolution_scale": 4})},
            ("sr_resolution_scale",),
        ),
    ],
    ids=["form_field", "extra_body", "extra_params"],
)
def test_video_multipart_rejects_unknown_fields(client, form, unknown):
    response = client.post(
        "/v1/videos",
        data={"prompt": "a cat", **form},
        files={"input_reference": ("in.png", _png(), "image/png")},
    )
    _assert_rejected(client, response, *unknown)


@pytest.mark.parametrize(
    "fields, sampling_params_cls, api",
    [
        ({"enable_cache_dit": True, "cache_dit_params": {}}, SamplingParams, "image"),
        ({"extra_body": {"num_layers": 2}}, MingImageLayerSamplingParams, "image"),
        # flatten_extra_params maps guardrails onto the declared use_guardrails
        ({"guardrails": False}, Cosmos3SamplingParams, "image"),
        ({"quality": "exact", "use_diffusion_decoder": True}, SamplingParams, "video"),
        (
            {"extra_json": '{"sr_resolution_scale": 4}'},
            Kandinsky6SRSamplingParams,
            "video",
        ),
    ],
)
def test_common_and_declared_extension_fields_are_accepted(
    fields, sampling_params_cls, api
):
    raise_on_unknown_request_fields(fields, sampling_params_cls, api)
