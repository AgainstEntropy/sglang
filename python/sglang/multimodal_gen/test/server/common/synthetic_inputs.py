"""Input images and control clips drawn at test time for the nightly B200 cases.

Cases name these paths at import; ``write_synthetic_inputs`` creates the files
before the first request. The server runs on the same host, so it reads the
control clips straight from disk. The blur control is the exception: the model
copies a blurred drawing back instead of generating, so it is the checkpoint's
own sample clip, fetched from its repo.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SYNTHETIC_INPUT_DIR = Path(tempfile.gettempdir()) / "sglang_diffusion_test_inputs"

HOUSE_EDIT_INPUT = SYNTHETIC_INPUT_DIR / "house_1024x1024.png"
HOUSE_I2V_INPUT = SYNTHETIC_INPUT_DIR / "house_832x480.png"
# The Sim-Transfer checkpoint's blur sample: a dancer in a timber yard, 832x480 at
# 30 fps; the case reads its first 33 frames.
SIM_TRANSFER_REPO = "nvidia/Cosmos3-Nano-Sim-Transfer"
SIM_TRANSFER_BLUR_SAMPLE = "assets/sample_data/example2/example2_blur.mp4"
SIM_TRANSFER_BLUR_CONTROL = SYNTHETIC_INPUT_DIR / "sim_transfer_example2_blur.mp4"
SIM_TRANSFER_BLUR_FRAMES = 33
SIM_TRANSFER_BLUR_FPS = 30
# Lane lines for a 2-camera rig, 29 frames at 30 fps each.
ROAD_CONTROL_FRAMES = 29
ROAD_CONTROL_FPS = 30
ROAD_CONTROLS = {
    "camera_front_wide_120fov": SYNTHETIC_INPUT_DIR / "road_front_832x480_29f.mp4",
    "camera_cross_left_120fov": SYNTHETIC_INPUT_DIR / "road_left_832x480_29f.mp4",
}


def draw_house(width: int, height: int, shift: int = 0) -> np.ndarray:
    """A red house in a green field under a blue sky, as uint8 [H, W, 3]."""
    image = Image.new("RGB", (width, height), (120, 170, 230))
    draw = ImageDraw.Draw(image)
    horizon = int(height * 0.6)
    draw.rectangle([0, horizon, width, height], fill=(70, 150, 60))
    sun = int(min(width, height) * 0.08)
    draw.ellipse([width - 3 * sun, sun, width - sun, 3 * sun], fill=(250, 220, 90))
    house_w = int(width * 0.3)
    house_h = int(height * 0.25)
    left = int(width * 0.2) + shift
    top = horizon - house_h // 2
    draw.rectangle([left, top, left + house_w, top + house_h], fill=(180, 40, 35))
    draw.polygon(
        [
            (left - house_w // 10, top),
            (left + house_w // 2, top - house_h * 2 // 3),
            (left + house_w + house_w // 10, top),
        ],
        fill=(90, 50, 30),
    )
    door_w = house_w // 6
    draw.rectangle(
        [
            left + house_w // 2 - door_w // 2,
            top + house_h // 2,
            left + house_w // 2 + door_w // 2,
            top + house_h,
        ],
        fill=(60, 35, 20),
    )
    window = house_w // 7
    draw.rectangle(
        [
            left + house_w // 6,
            top + house_h // 4,
            left + house_w // 6 + window,
            top + house_h // 4 + window,
        ],
        fill=(230, 230, 200),
    )
    return np.asarray(image)


def draw_road_clip(
    vanishing_x: float,
    width: int = 832,
    height: int = 480,
    num_frames: int = ROAD_CONTROL_FRAMES,
) -> np.ndarray:
    """Road edges and a dashed centre line on black, moving toward the camera.

    ``vanishing_x`` is the vanishing point as a fraction of the width; a
    side camera sees the road converge off-centre.
    """
    vx = int(width * vanishing_x)
    vy = int(height * 0.45)
    frames = []
    for index in range(num_frames):
        image = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        for edge_x in (vx - int(width * 0.7), vx + int(width * 0.7)):
            draw.line([(vx, vy), (edge_x, height)], fill=(255, 255, 255), width=6)
        # Dashes advance one step per frame along the centre line.
        for dash in range(8):
            start = ((dash * 0.125 + index * 0.02) % 1.0) ** 2
            end = min(start + 0.06, 1.0)
            points = [
                (vx + (width // 2 - vx) * t, vy + (height - vy) * t)
                for t in (start, end)
            ]
            draw.line(points, fill=(255, 210, 0), width=8)
        frames.append(np.asarray(image))
    return np.stack(frames)


def _write_atomically(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    try:
        write(tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _write_image(path: Path, image: np.ndarray) -> None:
    _write_atomically(path, lambda name: Image.fromarray(image).save(name))


def _encode_clip(name: str, frames: np.ndarray, fps: int) -> None:
    import av

    with av.open(name, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames.shape[1:3]
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            container.mux(
                stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24"))
            )
        container.mux(stream.encode())


def _write_clip(path: Path, frames: np.ndarray, fps: int) -> None:
    _write_atomically(path, lambda name: _encode_clip(name, frames, fps))


def write_synthetic_inputs() -> None:
    """Create every input file above that does not exist yet."""
    if not HOUSE_EDIT_INPUT.exists():
        _write_image(HOUSE_EDIT_INPUT, draw_house(1024, 1024))
    if not HOUSE_I2V_INPUT.exists():
        _write_image(HOUSE_I2V_INPUT, draw_house(832, 480))
    if not SIM_TRANSFER_BLUR_CONTROL.exists():
        from huggingface_hub import hf_hub_download

        sample = hf_hub_download(
            repo_id=SIM_TRANSFER_REPO, filename=SIM_TRANSFER_BLUR_SAMPLE
        )
        _write_atomically(
            SIM_TRANSFER_BLUR_CONTROL, lambda name: shutil.copyfile(sample, name)
        )
    for vanishing_x, path in zip((0.5, 0.8), ROAD_CONTROLS.values()):
        if not path.exists():
            _write_clip(path, draw_road_clip(vanishing_x), ROAD_CONTROL_FPS)
