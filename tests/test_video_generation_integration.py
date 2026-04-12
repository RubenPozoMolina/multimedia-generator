import logging
import os

import pytest
from PIL import Image

from utils.video_utils import VideoUtils, models

logger = logging.getLogger(__name__)

text_to_video_models = [m for m in models if "Wan" not in m["name"]]
i2v_only_models = [m for m in models if "Wan" in m["name"]]


@pytest.fixture
def output_dir():
    yield "output"


@pytest.fixture
def sample_image():
    image = Image.new("RGB", (704, 480), color=(128, 200, 255))
    return image


@pytest.mark.integration
@pytest.mark.parametrize("model_entry", text_to_video_models, ids=[m["name"] for m in text_to_video_models])
def test_text_to_video_real(model_entry, output_dir):
    model_name = model_entry["name"]

    video_utils = VideoUtils(model_name, output_path=output_dir)
    logger.info("Using model: %s", model_name)

    output_file = video_utils.text_to_video(
        "a white cat sitting on a table",
        negative_prompt="low quality, blurry",
        height=256,
        width=256,
        num_frames=9,
        num_inference_steps=5,
        guidance_scale=3.0,
        seed=42
    )
    logger.info("Output file: %s", output_file)

    assert os.path.isfile(output_file), f"Output file does not exist: {output_file}"
    assert output_file.endswith(".mp4")

    file_size = os.path.getsize(output_file)
    assert file_size > 0, "Generated video file is empty"

    logger.info("Generated file: %s (size: %d bytes)", output_file, file_size)


@pytest.mark.integration
@pytest.mark.parametrize("model_entry", models, ids=[m["name"] for m in models])
def test_image_to_video_real(model_entry, output_dir, sample_image):
    model_name = model_entry["name"]

    video_utils = VideoUtils(model_name, output_path=output_dir)
    logger.info("Using model: %s", model_name)

    output_file = video_utils.image_to_video(
        sample_image,
        "a white cat sitting on a table",
        negative_prompt="low quality, blurry",
        height=256,
        width=256,
        num_frames=9,
        num_inference_steps=5,
        guidance_scale=3.0,
        seed=42
    )
    logger.info("Output file: %s", output_file)

    assert os.path.isfile(output_file), f"Output file does not exist: {output_file}"
    assert output_file.endswith(".mp4")

    file_size = os.path.getsize(output_file)
    assert file_size > 0, "Generated video file is empty"

    logger.info("Generated file: %s (size: %d bytes)", output_file, file_size)
