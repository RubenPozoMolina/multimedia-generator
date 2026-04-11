import logging
import os

import pytest
from PIL import Image

from utils.image_utils import ImageUtils, models

logger = logging.getLogger(__name__)


@pytest.fixture
def output_dir():
    yield "output"


@pytest.mark.integration
@pytest.mark.parametrize("model_entry", models, ids=[m["name"] for m in models])
def test_text_to_image_real(model_entry, output_dir):
    model_name = model_entry["name"]

    image_utils = ImageUtils(model_name, output_path=output_dir)
    logger.info("Using model: %s", model_name)

    output_file = image_utils.text_to_image(
        "a white cat sitting on a table",
        negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        height=256,
        width=256,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42
    )
    logger.info("Output file: %s", output_file)

    assert os.path.isfile(output_file), f"Output file does not exist: {output_file}"
    assert output_file.endswith(".png")

    img = Image.open(output_file)
    assert img.size[0] > 0 and img.size[1] > 0, "Generated image has invalid dimensions"
    assert img.mode in ("RGB", "RGBA"), f"Unexpected image mode: {img.mode}"

    logger.info("Generated file: %s (size: %sx%s, mode: %s)", output_file, img.size[0], img.size[1], img.mode)
