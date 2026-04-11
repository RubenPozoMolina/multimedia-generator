import logging
from unittest.mock import patch, MagicMock

import pytest

from utils.image_models.base_image_model import BaseImageModel
from utils.image_models.flux_model import FluxModel
from utils.image_models.sdxl_model import SDXLModel
from utils.image_models.stable_difusion_model import StableDiffusionModel
from utils.image_utils import ImageUtils, models

logger = logging.getLogger(__name__)


class TestBaseImageModel:

    def test_init_sets_device_and_output_path(self):
        model = BaseImageModel(model_id="test-model", output_path="test_output")
        assert model.model_id == "test-model"
        assert model.output_path == "test_output"
        assert model.device in ("cuda", "cpu")

    def test_init_defaults(self):
        model = BaseImageModel()
        assert model.model_id is None
        assert model.output_path == "output"

    def test_load_model_raises_not_implemented(self):
        model = BaseImageModel()
        with pytest.raises(NotImplementedError):
            model.load_model("any-model")

    def test_get_file_name_generates_png(self):
        model = BaseImageModel(output_path="output")
        file_name = model.get_file_name(None)
        assert str(file_name).endswith(".png")

    def test_get_file_name_uses_custom_name(self):
        model = BaseImageModel(output_path="output")
        file_name = model.get_file_name("custom_image.png")
        assert file_name == "custom_image.png"

    def test_text_to_image_calls_pipeline_for_non_flux(self):
        model = BaseImageModel(model_id="CompVis/stable-diffusion-v1-4", output_path="output")
        mock_image = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.images = [mock_image]
        model.pipeline = mock_pipeline

        result = model.text_to_image("a cat", output_file_name="out.png")

        mock_pipeline.assert_called_once_with(
            "a cat",
            negative_prompt="",
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=50,
            seed=None
        )
        mock_image.save.assert_called_once()
        assert str(result) == "out.png"

    def test_text_to_image_calls_pipeline_for_flux(self):
        model = BaseImageModel(model_id="black-forest-labs/FLUX.1-dev", output_path="output")
        mock_image = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.images = [mock_image]
        model.pipeline = mock_pipeline

        result = model.text_to_image("a cat", output_file_name="out.png")

        mock_pipeline.assert_called_once_with(
            "a cat",
            negative_prompt="",
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=50,
            generator=None
        )
        mock_image.save.assert_called_once()
        assert str(result) == "out.png"

    def test_image_to_image_calls_pipeline(self):
        model = BaseImageModel(model_id="test-model", output_path="output")
        mock_image = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.return_value.images = [mock_image]
        model.pipeline = mock_pipeline

        result = model.image_to_image("input_image", output_file_name="out.png")

        mock_pipeline.assert_called_once_with("input_image")
        mock_image.save.assert_called_once()
        assert str(result) == "out.png"


class TestImageUtils:

    def test_models_list_contains_expected_entries(self):
        model_names = [m["name"] for m in models]
        assert "CompVis/stable-diffusion-v1-4" in model_names
        assert "Lykon/DreamShaper" in model_names
        assert "black-forest-labs/FLUX.1-dev" in model_names
        assert "black-forest-labs/FLUX.1-schnell" in model_names
        assert "stabilityai/stable-diffusion-xl-base-1.0" in model_names

    def test_get_model_returns_stable_diffusion(self):
        image_utils = ImageUtils.__new__(ImageUtils)
        image_utils.model = BaseImageModel()
        result = image_utils.get_model("CompVis/stable-diffusion-v1-4")
        assert isinstance(result, StableDiffusionModel)

    def test_get_model_returns_flux(self):
        image_utils = ImageUtils.__new__(ImageUtils)
        image_utils.model = BaseImageModel()
        result = image_utils.get_model("black-forest-labs/FLUX.1-dev")
        assert isinstance(result, FluxModel)

    def test_get_model_returns_sdxl(self):
        image_utils = ImageUtils.__new__(ImageUtils)
        image_utils.model = BaseImageModel()
        result = image_utils.get_model("stabilityai/stable-diffusion-xl-base-1.0")
        assert isinstance(result, SDXLModel)

    def test_get_model_raises_for_unknown_model(self):
        image_utils = ImageUtils.__new__(ImageUtils)
        image_utils.model = BaseImageModel()
        with pytest.raises(ValueError, match="not found"):
            image_utils.get_model("unknown/model")

    def test_get_model_returns_dreamshaper_as_stable_diffusion(self):
        image_utils = ImageUtils.__new__(ImageUtils)
        image_utils.model = BaseImageModel()
        result = image_utils.get_model("Lykon/DreamShaper")
        assert isinstance(result, StableDiffusionModel)

    def test_get_model_custom_output_path(self):
        image_utils = ImageUtils.__new__(ImageUtils)
        image_utils.model = BaseImageModel()
        result = image_utils.get_model("CompVis/stable-diffusion-v1-4", output_path="custom_output")
        assert result.output_path == "custom_output"

    def test_text_to_image_delegates_to_model(self):
        image_utils = ImageUtils.__new__(ImageUtils)
        mock_model = MagicMock(spec=BaseImageModel)
        mock_model.text_to_image.return_value = "output/image.png"
        image_utils.model = mock_model

        result = image_utils.text_to_image("a cat sitting")
        assert result == "output/image.png"
        mock_model.text_to_image.assert_called_once_with(
            "a cat sitting",
            negative_prompt="",
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=50,
            output_file_name=None
        )

    def test_text_to_image_passes_all_parameters(self):
        image_utils = ImageUtils.__new__(ImageUtils)
        mock_model = MagicMock(spec=BaseImageModel)
        mock_model.text_to_image.return_value = "output/custom.png"
        image_utils.model = mock_model

        result = image_utils.text_to_image(
            "a dog",
            negative_prompt="bad quality",
            height=768,
            width=768,
            guidance_scale=10.0,
            num_inference_steps=30,
            seed=42,
            output_path="output/custom.png"
        )
        assert result == "output/custom.png"
        mock_model.text_to_image.assert_called_once_with(
            "a dog",
            negative_prompt="bad quality",
            height=768,
            width=768,
            guidance_scale=10.0,
            num_inference_steps=30,
            output_file_name="output/custom.png"
        )
