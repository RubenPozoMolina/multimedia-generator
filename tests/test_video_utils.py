import logging
from unittest.mock import patch, MagicMock

import pytest

from utils.video_models.base_video_model import BaseVideoModel
from utils.video_models.ltx_video_model import LTXVideoModel
from utils.video_models.wan_model import WanModel
from utils.video_utils import VideoUtils, models

logger = logging.getLogger(__name__)


class TestBaseVideoModel:

    def test_init_sets_device_and_output_path(self):
        model = BaseVideoModel(model_id="test-model", output_path="test_output")
        assert model.model_id == "test-model"
        assert model.output_path == "test_output"
        assert model.device in ("cuda", "cpu")

    def test_load_model_raises_not_implemented(self):
        model = BaseVideoModel()
        with pytest.raises(NotImplementedError):
            model.load_model("any-model")

    def test_text_to_video_raises_not_implemented(self):
        model = BaseVideoModel()
        with pytest.raises(NotImplementedError):
            model.text_to_video("a prompt")

    def test_image_to_video_raises_not_implemented(self):
        model = BaseVideoModel()
        with pytest.raises(NotImplementedError):
            model.image_to_video("image", "a prompt")

    def test_get_file_name_generates_mp4(self):
        model = BaseVideoModel(output_path="output")
        file_name = model.get_file_name(None)
        assert str(file_name).endswith(".mp4")

    def test_get_file_name_uses_custom_name(self):
        model = BaseVideoModel(output_path="output")
        file_name = model.get_file_name("custom_video.mp4")
        assert file_name == "custom_video.mp4"


class TestVideoUtils:

    def test_models_list_contains_expected_entries(self):
        model_names = [m["name"] for m in models]
        assert "Lightricks/LTX-Video" in model_names
        assert "Wan-AI/Wan2.2-I2V-A14B-Diffusers" in model_names

    def test_get_model_returns_ltx_for_ltx_id(self):
        video_utils = VideoUtils.__new__(VideoUtils)
        video_utils.model = BaseVideoModel()
        result = video_utils.get_model("Lightricks/LTX-Video")
        assert isinstance(result, LTXVideoModel)

    def test_get_model_returns_wan_for_wan_id(self):
        video_utils = VideoUtils.__new__(VideoUtils)
        video_utils.model = BaseVideoModel()
        result = video_utils.get_model("Wan-AI/Wan2.2-I2V-A14B-Diffusers")
        assert isinstance(result, WanModel)

    def test_get_model_raises_for_unknown_model(self):
        video_utils = VideoUtils.__new__(VideoUtils)
        video_utils.model = BaseVideoModel()
        with pytest.raises(ValueError, match="not found"):
            video_utils.get_model("unknown/model")

    @patch.object(LTXVideoModel, "load_model")
    def test_text_to_video_delegates_to_model(self, mock_load):
        video_utils = VideoUtils.__new__(VideoUtils)
        mock_model = MagicMock(spec=LTXVideoModel)
        mock_model.text_to_video.return_value = "output/video.mp4"
        video_utils.model = mock_model

        result = video_utils.text_to_video("a cat running")
        assert result == "output/video.mp4"
        mock_model.text_to_video.assert_called_once()

    @patch.object(WanModel, "load_model")
    def test_image_to_video_delegates_to_model(self, mock_load):
        video_utils = VideoUtils.__new__(VideoUtils)
        mock_model = MagicMock(spec=WanModel)
        mock_model.image_to_video.return_value = "output/video.mp4"
        video_utils.model = mock_model

        result = video_utils.image_to_video("image.png", "a cat running")
        assert result == "output/video.mp4"
        mock_model.image_to_video.assert_called_once()
