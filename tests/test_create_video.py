import logging
from unittest.mock import patch, MagicMock

import pytest

logger = logging.getLogger(__name__)


class TestCreateVideo:

    @patch("scripts.create_video.VideoUtils")
    def test_main_with_required_args(self, mock_video_utils_cls):
        mock_instance = MagicMock()
        mock_instance.text_to_video.return_value = "output/video.mp4"
        mock_video_utils_cls.return_value = mock_instance

        with patch("sys.argv", ["create_video.py", "--prompt", "A cat walking"]):
            from scripts.create_video import main
            main()

        mock_video_utils_cls.assert_called_once_with(model_id="Lightricks/LTX-Video")
        mock_instance.text_to_video.assert_called_once_with(
            "A cat walking",
            negative_prompt="",
            height=480,
            width=704,
            num_frames=81,
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=None,
            output_path=None
        )

    @patch("scripts.create_video.VideoUtils")
    def test_main_with_all_args(self, mock_video_utils_cls):
        mock_instance = MagicMock()
        mock_instance.text_to_video.return_value = "output/custom.mp4"
        mock_video_utils_cls.return_value = mock_instance

        with patch("sys.argv", [
            "create_video.py",
            "--prompt", "A sunset",
            "--output", "output/custom.mp4",
            "--model", "custom/model",
            "--height", "360",
            "--width", "640",
            "--num-frames", "30",
            "--num-inference-steps", "25",
            "--guidance-scale", "3.0",
            "--negative-prompt", "blurry",
            "--seed", "42"
        ]):
            from scripts.create_video import main
            main()

        mock_video_utils_cls.assert_called_once_with(model_id="custom/model")
        mock_instance.text_to_video.assert_called_once_with(
            "A sunset",
            negative_prompt="blurry",
            height=360,
            width=640,
            num_frames=30,
            num_inference_steps=25,
            guidance_scale=3.0,
            seed=42,
            output_path="output/custom.mp4"
        )

    @patch("scripts.create_video.VideoUtils")
    def test_main_uses_default_model(self, mock_video_utils_cls):
        mock_instance = MagicMock()
        mock_instance.text_to_video.return_value = "output/video.mp4"
        mock_video_utils_cls.return_value = mock_instance

        with patch("sys.argv", ["create_video.py", "--prompt", "A dog"]):
            from scripts.create_video import main
            main()

        mock_video_utils_cls.assert_called_once_with(model_id="Lightricks/LTX-Video")

    @patch("scripts.create_video.VideoUtils")
    def test_main_custom_model(self, mock_video_utils_cls):
        mock_instance = MagicMock()
        mock_instance.text_to_video.return_value = "output/video.mp4"
        mock_video_utils_cls.return_value = mock_instance

        with patch("sys.argv", ["create_video.py", "--prompt", "A bird", "--model", "Wan-AI/Wan2.2-I2V-A14B-Diffusers"]):
            from scripts.create_video import main
            main()

        mock_video_utils_cls.assert_called_once_with(model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers")

    def test_main_missing_prompt_raises_error(self):
        with patch("sys.argv", ["create_video.py"]):
            with pytest.raises(SystemExit):
                from scripts.create_video import main
                main()
