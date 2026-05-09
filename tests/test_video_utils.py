import logging
from unittest.mock import patch, MagicMock

import pytest

from utils.video_models.base_video_model import BaseVideoModel
from utils.video_models.ltx_video_model import LTXVideoModel
from utils.video_models.ltx2_model import LTX2Model
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

    def test_align_dimension_already_divisible(self):
        assert BaseVideoModel.align_dimension(480) == 480
        assert BaseVideoModel.align_dimension(704) == 704
        assert BaseVideoModel.align_dimension(32) == 32

    def test_align_dimension_rounds_to_nearest(self):
        assert BaseVideoModel.align_dimension(360) == 352
        assert BaseVideoModel.align_dimension(640) == 640
        assert BaseVideoModel.align_dimension(500) == 512

    def test_align_dimension_minimum_is_divisor(self):
        assert BaseVideoModel.align_dimension(1) == 32
        assert BaseVideoModel.align_dimension(15) == 32


class TestVideoUtils:

    def test_models_list_contains_expected_entries(self):
        model_names = [m["name"] for m in models]
        assert "Lightricks/LTX-Video" in model_names
        assert "Lightricks/LTX-2" in model_names
        assert "Wan-AI/Wan2.2-I2V-A14B-Diffusers" in model_names

    def test_get_model_returns_ltx_for_ltx_id(self):
        video_utils = VideoUtils.__new__(VideoUtils)
        video_utils.model = BaseVideoModel()
        result = video_utils.get_model("Lightricks/LTX-Video")
        assert isinstance(result, LTXVideoModel)

    def test_get_model_returns_ltx2_for_ltx2_id(self):
        video_utils = VideoUtils.__new__(VideoUtils)
        video_utils.model = BaseVideoModel()
        result = video_utils.get_model("Lightricks/LTX-2")
        assert isinstance(result, LTX2Model)

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

    def test_concatenate_videos_raises_on_empty_list(self):
        with pytest.raises(ValueError, match="No video paths provided"):
            VideoUtils.concatenate_videos([], "output/final.mp4")

    @patch("utils.video_utils.Path.exists", return_value=True)
    @patch("utils.video_utils.concatenate_videoclips")
    @patch("utils.video_utils.VideoFileClip")
    def test_concatenate_videos_calls_moviepy(self, mock_vfc, mock_concat, mock_exists, tmp_path):
        mock_clip_1 = MagicMock()
        mock_clip_1.fps = 30
        mock_clip_2 = MagicMock()
        mock_vfc.side_effect = [mock_clip_1, mock_clip_2]

        mock_final = MagicMock()
        mock_concat.return_value = mock_final

        output_file = str(tmp_path / "final.mp4")
        result = VideoUtils.concatenate_videos(["a.mp4", "b.mp4"], output_file)

        assert mock_vfc.call_count == 2
        mock_concat.assert_called_once_with([mock_clip_1, mock_clip_2])
        mock_final.write_videofile.assert_called_once_with(output_file, fps=30, logger=None)
        assert result == output_file
        mock_clip_1.close.assert_called_once()
        mock_clip_2.close.assert_called_once()

    @patch("utils.video_utils.Path.exists", return_value=True)
    @patch("utils.video_utils.concatenate_videoclips")
    @patch("utils.video_utils.VideoFileClip")
    def test_concatenate_videos_uses_custom_fps(self, mock_vfc, mock_concat, mock_exists, tmp_path):
        mock_clip = MagicMock()
        mock_clip.fps = 30
        mock_vfc.return_value = mock_clip

        mock_final = MagicMock()
        mock_concat.return_value = mock_final

        output_file = str(tmp_path / "final.mp4")
        VideoUtils.concatenate_videos(["a.mp4"], output_file, fps=60)

        mock_final.write_videofile.assert_called_once_with(output_file, fps=60, logger=None)

    @patch("utils.video_utils.Path.exists", return_value=True)
    @patch("utils.video_utils.VideoFileClip")
    def test_extract_last_frame_returns_pil_image(self, mock_vfc, mock_exists):
        import numpy as np
        from PIL import Image

        mock_clip = MagicMock()
        mock_clip.duration = 5.0
        mock_clip.fps = 30
        mock_clip.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_vfc.return_value = mock_clip

        result = VideoUtils.extract_last_frame("video.mp4")

        assert isinstance(result, Image.Image)
        assert result.size == (640, 480)
        mock_clip.get_frame.assert_called_once()
        mock_clip.close.assert_called_once()

    def test_extract_last_frame_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            VideoUtils.extract_last_frame("nonexistent_video.mp4")

    def test_concatenate_videos_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            VideoUtils.concatenate_videos(["nonexistent.mp4"], "output.mp4")

    def test_add_audio_to_video_raises_on_missing_video(self):
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            VideoUtils.add_audio_to_video("nonexistent.mp4", "audio.mp3", "output.mp4")

    def test_add_audio_to_video_raises_on_missing_audio(self, tmp_path):
        video_file = tmp_path / "video.mp4"
        video_file.touch()
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            VideoUtils.add_audio_to_video(str(video_file), "nonexistent.mp3", "output.mp4")

    @patch("utils.video_utils.Path.exists", return_value=True)
    @patch("utils.video_utils.AudioFileClip")
    @patch("utils.video_utils.VideoFileClip")
    def test_add_audio_to_video_creates_video_with_audio(self, mock_vfc, mock_afc, mock_exists, tmp_path):
        mock_video_clip = MagicMock()
        mock_video_clip.fps = 24
        mock_video_with_audio = MagicMock()
        mock_video_clip.with_audio.return_value = mock_video_with_audio
        mock_vfc.return_value = mock_video_clip

        mock_audio_clip = MagicMock()
        mock_afc.return_value = mock_audio_clip

        output_file = str(tmp_path / "output_with_audio.mp4")
        result = VideoUtils.add_audio_to_video("video.mp4", "audio.mp3", output_file)

        mock_vfc.assert_called_once_with("video.mp4")
        mock_afc.assert_called_once_with("audio.mp3")
        mock_video_clip.with_audio.assert_called_once_with(mock_audio_clip)
        mock_video_with_audio.write_videofile.assert_called_once_with(output_file, fps=24, logger=None)
        assert result == output_file
        mock_audio_clip.close.assert_called_once()
        mock_video_clip.close.assert_called_once()

    @patch("utils.video_utils.Path.exists", return_value=True)
    @patch("utils.video_utils.AudioFileClip")
    @patch("utils.video_utils.VideoFileClip")
    def test_add_audio_to_video_creates_output_directory(self, mock_vfc, mock_afc, mock_exists, tmp_path):
        mock_video_clip = MagicMock()
        mock_video_clip.fps = 30
        mock_video_clip.with_audio.return_value = MagicMock()
        mock_vfc.return_value = mock_video_clip
        mock_afc.return_value = MagicMock()

        nested_output = str(tmp_path / "nested" / "dir" / "output.mp4")
        VideoUtils.add_audio_to_video("video.mp4", "audio.mp3", nested_output)

        assert (tmp_path / "nested" / "dir").is_dir()
