import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest
from PIL import Image

from scripts.create_video_from_screenplay import ScreenplayProcessor

logger = logging.getLogger(__name__)


VALID_SCREENPLAY = {
    "name": "TestMovie",
    "height": 360,
    "width": 640,
    "fps": 30,
    "negative_prompt": "blurry, low quality",
    "num_inference_steps": 30,
    "video_model": "Lightricks/LTX-Video",
    "common_prompt": "A solid black background",
    "scenes": [
        {"name": "Intro", "duration": 5, "prompt": "A sunrise"},
        {"name": "Middle", "duration": 3, "prompt": "A forest"},
    ]
}


@pytest.fixture
def screenplay_file(tmp_path):
    file_path = tmp_path / "screenplay.json"
    file_path.write_text(json.dumps(VALID_SCREENPLAY), encoding="utf-8")
    return file_path


class TestScreenplayValidation:

    def test_missing_name_raises_error(self):
        with pytest.raises(ValueError, match="name"):
            ScreenplayProcessor._validate_screenplay({"scenes": [{"prompt": "x"}]})

    def test_missing_scenes_raises_error(self):
        with pytest.raises(ValueError, match="scenes"):
            ScreenplayProcessor._validate_screenplay({"name": "Test"})

    def test_empty_scenes_raises_error(self):
        with pytest.raises(ValueError, match="at least one scene"):
            ScreenplayProcessor._validate_screenplay({"name": "Test", "scenes": []})

    def test_scene_missing_prompt_raises_error(self):
        with pytest.raises(ValueError, match="Scene 0 is missing required field: 'prompt'"):
            ScreenplayProcessor._validate_screenplay({"name": "Test", "scenes": [{"name": "x"}]})

    def test_valid_screenplay_passes(self):
        ScreenplayProcessor._validate_screenplay(VALID_SCREENPLAY)


class TestScreenplayProcessor:

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_load_screenplay(self, mock_video_utils_cls, screenplay_file):
        processor = ScreenplayProcessor(screenplay_file)
        assert processor.screenplay["name"] == "TestMovie"
        assert len(processor.screenplay["scenes"]) == 2

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_uses_video_model_from_screenplay(self, mock_video_utils_cls, screenplay_file):
        processor = ScreenplayProcessor(screenplay_file)
        assert processor.model_id == "Lightricks/LTX-Video"
        mock_video_utils_cls.assert_called_once_with("Lightricks/LTX-Video", output_path=str(processor.output_path))

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_model_id_argument_overrides_screenplay(self, mock_video_utils_cls, screenplay_file):
        processor = ScreenplayProcessor(screenplay_file, model_id="custom/model")
        assert processor.model_id == "custom/model"
        mock_video_utils_cls.assert_called_once_with("custom/model", output_path=str(processor.output_path))

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_calculate_num_frames(self, mock_video_utils_cls, screenplay_file):
        processor = ScreenplayProcessor(screenplay_file)
        assert processor._calculate_num_frames(5, 30) == 150
        assert processor._calculate_num_frames(3, 24) == 72

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_build_full_prompt_with_common_prompt(self, mock_video_utils_cls, screenplay_file):
        processor = ScreenplayProcessor(screenplay_file)
        result = processor._build_full_prompt("A sunrise")
        assert result == "A solid black background, A sunrise"

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_build_full_prompt_without_common_prompt(self, mock_video_utils_cls, tmp_path):
        screenplay = {"name": "Test", "scenes": [{"prompt": "A cat"}]}
        file_path = tmp_path / "test.json"
        file_path.write_text(json.dumps(screenplay), encoding="utf-8")
        processor = ScreenplayProcessor(file_path, output_path=str(tmp_path))
        result = processor._build_full_prompt("A cat")
        assert result == "A cat"

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_process_first_scene_uses_text_to_video(self, mock_video_utils_cls, screenplay_file, tmp_path):
        mock_video_utils = MagicMock()
        mock_video_utils.text_to_video.return_value = str(tmp_path / "TestMovie_Intro.mp4")
        mock_video_utils.image_to_video.return_value = str(tmp_path / "TestMovie_Middle.mp4")
        mock_video_utils_cls.return_value = mock_video_utils

        mock_frame = Image.new("RGB", (640, 360))
        mock_video_utils_cls.extract_last_frame.return_value = mock_frame

        processor = ScreenplayProcessor(screenplay_file, output_path=str(tmp_path))
        processor.process()

        assert mock_video_utils.text_to_video.call_count == 1
        first_call = mock_video_utils.text_to_video.call_args
        assert first_call[0][0] == "A solid black background, A sunrise"

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_process_subsequent_scenes_use_image_to_video(self, mock_video_utils_cls, screenplay_file, tmp_path):
        mock_video_utils = MagicMock()
        mock_video_utils.text_to_video.return_value = str(tmp_path / "TestMovie_Intro.mp4")
        mock_video_utils.image_to_video.return_value = str(tmp_path / "TestMovie_Middle.mp4")
        mock_video_utils_cls.return_value = mock_video_utils

        mock_frame = Image.new("RGB", (640, 360))
        mock_video_utils_cls.extract_last_frame.return_value = mock_frame

        processor = ScreenplayProcessor(screenplay_file, output_path=str(tmp_path))
        processor.process()

        assert mock_video_utils.image_to_video.call_count == 1
        i2v_call = mock_video_utils.image_to_video.call_args
        assert i2v_call[0][0] == mock_frame
        assert i2v_call[0][1] == "A solid black background, A forest"
        assert i2v_call[1]["num_frames"] == 90
        assert i2v_call[1]["num_inference_steps"] == 30

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_process_extracts_last_frame_after_each_scene(self, mock_video_utils_cls, screenplay_file, tmp_path):
        mock_video_utils = MagicMock()
        intro_path = str(tmp_path / "TestMovie_Intro.mp4")
        middle_path = str(tmp_path / "TestMovie_Middle.mp4")
        mock_video_utils.text_to_video.return_value = intro_path
        mock_video_utils.image_to_video.return_value = middle_path
        mock_video_utils_cls.return_value = mock_video_utils

        mock_frame = Image.new("RGB", (640, 360))
        mock_video_utils_cls.extract_last_frame.return_value = mock_frame

        processor = ScreenplayProcessor(screenplay_file, output_path=str(tmp_path))
        processor.process()

        assert mock_video_utils_cls.extract_last_frame.call_count == 2
        mock_video_utils_cls.extract_last_frame.assert_any_call(intro_path)
        mock_video_utils_cls.extract_last_frame.assert_any_call(middle_path)

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_process_concatenates_all_scenes(self, mock_video_utils_cls, screenplay_file, tmp_path):
        mock_video_utils = MagicMock()
        mock_video_utils.text_to_video.return_value = str(tmp_path / "TestMovie_Intro.mp4")
        mock_video_utils.image_to_video.return_value = str(tmp_path / "TestMovie_Middle.mp4")
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (640, 360))

        processor = ScreenplayProcessor(screenplay_file, output_path=str(tmp_path))
        processor.process()

        mock_video_utils_cls.concatenate_videos.assert_called_once()
        concat_args = mock_video_utils_cls.concatenate_videos.call_args
        assert len(concat_args[0][0]) == 2
        assert concat_args[0][1].endswith("TestMovie_final.mp4")
        assert concat_args[1]["fps"] == 30

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_process_uses_default_values(self, mock_video_utils_cls, tmp_path):
        minimal_screenplay = {
            "name": "Minimal",
            "scenes": [{"prompt": "A cat"}]
        }
        file_path = tmp_path / "minimal.json"
        file_path.write_text(json.dumps(minimal_screenplay), encoding="utf-8")

        mock_video_utils = MagicMock()
        mock_video_utils.text_to_video.return_value = str(tmp_path / "Minimal_scene_0.mp4")
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (704, 480))

        processor = ScreenplayProcessor(file_path, output_path=str(tmp_path))
        processor.process()

        call_kwargs = mock_video_utils.text_to_video.call_args[1]
        assert call_kwargs["height"] == 480
        assert call_kwargs["width"] == 704
        assert call_kwargs["negative_prompt"] == ""
        assert call_kwargs["num_frames"] == 120  # 5s * 24fps
        assert call_kwargs["num_inference_steps"] == 50

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_scene_overrides_global_settings(self, mock_video_utils_cls, tmp_path):
        screenplay = {
            "name": "Override",
            "height": 360,
            "width": 640,
            "fps": 30,
            "negative_prompt": "global negative",
            "scenes": [
                {
                    "prompt": "A dog",
                    "height": 720,
                    "width": 1280,
                    "fps": 60,
                    "negative_prompt": "scene negative",
                    "duration": 2,
                    "num_inference_steps": 10,
                    "seed": 42
                }
            ]
        }
        file_path = tmp_path / "override.json"
        file_path.write_text(json.dumps(screenplay), encoding="utf-8")

        mock_video_utils = MagicMock()
        mock_video_utils.text_to_video.return_value = str(tmp_path / "Override_scene_0.mp4")
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (1280, 720))

        processor = ScreenplayProcessor(file_path, output_path=str(tmp_path))
        processor.process()

        call_kwargs = mock_video_utils.text_to_video.call_args[1]
        assert call_kwargs["height"] == 720
        assert call_kwargs["width"] == 1280
        assert call_kwargs["negative_prompt"] == "scene negative"
        assert call_kwargs["num_frames"] == 120  # 2s * 60fps
        assert call_kwargs["num_inference_steps"] == 10
        assert call_kwargs["seed"] == 42

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_invalid_file_raises_error(self, mock_video_utils_cls, tmp_path):
        file_path = tmp_path / "bad.json"
        file_path.write_text("not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            ScreenplayProcessor(file_path)
