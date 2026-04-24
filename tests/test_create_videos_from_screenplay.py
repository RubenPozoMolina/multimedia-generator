import json
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest
from PIL import Image

from scripts.create_video_from_screenplay import ScreenplayProcessor, _calculate_num_frames

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

    def test_load_screenplay(self, screenplay_file):
        processor = ScreenplayProcessor(screenplay_file)
        assert processor.screenplay["name"] == "TestMovie"
        assert len(processor.screenplay["scenes"]) == 2

    def test_uses_video_model_from_screenplay(self, screenplay_file):
        processor = ScreenplayProcessor(screenplay_file)
        assert processor.model_id == "Lightricks/LTX-Video"
        assert processor.video_utils is None

    def test_calculate_num_frames(self):
        assert _calculate_num_frames(5, 30) == 150
        assert _calculate_num_frames(3, 24) == 72

    def test_build_full_prompt_with_common_prompt(self, screenplay_file):
        processor = ScreenplayProcessor(screenplay_file)
        result = processor._build_full_prompt("A sunrise")
        assert result == "A solid black background, A sunrise"

    def test_build_full_prompt_without_common_prompt(self, tmp_path):
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

    @patch("scripts.create_video_from_screenplay.ImageUtils")
    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_first_scene_uses_image_model_when_specified(self, mock_video_utils_cls, mock_image_utils_cls, tmp_path):
        screenplay = {
            "name": "ImageModelTest",
            "height": 360,
            "width": 640,
            "fps": 30,
            "image_model": "black-forest-labs/FLUX.1-schnell",
            "scenes": [
                {"name": "Intro", "duration": 5, "prompt": "A sunrise"},
                {"name": "Middle", "duration": 3, "prompt": "A forest"},
            ]
        }
        file_path = tmp_path / "screenplay.json"
        file_path.write_text(json.dumps(screenplay), encoding="utf-8")

        mock_image_utils = MagicMock()
        def create_and_return_image(*args, **kwargs):
            path = kwargs.get("output_path", "")
            Image.new("RGB", (640, 360)).save(path)
            return path
        mock_image_utils.text_to_image.side_effect = create_and_return_image
        mock_image_utils_cls.return_value = mock_image_utils

        mock_video_utils = MagicMock()
        mock_video_utils.image_to_video.side_effect = [
            str(tmp_path / "ImageModelTest_Intro.mp4"),
            str(tmp_path / "ImageModelTest_Middle.mp4"),
        ]
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (640, 360))

        processor = ScreenplayProcessor(file_path, output_path=str(tmp_path))
        processor.process()

        assert mock_image_utils_cls.call_count == 2
        assert mock_image_utils.text_to_image.call_count == 2
        assert mock_video_utils.text_to_video.call_count == 0
        assert mock_video_utils.image_to_video.call_count == 2

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_first_scene_uses_text_to_video_without_image_model(self, mock_video_utils_cls, screenplay_file, tmp_path):
        mock_video_utils = MagicMock()
        mock_video_utils.text_to_video.return_value = str(tmp_path / "TestMovie_Intro.mp4")
        mock_video_utils.image_to_video.return_value = str(tmp_path / "TestMovie_Middle.mp4")
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (640, 360))

        processor = ScreenplayProcessor(screenplay_file, output_path=str(tmp_path))
        processor.process()

        assert mock_video_utils.text_to_video.call_count == 1
        assert mock_video_utils.image_to_video.call_count == 1

    @patch.object(ScreenplayProcessor, "_get_audio_duration", return_value=5.0)
    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_process_adds_audio_when_specified(self, mock_video_utils_cls, mock_audio_dur, tmp_path):
        screenplay = {
            "name": "AudioTest",
            "scenes": [{"prompt": "A cat", "duration": 5}]
        }
        file_path = tmp_path / "screenplay.json"
        file_path.write_text(json.dumps({**screenplay, "audio": "song.mp3"}), encoding="utf-8")

        mock_video_utils = MagicMock()
        mock_video_utils.text_to_video.return_value = str(tmp_path / "AudioTest_scene_0.mp4")
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (704, 480))
        mock_video_utils_cls.concatenate_videos.return_value = str(tmp_path / "AudioTest_final.mp4")
        mock_video_utils_cls.add_audio_to_video.return_value = str(tmp_path / "AudioTest_final_audio.mp4")

        processor = ScreenplayProcessor(file_path, output_path=str(tmp_path))
        result = processor.process()

        mock_video_utils_cls.add_audio_to_video.assert_called_once_with(
            str(tmp_path / "AudioTest_final.mp4"),
            str(tmp_path / "song.mp3"),
            str(tmp_path / "AudioTest_final_audio.mp4")
        )
        assert result == str(tmp_path / "AudioTest_final_audio.mp4")

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_process_skips_audio_when_not_specified(self, mock_video_utils_cls, screenplay_file, tmp_path):
        mock_video_utils = MagicMock()
        mock_video_utils.text_to_video.return_value = str(tmp_path / "TestMovie_Intro.mp4")
        mock_video_utils.image_to_video.return_value = str(tmp_path / "TestMovie_Middle.mp4")
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (640, 360))
        mock_video_utils_cls.concatenate_videos.return_value = str(tmp_path / "TestMovie_final.mp4")

        processor = ScreenplayProcessor(screenplay_file, output_path=str(tmp_path))
        result = processor.process()

        mock_video_utils_cls.add_audio_to_video.assert_not_called()
        assert result == str(tmp_path / "TestMovie_final.mp4")

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_long_scene_generates_single_chunk(self, mock_video_utils_cls, tmp_path):
        screenplay = {
            "name": "LongScene",
            "fps": 30,
            "scenes": [{"prompt": "A landscape", "duration": 10}]
        }
        file_path = tmp_path / "screenplay.json"
        file_path.write_text(json.dumps(screenplay), encoding="utf-8")

        mock_video_utils = MagicMock()
        mock_video_utils.text_to_video.return_value = str(tmp_path / "chunk0.mp4")
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (704, 480))
        mock_video_utils_cls.concatenate_videos.return_value = str(tmp_path / "LongScene_final.mp4")

        processor = ScreenplayProcessor(file_path, output_path=str(tmp_path))
        processor.process()

        assert mock_video_utils.text_to_video.call_count == 1
        assert mock_video_utils.text_to_video.call_args[1]["num_frames"] == 300

    @patch("scripts.create_video_from_screenplay.ImageUtils")
    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_all_images_generated_before_videos(self, mock_video_utils_cls, mock_image_utils_cls, tmp_path):
        screenplay = {
            "name": "OrderTest",
            "image_model": "black-forest-labs/FLUX.1-schnell",
            "scenes": [
                {"name": "Scene1", "duration": 3, "prompt": "A mountain"},
                {"name": "Scene2", "duration": 3, "prompt": "A river"},
                {"name": "Scene3", "duration": 3, "prompt": "A valley"},
            ]
        }
        file_path = tmp_path / "screenplay.json"
        file_path.write_text(json.dumps(screenplay), encoding="utf-8")

        call_order = []

        mock_image_utils = MagicMock()
        def track_image_call(*args, **kwargs):
            scene_name = kwargs.get("output_path", "").split("/")[-1].replace("_frame.png", "")
            call_order.append(f"image_{scene_name}")
            image_path = str(tmp_path / f"{scene_name}_frame.png")
            Image.new("RGB", (704, 480)).save(image_path)
            return image_path
        mock_image_utils.text_to_image.side_effect = track_image_call
        mock_image_utils_cls.return_value = mock_image_utils

        mock_video_utils = MagicMock()
        def track_video_call(*args, **kwargs):
            call_order.append("video")
            return str(tmp_path / "video.mp4")
        mock_video_utils.image_to_video.side_effect = track_video_call
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (704, 480))

        processor = ScreenplayProcessor(file_path, output_path=str(tmp_path))
        processor.process()

        image_calls = [c for c in call_order if c.startswith("image_")]
        video_calls = [c for c in call_order if c == "video"]
        assert len(image_calls) == 3
        assert len(video_calls) == 3
        last_image_index = max(i for i, c in enumerate(call_order) if c.startswith("image_"))
        first_video_index = min(i for i, c in enumerate(call_order) if c == "video")
        assert last_image_index < first_video_index

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_invalid_file_raises_error(self, mock_video_utils_cls, tmp_path):
        file_path = tmp_path / "bad.json"
        file_path.write_text("not json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            ScreenplayProcessor(file_path)

    def test_recalculate_scene_durations_scales_proportionally(self):
        configs = [
            {"scene_name": "A", "duration": 10},
            {"scene_name": "B", "duration": 10},
        ]
        ScreenplayProcessor._recalculate_scene_durations(configs, 30.0)
        assert configs[0]["duration"] == 15.0
        assert configs[1]["duration"] == 15.0

    def test_recalculate_scene_durations_last_scene_absorbs_remainder(self):
        configs = [
            {"scene_name": "A", "duration": 10},
            {"scene_name": "B", "duration": 10},
            {"scene_name": "C", "duration": 10},
        ]
        ScreenplayProcessor._recalculate_scene_durations(configs, 100.0)
        total = sum(c["duration"] for c in configs)
        assert total == 100.0

    def test_recalculate_scene_durations_single_scene(self):
        configs = [{"scene_name": "Only", "duration": 5}]
        ScreenplayProcessor._recalculate_scene_durations(configs, 12.5)
        assert configs[0]["duration"] == 12.5

    def test_recalculate_scene_durations_zero_total_skips(self):
        configs = [{"scene_name": "A", "duration": 0}]
        ScreenplayProcessor._recalculate_scene_durations(configs, 10.0)
        assert configs[0]["duration"] == 0

    @patch.object(ScreenplayProcessor, "_get_audio_duration", return_value=15.0)
    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_process_recalculates_durations_when_audio_present(self, mock_video_utils_cls, mock_audio_dur, tmp_path):
        screenplay = {
            "name": "RecalcTest",
            "fps": 10,
            "audio": "song.mp3",
            "scenes": [
                {"prompt": "Scene A", "duration": 4},
                {"prompt": "Scene B", "duration": 6},
            ]
        }
        file_path = tmp_path / "screenplay.json"
        file_path.write_text(json.dumps(screenplay), encoding="utf-8")

        mock_video_utils = MagicMock()
        mock_video_utils.text_to_video.return_value = str(tmp_path / "scene0.mp4")
        mock_video_utils.image_to_video.return_value = str(tmp_path / "scene1.mp4")
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (704, 480))
        mock_video_utils_cls.concatenate_videos.return_value = str(tmp_path / "final.mp4")
        mock_video_utils_cls.add_audio_to_video.return_value = str(tmp_path / "final_audio.mp4")

        processor = ScreenplayProcessor(file_path, output_path=str(tmp_path))
        processor.process()

        first_call_frames = mock_video_utils.text_to_video.call_args[1]["num_frames"]
        second_call_frames = mock_video_utils.image_to_video.call_args[1]["num_frames"]
        assert first_call_frames == 60
        assert second_call_frames == 90


class TestSkipExistingFiles:

    @patch("scripts.create_video_from_screenplay.ImageUtils")
    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_skip_image_generation_when_file_exists(self, mock_video_utils_cls, mock_image_utils_cls, tmp_path):
        screenplay = {
            "name": "SkipImg",
            "image_model": "some-model",
            "scenes": [
                {"name": "existing_scene", "duration": 3, "prompt": "A cat"},
            ]
        }
        file_path = tmp_path / "screenplay.json"
        file_path.write_text(json.dumps(screenplay), encoding="utf-8")

        existing_image = tmp_path / "existing_scene_frame.png"
        Image.new("RGB", (704, 480)).save(str(existing_image))

        mock_video_utils = MagicMock()
        mock_video_utils.image_to_video.return_value = str(tmp_path / "video.mp4")
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (704, 480))
        mock_video_utils_cls.concatenate_videos.return_value = str(tmp_path / "final.mp4")

        processor = ScreenplayProcessor(file_path, output_path=str(tmp_path))
        processor.process()

        mock_image_utils_cls.assert_not_called()

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_skip_video_chunk_when_file_exists(self, mock_video_utils_cls, tmp_path):
        screenplay = {
            "name": "SkipVid",
            "scenes": [
                {"name": "scene_a", "duration": 3, "prompt": "A dog"},
            ]
        }
        file_path = tmp_path / "screenplay.json"
        file_path.write_text(json.dumps(screenplay), encoding="utf-8")

        existing_chunk = tmp_path / "SkipVid_scene_a_chunk0.mp4"
        existing_chunk.write_text("fake video")

        mock_video_utils = MagicMock()
        mock_video_utils_cls.return_value = mock_video_utils
        mock_frame = Image.new("RGB", (704, 480))
        mock_video_utils_cls.extract_last_frame.return_value = mock_frame
        mock_video_utils_cls.concatenate_videos.return_value = str(tmp_path / "final.mp4")

        processor = ScreenplayProcessor(file_path, output_path=str(tmp_path))
        processor.process()

        mock_video_utils.text_to_video.assert_not_called()
        mock_video_utils.image_to_video.assert_not_called()

    @patch("scripts.create_video_from_screenplay.VideoUtils")
    def test_generates_video_when_chunk_does_not_exist(self, mock_video_utils_cls, tmp_path):
        screenplay = {
            "name": "GenVid",
            "scenes": [
                {"name": "new_scene", "duration": 3, "prompt": "A bird"},
            ]
        }
        file_path = tmp_path / "screenplay.json"
        file_path.write_text(json.dumps(screenplay), encoding="utf-8")

        mock_video_utils = MagicMock()
        mock_video_utils.text_to_video.return_value = str(tmp_path / "GenVid_new_scene_chunk0.mp4")
        mock_video_utils_cls.return_value = mock_video_utils
        mock_video_utils_cls.extract_last_frame.return_value = Image.new("RGB", (704, 480))
        mock_video_utils_cls.concatenate_videos.return_value = str(tmp_path / "final.mp4")

        processor = ScreenplayProcessor(file_path, output_path=str(tmp_path))
        processor.process()

        mock_video_utils.text_to_video.assert_called_once()
