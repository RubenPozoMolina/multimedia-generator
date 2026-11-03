import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageFilter

from utils.character_animation import (
    CharacterAnimationProcessor,
    CharacterScreenplay,
    EvaluationReport,
    EvaluationThresholds,
    FrameAnimator,
    SceneEvaluation,
    SceneSpec,
    VideoQualityEvaluator,
)


def write_screenplay(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "screenplay.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def save_frames(directory: Path, frames) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray(frame).save(directory / f"{index:06d}.png")
    return directory


def moving_square_frames(count=8, size=64, brightness_step=0):
    """Frames with a white square moving horizontally over a grey background."""
    frames = []
    for index in range(count):
        base = 30 + index * brightness_step
        array = np.full((size, size, 3), base, dtype=np.uint8)
        left = 2 + index * 5
        array[20:44, left:left + 10] = 220
        frames.append(array)
    return frames


def checkerboard(size=96, block=8, inverted=False):
    rows, columns = size, size
    pattern = (np.indices((rows, columns)).sum(axis=0) // block) % 2
    if inverted:
        pattern = 1 - pattern
    grey = (pattern * 220 + 20).astype(np.uint8)
    return np.repeat(grey[:, :, None], 3, axis=2)


def blur(array, radius=10):
    image = Image.fromarray(array).filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(image, dtype=np.uint8)


class TestVideoQualityEvaluator:

    def test_static_frames_require_retry(self, tmp_path):
        frames = [np.full((48, 64, 3), 40, dtype=np.uint8) for _ in range(6)]
        directory = save_frames(tmp_path / "frames", frames)

        evaluation = VideoQualityEvaluator().evaluate_frame_directory(directory)

        assert evaluation.needs_retry
        assert evaluation.static_ratio == 1.0
        assert any(problem.startswith("static_motion") for problem in evaluation.problems)

    def test_moving_frames_pass(self, tmp_path):
        directory = save_frames(tmp_path / "frames", moving_square_frames())

        evaluation = VideoQualityEvaluator().evaluate_frame_directory(directory)

        assert not evaluation.needs_retry, evaluation.problems
        assert evaluation.static_ratio < 0.5
        assert evaluation.mean_motion > 1.0

    def test_degraded_sharpness_is_detected(self, tmp_path):
        sharp = [
            np.roll(checkerboard(), index * 3, axis=1) for index in range(3)
        ]
        blurred = [blur(np.roll(checkerboard(), index * 3, axis=1)) for index in range(3)]
        directory = save_frames(tmp_path / "frames", sharp + blurred)

        evaluation = VideoQualityEvaluator().evaluate_frame_directory(directory)

        assert evaluation.sharpness_retention < 0.6
        assert any(problem.startswith("degraded_sharpness") for problem in evaluation.problems)

    def test_brightness_drift_is_detected(self, tmp_path):
        directory = save_frames(
            tmp_path / "frames", moving_square_frames(count=6, brightness_step=30)
        )

        evaluation = VideoQualityEvaluator().evaluate_frame_directory(directory)

        assert any(problem.startswith("brightness_drift") for problem in evaluation.problems)

    def test_identity_drift_is_detected(self, tmp_path):
        frames = []
        for index in range(6):
            frame = np.roll(checkerboard(), index * 3, axis=1)
            if index >= 3:
                frame = 255 - frame
            frames.append(frame)
        directory = save_frames(tmp_path / "frames", frames)

        evaluation = VideoQualityEvaluator().evaluate_frame_directory(directory)

        assert evaluation.keyframe_similarity < 0.5
        assert any(problem.startswith("identity_drift") for problem in evaluation.problems)

    def test_insufficient_frames_are_rejected(self, tmp_path):
        directory = save_frames(
            tmp_path / "frames",
            [np.full((32, 32, 3), 10, dtype=np.uint8)],
        )

        evaluation = VideoQualityEvaluator().evaluate_frame_directory(directory)

        assert evaluation.problems == ("insufficient_frames",)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VideoQualityEvaluator().evaluate_frame_directory(tmp_path / "missing")

    def test_frames_are_read_in_filename_order(self, tmp_path):
        directory = tmp_path / "frames"
        directory.mkdir()
        for index in (2, 0, 1):
            array = moving_square_frames()[index]
            Image.fromarray(array).save(directory / f"{index:06d}.png")

        evaluation = VideoQualityEvaluator().evaluate_frame_directory(directory)

        assert evaluation.analyzed_frames == 3
        assert [frame.index for frame in evaluation.frames] == [0, 1, 2]

    def test_sampling_limits_analysed_frames(self, tmp_path):
        frames = [np.full((48, 64, 3), value, dtype=np.uint8) for value in range(20)]
        directory = save_frames(tmp_path / "frames", frames)
        evaluator = VideoQualityEvaluator(EvaluationThresholds(max_frames=5))

        evaluation = evaluator.evaluate_frame_directory(directory)

        assert evaluation.analyzed_frames == 5
        assert evaluation.frame_count == 20

    def test_video_is_sampled_with_moviepy(self, tmp_path):
        clip = MagicMock()
        clip.duration = 1.0
        clip.fps = 6.0
        clip.get_frame.side_effect = lambda time: moving_square_frames()[3]
        video_path = tmp_path / "scene.mp4"
        video_path.write_bytes(b"video")

        with patch("moviepy.VideoFileClip", return_value=clip):
            evaluation = VideoQualityEvaluator().evaluate_video(video_path, name="scene")

        assert evaluation.name == "scene"
        assert evaluation.analyzed_frames == 6
        assert clip.get_frame.call_count == 6
        clip.close.assert_called_once()

    def test_output_directory_discovers_scenes(self, tmp_path):
        frames = [np.full((48, 64, 3), 40, dtype=np.uint8) for _ in range(4)]
        save_frames(tmp_path / "out" / "scenes" / "01_opening" / "frames", frames)

        report = VideoQualityEvaluator().evaluate_output(tmp_path / "out")

        assert isinstance(report, EvaluationReport)
        assert len(report.scenes) == 1
        assert report.scenes[0].name == "01_opening"
        assert report.needs_retry
        assert report.failed_scenes[0].name == "01_opening"
        assert "RETRY REQUIRED" in report.format_report()

    def test_report_is_serializable(self, tmp_path):
        directory = save_frames(tmp_path / "frames", moving_square_frames())
        evaluation = VideoQualityEvaluator().evaluate_frame_directory(directory)
        report = EvaluationReport(target="demo", scenes=(evaluation,))

        data = report.as_dict()

        assert data["target"] == "demo"
        assert data["needs_retry"] is False
        assert data["scenes"][0]["verdict"] == "ok"
        assert "frames" not in data["scenes"][0]


def evaluation(name, problems=()):
    return SceneEvaluation(
        name=name,
        source=name,
        frame_count=4,
        analyzed_frames=4,
        mean_motion=0.0 if problems else 4.0,
        min_motion=0.0 if problems else 2.0,
        static_ratio=1.0 if problems else 0.0,
        sharpness_start=10.0,
        sharpness_end=10.0,
        sharpness_retention=1.0,
        brightness_drift=0.0,
        saturation_drift=0.0,
        keyframe_similarity=1.0,
        problems=tuple(problems),
    )


class FlakyEvaluator:
    """Returns a failure first and a pass afterwards."""

    def __init__(self):
        self.calls = 0

    def evaluate_frame_directory(self, directory, name=None):
        self.calls += 1
        if self.calls == 1:
            return evaluation(name or "scene", ("static_motion (100% identical frames)",))
        return evaluation(name or "scene")


class RecordingAnimator:
    def __init__(self, size=(64, 36)):
        self.size = size
        self.calls = []

    def animate_scene(self, scene, keyframe_path, frames_dir):
        self.calls.append(scene)
        frames_dir = Path(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for index in range(scene.frame_count):
            path = frames_dir / f"{index:06d}.png"
            Image.new("RGB", self.size, color=(index * 10, 20, 30)).save(path)
            frames.append(str(path))
        return frames

    def create_keyframe(self, scene, references, descriptions, style_prompt, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", self.size).save(output_path)
        return str(output_path)

    def close(self):
        pass


SCREENPLAY = {
    "name": "QualityDemo",
    "width": 64,
    "height": 36,
    "fps": 2,
    "characters": [{"name": "Hero", "description": "a knight"}],
    "scenes": [
        {
            "name": "Opening",
            "prompt": "a castle",
            "duration": 2,
            "characters": ["Hero"],
            "seed": 7,
        }
    ],
}


class TestProcessorRetry:

    def test_scene_is_repeated_with_strong_motion(self, tmp_path):
        screenplay = write_screenplay(tmp_path, SCREENPLAY)
        animator = RecordingAnimator()
        evaluator = FlakyEvaluator()
        processor = CharacterAnimationProcessor(
            screenplay,
            output_path=tmp_path / "out",
            frame_animator=animator,
            quality_evaluator=evaluator,
            max_retries=1,
        )
        scene = processor.screenplay.build_scenes()[0]
        processor.keyframe_path(scene).parent.mkdir(parents=True)
        Image.new("RGB", (64, 36)).save(processor.keyframe_path(scene))

        frames = processor._animate_with_retries(scene, animator)

        assert len(animator.calls) == 2
        assert animator.calls[0].motion_strength == "normal"
        assert animator.calls[1].motion_strength == "strong"
        assert animator.calls[1].seed != scene.seed
        assert evaluator.calls == 2
        assert len(frames) == scene.frame_count
        assert processor.last_evaluations[-1].needs_retry is False

    def test_retries_are_bounded(self, tmp_path):
        screenplay = write_screenplay(tmp_path, SCREENPLAY)
        animator = RecordingAnimator()
        processor = CharacterAnimationProcessor(
            screenplay,
            output_path=tmp_path / "out",
            frame_animator=animator,
            quality_evaluator=FlakyEvaluator(),
            max_retries=0,
        )
        scene = processor.screenplay.build_scenes()[0]
        processor.keyframe_path(scene).parent.mkdir(parents=True)
        Image.new("RGB", (64, 36)).save(processor.keyframe_path(scene))

        processor._animate_with_retries(scene, animator)

        assert len(animator.calls) == 1
        assert processor.last_evaluations[-1].needs_retry is True


class TestMotionStrength:

    def test_screenplay_accepts_motion_strength(self, tmp_path):
        data = {**SCREENPLAY, "motion_strength": "strong"}
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, data))
        assert screenplay.build_scenes()[0].motion_strength == "strong"

    def test_scene_overrides_motion_strength(self, tmp_path):
        data = {
            **SCREENPLAY,
            "motion_strength": "strong",
            "scenes": [{**SCREENPLAY["scenes"][0], "motion_strength": "normal"}],
        }
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, data))
        assert screenplay.build_scenes()[0].motion_strength == "normal"

    def test_invalid_motion_strength_raises(self, tmp_path):
        data = {**SCREENPLAY, "motion_strength": "wild"}
        with pytest.raises(ValueError, match="motion_strength"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_strong_motion_changes_frame_prompt(self):
        base = dict(
            index=0,
            name="Opening",
            prompt="a castle",
            duration=1,
            fps=2,
            height=36,
            width=64,
            negative_prompt="",
            num_inference_steps=4,
            true_cfg_scale=4.0,
            seed=1,
            character_names=("Hero",),
            motion_prompt="the knight runs",
            anchor_interval=2,
        )
        normal = FrameAnimator.build_frame_prompt(SceneSpec(**base), 0.5)
        strong = FrameAnimator.build_frame_prompt(
            SceneSpec(**base, motion_strength="strong"), 0.5
        )

        assert "only slightly" in normal
        assert "clearly different animation moment" in strong
        assert "only slightly" not in strong
