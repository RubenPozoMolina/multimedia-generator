"""Objective quality checks for generated character animations.

Qwen-Image-Edit can silently return almost identical frames: the resulting
video then looks like a static image that slowly degrades.  This module measures
motion, degradation and identity drift so a run can be repeated automatically
instead of being reviewed by hand.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FINAL_VIDEO_SUFFIXES = (
    "_frames_final_subtitles.mp4",
    "_frames_final_audio.mp4",
    "_frames_final.mp4",
)

# A pixel is considered changed when its grey level differs by more than this
# many 8-bit levels.  It filters out compression noise while keeping real
# motion, and is intentionally expressed in pixel units for readability.
CHANGE_THRESHOLD = 8


@dataclass(frozen=True)
class EvaluationThresholds:
    """Limits used to decide whether a generated video is acceptable."""

    analysis_width: int = 320
    min_motion_mean: float = 1.0
    min_changed_ratio: float = 0.01
    max_static_ratio: float = 0.5
    min_sharpness_retention: float = 0.6
    max_brightness_drift: float = 0.10
    max_saturation_drift: float = 0.25
    min_keyframe_similarity: float = 0.5
    max_frames: int = 240


@dataclass(frozen=True)
class FrameAnalysis:
    """Per-frame measurements used to build a scene verdict."""

    index: int
    mean_diff: float = 0.0
    changed_ratio: float = 0.0
    sharpness: float = 0.0
    brightness: float = 0.0
    saturation: float = 0.0
    keyframe_similarity: float = 1.0
    is_static: bool = False


@dataclass(frozen=True)
class SceneEvaluation:
    """Aggregated quality verdict for one scene or video."""

    name: str
    source: str
    frame_count: int
    analyzed_frames: int
    mean_motion: float
    min_motion: float
    static_ratio: float
    sharpness_start: float
    sharpness_end: float
    sharpness_retention: float
    brightness_drift: float
    saturation_drift: float
    keyframe_similarity: float
    problems: tuple[str, ...]
    frames: tuple[FrameAnalysis, ...] = field(default=(), repr=False)

    @property
    def needs_retry(self) -> bool:
        """Whether the scene should be generated again."""
        return bool(self.problems)

    @property
    def verdict(self) -> str:
        """Short human-readable verdict."""
        return "RETRY" if self.needs_retry else "ok"

    def as_dict(self) -> dict:
        """Return a serializable summary without the per-frame samples."""
        data = asdict(self)
        data.pop("frames", None)
        data["needs_retry"] = self.needs_retry
        data["verdict"] = self.verdict
        return data

    def format_line(self) -> str:
        """Return one aligned table row."""
        problems = "; ".join(self.problems) if self.problems else "ok"
        return (
            f"{self.name:<32} {self.analyzed_frames:>6} "
            f"{self.mean_motion:>7.2f} {self.static_ratio:>6.0%} "
            f"{self.sharpness_retention:>6.0%} {self.keyframe_similarity:>7.2f} "
            f"{problems}"
        )


@dataclass(frozen=True)
class EvaluationReport:
    """Quality report for every scene plus the final montage."""

    target: str
    scenes: tuple[SceneEvaluation, ...]
    final_video: SceneEvaluation | None = None

    @property
    def needs_retry(self) -> bool:
        """Whether at least one scene or the final video must be repeated."""
        evaluations = list(self.scenes)
        if self.final_video is not None:
            evaluations.append(self.final_video)
        return any(evaluation.needs_retry for evaluation in evaluations)

    @property
    def failed_scenes(self) -> tuple[SceneEvaluation, ...]:
        """Scenes whose quality verdict requires a retry."""
        return tuple(scene for scene in self.scenes if scene.needs_retry)

    def as_dict(self) -> dict:
        """Return a JSON-serializable report."""
        return {
            "target": self.target,
            "needs_retry": self.needs_retry,
            "scenes": [scene.as_dict() for scene in self.scenes],
            "final_video": (
                self.final_video.as_dict() if self.final_video is not None else None
            ),
        }

    def format_report(self) -> str:
        """Return a readable multi-line report."""
        header = (
            f"{'target':<32} {'frames':>6} {'motion':>7} {'static':>6} "
            f"{'sharp':>6} {'ident':>7} problems"
        )
        lines = [f"Quality report: {self.target}", header, "-" * len(header)]
        for scene in self.scenes:
            lines.append(scene.format_line())
        if self.final_video is not None:
            lines.append(self.final_video.format_line())
        lines.append("-" * len(header))
        state = "RETRY REQUIRED" if self.needs_retry else "OK"
        lines.append(f"Verdict: {state}")
        return "\n".join(lines)


class VideoQualityEvaluator:
    """Measure motion, degradation and drift in generated animations.

    The evaluator accepts image sequences or videos.  Frame directories are
    preferred because they are the exact artifacts produced by the pipeline and
    avoid an unnecessary video decode.
    """

    def __init__(
        self,
        thresholds: EvaluationThresholds | None = None,
        logger: logging.Logger | None = None,
    ):
        self.thresholds = thresholds or EvaluationThresholds()
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_frame_directory(
        self, directory: str | Path, name: str | None = None
    ) -> SceneEvaluation:
        """Evaluate every (or a sample of) frame image in a directory."""
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Frame directory not found: {path}")
        frame_paths = sorted(
            entry
            for entry in path.iterdir()
            if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES
        )
        if not frame_paths:
            raise ValueError(f"No frame images found in {path}")
        return self.evaluate_frames(frame_paths, name=name or path.name, source=str(path))

    def evaluate_frames(
        self,
        frames: Sequence[str | Path | np.ndarray | Image.Image],
        name: str = "scene",
        source: str = "",
    ) -> SceneEvaluation:
        """Evaluate an ordered sequence of frames."""
        arrays = self._load_frames(frames)
        return self._analyse(
            name=name,
            source=source or name,
            arrays=arrays,
            frame_count=len(frames),
        )

    def evaluate_video(self, video_path: str | Path, name: str | None = None) -> SceneEvaluation:
        """Evaluate a video file by sampling frames across its duration."""
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        from moviepy import VideoFileClip

        clip = VideoFileClip(str(path))
        try:
            duration = float(clip.duration or 0.0)
            fps = float(clip.fps or 0.0)
            if duration <= 0 or fps <= 0:
                raise ValueError(f"Cannot read duration or fps from {path}")
            sample_count = min(
                self.thresholds.max_frames, max(2, int(round(duration * fps)))
            )
            last_time = max(0.0, duration - 1.0 / fps)
            times = np.linspace(0.0, last_time, sample_count)
            arrays = [np.asarray(clip.get_frame(float(t)), dtype=np.uint8) for t in times]
        finally:
            clip.close()

        return self._analyse(
            name=name or path.stem,
            source=str(path),
            arrays=arrays,
            frame_count=int(round(duration * fps)),
        )

    def evaluate_output(self, output_path: str | Path) -> EvaluationReport:
        """Evaluate every scene folder and the final montage of an output run."""
        output = Path(output_path)
        evaluations = self._evaluate_scene_directories(output / "scenes")
        final_video = self._find_final_video(output)
        final_evaluation = None
        if final_video is not None:
            try:
                final_evaluation = self.evaluate_video(final_video, name=final_video.stem)
            except Exception as error:  # pragma: no cover - depends on media files
                self.logger.warning("Could not evaluate final video %s: %s", final_video, error)
        return EvaluationReport(
            target=str(output), scenes=tuple(evaluations), final_video=final_evaluation
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def _evaluate_scene_directories(self, scenes_dir: Path) -> list[SceneEvaluation]:
        if not scenes_dir.exists():
            return []

        evaluations: list[SceneEvaluation] = []
        for directory in sorted(entry for entry in scenes_dir.iterdir() if entry.is_dir()):
            frames_dir = directory / "frames"
            try:
                if frames_dir.exists():
                    evaluations.append(
                        self.evaluate_frame_directory(frames_dir, name=directory.name)
                    )
                    continue
                videos = sorted(directory.glob("*.mp4"))
                if videos:
                    evaluations.append(
                        self.evaluate_video(videos[0], name=directory.name)
                    )
            except Exception as error:  # pragma: no cover - depends on media files
                self.logger.warning("Could not evaluate scene %s: %s", directory, error)
        return evaluations

    @staticmethod
    def _find_final_video(output: Path) -> Path | None:
        for suffix in FINAL_VIDEO_SUFFIXES:
            matches = sorted(output.glob(f"*{suffix}"))
            if matches:
                return matches[0]
        return None

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def _load_frames(
        self, frames: Sequence[str | Path | np.ndarray | Image.Image]
    ) -> list[np.ndarray]:
        if not frames:
            raise ValueError("No frames to evaluate")

        sources = list(frames)
        if len(sources) > self.thresholds.max_frames:
            indices = np.linspace(
                0, len(sources) - 1, self.thresholds.max_frames
            ).round().astype(int)
            sources = [sources[index] for index in dict.fromkeys(indices)]

        return [self._load_frame(source) for source in sources]

    def _load_frame(self, source: str | Path | np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(source, np.ndarray):
            array = source
        elif isinstance(source, Image.Image):
            array = np.asarray(source.convert("RGB"), dtype=np.uint8)
        else:
            with Image.open(source) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)

        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("Frames must be RGB images")
        return array[:, :, :3]

    def _prepare(self, array: np.ndarray) -> np.ndarray:
        height, width = array.shape[:2]
        target_width = max(16, int(self.thresholds.analysis_width))
        target_height = max(16, int(round(height * target_width / max(1, width))))
        if (width, height) == (target_width, target_height):
            return array
        image = Image.fromarray(array).resize(
            (target_width, target_height), Image.Resampling.BILINEAR
        )
        return np.asarray(image, dtype=np.uint8)

    def _analyse(
        self,
        name: str,
        source: str,
        arrays: Sequence[np.ndarray],
        frame_count: int | None = None,
    ) -> SceneEvaluation:
        prepared = [self._prepare(array) for array in arrays]
        greys = [self._to_gray(array) for array in prepared]
        keyframe = greys[0]

        analyses: list[FrameAnalysis] = []
        previous: np.ndarray | None = None
        for index, (rgb, grey) in enumerate(zip(prepared, greys)):
            mean_diff = 0.0
            changed_ratio = 0.0
            is_static = False
            if previous is not None:
                diff = np.abs(grey - previous)
                mean_diff = float(diff.mean())
                changed_ratio = float((diff > CHANGE_THRESHOLD).mean())
                is_static = (
                    mean_diff < self.thresholds.min_motion_mean
                    and changed_ratio < self.thresholds.min_changed_ratio
                )

            keyframe_similarity = (
                1.0 if index == 0 else self._local_ssim(grey, keyframe)
            )
            analyses.append(
                FrameAnalysis(
                    index=index,
                    mean_diff=mean_diff,
                    changed_ratio=changed_ratio,
                    sharpness=self._sharpness(grey),
                    brightness=float(grey.mean()),
                    saturation=float((rgb.max(axis=2) - rgb.min(axis=2)).mean()),
                    keyframe_similarity=keyframe_similarity,
                    is_static=is_static,
                )
            )
            previous = grey

        return self._summarise(
            name=name,
            source=source,
            analyses=analyses,
            frame_count=frame_count if frame_count is not None else len(analyses),
        )

    def _summarise(
        self,
        name: str,
        source: str,
        analyses: Sequence[FrameAnalysis],
        frame_count: int,
    ) -> SceneEvaluation:
        pairs = list(analyses[1:])
        motion = [analysis.mean_diff for analysis in pairs]
        static_ratio = (
            sum(1 for analysis in pairs if analysis.is_static) / len(pairs)
            if pairs
            else 1.0
        )
        mean_motion = float(np.mean(motion)) if motion else 0.0
        min_motion = float(np.min(motion)) if motion else 0.0

        window = max(1, len(analyses) // 4)
        sharpness_start = self._median(
            [analysis.sharpness for analysis in analyses[:window]]
        )
        sharpness_end = self._median(
            [analysis.sharpness for analysis in analyses[-window:]]
        )
        sharpness_retention = (
            sharpness_end / sharpness_start if sharpness_start > 1e-6 else 1.0
        )
        brightness_drift = abs(
            self._median([analysis.brightness for analysis in analyses[-window:]])
            - self._median([analysis.brightness for analysis in analyses[:window]])
        ) / 255.0
        saturation_drift = abs(
            self._median([analysis.saturation for analysis in analyses[-window:]])
            - self._median([analysis.saturation for analysis in analyses[:window]])
        ) / 255.0
        keyframe_similarity = self._median(
            [analysis.keyframe_similarity for analysis in analyses[-window:]]
        )

        problems = self._collect_problems(
            analyzed_frames=len(analyses),
            static_ratio=static_ratio,
            sharpness_retention=sharpness_retention,
            brightness_drift=brightness_drift,
            saturation_drift=saturation_drift,
            keyframe_similarity=keyframe_similarity,
        )
        return SceneEvaluation(
            name=name,
            source=source,
            frame_count=frame_count,
            analyzed_frames=len(analyses),
            mean_motion=mean_motion,
            min_motion=min_motion,
            static_ratio=static_ratio,
            sharpness_start=sharpness_start,
            sharpness_end=sharpness_end,
            sharpness_retention=sharpness_retention,
            brightness_drift=brightness_drift,
            saturation_drift=saturation_drift,
            keyframe_similarity=keyframe_similarity,
            problems=tuple(problems),
            frames=tuple(analyses),
        )

    def _collect_problems(
        self,
        analyzed_frames: int,
        static_ratio: float,
        sharpness_retention: float,
        brightness_drift: float,
        saturation_drift: float,
        keyframe_similarity: float,
    ) -> list[str]:
        thresholds = self.thresholds
        if analyzed_frames < 3:
            return ["insufficient_frames"]

        problems: list[str] = []
        if static_ratio > thresholds.max_static_ratio:
            problems.append(f"static_motion ({static_ratio:.0%} identical frames)")
        if sharpness_retention < thresholds.min_sharpness_retention:
            problems.append(f"degraded_sharpness (retention {sharpness_retention:.0%})")
        if brightness_drift > thresholds.max_brightness_drift:
            problems.append(f"brightness_drift ({brightness_drift:.0%})")
        if saturation_drift > thresholds.max_saturation_drift:
            problems.append(f"color_drift ({saturation_drift:.0%})")
        if keyframe_similarity < thresholds.min_keyframe_similarity:
            problems.append(f"identity_drift (similarity {keyframe_similarity:.2f})")
        return problems

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_gray(rgb: np.ndarray) -> np.ndarray:
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        return rgb.astype(np.float32) @ weights

    @staticmethod
    def _sharpness(grey: np.ndarray) -> float:
        """Variance of the Laplacian: lower values mean a blurrier frame."""
        if grey.shape[0] < 3 or grey.shape[1] < 3:
            return 0.0
        laplacian = (
            -4.0 * grey[1:-1, 1:-1]
            + grey[:-2, 1:-1]
            + grey[2:, 1:-1]
            + grey[1:-1, :-2]
            + grey[1:-1, 2:]
        )
        return float(laplacian.var())

    @staticmethod
    def _pearson(first: np.ndarray, second: np.ndarray) -> float:
        """Normalized correlation used as a fallback for tiny images."""
        left = first - first.mean()
        right = second - second.mean()
        denominator = float(np.sqrt((left * left).sum() * (right * right).sum()))
        if denominator < 1e-8:
            return 1.0 if np.allclose(first, second) else 0.0
        return float((left * right).sum() / denominator)

    @classmethod
    def _local_ssim(cls, first: np.ndarray, second: np.ndarray, window: int = 8) -> float:
        """Local structural similarity, robust to localized motion.

        A global correlation drops when a character moves to a different part of
        the frame, even if its identity is intact.  Averaging SSIM over local
        windows keeps unchanged regions dominant while still detecting global
        style or identity changes.
        """
        window = max(1, min(window, first.shape[0], first.shape[1]))
        if window < 2:
            return cls._pearson(first, second)

        first = first.astype(np.float32)
        second = second.astype(np.float32)
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        mean_first = cls._box_mean(first, window)
        mean_second = cls._box_mean(second, window)
        variance_first = cls._box_mean(first * first, window) - mean_first**2
        variance_second = cls._box_mean(second * second, window) - mean_second**2
        covariance = cls._box_mean(first * second, window) - mean_first * mean_second

        numerator = (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
        denominator = (
            (mean_first**2 + mean_second**2 + c1)
            * (variance_first + variance_second + c2)
        )
        return float(np.mean(numerator / denominator))

    @staticmethod
    def _box_mean(array: np.ndarray, window: int) -> np.ndarray:
        """Mean over a square sliding window, computed with a view."""
        pad_before = (window - 1) // 2
        pad_after = window - 1 - pad_before
        padded = np.pad(
            array,
            ((pad_before, pad_after), (pad_before, pad_after)),
            mode="reflect",
        )
        windows = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
        return windows.mean(axis=(-1, -2), dtype=np.float32)

    @staticmethod
    def _median(values: Sequence[float]) -> float:
        return float(np.median(values)) if values else 0.0
