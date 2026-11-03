"""Orchestrator for character-consistent, frame-by-frame animation."""

from __future__ import annotations

import logging
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from moviepy import AudioFileClip, VideoFileClip
from PIL import Image

from utils.character_animation.character_designer import CharacterDesigner
from utils.character_animation.frame_animator import FrameAnimator
from utils.character_animation.frame_video_builder import FrameVideoBuilder
from utils.character_animation.keyframe_video_engine import KeyframeVideoEngine
from utils.character_animation.quality_evaluator import (
    EvaluationReport,
    SceneEvaluation,
    VideoQualityEvaluator,
)
from utils.character_animation.qwen_edit_engine import QwenImageEditEngine
from utils.character_animation.screenplay import (
    CharacterScreenplay,
    CharacterSpec,
    SceneSpec,
    join_prompts,
    scale_scene_durations,
)
from utils.video_utils import VideoUtils

logger = logging.getLogger(__name__)


class CharacterAnimationProcessor:
    """Generate a character-driven video using frame or keyframe animation.

    Pipeline stages:

    1. Create the character reference sheets once, at the beginning.
    2. Create every scene keyframe from those references.
    3. Either paint frames with Qwen-Image-Edit or generate a clip from scene
       start/end keyframes with an image-to-video model.
    4. Concatenate the scene clips and add audio/subtitles.

    Every generated artifact is cached on disk, so an interrupted run resumes
    from the last completed frame.
    """

    def __init__(
        self,
        screenplay: str | Path,
        output_path: str | Path = "output",
        logger: logging.Logger | None = None,
        character_model: str | None = None,
        frame_model: str | None = None,
        character_designer: Any | None = None,
        frame_animator: Any | None = None,
        video_builder: Any | None = None,
        video_utils: Any | None = None,
        quality_evaluator: Any | None = None,
        max_retries: int = 0,
        evaluate_output: bool = True,
        video_engine: Any | None = None,
        animation_mode: str | None = None,
        video_model: str | None = None,
    ):
        self.screenplay = CharacterScreenplay.load(screenplay, output_path)
        self.output_path = Path(output_path)
        self.logger = logger or logging.getLogger(__name__)

        self.character_model = character_model or self.screenplay.character_image_model
        self.frame_model = frame_model or self.screenplay.frame_image_model
        self.animation_mode = animation_mode or ("video" if video_model else None)
        if self.animation_mode is not None:
            self.animation_mode = str(self.animation_mode).strip().lower()
            if self.animation_mode not in {"frames", "video"}:
                raise ValueError(f"Invalid animation_mode: '{self.animation_mode}'")
        self.video_model = video_model

        self.character_designer = character_designer
        self.frame_animator = frame_animator
        self.video_engine = video_engine
        self.video_builder = video_builder or FrameVideoBuilder()
        self.video_utils = video_utils or VideoUtils
        self.quality_evaluator = quality_evaluator
        self.max_retries = max(0, int(max_retries))
        self.evaluate_output = evaluate_output
        self.last_evaluations: list[SceneEvaluation] = []

        self._owns_designer = False
        self._owns_animator = False
        self._owns_video_engine = False
        self._video_engine_model_id: str | None = None

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    @property
    def characters_dir(self) -> Path:
        return self.output_path / "characters"

    def scene_dir(self, scene: SceneSpec) -> Path:
        return self.output_path / "scenes" / f"{scene.index:02d}_{scene.slug}"

    def keyframe_path(self, scene: SceneSpec) -> Path:
        return self.scene_dir(scene) / "keyframe.png"

    def end_keyframe_path(self, scene: SceneSpec) -> Path:
        return self.scene_dir(scene) / "end_keyframe.png"

    def frames_dir(self, scene: SceneSpec) -> Path:
        return self.scene_dir(scene) / "frames"

    def scene_video_path(self, scene: SceneSpec) -> Path:
        suffix = "_video" if scene.animation_mode == "video" else ""
        return self.scene_dir(scene) / f"{scene.slug}{suffix}.mp4"

    def final_video_path(self) -> Path:
        return self.output_path / f"{self.screenplay.name}_{self._output_mode_suffix()}_final.mp4"

    def final_audio_path(self) -> Path:
        return self.output_path / f"{self.screenplay.name}_{self._output_mode_suffix()}_final_audio.mp4"

    def final_subtitles_path(self) -> Path:
        return self.output_path / f"{self.screenplay.name}_{self._output_mode_suffix()}_final_subtitles.mp4"

    def _output_mode_suffix(self) -> str:
        scenes = self.screenplay.build_scenes()
        self._apply_animation_overrides(scenes)
        return "video" if any(scene.animation_mode == "video" for scene in scenes) else "frames"

    # ------------------------------------------------------------------
    # Lazy services
    # ------------------------------------------------------------------
    def _get_designer(self) -> Any:
        if self.character_designer is None:
            self.character_designer = CharacterDesigner(
                self.character_model,
                output_path=self.output_path,
                logger=self.logger,
            )
            self._owns_designer = True
        return self.character_designer

    def _get_animator(self) -> Any:
        if self.frame_animator is None:
            engine = QwenImageEditEngine(self.frame_model, logger=self.logger)
            self.frame_animator = FrameAnimator(engine, logger=self.logger)
            self._owns_animator = True
        return self.frame_animator

    def _get_video_engine(self, model_id: str) -> Any:
        if self.video_engine is not None and not self._owns_video_engine:
            return self.video_engine
        if (
            self.video_engine is not None
            and self._video_engine_model_id == model_id
        ):
            return self.video_engine
        self._close_owned_video_engine()
        self.video_engine = KeyframeVideoEngine(model_id, str(self.output_path))
        self._video_engine_model_id = model_id
        self._owns_video_engine = True
        return self.video_engine

    def _close_owned_designer(self) -> None:
        if self._owns_designer and self.character_designer is not None:
            self.logger.info("Releasing character image model before frame generation")
            self.character_designer.close()
            self.character_designer = None
            self._owns_designer = False

    def _close_owned_animator(self) -> None:
        if self._owns_animator and self.frame_animator is not None:
            self.frame_animator.close()
            self.frame_animator = None
            self._owns_animator = False

    def _close_owned_video_engine(self) -> None:
        if self._owns_video_engine and self.video_engine is not None:
            close = getattr(self.video_engine, "close", None)
            if callable(close):
                close()
            self.video_engine = None
            self._video_engine_model_id = None
            self._owns_video_engine = False

    def _cleanup(self) -> None:
        self._close_owned_designer()
        self._close_owned_animator()
        self._close_owned_video_engine()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def create_characters(self) -> dict[str, list[str]]:
        """Create (or reuse) the character reference sheets and stop.

        This is the explicit first stage of the strategy: design the cast once,
        review the references, and only then spend time painting frames.
        """
        characters = self.screenplay.build_characters()
        try:
            return self._create_character_references(characters)
        finally:
            self._close_owned_designer()

    def process(self) -> str:
        """Run the full pipeline and return the final video path."""
        try:
            scenes = self.screenplay.build_scenes()
            self._apply_animation_overrides(scenes)
            self._apply_audio_duration(scenes)
            characters = self.screenplay.build_characters()
            self._log_plan(characters, scenes)

            references = self._create_character_references(characters)
            self._create_keyframes(scenes, references)
            self._create_end_keyframes(scenes)
            videos = self._animate_scenes(scenes)

            final_video = self._concatenate_videos(videos)
            final_video = self._add_audio(final_video)
            final_video = self._add_subtitles(final_video)
            return final_video
        finally:
            self._cleanup()

    def evaluate(self) -> EvaluationReport:
        """Evaluate the scenes and final video of this run."""
        return self._get_evaluator().evaluate_output(self.output_path)

    def format_plan(self) -> str:
        """Return a human-readable plan without loading any model."""
        scenes = self.screenplay.build_scenes()
        self._apply_animation_overrides(scenes)
        self._apply_audio_duration(scenes)
        characters = self.screenplay.build_characters()
        lines = [
            f"Screenplay: {self.screenplay.name}",
            f"Output: {self.output_path}",
            f"Character model: {self.character_model}",
            f"Frame model: {self.frame_model}",
            "",
            "Characters:",
        ]
        for character in characters:
            lines.append(f"  - {character.name}: {len(character.views)} view(s)")
        lines.append("")
        lines.append("Scenes:")
        painted_frames = 0
        video_frames = 0
        for scene in scenes:
            if scene.animation_mode == "video":
                video_frames += self._video_frame_count(scene)
            else:
                painted_frames += scene.frame_count
            cast = ", ".join(scene.character_names) or "(background)"
            video_settings = (
                f"; model {scene.video_model}"
                if scene.animation_mode == "video"
                else ""
            )
            lines.append(
                f"  - {scene.name}: {scene.frame_count} frames "
                f"@ {scene.fps:g} fps ({scene.duration:.2f}s) [{cast}] "
                f"[animation: {scene.animation_mode}{video_settings}]"
            )
        lines.append("")
        if painted_frames:
            lines.append(f"Total frames to paint: {painted_frames}")
        if video_frames:
            lines.append(f"Total video frames to generate: {video_frames}")
        return "\n".join(lines)

    def _apply_animation_overrides(self, scenes: Sequence[SceneSpec]) -> None:
        for scene in scenes:
            if self.animation_mode is not None:
                scene.animation_mode = self.animation_mode
            if self.video_model:
                scene.video_model = self.video_model

    def _log_plan(self, characters: Sequence[CharacterSpec], scenes: Sequence[SceneSpec]) -> None:
        total_frames = sum(scene.frame_count for scene in scenes)
        self.logger.info(
            "Plan: %d character(s), %d scene(s), %d frame(s)",
            len(characters),
            len(scenes),
            total_frames,
        )

    def _create_character_references(
        self, characters: Sequence[CharacterSpec]
    ) -> dict[str, list[str]]:
        designer = self._get_designer()
        self.logger.info("Creating character references in %s", self.characters_dir)
        references = designer.create_references(
            characters,
            self.characters_dir,
            style_prompt=self.screenplay.style_prompt,
            negative_prompt=self.screenplay.negative_prompt,
            num_inference_steps=self.screenplay.character_num_inference_steps,
            guidance_scale=self.screenplay.character_guidance_scale,
            height=self.screenplay.character_height,
            width=self.screenplay.character_width,
        )
        return references

    def _create_keyframes(
        self,
        scenes: Sequence[SceneSpec],
        references: dict[str, list[str]],
    ) -> None:
        # Characterless shots are generated with text-to-image first, while the
        # lighter model is still loaded.  The edit model is loaded afterwards.
        background_scenes = [scene for scene in scenes if not scene.character_names]
        if background_scenes:
            designer = self._get_designer()
            for scene in background_scenes:
                self._create_background_keyframe(scene, designer)
        self._close_owned_designer()

        primary_references = {
            name: paths[0] for name, paths in references.items() if paths
        }
        descriptions = self.screenplay.character_descriptions()
        character_scenes = [scene for scene in scenes if scene.character_names]
        for scene in character_scenes:
            destination = self.keyframe_path(scene)
            if destination.exists():
                self.logger.info("Found existing keyframe: %s", destination)
                continue
            if self._reuse_start_frame(scene, destination):
                continue
            animator = self._get_animator()
            animator.create_keyframe(
                scene,
                primary_references,
                descriptions,
                self.screenplay.style_prompt,
                destination,
            )

    def _create_end_keyframes(self, scenes: Sequence[SceneSpec]) -> None:
        for scene in scenes:
            if scene.animation_mode != "video" or not (scene.end_prompt or scene.end_frame):
                continue
            destination = self.end_keyframe_path(scene)
            if destination.exists():
                self.logger.info("Found existing end keyframe: %s", destination)
                continue
            if self.scene_video_path(scene).exists():
                self.logger.info(
                    "Reusing cached scene video without recreating its end keyframe: %s",
                    self.scene_video_path(scene),
                )
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            if scene.end_frame:
                source = self.screenplay.resolve_asset(scene.end_frame)
                if not source.exists():
                    raise FileNotFoundError(
                        f"End frame for scene '{scene.name}' not found: {source}"
                    )
                if source.resolve() != destination.resolve():
                    shutil.copyfile(source, destination)
                self.logger.info(
                    "Using supplied end frame for scene '%s': %s", scene.name, source
                )
                continue

            animator = self._get_animator()
            animator.create_end_keyframe(
                scene,
                self.keyframe_path(scene),
                destination,
                style_prompt=self.screenplay.style_prompt,
            )

    def _create_background_keyframe(self, scene: SceneSpec, designer: Any) -> None:
        destination = self.keyframe_path(scene)
        if destination.exists():
            self.logger.info("Found existing keyframe: %s", destination)
            return
        if self._reuse_start_frame(scene, destination):
            return

        prompt = self.screenplay.compose_prompt(scene.effective_keyframe_prompt)
        negative_prompt = join_prompts(
            scene.negative_prompt, "characters, people, text, watermark, logo"
        )
        self.logger.info("Generating background keyframe for scene: %s", scene.name)
        designer.create_background(
            prompt,
            destination,
            negative_prompt=negative_prompt,
            height=scene.height,
            width=scene.width,
            num_inference_steps=scene.num_inference_steps,
            guidance_scale=self.screenplay.character_guidance_scale,
            seed=scene.seed,
        )

    def _reuse_start_frame(self, scene: SceneSpec, destination: Path) -> bool:
        """Copy an externally supplied first frame into the scene folder."""
        if not scene.start_frame:
            return False
        source = self.screenplay.resolve_asset(scene.start_frame)
        if not source.exists():
            raise FileNotFoundError(
                f"Start frame for scene '{scene.name}' not found: {source}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        self.logger.info("Using supplied start frame for scene '%s': %s", scene.name, source)
        return True

    def _animate_scenes(self, scenes: Sequence[SceneSpec]) -> list[str]:
        videos: list[str] = []
        for scene in scenes:
            video_path = self.scene_video_path(scene)
            if video_path.exists():
                self.logger.info("Found existing scene video: %s", video_path)
                videos.append(str(video_path))
                if scene.animation_mode == "video" and self.evaluate_output:
                    self._extract_video_frames(video_path, scene)
                cached_evaluation = self._evaluate_frames(scene)
                if cached_evaluation is not None:
                    self.last_evaluations.append(cached_evaluation)
                continue

            if scene.animation_mode == "video":
                self._close_owned_animator()
                generated = self._animate_video_with_retries(scene)
                videos.append(str(generated))
                continue

            self.logger.warning(
                "Scene '%s' uses animation_mode='frames' (chained image edits): motion "
                "will be minimal and quality may drift. Set \"animation_mode\": \"video\" "
                "and an \"end_prompt\" in the screenplay for real motion.",
                scene.name,
            )
            self._close_owned_video_engine()
            animator = self._get_animator()
            frames = self._animate_with_retries(scene, animator)
            generated = self.video_builder.build(frames, video_path, scene.fps)
            videos.append(str(generated))
        self._log_quality_summary()
        return videos

    def _animate_video_with_retries(self, scene: SceneSpec) -> str:
        attempt = 0
        current_scene = scene
        while True:
            video_path = self.scene_video_path(scene)
            engine = self._get_video_engine(scene.video_model)
            start_image = self._load_image(self.keyframe_path(scene))
            end_path = self.end_keyframe_path(scene)
            end_image = self._load_image(end_path) if end_path.exists() else None
            frame_count = self._video_frame_count(current_scene)
            fps = frame_count / float(current_scene.duration)
            generated = engine.generate(
                scene=current_scene,
                start_image=start_image,
                end_image=end_image,
                output_path=str(video_path),
                num_frames=frame_count,
                fps=fps,
            )
            if self.evaluate_output:
                self._extract_video_frames(generated, scene)
            evaluation = self._evaluate_frames(scene)
            if evaluation is None or not evaluation.needs_retry:
                break
            if attempt >= self.max_retries:
                self.logger.error(
                    "Scene '%s' failed the quality check after %d attempt(s): %s",
                    scene.name,
                    attempt + 1,
                    "; ".join(evaluation.problems),
                )
                break
            attempt += 1
            self.logger.warning(
                "Scene '%s' failed the quality check (%s); retrying %d/%d",
                scene.name,
                "; ".join(evaluation.problems),
                attempt,
                self.max_retries,
            )
            self._clear_scene_animation(scene)
            current_scene = self._build_retry_scene(scene, attempt)
        if evaluation is not None:
            self.last_evaluations.append(evaluation)
        return str(generated)

    @staticmethod
    def _video_frame_count(scene: SceneSpec) -> int:
        max_frames = max(5, ((int(scene.video_max_frames) - 1) // 4) * 4 + 1)
        desired = min(max(5, scene.frame_count), max_frames)
        return max(5, ((desired - 1) // 4) * 4 + 1)

    def _extract_video_frames(self, video_path: str | Path, scene: SceneSpec) -> list[str]:
        """Extract a generated clip into the frame cache used by the evaluator."""
        frames_dir = self.frames_dir(scene)
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        clip = None
        frame_paths: list[str] = []
        try:
            clip = VideoFileClip(str(video_path))
            for index, frame in enumerate(clip.iter_frames(dtype="uint8")):
                destination = frames_dir / f"{index:06d}.png"
                Image.fromarray(frame).convert("RGB").save(destination)
                frame_paths.append(str(destination))
        except Exception as error:
            self.logger.warning(
                "Could not extract frames for quality evaluation from %s: %s",
                video_path,
                error,
            )
            if frames_dir.exists():
                shutil.rmtree(frames_dir)
            return []
        finally:
            if clip is not None:
                clip.close()
        return frame_paths

    @staticmethod
    def _load_image(path: str | Path) -> Image.Image:
        with Image.open(path) as image:
            return image.convert("RGB")

    def _animate_with_retries(self, scene: SceneSpec, animator: Any) -> list[str]:
        """Animate a scene and repeat it while the quality check fails.

        A retry changes the seed and switches the frame prompt to strong motion.
        Re-running with identical inputs would be pointless because generation
        is deterministic for a given seed.
        """
        attempt = 0
        current_scene = scene
        while True:
            frames = animator.animate_scene(
                current_scene, self.keyframe_path(scene), self.frames_dir(scene)
            )
            evaluation = self._evaluate_frames(scene)
            if evaluation is None or not evaluation.needs_retry:
                break
            if attempt >= self.max_retries:
                self.logger.error(
                    "Scene '%s' failed the quality check after %d attempt(s): %s",
                    scene.name,
                    attempt + 1,
                    "; ".join(evaluation.problems),
                )
                break
            attempt += 1
            self.logger.warning(
                "Scene '%s' failed the quality check (%s); retrying %d/%d with strong motion",
                scene.name,
                "; ".join(evaluation.problems),
                attempt,
                self.max_retries,
            )
            self._clear_scene_animation(scene)
            current_scene = self._build_retry_scene(scene, attempt)
        if evaluation is not None:
            self.last_evaluations.append(evaluation)
        return frames

    @staticmethod
    def _build_retry_scene(scene: SceneSpec, attempt: int) -> SceneSpec:
        """Return a variant that is likely to produce visible motion."""
        seed = int(scene.seed or 0) + attempt * 7919
        return replace(scene, seed=seed, motion_strength="strong")

    def _clear_scene_animation(self, scene: SceneSpec) -> None:
        frames_dir = self.frames_dir(scene)
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        video_path = self.scene_video_path(scene)
        if video_path.exists():
            video_path.unlink()

    def _get_evaluator(self) -> Any:
        if self.quality_evaluator is None:
            self.quality_evaluator = VideoQualityEvaluator(logger=self.logger)
        return self.quality_evaluator

    def _evaluate_frames(self, scene: SceneSpec) -> SceneEvaluation | None:
        """Evaluate a scene's frames, never breaking the generation run."""
        if not self.evaluate_output:
            return None
        frames_dir = self.frames_dir(scene)
        if not frames_dir.exists():
            return None
        try:
            evaluation = self._get_evaluator().evaluate_frame_directory(
                frames_dir, name=scene.name
            )
        except Exception as error:  # pragma: no cover - depends on frame files
            self.logger.warning("Could not evaluate scene '%s': %s", scene.name, error)
            return None
        return evaluation

    def _log_quality_summary(self) -> None:
        if not self.last_evaluations:
            return
        failed = [evaluation for evaluation in self.last_evaluations if evaluation.needs_retry]
        self.logger.info(
            "Quality check: %d scene(s) evaluated, %d need a retry",
            len(self.last_evaluations),
            len(failed),
        )
        for evaluation in failed:
            self.logger.warning(
                "Scene '%s' still needs a retry: %s",
                evaluation.name,
                "; ".join(evaluation.problems),
            )

    def _concatenate_videos(self, videos: Sequence[str]) -> str:
        destination = self.final_video_path()
        result = self.video_utils.concatenate_videos(
            list(videos), str(destination), fps=self.screenplay.fps
        )
        return str(result) if isinstance(result, (str, Path)) else str(destination)

    def _add_audio(self, video_path: str) -> str:
        audio_relative_path = self.screenplay.audio
        if not audio_relative_path:
            return video_path

        audio_path = self.screenplay.resolve_asset(audio_relative_path)
        destination = self.final_audio_path()
        result = self.video_utils.add_audio_to_video(
            video_path, str(audio_path), str(destination)
        )
        return str(result) if isinstance(result, (str, Path)) else str(destination)

    def _add_subtitles(self, video_path: str) -> str:
        subtitles_relative_path = self.screenplay.subtitles
        if not subtitles_relative_path:
            return video_path

        subtitles_path = self.screenplay.resolve_asset(subtitles_relative_path)
        audio_relative_path = self.screenplay.audio
        if self.screenplay.auto_sync_subtitles and audio_relative_path:
            audio_path = self.screenplay.resolve_asset(audio_relative_path)
            self.logger.info("Auto-syncing subtitles with audio")
            self.video_utils.sync_subtitles(str(audio_path), str(subtitles_path))

        destination = self.final_subtitles_path()
        result = self.video_utils.add_subtitles_to_video(
            video_path, str(subtitles_path), str(destination)
        )
        return str(result) if isinstance(result, (str, Path)) else str(destination)

    # ------------------------------------------------------------------
    # Audio length
    # ------------------------------------------------------------------
    def _apply_audio_duration(self, scenes: Sequence[SceneSpec]) -> None:
        audio_relative_path = self.screenplay.audio
        if not audio_relative_path:
            return
        audio_duration = self._get_audio_duration(
            self.screenplay.resolve_asset(audio_relative_path)
        )
        if not audio_duration:
            return
        self.logger.info("Scaling scene durations to audio duration: %.2fs", audio_duration)
        scale_scene_durations(scenes, audio_duration)

    @staticmethod
    def _get_audio_duration(audio_path: str | Path) -> float | None:
        """Read an audio duration with MoviePy, returning None on failure."""
        try:
            clip = AudioFileClip(str(audio_path))
            try:
                return float(clip.duration)
            finally:
                clip.close()
        except Exception as error:  # pragma: no cover - depends on media files
            logger.warning("Could not read audio duration from %s: %s", audio_path, error)
            return None
