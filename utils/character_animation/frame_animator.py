"""Frame-by-frame animation with Qwen-Image-Edit."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

from utils.character_animation.character_board import CharacterBoardBuilder
from utils.character_animation.screenplay import SceneSpec, join_prompts

logger = logging.getLogger(__name__)

IDENTITY_RULES = (
    "Treat each supplied reference as an immutable identity anchor. Keep every character "
    "exactly identical: same facial structure, age, gender, skin tone, eye color, hairstyle, "
    "facial hair, costume, accessories and body proportions. Change only the action, pose "
    "and expression requested by the scene. Never redesign, age, morph, duplicate, replace, "
    "swap faces or costumes, or change a character's species."
)

NEGATIVE_RULES = (
    "character redesign, different face, face swap, different clothes, hairstyle change, "
    "age change, gender change, species transformation, morphing, identity change, "
    "identity drift, duplicate character, reference sheet, collage, split screen, "
    "multiple panels, text, watermark, logo, deformed anatomy, extra fingers, extra limbs"
)

SUBTLE_MOTION_INSTRUCTION = (
    "Advance the pose, expression and camera framing only slightly. "
    "Preserve the composition, background, lighting and every character identity. "
    "Output one single animation frame."
)

STRONG_MOTION_INSTRUCTION = (
    "Create a clearly different animation moment: change the pose, arm positions, "
    "facial expression and camera framing noticeably, with large readable motion. "
    "Keep the same characters, costumes, setting, lighting and art style. "
    "Output one single animation frame."
)


class FrameAnimator:
    """Create a scene keyframe and then paint every subsequent frame.

    The first frame of a scene is anchored to the character references.  Each
    following frame is edited from the previous frame so motion stays smooth.
    Every ``scene.anchor_interval`` frames the animation is re-anchored to the
    scene keyframe, which prevents cumulative identity drift over long shots.
    """

    def __init__(
        self,
        engine,
        board_builder: CharacterBoardBuilder | None = None,
        logger: logging.Logger | None = None,
    ):
        self.engine = engine
        self.board_builder = board_builder or CharacterBoardBuilder()
        self.logger = logger or logging.getLogger(__name__)

    def create_keyframe(
        self,
        scene: SceneSpec,
        character_references: Mapping[str, str],
        character_descriptions: Mapping[str, str],
        style_prompt: str,
        output_path: str | Path,
    ) -> str:
        """Generate the identity-anchored first frame of a scene."""
        destination = Path(output_path)
        if destination.exists():
            self.logger.info("Found existing keyframe: %s", destination)
            return str(destination)

        names = list(scene.character_names)
        if not names:
            raise ValueError(
                f"Scene '{scene.name}' has no characters; use a text-to-image background instead"
            )

        # Include the cast in the filename so changing a scene's characters
        # never reuses a stale reference board.
        cast_key = hashlib.sha1("|".join(names).encode("utf-8")).hexdigest()[:8]
        board_path = destination.parent / "cast" / f"{scene.slug}_{cast_key}_cast.png"
        conditioning_path = self.board_builder.build(
            names, character_references, board_path
        )
        prompt = self.build_keyframe_prompt(
            scene, character_descriptions, style_prompt
        )
        self.logger.info("Generating keyframe for scene: %s", scene.name)

        conditioning_image = self._load_image(conditioning_path)
        generated = self.engine.edit(
            conditioning_image,
            prompt=prompt,
            negative_prompt=self.build_negative_prompt(scene),
            true_cfg_scale=scene.true_cfg_scale,
            num_inference_steps=scene.num_inference_steps,
            height=scene.height,
            width=scene.width,
            seed=scene.seed,
        )
        self._save_frame(generated, destination, scene)
        return str(destination)

    def create_end_keyframe(
        self,
        scene: SceneSpec,
        keyframe_path: str | Path,
        output_path: str | Path,
        style_prompt: str = "",
    ) -> str:
        """Edit the scene's starting keyframe once to create its final pose."""
        keyframe = Path(keyframe_path)
        if not keyframe.exists():
            raise FileNotFoundError(f"Keyframe not found: {keyframe}")
        destination = Path(output_path)
        if destination.exists():
            self.logger.info("Found existing end keyframe: %s", destination)
            return str(destination)
        if not scene.end_prompt:
            raise ValueError(f"Scene '{scene.name}' has no end_prompt")

        generated = self.engine.edit(
            self._load_image(keyframe),
            prompt=self.build_end_keyframe_prompt(scene, style_prompt),
            negative_prompt=self.build_negative_prompt(scene),
            true_cfg_scale=scene.true_cfg_scale,
            num_inference_steps=scene.num_inference_steps,
            height=scene.height,
            width=scene.width,
            # Reuse the scene seed so the endpoint edit does not introduce a new
            # random identity or composition unnecessarily.
            seed=scene.seed,
        )
        self._save_frame(generated, destination, scene)
        return str(destination)

    @staticmethod
    def build_end_keyframe_prompt(scene: SceneSpec, style_prompt: str = "") -> str:
        """Build an identity-preserving instruction for the final scene pose."""
        cast = (
            f"Characters present: {', '.join(scene.character_names)}."
            if scene.character_names
            else ""
        )
        return join_prompts(
            "Using the input image as the starting scene and identity reference, "
            "create its final moment",
            f"Scene: {join_prompts(style_prompt, scene.end_prompt)}",
            cast,
            IDENTITY_RULES,
            "Keep the same camera position, framing, setting, background, lighting and "
            "visual style. Do not add, remove or replace characters. Edit only the action, "
            "pose and expression needed to match the final moment; do not reinterpret the "
            "scene. Output one single animation frame.",
        )

    def animate_scene(
        self,
        scene: SceneSpec,
        keyframe_path: str | Path,
        frames_dir: str | Path,
    ) -> list[str]:
        """Generate all frames of a scene and return their ordered paths."""
        keyframe = Path(keyframe_path)
        if not keyframe.exists():
            raise FileNotFoundError(f"Keyframe not found: {keyframe}")

        output_dir = Path(frames_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        total_frames = scene.frame_count
        self.logger.info(
            "Animating scene '%s' (%d frames)", scene.name, total_frames
        )

        frame_paths: list[str] = []
        previous_frame: Path | None = None
        for index in range(total_frames):
            frame_path = output_dir / f"{index:06d}.png"
            frame_paths.append(str(frame_path))

            if frame_path.exists():
                previous_frame = frame_path
                continue

            if index == 0:
                # Frame zero is the already approved keyframe. Copying it keeps
                # the frame sequence self-contained for video encoding and
                # resuming.
                shutil.copyfile(keyframe, frame_path)
                previous_frame = frame_path
                continue

            anchor = scene.anchor_interval > 0 and index % scene.anchor_interval == 0
            source_path = keyframe if anchor else previous_frame
            if source_path is None:
                raise RuntimeError(
                    f"Cannot generate frame {index} of scene '{scene.name}' without a previous frame"
                )

            progress = index / max(1, total_frames - 1)
            prompt = self.build_frame_prompt(scene, progress, anchored=anchor)
            self.logger.info(
                "Scene '%s' frame %d/%d%s",
                scene.name,
                index + 1,
                total_frames,
                " (re-anchored)" if anchor else "",
            )
            generated = self.engine.edit(
                self._load_image(source_path),
                prompt=prompt,
                negative_prompt=self.build_negative_prompt(scene),
                true_cfg_scale=scene.true_cfg_scale,
                num_inference_steps=scene.num_inference_steps,
                height=scene.height,
                width=scene.width,
                seed=scene.seed,
            )
            self._save_frame(generated, frame_path, scene)
            previous_frame = frame_path

        return frame_paths

    @staticmethod
    def build_keyframe_prompt(
        scene: SceneSpec,
        character_descriptions: Mapping[str, str],
        style_prompt: str,
    ) -> str:
        """Build the prompt that turns character references into a scene."""
        full_scene_prompt = join_prompts(style_prompt, scene.effective_keyframe_prompt)
        names = list(scene.character_names)
        if len(names) == 1:
            name = names[0]
            description = character_descriptions.get(name, name)
            reference_text = (
                f"The input image is the identity reference for {name}: {description}. "
                f"Place this exact character in the scene."
            )
        else:
            character_list = "; ".join(
                f"{position + 1}) {name}: {character_descriptions.get(name, name)}"
                for position, name in enumerate(names)
            )
            reference_text = (
                "The input image is a character reference board. From left to right "
                f"the characters are: {character_list}. Turn this reference board into "
                "one single coherent scene, keeping each character design identical."
            )

        return join_prompts(
            reference_text,
            IDENTITY_RULES,
            f"Scene: {full_scene_prompt}",
            "Cinematic animation frame, clear staging, consistent lighting and background. "
            "Output one single frame, not a reference sheet, collage or split screen.",
        )

    @staticmethod
    def build_frame_prompt(scene: SceneSpec, progress: float, anchored: bool = False) -> str:
        """Build the per-frame edit instruction."""
        percent = max(0, min(100, int(round(progress * 100))))
        strong_motion = scene.motion_strength == "strong"
        if strong_motion:
            opening = (
                "Using this anchor keyframe as the identity and composition reference, "
                f"recreate the scene at {percent}% of the action with a clearly different pose."
                if anchored
                else "Continue this animation with a clearly visible change of pose and framing."
            )
        else:
            opening = (
                "Using this anchor keyframe as the identity and composition reference, "
                f"recreate the scene at {percent}% of the action."
                if anchored
                else "Continue this animation frame."
            )
        cast = (
            f"Characters present: {', '.join(scene.character_names)}."
            if scene.character_names
            else ""
        )
        return join_prompts(
            opening,
            IDENTITY_RULES,
            cast,
            f"Scene: {scene.effective_motion_prompt}",
            f"Moment: {percent}% of the action.",
            STRONG_MOTION_INSTRUCTION if strong_motion else SUBTLE_MOTION_INSTRUCTION,
        )

    @staticmethod
    def build_negative_prompt(scene: SceneSpec) -> str:
        """Combine the screenplay negative prompt with identity-specific rules."""
        return join_prompts(scene.negative_prompt, NEGATIVE_RULES)

    @staticmethod
    def _load_image(path: str | Path) -> Image.Image:
        with Image.open(path) as source:
            return source.convert("RGB")

    @staticmethod
    def _save_frame(image: Image.Image, output_path: Path, scene: SceneSpec) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if image.size != (scene.width, scene.height):
            image = image.resize(
                (scene.width, scene.height), resample=Image.Resampling.LANCZOS
            )
        image.convert("RGB").save(str(output_path))

    def close(self) -> None:
        """Release the frame engine if it exposes a destroy method."""
        destroy = getattr(self.engine, "destroy", None)
        if callable(destroy):
            destroy()
