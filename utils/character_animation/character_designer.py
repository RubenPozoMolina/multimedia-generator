"""Character reference generation for the frame-by-frame pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image

from utils.character_animation.screenplay import (
    DEFAULT_CHARACTER_GUIDANCE_SCALE,
    DEFAULT_CHARACTER_HEIGHT,
    DEFAULT_CHARACTER_STEPS,
    DEFAULT_CHARACTER_WIDTH,
    CharacterSpec,
    join_prompts,
)
from utils.image_utils import ImageUtils

logger = logging.getLogger(__name__)

# A neutral reference sheet is easier for Qwen-Image-Edit to use as an
# identity anchor than a dramatic frame with lighting and occlusions.
REFERENCE_SHEET_INSTRUCTIONS = (
    "full-body character design reference, neutral standing pose, centered composition, "
    "plain neutral grey background, consistent character design, clean silhouette, "
    "high detail, no text, no watermark, no logo, no extra characters"
)

BACKGROUND_INSTRUCTIONS = (
    "cinematic animation background frame, no characters, no text, no watermark, "
    "high detail, consistent art style"
)


class CharacterDesigner:
    """Generate and cache character reference images.

    The designer owns the text-to-image pipeline while the character sheets and
    any characterless scene backgrounds are being generated.  Call
    :meth:`close` before loading the heavier Qwen-Image-Edit model.
    """

    def __init__(
        self,
        model_id: str,
        output_path: str | Path = "output",
        image_utils_factory: Callable[..., ImageUtils] = ImageUtils,
        logger: logging.Logger | None = None,
    ):
        self.model_id = model_id
        self.output_path = Path(output_path)
        self._image_utils_factory = image_utils_factory
        self._image_utils: ImageUtils | None = None
        self.logger = logger or logging.getLogger(__name__)

    def _get_image_utils(self) -> ImageUtils:
        if self._image_utils is None:
            self.logger.info("Loading character image model: %s", self.model_id)
            self._image_utils = self._image_utils_factory(
                self.model_id, output_path=str(self.output_path)
            )
        return self._image_utils

    @staticmethod
    def build_reference_prompt(character: CharacterSpec, view: str, style_prompt: str = "") -> str:
        """Build the prompt used to create one character reference view."""
        if character.reference_prompt:
            return join_prompts(style_prompt, character.reference_prompt, view)
        return join_prompts(
            style_prompt,
            f"character design reference of {character.description}",
            view,
            REFERENCE_SHEET_INSTRUCTIONS,
        )

    @staticmethod
    def build_reference_negative_prompt(character: CharacterSpec, global_negative_prompt: str = "") -> str:
        """Build the negative prompt for character references."""
        return join_prompts(
            global_negative_prompt,
            character.negative_prompt,
            "multiple characters, collage, split screen, reference sheet grid, "
            "text, watermark, logo, deformed anatomy, extra fingers, extra limbs",
        )

    def generate(
        self,
        prompt: str,
        output_path: str | Path,
        negative_prompt: str = "",
        height: int = DEFAULT_CHARACTER_HEIGHT,
        width: int = DEFAULT_CHARACTER_WIDTH,
        num_inference_steps: int = DEFAULT_CHARACTER_STEPS,
        guidance_scale: float = DEFAULT_CHARACTER_GUIDANCE_SCALE,
        seed: int | None = None,
    ) -> str:
        """Generate a single text-to-image asset and return its path."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        generated = self._get_image_utils().text_to_image(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_path=str(destination),
        )
        return str(generated)

    def create_references(
        self,
        characters: Sequence[CharacterSpec],
        characters_dir: str | Path,
        style_prompt: str = "",
        negative_prompt: str = "",
        num_inference_steps: int = DEFAULT_CHARACTER_STEPS,
        guidance_scale: float = DEFAULT_CHARACTER_GUIDANCE_SCALE,
        height: int = DEFAULT_CHARACTER_HEIGHT,
        width: int = DEFAULT_CHARACTER_WIDTH,
    ) -> dict[str, list[str]]:
        """Create every configured view for every character.

        A reference image supplied by the user replaces the primary generated
        view.  Existing files are reused so an interrupted generation can
        resume without regenerating the whole cast.
        """
        base_dir = Path(characters_dir)
        references: dict[str, list[str]] = {}

        for character in characters:
            view_paths = character.reference_paths(base_dir)
            character_references: list[str] = []
            for index, (view, path) in enumerate(view_paths.items()):
                if index == 0 and character.has_supplied_reference:
                    self._copy_supplied_reference(character, path)
                    character_references.append(str(path))
                    continue

                if path.exists():
                    self.logger.info("Found existing character reference: %s", path)
                    character_references.append(str(path))
                    continue

                self.logger.info(
                    "Generating character reference: %s (%s)", character.name, view
                )
                generated = self.generate(
                    self.build_reference_prompt(character, view, style_prompt),
                    output_path=path,
                    negative_prompt=self.build_reference_negative_prompt(
                        character, negative_prompt
                    ),
                    height=int(character.height or height),
                    width=int(character.width or width),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=character.seed,
                )
                character_references.append(str(generated))
            references[character.name] = character_references

        return references

    def _copy_supplied_reference(self, character: CharacterSpec, destination: Path) -> str:
        """Copy a user-supplied image into the character reference cache."""
        source = Path(character.reference_image)
        if not source.is_file():
            raise FileNotFoundError(
                f"Reference image for character '{character.name}' not found: {source}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            with Image.open(source) as image:
                image.convert("RGB").save(destination)
        self.logger.info(
            "Using supplied reference for character %s: %s", character.name, source
        )
        return str(destination)

    def create_background(
        self,
        prompt: str,
        output_path: str | Path,
        negative_prompt: str = "",
        height: int = 480,
        width: int = 704,
        num_inference_steps: int = DEFAULT_CHARACTER_STEPS,
        guidance_scale: float = DEFAULT_CHARACTER_GUIDANCE_SCALE,
        seed: int | None = None,
    ) -> str:
        """Create a characterless scene keyframe with the text-to-image model."""
        return self.generate(
            join_prompts(prompt, BACKGROUND_INSTRUCTIONS),
            output_path=output_path,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    def close(self) -> None:
        """Release the text-to-image pipeline and free GPU memory."""
        if self._image_utils is not None:
            self._image_utils.destroy()
            self._image_utils = None
