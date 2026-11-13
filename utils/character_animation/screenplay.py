"""Character-driven screenplay model for frame-by-frame animation.

The model in this module is deliberately independent from the classic
screenplay processor.  It adds a new, richer schema (characters and motion
instructions) without changing the behaviour or the file format accepted by
``scripts/create_video_from_screenplay.py``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from utils.character_animation.character_inputs import CharacterInputLibrary
from utils.character_animation.naming import slugify

DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 704
DEFAULT_CHARACTER_HEIGHT = 1024
DEFAULT_CHARACTER_WIDTH = 768
DEFAULT_FPS = 24.0
DEFAULT_DURATION = 5.0
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_CHARACTER_STEPS = 30
DEFAULT_CHARACTER_GUIDANCE_SCALE = 7.5
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_ANCHOR_INTERVAL = 12
DEFAULT_REFERENCE_VIEW = "front view"
DEFAULT_CHARACTER_IMAGE_MODEL = "Qwen/Qwen-Image"
DEFAULT_FRAME_IMAGE_MODEL = "Qwen/Qwen-Image-Edit"
DEFAULT_VIDEO_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
ALLOWED_MOTION_STRENGTHS = {"normal", "strong"}
ALLOWED_ANIMATION_MODES = {"frames", "video"}


def join_prompts(*parts: str | None) -> str:
    """Join non-empty prompt fragments with a comma."""
    return ", ".join(part.strip() for part in parts if part and part.strip())


@dataclass
class CharacterSpec:
    """A character that must keep its identity across the whole video.

    ``views`` lets the design stage create more than one reference image for
    the same character (for example a front, a three-quarter and a profile
    view).  The first view is the primary identity reference used to build
    scene keyframes.
    """

    name: str
    description: str
    views: tuple[str, ...] = (DEFAULT_REFERENCE_VIEW,)
    reference_prompt: str = ""
    negative_prompt: str = ""
    seed: int | None = None
    height: int | None = None
    width: int | None = None
    reference_image: str = ""

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.description = self.description.strip()
        self.reference_image = str(self.reference_image or "").strip()
        if not self.name:
            raise ValueError("Character is missing required field: 'name'")
        if not self.description:
            raise ValueError(f"Character '{self.name}' is missing required field: 'description'")

        raw_views = (self.views,) if isinstance(self.views, str) else self.views
        views = tuple(view.strip() for view in raw_views if view and view.strip())
        if not views:
            views = (DEFAULT_REFERENCE_VIEW,)
        if len(set(views)) != len(views):
            raise ValueError(f"Character '{self.name}' has duplicated views")
        view_slugs = [slugify(view) for view in views]
        if len(set(view_slugs)) != len(view_slugs):
            raise ValueError(
                f"Character '{self.name}' has views that resolve to the same filename"
            )
        self.views = views
        if self.seed is not None:
            self.seed = int(self.seed)

    @property
    def slug(self) -> str:
        """Identifier used for files and directories."""
        return slugify(self.name)

    @property
    def primary_view(self) -> str:
        """Name of the reference view used for identity conditioning."""
        return self.views[0]

    def reference_paths(self, characters_dir: str | Path) -> dict[str, Path]:
        """Return one output path per configured view."""
        base_dir = Path(characters_dir)
        return {
            view: base_dir / f"{self.slug}_{slugify(view)}.png"
            for view in self.views
        }

    def primary_reference_path(self, characters_dir: str | Path) -> Path:
        """Return the path of the primary identity reference."""
        return self.reference_paths(characters_dir)[self.primary_view]

    @property
    def has_supplied_reference(self) -> bool:
        """Whether the primary reference comes from an external image."""
        return bool(self.reference_image)


@dataclass
class SceneSpec:
    """A resolved scene: every global value has already been applied."""

    index: int
    name: str
    prompt: str
    duration: float
    fps: float
    height: int
    width: int
    negative_prompt: str
    num_inference_steps: int
    true_cfg_scale: float
    seed: int | None
    character_names: tuple[str, ...]
    motion_prompt: str = ""
    keyframe_prompt: str = ""
    start_frame: str | None = None
    anchor_interval: int = DEFAULT_ANCHOR_INTERVAL
    motion_strength: str = "normal"
    animation_mode: str = "frames"
    end_prompt: str = ""
    end_frame: str | None = None
    video_model: str = DEFAULT_VIDEO_MODEL
    video_guidance_scale: float = 5.0
    video_num_inference_steps: int = 40
    video_max_frames: int = 81

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.prompt = self.prompt.strip()
        if not self.name:
            raise ValueError(f"Scene {self.index} is missing required field: 'name'")
        if not self.prompt:
            raise ValueError(f"Scene {self.index} is missing required field: 'prompt'")
        if float(self.duration) <= 0:
            raise ValueError(f"Scene '{self.name}' must have a positive duration")
        if float(self.fps) <= 0:
            raise ValueError(f"Scene '{self.name}' must have a positive fps")
        if int(self.height) <= 0 or int(self.width) <= 0:
            raise ValueError(f"Scene '{self.name}' must have positive dimensions")
        if int(self.anchor_interval) < 0:
            raise ValueError(f"Scene '{self.name}' must have a non-negative anchor_interval")
        if self.seed is not None:
            self.seed = int(self.seed)
        normalized_strength = str(self.motion_strength).strip().lower()
        if normalized_strength not in ALLOWED_MOTION_STRENGTHS:
            raise ValueError(
                f"Scene '{self.name}' has invalid motion_strength: '{self.motion_strength}'"
            )
        self.motion_strength = normalized_strength
        self.animation_mode = str(self.animation_mode).strip().lower()
        if self.animation_mode not in ALLOWED_ANIMATION_MODES:
            raise ValueError(
                f"Scene '{self.name}' has invalid animation_mode: '{self.animation_mode}'"
            )
        self.end_prompt = str(self.end_prompt or "").strip()
        self.video_model = str(self.video_model or "").strip()
        if not self.video_model:
            raise ValueError(f"Scene '{self.name}' must have a non-empty video_model")
        if float(self.video_guidance_scale) <= 0:
            raise ValueError(f"Scene '{self.name}' must have a positive video_guidance_scale")
        if int(self.video_num_inference_steps) <= 0:
            raise ValueError(
                f"Scene '{self.name}' must have positive video_num_inference_steps"
            )
        if int(self.video_max_frames) < 5:
            raise ValueError(f"Scene '{self.name}' video_max_frames must be at least 5")

    @property
    def slug(self) -> str:
        """Identifier used for files and directories."""
        return slugify(self.name)

    @property
    def frame_count(self) -> int:
        """Number of frames required by the scene duration and fps."""
        return max(1, int(round(float(self.duration) * float(self.fps))))

    @property
    def effective_motion_prompt(self) -> str:
        """Motion instruction, falling back to the scene prompt."""
        return self.motion_prompt or self.prompt

    @property
    def effective_keyframe_prompt(self) -> str:
        """Keyframe instruction, falling back to the scene prompt."""
        return self.keyframe_prompt or self.prompt


class CharacterScreenplay:
    """Parse and validate a screenplay that uses the character strategy."""

    def __init__(self, data: dict[str, Any], source_path: str | Path, output_path: str | Path = "output"):
        self.data = data
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.character_inputs = CharacterInputLibrary.from_screenplay(
            self.source_path, data
        )
        self._characters: list[CharacterSpec] | None = None
        self._scenes: list[SceneSpec] | None = None
        self.validate()

    @classmethod
    def load(cls, screenplay: str | Path, output_path: str | Path = "output") -> "CharacterScreenplay":
        """Load a screenplay JSON file from disk."""
        path = Path(screenplay)
        with open(path, "r", encoding="utf-8") as screenplay_file:
            data = json.load(screenplay_file)
        return cls(data, path, output_path)

    @property
    def name(self) -> str:
        return str(self.data["name"])

    @property
    def height(self) -> int:
        return int(self.data.get("height", DEFAULT_HEIGHT))

    @property
    def width(self) -> int:
        return int(self.data.get("width", DEFAULT_WIDTH))

    @property
    def fps(self) -> float:
        return float(self.data.get("fps", DEFAULT_FPS))

    @property
    def negative_prompt(self) -> str:
        return str(self.data.get("negative_prompt", "") or "")

    @property
    def motion_strength(self) -> str:
        """Default per-frame motion intensity: ``normal`` or ``strong``."""
        return str(self.data.get("motion_strength", "normal"))

    @property
    def animation_mode(self) -> str:
        default = "video" if self.data.get("video_model") else "frames"
        return str(self.data.get("animation_mode", default)).strip().lower()

    @property
    def video_model(self) -> str:
        return str(self.data.get("video_model", DEFAULT_VIDEO_MODEL) or "").strip()

    @property
    def video_guidance_scale(self) -> float:
        return float(self.data.get("video_guidance_scale", 5.0))

    @property
    def video_num_inference_steps(self) -> int:
        return int(self.data.get("video_num_inference_steps", 40))

    @property
    def video_max_frames(self) -> int:
        return int(self.data.get("video_max_frames", 81))

    @property
    def style_prompt(self) -> str:
        """Global style prefix.

        ``common_prompt`` is accepted as an alias so screenplays written for
        the classic processor can be migrated by adding a ``characters`` block.
        """

        return join_prompts(self.data.get("style_prompt"), self.data.get("common_prompt"))

    @property
    def character_image_model(self) -> str:
        return str(self.data.get("character_image_model", DEFAULT_CHARACTER_IMAGE_MODEL))

    @property
    def frame_image_model(self) -> str:
        return str(self.data.get("frame_image_model", DEFAULT_FRAME_IMAGE_MODEL))

    @property
    def character_width(self) -> int:
        return int(self.data.get("character_width", DEFAULT_CHARACTER_WIDTH))

    @property
    def character_height(self) -> int:
        return int(self.data.get("character_height", DEFAULT_CHARACTER_HEIGHT))

    @property
    def character_num_inference_steps(self) -> int:
        return int(self.data.get("character_num_inference_steps", DEFAULT_CHARACTER_STEPS))

    @property
    def character_guidance_scale(self) -> float:
        return float(
            self.data.get("character_guidance_scale", DEFAULT_CHARACTER_GUIDANCE_SCALE)
        )

    @property
    def audio(self) -> str | None:
        value = self.data.get("audio")
        return str(value) if value else None

    @property
    def subtitles(self) -> str | None:
        value = self.data.get("subtitles")
        return str(value) if value else None

    @property
    def auto_sync_subtitles(self) -> bool:
        return bool(self.data.get("auto_sync_subtitles", False))

    def validate(self) -> None:
        """Validate the complete screenplay before any expensive work starts."""
        if "name" not in self.data:
            raise ValueError("Screenplay is missing required field: 'name'")
        if "scenes" not in self.data:
            raise ValueError("Screenplay is missing required field: 'scenes'")
        if not isinstance(self.data["scenes"], list) or not self.data["scenes"]:
            raise ValueError("Screenplay must contain at least one scene")
        if "characters" not in self.data:
            raise ValueError("Screenplay is missing required field: 'characters'")
        if not isinstance(self.data["characters"], list) or not self.data["characters"]:
            raise ValueError("Screenplay must contain at least one character")
        if (
            str(self.data.get("motion_strength", "normal")).strip().lower()
            not in ALLOWED_MOTION_STRENGTHS
        ):
            raise ValueError(
                f"Invalid global motion_strength: '{self.data.get('motion_strength')}'"
            )
        if self.animation_mode not in ALLOWED_ANIMATION_MODES:
            raise ValueError(f"Invalid animation_mode: '{self.animation_mode}'")
        if not self.video_model:
            raise ValueError("video_model must be a non-empty model id")
        if self.video_guidance_scale <= 0:
            raise ValueError("video_guidance_scale must be positive")
        if self.video_num_inference_steps <= 0:
            raise ValueError("video_num_inference_steps must be positive")
        if self.video_max_frames < 5:
            raise ValueError("video_max_frames must be at least 5")

        characters = self._parse_characters()
        character_names = {character.name for character in characters}
        scene_names: set[str] = set()
        scene_slugs: set[str] = set()

        for index, scene in enumerate(self.data["scenes"]):
            if not isinstance(scene, dict):
                raise ValueError(f"Scene {index} must be an object")
            if not scene.get("prompt"):
                raise ValueError(f"Scene {index} is missing required field: 'prompt'")
            scene_mode = str(self._scene_animation_mode(scene)).strip().lower()
            if scene_mode not in ALLOWED_ANIMATION_MODES:
                raise ValueError(
                    f"Scene {index} has invalid animation_mode: '{scene_mode}'"
                )
            scene_name = str(scene.get("name", f"scene_{index}"))
            if scene_name in scene_names:
                raise ValueError(f"Duplicated scene name: '{scene_name}'")
            # Scenes are stored in folders named after their slug, so names
            # differing only in punctuation or case would share assets.
            scene_slug = slugify(scene_name)
            if scene_slug in scene_slugs:
                raise ValueError(
                    f"Scene name '{scene_name}' resolves to a duplicated folder name"
                )
            scene_names.add(scene_name)
            scene_slugs.add(scene_slug)
            self._resolve_scene_characters(scene, index, characters, character_names)

    def _parse_characters(self) -> list[CharacterSpec]:
        characters: list[CharacterSpec] = []
        names: set[str] = set()
        slugs: set[str] = set()
        for index, raw_character in enumerate(self.data["characters"]):
            if not isinstance(raw_character, dict):
                raise ValueError(f"Character {index} must be an object")
            views = raw_character.get("views", [DEFAULT_REFERENCE_VIEW])
            if isinstance(views, str):
                views = [views]
            name = str(raw_character.get("name", ""))
            character = CharacterSpec(
                name=name,
                description=str(raw_character.get("description", "")),
                views=tuple(str(view) for view in views),
                reference_prompt=str(raw_character.get("reference_prompt", "") or ""),
                negative_prompt=str(raw_character.get("negative_prompt", "") or ""),
                seed=raw_character.get("seed"),
                height=raw_character.get("height"),
                width=raw_character.get("width"),
                reference_image=self._resolve_reference_image(raw_character, name),
            )
            if character.name in names:
                raise ValueError(f"Duplicated character name: '{character.name}'")
            if character.slug in slugs:
                raise ValueError(
                    f"Character name '{character.name}' resolves to a duplicated filename"
                )
            names.add(character.name)
            slugs.add(character.slug)
            characters.append(character)
        return characters

    def _resolve_reference_image(self, raw_character: dict[str, Any], name: str) -> str:
        declared = raw_character.get("reference_image") or raw_character.get("image")
        if declared:
            path = self.resolve_asset(str(declared))
            if not path.is_file():
                raise FileNotFoundError(
                    f"Reference image for character '{name}' not found: {path}"
                )
            return str(path)

        supplied = self.character_inputs.resolve(name)
        return str(supplied) if supplied is not None else ""

    def _scene_animation_mode(self, scene: dict[str, Any]) -> str:
        if "animation_mode" in scene:
            return str(scene["animation_mode"])
        if scene.get("video_model") and "animation_mode" not in self.data:
            return "video"
        return self.animation_mode

    @staticmethod
    def _resolve_scene_characters(
        scene: dict[str, Any],
        index: int,
        characters: Sequence[CharacterSpec],
        character_names: set[str],
    ) -> tuple[str, ...]:
        raw = scene.get("characters")
        if raw is None:
            # The first character is the protagonist and the default subject
            # for scenes that do not explicitly list their cast.
            return (characters[0].name,)
        if isinstance(raw, str):
            if raw.strip().lower() == "all":
                return tuple(character.name for character in characters)
            raw = [raw]
        if not isinstance(raw, list):
            raise ValueError(f"Scene {index} field 'characters' must be a list or 'all'")
        resolved: list[str] = []
        for name in raw:
            name = str(name).strip()
            if name not in character_names:
                raise ValueError(f"Scene {index} references unknown character: '{name}'")
            if name not in resolved:
                resolved.append(name)
        return tuple(resolved)

    def build_characters(self) -> list[CharacterSpec]:
        """Return the validated character definitions."""
        if self._characters is None:
            self._characters = self._parse_characters()
        return list(self._characters)

    def build_scenes(self) -> list[SceneSpec]:
        """Resolve every scene, applying global defaults and overrides."""
        if self._scenes is not None:
            return copy.deepcopy(self._scenes)

        characters = self.build_characters()
        character_names = {character.name for character in characters}
        global_height = self.height
        global_width = self.width
        global_fps = self.fps
        global_negative = self.negative_prompt
        global_steps = int(self.data.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS))
        global_true_cfg = float(self.data.get("true_cfg_scale", DEFAULT_TRUE_CFG_SCALE))
        global_anchor = int(self.data.get("anchor_interval", DEFAULT_ANCHOR_INTERVAL))
        global_motion_strength = self.motion_strength
        global_video_model = self.video_model
        global_video_guidance = self.video_guidance_scale
        global_video_steps = self.video_num_inference_steps
        global_video_max_frames = self.video_max_frames

        scenes: list[SceneSpec] = []
        for index, scene in enumerate(self.data["scenes"]):
            scene_name = str(scene.get("name", f"scene_{index}"))
            character_names_for_scene = self._resolve_scene_characters(
                scene, index, characters, character_names
            )
            scenes.append(
                SceneSpec(
                    index=index,
                    name=scene_name,
                    prompt=str(scene["prompt"]),
                    duration=float(scene.get("duration", DEFAULT_DURATION)),
                    fps=float(scene.get("fps", global_fps)),
                    height=int(scene.get("height", global_height)),
                    width=int(scene.get("width", global_width)),
                    negative_prompt=str(scene.get("negative_prompt", global_negative) or ""),
                    num_inference_steps=int(scene.get("num_inference_steps", global_steps)),
                    true_cfg_scale=float(scene.get("true_cfg_scale", global_true_cfg)),
                    seed=scene.get("seed"),
                    character_names=character_names_for_scene,
                    motion_prompt=str(scene.get("motion_prompt", "") or ""),
                    keyframe_prompt=str(scene.get("keyframe_prompt", "") or ""),
                    start_frame=str(scene["start_frame"]) if scene.get("start_frame") else None,
                    anchor_interval=int(scene.get("anchor_interval", global_anchor)),
                    motion_strength=str(
                        scene.get("motion_strength", global_motion_strength)
                    ),
                    animation_mode=self._scene_animation_mode(scene),
                    end_prompt=str(scene.get("end_prompt", "") or ""),
                    end_frame=str(scene["end_frame"]) if scene.get("end_frame") else None,
                    video_model=str(scene.get("video_model") or global_video_model),
                    video_guidance_scale=float(
                        scene.get("video_guidance_scale", global_video_guidance)
                    ),
                    video_num_inference_steps=int(
                        scene.get("video_num_inference_steps", global_video_steps)
                    ),
                    video_max_frames=int(
                        scene.get("video_max_frames", global_video_max_frames)
                    ),
                )
            )
        self._scenes = scenes
        # Return copies because callers (for example the audio-duration
        # scaler) mutate durations per run.
        return copy.deepcopy(scenes)

    def resolve_asset(self, relative_path: str) -> Path:
        """Resolve a path declared in the screenplay relative to its folder."""
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return self.source_path.parent / path

    def compose_prompt(self, prompt: str) -> str:
        """Prepend the global style prompt to a scene prompt."""
        return join_prompts(self.style_prompt, prompt)

    def character_descriptions(self) -> dict[str, str]:
        """Return a name-to-description mapping for prompt construction."""
        return {character.name: character.description for character in self.build_characters()}


def scale_scene_durations(scenes: Sequence[SceneSpec], audio_duration: float) -> None:
    """Scale scene durations so the montage matches the audio length.

    The last scene absorbs accumulated rounding error, mirroring the behaviour
    of the classic screenplay processor.
    """
    total_duration = sum(float(scene.duration) for scene in scenes)
    if total_duration <= 0 or audio_duration <= 0 or not scenes:
        return

    scale = float(audio_duration) / total_duration
    for scene in scenes:
        scene.duration = float(scene.duration) * scale

    scenes[-1].duration = float(audio_duration) - sum(
        float(scene.duration) for scene in scenes[:-1]
    )
