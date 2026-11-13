import json
import logging
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from utils.character_animation import (
    CharacterAnimationProcessor,
    CharacterBoardBuilder,
    CharacterDesigner,
    CharacterScreenplay,
    FrameAnimator,
    FrameVideoBuilder,
    QwenImageEditEngine,
    scale_scene_durations,
    slugify,
)
from utils.character_animation.keyframe_video_engine import KeyframeVideoEngine
from utils.character_animation.screenplay import CharacterSpec, SceneSpec

logger = logging.getLogger(__name__)

MINIMAL_SCREENPLAY = {
    "name": "Demo",
    "width": 640,
    "height": 360,
    "fps": 24,
    "characters": [
        {"name": "Hero", "description": "a tall knight in blue armor"},
        {"name": "Mage", "description": "an old mage with a white beard"},
    ],
    "scenes": [
        {
            "name": "Opening",
            "duration": 2,
            "prompt": "a castle on a hill",
            "motion_prompt": "the knight raises a sword",
            "characters": ["Hero"],
        }
    ],
}


def write_screenplay(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "screenplay.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def save_image(path: Path, size=(64, 36)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path)
    return str(path)


class TestCharacterScreenplay:

    def test_missing_name_raises(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY}
        data.pop("name")
        with pytest.raises(ValueError, match="name"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_missing_scenes_raises(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY}
        data.pop("scenes")
        with pytest.raises(ValueError, match="scenes"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_missing_characters_raises(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY}
        data.pop("characters")
        with pytest.raises(ValueError, match="characters"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_empty_characters_raises(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY, "characters": []}
        with pytest.raises(ValueError, match="at least one character"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_character_requires_description(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY, "characters": [{"name": "Hero"}]}
        with pytest.raises(ValueError, match="description"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_duplicate_character_names_raise(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "characters": [
                {"name": "Hero", "description": "a knight"},
                {"name": "Hero", "description": "another knight"},
            ],
        }
        with pytest.raises(ValueError, match="Duplicated character"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_scene_missing_prompt_raises(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY, "scenes": [{"name": "Bad"}]}
        with pytest.raises(ValueError, match="Scene 0 is missing required field: 'prompt'"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_animation_mode_defaults_to_legacy_frames(self, tmp_path):
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, MINIMAL_SCREENPLAY))
        assert screenplay.build_scenes()[0].animation_mode == "frames"

    def test_video_model_defaults_animation_mode_to_video(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY, "video_model": "test/video"}
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, data))
        assert screenplay.build_scenes()[0].animation_mode == "video"
        assert screenplay.build_scenes()[0].video_model == "test/video"

    def test_scene_video_model_defaults_only_that_scene_to_video(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "scenes": [
                {**MINIMAL_SCREENPLAY["scenes"][0], "video_model": "test/video"}
            ],
        }
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, data))
        scene = screenplay.build_scenes()[0]
        assert scene.animation_mode == "video"
        assert scene.video_model == "test/video"

    def test_invalid_animation_mode_raises(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY, "animation_mode": "unknown"}
        with pytest.raises(ValueError, match="animation_mode"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_unknown_scene_character_raises(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "scenes": [{"name": "Bad", "prompt": "x", "characters": ["Ghost"]}],
        }
        with pytest.raises(ValueError, match="unknown character"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_duplicate_scene_names_raise(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "scenes": [
                {"name": "Same", "prompt": "a"},
                {"name": "Same", "prompt": "b"},
            ],
        }
        with pytest.raises(ValueError, match="Duplicated scene"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_scene_names_with_same_slug_raise(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "scenes": [
                {"name": "Night Sky", "prompt": "a"},
                {"name": "night-sky", "prompt": "b"},
            ],
        }
        with pytest.raises(ValueError, match="duplicated folder name"):
            CharacterScreenplay.load(write_screenplay(tmp_path, data))

    def test_builds_scenes_and_applies_defaults(self, tmp_path):
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, MINIMAL_SCREENPLAY))
        scenes = screenplay.build_scenes()
        assert len(scenes) == 1
        scene = scenes[0]
        assert scene.name == "Opening"
        assert scene.width == 640
        assert scene.height == 360
        assert scene.fps == 24
        assert scene.frame_count == 48
        assert scene.character_names == ("Hero",)
        assert scene.motion_prompt == "the knight raises a sword"

    def test_scene_without_characters_uses_first_character(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "scenes": [{"name": "Solo", "prompt": "a forest"}],
        }
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, data))
        assert screenplay.build_scenes()[0].character_names == ("Hero",)

    def test_scene_characters_all_keyword(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "scenes": [{"name": "Group", "prompt": "the party", "characters": "all"}],
        }
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, data))
        assert screenplay.build_scenes()[0].character_names == ("Hero", "Mage")

    def test_explicit_empty_characters_means_background(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "scenes": [{"name": "Landscape", "prompt": "a valley", "characters": []}],
        }
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, data))
        assert screenplay.build_scenes()[0].character_names == ()

    def test_scene_overrides_global_values(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "negative_prompt": "global negative",
            "num_inference_steps": 20,
            "anchor_interval": 10,
            "scenes": [
                {
                    "name": "Override",
                    "prompt": "a dog",
                    "width": 1280,
                    "height": 720,
                    "fps": 30,
                    "duration": 1,
                    "negative_prompt": "scene negative",
                    "num_inference_steps": 5,
                    "true_cfg_scale": 3.0,
                    "seed": 42,
                    "anchor_interval": 4,
                }
            ],
        }
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, data))
        scene = screenplay.build_scenes()[0]
        assert scene.width == 1280
        assert scene.height == 720
        assert scene.fps == 30
        assert scene.frame_count == 30
        assert scene.negative_prompt == "scene negative"
        assert scene.num_inference_steps == 5
        assert scene.true_cfg_scale == 3.0
        assert scene.seed == 42
        assert scene.anchor_interval == 4

    def test_common_prompt_is_accepted_as_style_alias(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY, "common_prompt": "dark fantasy style"}
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, data))
        assert screenplay.style_prompt == "dark fantasy style"
        assert screenplay.compose_prompt("a castle") == "dark fantasy style, a castle"

    def test_character_views_are_normalised(self):
        character = CharacterSpec(name="Hero", description="a knight", views=("front", "side"))
        assert character.primary_view == "front"
        assert set(character.reference_paths("chars").values()) == {
            Path("chars/hero_front.png"),
            Path("chars/hero_side.png"),
        }

    def test_slugify(self):
        assert slugify("El Monje / Héroe") == "el_monje_h_roe"

    def test_scaling_scene_durations(self):
        scenes = [
            SceneSpec(0, "A", "a", 10, 24, 360, 640, "", 10, 4.0, None, ()),
            SceneSpec(1, "B", "b", 10, 24, 360, 640, "", 10, 4.0, None, ()),
        ]
        scale_scene_durations(scenes, 30.0)
        assert scenes[0].duration == 15.0
        assert scenes[1].duration == 15.0
        assert sum(scene.duration for scene in scenes) == 30.0

    def test_scaling_last_scene_absorbs_remainder(self):
        scenes = [
            SceneSpec(0, "A", "a", 10, 24, 360, 640, "", 10, 4.0, None, ()),
            SceneSpec(1, "B", "b", 10, 24, 360, 640, "", 10, 4.0, None, ()),
            SceneSpec(2, "C", "c", 10, 24, 360, 640, "", 10, 4.0, None, ()),
        ]
        scale_scene_durations(scenes, 100.0)
        assert sum(scene.duration for scene in scenes) == 100.0

    def test_build_scenes_returns_independent_copies(self, tmp_path):
        screenplay = CharacterScreenplay.load(write_screenplay(tmp_path, MINIMAL_SCREENPLAY))
        first = screenplay.build_scenes()
        scale_scene_durations(first, 10.0)
        second = screenplay.build_scenes()
        assert second[0].duration == 2.0


class TestCharacterInputReferences:

    def _screenplay(self, tmp_path, characters):
        data = {
            **MINIMAL_SCREENPLAY,
            "characters": characters,
            "scenes": [{"name": "Opening", "prompt": "a stage", "characters": [characters[0]["name"]]}],
        }
        return write_screenplay(tmp_path, data)

    def test_input_folder_is_discovered_automatically(self, tmp_path):
        input_dir = tmp_path / "input"
        image = Path(save_image(input_dir / "axel.jpeg"))
        (input_dir / "characters.json").write_text(
            json.dumps([{"name": "Axel", "path": "input/axel.jpeg"}]),
            encoding="utf-8",
        )
        screenplay = CharacterScreenplay.load(
            self._screenplay(
                tmp_path, [{"name": "Axel", "description": "a young man"}]
            )
        )

        character = screenplay.build_characters()[0]
        assert character.has_supplied_reference
        assert Path(character.reference_image) == image

    def test_character_names_match_ignoring_case_and_punctuation(self, tmp_path):
        input_dir = tmp_path / "input"
        image = Path(save_image(input_dir / "bartender.jpeg"))
        (input_dir / "characters.json").write_text(
            json.dumps([{"name": "BarTender", "path": "input/bartender.jpeg"}]),
            encoding="utf-8",
        )
        screenplay = CharacterScreenplay.load(
            self._screenplay(
                tmp_path, [{"name": "Bartender", "description": "a waiter"}]
            )
        )

        assert Path(screenplay.build_characters()[0].reference_image) == image

    def test_images_named_after_characters_work_without_a_manifest(self, tmp_path):
        image = Path(save_image(tmp_path / "input" / "hero.jpeg"))
        screenplay = CharacterScreenplay.load(
            self._screenplay(
                tmp_path, [{"name": "Hero", "description": "a knight"}]
            )
        )

        assert Path(screenplay.build_characters()[0].reference_image) == image

    def test_explicit_reference_image_is_resolved_from_the_screenplay(self, tmp_path):
        image = Path(save_image(tmp_path / "refs" / "hero.png"))
        screenplay = CharacterScreenplay.load(
            self._screenplay(
                tmp_path,
                [
                    {
                        "name": "Hero",
                        "description": "a knight",
                        "reference_image": "refs/hero.png",
                    }
                ],
            )
        )

        assert Path(screenplay.build_characters()[0].reference_image) == image

    def test_missing_explicit_reference_image_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Reference image"):
            CharacterScreenplay.load(
                self._screenplay(
                    tmp_path,
                    [
                        {
                            "name": "Hero",
                            "description": "a knight",
                            "reference_image": "refs/missing.png",
                        }
                    ],
                )
            )


class TestCharacterDesigner:

    @staticmethod
    def fake_image_utils():
        utils = MagicMock()

        def text_to_image(prompt, output_path=None, **kwargs):
            save_image(Path(output_path))
            return str(output_path)

        utils.text_to_image.side_effect = text_to_image
        return utils

    def test_creates_one_reference_per_view(self, tmp_path):
        image_utils = self.fake_image_utils()
        designer = CharacterDesigner(
            "text-model",
            output_path=tmp_path,
            image_utils_factory=lambda *args, **kwargs: image_utils,
        )
        characters = [
            CharacterSpec("Hero", "a knight", views=("front view", "side view")),
            CharacterSpec("Mage", "an old mage"),
        ]

        references = designer.create_references(
            characters,
            tmp_path / "characters",
            style_prompt="dark fantasy",
            num_inference_steps=10,
        )

        assert image_utils.text_to_image.call_count == 3
        assert len(references["Hero"]) == 2
        assert len(references["Mage"]) == 1
        for paths in references.values():
            for path in paths:
                assert Path(path).exists()
        first_call = image_utils.text_to_image.call_args_list[0]
        assert "a knight" in first_call.args[0]
        assert "front view" in first_call.args[0]
        assert first_call.kwargs["output_path"].endswith("hero_front_view.png")

    def test_reuses_existing_references(self, tmp_path):
        image_utils = self.fake_image_utils()
        designer = CharacterDesigner(
            "text-model",
            output_path=tmp_path,
            image_utils_factory=lambda *args, **kwargs: image_utils,
        )
        character = CharacterSpec("Hero", "a knight")
        existing = character.primary_reference_path(tmp_path / "characters")
        save_image(existing)

        references = designer.create_references([character], tmp_path / "characters")

        image_utils.text_to_image.assert_not_called()
        assert references["Hero"] == [str(existing)]

    def test_supplied_reference_replaces_generated_primary_view(self, tmp_path):
        factory = MagicMock()
        designer = CharacterDesigner(
            "text-model",
            output_path=tmp_path,
            image_utils_factory=factory,
        )
        source = tmp_path / "input" / "hero.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 36), color=(200, 10, 10)).save(source)
        character = CharacterSpec("Hero", "a knight", reference_image=str(source))

        references = designer.create_references([character], tmp_path / "characters")

        factory.assert_not_called()
        assert references["Hero"] == [
            str(character.primary_reference_path(tmp_path / "characters"))
        ]
        with Image.open(references["Hero"][0]) as reference:
            assert reference.convert("RGB").getpixel((0, 0)) == (200, 10, 10)

    def test_supplied_reference_overrides_cached_sheet(self, tmp_path):
        image_utils = self.fake_image_utils()
        designer = CharacterDesigner(
            "text-model",
            output_path=tmp_path,
            image_utils_factory=lambda *args, **kwargs: image_utils,
        )
        source = tmp_path / "input" / "hero.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 36), color=(200, 10, 10)).save(source)
        character = CharacterSpec("Hero", "a knight", reference_image=str(source))
        cached = character.primary_reference_path(tmp_path / "characters")
        save_image(cached)

        designer.create_references([character], tmp_path / "characters")

        image_utils.text_to_image.assert_not_called()
        with Image.open(cached) as reference:
            assert reference.convert("RGB").getpixel((0, 0)) == (200, 10, 10)

    def test_extra_views_are_generated_after_supplied_primary(self, tmp_path):
        image_utils = self.fake_image_utils()
        designer = CharacterDesigner(
            "text-model",
            output_path=tmp_path,
            image_utils_factory=lambda *args, **kwargs: image_utils,
        )
        source = tmp_path / "input" / "hero.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        save_image(source)
        character = CharacterSpec(
            "Hero",
            "a knight",
            views=("front view", "side view"),
            reference_image=str(source),
        )

        references = designer.create_references([character], tmp_path / "characters")

        assert len(references["Hero"]) == 2
        assert image_utils.text_to_image.call_count == 1
        assert "side view" in image_utils.text_to_image.call_args.args[0]

    def test_close_destroys_image_utils(self, tmp_path):
        image_utils = self.fake_image_utils()
        designer = CharacterDesigner(
            "text-model",
            output_path=tmp_path,
            image_utils_factory=lambda *args, **kwargs: image_utils,
        )
        designer.create_references(
            [CharacterSpec("Hero", "a knight")], tmp_path / "characters"
        )
        designer.close()
        image_utils.destroy.assert_called_once()

    def test_background_prompt_has_no_character_instructions(self, tmp_path):
        image_utils = self.fake_image_utils()
        designer = CharacterDesigner(
            "text-model",
            output_path=tmp_path,
            image_utils_factory=lambda *args, **kwargs: image_utils,
        )
        output = tmp_path / "bg.png"
        designer.create_background("a valley", output)
        prompt = image_utils.text_to_image.call_args.args[0]
        assert "a valley" in prompt
        assert "no characters" in prompt


class TestCharacterBoardBuilder:

    def test_single_character_returns_reference_directly(self, tmp_path):
        reference = save_image(tmp_path / "hero.png")
        builder = CharacterBoardBuilder()
        result = builder.build(["Hero"], {"Hero": reference}, tmp_path / "board.png")
        assert result == reference
        assert not (tmp_path / "board.png").exists()

    def test_multiple_characters_are_composed(self, tmp_path):
        hero = save_image(tmp_path / "hero.png", (64, 64))
        mage = save_image(tmp_path / "mage.png", (32, 64))
        builder = CharacterBoardBuilder(gap=10, panel_height=100)
        result = builder.build(
            ["Hero", "Mage"], {"Hero": hero, "Mage": mage}, tmp_path / "board.png"
        )
        assert Path(result).exists()
        with Image.open(result) as board:
            assert board.height == 120
            assert board.width > 100

    def test_existing_board_is_refreshed_when_references_change(self, tmp_path):
        hero = save_image(tmp_path / "hero.png", (64, 64))
        mage = save_image(tmp_path / "mage.png", (64, 64))
        output = tmp_path / "board.png"
        builder = CharacterBoardBuilder(gap=10, panel_height=100)
        builder.build(["Hero", "Mage"], {"Hero": hero, "Mage": mage}, output)
        Image.new("RGB", (64, 64), color=(240, 10, 20)).save(hero)

        result = builder.build(
            ["Hero", "Mage"], {"Hero": hero, "Mage": mage}, output
        )

        with Image.open(result) as board:
            assert board.getpixel((20, 20)) == (240, 10, 20)

    def test_missing_reference_raises(self, tmp_path):
        builder = CharacterBoardBuilder()
        with pytest.raises(ValueError, match="Missing character references"):
            builder.build(["Hero"], {}, tmp_path / "board.png")


class TestFrameAnimator:

    @staticmethod
    def scene(**overrides):
        defaults = dict(
            index=0,
            name="Opening",
            prompt="a castle",
            duration=1,
            fps=3,
            height=36,
            width=64,
            negative_prompt="blurry",
            num_inference_steps=4,
            true_cfg_scale=4.0,
            seed=7,
            character_names=("Hero",),
            motion_prompt="the knight raises a sword",
            anchor_interval=2,
        )
        defaults.update(overrides)
        return SceneSpec(**defaults)

    def test_create_keyframe_uses_character_reference(self, tmp_path):
        reference = save_image(tmp_path / "characters" / "hero.png")
        engine = MagicMock()
        engine.edit.return_value = Image.new("RGB", (32, 18))
        animator = FrameAnimator(engine)
        scene = self.scene()

        result = animator.create_keyframe(
            scene,
            {"Hero": reference},
            {"Hero": "a tall knight"},
            "dark fantasy",
            tmp_path / "scene" / "keyframe.png",
        )

        assert Path(result).exists()
        engine.edit.assert_called_once()
        prompt = engine.edit.call_args.kwargs["prompt"]
        assert "identity reference" in prompt
        assert "a tall knight" in prompt
        assert "dark fantasy" in prompt
        with Image.open(result) as image:
            assert image.size == (64, 36)

    def test_keyframe_for_multiple_characters_uses_board(self, tmp_path):
        hero = save_image(tmp_path / "hero.png", (64, 64))
        mage = save_image(tmp_path / "mage.png", (64, 64))
        engine = MagicMock()
        engine.edit.return_value = Image.new("RGB", (64, 36))
        animator = FrameAnimator(engine)
        scene = self.scene(character_names=("Hero", "Mage"))

        animator.create_keyframe(
            scene,
            {"Hero": hero, "Mage": mage},
            {"Hero": "a knight", "Mage": "a mage"},
            "",
            tmp_path / "scene" / "keyframe.png",
        )

        prompt = engine.edit.call_args.kwargs["prompt"]
        assert "reference board" in prompt
        assert "1) Hero" in prompt
        assert "2) Mage" in prompt
        conditioning = engine.edit.call_args.args[0]
        assert conditioning.width > 64

    def test_create_end_keyframe_edits_only_from_start_keyframe(self, tmp_path):
        keyframe = save_image(tmp_path / "keyframe.png", (64, 36))
        engine = MagicMock()
        engine.edit.return_value = Image.new("RGB", (64, 36), color=(1, 2, 3))
        animator = FrameAnimator(engine)
        scene = self.scene(end_prompt="the knight lowers the sword")

        result = animator.create_end_keyframe(
            scene, keyframe, tmp_path / "end_keyframe.png", style_prompt="dark fantasy"
        )

        assert Path(result).exists()
        engine.edit.assert_called_once()
        assert engine.edit.call_args.args[0].size == (64, 36)
        prompt = engine.edit.call_args.kwargs["prompt"]
        assert "the knight lowers the sword" in prompt
        assert "dark fantasy" in prompt
        assert "same camera position" in prompt
        assert "Do not add, remove or replace characters" in prompt
        assert engine.edit.call_args.kwargs["seed"] == scene.seed

    def test_keyframe_video_engine_forwards_required_endpoint(self):
        video_utils = MagicMock()
        video_utils.image_to_video.return_value = "scene.mp4"
        engine = KeyframeVideoEngine(
            "test/video",
            "output",
            video_utils_factory=lambda *args, **kwargs: video_utils,
        )
        scene = self.scene(video_guidance_scale=5.0, video_num_inference_steps=12)
        start = Image.new("RGB", (64, 36))
        end = Image.new("RGB", (64, 36))

        result = engine.generate(
            scene, start, end, "scene.mp4", num_frames=21, fps=21.0
        )

        assert result == "scene.mp4"
        args, kwargs = video_utils.image_to_video.call_args
        assert args[0] is start
        assert args[1].startswith(scene.effective_motion_prompt)
        assert "no morphing or identity swaps" in args[1]
        assert kwargs["last_image"] is end
        assert "identity drift" in kwargs["negative_prompt"]
        assert kwargs["num_frames"] == 21
        assert kwargs["fps"] == 21.0

    def test_keyframe_video_engine_rejects_missing_endpoint(self):
        video_utils = MagicMock()
        engine = KeyframeVideoEngine(
            "test/video",
            "output",
            video_utils_factory=lambda *args, **kwargs: video_utils,
        )
        scene = self.scene()
        start = Image.new("RGB", (64, 36))

        with pytest.raises(ValueError, match="requires an end image"):
            engine.generate(scene, start, None, "scene.mp4", num_frames=21, fps=21.0)

        video_utils.image_to_video.assert_not_called()

    def test_animate_scene_generates_every_frame(self, tmp_path):
        keyframe = save_image(tmp_path / "keyframe.png", (64, 36))
        engine = MagicMock()
        engine.edit.side_effect = lambda *args, **kwargs: Image.new("RGB", (64, 36), color=(1, 2, 3))
        animator = FrameAnimator(engine)
        scene = self.scene(duration=1, fps=3)

        frames = animator.animate_scene(scene, keyframe, tmp_path / "frames")

        assert len(frames) == 3
        assert all(Path(frame).exists() for frame in frames)
        assert engine.edit.call_count == 2
        assert engine.edit.call_args_list[0].kwargs["prompt"].startswith("Continue this animation")
        assert "anchor keyframe" in engine.edit.call_args_list[1].kwargs["prompt"]

    def test_animate_scene_resumes_from_existing_frames(self, tmp_path):
        keyframe = save_image(tmp_path / "keyframe.png", (64, 36))
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        save_image(frames_dir / "000000.png", (64, 36))
        save_image(frames_dir / "000001.png", (64, 36))
        engine = MagicMock()
        engine.edit.return_value = Image.new("RGB", (64, 36))
        animator = FrameAnimator(engine)
        scene = self.scene(duration=1, fps=4)

        frames = animator.animate_scene(scene, keyframe, frames_dir)

        assert len(frames) == 4
        assert engine.edit.call_count == 2
        assert engine.edit.call_args_list[0].args[0].size == (64, 36)

    def test_negative_prompt_contains_identity_rules(self):
        scene = self.scene(negative_prompt="blurry")
        negative = FrameAnimator.build_negative_prompt(scene)
        assert "blurry" in negative
        assert "identity change" in negative
        assert "collage" in negative


class TestQwenImageEditEngine:

    def test_edit_delegates_to_pipeline(self):
        pipeline = MagicMock()
        pipeline.return_value.images = [Image.new("RGB", (64, 36))]
        engine = QwenImageEditEngine(pipeline=pipeline, device="cpu")

        result = engine.edit(
            Image.new("RGB", (64, 36)),
            "prompt",
            negative_prompt="negative",
            num_inference_steps=3,
            height=36,
            width=64,
        )

        assert result.size == (64, 36)
        kwargs = pipeline.call_args.kwargs
        assert kwargs["prompt"] == "prompt"
        assert kwargs["negative_prompt"] == "negative"
        assert kwargs["num_inference_steps"] == 3
        assert kwargs["height"] == 36
        assert kwargs["width"] == 64
        assert kwargs["max_sequence_length"] == 512

    def test_edit_rejects_non_pil_image(self):
        engine = QwenImageEditEngine(pipeline=MagicMock(), device="cpu")
        with pytest.raises(TypeError):
            engine.edit("not-an-image", "prompt")

    def test_destroy_releases_pipeline(self):
        engine = QwenImageEditEngine(pipeline=MagicMock(), device="cpu")
        engine.destroy()
        assert engine.pipeline is None


class TestFrameVideoBuilder:

    def test_build_encodes_frames(self, tmp_path):
        frames = [
            save_image(tmp_path / "frames" / "000000.png"),
            save_image(tmp_path / "frames" / "000001.png"),
        ]
        with patch("moviepy.video.io.ImageSequenceClip.ImageSequenceClip") as clip_cls:
            clip = clip_cls.return_value
            result = FrameVideoBuilder().build(frames, tmp_path / "scene.mp4", 12)
        assert result.endswith("scene.mp4")
        clip_cls.assert_called_once_with(frames, fps=12.0)
        clip.write_videofile.assert_called_once()
        clip.close.assert_called_once()

    def test_build_rejects_empty_frames(self, tmp_path):
        with pytest.raises(ValueError, match="No frame paths"):
            FrameVideoBuilder().build([], tmp_path / "scene.mp4", 12)

    def test_build_rejects_missing_frame(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FrameVideoBuilder().build(
                [tmp_path / "missing.png"], tmp_path / "scene.mp4", 12
            )


class FakeDesigner:
    def __init__(self, calls):
        self.calls = calls

    def create_references(self, characters, characters_dir, **kwargs):
        self.calls.append("references")
        references = {}
        for character in characters:
            paths = character.reference_paths(characters_dir)
            character_references = []
            for path in paths.values():
                save_image(path)
                character_references.append(str(path))
            references[character.name] = character_references
        return references

    def create_background(self, prompt, output_path, **kwargs):
        self.calls.append(f"background:{Path(output_path).parent.name}")
        save_image(Path(output_path))
        return str(output_path)

    def close(self):
        self.calls.append("close_designer")


class FakeAnimator:
    def __init__(self, calls):
        self.calls = calls

    def create_keyframe(self, scene, references, descriptions, style_prompt, output_path):
        self.calls.append(f"keyframe:{scene.name}")
        save_image(Path(output_path), (scene.width, scene.height))
        return str(output_path)

    def create_end_keyframe(self, scene, keyframe_path, output_path, style_prompt=""):
        self.calls.append(f"end_keyframe:{scene.name}")
        save_image(Path(output_path), (scene.width, scene.height))
        return str(output_path)

    def animate_scene(self, scene, keyframe_path, frames_dir):
        self.calls.append(f"frames:{scene.name}:{scene.frame_count}")
        frames = []
        for index in range(scene.frame_count):
            path = Path(frames_dir) / f"{index:06d}.png"
            save_image(path, (scene.width, scene.height))
            frames.append(str(path))
        return frames

    def close(self):
        self.calls.append("close_animator")


class FakeVideoBuilder:
    def __init__(self, calls):
        self.calls = calls

    def build(self, frames, output_path, fps):
        self.calls.append(f"video:{Path(output_path).stem}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"video")
        return str(output_path)


class FakeKeyframeVideoEngine:
    def __init__(self, calls):
        self.calls = calls
        self.arguments = []

    def generate(
        self, scene, start_image, end_image, output_path, num_frames, fps
    ):
        self.calls.append(f"video_generation:{scene.name}")
        self.arguments.append(
            (scene, start_image, end_image, output_path, num_frames, fps)
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"video")
        return str(output_path)


class FakeVideoUtils:
    calls = None

    @classmethod
    def concatenate_videos(cls, videos, output_file, fps=None):
        cls.calls.append("concat")
        Path(output_file).write_bytes(b"concat")
        return output_file

    @classmethod
    def add_audio_to_video(cls, video_path, audio_path, output_path):
        cls.calls.append("audio")
        Path(output_path).write_bytes(b"audio")
        return output_path

    @classmethod
    def sync_subtitles(cls, audio_path, subtitles_path, output_path=None):
        cls.calls.append("sync_subtitles")
        return subtitles_path

    @classmethod
    def add_subtitles_to_video(cls, video_path, subtitles_path, output_path):
        cls.calls.append("subtitles")
        Path(output_path).write_bytes(b"subtitles")
        return output_path


class TestCharacterAnimationProcessor:

    @pytest.fixture
    def screenplay_file(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "scenes": [
                {
                    "name": "Opening",
                    "duration": 1,
                    "prompt": "a castle",
                    "motion_prompt": "the knight raises a sword",
                    "characters": ["Hero", "Mage"],
                },
                {
                    "name": "Landscape",
                    "duration": 1,
                    "prompt": "a valley",
                    "characters": [],
                },
            ],
        }
        return write_screenplay(tmp_path, data)

    @contextmanager
    def _processor(self, screenplay_file, tmp_path, calls, video_engine=None):
        FakeVideoUtils.calls = []
        with patch(
            "utils.character_animation.processor.CharacterDesigner",
            side_effect=lambda *args, **kwargs: FakeDesigner(calls),
        ), patch("utils.character_animation.processor.QwenImageEditEngine"), patch(
            "utils.character_animation.processor.FrameAnimator",
            side_effect=lambda *args, **kwargs: FakeAnimator(calls),
        ):
            yield CharacterAnimationProcessor(
                screenplay_file,
                output_path=tmp_path / "out",
                video_builder=FakeVideoBuilder(calls),
                video_utils=FakeVideoUtils,
                video_engine=video_engine,
                evaluate_output=False,
            )

    def test_full_pipeline_order(self, screenplay_file, tmp_path):
        calls = []
        with self._processor(screenplay_file, tmp_path, calls) as processor:
            result = processor.process()

        assert result.endswith("Demo_frames_final.mp4")
        assert calls[0] == "references"
        assert "close_designer" in calls
        assert calls.index("references") < calls.index("background:01_landscape")
        assert calls.index("background:01_landscape") < calls.index("close_designer")
        assert calls.index("close_designer") < calls.index("keyframe:Opening")
        assert calls.index("keyframe:Opening") < calls.index("frames:Opening:24")
        assert calls.index("frames:Opening:24") < calls.index("video:opening")
        assert calls.index("frames:Landscape:24") < calls.index("video:landscape")
        assert calls[-1] == "close_animator"
        assert FakeVideoUtils.calls == ["concat"]

    def test_audio_and_subtitles_are_added(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "audio": "song.mp3",
            "subtitles": "subs.json",
            "auto_sync_subtitles": True,
            "scenes": [{"name": "Only", "duration": 1, "prompt": "a castle"}],
        }
        screenplay_file = write_screenplay(tmp_path, data)
        (tmp_path / "song.mp3").write_bytes(b"audio")
        (tmp_path / "subs.json").write_text("{}", encoding="utf-8")
        calls = []
        with self._processor(screenplay_file, tmp_path, calls) as processor:
            with patch.object(CharacterAnimationProcessor, "_get_audio_duration", return_value=None):
                result = processor.process()

        assert result.endswith("Demo_frames_final_subtitles.mp4")
        assert FakeVideoUtils.calls == ["concat", "audio", "sync_subtitles", "subtitles"]
        assert (tmp_path / "out" / "Demo_frames_final_subtitles.mp4").exists()

    def test_subtitles_can_be_overridden_without_changing_screenplay(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY, "subtitles": "horizontal.json"}
        screenplay_file = write_screenplay(tmp_path, data)
        vertical = tmp_path / "vertical.json"
        vertical.write_text("{}", encoding="utf-8")
        video_utils = MagicMock()
        video_utils.add_subtitles_to_video.return_value = "result.mp4"
        processor = CharacterAnimationProcessor(
            screenplay_file,
            output_path=tmp_path / "out",
            video_utils=video_utils,
            subtitles="vertical.json",
        )

        result = processor._add_subtitles("input.mp4")

        assert result == "result.mp4"
        assert video_utils.add_subtitles_to_video.call_args.args[1] == str(vertical)

    def test_durations_are_scaled_to_audio(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "audio": "song.mp3",
            "scenes": [
                {"name": "A", "duration": 1, "prompt": "a castle"},
                {"name": "B", "duration": 1, "prompt": "a valley"},
            ],
        }
        screenplay_file = write_screenplay(tmp_path, data)
        (tmp_path / "song.mp3").write_bytes(b"audio")
        calls = []
        with self._processor(screenplay_file, tmp_path, calls) as processor:
            with patch.object(CharacterAnimationProcessor, "_get_audio_duration", return_value=4.0):
                processor.process()

        assert "frames:A:48" in calls
        assert "frames:B:48" in calls

    def test_characters_only_stage_stops_before_keyframes(self, screenplay_file, tmp_path):
        calls = []
        with self._processor(screenplay_file, tmp_path, calls) as processor:
            references = processor.create_characters()

        assert set(references) == {"Hero", "Mage"}
        assert calls == ["references", "close_designer"]
        assert not any(call.startswith("keyframe:") for call in calls)

    def test_existing_scene_video_is_reused(self, tmp_path):
        screenplay_file = write_screenplay(tmp_path, MINIMAL_SCREENPLAY)
        calls = []
        with self._processor(screenplay_file, tmp_path, calls) as processor:
            scene = processor.screenplay.build_scenes()[0]
            video_path = processor.scene_video_path(scene)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"existing")

            processor.process()

        assert not any(call.startswith("frames:") for call in calls)
        assert "concat" in FakeVideoUtils.calls

    def test_supplied_start_frame_skips_keyframe_generation(self, tmp_path):
        source = save_image(tmp_path / "start.png", (640, 360))
        data = {
            **MINIMAL_SCREENPLAY,
            "scenes": [
                {
                    "name": "Opening",
                    "duration": 1,
                    "prompt": "a castle",
                    "start_frame": "start.png",
                }
            ],
        }
        screenplay_file = write_screenplay(tmp_path, data)
        calls = []
        with self._processor(screenplay_file, tmp_path, calls) as processor:
            processor.process()

            assert not any(call.startswith("keyframe:") for call in calls)
            scene = processor.screenplay.build_scenes()[0]
            assert processor.keyframe_path(scene).exists()

    def test_format_plan_does_not_load_models(self, screenplay_file, tmp_path):
        calls = []
        processor = CharacterAnimationProcessor(
            screenplay_file,
            output_path=tmp_path / "out",
            video_builder=FakeVideoBuilder(calls),
            video_utils=FakeVideoUtils,
        )
        plan = processor.format_plan()
        assert "Characters:" in plan
        assert "Total frames to paint: 48" in plan

    def test_vertical_orientation_swaps_dimensions_and_updates_prompts(self, tmp_path):
        screenplay_file = write_screenplay(tmp_path, MINIMAL_SCREENPLAY)
        processor = CharacterAnimationProcessor(
            screenplay_file,
            output_path=tmp_path / "out",
            orientation="vertical",
        )

        scenes = processor.screenplay.build_scenes()
        processor._apply_animation_overrides(scenes)
        scene = scenes[0]

        assert (scene.width, scene.height) == (360, 640)
        assert scene.prompt.startswith("Vertical portrait composition")
        assert scene.motion_prompt.startswith("Vertical portrait composition")
        processor._apply_animation_overrides(scenes)
        assert scene.prompt.count("Vertical portrait composition") == 1
        assert "Format: 360x640 (vertical)" in processor.format_plan()

    def test_explicit_dimensions_override_orientation(self, tmp_path):
        screenplay_file = write_screenplay(tmp_path, MINIMAL_SCREENPLAY)
        processor = CharacterAnimationProcessor(
            screenplay_file,
            output_path=tmp_path / "out",
            orientation="horizontal",
            width=1280,
            height=720,
        )

        scenes = processor.screenplay.build_scenes()
        processor._apply_animation_overrides(scenes)

        assert (scenes[0].width, scenes[0].height) == (1280, 720)

    def test_video_mode_requires_an_end_prompt_or_frame_before_models(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY, "animation_mode": "video"}
        screenplay_file = write_screenplay(tmp_path, data)
        calls = []

        with self._processor(screenplay_file, tmp_path, calls) as processor:
            with pytest.raises(ValueError, match="Every video scene needs"):
                processor.format_plan()

        assert calls == []

    def test_video_mode_rejects_missing_end_frame_before_models(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "animation_mode": "video",
            "scenes": [
                {
                    "name": "Ending",
                    "duration": 1,
                    "prompt": "a castle",
                    "end_frame": "missing.png",
                }
            ],
        }
        screenplay_file = write_screenplay(tmp_path, data)
        calls = []

        with self._processor(screenplay_file, tmp_path, calls) as processor:
            with pytest.raises(FileNotFoundError, match="End frame for scene 'Ending'"):
                processor.format_plan()

        assert calls == []

    def test_video_mode_uses_end_keyframe_and_releases_qwen_first(self, tmp_path):
        data = {
            **MINIMAL_SCREENPLAY,
            "animation_mode": "video",
            "scenes": [
                {
                    "name": "Action",
                    "duration": 1,
                    "prompt": "a castle",
                    "motion_prompt": "the knight raises a sword",
                    "end_prompt": "the knight lowers the sword",
                    "characters": ["Hero"],
                }
            ],
        }
        screenplay_file = write_screenplay(tmp_path, data)
        calls = []
        video_engine = FakeKeyframeVideoEngine(calls)
        with self._processor(screenplay_file, tmp_path, calls, video_engine) as processor:
            result = processor.process()
            scene = processor.screenplay.build_scenes()[0]
            processor._apply_animation_overrides([scene])

        assert result.endswith("Demo_video_final.mp4")
        assert processor.scene_video_path(scene).name == "action_video.mp4"
        assert "end_keyframe:Action" in calls
        assert calls.index("close_animator") < calls.index("video_generation:Action")
        _, start_image, end_image, _, num_frames, fps = video_engine.arguments[0]
        assert start_image.size == end_image.size == (640, 360)
        assert num_frames == 21
        assert fps == pytest.approx(21.0)

    def test_video_mode_accepts_supplied_end_frame(self, tmp_path):
        source = save_image(tmp_path / "ending.png", (640, 360))
        data = {
            **MINIMAL_SCREENPLAY,
            "animation_mode": "video",
            "scenes": [
                {
                    "name": "Ending",
                    "duration": 1,
                    "prompt": "a castle",
                    "characters": ["Hero"],
                    "end_frame": "ending.png",
                }
            ],
        }
        screenplay_file = write_screenplay(tmp_path, data)
        calls = []
        video_engine = FakeKeyframeVideoEngine(calls)
        with self._processor(screenplay_file, tmp_path, calls, video_engine) as processor:
            processor.process()

        assert "end_keyframe:Ending" not in calls
        _, _, end_image, _, _, _ = video_engine.arguments[0]
        assert end_image.size == (640, 360)

    def test_video_frame_extraction_is_cached_for_quality_checks(self, tmp_path):
        import numpy as np

        screenplay_file = write_screenplay(tmp_path, MINIMAL_SCREENPLAY)
        processor = CharacterAnimationProcessor(
            screenplay_file, output_path=tmp_path / "out", evaluate_output=True
        )
        scene = processor.screenplay.build_scenes()[0]
        clip = MagicMock()
        clip.iter_frames.return_value = [np.zeros((36, 64, 3), dtype=np.uint8)]

        with patch(
            "utils.character_animation.processor.VideoFileClip", return_value=clip
        ):
            frames = processor._extract_video_frames("scene.mp4", scene)

        assert len(frames) == 1
        assert Path(frames[0]).exists()
        clip.close.assert_called_once()

    def test_validation_happens_before_models(self, tmp_path):
        data = {**MINIMAL_SCREENPLAY}
        data.pop("characters")
        screenplay_file = write_screenplay(tmp_path, data)
        with pytest.raises(ValueError, match="characters"):
            CharacterAnimationProcessor(screenplay_file, output_path=tmp_path / "out")


class TestCervezaScreenplay:

    def test_single_canonical_screenplay_uses_the_supplied_group(self):
        screenplay_path = (
            Path(__file__).parents[1]
            / "examples"
            / "largo_viaje"
            / "el_sabio"
            / "cerveza"
            / "screenplay.json"
        )
        screenplay = CharacterScreenplay.load(screenplay_path)
        characters = screenplay.build_characters()
        scenes = screenplay.build_scenes()

        assert len(scenes) == 49
        assert all(scene.animation_mode == "video" for scene in scenes)
        assert all(scene.motion_prompt and scene.end_prompt for scene in scenes)
        assert {character.name for character in characters} == {
            "Axel",
            "BarTender",
            "Druid",
            "Monk",
            "Patricia",
            "Pedro",
            "Prima",
            "Rubo",
            "Saul",
        }
        assert all(character.has_supplied_reference for character in characters)
        assert set(scenes[-1].character_names) == {
            character.name for character in characters
        }
        assert not list(screenplay_path.parent.glob("screenplay_*.json"))
