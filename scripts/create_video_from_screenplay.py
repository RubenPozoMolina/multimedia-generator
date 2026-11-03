import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Allow running this file directly (python scripts/create_video_from_screenplay.py)
# while keeping the project root importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusers.utils import load_image
from moviepy import AudioFileClip

from utils.image_utils import ImageUtils
from utils.video_utils import VideoUtils

DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 704
DEFAULT_FPS = 24
DEFAULT_DURATION = 5
DEFAULT_STEPS = 50
DEFAULT_VIDEO_MODEL = "Lightricks/LTX-Video"


def _calculate_num_frames(duration: float, fps: float) -> int:
    """Return the number of frames needed to cover a scene duration."""
    return max(1, int(round(float(duration) * float(fps))))


class ScreenplayProcessor:
    """Generate a video from a JSON screenplay.

    The processor resolves scene-level overrides, optionally generates one
    image per scene, turns every scene into a video, concatenates the result
    and finally adds the audio and subtitles declared by the screenplay.
    """

    def __init__(self, screenplay: str | os.PathLike, output_path: str = "output", logger: logging.Logger | None = None):
        self.screenplay_path = Path(screenplay)
        self.output_path = str(output_path)
        self.logger = logger or logging.getLogger(__name__)
        self.screenplay: dict[str, Any] = {}
        self.model_id = DEFAULT_VIDEO_MODEL
        self.video_utils: VideoUtils | None = None
        self._load()

    def _load(self) -> None:
        self.logger.info("Loading screenplay: %s", self.screenplay_path)
        with open(self.screenplay_path, "r", encoding="utf-8") as screenplay_file:
            self.screenplay = json.load(screenplay_file)

        self._validate_screenplay(self.screenplay)
        self.model_id = self.screenplay.get("video_model", DEFAULT_VIDEO_MODEL)
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_screenplay(screenplay: dict[str, Any]) -> None:
        if "name" not in screenplay:
            raise ValueError("Screenplay is missing required field: 'name'")
        if "scenes" not in screenplay:
            raise ValueError("Screenplay is missing required field: 'scenes'")
        if not screenplay["scenes"]:
            raise ValueError("Screenplay must contain at least one scene")

        for index, scene in enumerate(screenplay["scenes"]):
            if not scene.get("prompt"):
                raise ValueError(f"Scene {index} is missing required field: 'prompt'")

    def _build_full_prompt(self, scene_prompt: str) -> str:
        common_prompt = (self.screenplay.get("common_prompt") or "").strip()
        if common_prompt:
            return f"{common_prompt}, {scene_prompt.strip()}"
        return scene_prompt

    def _resolve_asset(self, relative_path: str) -> str:
        return str(self.screenplay_path.parent / relative_path)

    @staticmethod
    def _get_audio_duration(audio_path: str) -> float | None:
        """Read the audio duration with MoviePy, returning None on failure."""
        try:
            clip = AudioFileClip(str(audio_path))
            try:
                return float(clip.duration)
            finally:
                clip.close()
        except Exception as error:  # pragma: no cover - depends on external media files
            logging.getLogger(__name__).warning("Could not read audio duration from %s: %s", audio_path, error)
            return None

    @staticmethod
    def _recalculate_scene_durations(configs: list[dict[str, Any]], audio_duration: float) -> None:
        """Scale scene durations so the montage matches the audio length."""
        total_duration = sum(float(config.get("duration", 0)) for config in configs)
        if total_duration <= 0 or audio_duration <= 0:
            return

        scale = float(audio_duration) / total_duration
        for config in configs:
            config["duration"] = float(config.get("duration", 0)) * scale

        # The last scene absorbs the accumulated rounding error so the final
        # montage is as close as possible to the audio duration.
        configs[-1]["duration"] = float(audio_duration) - sum(
            float(config["duration"]) for config in configs[:-1]
        )

    def _build_scene_configs(self) -> list[dict[str, Any]]:
        global_height = int(self.screenplay.get("height", DEFAULT_HEIGHT))
        global_width = int(self.screenplay.get("width", DEFAULT_WIDTH))
        global_fps = float(self.screenplay.get("fps", DEFAULT_FPS))
        global_negative_prompt = self.screenplay.get("negative_prompt", "")
        global_steps = int(self.screenplay.get("num_inference_steps", DEFAULT_STEPS))

        configs: list[dict[str, Any]] = []
        for index, scene in enumerate(self.screenplay["scenes"]):
            scene_name = scene.get("name", f"scene_{index}")
            configs.append(
                {
                    "index": index,
                    "scene_name": scene_name,
                    "prompt": self._build_full_prompt(scene["prompt"]),
                    "duration": float(scene.get("duration", DEFAULT_DURATION)),
                    "height": int(scene.get("height", global_height)),
                    "width": int(scene.get("width", global_width)),
                    "fps": float(scene.get("fps", global_fps)),
                    "negative_prompt": scene.get("negative_prompt", global_negative_prompt) or "",
                    "num_inference_steps": int(scene.get("num_inference_steps", global_steps)),
                    "seed": scene.get("seed"),
                    "image_path": str(Path(self.output_path) / f"{scene_name}_frame.png"),
                    "video_path": str(
                        Path(self.output_path) / f"{self.screenplay['name']}_{scene_name}_chunk0.mp4"
                    ),
                }
            )
        return configs

    def _generate_images(self, configs: list[dict[str, Any]]) -> None:
        image_model = self.screenplay.get("image_model")
        if not image_model:
            return

        image_utils_instances: list[ImageUtils] = []
        try:
            for config in configs:
                if os.path.exists(config["image_path"]):
                    self.logger.info("Found existing image: %s", config["image_path"])
                    continue

                # A per-scene instance keeps the model lifecycle simple; the
                # ImageUtils cache makes subsequent instances reuse the loaded
                # pipeline instead of loading the weights again.
                image_utils = ImageUtils(image_model, self.output_path)
                image_utils_instances.append(image_utils)
                self.logger.info("Generating image for scene: %s", config["scene_name"])
                generated_image = image_utils.text_to_image(
                    config["prompt"],
                    negative_prompt=config["negative_prompt"],
                    height=config["height"],
                    width=config["width"],
                    num_inference_steps=config["num_inference_steps"],
                    seed=config["seed"],
                    output_path=config["image_path"],
                )
                self.logger.info("Generated image: %s", generated_image)
        finally:
            for image_utils in image_utils_instances:
                image_utils.destroy()

    def _generate_videos(self, configs: list[dict[str, Any]]) -> list[str]:
        videos: list[str] = []
        previous_frame = None
        image_model = self.screenplay.get("image_model")
        self.video_utils = None

        try:
            for config in configs:
                video_path = config["video_path"]
                if not os.path.exists(video_path):
                    if self.video_utils is None:
                        self.video_utils = VideoUtils(self.model_id, self.output_path)

                    num_frames = _calculate_num_frames(config["duration"], config["fps"])
                    self.logger.info("Generating video for scene: %s", config["scene_name"])

                    if image_model:
                        image = load_image(config["image_path"])
                        generated_video = self.video_utils.image_to_video(
                            image,
                            config["prompt"],
                            negative_prompt=config["negative_prompt"],
                            height=config["height"],
                            width=config["width"],
                            num_inference_steps=config["num_inference_steps"],
                            num_frames=num_frames,
                            fps=config["fps"],
                            seed=config["seed"],
                            output_path=video_path,
                        )
                    elif config["index"] == 0:
                        generated_video = self.video_utils.text_to_video(
                            config["prompt"],
                            negative_prompt=config["negative_prompt"],
                            height=config["height"],
                            width=config["width"],
                            num_inference_steps=config["num_inference_steps"],
                            num_frames=num_frames,
                            seed=config["seed"],
                            output_path=video_path,
                        )
                    else:
                        generated_video = self.video_utils.image_to_video(
                            previous_frame,
                            config["prompt"],
                            negative_prompt=config["negative_prompt"],
                            height=config["height"],
                            width=config["width"],
                            num_inference_steps=config["num_inference_steps"],
                            num_frames=num_frames,
                            fps=config["fps"],
                            seed=config["seed"],
                            output_path=video_path,
                        )

                    if isinstance(generated_video, (str, os.PathLike)):
                        video_path = str(generated_video)
                    self.logger.info("Generated video: %s", video_path)
                    self.video_utils.clean_cache()
                else:
                    self.logger.info("Found existing video: %s", video_path)

                videos.append(video_path)
                previous_frame = VideoUtils.extract_last_frame(video_path)
        finally:
            if self.video_utils:
                self.logger.info("Destroying video model to free memory")
                self.video_utils.destroy()

        return videos

    def _concatenate_videos(self, videos: list[str]) -> str:
        final_output_filename = str(Path(self.output_path) / f"{self.screenplay['name']}_final.mp4")
        fps = self.screenplay.get("fps", DEFAULT_FPS)
        result = VideoUtils.concatenate_videos(videos, final_output_filename, fps=fps)
        return str(result) if isinstance(result, (str, os.PathLike)) else final_output_filename

    def _add_audio(self, video_path: str) -> str:
        audio_relative_path = self.screenplay.get("audio")
        if not audio_relative_path:
            return video_path

        audio_path = self._resolve_asset(audio_relative_path)
        audio_output = str(Path(self.output_path) / f"{self.screenplay['name']}_final_audio.mp4")
        result = VideoUtils.add_audio_to_video(video_path, audio_path, audio_output)
        return str(result) if isinstance(result, (str, os.PathLike)) else audio_output

    def _add_subtitles(self, video_path: str) -> str:
        subtitles_relative_path = self.screenplay.get("subtitles")
        if not subtitles_relative_path:
            return video_path

        subtitles_path = self._resolve_asset(subtitles_relative_path)
        audio_relative_path = self.screenplay.get("audio")

        if self.screenplay.get("auto_sync_subtitles") and audio_relative_path:
            audio_path = self._resolve_asset(audio_relative_path)
            self.logger.info("Auto-syncing subtitles with audio")
            VideoUtils.sync_subtitles(audio_path, subtitles_path)

        subtitles_output = str(Path(self.output_path) / f"{self.screenplay['name']}_final_subtitles.mp4")
        result = VideoUtils.add_subtitles_to_video(video_path, subtitles_path, subtitles_output)
        return str(result) if isinstance(result, (str, os.PathLike)) else subtitles_output

    def process(self) -> str:
        """Generate the final video and return its path."""
        configs = self._build_scene_configs()

        audio_relative_path = self.screenplay.get("audio")
        if audio_relative_path:
            audio_duration = self._get_audio_duration(self._resolve_asset(audio_relative_path))
            if audio_duration:
                self.logger.info("Scaling scene durations to audio duration: %.2fs", audio_duration)
                self._recalculate_scene_durations(configs, audio_duration)

        self._generate_images(configs)
        videos = self._generate_videos(configs)
        final_video = self._concatenate_videos(videos)
        final_video = self._add_audio(final_video)
        final_video = self._add_subtitles(final_video)
        return final_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate videos from a screenplay JSON file")
    parser.add_argument("--screenplay", type=str, required=True, help="Path to the screenplay JSON file")
    parser.add_argument("--output", type=str, default="output", help="Output directory for generated videos")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    processor = ScreenplayProcessor(args.screenplay, output_path=args.output, logger=logger)
    final_video = processor.process()
    logger.info("Final video: %s", final_video)


if __name__ == "__main__":
    main()
