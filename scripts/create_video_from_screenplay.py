import argparse
import json
import sys
import logging
import warnings
from pathlib import Path

from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    import diffusers.utils.logging as diffusers_logging
    diffusers_logging.set_verbosity_error()
    diffusers_logging.disable_progress_bar()
except ImportError:
    diffusers_logging = None

try:
    import transformers.utils.logging as transformers_logging
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
except ImportError:
    transformers_logging = None

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from utils.image_utils import ImageUtils
from utils.video_utils import VideoUtils

DEFAULT_MODEL = "Lightricks/LTX-Video"


class ScreenplayProcessor:

    def __init__(self, screenplay_path, model_id=None, output_path="output"):
        self.screenplay_path = Path(screenplay_path)
        self.output_path = Path(output_path)
        self.screenplay = self._load_screenplay()
        self.model_id = model_id or self.screenplay.get("video_model", DEFAULT_MODEL)
        self.video_utils = VideoUtils(self.model_id, output_path=str(self.output_path))
        self.image_model_id = self.screenplay.get("image_model")

    def _load_screenplay(self):
        logger.info("Loading screenplay from %s", self.screenplay_path)
        with open(self.screenplay_path, "r", encoding="utf-8") as file:
            screenplay = json.load(file)
        self._validate_screenplay(screenplay)
        logger.info("Screenplay '%s' loaded with %d scenes", screenplay["name"], len(screenplay["scenes"]))
        return screenplay

    @staticmethod
    def _validate_screenplay(screenplay):
        required_fields = ["name", "scenes"]
        for field in required_fields:
            if field not in screenplay:
                raise ValueError(f"Screenplay is missing required field: '{field}'")
        if not isinstance(screenplay["scenes"], list) or len(screenplay["scenes"]) == 0:
            raise ValueError("Screenplay must contain at least one scene")
        for index, scene in enumerate(screenplay["scenes"]):
            if "prompt" not in scene:
                raise ValueError(f"Scene {index} is missing required field: 'prompt'")

    def _calculate_num_frames(self, duration, fps):
        return int(duration * fps)

    def _generate_initial_image(self, prompt, negative_prompt="", height=480, width=704,
                                num_inference_steps=50, seed=None):
        image_utils = ImageUtils(self.image_model_id, output_path=str(self.output_path))
        image_path = image_utils.text_to_image(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_path=str(self.output_path / "initial_frame.png")
        )
        logger.info("Initial image generated at %s", image_path)
        return Image.open(image_path)

    def _build_full_prompt(self, scene_prompt):
        common_prompt = self.screenplay.get("common_prompt", "")
        if common_prompt:
            return f"{common_prompt}, {scene_prompt}"
        return scene_prompt

    def process(self):
        name = self.screenplay["name"]
        height = self.screenplay.get("height", 480)
        width = self.screenplay.get("width", 704)
        fps = self.screenplay.get("fps", 24)
        negative_prompt = self.screenplay.get("negative_prompt", "")
        num_inference_steps = self.screenplay.get("num_inference_steps", 50)
        scenes = self.screenplay["scenes"]

        logger.info("Processing screenplay '%s' (%d scenes, %dx%d, %d fps)", name, len(scenes), width, height, fps)

        generated_files = []
        last_frame = None
        for index, scene in enumerate(scenes):
            scene_name = scene.get("name", f"scene_{index}")
            prompt = self._build_full_prompt(scene["prompt"])
            duration = scene.get("duration", 5)
            scene_negative_prompt = scene.get("negative_prompt", negative_prompt)
            scene_height = scene.get("height", height)
            scene_width = scene.get("width", width)
            scene_fps = scene.get("fps", fps)
            num_frames = self._calculate_num_frames(duration, scene_fps)
            scene_num_inference_steps = scene.get("num_inference_steps", num_inference_steps)
            seed = scene.get("seed", None)

            logger.info("Generating scene %d/%d: '%s' (prompt: '%s', frames: %d)",
                        index + 1, len(scenes), scene_name, prompt, num_frames)

            output_file_name = str(self.output_path / f"{name}_{scene_name}.mp4")

            if last_frame is None and self.image_model_id:
                logger.info("Generating initial image with model '%s'", self.image_model_id)
                last_frame = self._generate_initial_image(
                    prompt,
                    negative_prompt=scene_negative_prompt,
                    height=scene_height,
                    width=scene_width,
                    num_inference_steps=scene_num_inference_steps,
                    seed=seed
                )

            if last_frame is not None:
                logger.info("Using last frame from previous scene for morphing effect")
                output_file = self.video_utils.image_to_video(
                    last_frame,
                    prompt,
                    negative_prompt=scene_negative_prompt,
                    height=scene_height,
                    width=scene_width,
                    num_frames=num_frames,
                    num_inference_steps=scene_num_inference_steps,
                    seed=seed,
                    output_path=output_file_name
                )
            else:
                output_file = self.video_utils.text_to_video(
                    prompt,
                    negative_prompt=scene_negative_prompt,
                    height=scene_height,
                    width=scene_width,
                    num_frames=num_frames,
                    num_inference_steps=scene_num_inference_steps,
                    seed=seed,
                    output_path=output_file_name
                )

            generated_files.append(output_file)
            last_frame = VideoUtils.extract_last_frame(output_file)
            logger.info("Scene '%s' saved to %s", scene_name, output_file)

        logger.info("All %d scenes generated successfully", len(generated_files))

        final_output = str(self.output_path / f"{name}_final.mp4")
        VideoUtils.concatenate_videos(generated_files, final_output, fps=fps)
        logger.info("Final video saved to %s", final_output)

        return final_output


def main():
    parser = argparse.ArgumentParser(description="Generate videos from a screenplay JSON file")
    parser.add_argument("--screenplay", type=str, required=True, help="Path to the screenplay JSON file")
    parser.add_argument("--output", type=str, default="output", help="Output directory for generated videos")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model ID")

    args = parser.parse_args()

    processor = ScreenplayProcessor(args.screenplay, model_id=args.model, output_path=args.output)
    final_video = processor.process()

    logger.info("Final video: %s", final_video)


if __name__ == "__main__":
    main()
