import argparse
import sys
import logging
import warnings
from pathlib import Path

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

from utils.video_utils import VideoUtils


def main():
    parser = argparse.ArgumentParser(description="Generate videos using diffusion models")
    parser.add_argument("--prompt", type=str, required=True, help="Description of the video to generate")
    parser.add_argument("--output", type=str, help="Output path for the video")
    parser.add_argument("--model", type=str, default="Lightricks/LTX-Video", help="HuggingFace model ID")
    parser.add_argument("--height", type=int, default=480, help="Video height in pixels")
    parser.add_argument("--width", type=int, default=704, help="Video width in pixels")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    logger.info("Loading model %s...", args.model)
    generator = VideoUtils(model_id=args.model)

    logger.info("Generating video for prompt: '%s'...", args.prompt)
    output_path = generator.text_to_video(
        args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output_path=args.output
    )

    logger.info("Video saved at: %s", output_path)


if __name__ == "__main__":
    main()
