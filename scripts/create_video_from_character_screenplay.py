"""Generate a character-consistent video from a character screenplay.

This entry point is additive: it uses a new screenplay schema with a
``characters`` block and never modifies the classic
``create_video_from_screenplay.py`` pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.character_animation import CharacterAnimationProcessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a character-consistent video from a character screenplay"
    )
    parser.add_argument("--screenplay", required=True, help="Path to the character screenplay JSON file")
    parser.add_argument("--output", default="output", help="Output directory for generated assets")
    parser.add_argument(
        "--character-model",
        default=None,
        help="Text-to-image model used to create the character reference sheets",
    )
    parser.add_argument(
        "--frame-model",
        default=None,
        help="Image-edit model used for scene keyframes and frame animation",
    )
    parser.add_argument(
        "--animation-mode",
        choices=["frames", "video"],
        default=None,
        help="Override the screenplay animation mode",
    )
    parser.add_argument(
        "--video-model",
        default=None,
        help="Override the screenplay image-to-video model",
    )
    parser.add_argument(
        "--orientation",
        choices=["horizontal", "vertical"],
        default=None,
        help="Override the screenplay orientation, swapping its dimensions when needed",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override the width of every scene",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override the height of every scene",
    )
    parser.add_argument(
        "--subtitles",
        default=None,
        help="Override the subtitles JSON path (relative to the screenplay)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the characters, scenes and frame counts without loading any model",
    )
    parser.add_argument(
        "--characters-only",
        action="store_true",
        help="Create the character reference sheets and stop before animating",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Repeat a scene with strong motion when the quality check fails (default: 1)",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Disable the automatic motion/degradation check",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    processor = CharacterAnimationProcessor(
        args.screenplay,
        output_path=args.output,
        logger=logger,
        character_model=args.character_model,
        frame_model=args.frame_model,
        animation_mode=args.animation_mode,
        video_model=args.video_model,
        orientation=args.orientation,
        width=args.width,
        height=args.height,
        subtitles=args.subtitles,
        max_retries=args.max_retries,
        evaluate_output=not args.no_evaluate,
    )

    if args.dry_run:
        print(processor.format_plan())
        return

    if args.characters_only:
        references = processor.create_characters()
        for name, paths in references.items():
            for path in paths:
                logger.info("Character %s: %s", name, path)
        return

    final_video = processor.process()
    logger.info("Final video: %s", final_video)


if __name__ == "__main__":
    main()
