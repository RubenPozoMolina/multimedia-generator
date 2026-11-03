"""Evaluate a generated character animation and decide whether to repeat it.

The evaluator measures motion between frames, image degradation and identity
drift.  It can inspect a full output directory, a single scene video or a frame
directory, and exits with status ``2`` when a retry is recommended so it can be
used in scripts.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.character_animation.quality_evaluator import (
    EvaluationReport,
    EvaluationThresholds,
    VideoQualityEvaluator,
)

RETRY_EXIT_CODE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate motion, degradation and identity drift of a character animation"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--output",
        help="Output directory produced by create_video_from_character_screenplay.py",
    )
    source.add_argument("--video", help="Path to a single video file")
    source.add_argument("--frames", help="Path to a directory with frame images")
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Write the full report as JSON to this path",
    )
    parser.add_argument(
        "--fail-on-retry",
        action="store_true",
        help=f"Exit with status {RETRY_EXIT_CODE} when a retry is recommended",
    )
    parser.add_argument(
        "--max-static-ratio",
        type=float,
        default=None,
        help="Maximum fraction of identical consecutive frames (default: 0.5)",
    )
    parser.add_argument(
        "--min-sharpness-retention",
        type=float,
        default=None,
        help="Minimum end/start sharpness ratio (default: 0.6)",
    )
    parser.add_argument(
        "--min-keyframe-similarity",
        type=float,
        default=None,
        help="Minimum correlation against the first frame (default: 0.5)",
    )
    parser.add_argument(
        "--max-brightness-drift",
        type=float,
        default=None,
        help="Maximum normalized brightness change (default: 0.10)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def build_thresholds(args: argparse.Namespace) -> EvaluationThresholds:
    defaults = EvaluationThresholds()
    return EvaluationThresholds(
        analysis_width=defaults.analysis_width,
        min_motion_mean=defaults.min_motion_mean,
        min_changed_ratio=defaults.min_changed_ratio,
        max_static_ratio=(
            args.max_static_ratio
            if args.max_static_ratio is not None
            else defaults.max_static_ratio
        ),
        min_sharpness_retention=(
            args.min_sharpness_retention
            if args.min_sharpness_retention is not None
            else defaults.min_sharpness_retention
        ),
        max_brightness_drift=(
            args.max_brightness_drift
            if args.max_brightness_drift is not None
            else defaults.max_brightness_drift
        ),
        max_saturation_drift=defaults.max_saturation_drift,
        min_keyframe_similarity=(
            args.min_keyframe_similarity
            if args.min_keyframe_similarity is not None
            else defaults.min_keyframe_similarity
        ),
        max_frames=defaults.max_frames,
    )


def build_report(args: argparse.Namespace, evaluator: VideoQualityEvaluator) -> EvaluationReport:
    if args.output:
        return evaluator.evaluate_output(args.output)
    if args.video:
        return EvaluationReport(
            target=args.video,
            scenes=(evaluator.evaluate_video(args.video),),
        )
    return EvaluationReport(
        target=args.frames,
        scenes=(evaluator.evaluate_frame_directory(args.frames),),
    )


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    evaluator = VideoQualityEvaluator(thresholds=build_thresholds(args))
    report = build_report(args, evaluator)
    print(report.format_report())

    if args.json_path:
        destination = Path(args.json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.fail_on_retry and report.needs_retry:
        raise SystemExit(RETRY_EXIT_CODE)


if __name__ == "__main__":
    main()
