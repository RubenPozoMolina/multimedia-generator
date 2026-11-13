#!/usr/bin/env bash
# Generate the full "Cerveza" music video in vertical 9:16 format (49 scenes,
# full audio and subtitles) for Shorts/Reels. Trim the final video afterwards
# if a shorter clip is needed.
#
# Usage:
#   scripts/create_cerveza_vertical.sh [extra options]
#
# Extra options are forwarded to create_video_from_character_screenplay.py,
# e.g. --dry-run, --characters-only, --no-evaluate, --log-level DEBUG.
#
# Environment overrides:
#   SCREENPLAY      screenplay JSON (default: cerveza/screenplay.json)
#   OUTPUT_DIR      output directory (default: output/cerveza_vertical_group)
#   SUBTITLES       subtitles JSON (default: cerveza/subtitles_vertical.json)
#   ANIMATION_MODE  video | frames (default: video)
#   VIDEO_MODEL     image-to-video model (default: Wan-AI/Wan2.2-I2V-A14B-Diffusers)
#   PYTHON          python interpreter (default: .venv/bin/python if present)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SCREENPLAY="${SCREENPLAY:-examples/largo_viaje/el_sabio/cerveza/screenplay.json}"
SUBTITLES="${SUBTITLES:-subtitles_vertical.json}"
OUTPUT_DIR="${OUTPUT_DIR:-output/cerveza_vertical_group}"
ANIMATION_MODE="${ANIMATION_MODE:-video}"
VIDEO_MODEL="${VIDEO_MODEL:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
        PYTHON="$PROJECT_ROOT/.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi

if [[ ! -f "$SCREENPLAY" ]]; then
    echo "Screenplay not found: $SCREENPLAY" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Screenplay: $SCREENPLAY"
echo "Output:     $OUTPUT_DIR"
echo "Mode:       $ANIMATION_MODE ($VIDEO_MODEL)"
echo "Python:     $PYTHON"

exec "$PYTHON" scripts/create_video_from_character_screenplay.py \
    --screenplay "$SCREENPLAY" \
    --output "$OUTPUT_DIR" \
    --orientation vertical \
    --subtitles "$SUBTITLES" \
    --animation-mode "$ANIMATION_MODE" \
    --video-model "$VIDEO_MODEL" \
    "$@"
