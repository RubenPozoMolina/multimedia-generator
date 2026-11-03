"""Encode ordered frame images into a video clip."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


class FrameVideoBuilder:
    """Render a list of PNG frames as an MP4 video with MoviePy."""

    def __init__(self, codec: str = "libx264", pixel_format: str = "yuv420p"):
        self.codec = codec
        self.pixel_format = pixel_format

    def build(
        self,
        frame_paths: Sequence[str | Path],
        output_path: str | Path,
        fps: float,
    ) -> str:
        """Encode the frames and return the resulting video path."""
        if not frame_paths:
            raise ValueError("No frame paths provided for video encoding")
        if float(fps) <= 0:
            raise ValueError("fps must be positive")

        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        resolved_frames = [str(Path(frame)) for frame in frame_paths]
        for frame in resolved_frames:
            if not Path(frame).exists():
                raise FileNotFoundError(f"Frame not found: {frame}")

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Encoding %d frames into %s", len(resolved_frames), destination)

        clip = ImageSequenceClip(resolved_frames, fps=float(fps))
        try:
            clip.write_videofile(
                str(destination),
                codec=self.codec,
                audio=False,
                logger=None,
                pixel_format=self.pixel_format,
            )
        finally:
            clip.close()
        return str(destination)
