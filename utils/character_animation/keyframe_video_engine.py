"""Generate a scene video from its identity-anchored start/end images."""

from __future__ import annotations

from typing import Any

from PIL import Image

from utils.video_utils import VideoUtils


class KeyframeVideoEngine:
    """Small adapter around the repository's image-to-video API."""

    def __init__(
        self,
        model_id: str,
        output_path: str,
        video_utils_factory: Any = VideoUtils,
    ):
        self.video_utils = video_utils_factory(model_id, output_path=output_path)

    def generate(
        self,
        scene,
        start_image: Image.Image,
        end_image: Image.Image | None,
        output_path: str,
        num_frames: int,
        fps: float,
    ) -> str:
        kwargs = {
            "negative_prompt": scene.negative_prompt,
            "height": scene.height,
            "width": scene.width,
            "num_frames": num_frames,
            "guidance_scale": scene.video_guidance_scale,
            "num_inference_steps": scene.video_num_inference_steps,
            "seed": scene.seed,
            "fps": fps,
            "output_path": output_path,
        }
        if end_image is not None:
            kwargs["last_image"] = end_image
        return self.video_utils.image_to_video(
            start_image,
            scene.effective_motion_prompt,
            **kwargs,
        )

    def close(self) -> None:
        """Release the underlying video model."""
        self.video_utils.destroy()