"""Generate a scene video from its identity-anchored start/end images."""

from __future__ import annotations

from typing import Any

from PIL import Image

from utils.character_animation.frame_animator import NEGATIVE_RULES
from utils.character_animation.screenplay import join_prompts
from utils.video_utils import VideoUtils

VIDEO_IDENTITY_INSTRUCTION = (
    "Move naturally from the start image to the exact end image. Preserve every "
    "character's face, age, hairstyle, clothing and body; no morphing or identity swaps."
)


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
        end_image: Image.Image,
        output_path: str,
        num_frames: int,
        fps: float,
    ) -> str:
        if end_image is None:
            raise ValueError(
                f"Scene '{scene.name}' requires an end image for start-to-end video generation"
            )
        kwargs = {
            "negative_prompt": join_prompts(scene.negative_prompt, NEGATIVE_RULES),
            "height": scene.height,
            "width": scene.width,
            "num_frames": num_frames,
            "guidance_scale": scene.video_guidance_scale,
            "num_inference_steps": scene.video_num_inference_steps,
            "seed": scene.seed,
            "fps": fps,
            "output_path": output_path,
        }
        kwargs["last_image"] = end_image
        return self.video_utils.image_to_video(
            start_image,
            join_prompts(scene.effective_motion_prompt, VIDEO_IDENTITY_INSTRUCTION),
            **kwargs,
        )

    def close(self) -> None:
        """Release the underlying video model."""
        self.video_utils.destroy()