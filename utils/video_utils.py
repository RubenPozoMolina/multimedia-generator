import logging
from pathlib import Path

from PIL import Image
from moviepy import VideoFileClip, concatenate_videoclips

from utils.video_models.base_video_model import BaseVideoModel
from utils.video_models.ltx_video_model import LTXVideoModel
from utils.video_models.wan_model import WanModel

logger = logging.getLogger(__name__)

models = [
    {
        "name": "Lightricks/LTX-Video",
        "model_class": LTXVideoModel
    },
    {
        "name": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "model_class": WanModel
    }
]


class VideoUtils:
    model = BaseVideoModel()

    def get_model(self, model_id, output_path="output") -> BaseVideoModel:
        for model_entry in models:
            if model_id == model_entry["name"]:
                self.model = model_entry["model_class"](model_id, output_path=output_path)
                return self.model
        raise ValueError(f"Model {model_id} not found in the list of supported models.")

    def __init__(self, model_id, output_path="output"):
        try:
            self.model = self.get_model(model_id, output_path=output_path)
            self.model.load_model(model_id)
        except Exception as e:
            logger.error("Error loading model: %s", e)
            raise

    def text_to_video(
            self,
            prompt,
            negative_prompt="",
            height=480,
            width=704,
            num_frames=81,
            guidance_scale=7.5,
            num_inference_steps=50,
            seed=None,
            output_path=None
    ):
        output_file = self.model.text_to_video(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_file_name=output_path
        )
        return str(output_file)

    def image_to_video(
            self,
            image,
            prompt,
            negative_prompt="",
            height=480,
            width=704,
            num_frames=81,
            guidance_scale=7.5,
            num_inference_steps=50,
            seed=None,
            output_path=None
    ):
        output_file = self.model.image_to_video(
            image,
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_file_name=output_path
        )
        return str(output_file)

    @staticmethod
    def extract_last_frame(video_path):
        clip = VideoFileClip(str(video_path))
        try:
            last_frame = clip.get_frame(clip.duration - (1.0 / clip.fps))
            return Image.fromarray(last_frame)
        finally:
            clip.close()

    @staticmethod
    def concatenate_videos(video_paths, output_file, fps=None):
        if not video_paths:
            raise ValueError("No video paths provided for concatenation.")

        logger.info("Concatenating %d videos into %s", len(video_paths), output_file)
        clips = []
        try:
            for video_path in video_paths:
                clip = VideoFileClip(str(video_path))
                clips.append(clip)

            final_clip = concatenate_videoclips(clips)

            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            final_fps = fps if fps is not None else clips[0].fps
            final_clip.write_videofile(str(output_path), fps=final_fps, logger=None)
            logger.info("Concatenated video saved to %s", output_path)
            return str(output_path)
        finally:
            for clip in clips:
                clip.close()
