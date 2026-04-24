import logging
from pathlib import Path

from PIL import Image
from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips

from utils.video_models.base_video_model import BaseVideoModel
from utils.video_models.ltx_video_model import LTXVideoModel
# from utils.video_models.ltx2_model import LTX2Model
from utils.video_models.wan_model import WanModel

logger = logging.getLogger(__name__)

models = [
    {
        "name": "Lightricks/LTX-Video",
        "model_class": LTXVideoModel
    },
    # {
    #     "name": "Lightricks/LTX-2",
    #     "model_class": LTX2Model
    # },
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
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
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
        for video_path in video_paths:
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
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

    @staticmethod
    def add_audio_to_video(video_path, audio_path, output_path):
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        video_path = str(video_path)
        audio_path = str(audio_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Adding audio '%s' to video '%s'", audio_path, video_path)
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        try:
            video_with_audio = video_clip.with_audio(audio_clip)
            video_with_audio.write_videofile(str(output_file), fps=video_clip.fps, logger=None)
            logger.info("Video with audio saved to %s", output_file)
            return str(output_file)
        finally:
            audio_clip.close()
            video_clip.close()
