import logging

import torch
from diffusers import LTXPipeline, LTXImageToVideoPipeline
from diffusers.utils import export_to_video

from utils.video_models.base_video_model import BaseVideoModel

logger = logging.getLogger(__name__)


class LTXVideoModel(BaseVideoModel):

    def load_model(self, model_id: str):
        self.model_id = model_id
        self.pipeline = LTXPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.to(self.device)

    def text_to_video(
            self,
            prompt,
            negative_prompt="",
            height=480,
            width=704,
            num_frames=161,
            guidance_scale=3.0,
            num_inference_steps=50,
            seed=None,
            fps=24,
            output_file_name=None
    ):
        height = self.align_dimension(height)
        width = self.align_dimension(width)
        output_file = self.get_file_name(output_file_name)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        video = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).frames[0]
        export_to_video(video, str(output_file), fps=fps)
        logger.info("Video saved to %s", output_file)
        return output_file

    def image_to_video(
            self,
            image,
            prompt,
            negative_prompt="",
            height=480,
            width=704,
            num_frames=161,
            guidance_scale=3.0,
            num_inference_steps=50,
            seed=None,
            fps=24,
            output_file_name=None
    ):
        height = self.align_dimension(height)
        width = self.align_dimension(width)
        i2v_pipeline = LTXImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        i2v_pipeline.to(self.device)
        output_file = self.get_file_name(output_file_name)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        video = i2v_pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).frames[0]
        export_to_video(video, str(output_file), fps=fps)
        logger.info("Video saved to %s", output_file)
        return output_file
