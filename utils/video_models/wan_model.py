import logging
import os

import numpy as np
import torch

from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

from utils.video_models.base_video_model import BaseVideoModel

logger = logging.getLogger(__name__)


class WanModel(BaseVideoModel):

    def text_to_video(self, prompt, **kwargs):
        raise NotImplementedError(
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers is an image-to-video only model. "
            "Use image_to_video instead."
        )

    def load_model(self, model_id: str):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.model_id = model_id
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dtype = torch.bfloat16

        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype
        )
        self.pipeline.to(self.device)
        self.pipeline.enable_attention_slicing()
        self.pipeline.vae.enable_tiling()

    def image_to_video(
            self,
            image,
            prompt,
            negative_prompt="",
            height=480,
            width=832,
            num_frames=81,
            guidance_scale=5.0,
            num_inference_steps=40,
            seed=None,
            fps=24,
            output_file_name=None
    ):
        max_area = height * width
        aspect_ratio = image.height / image.width
        mod_value = self.pipeline.vae_scale_factor_spatial * self.pipeline.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width, height))
        output_file = self.get_file_name(output_file_name)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        video = self.pipeline(
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
