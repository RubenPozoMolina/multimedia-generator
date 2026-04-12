import logging

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, WanImageToVideoPipeline
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
        self.model_id = model_id
        vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        self.pipeline = WanImageToVideoPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=8.0,
            use_dynamic_shifting=False,
        )
        self.pipeline.enable_model_cpu_offload()

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
            output_file_name=None
    ):
        output_file = self.get_file_name(output_file_name)
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
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
        export_to_video(video, str(output_file), fps=16)
        logger.info("Video saved to %s", output_file)
        return output_file
