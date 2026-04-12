import logging

import torch
from diffusers import QwenImageEditPipeline

from utils.image_models.base_image_model import BaseImageModel

logger = logging.getLogger(__name__)


class QwenImageEditModel(BaseImageModel):

    def load_model(self, model_id: str):
        self.model_id = model_id
        self.pipeline = QwenImageEditPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.enable_model_cpu_offload()

    def text_to_image(self, prompt, **kwargs):
        raise NotImplementedError(
            "Qwen/Qwen-Image-Edit is an image-to-image only model. "
            "Use image_to_image instead."
        )

    def image_to_image(
            self,
            image,
            prompt="",
            negative_prompt=" ",
            true_cfg_scale=4.0,
            num_inference_steps=50,
            height=512,
            width=512,
            seed=None,
            output_file_name=None
    ):
        output_file = self.get_file_name(output_file_name)
        generator = torch.manual_seed(seed) if seed is not None else None
        with torch.inference_mode():
            output = self.pipeline(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
            )
        output_image = output.images[0]
        output_image.save(str(output_file))
        logger.info("Image saved to %s", output_file)
        return output_file
