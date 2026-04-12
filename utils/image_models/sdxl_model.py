import torch
from diffusers import StableDiffusionXLPipeline

from utils.image_models.base_image_model import BaseImageModel


class SDXLModel(BaseImageModel):

    def load_model(self, model_id: str):
        self.model_id = model_id
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipeline.to(self.device)
