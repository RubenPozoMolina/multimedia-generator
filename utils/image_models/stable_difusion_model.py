import torch
from diffusers import StableDiffusionPipeline

from utils.image_models.base_image_model import BaseImageModel


class StableDiffusionModel(BaseImageModel):

    def load_model(self, model_id: str):
        self.model_id = model_id
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16
        )
        self.pipeline.to(self.device)