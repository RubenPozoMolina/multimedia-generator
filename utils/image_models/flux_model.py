import torch
from diffusers import FluxPipeline

from utils.image_models.base_image_model import BaseImageModel


class FluxModel(BaseImageModel):

    def load_model(self, model_id: str):
        self.model_id = model_id
        self.pipeline = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        self.pipeline.to(self.device)

