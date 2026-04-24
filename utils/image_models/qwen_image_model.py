import torch
from diffusers import DiffusionPipeline

from utils.image_models.base_image_model import BaseImageModel


class QwenImageModel(BaseImageModel):

    def load_model(self, model_id: str):
        self.model_id = model_id
        torch_type = torch.float32
        if self.device == 'cuda':
            torch_type = torch.bfloat16
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_type
        )
        self.pipeline.to(self.device)