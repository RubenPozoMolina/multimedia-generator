import logging

from utils.image_models.base_image_model import BaseImageModel
from utils.image_models.flux_model import FluxModel
from utils.image_models.sdxl_model import SDXLModel
from utils.image_models.stable_difusion_model import StableDiffusionModel

logger = logging.getLogger(__name__)

models = [
    {
        "name": "CompVis/stable-diffusion-v1-4",
        "model_class": StableDiffusionModel
    },
    {
        "name": "Lykon/DreamShaper",
        "model_class": StableDiffusionModel
    },
    {
        "name": "black-forest-labs/FLUX.1-dev",
        "model_class": FluxModel
    },
    {
        "name": "black-forest-labs/FLUX.1-schnell",
        "model_class": FluxModel
    },
    {
        "name": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_class": SDXLModel
    }
]


class ImageUtils:
    model = BaseImageModel()

    def get_model(self, model_id, output_path="output") -> BaseImageModel:
        for model in models:
            if model_id == model["name"]:
                self.model = model["model_class"](model_id, output_path=output_path)
                return self.model
        raise ValueError(f"Model {model_id} not found in the list of supported models.")

    def __init__(self, model_id, output_path="output"):
        try:
            self.model = self.get_model(model_id, output_path=output_path)
            self.model.load_model(model_id)
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def text_to_image(
            self,
            prompt,
            negative_prompt="",
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=50,
            seed=None,
            output_path=None
    ):
        output_file = self.model.text_to_image(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_file_name=output_path
        )
        return str(output_file)
