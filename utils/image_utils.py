import logging

from utils.image_models.base_image_model import BaseImageModel
from utils.image_models.flux_model import FluxModel
from utils.image_models.qwen_image_edit_model import QwenImageEditModel
from utils.image_models.qwen_image_model import QwenImageModel
from utils.image_models.sdxl_model import SDXLModel
from utils.image_models.stable_difusion_model import StableDiffusionModel

logger = logging.getLogger(__name__)

models = [
    {
        "name": "CompVis/stable-diffusion-v1-4",
        "model_class": StableDiffusionModel,
        "functionality": ["t2i"]
    },
    {
        "name": "Lykon/DreamShaper",
        "model_class": StableDiffusionModel,
        "functionality": ["t2i"]
    },
    {
        "name": "black-forest-labs/FLUX.1-dev",
        "model_class": FluxModel,
        "functionality": ["t2i"]
    },
    {
        "name": "black-forest-labs/FLUX.1-schnell",
        "model_class": FluxModel,
        "functionality": ["t2i"]
    },
    {
        "name": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_class": SDXLModel,
        "functionality": ["t2i"]
    },
    {
        "name": "Qwen/Qwen-Image-Edit",
        "model_class": QwenImageEditModel,
        "functionality": ["i2i"]
    },
    {
        "name": "Qwen/Qwen-Image",
        "model_class": QwenImageModel,
        "functionality": ["t2i"]
    }
]


class ImageUtils:
    model = BaseImageModel()

    def get_model(self, model_id, output_path="output") -> BaseImageModel:
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
            seed=seed,
            output_file_name=output_path
        )
        return str(output_file)

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
            output_path=None
    ):
        output_file = self.model.image_to_image(
            image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            seed=seed,
            output_file_name=output_path
        )
        return str(output_file)
