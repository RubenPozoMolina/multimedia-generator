import logging

from utils.video_models.base_video_model import BaseVideoModel
from utils.video_models.ltx_video_model import LTXVideoModel
from utils.video_models.wan_model import WanModel

logger = logging.getLogger(__name__)

models = [
    {
        "name": "Lightricks/LTX-Video",
        "model_class": LTXVideoModel
    },
    {
        "name": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "model_class": WanModel
    }
]


class VideoUtils:
    model = BaseVideoModel()

    def get_model(self, model_id, output_path="output") -> BaseVideoModel:
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
            logger.error("Error loading model: %s", e)
            raise

    def text_to_video(
            self,
            prompt,
            negative_prompt="",
            height=480,
            width=704,
            num_frames=81,
            guidance_scale=7.5,
            num_inference_steps=50,
            seed=None,
            output_path=None
    ):
        output_file = self.model.text_to_video(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_file_name=output_path
        )
        return str(output_file)

    def image_to_video(
            self,
            image,
            prompt,
            negative_prompt="",
            height=480,
            width=704,
            num_frames=81,
            guidance_scale=7.5,
            num_inference_steps=50,
            seed=None,
            output_path=None
    ):
        output_file = self.model.image_to_video(
            image,
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_file_name=output_path
        )
        return str(output_file)
