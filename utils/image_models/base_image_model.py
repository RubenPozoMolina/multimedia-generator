import torch
from pathlib import Path
from datetime import datetime


class BaseImageModel:
    device = None
    model = None
    model_id = None
    pipeline = None
    output_path = "output"

    def __init__(self, model_id=None, output_path="output"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.output_path = output_path

    def load_model(self, model_id):
        raise NotImplementedError("load_model method must be implemented in the child class.")

    def get_file_name(self, output_file_name):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return_value = Path(self.output_path) / f"{timestamp}.png"
        if output_file_name:
            return_value = output_file_name
        return return_value

    def text_to_image(
            self,
            prompt,
            negative_prompt="",
            height=512,
            width=512,
            guidance_scale=7.5,
            num_inference_steps=50,
            seed=None,
            output_file_name=None
    ):
        output_file = self.get_file_name(output_file_name)
        if "FLUX" in self.model_id:
            image = self.pipeline(
                prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=seed
            ).images[0]
        else:
            image = self.pipeline(
                prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed
            ).images[0]
        image.save(str(output_file))
        return output_file

    def image_to_image(self, image, output_file_name=None):
        output_file = self.get_file_name(output_file_name)
        image = self.pipeline(image).images[0]
        image.save(str(output_file))
        return output_file
