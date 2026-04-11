import torch
from pathlib import Path
from datetime import datetime


class BaseVideoModel:
    device = None
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
        return_value = Path(self.output_path) / f"{timestamp}.mp4"
        if output_file_name:
            return_value = output_file_name
        return return_value

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
            output_file_name=None
    ):
        raise NotImplementedError("text_to_video method must be implemented in the child class.")

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
            output_file_name=None
    ):
        raise NotImplementedError("image_to_video method must be implemented in the child class.")
