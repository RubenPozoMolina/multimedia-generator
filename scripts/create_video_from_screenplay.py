import argparse
import json
import logging
import os
import sys

from diffusers.utils import load_image

from utils.image_utils import ImageUtils
from utils.video_utils import VideoUtils


class VideoclipGenerator:

    screenplay = None
    screenplay_json = None

    def __init__(self, screenplay, output_path, logger=None):
        self.screenplay = screenplay
        self.output_path = output_path
        self.logger = logger or logging.getLogger(__name__)
        self.load()

    def load(self):
        self.logger.info("Loading screenplay: %s", self.screenplay)
        with open(self.screenplay, "r") as f:
            self.screenplay_json = json.load(f)


    def process(self):
        return_value = None
        try:
            # General params
            height = self.screenplay_json["height"]
            width = self.screenplay_json["width"]
            common_prompt = self.screenplay_json["common_prompt"]
            negative_prompt = self.screenplay_json["negative_prompt"]
            num_inference_steps = self.screenplay_json["num_inference_steps"]
            fps = self.screenplay_json["fps"]
            counter = 1

            # Generate images
            image_utils = None
            image_model = self.screenplay_json["image_model"]
            for scene in self.screenplay_json["scenes"]:
                self.logger.info("Processing image scene: %s", scene)
                scene_prompt = scene["prompt"]
                prompt = scene_prompt + common_prompt
                output_filename =  self.output_path + "/{:02d}_".format(counter) + scene["name"] + ".png"
                if not os.path.exists(output_filename):
                    if not image_utils:
                        image_utils = ImageUtils(image_model, self.output_path)
                    generated_image = image_utils.text_to_image(
                        prompt,
                        negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        output_path=output_filename
                    )
                    self.logger.info("Generated image: %s", generated_image)
                else:
                    self.logger.info("Found existing output file: %s", output_filename)
                counter += 1

            # Generate videos
            video_model = self.screenplay_json["video_model"]
            video_utils = None
            counter = 1
            videos = []
            for scene in self.screenplay_json["scenes"]:
                scene_prompt = scene["prompt"]
                prompt = scene_prompt + common_prompt
                image_output_filename =  self.output_path + "/{:02d}_".format(counter) + scene["name"] + ".png"
                output_filename =  self.output_path + "/{:02d}_".format(counter) + scene["name"] + ".mp4"
                num_frames = scene["duration"] * fps + 1
                self.logger.info("Processing video scene: %s", scene)
                image = load_image(image_output_filename)
                if not os.path.exists(output_filename):
                    if not video_utils:
                        video_utils = VideoUtils(
                            video_model,
                            self.output_path,
                        )
                    generated_video = video_utils.image_to_video(
                        image,
                        prompt,
                        negative_prompt,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        num_frames=num_frames,
                        output_path=output_filename
                    )
                    self.logger.info("Generated video: %s", generated_video)
                counter += 1
                videos.append(output_filename)

            ## Concat videos
            final_output_filename = self.output_path + "/final_"+ self.screenplay_json["name"] + ".mp4"
            if not video_utils:
                video_utils = VideoUtils(
                    video_model,
                    self.output_path,
                )
            return_value = video_utils.concatenate_videos(videos, final_output_filename)
        except Exception as e:
            self.logger.error("Error processing %s: %s", self.screenplay, e)
        return return_value


def main():
    parser = argparse.ArgumentParser(description="Generate videos from a screenplay JSON file")
    parser.add_argument("--screenplay", type=str, required=True, help="Path to the screenplay JSON file")
    parser.add_argument("--output", type=str, default="output", help="Output directory for generated videos")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    videoclip_generator = VideoclipGenerator(args.screenplay, output_path=args.output, logger=logger)
    final_video = videoclip_generator.process()

    logger.info("Final video: %s", final_video)


if __name__ == "__main__":
    main()
