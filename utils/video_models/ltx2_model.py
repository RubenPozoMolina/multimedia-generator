import logging

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2LatentUpsamplePipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.ltx2.export_utils import encode_video

from utils.video_models.base_video_model import BaseVideoModel

logger = logging.getLogger(__name__)


class LTX2Model(BaseVideoModel):

    STAGE_2_INFERENCE_STEPS = 3
    STAGE_2_GUIDANCE_SCALE = 1.0
    DEFAULT_FRAME_RATE = 24.0
    DISTILLED_LORA_ADAPTER = "stage_2_distilled"
    DISTILLED_LORA_WEIGHT_NAME = "ltx-2-19b-distilled-lora-384.safetensors"

    def __init__(self, model_id=None, output_path="output"):
        super().__init__(model_id, output_path)
        self.upsample_pipeline = None

    def load_model(self, model_id: str):
        self.model_id = model_id
        self.pipeline = LTX2Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
        )
        self.pipeline.enable_sequential_cpu_offload(device=self.device)

        latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            self.model_id,
            subfolder="latent_upsampler",
            torch_dtype=torch.bfloat16,
        )
        self.upsample_pipeline = LTX2LatentUpsamplePipeline(
            vae=self.pipeline.vae,
            latent_upsampler=latent_upsampler,
        )
        self.upsample_pipeline.enable_model_cpu_offload(device=self.device)

        self.pipeline.load_lora_weights(
            self.model_id,
            adapter_name=self.DISTILLED_LORA_ADAPTER,
            weight_name=self.DISTILLED_LORA_WEIGHT_NAME,
        )

        logger.info("LTX-2 model loaded from %s", model_id)

    def _run_two_stage(
            self,
            prompt,
            negative_prompt,
            height,
            width,
            num_frames,
            guidance_scale,
            num_inference_steps,
            generator,
            image=None,
    ):
        stage_1_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": self.DEFAULT_FRAME_RATE,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "output_type": "latent",
            "return_dict": False,
        }
        if image is not None:
            stage_1_kwargs["image"] = image

        self.pipeline.disable_lora()
        video_latent, audio_latent = self.pipeline(**stage_1_kwargs)

        upscaled_video_latent = self.upsample_pipeline(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]

        self.pipeline.set_adapters(self.DISTILLED_LORA_ADAPTER, 1.0)
        self.pipeline.vae.enable_tiling()

        original_scheduler = self.pipeline.scheduler
        self.pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipeline.scheduler.config,
            use_dynamic_shifting=False,
            shift_terminal=None,
        )

        video, audio = self.pipeline(
            latents=upscaled_video_latent,
            audio_latents=audio_latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.STAGE_2_INFERENCE_STEPS,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            guidance_scale=self.STAGE_2_GUIDANCE_SCALE,
            output_type="np",
            return_dict=False,
        )

        self.pipeline.scheduler = original_scheduler
        return video, audio

    def text_to_video(
            self,
            prompt,
            negative_prompt="",
            height=512,
            width=768,
            num_frames=121,
            guidance_scale=4.0,
            num_inference_steps=40,
            seed=None,
            fps=24,
            output_file_name=None
    ):
        height = self.align_dimension(height)
        width = self.align_dimension(width)
        output_file = self.get_file_name(output_file_name)
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None

        video, audio = self._run_two_stage(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        self._save_video_with_audio(video[0], audio[0], output_file, fps=fps)
        return output_file

    def image_to_video(
            self,
            image,
            prompt,
            negative_prompt="",
            height=512,
            width=768,
            num_frames=121,
            guidance_scale=4.0,
            num_inference_steps=40,
            seed=None,
            fps=24,
            output_file_name=None
    ):
        height = self.align_dimension(height)
        width = self.align_dimension(width)
        output_file = self.get_file_name(output_file_name)
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

        video, audio = self._run_two_stage(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            image=image,
        )

        self._save_video_with_audio(video[0], audio[0], output_file, fps=fps)
        logger.info(
            "Video saved to %s", output_file
        )
        return output_file

    def _save_video_with_audio(self, video, audio, output_file, fps=24):
        audio_sample_rate = self.pipeline.vocoder.config.output_sampling_rate
        encode_video(
            video,
            fps=fps,
            audio=audio.float().cpu(),
            audio_sample_rate=audio_sample_rate,
            output_path=str(output_file),
        )
        logger.info("Video with audio saved to %s", output_file)
