"""Qwen-Image-Edit engine used to paint animation frames one by one."""

from __future__ import annotations

import logging
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)


class QwenImageEditEngine:
    """Small wrapper around the Qwen-Image-Edit diffusers pipeline.

    The pipeline is loaded lazily so commands such as ``--dry-run`` and jobs
    that only reuse existing files never touch the GPU.  The wrapper also
    normalises device placement and memory-release behaviour, which are easy
    to get wrong when the pipeline is instantiated ad hoc.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen-Image-Edit",
        device: str | None = None,
        pipeline: Any | None = None,
        max_sequence_length: int = 512,
        logger: logging.Logger | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.pipeline = pipeline
        self.max_sequence_length = max_sequence_length
        self.logger = logger or logging.getLogger(__name__)

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def load_model(self) -> Any:
        """Load the edit pipeline on first use."""
        if self.pipeline is not None:
            return self.pipeline

        import torch
        from diffusers import QwenImageEditPipeline

        device = self._resolve_device()
        torch_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.logger.info("Loading frame image model: %s", self.model_id)
        pipeline = QwenImageEditPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )
        pipeline.to(device)

        # Frame-by-frame generation keeps several full-resolution images in
        # memory; slicing and VAE tiling make long scenes practical.
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, "vae") and hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()

        self.pipeline = pipeline
        return pipeline

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = " ",
        true_cfg_scale: float = 4.0,
        num_inference_steps: int = 40,
        height: int | None = None,
        width: int | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        """Edit one image and return the generated PIL image."""
        import torch

        pipeline = self.load_model()
        if not isinstance(image, Image.Image):
            raise TypeError("QwenImageEditEngine.edit expects a PIL image")
        image = image.convert("RGB")
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._resolve_device()).manual_seed(int(seed))

        with torch.inference_mode():
            output = pipeline(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=generator,
                max_sequence_length=self.max_sequence_length,
            )
        return output.images[0].convert("RGB")

    def destroy(self) -> None:
        """Destroy the pipeline and release GPU memory."""
        import torch

        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.logger.info("Frame image model destroyed.")
