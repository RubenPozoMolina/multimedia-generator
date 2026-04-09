import logging
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)

class ImageUtils:

    def __init__(self, model_id="Lykon/DreamShaper", output_path="output"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_path = Path(output_path)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing StableDiffusionPipeline on {self.device}...")
        try:
            # Intentar cargar con safetensors por defecto (más seguro)
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=True
            )
        except Exception as e:
            logger.warning(f"Failed to load with safetensors: {e}. Falling back to standard weights.")
            # Si falla (por archivos faltantes o corruptos), intentar sin forzar safetensors
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_safetensors=False
            )
        self.pipe = self.pipe.to(self.device)

    def text_to_image(self, prompt, output_path=None):
        if output_path:
            output_file = Path(output_path)
        else:
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            output_file = self.output_path / filename

        image = self.pipe(prompt).images[0]
        image.save(str(output_file))
        return str(output_file)