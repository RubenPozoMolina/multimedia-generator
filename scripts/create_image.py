import argparse
import sys
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configurar el registro (logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configurar logging de librerías de Hugging Face para que solo muestren errores
try:
    import diffusers.utils.logging as diffusers_logging
    diffusers_logging.set_verbosity_error()
    diffusers_logging.disable_progress_bar()
except ImportError:
    diffusers_logging = None

try:
    import transformers.utils.logging as transformers_logging
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
except ImportError:
    transformers_logging = None

# Añadir el directorio raíz al path para poder importar utils
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from utils.image_utils import ImageUtils

def main():
    parser = argparse.ArgumentParser(description="Generate images using DreamShaper")
    parser.add_argument("--prompt", type=str, required=True, help="Description of the image to generate")
    parser.add_argument("--output", type=str, help="Output path for the image")
    parser.add_argument("--model", type=str, default="Lykon/DreamShaper", help="HuggingFace model ID")

    args = parser.parse_args()

    logger.info(f"Loading model {args.model}...")
    generator = ImageUtils(model_id=args.model)

    logger.info(f"Generating image for prompt: '{args.prompt}'...")
    output_path = generator.text_to_image(args.prompt, args.output)

    logger.info(f"Image saved at: {output_path}")

if __name__ == "__main__":
    main()
