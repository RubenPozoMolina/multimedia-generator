# multimedia-generator

Tool to generate multimedia files using diffusion models.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

System dependencies:
```bash
pip install imagemagick ffmpeg
```

A CUDA-compatible GPU is required for model inference.

## Usage

### Generate Images

Generate an image from a text prompt using `create_image.py`:

```bash
python scripts/create_image.py --prompt "A futuristic city in the style of cyberpunk"
```

Specify an output path and model:

```bash
python scripts/create_image.py --prompt "A futuristic city" --output "output/city.png" --model "black-forest-labs/FLUX.1-dev"
```

#### Arguments

| Argument    | Required | Default            | Description                        |
|-------------|----------|--------------------|------------------------------------|
| `--prompt`  | Yes      | —                  | Description of the image to generate |
| `--output`  | No       | Auto-generated     | Output path for the image          |
| `--model`   | No       | `Lykon/DreamShaper`| HuggingFace model ID               |

#### Supported Image Models

| Model                                      | Functionality |
|--------------------------------------------|---------------|
| `CompVis/stable-diffusion-v1-4`            | text-to-image |
| `Lykon/DreamShaper`                        | text-to-image |
| `black-forest-labs/FLUX.1-dev`             | text-to-image |
| `black-forest-labs/FLUX.1-schnell`         | text-to-image |
| `stabilityai/stable-diffusion-xl-base-1.0` | text-to-image |
| `Qwen/Qwen-Image-Edit`                    | image-to-image |

### Generate Videos

Generate a video from a text prompt using `create_video.py`:

```bash
python scripts/create_video.py --prompt "A white cat sitting on a table"
```

With custom parameters:

```bash
python scripts/create_video.py \
  --prompt "A white cat sitting on a table" \
  --model "Lightricks/LTX-Video" \
  --height 480 \
  --width 704 \
  --num-frames 81 \
  --num-inference-steps 50 \
  --guidance-scale 7.5 \
  --negative-prompt "low quality, blurry" \
  --seed 42 \
  --output "output/cat.mp4"
```

#### Arguments

| Argument                | Required | Default                | Description                          |
|-------------------------|----------|------------------------|--------------------------------------|
| `--prompt`              | Yes      | —                      | Description of the video to generate |
| `--output`              | No       | Auto-generated         | Output path for the video            |
| `--model`               | No       | `Lightricks/LTX-Video` | HuggingFace model ID                 |
| `--height`              | No       | `480`                  | Video height in pixels               |
| `--width`               | No       | `704`                  | Video width in pixels                |
| `--num-frames`          | No       | `81`                   | Number of frames to generate         |
| `--num-inference-steps` | No       | `50`                   | Number of inference steps            |
| `--guidance-scale`      | No       | `7.5`                  | Guidance scale                       |
| `--negative-prompt`     | No       | `""`                   | Negative prompt                      |
| `--seed`                | No       | `None`                 | Random seed for reproducibility      |

#### Supported Video Models

| Model                                | Functionality    |
|--------------------------------------|------------------|
| `Lightricks/LTX-Video`              | text-to-video    |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers`  | image-to-video   |

### Generate Videos from a Screenplay

Generate multiple video scenes from a JSON screenplay file and concatenate them into a final video using `create_video_from_screenplay.py`:

```bash
python scripts/create_video_from_screenplay.py --screenplay examples/geometry/screenplay.json
```

With a custom output directory and model override:

```bash
python scripts/create_video_from_screenplay.py \
  --screenplay examples/geometry/screenplay.json \
  --output "output/geometry" \
  --model "Lightricks/LTX-Video"
```

#### Arguments

| Argument       | Required | Default                | Description                          |
|----------------|----------|------------------------|--------------------------------------|
| `--screenplay` | Yes      | —                      | Path to the screenplay JSON file     |
| `--output`     | No       | `output`               | Output directory for generated videos|
| `--model`      | No       | `Lightricks/LTX-Video` | HuggingFace model ID (overrides screenplay) |

#### Screenplay JSON Format

```json
{
  "name": "MyVideo",
  "height": 360,
  "width": 640,
  "fps": 30,
  "negative_prompt": "blurry, low quality",
  "num_inference_steps": 30,
  "video_model": "Lightricks/LTX-Video",
  "scenes": [
    {
      "name": "Scene1",
      "duration": 5,
      "prompt": "A sphere rotating in space"
    },
    {
      "name": "Scene2",
      "duration": 5,
      "prompt": "A cube floating in the sky",
      "num_inference_steps": 50
    }
  ]
}
```

#### Screenplay Fields

| Field                  | Level  | Required | Default                | Description                              |
|------------------------|--------|----------|------------------------|------------------------------------------|
| `name`                 | Global | Yes      | —                      | Name of the screenplay                   |
| `scenes`               | Global | Yes      | —                      | List of scenes                           |
| `height`               | Global | No       | `480`                  | Video height in pixels                   |
| `width`                | Global | No       | `704`                  | Video width in pixels                    |
| `fps`                  | Global | No       | `24`                   | Frames per second                        |
| `negative_prompt`      | Global | No       | `""`                   | Negative prompt for all scenes           |
| `num_inference_steps`  | Global | No       | `50`                   | Inference steps for all scenes           |
| `video_model`          | Global | No       | `Lightricks/LTX-Video` | Default video model                      |
| `image_model`          | Global | No       | `None`                 | Image model for generating an initial frame per scene (all images are generated before video creation) |
| `common_prompt`        | Global | No       | `""`                   | Common prompt prepended to all scene prompts |
| `audio`                | Global | No       | `None`                 | Audio file path (relative to screenplay) to overlay on the final video |
| `prompt`               | Scene  | Yes      | —                      | Text prompt for the scene                |
| `name`                 | Scene  | No       | `scene_{index}`        | Scene name (used in output filename)     |
| `duration`             | Scene  | No       | `5`                    | Duration in seconds                      |
| `negative_prompt`      | Scene  | No       | Global value           | Override negative prompt for this scene  |
| `height`               | Scene  | No       | Global value           | Override height for this scene           |
| `width`                | Scene  | No       | Global value           | Override width for this scene            |
| `fps`                  | Scene  | No       | Global value           | Override fps for this scene              |
| `num_inference_steps`  | Scene  | No       | Global value           | Override inference steps for this scene  |
| `seed`                 | Scene  | No       | `None`                 | Random seed for reproducibility          |

#### Examples

Generate the "La Plaga" music video (8 post-apocalyptic scenes with audio overlay):

```bash
python scripts/create_video_from_screenplay.py \
  --screenplay examples/largo_viaje/el_sabio/la_plaga/screenplay.json \
  --output output/la_plaga
```
You can see the result here: [La Plaga Music Video](https://youtu.be/ZfqbFIOXRTs?si=CJCsruSkxdxnZbcA)