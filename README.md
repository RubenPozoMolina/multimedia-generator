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
| `Lightricks/LTX-2`                  | text/image-to-video¹ |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers`  | image-to-video   |

¹ LTX-2 requires a `diffusers` version that includes `diffusers.pipelines.ltx2`;
it does not accept a final-image condition.

### Generate Videos from a Screenplay

Generate multiple video scenes from a JSON screenplay file and concatenate them into a final video using `create_video_from_screenplay.py`:

```bash
python scripts/create_video_from_screenplay.py --screenplay examples/geometry/screenplay.json
```

When the screenplay declares an `audio` file, scene durations are scaled proportionally so the final montage matches the track length. Existing images and video chunks are reused, which makes it safe to resume an interrupted generation.

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

Generate the "La Plaga" music video (52 post-apocalyptic scenes with audio overlay):

```bash
python scripts/create_video_from_screenplay.py \
  --screenplay examples/largo_viaje/el_sabio/la_plaga/screenplay.json \
  --output output/la_plaga
```
You can see the result here: [La Plaga Music Video](https://youtu.be/ZfqbFIOXRTs?si=CJCsruSkxdxnZbcA)

Generate the "Cerveza" animated heavy-metal tavern video (49 animated scenes with audio and subtitles):

```bash
python scripts/create_video_from_screenplay.py \
  --screenplay examples/largo_viaje/el_sabio/cerveza/screenplay.json \
  --output output/cerveza
```

The creative script and shot breakdown are documented in
[examples/largo_viaje/el_sabio/cerveza/guion.md](examples/largo_viaje/el_sabio/cerveza/guion.md).

### Character-Consistent Animation

The classic pipeline generates one video chunk per scene, which can make
characters mutate between shots. This alternative pipeline keeps a persistent
cast by creating the characters **before** animating anything:

1. Every character is designed once and stored as a reference sheet.
2. Each scene keyframe is composed from those references with
   `Qwen/Qwen-Image-Edit`.
3. By default, every subsequent frame is painted with Qwen-Image-Edit, re-anchoring
   periodically to the scene keyframe.
4. For video generation, Qwen-Image-Edit creates an optional final keyframe from
   the start image, then Wan generates motion between those images in one pass.

This is new, additive code. Existing character screenplays continue to use the
frame-by-frame path unless `animation_mode` or `video_model` opts into video
generation. The classic `create_video_from_screenplay.py` pipeline and JSON
format are untouched.

```bash
python scripts/create_video_from_character_screenplay.py \
  --screenplay examples/character_animation/screenplay.json \
  --output output/character_animation
```

Print the cast, scenes and total number of frames without loading a model:

```bash
python scripts/create_video_from_character_screenplay.py \
  --screenplay examples/character_animation/screenplay.json \
  --dry-run
```

Create only the cast first, review the reference sheets, and animate later with
the same command (the references are reused automatically):

```bash
python scripts/create_video_from_character_screenplay.py \
  --screenplay examples/character_animation/screenplay.json \
  --output output/character_animation \
  --characters-only
```

#### Arguments

| Argument            | Required | Default | Description                                              |
|---------------------|----------|---------|----------------------------------------------------------|
| `--screenplay`      | Yes      | —       | Path to the character screenplay JSON file               |
| `--output`          | No       | `output`| Output directory for references, frames and videos       |
| `--character-model` | No       | Screenplay value (`Qwen/Qwen-Image`) | Text-to-image model for character sheets |
| `--frame-model`     | No       | Screenplay value (`Qwen/Qwen-Image-Edit`) | Image-edit model for keyframes and frames |
| `--animation-mode`  | No       | Screenplay value, otherwise `frames` | `frames` or `video`; overrides screenplay |
| `--video-model`     | No       | Screenplay value, otherwise Wan2.2 | Image-to-video model; overrides screenplay |
| `--dry-run`         | No       | `False` | Print the plan without loading any model                 |
| `--characters-only` | No       | `False` | Create the reference sheets and stop before animating     |
| `--log-level`       | No       | `INFO`  | `DEBUG`, `INFO`, `WARNING` or `ERROR`                    |

#### Character Screenplay JSON Format

```json
{
  "name": "The Lantern Keeper",
  "width": 832,
  "height": 480,
  "fps": 24,
  "anchor_interval": 12,
  "animation_mode": "video",
  "video_model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
  "negative_prompt": "blurry, low quality, deformed hands, inconsistent character design",
  "style_prompt": "stylized 3D dark fantasy animation, cinematic volumetric lighting, film grain",
  "character_image_model": "Qwen/Qwen-Image",
  "frame_image_model": "Qwen/Qwen-Image-Edit",
  "audio": "song.mp3",
  "subtitles": "subtitles.json",
  "characters": [
    {
      "name": "Keeper",
      "description": "a skinny bald monk with big innocent eyes, brown hooded robe and rope belt",
      "views": ["front view", "three-quarter view"],
      "seed": 42
    },
    {
      "name": "Bartender",
      "description": "a huge moss-green troll with an underbite, cracked tusks and a braided beard"
    }
  ],
  "scenes": [
    {
      "name": "Tavern reveal",
      "duration": 4,
      "prompt": "the monk discovers a tavern built inside a colossal hollow tree",
      "motion_prompt": "the monk pushes the ferns aside and steps forward",
      "end_prompt": "the monk stands in the clearing and looks up at the tavern",
      "characters": ["Keeper", "Bartender"]
    },
    {
      "name": "Forest landscape",
      "duration": 3,
      "prompt": "a moonlit enchanted forest with fireflies",
      "characters": []
    }
  ]
}
```

#### Character Screenplay Fields

| Field                     | Level     | Required | Default | Description |
|---------------------------|-----------|----------|---------|-------------|
| `name`                    | Global    | Yes      | —       | Project name used in output filenames |
| `characters`              | Global    | Yes      | —       | Non-empty list of character definitions |
| `scenes`                  | Global    | Yes      | —       | Non-empty list of scenes |
| `style_prompt`            | Global    | No       | `""`    | Global style prefix added to every prompt (`common_prompt` is also accepted) |
| `negative_prompt`         | Global    | No       | `""`    | Negative prompt for all generations |
| `width` / `height`        | Global    | No       | `704` / `480` | Video dimensions |
| `fps`                     | Global    | No       | `24`    | Frames per second |
| `num_inference_steps`     | Global    | No       | `30`    | Edit steps per keyframe/frame |
| `true_cfg_scale`          | Global    | No       | `4.0`   | Qwen-Image-Edit classifier-free guidance |
| `anchor_interval`         | Global    | No       | `12`    | Re-anchor to the keyframe every N frames (`0` disables it) |
| `animation_mode`          | Global    | No       | `frames` (or `video` when `video_model` is set) | Select `frames` or image-to-video generation |
| `video_model`             | Global    | No       | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Image-to-video model (`last_image` conditioning is supported by Wan) |
| `video_guidance_scale`    | Global    | No       | `5.0`   | Image-to-video guidance scale |
| `video_num_inference_steps` | Global  | No       | `40`    | Image-to-video inference steps |
| `video_max_frames`        | Global    | No       | `81`    | Maximum video frames per scene; Wan uses frame counts of `4k+1` |
| `character_image_model`   | Global    | No       | `Qwen/Qwen-Image` | Text-to-image model for character sheets |
| `frame_image_model`       | Global    | No       | `Qwen/Qwen-Image-Edit` | Image-edit model for frames |
| `character_num_inference_steps` | Global | No  | `30`    | Steps for character/background generation |
| `character_guidance_scale`| Global    | No       | `7.5`   | Guidance scale for character/background generation |
| `character_width` / `character_height` | Global | No | `768` / `1024` | Portrait resolution for reference sheets; higher detail than the video frame improves identity |
| `audio`                   | Global    | No       | `None`  | Audio file relative to the screenplay; scene durations are scaled to its length |
| `subtitles`               | Global    | No       | `None`  | Subtitles JSON relative to the screenplay |
| `auto_sync_subtitles`     | Global    | No       | `False` | Re-align subtitles with Whisper before rendering them |
| `name`                    | Character | Yes      | —       | Unique character name |
| `description`             | Character | Yes      | —       | Exact visual identity: face, body, hair, clothing, colors |
| `views`                   | Character | No       | `["front view"]` | Reference views generated at the beginning. The first one is the primary identity anchor used for keyframes |
| `reference_prompt`        | Character | No       | `""`    | Overrides the automatic reference-sheet prompt |
| `negative_prompt`         | Character | No       | `""`    | Extra negative prompt for this character |
| `seed`                    | Character | No       | `None`  | Seed for the reference sheets |
| `height` / `width`        | Character | No       | `1024` / `768` | Reference-sheet dimensions (portrait) |
| `name`                    | Scene     | No       | `scene_{index}` | Unique scene name |
| `prompt`                  | Scene     | Yes      | —       | Scene description |
| `duration`                | Scene     | No       | `5`     | Scene duration in seconds |
| `characters`              | Scene     | No       | First character | Cast for this scene. Use `"all"` for the whole cast or `[]` for a characterless background |
| `motion_prompt`           | Scene     | No       | `prompt`| Action to advance from the first to the last frame |
| `keyframe_prompt`         | Scene     | No       | `prompt`| Description used to compose the first frame from the character references |
| `start_frame`             | Scene     | No       | `None`  | Existing image (relative to the screenplay) to use as the first frame instead of generating one |
| `end_prompt`              | Scene     | No       | `None`  | Final pose/action for Qwen to edit from the starting keyframe |
| `end_frame`               | Scene     | No       | `None`  | Existing image (relative to the screenplay) to use as the final video condition |
| `animation_mode`          | Scene     | No       | Global  | Override `frames` / `video` for this scene |
| `video_model`             | Scene     | No       | Global  | Override the image-to-video model; opts this scene into `video` unless the screenplay explicitly selects `frames` |
| `video_guidance_scale` / `video_num_inference_steps` | Scene | No | Global | Per-scene image-to-video settings |
| `video_max_frames`        | Scene     | No       | Global  | Per-scene maximum video frame count |
| `anchor_interval`         | Scene     | No       | Global  | Per-scene re-anchoring interval |
| `num_inference_steps`     | Scene     | No       | Global  | Per-scene edit steps |
| `true_cfg_scale`          | Scene     | No       | Global  | Per-scene edit guidance |
| `seed`                    | Scene     | No       | `None`  | Seed for keyframes and frames |
| `height` / `width` / `fps`| Scene     | No       | Global  | Per-scene overrides |

In `video` mode, `end_prompt` and `end_frame` are optional; without either,
the scene is generated as ordinary image-to-video from its starting keyframe.
The legacy `frames` mode remains the default for older screenplays. Wan's
`last_image` endpoint is used when an end image is available; LTX-Video rejects
that conditioning explicitly.

#### Output Layout

```text
output/character_animation/
├── characters/
│   ├── keeper_front_view.png
│   ├── keeper_three_quarter_view.png
│   └── bartender_front_view.png
├── scenes/
│   ├── 00_tavern_reveal/
│   │   ├── cast/tavern_reveal_<hash>_cast.png   # only for multi-character scenes
│   │   ├── keyframe.png
│   │   ├── end_keyframe.png                      # when video mode has an endpoint
│   │   ├── frames/000000.png …
│   │   ├── tavern_reveal.mp4                     # legacy frame mode
│   │   └── tavern_reveal_video.mp4               # keyframe video mode
│   └── 01_forest_landscape/
│       ├── keyframe.png
│       ├── frames/000000.png …
│       └── forest_landscape.mp4
└── The_Lantern_Keeper_{frames|video}_final.mp4
```

Every artifact is cached on disk. Interrupted runs resume from the last
completed reference, keyframe, frame, scene video and montage.

A migrated version of the "Cerveza" example is included at
[examples/largo_viaje/el_sabio/cerveza/screenplay_characters.json](examples/largo_viaje/el_sabio/cerveza/screenplay_characters.json).
It declares the monk, the bartender and the band as persistent characters and
assigns a cast to each of the 49 shots. It uses 12 fps to keep the frame count
practical; raise it to 24 fps for smoother motion at double the cost.

```bash
python scripts/create_video_from_character_screenplay.py \
  --screenplay examples/largo_viaje/el_sabio/cerveza/screenplay_characters.json \
  --output output/cerveza_frames
```

> **Cost warning:** a 4-second scene at 24 fps requires 96 Qwen-Image-Edit
> calls. Use `--dry-run` to review the total before starting a long job. The
> `anchor_interval` controls how often a frame is regenerated from the scene
> keyframe to stop identity drift; it does not skip frames.
