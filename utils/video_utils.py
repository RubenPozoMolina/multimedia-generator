import json
import logging
import os
from pathlib import Path

from PIL import Image
from moviepy import AudioFileClip, VideoFileClip, concatenate_videoclips, CompositeVideoClip, TextClip

from utils.video_models.base_video_model import BaseVideoModel
from utils.video_models.ltx_video_model import LTXVideoModel
# from utils.video_models.ltx2_model import LTX2Model
from utils.video_models.wan_model import WanModel

logger = logging.getLogger(__name__)

models = [
    {
        "name": "Lightricks/LTX-Video",
        "model_class": LTXVideoModel
    },
    # {
    #     "name": "Lightricks/LTX-2",
    #     "model_class": LTX2Model
    # },
    {
        "name": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "model_class": WanModel
    }
]


class VideoUtils:
    model = BaseVideoModel()

    def get_model(self, model_id, output_path="output") -> BaseVideoModel:
        for model_entry in models:
            if model_id == model_entry["name"]:
                self.model = model_entry["model_class"](model_id, output_path=output_path)
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
            fps=24,
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
            fps=fps,
            output_file_name=output_path
        )
        return str(output_file)

    @staticmethod
    def extract_last_frame(video_path):
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        clip = VideoFileClip(str(video_path))
        try:
            last_frame = clip.get_frame(clip.duration - (1.0 / clip.fps))
            return Image.fromarray(last_frame)
        finally:
            clip.close()

    @staticmethod
    def concatenate_videos(video_paths, output_file, fps=None):
        if not video_paths:
            raise ValueError("No video paths provided for concatenation.")

        logger.info("Concatenating %d videos into %s", len(video_paths), output_file)
        for video_path in video_paths:
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
        source_clips = []
        clips = []
        final_clip = None
        try:
            for video_path in video_paths:
                clip = VideoFileClip(str(video_path))
                source_clips.append(clip)
                clip_duration = getattr(clip, "duration", None)
                clip_fps = getattr(clip, "fps", None)
                if isinstance(clip_duration, (int, float)) and isinstance(clip_fps, (int, float)) and clip_duration > 0 and clip_fps > 0:
                    frame_duration = 1.0 / float(clip_fps)
                    safe_end = float(clip_duration) - (frame_duration * 0.5)
                    if safe_end > 0:
                        subclipped_method = getattr(clip, "subclipped", None)
                        if callable(subclipped_method):
                            safe_clip = subclipped_method(0, safe_end)
                        else:
                            subclip_method = getattr(clip, "subclip", None)
                            if callable(subclip_method):
                                safe_clip = subclip_method(0, safe_end)
                            else:
                                safe_clip = clip
                        clips.append(safe_clip)
                        continue
                clips.append(clip)

            final_clip = concatenate_videoclips(clips, method="compose")

            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            final_fps = fps if fps is not None else clips[0].fps
            final_clip.write_videofile(str(output_path), fps=final_fps, logger=None)
            logger.info("Concatenated video saved to %s", output_path)
            return str(output_path)
        finally:
            if final_clip is not None:
                final_clip.close()
            for clip in source_clips:
                clip.close()

    @staticmethod
    def add_audio_to_video(video_path, audio_path, output_path):
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        video_path = str(video_path)
        audio_path = str(audio_path)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Adding audio '%s' to video '%s'", audio_path, video_path)
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        try:
            video_with_audio = video_clip.with_audio(audio_clip)
            video_with_audio.write_videofile(str(output_file), fps=video_clip.fps, logger=None)
            logger.info("Video with audio saved to %s", output_file)
            return str(output_file)
        finally:
            audio_clip.close()
            video_clip.close()

    @staticmethod
    def sync_subtitles(audio_path, subtitles_path, output_path=None):
        """
        Automatically synchronize subtitles with an audio file using Whisper.
        """
        try:
            from scripts.sync_subtitles import align_subtitles
            if output_path is None:
                output_path = subtitles_path
            align_subtitles(audio_path, subtitles_path, output_path)
            return output_path
        except ImportError:
            logger.error("Could not import align_subtitles from scripts.sync_subtitles. Ensure dependencies are installed.")
            return None
        except Exception as e:
            logger.error("Error during subtitle synchronization: %s", e)
            return None

    @staticmethod
    def add_subtitles_to_video(video_path, subtitles_path, output_path):
        if not os.path.exists(video_path):
            logger.error("Error: %s not found.", video_path)
            return

        # 1. Load resources
        video = VideoFileClip(video_path)
        with open(subtitles_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. Extract common styles
        settings = data.get("global_settings") or data.get("font_settings", {})
        f_font = settings.get("font", "Arial")
        f_size = settings.get("fontsize", 40)
        f_color = settings.get("color", "white")

        # Convert position list to tuple for MoviePy
        raw_pos = settings.get("position", "bottom")
        if isinstance(raw_pos, list):
            f_pos = tuple(raw_pos)
            # Basic validation to keep subtitles within video height
            video_height = video.size[1]
            if len(f_pos) == 2 and isinstance(f_pos[1], (int, float)):
                if f_pos[1] >= video_height:
                    logger.warning("Subtitle position %s exceeds video height %s. Adjusting.", f_pos, video_height)
                    f_pos = (f_pos[0], int(video_height * 0.9))
        else:
            f_pos = raw_pos

        subtitle_clips = []
        video_width = video.size[0]

        def _get_font_path(font_name):
            if not font_name:
                return None
            if os.path.exists(font_name):
                return font_name

            # Try to find the font using fc-list
            import subprocess
            try:
                # 1. Broad search and manual filter
                result = subprocess.run(['fc-list'], capture_output=True, text=True)
                # Normalize: remove hyphens, spaces, style=
                search_term = font_name.lower().replace('-', '').replace(' ', '').replace('style', '')
                
                best_match = None
                for line in result.stdout.splitlines():
                    # fc-list output format: /path/to/font.ttf: Family Name:style=Style
                    parts = line.split(':')
                    if len(parts) >= 2:
                        path = parts[0].strip()
                        # Family and Style info
                        info = "".join(parts[1:]).lower().replace('-', '').replace(' ', '').replace('=', '').replace('style', '')
                        
                        # Exact match of normalized names
                        if search_term == info:
                            return path
                        
                        # Partial match as fallback
                        if search_term in info and not best_match:
                            best_match = path
                
                if best_match:
                    return best_match

            except Exception as e:
                logger.debug("Failed to resolve font path via fc-list: %s", e)

            return font_name

        def _create_text_clip(text):
            resolved_font = _get_font_path(f_font)
            base_kwargs = {
                "text": text,
                "font_size": f_size,
                "color": f_color,
                "method": "caption",
                "size": (int(video_width * 0.8), None)
            }

            if not resolved_font:
                return TextClip(**base_kwargs)

            try:
                return TextClip(font=resolved_font, **base_kwargs)
            except (OSError, ValueError) as error:
                logger.warning(
                    "Invalid font '%s' (resolved to '%s'). Falling back to default font. Details: %s",
                    f_font,
                    resolved_font,
                    error
                )
                return TextClip(**base_kwargs)

        # 3. Create clips
        for entry in data.get("subtitles", []):
            txt_clip = (_create_text_clip(entry["text"])
                        .with_start(entry["start"])
                        .with_end(entry["end"])
                        .with_position(f_pos))

            subtitle_clips.append(txt_clip)

        # 4. Final Composition
        final_video = CompositeVideoClip([video] + subtitle_clips)

        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=video.fps
        )