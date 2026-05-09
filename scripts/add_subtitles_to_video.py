import json
import os
from moviepy import VideoFileClip, TextClip, CompositeVideoClip


def burn_subtitles(video_path, json_path, output_path):
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    # 1. Load resources
    video = VideoFileClip(video_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Extract common styles
    settings = data.get("global_settings", {})
    f_font = settings.get("font", "Arial")
    f_size = settings.get("fontsize", 40)
    f_color = settings.get("color", "white")

    # Convert position list to tuple for MoviePy
    raw_pos = settings.get("position", "bottom")
    f_pos = tuple(raw_pos) if isinstance(raw_pos, list) else raw_pos

    subtitle_clips = []

    # 3. Create clips
    for entry in data.get("subtitles", []):
        txt_clip = (TextClip(
            text=entry["text"],
            font=f_font,
            font_size=f_size,
            color=f_color,
            method='caption',
            size=(video.width * 0.8, None)
        )
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


if __name__ == "__main__":
    burn_subtitles("plaga_video.mp4", "subtitles.json", "plaga_subtitled.mp4")