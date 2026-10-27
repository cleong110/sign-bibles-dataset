from pathlib import Path

from moviepy import VideoFileClip


def trim_video(input_path: Path, output_path: Path, duration: int = 60) -> None:
    # Convert Path objects to str to ensure compatibility across all MoviePy versions
    with VideoFileClip(str(input_path)) as video:
        # Note: Use video.subclip(0, duration) if you are on MoviePy v1.x
        trimmed_clip = video.subclipped(0, duration)
        trimmed_clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")


# Define Windows paths using raw strings (r"...") or forward slashes
input_file = Path(r"C:\Users\Colin Leong\projects\SIGN-B~1\SIGN_B~1\SHARD_~1\ASE_CH~3.MP4")

# Automatically generate output path in the same directory (e.g., input_first_60s.mp4)
output_file = input_file.parent / f"{input_file.stem}_first_60s{input_file.suffix}"

# Execute
trim_video(input_file, output_file)
