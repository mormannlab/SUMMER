# requires background installatoin of ffmpeg

from pathlib import Path
import tempfile
import shutil
from .wrapper import *


def map_frames(
    mp4_path: str | Path,
    version: str,           # "dvd" or "hd"
    output_dir: str | Path,
    wrapper_fps: int = 25,  # fps assumption baked into the wrapper functions
) -> Path:
    mp4_path = Path(mp4_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    version = version.lower()
    if version == "dvd":
        wrapper = wrapper_dvd
        total_frames = 125743
    elif version == "hd":
        wrapper = wrapper_hd
        total_frames = 125743
    else:
        raise ValueError(f"version must be 'dvd' or 'hd', got {version!r}")

    fps_result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(mp4_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    num, den = fps_result.stdout.strip().split("/")
    source_fps = int(num) / int(den)
    fps_fraction = f"{num}/{den}"   

    print(f"Source FPS: {source_fps:.4f} (wrapper assumes {wrapper_fps} fps)")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        frames_dir = tmp / "frames"
        frames_dir.mkdir()
        remapped_dir = tmp / "remapped"
        remapped_dir.mkdir()

        # extract frames from source video file
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(mp4_path),
                "-vsync", "0",
                str(frames_dir / "frame_%08d.png"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        source_frames = sorted(frames_dir.glob("frame_*.png"))
        n_frames_actual = len(source_frames)
        print(f"Extracted {n_frames_actual} frames from source video.")

        # apply the appropriate wrapper function to each source frame index to get the remapped index
        mapping: dict[int, Path] = {}
        skipped = 0

        for src_idx, src_path in enumerate(source_frames):
            # What timestamp does this source frame represent?
            timestamp = src_idx / source_fps

            # Which wrapper frame index corresponds to this timestamp?
            wrapper_idx = round(timestamp * wrapper_fps)

            if wrapper_idx > total_frames:
                skipped += 1
                continue

            new_idx = wrapper(wrapper_idx)

            if new_idx == -1:           # HD: frame absent in this version
                skipped += 1
                continue

            # If multiple source frames round to the same wrapper index,
            # keep the closest one (smallest rounding error)
            if new_idx not in mapping:
                mapping[new_idx] = src_path
            else:
                # Compare which source frame is closer to the wrapper timestamp
                existing_src_idx = source_frames.index(mapping[new_idx])
                existing_error = abs(existing_src_idx / source_fps - wrapper_idx / wrapper_fps)
                current_error  = abs(src_idx          / source_fps - wrapper_idx / wrapper_fps)
                if current_error < existing_error:
                    mapping[new_idx] = src_path

        print(f"Mapped {len(mapping)} frames ({skipped} skipped/dropped).")

        if not mapping:
            raise RuntimeError("No frames remained after remapping.")

        # write remapped frames in order with sequential numbering
        for out_seq, new_idx in enumerate(sorted(mapping.keys()), start=1):
            dst = remapped_dir / f"frame_{out_seq:08d}.png"
            shutil.copy2(mapping[new_idx], dst)

        print(f"Remapped frames written to output directory: {remapped_dir}")