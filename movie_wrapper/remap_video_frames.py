# remap_video_frames.py
#
# Remaps video frames from the DVD version to the version used
# in the experiment using pre-defined frame index mappings (see wrapper.py).
#
# Requires ffmpeg >= 5.0 and Python >= 3.10. No third-party Python packages
# needed. Tested with ffmpeg 7.0.2-static (johnvansickle.com/ffmpeg).
#
# The function creates a temporary directory to store intermediate frames,
# saved as JPEGs, then applies the wrapper and copies the remapped frames
# to the output directory.
#
# Temporary storage is roughly 5 GB (JPEG frames at q=2). Run from a disk
# with at least 10 GB free, or use --tmp-dir to point scratch space at a
# larger drive. Extraction runs at ~8-10 fps on a modern CPU, so expect
# 30-50 min depending on your machine.
#
# Usage:
#   python remap_video_frames.py <video_path> <output_dir>
#       [--wrapper-fps 25] [--tmp-dir /path/to/scratch] [--verbose]

import subprocess
import tempfile
import shutil
from pathlib import Path

from wrapper import wrapper_dvd


def remap_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    wrapper_fps: int = 25,
    tmp_dir: str | Path | None = None,  # if None, uses system default (e.g. /tmp on Unix)
    verbose: bool = False,
) -> None:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wrapper = wrapper_dvd
    total_frames = 125743

    # ── Detect frame rate from source video ───────────────────────────────
    try:
        fps_result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed:\n{e.stderr}") from e

    num, den = fps_result.stdout.strip().split("/")
    source_fps = int(num) / int(den)

    print(f"Source FPS: {source_fps:.4f} (wrapper assumes {wrapper_fps} fps)")

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmp_str:
        tmp = Path(tmp_str)
        frames_dir = tmp / "frames"
        frames_dir.mkdir()
        remapped_dir = tmp / "remapped"
        remapped_dir.mkdir()

        # ── 1. Decompose video → individual JPEG frames ───────────────────
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vsync", "vfr",
            "-q:v", "2",
            "-progress", "pipe:1",  # print progress stats to stdout
            "-nostats",             # suppress the default stderr stats line
            "-v", "error" if not verbose else "info",
            str(frames_dir / "frame_%08d.jpg"),
        ]

        try:
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for line in process.stdout:
                line = line.strip()
                if verbose and line:
                    print(f"  [ffmpeg] {line}")
                elif line.startswith("frame="):
                    # always print frame count even without --verbose
                    print(f"\r  {line}", end="", flush=True)
            process.wait()
            print()  # newline after the last \r
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, ffmpeg_cmd,
                    stderr=process.stderr.read()
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"ffmpeg frame extraction failed:\n{e.stderr}"
            ) from e

        source_frames = sorted(frames_dir.glob("frame_*.jpg"))
        n_frames_actual = len(source_frames)
        print(f"Extracted {n_frames_actual} frames from source video.")

        # ── 2. Convert source frame index → wrapper frame index ───────────
        mapping: dict[int, Path] = {}
        skipped = 0

        for src_idx, src_path in enumerate(source_frames):
            timestamp = src_idx / source_fps
            wrapper_idx = round(timestamp * wrapper_fps)

            if wrapper_idx > total_frames:
                skipped += 1
                continue

            new_idx = wrapper(wrapper_idx)

            if new_idx not in mapping:
                mapping[new_idx] = src_path
            else:
                existing_src_idx = source_frames.index(mapping[new_idx])
                existing_error = abs(existing_src_idx / source_fps - wrapper_idx / wrapper_fps)
                current_error  = abs(src_idx          / source_fps - wrapper_idx / wrapper_fps)
                if current_error < existing_error:
                    mapping[new_idx] = src_path

        print(f"Mapped {len(mapping)} frames ({skipped} skipped/dropped).")

        if not mapping:
            raise RuntimeError("No frames remained after remapping.")

        # ── 3. Copy remapped frames to output dir with sequential numbering ─
        for out_seq, new_idx in enumerate(sorted(mapping.keys()), start=1):
            dst = remapped_dir / f"frame_{out_seq:08d}.jpg"
            shutil.copy2(mapping[new_idx], dst)

        print(f"Remapped frames saved to: {remapped_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remap video frames to match wrapper indexing.")
    parser.add_argument("video_path", type=Path, help="Path to input video file.")
    parser.add_argument("output_dir", type=Path, help="Directory to save remapped frames.")
    parser.add_argument("--wrapper-fps", type=int, default=25, help="Frame rate the wrapper assumes (default: 25).")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Directory for temporary files (default: system temp).")
    parser.add_argument("--verbose", action="store_true", help="Print all ffmpeg progress output.")

    args = parser.parse_args()

    remap_video_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        wrapper_fps=args.wrapper_fps,
        tmp_dir=args.tmp_dir,
        verbose=args.verbose,
    )