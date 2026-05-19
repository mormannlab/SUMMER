# remap_video_frames.py
#
# Remaps video frames from the DVD version to the version used
# in the experiment using pre-defined frame index mappings (see wrapper.py).
#
# Requires ffmpeg >= 5.0 and Python >= 3.10. No third-party Python packages
# needed. Tested with ffmpeg 7.0.2-static (johnvansickle.com/ffmpeg).
#
# Frames are extracted directly to the output directory, then renamed in-place
# according to the wrapper mapping. Requires ~5 GB of free space in the output
# directory (JPEG frames at q=2). Extraction runs at ~8-10 fps on a modern CPU,
# so expect 30-50 min depending on your machine.
#
# Usage:
#   python remap_video_frames.py <video_path> <output_dir>
#       [--wrapper-fps 25] [--verbose]

import subprocess
from pathlib import Path

from wrapper import wrapper_dvd


def remap_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    wrapper_fps: int = 25,
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

    # ── 1. Extract frames directly to output dir ──────────────────────────
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vsync", "vfr",
        "-q:v", "2",
        "-progress", "pipe:1",
        "-nostats",
        "-v", "error" if not verbose else "info",
        str(output_dir / "frame_%08d.jpg"),
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
                print(f"\r  {line}", end="", flush=True)
        process.wait()
        print()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, ffmpeg_cmd,
                stderr=process.stderr.read()
            )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg frame extraction failed:\n{e.stderr}"
        ) from e

    source_frames = sorted(output_dir.glob("frame_*.jpg"))
    n_frames_actual = len(source_frames)
    print(f"Extracted {n_frames_actual} frames from source video.")

    # ── 2. Rename frames in-place according to wrapper mapping ────────────
    skipped = 0
    for src_idx, src_path in enumerate(source_frames):
        timestamp = src_idx / source_fps
        wrapper_idx = round(timestamp * wrapper_fps)

        if wrapper_idx > total_frames:
            src_path.unlink()
            skipped += 1
            continue

        new_idx = wrapper(wrapper_idx)
        new_path = output_dir / f"frame_{new_idx:08d}.jpg"
        src_path.rename(new_path)

    print(f"Remapped frames saved to: {output_dir} ({skipped} frames skipped/deleted).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Remap video frames to match wrapper indexing.")
    parser.add_argument("video_path", type=Path, help="Path to input video file.")
    parser.add_argument("output_dir", type=Path, help="Directory to save remapped frames.")
    parser.add_argument("--wrapper-fps", type=int, default=25, help="Frame rate the wrapper assumes (default: 25).")
    parser.add_argument("--verbose", action="store_true", help="Print all ffmpeg progress output.")

    args = parser.parse_args()

    remap_video_frames(
        video_path=args.video_path,
        output_dir=args.output_dir,
        wrapper_fps=args.wrapper_fps,
        verbose=args.verbose,
    )