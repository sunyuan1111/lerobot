#!/usr/bin/env python3
"""Copy a LeRobot dataset and resize its video files in place with PyAV."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import av


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("outputs/50_episodes"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fastwam_so101_50_224"))
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--crf", type=float, default=35)
    parser.add_argument("--g", type=int, default=2)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_without_videos(source_root: Path, output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {"videos", "images"} & set(names)

    shutil.copytree(source_root, output_root, ignore=ignore)
    (output_root / "videos").mkdir(parents=True, exist_ok=True)


def transcode_video(input_path: Path, output_path: Path, width: int, height: int, crf: float, g: int, preset: str) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    with av.open(str(input_path), "r") as in_container:
        in_stream = in_container.streams.video[0]
        rate = in_stream.average_rate or in_stream.base_rate or 30

        with av.open(str(output_path), "w") as out_container:
            out_stream = out_container.add_stream("libx264", rate=rate)
            out_stream.width = width
            out_stream.height = height
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {
                "crf": str(crf),
                "g": str(g),
                "preset": preset,
            }

            for frame in in_container.decode(in_stream):
                resized = frame.reformat(width=width, height=height, format="yuv420p")
                for packet in out_stream.encode(resized):
                    out_container.mux(packet)
                frame_count += 1

            for packet in out_stream.encode():
                out_container.mux(packet)

    return frame_count


def update_info(output_root: Path, width: int, height: int, crf: float, g: int, preset: str) -> None:
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for feature in info["features"].values():
        if feature.get("dtype") != "video":
            continue
        feature["shape"] = [height, width, 3]
        video_info = feature.setdefault("info", {})
        video_info["video.height"] = height
        video_info["video.width"] = width
        video_info["video.codec"] = "h264"
        video_info["video.pix_fmt"] = "yuv420p"
        video_info["video.crf"] = crf
        video_info["video.g"] = g
        video_info["video.preset"] = preset
        video_info["video.video_backend"] = "pyav"
    info_path.write_text(json.dumps(info, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    copy_without_videos(args.source_root, args.output_root, args.overwrite)

    videos_root = args.source_root / "videos"
    input_paths = sorted(videos_root.rglob("*.mp4"))
    if not input_paths:
        raise FileNotFoundError(f"No mp4 files found under {videos_root}")

    total_frames = 0
    for input_path in input_paths:
        rel_path = input_path.relative_to(args.source_root)
        output_path = args.output_root / rel_path
        frames = transcode_video(
            input_path=input_path,
            output_path=output_path,
            width=args.width,
            height=args.height,
            crf=args.crf,
            g=args.g,
            preset=args.preset,
        )
        total_frames += frames
        print(f"{rel_path}: {frames} frames -> {output_path}")

    update_info(args.output_root, args.width, args.height, args.crf, args.g, args.preset)
    print(f"created {args.output_root} with {len(input_paths)} video file(s), {total_frames} decoded frame(s)")


if __name__ == "__main__":
    main()
