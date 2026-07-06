#!/usr/bin/env python3
"""Create a tiny SO-101 LeRobot dataset for FastWAM smoke tests."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torchvision.transforms.functional as TVF

from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.feature_utils import DEFAULT_FEATURES
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_TASK = "Pick up the object and place it in the target area"


def image_hwc(image, height: int | None = None, width: int | None = None):
    if image.ndim == 3 and image.shape[0] == 3:
        if height is not None and width is not None:
            image = TVF.resize(image, [height, width], antialias=True)
        return image.permute(1, 2, 0).contiguous()
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("outputs/50_episodes"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fastwam_so101_smoke"))
    parser.add_argument("--source-repo-id", default="local/50_episodes")
    parser.add_argument("--output-repo-id", default="local/fastwam_so101_smoke")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--all-episodes", action="store_true")
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--crf", type=float, default=30)
    parser.add_argument("--g", type=int, default=2)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.output_root)

    selected_episodes = None if args.all_episodes else args.episodes
    if (args.resize_height is None) != (args.resize_width is None):
        raise ValueError("--resize-height and --resize-width must be passed together.")

    source = LeRobotDataset(
        args.source_repo_id,
        root=args.source_root,
        episodes=selected_episodes,
        video_backend="pyav",
        return_uint8=True,
    )
    features = {key: value for key, value in source.features.items() if key not in DEFAULT_FEATURES}
    if args.resize_height is not None and args.resize_width is not None:
        for key, value in features.items():
            if value["dtype"] == "video":
                value["shape"] = [args.resize_height, args.resize_width, 3]

    target = LeRobotDataset.create(
        repo_id=args.output_repo_id,
        root=args.output_root,
        fps=source.meta.fps,
        robot_type=source.meta.robot_type,
        features=features,
        use_videos=True,
        video_backend="pyav",
        rgb_encoder=RGBEncoderConfig(vcodec=args.vcodec, pix_fmt="yuv420p", crf=args.crf, g=args.g),
        encoder_threads=2,
    )

    current_episode = None
    copied_frames = 0
    copied_episodes = 0
    for item in source:
        episode_index = int(item["episode_index"].item())
        if current_episode is None:
            current_episode = episode_index
        elif episode_index != current_episode:
            target.save_episode(parallel_encoding=True)
            copied_episodes += 1
            current_episode = episode_index

        target.add_frame(
            {
                "action": item["action"],
                "observation.state": item["observation.state"],
                "observation.images.wrist": image_hwc(
                    item["observation.images.wrist"],
                    height=args.resize_height,
                    width=args.resize_width,
                ),
                "observation.images.front": image_hwc(
                    item["observation.images.front"],
                    height=args.resize_height,
                    width=args.resize_width,
                ),
                "task": args.task,
            }
        )
        copied_frames += 1

    if target.has_pending_frames():
        target.save_episode(parallel_encoding=True)
        copied_episodes += 1

    target.finalize()
    print(
        f"created {args.output_root} with {copied_episodes} episode(s), "
        f"{copied_frames} frame(s), fps={source.meta.fps}"
    )


if __name__ == "__main__":
    main()
