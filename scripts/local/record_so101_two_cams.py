#!/usr/bin/env python3
"""SO-101 双摄像头数据采集启动脚本。

平时只需要运行：

    python3 scripts/local/record_so101_two_cams.py

如果要改端口、相机、任务描述、episode 数量，优先改下面“常用参数”区域。
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


# =========================
# 常用参数：主要改这里
# =========================

# 主臂/从臂串口。你的当前连接是：主臂 ttyACM0，从臂 ttyACM1。
TELEOP_PORT = "/dev/ttyACM0"
ROBOT_PORT = "/dev/ttyACM1"

# 校准 ID。必须和 lerobot-calibrate 时使用的 id 保持一致。
TELEOP_ID = "my_leader"
ROBOT_ID = "my_follower"

# 双摄像头。wrist 是 eye-in-hand，front 是 eye-to-hand。
WRIST_CAMERA = "/dev/video2"
FRONT_CAMERA = "/dev/video4"

# 相机参数。你的双摄同时稳定的设置是 MJPG + 60fps。
# 数据集仍然按 DATASET_FPS=30 采样；相机跑 60fps 只是为了满足设备实际输出。
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60
CAMERA_FOURCC = "MJPG"
DATASET_FPS = 30

# 数据集名称和保存路径。重新录一批数据时，建议改 DATASET_NAME，避免目录已存在。
DATASET_NAME = "50_episodes"
DATASET_ROOT = Path("outputs") / DATASET_NAME
DATASET_REPO_ID = f"local/{DATASET_NAME}"

# 任务描述会写入数据集 metadata，后面训练语言条件模型时也会用到。
TASK_DESCRIPTION = "Pick up the object and place it in the target area"

# 先用 5 条测试。正式训练 ACT 时可以改成 50 或更多。
NUM_EPISODES = 50
EPISODE_TIME_S = 20
RESET_TIME_S = 10

# 是否打开 Rerun 可视化窗口。调试相机/动作时建议 True。
DISPLAY_DATA = True

# 本地采集先不要上传 Hugging Face Hub。
PUSH_TO_HUB = False

# 语音提示。Linux 上如果声音环境麻烦，可以保持 False。
PLAY_SOUNDS = False


def str_bool(value: bool) -> str:
    return "true" if value else "false"


def build_camera_config() -> str:
    cameras = {
        "wrist": {
            "type": "opencv",
            "index_or_path": WRIST_CAMERA,
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "fps": CAMERA_FPS,
            "fourcc": CAMERA_FOURCC,
        },
        "front": {
            "type": "opencv",
            "index_or_path": FRONT_CAMERA,
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "fps": CAMERA_FPS,
            "fourcc": CAMERA_FOURCC,
        },
    }
    return json.dumps(cameras)


def build_command() -> list[str]:
    return [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        "--teleop.type=so101_leader",
        f"--teleop.port={TELEOP_PORT}",
        f"--teleop.id={TELEOP_ID}",
        f"--robot.cameras={build_camera_config()}",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={DATASET_ROOT}",
        f"--dataset.push_to_hub={str_bool(PUSH_TO_HUB)}",
        f"--dataset.fps={DATASET_FPS}",
        f"--dataset.single_task={TASK_DESCRIPTION}",
        f"--dataset.num_episodes={NUM_EPISODES}",
        f"--dataset.episode_time_s={EPISODE_TIME_S}",
        f"--dataset.reset_time_s={RESET_TIME_S}",
        f"--display_data={str_bool(DISPLAY_DATA)}",
        f"--play_sounds={str_bool(PLAY_SOUNDS)}",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不真正开始录制。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="继续往已有数据集目录追加 episode。只有确认目录完整时才使用。",
    )
    parser.add_argument(
        "--allow-existing-root",
        action="store_true",
        help="允许输出目录已存在。通常只和 --resume 一起使用。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = build_command()

    if args.resume:
        cmd.append("--resume=true")

    print("即将运行：")
    print(" ".join(shlex.quote(part) for part in cmd))
    print()

    if args.dry_run:
        return 0

    if DATASET_ROOT.exists() and not (args.resume or args.allow_existing_root):
        print(f"输出目录已经存在：{DATASET_ROOT}", file=sys.stderr)
        print("请改 DATASET_NAME，或删除旧目录，或确认要追加时使用 --resume。", file=sys.stderr)
        return 1

    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
