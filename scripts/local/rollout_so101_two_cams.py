#!/usr/bin/env python3
"""SO-101 双摄像头策略推理/真机测试启动脚本。

平时只需要运行：

    python3 scripts/local/rollout_so101_two_cams.py

如果要换模型、端口、相机、运行时间，优先改下面“常用参数”区域。
临时比较 checkpoint 时，也可以用：

    python3 scripts/local/rollout_so101_two_cams.py \
        --policy-path outputs/train/act_50_10k/checkpoints/008000/pretrained_model
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

# 从臂串口。rollout 是模型直接控制从臂，不需要主臂。
ROBOT_PORT = "/dev/ttyACM1"

# 从臂校准 ID。必须和 lerobot-calibrate / 采集数据时一致。
ROBOT_ID = "my_follower"

# 要测试的模型。默认使用刚训练好的 ACT 最终 checkpoint。
# 对比不同 checkpoint 时，可以改成：
#   outputs/train/act_50_10k/checkpoints/008000/pretrained_model
#   outputs/train/diffusion_50_20k/checkpoints/last/pretrained_model
# POLICY_PATH = Path("outputs/train/act_50_10k/checkpoints/008000/pretrained_model")
POLICY_PATH = Path("outputs/train/diffusion_50_20k/checkpoints/last/pretrained_model")

# 双摄像头。名字必须和训练数据里的 feature 名字一致：wrist / front。
WRIST_CAMERA = "/dev/video2"
FRONT_CAMERA = "/dev/video4"

# 相机参数。你的双摄稳定设置是 MJPG + 60fps。
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60
CAMERA_FOURCC = "MJPG"

# 控制循环 FPS。通常和数据集 FPS 保持一致。
ROLLOUT_FPS = 30

# 真机运行时间，单位秒。0 表示无限运行，需要 Ctrl+C 手动停止。
ROLLOUT_DURATION_S = 0

# 任务描述。ACT 基本不依赖语言，但保留一致的 task 方便后续 VLA / diffusion 对比。
TASK_DESCRIPTION = "Pick up the object and place it in the target area"

# 是否打开 Rerun 可视化窗口。调试时建议 True。
DISPLAY_DATA = True

# 停止时是否回到启动位置。第一次测试建议 True；如果不想自动回位可改 False。
RETURN_TO_INITIAL_POSITION = True

# 语音提示。Linux 上声音环境容易麻烦，默认关掉。
PLAY_SOUNDS = False

# 额外 policy 参数。
# 例如 Diffusion 推理太慢时，可以取消下面这一行注释：
# EXTRA_POLICY_ARGS = ["--policy.num_inference_steps=16"]
EXTRA_POLICY_ARGS: list[str] = []

# 额外 rollout 参数。临时加实验参数时放这里。
EXTRA_ROLLOUT_ARGS: list[str] = []


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
    return json.dumps(cameras, separators=(",", ":"))


def build_command(policy_path: str | Path, duration_s: float) -> list[str]:
    return [
        "lerobot-rollout",
        "--strategy.type=base",
        f"--policy.path={policy_path}",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.cameras={build_camera_config()}",
        f"--fps={ROLLOUT_FPS}",
        f"--duration={duration_s}",
        f"--display_data={str_bool(DISPLAY_DATA)}",
        f"--task={TASK_DESCRIPTION}",
        f"--return_to_initial_position={str_bool(RETURN_TO_INITIAL_POSITION)}",
        f"--play_sounds={str_bool(PLAY_SOUNDS)}",
        *EXTRA_POLICY_ARGS,
        *EXTRA_ROLLOUT_ARGS,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不真正启动机械臂。")
    parser.add_argument(
        "--policy-path",
        default=str(POLICY_PATH),
        help="临时指定模型路径，不改代码也能比较不同 checkpoint。",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=ROLLOUT_DURATION_S,
        help="临时指定运行秒数。0 表示无限运行。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy_path = Path(args.policy_path)

    if not policy_path.exists():
        print(f"模型路径不存在：{policy_path}", file=sys.stderr)
        print("请先确认 POLICY_PATH，或用 --policy-path 指向一个 pretrained_model 目录。", file=sys.stderr)
        return 1

    cmd = build_command(policy_path, args.duration)

    print("即将运行：")
    print(" ".join(shlex.quote(part) for part in cmd))
    print()

    if args.dry_run:
        return 0

    print("真机 rollout 即将开始。请确认机械臂周围安全，必要时随时 Ctrl+C 停止。")
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
