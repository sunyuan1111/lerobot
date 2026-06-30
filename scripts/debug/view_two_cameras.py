#!/usr/bin/env python3
"""Preview two OpenCV cameras without connecting to a robot.

Defaults match the user's current SO-101 setup:
  wrist camera: /dev/video2
  front camera: /dev/video4

Controls:
  q or Esc  quit, when OpenCV GUI support is available
  s         save one snapshot from each camera, when OpenCV GUI support is available
  Ctrl-C    quit in Rerun or headless mode
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CameraState:
    name: str
    device: str
    cap: cv2.VideoCapture
    prev_gray: np.ndarray | None = None
    last_t: float = 0.0
    fps: float = 0.0
    brightness: float = 0.0
    frame_delta: float = 0.0
    frame: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrist", default="/dev/video2", help="Eye-in-hand camera device.")
    parser.add_argument("--front", default="/dev/video4", help="Eye-to-hand camera device.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--fourcc",
        default="MJPG",
        help="Requested camera pixel format. MJPG usually uses much less USB bandwidth than YUYV.",
    )
    parser.add_argument(
        "--display",
        choices=["auto", "window", "rerun", "headless"],
        default="auto",
        help="Use an OpenCV window, a Rerun window, save headless previews, or auto-detect.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Stop after this many frames. Use 0 to run until q/Esc/Ctrl-C.",
    )
    parser.add_argument(
        "--print-every-s",
        type=float,
        default=1.0,
        help="How often to print camera stats.",
    )
    parser.add_argument(
        "--save-every-s",
        type=float,
        default=2.0,
        help="How often to refresh latest.png in headless mode.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("outputs/camera_debug"),
        help="Directory for snapshots saved with the 's' key.",
    )
    return parser.parse_args()


def open_camera(
    name: str,
    device: str,
    width: int,
    height: int,
    fps: int,
    fourcc: str | None,
) -> CameraState:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {name} camera at {device}")

    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc_text = "".join(chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4))
    print(
        f"{name}: {device}, "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
        f"{cap.get(cv2.CAP_PROP_FPS):.1f} fps, fourcc={actual_fourcc_text!r}"
    )

    return CameraState(name=name, device=device, cap=cap)


def opencv_window_available() -> bool:
    """Return whether this OpenCV build can create GUI windows."""
    window_name = "__lerobot_camera_debug_probe__"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(window_name)
    except cv2.error:
        return False
    return True


def rerun_available() -> bool:
    try:
        import rerun  # noqa: F401
    except Exception:
        return False
    return True


def init_rerun_viewer():
    import os

    import rerun as rr

    os.environ.setdefault("RERUN_FLUSH_NUM_BYTES", "8000")
    rr.init("so101_camera_debug")
    rr.spawn(memory_limit=os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%"))
    return rr


def read_camera(state: CameraState) -> bool:
    ok, frame = state.cap.read()
    if not ok or frame is None:
        return False

    now = time.perf_counter()
    if state.last_t:
        dt = now - state.last_t
        if dt > 0:
            current_fps = 1.0 / dt
            state.fps = current_fps if state.fps == 0 else 0.9 * state.fps + 0.1 * current_fps
    state.last_t = now

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    state.brightness = float(gray.mean())
    if state.prev_gray is not None:
        state.frame_delta = float(cv2.absdiff(gray, state.prev_gray).mean())
    state.prev_gray = gray

    label = (
        f"{state.name} {state.device} | fps {state.fps:4.1f} | "
        f"brightness {state.brightness:5.1f} | delta {state.frame_delta:5.2f}"
    )
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    state.frame = frame
    return True


def save_snapshots(states: list[CameraState], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for state in states:
        if state.frame is None:
            continue
        path = save_dir / f"{stamp}_{state.name}.png"
        cv2.imwrite(str(path), state.frame)
        print(f"Saved {path}")


def save_preview(preview: np.ndarray, save_dir: Path, filename: str = "latest.png") -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    cv2.imwrite(str(path), preview)
    return path


def make_preview(states: list[CameraState]) -> np.ndarray | None:
    frames = [state.frame for state in states if state.frame is not None]
    if not frames:
        return None

    min_height = min(frame.shape[0] for frame in frames)
    resized = []
    for frame in frames:
        if frame.shape[0] != min_height:
            scale = min_height / frame.shape[0]
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), min_height))
        resized.append(frame)
    return cv2.hconcat(resized)


def print_stats(states: list[CameraState], frame_idx: int) -> None:
    parts = []
    for state in states:
        parts.append(
            f"{state.name}: fps={state.fps:4.1f}, brightness={state.brightness:5.1f}, "
            f"delta={state.frame_delta:5.2f}"
        )
    print(f"frame={frame_idx} | " + " | ".join(parts))


def log_rerun_frame(rr, states: list[CameraState], frame_idx: int) -> None:
    rr.set_time("frame", sequence=frame_idx)
    for state in states:
        if state.frame is None:
            continue

        rgb = cv2.cvtColor(state.frame, cv2.COLOR_BGR2RGB)
        rr.log(f"cameras/{state.name}", rr.Image(rgb))
        rr.log(f"stats/{state.name}/fps", rr.Scalars(state.fps))
        rr.log(f"stats/{state.name}/brightness", rr.Scalars(state.brightness))
        rr.log(f"stats/{state.name}/frame_delta", rr.Scalars(state.frame_delta))


def choose_display_mode(requested: str) -> str:
    if requested == "auto":
        if opencv_window_available():
            return "window"
        if rerun_available():
            return "rerun"
        return "headless"

    if requested == "window" and not opencv_window_available():
        raise RuntimeError(
            "This OpenCV build cannot create GUI windows. Use --display=rerun, "
            "or install a GUI-enabled OpenCV build."
        )

    if requested == "rerun" and not rerun_available():
        raise RuntimeError("Rerun is not installed. Install lerobot with the viz extra, or use --display=window.")

    return requested


def main() -> None:
    args = parse_args()
    states = [
        open_camera("wrist", args.wrist, args.width, args.height, args.fps, args.fourcc),
        open_camera("front", args.front, args.width, args.height, args.fps, args.fourcc),
    ]

    display_mode = choose_display_mode(args.display)
    rr = init_rerun_viewer() if display_mode == "rerun" else None

    if display_mode == "window":
        print("Previewing cameras only. Press 's' to save snapshots, 'q' or Esc to quit.")
    elif display_mode == "rerun":
        print("Previewing cameras in Rerun. Press Ctrl-C in this terminal to quit.")
    else:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        print(
            "No GUI viewer is available; running headless. "
            f"Refreshing {args.save_dir / 'latest.png'} every {args.save_every_s:g}s. "
            "Press Ctrl-C to quit."
        )

    frame_idx = 0
    last_print_t = 0.0
    last_save_t = 0.0

    try:
        while True:
            frame_idx += 1
            for state in states:
                if not read_camera(state):
                    print(f"Warning: failed to read from {state.name} at {state.device}")

            preview = make_preview(states)
            now = time.perf_counter()

            if preview is not None and display_mode == "window":
                cv2.imshow("SO-101 camera debug", preview)
            elif rr is not None:
                log_rerun_frame(rr, states, frame_idx)
            elif preview is not None and now - last_save_t >= args.save_every_s:
                path = save_preview(preview, args.save_dir)
                print(f"Saved preview to {path}")
                last_save_t = now

            if now - last_print_t >= args.print_every_s:
                print_stats(states, frame_idx)
                last_print_t = now

            if display_mode == "window":
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    save_snapshots(states, args.save_dir)

            if args.frames > 0 and frame_idx >= args.frames:
                break
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        for state in states:
            state.cap.release()
        if display_mode == "window":
            cv2.destroyAllWindows()
        elif rr is not None:
            rr.rerun_shutdown()


if __name__ == "__main__":
    main()
