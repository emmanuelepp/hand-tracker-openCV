"""Webcam loop: capture, calibrate, segment, count and draw the results (CLI entry point)."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

import cv2
import numpy as np

from hand_tracker import __version__
from hand_tracker.background import BackgroundModel
from hand_tracker.config import METHODS, ROI, Config
from hand_tracker.fingers import COUNTERS
from hand_tracker.segmentation import Segment, preprocess, segment

MAIN_WINDOW = "Finger Count"
MASK_WINDOW = "Thresholded"

KEY_ESC = 27
QUIT_KEYS = {KEY_ESC, ord("q")}
RESET_KEY = ord("r")

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args(argv: Sequence[str] | None = None) -> Config:
    defaults = Config()
    default_roi = defaults.roi
    parser = argparse.ArgumentParser(
        prog="hand-tracker",
        description="Real-time hand tracking and finger counting from a webcam.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--camera", type=int, default=defaults.camera, help="camera index (default: %(default)s)"
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        default=defaults.method,
        help="finger counting method (default: %(default)s)",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
        default=None,
        help="region of interest in pixels (default: "
        f"{default_roi.top} {default_roi.bottom} {default_roi.left} {default_roi.right})",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=defaults.threshold,
        help="minimum gray-level difference from the background (default: %(default)s)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=defaults.min_area,
        help="minimum hand area in pixels (default: %(default)s)",
    )
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=defaults.calibration_frames,
        help="frames used to learn the background (default: %(default)s)",
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="do not flip the camera image horizontally"
    )
    args = parser.parse_args(argv)

    if args.calibration_frames < 1:
        parser.error("--calibration-frames must be at least 1")
    if not 0 <= args.threshold <= 255:
        parser.error("--threshold must be between 0 and 255")
    roi = ROI(*args.roi) if args.roi else defaults.roi
    if roi.is_empty:
        parser.error("--roi must satisfy TOP < BOTTOM and LEFT < RIGHT")

    return Config(
        camera=args.camera,
        roi=roi,
        method=args.method,
        calibration_frames=args.calibration_frames,
        threshold=args.threshold,
        min_area=args.min_area,
        mirror=not args.no_mirror,
    )


def draw_overlay(
    frame: np.ndarray,
    roi: ROI,
    message: str,
    color: tuple[int, int, int],
    fps: float,
    hand: Segment | None = None,
) -> None:
    if hand is not None:
        cv2.drawContours(frame, [hand.contour], -1, BLUE, 2, offset=(roi.left, roi.top))
    cv2.rectangle(frame, (roi.left, roi.top), (roi.right, roi.bottom), RED, 3)
    cv2.putText(frame, message, (10, 30), FONT, 0.8, color, 2)
    cv2.putText(frame, f"{fps:.0f} FPS", (10, 60), FONT, 0.6, YELLOW, 2)
    cv2.putText(
        frame, "r: recalibrate  q/Esc: quit", (10, frame.shape[0] - 10), FONT, 0.5, YELLOW, 1
    )


def run(capture: cv2.VideoCapture, config: Config) -> None:
    background = BackgroundModel(config.accumulated_weight, config.calibration_frames)
    count_fingers = COUNTERS[config.method]
    roi: ROI | None = None
    fps = 0.0
    last = time.perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            print("Failed to read a frame from the camera.", file=sys.stderr)
            break

        if config.mirror:
            frame = cv2.flip(frame, 1)
        if roi is None:
            roi = config.roi.clip(*frame.shape[:2])
            if roi.is_empty:
                print("The ROI is outside the camera frame.", file=sys.stderr)
                break

        gray = preprocess(frame[roi.top : roi.bottom, roi.left : roi.right], config.blur_kernel)
        hand = None

        if not background.ready:
            background.update(gray)
            message = f"Calibrating {background.progress:.0%} - keep the box empty"
            color = YELLOW
            mask = np.zeros_like(gray)
        else:
            hand = segment(gray, background.image, config.threshold, config.min_area)
            if hand is None:
                message, color = "No hand detected", RED
                mask = np.zeros_like(gray)
            else:
                fingers = count_fingers(hand.mask, hand.contour)
                message = f"{fingers} finger{'s' if fingers != 1 else ''}"
                color = GREEN
                mask = hand.mask

        now = time.perf_counter()
        elapsed, last = now - last, now
        if elapsed > 0:
            fps = 0.9 * fps + 0.1 / elapsed if fps else 1 / elapsed

        draw_overlay(frame, roi, message, color, fps, hand)
        cv2.imshow(MAIN_WINDOW, frame)
        cv2.imshow(MASK_WINDOW, mask)

        key = cv2.waitKey(1) & 0xFF
        if key in QUIT_KEYS:
            break
        if key == RESET_KEY:
            background.reset()


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    capture = cv2.VideoCapture(config.camera)
    if not capture.isOpened():
        print(f"Could not open camera {config.camera}.", file=sys.stderr)
        return 1
    try:
        run(capture, config)
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0
