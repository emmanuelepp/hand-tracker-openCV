"""Synthetic hand masks for testing."""

from __future__ import annotations

import math

import cv2
import numpy as np

SIZE = (280, 300)
PALM_CENTER = (150, 190)
PALM_RADIUS = 45

FINGER_ANGLES = {
    "thumb": -70,
    "index": -25,
    "middle": -5,
    "ring": 15,
    "pinky": 35,
}
FINGER_LENGTHS = {"thumb": 65, "index": 85, "middle": 90, "ring": 85, "pinky": 70}
FINGER_WIDTH = 18

FINGER_ORDER = ["index", "middle", "ring", "pinky", "thumb"]


def hand_mask(fingers: int | list[str]) -> np.ndarray:
    """Draw a palm with a forearm entering from the bottom and the given extended fingers."""
    names = FINGER_ORDER[:fingers] if isinstance(fingers, int) else fingers
    mask = np.zeros(SIZE, dtype=np.uint8)
    cx, cy = PALM_CENTER

    cv2.ellipse(mask, PALM_CENTER, (PALM_RADIUS, PALM_RADIUS + 8), 0, 0, 360, 255, cv2.FILLED)
    cv2.rectangle(mask, (cx - 32, cy + 20), (cx + 32, SIZE[0] - 1), 255, cv2.FILLED)

    for name in names:
        angle = math.radians(FINGER_ANGLES[name])
        start_offset = PALM_RADIUS * 0.6
        length = FINGER_LENGTHS[name] + start_offset
        x0 = int(cx + start_offset * math.sin(angle))
        y0 = int(cy - start_offset * math.cos(angle))
        x1 = int(cx + length * math.sin(angle))
        y1 = int(cy - length * math.cos(angle))
        cv2.line(mask, (x0, y0), (x1, y1), 255, FINGER_WIDTH)
    return mask


def contour_of(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea)
