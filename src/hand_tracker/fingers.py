"""Finger counting from the segmented hand contour and its thresholded mask."""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

MAX_FINGERS = 5

RING_RADIUS_RATIO = 1.7
RING_THICKNESS = 10
WRIST_RATIO = 0.5

DEFECT_MAX_ANGLE = np.radians(110)
DEFECT_MIN_DEPTH_RATIO = 0.5
FINGER_REACH_RATIO = 2.0


def palm(mask: np.ndarray) -> tuple[tuple[int, int], float]:
    """Return the palm center and radius as the largest circle inscribed in the mask."""
    distances = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, radius, _, center = cv2.minMaxLoc(distances)
    return center, radius


def count_fingers_circle(mask: np.ndarray, contour: np.ndarray) -> int:
    """Count the finger segments that cross a ring around the palm.

    The ring is centered on the palm and scaled to its size, so it cuts through
    the fingers but not the palm itself. Segments whose lowest point falls in the
    bottom part of the ring are treated as the wrist, and long segments are
    ignored as they are not finger-sized.
    """
    center, palm_radius = palm(mask)
    radius = int(RING_RADIUS_RATIO * palm_radius)
    if radius <= 0:
        return 0

    ring = np.zeros(mask.shape[:2], dtype=np.uint8)
    cv2.circle(ring, center, radius, 255, RING_THICKNESS)
    crossings = cv2.bitwise_and(mask, mask, mask=ring)

    contours, _ = cv2.findContours(crossings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    wrist_limit = center[1] + WRIST_RATIO * radius
    max_points = 2 * np.pi * radius * 0.25

    count = 0
    for crossing in contours:
        _, y, _, h = cv2.boundingRect(crossing)
        if y + h < wrist_limit and len(crossing) < max_points:
            count += 1
    return min(count, MAX_FINGERS)


def count_fingers_defects(mask: np.ndarray, contour: np.ndarray) -> int:
    """Count fingers from the deep, sharp valleys between them (convexity defects).

    Each valley between two extended fingers adds one finger. With no valleys, a
    single finger is recognized when the upper half of the hand reaches well
    beyond the palm.
    """
    center, palm_radius = palm(mask)
    if palm_radius <= 0:
        return 0

    hull = cv2.convexHull(contour, returnPoints=False)
    try:
        defects = cv2.convexityDefects(contour, hull)
    except cv2.error:
        defects = None

    valleys = 0
    if defects is not None:
        points = contour.reshape(-1, 2).astype(np.float64)
        for start_idx, end_idx, far_idx, depth in defects.reshape(-1, 4):
            if depth / 256 < DEFECT_MIN_DEPTH_RATIO * palm_radius:
                continue
            start, end, far = points[start_idx], points[end_idx], points[far_idx]
            if far[1] > center[1] + palm_radius:
                continue
            if _angle(start, far, end) < DEFECT_MAX_ANGLE:
                valleys += 1

    if valleys:
        return min(valleys + 1, MAX_FINGERS)

    points = contour.reshape(-1, 2)
    upper = points[points[:, 1] <= center[1]]
    if not len(upper):
        return 0
    reach = np.linalg.norm(upper - np.array(center), axis=1).max()
    return int(reach > FINGER_REACH_RATIO * palm_radius)


def _angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    u, v = a - vertex, b - vertex
    norms = np.linalg.norm(u) * np.linalg.norm(v)
    if norms == 0:
        return np.pi
    return float(np.arccos(np.clip(np.dot(u, v) / norms, -1.0, 1.0)))


COUNTERS: dict[str, Callable[[np.ndarray, np.ndarray], int]] = {
    "circle": count_fingers_circle,
    "defects": count_fingers_defects,
}
