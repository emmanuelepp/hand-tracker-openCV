"""Hand segmentation: background difference, thresholding and largest-contour selection."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


@dataclass
class Segment:
    mask: np.ndarray
    contour: np.ndarray


def preprocess(image: np.ndarray, blur_kernel: int = 7) -> np.ndarray:
    """Convert a BGR image to blurred grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur_kernel > 1:
        kernel = blur_kernel if blur_kernel % 2 else blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return gray


def segment(
    gray: np.ndarray,
    background: np.ndarray,
    threshold: int = 25,
    min_area: float = 2000,
) -> Segment | None:
    """Return the mask and contour of the largest foreground region, if it is big enough."""
    diff = cv2.absdiff(background, gray)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None

    hand_mask = np.zeros_like(mask)
    cv2.drawContours(hand_mask, [contour], -1, 255, cv2.FILLED)
    return Segment(hand_mask, contour)
