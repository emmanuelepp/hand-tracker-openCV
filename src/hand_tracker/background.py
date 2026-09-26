"""Background model: running weighted average of the ROI used for subtraction."""

from __future__ import annotations

import cv2
import numpy as np


class BackgroundModel:
    """Averages the first ``frames`` grayscale images it receives."""

    def __init__(self, weight: float = 0.1, frames: int = 60) -> None:
        if not 0 < weight <= 1:
            raise ValueError("weight must be in (0, 1]")
        if frames < 1:
            raise ValueError("frames must be at least 1")
        self.weight = weight
        self.frames = frames
        self.reset()

    def reset(self) -> None:
        self._average: np.ndarray | None = None
        self._seen = 0

    @property
    def ready(self) -> bool:
        return self._seen >= self.frames

    @property
    def progress(self) -> float:
        return min(self._seen / self.frames, 1.0)

    @property
    def image(self) -> np.ndarray:
        if self._average is None:
            raise RuntimeError("background has not been calibrated yet")
        return cv2.convertScaleAbs(self._average)

    def update(self, gray: np.ndarray) -> None:
        if self._average is None or self._average.shape != gray.shape:
            self._average = gray.astype(np.float32)
            self._seen = 0
        else:
            cv2.accumulateWeighted(gray, self._average, self.weight)
        self._seen += 1
