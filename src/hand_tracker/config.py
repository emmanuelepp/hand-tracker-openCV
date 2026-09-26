"""Tunable parameters: camera index, ROI coordinates, background weight, thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field

METHODS = ("circle", "defects")


@dataclass(frozen=True)
class ROI:
    top: int = 20
    bottom: int = 300
    left: int = 300
    right: int = 600

    def clip(self, height: int, width: int) -> ROI:
        """Return a copy of the ROI limited to a frame of the given size."""
        top = min(max(0, self.top), height)
        left = min(max(0, self.left), width)
        bottom = min(max(top, self.bottom), height)
        right = min(max(left, self.right), width)
        return ROI(top, bottom, left, right)

    @property
    def is_empty(self) -> bool:
        return self.bottom <= self.top or self.right <= self.left


@dataclass(frozen=True)
class Config:
    camera: int = 0
    roi: ROI = field(default_factory=ROI)
    method: str = "circle"
    calibration_frames: int = 60
    accumulated_weight: float = 0.1
    threshold: int = 25
    min_area: int = 2000
    blur_kernel: int = 7
    mirror: bool = True
