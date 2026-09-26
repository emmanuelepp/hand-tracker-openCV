from pathlib import Path

import cv2
import numpy as np
import pytest

from hand_tracker.fingers import COUNTERS, palm

from synthetic import FINGER_ORDER, PALM_CENTER, contour_of, hand_mask

DEMO_DIR = Path(__file__).resolve().parents[1] / "assets" / "demo"
METHODS = sorted(COUNTERS)


def count(method: str, mask: np.ndarray) -> int:
    return COUNTERS[method](mask, contour_of(mask))


def demo_mask(name: str) -> np.ndarray:
    """Extract the binary hand mask from the "Thresholded" window in a demo screenshot."""
    image = cv2.imread(str(DEMO_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    window = image[: image.shape[0] // 2]
    dark = (window < 20).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    x, y, w, h = stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), :4]
    _, mask = cv2.threshold(window[y : y + h + 5, x : x + w], 127, 255, cv2.THRESH_BINARY)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour_of(mask)], -1, 255, cv2.FILLED)
    return filled


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fingers", range(6))
def test_counts_synthetic_hand(method, fingers):
    assert count(method, hand_mask(fingers)) == fingers


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scale", [0.6, 1.5])
@pytest.mark.parametrize("fingers", [0, 2, 5])
def test_is_scale_invariant(method, scale, fingers):
    mask = cv2.resize(hand_mask(fingers), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    assert count(method, mask) == fingers


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("fingers", [["index", "pinky"], ["thumb", "index"], ["thumb"]])
def test_counts_non_adjacent_fingers(method, fingers):
    assert count(method, hand_mask(fingers)) == len(fingers)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("name", "fingers"), [("one-finger", 1), ("two-fingers", 2), ("five-fingers", 5)]
)
def test_counts_demo_screenshots(method, name, fingers):
    assert count(method, demo_mask(name)) == fingers


def test_palm_is_found_in_the_palm():
    (x, y), radius = palm(hand_mask(len(FINGER_ORDER)))
    assert abs(x - PALM_CENTER[0]) < 15
    assert abs(y - PALM_CENTER[1]) < 25
    assert radius > 30


@pytest.mark.parametrize("method", METHODS)
def test_empty_mask_counts_zero(method):
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (40, 40), (42, 42), 255, cv2.FILLED)
    assert count(method, mask) == 0
