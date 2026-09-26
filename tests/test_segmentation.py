import cv2
import numpy as np

from hand_tracker.segmentation import preprocess, segment

from synthetic import hand_mask

BACKGROUND_LEVEL = 60
HAND_LEVEL = 180


def scene(mask: np.ndarray) -> np.ndarray:
    gray = np.full(mask.shape, BACKGROUND_LEVEL, dtype=np.uint8)
    gray[mask > 0] = HAND_LEVEL
    return gray


def test_segments_the_hand():
    mask = hand_mask(3)
    hand = segment(scene(mask), scene(np.zeros_like(mask)))
    assert hand is not None
    overlap = np.count_nonzero(hand.mask & mask) / np.count_nonzero(mask)
    assert overlap > 0.95


def test_returns_none_without_changes():
    background = np.full((200, 200), BACKGROUND_LEVEL, dtype=np.uint8)
    assert segment(background.copy(), background) is None


def test_ignores_small_noise():
    background = np.full((200, 200), BACKGROUND_LEVEL, dtype=np.uint8)
    frame = background.copy()
    cv2.circle(frame, (100, 100), 10, HAND_LEVEL, cv2.FILLED)
    assert segment(frame, background, min_area=2000) is None


def test_removes_speckles_and_keeps_largest_region():
    mask = hand_mask(2)
    frame = scene(mask)
    rng = np.random.default_rng(0)
    ys, xs = rng.integers(0, mask.shape[0], 200), rng.integers(0, mask.shape[1], 200)
    frame[ys, xs] = 255
    hand = segment(frame, scene(np.zeros_like(mask)))
    assert hand is not None
    assert np.count_nonzero(hand.mask & ~mask) < 0.02 * np.count_nonzero(mask)


def test_fills_holes_in_the_mask():
    mask = hand_mask(0)
    frame = scene(mask)
    frame[180:200, 140:160] = BACKGROUND_LEVEL
    hand = segment(frame, scene(np.zeros_like(mask)))
    assert hand is not None
    assert hand.mask[190, 150] == 255


def test_preprocess_returns_grayscale():
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    gray = preprocess(image, blur_kernel=6)
    assert gray.shape == (50, 60)
    assert gray.dtype == np.uint8
