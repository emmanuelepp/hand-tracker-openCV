import cv2
import numpy as np
import pytest

from hand_tracker import app
from hand_tracker.config import ROI, Config

from synthetic import hand_mask

FRAME_SHAPE = (480, 640, 3)


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def frame_with_hand(roi: ROI, fingers: int | None) -> np.ndarray:
    frame = np.full(FRAME_SHAPE, 60, dtype=np.uint8)
    if fingers is not None:
        mask = hand_mask(fingers)
        region = frame[roi.top : roi.bottom, roi.left : roi.right]
        region[mask > 0] = 180
    return frame


@pytest.fixture
def gui(monkeypatch):
    shown = []
    monkeypatch.setattr(cv2, "imshow", lambda name, image: shown.append((name, image.copy())))
    monkeypatch.setattr(cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    return shown


def test_parse_args_defaults():
    assert app.parse_args([]) == Config()


def test_parse_args_custom_values():
    config = app.parse_args(
        ["--camera", "2", "--method", "defects", "--roi", "0", "200", "10", "210", "--no-mirror"]
    )
    assert config.camera == 2
    assert config.method == "defects"
    assert config.roi == ROI(0, 200, 10, 210)
    assert not config.mirror


@pytest.mark.parametrize(
    "argv",
    [["--roi", "100", "50", "0", "10"], ["--threshold", "300"], ["--calibration-frames", "0"]],
)
def test_parse_args_rejects_invalid_values(argv):
    with pytest.raises(SystemExit):
        app.parse_args(argv)


def test_roi_is_clipped_to_the_frame():
    assert ROI(-10, 900, 500, 900).clip(480, 640) == ROI(0, 480, 500, 640)


@pytest.mark.parametrize("method", ["circle", "defects"])
def test_run_counts_fingers_end_to_end(gui, method):
    roi = ROI(20, 300, 300, 600)
    config = Config(roi=roi, method=method, calibration_frames=5, mirror=False)
    frames = [frame_with_hand(roi, None)] * 5 + [
        frame_with_hand(roi, 3),
        frame_with_hand(roi, None),
    ]

    app.run(FakeCapture(frames), config)

    masks = [image for name, image in gui if name == app.MASK_WINDOW]
    assert len(masks) == 7
    assert np.count_nonzero(masks[5]) > 0
    assert np.count_nonzero(masks[6]) == 0


def test_main_releases_camera_on_error(monkeypatch, gui):
    capture = FakeCapture([])

    def fail(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(app.cv2, "VideoCapture", lambda index: capture)
    monkeypatch.setattr(app, "run", fail)
    with pytest.raises(RuntimeError):
        app.main([])
    assert capture.released
