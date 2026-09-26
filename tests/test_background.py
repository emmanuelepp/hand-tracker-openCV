import numpy as np
import pytest

from hand_tracker.background import BackgroundModel


def frame(value: int, shape=(10, 10)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_becomes_ready_after_calibration_frames():
    model = BackgroundModel(weight=0.5, frames=3)
    for i in range(3):
        assert not model.ready
        assert model.progress == pytest.approx(i / 3)
        model.update(frame(100))
    assert model.ready
    assert model.progress == 1


def test_averages_frames():
    model = BackgroundModel(weight=0.5, frames=2)
    model.update(frame(100))
    model.update(frame(200))
    assert model.image.dtype == np.uint8
    assert np.all(model.image == 150)


def test_reset_restarts_calibration():
    model = BackgroundModel(frames=1)
    model.update(frame(10))
    model.reset()
    assert not model.ready
    with pytest.raises(RuntimeError):
        _ = model.image


def test_restarts_when_frame_size_changes():
    model = BackgroundModel(frames=2)
    model.update(frame(10))
    model.update(frame(10, shape=(5, 5)))
    assert not model.ready
    assert model.image.shape == (5, 5)


@pytest.mark.parametrize(("weight", "frames"), [(0, 10), (1.5, 10), (0.1, 0)])
def test_rejects_invalid_parameters(weight, frames):
    with pytest.raises(ValueError):
        BackgroundModel(weight, frames)
