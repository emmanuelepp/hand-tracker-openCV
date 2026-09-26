# Hand Tracker OpenCV

[![CI](https://github.com/emmanuelepp/hand-tracker-openCV/actions/workflows/ci.yml/badge.svg)](https://github.com/emmanuelepp/hand-tracker-openCV/actions/workflows/ci.yml)

Real-time hand tracking and finger counting from a webcam, built with Python and OpenCV using classic computer-vision techniques (background subtraction, contours, distance transform and convexity defects). No deep learning models required.

<p align="center">
  <img src="assets/demo/one-finger.png" alt="One finger detected" width="30%">
  <img src="assets/demo/two-fingers.png" alt="Two fingers detected" width="30%">
  <img src="assets/demo/five-fingers.png" alt="Five fingers detected" width="30%">
</p>

## Features

- Real-time hand detection inside a configurable region of interest (ROI).
- Two finger counting methods: **circle** (ring around the palm) and **defects** (valleys between fingers).
- Noise filtering with morphological operations and a minimum hand area.
- Background recalibration at any time, mirrored view and live FPS counter.
- Command-line options for camera, ROI, thresholds and counting method.

## How it works

1. **Background calibration.** For the first frames (60 by default) the ROI is averaged with `cv2.accumulateWeighted` to learn the empty background. Keep your hand out of the box during this step.
2. **Segmentation.** Each frame is converted to grayscale, blurred and subtracted from the background. The difference is thresholded, cleaned with morphological opening and closing, and the largest contour is taken as the hand if it is big enough.
3. **Palm detection.** The distance transform of the hand mask gives the largest inscribed circle: its center is the palm center and its radius the palm size. Every threshold below is relative to it, so counting does not depend on how close the hand is to the camera.
4. **Finger counting.**
   - `circle`: a ring 1.7× the palm radius is drawn around the palm and intersected with the mask. Each segment crossing the ring above the wrist is a finger.
   - `defects`: the convexity defects of the hand contour are the valleys between fingers. Each deep, sharp valley (angle under 110°) above the wrist separates two fingers. With no valleys, one finger is counted if the hand reaches well beyond the palm.

## Project structure

```
hand-tracker-openCV/
├── assets/demo/              # Screenshots used in this README (also used by the tests)
├── notebooks/
│   └── hand-detection.ipynb  # Original prototype
├── src/hand_tracker/
│   ├── __main__.py           # `python -m hand_tracker`
│   ├── app.py                # Webcam loop and command-line interface
│   ├── config.py             # Default parameters and ROI
│   ├── background.py         # Background model
│   ├── segmentation.py       # Hand segmentation
│   └── fingers.py            # Palm detection and finger counting methods
├── tests/                    # Unit and end-to-end tests (pytest)
├── .github/workflows/ci.yml  # Lint and tests on every push and pull request
└── pyproject.toml            # Metadata, dependencies and tool config
```

## Requirements

- Python 3.10+
- A webcam
- `opencv-python` and `numpy` (installed automatically)

## Installation

```bash
git clone https://github.com/emmanuelepp/hand-tracker-openCV.git
cd hand-tracker-openCV

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .                 # app only
pip install -e ".[dev]"          # + pytest and ruff
pip install -e ".[notebook]"     # + Jupyter, to run the original notebook
```

## Usage

```bash
hand-tracker
# or
python -m hand_tracker
```

Two windows open: **Finger Count** (camera feed with the ROI, hand contour and count) and **Thresholded** (hand mask).

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--camera N` | `0` | Camera index |
| `--method {circle,defects}` | `circle` | Finger counting method |
| `--roi TOP BOTTOM LEFT RIGHT` | `20 300 300 600` | Region of interest in pixels |
| `--threshold N` | `25` | Minimum gray-level difference from the background (0–255) |
| `--min-area N` | `2000` | Minimum hand area in pixels |
| `--calibration-frames N` | `60` | Frames used to learn the background |
| `--no-mirror` | | Do not flip the image horizontally |

Example:

```bash
hand-tracker --camera 1 --method defects --roi 50 350 280 580
```

### Controls

| Key | Action |
|-----|--------|
| `r` | Recalibrate the background |
| `q` / `Esc` | Quit |

### Tips

- Wait until calibration reaches 100% before putting your hand in the red box.
- Use a plain background that contrasts with your skin, and keep the camera and lighting steady.
- If the lighting changes or something moves in the background, press `r` with the box empty.
- If shadows or noise are detected as the hand, increase `--threshold` or `--min-area`.

## Limitations

- Requires a static camera and background; anything that moves inside the ROI is segmented as part of the hand.
- The hand must enter the ROI from the bottom, with the fingers pointing up.
- Fingers that touch each other can be counted as one.

## Development

```bash
pytest                 # run tests
ruff check .           # lint
ruff format .          # format
```

The tests do not need a camera: they use synthetic hand masks, the masks from the screenshots in `assets/demo/` and a fake video capture for the end-to-end loop.

## Roadmap

- [ ] Optional MediaPipe Hands backend.
- [ ] Skin-color segmentation as an alternative to background subtraction.
- [ ] Animated demo in the README.

## License

This project is licensed under the [MIT License](LICENSE).
