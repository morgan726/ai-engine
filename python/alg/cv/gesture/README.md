# RealSense Gesture Demo

This demo recognizes simple static and dynamic hand gestures with an Intel
RealSense RGB-D camera.

It does not use a trained model. The pipeline is:

1. Align depth to the color frame.
2. Segment the nearest hand-sized depth blob.
3. Classify static hand shapes from contour, convex hull, convexity defects,
   solidity, and aspect ratio.
4. Track the palm center in 3D and classify dynamic gestures from trajectory.

## Dependencies

Install the RealSense SDK and Python bindings first, then install OpenCV and
NumPy:

```bash
pip install pyrealsense2 opencv-python numpy
```

If `pyrealsense2` is not available for the current Python version, install the
RealSense SDK package for the system Python supported by your distribution.

## Run

```bash
python3 python/alg/cv/gesture/realsense_gesture_demo.py --show-mask
```

Useful options:

```bash
python3 python/alg/cv/gesture/realsense_gesture_demo.py \
  --min-depth 0.20 \
  --max-depth 1.20 \
  --depth-band 0.18 \
  --min-area 2500 \
  --history 1.2 \
  --show-mask
```

Press `q` or `Esc` to quit.

## Gestures

Static labels:

- `open palm`
- `fist`
- `two fingers`
- `three/four fingers`
- `pointing`

Dynamic labels:

- `swipe left`
- `swipe right`
- `wave`
- `move up`
- `move down`
- `push`
- `pull`

## Tuning

Use `--show-mask` first. A good mask should cover the hand but not the table,
body, or background.

- Increase `--min-depth` if the camera sees objects too close.
- Decrease `--max-depth` if background objects are selected.
- Decrease `--depth-band` in cluttered scenes.
- Increase `--min-area` if small objects are detected as hands.
- Increase `--history` if dynamic gestures are too hard to trigger.

Keep the hand as the closest object in front of the camera. This rule-based demo
is meant for quick validation; production gesture recognition should use a hand
landmark model or a trained RGB-D classifier.
