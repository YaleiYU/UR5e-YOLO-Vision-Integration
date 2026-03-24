# UR5e-YOLO-Vision-Integration

A complete guidance and integration framework for connecting a **Universal Robots UR5e** robotic arm with an **Intel RealSense** depth camera and a **YOLOv8** object detector to enable vision-guided manipulation tasks (e.g., pick-and-place on a platform).

---

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Prerequisites](#software-prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [UR5e Controller](#ur5e-controller)
  - [Camera Interface](#camera-interface)
  - [YOLO Detector](#yolo-detector)
  - [Hand-Eye Calibration](#hand-eye-calibration)
  - [Main Integration](#main-integration)
- [Example: Pick and Place](#example-pick-and-place)
- [Troubleshooting](#troubleshooting)

---

## Overview

```
Camera (RealSense D435)
       │
       ▼
YOLO Detector  ──────► 2-D pixel coordinates of detected object
       │
       ▼
Calibration (hand-eye)  ──► 3-D robot-base coordinates
       │
       ▼
UR5e Controller  ──────────► Move robot to target pose & actuate gripper
```

The pipeline:
1. The camera captures an RGB-D frame.
2. YOLOv8 detects objects of interest in the colour image.
3. The pixel centroid of a detection is back-projected to 3-D using the depth image and the RealSense intrinsics.
4. A pre-computed hand-eye transform converts the camera-frame point to the robot-base frame.
5. The UR5e RTDE interface moves the robot to the target, closes the gripper, and returns to home.

---

## Hardware Requirements

| Component | Notes |
|-----------|-------|
| Universal Robots UR5e | Firmware ≥ 5.1 |
| Intel RealSense D435 / D435i | USB 3.x port required |
| Gripper | Any digital output gripper (e.g., Robotiq 2F-85) |
| Host PC | Ubuntu 20.04 / 22.04 recommended |
| Network | UR5e and PC on the same Ethernet subnet |

---

## Software Prerequisites

The following software must already be installed on the **host PC** (as stated in the project brief):

- **Ubuntu 20.04 or 22.04**
- **Python 3.8+**
- **UR RTDE driver** – `ur_rtde` (installed via pip, see below)
- **Intel RealSense SDK 2.0** (`librealsense2`) – install from [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)
- **CUDA** *(optional)* – for GPU-accelerated YOLO inference

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YaleiYU/UR5e-YOLO-Vision-Integration.git
cd UR5e-YOLO-Vision-Integration

# 2. (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

---

## Configuration

Edit `config/settings.yaml` before running any script:

```yaml
robot:
  ip: "192.168.1.100"   # ← change to your UR5e IP address

camera:
  width: 640
  height: 480
  fps: 30

yolo:
  model: "yolov8n.pt"   # ← use your custom model path if needed
  confidence: 0.5
  target_class: "cup"   # ← object class to pick

calibration:
  extrinsic_file: "config/hand_eye_transform.npy"
```

---

## Project Structure

```
UR5e-YOLO-Vision-Integration/
├── config/
│   ├── settings.yaml            # Global configuration
│   └── hand_eye_transform.npy   # Saved hand-eye calibration (generated)
├── src/
│   ├── __init__.py
│   ├── ur5e_controller.py       # UR5e RTDE wrapper
│   ├── camera_interface.py      # RealSense / OpenCV camera wrapper
│   ├── yolo_detector.py         # YOLOv8 detector wrapper
│   ├── calibration.py           # Hand-eye calibration utilities
│   └── main.py                  # Full integration pipeline
├── examples/
│   └── pick_and_place.py        # End-to-end pick-and-place demo
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Run the full vision-guided pick-and-place pipeline
python src/main.py
```

or use the self-contained example:

```bash
python examples/pick_and_place.py
```

---

## Modules

### UR5e Controller

`src/ur5e_controller.py`

Wraps `ur_rtde.RTDEControlInterface` and `ur_rtde.RTDEReceiveInterface` to provide a simple API:

```python
from src.ur5e_controller import UR5eController

robot = UR5eController(ip="192.168.1.100")
robot.move_to_pose([x, y, z, rx, ry, rz])   # [m, m, m, rad, rad, rad]
robot.open_gripper()
robot.close_gripper()
robot.move_home()
robot.disconnect()
```

### Camera Interface

`src/camera_interface.py`

Wraps `pyrealsense2` to stream aligned colour + depth frames and to back-project a pixel to a 3-D point:

```python
from src.camera_interface import CameraInterface

cam = CameraInterface(width=640, height=480, fps=30)
color_image, depth_image = cam.get_frames()
point_3d = cam.pixel_to_3d(u=320, v=240, depth_image=depth_image)  # metres
cam.stop()
```

Falls back to an OpenCV `VideoCapture` when no RealSense device is detected.

### YOLO Detector

`src/yolo_detector.py`

Wraps `ultralytics.YOLO`:

```python
from src.yolo_detector import YOLODetector

detector = YOLODetector(model_path="yolov8n.pt", confidence=0.5)
detections = detector.detect(color_image)
# detections: list of {"class": str, "confidence": float,
#                       "bbox": [x1,y1,x2,y2], "centroid": (cx,cy)}
annotated = detector.annotate(color_image, detections)
```

### Hand-Eye Calibration

`src/calibration.py`

Provides utilities for *eye-to-hand* calibration (camera mounted on the robot base / workcell) and persisting the resulting transform:

```python
from src.calibration import HandEyeCalibration

cal = HandEyeCalibration()
# Collect correspondences while moving the robot
cal.add_sample(robot_pose, board_pose_in_camera)
T_base_cam = cal.compute()   # 4×4 homogeneous transform
cal.save("config/hand_eye_transform.npy")

# Later, load and use it
T = cal.load("config/hand_eye_transform.npy")
robot_point = cal.transform_point(camera_point, T)
```

### Main Integration

`src/main.py`

Runs the full loop:
1. Load config.
2. Connect to robot and camera.
3. Load hand-eye transform.
4. Capture frame → detect objects → pick target detection.
5. Back-project centroid to 3-D (camera frame) → transform to robot base.
6. Move robot to target, close gripper, lift, move home, open gripper.

---

## Example: Pick and Place

```bash
python examples/pick_and_place.py \
    --robot-ip 192.168.1.100 \
    --model yolov8n.pt \
    --target cup \
    --transform config/hand_eye_transform.npy
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ConnectionRefusedError` when connecting to robot | Wrong IP or robot in protective stop | Check IP in `settings.yaml`; release protective stop on teach pendant |
| `RuntimeError: No device connected` from RealSense | Camera not plugged in or wrong USB port | Use a USB 3.x port; check `rs-enumerate-devices` |
| YOLO model not found | Wrong path in `settings.yaml` | Set `yolo.model` to the absolute or relative path of your `.pt` file |
| Low detection confidence | Lighting or model mismatch | Tune `yolo.confidence`; use a domain-specific fine-tuned model |
| Transform mismatch (robot misses object) | Stale calibration | Re-run hand-eye calibration and save new `hand_eye_transform.npy` |
