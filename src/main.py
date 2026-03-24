"""
main.py
-------
Full vision-guided pick-and-place integration pipeline.

Running this script:
1. Loads configuration from ``config/settings.yaml``.
2. Connects to the UR5e robot via RTDE.
3. Starts the RealSense camera (falls back to webcam if unavailable).
4. Loads the pre-computed hand-eye transform.
5. Continuously captures frames and detects the target object with YOLOv8.
6. On the first valid detection:
   a. Back-projects the centroid pixel to a 3-D camera-frame point.
   b. Transforms the point to the robot-base frame.
   c. Executes a pick sequence: approach → descend → grasp → lift → home.
7. Opens the gripper and exits.

Usage
~~~~~
    python src/main.py

Optional flags
~~~~~~~~~~~~~~
    --config    Path to the YAML config file (default: config/settings.yaml)
    --dry-run   Run the vision pipeline without moving the robot
    --display   Show the annotated camera feed in an OpenCV window
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

# Ensure the project root is on sys.path when the script is invoked directly.
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.camera_interface import CameraInterface
from src.yolo_detector import YOLODetector
from src.calibration import HandEyeCalibration


def load_config(path: str) -> dict:
    """Load and return the YAML configuration file."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UR5e + Camera + YOLO vision-guided pick-and-place"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(_PROJECT_ROOT, "config", "settings.yaml"),
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run vision pipeline only — do not move the robot",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the annotated camera feed in an OpenCV window",
    )
    return parser.parse_args()


def build_pick_pose(
    target_xyz: np.ndarray,
    approach_height: float = 0.05,
    tool_orientation: list | None = None,
) -> tuple[list, list]:
    """Build approach and pick poses from a target 3-D point.

    Parameters
    ----------
    target_xyz:
        [x, y, z] of the target in the robot base frame (metres).
    approach_height:
        Height offset above the target for the approach waypoint (metres).
    tool_orientation:
        [rx, ry, rz] rotation vector for the TCP (radians).  Defaults to
        straight-down (tool pointing toward negative Z-axis in base frame).

    Returns
    -------
    (approach_pose, pick_pose) each as a 6-element list [x,y,z,rx,ry,rz].
    """
    if tool_orientation is None:
        tool_orientation = [3.14159, 0.0, 0.0]

    approach_pose = [
        target_xyz[0],
        target_xyz[1],
        target_xyz[2] + approach_height,
    ] + tool_orientation

    pick_pose = [
        target_xyz[0],
        target_xyz[1],
        target_xyz[2],
    ] + tool_orientation

    return approach_pose, pick_pose


def run_pipeline(cfg: dict, dry_run: bool = False, display: bool = False) -> None:
    """Execute the full detection-to-motion pipeline."""

    # ------------------------------------------------------------------ camera
    cam_cfg = cfg.get("camera", {})
    camera = CameraInterface(
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
        fps=cam_cfg.get("fps", 30),
    )

    # ------------------------------------------------------------------ YOLO
    yolo_cfg = cfg.get("yolo", {})
    detector = YOLODetector(
        model_path=yolo_cfg.get("model", "yolov8n.pt"),
        confidence=yolo_cfg.get("confidence", 0.5),
    )
    target_class = yolo_cfg.get("target_class", None)

    # -------------------------------------------------------------- calibration
    cal_cfg = cfg.get("calibration", {})
    transform_path = cal_cfg.get("extrinsic_file", "config/hand_eye_transform.npy")
    if not os.path.isabs(transform_path):
        transform_path = os.path.join(_PROJECT_ROOT, transform_path)

    if not os.path.exists(transform_path):
        print(
            f"[main] WARNING: hand-eye transform not found at {transform_path}.\n"
            "       Run src/calibration.py to generate it.  "
            "Using identity transform as a placeholder."
        )
        T_base_cam = np.eye(4, dtype=np.float64)
    else:
        T_base_cam = HandEyeCalibration.load(transform_path)
        print(f"[main] Loaded hand-eye transform from {transform_path}")

    approach_height = cal_cfg.get("approach_height", 0.05)

    # ------------------------------------------------------------------ robot
    robot = None
    if not dry_run:
        from src.ur5e_controller import UR5eController
        robot_cfg = cfg.get("robot", {})
        robot = UR5eController(
            ip=robot_cfg["ip"],
            speed=robot_cfg.get("speed", 0.1),
            acceleration=robot_cfg.get("acceleration", 0.5),
            tcp_offset=robot_cfg.get("tcp_offset"),
            home_joints=robot_cfg.get("home_joints"),
        )
        robot.move_home()
        robot.open_gripper()

    # ----------------------------------------------------------------- main loop
    print(
        f"[main] Starting detection loop.  "
        f"Target class: {target_class!r}.  "
        f"Press Ctrl+C to quit."
    )
    try:
        while True:
            color_image, depth_image = camera.get_frames()
            detections = detector.detect(color_image, target_classes=[target_class] if target_class else None)
            best = detector.best_detection(detections, target_class=target_class)

            if display:
                annotated = detector.annotate(color_image, detections)
                cv2.imshow("UR5e-YOLO-Vision-Integration", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if best is None:
                continue

            cx, cy = best["centroid"]
            print(
                f"[main] Detected '{best['class']}' "
                f"(conf={best['confidence']:.2f}) at pixel ({cx}, {cy})"
            )

            # Back-project to 3-D camera frame
            point_cam = camera.pixel_to_3d(cx, cy, depth_image)
            if point_cam is None:
                print("[main] Depth unavailable at centroid — skipping frame.")
                continue

            print(f"[main] Camera-frame point: {point_cam}")

            # Transform to robot base frame
            point_base = HandEyeCalibration.transform_point(point_cam, T_base_cam)
            print(f"[main] Robot-base-frame point: {point_base}")

            # Execute pick sequence
            if not dry_run and robot is not None:
                approach_pose, pick_pose = build_pick_pose(
                    point_base, approach_height=approach_height
                )
                print(f"[main] Moving to approach pose: {approach_pose}")
                robot.move_to_pose(approach_pose)
                print(f"[main] Descending to pick pose: {pick_pose}")
                robot.move_to_pose(pick_pose)
                robot.close_gripper()
                print("[main] Gripper closed — lifting object …")
                robot.move_to_pose(approach_pose)
                robot.move_home()
                robot.open_gripper()
                print("[main] Pick-and-place cycle complete.")
            else:
                print("[main] Dry-run: skipping robot motion.")

            # After a successful pick, exit the loop.
            break

    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")

    finally:
        camera.stop()
        if robot is not None:
            robot.disconnect()
        if display:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg, dry_run=args.dry_run, display=args.display)


if __name__ == "__main__":
    main()
