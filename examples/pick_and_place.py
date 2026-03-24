"""
pick_and_place.py
-----------------
Self-contained end-to-end example of a vision-guided pick-and-place cycle
using the UR5e robot arm, an Intel RealSense camera, and a YOLOv8 detector.

This script is intentionally verbose so that it can be read as a tutorial.
It mirrors the functionality of ``src/main.py`` but exposes every step
individually so you can adapt it to your specific workflow.

Usage
~~~~~
    python examples/pick_and_place.py \\
        --robot-ip   192.168.1.100 \\
        --model      yolov8n.pt \\
        --target     cup \\
        --transform  config/hand_eye_transform.npy

    # Without robot hardware (vision-only test):
    python examples/pick_and_place.py --dry-run --display
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

# Make sure the project root is importable even when running from examples/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.camera_interface import CameraInterface
from src.yolo_detector import YOLODetector
from src.calibration import HandEyeCalibration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pick-and-place demo: UR5e + RealSense + YOLOv8"
    )
    p.add_argument("--robot-ip", default="192.168.1.100", help="UR5e IP address")
    p.add_argument("--model", default="yolov8n.pt", help="Path to YOLOv8 .pt weights")
    p.add_argument("--target", default="cup", help="Object class to pick")
    p.add_argument(
        "--transform",
        default=os.path.join(_PROJECT_ROOT, "config", "hand_eye_transform.npy"),
        help="Path to the hand-eye extrinsic transform (.npy)",
    )
    p.add_argument(
        "--confidence", type=float, default=0.5, help="YOLO confidence threshold"
    )
    p.add_argument(
        "--approach-height",
        type=float,
        default=0.05,
        help="Approach height above target in metres",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip all robot motions (vision pipeline only)",
    )
    p.add_argument(
        "--display",
        action="store_true",
        help="Show the annotated camera feed in a window",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 – Initialise the camera
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Step 1: Initialising camera …")
    camera = CameraInterface(width=640, height=480, fps=30)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 – Load the YOLO detector
    # ──────────────────────────────────────────────────────────────────────────
    print("Step 2: Loading YOLOv8 detector …")
    detector = YOLODetector(model_path=args.model, confidence=args.confidence)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 – Load the hand-eye transform
    # ──────────────────────────────────────────────────────────────────────────
    print("Step 3: Loading hand-eye transform …")
    transform_path = args.transform
    if not os.path.isabs(transform_path):
        transform_path = os.path.join(_PROJECT_ROOT, transform_path)

    if os.path.exists(transform_path):
        T_base_cam = HandEyeCalibration.load(transform_path)
        print(f"  Loaded from {transform_path}")
    else:
        print(
            f"  WARNING: {transform_path} not found — using identity transform.\n"
            "  Run src/calibration.py to produce a real calibration."
        )
        T_base_cam = np.eye(4, dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4 – Connect to the robot (skipped in dry-run mode)
    # ──────────────────────────────────────────────────────────────────────────
    robot = None
    if not args.dry_run:
        print("Step 4: Connecting to UR5e …")
        from src.ur5e_controller import UR5eController
        robot = UR5eController(ip=args.robot_ip)
        robot.move_home()
        robot.open_gripper()
    else:
        print("Step 4: Dry-run mode — robot connection skipped.")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5 – Detection loop
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\nStep 5: Scanning for '{args.target}' …  (Ctrl+C to stop)\n")
    target_point_base: np.ndarray | None = None

    try:
        while True:
            color_image, depth_image = camera.get_frames()

            # Detect objects
            detections = detector.detect(
                color_image, target_classes=[args.target]
            )
            best = detector.best_detection(detections, target_class=args.target)

            if args.display:
                vis = detector.annotate(color_image, detections)
                cv2.imshow("Pick-and-place demo", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nUser pressed 'q' — exiting.")
                    break

            if best is None:
                continue

            cx, cy = best["centroid"]
            print(
                f"  Detected '{best['class']}' "
                f"conf={best['confidence']:.2f}  pixel=({cx},{cy})"
            )

            # Back-project centroid to 3-D camera frame
            p_cam = camera.pixel_to_3d(cx, cy, depth_image)
            if p_cam is None:
                print("  Depth not available at centroid — retrying …")
                continue

            # Convert to robot base frame using hand-eye transform
            p_base = HandEyeCalibration.transform_point(p_cam, T_base_cam)
            print(f"  Camera frame: {p_cam}")
            print(f"  Robot base  : {p_base}")
            target_point_base = p_base
            break  # found a target — proceed with pick

    except KeyboardInterrupt:
        print("\nInterrupted — no pick will be performed.")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6 – Execute pick-and-place
    # ──────────────────────────────────────────────────────────────────────────
    if target_point_base is not None and robot is not None:
        print("\nStep 6: Executing pick-and-place …")
        orient = [3.14159, 0.0, 0.0]  # tool pointing down

        approach_pose = [
            target_point_base[0],
            target_point_base[1],
            target_point_base[2] + args.approach_height,
        ] + orient

        pick_pose = [
            target_point_base[0],
            target_point_base[1],
            target_point_base[2],
        ] + orient

        print(f"  Moving to approach: {approach_pose}")
        robot.move_to_pose(approach_pose)

        print(f"  Descending to pick: {pick_pose}")
        robot.move_to_pose(pick_pose)

        print("  Closing gripper …")
        robot.close_gripper()

        print("  Lifting object …")
        robot.move_to_pose(approach_pose)

        print("  Returning to home …")
        robot.move_home()

        print("  Opening gripper — object deposited.")
        robot.open_gripper()

        print("\nPick-and-place complete!")
    elif target_point_base is not None and args.dry_run:
        print("\nStep 6: Dry-run — skipping robot motion.")
        print(f"  Would pick at base-frame coords: {target_point_base}")
    else:
        print("\nStep 6: No target detected — nothing to pick.")

    # ──────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ──────────────────────────────────────────────────────────────────────────
    camera.stop()
    if robot is not None:
        robot.disconnect()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
