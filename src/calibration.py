"""
calibration.py
--------------
Hand-eye calibration utilities for the UR5e + camera setup.

Two common configurations are supported:

``eye-to-hand``
    The camera is fixed to the workcell (not on the robot).  The transform
    T_base_cam is estimated so that a point expressed in camera coordinates
    can be converted to robot-base coordinates.

``eye-in-hand``
    The camera is mounted on the robot's end-effector.  The transform
    T_flange_cam is estimated using OpenCV's ``calibrateHandEye``.

Workflow (eye-to-hand example)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Move the robot to N different poses and note each TCP pose.
2. At each pose, detect the calibration board in the camera image and
   compute its pose relative to the camera.
3. Call :meth:`HandEyeCalibration.add_sample` for each (robot, board) pair.
4. Call :meth:`HandEyeCalibration.compute` to get the transform.
5. Save it with :meth:`HandEyeCalibration.save`.

Dependencies
~~~~~~~~~~~~
- numpy >= 1.24
- opencv-python >= 4.8
- scipy >= 1.11 (for Rodrigues / quaternion helpers)

Usage
~~~~~
    from src.calibration import HandEyeCalibration
    import numpy as np

    cal = HandEyeCalibration(method="eye-to-hand")
    cal.add_sample(robot_pose_4x4, board_in_camera_4x4)  # repeat N times
    T = cal.compute()
    cal.save("config/hand_eye_transform.npy")

    # Later, to transform a camera-frame point to the robot base frame:
    T_loaded = HandEyeCalibration.load("config/hand_eye_transform.npy")
    robot_point = HandEyeCalibration.transform_point(camera_point, T_loaded)
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def pose_vec_to_matrix(pose: List[float]) -> np.ndarray:
    """Convert a UR5e-style pose vector [x,y,z,rx,ry,rz] to a 4×4 matrix.

    The rotation part is encoded as a *rotation vector* (axis × angle) as used
    by the UR controller, which matches OpenCV's Rodrigues convention.

    Parameters
    ----------
    pose:
        [x, y, z, rx, ry, rz] in metres and radians.

    Returns
    -------
    4×4 homogeneous transform as a float64 numpy array.
    """
    pose = np.asarray(pose, dtype=np.float64)
    rvec = pose[3:6].reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pose[:3]
    return T


def matrix_to_pose_vec(T: np.ndarray) -> np.ndarray:
    """Convert a 4×4 homogeneous matrix to a UR5e pose vector [x,y,z,rx,ry,rz].

    Returns
    -------
    numpy array of shape (6,).
    """
    R = T[:3, :3]
    t = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([t, rvec.ravel()])


# ──────────────────────────────────────────────────────────────────────────────
# HandEyeCalibration class
# ──────────────────────────────────────────────────────────────────────────────

class HandEyeCalibration:
    """Collect robot / board pose pairs and compute the hand-eye transform.

    Parameters
    ----------
    method:
        ``"eye-to-hand"`` (camera fixed, default) or ``"eye-in-hand"``
        (camera on end-effector).
    cv_method:
        OpenCV hand-eye method constant.  Defaults to
        ``cv2.CALIB_HAND_EYE_TSAI`` which works well in most situations.
    """

    def __init__(
        self,
        method: str = "eye-to-hand",
        cv_method: int = cv2.CALIB_HAND_EYE_TSAI,
    ) -> None:
        if method not in ("eye-to-hand", "eye-in-hand"):
            raise ValueError("method must be 'eye-to-hand' or 'eye-in-hand'")
        self.method = method
        self.cv_method = cv_method
        self._robot_poses: List[np.ndarray] = []
        self._board_poses: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Sample collection
    # ------------------------------------------------------------------

    def add_sample(
        self,
        robot_pose: List[float] | np.ndarray,
        board_in_camera: np.ndarray,
    ) -> None:
        """Add one calibration sample.

        Parameters
        ----------
        robot_pose:
            Current TCP pose as either a 6-element pose vector
            ``[x,y,z,rx,ry,rz]`` or a 4×4 homogeneous matrix.
        board_in_camera:
            4×4 homogeneous transform of the calibration board in the camera
            frame (e.g., obtained from ``cv2.solvePnP``).
        """
        if isinstance(robot_pose, (list, tuple)):
            robot_pose = np.asarray(robot_pose, dtype=np.float64)

        if robot_pose.shape == (6,):
            T_robot = pose_vec_to_matrix(robot_pose)
        elif robot_pose.shape == (4, 4):
            T_robot = robot_pose.astype(np.float64)
        else:
            raise ValueError("robot_pose must be shape (6,) or (4,4)")

        self._robot_poses.append(T_robot)
        self._board_poses.append(np.asarray(board_in_camera, dtype=np.float64))

    @property
    def num_samples(self) -> int:
        """Number of collected calibration samples."""
        return len(self._robot_poses)

    def clear(self) -> None:
        """Remove all previously added samples."""
        self._robot_poses.clear()
        self._board_poses.clear()

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self) -> np.ndarray:
        """Compute the hand-eye transform from the collected samples.

        Returns
        -------
        4×4 homogeneous transform ``T_base_cam`` (eye-to-hand) or
        ``T_flange_cam`` (eye-in-hand) as a float64 numpy array.

        Raises
        ------
        ValueError
            If fewer than 3 samples have been collected.
        """
        if self.num_samples < 3:
            raise ValueError(
                f"At least 3 samples are required (have {self.num_samples})."
            )

        R_gripper2base, t_gripper2base = self._decompose(self._robot_poses)
        R_target2cam, t_target2cam = self._decompose(self._board_poses)

        if self.method == "eye-to-hand":
            # For eye-to-hand the convention requires the inverse of the robot pose.
            R_base2gripper = [R.T for R in R_gripper2base]
            t_base2gripper = [-R.T @ t for R, t in zip(R_gripper2base, t_gripper2base)]
            R_cam, t_cam = cv2.calibrateHandEye(
                R_base2gripper, t_base2gripper,
                R_target2cam, t_target2cam,
                method=self.cv_method,
            )
        else:
            R_cam, t_cam = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=self.cv_method,
            )

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_cam
        T[:3, 3] = t_cam.ravel()
        return T

    @staticmethod
    def _decompose(
        matrices: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Split a list of 4×4 matrices into rotation and translation lists."""
        Rs = [M[:3, :3] for M in matrices]
        ts = [M[:3, 3:4] for M in matrices]
        return Rs, ts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Compute and save the hand-eye transform to a ``.npy`` file.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``"config/hand_eye_transform.npy"``).
        """
        T = self.compute()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.save(path, T)
        print(f"[HandEyeCalibration] Transform saved → {path}")

    @staticmethod
    def load(path: str) -> np.ndarray:
        """Load a previously saved hand-eye transform.

        Parameters
        ----------
        path:
            Path to the ``.npy`` file produced by :meth:`save`.

        Returns
        -------
        4×4 float64 numpy array.
        """
        T = np.load(path)
        if T.shape != (4, 4):
            raise ValueError(f"Expected shape (4,4), got {T.shape} from {path}")
        return T

    # ------------------------------------------------------------------
    # Coordinate transform helpers
    # ------------------------------------------------------------------

    @staticmethod
    def transform_point(
        point_in_camera: np.ndarray,
        T_base_cam: np.ndarray,
    ) -> np.ndarray:
        """Transform a 3-D point from the camera frame to the robot base frame.

        Parameters
        ----------
        point_in_camera:
            3-element array [x, y, z] in metres (camera frame).
        T_base_cam:
            4×4 homogeneous transform loaded via :meth:`load`.

        Returns
        -------
        3-element array [x, y, z] in metres (robot base frame).
        """
        p = np.ones(4, dtype=np.float64)
        p[:3] = point_in_camera
        return (T_base_cam @ p)[:3]

    @staticmethod
    def transform_pose(
        pose_in_camera: np.ndarray,
        T_base_cam: np.ndarray,
    ) -> np.ndarray:
        """Transform a 4×4 pose matrix from camera frame to robot base frame.

        Parameters
        ----------
        pose_in_camera:
            4×4 homogeneous matrix in the camera frame.
        T_base_cam:
            Hand-eye transform loaded via :meth:`load`.

        Returns
        -------
        4×4 homogeneous matrix in the robot base frame.
        """
        return T_base_cam @ pose_in_camera
