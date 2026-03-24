"""
camera_interface.py
-------------------
Wrapper around the Intel RealSense D435/D435i SDK that streams aligned
colour and depth frames and provides a pixel-to-3D back-projection helper.

Falls back to a plain OpenCV ``VideoCapture`` (colour only, no depth) when no
RealSense device is connected, so the rest of the pipeline can be exercised
without hardware present.

Dependencies
~~~~~~~~~~~~
- pyrealsense2 >= 2.54  (pip install pyrealsense2)
- opencv-python >= 4.8  (pip install opencv-python)
- numpy >= 1.24         (pip install numpy)

Usage
~~~~~
    from src.camera_interface import CameraInterface

    cam = CameraInterface(width=640, height=480, fps=30)
    color_image, depth_image = cam.get_frames()

    # Back-project the pixel (320, 240) to a 3-D point in metres
    point_3d = cam.pixel_to_3d(u=320, v=240, depth_image=depth_image)
    cam.stop()
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
    _RS_AVAILABLE = True
except ImportError:
    _RS_AVAILABLE = False

import cv2


class CameraInterface:
    """Streams colour and depth frames from an Intel RealSense camera.

    If no RealSense device is present the class falls back to a plain
    OpenCV webcam capture (depth will be ``None``).

    Parameters
    ----------
    width:
        Frame width in pixels.
    height:
        Frame height in pixels.
    fps:
        Target frame rate.
    device_serial:
        Optional serial number to open a specific RealSense device when
        multiple cameras are connected.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        device_serial: Optional[str] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._intrinsics: Optional[rs.intrinsics] = None
        self._pipeline: Optional[rs.pipeline] = None
        self._cap: Optional[cv2.VideoCapture] = None

        if _RS_AVAILABLE and self._realsense_device_present(device_serial):
            self._start_realsense(device_serial)
        else:
            print(
                "[CameraInterface] No RealSense device found — "
                "falling back to OpenCV VideoCapture (colour only)."
            )
            self._start_opencv()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _realsense_device_present(serial: Optional[str]) -> bool:
        """Return ``True`` if at least one RealSense device is connected."""
        if not _RS_AVAILABLE:
            return False
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            return False
        if serial is not None:
            return any(d.get_info(rs.camera_info.serial_number) == serial for d in devices)
        return True

    def _start_realsense(self, serial: Optional[str]) -> None:
        """Configure and start the RealSense pipeline."""
        self._pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile = self._pipeline.start(config)

        # Store intrinsics for back-projection
        color_profile = (
            profile.get_stream(rs.stream.color).as_video_stream_profile()
        )
        self._intrinsics = color_profile.get_intrinsics()

        # Align depth to colour frame
        self._align = rs.align(rs.stream.color)
        print(
            f"[CameraInterface] RealSense started "
            f"({self.width}×{self.height} @ {self.fps} fps)."
        )

    def _start_opencv(self) -> None:
        """Open the default webcam via OpenCV."""
        self._cap = cv2.VideoCapture(0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._cap.isOpened():
            raise RuntimeError(
                "[CameraInterface] Failed to open webcam with OpenCV."
            )
        print("[CameraInterface] OpenCV VideoCapture started.")

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def get_frames(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Capture and return the next (colour, depth) frame pair.

        Returns
        -------
        color_image:
            BGR image as a ``uint8`` numpy array of shape (H, W, 3).
        depth_image:
            Depth map as a ``uint16`` numpy array of shape (H, W) where each
            value is the distance in millimetres.  ``None`` when using the
            OpenCV fallback (no depth sensor available).
        """
        if self._pipeline is not None:
            return self._get_realsense_frames()
        return self._get_opencv_frame()

    def _get_realsense_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def _get_opencv_frame(self) -> Tuple[np.ndarray, None]:
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("[CameraInterface] Failed to read frame from webcam.")
        return frame, None

    # ------------------------------------------------------------------
    # 3-D back-projection
    # ------------------------------------------------------------------

    def pixel_to_3d(
        self,
        u: int,
        v: int,
        depth_image: np.ndarray,
        depth_scale: float = 0.001,
    ) -> Optional[np.ndarray]:
        """Back-project a pixel coordinate to a 3-D point in *camera* space.

        Parameters
        ----------
        u, v:
            Column and row of the pixel (origin at top-left).
        depth_image:
            Depth map returned by :meth:`get_frames` (uint16, millimetres).
        depth_scale:
            Multiply raw depth values by this factor to get metres
            (0.001 converts mm → m, which is the RealSense default).

        Returns
        -------
        numpy array [x, y, z] in metres, or ``None`` if depth is zero /
        invalid.
        """
        if depth_image is None:
            return None

        raw_depth = depth_image[int(v), int(u)]
        if raw_depth == 0:
            return None

        z = float(raw_depth) * depth_scale

        if self._intrinsics is not None:
            # Use the RealSense SDK de-projection for maximum accuracy.
            point = rs.rs2_deproject_pixel_to_point(
                self._intrinsics, [float(u), float(v)], z
            )
            return np.array(point, dtype=np.float64)

        # Fallback: use a simple pin-hole model with assumed focal lengths.
        cx = self.width / 2.0
        cy = self.height / 2.0
        fx = fy = max(self.width, self.height)  # rough estimate
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float64)

    # ------------------------------------------------------------------
    # Intrinsics access
    # ------------------------------------------------------------------

    def get_intrinsics_matrix(self) -> Optional[np.ndarray]:
        """Return the 3×3 camera intrinsics matrix K, or ``None`` if unknown."""
        if self._intrinsics is None:
            return None
        i = self._intrinsics
        return np.array(
            [
                [i.fx, 0.0, i.ppx],
                [0.0, i.fy, i.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Clean-up
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop the camera pipeline and release all resources."""
        if self._pipeline is not None:
            self._pipeline.stop()
            print("[CameraInterface] RealSense pipeline stopped.")
        if self._cap is not None:
            self._cap.release()
            print("[CameraInterface] OpenCV VideoCapture released.")
