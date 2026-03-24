"""
ur5e_controller.py
------------------
Thin wrapper around the ``ur_rtde`` RTDE interface that provides a simple,
high-level API for controlling the Universal Robots UR5e arm.

Dependencies
~~~~~~~~~~~~
- ur-rtde >= 1.5.7  (pip install ur-rtde)

Usage
~~~~~
    from src.ur5e_controller import UR5eController

    robot = UR5eController(ip="192.168.1.100")
    robot.move_to_pose([0.3, -0.1, 0.4, 3.14, 0.0, 0.0])
    robot.close_gripper()
    robot.move_home()
    robot.disconnect()
"""

from __future__ import annotations

import time
from typing import List

try:
    import rtde_control
    import rtde_receive
    _RTDE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RTDE_AVAILABLE = False


# Default joint configuration (radians) — elbow-up, tool pointing down.
_DEFAULT_HOME_JOINTS: List[float] = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]

# Default tool-centre-point offset for a ~170 mm gripper mounted on the flange.
_DEFAULT_TCP_OFFSET: List[float] = [0.0, 0.0, 0.17, 0.0, 0.0, 0.0]


class UR5eController:
    """High-level controller for the UR5e collaborative robot arm.

    Parameters
    ----------
    ip:
        IP address of the UR5e controller box.
    speed:
        Default TCP speed in m/s used for linear moves.
    acceleration:
        Default TCP acceleration in m/s² used for linear moves.
    tcp_offset:
        Tool-centre-point offset [x, y, z, rx, ry, rz] (m / rad) applied on
        top of the flange frame.  Adjust to match your gripper geometry.
    home_joints:
        Joint angles [q1..q6] (rad) used by :meth:`move_home`.
    gripper_open_output:
        Digital output index used to *open* the gripper (default 0).
    gripper_close_output:
        Digital output index used to *close* the gripper (default 1).
    """

    def __init__(
        self,
        ip: str,
        speed: float = 0.1,
        acceleration: float = 0.5,
        tcp_offset: List[float] | None = None,
        home_joints: List[float] | None = None,
        gripper_open_output: int = 0,
        gripper_close_output: int = 1,
    ) -> None:
        if not _RTDE_AVAILABLE:
            raise ImportError(
                "ur-rtde is not installed.  Run: pip install ur-rtde"
            )

        self.ip = ip
        self.speed = speed
        self.acceleration = acceleration
        self.tcp_offset = tcp_offset or _DEFAULT_TCP_OFFSET
        self.home_joints = home_joints or _DEFAULT_HOME_JOINTS
        self._gripper_open_output = gripper_open_output
        self._gripper_close_output = gripper_close_output

        print(f"[UR5eController] Connecting to robot at {ip} …")
        self._ctrl = rtde_control.RTDEControlInterface(ip)
        self._recv = rtde_receive.RTDEReceiveInterface(ip)

        # Apply TCP offset so all subsequent moves are in the TCP frame.
        self._ctrl.setTcp(self.tcp_offset)
        print("[UR5eController] Connected.")

    # ------------------------------------------------------------------
    # Motion primitives
    # ------------------------------------------------------------------

    def move_to_pose(
        self,
        pose: List[float],
        speed: float | None = None,
        acceleration: float | None = None,
        asynchronous: bool = False,
    ) -> None:
        """Move the TCP to *pose* using a linear (moveL) trajectory.

        Parameters
        ----------
        pose:
            Target pose as [x, y, z, rx, ry, rz] in metres and radians
            (robot base frame).
        speed:
            Override the default TCP speed (m/s).
        acceleration:
            Override the default TCP acceleration (m/s²).
        asynchronous:
            If ``True`` the call returns immediately without waiting for the
            motion to finish.
        """
        if len(pose) != 6:
            raise ValueError("pose must have exactly 6 elements [x,y,z,rx,ry,rz]")
        v = speed or self.speed
        a = acceleration or self.acceleration
        self._ctrl.moveL(pose, v, a, asynchronous)

    def move_joints(
        self,
        joints: List[float],
        speed: float = 1.05,
        acceleration: float = 1.4,
        asynchronous: bool = False,
    ) -> None:
        """Move all joints to the specified angles using a joint-space (moveJ) move.

        Parameters
        ----------
        joints:
            Target joint angles [q1..q6] in radians.
        speed:
            Joint speed in rad/s.
        acceleration:
            Joint acceleration in rad/s².
        asynchronous:
            If ``True`` the call returns immediately.
        """
        if len(joints) != 6:
            raise ValueError("joints must have exactly 6 elements")
        self._ctrl.moveJ(joints, speed, acceleration, asynchronous)

    def move_home(self) -> None:
        """Move the robot to its configured *home* joint configuration."""
        print("[UR5eController] Moving to home position …")
        self.move_joints(self.home_joints)

    # ------------------------------------------------------------------
    # Gripper control (via digital outputs)
    # ------------------------------------------------------------------

    def open_gripper(self, settle_time: float = 0.5) -> None:
        """Open the gripper by asserting the configured digital output.

        Parameters
        ----------
        settle_time:
            Time in seconds to wait after the signal is sent.
        """
        self._ctrl.setStandardDigitalOut(self._gripper_open_output, True)
        self._ctrl.setStandardDigitalOut(self._gripper_close_output, False)
        time.sleep(settle_time)

    def close_gripper(self, settle_time: float = 0.5) -> None:
        """Close the gripper by asserting the configured digital output.

        Parameters
        ----------
        settle_time:
            Time in seconds to wait after the signal is sent.
        """
        self._ctrl.setStandardDigitalOut(self._gripper_close_output, True)
        self._ctrl.setStandardDigitalOut(self._gripper_open_output, False)
        time.sleep(settle_time)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_tcp_pose(self) -> List[float]:
        """Return the current TCP pose [x, y, z, rx, ry, rz] in the base frame."""
        return self._recv.getActualTCPPose()

    def get_joint_positions(self) -> List[float]:
        """Return the current joint positions [q1..q6] in radians."""
        return self._recv.getActualQ()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def disconnect(self) -> None:
        """Stop the RTDE connection cleanly."""
        self._ctrl.stopScript()
        self._ctrl.disconnect()
        self._recv.disconnect()
        print("[UR5eController] Disconnected.")
