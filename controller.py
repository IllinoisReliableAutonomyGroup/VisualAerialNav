"""Controllers for the quadrotor plant. Includes a geometric position controller on SE(3) and a random controller for testing. In addition to returning control inputs, controllers can also return logging values such as computed pose estimates, input images and observations, etc."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, Tuple, TypeVar

import numpy as np

try:  # pragma: no cover
    from .math_utils import vee, normalize, rpy_to_rot
    from .plant import ControlInput, QuadrotorParams, QuadrotorState
except ImportError:  # pragma: no cover
    from math_utils import vee, normalize, rpy_to_rot
    from plant import ControlInput, QuadrotorParams, QuadrotorState
try:  # pragma: no cover
    import cv2
except ImportError:
    cv2 = None


@dataclass
class TrajectoryCommand:
    """Trajectory reference expressed in the world frame."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    yaw_rate: float = 0.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.velocity = np.asarray(self.velocity, dtype=float).reshape(3)
        self.acceleration = np.asarray(self.acceleration, dtype=float).reshape(3)


ObservationT = TypeVar("ObservationT")


class Controller(Protocol, Generic[ObservationT]):
    """Protocol that all controllers must satisfy. Returns the control input given an observation, time, and trajectory command and optionally logging values, e.g., computed pose estimates."""

    def compute_control(
        self,
        observation: ObservationT,
        t: float,
        command: TrajectoryCommand,
    ) -> Tuple[ControlInput, dict[str, Any]]:  # pragma: no cover - interface
        ...

# Backward compatibility alias
GeometricController = Controller


class CameraParamsLike(Protocol):
    width: int
    height: int
    fov: float


class CameraLike(Protocol):
    params: CameraParamsLike

    def intrinsics(self) -> np.ndarray:
        ...

class SE3PositionController:
    """Position-tracking geometric controller on SE(3)."""

    def __init__(
        self,
        params: QuadrotorParams,
        kp: np.ndarray | float = np.array([14.0, 14.0, 16.0]),      # position gain
        kd: np.ndarray | float = np.array([2.5, 2.5, 3.0]),      # velocity gain
        kR: np.ndarray | float = np.array([8.0, 8.0, 6.0]),     # rotation gain
        kOmega: np.ndarray | float = np.array([0.4, 0.4, 0.3]),  # angular velocity gain
    ):
        self.params = params
        self.kp = np.asarray(kp, dtype=float).reshape(3)
        self.kd = np.asarray(kd, dtype=float).reshape(3)
        self.kR = np.asarray(kR, dtype=float).reshape(3)
        self.kOmega = np.asarray(kOmega, dtype=float).reshape(3)

    def compute_control(
        self, observation: QuadrotorState, t: float, command: TrajectoryCommand
    ) -> Tuple[ControlInput, dict[str, Any]]:
        state = observation
        m = self.params.mass
        g = self.params.gravity
        e3 = np.array([0.0, 0.0, 1.0])

        pos_error = state.position - command.position # e_p = p^w - p^w_d (position error) 
        vel_error = state.velocity - command.velocity # e_v = v^w - v^w_d (velocity error)

        force_cmd = (
            -self.kp * pos_error
            - self.kd * vel_error
            + m * command.acceleration
            + m * g * e3
        )                                   # f_cmd = -Kp*e_p - Kd*e_v + m*\ddot{p}^w_d + m*g*e3 
        b3_des = normalize(force_cmd)       # b3_des = f_cmd / ||f_cmd||

        yaw = command.yaw
        xc = np.array([np.cos(yaw), np.sin(yaw), 0.0]) # desired x-axis in world frame
        if np.linalg.norm(np.cross(b3_des, xc)) < 1e-4:
            xc = normalize(xc + 0.1 * e3)              # avoid degeneracy when pointing up
        b2_des = normalize(np.cross(b3_des, xc))       # b2_des = (b3_des x x_c) / ||b3_des x x_c||
        b1_des = np.cross(b2_des, b3_des)              # b1_des = b2_des x b3_des
        R_des = np.column_stack((b1_des, b2_des, b3_des))

        omega_des = np.array([0.0, 0.0, command.yaw_rate])

        e_R_mat = 0.5 * (R_des.T @ state.rotation - state.rotation.T @ R_des)
        e_R = vee(e_R_mat)
        e_Omega = state.omega - state.rotation.T @ R_des @ omega_des

        thrust = float(np.dot(force_cmd, state.rotation[:, 2])) # thrust = f_cmd . b3 = f_cmd . R b3
        thrust = max(thrust, 0.0)

        torque = (
            -self.kR * e_R
            - self.kOmega * e_Omega
            + np.cross(state.omega, self.params.inertia @ state.omega)
        )

        vals = {
            "pos_error": pos_error,
            "vel_error": vel_error,
            "thrust": thrust,
            "torque": torque,
        }
        return ControlInput(thrust=thrust, torque=torque), vals


class RandomController:
    """A simple controller that outputs random thrust and torque (for testing only)."""

    def __init__(
        self,
        thrust_range: tuple[float, float] = (0.0, 15.0),
        torque_std: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.thrust_min, self.thrust_max = thrust_range
        self.torque_std = torque_std
        self.rng = rng or np.random.default_rng()

    def compute_control(
        self, observation: QuadrotorState, t: float, command: TrajectoryCommand
    ) -> Tuple[ControlInput, dict[str, Any]]:
        del observation, t, command
        thrust = float(self.rng.uniform(self.thrust_min, self.thrust_max))
        torque = self.rng.normal(scale=self.torque_std, size=3)
        return ControlInput(thrust=thrust, torque=torque), {}


@dataclass
class PBVSController:
    """Minimal PBVS stub: detect ArUco markers and return zero control."""

    def __init__(
        self,
        params: QuadrotorParams,
        aruco_dict: str = "DICT_APRILTAG_36h11",
        # DICT_APRILTAG_36h11 is a built-in AprilTag dictionary that ships with OpenCV
        # Contains parameters for detecting markers
        tag_map: list[dict[str, Any]] | None = None,
        # This is the list of known tags and their locaitons in the scene
        camera: CameraLike | None = None,
        camera_fov_deg: float = 60.0,
        image_size: tuple[int, int] = (320, 240),
        tag_size: float = 0.6,
        p_BC: np.ndarray | None = None,
        R_BC: np.ndarray | None = None,
        # These values seem to stabilize the non-a2rl drone
        # K_p: np.ndarray | float = np.array([7.0, 7.0, 8.0]),
        # K_v: np.ndarray | float = np.array([2.5, 2.5, 3.0]),
        # K_R: np.ndarray | float = np.array([4.0, 4.0, 3.0]),
        # K_w: np.ndarray | float = np.array([0.4, 0.4, 0.3]),
        # Best values for the a2rl drone
        K_p: np.ndarray | float = np.array([2.0, 2.0, 1.0]),
        K_v: np.ndarray | float = np.array([0.5, 0.5, 1.0]),
        K_R: np.ndarray | float = np.array([1.0, 1.0, 1.0]),
        K_w: np.ndarray | float = np.array([0.04, 0.04, 0.03]),
    ) -> None:
        if cv2 is None or not hasattr(cv2, "aruco"):
            raise ImportError("OpenCV with cv2.aruco is required for PBVSController.")
        # Store parameters of the quadrotor namely mass, inertia, gravity
        self.params = params
        self._dict_name = aruco_dict
        self.tag_map = tag_map or []
        # Fallback to a known dictionary if the requested one is missing
        dict_id = getattr(cv2.aruco, aruco_dict, cv2.aruco.DICT_4X4_50)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        # Set the parameters for aruco detection, thresholds for contour filtering, etc.
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.detector_params = cv2.aruco.DetectorParameters_create()
        elif hasattr(cv2.aruco, "DetectorParameters"):
            self.detector_params = cv2.aruco.DetectorParameters()
        else:  # pragma: no cover - very old OpenCV
            raise ImportError("cv2.aruco DetectorParameters API not found in this OpenCV build.")
        self.tag_lookup = self._build_tag_lookup(self.tag_map)
        self.tag_size = float(tag_size)
        self.camera = camera
        self.image_size = (int(camera.params.width), int(camera.params.height))
        self.camera_fov_deg = float(camera.params.fov)
        self._camera_matrix = np.asarray(camera.intrinsics(), dtype=float).reshape(3, 3)
        # Distortion coefficients are currently set to zero
        self._dist_coeffs = np.zeros((5, 1))
        
        self.p_BC = np.zeros(3) if p_BC is None else np.asarray(p_BC, dtype=float).reshape(3)
        self.R_BC = np.eye(3) if R_BC is None else np.asarray(R_BC, dtype=float).reshape(3, 3)
        self.R_CB = self.R_BC.T
        self.K_p = np.asarray(K_p, dtype=float).reshape(3)
        self.K_v = np.asarray(K_v, dtype=float).reshape(3)
        self.K_R = np.asarray(K_R, dtype=float).reshape(3)
        self.K_w = np.asarray(K_w, dtype=float).reshape(3)
        # previous position and time for velocity estimation
        self._prev_p_WB: np.ndarray | None = None
        self._prev_R_WB: np.ndarray | None = None
        self._prev_time: float | None = None
        self._p_WB_des: np.ndarray | None = None
        self._R_WB_des: np.ndarray | None = None

    @staticmethod
    def _compute_camera_matrix(image_size: tuple[int, int], fov_deg: float) -> np.ndarray:
        """Compute a pinhole intrinsics matrix from fov and image size."""
        width, height = image_size
        fov_rad = np.deg2rad(fov_deg)
        fy = 0.5 * height / np.tan(0.5 * fov_rad)
        # PyBullet uses vertical FOV and scales x by aspect, so fx == fy for square pixels.
        fx = fy
        cx = width / 2.0
        cy = height / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    @staticmethod
    def _build_tag_lookup(tag_map: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        """Map tag id -> pose (position, rotation matrix). 
        Uses tag_map constructed from world urdf file with april tags"""
        lookup: dict[int, dict[str, Any]] = {}
        for entry in tag_map:
            tag_id = entry.get("id")
            if tag_id is None:
                continue
            pos = np.asarray(entry.get("position", np.zeros(3)), dtype=float).reshape(3)
            rpy = np.asarray(entry.get("rpy", np.zeros(3)), dtype=float).reshape(3)
            lookup[int(tag_id)] = {"position": pos, "rotation": rpy_to_rot(rpy)}
        return lookup

    def _extract_frame(self, observation: Any) -> np.ndarray | None:
        """Pull a numpy image from various observation types."""
        if isinstance(observation, np.ndarray):
            return observation
        if hasattr(observation, "image"):
            return np.asarray(getattr(observation, "image"))
        if isinstance(observation, dict) and "image" in observation:
            return np.asarray(observation["image"])
        return None

    def compute_control(
        self, observation: Any, t: float, command: TrajectoryCommand
    ) -> Tuple[ControlInput, dict[str, Any]]:
        del command
        e3 = np.array([0.0, 0.0, 1.0])
        hover_thrust = float(self.params.mass * self.params.gravity)
        control = ControlInput(hover_thrust, np.zeros(3))

        if cv2 is None or not hasattr(cv2, "aruco"):
            return control, {
                "aruco": {"detected": False, "error": "cv2.aruco not available"},
            }

        frame = self._extract_frame(observation)
        if frame is None:
            return control, {
                "aruco": {"detected": False, "error": "no_image_in_observation"},
            }

        gray = frame
        if frame.ndim == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Update intrinsics if the incoming frame size differs from the configured size
        h, w = gray.shape[:2]
        if (w, h) != self.image_size:
            self.image_size = (w, h)
            self._camera_matrix = self._compute_camera_matrix(self.image_size, self.camera_fov_deg)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.detector_params
        )


        detection_log: dict[str, Any] = {
            "detected": ids is not None and len(ids) > 0,
            "ids": [] if ids is None else ids.flatten().tolist(),
            "num_markers": 0 if ids is None else int(len(ids)),
            "corners": [],
            "velocity_estimate": None,
        }
        if corners:
            detection_log["corners"] = [c.squeeze().tolist() for c in corners]

        if not detection_log["detected"]:
            return control, {"aruco": detection_log}

        # Estimate pose using the first detected marker that exists in the tag map
        try:
            retval = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.tag_size,
                self._camera_matrix,
                self._dist_coeffs,
            )
        except Exception as exc:  # pragma: no cover - safety
            detection_log["error"] = f"pose_estimation_failed: {exc}"
            return control, {"aruco": detection_log}

        if retval is None:
            detection_log["error"] = "pose_estimation_failed"
            return control, {"aruco": detection_log}

        rvecs, tvecs, _ = retval
        pose_logged = False
        for det_idx, tag_id in enumerate(detection_log["ids"]):
            if tag_id not in self.tag_lookup:
                continue
            rvec = rvecs[det_idx].reshape(3)
            tvec = tvecs[det_idx].reshape(3)
            # Compute transformation from Marker to Camera; conver to matrix from 
            # rotation vector rvec
            R_CM, _ = cv2.Rodrigues(rvec)
            t_CM = tvec

            # Lookup tag pose from tag dictionary 
            tag_pose = self.tag_lookup[tag_id]
            p_WM = tag_pose["position"]
            R_WM = tag_pose["rotation"]

            # Convert p_WM and R_WM to p_WC and E_WC
            R_MC = R_CM.T
            R_WC = R_WM @ R_MC
            p_WC = p_WM + R_WM @ (-R_MC @ t_CM)

            R_WB = R_WC @ self.R_CB
            p_WB = p_WC - R_WB @ self.p_BC

            # roll, pitch, yaw from rotation matrix (ZYX)
            sy = -R_WB[2, 0]
            cy = np.sqrt(max(0.0, 1 - sy * sy))
            if cy < 1e-6:
                roll = np.arctan2(-R_WB[0, 1], R_WB[1, 1])
                pitch = np.arctan2(sy, cy)
                yaw = 0.0
            else:
                roll = np.arctan2(R_WB[2, 1], R_WB[2, 2])
                pitch = np.arctan2(sy, cy)
                yaw = np.arctan2(R_WB[1, 0], R_WB[0, 0])

            detection_log["pose_estimate"] = {
                "tag_id": int(tag_id),
                "p_WC": p_WC.tolist(),
                "R_WC": R_WC.tolist(),
                "p_WB": p_WB.tolist(),
                "R_WB": R_WB.tolist(),
                "x": float(p_WB[0]),
                "y": float(p_WB[1]),
                "z": float(p_WB[2]),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
            }
            # First sample: no previous pose to differentiate; log NaNs.
            # Missing pose estimates: leave prev_* unchanged so the next valid detection uses the last pose.
            dt = None if self._prev_time is None else float(t - self._prev_time)
            dt_valid = dt if dt is not None and dt > 0.0 else None
            if self._prev_p_WB is None or dt_valid is None:
                v_WB = np.full(3, np.nan)
            else:
                v_WB = (p_WB - self._prev_p_WB) / dt_valid
            v_B = R_WB.T @ v_WB if np.all(np.isfinite(v_WB)) else np.full(3, np.nan)
            if self._prev_R_WB is None or dt_valid is None:
                R_delta = np.full((3, 3), np.nan)
                omega_B = np.full(3, np.nan)
            else:
                # Relative rotation from previous to current
                R_delta = self._prev_R_WB.T @ R_WB
                # cos(theta) = (tr(R) - 1)/2
                trace = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
                # rotation angle magnitude
                theta = float(np.arccos(trace)) 

                if theta < 1e-6:
                    # Use first-order approximation for small angles
                    phi = 0.5 * vee(R_delta - R_delta.T)
                else:
                    # phi = log(R_delta) = (theta / (2 sin theta)) (R_delta - R_delta^T)
                    phi = (theta / (2.0 * np.sin(theta))) * vee(R_delta - R_delta.T)
                omega_B = phi / dt_valid
            
            omega_speed = float(np.linalg.norm(omega_B)) if np.all(np.isfinite(omega_B)) else np.nan
            detection_log["velocity_estimate"] = {
                "v_WB": v_WB.tolist(),
                "v_B": v_B.tolist(),
                "dt": dt_valid,
            }
            detection_log["attitude_estimate"] = {
                "R_delta": R_delta.tolist(),
                "omega_B": omega_B.tolist(),
                "omega_speed": omega_speed,
                "dt": dt_valid,
            }
            if self._p_WB_des is None:
                self._p_WB_des = p_WB.copy()
                self._R_WB_des = R_WB.copy()
            e_p = p_WB - self._p_WB_des
            e_v = v_WB if np.all(np.isfinite(v_WB)) else np.zeros(3)
            if self._R_WB_des is None:
                e_R = np.zeros(3)
            else:
                e_R_mat = 0.5 * (
                    self._R_WB_des.T @ R_WB - R_WB.T @ self._R_WB_des
                )
                e_R = vee(e_R_mat)
            e_w = omega_B if np.all(np.isfinite(omega_B)) else np.zeros(3)
            force_cmd = -self.K_p * e_p - self.K_v * e_v + self.params.mass * self.params.gravity * e3
            thrust = float(np.dot(force_cmd, R_WB[:, 2]))
            thrust = max(thrust, 0.0)
            torque = -self.K_R * e_R - self.K_w * e_w
            control = ControlInput(thrust, torque)
            detection_log["control_error"] = {
                "e_p": e_p.tolist(),
                "e_v": e_v.tolist(),
                "e_R": e_R.tolist(),
                "e_w": e_w.tolist(),
            }
            self._prev_p_WB = p_WB
            self._prev_R_WB = R_WB
            self._prev_time = float(t)
            pose_logged = True
            break

        if not pose_logged:
            detection_log["error"] = "no_tag_pose_in_map"

        return control, {"aruco": detection_log}


@dataclass
class PBVSSmoothedController:
    """PBVS controller with causal EMA smoothing on derived velocities."""

    def __init__(
        self,
        params: QuadrotorParams,
        aruco_dict: str = "DICT_APRILTAG_36h11",
        tag_map: list[dict[str, Any]] | None = None,
        camera: CameraLike | None = None,
        camera_fov_deg: float = 60.0,
        image_size: tuple[int, int] = (320, 240),
        tag_size: float = 0.6,
        p_BC: np.ndarray | None = None,
        R_BC: np.ndarray | None = None,
        vel_time_constant: float = 0.2,
        omega_time_constant: float = 0.2,
        max_linear_speed: float = 5.0,
        max_angular_speed: float = 6.0,
        K_p: np.ndarray | float = np.array([2.0, 2.0, 1.0]),
        K_v: np.ndarray | float = np.array([0.5, 0.5, 1.0]),
        K_R: np.ndarray | float = np.array([1.0, 1.0, 1.0]),
        K_w: np.ndarray | float = np.array([0.04, 0.04, 0.03]),
    ) -> None:
        if cv2 is None or not hasattr(cv2, "aruco"):
            raise ImportError("OpenCV with cv2.aruco is required for PBVSSmoothedController.")
        self.params = params
        self._dict_name = aruco_dict
        self.tag_map = tag_map or []
        dict_id = getattr(cv2.aruco, aruco_dict, cv2.aruco.DICT_4X4_50)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            self.detector_params = cv2.aruco.DetectorParameters_create()
        elif hasattr(cv2.aruco, "DetectorParameters"):
            self.detector_params = cv2.aruco.DetectorParameters()
        else:  # pragma: no cover - very old OpenCV
            raise ImportError("cv2.aruco DetectorParameters API not found in this OpenCV build.")
        self.tag_lookup = PBVSController._build_tag_lookup(self.tag_map)
        self.tag_size = float(tag_size)
        self.camera = camera
        self.image_size = (int(camera.params.width), int(camera.params.height))
        self.camera_fov_deg = float(camera.params.fov)
        self._camera_matrix = np.asarray(camera.intrinsics(), dtype=float).reshape(3, 3)
        self._dist_coeffs = np.zeros((5, 1))

        self.p_BC = np.zeros(3) if p_BC is None else np.asarray(p_BC, dtype=float).reshape(3)
        self.R_BC = np.eye(3) if R_BC is None else np.asarray(R_BC, dtype=float).reshape(3, 3)
        self.R_CB = self.R_BC.T
        self.K_p = np.asarray(K_p, dtype=float).reshape(3)
        self.K_v = np.asarray(K_v, dtype=float).reshape(3)
        self.K_R = np.asarray(K_R, dtype=float).reshape(3)
        self.K_w = np.asarray(K_w, dtype=float).reshape(3)
        self.vel_time_constant = float(vel_time_constant)
        self.omega_time_constant = float(omega_time_constant)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self._prev_p_WB: np.ndarray | None = None
        self._prev_R_WB: np.ndarray | None = None
        self._prev_time: float | None = None
        self._p_WB_des: np.ndarray | None = None
        self._R_WB_des: np.ndarray | None = None
        self._v_WB_smoothed: np.ndarray | None = None
        self._omega_B_smoothed: np.ndarray | None = None

    @staticmethod
    def _compute_camera_matrix(image_size: tuple[int, int], fov_deg: float) -> np.ndarray:
        return PBVSController._compute_camera_matrix(image_size, fov_deg)

    def _extract_frame(self, observation: Any) -> np.ndarray | None:
        return PBVSController._extract_frame(self, observation)

    @staticmethod
    def _ema_alpha(dt: float, time_constant: float) -> float:
        if dt <= 0.0 or time_constant <= 0.0:
            return 1.0
        return float(1.0 - np.exp(-dt / time_constant))

    @staticmethod
    def _ema_update(prev: np.ndarray | None, measurement: np.ndarray, alpha: float) -> np.ndarray:
        if prev is None:
            return measurement.copy()
        return prev + alpha * (measurement - prev)

    @staticmethod
    def _is_outlier(vec: np.ndarray, max_norm: float) -> bool:
        if not np.all(np.isfinite(vec)):
            return True
        return float(np.linalg.norm(vec)) > max_norm

    def compute_control(
        self, observation: Any, t: float, command: TrajectoryCommand
    ) -> Tuple[ControlInput, dict[str, Any]]:
        del command
        e3 = np.array([0.0, 0.0, 1.0])
        hover_thrust = float(self.params.mass * self.params.gravity)
        control = ControlInput(hover_thrust, np.zeros(3))

        if cv2 is None or not hasattr(cv2, "aruco"):
            return control, {
                "aruco": {"detected": False, "error": "cv2.aruco not available"},
            }

        frame = self._extract_frame(observation)
        if frame is None:
            return control, {
                "aruco": {"detected": False, "error": "no_image_in_observation"},
            }

        gray = frame
        if frame.ndim == 3 and frame.shape[2] >= 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        h, w = gray.shape[:2]
        if (w, h) != self.image_size:
            self.image_size = (w, h)
            self._camera_matrix = self._compute_camera_matrix(self.image_size, self.camera_fov_deg)

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.detector_params
        )

        detection_log: dict[str, Any] = {
            "detected": ids is not None and len(ids) > 0,
            "ids": [] if ids is None else ids.flatten().tolist(),
            "num_markers": 0 if ids is None else int(len(ids)),
            "corners": [],
            "velocity_estimate": None,
        }
        if corners:
            detection_log["corners"] = [c.squeeze().tolist() for c in corners]

        if not detection_log["detected"]:
            return control, {"aruco": detection_log}

        try:
            retval = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.tag_size,
                self._camera_matrix,
                self._dist_coeffs,
            )
        except Exception as exc:  # pragma: no cover - safety
            detection_log["error"] = f"pose_estimation_failed: {exc}"
            return control, {"aruco": detection_log}

        if retval is None:
            detection_log["error"] = "pose_estimation_failed"
            return control, {"aruco": detection_log}

        rvecs, tvecs, _ = retval
        pose_logged = False
        for det_idx, tag_id in enumerate(detection_log["ids"]):
            if tag_id not in self.tag_lookup:
                continue
            rvec = rvecs[det_idx].reshape(3)
            tvec = tvecs[det_idx].reshape(3)
            R_CM, _ = cv2.Rodrigues(rvec)
            t_CM = tvec

            tag_pose = self.tag_lookup[tag_id]
            p_WM = tag_pose["position"]
            R_WM = tag_pose["rotation"]

            R_MC = R_CM.T
            R_WC = R_WM @ R_MC
            p_WC = p_WM + R_WM @ (-R_MC @ t_CM)

            R_WB = R_WC @ self.R_CB
            p_WB = p_WC - R_WB @ self.p_BC

            sy = -R_WB[2, 0]
            cy = np.sqrt(max(0.0, 1 - sy * sy))
            if cy < 1e-6:
                roll = np.arctan2(-R_WB[0, 1], R_WB[1, 1])
                pitch = np.arctan2(sy, cy)
                yaw = 0.0
            else:
                roll = np.arctan2(R_WB[2, 1], R_WB[2, 2])
                pitch = np.arctan2(sy, cy)
                yaw = np.arctan2(R_WB[1, 0], R_WB[0, 0])

            detection_log["pose_estimate"] = {
                "tag_id": int(tag_id),
                "p_WC": p_WC.tolist(),
                "R_WC": R_WC.tolist(),
                "p_WB": p_WB.tolist(),
                "R_WB": R_WB.tolist(),
                "x": float(p_WB[0]),
                "y": float(p_WB[1]),
                "z": float(p_WB[2]),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw),
            }
            dt = None if self._prev_time is None else float(t - self._prev_time)
            dt_valid = dt if dt is not None and dt > 0.0 else None
            if self._prev_p_WB is None or dt_valid is None:
                v_WB = np.full(3, np.nan)
            else:
                v_WB = (p_WB - self._prev_p_WB) / dt_valid
            v_B = R_WB.T @ v_WB if np.all(np.isfinite(v_WB)) else np.full(3, np.nan)
            if self._prev_R_WB is None or dt_valid is None:
                R_delta = np.full((3, 3), np.nan)
                omega_B = np.full(3, np.nan)
            else:
                R_delta = self._prev_R_WB.T @ R_WB
                trace = np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0)
                theta = float(np.arccos(trace))

                if theta < 1e-6:
                    phi = 0.5 * vee(R_delta - R_delta.T)
                else:
                    phi = (theta / (2.0 * np.sin(theta))) * vee(R_delta - R_delta.T)
                omega_B = phi / dt_valid

            v_outlier = self._is_outlier(v_WB, self.max_linear_speed)
            omega_outlier = self._is_outlier(omega_B, self.max_angular_speed)
            v_alpha = None
            omega_alpha = None
            if not v_outlier and dt_valid is not None:
                v_alpha = self._ema_alpha(dt_valid, self.vel_time_constant)
                self._v_WB_smoothed = self._ema_update(self._v_WB_smoothed, v_WB, v_alpha)
            if not omega_outlier and dt_valid is not None:
                omega_alpha = self._ema_alpha(dt_valid, self.omega_time_constant)
                self._omega_B_smoothed = self._ema_update(
                    self._omega_B_smoothed, omega_B, omega_alpha
                )

            v_WB_smoothed = self._v_WB_smoothed
            omega_B_smoothed = self._omega_B_smoothed
            v_B_smoothed = (
                R_WB.T @ v_WB_smoothed
                if v_WB_smoothed is not None and np.all(np.isfinite(v_WB_smoothed))
                else None
            )
            omega_speed = float(np.linalg.norm(omega_B)) if np.all(np.isfinite(omega_B)) else np.nan
            detection_log["velocity_estimate"] = {
                "v_WB": v_WB.tolist(),
                "v_B": v_B.tolist(),
                "v_WB_smoothed": None if v_WB_smoothed is None else v_WB_smoothed.tolist(),
                "v_B_smoothed": None if v_B_smoothed is None else v_B_smoothed.tolist(),
                "v_outlier": bool(v_outlier),
                "alpha": v_alpha,
                "dt": dt_valid,
            }
            detection_log["attitude_estimate"] = {
                "R_delta": R_delta.tolist(),
                "omega_B": omega_B.tolist(),
                "omega_B_smoothed": None if omega_B_smoothed is None else omega_B_smoothed.tolist(),
                "omega_speed": omega_speed,
                "omega_outlier": bool(omega_outlier),
                "alpha": omega_alpha,
                "dt": dt_valid,
            }
            if self._p_WB_des is None:
                self._p_WB_des = p_WB.copy()
                self._R_WB_des = R_WB.copy()
            e_p = p_WB - self._p_WB_des
            e_v = v_WB_smoothed if v_WB_smoothed is not None else np.zeros(3)
            if self._R_WB_des is None:
                e_R = np.zeros(3)
            else:
                e_R_mat = 0.5 * (
                    self._R_WB_des.T @ R_WB - R_WB.T @ self._R_WB_des
                )
                e_R = vee(e_R_mat)
            e_w = omega_B_smoothed if omega_B_smoothed is not None else np.zeros(3)
            force_cmd = -self.K_p * e_p - self.K_v * e_v + self.params.mass * self.params.gravity * e3
            thrust = float(np.dot(force_cmd, R_WB[:, 2]))
            thrust = max(thrust, 0.0)
            torque = -self.K_R * e_R - self.K_w * e_w
            control = ControlInput(thrust, torque)
            detection_log["control_error"] = {
                "e_p": e_p.tolist(),
                "e_v": e_v.tolist(),
                "e_R": e_R.tolist(),
                "e_w": e_w.tolist(),
            }
            self._prev_p_WB = p_WB
            self._prev_R_WB = R_WB
            self._prev_time = float(t)
            pose_logged = True
            break

        if not pose_logged:
            detection_log["error"] = "no_tag_pose_in_map"

        return control, {"aruco": detection_log}
