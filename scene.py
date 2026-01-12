"""Scene, camera classes, and drone visualization helpers for the PyBullet quadrotor sim. Updates the drone visual pose to match the plant state and renders camera images or generates observations."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

try:
    import cv2
    import cv2.aruco as aruco
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:  # pragma: no cover
    import pybullet as p
    import pybullet_data
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pybullet is required for visualization utilities.") from exc

try:  # pragma: no cover
    from .math_utils import normalize, rotation_matrix_to_quaternion
    from .plant import QuadrotorState
except ImportError:  # pragma: no cover
    from math_utils import normalize, rotation_matrix_to_quaternion
    from plant import QuadrotorState

#Aruco Scene Parameters

# AprilTag generation parameters
ARUCO_DICT_TYPE = "DICT_APRILTAG_25h9"  
ARUCO_TAG_ID = 0  
ARUCO_TAG_RESOLUTION = 512  # High resolution for tag image
ARUCO_BORDER_PERCENT = 0.15  # White border
ARUCO_TAG_SIZE = 0.5  # Physical size (m)
ARUCO_TAG_POSITION = [2.0, 0.0, 1.0]  # Location in world frame [x, y, z]
ARUCO_TAG_ORIENTATION = [0.0, 1.5708, 0.0]  # Orientation [roll, pitch, yaw] in radians

ARUCO_TEXTURE_REL_PATH = "worlds/april/tag25h9_generated.png"
A2RL_URDF_FILENAME = "assets/a2r_drone/urdf/a2r_drone.urdf"  

# Simulation start parameters
SIM_START_POSITION = [0.0, 0.0, 0.1]  
SIM_START_ORIENTATION = [0.0, 0.0, 0.0]  

CAMERA_OFFSET_BASIC = [0.0, 0.0, 0.06]  
CAMERA_OFFSET_A2RL = [0.05, 0.0, 0.08]  

# Camera basis in the body frame for rendering: x forward, y right, z up.
# This basis is used to build the PyBullet view direction (look-at + up).
def default_camera_basis_body() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam_forward_body = normalize(np.array([1.0, 0.1, 0.0]))
    cam_up_body = np.array([0.0, 0.0, 1.0])
    cam_right_body = normalize(np.cross(cam_forward_body, cam_up_body))
    cam_up_body = normalize(np.cross(cam_right_body, cam_forward_body))
    return cam_forward_body, cam_right_body, cam_up_body


def default_camera_rotation_body() -> np.ndarray:
    """Return camera->body rotation for the default render camera frame."""
    cam_forward_body, cam_right_body, cam_up_body = default_camera_basis_body()
    return np.column_stack((cam_forward_body, cam_right_body, cam_up_body))


# Render camera frame (x forward, y right, z up) -> OpenCV camera frame (x right, y down, z forward).
CAMERA_RENDER_TO_CV = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)

# Physical parameters for different drone types
# A2RL drone physical parameters (from a2r_drone.urdf)
A2RL_MASS = 0.870  # kg
A2RL_INERTIA = [0.008951, 0.003102, 0.01161]  # [Ixx, Iyy, Izz] in kg·m²

# Basic/default drone physical parameters
BASIC_MASS = 20.0  # kg (default from QuadrotorParams)
BASIC_INERTIA = [0.02, 0.02, 0.04]  # [Ixx, Iyy, Izz] in kg·m² (default from QuadrotorParams)

ARM_LENGTH = 0.3
ARM_HALF = ARM_LENGTH / 2.0
ARM_THICKNESS = 0.015
ARM_PARTS = (
    {
        "offset": np.array([ARM_HALF, 0.0, 0.0]),
        "half_extents": [ARM_HALF, ARM_THICKNESS, ARM_THICKNESS],
        "color": [0.85, 0.2, 0.2, 1.0],
    },
    {
        "offset": np.array([-ARM_HALF, 0.0, 0.0]),
        "half_extents": [ARM_HALF, ARM_THICKNESS, ARM_THICKNESS],
        "color": [0.85, 0.2, 0.2, 1.0],
    },
    {
        "offset": np.array([0.0, ARM_HALF, 0.0]),
        "half_extents": [ARM_THICKNESS, ARM_HALF, ARM_THICKNESS],
        "color": [0.2, 0.2, 0.85, 1.0],
    },
    {
        "offset": np.array([0.0, -ARM_HALF, 0.0]),
        "half_extents": [ARM_THICKNESS, ARM_HALF, ARM_THICKNESS],
        "color": [0.2, 0.2, 0.85, 1.0],
    },
)


def compute_camera_frame(
    position: np.ndarray, 
    rotation: np.ndarray,
    camera_offset: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the camera pose and basis vectors for a given drone pose.

    """
    if camera_offset is None:
        camera_offset = np.array(CAMERA_OFFSET_BASIC)
    else:
        camera_offset = np.asarray(camera_offset, dtype=float).reshape(3)
    
    # Camera position in world frame: position + rotation @ offset
    cam_position = position + rotation @ camera_offset
    
    # Camera looks forward with a slight lateral component (fixed in body frame).
    cam_forward_body, cam_right_body, cam_up_body = default_camera_basis_body()
    cam_forward = rotation @ cam_forward_body
    cam_right = rotation @ cam_right_body
    cam_up = rotation @ cam_up_body
    cam_target = cam_position + 2 * cam_forward
    return cam_position, cam_target, cam_forward, cam_up, cam_right


def compute_camera_intrinsics(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Compute OpenCV-style intrinsics from PyBullet's vertical FOV."""
    fov_rad = np.deg2rad(fov_deg)
    fy = 0.5 * height / np.tan(0.5 * fov_rad)
    # PyBullet scales x by aspect, so fx == fy in pixel units for square pixels.
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)


def setup_scene(
    board_pose: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Create the PyBullet scene with the ground board and optional AprilTag."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
        cameraDistance=3.0, cameraYaw=45.0, cameraPitch=-35.0, cameraTargetPosition=[0, 0, 0]
    )
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    script_dir = os.path.dirname(os.path.abspath(__file__))


    if tag_texture:
        half_extents = [0.5 * tag_size, 0.005, 0.5 * tag_size]
        vis_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
        )
        col_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents)
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[0.0, 0.0, 0.0],
        )
        try:
            tex_id = p.loadTexture(tag_texture)
            p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)
        except Exception as exc:  # pragma: no cover - texture optional
            print(f"Warning: failed to load tag texture '{tag_texture}': {exc}")


def create_drone_visual() -> tuple[int, list[tuple[int, np.ndarray]], int]:
    """Create a small body plus box visuals for the four arms."""
    base_radius = 0.04
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=base_radius)
    visual = p.createVisualShape(
        p.GEOM_SPHERE, radius=base_radius, rgbaColor=[0.1, 0.1, 0.1, 1.0]
    )
    body_id = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[0.0, 0.0, 0.0],
    )
    arm_bodies: list[tuple[int, np.ndarray]] = []
    for part in ARM_PARTS:
        arm_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=part["half_extents"],
            rgbaColor=part["color"],
        )
        arm_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=arm_visual,
            basePosition=[0.0, 0.0, 0.0],
        )
        arm_bodies.append((arm_id, part["offset"]))
    return body_id, arm_bodies  # , camera_id


def make_noisy_state_sensor(
    position_std: float = 0.05,
    velocity_std: float = 0.05,
    omega_std: float = 0.02,
    rng: np.random.Generator | None = None,
) -> Callable[[QuadrotorState, float], QuadrotorState]:
    """Return a sensor function that perturbs the true state with Gaussian noise."""
    _rng = rng or np.random.default_rng()

    def sensor(state: QuadrotorState, t: float) -> QuadrotorState:
        del t
        noisy = state.copy()
        if position_std > 0.0:
            noisy.position = noisy.position + _rng.normal(scale=position_std, size=3)
        if velocity_std > 0.0:
            noisy.velocity = noisy.velocity + _rng.normal(scale=velocity_std, size=3)
        if omega_std > 0.0:
            noisy.omega = noisy.omega + _rng.normal(scale=omega_std, size=3)
        return noisy

    return sensor


def load_aruco_scene(
    script_dir: str | None = None,
    tag_size: float | None = None,
    tag_position: List[float] | None = None,
    tag_orientation: List[float] | None = None,
    texture_path: str | None = None,
) -> None:
    """
    Load an AprilTag/ArUco scene into PyBullet.
    
    """
    if script_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    
    tag_size = tag_size or ARUCO_TAG_SIZE
    tag_position = tag_position or ARUCO_TAG_POSITION
    tag_orientation = tag_orientation or ARUCO_TAG_ORIENTATION
    texture_path = texture_path or ARUCO_TEXTURE_REL_PATH
    
    # Create worlds/april directory
    worlds_april_dir = os.path.join(script_dir, "worlds", "april")
    os.makedirs(worlds_april_dir, exist_ok=True)
    
    # Generate high-res tag
    if not CV2_AVAILABLE:
        raise ImportError("cv2.aruco is required for ArUco scene. Install opencv-contrib-python.")
    
    # Map dictionary type string to cv2 constant
    dict_map = {
        "DICT_APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11,
    }
    aruco_dict = aruco.getPredefinedDictionary(dict_map.get(ARUCO_DICT_TYPE, aruco.DICT_APRILTAG_25h9))
    tag_img = aruco.generateImageMarker(aruco_dict, ARUCO_TAG_ID, ARUCO_TAG_RESOLUTION)
    
    # Add white border
    border = int(ARUCO_TAG_RESOLUTION * ARUCO_BORDER_PERCENT)
    tag_img_bordered = cv2.copyMakeBorder(
        tag_img, border, border, border, border, cv2.BORDER_CONSTANT, value=255
    )
    
    
    if len(tag_img_bordered.shape) == 2:
        tag_img_bordered = cv2.cvtColor(tag_img_bordered, cv2.COLOR_GRAY2BGR)
    
    # Save texture
    if os.path.isabs(texture_path):
        tex_path = texture_path
    else:
        tex_path = os.path.join(script_dir, texture_path)
    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    cv2.imwrite(tex_path, tag_img_bordered)
    
    # To prevent mirroring issue
    obj_path = os.path.join(worlds_april_dir, "plane_tag.obj")
    with open(obj_path, 'w') as f:
        # Vertices
        f.write("v -0.5 -0.5 0\n")
        f.write("v 0.5 -0.5 0\n")
        f.write("v 0.5 0.5 0\n")
        f.write("v -0.5 0.5 0\n")
        # UVs: Flip U (1-u) to fix mirroring
        f.write("vt 1 0\n")  # 1
        f.write("vt 0 0\n")  # 2
        f.write("vt 0 1\n")  # 3
        f.write("vt 1 1\n")  # 4
        # Normals
        f.write("vn 0 0 1\n")
        f.write("vn 0 0 -1\n")
        # Front face
        f.write("f 1/1/1 2/2/1 3/3/1 4/4/1\n")
        # Back face (reversed winding)
        f.write("f 4/4/2 3/3/2 2/2/2 1/1/2\n")
    
    # Convert orientation to quaternion
    tag_orn = p.getQuaternionFromEuler(tag_orientation)
    
    # Create visual shape using MESH for proper UVs
    vis_shape = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[tag_size, tag_size, 1],
        rgbaColor=[1, 1, 1, 1],
        specularColor=[0, 0, 0],  # Matte finish
        visualFramePosition=[0, 0, 0]
    )
    
    # Create Collision Shape (Box is fine for collision)
    col_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[tag_size/2, tag_size/2, 0.001]
    )
    
    # Create multi-body
    body_id = p.createMultiBody(
        baseMass=0,  # Static object
        baseVisualShapeIndex=vis_shape,
        baseCollisionShapeIndex=col_shape,
        basePosition=tag_position,
        baseOrientation=tag_orn
    )
    
    # Load and apply texture
    tex_id = p.loadTexture(tex_path)
    p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)
    print(f"Loaded AprilTag from {tex_path} at {tag_position} using OBJ mesh")


@dataclass
class CameraParams:
    width: int = 320
    height: int = 240
    # Vertical FOV in degrees (PyBullet's computeProjectionMatrixFOV uses vertical FOV).
    fov: float = 60.0
    near: float = 0.01
    far: float = 10.0
    update_rate_hz: float = 60.0


@dataclass
class CameraObservation:
    """Single synchronized camera frame plus pose information."""

    time: float
    position: np.ndarray
    rotation: np.ndarray
    image: np.ndarray

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        self.rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        self.image = np.asarray(self.image)


@dataclass
class Camera:
    """Pinhole camera mounted on the drone."""

    params: CameraParams = field(default_factory=CameraParams)
    # camera->body rotation using the render frame (x forward, y right, z up).
    R_BC: np.ndarray = field(default_factory=default_camera_rotation_body)
    p_BC: np.ndarray = field(default_factory=lambda: np.zeros(3))  # camera origin in body frame
    renderer: int = p.ER_BULLET_HARDWARE_OPENGL
    _next_capture_time: float = 0.0
    _last_observation: CameraObservation | None = None

    def intrinsics(self) -> np.ndarray:
        """Return OpenCV-style intrinsics consistent with PyBullet's vertical FOV."""
        return compute_camera_intrinsics(self.params.width, self.params.height, self.params.fov)

    def R_BC_opencv(self) -> np.ndarray:
        """Return camera->body rotation in the OpenCV camera frame."""
        # Convert from render camera axes (x forward, y right, z up) to OpenCV (x right, y down, z forward).
        return self.R_BC @ CAMERA_RENDER_TO_CV.T

    def _camera_pose_world(self, state: QuadrotorState) -> tuple[np.ndarray, np.ndarray]:
        """Return camera position/orientation in world frame given body pose."""
        R_WB = state.rotation
        p_WB = state.position
        R_WC = R_WB @ self.R_BC
        p_WC = p_WB + R_WB @ self.p_BC
        return p_WC, R_WC

    def render(self, state: QuadrotorState, t: float) -> CameraObservation | None:
        """Render an image if it is time for the next capture; otherwise return the last one."""
        if t + 1e-12 < self._next_capture_time:
            return self._last_observation

        p_WC, R_WC = self._camera_pose_world(state)
        cam_forward = R_WC[:, 0]
        cam_up = R_WC[:, 2]
        cam_target = p_WC + cam_forward

        # PyBullet builds an OpenGL view matrix from a look-at target and up vector.
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=p_WC.tolist(),
            cameraTargetPosition=cam_target.tolist(),
            cameraUpVector=cam_up.tolist(),
        )
        aspect = self.params.width / self.params.height
        # PyBullet expects vertical FOV here; aspect scales x.
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.params.fov,
            aspect=aspect,
            nearVal=self.params.near,
            farVal=self.params.far,
        )
        img = p.getCameraImage(
            self.params.width,
            self.params.height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=self.renderer,
        )
        if img is None or img[2] is None:
            return None

        rgba = np.reshape(img[2], (self.params.height, self.params.width, 4))[:, :, :3]
        obs = CameraObservation(time=t, position=p_WC.copy(), rotation=R_WC.copy(), image=rgba)
        self._last_observation = obs
        self._next_capture_time = t + 1.0 / max(1e-6, self.params.update_rate_hz)
        return obs


class Scene:
    """Manage environment, drone visuals, and sensors."""

    def __init__(
        self,
        env_urdf: str | None = None,
        env_mesh: str | None = None,
        env_mesh_scale: Optional[List[float]] = None,
        drone_urdf: str | None = None,
        board_pose: tuple[np.ndarray, np.ndarray] | None = None,
        tag_texture: str | None = None,
        tag_size: float = 0.6,
        state_sensor_fn: Optional[Callable[[QuadrotorState, float], QuadrotorState]] = None,
        use_a2rl_drone: bool = False,
        use_aruco_scene: bool = False,
        drone_start_position: List[float] | None = None,
        drone_start_orientation: List[float] | None = None,
    ) -> None:
        self.env_urdf = env_urdf
        self.env_mesh = env_mesh
        self.env_mesh_scale = env_mesh_scale or [1.0, 1.0, 1.0]
        self.drone_urdf = drone_urdf
        self.board_pose = board_pose
        self.tag_texture = tag_texture
        self.tag_size = tag_size
        self.use_a2rl_drone = use_a2rl_drone
        self.use_aruco_scene = use_aruco_scene
        self.drone_start_position = drone_start_position or SIM_START_POSITION
        self.drone_start_orientation = drone_start_orientation or SIM_START_ORIENTATION
        self.camera_offset = np.array(CAMERA_OFFSET_A2RL if use_a2rl_drone else CAMERA_OFFSET_BASIC)
        self.drone_body_id: Optional[int] = None
        self.arm_bodies: list[tuple[int, np.ndarray]] = []
        self.cameras: List[Camera] = []
        self.state_sensor_fn = state_sensor_fn
        self._connect_and_load()

    def _maybe_apply_marker_textures(self, env_id: int, env_path: str, textures_dir: str) -> None:
        """If using the apriltag marker field URDF, manually apply textures to links."""
        basename = os.path.basename(env_path)
        if basename != "marker_field_apriltags.urdf":
            return
        # Apply to base if its name encodes a tag id
        body_info = p.getBodyInfo(env_id)
        base_name = body_info[0].decode("utf-8")
        self._apply_texture_if_tag(env_id, -1, base_name, textures_dir)
        for link_idx in range(p.getNumJoints(env_id)):
            info = p.getJointInfo(env_id, link_idx)
            link_name = info[12].decode("utf-8")
            self._apply_texture_if_tag(env_id, link_idx, link_name, textures_dir)

    @staticmethod
    def _apply_texture_if_tag(env_id: int, link_idx: int, link_name: str, textures_dir: str) -> None:
        # Expect names like floor_center_tag0 -> extract trailing integer
        suffix = "".join(ch for ch in link_name if ch.isdigit())
        if not suffix:
            return
        candidates = [
            os.path.join(textures_dir, "tags_36h11", f"tag_{suffix}.png"),
            os.path.join(textures_dir, f"apriltag_{suffix}.png"),
        ]
        tex_path = next((c for c in candidates if os.path.exists(c)), None)
        if tex_path is None:
            return
        try:
            tex_id = p.loadTexture(tex_path)
            p.changeVisualShape(
                env_id,
                link_idx,
                textureUniqueId=tex_id,
                rgbaColor=[1, 1, 1, 1],
                specularColor=[0, 0, 0],
            )
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"Warning: failed to apply texture {tex_path} to {link_name}: {exc}")

    def _connect_and_load(self) -> None:
        if not p.isConnected():
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0, cameraYaw=45.0, cameraPitch=-35.0, cameraTargetPosition=[0, 0, 0]
        )

        # Load environment
        if self.env_mesh:
            vis = p.createVisualShape(
                shapeType=p.GEOM_MESH, fileName=self.env_mesh, meshScale=self.env_mesh_scale
            )
            p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis)
        else:
            env_path = self.env_urdf or "plane.urdf"
            # If the environment is a URDF/SDF with relative assets (textures/meshes),
            # temporarily switch to the file's directory so relative paths resolve.
            env_dir = os.path.dirname(env_path)
            textures_dir = os.path.join(os.path.dirname(__file__), "textures")
            cwd = os.getcwd()
            try:
                if env_dir:
                    os.chdir(env_dir)
                    p.setAdditionalSearchPath(env_dir)
                if os.path.isdir(textures_dir):
                    p.setAdditionalSearchPath(textures_dir)
                if env_path.endswith(".sdf"):
                    p.loadSDF(os.path.basename(env_path))
                else:
                    env_id = p.loadURDF(os.path.basename(env_path))  # environment/base
                    self._maybe_apply_marker_textures(env_id, env_path, textures_dir)
            finally:
                os.chdir(cwd)

        # Load ArUco scene if requested
        if self.use_aruco_scene:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            load_aruco_scene(
                script_dir=script_dir,
                tag_size=self.tag_size,
                tag_position=ARUCO_TAG_POSITION,
                tag_orientation=ARUCO_TAG_ORIENTATION,
            )

        # Load drone
        if self.use_a2rl_drone:
            # Load A2RL drone URDF
            script_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_path = os.path.join(script_dir, A2RL_URDF_FILENAME)
            if not os.path.exists(urdf_path):
                raise FileNotFoundError(
                    f"Could not find A2RL URDF at: {urdf_path}\n"
                    f"Please ensure the file exists or update A2RL_URDF_FILENAME in scene.py"
                )
            start_orn = p.getQuaternionFromEuler(self.drone_start_orientation)
            self.drone_body_id = p.loadURDF(urdf_path, self.drone_start_position, start_orn)
            print(f"Loaded A2RL drone from {urdf_path}")
        elif self.drone_urdf:
            # Load custom drone URDF
            start_orn = p.getQuaternionFromEuler(self.drone_start_orientation)
            self.drone_body_id = p.loadURDF(self.drone_urdf, self.drone_start_position, start_orn)
        else:
            # Use basic drone visual
            self.drone_body_id, self.arm_bodies = create_drone_visual()

        # Optional tag/board (legacy support for basic tag loading)
        if self.board_pose:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            board_path = os.path.join(script_dir, "worlds", "basic.urdf")
            if os.path.exists(board_path):
                board_pos, board_quat = self.board_pose
                p.loadURDF(
                    board_path,
                    basePosition=board_pos.tolist(),
                    baseOrientation=board_quat.tolist(),
                    useFixedBase=True,
                )
            else:
                print(f"Warning: board URDF not found at {board_path}, skipping board loading")
        elif self.board_pose is None and self.env_urdf is None and not self.use_aruco_scene:
            # by default, do nothing extra
            pass

        
        if self.tag_texture and not self.use_aruco_scene:
            half_extents = [0.5 * self.tag_size, 0.005, 0.5 * self.tag_size]
            vis_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=half_extents,
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
            )
            col_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents)
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=[0.0, 0.0, 0.0],
            )
            try:
                tex_id = p.loadTexture(self.tag_texture)
                p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)
            except Exception as exc:  # pragma: no cover
                print(f"Warning: failed to load tag texture '{self.tag_texture}': {exc}")

    def add_camera(self, camera: Camera) -> None:
        self.cameras.append(camera)
    
    def get_camera_frame(self, position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get camera frame using the scene's configured camera offset.
        """
        return compute_camera_frame(position, rotation, self.camera_offset)

    def update_drone_pose(self, state: QuadrotorState) -> None:
        """Update the pose of the visual bodies to match the plant state."""
        quat = rotation_matrix_to_quaternion(state.rotation)
        if self.drone_body_id is not None:
            p.resetBasePositionAndOrientation(
                self.drone_body_id, state.position.tolist(), quat.tolist()
            )
        for arm_id, offset in self.arm_bodies:
            arm_pos = state.position + state.rotation @ offset
            p.resetBasePositionAndOrientation(arm_id, arm_pos.tolist(), quat.tolist())

    def observe(self, state: QuadrotorState, t: float) -> List[CameraObservation]:
        """Render all cameras and return the observations (if any updated at this t)."""
        observations: List[CameraObservation] = []
        for cam in self.cameras:
            obs = cam.render(state, t)
            if obs is not None:
                observations.append(obs)
        return observations

    def state_sensor(self, state: QuadrotorState, t: float) -> QuadrotorState:
        """Return state observation (optionally noisy via provided sensor fn)."""
        if self.state_sensor_fn is None:
            return state
        return self.state_sensor_fn(state, t)
