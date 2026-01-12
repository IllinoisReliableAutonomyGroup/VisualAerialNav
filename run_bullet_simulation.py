"""Run the quadrotor simulator and visualize the trajectory with PyBullet. The basic functionality is to create a simulator object with a quadrotor plant, sensor, and controller, and then run the simulation for a specified duration. Additional functionality includes visualizing the trajectory in PyBullet, initializing the quadrotor, and logging, and log playback with camera views."""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable, Optional

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

DEFAULT_INIT_OFFSET = np.array([0.15, -0.2, -0.15])

import pybullet as p


try:  # pragma: no cover
    from .math_utils import rotation_matrix_to_quaternion, yaw_rotation
    from .scene import (
        Scene,
        board_pose_from_initial_frame,
        default_camera_rotation_body,
        create_drone_visual,
        setup_scene,
        make_noisy_state_sensor,
        Camera,
        CameraParams,
        CameraObservation,
    )
    from .controller import (
        RandomController,
        SE3PositionController,
        TrajectoryCommand,
        PBVSController,
        PBVSSmoothedController,
    )
    from .plant import QuadrotorParams, QuadrotorPlant, QuadrotorState, ProcessNoise
    from .simulator import Simulator, SimulationLog
    from .run_simulation import make_circular_command, make_hover_command
    from .visualization import plot_trajectory
    from .feature_extractor import FeatureExtractor
    from .worlds.tag_map import load_tag_map
except ImportError:  # pragma: no cover
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from math_utils import rotation_matrix_to_quaternion, yaw_rotation
    from scene import (
        Scene,
        default_camera_rotation_body,
        create_drone_visual,
        setup_scene,
        make_noisy_state_sensor,
    )
    from scene import Camera, CameraParams, CameraObservation
    from controller import (
        RandomController,
        SE3PositionController,
        TrajectoryCommand,
        PBVSController,
        PBVSSmoothedController,
    )
    from plant import QuadrotorParams, QuadrotorPlant, QuadrotorState, ProcessNoise
    from simulator import Simulator, SimulationLog
    from run_simulation import make_circular_command, make_hover_command
    from visualization import plot_trajectory
    from feature_extractor import FeatureExtractor
    from worlds.tag_map import load_tag_map

def _make_camera_sensor(
    camera: Camera,
) -> Callable[[QuadrotorState, float], CameraObservation | None]:
    """Render a camera image from the drone's point of view using the shared Camera."""

    def _sensor(state: QuadrotorState, t: float) -> CameraObservation | None:
        return camera.render(state, t)

    return _sensor



def _playback_log(
    log: SimulationLog,
    sim_dt: float,
    scene: Scene,
    cam_width: int = 320,
    cam_height: int = 240,
    feature_extractor: FeatureExtractor | None = None,
) -> None:
    """Visualize the log by resetting the PyBullet body pose each step."""
    drone_id = scene.drone_body_id
    arm_bodies = scene.arm_bodies
    if drone_id is None:
        drone_id, arm_bodies = create_drone_visual()
    if not hasattr(scene, "cameras") or not scene.cameras:
        raise ValueError("Playback requires a configured Scene camera.")
    cam_fig = None
    cam_im = None
    feat_plot = None
    goal_plot = None
    goal_features: np.ndarray | None = None

    for idx, t in enumerate(log.time):
        pos = log.position[idx]
        R = log.rotation[idx]
        quat = rotation_matrix_to_quaternion(R)
        p.resetBasePositionAndOrientation(drone_id, pos.tolist(), quat.tolist())

        for arm_id, offset in arm_bodies:
            arm_pos = pos + R @ offset
            p.resetBasePositionAndOrientation(arm_id, arm_pos.tolist(), quat.tolist())

        if idx > 0:
            start = log.position[idx - 1].tolist()
            end = pos.tolist()
            p.addUserDebugLine(start, end, lineColorRGB=[0.1, 0.1, 0.9], lineWidth=2, lifeTime=5.0)

        # R is the body to world rotation; columns are body axes in world frame
        # Use recorded observation image if present in control logs; otherwise render
        rgba = None
        if idx < len(log.control_logs):
            entry = log.control_logs[idx]
            if isinstance(entry, dict) and "observation_image" in entry:
                rgba = np.asarray(entry["observation_image"])
        if rgba is None:
            render_camera = scene.cameras[0]
            playback_state = QuadrotorState(
                position=pos,
                velocity=np.zeros(3),
                rotation=R,
                omega=np.zeros(3),
            )
            obs = render_camera.render(playback_state, log.time[idx])
            if obs is None:
                continue
            rgba = obs.image

        # Overlay detected corners/tag ids if present in the control log
        overlay = rgba.copy()
        if cv2 is not None and idx < len(log.control_logs):
            entry = log.control_logs[idx]
            if isinstance(entry, dict):
                aruco_log = entry.get("aruco")
                if isinstance(aruco_log, dict):
                    corners = aruco_log.get("corners")
                    ids = aruco_log.get("ids", [])
                    if corners:
                        for c_idx, pts in enumerate(corners):
                            pts_arr = np.asarray(pts, dtype=np.int32)
                            if pts_arr.shape == (4, 2):
                                # OpenCV expects (N,1,2) int32 for polylines
                                poly = pts_arr.reshape((-1, 1, 2))
                                cv2.polylines(overlay, [poly], isClosed=True, color=(255, 255, 0), thickness=2)
                                if ids and c_idx < len(ids):
                                    tag_id = int(ids[c_idx])
                                    text = str(tag_id)
                                    pt = tuple(pts_arr[0])
                                    cv2.putText(overlay,text,pt,cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 0),2,lineType=cv2.LINE_AA,)
                        rgba = overlay
        if feature_extractor is not None:
            features = feature_extractor.extract(rgba)
            if goal_features is None and len(features):
                goal_features = features.copy()
                print(f"Recorded {len(goal_features)} goal features at t={t:.2f}s")
        else:
            features = np.zeros((0, 2))

       
        # cone_offset = np.array([0.12, 0.12, 0.02])
        # cone_pos = pos + R @ cone_offset
        # cam_basis = np.column_stack((cam_right, cam_up, cam_forward))
        # cone_quat = _rotation_matrix_to_quaternion(cam_basis)
        # p.resetBasePositionAndOrientation(cone_id, cone_pos.tolist(), cone_quat.tolist())
        if cam_fig is None:
            import matplotlib.pyplot as plt

            cam_fig, cam_ax = plt.subplots(figsize=(4.5, 3.5))
            cam_ax.set_title("Onboard camera view")
            cam_ax.axis("off")
            cam_im = cam_ax.imshow(rgba)
            feat_plot = cam_ax.scatter([], [], s=20, c="tab:red", marker="x", label="features")
            goal_plot = cam_ax.scatter([], [], s=25, c="tab:green", marker="+", label="goal")
            cam_ax.legend(loc="lower right", fontsize="xx-small")
            plt.show(block=False)
        else:
            cam_im.set_data(rgba)
            cam_im.figure.canvas.draw_idle()
            cam_im.figure.canvas.flush_events()
        if feat_plot is not None:
            if len(features):
                feat_plot.set_offsets(features)
            else:
                feat_plot.set_offsets(np.zeros((0, 2)))
        if goal_plot is not None and goal_features is not None and len(goal_features):
            goal_plot.set_offsets(goal_features)
        if cam_fig is not None:
            roll = np.rad2deg(np.arctan2(R[2, 1], R[2, 2]))
            pitch = np.rad2deg(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
            yaw = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))
            cam_fig.suptitle(
                f"t={t:.2f}s | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                f"| rpy=({roll:.1f},{pitch:.1f},{yaw:.1f})°",
                fontsize=9,
            )

        time.sleep(sim_dt)

    print("Simulation complete. Close the PyBullet window to exit.")
    # while p.isConnected():
    #     time.sleep(0.1)


def _plot_log(log: SimulationLog) -> None:
    """Plot true vs estimated pose (x, y, z, roll, pitch, yaw)."""
    import matplotlib.pyplot as plt

    t = log.time
    true_pos = log.position
    true_rot = log.rotation
    true_vel = log.velocity
    true_omega = log.omega

    # Compute true pose roll/pitch/yaw
    roll_true = np.arctan2(true_rot[:, 2, 1], true_rot[:, 2, 2])
    pitch_true = np.arctan2(-true_rot[:, 2, 0], np.sqrt(true_rot[:, 2, 1] ** 2 + true_rot[:, 2, 2] ** 2))
    yaw_true = np.arctan2(true_rot[:, 1, 0], true_rot[:, 0, 0])

    # Resample true pose at control times using nearest sim step
    t_ctrl = log.control_times
    idx_ctrl = np.searchsorted(t, t_ctrl, side="right") - 1
    idx_ctrl = np.clip(idx_ctrl, 0, len(t) - 1)
    true_at_ctrl = {
        "x": true_pos[idx_ctrl, 0],
        "y": true_pos[idx_ctrl, 1],
        "z": true_pos[idx_ctrl, 2],
        "roll": roll_true[idx_ctrl],
        "pitch": pitch_true[idx_ctrl],
        "yaw": yaw_true[idx_ctrl],
    }

    est = {"x": [], "y": [], "z": [], "roll": [], "pitch": [], "yaw": [], "t": []}
    est_vel = {"vx": [], "vy": [], "vz": [], "t": []}
    est_omega = {"wx": [], "wy": [], "wz": [], "t": []}
    for t_c, entry in zip(t_ctrl, log.control_logs):
        if not isinstance(entry, dict):
            continue
        pose = entry.get("aruco", {}).get("pose_estimate") if isinstance(entry.get("aruco"), dict) else None
        if not isinstance(pose, dict):
            continue
        est["t"].append(t_c)
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            est[key].append(float(pose.get(key, np.nan)))
        vel_est = entry.get("aruco", {}).get("velocity_estimate") if isinstance(entry.get("aruco"), dict) else None
        if isinstance(vel_est, dict) and isinstance(vel_est.get("v_WB"), list):
            v = vel_est.get("v_WB", [np.nan, np.nan, np.nan])
            est_vel["t"].append(t_c)
            est_vel["vx"].append(float(v[0]))
            est_vel["vy"].append(float(v[1]))
            est_vel["vz"].append(float(v[2]))
        att_est = entry.get("aruco", {}).get("attitude_estimate") if isinstance(entry.get("aruco"), dict) else None
        if isinstance(att_est, dict) and isinstance(att_est.get("omega_B"), list):
            w = att_est.get("omega_B", [np.nan, np.nan, np.nan])
            est_omega["t"].append(t_c)
            est_omega["wx"].append(float(w[0]))
            est_omega["wy"].append(float(w[1]))
            est_omega["wz"].append(float(w[2]))

    labels = [
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "vx",
        "vy",
        "vz",
        "wx",
        "wy",
        "wz",
        "thrust",
        "tau_x",
        "tau_y",
        "tau_z",
    ]
    true_series = {
        "x": true_at_ctrl["x"],
        "y": true_at_ctrl["y"],
        "z": true_at_ctrl["z"],
        "roll": true_at_ctrl["roll"],
        "pitch": true_at_ctrl["pitch"],
        "yaw": true_at_ctrl["yaw"],
        "vx": true_vel[idx_ctrl, 0],
        "vy": true_vel[idx_ctrl, 1],
        "vz": true_vel[idx_ctrl, 2],
        "wx": true_omega[idx_ctrl, 0],
        "wy": true_omega[idx_ctrl, 1],
        "wz": true_omega[idx_ctrl, 2],
        "thrust": log.thrust[idx_ctrl],
        "tau_x": log.torque[idx_ctrl, 0],
        "tau_y": log.torque[idx_ctrl, 1],
        "tau_z": log.torque[idx_ctrl, 2],
    }
    est_series = {
        "x": (est["t"], est["x"]),
        "y": (est["t"], est["y"]),
        "z": (est["t"], est["z"]),
        "roll": (est["t"], est["roll"]),
        "pitch": (est["t"], est["pitch"]),
        "yaw": (est["t"], est["yaw"]),
        "vx": (est_vel["t"], est_vel["vx"]),
        "vy": (est_vel["t"], est_vel["vy"]),
        "vz": (est_vel["t"], est_vel["vz"]),
        "wx": (est_omega["t"], est_omega["wx"]),
        "wy": (est_omega["t"], est_omega["wy"]),
        "wz": (est_omega["t"], est_omega["wz"]),
        "thrust": ([], []),
        "tau_x": ([], []),
        "tau_y": ([], []),
        "tau_z": ([], []),
    }

    fig, axes = plt.subplots(6, 3, figsize=(14, 14), sharex=True)
    flat_axes = axes.flatten()
    for idx, lbl in enumerate(labels):
        ax = flat_axes[idx]
        ax.plot(t_ctrl, true_series[lbl], label="true", color="tab:blue")
        est_t, est_vals = est_series[lbl]
        if est_t:
            ax.plot(est_t, est_vals, ".", label="estimate", color="tab:orange")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()
    for ax in flat_axes[len(labels):]:
        ax.axis("off")
    fig.supxlabel("time [s]")
    fig.tight_layout()
    plt.show()


def main() -> None:

    # Hand input command line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.2)
    parser.add_argument("--angular-rate", type=float, default=0.4)
    parser.add_argument("--controller-rate", type=float, default=200.0)
    parser.add_argument("--sim-dt", type=float, default=0.001)
    parser.add_argument("--noise", action="store_true", help="Use a noisy state sensor.")
    parser.add_argument(
        "--process-noise-accel",
        type=float,
        default=None,
        help="Process accel noise std (m/s^2).",
    )
    parser.add_argument(
        "--process-noise-omega",
        type=float,
        default=None,
        help="Process omega noise std (rad/s^2).",
    )
    parser.add_argument("--trajectory", choices=["circle", "hover"], default="circle")
    parser.add_argument("--hover-x", type=float, default=0.0)
    parser.add_argument("--hover-y", type=float, default=0.0)
    parser.add_argument("--hover-height", type=float, default=3.0)
    parser.add_argument(
        "--hover-yaw",
        type=float,
        default=None,
        help="Initial hover yaw in radians (default: plant default yaw)",
    )
    parser.add_argument("--playback-stride", type=int, default=1, help="Skip this many frames when replaying a log (>=1).")
    parser.add_argument("--max-features", "--max_features", type=int, default=200, help="Maximum features to extract per frame.")
    parser.add_argument(
        "--feature-min-distance",
        "--feature_min_distance",
        type=float,
        default=5.0,
        help="Minimum pixel spacing between detected features.",
    )
    parser.add_argument(
        "--controller",
        choices=["se3", "random", "pbvs", "smoothed-pbvs"],
        default="se3",
        help="Controller to use",
    )
    parser.add_argument(
        "--a2rl",
        action="store_true",
        help="Use A2RL drone URDF instead of basic drone visual",
    )
    parser.add_argument(
        "--aruco_scene",
        action="store_true",
        help="Load ArUco/AprilTag scene with generated tag",
    )
    parser.add_argument(
        "--basic_drone",
        action="store_true",
        help="Use basic drone visual (default if --a2rl not specified)",
    )
    args = parser.parse_args()

    # Determine scene configuration based on arguments
    use_a2rl = args.a2rl
    env_urdf_rel = "worlds/marker_field_apriltags.urdf"
    env_urdf_path = os.path.join(os.path.dirname(__file__), env_urdf_rel)
    
    # Create the plant model with appropriate physical parameters
    params = QuadrotorParams()
    if use_a2rl:
        # Use A2RL drone physical parameters
        from scene import A2RL_MASS, A2RL_INERTIA
        params.mass = A2RL_MASS
        params.inertia = np.diag(A2RL_INERTIA)
        params.inv_inertia = np.linalg.inv(params.inertia)
        print(f"Using A2RL drone physical parameters: mass={A2RL_MASS} kg, inertia={A2RL_INERTIA}")
    else:
        print(f"Using default drone physical parameters: mass={params.mass} kg")
    
    # Build the initial state once and pass it into the plant
    hover_target = np.array([args.hover_x, args.hover_y, args.hover_height])
    # Default initial state
    # init_state = QuadrotorState()
    init_state = QuadrotorState(rotation=yaw_rotation(np.deg2rad(120.0)))
    if args.trajectory == "hover":
        init_state.position = hover_target + DEFAULT_INIT_OFFSET
        if args.hover_yaw is not None:
            init_state.rotation = yaw_rotation(args.hover_yaw)
    else:
        circle_start = np.array([args.radius, 0.0, args.height])
        init_state.position = circle_start + DEFAULT_INIT_OFFSET
    init_state.velocity = np.zeros(3)
    # Alternative initial velocity
    # init_state.velocity = np.array([0.2, 0.4, 0.0])
    init_state.omega = np.zeros(3)

    process_noise = None
    if args.process_noise_accel is not None or args.process_noise_omega is not None:
        default_noise = ProcessNoise()
        accel_std = (
            default_noise.accel_std
            if args.process_noise_accel is None
            else args.process_noise_accel
        )
        omega_std = (
            default_noise.omega_std
            if args.process_noise_omega is None
            else args.process_noise_omega
        )
        process_noise = ProcessNoise(accel_std=accel_std, omega_std=omega_std)
    plant = QuadrotorPlant(params, noise=process_noise, initial_state=init_state)
    # estimator.reset(plant.state)
    # This feature extractor is going to go away and be part of the controller 
    feature_extractor = FeatureExtractor(
        max_features=args.max_features,
        min_distance=args.feature_min_distance,
    )
    # Setup the pybullet scene
    initial_pose = plant.state
    
    # Determine scene configuration based on arguments (use_a2rl already set above)
    use_aruco = args.aruco_scene
    use_basic_drone = args.basic_drone or (not use_a2rl)  # Default to basic if not a2rl
        
    # Create scene with appropriate configuration
    scene = Scene(
        env_urdf=env_urdf_rel,  # Use default plane.urdf from pybullet_data
        use_a2rl_drone=use_a2rl,
    )

    # Load tag poses from the same URDF used for the scene
    tag_map = load_tag_map(env_urdf_path)

    # Camera parameters consistent with the renderer.
    cam_width, cam_height, cam_fov = 320, 240, 60.0
    cam_params = CameraParams(
        width=cam_width,
        height=cam_height,
        fov=cam_fov,
        update_rate_hz=args.controller_rate,
    )
    # Render camera frame: x forward, y right, z up (used by PyBullet view matrix).
    R_BC_render = default_camera_rotation_body()
    p_BC = scene.camera_offset if hasattr(scene, "camera_offset") else np.zeros(3)
    camera = Camera(params=cam_params, R_BC=R_BC_render, p_BC=p_BC)
    scene.add_camera(camera)
    # Convert render camera axes -> OpenCV camera axes for pose estimation.
    R_BC_cv = camera.R_BC_opencv()

    # Create the sensor function
    sensor_fn = make_noisy_state_sensor() if args.noise else None

    # Create the controller (choose via args.controller)
    controller_choice = getattr(args, "controller", "se3")
    if controller_choice == "pbvs":
        controller = PBVSController(
            params,
            tag_map=tag_map,
            p_BC=p_BC,
            R_BC=R_BC_cv,
            camera=camera,
            tag_size=getattr(scene, "tag_size", 0.6),
        )
        sensor_fn = _make_camera_sensor(camera)
    elif controller_choice == "smoothed-pbvs":
        controller = PBVSSmoothedController(
            params,
            tag_map=tag_map,
            p_BC=p_BC,
            R_BC=R_BC_cv,
            camera=camera,
            tag_size=getattr(scene, "tag_size", 0.6),
        )
        sensor_fn = _make_camera_sensor(camera)
    elif controller_choice == "random":
        controller = RandomController()
    else:
        controller = SE3PositionController(params)
    simulator = Simulator(
        plant,
        controller,
        sim_dt=args.sim_dt,
        controller_rate_hz=args.controller_rate,
        sensor_rate_hz=args.controller_rate,
        sensor_fn=sensor_fn,
        scene=scene,
    )
    # Create the command function
    # This could also be part of a planner class later on
    if args.trajectory == "hover":
        command_fn = make_hover_command(position=hover_target, yaw=args.hover_yaw)
    else:
        command_fn = make_circular_command(
            radius=args.radius, height=args.height, angular_rate=args.angular_rate
        )
    # Run the simulation
    log = simulator.run(command_fn, duration=args.duration)
    print("Simulation finished.")
    # Plot and display
    # print(log)
    _plot_log(log)
    # print([(e["aruco"]["detected"], e["aruco"]["ids"]) for e in log.control_logs if isinstance(e, dict) and "aruco" in e])
    simulator.playback(
        log,
        #        feature_extractor=feature_extractor,
        stride=max(1, args.playback_stride),
    )


if __name__ == "__main__":
    main()
