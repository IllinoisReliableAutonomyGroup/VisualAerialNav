"""Simple simulator that glues the plant, sensors, and controller together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, TypeVar

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

try:  # pragma: no cover
    from .controller import Controller, TrajectoryCommand
    from .plant import ControlInput, QuadrotorPlant, QuadrotorState
    from .scene import Scene, create_drone_visual
    from .math_utils import rotation_matrix_to_quaternion
    import pybullet as p
except ImportError:  # pragma: no cover
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from controller import Controller, TrajectoryCommand
    from plant import ControlInput, QuadrotorPlant, QuadrotorState
    from scene import Scene, create_drone_visual
    from math_utils import rotation_matrix_to_quaternion
    import pybullet as p


ObservationT = TypeVar("ObservationT")


@dataclass
class SimulationLog:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    omega: np.ndarray
    thrust: np.ndarray
    torque: np.ndarray
    control_logs: List[Any] 
    control_times: np.ndarray


class Simulator:
    """Discrete-time simulator with independent controller, sensor, and plant rates.
    plant: QuadrotorPlant instance
    controller: Controller instance (consumes observations from the sensor_fn)
    sim_dt: Simulation time step
    controller_rate_hz: Controller update rate in Hz (if None, controller is updated every sim_dt)
    sensor_rate_hz: Sensor/observer update rate in Hz (if None, defaults to controller rate; if both are None, falls back to sim_dt)
    sensor_fn: callable to generate observations directly from the plant state (defaults to identity)."""

    def __init__(
        self,
        plant: QuadrotorPlant,
        controller: Controller[ObservationT],
        sim_dt: float = 0.08,
        controller_rate_hz: Optional[float] = 200.0,
        sensor_rate_hz: Optional[float] = None,
        sensor_fn: Optional[Callable[[QuadrotorState, float], ObservationT]] = None,
        scene: Optional[Scene] = None,
    ):
        self.plant = plant
        self.controller = controller
        self.sim_dt = sim_dt
        self.ctrl_period = sim_dt if controller_rate_hz is None else 1.0 / controller_rate_hz
        default_sensor_period = self.ctrl_period if controller_rate_hz is not None else sim_dt
        self.sensor_period = (
            default_sensor_period if sensor_rate_hz is None else 1.0 / sensor_rate_hz
        )
        self.scene = scene
        self.sensor_fn = sensor_fn

    def run(
        self,
        command_fn: Callable[[float], TrajectoryCommand],
        duration: float,
    ) -> SimulationLog:
        """Simulate for the requested duration."""
        times: List[float] = [0.0]
        states: List[QuadrotorState] = [self.plant.state.copy()]
        thrust_hist: List[float] = []
        torque_hist: List[np.ndarray] = []
        control_logs: List[Any] = []
        control_times: List[float] = []

        t = 0.0
        next_control_time = 0.0
        next_sensor_time = 0.0
        # default sensor returns the state (or uses scene's state sensor if provided)
        if self.sensor_fn is not None:
            sensor = self.sensor_fn
        elif self.scene is not None:
            sensor = self.scene.state_sensor  # type: ignore[assignment]
        else:
            sensor = lambda s, _t: s  # type: ignore[assignment]
        observation = sensor(self.plant.state, t)
        command = command_fn(t)
        control, ctrl_log = self.controller.compute_control(observation, t, command)
        obs_img = None
        if hasattr(observation, "image"):
            obs_img = np.asarray(getattr(observation, "image"))
        elif isinstance(observation, np.ndarray) and observation.ndim >= 2:
            obs_img = observation
        if obs_img is not None:
            ctrl_log = dict(ctrl_log)
            ctrl_log["observation_image"] = obs_img
        control_logs.append(ctrl_log)
        control_times.append(t)

        while t < duration - 1e-10:
            state = self.plant.step(control, self.sim_dt)
            t += self.sim_dt
            if t >= next_sensor_time - 1e-12:
                observation = sensor(state, t)
                next_sensor_time += self.sensor_period

            if t >= next_control_time - 1e-12:
                command = command_fn(t)
                control, ctrl_log = self.controller.compute_control(observation, t, command)
                obs_img = None
                if hasattr(observation, "image"):
                    obs_img = np.asarray(getattr(observation, "image"))
                elif isinstance(observation, np.ndarray) and observation.ndim >= 2:
                    obs_img = observation
                if obs_img is not None:
                    ctrl_log = dict(ctrl_log)
                    ctrl_log["observation_image"] = obs_img
                control_logs.append(ctrl_log)
                control_times.append(t)
                next_control_time += self.ctrl_period

            times.append(t)
            states.append(state)
            thrust_hist.append(control.thrust)
            torque_hist.append(control.torque.copy())

        position = np.vstack([s.position for s in states])
        velocity = np.vstack([s.velocity for s in states])
        rotation = np.stack([s.rotation for s in states], axis=0)
        omega = np.vstack([s.omega for s in states])
        thrust = np.array(thrust_hist + [thrust_hist[-1] if thrust_hist else 0.0])
        torque = np.vstack(
            torque_hist + [torque_hist[-1] if torque_hist else np.zeros(3)]
        )

        return SimulationLog(
            time=np.array(times),
            position=position,
            velocity=velocity,
            rotation=rotation,
            omega=omega,
            thrust=thrust,
            torque=torque,
            control_logs=control_logs,
            control_times=np.array(control_times),
        )

    def playback(
        self,
        log: SimulationLog,
        feature_extractor=None,
        cam_width: int = 320,
        cam_height: int = 240,
        stride: int = 1,
    ) -> None:
        """Replay a SimulationLog with interactive controls using the attached Scene."""
        if self.scene is None:
            print("No scene available for playback.")
            return
        stride = max(1, int(stride))


        scene = self.scene
        drone_id = scene.drone_body_id
        arm_bodies = scene.arm_bodies
        if drone_id is None:
            drone_id, arm_bodies = create_drone_visual()
        if not hasattr(scene, "cameras") or not scene.cameras:
            raise ValueError("Playback requires a configured Scene camera.")

        goal_features = None

        def render_frame(idx: int) -> None:
            """Render a single frame of the log at the requested index."""
            nonlocal goal_features
            if idx < 0 or idx >= len(log.time):
                return
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
                    return
                rgba = obs.image

            features = np.zeros((0, 2))
            if feature_extractor is not None and rgba is not None:
                features = feature_extractor.extract(rgba)
                if goal_features is None and len(features):
                    goal_features = features.copy()

            cam_im.set_data(rgba)
            cam_im.figure.canvas.draw_idle()
            if feat_plot is not None:
                feat_plot.set_offsets(features if len(features) else np.zeros((0, 2)))
            if goal_plot is not None and goal_features is not None and len(goal_features):
                goal_plot.set_offsets(goal_features)
            roll = np.rad2deg(np.arctan2(R[2, 1], R[2, 2]))
            pitch = np.rad2deg(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
            yaw = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))
            cam_fig.suptitle(
                f"t={log.time[idx]:.2f}s | pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                f"| rpy=({roll:.1f},{pitch:.1f},{yaw:.1f})°",
                fontsize=9,
            )

        cam_fig, cam_ax = plt.subplots(figsize=(5, 4))
        plt.subplots_adjust(bottom=0.25)
        cam_ax.set_title("Onboard camera view")
        cam_ax.axis("off")
        cam_im = cam_ax.imshow(np.zeros((cam_height, cam_width, 3), dtype=np.uint8))
        feat_plot = cam_ax.scatter([], [], s=20, c="tab:red", marker="x", label="features")
        goal_plot = cam_ax.scatter([], [], s=25, c="tab:green", marker="+", label="goal")
        cam_ax.legend(loc="lower right", fontsize="xx-small")

        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        slider = Slider(ax_slider, "Frame", 0, len(log.time) - 1, valinit=0, valstep=stride)

        ax_play = plt.axes([0.15, 0.02, 0.1, 0.04])
        ax_pause = plt.axes([0.27, 0.02, 0.1, 0.04])
        ax_rewind = plt.axes([0.39, 0.02, 0.1, 0.04])
        btn_play = Button(ax_play, "Play")
        btn_pause = Button(ax_pause, "Pause")
        btn_rewind = Button(ax_rewind, "Rewind")

        playing = {"state": False}

        def on_slider(val):
            render_frame(int(val))

        def on_play(event):
            playing["state"] = True

        def on_pause(event):
            playing["state"] = False

        def on_rewind(event):
            playing["state"] = False
            slider.set_val(0)

        slider.on_changed(on_slider)
        btn_play.on_clicked(on_play)
        btn_pause.on_clicked(on_pause)
        btn_rewind.on_clicked(on_rewind)

        timer = cam_fig.canvas.new_timer(interval=int(self.sim_dt * 1000))

        def advance_frame():
            if not playing["state"]:
                return
            next_val = slider.val + stride
            if next_val >= len(log.time):
                playing["state"] = False
                return
            slider.set_val(next_val)

        timer.add_callback(advance_frame)
        timer.start()

        render_frame(0)
        plt.show()
