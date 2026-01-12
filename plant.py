"""Quadrotor plant dynamics. Defines the rigid-body model and RK4 integration. Provides a QuadrotorPlant class for instantiating a quadrotor model with specified parameters. The plant takes in thrust and torque control inputs and advances the state using 4th-order Runge-Kutta (RK4) integration represented in the world frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:  # pragma: no cover
    from .math_utils import hat, project_to_so3, yaw_rotation
except ImportError:  # pragma: no cover
    from math_utils import hat, project_to_so3, yaw_rotation

DEFAULT_YAW_DEG = 30.0
DEFAULT_YAW_RAD = float(np.deg2rad(DEFAULT_YAW_DEG))


@dataclass
class QuadrotorParams:
    """Physical parameters for the quadrotor model."""

    mass: float = 20.0
    inertia: np.ndarray = field(
        default_factory=lambda: np.diag([0.02, 0.02, 0.04])
    )
    gravity: float = 9.81

    def __post_init__(self) -> None:
        self.inertia = np.asarray(self.inertia, dtype=float).reshape(3, 3)
        self.inv_inertia = np.linalg.inv(self.inertia)


@dataclass
class QuadrotorState:
    """State of the quadrotor expressed in the world frame."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: yaw_rotation(DEFAULT_YAW_RAD))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "QuadrotorState":
        '''Create a deep copy of the state.'''
        return QuadrotorState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            rotation=self.rotation.copy(),
            omega=self.omega.copy(),
        )


@dataclass
class QuadrotorStateDerivative:
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    omega: np.ndarray


@dataclass
class ControlInput:
    """Control input comprised of thrust along body z and body torques."""

    thrust: float
    torque: np.ndarray

    def __post_init__(self) -> None:
        self.torque = np.asarray(self.torque, dtype=float).reshape(3)


@dataclass
class ProcessNoise:
    """Optional process-noise standard deviations."""

    accel_std: float = 0  # applies to translational acceleration (m/s^2)
    omega_std: float = 0  # applies to angular acceleration (rad/s^2)


class QuadrotorPlant:
    """Quadrotor rigid-body dynamics with RK4 integration."""

    def __init__(
        self,
        params: QuadrotorParams,
        noise: Optional[ProcessNoise] = None,
        initial_state: Optional[QuadrotorState] = None,
    ):
        self.params = params
        self.state = initial_state.copy() if initial_state is not None else QuadrotorState()
        self.noise = noise or ProcessNoise()

    def reset(self, state: Optional[QuadrotorState] = None) -> None:
        """Reset the plant to a provided state or to hover at the origin."""
        self.state = state.copy() if state is not None else QuadrotorState()

    def _state_derivative(
        self, state: QuadrotorState, control: ControlInput
    ) -> QuadrotorStateDerivative:
        """Compute the time derivative of the state."""
        m = self.params.mass
        g = self.params.gravity
        e3 = np.array([0.0, 0.0, 1.0])

        b3 = state.rotation[:, 2]   # body z-axis in world frame
        pos_dot = state.velocity
        vel_dot = -g * e3 + (control.thrust / m) * b3   # acceleration

        rot_dot = state.rotation @ hat(state.omega)  # rotation matrix derivative
        omega_dot = (
            self.params.inv_inertia
            @ (
                control.torque
                - np.cross(state.omega, self.params.inertia @ state.omega)
            )
        )

        return QuadrotorStateDerivative(
            position=pos_dot,
            velocity=vel_dot,
            rotation=rot_dot,
            omega=omega_dot,
        )

    @staticmethod
    def _combine(
        state: QuadrotorState,
        deriv: QuadrotorStateDerivative,
        scale: float,   # scaling factor advancing the state = state + scale * deriv
    ) -> QuadrotorState:
        return QuadrotorState(
            position=state.position + scale * deriv.position,
            velocity=state.velocity + scale * deriv.velocity,
            rotation=state.rotation + scale * deriv.rotation,
            omega=state.omega + scale * deriv.omega,
        )

    def step(self, control: ControlInput, dt: float) -> QuadrotorState:
        """Advance the dynamics by dt seconds using RK4."""
        s0 = self.state
        k1 = self._state_derivative(s0, control)
        k2 = self._state_derivative(self._combine(s0, k1, 0.5 * dt), control)
        k3 = self._state_derivative(self._combine(s0, k2, 0.5 * dt), control)
        k4 = self._state_derivative(self._combine(s0, k3, dt), control)

        avg = QuadrotorStateDerivative(
            position=(k1.position + 2 * k2.position + 2 * k3.position + k4.position)
            / 6.0,
            velocity=(k1.velocity + 2 * k2.velocity + 2 * k3.velocity + k4.velocity)
            / 6.0,
            rotation=(k1.rotation + 2 * k2.rotation + 2 * k3.rotation + k4.rotation)
            / 6.0,
            omega=(k1.omega + 2 * k2.omega + 2 * k3.omega + k4.omega) / 6.0,
        )
        accel_noise = (
            np.random.normal(scale=self.noise.accel_std, size=3)
            if self.noise.accel_std > 0
            else 0.0
        )
        omega_noise = (
            np.random.normal(scale=self.noise.omega_std, size=3)
            if self.noise.omega_std > 0
            else 0.0
        )

        next_state = QuadrotorState(
            position=s0.position + dt * avg.position,
            velocity=s0.velocity + dt * (avg.velocity + accel_noise),
            rotation=project_to_so3(s0.rotation + dt * avg.rotation),
            omega=s0.omega + dt * (avg.omega + omega_noise),
        )
        self.state = next_state
        return self.state.copy()


def _random_control(
    params: QuadrotorParams, thrust_std: float, torque_std: float
) -> ControlInput:
    """Draw a random hover-ish control input."""
    thrust = np.random.normal(loc=params.mass * params.gravity, scale=thrust_std)
    torque = np.random.normal(scale=torque_std, size=3)
    return ControlInput(thrust=float(max(thrust, 0.0)), torque=torque)


def _animate_path(
    positions: np.ndarray,
    rotations: np.ndarray,
    interval_ms: int = 40,
    arm_length: float = 0.35,
) -> None:
    """Animate the random flight path with a quadrotor cross."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Random control flight")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    min_vals = positions.min(axis=0)
    max_vals = positions.max(axis=0)
    center = 0.5 * (min_vals + max_vals)
    span = float(np.max(max_vals - min_vals))
    global_margin = max(2.0, 0.6 * span)
    ax.set_xlim(center[0] - global_margin, center[0] + global_margin)
    ax.set_ylim(center[1] - global_margin, center[1] + global_margin)
    ax.set_zlim(center[2] - global_margin, center[2] + global_margin)

    path_line, = ax.plot([], [], [], lw=2, color="tab:blue", alpha=0.7)
    colors = ["tab:red", "tab:red", "tab:green", "tab:green"]
    segments_body = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        ]
    ) * arm_length
    body_lines = [
        ax.plot([], [], [], color=c, lw=3, solid_capstyle="round")[0]
        for c in colors
    ]

    local_margin = max(1.0, 4.0 * arm_length)

    def update(frame: int):
        path = positions[: frame + 1]
        R = rotations[frame]
        p = positions[frame]

        path_line.set_data(path[:, 0], path[:, 1])
        path_line.set_3d_properties(path[:, 2])

        for seg_body, line in zip(segments_body, body_lines):
            seg_world = (R @ seg_body.T).T + p
            line.set_data(seg_world[:, 0], seg_world[:, 1])
            line.set_3d_properties(seg_world[:, 2])

        ax.set_xlim(p[0] - local_margin, p[0] + local_margin)
        ax.set_ylim(p[1] - local_margin, p[1] + local_margin)
        ax.set_zlim(p[2] - local_margin, p[2] + local_margin)
        return [path_line, *body_lines]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    plt.show()
    return ani


def main(
    duration: float = 8.0,
    dt: float = 0.01,
    animate: bool = True,
    accel_noise: float = 1.0,
    omega_noise: float = 1.0,
) -> None:
    """Simulate free flight with random controls, plot traces, and animate."""
    params = QuadrotorParams()
    plant = QuadrotorPlant(
        params, noise=ProcessNoise(accel_std=accel_noise, omega_std=omega_noise)
    )

    times = [0.0]
    positions = [plant.state.position.copy()]
    velocities = [plant.state.velocity.copy()]
    rotations = [plant.state.rotation.copy()]

    thrust_hist = []
    torque_hist = []

    steps = int(duration / dt)
    for k in range(steps):
        control = _random_control(
            params,
            thrust_std=0.0100 * params.mass * params.gravity,
            torque_std=0.0500,
        )
        state = plant.step(control, dt)
        times.append((k + 1) * dt)
        positions.append(state.position.copy())
        velocities.append(state.velocity.copy())
        rotations.append(state.rotation.copy())
        thrust_hist.append(control.thrust)
        torque_hist.append(control.torque.copy())

    import matplotlib.pyplot as plt

    times = np.array(times)
    positions = np.vstack(positions)
    velocities = np.vstack(velocities)
    rotations = np.stack(rotations, axis=0)

    if animate:
        _ = _animate_path(positions, rotations)

    fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(times, positions[:, 0], label="x")
    axes[0].plot(times, positions[:, 1], label="y")
    axes[0].plot(times, positions[:, 2], label="z")
    axes[0].set_ylabel("position [m]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, velocities[:, 0], label="vx")
    axes[1].plot(times, velocities[:, 1], label="vy")
    axes[1].plot(times, velocities[:, 2], label="vz")
    axes[1].set_ylabel("velocity [m/s]")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times[1:], thrust_hist, label="thrust")
    axes[2].set_ylabel("thrust [N]")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
