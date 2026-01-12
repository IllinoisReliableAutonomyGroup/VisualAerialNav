"""Example entrypoint for the quadrotor simulator using only matplotlib-based visualization."""

from __future__ import annotations

import argparse
import numpy as np

try:  # pragma: no cover
    from .controller import SE3PositionController, TrajectoryCommand
    from .plant import QuadrotorParams, QuadrotorPlant, QuadrotorState
    from .simulator import Simulator
    from .visualization import animate_log, plot_trajectory
except ImportError:  # pragma: no cover
    from controller import SE3PositionController, TrajectoryCommand
    from plant import QuadrotorParams, QuadrotorPlant, QuadrotorState
    from simulator import Simulator
    from visualization import animate_log, plot_trajectory


def make_circular_command(
    radius: float, height: float, angular_rate: float
):
    """Create a callable that returns a circular trajectory command."""

    def command(t: float) -> TrajectoryCommand:
        x = radius * np.cos(angular_rate * t)
        y = radius * np.sin(angular_rate * t)
        z = height
        vx = -radius * angular_rate * np.sin(angular_rate * t)
        vy = radius * angular_rate * np.cos(angular_rate * t)
        vz = 0.0
        ax = -radius * angular_rate**2 * np.cos(angular_rate * t)
        ay = -radius * angular_rate**2 * np.sin(angular_rate * t)
        az = 0.0
        yaw = np.arctan2(vy, vx)
        return TrajectoryCommand(
            position=np.array([x, y, z]),
            velocity=np.array([vx, vy, vz]),
            acceleration=np.array([ax, ay, az]),
            yaw=yaw,
            yaw_rate=0.0,
        )

    return command


def make_hover_command(
    position: np.ndarray | None = None,
    yaw: float = 0.0,
) -> callable:
    """Return a constant hover command around the requested position."""
    hover_position = (
        np.array([0.0, 0.0, 3.0]) if position is None else np.asarray(position, dtype=float)
    )

    def command(t: float) -> TrajectoryCommand:
        del t
        return TrajectoryCommand(
            position=hover_position,
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            yaw=yaw,
            yaw_rate=0.0,
        )

    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--height", type=float, default=1.2)
    parser.add_argument("--angular-rate", type=float, default=0.4)
    parser.add_argument("--controller-rate", type=float, default=200.0)
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument(
        "--trajectory",
        choices=["circle", "hover"],
        default="circle",
        help="Select the high-level planner.",
    )
    parser.add_argument("--hover-x", type=float, default=0.0)
    parser.add_argument("--hover-y", type=float, default=0.0)
    parser.add_argument("--hover-height", type=float, default=3.0)
    parser.add_argument("--hover-yaw", type=float, default=0.0)
    args = parser.parse_args()

    params = QuadrotorParams()
    hover_target = np.array([args.hover_x, args.hover_y, args.hover_height])
    initial_state = None
    if args.trajectory == "hover":
        initial_state = QuadrotorState()
        initial_state.position = hover_target + np.array([0.1, -0.1, -0.2])
        initial_state.velocity = np.zeros(3)
        initial_state.omega = np.zeros(3)
    plant = QuadrotorPlant(params, initial_state=initial_state)
    controller = SE3PositionController(params)
    simulator = Simulator(
        plant,
        controller,
        sim_dt=args.sim_dt,
        controller_rate_hz=args.controller_rate,
        sensor_rate_hz=args.controller_rate,
    )
    if args.trajectory == "hover":
        command_fn = make_hover_command(position=hover_target, yaw=args.hover_yaw)
    else:
        command_fn = make_circular_command(
            radius=args.radius, height=args.height, angular_rate=args.angular_rate
        )
    log = simulator.run(command_fn, duration=args.duration)

    print(
        f"Simulation finished. Final position: {log.position[-1]}, final velocity: {log.velocity[-1]}"
    )
    try:
        plot_trajectory(log)
        animate_log(log)
    except ImportError as exc:
        print("matplotlib is required for visualization:", exc)


if __name__ == "__main__":
    main()
