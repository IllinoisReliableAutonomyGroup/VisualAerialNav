"""Visualization helpers for quadrotor simulations."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover
    from .simulator import SimulationLog
except ImportError:  # pragma: no cover
    from simulator import SimulationLog


def plot_trajectory(log: SimulationLog, axes=None):
    """Plot the xyz position traces."""
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        fig.suptitle("Quadrotor position")

    labels = ["x", "y", "z"]
    for idx in range(3):
        axes[idx].plot(log.time, log.position[:, idx])
        axes[idx].set_ylabel(f"{labels[idx]} [m]")
        axes[idx].grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    return axes


def animate_log(
    log: SimulationLog,
    skip: int = 2,
    arm_length: float = 0.3,
    interval_ms: int = 5,
    repeat: bool = True,
    save_path: str | None = None,
):
    """Animate the quadrotor states using matplotlib."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if skip < 1:
        skip = 1

    frames = list(range(0, len(log.time), skip))
    if frames[-1] != len(log.time) - 1:
        frames.append(len(log.time) - 1)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    path, = ax.plot([], [], [], color="gray", lw=1.0, alpha=0.6)

    colors = ["tab:red", "tab:red", "tab:blue", "tab:blue"]
    segments_body = np.array(
        [
            [[0.0, 0.0, 0.0], [arm_length, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [-arm_length, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, arm_length, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, -arm_length, 0.0]],
        ]
    )
    body_lines = [
        ax.plot([], [], [], color=c, lw=2.5, solid_capstyle="round")[0]
        for c in colors
    ]

    pos = log.position
    limits = np.array([pos.min(axis=0), pos.max(axis=0)])
    span = limits[1] - limits[0]
    margin = 2.0 * np.maximum(0.5, 0.2 * span)
    center = np.mean(limits, axis=0)
    for idx, axis in enumerate("xyz"):
        getattr(ax, f"set_{axis}lim")(center[idx] - margin[idx], center[idx] + margin[idx])

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Quadrotor flight")

    def update(frame_idx: int):
        idx = frames[frame_idx]
        R = log.rotation[idx]
        p = log.position[idx]
        history = pos[: idx + 1]
        path.set_data(history[:, 0], history[:, 1])
        path.set_3d_properties(history[:, 2])

        for seg_body, line in zip(segments_body, body_lines):
            seg_world = (R @ seg_body.T).T + p
            line.set_data(seg_world[:, 0], seg_world[:, 1])
            line.set_3d_properties(seg_world[:, 2])
        return body_lines + [path]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=repeat,
    )

    if save_path:
        ani.save(save_path)
    else:
        plt.show()
    return ani
