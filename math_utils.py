"""Small collection of SO(3) utilities used by the quadrotor simulation."""

from __future__ import annotations

import numpy as np


def hat(vec: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a 3-vector."""
    v = np.asarray(vec).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def vee(mat: np.ndarray) -> np.ndarray:
    """Return the vector that corresponds to a skew-symmetric matrix."""
    return np.array([mat[2, 1], mat[0, 2], mat[1, 0]])


def normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize a vector with numerical guard."""
    v = np.asarray(vec).reshape(3)
    norm = np.linalg.norm(v)
    if norm < eps:
        return v.copy()
    return v / norm


def project_to_so3(mat: np.ndarray) -> np.ndarray:
    """Project a matrix onto SO(3) via SVD."""
    u, _, vh = np.linalg.svd(mat)
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vh
    return r


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a body-to-world rotation matrix into a (x, y, z, w) quaternion."""
    q = np.empty(4)
    trace = np.trace(R)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s
    return q


def yaw_rotation(angle: float) -> np.ndarray:
    """Return a rotation matrix for a yaw about the z-axis."""
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rpy_to_rot(rpy: np.ndarray) -> np.ndarray:
    """Convert roll, pitch, yaw to rotation matrix (ZYX convention)."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def rot_to_rpy(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to roll, pitch, yaw (ZYX convention)."""
    sy = -R[2, 0]
    cy = np.sqrt(max(0.0, 1 - sy * sy))
    if cy < 1e-6:
        roll = np.arctan2(-R[0, 1], R[1, 1])
        pitch = np.arctan2(sy, cy)
        yaw = 0.0
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(sy, cy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])
