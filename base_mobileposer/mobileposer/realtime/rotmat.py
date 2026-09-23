"""Small rotation-matrix <-> quaternion helpers shared by the realtime bridge.

mobileposer.articulate.math only goes quaternion(wxyz) -> matrix, not the
reverse, and the wire format used here (SolarXR, and the outgoing stream to
Unity) is quaternion(x, y, z, w). Kept separate from calibration.py/infer.py
so both can share it without a circular import.
"""
from __future__ import annotations

import numpy as np

# Same basis-change matrix mobileposer.fit_ours_smpl uses (as
# _UNITY_TO_SMPL_BASIS) to convert between Unity's left-handed and SMPL's
# right-handed convention: for a rotation matrix, conjugate by this fixed
# mirror; it is its own inverse, so the same expression converts either way.
UNITY_SMPL_BASIS = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)


def smpl_matrix_to_unity_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """Single SMPL-convention (right-handed) local rotation matrix -> a
    quaternion (x, y, z, w) directly usable as a Unity Quaternion."""
    unity_matrix = UNITY_SMPL_BASIS @ m @ UNITY_SMPL_BASIS
    return matrix_to_quat_xyzw(unity_matrix)


def batch_smpl_matrix_to_unity_quat_xyzw(matrices: np.ndarray) -> np.ndarray:
    """(N, 3, 3) SMPL-convention local rotations -> (N, 4) Unity quaternions."""
    return np.stack([smpl_matrix_to_unity_quat_xyzw(matrices[i]) for i in range(matrices.shape[0])])


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    """Single 3x3 proper rotation matrix -> unit quaternion (x, y, z, w).

    Standard branching (Shepperd's method) construction: numerically stable
    for every rotation, including 180-degree ones, unlike a single closed-form
    sqrt/copysign formula.
    """
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def batch_matrix_to_quat_xyzw(matrices: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 4). N is small (24 SMPL joints/frame) -- a Python loop
    over matrix_to_quat_xyzw is plenty fast at 30 Hz."""
    return np.stack([matrix_to_quat_xyzw(matrices[i]) for i in range(matrices.shape[0])])
