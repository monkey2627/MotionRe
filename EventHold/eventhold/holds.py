"""Offline candidate mining from reference pose. Never an inference input."""
import numpy as np
from scipy.spatial.transform import Rotation

REGIONS = {"trunk": [3, 6, 9], "lower_body": [1, 2, 4, 5]}


def intervals(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def mine(pose_axis_angle, valid, hz=60.0, threshold_deg_s=5.0, min_seconds=2.0):
    pose = np.asarray(pose_axis_angle).reshape(-1, 24, 3)
    finite = np.isfinite(pose).all(axis=(1, 2))
    safe = np.where(np.isfinite(pose), pose, 0)
    rot = Rotation.from_rotvec(safe.reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
    delta = np.swapaxes(rot[:-1], -1, -2) @ rot[1:]
    speed = np.zeros((len(rot), 24))
    speed[1:] = Rotation.from_matrix(delta.reshape(-1, 3, 3)).magnitude().reshape(-1, 24) * hz * 180 / np.pi
    admissible = np.asarray(valid).all(axis=1) & finite
    admissible[1:] &= finite[:-1]
    admissible[0] = False
    rows = []
    for region, ids in REGIONS.items():
        stationary = (speed[:, ids].max(axis=1) < threshold_deg_s) & admissible
        for start, end in intervals(stationary):
            duration = float((end - start) / hz)
            if duration >= min_seconds:
                rows.append({"region": region, "start_frame": int(start), "end_frame_exclusive": int(end),
                             "duration_seconds": duration, "max_speed_deg_s": float(speed[start:end, ids].max()),
                             "annotation_status": "candidate_not_reviewed", "pose_semantics": "unknown"})
    return rows
