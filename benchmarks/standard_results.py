"""Canonical result contract for DIP and AMASS drift comparisons.

All adapters write per-frame mean errors before any plotting.  This prevents
method-specific figures or checkpoint times from becoming the comparison
source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np


JOINT_GROUPS = {
    "all": tuple(range(24)),
    "lumbar": (3, 6, 9),
    "hips": (1, 2),
    "knees": (4, 5),
    "upper_arms": (16, 17),
    "forearms": (18, 19),
}


def root_aligned_translation_error(prediction, target) -> np.ndarray:
    """Per-frame Euclidean root-translation error after initial-frame alignment."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return np.linalg.norm(
        (prediction - prediction[:1]) - (target - target[:1]), axis=-1
    )


def write_standard_result(
    output_dir: Path,
    method: str,
    suite: str,
    rotation_deg: np.ndarray,
    frame_count: np.ndarray,
    fps: int,
    sensor_count: int,
    sensor_slots: Sequence[int],
    translation_m: Optional[np.ndarray] = None,
) -> Path:
    """Write canonical arrays and a concise, machine-readable summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rotation_deg = np.asarray(rotation_deg, dtype=np.float64)
    frame_count = np.asarray(frame_count, dtype=np.int64)
    if rotation_deg.ndim != 2 or rotation_deg.shape[1] != 24:
        raise ValueError("rotation_deg must have shape [T, 24]")
    if frame_count.shape != (rotation_deg.shape[0],):
        raise ValueError("frame_count must have shape [T]")
    if translation_m is None:
        translation_m = np.full(rotation_deg.shape[0], np.nan, dtype=np.float64)
    translation_m = np.asarray(translation_m, dtype=np.float64)
    if translation_m.shape != (rotation_deg.shape[0],):
        raise ValueError("translation_m must have shape [T]")

    np.savez_compressed(
        output_dir / "standard_metrics.npz",
        rotation_deg=rotation_deg,
        translation_m=translation_m,
        frame_count=frame_count,
        fps=int(fps),
        sensor_count=int(sensor_count),
        sensor_slots=np.asarray(sensor_slots, dtype=np.int64),
    )
    valid = frame_count > 0
    summary = {
        "method": method,
        "suite": suite,
        "fps": int(fps),
        "sensor_count": int(sensor_count),
        "sensor_slots": list(sensor_slots),
        "evaluated_frames": int(valid.sum()),
        "translation_available": bool(np.isfinite(translation_m[valid]).any()),
        "mean_rotation_deg": {
            name: float(np.nanmean(rotation_deg[valid][:, joints])) if valid.any() else None
            for name, joints in JOINT_GROUPS.items()
        },
        "mean_translation_m": (
            float(np.nanmean(translation_m[valid]))
            if np.isfinite(translation_m[valid]).any() else None
        ),
    }
    path = output_dir / "standard_metrics.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
