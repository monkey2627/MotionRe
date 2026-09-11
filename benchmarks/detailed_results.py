"""Per-sequence benchmark result contract and robust aggregate statistics.

``standard_metrics`` remains the compact, frame-aligned comparison format.
This module preserves the information it intentionally omits: sequence identity,
action label, variable-length error arrays, and recoverable failures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


MANDATORY_ACTIONS = ("lying", "crawling", "transitions", "interaction", "sports")
JOINT_GROUPS = {
    "all": tuple(range(24)),
    "lumbar": (3, 6, 9),
    "knees": (4, 5),
    "upper_arms": (16, 17),
}


def safe_sequence_name(index: int, source: str, configuration: Optional[str] = None) -> str:
    """Produce a stable, filesystem-safe sequence result filename."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("_")[:80]
    prefix = "sequence_{:05d}_{}".format(index, stem or "unknown")
    return "{}_{}".format(prefix, configuration) if configuration else prefix


def translation_metrics(translation_m: Optional[np.ndarray], fps: int) -> Dict[str, Optional[float]]:
    """Compute root-translation RMSE, endpoint error, and least-squares drift rate."""
    if translation_m is None:
        return {"translation_rmse_m": None, "translation_endpoint_m": None,
                "translation_drift_m_per_s": None}
    error = np.asarray(translation_m, dtype=np.float64)
    valid = np.isfinite(error)
    if not valid.any():
        return {"translation_rmse_m": None, "translation_endpoint_m": None,
                "translation_drift_m_per_s": None}
    values = error[valid]
    times = np.flatnonzero(valid).astype(np.float64) / float(fps)
    drift = 0.0
    if len(values) >= 2 and np.ptp(times) > 0:
        drift = float(np.polyfit(times, values, 1)[0])
    return {
        "translation_rmse_m": float(np.sqrt(np.mean(values ** 2))),
        "translation_endpoint_m": float(values[-1]),
        "translation_drift_m_per_s": drift,
    }


def foot_sliding_metrics(
    predicted_joints: Optional[np.ndarray],
    target_joints: Optional[np.ndarray],
    fps: int,
    foot_joints: Sequence[int] = (10, 11),
    contact_height_m: float = 0.06,
    contact_speed_m_per_s: float = 0.20,
) -> Dict[str, Optional[float]]:
    """Estimate predicted horizontal foot speed during GT foot-contact frames.

    This is a kinematic proxy, not a force/contact-label metric.  It is only
    written when both predicted and target joint trajectories are available.
    """
    if predicted_joints is None or target_joints is None:
        return {"contact_foot_sliding_m_per_s": None, "contact_frame_count": None}
    pred = np.asarray(predicted_joints, dtype=np.float64)
    target = np.asarray(target_joints, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[1] <= max(foot_joints):
        raise ValueError("joint trajectories must have matching [T, J, 3] shape")
    if len(pred) < 2:
        return {"contact_foot_sliding_m_per_s": None, "contact_frame_count": 0}
    gt_feet = target[:, foot_joints]
    gt_speed = np.linalg.norm(np.diff(gt_feet, axis=0), axis=-1) * float(fps)
    ground = np.nanmin(target[..., 1], axis=1, keepdims=True)
    near_ground = gt_feet[:-1, :, 1] - ground[:-1] <= contact_height_m
    contact = near_ground & (gt_speed <= contact_speed_m_per_s)
    pred_speed = np.linalg.norm(np.diff(pred[:, foot_joints, :][:, :, (0, 2)], axis=0), axis=-1) * float(fps)
    if not contact.any():
        return {"contact_foot_sliding_m_per_s": None, "contact_frame_count": 0}
    return {"contact_foot_sliding_m_per_s": float(pred_speed[contact].mean()),
            "contact_frame_count": int(contact.sum())}


def _rotation_summary(rotation_deg: np.ndarray) -> Dict[str, float]:
    return {name: float(np.nanmean(rotation_deg[:, joints])) for name, joints in JOINT_GROUPS.items()}


def write_sequence_result(
    output_dir: Path,
    index: int,
    source: str,
    action: str,
    rotation_deg: Optional[np.ndarray],
    translation_m: Optional[np.ndarray],
    fps: int,
    failure_reason: Optional[str] = None,
    predicted_joints: Optional[np.ndarray] = None,
    target_joints: Optional[np.ndarray] = None,
    configuration: Optional[str] = None,
) -> Dict[str, object]:
    """Persist one sequence's raw arrays and return its JSON-serialisable row."""
    output_dir = Path(output_dir)
    arrays_dir = output_dir / "detailed_metrics"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    row: Dict[str, object] = {
        "sequence_index": int(index), "source": str(source), "action": str(action or "other"),
        "status": "failed" if failure_reason else "passed", "failure_reason": failure_reason,
        "valid_frames": 0, "rotation": None,
        "translation_rmse_m": None, "translation_endpoint_m": None,
        "translation_drift_m_per_s": None, "contact_foot_sliding_m_per_s": None,
        "contact_frame_count": None, "array_file": None,
        "configuration": configuration,
    }
    if failure_reason:
        return row
    if rotation_deg is None:
        raise ValueError("rotation_deg is required for a passed sequence")
    rotation = np.asarray(rotation_deg, dtype=np.float64)
    if rotation.ndim != 2 or rotation.shape[1] != 24:
        raise ValueError("rotation_deg must have shape [T, 24]")
    translation = None if translation_m is None else np.asarray(translation_m, dtype=np.float64)
    if translation is not None and translation.shape != (len(rotation),):
        raise ValueError("translation_m must have shape [T]")
    array_name = safe_sequence_name(index, source, configuration) + ".npz"
    payload = {"rotation_deg": rotation, "fps": np.asarray(fps, dtype=np.int32)}
    if translation is not None:
        payload["translation_m"] = translation
    np.savez_compressed(arrays_dir / array_name, **payload)
    row.update({"valid_frames": int(len(rotation)), "rotation": _rotation_summary(rotation),
                "array_file": str(Path("detailed_metrics") / array_name)})
    row["terminal_rotation_all_deg"] = float(np.nanmean(rotation[-1]))
    row.update(translation_metrics(translation, fps))
    row.update(foot_sliding_metrics(predicted_joints, target_joints, fps))
    return row


def write_detailed_index(output_dir: Path, method: str, suite: str, fps: int,
                         records: Iterable[Dict[str, object]]) -> Path:
    """Write a JSONL index, preserving successful rows and sequence failures."""
    output_dir = Path(output_dir)
    records = list(records)
    path = output_dir / "detailed_metrics.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
    metadata = {
        "method": method, "suite": suite, "fps": int(fps),
        "record_count": len(records), "failed_sequences": sum(r["status"] != "passed" for r in records),
        "mandatory_actions": list(MANDATORY_ACTIONS), "index": path.name,
    }
    (output_dir / "detailed_metrics_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def action_summary(records: Iterable[Dict[str, object]], mandatory_actions: Sequence[str] = MANDATORY_ACTIONS) -> List[Dict[str, object]]:
    """Compute means/P90/failure rates by action, including missing required actions."""
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("action") or "other"), []).append(record)
    for action in mandatory_actions:
        grouped.setdefault(action, [])
    rows = []
    for action in sorted(grouped):
        group = grouped[action]
        successful = [r for r in group if r.get("status") == "passed"]
        row: Dict[str, object] = {"action": action, "sequences": len(group),
                                  "successful_sequences": len(successful),
                                  "failure_rate": None if not group else 1.0 - len(successful) / len(group)}
        for metric, key in (("rotation_all_deg", ("rotation", "all")),
                            ("rotation_lumbar_deg", ("rotation", "lumbar")),
                            ("translation_rmse_m", None),
                            ("translation_drift_m_per_s", None),
                            ("translation_endpoint_m", None),
                            ("terminal_rotation_all_deg", None),
                            ("contact_foot_sliding_m_per_s", None)):
            values = []
            for item in successful:
                value = item[key[0]].get(key[1]) if key else item.get(metric)
                if value is not None and np.isfinite(value):
                    values.append(float(value))
            row[metric + "_mean"] = float(np.mean(values)) if values else None
            row[metric + "_p90"] = float(np.percentile(values, 90)) if values else None
        rows.append(row)
    return rows
