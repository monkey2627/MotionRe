"""Reproducible G0 inventory and training/validation-only hold mining."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .dip import DIP_HZ, DIP_INDEX, load_original, measurements, split_for_subject
from .holds import mine
from .records import FIVE


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("reports/g0"))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = args.root.resolve()
    raw = root / "base_mobileposer/data/raw/DIP_IMU"
    rows, candidates, errors = [], [], []
    for path in sorted(raw.glob("s_*/*.pkl")):
        try:
            d = load_original(path)
            rec = measurements(d, path)
            split = split_for_subject(path.parent.name)
            row = {"source": str(path), "sha256": sha256(path), "subject_id": path.parent.name,
                   "sequence_id": path.stem, "split": split, "frames": len(rec.timestamps),
                   "sample_hz": DIP_HZ, "duration_seconds": (len(rec.timestamps) - 1) / DIP_HZ,
                   "five_sensor_valid_frame_fraction": float(rec.valid.all(axis=1).mean()),
                   "nan_target_frame_fraction": float((~np.isfinite(d['gt']).all(axis=1)).mean()),
                   "root_translation_gt": "absent", "source_mapping": json.dumps(DIP_INDEX),
                   "hold_mining": "not_inspected_locked_test" if split == "test_locked" else "candidate_only"}
            rows.append(row)
            if split != "test_locked":
                for hold in mine(d["gt"], rec.valid):
                    candidates.append({"source": str(path), "subject_id": path.parent.name,
                                       "sequence_id": path.stem, "split": split, **hold})
        except Exception as error:
            errors.append({"source": str(path), "error": type(error).__name__ + ": " + str(error)})
    if not rows:
        raise RuntimeError("No valid original DIP records found; refusing empty audit")
    write_csv(out / "sequence_manifest.csv", rows, list(rows[0]))
    hold_fields = ["source", "subject_id", "sequence_id", "split", "region", "start_frame",
                   "end_frame_exclusive", "duration_seconds", "max_speed_deg_s", "annotation_status", "pose_semantics"]
    write_csv(out / "hold_candidates.csv", candidates, hold_fields)

    # Inventory local captures by declared semantics, not by reinterpretation of
    # Unity Euler angles or treating inferred tracker pose as optical reference.
    local_rows = []
    for path in sorted((root / "base_mobileposer/data/raw/ours").rglob("*.raw-imu.json")):
        try:
            d = json.loads(path.read_text())
            frames = d.get("frames", [])
            sensors = frames[0].get("sensors", []) if frames else []
            ts = np.array([f.get("timeSeconds", np.nan) for f in frames])
            local_rows.append({"source": str(path), "frames": len(frames),
                               "sample_hz_declared": d.get("sampleRate"),
                               "duration_seconds_declared": d.get("durationSeconds"),
                               "timestamps_strict": bool(np.isfinite(ts).all() and np.all(np.diff(ts) > 0)),
                               "roles_first_frame": [s.get("trackerRole") for s in sensors],
                               "acceleration_kinds": sorted(set(str(s.get("accelerationKind")) for s in sensors)),
                               "measured_gyro_declared": any(s.get("hasAngularVelocity", False) for s in sensors),
                               "independent_pose_reference": "not_verified",
                               "benchmark_eligible": False})
        except Exception as error:
            local_rows.append({"source": str(path), "error": str(error), "benchmark_eligible": False})

    sources = ["PNP/net.py", "PNP/test.py", "PNP/process.py", "PNP/evaluate_dip.py",
               "NoUse/DynaIP/utils/data.py", "NoUse/DynaIP/model/model.py", "NoUse/DynaIP/eval.py",
               "GlobalPose/net.py", "benchmarks/bridge_data.py",
               "base_mobileposer/mobileposer/process.py", "base_mobileposer/mobileposer/no_head_layouts.py"]
    fingerprint = {name: sha256(root / name) for name in sources if (root / name).is_file()}
    count_split = {s: sum(r['split'] == s for r in rows) for s in ['train', 'validation', 'test_locked']}
    summary = {"stage": "G0_source_inventory", "data_kind": "original_DIP_real_IMU",
               "target_layout": list(FIVE), "raw_DIP_source_indices": [DIP_INDEX[n] for n in FIVE],
               "sensor_mapping_verification": "cross_checked_local_PNP_process;upstream_provenance_pending",
               "counts_by_split": count_split, "sequences": len(rows), "errors": errors,
               "hold_candidates": len(candidates),
               "hold_candidates_ge_5s": sum(r['duration_seconds'] >= 5 for r in candidates),
               "hold_candidates_ge_20s": sum(r['duration_seconds'] >= 20 for r in candidates),
               "hold_candidates_ge_60s": sum(r['duration_seconds'] >= 60 for r in candidates),
               "hold_candidates_are_not_verified_sitting_or_squatting": True,
               "test_pose_not_used_for_mining_or_model_selection": True,
               "local_capture_inventory": local_rows, "local_source_sha256": fingerprint,
               "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
               "not_certified": ["external_model_inference", "upstream_revision_parity",
                                 "five_IMU_failure_hypothesis", "local_capture_independent_ground_truth"]}
    (out / "data_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    # Highest-duration candidates are for human review, not result selection.
    review = sorted(candidates, key=lambda r: (-r['duration_seconds'], r['source'], r['region']))[:30]
    write_csv(out / "review_queue.csv", review, hold_fields)
    print(json.dumps({k: summary[k] for k in ['sequences','counts_by_split','hold_candidates',
                                             'hold_candidates_ge_5s','hold_candidates_ge_20s',
                                             'hold_candidates_ge_60s','errors']}, ensure_ascii=False))


if __name__ == "__main__":
    main()
