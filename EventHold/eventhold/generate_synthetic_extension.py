"""Generate non-overlapping five-IMU AMASS extension sources.

This keeps the existing pilot generator and synthesis semantics, but selects
only native-confirmed transition candidates from subject groups absent from the
pilot.  The output remains exploration data, not a benchmark.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .audit import sha256
from .generate_five_imu_synthetic import SITE_JOINTS, SITE_NAMES, SITE_VERTICES, synthesize


def read_rows(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--mobileposer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-duration-seconds", type=float, default=10.0)
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new extension output directory")
    if args.min_duration_seconds <= 0 or args.max_files <= 0:
        raise ValueError("Duration and max-files must be positive")

    pilot = json.loads(args.pilot_manifest.read_text())
    pilot_sources = {row["source"] for row in pilot["manifest"]}
    pilot_subjects = set()
    for row in read_rows(args.candidate_csv):
        if row["source"] in pilot_sources:
            pilot_subjects.add(row["subject_key"])

    selected = {}
    for row in read_rows(args.candidate_csv):
        if row["kind"] != "seated_like" or row["complete_transition_proxy"] != "True":
            continue
        if float(row["duration_seconds"]) < args.min_duration_seconds:
            continue
        if row["source"] in pilot_sources or row["subject_key"] in pilot_subjects:
            continue
        selected.setdefault(row["source"], row)
    chosen = sorted(selected.values(), key=lambda row: (row["subject_key"], row["source"]))[: args.max_files]
    if not chosen:
        raise ValueError("No non-overlapping extension sources matched the selection rules")

    sys.path.insert(0, str(args.mobileposer_root))
    from mobileposer.articulate.model import ParametricModel

    torch.set_num_threads(2)
    model_path = args.mobileposer_root / "mobileposer/smpl/basicmodel_m.pkl"
    model = ParametricModel(str(model_path))
    args.output.mkdir(parents=True)
    manifest = []
    errors = []
    for index, row in enumerate(chosen):
        source = Path(row["source"])
        try:
            generated = synthesize(source, model, max_seconds=args.max_seconds)
            artifact = f"seq_{index:02d}.npz"
            np.savez_compressed(
                args.output / artifact,
                orientation=generated["orientation"],
                acceleration=generated["acceleration"],
                target=generated["target"],
                valid=generated["valid"],
                source_indices=generated["source_indices"],
            )
            manifest.append(
                {
                    "id": artifact,
                    "source": str(source),
                    "source_sha256": sha256(source),
                    "subject_key": row["subject_key"],
                    "frames": len(generated["valid"]),
                    "fps": 60.0,
                    "native_fps": generated["native_fps"],
                    "invalid_boundary_frames": int((~generated["valid"]).sum()),
                    "site_names": list(SITE_NAMES),
                    "site_joints": list(SITE_JOINTS),
                    "site_vertices": list(SITE_VERTICES),
                    "synthetic_acceleration": "centered_second_difference_m_per_s2",
                    "candidate_start_seconds": float(row["start_seconds"]),
                    "candidate_duration_seconds": float(row["duration_seconds"]),
                    "complete_transition_proxy": True,
                }
            )
        except Exception as error:
            errors.append({"source": str(source), "error": f"{type(error).__name__}: {error}"})
    if not manifest:
        raise RuntimeError(f"All selected extension sources failed: {errors}")

    report = {
        "status": "bounded_five_imu_synthetic_extension_not_real_imu",
        "files": len(manifest),
        "errors": errors,
        "manifest": manifest,
        "layout": list(SITE_NAMES),
        "head_input": False,
        "pilot_source_overlap": sorted(set(row["source"] for row in manifest) & pilot_sources),
        "pilot_subject_overlap": sorted(set(row["subject_key"] for row in manifest) & pilot_subjects),
        "selection": {
            "native_confirmation": True,
            "complete_transition_proxy": True,
            "min_duration_seconds": args.min_duration_seconds,
            "max_seconds": args.max_seconds,
            "excluded_native_rate_not_integer_downsample": True,
        },
        "model_file": str(model_path),
        "model_sha256": sha256(model_path),
        "code_sha256": sha256(Path(__file__)),
        "limitations": [
            "AMASS fitted motion is not measured IMU",
            "candidate semantics are geometry-confirmed proxies, not action annotations",
            "centered acceleration labels are offline and boundary frames are invalid",
            "only three new subject groups were available under the current rate and non-overlap rules",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(chosen[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in chosen:
            writer.writerow(row)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
