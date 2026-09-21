"""Region-wise initial/late diagnostics for synthetic B0/M1 runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .audit import sha256
from .baseline import CausalBaseline, geodesic
from .records import FIVE, ImuRecord
from .adapters import eventhold_features
from .region_event_model import RegionEventWrite


FPS = 60
BODY = tuple(i for i in range(1, 22) if i not in (12, 15))
LOWER = tuple(range(1, 12))
UPPER = (13, 14, 16, 17, 18, 19, 20, 21)
REGIONS = {"all": BODY, "lower": LOWER, "upper": UPPER}


def load_sequence(path):
    with np.load(path, allow_pickle=False) as data:
        valid = np.asarray(data["valid"], dtype=bool)
        sensor_valid = valid[:, None].repeat(5, axis=1)
        record = ImuRecord(
            np.arange(len(valid), dtype=float) / FPS,
            FIVE,
            data["orientation"].astype(np.float32),
            data["acceleration"].astype(np.float32),
            sensor_valid,
            str(path),
        ).validate()
        target = torch.from_numpy(data["target"].astype(np.float32))
    return torch.from_numpy(eventhold_features(record)).float(), target, valid


def load_model(run):
    manifest = json.loads((run / "run_manifest.json").read_text())
    model = CausalBaseline(manifest["hidden"]) if manifest["model"] == "b0" else RegionEventWrite(manifest["hidden"])
    checkpoint = torch.load(run / "last.pt", map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, manifest


def infer(model, features):
    predictions, lower_gates, upper_gates = [], [], []
    state = None
    with torch.no_grad():
        for start in range(0, len(features), 240):
            x = features[start : start + 240][None]
            if isinstance(model, RegionEventWrite):
                prediction, state, lower, upper = model(x, state, return_gates=True)
                lower_gates.append(lower[0].mean(-1))
                upper_gates.append(upper[0].mean(-1))
            else:
                prediction, state = model(x, state)
            predictions.append(prediction[0])
    return torch.cat(predictions), (
        None if not lower_gates else torch.cat(lower_gates).numpy(),
        None if not upper_gates else torch.cat(upper_gates).numpy(),
    )


def segment_mean(values, start, end):
    values = np.asarray(values)
    return float(np.mean(values[start:end]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")

    model, manifest = load_model(args.run)
    split = json.loads((args.split / "split.json").read_text())
    with args.candidate_csv.open(encoding="utf-8-sig") as handle:
        candidates = list(csv.DictReader(handle))
    by_source = {}
    for row in candidates:
        if row["kind"] == "seated_like":
            by_source.setdefault(row["source"], []).append(row)

    rows = []
    for entry in split["rows"]:
        if entry["split"] != "holdout_exploration":
            continue
        source_candidates = by_source.get(entry["source"], [])
        if len(source_candidates) != 1:
            raise ValueError(f"Expected one seated_like candidate for {entry['source']}")
        candidate = source_candidates[0]
        features, target, valid = load_sequence(args.data / entry["id"])
        prediction, gates = infer(model, features)
        start = max(0, round(float(candidate["start_seconds"]) * FPS))
        end = min(len(features), round((float(candidate["start_seconds"]) + float(candidate["duration_seconds"])) * FPS))
        window = min(10 * FPS, end - start)
        early = (start, start + window)
        late = (max(start, end - window), end)
        row = {"file": entry["id"], "subject_key": entry["subject_key"], "model": manifest["model"], "frames": end - start}
        for name, joints in REGIONS.items():
            errors = geodesic(prediction[:, joints], target[:, joints]).mean(-1).numpy() * 180 / np.pi
            row[f"{name}_early10s_deg"] = segment_mean(errors, *early)
            row[f"{name}_late10s_deg"] = segment_mean(errors, *late)
            row[f"{name}_late_minus_early_deg"] = row[f"{name}_late10s_deg"] - row[f"{name}_early10s_deg"]
        if gates[0] is None:
            row["lower_gate_mean"] = None
            row["upper_gate_mean"] = None
            row["lower_gate_late_minus_early"] = None
            row["upper_gate_late_minus_early"] = None
        else:
            row["lower_gate_mean"] = segment_mean(gates[0], start, end)
            row["upper_gate_mean"] = segment_mean(gates[1], start, end)
            row["lower_gate_late_minus_early"] = segment_mean(gates[0], *late) - segment_mean(gates[0], *early)
            row["upper_gate_late_minus_early"] = segment_mean(gates[1], *late) - segment_mean(gates[1], *early)
        rows.append(row)

    if not rows:
        raise RuntimeError("No holdout candidates")
    args.output.mkdir(parents=True)
    with (args.output / "regions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "regionwise_initial_vs_late_synthetic_exploration",
        "model": manifest["model"],
        "rows": rows,
        "means": {
            name: {
                "early10s_deg": float(np.mean([row[f"{name}_early10s_deg"] for row in rows])),
                "late10s_deg": float(np.mean([row[f"{name}_late10s_deg"] for row in rows])),
                "late_minus_early_deg": float(np.mean([row[f"{name}_late_minus_early_deg"] for row in rows])),
            }
            for name in REGIONS
        },
        "statistical_inference": {"performed": False, "reason": "One holdout sequence per LOSO fold; frames are not independent samples."},
        "head_input": False,
        "test_accessed": False,
        "run_sha256": sha256(args.run / "run_manifest.json"),
        "code_sha256": sha256(Path(__file__)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
