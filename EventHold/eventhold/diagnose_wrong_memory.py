"""Controlled recurrent-state injection for B0/M1 recovery diagnostics.

This is an exploratory synthetic stress test.  It changes model state while
leaving the five-IMU observation stream untouched.  Ground truth may select a
worst prior donor in the explicitly oracle-only stress condition; it never
enters model inference.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .adapters import eventhold_features
from .audit import sha256
from .baseline import CausalBaseline, geodesic
from .records import FIVE, ImuRecord
from .region_event_model import RegionEventWrite


FPS = 60
BODY = tuple(i for i in range(1, 22) if i not in (12, 15))
LOWER = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
UPPER = (13, 14, 16, 17, 18, 19, 20, 21)
JOINTS = {"all": BODY, "lower": LOWER, "upper": UPPER}


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


def initial_state(model, reference):
    if isinstance(model, CausalBaseline):
        return reference.new_zeros((model.gru.num_layers, 1, model.gru.hidden_size))
    if isinstance(model, RegionEventWrite):
        return reference.new_zeros((1, model.hidden))
    raise TypeError("Unsupported model for controlled state injection")


def inject_state(model, clean_state, donor_state, scope):
    """Return a transplanted state without modifying either source tensor."""
    if clean_state.shape != donor_state.shape:
        raise ValueError("Clean and donor states must have identical shapes")
    if isinstance(model, CausalBaseline):
        if scope != "all":
            raise ValueError("B0 has no region-addressable recurrent state")
        return donor_state.clone()
    if not isinstance(model, RegionEventWrite):
        raise TypeError("Unsupported model for controlled state injection")
    if scope not in ("all", "lower", "upper"):
        raise ValueError("M1 scope must be all, lower, or upper")
    result = clean_state.clone()
    width = model.hidden // 3
    if scope == "all":
        result.copy_(donor_state)
    elif scope == "lower":
        result[:, :width] = donor_state[:, :width]
    else:
        result[:, width : 2 * width] = donor_state[:, width : 2 * width]
    return result


def clean_trace(model, features):
    state = initial_state(model, features)
    states_before, predictions = [], []
    with torch.no_grad():
        for frame in features:
            states_before.append(state.clone())
            prediction, state = model(frame.view(1, 1, -1), state)
            predictions.append(prediction[0, 0])
    return torch.stack(predictions), states_before


def run_from_state(model, features, state):
    with torch.no_grad():
        prediction, _ = model(features[None], state)
    return prediction[0]


def first_sustained_below(values, valid, threshold, sustain, start=0):
    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or sustain < 1:
        raise ValueError("Invalid recovery arrays or sustain length")
    good = valid & np.isfinite(values) & (values <= threshold)
    run = 0
    for index in range(max(0, start), len(good)):
        run = run + 1 if good[index] else 0
        if run >= sustain:
            return index - sustain + 1
    return None


def donor_indices(event_start, stride):
    if event_start < 0 or stride < 1:
        raise ValueError("Invalid event start or donor stride")
    candidates = list(range(0, event_start + 1, stride))
    if not candidates or candidates[-1] != event_start:
        candidates.append(event_start)
    return candidates


def select_donor(
    model,
    policy,
    scope,
    states_before,
    features,
    target,
    clean_prediction,
    event_start,
    injection_frame,
    selection_frames,
    stride,
):
    if policy == "pre_event":
        return event_start, states_before[event_start]
    if policy != "worst_prior":
        raise ValueError("Unknown donor policy")
    stop = min(len(features), injection_frame + selection_frames)
    joints = JOINTS[scope]
    clean_error = geodesic(
        clean_prediction[injection_frame:stop, joints], target[injection_frame:stop, joints]
    ).mean().item()
    best = None
    for index in donor_indices(event_start, stride):
        state = inject_state(model, states_before[injection_frame], states_before[index], scope)
        prediction = run_from_state(model, features[injection_frame:stop], state)
        error = geodesic(prediction[:, joints], target[injection_frame:stop, joints]).mean().item()
        score = error - clean_error
        if best is None or score > best[0]:
            best = (score, index, states_before[index])
    return best[1], best[2]


def event_metrics(
    model,
    scope,
    clean_prediction,
    injected_prediction,
    target,
    valid,
    injection_frame,
    event_end,
    donor_frame,
    subject,
    filename,
    manipulation_frames,
    minimum_excess_deg,
    minimum_delta_deg,
    recovery_thresholds_deg,
    sustain_frames,
    state_distance,
):
    joints = JOINTS[scope]
    clean = clean_prediction[injection_frame:event_end]
    injected = injected_prediction[: event_end - injection_frame]
    truth = target[injection_frame:event_end]
    frame_valid = valid[injection_frame:event_end]
    output_delta = geodesic(injected[:, joints], clean[:, joints]).mean(-1).numpy() * 180 / np.pi
    clean_error = geodesic(clean[:, joints], truth[:, joints]).mean(-1).numpy() * 180 / np.pi
    injected_error = geodesic(injected[:, joints], truth[:, joints]).mean(-1).numpy() * 180 / np.pi
    excess_error = injected_error - clean_error
    check = min(manipulation_frames, len(output_delta))
    check_valid = frame_valid[:check]
    initial_delta = float(np.mean(output_delta[:check][check_valid]))
    initial_excess = float(np.mean(excess_error[:check][check_valid]))
    effective = initial_delta >= minimum_delta_deg and initial_excess >= minimum_excess_deg
    row = {
        "file": filename,
        "subject_key": subject,
        "scope": scope,
        "injection_frame": injection_frame,
        "donor_frame": donor_frame,
        "followup_seconds": (event_end - injection_frame) / FPS,
        "state_l2": state_distance,
        "initial_output_delta_deg": initial_delta,
        "initial_gt_excess_deg": initial_excess,
        "mean_output_delta_deg": float(np.mean(output_delta[frame_valid])),
        "mean_gt_excess_deg": float(np.mean(excess_error[frame_valid])),
        "end_output_delta_deg": float(np.mean(output_delta[-min(FPS, len(output_delta)) :])),
        "end_gt_excess_deg": float(np.mean(excess_error[-min(FPS, len(excess_error)) :])),
        "output_delta_auc_deg_s": float(np.sum(output_delta[frame_valid]) / FPS),
        "effective_wrong_injection": bool(effective),
    }
    recovery_start = min(check, len(output_delta))
    for threshold in recovery_thresholds_deg:
        recovered_at = first_sustained_below(
            output_delta, frame_valid, threshold, sustain_frames, recovery_start
        )
        key = "%gdeg" % threshold
        row[f"recovery_{key}_seconds"] = (
            None if recovered_at is None else recovered_at / FPS
        )
        row[f"unrecovered_{key}"] = bool(effective and recovered_at is None)
    trajectories = {
        "output_delta_deg": output_delta,
        "clean_gt_error_deg": clean_error,
        "injected_gt_error_deg": injected_error,
        "gt_excess_deg": excess_error,
        "valid": frame_valid,
    }
    return row, trajectories


def summarize(rows, primary_threshold):
    key = "%gdeg" % primary_threshold
    groups = {}
    for scope in sorted(set(row["scope"] for row in rows)):
        selected = [row for row in rows if row["scope"] == scope]
        effective = [row for row in selected if row["effective_wrong_injection"]]
        recovered = [row for row in effective if row[f"recovery_{key}_seconds"] is not None]
        by_subject = {}
        for subject in sorted(set(row["subject_key"] for row in selected)):
            subject_rows = [row for row in selected if row["subject_key"] == subject]
            subject_effective = [row for row in subject_rows if row["effective_wrong_injection"]]
            subject_unrecovered = [row for row in subject_effective if row[f"unrecovered_{key}"]]
            by_subject[subject] = {
                "events": len(subject_rows),
                "effective_wrong_injections": len(subject_effective),
                "unrecovered": len(subject_unrecovered),
                "unrecovered_fraction": (
                    None if not subject_effective else len(subject_unrecovered) / len(subject_effective)
                ),
            }
        groups[scope] = {
            "events": len(selected),
            "subjects": len(by_subject),
            "effective_wrong_injections": len(effective),
            "recovered": len(recovered),
            "unrecovered": len(effective) - len(recovered),
            "unrecovered_fraction": None if not effective else 1 - len(recovered) / len(effective),
            "median_recovery_seconds_among_recovered": (
                None
                if not recovered
                else float(np.median([row[f"recovery_{key}_seconds"] for row in recovered]))
            ),
            "mean_initial_output_delta_deg": float(
                np.mean([row["initial_output_delta_deg"] for row in selected])
            ),
            "mean_initial_gt_excess_deg": float(
                np.mean([row["initial_gt_excess_deg"] for row in selected])
            ),
            "by_subject": by_subject,
        }
    return groups


def load_candidates(path):
    with path.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        if row["kind"] == "seated_like":
            result.setdefault(row["source"], []).append(row)
    return result


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--donor-policy", choices=("pre_event", "worst_prior"), required=True)
    parser.add_argument("--inject-after-seconds", type=float, default=5.0)
    parser.add_argument("--selection-seconds", type=float, default=0.5)
    parser.add_argument("--donor-stride-frames", type=int, default=30)
    parser.add_argument("--manipulation-seconds", type=float, default=0.5)
    parser.add_argument("--minimum-excess-deg", type=float, default=1.0)
    parser.add_argument("--minimum-delta-deg", type=float, default=2.0)
    parser.add_argument("--recovery-thresholds-deg", type=float, nargs="+", default=(1.0, 2.0, 5.0))
    parser.add_argument("--primary-recovery-deg", type=float, default=2.0)
    parser.add_argument("--sustain-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    if args.primary_recovery_deg not in args.recovery_thresholds_deg:
        raise ValueError("Primary recovery threshold must be in sensitivity thresholds")
    if args.inject_after_seconds <= 0 or args.selection_seconds <= 0 or args.manipulation_seconds <= 0:
        raise ValueError("Durations must be positive")

    torch.set_num_threads(2)
    run_manifest = json.loads((args.run / "run_manifest.json").read_text())
    checkpoint = torch.load(args.run / "last.pt", map_location="cpu")
    model = (
        CausalBaseline(run_manifest["hidden"])
        if run_manifest["model"] == "b0"
        else RegionEventWrite(run_manifest["hidden"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scopes = ("all",) if run_manifest["model"] == "b0" else ("lower", "upper", "all")
    split = json.loads((args.split / "split.json").read_text())
    candidates = load_candidates(args.candidate_csv)
    rows, trajectory_store = [], {}
    selection_frames = max(1, round(args.selection_seconds * FPS))
    manipulation_frames = max(1, round(args.manipulation_seconds * FPS))
    sustain_frames = max(1, round(args.sustain_seconds * FPS))
    inject_offset = round(args.inject_after_seconds * FPS)

    for entry in split["rows"]:
        if entry["split"] != "holdout_exploration":
            continue
        matching = candidates.get(entry["source"], [])
        if len(matching) != 1:
            raise ValueError(f"Expected one seated_like candidate for {entry['source']}, got {len(matching)}")
        candidate = matching[0]
        features, target, valid = load_sequence(args.data / entry["id"])
        clean_prediction, states_before = clean_trace(model, features)
        event_start = max(0, round(float(candidate["start_seconds"]) * FPS))
        event_end = min(
            len(features),
            round((float(candidate["start_seconds"]) + float(candidate["duration_seconds"])) * FPS),
        )
        injection_frame = event_start + inject_offset
        if injection_frame + max(manipulation_frames, sustain_frames) > event_end:
            continue
        for scope in scopes:
            donor_frame, donor_state = select_donor(
                model,
                args.donor_policy,
                scope,
                states_before,
                features,
                target,
                clean_prediction,
                event_start,
                injection_frame,
                selection_frames,
                args.donor_stride_frames,
            )
            clean_state = states_before[injection_frame]
            state = inject_state(model, clean_state, donor_state, scope)
            injected = run_from_state(model, features[injection_frame:event_end], state)
            changed = state - clean_state
            row, trajectories = event_metrics(
                model,
                scope,
                clean_prediction,
                injected,
                target,
                valid,
                injection_frame,
                event_end,
                donor_frame,
                entry["subject_key"],
                entry["id"],
                manipulation_frames,
                args.minimum_excess_deg,
                args.minimum_delta_deg,
                args.recovery_thresholds_deg,
                sustain_frames,
                float(torch.linalg.vector_norm(changed).item()),
            )
            row.update(
                {
                    "event_start_frame": event_start,
                    "event_end_frame_exclusive": event_end,
                    "standing_before_15s": candidate["standing_before_15s"].lower() == "true",
                    "complete_transition_proxy": candidate["complete_transition_proxy"].lower() == "true",
                }
            )
            rows.append(row)
            prefix = f"{Path(entry['id']).stem}_{scope}"
            for name, values in trajectories.items():
                trajectory_store[f"{prefix}_{name}"] = values

    if not rows:
        raise RuntimeError("No eligible holdout events")
    args.output.mkdir(parents=True)
    write_csv(args.output / "events.csv", rows)
    np.savez_compressed(args.output / "trajectories.npz", **trajectory_store)
    protocol = {
        "state_only_intervention": True,
        "imu_stream_modified": False,
        "donor_policy": args.donor_policy,
        "donor_causality": "state_before_candidate_start_or_earlier",
        "oracle_donor_selection": args.donor_policy == "worst_prior",
        "inject_after_seconds": args.inject_after_seconds,
        "selection_seconds": args.selection_seconds,
        "donor_stride_frames": args.donor_stride_frames,
        "manipulation_window_seconds": args.manipulation_seconds,
        "effective_wrong_injection_rule": {
            "minimum_initial_output_delta_deg": args.minimum_delta_deg,
            "minimum_initial_gt_excess_deg": args.minimum_excess_deg,
        },
        "recovery_rule": {
            "metric": "region_mean_geodesic_distance_to_paired_clean_trajectory",
            "thresholds_deg": args.recovery_thresholds_deg,
            "primary_threshold_deg": args.primary_recovery_deg,
            "sustain_seconds": args.sustain_seconds,
            "unrecovered_is_right_censored": True,
        },
    }
    summary = {
        "status": "controlled_wrong_memory_synthetic_exploration_not_real_performance",
        "model": run_manifest["model"],
        "protocol": protocol,
        "summary_by_scope": summarize(rows, args.primary_recovery_deg),
        "statistical_inference": {
            "performed": False,
            "reason": "Only three holdout subject groups; frames and repeated events are not independent samples.",
        },
        "limitations": [
            "AMASS-derived synthetic IMUs are not real wearable measurements.",
            "The trained models used only two exploration sequences and have high absolute pose error.",
            "State transplantation tests recovery from an intervention, not the natural frequency of wrong memories.",
            "worst_prior uses ground truth only to construct an oracle stress case and is not deployable.",
            "M1 hidden partitions are gated by region but the dense pose head is not anatomically isolated.",
        ],
        "artifacts": {"events": "events.csv", "trajectories": "trajectories.npz"},
        "hashes": {
            "run_manifest": sha256(args.run / "run_manifest.json"),
            "checkpoint": sha256(args.run / "last.pt"),
            "split": sha256(args.split / "split.json"),
            "candidate_csv": sha256(args.candidate_csv),
            "code": sha256(Path(__file__)),
        },
        "head_input": False,
        "test_accessed": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
