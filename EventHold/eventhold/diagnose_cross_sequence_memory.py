"""Truth-free cross-sequence hidden-state stress test.

Donor states are collected from training-exploration sequences and selected by
masked hidden-state distance only.  The held-out target is used only after
inference to score the intervention; it never chooses a donor.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audit import sha256
from .baseline import CausalBaseline
from .diagnose_wrong_memory import (
    FPS,
    JOINTS,
    RegionEventWrite,
    BODY,
    clean_trace,
    event_metrics,
    inject_state,
    load_candidates,
    load_sequence,
    run_from_state,
    summarize,
    write_csv,
)


def state_mask(model, scope, state):
    if isinstance(model, CausalBaseline):
        return torch.ones_like(state)
    width = model.hidden // 3
    mask = torch.zeros_like(state)
    if scope == "lower":
        mask[:, :width] = 1
    elif scope == "upper":
        mask[:, width : 2 * width] = 1
    elif scope == "all":
        mask[:] = 1
    else:
        raise ValueError(f"Unknown scope: {scope}")
    return mask


def select_cross_donor(model, scope, clean_state, donor_pool):
    """Choose the farthest training-sequence state without target access."""
    if not donor_pool:
        raise ValueError("Donor pool is empty")
    mask = state_mask(model, scope, clean_state)
    candidates = []
    for source, frame, donor_state in donor_pool:
        if donor_state.shape != clean_state.shape:
            raise ValueError("Donor and target recurrent state shapes differ")
        distance = torch.linalg.vector_norm((donor_state - clean_state) * mask).item()
        candidates.append((distance, source, frame, donor_state))
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))


def collect_donor_pool(model, split, data_root, stride):
    pool = []
    for entry in split["rows"]:
        if entry["split"] != "train_exploration":
            continue
        features, _, _ = load_sequence(data_root / entry["id"])
        _, states = clean_trace(model, features)
        for frame in range(0, len(states), stride):
            pool.append((entry["id"], frame, states[frame].clone()))
    return pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inject-after-seconds", type=float, default=5.0)
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
    if args.donor_stride_frames < 1 or args.inject_after_seconds <= 0:
        raise ValueError("Invalid donor stride or injection delay")
    if args.primary_recovery_deg not in args.recovery_thresholds_deg:
        raise ValueError("Primary recovery threshold must be in sensitivity thresholds")

    torch.set_num_threads(2)
    manifest = json.loads((args.run / "run_manifest.json").read_text())
    checkpoint = torch.load(args.run / "last.pt", map_location="cpu")
    model = CausalBaseline(manifest["hidden"]) if manifest["model"] == "b0" else RegionEventWrite(manifest["hidden"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scopes = ("all",) if manifest["model"] == "b0" else ("lower", "upper", "all")
    split = json.loads((args.split / "split.json").read_text())
    candidates = load_candidates(args.candidate_csv)
    pool = collect_donor_pool(model, split, args.data, args.donor_stride_frames)
    manipulation_frames = max(1, round(args.manipulation_seconds * FPS))
    sustain_frames = max(1, round(args.sustain_seconds * FPS))
    rows, trajectories = [], {}

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
        event_end = min(len(features), round((float(candidate["start_seconds"]) + float(candidate["duration_seconds"])) * FPS))
        injection_frame = event_start + round(args.inject_after_seconds * FPS)
        if injection_frame + max(manipulation_frames, sustain_frames) > event_end:
            continue
        for scope in scopes:
            donor_distance, donor_source, donor_frame, donor_state = select_cross_donor(
                model, scope, states_before[injection_frame], pool
            )
            clean_state = states_before[injection_frame]
            injected_state = inject_state(model, clean_state, donor_state, scope)
            injected_prediction = run_from_state(model, features[injection_frame:event_end], injected_state)
            row, trace = event_metrics(
                model,
                scope,
                clean_prediction,
                injected_prediction,
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
                donor_distance,
            )
            row.update({"donor_source": donor_source, "donor_selection": "max_hidden_distance_no_truth"})
            rows.append(row)
            prefix = f"{Path(entry['id']).stem}_{scope}"
            for name, values in trace.items():
                trajectories[f"{prefix}_{name}"] = values

    if not rows:
        raise RuntimeError("No eligible holdout events")
    args.output.mkdir(parents=True)
    write_csv(args.output / "events.csv", rows)
    np.savez_compressed(args.output / "trajectories.npz", **trajectories)
    protocol = {
        "state_only_intervention": True,
        "imu_stream_modified": False,
        "donor_policy": "max_cross_sequence",
        "donor_pool_split": "train_exploration_only",
        "donor_selection_uses_target_truth": False,
        "donor_selection_metric": "masked_hidden_state_l2",
        "holdout_target_used_for_donor": False,
        "oracle_donor_selection": False,
        "inject_after_seconds": args.inject_after_seconds,
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
        "status": "truth_free_cross_sequence_wrong_memory_synthetic_exploration_not_real_performance",
        "model": manifest["model"],
        "protocol": protocol,
        "summary_by_scope": summarize(rows, args.primary_recovery_deg),
        "donor_pool_states": len(pool),
        "statistical_inference": {"performed": False, "reason": "Only three holdout subject groups; frames and repeated events are not independent samples."},
        "limitations": [
            "AMASS-derived synthetic IMUs are not real wearable measurements.",
            "The trained models used only two exploration sequences and have high absolute pose error.",
            "Cross-sequence state transplantation tests recoverability from an intervention, not natural error frequency.",
            "Selecting the farthest hidden state is a stress rule, not a deployment policy.",
            "M1 hidden partitions are gated by region but the dense pose head is not anatomically isolated.",
        ],
        "artifacts": {"events": "events.csv", "trajectories": "trajectories.npz"},
        "hashes": {"run_manifest": sha256(args.run / "run_manifest.json"), "checkpoint": sha256(args.run / "last.pt"), "split": sha256(args.split / "split.json"), "candidate_csv": sha256(args.candidate_csv), "code": sha256(Path(__file__))},
        "head_input": False,
        "test_accessed": False,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
