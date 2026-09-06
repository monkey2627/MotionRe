#!/usr/bin/env python3
"""
Unified per-body-part geodesic error evaluation for multiple methods / sensor combos.

Input:  result .pt files, each containing:
          pred: List[Tensor[T, 24, 3, 3]]   — predicted local rotation matrices
          gt:   List[Tensor[T, 24, 3, 3]]   — ground-truth local rotation matrices

Output: comparison table (stdout + optional CSV)

Body part → SMPL joint mapping
  腰 Lumbar:      joint 3  (Spine1)
  胸 Thorax:      joints 6, 9 (Spine2, Spine3)
  大臂 UpperArm:  joints 16, 17 (L/R Shoulder — drives upper arm)
  大腿 Thigh:     joints 1, 2  (L/R Hip — drives thigh)
  小腿 LowerLeg:  joints 4, 5  (L/R Knee)
  手腕 Wrist:     joints 18, 19 (L/R Elbow — proximal to forearm sensor)

Usage:
  # Single result file:
  python -m mobileposer.evaluate_per_bodypart \
      --results results/mobileposer/mobileposer_all.pt

  # Multiple methods, with labels:
  python -m mobileposer.evaluate_per_bodypart \
      --results \
        "results/mobileposer/mobileposer_all.pt:MobilePoser(5传感器)" \
        "results/mobileposer/mobileposer_lw_rw_h.pt:MobilePoser(双腕+头)" \
        "results/tip/tip.pt:TIP(6传感器)" \
      --csv results/comparison.csv

  # Batch-evaluate all MobilePoser combos in a directory:
  python -m mobileposer.evaluate_per_bodypart \
      --dir results/mobileposer --csv results/mobileposer_comparison.csv
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch


BODY_PART_JOINTS = {
    '腰 Lumbar':     [3],
    '胸 Thorax':     [6, 9],
    '大臂 UpperArm': [16, 17],
    '大腿 Thigh':    [1, 2],
    '小腿 LowerLeg': [4, 5],
    '手腕 Wrist':    [18, 19],
}

# Joints the model actually predicts (non-ignored); used for "All" metric.
# joint_set.ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]
_PREDICTED_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]


def geodesic_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    Geodesic angular error in degrees.
    R_pred, R_gt: [..., 3, 3]
    Returns: [...] tensor of per-joint-frame errors in degrees.
    """
    R_diff = R_pred.transpose(-1, -2) @ R_gt
    trace  = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_a  = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.acos(cos_a) * (180.0 / math.pi)


def compute_bodypart_errors(pred_list, gt_list):
    """
    Compute mean geodesic error per body part, averaged over all frames and sequences.

    pred_list, gt_list: List[Tensor[T, 24, 3, 3]]
    Returns dict: {part_name: mean_error_degrees, ..., '全身 All': mean_error_degrees}
    """
    part_accum = {k: 0.0 for k in BODY_PART_JOINTS}
    part_count = {k: 0   for k in BODY_PART_JOINTS}
    all_accum, all_count = 0.0, 0

    for pred, gt in zip(pred_list, gt_list):
        pred = pred.float()
        gt   = gt.float()
        T    = pred.shape[0]

        for part_name, joint_ids in BODY_PART_JOINTS.items():
            err = geodesic_error_deg(pred[:, joint_ids], gt[:, joint_ids])  # [T, n]
            part_accum[part_name] += err.sum().item()
            part_count[part_name] += T * len(joint_ids)

        # "All" = predicted joint set only
        err_all = geodesic_error_deg(pred[:, _PREDICTED_JOINTS], gt[:, _PREDICTED_JOINTS])
        all_accum += err_all.sum().item()
        all_count += T * len(_PREDICTED_JOINTS)

    result = {k: part_accum[k] / part_count[k] for k in BODY_PART_JOINTS}
    result['全身 All'] = all_accum / all_count
    return result


def load_result_file(path: str):
    """Load a result .pt file; returns (pred_list, gt_list)."""
    data = torch.load(path, map_location='cpu')
    return data['pred'], data['gt']


def parse_result_specs(specs):
    """
    Parse 'path:label' strings. Label is optional (defaults to file stem).
    Returns list of (path, label).
    """
    entries = []
    for spec in specs:
        if ':' in spec:
            # Split on first colon only (Windows paths have colons after drive letter)
            # Handle "C:\path\to\file.pt:Label" correctly
            # Strategy: find the last colon that's followed by a non-path character
            # Simpler: split from the right
            idx = spec.rfind(':')
            # Check if what follows the colon looks like a path component (single letter = drive)
            before, after = spec[:idx], spec[idx+1:]
            if len(before) == 1:  # Windows drive letter like "C"
                path, label = spec, Path(spec).stem
            else:
                path, label = before, after
        else:
            path, label = spec, Path(spec).stem
        entries.append((path, label))
    return entries


def print_table(all_results: dict):
    """Print a formatted comparison table sorted by overall error."""
    part_names = list(BODY_PART_JOINTS.keys()) + ['全身 All']
    sorted_items = sorted(all_results.items(), key=lambda x: x[1]['全身 All'])

    # Column widths
    label_w = max(6, max(len(lbl) for lbl in all_results)) + 2
    col_w   = 14

    header_parts = [f"  {p:>{col_w-2}}" for p in part_names]
    header = f"{'方法':{label_w}}" + "".join(header_parts)
    sep    = "=" * len(header)

    print("\n" + sep)
    print(header)
    print("-" * len(header))
    for label, errors in sorted_items:
        row = f"{label:{label_w}}" + "".join(f"  {errors[p]:>11.2f} °" for p in part_names)
        print(row)
    print(sep + "\n")


def save_csv(all_results: dict, csv_path: str):
    part_names = list(BODY_PART_JOINTS.keys()) + ['全身 All']
    sorted_items = sorted(all_results.items(), key=lambda x: x[1]['全身 All'])
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['方法'] + part_names)
        for label, errors in sorted_items:
            writer.writerow([label] + [f"{errors[p]:.4f}" for p in part_names])
    print(f"Saved CSV → {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Per-body-part geodesic error evaluation')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--results', nargs='+', metavar='PATH[:LABEL]',
                       help='Result .pt files (optionally with labels after colon)')
    group.add_argument('--dir', type=str, metavar='DIR',
                       help='Directory of .pt result files (all files evaluated)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Save comparison table to this CSV path')
    parser.add_argument('--pattern', type=str, default='*.pt',
                        help='Glob pattern when using --dir (default: *.pt)')
    args = parser.parse_args()

    # Gather (path, label) pairs
    if args.results:
        entries = parse_result_specs(args.results)
    else:
        dir_path = Path(args.dir)
        files    = sorted(dir_path.glob(args.pattern))
        if not files:
            raise FileNotFoundError(f"No files matching '{args.pattern}' in {dir_path}")
        entries = [(str(f), f.stem) for f in files]

    # Evaluate each entry
    all_results = {}
    for path, label in entries:
        print(f"Evaluating: {label}  ←  {path}")
        pred_list, gt_list = load_result_file(path)
        errors = compute_bodypart_errors(pred_list, gt_list)
        all_results[label] = errors

        # Quick per-part preview
        parts_str = "  |  ".join(f"{k}: {v:.2f}°" for k, v in errors.items())
        print(f"  {parts_str}")

    # Print comparison table
    print_table(all_results)

    # Optionally save CSV
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or '.', exist_ok=True)
        save_csv(all_results, args.csv)


if __name__ == '__main__':
    main()
