"""Temporal diagnostic for the trained surface-IMU MobilePoser layouts.

This reuses the original MobilePoser checkpoint and data format. It does not
train or alter a model; it reports early/late and fixed-window errors per
sequence so long-sequence degradation can be checked before adding M1.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch


PARTS = {
    "lower": [1, 2, 4, 5],
    "trunk": [3, 6, 9],
    "upper": [16, 17, 18, 19],
    "body": [1, 2, 3, 4, 5, 6, 9, 16, 17, 18, 19],
}


def geodesic_deg(pred, gt):
    diff = pred.transpose(-1, -2) @ gt
    trace = diff[..., 0, 0] + diff[..., 1, 1] + diff[..., 2, 2]
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)) * (180.0 / math.pi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mobileposer-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layouts", nargs="+", required=True)
    p.add_argument("--min-seconds", type=float, default=20.0)
    p.add_argument("--window-seconds", type=float, default=10.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory")
    sys.path.insert(0, str(args.mobileposer_root))
    from mobileposer.config import model_config
    from mobileposer.no_head_layouts import LAYOUTS
    from mobileposer.utils.model_utils import load_model
    from mobileposer.evaluate_no_head_5imu import load_sequences, prepare_imu

    device = torch.device(args.device)
    model_config.device = device
    args.output.mkdir(parents=True)
    rows = []
    for layout in args.layouts:
        if layout not in LAYOUTS:
            raise ValueError(f"Unknown layout: {layout}")
        checkpoint = args.checkpoint_root / layout / "1" / "base_model.pth"
        data_dir = args.data_root / layout
        model = load_model(str(checkpoint)).to(device).eval()
        sequences = load_sequences(data_dir)
        for index, seq in enumerate(sequences):
            fps = 30.0
            n = min(len(seq["pose"]), len(seq["acc"]))
            if n < args.min_seconds * fps:
                continue
            pose_gt = seq["pose"][:n].float()
            imu = prepare_imu(seq["acc"][:n], seq["ori"][:n]).to(device)
            model.reset()
            with torch.no_grad():
                pred, _, _, _ = model.forward_offline(imu.unsqueeze(0), [n])
            err = geodesic_deg(pred[0].cpu(), pose_gt)
            w = max(1, int(round(args.window_seconds * fps)))
            for part, joints in PARTS.items():
                values = err[:, joints].mean(dim=1)
                early = values[:w]
                late = values[-w:]
                rows.append({
                    "layout": layout,
                    "sequence_index": index,
                    "source": seq["source"],
                    "frames": n,
                    "duration_seconds": n / fps,
                    "part": part,
                    "early_deg": float(early.mean()),
                    "late_deg": float(late.mean()),
                    "late_minus_early_deg": float(late.mean() - early.mean()),
                    "min_window_deg": float(values.unfold(0, w, w).mean(dim=-1).min()),
                    "max_window_deg": float(values.unfold(0, w, w).mean(dim=-1).max()),
                })
        del model
    fields = list(rows[0]) if rows else ["layout"]
    with (args.output / "temporal_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "trained_mobileposer_temporal_diagnostic",
        "layouts": args.layouts,
        "min_seconds": args.min_seconds,
        "window_seconds": args.window_seconds,
        "rows": len(rows),
        "interpretation": "early/late is a temporal drift diagnostic, not proof of a hold event",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
