"""Validate tracker.json position semantics against the HumanPose joint cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from mobileposer.fit_ours_smpl import _HUMANPOSE_BONES, _load_humanpose_positions

S = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
CANDIDATES = {
    "LEFT_ELBOW": ["humerus_l", "ulna_l", "hand_l"],
    "RIGHT_ELBOW": ["humerus_r", "ulna_r", "hand_r"],
    "LEFT_KNEE": ["femur_l", "tibia_l", "talus_l"],
    "RIGHT_KNEE": ["femur_r", "tibia_r", "talus_r"],
    "LEFT_FOOT": ["talus_l", "toes_l"],
    "RIGHT_FOOT": ["talus_r", "toes_r"],
    "WAIST": ["pelvis", "lumbar_body", "thorax"],
    "CHEST": ["lumbar_body", "thorax", "head"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", type=Path, required=True)
    ap.add_argument("--humanpose", type=Path, required=True)
    args = ap.parse_args()
    package = json.loads(args.tracker.read_text(encoding="utf-8"))
    hp, _ = _load_humanpose_positions(args.humanpose)
    names = {n: i for i, n in enumerate(_HUMANPOSE_BONES)}
    frames = package["frames"]
    tracker = {}
    for role in CANDIDATES:
        values = []
        for frame in frames:
            item = next((x for x in frame.get("trackers", []) if x.get("name", "").split("//")[-1] == role), None)
            values.append([item["position"][a] for a in ("x", "y", "z")] if item else [float("nan")]*3)
        tracker[role] = torch.tensor(values, dtype=torch.float32) @ S.T
    n = min(len(hp), len(frames))
    print(f"frames tracker={len(frames)} humanpose={len(hp)} compare={n}")
    for role, candidates in CANDIDATES.items():
        x = tracker[role][:n]
        valid = torch.isfinite(x).all(1)
        if valid.sum() < 3:
            continue
        print(f"\n{role} valid={int(valid.sum())}")
        for bone in candidates:
            y = hp[:n, names[bone]]
            mask = valid & torch.isfinite(y).all(1)
            # Compare trajectories after the only ambiguity-free calibration:
            # one constant translation. Also report best isotropic scale.
            xx, yy = x[mask], y[mask]
            delta = yy.mean(0) - xx.mean(0)
            err = (xx + delta - yy).norm(dim=1)
            xc, yc = xx - xx.mean(0), yy - yy.mean(0)
            scale = (xc * yc).sum() / xc.square().sum().clamp_min(1e-8)
            err_scaled = (xx * scale + (yy - xx.mul(scale).mean(0)) - yy).norm(dim=1)
            corr = torch.nn.functional.cosine_similarity(xc.flatten(), yc.flatten(), dim=0)
            print(f"  {bone:12s} offset_rmse={err.mean():.4f}m p95={err.quantile(.95):.4f} "
                  f"scale={scale:.3f} scaled_rmse={err_scaled.mean():.4f} corr={corr:.3f}")

if __name__ == "__main__":
    main()
