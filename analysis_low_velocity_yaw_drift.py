"""
Validation experiment: does global-yaw reconstruction error accumulate faster
during low-body-velocity (quasi-static) segments than during normal-motion
segments, once we control for the general "slow motion is harder to estimate"
confound?

Hypothesis (from TIC / MobilePoser / Ultra Inertial Poser's own stated
limitations): sparse-IMU pose networks cannot correct global yaw (heading)
drift when the body isn't moving enough to give the network cross-sensor /
kinematic-chain cues. If true, injected gyro-drift noise should produce a
*larger* yaw-error penalty during low-velocity windows than during
high-velocity windows of the same sequence.

Design (isolates the drift-correction effect from the "slow motion is
generally harder to estimate" confound):
  1. Run the SAME checkpoint on the SAME sequences twice: once with clean
     synthetic IMU (no injected drift), once with realistic gyro-drift noise
     added (drift_eval_common.add_imu_noise, same model as evaluate_drift.py).
  2. For each frame, delta[t] = yaw_err_noisy[t] - yaw_err_clean[t].
     This isolates the marginal effect of the injected drift, controlling for
     baseline estimation difficulty during slow motion (which affects both
     conditions equally).
  3. Bin into non-overlapping windows; regress a local slope of delta within
     each window (growth rate), and record the window's mean body angular
     speed (ground-truth-derived, independent of the noisy input).
  4. Test whether low-speed windows show larger delta-growth than high-speed
     windows (Spearman correlation, quartile Mann-Whitney U, per-sequence
     paired Wilcoxon).

Run from code/ (repo root, one level above base_mobileposer/):
    conda activate mobileposer
    python analysis_low_velocity_yaw_drift.py \
        --model base_mobileposer/checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth \
        --layout wrists_shanks_waist --max_seqs 40 --min_frames 1800
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_DIR))
sys.path.insert(0, str(_CODE_DIR / 'base_mobileposer'))

from drift_eval_common import load_long_sequences, angle_between_rotmats, add_imu_noise
from mobileposer.config import amass, model_config
from mobileposer.utils.model_utils import load_model


HELD_OUT_PREFIXES = ('BMLhandball', 'BioMotionLab_NTroje')  # per project's own train/holdout split


def prepare_imu(acc: torch.Tensor, ori: torch.Tensor, acc_scale: float) -> torch.Tensor:
    """All 5 custom-layout slots used (no combo masking, matches all_5imu training)."""
    acc5 = (acc[:, :5] / acc_scale).flatten(1)
    ori5 = ori[:, :5].flatten(1)
    return torch.cat([acc5, ori5], dim=1)


@torch.no_grad()
def run_inference(model, imu_input: torch.Tensor, device) -> torch.Tensor:
    model.reset()
    imu = imu_input.to(device).unsqueeze(0)
    pose_pred, _, _, _ = model.forward_offline(imu, [imu.shape[1]])
    return pose_pred.cpu()


def extract_yaw_deg(R_root: torch.Tensor) -> torch.Tensor:
    """R_root: [T,3,3] global (=local, since it's the SMPL root joint).
    World frame is Y-up (amass_rot in process.py). Yaw = heading of the
    body-local +X axis projected onto the world XZ (horizontal) plane."""
    x_world = R_root[:, :, 0]              # [T,3] = column 0 of R
    yaw = torch.atan2(x_world[:, 2], x_world[:, 0])
    return torch.rad2deg(yaw)


def wrapped_abs_diff_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = (a - b + 180.0) % 360.0 - 180.0
    return d.abs()


def body_speed_deg_s(gt_pose: torch.Tensor, fps: int) -> torch.Tensor:
    """Whole-body motion proxy: mean per-joint angular speed (deg/s) over all
    24 joints, from GT local rotations. Independent of the noisy IMU input."""
    step = angle_between_rotmats(gt_pose[:-1], gt_pose[1:])   # [T-1, 24]
    speed = step.mean(dim=1) * fps                             # [T-1]
    return torch.cat([speed[:1], speed])                       # pad to T


def window_slope(err: np.ndarray) -> float:
    """Robust local growth estimate: linear-regression slope * window length."""
    n = len(err)
    if n < 3:
        return float(err[-1] - err[0])
    t = np.arange(n)
    slope, _ = np.polyfit(t, err, 1)
    return float(slope * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--layout', default='wrists_shanks_waist')
    ap.add_argument('--amass_dir', default=None,
                     help='Default: base_mobileposer/data/no_head_5imu_surface_processed/<layout>')
    ap.add_argument('--min_frames', type=int, default=1800)   # 60s @ 30fps
    ap.add_argument('--max_seqs', type=int, default=0)        # 0 = all held-out
    ap.add_argument('--window_s', type=float, default=2.0)    # analysis window length
    ap.add_argument('--drift', type=float, default=0.5)       # deg/sqrt(s), consumer MEMS
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default='low_velocity_yaw_drift_results')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    fps = 30
    device = args.device or model_config.device
    amass_dir = Path(args.amass_dir) if args.amass_dir else (
        _CODE_DIR / 'base_mobileposer' / 'data' / 'no_head_5imu_surface_processed' / args.layout)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Device        : {device}')
    print(f'AMASS dir     : {amass_dir}')
    print(f'Loading model : {args.model}')
    model = load_model(args.model).to(device)
    model.eval()

    all_seqs = load_long_sequences(args.min_frames, 0, amass_dir=amass_dir)
    seqs = [s for s in all_seqs if s['source'].startswith(HELD_OUT_PREFIXES)]
    if args.max_seqs > 0:
        seqs = seqs[:args.max_seqs]
    print(f'Held-out sequences (from {HELD_OUT_PREFIXES}): {len(seqs)}')
    if not seqs:
        print('No held-out sequences found — check --amass_dir / HELD_OUT_PREFIXES.')
        return

    W = int(round(args.window_s * fps))
    rows = []              # pooled window records: (seq_idx, mean_speed, growth_clean, growth_noisy, delta)
    per_seq_deltas = {}    # seq_idx -> list of (mean_speed, delta)
    yaw_err_end_clean, yaw_err_end_noisy = [], []

    for si, seq in enumerate(seqs):
        gt_pose = seq['pose']            # [T,24,3,3] local; root local == root global
        acc, ori = seq['acc'], seq['ori']
        T = gt_pose.shape[0]

        # --- clean condition ---
        imu_clean = prepare_imu(acc, ori, amass.acc_scale)
        pose_clean = run_inference(model, imu_clean, device)[:T]

        # --- noisy condition (independent draw per sequence, seeded for repro) ---
        torch.manual_seed(args.seed + si)
        ori_n, acc_n = add_imu_noise(ori, acc, fps=fps, drift_deg_per_sqrt_s=args.drift)
        imu_noisy = prepare_imu(acc_n, ori_n, amass.acc_scale)
        pose_noisy = run_inference(model, imu_noisy, device)[:T]

        yaw_gt = extract_yaw_deg(gt_pose[:, 0])
        yaw_clean = extract_yaw_deg(pose_clean[:, 0])
        yaw_noisy = extract_yaw_deg(pose_noisy[:, 0])
        err_clean = wrapped_abs_diff_deg(yaw_clean, yaw_gt).numpy()
        err_noisy = wrapped_abs_diff_deg(yaw_noisy, yaw_gt).numpy()
        yaw_err_end_clean.append(float(err_clean[-1]))
        yaw_err_end_noisy.append(float(err_noisy[-1]))

        speed = body_speed_deg_s(gt_pose, fps).numpy()

        n_win = T // W
        seq_records = []
        for wi in range(n_win):
            sl = slice(wi * W, (wi + 1) * W)
            mean_speed = float(speed[sl].mean())
            g_clean = window_slope(err_clean[sl])
            g_noisy = window_slope(err_noisy[sl])
            delta = g_noisy - g_clean
            rows.append((si, mean_speed, g_clean, g_noisy, delta))
            seq_records.append((mean_speed, delta))
        per_seq_deltas[si] = seq_records

        if (si + 1) % 10 == 0 or si == len(seqs) - 1:
            print(f'  [{si+1}/{len(seqs)}] {seq["source"]}  '
                  f'end_err clean={err_clean[-1]:.1f} deg  noisy={err_noisy[-1]:.1f} deg')

    rows = np.array([(r[1], r[2], r[3], r[4]) for r in rows])  # speed, g_clean, g_noisy, delta
    speed_all, g_clean_all, g_noisy_all, delta_all = rows.T

    print(f'\nTotal windows: {len(rows)}  (window = {args.window_s}s, {W} frames)')
    print(f'Mean end-of-sequence yaw error — clean: {np.mean(yaw_err_end_clean):.2f} deg  '
          f'noisy: {np.mean(yaw_err_end_noisy):.2f} deg  '
          f'(positive control: noise should clearly increase error)')

    # --- Spearman correlation: speed vs delta ---
    rho, p_rho = stats.spearmanr(speed_all, delta_all)
    print(f'\nSpearman(speed, delta) = {rho:.4f}  (p={p_rho:.3e})')
    print('  Hypothesis predicts rho < 0 (low speed -> larger noise-driven yaw-error growth).')

    # --- Quartile split ---
    q1, q3 = np.percentile(speed_all, [25, 75])
    low = delta_all[speed_all <= q1]
    high = delta_all[speed_all >= q3]
    u_stat, p_u = stats.mannwhitneyu(low, high, alternative='greater')
    print(f'\nLow-speed windows  (<=P25={q1:.2f} deg/s): n={len(low)}  '
          f'mean delta={low.mean():.3f}  median={np.median(low):.3f}')
    print(f'High-speed windows (>=P75={q3:.2f} deg/s): n={len(high)}  '
          f'mean delta={high.mean():.3f}  median={np.median(high):.3f}')
    print(f'Mann-Whitney U (low > high): U={u_stat:.1f}  p={p_u:.3e}')

    # --- Per-sequence paired test (median split within each sequence) ---
    paired_diffs = []
    for si, recs in per_seq_deltas.items():
        if len(recs) < 4:
            continue
        spd = np.array([r[0] for r in recs])
        dlt = np.array([r[1] for r in recs])
        med = np.median(spd)
        lo = dlt[spd <= med]
        hi = dlt[spd > med]
        if len(lo) >= 2 and len(hi) >= 2:
            paired_diffs.append(lo.mean() - hi.mean())
    paired_diffs = np.array(paired_diffs)
    if len(paired_diffs) >= 5:
        w_stat, p_w = stats.wilcoxon(paired_diffs, alternative='greater')
        print(f'\nPer-sequence paired (low-half delta − high-half delta), n_seq={len(paired_diffs)}:')
        print(f'  mean diff={paired_diffs.mean():.3f}  Wilcoxon W={w_stat:.1f}  p={p_w:.3e}')
    else:
        print(f'\nToo few sequences with enough windows for paired test (n={len(paired_diffs)}).')

    # --- save raw data + plot ---
    np.savez(out_dir / 'raw_data.npz', speed=speed_all, g_clean=g_clean_all,
             g_noisy=g_noisy_all, delta=delta_all, paired_diffs=paired_diffs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(speed_all, delta_all, s=8, alpha=0.4)
    axes[0].set_xlabel('Mean body angular speed in window (deg/s)')
    axes[0].set_ylabel('Delta yaw-error growth (noisy - clean, deg per window)')
    axes[0].set_title(f'Spearman rho={rho:.3f}  p={p_rho:.2e}')
    axes[0].grid(alpha=0.3)

    axes[1].boxplot([low, high], labels=['low speed\n(<=P25)', 'high speed\n(>=P75)'])
    axes[1].set_ylabel('Delta yaw-error growth (deg per window)')
    axes[1].set_title(f'Mann-Whitney p={p_u:.2e}')
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_dir / 'low_velocity_yaw_drift.png', dpi=150)
    print(f'\nSaved: {out_dir / "low_velocity_yaw_drift.png"}')
    print(f'Saved: {out_dir / "raw_data.npz"}')


if __name__ == '__main__':
    main()
