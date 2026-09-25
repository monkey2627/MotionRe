"""
Final confirmation test for the "low-velocity yaw drift" hypothesis, addressing
the caveat raised after analysis_low_velocity_yaw_drift.py's null result: AMASS
clips rarely contain genuinely static "standing still" segments, so the P25
speed bucket there may not represent true quasi-static conditions.

Here we directly construct the scenario TIC / MobilePoser describe: splice an
artificial "frozen pose" hold (pose, orientation, translation held perfectly
constant — genuine zero motion) of H frames into each held-out sequence at a
fixed point t0, and compare the noise-induced yaw-error growth rate during
that hold against the growth rate during the corresponding real-motion window
of the SAME LENGTH from the unmodified sequence at the same t0.

Paired design per sequence: delta_hold - delta_normal_motion, tested with a
one-sided Wilcoxon signed-rank test across sequences (H1: hold > normal).

Run from code/ (repo root):
    conda activate mobileposer
    python analysis_frozen_hold_yaw_drift.py \
        --model base_mobileposer/checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth \
        --hold_s 8
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

from drift_eval_common import load_long_sequences, add_imu_noise
from mobileposer.config import amass, model_config
from mobileposer.utils.model_utils import load_model

from analysis_low_velocity_yaw_drift import (
    prepare_imu, run_inference, extract_yaw_deg, wrapped_abs_diff_deg,
    body_speed_deg_s, window_slope, HELD_OUT_PREFIXES,
)


def splice_hold(pose, tran, ori, acc, t0: int, H: int):
    """Insert an H-frame frozen-pose hold right after frame t0."""
    hold_pose = pose[t0:t0 + 1].repeat(H, 1, 1, 1)
    hold_tran = tran[t0:t0 + 1].repeat(H, 1)
    hold_ori  = ori[t0:t0 + 1].repeat(H, 1, 1, 1)
    hold_acc  = acc[t0:t0 + 1].repeat(H, 1, 1)
    f_pose = torch.cat([pose[:t0], hold_pose, pose[t0:]], dim=0)
    f_tran = torch.cat([tran[:t0], hold_tran, tran[t0:]], dim=0)
    f_ori  = torch.cat([ori[:t0],  hold_ori,  ori[t0:]],  dim=0)
    f_acc  = torch.cat([acc[:t0],  hold_acc,  acc[t0:]],  dim=0)
    return f_pose, f_tran, f_ori, f_acc


def yaw_err_pair(model, pose, tran, ori, acc, device, seed):
    """Run clean + noisy inference, return (err_clean, err_noisy) yaw-error arrays."""
    T = pose.shape[0]
    imu_clean = prepare_imu(acc, ori, amass.acc_scale)
    pose_clean = run_inference(model, imu_clean, device)[:T]

    torch.manual_seed(seed)
    ori_n, acc_n = add_imu_noise(ori, acc, fps=30, drift_deg_per_sqrt_s=0.5)
    imu_noisy = prepare_imu(acc_n, ori_n, amass.acc_scale)
    pose_noisy = run_inference(model, imu_noisy, device)[:T]

    yaw_gt = extract_yaw_deg(pose[:, 0])
    err_clean = wrapped_abs_diff_deg(extract_yaw_deg(pose_clean[:, 0]), yaw_gt).numpy()
    err_noisy = wrapped_abs_diff_deg(extract_yaw_deg(pose_noisy[:, 0]), yaw_gt).numpy()
    return err_clean, err_noisy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--layout', default='wrists_shanks_waist')
    ap.add_argument('--amass_dir', default=None)
    ap.add_argument('--min_frames', type=int, default=1800)
    ap.add_argument('--hold_s', type=float, default=8.0)
    ap.add_argument('--t0_frac', type=float, default=1.0 / 3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', default='frozen_hold_yaw_drift_results')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    fps = 30
    H = int(round(args.hold_s * fps))
    device = args.device or model_config.device
    amass_dir = Path(args.amass_dir) if args.amass_dir else (
        _CODE_DIR / 'base_mobileposer' / 'data' / 'no_head_5imu_surface_processed' / args.layout)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Device : {device}   Hold length: {H} frames ({args.hold_s}s)')
    model = load_model(args.model).to(device)
    model.eval()

    all_seqs = load_long_sequences(args.min_frames, 0, amass_dir=amass_dir)
    seqs = [s for s in all_seqs if s['source'].startswith(HELD_OUT_PREFIXES)]
    print(f'Held-out sequences: {len(seqs)}')

    rows = []   # (delta_hold, delta_normal, gt_speed_during_normal_window)
    for si, seq in enumerate(seqs):
        pose, tran, ori, acc = seq['pose'], seq['tran'], seq['ori'], seq['acc']
        T = pose.shape[0]
        t0 = int(T * args.t0_frac)
        if t0 + H >= T:
            continue

        # --- normal (unmodified) condition ---
        err_c_norm, err_n_norm = yaw_err_pair(model, pose, tran, ori, acc, device, args.seed + si)
        gt_speed_window = float(body_speed_deg_s(pose, fps)[t0:t0 + H].mean())

        # --- frozen-hold condition ---
        f_pose, f_tran, f_ori, f_acc = splice_hold(pose, tran, ori, acc, t0, H)
        err_c_hold, err_n_hold = yaw_err_pair(model, f_pose, f_tran, f_ori, f_acc, device, args.seed + si + 10000)

        delta_normal = window_slope(err_n_norm[t0:t0 + H]) - window_slope(err_c_norm[t0:t0 + H])
        delta_hold   = window_slope(err_n_hold[t0:t0 + H]) - window_slope(err_c_hold[t0:t0 + H])
        rows.append((delta_hold, delta_normal, gt_speed_window))

        if (si + 1) % 10 == 0 or si == len(seqs) - 1:
            print(f'  [{si+1}/{len(seqs)}] {seq["source"]}  '
                  f'delta_hold={delta_hold:+.3f}  delta_normal={delta_normal:+.3f}  '
                  f'(normal-window GT speed={gt_speed_window:.1f} deg/s)')

    rows = np.array(rows)
    delta_hold, delta_normal, gt_speed = rows[:, 0], rows[:, 1], rows[:, 2]
    paired_diff = delta_hold - delta_normal

    print(f'\nn_sequences = {len(rows)}')
    print(f'Normal-window GT speed: mean={gt_speed.mean():.1f}  median={np.median(gt_speed):.1f} deg/s '
          f'(sanity check: this should be clearly non-trivial motion, not near-zero)')
    print(f'\nmean delta_hold   = {delta_hold.mean():+.4f}  (median {np.median(delta_hold):+.4f})')
    print(f'mean delta_normal = {delta_normal.mean():+.4f}  (median {np.median(delta_normal):+.4f})')
    print(f'mean paired diff (hold - normal) = {paired_diff.mean():+.4f}  (median {np.median(paired_diff):+.4f})')

    w_stat, p_w = stats.wilcoxon(paired_diff, alternative='greater')
    print(f'\nWilcoxon signed-rank (H1: hold > normal): W={w_stat:.1f}  p={p_w:.3e}')
    frac_positive = float((paired_diff > 0).mean())
    print(f'Fraction of sequences with paired_diff > 0: {frac_positive:.2f}')

    np.savez(out_dir / 'raw_data.npz', delta_hold=delta_hold, delta_normal=delta_normal,
             gt_speed=gt_speed, paired_diff=paired_diff)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([delta_normal, delta_hold], labels=['normal motion\nwindow', 'frozen-pose\nhold window'])
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_ylabel('Noise-driven yaw-error growth (deg per window)')
    ax.set_title(f'Frozen-hold vs normal-motion window (n={len(rows)})\nWilcoxon p={p_w:.2e}')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out_dir / 'frozen_hold_comparison.png', dpi=150)
    print(f'\nSaved: {out_dir / "frozen_hold_comparison.png"}')


if __name__ == '__main__':
    main()
