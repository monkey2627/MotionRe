#!/usr/bin/env python3
"""
DynaIP (CVPR 2024) drift benchmark.
Mirrors base_mobileposer/mobileposer/evaluate_drift.py — same metrics, segment
definitions, and plot layout.

Run from DynaIP/ directory on the server:
    python evaluate_drift.py \
        --min_frames 900 --max_seqs 3 --max_seconds 30    # smoke test
    python evaluate_drift.py \
        --min_frames 1800 --max_seqs 30                   # full run
"""
import sys
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import articulate as art
import utils.config as cfg
from model.model import Poser

# ── Shared evaluation constants / utilities (identical across all methods) ────
_CODE_DIR = Path(_SCRIPT_DIR).parent
sys.path.insert(0, str(_CODE_DIR))
from drift_eval_common import (
    PRIMARY_SEGMENTS, SECONDARY_SEGMENTS, SEGMENTS, SEG_EN,
    LUMBAR_JOINTS, FPS, SENSOR_TO_JOINT, DATA_PATH,
    load_long_sequences, angle_between_rotmats, moving_average,
)

# Reorder our sensors [L_wrist, R_wrist, L_hip, R_hip, Head, Pelvis]
# to DynaIP's expected slots [Root, LeftLowerLeg, RightLowerLeg, Head, LeftForeArm, RightForeArm]
DYNAIP_REORDER = [5, 2, 3, 4, 0, 1]

# FK baseline: {sensor_idx → smpl_joint}  (head excluded, same as MobilePoser)
FK_SENSOR_MAP = {0: 18, 1: 19, 2: 1, 3: 2, 5: 0}

SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

RESULTS_DIR = Path(_SCRIPT_DIR) / 'drift_results'


# ─── DynaIP inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def eval_dynaip(net, acc: torch.Tensor, ori: torch.Tensor,
                gt_pose: torch.Tensor, device: str) -> torch.Tensor:
    """
    Returns rot_err [T, 24] (degrees), LOCAL rotation comparison.

    acc:     [T, 6, 3]
    ori:     [T, 6, 3, 3] global rotation matrices
    gt_pose: [T, 24, 3, 3] LOCAL ground-truth rotation matrices
    """
    T = acc.shape[0]

    ori_re   = ori[:, DYNAIP_REORDER]             # [T, 6, 3, 3]
    acc_re   = acc[:, DYNAIP_REORDER]             # [T, 6, 3]
    ori_flat = ori_re.reshape(T, 6, 9)            # [T, 6, 9]
    imu = torch.cat([ori_flat, acc_re], dim=-1).unsqueeze(0).to(device)  # [1, T, 6, 12]

    v_init = torch.zeros(1, 6, 3, device=device)
    t6d    = torch.tensor([[1., 0., 0., 0., 1., 0.]], device=device)
    p_init = t6d.unsqueeze(0).expand(1, 11, -1).contiguous()             # [1, 11, 6]

    net.eval()
    _, glb_smpl = net.predict(imu, v_init, p_init)    # [T, 24, 3, 3] global, on CPU

    parents = torch.tensor(SMPL_PARENTS)
    R_local = art.math.inverse_kinematics_R(glb_smpl, parents)           # [T, 24, 3, 3]

    return angle_between_rotmats(R_local, gt_pose)    # [T, 24]


def eval_fk(ori: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
    """Sensor-orientation FK baseline → rot_err [T, 24]."""
    T = ori.shape[0]
    pose_local = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, -1, -1).clone()
    root_ori = ori[:, 5]
    for s_idx, j_idx in FK_SENSOR_MAP.items():
        pose_local[:, j_idx] = root_ori.transpose(-1, -2) @ ori[:, s_idx]
    return angle_between_rotmats(pose_local, gt_pose)


# ─── evaluation loop ───────────────────────────────────────────────────────────

def evaluate_all(sequences, net, max_frames: int, device: str) -> dict:
    rot_sum  = np.zeros((max_frames, 24))
    fk_sum   = np.zeros((max_frames, 24))
    count    = np.zeros(max_frames)

    for seq in tqdm(sequences, desc='Evaluating'):
        T       = min(seq['pose'].shape[0], max_frames)
        gt_pose = seq['pose'][:T]
        acc     = seq['acc'][:T]
        ori     = seq['ori'][:T]

        try:
            rot_dip = eval_dynaip(net, acc, ori, gt_pose, device)
        except Exception as e:
            print(f'\n  Warning: skipped {seq["source"]} — {e}')
            continue

        rot_fk = eval_fk(ori, gt_pose)

        rot_sum[:T] += rot_dip.numpy()
        fk_sum[:T]  += rot_fk.numpy()
        count[:T]   += 1.0

    valid      = count > 0
    safe_count = np.maximum(count[:, None], 1)
    return {
        'rot':       np.where(valid[:, None], rot_sum / safe_count, 0.0),
        'fk_rot':    np.where(valid[:, None], fk_sum  / safe_count, 0.0),
        'tran':      np.zeros(max_frames),   # DynaIP has no translation output
        'n_sensors': 6,
        'n_seqs':    int(count[0]) if count[0] > 0 else 0,
    }


# ─── plotting ──────────────────────────────────────────────────────────────────

def plot_timeseries(res: dict, max_frames: int, fps: int, out_dir: str):
    """Figure 1: per-segment angular error over time."""
    t       = np.arange(max_frames) / fps
    max_sec = max_frames / fps

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for ax_i, (seg_name, joint_idx) in enumerate(SEGMENTS.items()):
        ax         = axes[ax_i]
        is_primary = seg_name in PRIMARY_SEGMENTS

        y_dip = moving_average(res['rot'][:, joint_idx].mean(axis=1))
        y_fk  = moving_average(res['fk_rot'][:, joint_idx].mean(axis=1))

        lw = 2.0 if is_primary else 1.2
        ax.plot(t, y_dip, label='DynaIP (6s)', color='#1565C0', linewidth=lw)
        ax.axhline(y_fk[max_frames // 2], color='gray', linestyle='--',
                   linewidth=1.2, label='FK baseline')

        prefix = '[*] ' if is_primary else ''
        ax.set_title(f'{prefix}{SEG_EN[seg_name]}',
                     fontsize=11, fontweight='bold' if is_primary else 'normal')
        ax.set_ylabel('Angle error (deg)', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_sec)
        ax.set_ylim(bottom=0)

    fig.suptitle('DynaIP — Rotation drift by body segment  [*]=lumbar-rehab primary',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_drift_timeseries.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_segment_bars(res: dict, max_frames: int, fps: int,
                      checkpoint_s: float, out_dir: str):
    """Figure 2: DynaIP vs FK bar chart at checkpoint time per segment."""
    fps_cp    = min(int(checkpoint_s * fps), max_frames - 1)
    seg_names = list(SEGMENTS.keys())
    x = np.arange(len(seg_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    dip_vals = [res['rot'][fps_cp, joint_idx].mean()    for joint_idx in SEGMENTS.values()]
    fk_vals  = [res['fk_rot'][fps_cp, joint_idx].mean() for joint_idx in SEGMENTS.values()]

    ax.bar(x - w / 2, dip_vals, w, label='DynaIP (6s)',  color='#1565C0', alpha=0.85)
    ax.bar(x + w / 2, fk_vals,  w, label='FK baseline',  color='salmon',  alpha=0.85)

    for si, seg_name in enumerate(seg_names):
        if seg_name in PRIMARY_SEGMENTS:
            ax.axvspan(si - 0.48, si + 0.48, alpha=0.07, color='gold', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'[*]{SEG_EN[s]}' if s in PRIMARY_SEGMENTS else SEG_EN[s] for s in seg_names],
        fontsize=9, rotation=10)
    ax.set_ylabel('Mean angle error (deg)', fontsize=11)
    ax.set_title(f'DynaIP vs FK at {checkpoint_s}s  [*]=lumbar primary', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_segment_bars.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_lumbar_detail(res: dict, max_frames: int, fps: int,
                       checkpoint_s: float, out_dir: str):
    """Figure 3: detailed lumbar joint breakdown at checkpoint."""
    fps_cp = min(int(checkpoint_s * fps), max_frames - 1)
    t = np.arange(max_frames) / fps

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-joint error curves
    ax = axes[0]
    lumbar_names = {1: 'L_Hip(j1)', 2: 'R_Hip(j2)', 3: 'Spine1(j3)',
                    6: 'Spine2(j6)', 9: 'Spine3(j9)'}
    colors = ['#1565C0', '#C62828', '#2E7D32', '#F57F17', '#6A1B9A']
    for ji, (j, lbl) in enumerate(lumbar_names.items()):
        ax.plot(t, moving_average(res['rot'][:, j]), label=lbl, color=colors[ji], linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle error (deg)')
    ax.set_title('DynaIP — Lumbar joint error over time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_frames / fps)
    ax.set_ylim(bottom=0)

    # Right: bar chart of individual lumbar joints at checkpoint
    ax = axes[1]
    lumbar_j = list(lumbar_names.keys())
    dip_j  = [res['rot'][fps_cp, j]    for j in lumbar_j]
    fk_j   = [res['fk_rot'][fps_cp, j] for j in lumbar_j]
    x = np.arange(len(lumbar_j))
    w = 0.35
    ax.bar(x - w / 2, dip_j, w, label='DynaIP', color='#1565C0', alpha=0.85)
    ax.bar(x + w / 2, fk_j,  w, label='FK',     color='salmon',  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(list(lumbar_names.values()), fontsize=9)
    ax.set_ylabel('Angle error (deg)')
    ax.set_title(f'Lumbar joints at {checkpoint_s}s')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_lumbar_detail.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_translation(res: dict, max_frames: int, fps: int, out_dir: str):
    """Figure 4: placeholder — DynaIP does not output root translation."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5,
            'DynaIP does not predict root translation.\nTranslation drift plot unavailable.',
            ha='center', va='center', transform=ax.transAxes, fontsize=13,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_axis_off()
    ax.set_title('Translation Drift (N/A for DynaIP)')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_translation_drift.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


# ─── summary ───────────────────────────────────────────────────────────────────

def print_summary(res: dict, max_frames: int, fps: int):
    checkpoints = sorted(set(cp for cp in [10, 20, 30, int(max_frames / fps)]
                             if cp <= max_frames / fps))
    seg_names    = list(SEGMENTS.keys())
    primary_flag = ['*' if s in PRIMARY_SEGMENTS else ' ' for s in seg_names]
    col_w = 8

    hdr = f"{'Method':<14} {'t(s)':>5}  "
    hdr += ''.join(f"{f}{SEG_EN[s]:>{col_w}}" for f, s in zip(primary_flag, seg_names))
    hdr += f"  {'Lumbar':>{col_w}}  {'Tran(m)':>{col_w}}"
    sep = '─' * len(hdr)

    print(f'\n{sep}')
    print(hdr)
    print(sep)

    for cp in checkpoints:
        frame = min(int(cp * fps), max_frames - 1)
        row = f"{'DynaIP':<14} {cp:>5}  "
        for joint_idx in SEGMENTS.values():
            row += f"  {res['rot'][frame, joint_idx].mean():>{col_w}.1f}"
        lumbar = res['rot'][frame, LUMBAR_JOINTS].mean()
        row += f"  {lumbar:>{col_w}.1f}  {'N/A':>{col_w}}"
        print(row)

    print(f"\n  FK baseline:")
    frame30 = min(int(30 * fps), max_frames - 1)
    row = f"{'FK':<14} {30:>5}  "
    for joint_idx in SEGMENTS.values():
        row += f"  {res['fk_rot'][frame30, joint_idx].mean():>{col_w}.1f}"
    lumbar_fk = res['fk_rot'][frame30, LUMBAR_JOINTS].mean()
    row += f"  {lumbar_fk:>{col_w}.1f}  {'N/A':>{col_w}}"
    print(row)
    print(sep)


def save_npz(res: dict, out_path: Path):
    np.savez_compressed(str(out_path),
                        rot=res['rot'],
                        fk_rot=res['fk_rot'],
                        tran=res['tran'])
    print(f'Saved: {out_path}')


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DynaIP drift benchmark')
    parser.add_argument('--model',       default='weights/DynaIP.pth')
    parser.add_argument('--min_frames',  type=int,   default=1800)
    parser.add_argument('--max_seqs',    type=int,   default=0,
                        help='Max sequences (0 = all qualifying)')
    parser.add_argument('--max_seconds', type=float, default=120.0)
    parser.add_argument('--checkpoint_s', type=float, default=60.0,
                        help='Checkpoint time for bar charts')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    max_frames = int(args.max_seconds * FPS)
    device = args.device
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = str(RESULTS_DIR)

    print(f'Loading DynaIP weights from {args.model}')
    net = Poser().to(device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    net.eval()

    sequences = load_long_sequences(args.min_frames, args.max_seqs)
    print(f'  {len(sequences)} sequences loaded')

    res = evaluate_all(sequences, net, max_frames, device)
    print(f'\nEvaluated {res["n_seqs"]} sequences ({max_frames / FPS:.0f}s each)')

    print('\nGenerating figures...')
    plot_timeseries(res,     max_frames, FPS, out_dir)
    plot_segment_bars(res,   max_frames, FPS, args.checkpoint_s, out_dir)
    plot_lumbar_detail(res,  max_frames, FPS, args.checkpoint_s, out_dir)
    plot_translation(res,    max_frames, FPS, out_dir)

    print_summary(res, max_frames, FPS)
    save_npz(res, RESULTS_DIR / 'dynaip_drift_results.npz')
    print(f'\nAll outputs → {RESULTS_DIR}')


if __name__ == '__main__':
    main()
