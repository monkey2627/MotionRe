"""
Temporal rotation and translation drift evaluation for PNP — lumbar-first.

PNP (SIGGRAPH 2024) is a fixed 6-sensor method with physics-based optimization.
It requires acceleration, angular velocity, and orientation per sensor per frame,
processed frame-by-frame through a stateful RNN.

Sensor layout (matches base_mobileposer .pt data):
    0 → joint 18  left wrist     3 → joint  2  right hip
    1 → joint 19  right wrist    4 → joint 15  head
    2 → joint  1  left hip       5 → joint  0  pelvis (root / RIR)

Angular velocity is estimated by numerical differentiation of orientation:
    Ω[t] = R[t]^T · dR/dt  (antisymmetric part gives ω)

Gravity (0, −9.8, 0) in world frame is added to acceleration because a real
IMU measures linear acceleration + gravitational force, while our synthesised
AMASS data contains only linear acceleration.

Run from PNP/:
    python evaluate_drift.py \
        --amass_dir ../base_mobileposer/data/processed_datasets \
        --min_frames 900 --max_seqs 3 --max_seconds 30   # smoke test

    python evaluate_drift.py \
        --min_frames 1800 --max_seqs 30                  # full benchmark
"""

import sys
import os
import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── PNP internals ─────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent       # PNP/
sys.path.insert(0, str(_SCRIPT_DIR))

from net import PNP
import articulate as art

# ── Shared evaluation constants / utilities (identical across all methods) ────
_CODE_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_CODE_DIR))
from drift_eval_common import (
    PRIMARY_SEGMENTS, SECONDARY_SEGMENTS, SEGMENTS, SEG_EN,
    LUMBAR_JOINTS, FPS, SENSOR_TO_JOINT, DATA_PATH,
    load_long_sequences, angle_between_rotmats, moving_average, add_imu_noise,
)

SMPL_KINTREE = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(4,7),(5,8),(7,10),(8,11),
    (3,6),(6,9),(9,12),(9,13),(9,14),(12,15),
    (13,16),(14,17),(16,18),(17,19),(18,20),(19,21),(20,22),(21,23),
]
_LUMBAR_BONES = {(0,3),(3,6),(6,9)}

# PNP is fixed 6-sensor — single "combo"
PNP_SENSOR_INDICES = [0, 1, 2, 3, 4, 5]
FK_SENSOR_INDICES  = [0, 1, 2, 3, 5]       # no-head FK baseline

_SMPL_FILE = str(_SCRIPT_DIR / 'models' / 'SMPL_male.pkl')


# angle_between_rotmats, moving_average → imported from drift_eval_common

def compute_angular_velocity(ori: torch.Tensor, fps: int = FPS) -> torch.Tensor:
    """Numerical differentiation: ω = vee(R^T · dR/dt).  ori: [T, 6, 3, 3]"""
    T = ori.shape[0]
    w = torch.zeros_like(ori[..., 0])           # [T, 6, 3]

    for t in range(T):
        if t == 0:
            dR = (ori[1] - ori[0]) * fps
        elif t == T - 1:
            dR = (ori[-1] - ori[-2]) * fps
        else:
            dR = (ori[t + 1] - ori[t - 1]) * (fps / 2.0)

        Omega = ori[t].transpose(-1, -2) @ dR   # [6, 3, 3] skew-symmetric
        w[t, :, 0] = Omega[:, 2, 1]
        w[t, :, 1] = Omega[:, 0, 2]
        w[t, :, 2] = Omega[:, 1, 0]

    return w


# load_long_sequences → imported from drift_eval_common


# ─────────────────────────────────────────────────────────────────────────────
# FK baseline
# ─────────────────────────────────────────────────────────────────────────────

def fk_baseline(ori: torch.Tensor, combo_indices: list,
                bodymodel) -> torch.Tensor:
    T      = ori.shape[0]
    parent = bodymodel.parent
    sensor_joints = {SENSOR_TO_JOINT[5]}
    for s in combo_indices:
        if s != 4:
            sensor_joints.add(SENSOR_TO_JOINT[s])
    R_global = torch.eye(3).view(1, 1, 3, 3).expand(T, 24, -1, -1).clone()
    R_global[:, SENSOR_TO_JOINT[5]] = ori[:, 5]
    for s in combo_indices:
        if s != 4:
            R_global[:, SENSOR_TO_JOINT[s]] = ori[:, s]
    for j in range(1, 24):
        if j not in sensor_joints:
            R_global[:, j] = R_global[:, parent[j]]
    return bodymodel.inverse_kinematics_R(R_global.view(T, -1)).view(T, 24, 3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# PNP inference
# ─────────────────────────────────────────────────────────────────────────────

_GRAVITY = torch.tensor([0., -9.8, 0.])   # world-frame gravity (SMPL Y-up)


@torch.no_grad()
def eval_pnp(model, acc: torch.Tensor, ori: torch.Tensor,
             gt_pose: torch.Tensor, gt_tran: torch.Tensor, device) -> tuple:
    """
    acc : [T, 6, 3]  linear acceleration (synthesised, no gravity)
    ori : [T, 6, 3, 3]  global rotation matrices
    Returns: rot_err [T, 24] degrees, tran_err [T] metres
    """
    T   = gt_pose.shape[0]
    acc_g = (acc + _GRAVITY).to(device)    # add gravity (real-IMU convention)
    ori_d = ori.to(device)
    w     = compute_angular_velocity(ori, FPS).to(device)

    model.rnn_initialize()                 # reset stateful RNN

    pose_list, tran_list = [], []
    for t in range(T):
        p, tr = model.forward_frame(acc_g[t], w[t], ori_d[t])
        pose_list.append(p.detach().cpu())
        tran_list.append(tr.detach().cpu())
    pose_pred = torch.stack(pose_list)   # [T, 24, 3, 3] on CPU
    tran_pred = torch.stack(tran_list)   # [T, 3] on CPU

    rot_err  = angle_between_rotmats(pose_pred, gt_pose)
    tran_err = ((tran_pred - tran_pred[:1]) - (gt_tran - gt_tran[:1])).norm(dim=-1)

    return rot_err, tran_err


def eval_fk(ori, gt_pose, combo_indices, bodymodel):
    T = gt_pose.shape[0]
    pose_fk = fk_baseline(ori, combo_indices, bodymodel)
    return angle_between_rotmats(pose_fk[:T], gt_pose)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(sequences, model, bodymodel, device, max_frames: int,
                 out_dir: str, seq_callback=None) -> dict:
    _CKPT = os.path.join(out_dir, '.eval_ckpt_pnp.npz')

    rot_sum  = np.zeros((max_frames, 24))
    tran_sum = np.zeros(max_frames)
    fk_sum   = np.zeros((max_frames, 24))
    count    = np.zeros(max_frames)
    start_idx = 0

    if os.path.exists(_CKPT):
        ck = np.load(_CKPT)
        rot_sum   = ck['rot_sum']
        tran_sum  = ck['tran_sum']
        fk_sum    = ck['fk_sum']
        count     = ck['count']
        start_idx = int(ck['seqs_done'])
        print(f"  [Resume] checkpoint loaded: {start_idx}/{len(sequences)} sequences done.")

    for idx, seq in enumerate(tqdm.tqdm(sequences, desc='Evaluating')):
        if idx < start_idx:
            continue
        T       = min(seq['pose'].shape[0], max_frames)
        gt_pose = seq['pose'][:T]
        gt_tran = seq['tran'][:T]
        acc     = seq['acc'][:T]
        ori     = seq['ori'][:T]

        try:
            rot_ml, tran_ml = eval_pnp(model, acc, ori, gt_pose, gt_tran, device)
        except Exception as e:
            print(f"\n  Warning: skipped {seq['source']} — {e}")
        else:
            rot_fk = eval_fk(ori, gt_pose, FK_SENSOR_INDICES, bodymodel)
            rot_sum[:T]  += rot_ml.numpy()
            tran_sum[:T] += tran_ml.numpy()
            fk_sum[:T]   += rot_fk.numpy()
            count[:T]    += 1.0

        np.savez(_CKPT, rot_sum=rot_sum, tran_sum=tran_sum, fk_sum=fk_sum,
                 count=count, seqs_done=idx + 1)

        if seq_callback is not None:
            seq_callback(idx, seq)

    if os.path.exists(_CKPT):
        os.remove(_CKPT)

    valid      = count > 0
    safe_cnt   = np.maximum(count[:, None], 1)
    rot_avg    = np.where(valid[:, None], rot_sum  / safe_cnt, 0.0)
    tran_avg   = np.where(valid,          tran_sum / np.maximum(count, 1), 0.0)
    fk_avg     = np.where(valid[:, None], fk_sum   / safe_cnt, 0.0)

    return {'rot': rot_avg, 'tran': tran_avg, 'fk_rot': fk_avg, 'count': count,
            'n_sensors': 6, 'n_seqs': int(count[0]) if count[0] > 0 else 0}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting  (identical structure to other evaluate_drift scripts)
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(res, max_frames, fps, out_dir):
    t = np.arange(max_frames) / fps
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax_i, (seg, joints) in enumerate(SEGMENTS.items()):
        ax = axes.flatten()[ax_i]
        is_p = seg in PRIMARY_SEGMENTS
        ax.plot(t, moving_average(res['rot'][:, joints].mean(1)),
                label='PNP (6 sensors)', color='#C62828', lw=2.0 if is_p else 1.4)
        ax.plot(t, moving_average(res['fk_rot'][:, joints].mean(1)),
                label='FK baseline (5 sensors)', color='gray', ls='--', lw=1.2)
        ax.set_title(f"{'[*] ' if is_p else ''}{SEG_EN[seg]}", fontsize=11,
                     fontweight='bold' if is_p else 'normal')
        ax.set_ylabel('Angle error (deg)', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_frames / fps); ax.set_ylim(bottom=0)
    fig.suptitle('PNP rotation drift by body segment  [*]=lumbar-rehab primary  (fixed 6 sensors)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_drift_timeseries.png')
    plt.savefig(path, dpi=150); plt.close(); print(f"Saved: {path}")


def plot_combo_comparison(res, max_frames, fps, checkpoint_s, out_dir):
    fp    = min(int(checkpoint_s * fps), max_frames - 1)
    segs  = list(SEGMENTS.keys())
    x     = np.arange(len(segs)); bw = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - bw/2, [res['rot'][fp, j].mean() for j in SEGMENTS.values()],
           bw, label='PNP (6 sensors)', color='#C62828', alpha=0.85)
    ax.bar(x + bw/2, [res['fk_rot'][fp, j].mean() for j in SEGMENTS.values()],
           bw, label='FK baseline (5 sensors)', color='gray', alpha=0.75)
    for si, s in enumerate(segs):
        if s in PRIMARY_SEGMENTS:
            ax.axvspan(si - 0.48, si + 0.48, alpha=0.06, color='gold', zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"[*]{SEG_EN[s]}" if s in PRIMARY_SEGMENTS else SEG_EN[s]
                        for s in segs], fontsize=9, rotation=10)
    ax.set_ylabel('Mean angle error (deg)', fontsize=11)
    ax.set_title(f'PNP vs FK baseline at {checkpoint_s}s  --  [*]=lumbar primary',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3); ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_combo_comparison.png')
    plt.savefig(path, dpi=150); plt.close(); print(f"Saved: {path}")


def plot_sensor_count(res, max_frames, fps, checkpoint_s, out_dir):
    fp = min(int(checkpoint_s * fps), max_frames - 1)
    dip = float(res['rot'][fp, LUMBAR_JOINTS].mean())
    fk  = float(res['fk_rot'][fp, LUMBAR_JOINTS].mean())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([6], [dip], 0.4, label=f'PNP  {dip:.1f} deg', color='#C62828', alpha=0.85)
    ax.scatter([6], [dip], color='#C62828', s=80, zorder=4)
    ax.axhline(fk, color='black', ls='--', lw=1.5, label=f'FK baseline  {fk:.1f} deg')
    ax.set_xticks([6]); ax.set_xticklabels(['6 sensors\n(PNP)'])
    ax.set_xlabel('Number of sensors', fontsize=12)
    ax.set_ylabel('Lumbar error (deg)', fontsize=12)
    ax.set_title(f'Sensor count vs lumbar accuracy at {checkpoint_s}s\n(PNP: fixed 6-sensor)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0); ax.set_xlim(4, 8)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_sensor_count_vs_lumbar.png')
    plt.savefig(path, dpi=150); plt.close(); print(f"Saved: {path}")


def plot_translation(res, max_frames, fps, out_dir):
    t = np.arange(max_frames) / fps
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, moving_average(res['tran']),
            label='PNP (6 sensors)', color='#C62828', lw=1.8)
    ax.set_title('PNP global translation drift (fixed 6 sensors)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Position error (m)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_frames / fps); ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_translation_drift.png')
    plt.savefig(path, dpi=150); plt.close(); print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(res, max_frames, fps):
    cps    = sorted({cp for cp in [30, 60, 90, int(max_frames / fps)]
                     if cp <= max_frames / fps})
    segs   = list(SEGMENTS.keys())
    flags  = ['*' if s in PRIMARY_SEGMENTS else ' ' for s in segs]
    col_w  = 8
    header = f"{'Method':<14} {'Sensors':>9} {'Time':>5}  "
    header += ''.join(f"{f}{SEG_EN[s]:>{col_w}}" for f, s in zip(flags, segs))
    header += f"  {'Lumbar':>{col_w}}  {'Tran(m)':>{col_w}}"
    sep = '-' * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for cp in cps:
        fr  = min(int(cp * fps), max_frames - 1)
        row = f"{'PNP':<14} {'6 (fixed)':>9} {cp:>4}s  "
        for ji in SEGMENTS.values():
            row += f"  {float(res['rot'][fr, ji].mean()):>{col_w}.1f}"
        lumbar = float(res['rot'][fr, LUMBAR_JOINTS].mean())
        tran   = float(res['tran'][fr])
        row   += f"  {lumbar:>{col_w}.1f}  {tran:>{col_w}.3f}"
        print(row)
    print()
    row_fk = f"{'FK-baseline':<14} {'5 (no hd)':>9} {'end':>5}  "
    for ji in SEGMENTS.values():
        row_fk += f"  {float(res['fk_rot'][-1, ji].mean()):>{col_w}.1f}"
    fk_l = float(res['fk_rot'][-1, LUMBAR_JOINTS].mean())
    row_fk += f"  {fk_l:>{col_w}.1f}  {'N/A':>{col_w}}"
    print(row_fk)
    print(sep)
    print("*=lumbar primary  Lumbar=joints[1,2,3,6,9] mean  Tran=root-relative drift")
    print("Note: PNP is a fixed 6-sensor method (incl. head). No sparse-sensor evaluation.")


# ─────────────────────────────────────────────────────────────────────────────
# Video (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_skel(ax, joints, color, title, lumbar_err=None):
    ax.cla()
    j = joints - joints[0]
    for (a, b) in SMPL_KINTREE:
        is_lmb = (a, b) in _LUMBAR_BONES or (b, a) in _LUMBAR_BONES
        c, lw  = ('#FF6F00', 3.5) if is_lmb else (color, 1.8)
        ax.plot([j[a, 0], j[b, 0]], [j[a, 1], j[b, 1]], '-', color=c, lw=lw)
    ax.scatter(j[:, 0], j[:, 1], c=color, s=18, zorder=5)
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.25, 1.85)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title + (f'\nlumbar: {lumbar_err:.1f}°' if lumbar_err else ''),
                 fontsize=9, pad=3)


def generate_video(seq, model, bodymodel, device, fps, out_dir,
                   max_seconds=30, render_fps=10, seq_idx=0):
    try:
        from matplotlib.animation import FFMpegWriter
    except Exception as e:
        print(f"  Skipped (FFMpegWriter): {e}"); return

    stride = max(1, round(fps / render_fps))
    T      = min(seq['pose'].shape[0], int(max_seconds * fps))
    gt_pose, gt_tran = seq['pose'][:T], seq['tran'][:T]
    acc, ori = seq['acc'][:T], seq['ori'][:T]

    model.rnn_initialize()
    acc_g = (acc + _GRAVITY).to(device)
    ori_d = ori.to(device)
    w     = compute_angular_velocity(ori, fps).to(device)
    pose_ml_list, tran_ml_list = [], []
    for t in range(T):
        p, tr = model.forward_frame(acc_g[t], w[t], ori_d[t])
        pose_ml_list.append(p.detach().cpu())
        tran_ml_list.append(tr.detach().cpu())
    pose_ml = torch.stack(pose_ml_list)
    tran_ml = torch.stack(tran_ml_list)
    rot_ml = angle_between_rotmats(pose_ml, gt_pose)
    pose_fk = fk_baseline(ori, FK_SENSOR_INDICES, bodymodel)[:T]

    with torch.no_grad():
        _, gt_j  = bodymodel.forward_kinematics(gt_pose,  tran=gt_tran)
        _, ml_j  = bodymodel.forward_kinematics(pose_ml, tran=tran_ml)
        _, fk_j  = bodymodel.forward_kinematics(pose_fk,  tran=gt_tran)
    gt_j, ml_j, fk_j = gt_j.numpy(), ml_j.numpy(), fk_j.numpy()

    lumbar_ml = angle_between_rotmats(pose_ml.cpu()[:, LUMBAR_JOINTS],
                                      gt_pose[:, LUMBAR_JOINTS]).mean(-1).numpy()
    lumbar_fk = angle_between_rotmats(pose_fk[:, LUMBAR_JOINTS],
                                      gt_pose[:, LUMBAR_JOINTS]).mean(-1).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.patch.set_facecolor('#111122')
    for ax in axes: ax.set_facecolor('#111122')
    vp = os.path.join(out_dir, f'video_pnp_6s_{seq_idx:04d}.mp4')
    writer = FFMpegWriter(fps=render_fps, metadata={'title': 'drift-pnp-6s'})
    print(f"  Writing {len(range(0,T,stride))} frames → {vp}")
    with writer.saving(fig, vp, dpi=100):
        for t in range(0, T, stride):
            _draw_skel(axes[0], gt_j[t],  '#43A047', 'Ground Truth')
            _draw_skel(axes[1], ml_j[t],  '#C62828', 'PNP (6 sensors)', lumbar_ml[t])
            _draw_skel(axes[2], fk_j[t],  '#E53935', 'FK baseline', lumbar_fk[t])
            fig.suptitle(f't = {t/fps:.1f}s    orange = lumbar spine',
                         color='white', fontsize=11)
            writer.grab_frame()
    plt.close(fig); print(f"  Saved: {vp}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='PNP drift evaluation — lumbar focus.')
    parser.add_argument('--combos', nargs='+', default=['all'],
                        help='Ignored — PNP is a fixed 6-sensor method.')
    parser.add_argument('--amass_dir', default=None)
    parser.add_argument('--min_frames', type=int, default=1800)
    parser.add_argument('--max_seqs',   type=int, default=10)
    parser.add_argument('--max_seconds', type=int, default=120)
    parser.add_argument('--compare_at', type=int, default=60)
    parser.add_argument('--out_dir', default='drift_results')
    parser.add_argument('--video_seconds', type=int, default=30)
    parser.add_argument('--video_fps',     type=int, default=10)
    parser.add_argument('--no_video', action='store_true',
                        help='Skip video generation (metrics and figures only)')
    parser.add_argument('--no_noise', action='store_true',
                        help='Disable IMU noise simulation (use clean synthetic data)')
    parser.add_argument('--drift', type=float, default=0.5,
                        help='Gyro random-walk rate °/√s (default 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for reproducible noise (default 42)')
    args = parser.parse_args()

    fps        = FPS
    max_frames = args.max_seconds * fps
    amass_dir  = (Path(args.amass_dir) if args.amass_dir
                  else _SCRIPT_DIR.parent / 'base_mobileposer'
                       / 'data' / 'processed_datasets')
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 62)
    print('  PNP Drift Evaluation — lumbar-rehabilitation focus')
    print('=' * 62)
    print(f"  AMASS dir  : {amass_dir}")
    print(f"  Device     : {device}")
    print(f"  Min frames : {args.min_frames} ({args.min_frames/fps:.0f}s)")
    print(f"  Window     : {args.max_seconds}s  |  checkpoint: {args.compare_at}s")
    print('  NOTE: PNP is fixed 6-sensor (incl. head). --combos ignored.')
    print('  NOTE: angular velocity estimated via finite difference of orientation.')
    print('=' * 62)

    os.makedirs(args.out_dir, exist_ok=True)
    sequences = load_long_sequences(args.min_frames, args.max_seqs, amass_dir=amass_dir)
    if not sequences:
        print("No qualifying sequences. Adjust --min_frames or --amass_dir."); return

    if not args.no_noise:
        torch.manual_seed(args.seed)
        print(f"\nApplying IMU noise  drift={args.drift}°/√s  noise=0.5°  acc=0.1m/s²"
              f"  seed={args.seed}")
        print(f"  Expected drift std after 60s: {args.drift*(60**0.5):.1f}°"
              f"  | after 120s: {args.drift*(120**0.5):.1f}°")
        for seq in sequences:
            seq['ori'], seq['acc'] = add_imu_noise(
                seq['ori'], seq['acc'], fps=fps,
                drift_deg_per_sqrt_s=args.drift)
    else:
        print("\nIMU noise disabled (--no_noise).  Using clean synthetic data.")

    print('\nLoading PNP model ...')
    # PNP __init__ loads weights from data/weights/PNP/weights.pt automatically
    model = PNP().eval().to(device)
    bodymodel = art.ParametricModel(_SMPL_FILE)

    def _video_cb(i, seq):
        print(f"  [{i+1}/{len(sequences)}] video  seq={seq['source']}")
        try:
            generate_video(seq, model, bodymodel, device, fps, args.out_dir,
                           args.video_seconds, args.video_fps, seq_idx=i)
        except Exception as e:
            print(f'    Video failed: {e}')

    print(f"\nRunning evaluation over {len(sequences)} sequences ...")
    res = evaluate_all(sequences, model, bodymodel, device, max_frames, args.out_dir,
                       seq_callback=_video_cb if not args.no_video else None)
    print(f"  Evaluated {res['n_seqs']} sequences.")

    print('\nGenerating figures ...')
    plot_timeseries(res, max_frames, fps, args.out_dir)
    plot_combo_comparison(res, max_frames, fps, args.compare_at, args.out_dir)
    plot_sensor_count(res, max_frames, fps, args.compare_at, args.out_dir)
    plot_translation(res, max_frames, fps, args.out_dir)

    print_summary(res, max_frames, fps)

    np.savez(os.path.join(args.out_dir, 'drift_data.npz'),
             pnp_6s_rot=res['rot'], pnp_6s_tran=res['tran'],
             pnp_6s_fk_rot=res['fk_rot'], fps=fps, combos=['pnp_6s'])
    import sys as _sys
    _sys.path.insert(0, str(_DIR.parent))
    from benchmarks.standard_results import write_standard_result
    write_standard_result(
        Path(args.out_dir), 'pnp', 'drift', res['rot'], res['count'], fps,
        6, [0, 1, 2, 3, 4, 5], res['tran'],
    )
    print(f"\nAll outputs in: {os.path.abspath(args.out_dir)}/")


if __name__ == '__main__':
    main()
