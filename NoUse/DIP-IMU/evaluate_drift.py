"""
Temporal rotation drift evaluation for DIP-IMU — lumbar-rehabilitation focus.

DIP-IMU (SIGGRAPH Asia 2018) is a FIXED 6-sensor method trained with a specific
sensor set (left wrist, right wrist, left hip, right hip, head, pelvis).  It
does NOT support variable/sparse sensor configurations — there is no combo
masking in the architecture or training.

Therefore this script evaluates DIP-IMU in its native 6-sensor mode ('dip_6s')
and shows the 5-sensor FK baseline (no head) for reference.  For sparse-sensor
comparison, see base_mobileposer/mobileposer/evaluate_drift.py.

Sensor index → SMPL joint mapping (shared with MobilePoser data):
    0 → joint 18  left wrist    |  3 → joint  2  right hip
    1 → joint 19  right wrist   |  4 → joint 15  head
    2 → joint  1  left hip      |  5 → joint  0  pelvis (root)

DIP predicts 15 SMPL_MAJOR_JOINTS = [1,2,3,4,5,6,9,12,13,14,15,16,17,18,19].
Lumbar joints [1,2,3,6,9] are ALL within this set — evaluation is meaningful.

Run from DIP-IMU/:
    # smoke test (3 seqs, 30 s window)
    python evaluate_drift.py \\
        --model train_and_eval/models/<model_dir> \\
        --amass_dir ../base_mobileposer/data/processed_datasets \\
        --min_frames 900 --max_seqs 3 --max_seconds 30

    # full benchmark
    python evaluate_drift.py \\
        --model train_and_eval/models/<model_dir> \\
        --min_frames 1800 --max_seqs 30
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

# ── TensorFlow: support both TF 1.x and TF 2.x compat mode ──────────────────
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf  # TF 1.x

# ── DIP-IMU internal modules ─────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent           # DIP-IMU/
_TRAIN_EVAL  = _SCRIPT_DIR / 'train_and_eval'
if str(_TRAIN_EVAL) not in sys.path:
    sys.path.insert(0, str(_TRAIN_EVAL))

from configuration import Configuration
from constants    import Constants as _DIPConstants
from utils        import SMPL_MAJOR_JOINTS, SMPL_NR_JOINTS, smpl_reduced_to_full

_C = _DIPConstants()

# ── MobilePoser articulate library (shared SMPL body model) ──────────────────
_BASE_MP = _SCRIPT_DIR.parent / 'base_mobileposer'
sys.path.insert(0, str(_BASE_MP))
import mobileposer.articulate as art
from mobileposer.config import datasets, paths

# ── Shared evaluation constants / utilities (identical across all methods) ────
_CODE_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_CODE_DIR))
from drift_eval_common import (
    PRIMARY_SEGMENTS, SECONDARY_SEGMENTS, SEGMENTS, SEG_EN,
    LUMBAR_JOINTS, FPS, SENSOR_TO_JOINT, DATA_PATH,
    load_long_sequences, angle_between_rotmats, moving_average, add_imu_noise,
)

# DIP-IMU: 6 fixed sensors (incl. head)
DIP_SENSOR_INDICES = [0, 1, 2, 3, 4, 5]
# FK baseline: same sensors but exclude head (sensor 4)
FK_SENSOR_INDICES  = [0, 1, 2, 3, 5]

SMPL_KINTREE = [
    (0,1),(0,2),(0,3),
    (1,4),(2,5),(4,7),(5,8),(7,10),(8,11),
    (3,6),(6,9),(9,12),(9,13),(9,14),(12,15),
    (13,16),(14,17),(16,18),(17,19),(18,20),(19,21),(20,22),(21,23),
]
_LUMBAR_BONES = {(0,3),(3,6),(6,9)}


# angle_between_rotmats, moving_average, load_long_sequences → imported from drift_eval_common


# ─────────────────────────────────────────────────────────────────────────────
# DIP-IMU root-relative normalization
# (replicated from live_demo/inference_server.py :: normalize())
# ─────────────────────────────────────────────────────────────────────────────

def normalize_dip(ori_flat: np.ndarray, acc_flat: np.ndarray,
                  root_idx: int = 5) -> tuple:
    """
    Normalize 6-sensor IMU relative to the root sensor.

    ori_flat : (T, 54)  = 6 sensors × 9-D flattened 3×3 rotation matrices
    acc_flat : (T, 18)  = 6 sensors × 3-D acceleration vectors

    Returns (ori_n, acc_n) with shapes (T, 45) and (T, 15).
    Root sensor (index 5) is removed from output; 5 remaining sensors kept.
    """
    T = ori_flat.shape[0]
    oris = ori_flat.reshape(T, -1, 3, 3)        # (T, 6, 3, 3)
    accs = acc_flat.reshape(T, -1, 3, 1)         # (T, 6, 3, 1)

    # root orientation inverse  (R^T = R^{-1} for rotation matrices)
    root_inv = oris[:, root_idx].transpose(0, 2, 1)  # (T, 3, 3)

    # normalize orientations relative to root, drop root column
    oris_n = np.matmul(root_inv[:, np.newaxis], oris)[:, :5]   # (T, 5, 3, 3)

    # subtract root acceleration, rotate into root frame, drop root column
    accs_s = accs - accs[:, root_idx:root_idx+1]                # (T, 6, 3, 1)
    accs_n = np.matmul(root_inv[:, np.newaxis], accs_s[:, :5])  # (T, 5, 3, 1)

    return oris_n.reshape(T, -1), accs_n.reshape(T, -1)


# ─────────────────────────────────────────────────────────────────────────────
# FK baseline  (identical to MobilePoser evaluate_drift)
# ─────────────────────────────────────────────────────────────────────────────

def fk_baseline(ori: torch.Tensor, combo_indices: list,
                bodymodel: art.model.ParametricModel) -> torch.Tensor:
    """Full-body local pose from sensor orientations + parent propagation + IK."""
    T      = ori.shape[0]
    parent = bodymodel.parent

    sensor_joints = {SENSOR_TO_JOINT[5]}          # pelvis always present
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
# DIP-IMU model loading
# (adapted from live_demo/inference_server.py :: load_model())
# ─────────────────────────────────────────────────────────────────────────────

def load_dip_model(model_dir: str, sess):
    """Load DIP-IMU TF model.  model_dir must contain config.json + stats.npz."""
    tf.reset_default_graph()

    config_path = os.path.abspath(os.path.join(model_dir, 'config.json'))
    assert os.path.exists(config_path), f"config.json not found in {model_dir}"
    config_dict = Configuration.from_json(config_path)
    config_dict['system'] = 'local'

    config = Configuration(**config_dict)
    config.set('eval_dir', os.path.join(model_dir, 'evaluation'), override=True)
    config.set('model_dir', model_dir, override=True)
    os.makedirs(config.get('eval_dir'), exist_ok=True)

    data_placeholders = {
        _C.PL_INPUT:   tf.placeholder(tf.float32,
                                       shape=[1, None, _C.SIZE_ORI + _C.SIZE_ACC]),
        _C.PL_TARGET:  tf.placeholder(tf.float32,
                                       shape=[1, None, _C.SIZE_SMPL]),
        _C.PL_SEQ_LEN: tf.placeholder(tf.int32, shape=[1]),
    }

    stat_path = os.path.join(model_dir, 'stats.npz')
    assert os.path.exists(stat_path), f"stats.npz not found in {model_dir}"
    stats = dict(np.load(stat_path))

    model_cls = config.model_cls
    model = model_cls(
        config=config,
        session=sess,
        reuse=False,
        mode='sampling',
        placeholders=data_placeholders,
        input_dims=[_C.SIZE_ORI + _C.SIZE_ACC],
        target_dims=[_C.SIZE_SMPL],
        data_stats=stats,
    )
    model.build_graph()

    saver   = tf.train.Saver()
    ckpt    = tf.train.latest_checkpoint(model_dir)
    assert ckpt is not None, f"No checkpoint found in {model_dir}"
    print(f"  Restoring: {ckpt}")
    saver.restore(sess, ckpt)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Per-sequence evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_dip(model, acc: torch.Tensor, ori: torch.Tensor,
             gt_pose: torch.Tensor) -> torch.Tensor:
    """
    Run DIP-IMU inference on one sequence.

    acc     : [T, 6, 3]   raw acceleration (m/s²)
    ori     : [T, 6, 3, 3] global rotation matrices
    gt_pose : [T, 24, 3, 3] local rotation matrices (ground truth)
    Returns : rot_err [T, 24] in degrees
    """
    T = gt_pose.shape[0]

    # Flatten to DIP's expected format
    ori_flat = ori.numpy().reshape(T, -1)   # (T, 54)
    acc_flat = acc.numpy().reshape(T, -1)   # (T, 18)

    # Root-relative normalization (root = sensor 5 = pelvis)
    ori_n, acc_n = normalize_dip(ori_flat, acc_flat, root_idx=5)  # (T,45), (T,15)

    # DIP inference; internally applies zero-mean/unit-std from stats.npz
    pred, _, _ = model.inference_step(ori_n, acc_n, previous_state=None)
    # pred: (T, 135) — 15 major joints × 9-D rotation matrices (local)

    # Expand 15 predicted joints → 24 joints (missing joints = identity)
    pred_full = smpl_reduced_to_full(pred)   # (T, 216)

    # Replace joint 0 (pelvis) with the actual pelvis sensor global orientation.
    # In SMPL, joint 0 local rotation == its global orientation (it is the root).
    # The raw (non-normalized) pelvis sensor is sensor index 5.
    root_ori = ori_flat[:, 5 * 9 : 6 * 9]   # (T, 9)
    pred_full[:, 0:9] = root_ori

    # Reshape to [T, 24, 3, 3] and convert to torch for error computation
    pred_t = torch.from_numpy(pred_full.reshape(T, 24, 3, 3)).float()
    return angle_between_rotmats(pred_t, gt_pose)


def eval_fk(ori: torch.Tensor, gt_pose: torch.Tensor,
            combo_indices: list, bodymodel) -> torch.Tensor:
    T = gt_pose.shape[0]
    pose_fk = fk_baseline(ori, combo_indices, bodymodel)
    return angle_between_rotmats(pose_fk[:T], gt_pose)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(sequences, model, bodymodel, max_frames: int) -> dict:
    """Accumulate average per-frame error over all sequences."""
    dip_rot_sum = np.zeros((max_frames, 24))
    fk_rot_sum  = np.zeros((max_frames, 24))
    count       = np.zeros(max_frames)

    for seq in tqdm.tqdm(sequences, desc='Evaluating sequences'):
        T       = min(seq['pose'].shape[0], max_frames)
        gt_pose = seq['pose'][:T]
        acc     = seq['acc'][:T]
        ori     = seq['ori'][:T]

        try:
            rot_dip = eval_dip(model, acc, ori, gt_pose)
        except Exception as e:
            print(f"\n  Warning: skipped {seq['source']} — {e}")
            continue

        rot_fk = eval_fk(ori, gt_pose, FK_SENSOR_INDICES, bodymodel)

        dip_rot_sum[:T] += rot_dip.numpy()
        fk_rot_sum[:T]  += rot_fk.numpy()
        count[:T]       += 1.0

    valid        = count > 0
    safe_count   = np.maximum(count[:, None], 1)
    dip_rot_avg  = np.where(valid[:, None], dip_rot_sum / safe_count, 0.0)
    fk_rot_avg   = np.where(valid[:, None], fk_rot_sum  / safe_count, 0.0)

    return {
        'rot':       dip_rot_avg,            # [T, 24]  DIP-IMU
        'tran':      np.full(max_frames, np.nan),  # no translation output
        'fk_rot':    fk_rot_avg,             # [T, 24]  FK baseline
        'count':     count,
        'n_sensors': 6,
        'n_seqs':    int(count[0]) if count[0] > 0 else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting  (adapted from MobilePoser evaluate_drift)
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(res: dict, max_frames: int, fps: int, out_dir: str):
    """Figure 1: per-segment error curves over time."""
    t       = np.arange(max_frames) / fps
    max_sec = max_frames / fps

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for ax_i, (seg_name, joint_idx) in enumerate(SEGMENTS.items()):
        ax         = axes[ax_i]
        is_primary = seg_name in PRIMARY_SEGMENTS

        y_dip = moving_average(res['rot'][:, joint_idx].mean(axis=1))
        y_fk  = moving_average(res['fk_rot'][:, joint_idx].mean(axis=1))
        lw    = 2.0 if is_primary else 1.4

        ax.plot(t, y_dip, label='DIP-IMU (6 sensors)',
                color='#1565C0', linewidth=lw)
        ax.plot(t, y_fk,  label='FK baseline (5 sensors, no head)',
                color='gray', linestyle='--', linewidth=1.2)

        prefix = '[*] ' if is_primary else ''
        ax.set_title(f"{prefix}{SEG_EN[seg_name]}", fontsize=11,
                     fontweight='bold' if is_primary else 'normal')
        ax.set_ylabel('Angle error (deg)', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_sec)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        'DIP-IMU rotation drift by body segment  [*]=lumbar-rehab primary  '
        '(fixed 6 sensors incl. head)',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_drift_timeseries.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_combo_comparison(res: dict, max_frames: int, fps: int,
                          checkpoint_s: int, out_dir: str):
    """Figure 2: grouped bar chart — DIP vs FK at checkpoint_s."""
    fps_cp    = min(int(checkpoint_s * fps), max_frames - 1)
    seg_names = list(SEGMENTS.keys())
    x         = np.arange(len(seg_names))
    bar_w     = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))

    dip_vals = [res['rot'][fps_cp, j].mean() for j in SEGMENTS.values()]
    fk_vals  = [res['fk_rot'][fps_cp, j].mean() for j in SEGMENTS.values()]

    ax.bar(x - bar_w / 2, dip_vals, bar_w,
           label='DIP-IMU (6 sensors)', color='#1565C0', alpha=0.85)
    ax.bar(x + bar_w / 2, fk_vals,  bar_w,
           label='FK baseline (5 sensors, no head)', color='gray', alpha=0.75)

    for si, seg_name in enumerate(seg_names):
        if seg_name in PRIMARY_SEGMENTS:
            ax.axvspan(si - 0.48, si + 0.48, alpha=0.06, color='gold', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"[*]{SEG_EN[s]}" if s in PRIMARY_SEGMENTS else SEG_EN[s]
         for s in seg_names],
        fontsize=9, rotation=10)
    ax.set_ylabel('Mean angle error (deg)', fontsize=11)
    ax.set_title(
        f'DIP-IMU vs FK baseline at {checkpoint_s}s  --  [*]=lumbar primary',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_combo_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_sensor_count(res: dict, max_frames: int, fps: int,
                      checkpoint_s: int, out_dir: str):
    """Figure 3: lumbar score at checkpoint_s (single-point for fixed sensor count)."""
    fps_cp = min(int(checkpoint_s * fps), max_frames - 1)

    dip_lumbar = float(res['rot'][fps_cp, LUMBAR_JOINTS].mean())
    fk_lumbar  = float(res['fk_rot'][fps_cp, LUMBAR_JOINTS].mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([6], [dip_lumbar], width=0.4, color='#1565C0', alpha=0.85,
           label=f'DIP-IMU 6-sensor  {dip_lumbar:.1f} deg')
    ax.scatter([6], [dip_lumbar], color='#1565C0', s=80, zorder=4)
    ax.bar([5], [fk_lumbar], width=0.4, color='gray', alpha=0.75,
           label=f'FK baseline 5-sensor  {fk_lumbar:.1f} deg')
    ax.scatter([5], [fk_lumbar], color='gray', s=80, zorder=4)

    ax.set_xticks([5, 6])
    ax.set_xticklabels(['5 sensors\n(FK baseline)', '6 sensors\n(DIP-IMU)'])
    ax.set_xlabel('Number of sensors', fontsize=12)
    ax.set_ylabel('Lumbar error (deg)', fontsize=12)
    ax.set_title(
        f'Sensor count vs lumbar accuracy at {checkpoint_s}s\n'
        '(DIP-IMU: fixed 6-sensor method, no sparse-sensor support)',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_xlim(4, 7)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_sensor_count_vs_lumbar.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_translation(res: dict, max_frames: int, fps: int, out_dir: str):
    """Figure 4: DIP-IMU has no global translation output."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5,
            "DIP-IMU does not predict global translation.\n"
            "No translation drift data available.",
            ha='center', va='center', fontsize=16, color='gray',
            transform=ax.transAxes)
    ax.set_title('Global translation drift — DIP-IMU',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Position error (m)', fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_translation_drift.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(res: dict, max_frames: int, fps: int):
    checkpoints = sorted(set(
        cp for cp in [30, 60, 90, int(max_frames / fps)]
        if cp <= max_frames / fps))

    seg_names    = list(SEGMENTS.keys())
    primary_flag = ['*' if s in PRIMARY_SEGMENTS else ' ' for s in seg_names]
    col_w        = 8

    header = f"{'Method':<16} {'Sensors':>9} {'Time':>5}  "
    header += ''.join(f"{f}{SEG_EN[s]:>{col_w}}" for f, s in zip(primary_flag, seg_names))
    header += f"  {'Lumbar':>{col_w}}"
    sep = '-' * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for cp in checkpoints:
        frame = min(int(cp * fps), max_frames - 1)
        row   = f"{'DIP-IMU':<16} {'6 (fixed)':>9} {cp:>4}s  "
        for joint_idx in SEGMENTS.values():
            row += f"  {float(res['rot'][frame, joint_idx].mean()):>{col_w}.1f}"
        lumbar = float(res['rot'][frame, LUMBAR_JOINTS].mean())
        row   += f"  {lumbar:>{col_w}.1f}"
        print(row)

    print()
    row_fk = f"{'FK-baseline':<16} {'5 (no hd)':>9} {'end':>5}  "
    for joint_idx in SEGMENTS.values():
        row_fk += f"  {float(res['fk_rot'][-1, joint_idx].mean()):>{col_w}.1f}"
    fk_lumbar = float(res['fk_rot'][-1, LUMBAR_JOINTS].mean())
    row_fk   += f"  {fk_lumbar:>{col_w}.1f}"
    print(row_fk)

    print(sep)
    print("*=lumbar-rehab primary  Lumbar=joints[1,2,3,6,9] mean (deg)")
    print("Note: DIP-IMU has no translation estimate.")
    print("Note: DIP-IMU is a fixed 6-sensor method; no sparse-sensor evaluation.")


# ─────────────────────────────────────────────────────────────────────────────
# Video skeleton generation  (optional)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_skel(ax, joints, color, title, lumbar_err=None):
    ax.cla()
    j = joints - joints[0]
    for (a, b) in SMPL_KINTREE:
        is_lmb = (a, b) in _LUMBAR_BONES or (b, a) in _LUMBAR_BONES
        c, lw  = ('#FF6F00', 3.5) if is_lmb else (color, 1.8)
        ax.plot([j[a, 0], j[b, 0]], [j[a, 1], j[b, 1]], '-', color=c, lw=lw)
    ax.scatter(j[:, 0], j[:, 1], c=color, s=18, zorder=5)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.25, 1.85)
    ax.set_aspect('equal')
    ax.axis('off')
    lbl = title + (f'\nlumbar: {lumbar_err:.1f}' if lumbar_err is not None else '')
    ax.set_title(lbl, fontsize=9, pad=3)


def generate_video(seq, model, bodymodel, device_str,
                   fps, out_dir, max_seconds=30, render_fps=10, seq_idx=0):
    """Side-by-side video: GT (green) | DIP-IMU (blue) | FK (red)."""
    try:
        from matplotlib.animation import FFMpegWriter
    except Exception as e:
        print(f"  Skipped (FFMpegWriter unavailable): {e}")
        return

    stride  = max(1, round(fps / render_fps))
    T       = min(seq['pose'].shape[0], int(max_seconds * fps))
    gt_pose = seq['pose'][:T]
    gt_tran = seq['tran'][:T]
    acc     = seq['acc'][:T]
    ori     = seq['ori'][:T]

    # DIP inference
    ori_flat = ori.numpy().reshape(T, -1)
    acc_flat = acc.numpy().reshape(T, -1)
    ori_n, acc_n = normalize_dip(ori_flat, acc_flat)
    pred, _, _ = model.inference_step(ori_n, acc_n, previous_state=None)
    pred_full = smpl_reduced_to_full(pred)
    pred_full[:, 0:9] = ori_flat[:, 5*9:6*9]
    pose_dip = torch.from_numpy(pred_full.reshape(T, 24, 3, 3)).float()

    # FK baseline
    pose_fk = fk_baseline(ori, FK_SENSOR_INDICES, bodymodel)[:T]

    # Forward kinematics for joint positions
    with torch.no_grad():
        _, gt_joints  = bodymodel.forward_kinematics(gt_pose,  tran=gt_tran)
        _, dip_joints = bodymodel.forward_kinematics(pose_dip, tran=gt_tran)
        _, fk_joints  = bodymodel.forward_kinematics(pose_fk,  tran=gt_tran)
    gt_joints  = gt_joints.cpu().numpy()
    dip_joints = dip_joints.cpu().numpy()
    fk_joints  = fk_joints.cpu().numpy()

    lumbar_dip = angle_between_rotmats(
        pose_dip[:, LUMBAR_JOINTS], gt_pose[:, LUMBAR_JOINTS]).mean(-1).numpy()
    lumbar_fk  = angle_between_rotmats(
        pose_fk[:, LUMBAR_JOINTS], gt_pose[:, LUMBAR_JOINTS]).mean(-1).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.patch.set_facecolor('#111122')
    for ax in axes:
        ax.set_facecolor('#111122')

    video_path = os.path.join(out_dir, f'video_dip_imu_{seq.get("action", "all")}_{seq.get("source", seq_idx).replace("/", "_").replace("[", "_").replace("]", "")}.mp4')
    writer = FFMpegWriter(fps=render_fps, metadata={'title': 'drift-dip-6s'})
    frames = list(range(0, T, stride))
    print(f"  {len(frames)} frames at {render_fps}fps -> {video_path}")

    with writer.saving(fig, video_path, dpi=100):
        for t in frames:
            _draw_skel(axes[0], gt_joints[t],  '#43A047', 'Ground Truth')
            _draw_skel(axes[1], dip_joints[t], '#1E88E5',
                       'DIP-IMU (6 sensors)', lumbar_dip[t])
            _draw_skel(axes[2], fk_joints[t],  '#E53935',
                       'FK baseline (5 sensors)', lumbar_fk[t])
            fig.suptitle(f't = {t/fps:.1f}s    orange = lumbar spine',
                         color='white', fontsize=11)
            writer.grab_frame()

    plt.close(fig)
    print(f"  Saved: {video_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DIP-IMU drift evaluation — lumbar-rehabilitation focus.')
    parser.add_argument('--model', required=True,
                        help='Path to DIP-IMU model directory '
                             '(must contain config.json, stats.npz, and checkpoint files)')
    parser.add_argument('--combos', nargs='+', default=['all'],
                        help='Ignored — DIP-IMU is a fixed 6-sensor method.')
    parser.add_argument('--no_noise', action='store_true',
                        help='Disable IMU noise simulation (use clean synthetic data)')
    parser.add_argument('--drift', type=float, default=0.5,
                        help='Gyro random-walk rate °/√s (default 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for reproducible noise (default 42)')
    parser.add_argument('--amass_dir', default=None,
                        help='Dir with processed AMASS .pt files '
                             '(default: ../base_mobileposer/data/processed_datasets)')
    parser.add_argument('--action_manifest', default=None)
    parser.add_argument('--max_per_action', type=int, default=100)
    parser.add_argument('--min_frames', type=int, default=1800,
                        help='Min sequence length in frames (default 1800 = 60s @ 30fps)')
    parser.add_argument('--max_seqs', type=int, default=10,
                        help='Max sequences to use (0 = all qualifying)')
    parser.add_argument('--max_seconds', type=int, default=120,
                        help='Evaluation time window in seconds (default 120)')
    parser.add_argument('--compare_at', type=int, default=60,
                        help='Checkpoint time (s) for bar chart / scatter (default 60)')
    parser.add_argument('--out_dir', default='drift_results',
                        help='Output directory (default: drift_results)')
    parser.add_argument('--video_seconds', type=int, default=30,
                        help='Video length in seconds (default 30)')
    parser.add_argument('--video_fps', type=int, default=10,
                        help='Render FPS for video (default 10)')
    parser.add_argument('--no_video', action='store_true',
                        help='Skip video generation (metrics and figures only)')
    args = parser.parse_args()

    fps        = datasets.fps
    max_frames = args.max_seconds * fps
    amass_dir  = (Path(args.amass_dir) if args.amass_dir
                  else _SCRIPT_DIR.parent / 'base_mobileposer'
                       / 'data' / 'processed_datasets')
    model_dir  = os.path.abspath(args.model)

    print('=' * 62)
    print('  DIP-IMU Drift Evaluation — lumbar-rehabilitation focus')
    print('=' * 62)
    print(f"  Model dir  : {model_dir}")
    print(f"  AMASS dir  : {amass_dir}")
    print(f"  Min frames : {args.min_frames} ({args.min_frames / fps:.0f}s @ {fps}fps)")
    print(f"  Window     : {args.max_seconds}s  |  checkpoint: {args.compare_at}s")
    print(f"  Sensor cfg : fixed 6 (DIP-IMU native) — no combo variation")
    print('  NOTE: DIP-IMU is a TF-based fixed-sensor method.')
    print('        --combos argument is ignored.')
    print('=' * 62)

    os.makedirs(args.out_dir, exist_ok=True)

    sequences = load_long_sequences(args.min_frames, 0 if args.action_manifest else args.max_seqs, amass_dir=amass_dir,
                                    action_manifest=args.action_manifest,
                                    max_per_action=args.max_per_action)
    if not sequences:
        print("No qualifying sequences found. Adjust --min_frames or --amass_dir.")
        return

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

    smpl_file = str(_SCRIPT_DIR.parent / 'base_mobileposer'
                    / 'mobileposer' / 'smpl' / 'basicmodel_m.pkl')
    bodymodel = art.model.ParametricModel(smpl_file)

    print(f"\nLoading DIP-IMU model ...")
    sess  = tf.Session()
    model = load_dip_model(model_dir, sess)

    print(f"\nRunning evaluation over {len(sequences)} sequences ...")
    res = evaluate_all(sequences, model, bodymodel, max_frames)
    print(f"  Evaluated {res['n_seqs']} sequences.")

    print("\nGenerating figures ...")
    plot_timeseries(res, max_frames, fps, args.out_dir)
    plot_combo_comparison(res, max_frames, fps, args.compare_at, args.out_dir)
    plot_sensor_count(res, max_frames, fps, args.compare_at, args.out_dir)
    plot_translation(res, max_frames, fps, args.out_dir)

    print_summary(res, max_frames, fps)

    npz_path = os.path.join(args.out_dir, 'drift_data.npz')
    np.savez(npz_path,
             dip_6s_rot=res['rot'],
             dip_6s_fk_rot=res['fk_rot'],
             fps=fps,
             combos=['dip_6s'],
             n_seqs=res['n_seqs'])
    from benchmarks.standard_results import write_standard_result
    write_standard_result(
        Path(args.out_dir), 'dip-imu', 'drift', res['rot'], res['count'], fps,
        6, [0, 1, 2, 3, 4, 5], res['tran'],
    )
    print(f"\nRaw arrays saved: {npz_path}")

    if not args.no_video:
        print(f'\nGenerating videos for {len(sequences)} sequences ...')
        for i, vid_seq in enumerate(sequences):
            print(f"  [{i+1}/{len(sequences)}] seq={vid_seq['source']}")
            try:
                generate_video(vid_seq, model, bodymodel, 'cpu',
                               fps, args.out_dir,
                               max_seconds=args.video_seconds,
                               render_fps=args.video_fps, seq_idx=i)
            except Exception as e:
                print(f'    Video failed: {e}')

    sess.close()
    print(f"\nAll outputs in: {os.path.abspath(args.out_dir)}/")


if __name__ == '__main__':
    main()
