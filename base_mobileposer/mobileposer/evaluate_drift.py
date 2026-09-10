"""
Temporal rotation and translation drift evaluation — multi-combo, lumbar-first.

Product focus: lumbar rehabilitation. Primary metrics are on lumbar/pelvic region.
Compares all no-head sensor combos against FK baseline on long AMASS sequences.

Run from code/base_mobileposer/:
    # Quick smoke test (1 combo, 3 sequences, 30s)
    python -m mobileposer.evaluate_drift --model checkpoints/weights.pth \
        --combos rp --min_frames 900 --max_seqs 3 --max_seconds 30

    # Full benchmark (all 6 no-head combos, 30 sequences, 120s)
    python -m mobileposer.evaluate_drift --model checkpoints/weights.pth \
        --combos all --min_frames 1800 --max_seqs 30
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
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobileposer.config import amass, datasets, model_config, paths, joint_set
import mobileposer.articulate as art
from mobileposer.utils.model_utils import load_model

# ── Shared evaluation constants / utilities (identical across all methods) ────
_CODE_DIR = Path(__file__).resolve().parents[2]  # code/
sys.path.insert(0, str(_CODE_DIR))
from drift_eval_common import (
    load_long_sequences, angle_between_rotmats, moving_average, add_imu_noise,
)


# ---------------------------------------------------------------------------
# Segment definitions  (腰部优先)
# ---------------------------------------------------------------------------

# PRIMARY: directly relevant to lumbar rehabilitation
PRIMARY_SEGMENTS = {
    '腰':   [3],       # Spine1  (joint 0 forced to identity by model → excluded)
    '胸':   [6, 9],    # Spine2, Spine3
    '大腿': [1, 2],    # L/R Hip  (pelvic-motion proxy, closest sensor location)
}
# SECONDARY: useful for full-body context
SECONDARY_SEGMENTS = {
    '小腿': [4, 5],    # L/R Knee
    '大臂': [16, 17],  # L/R Shoulder (upper arm)
    '手腕': [18, 19],  # L/R Elbow (forearm; sensor placement location)
}
SEGMENTS = {**PRIMARY_SEGMENTS, **SECONDARY_SEGMENTS}

# "腰部综合分": mean error across all lumbar-relevant joints
LUMBAR_JOINTS = [1, 2, 3, 6, 9]

# Sensor index → SMPL joint index (from process.py ji_mask)
SENSOR_TO_JOINT = [18, 19, 1, 2, 15, 0]

# All combos without head sensor (index 4), from config.py amass.combos
# Native MobilePoser: all five wearable slots; pelvis remains the reference.
FULL_COMBOS = {'full_5s': [0, 1, 2, 3, 4]}

# Color palette per combo (consistent across all plots)
_PALETTE = ['#1565C0', '#C62828', '#2E7D32', '#F57F17', '#6A1B9A', '#00838F']
COMBO_COLORS = {name: _PALETTE[i % len(_PALETTE)]
                for i, name in enumerate(FULL_COMBOS)}

SENSOR_COUNT_COLORS = {1: '#EF5350', 2: '#42A5F5'}

# SMPL kinematic tree — (parent, child) bone pairs
SMPL_KINTREE = [
    (0,1),(0,2),(0,3),       # pelvis → L/R hip, spine1
    (1,4),(2,5),              # hip → knee
    (4,7),(5,8),              # knee → ankle
    (7,10),(8,11),            # ankle → foot
    (3,6),(6,9),              # spine chain
    (9,12),(9,13),(9,14),    # spine3 → neck / L-R collar
    (12,15),                  # neck → head
    (13,16),(14,17),          # collar → shoulder
    (16,18),(17,19),          # shoulder → elbow
    (18,20),(19,21),          # elbow → wrist
    (20,22),(21,23),          # wrist → hand
]
# Lumbar spine bones — highlighted in orange
_LUMBAR_BONES = {(0,3),(3,6),(6,9)}

# English labels for matplotlib plots (server has no CJK font)
SEG_EN = {
    '腰':   'Lumbar(j3)',
    '胸':   'Thoracic(j6,9)',
    '大腿': 'Hip(j1,2)',
    '小腿': 'Knee(j4,5)',
    '大臂': 'UpperArm(j16,17)',
    '手腕': 'Forearm(j18,19)',
}


# ---------------------------------------------------------------------------
# Math helpers  (angle_between_rotmats, moving_average → drift_eval_common)
# ---------------------------------------------------------------------------

def lumbar_score(rot_avg: np.ndarray) -> float:
    """Mean angular error over lumbar-relevant joints. rot_avg: [T, 24]"""
    return float(rot_avg[:, LUMBAR_JOINTS].mean())


# ---------------------------------------------------------------------------
# IMU input preparation  (replicates data.py _process_combo_data)
# ---------------------------------------------------------------------------

def prepare_imu(acc: torch.Tensor, ori: torch.Tensor,
                combo_indices: list) -> torch.Tensor:
    """Build 60-D IMU input for MobilePoser. acc:[T,6,3] ori:[T,6,3,3] → [T,60]"""
    acc5 = acc[:, :5] / amass.acc_scale
    ori5 = ori[:, :5]
    combo_acc = torch.zeros_like(acc5)
    combo_ori = torch.zeros_like(ori5)
    combo_acc[:, combo_indices] = acc5[:, combo_indices]
    combo_ori[:, combo_indices] = ori5[:, combo_indices]
    return torch.cat([combo_acc.flatten(1), combo_ori.flatten(1)], dim=1)


# ---------------------------------------------------------------------------
# FK baseline
# ---------------------------------------------------------------------------

def fk_baseline(ori: torch.Tensor, combo_indices: list,
                bodymodel: art.model.ParametricModel) -> torch.Tensor:
    """Full-body local pose from sensor orientations + parent propagation + IK.

    ori: [T, 6, 3, 3].  Returns: [T, 24, 3, 3] local rotation matrices.
    """
    T = ori.shape[0]
    parent = bodymodel.parent

    sensor_joints = {SENSOR_TO_JOINT[5]}          # pelvis always included
    for s in combo_indices:
        if s != 4:
            sensor_joints.add(SENSOR_TO_JOINT[s])

    R_global = torch.eye(3).view(1, 1, 3, 3).expand(T, 24, -1, -1).clone()
    R_global[:, SENSOR_TO_JOINT[5]] = ori[:, 5]   # pelvis
    for s in combo_indices:
        if s != 4:
            R_global[:, SENSOR_TO_JOINT[s]] = ori[:, s]

    # Propagate parent orientation to non-sensor joints
    for j in range(1, 24):
        if j not in sensor_joints:
            R_global[:, j] = R_global[:, parent[j]]

    return bodymodel.inverse_kinematics_R(
        R_global.view(T, -1)).view(T, 24, 3, 3)


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_mobileposer(model, imu_input, gt_pose, gt_tran, device):
    """Returns rot_err [T,24] in degrees and tran_err [T] in metres."""
    model.reset()
    imu = imu_input.to(device).unsqueeze(0)
    pose_pred, _, tran_pred, _ = model.forward_offline(imu, [imu.shape[1]])
    pose_pred = pose_pred.cpu()
    tran_pred = tran_pred.cpu()
    T = gt_pose.shape[0]
    rot_err  = angle_between_rotmats(pose_pred[:T], gt_pose)
    tran_err = (tran_pred[:T] - (gt_tran - gt_tran[:1])).norm(dim=-1)
    return rot_err, tran_err


def eval_fk(ori, gt_pose, combo_indices, bodymodel):
    """Returns rot_err [T,24] in degrees for the FK baseline."""
    pose_fk = fk_baseline(ori, combo_indices, bodymodel)
    T = gt_pose.shape[0]
    return angle_between_rotmats(pose_fk[:T], gt_pose)


# ---------------------------------------------------------------------------
# Run evaluation for one combo
# ---------------------------------------------------------------------------

def evaluate_combo(combo_name, combo_indices, sequences, model,
                   bodymodel, device, max_frames, out_dir):
    """Evaluate one combo over all sequences. Returns averaged error arrays."""
    _CKPT = os.path.join(out_dir, f'.eval_ckpt_{combo_name}.npz')

    ml_rot_sum  = np.zeros((max_frames, 24))
    ml_tran_sum = np.zeros(max_frames)
    fk_rot_sum  = np.zeros((max_frames, 24))
    count       = np.zeros(max_frames)
    start_idx   = 0

    if os.path.exists(_CKPT):
        ck = np.load(_CKPT)
        ml_rot_sum  = ck['ml_rot_sum']
        ml_tran_sum = ck['ml_tran_sum']
        fk_rot_sum  = ck['fk_rot_sum']
        count       = ck['count']
        start_idx   = int(ck['seqs_done'])
        print(f"    [Resume] combo={combo_name}: {start_idx}/{len(sequences)} done")

    for idx, seq in enumerate(sequences):
        if idx < start_idx:
            continue
        T = min(seq['pose'].shape[0], max_frames)
        gt_pose = seq['pose'][:T]
        gt_tran = seq['tran'][:T]
        acc     = seq['acc'][:T]
        ori     = seq['ori'][:T]

        try:
            imu = prepare_imu(acc, ori, combo_indices)
            rot_ml, tran_ml = eval_mobileposer(model, imu, gt_pose, gt_tran, device)
        except Exception as e:
            print(f"  Warning: skipped {seq['source']} ({combo_name}) — {e}")
        else:
            rot_fk = eval_fk(ori, gt_pose, combo_indices, bodymodel)
            ml_rot_sum[:T]  += rot_ml.numpy()
            ml_tran_sum[:T] += tran_ml.numpy()
            fk_rot_sum[:T]  += rot_fk.numpy()
            count[:T]       += 1.0

        np.savez(_CKPT, ml_rot_sum=ml_rot_sum, ml_tran_sum=ml_tran_sum,
                 fk_rot_sum=fk_rot_sum, count=count, seqs_done=idx + 1)

    if os.path.exists(_CKPT):
        os.remove(_CKPT)

    valid = count > 0
    ml_rot_avg  = np.where(valid[:, None], ml_rot_sum  / np.maximum(count[:, None], 1), 0.0)
    ml_tran_avg = np.where(valid,          ml_tran_sum / np.maximum(count, 1),          0.0)
    fk_rot_avg  = np.where(valid[:, None], fk_rot_sum  / np.maximum(count[:, None], 1), 0.0)

    return {
        'rot':      ml_rot_avg,
        'tran':     ml_tran_avg,
        'fk_rot':   fk_rot_avg,
        'count':    count,
        'n_sensors': len(combo_indices),
        'n_seqs':   int(count[0]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_timeseries(all_results, max_frames, fps, out_dir):
    """Figure 1: drift curves per body segment, one line per combo."""
    t = np.arange(max_frames) / fps
    max_sec = max_frames / fps

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for ax_i, (seg_name, joint_idx) in enumerate(SEGMENTS.items()):
        ax = axes[ax_i]
        is_primary = seg_name in PRIMARY_SEGMENTS

        for combo_name, res in all_results.items():
            y = moving_average(res['rot'][:, joint_idx].mean(axis=1))
            lw = 2.0 if is_primary else 1.2
            ax.plot(t, y, label=f"{combo_name}({res['n_sensors']}s)",
                    color=COMBO_COLORS[combo_name], linewidth=lw)

        # FK baseline: average across all combos
        fk_mean = np.mean([res['fk_rot'][:, joint_idx].mean(axis=1)
                           for res in all_results.values()], axis=0)
        ax.axhline(moving_average(fk_mean)[max_frames // 2],
                   color='gray', linestyle='--', linewidth=1.2, label='FK baseline')

        prefix = '[*] ' if is_primary else ''
        title = f"{prefix}{SEG_EN[seg_name]}"
        ax.set_title(title, fontsize=11, fontweight='bold' if is_primary else 'normal')
        ax.set_ylabel('Angle error (deg)', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_sec)
        ax.set_ylim(bottom=0)

    fig.suptitle('Rotation drift by body segment  [*]=lumbar-rehab primary  (no head sensor)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_drift_timeseries.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_combo_comparison(all_results, max_frames, fps, checkpoint_s, out_dir):
    """Figure 2: grouped bar chart — all combos × all segments at checkpoint_s."""
    fps_cp  = min(int(checkpoint_s * fps), max_frames - 1)
    seg_names  = list(SEGMENTS.keys())
    combo_names = list(all_results.keys())
    n_combos   = len(combo_names)
    n_segs     = len(seg_names)

    x = np.arange(n_segs)
    bar_w = 0.8 / n_combos

    fig, ax = plt.subplots(figsize=(14, 6))

    for ci, (combo_name, res) in enumerate(all_results.items()):
        vals = [res['rot'][fps_cp, joint_idx].mean()
                for joint_idx in SEGMENTS.values()]
        offset = (ci - n_combos / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w,
               label=f"{combo_name}({res['n_sensors']}s)",
               color=COMBO_COLORS[combo_name], alpha=0.85)

    # FK baseline horizontal lines per segment
    for si, (seg_name, joint_idx) in enumerate(SEGMENTS.items()):
        fk_val = np.mean([res['fk_rot'][fps_cp, joint_idx].mean()
                          for res in all_results.values()])
        ax.plot([si - 0.4, si + 0.4], [fk_val, fk_val],
                color='black', linestyle='--', linewidth=1.5)

    # Mark primary segments
    for si, seg_name in enumerate(seg_names):
        if seg_name in PRIMARY_SEGMENTS:
            ax.axvspan(si - 0.45, si + 0.45, alpha=0.06, color='gold', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"[*]{SEG_EN[s]}" if s in PRIMARY_SEGMENTS else SEG_EN[s] for s in seg_names],
        fontsize=9, rotation=10)
    ax.set_ylabel('Mean angle error (deg)', fontsize=11)
    ax.set_title(f'Combo error comparison at {checkpoint_s}s  --  [*]=lumbar primary  --  dashed=FK baseline',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_combo_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_sensor_count(all_results, max_frames, fps, checkpoint_s, out_dir):
    """Figure 3: lumbar score vs sensor count (scatter + mean±std)."""
    fps_cp = min(int(checkpoint_s * fps), max_frames - 1)

    by_count = {}
    for combo_name, res in all_results.items():
        n = res['n_sensors']
        score = float(res['rot'][fps_cp, LUMBAR_JOINTS].mean())
        by_count.setdefault(n, []).append((combo_name, score))

    counts = sorted(by_count.keys())
    fig, ax = plt.subplots(figsize=(7, 5))

    for n in counts:
        names, scores = zip(*by_count[n])
        mu, sigma = np.mean(scores), np.std(scores)
        color = SENSOR_COUNT_COLORS.get(n, 'gray')

        jitter = np.random.uniform(-0.05, 0.05, len(scores))
        ax.scatter([n + j for j in jitter], scores, color=color,
                   alpha=0.7, s=60, zorder=3)
        for xi, yi, lbl in zip([n + j for j in jitter], scores, names):
            ax.annotate(lbl, (xi, yi), textcoords='offset points',
                        xytext=(4, 2), fontsize=7, color=color)

        ax.bar(n, mu, width=0.3, color=color, alpha=0.3, zorder=2)
        ax.errorbar(n, mu, yerr=sigma, fmt='D', color=color,
                    markersize=7, capsize=5, linewidth=2, zorder=4,
                    label=f"{n} sensor(s)  mean={mu:.1f}deg")

    fk_lumbar = np.mean([res['fk_rot'][fps_cp, LUMBAR_JOINTS].mean()
                         for res in all_results.values()])
    ax.axhline(fk_lumbar, color='black', linestyle='--', linewidth=1.5,
               label=f'FK baseline  {fk_lumbar:.1f}deg')

    ax.set_xticks(counts)
    ax.set_xlabel('Number of sensors', fontsize=12)
    ax.set_ylabel('Lumbar error (deg)', fontsize=12)
    ax.set_title(f'Sensor count vs lumbar accuracy at {checkpoint_s}s', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_sensor_count_vs_lumbar.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_translation(all_results, max_frames, fps, out_dir):
    """Figure 4: translation drift per combo."""
    t = np.arange(max_frames) / fps
    max_sec = max_frames / fps
    fig, ax = plt.subplots(figsize=(10, 5))
    for combo_name, res in all_results.items():
        y = moving_average(res['tran'])
        ax.plot(t, y, label=f"{combo_name}({res['n_sensors']}s)",
                color=COMBO_COLORS[combo_name], linewidth=1.8)
    ax.set_title('Global translation drift (no head sensor)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Position error (m)', fontsize=11)
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_sec)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_translation_drift.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def _draw_skel(ax, joints, color, title, lumbar_err=None, view_limits=None,
               dimensions=(0, 1)):
    """Render one projection of the full SMPL skeleton."""
    ax.cla()
    j = joints
    x_dim, y_dim = dimensions
    for (a, b) in SMPL_KINTREE:
        is_lumbar = (a, b) in _LUMBAR_BONES or (b, a) in _LUMBAR_BONES
        c  = '#FF6F00' if is_lumbar else color
        lw = 3.5      if is_lumbar else 1.8
        ax.plot([j[a, x_dim], j[b, x_dim]], [j[a, y_dim], j[b, y_dim]], '-', color=c, lw=lw)
    ax.scatter(j[:, x_dim], j[:, y_dim], c=color, s=18, zorder=5)
    if view_limits is None:
        view_limits = (-1.2, 1.2, -0.1, 2.0)
    ax.set_xlim(view_limits[0], view_limits[1])
    ax.set_ylim(view_limits[2], view_limits[3])
    ax.set_aspect('equal')
    ax.axis('off')
    lbl = title + (f'\nlumbar: {lumbar_err:.1f}°' if lumbar_err is not None else '')
    ax.set_title(lbl, fontsize=10, color='white', pad=2, fontweight='bold')


def _canonical(combo: str) -> str:
    """Canonical combo name for cross-method comparison: h → hd."""
    return '_'.join('hd' if p == 'h' else p for p in combo.split('_'))


def generate_video(combo_name, combo_indices, seq, model, bodymodel, device,
                   fps, out_dir, max_seconds=30, render_fps=10, seq_idx=0):
    """
    Side-by-side skeleton video: GT (green) | MobilePoser (blue) | FK (red).
    Orange bones = lumbar chain.  Requires ffmpeg on PATH.
    """
    try:
        from matplotlib.animation import FFMpegWriter
    except Exception as e:
        print(f"  Skipped (FFMpegWriter unavailable): {e}")
        return

    stride  = max(1, round(fps / render_fps))
    T       = min(seq['pose'].shape[0], int(max_seconds * fps))

    gt_pose = seq['pose'][:T]          # [T, 24, 3, 3]
    gt_tran = seq['tran'][:T]          # [T, 3]
    acc     = seq['acc'][:T]
    ori     = seq['ori'][:T]

    # ── MobilePoser inference ──
    model.reset()
    with torch.no_grad():
        imu_in = prepare_imu(acc, ori, combo_indices).to(device).unsqueeze(0)
        pose_ml, _, tran_ml, _ = model.forward_offline(imu_in, [imu_in.shape[1]])
    pose_ml = pose_ml.cpu()[:T]                          # [T, 24, 3, 3]
    tran_ml = tran_ml.cpu()[:T] - tran_ml.cpu()[:1] + gt_tran[:1]

    # ── FK baseline ──
    pose_fk = fk_baseline(ori, combo_indices, bodymodel)[:T]

    # ── Joint positions via forward kinematics ──
    with torch.no_grad():
        _, gt_joints = bodymodel.forward_kinematics(gt_pose,  tran=gt_tran)
        _, ml_joints = bodymodel.forward_kinematics(pose_ml,  tran=tran_ml)
        _, fk_joints = bodymodel.forward_kinematics(pose_fk,  tran=gt_tran)
    gt_joints = gt_joints.cpu().numpy()   # [T, 24, 3]
    ml_joints = ml_joints.cpu().numpy()
    fk_joints = fk_joints.cpu().numpy()

    # Derive one shared camera window from the whole sequence. The previous
    # fixed portrait window clipped feet/legs or the torso for lying,
    # crawling, and other non-upright motions. Root-centering matches the
    # drawing logic, while shared limits keep the three panels comparable.
    sampled = np.concatenate((gt_joints[::stride], ml_joints[::stride],
                              fk_joints[::stride]), axis=0)
    # Match render_amass.py: centre horizontal root drift, shift the floor to
    # zero, then derive robust percentile camera extents for all projections.
    all_joints = np.concatenate((gt_joints, ml_joints, fk_joints), axis=0)
    all_joints[:, :, 0] -= np.median(all_joints[:, 0, 0])
    all_joints[:, :, 2] -= np.median(all_joints[:, 0, 2])
    all_joints[:, :, 1] -= np.percentile(all_joints[:, :, 1], 1.0)
    horizontal_radius = max(1.2, float(np.percentile(np.abs(all_joints[:, :, [0, 2]]), 99.5)) + 0.15)
    vertical_max = max(2.0, float(np.percentile(all_joints[:, :, 1], 99.5)) + 0.15)
    gt_joints, ml_joints, fk_joints = np.split(all_joints, 3, axis=0)
    view_limits = (horizontal_radius, vertical_max)

    # ── Per-frame lumbar error ──
    lumbar_ml = angle_between_rotmats(
        pose_ml[:, LUMBAR_JOINTS], gt_pose[:, LUMBAR_JOINTS]).mean(-1).numpy()
    lumbar_fk = angle_between_rotmats(
        pose_fk[:, LUMBAR_JOINTS], gt_pose[:, LUMBAR_JOINTS]).mean(-1).numpy()

    # ── Render ──
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.patch.set_facecolor('#111122')
    fig.subplots_adjust(top=0.78, bottom=0.04, left=0.02, right=0.98, wspace=0.04)
    for ax in axes:
        ax.set_facecolor('#111122')

    # Keep method, action and source in the filename so videos remain
    # attributable when outputs from several methods are collected together.
    def _safe(value):
        return ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in str(value))[:120]
    action = _safe(seq.get('action', 'all'))
    source = _safe(seq.get('source', f'seq{seq_idx:04d}'))
    video_path = os.path.join(
        out_dir,
        f'video_mobileposer_{_canonical(combo_name)}_{action}_{source}_{seq_idx:04d}.mp4',
    )
    writer = FFMpegWriter(fps=render_fps,
                          metadata={'title': f'drift-mobileposer-{_canonical(combo_name)}'})
    frames = list(range(0, T, stride))
    print(f"  {len(frames)} frames at {render_fps}fps → {video_path}")

    shared_vl = (-view_limits[0], view_limits[0], -0.1, view_limits[1])
    with writer.saving(fig, video_path, dpi=100):
        for t in frames:
            _draw_skel(axes[0], gt_joints[t], '#43A047', 'Ground Truth',
                       view_limits=shared_vl, dimensions=(0, 1))
            _draw_skel(axes[1], ml_joints[t],  '#1E88E5',
                       f'MobilePoser ({combo_name})', lumbar_ml[t],
                       shared_vl, (0, 1))
            _draw_skel(axes[2], fk_joints[t],  '#E53935',
                       'FK baseline', lumbar_fk[t],
                       shared_vl, (0, 1))
            source_label = str(seq.get('source', seq_idx))
            if len(source_label) > 42:
                source_label = source_label[:39] + '...'
            fig.suptitle(
                f'MobilePoser | {seq.get("action", "all")} | '
                f'{source_label} | t={t/fps:.1f}s',
                y=0.96, color='white', fontsize=13, fontweight='bold',
            )
            writer.grab_frame()
            if t % (fps * 10) == 0:
                print(f"    {t/fps:.0f}s / {T/fps:.0f}s")

    plt.close(fig)
    print(f"  Saved: {video_path}")


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_summary(all_results, max_frames, fps):
    checkpoints = [30, 60, 90, int(max_frames / fps)]
    checkpoints = sorted(set(cp for cp in checkpoints if cp <= max_frames / fps))

    seg_names = list(SEGMENTS.keys())
    primary_flag = ['★' if s in PRIMARY_SEGMENTS else ' ' for s in seg_names]
    col_w = 7

    header = f"{'组合':<10} {'传感器':>5} {'时间':>5}  "
    header += ''.join(f"{f}{s:>{col_w-1}}" for f, s in zip(primary_flag, seg_names))
    header += f"  {'腰部综合':>{col_w}}  {'位移(m)':>{col_w}}"
    sep = '─' * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for combo_name, res in all_results.items():
        for cp in checkpoints:
            frame = min(int(cp * fps), max_frames - 1)
            row = f"{combo_name:<10} {res['n_sensors']:>5}个 {cp:>4}s  "
            samples = int(res['count'][frame])
            if samples == 0:
                row += f"{'N/A':>{col_w}}" * (len(SEGMENTS) + 2)
                row += "  (no sequence reaches this checkpoint)"
                print(row)
                continue
            for joint_idx in SEGMENTS.values():
                val = float(res['rot'][frame, joint_idx].mean())
                row += f"  {val:>{col_w}.1f}"
            lumbar = float(res['rot'][frame, LUMBAR_JOINTS].mean())
            tran   = float(res['tran'][frame])
            row += f"  {lumbar:>{col_w}.1f}  {tran:>{col_w}.3f}  (n={samples})"
            print(row)
        print()

    print(f"{'FK-base':<10} {'─':>5}  {'avg':>4}  ", end='')
    valid_frames = [np.flatnonzero(res['count'] > 0)[-1] for res in all_results.values()]
    for joint_idx in SEGMENTS.values():
        fk_val = np.mean([res['fk_rot'][frame, joint_idx].mean()
                          for res, frame in zip(all_results.values(), valid_frames)])
        print(f"  {fk_val:>{col_w}.1f}", end='')
    fk_lumbar = np.mean([res['fk_rot'][frame, LUMBAR_JOINTS].mean()
                         for res, frame in zip(all_results.values(), valid_frames)])
    print(f"  {fk_lumbar:>{col_w}.1f}  {'N/A':>{col_w}}")
    print(sep)
    print("★=腰部康复核心段  腰部综合=joints[1,2,3,6,9]均值  位移仅MobilePoser有")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Multi-combo drift evaluation (lumbar-rehabilitation focus).')
    parser.add_argument('--model', required=True,
                        help='Path to MobilePoser weights (.pth or .ckpt)')
    parser.add_argument('--combos', nargs='+', default=['all'],
                        help='"all" or a subset, e.g.: --combos lp rp lw_rp rw_lp')
    parser.add_argument('--amass_dir', default=None,
                        help='Dir with processed AMASS .pt files '
                             '(default: paths.processed_datasets from config)')
    parser.add_argument('--action_manifest', default=None)
    parser.add_argument('--max_per_action', type=int, default=100)
    parser.add_argument('--min_frames', type=int, default=1800,
                        help='Min sequence length in frames (default 1800 = 60s)')
    parser.add_argument('--max_seqs', type=int, default=10,
                        help='Max sequences to use (default 0 = all qualifying)')
    parser.add_argument('--max_seconds', type=int, default=120,
                        help='Time window for plots in seconds (default 120)')
    parser.add_argument('--compare_at', type=int, default=60,
                        help='Time checkpoint (s) for bar chart and sensor-count plot (default 60)')
    parser.add_argument('--out_dir', default='drift_results',
                        help='Output directory (default: drift_results)')
    parser.add_argument('--video_seconds', type=int, default=30,
                        help='Video length in seconds (default 30)')
    parser.add_argument('--video_fps', type=int, default=10,
                        help='Render FPS for video (default 10; lower = faster)')
    parser.add_argument('--no_video', action='store_true',
                        help='Skip video generation (metrics and figures only)')
    parser.add_argument('--no_noise', action='store_true',
                        help='Disable IMU noise simulation (use clean synthetic data). '
                             'Without this flag noise is applied by default to expose '
                             'the gap between synthetic and real-IMU conditions.')
    parser.add_argument('--drift', type=float, default=0.5,
                        help='Gyro random-walk rate °/√s (default 0.5 ≈ consumer MEMS). '
                             '0.2=high-quality, 0.5=consumer, 1.5=low-cost')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for reproducible noise (default 42)')
    args = parser.parse_args()

    if args.combos == ['all']:
        selected = FULL_COMBOS
    else:
        invalid = [c for c in args.combos if c not in FULL_COMBOS]
        if invalid:
            raise ValueError(f"Unknown combo(s): {invalid}. "
                             f"Valid native full combo: {list(FULL_COMBOS)}")
        selected = {k: FULL_COMBOS[k] for k in args.combos}

    device     = model_config.device
    fps        = datasets.fps
    max_frames = args.max_seconds * fps
    amass_dir  = Path(args.amass_dir) if args.amass_dir else paths.processed_datasets
    action_manifest = args.action_manifest
    if action_manifest is None:
        for candidate in (
                paths.root_dir / 'data' / 'classification_manifest.csv',
                paths.root_dir / 'data' / 'rendered' / 'AMASS_by_action' / 'classification_manifest.csv'):
            if candidate.exists():
                action_manifest = str(candidate)
                break

    print(f"Device      : {device}")
    print(f"AMASS dir   : {amass_dir}")
    print(f"Combos      : {list(selected.keys())}")
    print(f"Min length  : {args.min_frames} frames ({args.min_frames/fps:.0f}s)")
    print(f"Action list : {action_manifest or 'disabled'}")

    print(f"\nLoading model: {args.model}")
    model = load_model(args.model).to(device)
    model.eval()

    bodymodel = art.model.ParametricModel(str(paths.smpl_file))

    sequences = load_long_sequences(args.min_frames, 0 if action_manifest else args.max_seqs, amass_dir=amass_dir,
                                    action_manifest=action_manifest,
                                    max_per_action=args.max_per_action)
    if not sequences:
        print("No sequences found. Adjust --min_frames or --amass_dir.")
        return

    if not args.no_noise:
        torch.manual_seed(args.seed)
        print(f"\nApplying IMU noise  drift={args.drift}°/√s  noise=0.5°  acc=0.1m/s²"
              f"  seed={args.seed}")
        print(f"  Expected orientation drift std after 60s : "
              f"{args.drift * (60 ** 0.5):.1f}°  |  after 120s : "
              f"{args.drift * (120 ** 0.5):.1f}°")
        for seq in sequences:
            seq['ori'], seq['acc'] = add_imu_noise(
                seq['ori'], seq['acc'], fps=fps,
                drift_deg_per_sqrt_s=args.drift)
    else:
        print("\nIMU noise disabled (--no_noise).  Using clean synthetic data.")

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}
    for combo_name, combo_indices in selected.items():
        print(f"\n── Evaluating combo: {combo_name}  "
              f"sensors={combo_indices}  n={len(combo_indices)} ──")
        all_results[combo_name] = evaluate_combo(
            combo_name, combo_indices, sequences,
            model, bodymodel, device, max_frames, args.out_dir)

    print("\nGenerating figures …")
    plot_timeseries(all_results, max_frames, fps, args.out_dir)
    plot_combo_comparison(all_results, max_frames, fps, args.compare_at, args.out_dir)
    plot_sensor_count(all_results, max_frames, fps, args.compare_at, args.out_dir)
    plot_translation(all_results, max_frames, fps, args.out_dir)

    print_summary(all_results, max_frames, fps)

    save_dict = {}
    for combo_name, res in all_results.items():
        save_dict[f'{combo_name}_rot']  = res['rot']
        save_dict[f'{combo_name}_tran'] = res['tran']
        save_dict[f'{combo_name}_fk']   = res['fk_rot']
    save_dict['fps']    = fps
    save_dict['combos'] = list(all_results.keys())
    np_path = os.path.join(args.out_dir, 'drift_data.npz')
    np.savez(np_path, **save_dict)
    print(f"\nRaw arrays saved: {np_path}")

    from pathlib import Path as _Path
    import sys as _sys
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from benchmarks.standard_results import write_standard_result
    result = all_results['full_5s']
    write_standard_result(
        _Path(args.out_dir), 'mobileposer', 'drift', result['rot'], result['count'],
        fps, 5, FULL_COMBOS['full_5s'], result['tran'],
    )

    if not args.no_video:
        print(f'\nGenerating videos for {len(sequences)} sequences × {len(selected)} combos ...')
        for i, vid_seq in enumerate(sequences):
            for combo_name, combo_indices in selected.items():
                print(f"  [{i+1}/{len(sequences)}] combo={combo_name}  seq={vid_seq['source']}")
                try:
                    generate_video(
                        combo_name, combo_indices, vid_seq,
                        model, bodymodel, device, fps, args.out_dir,
                        max_seconds=args.video_seconds,
                        render_fps=args.video_fps,
                        seq_idx=i,
                    )
                except Exception as e:
                    print(f'    Failed: {e}')

    print(f"\nAll outputs in:  {os.path.abspath(args.out_dir)}/")


if __name__ == '__main__':
    main()
