#!/usr/bin/env python3
"""
IMUCoCo (UIST 2025) drift benchmark.
Mirrors base_mobileposer/mobileposer/evaluate_drift.py — same metrics, segment
definitions, and plot layout.  IMUCoCo's unique ability to handle variable sensor
counts is exercised via multiple combos.

Run from IMUCoCo/ directory on the server:
    python evaluate_drift.py \
        --combos rw_rp,full_6s --min_frames 900 --max_seqs 3 --max_seconds 30  # smoke test
    python evaluate_drift.py \
        --min_frames 1800 --max_seqs 30                                          # full run
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
from models.imucoco import IMUCoCo
from models.dtp import Poser
from utils import imu_config
import path_config

# ── Shared evaluation constants / utilities (identical across all methods) ────
_CODE_DIR = Path(_SCRIPT_DIR).parent
sys.path.insert(0, str(_CODE_DIR))
from drift_eval_common import (
    PRIMARY_SEGMENTS, SECONDARY_SEGMENTS, SEGMENTS, SEG_EN,
    LUMBAR_JOINTS, FPS, SENSOR_TO_JOINT, DATA_PATH,
    load_long_sequences, angle_between_rotmats, moving_average,
)

# Our AMASS sensor order: [L_wrist(0), R_wrist(1), L_hip(2), R_hip(3), Head(4), Pelvis(5)]
# SMPL mesh vertex IDs for each sensor location (T-pose):
#   llowerarm=1962, rlowerarm=5431, lupperleg=947, rupperleg=4433, head=412, pelvis=3021
SENSOR_VERTEX_IDS = [1962, 5431, 947, 4433, 412, 3021]

# FK baseline: {sensor_idx → smpl_joint}  (head excluded)
FK_SENSOR_MAP = {0: 18, 1: 19, 2: 1, 3: 2, 5: 0}

# Combos: sensor index subsets from our 6-sensor AMASS format
COMBOS = {
    'lp':     {'indices': [2],           'n': 1},  # L hip
    'rp':     {'indices': [3],           'n': 1},  # R hip
    'lw_lp':  {'indices': [0, 2],        'n': 2},  # L wrist + L hip
    'lw_rp':  {'indices': [0, 3],        'n': 2},  # L wrist + R hip
    'rw_lp':  {'indices': [1, 2],        'n': 2},  # R wrist + L hip
    'rw_rp':  {'indices': [1, 3],        'n': 2},  # R wrist + R hip
    'full_6s': {'indices': list(range(6)), 'n': 6}, # all 6 sensors
}

_PALETTE = ['#1565C0', '#C62828', '#2E7D32', '#F57F17', '#6A1B9A', '#00838F', '#E65100']
COMBO_COLORS = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(COMBOS)}
SENSOR_COUNT_COLORS = {1: '#EF5350', 2: '#42A5F5', 6: '#2E7D32'}

RESULTS_DIR = Path(_SCRIPT_DIR) / 'drift_results'

SMPL_KINTREE = [
    (0,1),(0,2),(0,3),(1,4),(2,5),(4,7),(5,8),(7,10),(8,11),
    (3,6),(6,9),(9,12),(9,13),(9,14),(12,15),
    (13,16),(14,17),(16,18),(17,19),(18,20),(19,21),(20,22),(21,23),
]
_LUMBAR_BONES = {(0,3),(3,6),(6,9)}


# ─── model loading ─────────────────────────────────────────────────────────────

def load_models(device: str):
    """Load IMUCoCo encoder, Poser, body_model, and vertex_coordinates."""
    vc_cat = torch.tensor(imu_config.vertex_coordinates_with_category).float().to(device)
    jc_cat = torch.tensor(imu_config.joint_coordinates_with_category).float().to(device)
    coord_max = torch.max(vc_cat[:, 1:], dim=0).values
    coord_min = torch.min(vc_cat[:, 1:], dim=0).values
    vertex_coords = torch.tensor(imu_config.vertex_coordinates).float().to(device)  # [6890, 3]

    imucoco = IMUCoCo(
        coordinate_origins=jc_cat,
        coordinate_max=coord_max,
        coordinate_min=coord_min,
        smpl_mesh_coordinates=vc_cat,
        n_hidden=128, n_kr_hidden=32,
        n_mfe_layers=2, n_jnm_layers=3, n_sce_freq=4, n_sce_emb=40,
        online_mode=False,
        joint_node_allocation_map=path_config.saved_imucoco_loss_map_path,
        joint_node_max_err_tolerance=-1,
    ).to(device)
    imucoco.load_state_dict(
        torch.load(path_config.saved_imucoco_checkpoint_path, map_location=device), strict=False)
    imucoco.freeze()
    imucoco.eval()

    poser = Poser(joint_feature_dim=128, n_hidden=300, n_glb=40,
                  num_layer=3, n_total_devices=24, load_tran_module=True).to(device)
    poser.load_state_dict(
        torch.load(path_config.saved_hpe_checkpoint_path, map_location=device), strict=False)
    poser.eval()

    body_model = art.ParametricModel('smpl/SMPL_MALE.pkl', device='cpu')

    return imucoco, poser, body_model, vertex_coords


# ─── IMUCoCo inference ─────────────────────────────────────────────────────────

def _rotmat_to_r6d(R: torch.Tensor) -> torch.Tensor:
    """R: [..., 3, 3] → r6d [..., 6] (first two rows flattened)."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


@torch.no_grad()
def eval_imucoco(imucoco, poser, body_model, vertex_coords,
                 acc: torch.Tensor, ori: torch.Tensor,
                 gt_pose: torch.Tensor, gt_tran: torch.Tensor,
                 sensor_indices: list, device: str):
    """
    Run IMUCoCo + Poser for one sequence with a given sensor subset.

    Returns: (rot_err [T, 24], tran_err [T])
    """
    T = acc.shape[0]

    # 1. Set sensor placement codes for this combo
    vids   = [SENSOR_VERTEX_IDS[s] for s in sensor_indices]
    coords = vertex_coords[vids]                           # [D, 3]
    imucoco.set_current_device_coordinates(coords)
    imucoco.buffer_placement_codes_with_current_devices(parallel=False)

    # 2. Build 9D IMU input: [r6d_6D | acc_3D]
    ori_sel = ori[:, sensor_indices]                       # [T, D, 3, 3]
    acc_sel = acc[:, sensor_indices]                       # [T, D, 3]
    r6d     = _rotmat_to_r6d(ori_sel)                     # [T, D, 6]
    imu9    = torch.cat([r6d, acc_sel], dim=-1)            # [T, D, 9]
    imu9    = imu9.unsqueeze(0).to(device)                 # [1, T, D, 9]

    # 3. Encode → joint node features
    feat_m = imucoco.inference_time_forward_mesh(imu9)     # [1, T, 24, 128]

    # 4. Compute glb_init from GT first frame (local → global FK → r6d)
    first_local = gt_pose[0:1].cpu()                       # [1, 24, 3, 3]
    glb_0, _    = body_model.forward_kinematics(first_local, calc_mesh=False)  # [1, 24, 3, 3]
    glb_init    = _rotmat_to_r6d(glb_0).to(device)        # [1, 24, 6]

    vel_init = torch.zeros(1, 24, 3, device=device)
    seq_len  = torch.tensor([T])

    # 5. Poser inference
    _, pose_local_pred, tran_out = poser.forward(
        x=feat_m, v_init=vel_init, glb_init=glb_init,
        seq_len=seq_len, compute_tran='transpose',
    )
    # pose_local_pred: [1, T, 24, 3, 3]  LOCAL
    # tran_out:        [T, 3]             root translation

    pred_local = pose_local_pred[0].cpu()                  # [T, 24, 3, 3]
    rot_err = angle_between_rotmats(pred_local, gt_pose)   # [T, 24]

    tran_pred = tran_out.cpu()                             # [T, 3]
    tran_gt   = gt_tran.cpu()
    tran_err  = (tran_pred - tran_pred[0:1] - (tran_gt - tran_gt[0:1])).norm(dim=-1)  # [T]

    return rot_err, tran_err


def eval_fk(ori: torch.Tensor, gt_pose: torch.Tensor) -> torch.Tensor:
    """Sensor-orientation FK baseline → rot_err [T, 24]."""
    T = ori.shape[0]
    pose_local = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, -1, -1).clone()
    root_ori = ori[:, 5]
    for s_idx, j_idx in FK_SENSOR_MAP.items():
        pose_local[:, j_idx] = root_ori.transpose(-1, -2) @ ori[:, s_idx]
    return angle_between_rotmats(pose_local, gt_pose)


# ─── evaluation loop ───────────────────────────────────────────────────────────

def evaluate_all(sequences, imucoco, poser, body_model, vertex_coords,
                 active_combos: dict, max_frames: int, device: str,
                 out_dir: str) -> dict:
    """Accumulate running-mean per-frame error for each combo."""
    _CKPT = os.path.join(out_dir, '.eval_ckpt_imucoco.npz')

    all_results = {
        combo_name: {
            'rot_sum':   np.zeros((max_frames, 24)),
            'fk_sum':    np.zeros((max_frames, 24)),
            'tran_sum':  np.zeros(max_frames),
            'count':     np.zeros(max_frames),
            'n_sensors': info['n'],
        }
        for combo_name, info in active_combos.items()
    }
    start_idx = 0

    if os.path.exists(_CKPT):
        ck = np.load(_CKPT)
        start_idx = int(ck['seqs_done'])
        for combo_name in active_combos:
            r = all_results[combo_name]
            r['rot_sum']  = ck[f'{combo_name}_rot_sum']
            r['fk_sum']   = ck[f'{combo_name}_fk_sum']
            r['tran_sum'] = ck[f'{combo_name}_tran_sum']
            r['count']    = ck[f'{combo_name}_count']
        print(f"  [Resume] checkpoint loaded: {start_idx}/{len(sequences)} sequences done.")

    for idx, seq in enumerate(tqdm(sequences, desc='Sequences')):
        if idx < start_idx:
            continue
        T       = min(seq['pose'].shape[0], max_frames)
        gt_pose = seq['pose'][:T]
        acc     = seq['acc'][:T]
        ori     = seq['ori'][:T]
        tran    = seq['tran'][:T]

        fk_err = eval_fk(ori, gt_pose)

        for combo_name, info in active_combos.items():
            try:
                rot_err, tran_err = eval_imucoco(
                    imucoco, poser, body_model, vertex_coords,
                    acc, ori, gt_pose, tran, info['indices'], device,
                )
            except Exception as e:
                print(f'\n  Warning: {combo_name} seq {seq["source"]} — {e}')
                continue

            r = all_results[combo_name]
            r['rot_sum'][:T]  += rot_err.numpy()
            r['fk_sum'][:T]   += fk_err.numpy()
            r['tran_sum'][:T] += tran_err.numpy()
            r['count'][:T]    += 1.0

        ck_data = {'seqs_done': idx + 1}
        for combo_name, r in all_results.items():
            ck_data[f'{combo_name}_rot_sum']  = r['rot_sum']
            ck_data[f'{combo_name}_fk_sum']   = r['fk_sum']
            ck_data[f'{combo_name}_tran_sum'] = r['tran_sum']
            ck_data[f'{combo_name}_count']    = r['count']
        np.savez(_CKPT, **ck_data)

    if os.path.exists(_CKPT):
        os.remove(_CKPT)

    final = {}
    for combo_name, r in all_results.items():
        valid  = r['count'] > 0
        safe_c = np.maximum(r['count'][:, None], 1)
        safe_1 = np.maximum(r['count'], 1)
        final[combo_name] = {
            'rot':     np.where(valid[:, None], r['rot_sum']  / safe_c, 0.0),
            'fk_rot':  np.where(valid[:, None], r['fk_sum']   / safe_c, 0.0),
            'tran':    np.where(valid,          r['tran_sum'] / safe_1,  0.0),
            'n_sensors': r['n_sensors'],
            'n_seqs':  int(r['count'][0]) if r['count'][0] > 0 else 0,
        }
    return final


# ─── plotting ──────────────────────────────────────────────────────────────────

def plot_timeseries(all_results: dict, max_frames: int, fps: int, out_dir: str):
    """Figure 1: per-segment drift curves, one line per combo."""
    t       = np.arange(max_frames) / fps
    max_sec = max_frames / fps

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()

    for ax_i, (seg_name, joint_idx) in enumerate(SEGMENTS.items()):
        ax         = axes[ax_i]
        is_primary = seg_name in PRIMARY_SEGMENTS

        for combo_name, res in all_results.items():
            y  = moving_average(res['rot'][:, joint_idx].mean(axis=1))
            lw = 2.0 if is_primary else 1.2
            ax.plot(t, y, label=f"{combo_name}({res['n_sensors']}s)",
                    color=COMBO_COLORS[combo_name], linewidth=lw)

        fk_mean = np.mean([res['fk_rot'][:, joint_idx].mean(axis=1)
                           for res in all_results.values()], axis=0)
        ax.axhline(moving_average(fk_mean)[max_frames // 2],
                   color='gray', linestyle='--', linewidth=1.2, label='FK baseline')

        prefix = '[*] ' if is_primary else ''
        ax.set_title(f'{prefix}{SEG_EN[seg_name]}',
                     fontsize=11, fontweight='bold' if is_primary else 'normal')
        ax.set_ylabel('Angle error (deg)', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_sec)
        ax.set_ylim(bottom=0)

    fig.suptitle('IMUCoCo — Rotation drift by body segment  [*]=lumbar-rehab primary',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_drift_timeseries.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_combo_comparison(all_results: dict, max_frames: int, fps: int,
                          checkpoint_s: float, out_dir: str):
    """Figure 2: grouped bar chart — all combos × all segments at checkpoint_s."""
    fps_cp      = min(int(checkpoint_s * fps), max_frames - 1)
    seg_names   = list(SEGMENTS.keys())
    combo_names = list(all_results.keys())
    n_combos    = len(combo_names)
    x           = np.arange(len(seg_names))
    bar_w       = 0.8 / n_combos

    fig, ax = plt.subplots(figsize=(14, 6))

    for ci, (combo_name, res) in enumerate(all_results.items()):
        vals   = [res['rot'][fps_cp, joint_idx].mean() for joint_idx in SEGMENTS.values()]
        offset = (ci - n_combos / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w,
               label=f"{combo_name}({res['n_sensors']}s)",
               color=COMBO_COLORS[combo_name], alpha=0.85)

    for si, (seg_name, joint_idx) in enumerate(SEGMENTS.items()):
        fk_val = np.mean([res['fk_rot'][fps_cp, joint_idx].mean()
                          for res in all_results.values()])
        ax.plot([si - 0.4, si + 0.4], [fk_val, fk_val],
                color='black', linestyle='--', linewidth=1.5)

    for si, seg_name in enumerate(seg_names):
        if seg_name in PRIMARY_SEGMENTS:
            ax.axvspan(si - 0.45, si + 0.45, alpha=0.06, color='gold', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'[*]{SEG_EN[s]}' if s in PRIMARY_SEGMENTS else SEG_EN[s] for s in seg_names],
        fontsize=9, rotation=10)
    ax.set_ylabel('Mean angle error (deg)', fontsize=11)
    ax.set_title(f'Combo comparison at {checkpoint_s}s  [*]=lumbar primary  dashed=FK',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig2_combo_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_sensor_count(all_results: dict, max_frames: int, fps: int,
                      checkpoint_s: float, out_dir: str):
    """Figure 3: lumbar score vs sensor count (scatter + mean±std)."""
    fps_cp = min(int(checkpoint_s * fps), max_frames - 1)

    by_count: dict = {}
    for combo_name, res in all_results.items():
        n     = res['n_sensors']
        score = float(res['rot'][fps_cp, LUMBAR_JOINTS].mean())
        by_count.setdefault(n, []).append((combo_name, score))

    counts = sorted(by_count.keys())
    fig, ax = plt.subplots(figsize=(7, 5))

    np.random.seed(0)
    for n in counts:
        names, scores = zip(*by_count[n])
        mu, sigma = np.mean(scores), np.std(scores)
        color = SENSOR_COUNT_COLORS.get(n, 'gray')
        jitter = np.random.uniform(-0.05, 0.05, len(scores))
        ax.scatter([n + j for j in jitter], scores, color=color, alpha=0.7, s=60, zorder=3)
        for xi, yi, lbl in zip([n + j for j in jitter], scores, names):
            ax.annotate(lbl, (xi, yi), textcoords='offset points',
                        xytext=(4, 2), fontsize=7, color=color)
        ax.bar(n, mu, width=0.3, color=color, alpha=0.3, zorder=2)
        ax.errorbar(n, mu, yerr=sigma, fmt='D', color=color,
                    markersize=7, capsize=5, linewidth=2, zorder=4,
                    label=f'{n} sensor(s)  mean={mu:.1f}°')

    fk_lumbar = np.mean([res['fk_rot'][fps_cp, LUMBAR_JOINTS].mean()
                         for res in all_results.values()])
    ax.axhline(fk_lumbar, color='black', linestyle='--', linewidth=1.5,
               label=f'FK baseline  {fk_lumbar:.1f}°')

    ax.set_xticks(counts)
    ax.set_xlabel('Number of sensors', fontsize=12)
    ax.set_ylabel('Lumbar error (deg)', fontsize=12)
    ax.set_title(f'Sensor count vs lumbar accuracy at {checkpoint_s}s', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_sensor_count_vs_lumbar.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved: {path}')


def plot_translation(all_results: dict, max_frames: int, fps: int, out_dir: str):
    """Figure 4: root translation drift per combo."""
    t       = np.arange(max_frames) / fps
    max_sec = max_frames / fps

    fig, ax = plt.subplots(figsize=(10, 5))
    for combo_name, res in all_results.items():
        y = moving_average(res['tran'])
        ax.plot(t, y, label=f"{combo_name}({res['n_sensors']}s)",
                color=COMBO_COLORS[combo_name], linewidth=1.8)

    ax.set_title('IMUCoCo — Global translation drift', fontsize=13, fontweight='bold')
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
    print(f'Saved: {path}')


# ─── video generation ──────────────────────────────────────────────────────────

@torch.no_grad()
def _infer_imucoco_poses(imucoco, poser, body_model, vertex_coords,
                          acc, ori, gt_pose, gt_tran, sensor_indices, device):
    """Like eval_imucoco but returns (pose_local [T,24,3,3], tran [T,3])."""
    T = acc.shape[0]
    vids   = [SENSOR_VERTEX_IDS[s] for s in sensor_indices]
    coords = vertex_coords[vids]
    imucoco.set_current_device_coordinates(coords)
    imucoco.buffer_placement_codes_with_current_devices(parallel=False)

    ori_sel = ori[:, sensor_indices]
    r6d     = _rotmat_to_r6d(ori_sel)
    imu9    = torch.cat([r6d, acc[:, sensor_indices]], dim=-1).unsqueeze(0).to(device)
    feat_m  = imucoco.inference_time_forward_mesh(imu9)

    first_local = gt_pose[0:1].cpu()
    glb_0, _    = body_model.forward_kinematics(first_local, calc_mesh=False)
    glb_init    = _rotmat_to_r6d(glb_0).to(device)
    vel_init    = torch.zeros(1, 24, 3, device=device)

    _, pose_local_pred, tran_out = poser.forward(
        x=feat_m, v_init=vel_init, glb_init=glb_init,
        seq_len=torch.tensor([T]), compute_tran='transpose')

    return pose_local_pred[0].cpu(), tran_out.cpu()   # [T,24,3,3], [T,3]


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
    ax.set_title(title + (f'\nlumbar: {lumbar_err:.1f}°' if lumbar_err is not None else ''),
                 fontsize=9, pad=3)


def _canonical(combo: str) -> str:
    """Canonical combo name for cross-method comparison: full_6s → 6s."""
    return '6s' if combo == 'full_6s' else combo


def generate_video_combo(combo_name, sensor_indices, seq,
                          imucoco, poser, body_model, vertex_coords, device,
                          fps, out_dir, max_seconds=30, render_fps=10, seq_idx=0):
    """Side-by-side skeleton video: GT (green) | IMUCoCo (blue) | FK (red)."""
    try:
        from matplotlib.animation import FFMpegWriter
    except Exception as e:
        print(f'  Skipped (FFMpegWriter unavailable): {e}')
        return

    stride  = max(1, round(fps / render_fps))
    T       = min(seq['pose'].shape[0], int(max_seconds * fps))
    gt_pose = seq['pose'][:T]
    gt_tran = seq['tran'][:T]
    acc     = seq['acc'][:T]
    ori     = seq['ori'][:T]

    pose_pred, tran_pred = _infer_imucoco_poses(
        imucoco, poser, body_model, vertex_coords,
        acc, ori, gt_pose, gt_tran, sensor_indices, device)

    # root-align predicted translation
    tran_pred = tran_pred - tran_pred[0:1] + gt_tran[0:1]

    # FK baseline local poses
    pose_fk = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(T, 24, -1, -1).clone()
    root_ori = ori[:, 5]
    for s_idx, j_idx in FK_SENSOR_MAP.items():
        pose_fk[:, j_idx] = root_ori.transpose(-1, -2) @ ori[:, s_idx]

    with torch.no_grad():
        _, gt_j   = body_model.forward_kinematics(gt_pose,   tran=gt_tran)
        _, pred_j = body_model.forward_kinematics(pose_pred, tran=tran_pred)
        _, fk_j   = body_model.forward_kinematics(pose_fk,   tran=gt_tran)
    gt_j, pred_j, fk_j = gt_j.cpu().numpy(), pred_j.cpu().numpy(), fk_j.cpu().numpy()

    lumbar_pred = angle_between_rotmats(
        pose_pred[:, LUMBAR_JOINTS], gt_pose[:, LUMBAR_JOINTS]).mean(-1).numpy()
    lumbar_fk   = angle_between_rotmats(
        pose_fk[:, LUMBAR_JOINTS], gt_pose[:, LUMBAR_JOINTS]).mean(-1).numpy()

    n_sens = len(sensor_indices)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.patch.set_facecolor('#111122')
    for ax in axes:
        ax.set_facecolor('#111122')

    vp = os.path.join(out_dir, f'video_imucoco_{_canonical(combo_name)}_{seq_idx:04d}.mp4')
    writer = FFMpegWriter(fps=render_fps, metadata={'title': f'drift-imucoco-{_canonical(combo_name)}'})
    print(f'  Writing {len(range(0, T, stride))} frames → {vp}')
    with writer.saving(fig, vp, dpi=100):
        for t in range(0, T, stride):
            _draw_skel(axes[0], gt_j[t],   '#43A047', 'Ground Truth')
            _draw_skel(axes[1], pred_j[t], '#1E88E5',
                       f'IMUCoCo {combo_name}({n_sens}s)', lumbar_pred[t])
            _draw_skel(axes[2], fk_j[t],   '#E53935', 'FK baseline', lumbar_fk[t])
            fig.suptitle(f't = {t/fps:.1f}s    orange = lumbar spine',
                         color='white', fontsize=11)
            writer.grab_frame()
    plt.close(fig)
    print(f'  Saved: {vp}')


# ─── summary ───────────────────────────────────────────────────────────────────

def print_summary(all_results: dict, max_frames: int, fps: int):
    checkpoints = sorted(set(cp for cp in [10, 20, 30, int(max_frames / fps)]
                             if cp <= max_frames / fps))
    seg_names    = list(SEGMENTS.keys())
    primary_flag = ['*' if s in PRIMARY_SEGMENTS else ' ' for s in seg_names]
    col_w = 8

    hdr = f"{'Combo':<12} {'#s':>3} {'t(s)':>5}  "
    hdr += ''.join(f"{f}{SEG_EN[s]:>{col_w}}" for f, s in zip(primary_flag, seg_names))
    hdr += f"  {'Lumbar':>{col_w}}  {'Tran(m)':>{col_w}}"
    sep = '─' * len(hdr)

    print(f'\n{sep}')
    print(hdr)
    print(sep)

    for combo_name, res in all_results.items():
        for cp in checkpoints:
            frame = min(int(cp * fps), max_frames - 1)
            row = f"{combo_name:<12} {res['n_sensors']:>3} {cp:>5}  "
            for joint_idx in SEGMENTS.values():
                row += f"  {res['rot'][frame, joint_idx].mean():>{col_w}.1f}"
            lumbar = res['rot'][frame, LUMBAR_JOINTS].mean()
            tran   = res['tran'][frame]
            row += f"  {lumbar:>{col_w}.1f}  {tran:>{col_w}.3f}"
            print(row)

    # FK row
    frame30 = min(int(30 * fps), max_frames - 1)
    print(f'\n  FK baseline:')
    row = f"{'FK':<12} {'6':>3} {30:>5}  "
    first_res = next(iter(all_results.values()))
    for joint_idx in SEGMENTS.values():
        row += f"  {first_res['fk_rot'][frame30, joint_idx].mean():>{col_w}.1f}"
    lumbar_fk = first_res['fk_rot'][frame30, LUMBAR_JOINTS].mean()
    row += f"  {lumbar_fk:>{col_w}.1f}  {'N/A':>{col_w}}"
    print(row)
    print(sep)


def save_npz(all_results: dict, out_path: Path):
    arrays = {}
    for combo_name, res in all_results.items():
        arrays[f'{combo_name}_rot']     = res['rot']
        arrays[f'{combo_name}_fk_rot']  = res['fk_rot']
        arrays[f'{combo_name}_tran']    = res['tran']
    np.savez_compressed(str(out_path), **arrays)
    print(f'Saved: {out_path}')


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='IMUCoCo drift benchmark')
    parser.add_argument('--min_frames',   type=int,   default=1800)
    parser.add_argument('--max_seqs',     type=int,   default=0,
                        help='Max sequences (0 = all qualifying)')
    parser.add_argument('--max_seconds',  type=float, default=120.0)
    parser.add_argument('--checkpoint_s', type=float, default=60.0)
    parser.add_argument('--combos',       type=str,   default='all',
                        help='Comma-separated combo names or "all"')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--video_seconds', type=int, default=30)
    parser.add_argument('--video_fps',     type=int, default=10)
    parser.add_argument('--no_video', action='store_true',
                        help='Skip video generation (metrics and figures only)')
    args = parser.parse_args()

    max_frames = int(args.max_seconds * FPS)
    device = args.device
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = str(RESULTS_DIR)

    if args.combos == 'all':
        active_combos = COMBOS
    else:
        keys = [k.strip() for k in args.combos.split(',')]
        active_combos = {k: COMBOS[k] for k in keys if k in COMBOS}
        if not active_combos:
            raise ValueError(f'No valid combos in: {args.combos}')

    print('Loading models...')
    imucoco, poser, body_model, vertex_coords = load_models(device)

    sequences = load_long_sequences(args.min_frames, args.max_seqs)
    print(f'  {len(sequences)} sequences loaded')

    all_results = evaluate_all(
        sequences, imucoco, poser, body_model, vertex_coords,
        active_combos, max_frames, device, out_dir,
    )

    n_seqs = next(iter(all_results.values()))['n_seqs']
    print(f'\nEvaluated {n_seqs} sequences, {len(active_combos)} combos, {max_frames / FPS:.0f}s each')

    print('\nGenerating figures...')
    plot_timeseries(all_results,      max_frames, FPS, out_dir)
    plot_combo_comparison(all_results, max_frames, FPS, args.checkpoint_s, out_dir)
    plot_sensor_count(all_results,    max_frames, FPS, args.checkpoint_s, out_dir)
    plot_translation(all_results,     max_frames, FPS, out_dir)

    print_summary(all_results, max_frames, FPS)
    save_npz(all_results, RESULTS_DIR / 'imucoco_drift_results.npz')

    if not args.no_video:
        print(f'\nGenerating videos for {len(sequences)} sequences × {len(active_combos)} combos ...')
        for i, vid_seq in enumerate(sequences):
            for combo_name, info in active_combos.items():
                print(f"  [{i+1}/{len(sequences)}] combo={combo_name}  seq={vid_seq['source']}")
                try:
                    generate_video_combo(
                        combo_name, info['indices'], vid_seq,
                        imucoco, poser, body_model, vertex_coords, device,
                        FPS, out_dir, args.video_seconds, args.video_fps, seq_idx=i)
                except Exception as e:
                    print(f'    Video failed: {e}')

    print(f'\nAll outputs → {RESULTS_DIR}')


if __name__ == '__main__':
    main()
