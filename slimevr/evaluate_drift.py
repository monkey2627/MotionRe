"""
Temporal rotation drift evaluation for SlimeVR (FK parent propagation) — lumbar-first.

SlimeVR has NO learned model. Its pose estimation algorithm is purely geometric:
  1. Direct sensor orientation → joint assignment for instrumented joints.
  2. Parent propagation (inheriting parent's global rotation) for all other joints.
  3. Inverse kinematics converts global rotations back to local space.

This is identical to the FK baseline used in MobilePoser evaluation. The script
therefore evaluates how different sensor configurations (combos) affect FK accuracy.
Since translation is not estimated by this method, fig4 is a placeholder.

Run from code/slimevr/:
    # Quick smoke test (1 combo, 3 sequences, 30s)
    python evaluate_drift.py --combos rp --min_frames 900 --max_seqs 3 --max_seconds 30

    # Full benchmark (all 6 no-head combos, 30 sequences, 120s)
    python evaluate_drift.py --combos all --min_frames 1800 --max_seqs 30
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

# Add base_mobileposer to path so we can reuse config, articulate, etc.
_BASE = Path(__file__).resolve().parents[1] / 'base_mobileposer'
sys.path.insert(0, str(_BASE))

from mobileposer.config import amass, datasets, model_config, paths, joint_set
import mobileposer.articulate as art


# ---------------------------------------------------------------------------
# Segment definitions  (lumbar-first)
# ---------------------------------------------------------------------------

PRIMARY_SEGMENTS = {
    'Lumbar':   [3],
    'Thoracic': [6, 9],
    'Hip':      [1, 2],
}
SECONDARY_SEGMENTS = {
    'Knee':     [4, 5],
    'UpperArm': [16, 17],
    'Forearm':  [18, 19],
}
SEGMENTS = {**PRIMARY_SEGMENTS, **SECONDARY_SEGMENTS}

LUMBAR_JOINTS = [1, 2, 3, 6, 9]

SENSOR_TO_JOINT = [18, 19, 1, 2, 15, 0]

NO_HEAD_COMBOS = {k: v for k, v in amass.combos.items() if 4 not in v}

_PALETTE = ['#1565C0', '#C62828', '#2E7D32', '#F57F17', '#6A1B9A', '#00838F']
COMBO_COLORS = {name: _PALETTE[i % len(_PALETTE)]
                for i, name in enumerate(NO_HEAD_COMBOS)}

SENSOR_COUNT_COLORS = {1: '#EF5350', 2: '#42A5F5'}

SMPL_KINTREE = [
    (0,1),(0,2),(0,3),
    (1,4),(2,5),
    (4,7),(5,8),
    (7,10),(8,11),
    (3,6),(6,9),
    (9,12),(9,13),(9,14),
    (12,15),
    (13,16),(14,17),
    (16,18),(17,19),
    (18,20),(19,21),
    (20,22),(21,23),
]
_LUMBAR_BONES = {(0,3),(3,6),(6,9)}

SEG_EN = {
    'Lumbar':   'Lumbar(j3)',
    'Thoracic': 'Thoracic(j6,9)',
    'Hip':      'Hip(j1,2)',
    'Knee':     'Knee(j4,5)',
    'UpperArm': 'UpperArm(j16,17)',
    'Forearm':  'Forearm(j18,19)',
}


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def angle_between_rotmats(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angular error in degrees. R1, R2: [..., 3, 3] → [...]"""
    R = R1.transpose(-1, -2) @ R2
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def moving_average(x: np.ndarray, window: int = 15) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='same')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_long_sequences(amass_dir: Path, min_frames: int, max_seqs: int):
    """Load full (unwindowed) sequences that are at least min_frames long."""
    pt_files = sorted(amass_dir.glob('*.pt'))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {amass_dir}.")

    unlimited = (max_seqs <= 0)
    seqs = []
    print(f"Scanning {len(pt_files)} AMASS files for sequences >= {min_frames} frames "
          f"({min_frames / datasets.fps:.0f}s) ...")
    for fpath in pt_files:
        try:
            data = torch.load(fpath, map_location='cpu')
        except Exception as e:
            print(f"  Skip {fpath.name}: {e}")
            continue
        for i, (acc, ori, pose, tran) in enumerate(
                zip(data['acc'], data['ori'], data['pose'], data['tran'])):
            if pose.shape[0] >= min_frames:
                seqs.append({
                    'acc':    acc.float(),
                    'ori':    ori.float(),
                    'pose':   pose.float(),
                    'tran':   tran.float(),
                    'source': f"{fpath.stem}[{i}]",
                })
            if not unlimited and len(seqs) >= max_seqs:
                break
        if not unlimited and len(seqs) >= max_seqs:
            break
    print(f"  -> {len(seqs)} qualifying sequences found.")
    return seqs


# ---------------------------------------------------------------------------
# SlimeVR algorithm: FK parent propagation
# ---------------------------------------------------------------------------

def slimevr_predict(ori: torch.Tensor, combo_indices: list,
                    bodymodel: art.model.ParametricModel) -> torch.Tensor:
    """
    SlimeVR pose estimation:
      - Instrumented joints get sensor orientation directly.
      - Non-instrumented joints inherit parent's global orientation.
      - Convert global -> local via inverse_kinematics_R.

    ori: [T, 6, 3, 3].  Returns: [T, 24, 3, 3] local rotation matrices.
    """
    T = ori.shape[0]
    parent = bodymodel.parent

    sensor_joints = {SENSOR_TO_JOINT[5]}           # pelvis always present
    for s in combo_indices:
        if s != 4:
            sensor_joints.add(SENSOR_TO_JOINT[s])

    R_global = torch.eye(3).view(1, 1, 3, 3).expand(T, 24, -1, -1).clone()
    R_global[:, SENSOR_TO_JOINT[5]] = ori[:, 5]    # pelvis
    for s in combo_indices:
        if s != 4:
            R_global[:, SENSOR_TO_JOINT[s]] = ori[:, s]

    for j in range(1, 24):
        if j not in sensor_joints:
            R_global[:, j] = R_global[:, parent[j]]

    return bodymodel.inverse_kinematics_R(
        R_global.view(T, -1)).view(T, 24, 3, 3)


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_slimevr(ori, gt_pose, combo_indices, bodymodel):
    """Returns rot_err [T, 24] in degrees (no translation estimate)."""
    pose_pred = slimevr_predict(ori, combo_indices, bodymodel)
    T = gt_pose.shape[0]
    return angle_between_rotmats(pose_pred[:T], gt_pose)


# ---------------------------------------------------------------------------
# Run evaluation for one combo
# ---------------------------------------------------------------------------

def evaluate_combo(combo_name, combo_indices, sequences, bodymodel, max_frames,
                   out_dir):
    """Evaluate one combo over all sequences. Returns averaged error arrays."""
    _CKPT = os.path.join(out_dir, f'.eval_ckpt_{combo_name}.npz')

    rot_sum   = np.zeros((max_frames, 24))
    count     = np.zeros(max_frames)
    start_idx = 0

    if os.path.exists(_CKPT):
        ck = np.load(_CKPT)
        rot_sum   = ck['rot_sum']
        count     = ck['count']
        start_idx = int(ck['seqs_done'])
        print(f"    [Resume] combo={combo_name}: {start_idx}/{len(sequences)} done")

    for idx, seq in enumerate(tqdm.tqdm(sequences, desc=f"  {combo_name}", leave=False)):
        if idx < start_idx:
            continue
        T = min(seq['pose'].shape[0], max_frames)
        gt_pose = seq['pose'][:T]
        ori     = seq['ori'][:T]

        rot_err = eval_slimevr(ori, gt_pose, combo_indices, bodymodel)
        rot_sum[:T] += rot_err.numpy()
        count[:T]   += 1.0

        np.savez(_CKPT, rot_sum=rot_sum, count=count, seqs_done=idx + 1)

    if os.path.exists(_CKPT):
        os.remove(_CKPT)

    valid = count > 0
    rot_avg = np.where(valid[:, None], rot_sum / np.maximum(count[:, None], 1), 0.0)

    return {
        'rot':       rot_avg,
        'tran':      np.zeros(max_frames),
        'fk_rot':    rot_avg,
        'n_sensors': len(combo_indices),
        'n_seqs':    int(count[0]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_SUBTITLE = '(SlimeVR = FK parent propagation — no learned model)'


def plot_timeseries(all_results, max_frames, fps, out_dir):
    """Figure 1: drift curves per body segment, one line per combo."""
    t = np.arange(max_frames) / fps

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

        # Reference: mean across all combos (= aggregate FK)
        mean_y = np.mean([res['rot'][:, joint_idx].mean(axis=1)
                          for res in all_results.values()], axis=0)
        ax.plot(t, moving_average(mean_y), color='gray', linestyle='--',
                linewidth=1.2, label='Combo mean (ref)')

        prefix = '[*] ' if is_primary else ''
        ax.set_title(f"{prefix}{SEG_EN[seg_name]}",
                     fontsize=11, fontweight='bold' if is_primary else 'normal')
        ax.set_ylabel('Angle error (deg)', fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_frames / fps)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        f'SlimeVR rotation drift by body segment  [*]=lumbar-rehab primary  '
        f'(no head sensor)\n{_SUBTITLE}',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig1_drift_timeseries.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_combo_comparison(all_results, max_frames, fps, checkpoint_s, out_dir):
    """Figure 2: grouped bar chart — all combos × all segments at checkpoint_s."""
    fps_cp      = min(int(checkpoint_s * fps), max_frames - 1)
    seg_names   = list(SEGMENTS.keys())
    combo_names = list(all_results.keys())
    n_combos    = len(combo_names)
    n_segs      = len(seg_names)

    x     = np.arange(n_segs)
    bar_w = 0.8 / n_combos

    fig, ax = plt.subplots(figsize=(14, 6))

    for ci, (combo_name, res) in enumerate(all_results.items()):
        vals = [res['rot'][fps_cp, joint_idx].mean()
                for joint_idx in SEGMENTS.values()]
        offset = (ci - n_combos / 2 + 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w,
               label=f"{combo_name}({res['n_sensors']}s)",
               color=COMBO_COLORS[combo_name], alpha=0.85)

    # Reference: combo mean per segment
    for si, (seg_name, joint_idx) in enumerate(SEGMENTS.items()):
        ref_val = np.mean([res['rot'][fps_cp, joint_idx].mean()
                           for res in all_results.values()])
        ax.plot([si - 0.4, si + 0.4], [ref_val, ref_val],
                color='black', linestyle='--', linewidth=1.5)

    for si, seg_name in enumerate(seg_names):
        if seg_name in PRIMARY_SEGMENTS:
            ax.axvspan(si - 0.45, si + 0.45, alpha=0.06, color='gold', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"[*]{SEG_EN[s]}" if s in PRIMARY_SEGMENTS else SEG_EN[s] for s in seg_names],
        fontsize=9, rotation=10)
    ax.set_ylabel('Mean angle error (deg)', fontsize=11)
    ax.set_title(
        f'SlimeVR combo comparison at {checkpoint_s}s  --  [*]=lumbar primary  '
        f'--  dashed=combo mean\n{_SUBTITLE}',
        fontsize=10, fontweight='bold')
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

    ax.set_xticks(counts)
    ax.set_xlabel('Number of sensors', fontsize=12)
    ax.set_ylabel('Lumbar error (deg)', fontsize=12)
    ax.set_title(
        f'SlimeVR: sensor count vs lumbar accuracy at {checkpoint_s}s\n{_SUBTITLE}',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, 'fig3_sensor_count_vs_lumbar.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_translation(all_results, max_frames, fps, out_dir):
    """Figure 4: placeholder — SlimeVR does not estimate global translation."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5,
            'Translation estimation not available.\n'
            'SlimeVR outputs rotation only.\n'
            '(No root-velocity or foot-contact integration.)',
            ha='center', va='center', fontsize=16, color='gray',
            transform=ax.transAxes)
    ax.set_axis_off()
    fig.suptitle('Global Translation Drift — SlimeVR (N/A)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'fig4_translation_drift.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

def _draw_skel(ax, joints, color, title, lumbar_err=None):
    ax.cla()
    j = joints - joints[0]
    for (a, b) in SMPL_KINTREE:
        is_lumbar = (a, b) in _LUMBAR_BONES or (b, a) in _LUMBAR_BONES
        c  = '#FF6F00' if is_lumbar else color
        lw = 3.5      if is_lumbar else 1.8
        ax.plot([j[a,0], j[b,0]], [j[a,1], j[b,1]], '-', color=c, lw=lw)
    ax.scatter(j[:,0], j[:,1], c=color, s=18, zorder=5)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.25, 1.85)
    ax.set_aspect('equal')
    ax.axis('off')
    lbl = title + (f'\nlumbar: {lumbar_err:.1f}deg' if lumbar_err is not None else '')
    ax.set_title(lbl, fontsize=9, pad=3)


def generate_video(combo_name, combo_indices, seq, bodymodel,
                   fps, out_dir, max_seconds=30, render_fps=10, seq_idx=0):
    """
    Side-by-side skeleton video: GT (green) | SlimeVR/FK (blue).
    Orange bones = lumbar chain.  Requires ffmpeg on PATH.
    """
    try:
        from matplotlib.animation import FFMpegWriter
    except Exception as e:
        print(f"  Skipped (FFMpegWriter unavailable): {e}")
        return

    stride  = max(1, round(fps / render_fps))
    T       = min(seq['pose'].shape[0], int(max_seconds * fps))

    gt_pose = seq['pose'][:T]
    gt_tran = seq['tran'][:T]
    ori     = seq['ori'][:T]

    with torch.no_grad():
        pose_pred = slimevr_predict(ori, combo_indices, bodymodel)[:T]
        _, gt_joints   = bodymodel.forward_kinematics(gt_pose,   tran=gt_tran)
        _, pred_joints = bodymodel.forward_kinematics(pose_pred, tran=gt_tran)

    gt_joints   = gt_joints.cpu().numpy()
    pred_joints = pred_joints.cpu().numpy()

    lumbar_err = angle_between_rotmats(
        pose_pred[:, LUMBAR_JOINTS], gt_pose[:, LUMBAR_JOINTS]).mean(-1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    fig.patch.set_facecolor('#111122')
    for ax in axes:
        ax.set_facecolor('#111122')

    video_path = os.path.join(out_dir, f'video_slimevr_{combo_name}_{seq_idx:04d}.mp4')
    writer = FFMpegWriter(fps=render_fps,
                          metadata={'title': f'drift-slimevr-{combo_name}'})
    frames = list(range(0, T, stride))
    print(f"  {len(frames)} frames at {render_fps}fps -> {video_path}")

    with writer.saving(fig, video_path, dpi=100):
        for t in frames:
            _draw_skel(axes[0], gt_joints[t],   '#43A047', 'Ground Truth')
            _draw_skel(axes[1], pred_joints[t], '#1E88E5',
                       f'SlimeVR ({combo_name})', lumbar_err[t])
            fig.suptitle(f't = {t/fps:.1f}s    orange = lumbar spine',
                         color='white', fontsize=11)
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

    seg_names    = list(SEGMENTS.keys())
    primary_flag = ['*' if s in PRIMARY_SEGMENTS else ' ' for s in seg_names]
    col_w = 10

    header  = f"{'Combo':<10} {'Sensors':>7} {'Time':>5}  "
    header += ''.join(f"{f}{s:>{col_w-1}}" for f, s in zip(primary_flag, seg_names))
    header += f"  {'Lumbar':>{col_w}}  {'Tran(m)':>{col_w}}"
    sep = '-' * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for combo_name, res in all_results.items():
        for cp in checkpoints:
            frame = min(int(cp * fps), max_frames - 1)
            row  = f"{combo_name:<10} {res['n_sensors']:>7} {cp:>4}s  "
            for joint_idx in SEGMENTS.values():
                val = float(res['rot'][frame, joint_idx].mean())
                row += f"  {val:>{col_w}.1f}"
            lumbar = float(res['rot'][frame, LUMBAR_JOINTS].mean())
            row += f"  {lumbar:>{col_w}.1f}  {'N/A':>{col_w}}"
            print(row)
        print()

    print(f"{'Mean':<10} {'  -':>7}  {'all':>4}  ", end='')
    for joint_idx in SEGMENTS.values():
        mean_val = np.mean([res['rot'][-1, joint_idx].mean()
                            for res in all_results.values()])
        print(f"  {mean_val:>{col_w}.1f}", end='')
    mean_lumbar = np.mean([res['rot'][-1, LUMBAR_JOINTS].mean()
                           for res in all_results.values()])
    print(f"  {mean_lumbar:>{col_w}.1f}  {'N/A':>{col_w}}")
    print(sep)
    print("*=lumbar-rehab primary  Lumbar=joints[1,2,3,6,9] mean")
    print("SlimeVR = FK parent propagation — no learned model, no translation estimate")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SlimeVR (FK parent propagation) drift evaluation — lumbar focus.')
    parser.add_argument('--model', default=None,
                        help='Unused for SlimeVR (no learned model). Accepted for '
                             'interface consistency but silently ignored.')
    parser.add_argument('--combos', nargs='+', default=['all'],
                        help='"all" or specific names: --combos lp rp lw_rp rw_lp')
    parser.add_argument('--amass_dir', default=None,
                        help='Dir with processed AMASS .pt files '
                             '(default: paths.processed_datasets from config)')
    parser.add_argument('--min_frames', type=int, default=1800,
                        help='Min sequence length in frames (default 1800 = 60s)')
    parser.add_argument('--max_seqs', type=int, default=0,
                        help='Max sequences to use (default 0 = all qualifying)')
    parser.add_argument('--max_seconds', type=int, default=120,
                        help='Time window for plots in seconds (default 120)')
    parser.add_argument('--compare_at', type=int, default=60,
                        help='Time checkpoint (s) for bar/scatter plots (default 60)')
    parser.add_argument('--out_dir', default='drift_results',
                        help='Output directory (default: drift_results)')
    parser.add_argument('--video_seconds', type=int, default=30,
                        help='Video length in seconds (default 30)')
    parser.add_argument('--video_fps', type=int, default=10,
                        help='Render FPS for video (default 10; lower = faster)')
    parser.add_argument('--no_video', action='store_true',
                        help='Skip video generation (metrics and figures only)')
    args = parser.parse_args()

    if args.model is not None:
        print("[INFO] --model argument is ignored for SlimeVR (no learned model).")

    if args.combos == ['all']:
        selected = NO_HEAD_COMBOS
    else:
        invalid = [c for c in args.combos if c not in NO_HEAD_COMBOS]
        if invalid:
            raise ValueError(f"Unknown combo(s): {invalid}. "
                             f"Valid no-head combos: {list(NO_HEAD_COMBOS)}")
        selected = {k: NO_HEAD_COMBOS[k] for k in args.combos}

    fps        = datasets.fps
    max_frames = args.max_seconds * fps
    amass_dir  = Path(args.amass_dir) if args.amass_dir else paths.processed_datasets

    print("Method      : SlimeVR (FK parent propagation, no neural network)")
    print(f"AMASS dir   : {amass_dir}")
    print(f"Combos      : {list(selected.keys())}")
    print(f"Min length  : {args.min_frames} frames ({args.min_frames/fps:.0f}s)")

    bodymodel = art.model.ParametricModel(str(paths.smpl_file))

    sequences = load_long_sequences(amass_dir, args.min_frames, args.max_seqs)
    if not sequences:
        print("No sequences found. Adjust --min_frames or --amass_dir.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}
    for combo_name, combo_indices in selected.items():
        print(f"\n-- Evaluating combo: {combo_name}  "
              f"sensors={combo_indices}  n={len(combo_indices)} --")
        all_results[combo_name] = evaluate_combo(
            combo_name, combo_indices, sequences, bodymodel, max_frames, args.out_dir)

    print("\nGenerating figures ...")
    plot_timeseries(all_results, max_frames, fps, args.out_dir)
    plot_combo_comparison(all_results, max_frames, fps, args.compare_at, args.out_dir)
    plot_sensor_count(all_results, max_frames, fps, args.compare_at, args.out_dir)
    plot_translation(all_results, max_frames, fps, args.out_dir)

    print_summary(all_results, max_frames, fps)

    save_dict = {}
    for combo_name, res in all_results.items():
        save_dict[f'{combo_name}_rot'] = res['rot']
        save_dict[f'{combo_name}_fk']  = res['fk_rot']
    save_dict['fps']    = fps
    save_dict['combos'] = list(all_results.keys())
    np_path = os.path.join(args.out_dir, 'drift_data.npz')
    np.savez(np_path, **save_dict)
    print(f"\nRaw arrays saved: {np_path}")

    if not args.no_video:
        print(f'\nGenerating videos for {len(sequences)} sequences ...')
        for i, vid_seq in enumerate(sequences):
            print(f"  [{i+1}/{len(sequences)}] seq={vid_seq['source']}")
            for combo_name, combo_indices in selected.items():
                try:
                    generate_video(
                        combo_name, combo_indices, vid_seq,
                        bodymodel, fps, args.out_dir,
                        max_seconds=args.video_seconds,
                        render_fps=args.video_fps,
                        seq_idx=i,
                    )
                except Exception as e:
                    print(f'    Video failed ({combo_name}): {e}')

    print(f"\nAll outputs in:  {os.path.abspath(args.out_dir)}/")


if __name__ == '__main__':
    main()
