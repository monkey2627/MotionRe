"""
Evaluate the six no-head 5-IMU models and render GT/Prediction videos.

Example:
  python -m mobileposer.evaluate_no_head_5imu \
      --checkpoint-root checkpoints/no_head_5imu \
      --data-root data/no_head_5imu_processed \
      --output-root results/no_head_5imu_eval \
      --video-count 2

The video is a side-by-side skeleton comparison:
  Ground Truth | Prediction
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from mobileposer.config import model_config, paths
from mobileposer.no_head_layouts import LAYOUTS
from mobileposer.utils.model_utils import load_model
import mobileposer.articulate as art


FPS = 30
# no_head_layouts rotates AMASS from its source basis into the model's training
# basis.  Videos are easier to inspect in the Unity-style y-up basis.
TRAINING_WORLD_ROT = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
JOINT_NAMES = {
    0: "pelvis", 1: "l_hip", 2: "r_hip", 3: "spine1", 4: "l_knee",
    5: "r_knee", 6: "spine2", 7: "l_ankle", 8: "r_ankle", 9: "spine3",
    10: "l_foot", 11: "r_foot", 12: "neck", 13: "l_clavicle",
    14: "r_clavicle", 15: "head", 16: "l_shoulder", 17: "r_shoulder",
    18: "l_elbow", 19: "r_elbow", 20: "l_wrist", 21: "r_wrist",
    22: "l_hand", 23: "r_hand",
}
EDGES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (4, 7), (5, 8),
    (7, 10), (8, 11), (3, 6), (6, 9), (9, 12), (9, 13), (9, 14),
    (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20),
    (19, 21), (20, 22), (21, 23),
]
PARTS = {
    "lumbar": [1, 2, 3, 6, 9],
    "thoracic": [6, 9],
    "hips": [1, 2],
    "knees": [4, 5],
    "upperarms": [16, 17],
    "forearms": [18, 19],
    "all_predicted": [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19],
}


def geodesic_deg(pred, gt):
    diff = pred.transpose(-1, -2) @ gt
    trace = diff[..., 0, 0] + diff[..., 1, 1] + diff[..., 2, 2]
    return torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)) * (180.0 / math.pi)


def prepare_imu(acc, ori):
    acc5 = acc[:, :5].float() / 30.0
    ori5 = ori[:, :5].float()
    return torch.cat([acc5.flatten(1), ori5.flatten(1)], dim=1)


def load_sequences(data_dir: Path, max_seq=None):
    sequences = []
    for pt_path in sorted(data_dir.glob("*.pt")):
        data = torch.load(pt_path, map_location="cpu")
        metadata = data.get("metadata") or []
        for local_idx, (acc, ori, pose, tran) in enumerate(
            zip(data["acc"], data["ori"], data["pose"], data["tran"])
        ):
            source = f"{pt_path.stem}[{local_idx}]"
            if local_idx < len(metadata) and isinstance(metadata[local_idx], dict):
                source = metadata[local_idx].get("source", source)
            sequences.append({
                "source": source,
                "acc": acc,
                "ori": ori,
                "pose": pose,
                "tran": tran,
            })
            if max_seq is not None and len(sequences) >= max_seq:
                return sequences
    return sequences


def _project(joints, view):
    if view == "front":
        return joints[:, [0, 1]]
    if view == "side":
        return joints[:, [2, 1]]
    return joints[:, [0, 2]]


def _normalize_pair_for_video(gt_joints, pred_joints):
    """Use one GT-relative coordinate frame so GT/pred disagreement remains visible."""
    gt = gt_joints.astype(np.float32, copy=True)
    pred = pred_joints.astype(np.float32, copy=True)

    pelvis_ground = gt[:, 0, [0, 2]].copy()
    for positions in (gt, pred):
        positions[:, :, 0] -= pelvis_ground[:, 0:1]
        positions[:, :, 2] -= pelvis_ground[:, 1:2]

    floor_y = float(np.percentile(gt[:, :, 1], 1.0))
    gt[:, :, 1] -= floor_y
    pred[:, :, 1] -= floor_y
    return gt, pred


def _draw_skeleton(ax, joints, color, label, view, linewidth=2.0, alpha=1.0):
    xy = _project(joints, view)
    for a, b in EDGES:
        ax.plot(
            xy[[a, b], 0],
            xy[[a, b], 1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            label=label if a == EDGES[0][0] and b == EDGES[0][1] else None,
        )
    ax.scatter(xy[:, 0], xy[:, 1], color=color, s=11, alpha=alpha, zorder=3)


def _setup_axis(ax, title, x_limits, y_limits, show_floor=True):
    ax.set_facecolor("#101418")
    ax.set_title(title, color="white", fontsize=10, pad=5)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if show_floor:
        ax.axhline(0.0, color="#5A626B", linewidth=0.8, alpha=0.8)
    else:
        ax.axhline(0.0, color="#343B43", linewidth=0.6, alpha=0.8)
        ax.axvline(0.0, color="#343B43", linewidth=0.6, alpha=0.8)


def render_comparison_video(gt_joints, pred_joints, output_path: Path,
                            render_fps=10, max_seconds=30):
    import cv2

    n = min(len(gt_joints), len(pred_joints), int(max_seconds * FPS))
    if n == 0:
        return
    stride = max(1, round(FPS / render_fps))
    frames = []
    gt_joints, pred_joints = _normalize_pair_for_video(
        np.asarray(gt_joints[:n]), np.asarray(pred_joints[:n])
    )

    ground = np.concatenate(
        [gt_joints[:, :, [0, 2]], pred_joints[:, :, [0, 2]]],
        axis=1,
    )
    horizontal = max(1.2, float(np.percentile(np.abs(ground), 99.5)) + 0.15)
    y_values = np.concatenate([gt_joints[:, :, 1].reshape(-1), pred_joints[:, :, 1].reshape(-1)])
    y_min = min(-0.1, float(np.percentile(y_values, 0.5)) - 0.15)
    y_max = max(2.0, float(np.percentile(y_values, 99.5)) + 0.15)
    vertical_limits = (y_min, y_max)
    ground_limits = (-horizontal, horizontal)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, n, stride):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), dpi=100)
        fig.patch.set_facecolor("#101418")
        gt = gt_joints[i]
        pred = pred_joints[i]
        panels = [
            ("Front: x / y", "front", ground_limits, vertical_limits, True),
            ("Side: z / y", "side", ground_limits, vertical_limits, True),
            ("Top: x / z", "top", ground_limits, ground_limits, False),
        ]
        for ax, (title, view, x_limits, y_limits, show_floor) in zip(axes, panels):
            _setup_axis(ax, title, x_limits, y_limits, show_floor=show_floor)
            _draw_skeleton(ax, gt, "#42A5F5", "GT", view, linewidth=2.4, alpha=0.95)
            _draw_skeleton(ax, pred, "#EF5350", "Pred", view, linewidth=1.9, alpha=0.9)
            legend = ax.legend(loc="upper right", frameon=False, fontsize=8)
            for text in legend.get_texts():
                text.set_color("white")
        fig.suptitle(
            f"GT vs Prediction    frame={i}    time={i / FPS:.2f}s",
            color="white",
            fontsize=11,
            y=0.98,
        )
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(frame)
        plt.close(fig)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        render_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open MP4 writer: {output_path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  video: {output_path}")


def _to_y_up_for_video(joints: torch.Tensor) -> torch.Tensor:
    """Undo the training-world rotation for y-up visual inspection only."""
    inverse_rotation = TRAINING_WORLD_ROT.transpose(0, 1).to(joints)
    return joints @ inverse_rotation.transpose(0, 1)


@torch.no_grad()
def evaluate_model(model, sequences, bodymodel, device, video_count,
                   video_dir, render_fps, max_seconds, overwrite, video_y_up):
    sums = {name: 0.0 for name in PARTS}
    count = {name: 0 for name in PARTS}
    tran_sum = 0.0
    tran_count = 0

    for seq_idx, seq in enumerate(tqdm(sequences, desc="sequences")):
        pose_gt = seq["pose"].float()
        tran_gt = seq["tran"].float()
        imu = prepare_imu(seq["acc"], seq["ori"]).to(device)

        model.reset()
        pose_pred, _, tran_pred, _ = model.forward_offline(
            imu.unsqueeze(0), [imu.shape[0]]
        )
        pose_pred = pose_pred.cpu()
        tran_pred = tran_pred.cpu()
        T = min(len(pose_gt), len(pose_pred))
        pose_gt = pose_gt[:T]
        pose_pred = pose_pred[:T]
        tran_gt = tran_gt[:T]
        tran_pred = tran_pred[:T]

        rot_err = geodesic_deg(pose_pred, pose_gt)
        for name, joints in PARTS.items():
            value = rot_err[:, joints].sum().item()
            sums[name] += value
            count[name] += T * len(joints)

        tran_err = (tran_pred - tran_pred[:1] - (tran_gt - tran_gt[:1])).norm(dim=-1)
        tran_sum += tran_err.sum().item()
        tran_count += T

        if seq_idx < video_count:
            video_path = video_dir / f"{seq_idx:04d}.mp4"
            if overwrite or not video_path.exists():
                _, gt_joints = bodymodel.forward_kinematics(pose_gt, tran=tran_gt)
                _, pred_joints = bodymodel.forward_kinematics(pose_pred, tran=tran_pred)
                if video_y_up:
                    gt_joints = _to_y_up_for_video(gt_joints)
                    pred_joints = _to_y_up_for_video(pred_joints)
                render_comparison_video(
                    gt_joints.cpu().numpy(),
                    pred_joints.cpu().numpy(),
                    video_path,
                    render_fps=render_fps,
                    max_seconds=max_seconds,
                )

    result = {name: sums[name] / max(count[name], 1) for name in PARTS}
    result["translation_mean_m"] = tran_sum / max(tran_count, 1)
    result["n_sequences"] = len(sequences)
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate no-head 5-IMU models with GT/pred videos.")
    parser.add_argument("--checkpoint-root", default=str(paths.root_dir / "checkpoints/no_head_5imu"))
    parser.add_argument("--data-root", default=str(paths.root_dir / "data/no_head_5imu_processed"))
    parser.add_argument("--output-root", default=str(paths.root_dir / "results/no_head_5imu_eval"))
    parser.add_argument("--layouts", nargs="+", default=["all"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seq", type=int, default=None)
    parser.add_argument("--video-count", type=int, default=2)
    parser.add_argument("--max-seconds", type=int, default=30)
    parser.add_argument("--render-fps", type=int, default=10)
    parser.add_argument("--video-y-up", action="store_true",
                        help="Undo the no-head training-world rotation before rendering videos.")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--videos-only", action="store_true",
                        help="Only render GT/pred videos for the first --video-count sequences; do not write metrics or comparison.csv.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.videos_only and args.no_video:
        parser.error("--videos-only cannot be used with --no-video")
    if args.videos_only and args.video_count <= 0:
        parser.error("--videos-only requires --video-count > 0")

    layout_names = list(LAYOUTS) if args.layouts == ["all"] else args.layouts
    unknown = [x for x in layout_names if x not in LAYOUTS]
    if unknown:
        raise ValueError(f"Unknown layouts: {unknown}. Available: {list(LAYOUTS)}")

    checkpoint_root = Path(args.checkpoint_root)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    # MobilePoserNet uses model_config.device for internal temporary tensors.
    # Keep it consistent with the command-line device, especially for CPU runs.
    model_config.device = device
    bodymodel = art.model.ParametricModel(paths.smpl_file, device="cpu")
    all_results = {}

    for layout_name in layout_names:
        model_path = checkpoint_root / layout_name / "1" / "base_model.pth"
        data_dir = data_root / layout_name
        layout_out = output_root / layout_name
        video_dir = layout_out / "videos"
        if not model_path.exists():
            print(f"[skip] {layout_name}: missing {model_path}")
            continue
        if not data_dir.exists():
            print(f"[skip] {layout_name}: missing {data_dir}")
            continue

        print(f"\n=== {layout_name} ===")
        print(f"model: {model_path}")
        model = load_model(str(model_path)).to(device).eval()
        sequence_limit = args.max_seq
        if args.videos_only:
            sequence_limit = args.video_count if args.max_seq is None else min(args.max_seq, args.video_count)
        sequences = load_sequences(data_dir, sequence_limit)
        if not sequences:
            print(f"[skip] {layout_name}: no sequences")
            continue

        result = evaluate_model(
            model, sequences, bodymodel, device,
            0 if args.no_video else args.video_count,
            video_dir, args.render_fps, args.max_seconds, args.overwrite,
            args.video_y_up,
        )
        if args.videos_only:
            print(f"[videos-only] {layout_name}: rendered up to {args.video_count} videos; metrics were not written.")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        all_results[layout_name] = result
        layout_out.mkdir(parents=True, exist_ok=True)
        (layout_out / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.videos_only:
        print("\nVideos-only run finished. Existing metrics.json and comparison.csv were left unchanged.")
        return

    csv_path = output_root / "comparison.csv"
    fields = ["layout"] + list(PARTS) + ["translation_mean_m", "n_sequences"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for layout_name, result in all_results.items():
            writer.writerow({"layout": layout_name, **result})

    print("\n=== Summary ===")
    for layout_name, result in sorted(all_results.items(), key=lambda x: x[1]["all_predicted"]):
        print(
            f"{layout_name:28s} "
            f"all={result['all_predicted']:.3f} deg  "
            f"lumbar={result['lumbar']:.3f} deg  "
            f"tran={result['translation_mean_m']:.4f} m"
        )
    print(f"Saved comparison: {csv_path}")


if __name__ == "__main__":
    main()
