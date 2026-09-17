"""Render FBX sampled skeleton and fitted SMPL GT side by side."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mobileposer.articulate.model import ParametricModel
from mobileposer.config import paths
from mobileposer.evaluate_no_head_5imu import EDGES, FPS, _draw_skeleton, _normalize_pair_for_video, _setup_axis
from mobileposer.fit_ours_smpl import _FBX_BONES, _load_fbx_positions


FBX_TO_SMPL_EDGES = [
    ("pelvis", "femur_l"), ("femur_l", "tibia_l"), ("tibia_l", "talus_l"),
    ("talus_l", "toes_l"), ("pelvis", "femur_r"), ("femur_r", "tibia_r"),
    ("tibia_r", "talus_r"), ("talus_r", "toes_r"), ("pelvis", "lumbar_body"),
    ("lumbar_body", "thorax"), ("thorax", "head"), ("thorax", "humerus_l"),
    ("humerus_l", "ulna_l"), ("ulna_l", "hand_l"), ("thorax", "humerus_r"),
    ("humerus_r", "ulna_r"), ("ulna_r", "hand_r"),
]


def _fbx_to_24_joints(fbx_positions: torch.Tensor) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(_FBX_BONES)}
    out = torch.zeros(fbx_positions.shape[0], 24, 3)
    mapping = {
        0: "pelvis", 1: "femur_l", 2: "femur_r", 3: "lumbar_body",
        4: "tibia_l", 5: "tibia_r", 7: "talus_l", 8: "talus_r",
        9: "thorax", 10: "toes_l", 11: "toes_r", 12: "thorax",
        15: "head", 16: "humerus_l", 17: "humerus_r", 18: "ulna_l",
        19: "ulna_r", 20: "hand_l", 21: "hand_r", 22: "hand_l", 23: "hand_r",
    }
    for smpl_idx, fbx_name in mapping.items():
        out[:, smpl_idx] = fbx_positions[:, name_to_idx[fbx_name]]
    return out.numpy()


def _render_video(fbx_joints, smpl_joints, output_path: Path, render_fps: int, max_seconds: int):
    import cv2

    n = min(len(fbx_joints), len(smpl_joints), max_seconds * FPS)
    stride = max(1, round(FPS / render_fps))
    fbx_joints, smpl_joints = _normalize_pair_for_video(fbx_joints[:n], smpl_joints[:n])

    ground = np.concatenate([fbx_joints[:, :, [0, 2]], smpl_joints[:, :, [0, 2]]], axis=1)
    horizontal = max(1.2, float(np.percentile(np.abs(ground), 99.5)) + 0.15)
    y_values = np.concatenate([fbx_joints[:, :, 1].reshape(-1), smpl_joints[:, :, 1].reshape(-1)])
    vertical_limits = (
        min(-0.1, float(np.percentile(y_values, 0.5)) - 0.15),
        max(2.0, float(np.percentile(y_values, 99.5)) + 0.15),
    )
    ground_limits = (-horizontal, horizontal)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(0, n, stride):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), dpi=100)
        fig.patch.set_facecolor("#101418")
        panels = [
            ("Front: x / y", "front", ground_limits, vertical_limits, True),
            ("Side: z / y", "side", ground_limits, vertical_limits, True),
            ("Top: x / z", "top", ground_limits, ground_limits, False),
        ]
        for ax, (title, view, x_limits, y_limits, show_floor) in zip(axes, panels):
            _setup_axis(ax, title, x_limits, y_limits, show_floor=show_floor)
            _draw_skeleton(ax, fbx_joints[i], "#42A5F5", "FBX", view, linewidth=2.4, alpha=0.95)
            _draw_skeleton(ax, smpl_joints[i], "#EF5350", "SMPL fit", view, linewidth=1.9, alpha=0.9)
            legend = ax.legend(loc="upper right", frameon=False, fontsize=8)
            for text in legend.get_texts():
                text.set_color("white")
        fig.suptitle(f"FBX skeleton vs fitted SMPL GT    frame={i}    time={i / FPS:.2f}s", color="white", fontsize=11, y=0.98)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), render_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open MP4 writer: {output_path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render ours FBX skeleton against fitted SMPL GT.")
    parser.add_argument("--fbx-cache", type=Path, default=paths.processed_datasets / "ours_fbx_cache/1.fbx_joints.npz")
    parser.add_argument("--ours-smpl", type=Path, default=paths.eval_dir / "ours_smpl.pt")
    parser.add_argument("--output", type=Path, default=paths.root_dir / "results/ours_gt_debug/fbx_vs_smpl_gt.mp4")
    parser.add_argument("--max-seconds", type=int, default=30)
    parser.add_argument("--render-fps", type=int, default=10)
    args = parser.parse_args()

    fbx_positions, _ = _load_fbx_positions(args.fbx_cache)
    fbx_joints = _fbx_to_24_joints(fbx_positions)

    data = torch.load(args.ours_smpl, map_location="cpu")
    pose = data["pose"][0]
    tran = data["tran"][0]
    body = ParametricModel(paths.smpl_file)
    _, smpl_joints = body.forward_kinematics(pose, tran=tran)
    _render_video(fbx_joints, smpl_joints.numpy(), args.output, args.render_fps, args.max_seconds)


if __name__ == "__main__":
    main()
