"""Render source skeleton and fitted SMPL GT side by side."""

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
from mobileposer.fit_ours_smpl import _HUMANPOSE_BONES, _load_fbx_positions, _load_humanpose_positions


FBX_TO_SMPL_EDGES = [
    ("pelvis", "femur_l"), ("femur_l", "tibia_l"), ("tibia_l", "talus_l"),
    ("talus_l", "toes_l"), ("pelvis", "femur_r"), ("femur_r", "tibia_r"),
    ("tibia_r", "talus_r"), ("talus_r", "toes_r"), ("pelvis", "lumbar_body"),
    ("lumbar_body", "thorax"), ("thorax", "head"), ("thorax", "humerus_l"),
    ("humerus_l", "ulna_l"), ("ulna_l", "hand_l"), ("thorax", "humerus_r"),
    ("humerus_r", "ulna_r"), ("ulna_r", "hand_r"),
]


def _source_to_24_joints(source_positions: torch.Tensor) -> np.ndarray:
    name_to_idx = {name: idx for idx, name in enumerate(_HUMANPOSE_BONES)}
    out = torch.zeros(source_positions.shape[0], 24, 3)
    def p(name):
        return source_positions[:, name_to_idx[name]]

    # Fill the intermediate SMPL joints too. Leaving clavicles/neck at zero
    # makes an otherwise valid FBX skeleton look folded toward the origin.
    out[:, 0] = p("pelvis")
    out[:, 1] = p("femur_l")
    out[:, 2] = p("femur_r")
    out[:, 3] = p("lumbar_body")
    out[:, 4] = p("tibia_l")
    out[:, 5] = p("tibia_r")
    out[:, 6] = (p("lumbar_body") + p("thorax")) * 0.5
    out[:, 7] = p("talus_l")
    out[:, 8] = p("talus_r")
    out[:, 9] = p("thorax")
    out[:, 10] = p("toes_l")
    out[:, 11] = p("toes_r")
    out[:, 12] = (p("thorax") + p("head")) * 0.5
    out[:, 13] = (p("thorax") + p("humerus_l")) * 0.5
    out[:, 14] = (p("thorax") + p("humerus_r")) * 0.5
    out[:, 15] = p("head")
    out[:, 16] = p("humerus_l")
    out[:, 17] = p("humerus_r")
    out[:, 18] = p("ulna_l")
    out[:, 19] = p("ulna_r")
    out[:, 20] = p("hand_l")
    out[:, 21] = p("hand_r")
    out[:, 22] = p("hand_l")
    out[:, 23] = p("hand_r")
    return out.numpy()


def _load_source_positions(path: Path):
    if path.suffix.lower() == ".json":
        return _load_humanpose_positions(path)
    return _load_fbx_positions(path)


def _render_video(source_joints, smpl_joints, output_path: Path, render_fps: int, max_seconds: int, source_label: str, smpl_label: str):
    import cv2

    n = min(len(source_joints), len(smpl_joints), max_seconds * FPS)
    stride = max(1, round(FPS / render_fps))
    source_joints, smpl_joints = _normalize_pair_for_video(source_joints[:n], smpl_joints[:n])

    ground = np.concatenate([source_joints[:, :, [0, 2]], smpl_joints[:, :, [0, 2]]], axis=1)
    horizontal = max(1.2, float(np.percentile(np.abs(ground), 99.5)) + 0.15)
    y_values = np.concatenate([source_joints[:, :, 1].reshape(-1), smpl_joints[:, :, 1].reshape(-1)])
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
            _draw_skeleton(ax, source_joints[i], "#42A5F5", source_label, view, linewidth=2.4, alpha=0.95)
            _draw_skeleton(ax, smpl_joints[i], "#EF5350", smpl_label, view, linewidth=1.9, alpha=0.9)
            legend = ax.legend(loc="upper right", frameon=False, fontsize=8)
            for text in legend.get_texts():
                text.set_color("white")
        fig.suptitle(f"{source_label} skeleton vs {smpl_label}    frame={i}    time={i / FPS:.2f}s", color="white", fontsize=11, y=0.98)
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
    parser = argparse.ArgumentParser(description="Render ours source skeleton against fitted SMPL GT.")
    parser.add_argument("--source-cache", "--fbx-cache", dest="source_cache", type=Path,
                        default=paths.processed_datasets / "ours_humanpose_cache/1.humanpose_joints.json")
    parser.add_argument("--ours-smpl", type=Path, default=paths.eval_dir / "ours_smpl.pt")
    parser.add_argument("--output", type=Path, default=paths.root_dir / "results/ours_gt_debug/humanpose_vs_smpl_gt.mp4")
    parser.add_argument("--max-seconds", type=int, default=30)
    parser.add_argument("--render-fps", type=int, default=10)
    args = parser.parse_args()

    source_positions, source_meta = _load_source_positions(args.source_cache)
    source_joints = _source_to_24_joints(source_positions)
    source_label = "HumanPose" if source_meta.get("source_type") == "unity_humanpose" else "FBX"

    data = torch.load(args.ours_smpl, map_location="cpu")
    pose = data["pose"][0]
    tran = data["tran"][0]
    shape = data.get("shape", [None])[0]
    body = ParametricModel(paths.smpl_file)
    _, smpl_joints = body.forward_kinematics(pose, shape=shape, tran=tran)
    metadata = data.get("metadata") or []
    is_retarget = bool(metadata) and metadata[0].get("source_type") == "unity_humanpose_rotation_retarget"
    smpl_label = "SMPL rotation retarget" if is_retarget else "SMPL fit"
    _render_video(source_joints, smpl_joints.detach().numpy(), args.output, args.render_fps, args.max_seconds, source_label, smpl_label)


if __name__ == "__main__":
    main()
