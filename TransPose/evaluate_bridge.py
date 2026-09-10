"""Run TransPose on the shared MobilePoser DIP or AMASS processed sequences."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


_DIR = Path(__file__).resolve().parent


def _find_repo_root() -> Path:
    for candidate in (_DIR, *_DIR.parents):
        code = candidate / "code"
        if (candidate / "benchmarks").is_dir() and (
            (candidate / "base_mobileposer").is_dir()
            or (code / "base_mobileposer").is_dir()
        ):
            return candidate
    raise RuntimeError("Could not locate MotionRe repository root")


_ROOT = _find_repo_root()
_CODE = _ROOT / "code" if (_ROOT / "code" / "base_mobileposer").is_dir() else _ROOT
_BASE = _CODE / "base_mobileposer"
os.chdir(str(_DIR))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_BASE))

from benchmarks.bridge_data import load_dip_sequences
from benchmarks.standard_results import write_standard_result
from benchmarks.video import render_comparison_video
from drift_eval_common import angle_between_rotmats, load_long_sequences
import mobileposer.articulate as art
from mobileposer.config import paths


FPS = 30


def _field(sequence, name):
    return sequence[name] if isinstance(sequence, dict) else getattr(sequence, name)


def _load_model(weights: Path, device: torch.device):
    import config

    config.paths.weights_file = str(weights)
    from net import TransPoseNet

    return TransPoseNet().to(device).eval()


@torch.no_grad()
def evaluate(sequences, model, device: torch.device, max_frames: int):
    from utils import normalize_and_concat

    rotation_sum = np.zeros((max_frames, 24), dtype=np.float64)
    translation_sum = np.zeros(max_frames, dtype=np.float64)
    count = np.zeros(max_frames, dtype=np.int64)
    for sequence in tqdm(sequences, desc="TransPose"):
        pose = _field(sequence, "pose")
        acc = _field(sequence, "acc")
        ori = _field(sequence, "ori")
        tran = _field(sequence, "tran")
        length = min(len(pose), max_frames)
        model.reset()
        imu = normalize_and_concat(acc[:length], ori[:length]).to(device)
        pose_pred, tran_pred = model.forward_offline(imu)
        pose_pred, tran_pred = pose_pred.cpu(), tran_pred.cpu()
        rotation_sum[:length] += angle_between_rotmats(
            pose_pred, pose[:length]
        ).numpy()
        translation_sum[:length] += (tran_pred - (
            tran[:length] - tran[:1]
        )).norm(dim=-1).numpy()
        count[:length] += 1
    valid = count > 0
    rotation = np.full_like(rotation_sum, np.nan)
    translation = np.full(max_frames, np.nan)
    rotation[valid] = rotation_sum[valid] / count[valid, None]
    translation[valid] = translation_sum[valid] / count[valid]
    return rotation, translation, count


def main() -> int:
    parser = argparse.ArgumentParser(description="TransPose bridge evaluation on shared processed data")
    parser.add_argument("--suite", choices=("dip", "drift"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=_BASE / "data/processed_datasets")
    parser.add_argument("--min-frames", type=int, default=300)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--action-manifest", type=Path, default=None)
    parser.add_argument("--max-per-action", type=int, default=100)
    parser.add_argument("--video-seconds", type=int, default=30)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    max_frames = args.max_seconds * FPS
    if args.suite == "dip":
        sequences = load_dip_sequences(args.processed_root, args.min_frames, max_frames)
    else:
        sequences = load_long_sequences(
            args.min_frames, args.max_seqs or 0,
            amass_dir=args.processed_root,
            action_manifest=args.action_manifest,
            max_per_action=args.max_per_action,
        )
    if not sequences:
        raise RuntimeError("No eligible sequences found")
    device = torch.device(args.device)
    model = _load_model(args.model, device)
    rotation, translation, count = evaluate(sequences, model, device, max_frames)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "transpose_{}.npz".format(args.suite)
    np.savez_compressed(output, rotation=rotation, translation=translation, count=count, fps=FPS)
    write_standard_result(
        args.out_dir, "transpose", args.suite, rotation, count, FPS, 6,
        [0, 1, 2, 3, 4, 5], translation,
    )
    if not args.no_video and args.suite == "drift":
        bodymodel = art.model.ParametricModel(str(paths.smpl_file))
        for index, sequence in enumerate(sequences):
            pose = _field(sequence, "pose")
            acc = _field(sequence, "acc")
            ori = _field(sequence, "ori")
            tran = _field(sequence, "tran")
            length = min(len(pose), args.video_seconds * FPS)
            model.reset()
            from utils import normalize_and_concat
            imu = normalize_and_concat(
                acc[:length], ori[:length]
            ).to(device)
            pose_pred, tran_pred = model.forward_offline(imu)
            pose_pred, tran_pred = pose_pred.cpu(), tran_pred.cpu()
            pose_fk = torch.eye(3).view(1, 1, 3, 3).expand(length, 24, 3, 3).clone()
            root_ori = ori[:length, 5]
            for sensor, joint in {0: 18, 1: 19, 2: 1, 3: 2, 5: 0}.items():
                pose_fk[:, joint] = root_ori.transpose(-1, -2) @ ori[:length, sensor]
            tran_pred = tran_pred - tran_pred[:1] + tran[:1]
            with torch.no_grad():
                _, gt_joints = bodymodel.forward_kinematics(
                    pose[:length], tran=tran[:length]
                )
                _, pred_joints = bodymodel.forward_kinematics(
                    pose_pred, tran=tran_pred
                )
                _, fk_joints = bodymodel.forward_kinematics(
                    pose_fk, tran=tran[:length]
                )
            errors = angle_between_rotmats(
                pose_pred, pose[:length]
            )[:, [1, 2, 3, 6, 9]].mean(-1).numpy()
            fk_errors = angle_between_rotmats(
                pose_fk, pose[:length]
            )[:, [1, 2, 3, 6, 9]].mean(-1).numpy()
            render_comparison_video(
                gt_joints=gt_joints.numpy(),
                method_joints=pred_joints.numpy(),
                fk_joints=fk_joints.numpy(),
                method="TransPose", combo="full_6s",
                sequence=sequence, fps=FPS, out_dir=args.out_dir,
                seq_idx=index, max_seconds=args.video_seconds,
                render_fps=args.video_fps, method_errors=errors,
                fk_errors=fk_errors,
            )
    print("Saved: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
