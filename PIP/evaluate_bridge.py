"""Run PIP on shared MobilePoser DIP or AMASS processed sequences.

PIP receives the six sensors in the shared order and performs its own pelvis
relative normalisation in ``net.normalize_and_concat``.  Its native physics
optimiser is preserved unchanged.
"""

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


def _load_model(weights: Path):
    import config

    config.paths.weights_file = str(weights)
    from net import PIP

    return PIP().eval()


@torch.no_grad()
def evaluate(sequences, model, max_frames: int, out_dir: Path = None):
    _CKPT = str(out_dir / ".eval_ckpt_pip.npz") if out_dir else None

    rotation_sum = np.zeros((max_frames, 24), dtype=np.float64)
    translation_sum = np.zeros(max_frames, dtype=np.float64)
    count = np.zeros(max_frames, dtype=np.int64)
    start_idx = 0

    if _CKPT and os.path.exists(_CKPT):
        ck = np.load(_CKPT)
        rotation_sum    = ck["rotation_sum"]
        translation_sum = ck["translation_sum"]
        count           = ck["count"]
        start_idx       = int(ck["seqs_done"])
        print(f"  [Resume] PIP: {start_idx}/{len(sequences)} seqs done")

    for idx, sequence in enumerate(tqdm(sequences, desc="PIP")):
        if idx < start_idx:
            continue
        pose = _field(sequence, "pose")
        acc = _field(sequence, "acc")
        ori = _field(sequence, "ori")
        tran = _field(sequence, "tran")
        length = min(len(pose), max_frames)
        try:
            pose_pred, tran_pred = model.predict(
                acc[:length], ori[:length], pose[:1]
            )
        except Exception as exc:
            source = _field(sequence, "source") if not isinstance(sequence, dict) else sequence.get("source", "unknown")
            print(f"\n  skip {source}: {type(exc).__name__}: {exc}")
        else:
            rotation_sum[:length] += angle_between_rotmats(
                pose_pred.cpu(), pose[:length]
            ).numpy()
            translation_sum[:length] += (tran_pred.cpu() - (
                tran[:length] - tran[:1]
            )).norm(dim=-1).numpy()
            count[:length] += 1
        if _CKPT:
            np.savez(_CKPT, rotation_sum=rotation_sum, translation_sum=translation_sum,
                     count=count, seqs_done=idx + 1)

    if _CKPT and os.path.exists(_CKPT):
        os.remove(_CKPT)

    valid = count > 0
    rotation = np.full_like(rotation_sum, np.nan)
    translation = np.full(max_frames, np.nan)
    rotation[valid] = rotation_sum[valid] / count[valid, None]
    translation[valid] = translation_sum[valid] / count[valid]
    return rotation, translation, count


def main() -> int:
    parser = argparse.ArgumentParser(description="PIP bridge evaluation on shared processed data")
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
    model = _load_model(args.model)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rotation, translation, count = evaluate(sequences, model, max_frames, out_dir=args.out_dir)
    output = args.out_dir / "pip_{}.npz".format(args.suite)
    np.savez_compressed(output, rotation=rotation, translation=translation, count=count, fps=FPS)
    write_standard_result(
        args.out_dir, "pip", args.suite, rotation, count, FPS, 6,
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
            try:
                pose_pred, tran_pred = model.predict(
                    acc[:length], ori[:length], pose[:1]
                )
            except Exception as exc:
                source = _field(sequence, "source") if not isinstance(sequence, dict) else sequence.get("source", "unknown")
                print(f"\n  skip video {source}: {type(exc).__name__}: {exc}")
                continue
            pose_fk = torch.eye(3).view(1, 1, 3, 3).expand(length, 24, 3, 3).clone()
            root_ori = ori[:length, 5]
            for sensor, joint in {0: 18, 1: 19, 2: 1, 3: 2, 5: 0}.items():
                pose_fk[:, joint] = root_ori.transpose(-1, -2) @ ori[:length, sensor]
            tran_pred = tran_pred.cpu()
            tran_pred = tran_pred - tran_pred[:1] + tran[:1]
            with torch.no_grad():
                _, gt_joints = bodymodel.forward_kinematics(
                    pose[:length], tran=tran[:length]
                )
                _, pred_joints = bodymodel.forward_kinematics(
                    pose_pred.cpu(), tran=tran_pred
                )
                _, fk_joints = bodymodel.forward_kinematics(
                    pose_fk, tran=tran[:length]
                )
            errors = angle_between_rotmats(
                pose_pred.cpu(), pose[:length]
            )[:, [1, 2, 3, 6, 9]].mean(-1).numpy()
            fk_errors = angle_between_rotmats(
                pose_fk, pose[:length]
            )[:, [1, 2, 3, 6, 9]].mean(-1).numpy()
            render_comparison_video(
                gt_joints=gt_joints.numpy(),
                method_joints=pred_joints.numpy(),
                fk_joints=fk_joints.numpy(),
                method="PIP", combo="full_6s",
                sequence=sequence, fps=FPS, out_dir=args.out_dir,
                seq_idx=index, max_seconds=args.video_seconds,
                render_fps=args.video_fps, method_errors=errors,
                fk_errors=fk_errors,
            )
    print("Saved: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
