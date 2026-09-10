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
_CODE = _DIR.parents[1]
_BASE = _CODE / "base_mobileposer"
os.chdir(str(_DIR))
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_BASE))

from benchmarks.bridge_data import load_amass_sequences, load_dip_sequences
from benchmarks.standard_results import write_standard_result
from drift_eval_common import angle_between_rotmats


FPS = 30


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
        length = min(len(sequence.pose), max_frames)
        model.reset()
        imu = normalize_and_concat(sequence.acc[:length], sequence.ori[:length]).to(device)
        pose_pred, tran_pred = model.forward_offline(imu)
        pose_pred, tran_pred = pose_pred.cpu(), tran_pred.cpu()
        rotation_sum[:length] += angle_between_rotmats(
            pose_pred, sequence.pose[:length]
        ).numpy()
        translation_sum[:length] += (tran_pred - (
            sequence.tran[:length] - sequence.tran[:1]
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    max_frames = args.max_seconds * FPS
    if args.suite == "dip":
        sequences = load_dip_sequences(args.processed_root, args.min_frames, max_frames)
    else:
        sequences = load_amass_sequences(args.processed_root, args.min_frames, max_frames, args.max_seqs)
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
    print("Saved: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
