"""Evaluate a native DIP-IMU TensorFlow checkpoint on real DIP-IMU signals."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


_DIR = Path(__file__).resolve().parent
_BASE = _DIR.parents[1] / "base_mobileposer"
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_BASE))

import evaluate_drift as native
from benchmarks.standard_results import write_standard_result
from mobileposer.config import datasets, paths


FPS = int(datasets.fps)
SEGMENTS = {
    "lumbar": [3, 6, 9],
    "hips": [1, 2],
    "knees": [4, 5],
    "upper_arms": [16, 17],
    "forearms": [18, 19],
}


def load_dip(min_frames: int):
    data = torch.load(paths.processed_datasets / "eval" / "dip_test.pt", map_location="cpu")
    sequences = []
    for acc, ori, pose in zip(data["acc"], data["ori"], data["pose"]):
        if pose.shape[0] >= min_frames:
            sequences.append((acc.float(), ori.float(), pose.float()))
    print("DIP-IMU test: {}/{} sequences >= {} frames".format(
        len(sequences), len(data["pose"]), min_frames
    ))
    return sequences


def main() -> int:
    parser = argparse.ArgumentParser(description="Native DIP-IMU evaluation on real DIP-IMU data")
    parser.add_argument("--model", required=True, help="DIP-IMU model directory")
    parser.add_argument("--min_frames", type=int, default=300)
    parser.add_argument("--max_seconds", type=int, default=60)
    parser.add_argument("--out_dir", default="dip_results")
    args = parser.parse_args()

    sequences = load_dip(args.min_frames)
    if not sequences:
        return 0
    model_dir = os.path.abspath(args.model)
    session = native.tf.Session()
    model = native.load_dip_model(model_dir, session)
    max_frames = args.max_seconds * FPS
    rotation_sum = np.zeros((max_frames, 24), dtype=np.float64)
    count = np.zeros(max_frames, dtype=np.int64)
    for acc, ori, pose in tqdm(sequences, desc="DIP-IMU / DIP-IMU"):
        length = min(len(pose), max_frames)
        error = native.eval_dip(model, acc[:length], ori[:length], pose[:length]).numpy()
        rotation_sum[:length] += error
        count[:length] += 1
    session.close()

    valid = count > 0
    rotation = np.full_like(rotation_sum, np.nan)
    rotation[valid] = rotation_sum[valid] / count[valid, None]
    for seconds in (10, 30, 60):
        frame = seconds * FPS - 1
        if frame < len(count) and count[frame]:
            print("@{}s: lumbar5={:.2f}deg".format(
                seconds, float(rotation[frame, native.LUMBAR_JOINTS].mean())
            ))
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "dip_imu_dip.npz", rotation=rotation, count=count, fps=FPS)
    write_standard_result(
        output, "dip-imu", "dip", rotation, count, FPS, 6,
        [0, 1, 2, 3, 4, 5],
    )
    print("Saved: {}".format(output / "dip_imu_dip.npz"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
