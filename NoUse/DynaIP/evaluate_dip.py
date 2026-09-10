"""Evaluate DynaIP on the shared preprocessed DIP-IMU test set.

The adapter intentionally reuses DynaIP's native inference implementation in
``evaluate_drift.py``.  It does not add synthetic drift/noise: DIP contains
the recorded IMU signals and is the real-sensor counterpart to the AMASS
drift benchmark.
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
_CODE = _DIR.parents[1]
_BASE = _CODE / "base_mobileposer"
os.chdir(str(_DIR))
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_BASE))

import articulate as art
import utils.config as cfg
from model.model import Poser
from drift_eval_common import LUMBAR_JOINTS, angle_between_rotmats
from benchmarks.standard_results import write_standard_result
from evaluate_drift import _infer_dynaip_local
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
    for acc, ori, pose, tran in zip(data["acc"], data["ori"], data["pose"], data["tran"]):
        if pose.shape[0] >= min_frames:
            sequences.append((acc.float(), ori.float(), pose.float(), tran.float()))
    print("DIP-IMU test: {}/{} sequences >= {} frames".format(
        len(sequences), len(data["pose"]), min_frames
    ))
    return sequences


@torch.no_grad()
def evaluate(sequences, model, device: str, max_frames: int):
    rotation_sum = np.zeros((max_frames, 24), dtype=np.float64)
    count = np.zeros(max_frames, dtype=np.int64)
    for acc, ori, pose, _ in tqdm(sequences, desc="DynaIP / DIP-IMU"):
        length = min(len(pose), max_frames)
        prediction = _infer_dynaip_local(model, acc[:length], ori[:length], device).cpu()
        error = angle_between_rotmats(prediction, pose[:length]).numpy()
        rotation_sum[:length] += error
        count[:length] += 1
    valid = count > 0
    rotation = np.full_like(rotation_sum, np.nan)
    rotation[valid] = rotation_sum[valid] / count[valid, None]
    return rotation, count


def print_summary(rotation: np.ndarray, count: np.ndarray) -> None:
    for seconds in (10, 30, 60):
        frame = seconds * FPS - 1
        if frame >= len(count) or not count[frame]:
            continue
        values = {name: float(rotation[frame, joints].mean()) for name, joints in SEGMENTS.items()}
        values["lumbar5"] = float(rotation[frame, LUMBAR_JOINTS].mean())
        print("@{}s: {}".format(seconds, "  ".join(
            "{}={:.2f}deg".format(name, value) for name, value in values.items()
        )))


def main() -> int:
    parser = argparse.ArgumentParser(description="DynaIP evaluation on real DIP-IMU data")
    parser.add_argument("--model", default="../../DynaIP/weights/DynaIP.pth")
    parser.add_argument("--min_frames", type=int, default=300)
    parser.add_argument("--max_seconds", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="dip_results")
    args = parser.parse_args()

    sequences = load_dip(args.min_frames)
    if not sequences:
        return 0
    model = Poser().to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.eval()
    rotation, count = evaluate(sequences, model, args.device, args.max_seconds * FPS)
    print_summary(rotation, count)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "dynaip_dip.npz", rotation=rotation, count=count, fps=FPS)
    write_standard_result(
        output, "dynaip", "dip", rotation, count, FPS, 6,
        [0, 1, 2, 3, 4, 5],
    )
    print("Saved: {}".format(output / "dynaip_dip.npz"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
