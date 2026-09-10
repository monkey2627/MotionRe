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


def _load_model(weights: Path):
    import config

    config.paths.weights_file = str(weights)
    from net import PIP

    return PIP().eval()


@torch.no_grad()
def evaluate(sequences, model, max_frames: int):
    rotation_sum = np.zeros((max_frames, 24), dtype=np.float64)
    translation_sum = np.zeros(max_frames, dtype=np.float64)
    count = np.zeros(max_frames, dtype=np.int64)
    for sequence in tqdm(sequences, desc="PIP"):
        length = min(len(sequence.pose), max_frames)
        pose_pred, tran_pred = model.predict(
            sequence.acc[:length], sequence.ori[:length], sequence.pose[:1]
        )
        rotation_sum[:length] += angle_between_rotmats(
            pose_pred.cpu(), sequence.pose[:length]
        ).numpy()
        translation_sum[:length] += (tran_pred.cpu() - (
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
    parser = argparse.ArgumentParser(description="PIP bridge evaluation on shared processed data")
    parser.add_argument("--suite", choices=("dip", "drift"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=_BASE / "data/processed_datasets")
    parser.add_argument("--min-frames", type=int, default=300)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    max_frames = args.max_seconds * FPS
    if args.suite == "dip":
        sequences = load_dip_sequences(args.processed_root, args.min_frames, max_frames)
    else:
        sequences = load_amass_sequences(
            args.processed_root, args.min_frames, max_frames, args.max_seqs
        )
    if not sequences:
        raise RuntimeError("No eligible sequences found")
    model = _load_model(args.model)
    rotation, translation, count = evaluate(sequences, model, max_frames)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "pip_{}.npz".format(args.suite)
    np.savez_compressed(output, rotation=rotation, translation=translation, count=count, fps=FPS)
    write_standard_result(
        args.out_dir, "pip", args.suite, rotation, count, FPS, 6,
        [0, 1, 2, 3, 4, 5], translation,
    )
    print("Saved: {}".format(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
