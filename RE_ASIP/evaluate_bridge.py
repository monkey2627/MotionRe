"""ASIP evaluation on the shared six-IMU bridge format.

ASIP predicts local SMPL rotations from six IMUs and does not predict root
translation.  The bridge therefore reports rotation metrics only and keeps
translation explicitly unavailable.
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
_ROOT = next(
    p for p in (_DIR, *_DIR.parents)
    if (p / "benchmarks").is_dir()
    and ((p / "code" / "base_mobileposer").is_dir() or (p / "base_mobileposer").is_dir())
)
_CODE = _ROOT / "code" if (_ROOT / "code" / "base_mobileposer").is_dir() else _ROOT
_BASE = _CODE / "base_mobileposer"
os.chdir(str(_DIR / "common"))
sys.path.insert(0, str(_DIR / "common"))
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_BASE))

from benchmarks.bridge_data import load_dip_sequences
from benchmarks.detailed_results import write_detailed_index, write_sequence_result
from benchmarks.standard_results import write_standard_result
from drift_eval_common import angle_between_rotmats, load_long_sequences

FPS = 30
WINDOW = 30
REDUCED = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]


def _field(seq, name):
    return seq[name] if isinstance(seq, dict) else getattr(seq, name)


def _make_input(acc: torch.Tensor, ori: torch.Tensor) -> torch.Tensor:
    """Convert shared world-frame data to ASIP's pelvis-relative [T,6,12]."""
    root = ori[:, 5]
    rel_ori = torch.cat((root.transpose(-1, -2).unsqueeze(1).matmul(ori[:, :5]), root.unsqueeze(1)), dim=1)
    rel_acc = torch.cat(((acc[:, :5] - acc[:, 5:]).unsqueeze(-2).matmul(root).squeeze(-2), acc[:, 5:]), dim=1)
    # ASIP's original preprocessing concatenates nine orientation values first.
    return torch.cat((rel_ori.flatten(2), rel_acc), dim=-1).float()


def _r6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def _load_model(weights: Path, device: torch.device):
    from modules import Inertial_PoseTransformer

    model = Inertial_PoseTransformer(
        num_frame=WINDOW, in_num_joints=6, out_num_joints=15,
        in_chans=12, embed_dim_ratio=32, depth=4, num_heads=8,
        mlp_ratio=2.0, qkv_bias=True, qk_scale=None, drop_path_rate=0.1,
        with_spatial_block=True, with_spatial_pos_embed=True,
        with_temporal_pos_embed=True, with_ssms=True, with_ssmt=True,
    ).to(device).eval()
    state = torch.load(str(weights), map_location=device)
    if isinstance(state, dict) and "model_pos" in state:
        state = state["model_pos"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model


@torch.no_grad()
def _predict(model, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.shape[0] < WINDOW:
        raise ValueError("sequence has fewer than {} frames".format(WINDOW))
    windows = x.unfold(0, WINDOW, 1).permute(0, 3, 1, 2).contiguous()
    outputs = []
    for batch in windows.split(256):
        y = model(batch.to(device))[:, -1]
        matrices = _r6d_to_matrix(y.view(-1, 15, 6)).cpu()
        pose = torch.eye(3).view(1, 1, 3, 3).expand(len(batch), 24, 3, 3).clone()
        pose[:, REDUCED] = matrices
        pose[:, 0] = x[WINDOW - 1::1][:len(batch), 5, :9].view(-1, 3, 3)
        outputs.append(pose)
    return torch.cat(outputs)


def evaluate(sequences, model, device, max_frames: int, out_dir: Path):
    rotation_sum = np.zeros((max_frames, 24), dtype=np.float64)
    count = np.zeros(max_frames, dtype=np.int64)
    records = []
    for idx, seq in enumerate(tqdm(sequences, desc="ASIP")):
        pose = _field(seq, "pose")
        length = min(len(pose), max_frames)
        source = _field(seq, "source") if not isinstance(seq, dict) else seq.get("source", "unknown")
        action = seq.get("action", "other") if isinstance(seq, dict) else "other"
        try:
            pred = _predict(model, _make_input(_field(seq, "acc")[:length], _field(seq, "ori")[:length]), device)
            target = pose[WINDOW - 1:length]
            error = angle_between_rotmats(pred, target).numpy()
            valid_len = len(error)
            rotation_sum[WINDOW - 1:WINDOW - 1 + valid_len] += error
            count[WINDOW - 1:WINDOW - 1 + valid_len] += 1
            records.append(write_sequence_result(out_dir, idx, source, action, error, None, FPS, configuration="full_6s"))
        except Exception as exc:
            records.append(write_sequence_result(out_dir, idx, source, action, None, None, FPS,
                                                 failure_reason="{}: {}".format(type(exc).__name__, exc),
                                                 configuration="full_6s"))
    rotation = np.full_like(rotation_sum, np.nan)
    valid = count > 0
    rotation[valid] = rotation_sum[valid] / count[valid, None]
    return rotation, count, records


def main() -> int:
    parser = argparse.ArgumentParser(description="ASIP evaluation on shared processed data")
    parser.add_argument("--suite", choices=("dip", "drift"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, default=_BASE / "data/processed_datasets")
    parser.add_argument("--min-frames", type=int, default=300)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--action-manifest", type=Path, default=None)
    parser.add_argument("--max-per-action", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    max_frames = args.max_seconds * FPS
    sequences = (load_dip_sequences(args.processed_root, args.min_frames, max_frames) if args.suite == "dip" else
                 load_long_sequences(args.min_frames, args.max_seqs or 0, amass_dir=args.processed_root,
                                     action_manifest=args.action_manifest, max_per_action=args.max_per_action))
    if not sequences:
        raise RuntimeError("No eligible sequences found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rotation, count, records = evaluate(sequences, _load_model(args.model, torch.device(args.device)),
                                        torch.device(args.device), max_frames, args.out_dir)
    np.savez_compressed(args.out_dir / "asip_{}.npz".format(args.suite), rotation=rotation, count=count, fps=FPS)
    write_standard_result(args.out_dir, "asip", args.suite, rotation, count, FPS, 6, [0, 1, 2, 3, 4, 5])
    write_detailed_index(args.out_dir, "asip", args.suite, FPS, records)
    print("Saved ASIP results to {}".format(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
