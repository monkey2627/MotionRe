"""GlobalPose bridge evaluation on shared MobilePoser DIP or AMASS sequences.

GlobalPose (SIGGRAPH Asia 2025) takes per-frame body-frame acceleration, angular
velocity, and bone orientation.  For synthetic AMASS data the sensors are
perfectly calibrated, so:

    RMB[t,s]  = ori[t,s]          (sensor/body orientation in world frame)
    aM[t,s]   = acc[t,s]/30 + g   (net world-frame accel + gravity, m/s²)
    wM[t,s]   = angular_velocity computed from ori via finite differences

For DIP-IMU real data, the native GlobalPose test file is loaded if it exists
at  data/test_datasets/dipimu.pt; otherwise the bridge data is used with the
same conversion (real IMU data convention).

Run from code/GlobalPose/:
    python evaluate_bridge.py --suite dip  --model data/weights.pt --out-dir /path
    python evaluate_bridge.py --suite drift --model data/weights.pt --out-dir /path
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
        if (candidate / "benchmarks").is_dir() and (
            (candidate / "base_mobileposer").is_dir()
            or (candidate / "code" / "base_mobileposer").is_dir()
        ):
            return candidate
    raise RuntimeError("Could not locate MotionRe repository root")


_ROOT = _find_repo_root()
_CODE = _ROOT / "code" if (_ROOT / "code" / "base_mobileposer").is_dir() else _ROOT
_BASE = _CODE / "base_mobileposer"

os.chdir(str(_DIR))
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_BASE))

from benchmarks.bridge_data import load_dip_sequences, load_amass_sequences
from benchmarks.standard_results import write_standard_result
from drift_eval_common import angle_between_rotmats, load_long_sequences

FPS = 30
_GRAVITY = torch.tensor([0.0, -9.8, 0.0])   # world-frame, SMPL Y-up
_ACC_SCALE = 30.0                             # MobilePoser acc normalization


def _field(seq, name):
    return seq[name] if isinstance(seq, dict) else getattr(seq, name)


# ── data helpers ──────────────────────────────────────────────────────────────

def _load_native_dip(weights_dir: Path):
    """Load GlobalPose's own DIP-IMU test set if available."""
    candidate = weights_dir / "data" / "test_datasets" / "dipimu.pt"
    if not candidate.exists():
        candidate = _DIR / "data" / "test_datasets" / "dipimu.pt"
    if candidate.exists():
        return torch.load(str(candidate), map_location="cpu"), True
    return None, False


# ── angular velocity ──────────────────────────────────────────────────────────

def _angular_velocity(ori: torch.Tensor, fps: float) -> torch.Tensor:
    """Numerical finite-difference angular velocity from ori [T,S,3,3] → [T,S,3]."""
    T = ori.shape[0]
    w = torch.zeros(*ori.shape[:-2], 3)
    for t in range(T):
        if t == 0:
            dR = (ori[1] - ori[0]) * fps
        elif t == T - 1:
            dR = (ori[-1] - ori[-2]) * fps
        else:
            dR = (ori[t + 1] - ori[t - 1]) * (fps / 2.0)
        Om = ori[t].transpose(-1, -2) @ dR   # [..., 3, 3] skew-symmetric
        w[t, ..., 0] = Om[..., 2, 1]
        w[t, ..., 1] = Om[..., 0, 2]
        w[t, ..., 2] = Om[..., 1, 0]
    return w


# ── bridge data → GlobalPose input ────────────────────────────────────────────

def _to_globalpose_inputs(acc: torch.Tensor, ori: torch.Tensor):
    """Convert bridge-format tensors to GlobalPose model inputs.

    Bridge format:
        acc  [T, 6, 3]    net linear accel, world frame, scaled ×30
        ori  [T, 6, 3, 3] sensor orientation (world ← sensor)

    GlobalPose inputs per frame:
        aM   [6, 3]   net accel + gravity in world frame (m/s²)
        wM   [6, 3]   angular velocity in world frame (rad/s)
        RMB  [6, 3, 3] body/sensor orientation in world frame
    """
    aM = acc / _ACC_SCALE + _GRAVITY.view(1, 1, 3)   # [T, 6, 3]
    wM = _angular_velocity(ori, FPS)                   # [T, 6, 3]
    RMB = ori                                          # [T, 6, 3, 3]
    return aM, wM, RMB


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model(weights: Path, device: torch.device):
    from net import GPNet
    net = GPNet().to(device).eval()
    state = torch.load(str(weights), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state, strict=False)
    return net


# ── per-sequence inference ────────────────────────────────────────────────────

@torch.no_grad()
def _run_bridge_seq(model, acc, ori, gt_pose, device):
    """Run GlobalPose on one bridge sequence.  Returns rot_err [T,24] in degrees."""
    T = gt_pose.shape[0]
    aM, wM, RMB = _to_globalpose_inputs(acc[:T], ori[:T])

    # GlobalPose is stateful; initialize with ground-truth first pose
    model.rnn_initialize(gt_pose[:1].to(device))

    pose_list = []
    for t in range(T):
        p, _ = model.forward_frame(
            aM[t].to(device),
            wM[t].to(device),
            RMB[t].to(device),
        )
        pose_list.append(p.detach().cpu())

    pose_pred = torch.stack(pose_list)   # [T, 24, 3, 3]
    return angle_between_rotmats(pose_pred, gt_pose).numpy()


@torch.no_grad()
def _run_native_seq(model, aS, wS, RIS, RIM, RSB, gt_pose, device):
    """Run GlobalPose on one sequence using GlobalPose's native calibrated format."""
    T = gt_pose.shape[0]
    g = _GRAVITY.to(device)

    RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB).to(device)      # [T, 6, 3, 3]
    aM = (RIM.transpose(1, 2).matmul(RIS)
          .matmul(aS.unsqueeze(-1)).squeeze(-1) + g).to(device)         # [T, 6, 3]
    wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1).to(device)

    model.rnn_initialize(gt_pose[:1].to(device))

    pose_list = []
    for t in range(T):
        p, _ = model.forward_frame(aM[t], wM[t], RMB[t])
        pose_list.append(p.detach().cpu())

    pose_pred = torch.stack(pose_list)
    return angle_between_rotmats(pose_pred, gt_pose).numpy()


# ── evaluation loop (bridge sequences) ───────────────────────────────────────

def evaluate_bridge(sequences, model, device, max_frames: int, out_dir: Path):
    """Evaluate on bridge-format sequences with per-sequence checkpointing."""
    _CKPT = str(out_dir / ".eval_ckpt_globalpose.npz")

    rot_sum = np.zeros((max_frames, 24), dtype=np.float64)
    count   = np.zeros(max_frames, dtype=np.int64)
    start_idx = 0

    if os.path.exists(_CKPT):
        ck = np.load(_CKPT)
        rot_sum   = ck["rot_sum"]
        count     = ck["count"]
        start_idx = int(ck["seqs_done"])
        print(f"  [Resume] GlobalPose: {start_idx}/{len(sequences)} seqs done")

    for idx, seq in enumerate(tqdm(sequences, desc="GlobalPose")):
        if idx < start_idx:
            continue
        pose = _field(seq, "pose")
        acc  = _field(seq, "acc")
        ori  = _field(seq, "ori")
        T = min(len(pose), max_frames)
        try:
            rot_err = _run_bridge_seq(model, acc[:T], ori[:T], pose[:T], device)
            rot_sum[:T] += rot_err
            count[:T]   += 1
        except Exception as exc:
            src = _field(seq, "source") if not isinstance(seq, dict) else seq.get("source", "?")
            print(f"\n  skip {src}: {type(exc).__name__}: {exc}")
        np.savez(_CKPT, rot_sum=rot_sum, count=count, seqs_done=idx + 1)

    if os.path.exists(_CKPT):
        os.remove(_CKPT)

    valid = count > 0
    rotation = np.full_like(rot_sum, np.nan)
    rotation[valid] = rot_sum[valid] / count[valid, None]
    return rotation, count


# ── evaluation loop (native GlobalPose DIP format) ───────────────────────────

def evaluate_native_dip(data, model, device, max_frames: int, out_dir: Path):
    """Evaluate on GlobalPose native DIP-IMU test data with checkpointing."""
    n_seqs = len(data["pose"])
    _CKPT = str(out_dir / ".eval_ckpt_globalpose_native.npz")

    rot_sum = np.zeros((max_frames, 24), dtype=np.float64)
    count   = np.zeros(max_frames, dtype=np.int64)
    start_idx = 0

    if os.path.exists(_CKPT):
        ck = np.load(_CKPT)
        rot_sum   = ck["rot_sum"]
        count     = ck["count"]
        start_idx = int(ck["seqs_done"])
        print(f"  [Resume] GlobalPose native DIP: {start_idx}/{n_seqs} seqs done")

    import articulate as art
    for idx in tqdm(range(n_seqs), desc="GlobalPose/DIP-native"):
        if idx < start_idx:
            continue
        pose = art.math.axis_angle_to_rotation_matrix(data["pose"][idx]).view(-1, 24, 3, 3)
        T = min(pose.shape[0], max_frames)
        try:
            rot_err = _run_native_seq(
                model,
                data["aS"][idx][:T],
                data["wS"][idx][:T],
                data["RIS"][idx][:T] if data["RIS"][idx].ndim == 3 else data["RIS"][idx].unsqueeze(0).expand(T, -1, -1, -1),
                data["RIM"][idx][:T],
                data["RSB"][idx][:T] if data["RSB"][idx].ndim == 3 else data["RSB"][idx].unsqueeze(0).expand(T, -1, -1, -1),
                pose[:T],
                device,
            )
            rot_sum[:T] += rot_err
            count[:T]   += 1
        except Exception as exc:
            print(f"\n  skip seq {idx}: {type(exc).__name__}: {exc}")
        np.savez(_CKPT, rot_sum=rot_sum, count=count, seqs_done=idx + 1)

    if os.path.exists(_CKPT):
        os.remove(_CKPT)

    valid = count > 0
    rotation = np.full_like(rot_sum, np.nan)
    rotation[valid] = rot_sum[valid] / count[valid, None]
    return rotation, count


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="GlobalPose bridge evaluation on shared processed data"
    )
    parser.add_argument("--suite", choices=("dip", "drift"), required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="Path to GlobalPose weights (.pt)")
    parser.add_argument("--processed-root", type=Path,
                        default=_BASE / "data/processed_datasets")
    parser.add_argument("--min-frames", type=int, default=300)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--action-manifest", type=Path, default=None)
    parser.add_argument("--max-per-action", type=int, default=100)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    max_frames = args.max_seconds * FPS
    device = torch.device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GlobalPose model from {args.model} ...")
    model = _load_model(args.model, device)

    if args.suite == "dip":
        # Prefer native DIP data if available (original GlobalPose calibrated format)
        native_data, has_native = _load_native_dip(args.model.parent)
        if has_native:
            print("Using native GlobalPose DIP-IMU test data.")
            rotation, count = evaluate_native_dip(
                native_data, model, device, max_frames, args.out_dir
            )
        else:
            print("Native DIP data not found; using MobilePoser bridge format.")
            sequences = load_dip_sequences(
                args.processed_root, args.min_frames, max_frames
            )
            if not sequences:
                raise RuntimeError("No eligible DIP sequences found")
            rotation, count = evaluate_bridge(
                sequences, model, device, max_frames, args.out_dir
            )
    else:
        sequences = load_long_sequences(
            args.min_frames,
            args.max_seqs or 0,
            amass_dir=args.processed_root,
            action_manifest=args.action_manifest,
            max_per_action=args.max_per_action,
        )
        if not sequences:
            raise RuntimeError("No eligible AMASS sequences found")
        rotation, count = evaluate_bridge(
            sequences, model, device, max_frames, args.out_dir
        )

    output = args.out_dir / f"globalpose_{args.suite}.npz"
    np.savez_compressed(
        output, rotation=rotation, count=count, fps=FPS
    )
    write_standard_result(
        args.out_dir, "globalpose", args.suite,
        rotation, count, FPS, 6, [0, 1, 2, 3, 4, 5],
    )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
