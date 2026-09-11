"""WheelPoser bridge evaluation on shared MobilePoser DIP or AMASS sequences.

WheelPoser (UIST 2025) is an upper-body-only IMU pose estimator designed for
wheelchair users.  It uses 4 sensors: left wrist (idx 0), right wrist (idx 1),
head (idx 4), and pelvis/root (idx 5) in the MobilePoser sensor layout.

Sensor mapping:
    Bridge slot 0 → lw  → WheelPoser joint_set.WheelPoser[0]  (left wrist)
    Bridge slot 1 → rw  → WheelPoser joint_set.WheelPoser[1]  (right wrist)
    Bridge slot 4 → hd  → WheelPoser joint_set.WheelPoser[2]  (head)
    Bridge slot 5 → root→ WheelPoser joint_set.WheelPoser[3]  (pelvis)

Output: 16 upper-body joints (joint_set.upper_body).
Lower-body joints are zeroed (identity rotation) for standard_results.

Run from code/WheelPoser/:
    python evaluate_bridge.py --suite dip  --checkpoint-dir checkpoints/3_Stage_500 \
        --out-dir /path/to/results
    python evaluate_bridge.py --suite drift --checkpoint-dir checkpoints/3_Stage_500 \
        --out-dir /path/to/results
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
_ACC_SCALE = 30.0

# Bridge sensor slots used by WheelPoser
_BRIDGE_SLOTS = [0, 1, 4, 5]   # lw, rw, head, root

# SMPL upper-body joint indices (WheelPoser's pred_joints_set)
_UPPER_BODY = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

_LEAVE = "14"   # default leave-one-out experiment tag (leave subject 14 out)


def _field(seq, name):
    return seq[name] if isinstance(seq, dict) else getattr(seq, name)


# ── model loading ─────────────────────────────────────────────────────────────

def _get_checkpoints(experiment_name: str, model_names: list, leave_one_out: str,
                     ckpt_root: Path) -> dict:
    """Find best checkpoint files for each model stage."""
    result = {}
    for model_name in model_names:
        pattern = f"{experiment_name}*{model_name}*leave_{leave_one_out}_out*"
        candidates = sorted(ckpt_root.glob(f"{pattern}/**/*.ckpt"))
        if not candidates:
            # Also try without leave-one-out pattern
            candidates = sorted(ckpt_root.glob(f"{experiment_name}*{model_name}*/**/*.ckpt"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found for {model_name} under {ckpt_root} "
                f"(pattern: {pattern})"
            )
        result[model_name] = str(candidates[-1])
    return result


def _load_model(ckpt_dir: Path, leave: str, device: torch.device):
    """Load Three_Stage_Global_WheelPoser from checkpoints."""
    from src.config import Config, joint_set
    from src.models.utils import get_model
    from src.models.LSTMs.Three_Stage_Global.Three_Stage_Global_WheelPoser_Wrapper import (
        Three_Stage_Global_WheelPoser,
    )

    experiment = "3_Stage_500"
    stage_names = [
        f"IMU2Leaf_WheelPoser_AMASS",
        f"Leaf2Full_WheelPoser_AMASS",
        f"Full2Pose_WheelPoser_AMASS",
    ]
    fine_names = [
        f"IMU2Leaf_WheelPoser_WHEELPOSER",
        f"Leaf2Full_WheelPoser_WHEELPOSER",
        f"Full2Pose_WheelPoser_WHEELPOSER",
    ]

    ckpt_root = ckpt_dir if ckpt_dir.exists() else _DIR / "checkpoints"

    amass_ckpts = _get_checkpoints(experiment, stage_names, leave, ckpt_root)
    fine_ckpts  = _get_checkpoints(experiment, fine_names,  leave, ckpt_root)

    def _cfg(joints_set, exp_setup=None, **kw):
        return Config(
            experiment=experiment, model="bridge_eval",
            project_root_dir=str(_DIR),
            joints_set=joints_set,
            pred_joints_set=joint_set.upper_body,
            normalize=True, r6d=True, loss_type="mse",
            use_joint_loss=False, mkdir=False,
            upper_body_only=True,
            exp_setup=exp_setup,
            **kw,
        )

    amass_i2l = get_model(_cfg(joint_set.WheelPoser)).load_from_checkpoint(
        amass_ckpts[stage_names[0]], config=_cfg(joint_set.WheelPoser))
    amass_l2f = get_model(_cfg(joint_set.WheelPoser)).load_from_checkpoint(
        amass_ckpts[stage_names[1]], config=_cfg(joint_set.WheelPoser))
    amass_f2p = get_model(_cfg(joint_set.WheelPoser)).load_from_checkpoint(
        amass_ckpts[stage_names[2]], config=_cfg(joint_set.WheelPoser))

    fine_cfg = _cfg(joint_set.WheelPoser, exp_setup=f"leave_{leave}_out", upsample_copies=7)
    fine_i2l = get_model(fine_cfg, pretrained=amass_i2l).load_from_checkpoint(
        fine_ckpts[fine_names[0]], config=fine_cfg, pretrained_model=amass_i2l)
    fine_l2f = get_model(fine_cfg, pretrained=amass_l2f).load_from_checkpoint(
        fine_ckpts[fine_names[1]], config=fine_cfg, pretrained_model=amass_l2f)
    fine_f2p = get_model(fine_cfg, pretrained=amass_f2p).load_from_checkpoint(
        fine_ckpts[fine_names[2]], config=fine_cfg, pretrained_model=amass_f2p)

    for m in (amass_i2l, amass_l2f, amass_f2p, fine_i2l, fine_l2f, fine_f2p):
        m.eval()

    wheelposer = Three_Stage_Global_WheelPoser(
        config=fine_cfg,
        imu2leaf=fine_i2l,
        leaf2full=fine_l2f,
        full2pose=fine_f2p,
        num_past_frame=20,
        num_future_frame=5,
        physics=False,
    ).to(device)
    return wheelposer


# ── IMU feature preparation ───────────────────────────────────────────────────

def _prepare_imu(acc: torch.Tensor, ori: torch.Tensor) -> torch.Tensor:
    """Build WheelPoser's 48-D input from bridge data.

    Bridge slots [0,1,4,5] → [lw, rw, hd, root].
    WheelPoser expects 4 × 12D (6D orientation + 3D acc + 3D zeros) = 48D.
    We use the 6D rotation representation (first two columns of 3×3).
    """
    # Select the four sensor slots
    sel_acc = acc[:, _BRIDGE_SLOTS, :]           # [T, 4, 3]
    sel_ori = ori[:, _BRIDGE_SLOTS, :, :]        # [T, 4, 3, 3]

    # Normalize acceleration (WheelPoser uses acc_scale = 30 internally)
    sel_acc_norm = sel_acc / _ACC_SCALE           # [T, 4, 3]

    # 6D rotation: first two columns of 3×3 → [T, 4, 6]
    r6d = sel_ori[..., :2, :].transpose(-1, -2).reshape(sel_ori.shape[0], 4, 6)

    # Concatenate: [r6d | acc | zeros] → [T, 4, 12]
    zeros = torch.zeros(*sel_acc_norm.shape, device=acc.device)
    imu_feat = torch.cat([r6d, sel_acc_norm, zeros], dim=-1)   # [T, 4, 12]

    return imu_feat.reshape(imu_feat.shape[0], -1)             # [T, 48]


# ── per-sequence inference ────────────────────────────────────────────────────

@torch.no_grad()
def _run_seq(model, acc: torch.Tensor, ori: torch.Tensor,
             gt_pose: torch.Tensor, device: torch.device) -> np.ndarray:
    """Run WheelPoser offline on one sequence.  Returns rot_err [T, 24] degrees."""
    imu_input = _prepare_imu(acc, ori).to(device)    # [T, 48]
    pred_pose = model.forward_offline(imu_input)      # [T, 24, 3, 3] on CPU

    T = gt_pose.shape[0]

    # Build full-body prediction: upper body from WheelPoser, lower body = identity
    pred_full = torch.eye(3).view(1, 1, 3, 3).expand(T, 24, 3, 3).clone()
    pred_full[:, _UPPER_BODY] = pred_pose[:T, _UPPER_BODY]

    return angle_between_rotmats(pred_full, gt_pose[:T]).numpy()


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(sequences, model, device: torch.device, max_frames: int,
             out_dir: Path) -> tuple:
    """Evaluate with per-sequence checkpointing."""
    _CKPT = str(out_dir / ".eval_ckpt_wheelposer.npz")

    rot_sum = np.zeros((max_frames, 24), dtype=np.float64)
    count   = np.zeros(max_frames, dtype=np.int64)
    start_idx = 0

    if os.path.exists(_CKPT):
        ck = np.load(_CKPT)
        rot_sum   = ck["rot_sum"]
        count     = ck["count"]
        start_idx = int(ck["seqs_done"])
        print(f"  [Resume] WheelPoser: {start_idx}/{len(sequences)} seqs done")

    for idx, seq in enumerate(tqdm(sequences, desc="WheelPoser")):
        if idx < start_idx:
            continue
        pose = _field(seq, "pose")
        acc  = _field(seq, "acc")
        ori  = _field(seq, "ori")
        T = min(len(pose), max_frames)
        try:
            rot_err = _run_seq(model, acc[:T], ori[:T], pose[:T], device)
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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="WheelPoser bridge evaluation on shared processed data"
    )
    parser.add_argument("--suite", choices=("dip", "drift"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=_DIR / "checkpoints",
                        help="Root directory containing WheelPoser checkpoint folders")
    parser.add_argument("--leave", default=_LEAVE,
                        help=f"Leave-one-out subject tag (default: {_LEAVE})")
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

    print(f"Loading WheelPoser (leave-{args.leave}-out) from {args.checkpoint_dir} ...")
    model = _load_model(args.checkpoint_dir, args.leave, device)

    if args.suite == "dip":
        sequences = load_dip_sequences(
            args.processed_root, args.min_frames, max_frames
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
        raise RuntimeError("No eligible sequences found")

    rotation, count = evaluate(sequences, model, device, max_frames, args.out_dir)

    output = args.out_dir / f"wheelposer_{args.suite}.npz"
    np.savez_compressed(output, rotation=rotation, count=count, fps=FPS)
    write_standard_result(
        args.out_dir, "wheelposer", args.suite,
        rotation, count, FPS,
        sensor_count=4, sensor_slots=_BRIDGE_SLOTS,
    )
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
