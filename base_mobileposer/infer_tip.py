#!/usr/bin/env python3
"""
TIP (ECCV 2022) inference on AMASS data for per-body-part error evaluation.

Runs the TIP kinematic model autoregressively on its pre-processed AMASS data
WITHOUT pybullet — skips root position correction (rotation errors unaffected).

Output format (same as infer_mobileposer.py):
  {pred: List[Tensor[T,24,3,3]], gt: List[Tensor[T,24,3,3]]}
  Joints 10,11,20-23 (toes, wrists, hands) are set to identity (TIP doesn't predict them).

Dependencies:
  pip install fairmotion einops torch

Usage (run from code/base_mobileposer/):
  conda activate mobileposer

  # Run on TIP's pre-processed AMASS data (uses model-with-dip9and10.pt by default):
  python infer_tip.py \
      --model ../TIP/output/model-with-dip9and10.pt \
      --data-dir ../TIP/data \
      --output-dir results/tip

  # Quick smoke test (10 files):
  python infer_tip.py --model ../TIP/output/model-with-dip9and10.pt --max-files 10

TIP joint → SMPL joint mapping (18 predicted joints):
  TIP predicts: root(0), lhip(1), rhip(2), lowerback(3), lknee(4), rknee(5),
                upperback(6), lankle(7), rankle(8), chest(9),
                lowerneck(12), lclavicle(13), rclavicle(14), upperneck(15),
                lshoulder(16), rshoulder(17), lelbow(18), relbow(19)
  NOT predicted: ltoe(10), rtoe(11), lwrist(20), rwrist(21), lhand(22), rhand(23)
"""
import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

# Add TIP directory to path for model import
TIP_DIR = Path(__file__).resolve().parent.parent / 'TIP'
sys.path.insert(0, str(TIP_DIR))


# ---------------------------------------------------------------------------
# Pure-numpy utility functions (extracted from TIP/data_utils.py to avoid
# the top-level 'import pybullet' that would fail without pybullet installed)
# ---------------------------------------------------------------------------

def _imu_rotate_to_local(batch_imu: np.ndarray) -> np.ndarray:
    """Rotate all IMU readings from global frame to root's local frame. [T, 72]→[T, 72]"""
    batch_imu = batch_imu.copy()
    root_r    = batch_imu[:, :9].reshape(-1, 3, 3)          # [T, 3, 3]
    other_r   = batch_imu[:, 9: 6*9].reshape(-1, 5, 3, 3)  # [T, 5, 3, 3]
    root_r_inv = np.linalg.inv(root_r)

    other_r_local = other_r.copy()
    for i in range(5):
        other_r_local[:, i] = np.matmul(root_r_inv, other_r[:, i])
    batch_imu[:, 9: 6*9] = other_r_local.reshape(-1, 5*9)

    acc_other = batch_imu[:, 6*9+3:].reshape(-1, 5, 3)  # [T, 5, 3]
    for i in range(5):
        acc_other[:, i] = np.einsum('bij,bj->bi', root_r_inv, acc_other[:, i])
    batch_imu[:, 6*9+3:] = acc_other.reshape(-1, 15)

    return batch_imu


def _aa_to_rot_mat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle → rotation matrix. aa: [N, 3] → [N, 3, 3]"""
    # Rodrigues formula
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)  # [N, 1]
    axis  = aa / (angle + 1e-9)                          # [N, 3]
    c, s  = np.cos(angle), np.sin(angle)                 # [N, 1]
    t     = 1.0 - c                                      # [N, 1]

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    R = np.stack([
        t*x*x + c,    t*x*y - s*z,  t*x*z + s*y,
        t*x*y + s*z,  t*y*y + c,    t*y*z - s*x,
        t*x*z - s*y,  t*y*z + s*x,  t*z*z + c,
    ], axis=-1).reshape(-1, 3, 3)

    # Identity for near-zero rotations
    near_zero = (angle[..., 0] < 1e-9)
    R[near_zero] = np.eye(3)
    return R


def _rot_mat_to_aa(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle. R: [..., 3, 3] → [..., 3]"""
    orig_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_a = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)  # [N]
    sin_a = np.sin(angle)
    # skew-symmetric part
    w = np.stack([R[:, 2, 1] - R[:, 1, 2],
                  R[:, 0, 2] - R[:, 2, 0],
                  R[:, 1, 0] - R[:, 0, 1]], axis=-1) / (2.0 * sin_a[:, None] + 1e-9)
    aa = w * angle[:, None]
    # identity for near-zero angles
    near_zero = (angle < 1e-9)
    aa[near_zero] = 0.0
    return aa.reshape(*orig_shape, 3)


def _2axis_to_rot_mat_batch(rm_2axis: np.ndarray) -> np.ndarray:
    """
    Convert TIP's 2-axis representation to 3×3 rotation matrices.
    rm_2axis: [B, N_joints * 6] → returns [B, N_joints, 3, 3]
    """
    B = rm_2axis.shape[0]
    N = rm_2axis.shape[1] // 6
    rm = rearrange(rm_2axis, 'b (n r1 r2) -> (b n) r1 r2', r1=3, r2=2)  # [B*N, 3, 2]
    a1 = rm[:, :, 0]
    a1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-6)
    a2 = rm[:, :, 1]
    a2 = a2 / (np.linalg.norm(a2, axis=1, keepdims=True) + 1e-6)
    a3 = np.cross(a1, a2)
    R_full = np.stack([a1, a2, a3], axis=-1)  # [B*N, 3, 3]
    return R_full.reshape(B, N, 3, 3)


def _2axis_to_aa_batch(rm_2axis: np.ndarray) -> np.ndarray:
    """Convert 2-axis repr to axis-angle. rm_2axis: [B, N*6] → [B, N*3]"""
    R = _2axis_to_rot_mat_batch(rm_2axis)   # [B, N, 3, 3]
    B, N = R.shape[:2]
    aa = _rot_mat_to_aa(R.reshape(B*N, 3, 3))  # [B*N, 3]
    return aa.reshape(B, N*3)


def _aa_to_2axis_batch(batch_s: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle batch to 2-axis representation (for state input).
    batch_s: [B, n_dofs] where n_dofs=57 (18 joint AA + 3 root vel)
    Returns: [B, 18*6 + 3]
    """
    n_dofs = 57
    aa = batch_s[:, :n_dofs - 3].reshape(-1, 3)  # [B*18, 3]
    R  = _aa_to_rot_mat(aa)                       # [B*18, 3, 3]
    r  = R[:, :, :2].reshape(-1, 6).reshape(batch_s.shape[0], -1)  # [B, 18*6]
    return np.concatenate((r, batch_s[:, -3:]), axis=1)  # [B, 18*6+3]


# ---------------------------------------------------------------------------
# TIP joint → SMPL joint index mapping
# ---------------------------------------------------------------------------
# TIP qdq layout (indices into qdq[3:57] = 54 values = 18 joints × 3):
#   TIP slot i → SMPL joint TIP_SLOT_TO_SMPL[i]
TIP_SLOT_TO_SMPL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]

# SMPL joints NOT predicted by TIP (will be identity in output)
TIP_MISSING_SMPL = [10, 11, 20, 21, 22, 23]

N_SMPL_JOINTS = 24
N_TIP_JOINTS  = 18  # joints predicted by TIP
N_DOFS        = 57  # TIP's n_dofs
IMU_SMOOTH_N  = 5   # smoothing half-window
ACC_WIN_LEN   = IMU_SMOOTH_N * 2 + 1


# ---------------------------------------------------------------------------
# TIP model loader
# ---------------------------------------------------------------------------

def load_tip_model(weights_path: str, five_sbp: bool = True):
    """Load TIP TF_RNN_Past_State model from weights file."""
    from simple_transformer_with_state import TF_RNN_Past_State

    input_channels_imu = 6 * (9 + 3)  # 72
    if five_sbp:
        output_channels = N_TIP_JOINTS * 6 + 3 + 20
    else:
        output_channels = N_TIP_JOINTS * 6 + 3 + 8

    model = TF_RNN_Past_State(
        input_channels_imu, output_channels,
        rnn_hid_size=512,
        tf_hid_size=1024, tf_in_dim=256,
        n_heads=16, tf_layers=4,
        dropout=0.0, in_dropout=0.0,
        past_state_dropout=0.8,
        with_acc_sum=False,
    )
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


# ---------------------------------------------------------------------------
# Core inference loop (no pybullet)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_tip_sequence(model, imu: np.ndarray, s_gt: np.ndarray,
                     max_win: int = 40, device: torch.device = None,
                     five_sbp: bool = True) -> np.ndarray:
    """
    Run TIP inference on a single sequence autoregressively.

    Args:
        model:   TF_RNN_Past_State (eval mode, on device)
        imu:     [T, 72]  raw IMU readings (from TIP's pickle data)
        s_gt:    [T, 114] nimble_qdq GT states
        max_win: maximum transformer context window (default 40)
        device:  torch device

    Returns:
        s_pred: [T, 114] predicted states in qdq format
    """
    if device is None:
        device = next(model.parameters()).device

    n_sbps      = 5 if five_sbp else 2
    output_rot  = N_TIP_JOINTS * 6  # 108

    T = imu.shape[0]
    s_pred = np.zeros((T, N_DOFS * 2))
    s_pred[0] = s_gt[0]

    raw_imu_buf   = []
    smooth_imu_buf = []
    s_and_c_buf   = []
    last_s        = None

    # Initialize state buffer from GT frame 0
    c_init = np.zeros(n_sbps * 4)
    s_and_c_init = np.concatenate([
        _aa_to_2axis_batch(s_gt[0:1, 3:N_DOFS + 3])[0],
        c_init,
    ])
    s_and_c_buf.append(s_and_c_init)

    for t in range(T - 1):
        cur_imu = imu[t].copy()

        # --- IMU smoothing (running mean of accelerations) ---
        if len(raw_imu_buf) == 0:
            for _ in range(IMU_SMOOTH_N):
                raw_imu_buf.append(cur_imu.copy())
        raw_imu_buf.append(cur_imu.copy())

        if len(raw_imu_buf) >= ACC_WIN_LEN:
            win = np.array(raw_imu_buf[-ACC_WIN_LEN:])
            smoothed = np.concatenate([
                raw_imu_buf[-IMU_SMOOTH_N - 1][: 6 * 9],          # rotations: unsmoothed
                np.mean(win[:, 6*9 : 6*9 + 18], axis=0),          # accelerations: smoothed
            ])
            smooth_imu_buf.append(smoothed)

        if len(smooth_imu_buf) < 1:
            s_pred[t + 1] = s_gt[0]
            continue

        assert len(s_and_c_buf) == len(smooth_imu_buf)

        # --- Build model inputs ---
        in_imu   = np.array(smooth_imu_buf[-max_win:])      # [L, 72]
        in_imu   = _imu_rotate_to_local(in_imu)             # local frame
        L        = in_imu.shape[0]
        in_s_c   = np.array(s_and_c_buf[-L:])               # [L, 18*6+3+n_sbps*4]

        x_imu  = torch.from_numpy(in_imu).float().unsqueeze(0).to(device)   # [1, L, 72]
        x_s_c  = torch.from_numpy(in_s_c).float().unsqueeze(0).to(device)   # [1, L, ...]

        y = model(x_imu, x_s_c)                                              # [1, L, output_ch]
        out = y.squeeze(0)[-1].cpu().numpy()                                  # [output_ch]

        # --- Extract predicted state ---
        st_2axis_and_v = out[:output_rot + 3]   # [18*6 + 3] = 2-axis rotations + root vel
        c_t            = out[output_rot + 3:]   # [n_sbps * 4] SBP constraints

        # Post-processing smoothing (lightweight EWA)
        if last_s is not None:
            st_2axis_and_v = 0.5 * (st_2axis_and_v + last_s)
        last_s = st_2axis_and_v.copy()

        root_v  = st_2axis_and_v[-3:]                                     # [3] root velocity
        st_2aa  = _2axis_to_aa_batch(st_2axis_and_v[:-3][np.newaxis, :])[0]  # [18*3]

        # Build qdq output
        s_t = np.zeros(N_DOFS * 2)
        s_t[:3]        = s_pred[t, :3] + root_v / 60.0                  # root xyz (rough)
        s_t[3:6]       = _rot_mat_to_aa(
                            np.reshape(in_imu[-1, :9], (3, 3))[np.newaxis])[0]  # root rot from IMU
        s_t[6:N_DOFS]  = st_2aa[3:]                                       # 17 non-root joints
        s_t[N_DOFS: N_DOFS + 3] = root_v

        s_pred[t + 1] = s_t

        # Record new state for next step
        s_and_c_new = np.concatenate([
            _aa_to_2axis_batch(s_t[np.newaxis, 3:N_DOFS + 3])[0],
            c_t,
        ])
        s_and_c_buf.append(s_and_c_new)

    # Trim smoothing delay (IMU_SMOOTH_N + 2 frames at start are invalid)
    trim = IMU_SMOOTH_N + 2
    s_pred[:-trim] = s_pred[trim:]
    s_pred[-trim:] = s_pred[-trim - 1]

    return s_pred


def qdq_to_rotation_matrices(qdq_batch: np.ndarray) -> torch.Tensor:
    """
    Convert TIP's qdq format [T, 114] to SMPL rotation matrices [T, 24, 3, 3].
    Missing joints (10,11,20-23) are set to identity.

    qdq[0:3]  = root XYZ (ignored for rotation)
    qdq[3:6]  = root rotation axis-angle  → SMPL joint 0
    qdq[6:57] = 17 non-root joint AA      → SMPL joints per TIP_SLOT_TO_SMPL[1:]
    """
    T = qdq_batch.shape[0]
    all_aa = qdq_batch[:, 3:N_DOFS]  # [T, 54] = 18 joints × 3
    all_aa_flat = all_aa.reshape(T * N_TIP_JOINTS, 3)
    all_R = _aa_to_rot_mat(all_aa_flat).reshape(T, N_TIP_JOINTS, 3, 3)  # [T, 18, 3, 3]

    R_smpl = np.tile(np.eye(3), (T, N_SMPL_JOINTS, 1, 1))  # [T, 24, 3, 3] = identity
    for tip_slot, smpl_idx in enumerate(TIP_SLOT_TO_SMPL):
        R_smpl[:, smpl_idx] = all_R[:, tip_slot]

    return torch.from_numpy(R_smpl.astype(np.float32))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TIP_DATA_SUBDIRS = [
    'syn_AMASS_CMU_v0', 'syn_Eyes_Japan_Dataset_v0',
    'syn_KIT_v0', 'syn_HUMAN4D_v0', 'syn_ACCAD_v0',
    'syn_DFaust_67_v0', 'syn_HumanEva_v0', 'syn_MPI_Limits_v0',
    'syn_MPI_mosh_v0', 'syn_SFU_v0', 'syn_Transitions_mocap_v0',
    'syn_DanceDB_v0',
]


def iter_tip_files(data_dir: Path, subdirs=None, max_files=None):
    """Yield .pkl file paths from TIP's data directory."""
    subdirs = subdirs or TIP_DATA_SUBDIRS
    count = 0
    for sub in subdirs:
        sub_path = data_dir / sub
        if not sub_path.is_dir():
            continue
        for pkl_file in sorted(sub_path.glob('*.pkl')):
            yield pkl_file
            count += 1
            if max_files is not None and count >= max_files:
                return


def load_tip_file(pkl_path: Path):
    """Load one TIP pickle file. Returns (imu [T,72], s_gt [T,114]) or None if too short."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    imu  = data['imu']          # [T, 72]
    s_gt = data['nimble_qdq']   # [T, 114]
    MIN_LEN = int(2.5 / (1.0 / 60))  # 2.5 seconds @ 60 Hz = 150 frames
    if s_gt.shape[0] < MIN_LEN:
        return None, None
    return imu, s_gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='TIP inference (no pybullet required)')
    parser.add_argument('--model', type=str,
                        default='../TIP/output/model-with-dip9and10.pt',
                        help='Path to TIP weights .pt file')
    parser.add_argument('--data-dir', type=str,
                        default='../TIP/data',
                        help='TIP data root directory (contains syn_AMASS_CMU_v0/ etc.)')
    parser.add_argument('--output-dir', type=str, default='results/tip',
                        help='Directory for output .pt files')
    parser.add_argument('--subdirs', nargs='+', default=None,
                        help='Which TIP data subdirs to use (default: all AMASS subdirs)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Max pickle files to load (for speed tests)')
    parser.add_argument('--two-sbp', action='store_true',
                        help='Use 2-SBP model (default: 5-SBP)')
    args = parser.parse_args()

    five_sbp = not args.two_sbp
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Loading TIP model from {args.model} ({device})')
    try:
        model = load_tip_model(args.model, five_sbp=five_sbp)
    except RuntimeError as e:
        if five_sbp:
            print(f'5-SBP load failed ({e}); retrying with 2-SBP...')
            model = load_tip_model(args.model, five_sbp=False)
            five_sbp = False
        else:
            raise
    model = model.to(device).eval()
    print(f'  SBP mode: {"5" if five_sbp else "2"}')

    data_dir = Path(args.data_dir)
    pkl_files = list(iter_tip_files(data_dir, args.subdirs, args.max_files))
    print(f'Found {len(pkl_files)} pkl files')

    preds, gts = [], []
    for pkl_path in pkl_files:
        imu, s_gt = load_tip_file(pkl_path)
        if imu is None:
            continue

        print(f'  {pkl_path.name}  [T={imu.shape[0]}]', end='', flush=True)
        try:
            s_pred = run_tip_sequence(model, imu, s_gt, device=device, five_sbp=five_sbp)
        except Exception as e:
            print(f'  ERROR: {e}')
            continue

        pred_rot = qdq_to_rotation_matrices(s_pred)  # [T, 24, 3, 3]
        gt_rot   = qdq_to_rotation_matrices(s_gt)    # [T, 24, 3, 3]

        preds.append(pred_rot)
        gts.append(gt_rot)
        print(f'  OK')

    if len(preds) == 0:
        print('No sequences processed. Check data path and model configuration.')
        return

    out_path = out_dir / 'tip.pt'
    torch.save({'pred': preds, 'gt': gts,
                'n_sequences': len(preds), 'method': 'TIP'}, out_path)
    print(f'\nSaved {len(preds)} sequences → {out_path}')
    print('Evaluate with:')
    print(f'  python -m mobileposer.evaluate_per_bodypart '
          f'--results results/tip/tip.pt:TIP')


if __name__ == '__main__':
    main()
