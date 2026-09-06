"""
Shared constants and utilities for all evaluate_drift.py scripts.

Import from each method's evaluate_drift.py:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # code/
    from drift_eval_common import (
        PRIMARY_SEGMENTS, SECONDARY_SEGMENTS, SEGMENTS, SEG_EN,
        LUMBAR_JOINTS, FPS, SENSOR_TO_JOINT, DATA_PATH,
        load_long_sequences, angle_between_rotmats, moving_average,
    )

All five methods (MobilePoser, DIP-IMU, PNP, DynaIP, IMUCoCo) import from here
to guarantee that:
  - The same AMASS sequences are loaded in the same order
  - The same joint groups are analysed
  - Identical math helpers are used
"""

from pathlib import Path
import numpy as np
import torch

# ─── paths ─────────────────────────────────────────────────────────────────────
# All scripts run on a server where code/ maps to MotionRe/.
# DATA_PATH is relative to this file (drift_eval_common.py lives in code/).
DATA_PATH = Path(__file__).resolve().parent / 'base_mobileposer' / 'data' / 'processed_datasets'

# ─── timing ────────────────────────────────────────────────────────────────────
FPS = 30

# ─── sensor → SMPL joint mapping ───────────────────────────────────────────────
# Sensor order in AMASS .pt files: [L_wrist, R_wrist, L_hip, R_hip, Head, Pelvis]
# Maps to SMPL joints:             [18,      19,      1,     2,     15,   0     ]
SENSOR_TO_JOINT = [18, 19, 1, 2, 15, 0]

# ─── body-segment definitions  (IDENTICAL across all five methods) ──────────────
# Values are SMPL joint index lists — these are the joints that define each segment.
# Keys are English names used in plot titles and table headers.
PRIMARY_SEGMENTS = {
    'Lumbar':   [3],        # Spine1
    'Thoracic': [6, 9],     # Spine2, Spine3
    'Hip':      [1, 2],     # L/R Hip
}
SECONDARY_SEGMENTS = {
    'Knee':     [4, 5],     # L/R Knee
    'UpperArm': [16, 17],   # L/R Shoulder / upper arm
    'Forearm':  [18, 19],   # L/R Forearm (sensor attachment points)
}
SEGMENTS = {**PRIMARY_SEGMENTS, **SECONDARY_SEGMENTS}

# Human-readable display strings for each segment key
SEG_EN = {
    'Lumbar':   'Lumbar(j3)',
    'Thoracic': 'Thoracic(j6,9)',
    'Hip':      'Hip(j1,2)',
    'Knee':     'Knee(j4,5)',
    'UpperArm': 'UpperArm(j16,17)',
    'Forearm':  'Forearm(j18,19)',
}

# Lumbar composite score — average angular error over these 5 joints
LUMBAR_JOINTS = [1, 2, 3, 6, 9]

# ─── canonical sequence loader ─────────────────────────────────────────────────

def load_long_sequences(min_frames: int, max_seqs: int,
                        amass_dir: Path = None) -> list:
    """
    Load unwindowed AMASS sequences that are at least min_frames long.

    Scans ``amass_dir`` (defaults to DATA_PATH) in sorted order so that all
    methods evaluate on *exactly the same* sequences given the same parameters.

    Parameters
    ----------
    min_frames : int
        Minimum sequence length in frames.
    max_seqs : int
        Maximum number of sequences to return.  0 = no limit.
    amass_dir : Path, optional
        Override the default DATA_PATH.  Used when --amass_dir CLI arg is set.

    Returns
    -------
    list of dicts, each with keys: acc [T,6,3], ori [T,6,3,3],
        pose [T,24,3,3] (LOCAL rotmats), tran [T,3], source (str).
    """
    dir_ = amass_dir if amass_dir is not None else DATA_PATH
    pt_files = sorted(dir_.glob('*.pt'))
    if not pt_files:
        raise FileNotFoundError(
            f'No .pt files found in {dir_}.\n'
            'Expected AMASS subsets processed by MobilePoser (process.py).'
        )

    unlimited = (max_seqs <= 0)
    seqs: list = []
    print(f'Scanning {len(pt_files)} AMASS files '
          f'(min {min_frames} frames = {min_frames / FPS:.0f}s) ...')

    for fpath in pt_files:
        try:
            try:
                data = torch.load(fpath, map_location='cpu', weights_only=False)
            except TypeError:
                data = torch.load(fpath, map_location='cpu')
        except Exception as e:
            print(f'  Skip {fpath.name}: {e}')
            continue

        for i, (acc, ori, pose, tran) in enumerate(
                zip(data['acc'], data['ori'], data['pose'], data['tran'])):
            if pose.shape[0] >= min_frames:
                seqs.append({
                    'acc':    acc.float(),    # [T, 6, 3]
                    'ori':    ori.float(),    # [T, 6, 3, 3]
                    'pose':   pose.float(),   # [T, 24, 3, 3]  LOCAL rotation matrices
                    'tran':   tran.float(),   # [T, 3]
                    'source': f'{fpath.stem}[{i}]',
                })
            if not unlimited and len(seqs) >= max_seqs:
                break
        if not unlimited and len(seqs) >= max_seqs:
            break

    print(f'  → {len(seqs)} qualifying sequences found.')
    return seqs


# ─── shared math helpers ───────────────────────────────────────────────────────

def angle_between_rotmats(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Angular error in degrees between two sets of rotation matrices.

    R1, R2 : [..., 3, 3]
    Returns : [...] in degrees
    """
    R     = R1.transpose(-1, -2) @ R2
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0, 1.0)))


def moving_average(x: np.ndarray, window: int = 15) -> np.ndarray:
    """Symmetric moving average (same as used in MobilePoser evaluate_drift)."""
    return np.convolve(x, np.ones(window) / window, mode='same')
