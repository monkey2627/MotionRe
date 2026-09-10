"""
Shared constants and utilities for all evaluate_drift.py scripts.

Import from each method's evaluate_drift.py:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # code/
    from drift_eval_common import (
        PRIMARY_SEGMENTS, SECONDARY_SEGMENTS, SEGMENTS, SEG_EN,
        LUMBAR_JOINTS, FPS, SENSOR_TO_JOINT, DATA_PATH,
        load_long_sequences, angle_between_rotmats, moving_average,
        add_imu_noise,
    )

All five methods (MobilePoser, DIP-IMU, PNP, DynaIP, IMUCoCo) import from here
to guarantee that:
  - The same AMASS sequences are loaded in the same order
  - The same joint groups are analysed
  - Identical math helpers are used
"""

from pathlib import Path
import csv
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

def _action_selection(manifest_path: Path, raw_amass: Path,
                      max_per_action: int) -> dict:
    selected = {}
    with manifest_path.open('r', newline='', encoding='utf-8-sig') as handle:
        rows = csv.DictReader(handle)
        grouped = {}
        for row in rows:
            category = (row.get('category') or 'other').strip()
            source = (row.get('raw_motion') or '').strip()
            if source:
                try:
                    relative = Path(source).resolve().relative_to(raw_amass.resolve())
                except ValueError:
                    continue
                grouped.setdefault(category, []).append((relative.parts[0], relative.as_posix()))
        for category, entries in grouped.items():
            for dataset, source in sorted(entries)[:max_per_action]:
                selected.setdefault(dataset, set()).add((category, source))
    return selected


def load_long_sequences(min_frames: int, max_seqs: int,
                        amass_dir: Path = None,
                        action_manifest: Path = None,
                        raw_amass: Path = None,
                        max_per_action: int = 100) -> list:
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
    action_map = None
    raw_root = raw_amass
    if action_manifest:
        raw_root = raw_root or dir_.parent.parent / 'data' / 'raw' / 'AMASS'
        action_map = _action_selection(Path(action_manifest), Path(raw_root), max_per_action)
    seqs: list = []
    print(f'Scanning {len(pt_files)} AMASS files '
          f'(min {min_frames} frames = {min_frames / FPS:.0f}s) ...')

    if action_map is not None:
        min_frames = 1
    for fpath in pt_files:
        try:
            try:
                data = torch.load(fpath, map_location='cpu', weights_only=False)
            except TypeError:
                data = torch.load(fpath, map_location='cpu')
        except Exception as e:
            print(f'  Skip {fpath.name}: {e}')
            continue

        raw_candidates = []
        allowed = set()
        if action_map is not None:
            raw_candidates = sorted((Path(raw_root) / fpath.stem).rglob('*_poses.npz'))
            allowed = action_map.get(fpath.stem, set())
        for i, (acc, ori, pose, tran) in enumerate(
                zip(data['acc'], data['ori'], data['pose'], data['tran'])):
            action = 'all'
            if action_map is not None:
                if i >= len(raw_candidates):
                    continue
                raw_rel = raw_candidates[i].relative_to(Path(raw_root)).as_posix()
                actions = [category for category, source in allowed if source == raw_rel]
                if not actions:
                    continue
                action = actions[0]
            if pose.shape[0] >= min_frames:
                seqs.append({
                    'acc':    acc.float(),    # [T, 6, 3]
                    'ori':    ori.float(),    # [T, 6, 3, 3]
                    'pose':   pose.float(),   # [T, 24, 3, 3]  LOCAL rotation matrices
                    'tran':   tran.float(),   # [T, 3]
                    'source': f'{fpath.stem}[{i}]',
                    'action': action,
                })
            if not unlimited and len(seqs) >= max_seqs:
                break
        if not unlimited and len(seqs) >= max_seqs:
            break

    print(f'  → {len(seqs)} qualifying sequences found.')
    return seqs


# ─── shared math helpers ───────────────────────────────────────────────────────

def _exp_so3_batch(omega: torch.Tensor) -> torch.Tensor:
    """
    Batch Rodrigues formula: [..., 3] → [..., 3, 3].
    Maps axis-angle vectors to rotation matrices.
    Near-zero vectors map to identity via the clamp trick (numerically stable).
    """
    shape = omega.shape[:-1]
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-9)   # [..., 1]
    k     = omega / theta                                        # [..., 3]

    K = torch.zeros(*shape, 3, 3, dtype=omega.dtype)
    K[..., 0, 1] = -k[..., 2];  K[..., 0, 2] =  k[..., 1]
    K[..., 1, 0] =  k[..., 2];  K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1];  K[..., 2, 1] =  k[..., 0]

    sin_t = theta[..., None].sin()   # [..., 1, 1]
    cos_t = theta[..., None].cos()
    I     = torch.eye(3, dtype=omega.dtype).expand(*shape, 3, 3)
    return I + sin_t * K + (1.0 - cos_t) * (K @ K)


def add_imu_noise(ori: torch.Tensor, acc: torch.Tensor,
                  fps: int = FPS,
                  drift_deg_per_sqrt_s: float = 0.5,
                  noise_deg: float = 0.5,
                  acc_noise_ms2: float = 0.1) -> tuple:
    """
    Add realistic MEMS-IMU noise to synthesised orientation and acceleration.

    Parameters
    ----------
    ori : [T, N, 3, 3]  global rotation matrices (all N sensor slots, incl. pelvis)
    acc : [T, N, 3]     linear accelerations (m/s²)
    fps : int           sample rate of the data
    drift_deg_per_sqrt_s : float
        Gyroscope random-walk coefficient (°/√s).
        Governs how quickly orientation error accumulates over time.
        Typical values:
          0.2  → high-quality IMU (Xsens MTi)       ~2.2° std after 120 s
          0.5  → decent consumer MEMS               ~5.5° std after 120 s  [default]
          1.5  → cheap phone/watch MEMS             ~16°  std after 120 s
    noise_deg : float
        Per-frame i.i.d. orientation noise (°). Models vibration and read-out noise.
    acc_noise_ms2 : float
        Per-frame Gaussian acceleration noise (m/s²). Typical MEMS floor: 0.05–0.2.

    Noise model
    -----------
    For each sensor independently:
      R_out[t] = D[t] @ R_true[t] @ ΔR_noise[t]
    where
      D[t]         = D[t-1] @ exp_so3(ε_drift[t]),  ε_drift ~ N(0, σ_drift² I)
      ΔR_noise[t]  = exp_so3(ε_noise[t]),            ε_noise ~ N(0, σ_noise² I)
      σ_drift      = drift_deg_per_sqrt_s [rad/√s] / √fps  (per-frame random walk step)
      σ_noise      = noise_deg [rad]

    D[t] is a slowly-drifting rotation that mimics gyro integration error and grows
    as √t.  ΔR_noise[t] is a fast-varying per-frame perturbation.
    """
    T, N = ori.shape[0], ori.shape[1]

    sigma_drift = float(np.deg2rad(drift_deg_per_sqrt_s)) / float(np.sqrt(fps))
    sigma_noise = float(np.deg2rad(noise_deg))

    # Pre-generate all random increments at once for efficiency
    drift_omegas = torch.randn(T, N, 3) * sigma_drift   # [T, N, 3]
    noise_omegas = torch.randn(T, N, 3) * sigma_noise   # [T, N, 3]

    # Convert to rotation matrices in batch: [T, N, 3, 3]
    drift_Rs = _exp_so3_batch(drift_omegas)
    noise_Rs = _exp_so3_batch(noise_omegas)

    ori_out = ori.clone()
    # Cumulative drift per sensor: [N, 3, 3], starts at identity
    D = torch.eye(3).unsqueeze(0).repeat(N, 1, 1)

    for t in range(T):
        D = torch.bmm(D, drift_Rs[t])                           # update drift [N,3,3]
        ori_out[t] = torch.bmm(torch.bmm(D, ori[t]), noise_Rs[t])  # apply

    acc_out = acc + torch.randn_like(acc) * acc_noise_ms2
    return ori_out, acc_out


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
