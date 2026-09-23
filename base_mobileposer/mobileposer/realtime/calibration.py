"""Per-tracker calibration and per-frame conversion to mobileposer's
no_head_5imu_surface input convention.

Ports the *validated* conventions from `mobileposer/test_ours_surface.py` (an
offline script that ran real physical-IMU captures under `data/raw/ours`
through these exact checkpoints and quantitatively compared the result
against the same session's Unity avatar playback -- see its
`report.json`/README.md for actual cm-level agreement numbers), adapted from
whole-recording batch processing to a live pipeline with a short startup
calibration window instead of a whole-session fit.

Two conversions, independently confirmed:

1. Orientation -- `rotation_reference_adjusted` already carries SlimeVR's own
   mounting/reset calibration (the "位姿校准" step done in the SlimeVR-Server
   app). Only a KNOWN, fixed +-90 degree rotation about Z is needed for
   forearm/upper-arm sensors (SlimeVR's internal arm-bone reference points
   down/N-pose; SMPL's points sideways/T-pose) -- ported verbatim from
   test_ours_surface.py's `bone_orientation()`. No per-session fitting
   involved; bit-exact-verified (~3e-8 max abs error) against
   test_ours_surface.py on real data from data/raw/ours.

2. Acceleration -- REQUIRES a short startup calibration window with some
   rotational motion. This is not a simplification that could be avoided:
   SlimeVR-Server computes `linear_acceleration` by rotating the tracker's
   local reading with the RAW rotation (Tracker.kt:456-460,
   TrackerResetsHandler.kt:196), while `rotation_reference_adjusted` is
   related to that same raw rotation by
   `adjusted(t) = LeftConst @ raw(t) @ RightConst`
   (verified against TrackerResetsHandler.kt:180-254: LeftConst bundles
   gyroFix/yawFix/mountRotFix^-1/constraintFix [+ a slowly-time-varying drift
   term if drift compensation is enabled]; RightConst bundles
   mountingOrientation/attachmentFix/mountRotFix/tposeDownFix). Only
   `mountingOrientation` of RightConst's four factors is exposed over
   SolarXR -- `attachmentFix`, `mountRotFix`, `tposeDownFix` are private
   TrackerResetsHandler fields never serialized in the protocol, so
   RightConst cannot be reconstructed from any amount of live data. The one
   thing that IS extractable is `LeftConst` (~="Yaw"), via the same
   delta-trajectory trick test_ours_surface.py uses: comparing
   `raw(t) @ raw(0)^-1` against `adjusted(t) @ adjusted(0)^-1` makes the
   entire (unobservable) RightConst cancel algebraically. This needs
   observable rotational motion during the window (a perfectly static pose
   is mathematically unobservable -- fit_heading raises if so), hence a
   short "please rotate" startup phase instead of the whole-session batch
   this was ported from.

   Once fitted, applying the correction each frame is a single fixed
   matrix-vector product (see `apply_heading`) -- no ongoing cost, and no
   further use of the raw rotation is needed after calibration.

Coordinate system: SolarXR/SlimeVR-Server space -> SMPL space is
`SERVER_TO_SMPL = diag(-1, 1, -1)` (reflect Z going Server->Unity/VMC, then
reflect X going Unity->SMPL; both are Y-up, so AMASS's Z-up->Y-up rotation
must NOT be applied here -- confirmed independently by loading a raw AMASS
.npz directly: AMASS is Z-up, this live SolarXR/Unity pipeline is Y-up
throughout, they are different conventions).
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

SERVER_TO_SMPL = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])

# +-90 degree rotation about Z, applied to forearm/upper-arm sensors only,
# before the SERVER_TO_SMPL conjugation. Ported from
# mobileposer/test_ours_surface.py::bone_orientation.
_ARM_LABELS_POSITIVE = {"lw", "lu"}
_ARM_LABELS_NEGATIVE = {"rw", "ru"}

_ROT_Z_POS90 = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_ROT_Z_NEG90 = _ROT_Z_POS90.transpose(-1, -2)


def _quat_xyzw_to_matrix(quats: np.ndarray) -> torch.Tensor:
    """quats: (..., 4) array of (x, y, z, w) -> (..., 3, 3) rotation matrices."""
    q = torch.as_tensor(quats, dtype=torch.float32)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = (x * x + y * y + z * z + w * w).clamp_min(1e-12).sqrt()
    x, y, z, w = x / n, y / n, z / n, w / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rows = [
        torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
        torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
        torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
    ]
    return torch.stack(rows, dim=-2)


def _bone_offset(label: str) -> torch.Tensor:
    if label in _ARM_LABELS_POSITIVE:
        return _ROT_Z_POS90
    if label in _ARM_LABELS_NEGATIVE:
        return _ROT_Z_NEG90
    return torch.eye(3)


def bone_orientation(reference_adjusted_quat_xyzw: np.ndarray, labels: List[str]) -> torch.Tensor:
    """reference_adjusted_quat_xyzw: (n_labels, 4) -> (n_labels, 3, 3) SMPL-frame
    rotations ready to feed the model, per mobileposer/test_ours_surface.py's
    bone_orientation(). No calibration state involved."""
    adjusted = _quat_xyzw_to_matrix(reference_adjusted_quat_xyzw)
    basis = SERVER_TO_SMPL.to(adjusted)
    offsets = torch.stack([_bone_offset(label).to(adjusted) for label in labels])
    return basis @ (adjusted @ offsets) @ basis.transpose(-1, -2)


class UnobservableHeadingError(RuntimeError):
    """Raised when a calibration window has too little rotational motion (or
    too inconsistent a raw<->adjusted relationship) to fit a reliable yaw."""


def _yaw_matrix(angle: float) -> np.ndarray:
    return Rotation.from_rotvec([0.0, angle, 0.0]).as_matrix()


def fit_heading(raw_quat_xyzw: np.ndarray, reference_adjusted_quat_xyzw: np.ndarray) -> Dict:
    """raw_quat_xyzw, reference_adjusted_quat_xyzw: (n_frames, 4) for ONE
    tracker, collected over a startup window with some rotational motion.

    Returns {"heading": (3,3) ndarray, "diagnostic": {...}}. Ported from
    mobileposer/test_ours_surface.py::fit_heading, operating on a short
    live window instead of a whole recording.

    Solves adjusted(t) = Yaw @ raw(t) @ mounting for the LEFT-multiplied
    Yaw only, via delta trajectories relative to frame 0 (this makes the
    unobservable RightConst/mounting term cancel exactly, see module
    docstring) -- NOT a body-pose-derived calibration, and it needs
    rotational motion during the window to be solvable at all.
    """
    raw = Rotation.from_quat(raw_quat_xyzw).as_matrix()
    adjusted = Rotation.from_quat(reference_adjusted_quat_xyzw).as_matrix()

    a = raw @ raw[0].T
    b = adjusted @ adjusted[0].T

    def objective(angle: float) -> float:
        h = _yaw_matrix(angle)
        return float(np.mean((h @ a @ h.T - b) ** 2))

    grid = np.linspace(-np.pi, np.pi, 181)
    values = np.array([objective(x) for x in grid])
    best = grid[values.argmin()]
    fit = minimize_scalar(objective, bounds=(best - np.pi / 90, best + np.pi / 90), method="bounded")
    h = _yaw_matrix(fit.x)
    residual = Rotation.from_matrix((h @ a @ h.T).transpose(0, 2, 1) @ b).magnitude()
    diagnostic = {
        "yaw_deg": float(np.rad2deg(fit.x)),
        "delta_residual_mean_deg": float(np.rad2deg(residual).mean()),
        "delta_residual_p95_deg": float(np.quantile(np.rad2deg(residual), 0.95)),
        "objective_contrast": float(np.ptp(values)),
        "frames": len(raw_quat_xyzw),
    }
    if diagnostic["objective_contrast"] < 1e-5:
        raise UnobservableHeadingError(
            f"No detectable rotation during calibration window ({diagnostic['frames']} frames) -- "
            "hold the tracker still is not enough, it must actually rotate for this fit to work."
        )
    if diagnostic["delta_residual_p95_deg"] > 10:
        raise UnobservableHeadingError(f"Inconsistent raw<->reference-adjusted relationship: {diagnostic}")
    return {"heading": h, "diagnostic": diagnostic}


def apply_heading(heading: torch.Tensor, accel_xyz: np.ndarray) -> torch.Tensor:
    """heading: (n_labels, 3, 3) fitted per-tracker Yaw matrices;
    accel_xyz: (n_labels, 3) linear_acceleration in m/s^2, gravity removed,
    straight from SolarXR (already the raw-rotation-frame world acceleration
    SlimeVR-Server computed internally) -> (n_labels, 3) SMPL-frame world
    acceleration (still needs /30 scaling before feeding the model, see
    infer.py::build_feature -- matching amass.acc_scale). Pure per-frame
    application, no fitting -- O(1) per call.
    """
    accel = torch.as_tensor(accel_xyz, dtype=torch.float32)
    basis = SERVER_TO_SMPL.to(accel)
    world = torch.einsum("nij,nj->ni", heading.to(accel), accel)
    return torch.einsum("ij,nj->ni", basis, world)


@dataclasses.dataclass(frozen=True)
class HeadingCalibration:
    labels: List[str]
    heading: torch.Tensor  # (n_labels, 3, 3)
    diagnostics: Dict[str, Dict]

    def apply(self, accel_xyz: np.ndarray) -> torch.Tensor:
        return apply_heading(self.heading, accel_xyz)


def calibrate_heading(
    labels: List[str],
    raw_quat_xyzw_by_label: Dict[str, np.ndarray],
    reference_adjusted_quat_xyzw_by_label: Dict[str, np.ndarray],
) -> HeadingCalibration:
    """labels: layout label order. Both dicts map label -> (n_frames, 4)
    collected over the SAME startup window. Raises UnobservableHeadingError
    (naming which label failed) if any tracker's window wasn't usable."""
    headings, diagnostics = [], {}
    for label in labels:
        try:
            result = fit_heading(raw_quat_xyzw_by_label[label], reference_adjusted_quat_xyzw_by_label[label])
        except UnobservableHeadingError as exc:
            raise UnobservableHeadingError(f"{label}: {exc}") from exc
        headings.append(torch.as_tensor(result["heading"], dtype=torch.float32))
        diagnostics[label] = result["diagnostic"]
    return HeadingCalibration(labels=list(labels), heading=torch.stack(headings), diagnostics=diagnostics)
