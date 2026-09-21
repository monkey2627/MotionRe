from dataclasses import dataclass
from typing import Tuple

import numpy as np


FIVE = ("pelvis", "left_forearm", "right_forearm", "left_shank", "right_shank")
PNP_SIX = ("left_forearm", "right_forearm", "left_shank", "right_shank", "head", "pelvis")
DYNA_SIX = ("pelvis", "left_shank", "right_shank", "head", "left_forearm", "right_forearm")


@dataclass
class ImuRecord:
    """Measurements only. Ground-truth pose must never enter method adapters.

    `valid` describes actually available source measurements, even if finite
    placeholders/causal last values are stored for missing measurements.
    """

    timestamps: np.ndarray
    names: Tuple[str, ...]
    orientation: np.ndarray
    acceleration: np.ndarray
    valid: np.ndarray
    source: str
    orientation_kind: str = "calibrated_bone_to_world"
    acceleration_kind: str = "world_linear_m_s2"

    def validate(self):
        t, n = len(self.timestamps), len(self.names)
        if t < 1 or len(set(self.names)) != n:
            raise ValueError("Empty sequence or duplicate sensor names")
        if self.orientation.shape != (t, n, 3, 3) or self.acceleration.shape != (t, n, 3):
            raise ValueError("Measurement shape and named layout disagree")
        if self.valid.shape != (t, n) or self.valid.dtype != np.bool_:
            raise ValueError("valid must be a Boolean [T,N] source-availability mask")
        if not np.isfinite(self.timestamps).all() or np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Timestamps must be finite and strictly increasing")
        if not np.isfinite(self.orientation).all() or not np.isfinite(self.acceleration).all():
            raise ValueError("Use explicit masked finite placeholders, never silent NaNs")
        if self.orientation_kind != "calibrated_bone_to_world":
            raise ValueError("Bone and uncalibrated sensor orientations are not interchangeable")
        if self.acceleration_kind != "world_linear_m_s2":
            raise ValueError("Expected world linear acceleration in m/s^2, not raw specific force")
        r = self.orientation[self.valid]
        if len(r) and (not np.allclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=2e-3)
                       or not np.allclose(np.linalg.det(r), 1, atol=2e-3)):
            raise ValueError("Valid orientations must lie in SO(3)")
        return self

    def select(self, names):
        self.validate()
        try:
            ids = [self.names.index(name) for name in names]
        except ValueError as error:
            raise ValueError("Required physical sensor missing; no slot substitution allowed") from error
        return ImuRecord(self.timestamps.copy(), tuple(names), self.orientation[:, ids].copy(),
                         self.acceleration[:, ids].copy(), self.valid[:, ids].copy(),
                         self.source, self.orientation_kind, self.acceleration_kind)

    def require_rate(self, hz):
        if len(self.timestamps) < 2:
            raise ValueError("At least two timestamps required to validate sampling rate")
        if not np.allclose(np.diff(self.timestamps), 1 / hz, rtol=1e-4, atol=1e-7):
            raise ValueError("Method requires %g Hz; resampling must be explicit" % hz)


def causal_fill(orientation, acceleration):
    """Forward fill only; never mark filled/missing observations as measured."""
    r, a = np.array(orientation, copy=True), np.array(acceleration, copy=True)
    valid = np.isfinite(r).all(axis=(-1, -2)) & np.isfinite(a).all(axis=-1)
    for t in range(len(r)):
        missing = ~valid[t]
        r[t, missing] = r[t - 1, missing] if t else np.eye(3)
        a[t, missing] = a[t - 1, missing] if t else 0
    return r, a, valid
