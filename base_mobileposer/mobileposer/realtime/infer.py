"""Real-time mobileposer inference: turn calibrated per-tracker samples into
the 45-frame sliding window mobile_export.MobilePose expects, and run it.

Reuses mobile_export.MobilePose as-is -- it loads the exact same
"no_head_5imu_surface" checkpoints as the android export path, and is already
validated there to be bit-exact with mobileposer.models.MobilePoserNet within
1e-4 -- instead of re-implementing the joints/pose regression and
kinematic-tree (global -> local) reconstruction.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch

from mobileposer.config import paths
from mobileposer.mobile_export import MobilePose, OUTPUT_INDEX, WINDOW
from mobileposer.realtime.calibration import HeadingCalibration, bone_orientation
from mobileposer.realtime.layout import ResolvedLayout

# Real per-layout checkpoints live only on the training server (see CLAUDE.md's
# local<->server path mapping); this must be synced to the local machine first.
DEFAULT_CHECKPOINT_ROOT = paths.checkpoint / "no_head_5imu_surface"

# The model's pose output at window index OUTPUT_INDEX (40 of 45) reconstructs
# the pose ~ (WINDOW - 1 - OUTPUT_INDEX) = 4 frames behind the most recent
# input frame -- about 133ms of inherent lag at 30 Hz. This is a property of
# the trained model (model_config.future_frames=5 lookahead context), not a
# bug to fix here.
OUTPUT_LAG_FRAMES = WINDOW - 1 - OUTPUT_INDEX


def checkpoint_path(layout_name: str, checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT) -> Path:
    return checkpoint_root / layout_name / "1" / "base_model.pth"


def load_model(layout_name: str, checkpoint_root: Path = DEFAULT_CHECKPOINT_ROOT) -> MobilePose:
    path = checkpoint_path(layout_name, checkpoint_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. This layout's checkpoint only exists on the "
            f"training server -- sync it to {checkpoint_root} first."
        )
    return MobilePose().load_combined(path)


def build_feature(calibrated_rotation: torch.Tensor, aligned_accel: torch.Tensor) -> np.ndarray:
    """calibrated_rotation: (n_labels, 3, 3); aligned_accel: (n_labels, 3) -> (60,) float32.

    Matches mobile_export.py::make_fixture's exact layout: all five
    accelerations concatenated first (scaled by 1/30, mobileposer.config
    amass.acc_scale), THEN all five orientations -- NOT interleaved per
    sensor slot, even though the model is often described as "5 x 12D".
    """
    acc = (aligned_accel.reshape(-1) / 30.0).numpy()
    ori = calibrated_rotation.reshape(-1).numpy()
    return np.concatenate([acc, ori]).astype(np.float32)


class SlidingWindow:
    """Push-based re-implementation of mobile_export.py::windows()'s per-step
    logic (that helper is an iterator over a whole pre-collected sequence,
    which doesn't fit a live push-one-frame-at-a-time loop)."""

    def __init__(self, size: int = WINDOW) -> None:
        self.size = size
        self._window: np.ndarray | None = None

    def push(self, frame: np.ndarray) -> np.ndarray:
        """frame: (60,) float32 -> (1, size, 60) float32, ready for the model."""
        if self._window is None:
            self._window = np.repeat(frame[None], self.size, axis=0)
        else:
            self._window = np.concatenate([self._window[1:], frame[None]])
        return self._window[None].astype(np.float32)


@dataclasses.dataclass
class InferenceSession:
    """One loaded model + its sliding window, for one resolved layout.

    Orientation conversion is stateless per-frame (see calibration.py). The
    acceleration correction needs a one-time HeadingCalibration, fitted from a
    short startup window with rotational motion (see run.py) before this can
    be constructed -- unlike orientation, there is no calibration-free
    shortcut for acceleration (see calibration.py's module docstring for why).
    """

    layout: ResolvedLayout
    model: MobilePose
    heading: HeadingCalibration
    window: SlidingWindow = dataclasses.field(default_factory=SlidingWindow)

    def step(self, quats_xyzw: np.ndarray, accel_xyz: np.ndarray) -> torch.Tensor:
        """quats_xyzw: (n_labels, 4) rotation_reference_adjusted quats;
        accel_xyz: (n_labels, 3) linear_acceleration (m/s^2, gravity removed)
        straight from SolarXR -- both in self.layout.labels order. Returns
        (24, 3, 3) local rotation matrices for the current window; see
        OUTPUT_LAG_FRAMES for the inherent delay.
        """
        ori = bone_orientation(quats_xyzw, self.layout.labels)
        accel = self.heading.apply(accel_xyz)
        feature = build_feature(ori, accel)
        window = self.window.push(feature)
        with torch.no_grad():
            return self.model(torch.from_numpy(window))
