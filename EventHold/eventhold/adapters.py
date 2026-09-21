"""Method-specific conversions. No universal model tensor, no slot guessing."""
import numpy as np
from scipy.spatial.transform import Rotation

from .records import FIVE, PNP_SIX, DYNA_SIX


def eventhold_features(record):
    r = record.select(FIVE)
    root = r.orientation[:, 0]
    ori = np.empty_like(r.orientation)
    ori[:, 0] = root
    ori[:, 1:] = np.swapaxes(root[:, None], -1, -2) @ r.orientation[:, 1:]
    acc = r.acceleration.copy()
    acc[:, 1:] -= acc[:, :1]
    acc = np.einsum("tni,tij->tnj", acc, root)
    # Missing values never become genuine measurements in the feature stream.
    block = np.concatenate([ori.reshape(len(root), 5, 9), acc / 30.0,
                            r.valid[..., None].astype(np.float32)], axis=-1)
    dt = np.r_[0.0, np.diff(r.timestamps)].astype(np.float32)
    return np.concatenate([block.reshape(len(root), -1), dt[:, None]], axis=-1)


def dynaip_features(record):
    """Native six-sensor normalize_imu semantics; root orientation remains global."""
    r = record.select(DYNA_SIX)
    r.require_rate(60)
    if not r.valid.all():
        raise ValueError("DynaIP native profile has no missing-measurement mask")
    root = r.orientation[:, 0]
    ori = r.orientation.copy()
    ori[:, 1:] = np.swapaxes(root[:, None], -1, -2) @ ori[:, 1:]
    acc = r.acceleration.copy()
    acc[:, 1:] -= acc[:, :1]
    acc = np.einsum("tni,tij->tnj", acc, root)
    return np.concatenate([ori.reshape(len(root), 6, 9), acc], axis=-1)


def pnp_from_native(aS, wS, RIS, RIM, RSB, gravity):
    """Native PNP input: world linear a, world angular w, bone-to-world R.

    RIS sensor->inertial reference; RIM model/world->inertial reference;
    RSB bone->sensor. The world gravity vector must match that native record.
    This is not a causal certification of the upstream aS/wS generation.
    """
    aS, wS, RIS, RIM, RSB, gravity = [np.asarray(x) for x in (aS, wS, RIS, RIM, RSB, gravity)]
    if aS.ndim != 3 or aS.shape[1:] != (6, 3) or wS.shape != aS.shape:
        raise ValueError("PNP native requires six sensor-local vector streams")
    if RIS.shape != aS.shape[:2] + (3, 3) or RIM.shape != (6, 3, 3) or RSB.shape != (6, 3, 3):
        raise ValueError("Calibration matrix shapes do not match native PNP contract")
    if gravity.shape != (3,) or not all(np.isfinite(x).all() for x in (aS, wS, RIS, RIM, RSB, gravity)):
        raise ValueError("PNP conversion requires explicit finite observations and world gravity")
    sensor_to_world = np.swapaxes(RIM, -1, -2)[None] @ RIS
    a = np.einsum("tnij,tnj->tni", sensor_to_world, aS) + gravity
    w = np.einsum("tnij,tnj->tni", sensor_to_world, wS)
    return a, w, sensor_to_world @ RSB[None]


def angular_velocity_world_causal(orientation, timestamps):
    """Backward SO(3) increment in world coordinates, interval-average estimate.

    First frame has zero placeholder AND invalid flag. Not a measured gyro.
    """
    if len(orientation) != len(timestamps) or len(timestamps) < 2:
        raise ValueError("At least two synchronized frames required")
    dt = np.diff(timestamps)
    if np.any(dt <= 0):
        raise ValueError("Nonpositive dt")
    dr = orientation[1:] @ np.swapaxes(orientation[:-1], -1, -2)
    w = np.zeros(orientation.shape[:-2] + (3,), dtype=orientation.dtype)
    w[1:] = Rotation.from_matrix(dr.reshape(-1, 3, 3)).as_rotvec().reshape(w[1:].shape) / dt[:, None, None]
    valid = np.ones(w.shape[:2], dtype=bool)
    valid[0] = False
    return w, valid


def pnp_from_world_record(record):
    """Causal derived-gyro profile, separate from published native gyro pipeline."""
    r = record.select(PNP_SIX)
    r.require_rate(60)
    if not r.valid.all():
        raise ValueError("PNP native profile cannot silently consume filled measurements")
    w, gyro_valid = angular_velocity_world_causal(r.orientation, r.timestamps)
    return {"a": r.acceleration.copy(), "w": w, "R": r.orientation.copy(),
            "gyro_valid": gyro_valid, "gyro_source": "backward_world_SO3_difference",
            "sampling_hz": 60, "causal": True, "native_gyro_equivalent": False}
