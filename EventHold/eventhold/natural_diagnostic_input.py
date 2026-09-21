"""Bounded Natural Motion diagnostic conventions, not full SMPL retargeting.

Xsens XYZ -> diagnostic SMPL YZX. At the canonical T-pose this maps
forward,left,up to left,up,forward; global X need not follow the wearer.
Only pelvis->thigh and thigh->shank local rotations are compared. Different
anatomical rest definitions remain a cross-dataset limitation.
"""
import numpy as np
from scipy.signal import firwin, lfilter
from .mounting import check_rotations

BASIS = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
LOWER_SMPL = [1, 2, 4, 5]
LOWER_NAMES = ['left_hip', 'right_hip', 'left_knee', 'right_knee']
TAPS = firwin(65, 20., fs=240., window=('kaiser', 8.))
DELAY_FRAMES = 32


def lower_reference(global_rotations, names):
    r = check_rotations(global_rotations)
    pairs = [('Pelvis', 'LeftUpperLeg'), ('Pelvis', 'RightUpperLeg'),
             ('LeftUpperLeg', 'LeftLowerLeg'), ('RightUpperLeg', 'RightLowerLeg')]
    result = [r[:, names.index(p)].swapaxes(-1, -2) @ r[:, names.index(c)]
              for p, c in pairs]
    return BASIS @ np.stack(result, axis=1) @ BASIS.T


def causal_decimate(rotations, acceleration, frame_index, time_ms):
    """65-tap trailing FIR then SO(3) projection, decimate by four.

    Output availability is the last source sample, not the delayed signal
    time. Discard all startup outputs without a full window. Rotation matrix
    entries are filtered before projection; the nonlinear projection is NOT
    guaranteed bandlimited. No filtfilt, centered future input or interpolation.
    Requires contiguous nominal 240 Hz frames with millisecond timestamp rounding.
    Caller must supply only post-calibration measurements.
    """
    r = check_rotations(rotations)
    a = np.asarray(acceleration, dtype=float)
    fi, tm = np.asarray(frame_index), np.asarray(time_ms, dtype=float)
    if r.ndim != 4 or a.shape != r.shape[:-2] + (3,) or len(r) < 65:
        raise ValueError('At least 65 synchronized frames required')
    if fi.shape != (len(r),) or tm.shape != fi.shape or not np.isfinite(a).all():
        raise ValueError('Invalid measurements or time shapes')
    if not np.all(np.diff(fi) == 1) or not np.isfinite(tm).all():
        raise ValueError('Contiguous valid native frames required')
    elapsed = (fi-fi[0])/240.
    if np.max(np.abs((tm-tm[0])/1000.-elapsed)) > .0011:
        raise ValueError('Timestamps disagree with nominal 240 Hz')
    ids = np.arange(64, len(r), 4)
    filtered = lfilter(TAPS, [1.], r, axis=0)[ids]
    u, s, vt = np.linalg.svd(filtered)
    if np.any(s[..., -1] < .1):
        raise ValueError('Filtered rotation near singular; reject rather than guess')
    fix = np.broadcast_to(np.eye(3), filtered.shape).copy()
    fix[..., -1, -1] = np.linalg.det(u@vt)
    out = u@fix@vt
    return {'orientation': out, 'acceleration': lfilter(TAPS, [1.], a, axis=0)[ids],
            'source_indices': ids, 'nominal_availability_seconds': tm[0]/1000.+elapsed[ids],
            'recorded_availability_seconds': tm[ids]/1000.,
            'signal_delay_seconds': DELAY_FRAMES/240.}
