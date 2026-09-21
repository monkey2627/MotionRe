"""Explicit fixed mounting calibration, with provenance checked at application.

R_WS @ R_SB = R_WB. Expected bone orientation must come from an independently
specified calibration pose for deployable use. Reference-derived offsets are
only allowed through the explicitly privileged diagnostic profile.
"""
from dataclasses import dataclass
import numpy as np


def check_rotations(r):
    r=np.asarray(r,dtype=np.float64)
    if r.shape[-2:]!=(3,3) or not np.isfinite(r).all():
        raise ValueError('Finite rotation matrices required')
    if not np.allclose(r.swapaxes(-1,-2)@r,np.eye(3),atol=2e-5) or not np.allclose(np.linalg.det(r),1,atol=2e-5):
        raise ValueError('Invalid SO(3) matrices')
    return r


@dataclass
class MountingCalibration:
    bone_to_sensor: np.ndarray
    sensor_names: tuple
    provenance: str
    calibration_frames: int
    source_note: str


def fit_mounting(sensor_prefix,expected_bone_prefix,sensor_names,*,provenance,source_note):
    """Chordal SO(3) mean of R_WS^T R_WB, using only supplied prefix frames.

    `commanded_pose` means the expected WORLD orientations and held pose were
    independently specified/verified; low IMU motion cannot establish that.
    This routine cannot verify how a caller acquired those expected rotations.
    """
    if provenance not in ('commanded_pose','reference_assisted_diagnostic'):
        raise ValueError('Explicit supported calibration provenance required')
    if not source_note.strip():raise ValueError('Calibration provenance note required')
    rs=check_rotations(sensor_prefix);rb=check_rotations(expected_bone_prefix)
    if rs.ndim!=4 or rs.shape!=rb.shape or len(rs)<2 or rs.shape[1]!=len(sensor_names):
        raise ValueError('Synchronized [prefix_frames,sensors,3,3] required')
    if len(set(sensor_names))!=len(sensor_names):raise ValueError('Duplicate sensor names')
    cross=(rs.swapaxes(-1,-2)@rb).mean(axis=0)
    u,_,vt=np.linalg.svd(cross)
    correction=np.tile(np.eye(3),(len(sensor_names),1,1))
    correction[:,-1,-1]=np.linalg.det(u@vt)
    offset=u@correction@vt
    return MountingCalibration(offset,tuple(sensor_names),provenance,len(rs),source_note)


def apply_mounting(sensor_orientation,calibration,sensor_names,*,allow_reference_diagnostic=False):
    if tuple(sensor_names)!=calibration.sensor_names:
        raise ValueError('Sensor identity/order differs from calibration')
    if calibration.provenance=='reference_assisted_diagnostic' and not allow_reference_diagnostic:
        raise ValueError('Reference-assisted calibration forbidden in deployable profile')
    if calibration.provenance not in ('commanded_pose','reference_assisted_diagnostic'):
        raise ValueError('Unknown calibration provenance')
    rs=check_rotations(sensor_orientation);offset=check_rotations(calibration.bone_to_sensor)
    if rs.ndim!=4 or rs.shape[1]!=len(sensor_names) or offset.shape!=(len(sensor_names),3,3):
        raise ValueError('Sensor shapes do not match calibration')
    return rs@offset[None]


def change_basis(rotations,vectors,basis):
    """Change BOTH world and local axis convention for orientation matrices.

    Does not perform mounting calibration or add/remove gravity.
    """
    r=check_rotations(rotations);c=check_rotations(basis)
    v=np.asarray(vectors,dtype=float)
    if c.shape!=(3,3) or v.shape!=r.shape[:-2]+(3,) or not np.isfinite(v).all():
        raise ValueError('Mismatched vectors or basis')
    return c@r@c.T, np.einsum('ij,...j->...i',c,v)
