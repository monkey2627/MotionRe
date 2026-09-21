"""Offline synthetic measurement generation; never an online sensor adapter."""
import numpy as np
from scipy.signal import firwin,lfilter

JOINTS=[0,18,19,4,5]
VERTICES=[3021,1961,5424,1176,4662]


def body_pose(pose):
    pose=np.asarray(pose)
    if pose.ndim!=2 or pose.shape[1]!=156 or not np.isfinite(pose).all():
        raise ValueError('Explicit SMPLH 52-joint axis-angle required')
    p=pose.reshape(-1,52,3)
    return p[:,list(range(23))+[37]].copy()


def synthesize_at_60hz(global_orientation,vertex_position,native_fps):
    """Filter native samples, align to FIR center, then centered acceleration.

    Only exact 60/120 Hz admitted in pilot. 65-tap 20 Hz FIR on positions and
    rotation entries, then SO3 projection. No startup padding is reported as
    measured. Acceleration stencil is +/-4 target frames; boundary discarded.
    The reference trajectory lookahead is explicit; network causality separate.
    """
    if native_fps not in (60,120):raise ValueError('Pilot admits exact60/120Hz only')
    r=np.asarray(global_orientation,dtype=float);v=np.asarray(vertex_position,dtype=float)
    if r.ndim!=4 or r.shape[1:]!=(5,3,3) or v.shape!=r.shape[:2]+(3,) or len(r)<90:
        raise ValueError('Need enough synchronized five-sensor samples')
    if not np.isfinite(r).all() or not np.isfinite(v).all():raise ValueError('Finite inputs required')
    taps=firwin(65,20,fs=native_fps,window=('kaiser',8.))
    step=int(native_fps/60);ends=np.arange(64,len(r),step);centers=ends-32
    smooth_r=lfilter(taps,[1.],r,axis=0)[ends]
    u,s,vt=np.linalg.svd(smooth_r)
    if np.any(s[...,-1]<.1):raise ValueError('Filtered rotations near singular')
    fix=np.broadcast_to(np.eye(3),smooth_r.shape).copy();fix[...,-1,-1]=np.linalg.det(u@vt)
    smooth_r=u@fix@vt;smooth_v=lfilter(taps,[1.],v,axis=0)[ends]
    h=4
    acc=(smooth_v[2*h:]+smooth_v[:-2*h]-2*smooth_v[h:-h])*(60/h)**2
    return {'orientation':smooth_r[h:-h].astype(np.float32),'acceleration':acc.astype(np.float32),
            'native_center_indices':centers[h:-h],'timestamps':centers[h:-h]/native_fps,
            'offline_reference_future_support_seconds':32/native_fps+h/60,
            'offline_reference_past_support_seconds':32/native_fps+h/60,
            'rotation_filter':'matrix FIR then SO3; nonlinear projection not strictly bandlimited'}
