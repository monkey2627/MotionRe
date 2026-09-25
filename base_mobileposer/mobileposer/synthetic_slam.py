"""Synthesize a waist-mounted camera's visual(-inertial) SLAM output.

The camera faces outward at the environment (not the wearer), rigidly
mounted at the pelvis, so its own 6-DoF trajectory *is* the pelvis
trajectory (EgoLocate-style: 第一人称/腰部相机+IMU 联合做 SLAM+动捕, not an
external camera watching the body). We don't simulate images or scene
geometry -- AMASS has no room/environment to render, and we don't need one
-- we instead inject a noise process on the ground-truth pelvis trajectory,
calibrated to match published body-worn SLAM+IMU fusion error magnitudes:

  - EgoLocate (SIGGRAPH 2023, R07) reports ~0.66 m average localization
    error on TotalCapture with 6 IMUs + a body-worn phone camera doing
    ORB-SLAM3-based visual SLAM (vs. 0.72 m for offline DROID-SLAM).
  - Its follow-up EgoHDM (2024) improves this to ~0.07 m with tighter
    fusion -- the achievable range spans roughly an order of magnitude
    depending on fusion quality, not a hard floor set by the sensor.
  - Bare visual-inertial odometry (no human-motion context) on EuRoC:
    ORB-SLAM3 reports ATE 0.014-0.082 m, scale error 0.2-1.1%.

Rehab exercises happen in a small, repeatedly-revisited room, so we model
the error as *bounded* (mean-reverting / Ornstein-Uhlenbeck), not
unboundedly growing like raw dead-reckoning: a small-room SLAM system
continually re-recognizes already-mapped surroundings (the loop-closure
regime), so absolute error stays roughly stationary rather than diverging
over a session. This is a deliberately simple stand-in for a real SLAM
error process, not a simulator -- treat the calibrated magnitude, not the
exact process shape, as the load-bearing assumption; sensitivity to this is
exactly what stage 4 (robustness sweep) is for.
"""
import math

import torch


def _ou_process(shape, fps, time_constant_s, stationary_rms, generator=None):
    """Isotropic Ornstein-Uhlenbeck noise: mean-reverting random walk with a
    fixed stationary RMS magnitude, independent per trailing dimension.

    shape: (T, D). Returns [T, D] noise trace, noise[0] drawn from the
    stationary distribution (not zero) so short sequences aren't biased low.
    """
    T, D = shape
    dt = 1.0 / fps
    theta = 1.0 / time_constant_s
    per_axis_rms = stationary_rms / math.sqrt(D)
    sigma = per_axis_rms * math.sqrt(2 * theta)
    noise = torch.zeros(T, D)
    noise[0] = torch.randn(D, generator=generator) * per_axis_rms
    step_std = sigma * math.sqrt(dt)
    decay = 1 - theta * dt
    for t in range(1, T):
        noise[t] = noise[t - 1] * decay + torch.randn(D, generator=generator) * step_std
    return noise


def synthesize_slam_position(tran_gt, fps=30.0, target_rms_m=0.66, time_constant_s=2.0, seed=None):
    """tran_gt: [T,3] ground-truth pelvis position (meters).
    Returns [T,3] synthetic body-worn-SLAM position estimate.

    target_rms_m defaults to EgoLocate's own reported 0.66 m; pass a smaller
    value (down to ~0.07 m, EgoHDM's reported number) to probe the
    achievable-with-good-fusion end of the range in stage 1's ceiling check.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    error = _ou_process(tran_gt.shape, fps, time_constant_s, target_rms_m, generator)
    return tran_gt + error


def synthesize_slam_orientation(root_ori_gt, fps=30.0, target_rms_deg=3.0, time_constant_s=5.0, seed=None):
    """root_ori_gt: [T,3,3] ground-truth pelvis global rotation.
    Returns [T,3,3] synthetic estimate, perturbed by a small mean-reverting
    rotation-vector noise (axis-angle applied on the left: noisy = R_err @ gt).

    No specific published number was found for body-worn-camera-SLAM heading
    accuracy in this exact scenario (unlike the position numbers, which come
    directly from EgoLocate); target_rms_deg=3.0 is a conservative
    placeholder representing "camera-anchored yaw drifts much less than
    uncalibrated pure-IMU yaw, but isn't perfect" -- this parameter must be
    swept in stage 4's robustness check, not treated as load-bearing.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    T = root_ori_gt.shape[0]
    rotvec_error = _ou_process((T, 3), fps, time_constant_s, math.radians(target_rms_deg), generator)
    angle = rotvec_error.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = rotvec_error / angle
    cos, sin = torch.cos(angle), torch.sin(angle)
    zero = torch.zeros_like(axis[:, :1])
    kx, ky, kz = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
    K = torch.stack([
        torch.cat([zero, -kz, ky], dim=-1),
        torch.cat([kz, zero, -kx], dim=-1),
        torch.cat([-ky, kx, zero], dim=-1),
    ], dim=-2)
    eye = torch.eye(3).expand(T, 3, 3)
    R_err = eye + sin.unsqueeze(-1) * K + (1 - cos).unsqueeze(-1) * (K @ K)
    return R_err @ root_ori_gt


def synthesize_slam_joint(tran_gt, ori_gt, fps=30.0, target_rms_m=0.66, target_rms_deg=3.0,
                           quality_time_constant_s=6.0, quality_strength=1.5, seed=None):
    """Jointly synthesize position + orientation SLAM signals sharing ONE
    underlying 'tracking quality' latent process, instead of two fully
    independent noise draws (synthesize_slam_position/_orientation above).

    Motivation: real visual(-inertial) SLAM quality is a shared hidden state
    -- feature-poor scenes, motion blur, occlusion, fast rotation -- that
    degrades position AND heading estimates AT THE SAME TIME, not on
    independent per-DOF schedules. A fusion network that only sees one axis
    at a time can't exploit this; a joint network conditioned on both axis's
    residuals simultaneously can, in principle, use the position channel's
    current disagreement as an early warning for the orientation channel
    (and vice versa) -- this is the hypothesis under test for
    JointAnchorFusion (see joint_anchor_fusion.py).

    Returns (slam_pos [T,3], slam_ori [T,3,3], quality_scale [T]) -- the last
    is returned for diagnostic/oracle comparisons, not meant as a model input.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    T = tran_gt.shape[0]

    quality = _ou_process((T, 1), fps, quality_time_constant_s, 1.0, generator).squeeze(-1)
    scale = torch.exp(quality_strength * quality).clamp(0.3, 4.0)   # [T], shared multiplier

    pos_error = _ou_process(tran_gt.shape, fps, 2.0, target_rms_m, generator) * scale.unsqueeze(-1)
    slam_pos = tran_gt + pos_error

    rotvec_error = _ou_process((T, 3), fps, 5.0, math.radians(target_rms_deg), generator) * scale.unsqueeze(-1)
    angle = rotvec_error.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = rotvec_error / angle
    cos, sin = torch.cos(angle), torch.sin(angle)
    zero = torch.zeros_like(axis[:, :1])
    kx, ky, kz = axis[:, 0:1], axis[:, 1:2], axis[:, 2:3]
    K = torch.stack([
        torch.cat([zero, -kz, ky], dim=-1),
        torch.cat([kz, zero, -kx], dim=-1),
        torch.cat([-ky, kx, zero], dim=-1),
    ], dim=-2)
    eye = torch.eye(3).expand(T, 3, 3)
    R_err = eye + sin.unsqueeze(-1) * K + (1 - cos).unsqueeze(-1) * (K @ K)
    slam_ori = R_err @ ori_gt

    return slam_pos, slam_ori, scale
