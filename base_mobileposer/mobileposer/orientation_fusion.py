"""Learned fusion of the existing IMU-only root/global orientation estimate
with a synthetic waist-camera SLAM heading (see synthetic_slam.py's
synthesize_slam_orientation) -- the orientation-axis counterpart of
camera_fusion.py's CameraFusionGRU (which only handles translation).

Together with camera_fusion.py, completes the full 6-DOF drift-anchoring
system: translation via convex-combination blending in R^3, orientation via
learned-weight SLERP-style blending in SO(3) (rotations can't be linearly
combined, so the network predicts a blend factor along the geodesic between
the two rotation estimates instead of a per-axis convex weight).
"""
import torch
from torch import nn


def _so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> axis-angle. R: [...,3,3] -> [...,3]."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_angle = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)                                   # [...]
    sin_angle = torch.sin(angle).clamp_min(1e-6)
    rx = R[..., 2, 1] - R[..., 1, 2]
    ry = R[..., 0, 2] - R[..., 2, 0]
    rz = R[..., 1, 0] - R[..., 0, 1]
    axis = torch.stack([rx, ry, rz], dim=-1) / (2.0 * sin_angle).unsqueeze(-1)
    return axis * angle.unsqueeze(-1)


def _so3_exp(v: torch.Tensor) -> torch.Tensor:
    """Axis-angle -> rotation matrix (Rodrigues). v: [...,3] -> [...,3,3]."""
    shape = v.shape[:-1]
    theta = v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = v / theta
    K = torch.zeros(*shape, 3, 3, dtype=v.dtype, device=v.device)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2];  K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    sin_t = theta.unsqueeze(-1).sin()
    cos_t = theta.unsqueeze(-1).cos()
    I = torch.eye(3, dtype=v.dtype, device=v.device).expand(*shape, 3, 3)
    return I + sin_t * K + (1.0 - cos_t) * (K @ K)


class OrientationFusionGRU(nn.Module):
    """imu_ori, slam_ori: [B,T,3,3] global rotation matrices (root joint).
    Outputs a per-frame scalar blend factor along the geodesic from imu_ori
    to slam_ori (0 = fully trust IMU, mirrors camera_fusion.py's convention
    where weight=1 means trust IMU -- here we predict `blend` = 1-weight
    directly for a cleaner geodesic-interpolation formula)."""

    def __init__(self, hidden=32):
        super().__init__()
        self.gru = nn.GRU(9, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.bias)  # start near sigmoid(0)=0.5

    def _features(self, imu_ori, slam_ori):
        imu_rel = torch.zeros(*imu_ori.shape[:2], 3, dtype=imu_ori.dtype, device=imu_ori.device)
        imu_rel[:, 1:] = _so3_log(imu_ori[:, :-1].transpose(-1, -2) @ imu_ori[:, 1:])
        slam_rel = torch.zeros_like(imu_rel)
        slam_rel[:, 1:] = _so3_log(slam_ori[:, :-1].transpose(-1, -2) @ slam_ori[:, 1:])
        discrepancy = _so3_log(imu_ori.transpose(-1, -2) @ slam_ori)
        return torch.cat([imu_rel, slam_rel, discrepancy], dim=-1)

    def forward(self, imu_ori, slam_ori, state=None):
        if imu_ori.shape != slam_ori.shape or imu_ori.shape[-2:] != (3, 3):
            raise ValueError('Expected imu_ori/slam_ori of shape [B,T,3,3]')
        feat = self._features(imu_ori, slam_ori)
        h, state = self.gru(feat, state)
        weight = torch.sigmoid(self.head(h))            # [B,T,1], 1 = trust IMU
        blend = 1.0 - weight                             # geodesic fraction toward SLAM
        rel_rotvec = _so3_log(imu_ori.transpose(-1, -2) @ slam_ori)   # [B,T,3]
        step = _so3_exp(rel_rotvec * blend)               # [B,T,3,3]
        fused = imu_ori @ step
        return fused, weight, state
