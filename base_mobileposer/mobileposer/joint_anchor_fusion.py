"""JointAnchorFusion: a single shared-hidden-state GRU that fuses BOTH the
translation and orientation channels together, instead of the two
independent networks in camera_fusion.py (CameraFusionGRU) and
orientation_fusion.py (OrientationFusionGRU).

Hypothesis under test: real visual SLAM tracking quality is a shared latent
state that degrades position and heading estimates simultaneously (see
synthetic_slam.synthesize_slam_joint's motivation docstring). If so, a joint
model conditioned on both axes' residuals at once should out-perform two
independent single-axis models of matched total capacity, because it can use
one channel's current disagreement as an early-warning signal for the other.
"""
import torch
from torch import nn

from mobileposer.orientation_fusion import _so3_log, _so3_exp


class JointAnchorFusion(nn.Module):
    """auxiliary_quality=True adds a third head that explicitly predicts the
    shared 'tracking quality' scale (synthetic_slam.synthesize_slam_joint's
    returned `scale`), supervised with an auxiliary loss. Motivation: the
    implicit shared-hidden-state design (Pillar 4) helped orientation cleanly
    but left position's gain statistically fragile -- forcing the shared
    representation to explicitly reconstruct the common quality signal, rather
    than hoping the network discovers it as a useful implicit feature, should
    make BOTH heads draw on the same explicit reliability estimate more
    consistently."""

    def __init__(self, hidden=40, auxiliary_quality=False):
        super().__init__()
        self.auxiliary_quality = auxiliary_quality
        # position features (9): imu_disp(3), slam_disp(3), discrepancy(3)
        # orientation features (9): imu_rel(3), slam_rel(3), discrepancy(3)
        self.gru = nn.GRU(18, hidden, batch_first=True)
        self.pos_head = nn.Linear(hidden, 3)   # per-axis position trust weight
        self.ori_head = nn.Linear(hidden, 1)   # scalar orientation trust weight
        nn.init.zeros_(self.pos_head.bias)
        nn.init.zeros_(self.ori_head.bias)
        if auxiliary_quality:
            self.quality_head = nn.Linear(hidden, 1)   # predicts log(scale)

    @staticmethod
    def _pos_features(imu_pos, slam_pos):
        imu_disp = torch.zeros_like(imu_pos)
        imu_disp[:, 1:] = imu_pos[:, 1:] - imu_pos[:, :-1]
        slam_disp = torch.zeros_like(slam_pos)
        slam_disp[:, 1:] = slam_pos[:, 1:] - slam_pos[:, :-1]
        discrepancy = imu_pos - slam_pos
        return torch.cat([imu_disp, slam_disp, discrepancy], dim=-1)

    @staticmethod
    def _ori_features(imu_ori, slam_ori):
        imu_rel = torch.zeros(*imu_ori.shape[:2], 3, dtype=imu_ori.dtype, device=imu_ori.device)
        imu_rel[:, 1:] = _so3_log(imu_ori[:, :-1].transpose(-1, -2) @ imu_ori[:, 1:])
        slam_rel = torch.zeros_like(imu_rel)
        slam_rel[:, 1:] = _so3_log(slam_ori[:, :-1].transpose(-1, -2) @ slam_ori[:, 1:])
        discrepancy = _so3_log(imu_ori.transpose(-1, -2) @ slam_ori)
        return torch.cat([imu_rel, slam_rel, discrepancy], dim=-1)

    def forward(self, imu_pos, slam_pos, imu_ori, slam_ori, state=None):
        """imu_pos/slam_pos: [B,T,3]. imu_ori/slam_ori: [B,T,3,3].
        Returns (fused_pos [B,T,3], fused_ori [B,T,3,3], pos_weight [B,T,3],
        ori_weight [B,T,1], state)."""
        pos_feat = self._pos_features(imu_pos, slam_pos)
        ori_feat = self._ori_features(imu_ori, slam_ori)
        feat = torch.cat([pos_feat, ori_feat], dim=-1)
        h, state = self.gru(feat, state)

        pos_weight = torch.sigmoid(self.pos_head(h))     # [B,T,3], 1=trust IMU
        fused_pos = pos_weight * imu_pos + (1 - pos_weight) * slam_pos

        ori_weight = torch.sigmoid(self.ori_head(h))     # [B,T,1], 1=trust IMU
        blend = 1.0 - ori_weight
        rel_rotvec = _so3_log(imu_ori.transpose(-1, -2) @ slam_ori)
        step = _so3_exp(rel_rotvec * blend)
        fused_ori = imu_ori @ step

        pred_log_quality = self.quality_head(h).squeeze(-1) if self.auxiliary_quality else None
        return fused_pos, fused_ori, pos_weight, ori_weight, state, pred_log_quality
