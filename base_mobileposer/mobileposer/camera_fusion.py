"""Learned fusion of the existing IMU-only translation estimate with a
synthetic waist-mounted-camera SLAM position (see synthetic_slam.py).

Generalizes the validated fixed-weight blend (w * imu_pos + (1-w) *
slam_pos, optimal w found by sweep to cut translation error ~48% on a
59-sequence AMASS check: 0.427m -> 0.221m) into a causal, per-axis,
time-varying weight predicted by a small GRU -- the fixed sweep already
proved there's a real gain available; this network's only job is to do
better than a single global constant by conditioning the blend on how much
the two signals currently agree/disagree.

Deliberately outputs a *convex combination* of the two inputs (sigmoid
weight), not a free-form residual: bounded by construction to the segment
between the two signals, so it can't extrapolate to something worse than
both -- keeps this a "learn to combine two decent guesses" model, not a
generic corrector that could in principle diverge from either.
"""
import torch
from torch import nn


class CameraFusionGRU(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = hidden
        # features per frame: imu displacement(3), slam displacement(3), current imu-slam discrepancy(3)
        self.gru = nn.GRU(9, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 3)
        nn.init.zeros_(self.head.bias)  # start near sigmoid(0)=0.5, no prior toward either signal

    def _features(self, imu_pos, slam_pos):
        imu_disp = torch.zeros_like(imu_pos)
        imu_disp[:, 1:] = imu_pos[:, 1:] - imu_pos[:, :-1]
        slam_disp = torch.zeros_like(slam_pos)
        slam_disp[:, 1:] = slam_pos[:, 1:] - slam_pos[:, :-1]
        discrepancy = imu_pos - slam_pos
        return torch.cat([imu_disp, slam_disp, discrepancy], dim=-1)

    def forward(self, imu_pos, slam_pos, state=None):
        """imu_pos, slam_pos: [B,T,3], both already expressed relative to the
        same sequence-start reference point. Returns (fused[B,T,3], weight[B,T,3], state)."""
        if imu_pos.shape != slam_pos.shape or imu_pos.ndim != 3 or imu_pos.shape[-1] != 3:
            raise ValueError('Expected imu_pos/slam_pos of shape [B,T,3]')
        feat = self._features(imu_pos, slam_pos)
        h, state = self.gru(feat, state)
        weight = torch.sigmoid(self.head(h))  # [B,T,3] in (0,1), 1 = fully trust IMU
        fused = weight * imu_pos + (1 - weight) * slam_pos
        return fused, weight, state
