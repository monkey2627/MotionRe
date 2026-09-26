"""Noise-level robustness sweep: evaluate an ALREADY-TRAINED independent
fusion network (e.g. camera_fusion_v2 -- trained at ONE calibrated noise
level) across a RANGE of test-time SLAM noise levels, using cached
(tran_pred/tran_gt, ori_pred/ori_gt) data (rebuild via --cache-file on
train_camera_fusion.py / train_orientation_fusion.py if not present).

This is a deployment-realistic check: in practice you train/ship one model,
not one per session, so what matters is whether it still helps (vs pure IMU)
across the plausible range of real-world camera-SLAM quality, not just at the
one point it happened to be calibrated on. This sweep is what revealed that a
fixed-noise-trained orientation model can actively hurt outside its training
band -- see train_orientation_fusion.py's --rms-deg-range domain-randomization
fix (runtime_orientation_fusion_dr/) for the robust replacement.

Run from base_mobileposer/:
    python -m mobileposer.robustness_sweep \
        --tran-cache /tmp/tran_cache.pt --ori-cache /tmp/ori_cache.pt \
        --camera-fusion-ckpt runtime_camera_fusion_v2/last.pt \
        --orientation-fusion-ckpt runtime_orientation_fusion_dr/last.pt \
        --output robustness_sweep_results
"""
import argparse
import json
from pathlib import Path

import torch

from mobileposer.synthetic_slam import synthesize_slam_position, synthesize_slam_orientation
from mobileposer.camera_fusion import CameraFusionGRU
from mobileposer.orientation_fusion import OrientationFusionGRU


def angle_between_rotmats(R1, R2):
    R = R1.transpose(-1, -2) @ R2
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tran-cache', type=Path, required=True)
    ap.add_argument('--ori-cache', type=Path, required=True)
    ap.add_argument('--camera-fusion-ckpt', type=Path, required=True)
    ap.add_argument('--orientation-fusion-ckpt', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pos-rms-grid', type=float, nargs='+', default=[0.03, 0.07, 0.15, 0.30, 0.66, 1.0])
    ap.add_argument('--ori-rms-grid', type=float, nargs='+', default=[1.0, 2.0, 3.0, 5.0, 8.0])
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tran_cache = torch.load(args.tran_cache, map_location='cpu', weights_only=False)['cached']['holdout']
    ori_cache = torch.load(args.ori_cache, map_location='cpu', weights_only=False)['cached']['holdout']

    cam_blob = torch.load(args.camera_fusion_ckpt, map_location='cpu', weights_only=False)
    cam_net = CameraFusionGRU(hidden=cam_blob['manifest']['hidden']).to(device)
    cam_net.load_state_dict(cam_blob['state_dict']); cam_net.eval()

    ori_blob = torch.load(args.orientation_fusion_ckpt, map_location='cpu', weights_only=False)
    ori_net = OrientationFusionGRU(hidden=ori_blob['manifest']['hidden']).to(device)
    ori_net.load_state_dict(ori_blob['state_dict']); ori_net.eval()

    print(f"Trained at: pos_rms={cam_blob['manifest']['slam_rms_m']}m  "
          f"ori_rms={ori_blob['manifest']['slam_rms_deg']}deg")

    results = {'position': [], 'orientation': []}

    print('\n--- Position sweep ---')
    with torch.no_grad():
        for rms in args.pos_rms_grid:
            baseline_errs, fused_errs = [], []
            for subset, source, tran_pred, tran_gt in tran_cache:
                slam_pos = synthesize_slam_position(tran_gt, fps=30.0, target_rms_m=rms,
                                                      seed=hash(source) % 100000)
                imu_in, slam_in = tran_pred.unsqueeze(0).to(device), slam_pos.unsqueeze(0).to(device)
                fused, _, _ = cam_net(imu_in, slam_in)
                baseline_errs.append((tran_pred - tran_gt).norm(dim=-1).mean().item())
                fused_errs.append((fused.squeeze(0).cpu() - tran_gt).norm(dim=-1).mean().item())
            b, f = sum(baseline_errs) / len(baseline_errs), sum(fused_errs) / len(fused_errs)
            imp = (1 - f / b) * 100
            row = {'test_rms_m': rms, 'baseline_m': b, 'fused_m': f, 'improvement_pct': imp}
            results['position'].append(row)
            print(json.dumps(row))

    print('\n--- Orientation sweep ---')
    with torch.no_grad():
        for rms in args.ori_rms_grid:
            baseline_errs, fused_errs = [], []
            for subset, source, ori_pred, ori_gt in ori_cache:
                slam_ori = synthesize_slam_orientation(ori_gt, fps=30.0, target_rms_deg=rms,
                                                          seed=hash(source) % 100000)
                imu_in, slam_in = ori_pred.unsqueeze(0).to(device), slam_ori.unsqueeze(0).to(device)
                fused, _, _ = ori_net(imu_in, slam_in)
                baseline_errs.append(angle_between_rotmats(ori_pred, ori_gt).mean().item())
                fused_errs.append(angle_between_rotmats(fused.squeeze(0).cpu(), ori_gt).mean().item())
            b, f = sum(baseline_errs) / len(baseline_errs), sum(fused_errs) / len(fused_errs)
            imp = (1 - f / b) * 100
            row = {'test_rms_deg': rms, 'baseline_deg': b, 'fused_deg': f, 'improvement_pct': imp}
            results['orientation'].append(row)
            print(json.dumps(row))

    (args.output / 'sweep_result.json').write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {args.output / 'sweep_result.json'}")


if __name__ == '__main__':
    main()
