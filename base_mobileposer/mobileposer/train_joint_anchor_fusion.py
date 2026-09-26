"""Train JointAnchorFusion (shared-hidden-state position+orientation fusion)
and compare it against the two INDEPENDENT baseline networks (CameraFusionGRU,
OrientationFusionGRU) re-evaluated under the SAME correlated-failure synthetic
SLAM signal (synthetic_slam.synthesize_slam_joint), for a fair apples-to-apples
test of the "shared tracking-quality state helps" hypothesis.

Requires the two per-axis caches (rebuild via --cache-file on train_camera_fusion.py /
train_orientation_fusion.py if not present -- the original /tmp caches from this session
were cleaned up as temporary artifacts) and the two trained independent-baseline checkpoints:
    runtime_camera_fusion_v2/last.pt
    runtime_orientation_fusion_dr/last.pt   (domain-randomized, robust version -- see
                                              orientation_fusion.py's robustness fix;
                                              supersedes the old fixed-noise checkpoint)

Run from base_mobileposer/:
    python -m mobileposer.train_joint_anchor_fusion \
        --tran-cache /tmp/tran_cache.pt --ori-cache /tmp/ori_cache.pt \
        --camera-fusion-ckpt runtime_camera_fusion_v2/last.pt \
        --orientation-fusion-ckpt runtime_orientation_fusion_dr/last.pt \
        --output runtime_joint_anchor_fusion
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from mobileposer.synthetic_slam import synthesize_slam_joint
from mobileposer.joint_anchor_fusion import JointAnchorFusion
from mobileposer.camera_fusion import CameraFusionGRU
from mobileposer.orientation_fusion import OrientationFusionGRU


def angle_between_rotmats(R1, R2):
    R = R1.transpose(-1, -2) @ R2
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)))


def merge_caches(tran_cache, ori_cache, fold):
    ori_by_source = {source: (ori_pred, ori_gt) for _, source, ori_pred, ori_gt in ori_cache[fold]}
    merged = []
    for subset, source, tran_pred, tran_gt in tran_cache[fold]:
        if source not in ori_by_source:
            continue
        ori_pred, ori_gt = ori_by_source[source]
        T = min(len(tran_gt), len(ori_gt))
        merged.append((subset, source, tran_pred[:T], tran_gt[:T], ori_pred[:T], ori_gt[:T]))
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tran-cache', type=Path, required=True)
    ap.add_argument('--ori-cache', type=Path, required=True)
    ap.add_argument('--camera-fusion-ckpt', type=Path, required=True)
    ap.add_argument('--orientation-fusion-ckpt', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--hidden', type=int, default=40)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--ori-loss-weight', type=float, default=20.0,
                     help='Balances position-MSE (meters^2, small) against orientation-MSE '
                          '(rotation-matrix entries, different scale).')
    ap.add_argument('--slam-rms-m', type=float, default=0.66)
    ap.add_argument('--slam-rms-deg', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--auxiliary-quality', action='store_true',
                     help='Add an explicit auxiliary head+loss supervising the shared '
                          'tracking-quality scale, instead of hoping the network discovers '
                          'it implicitly (see joint_anchor_fusion.py docstring).')
    ap.add_argument('--quality-loss-weight', type=float, default=0.5)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tran_blob = torch.load(args.tran_cache, map_location='cpu', weights_only=False)
    ori_blob = torch.load(args.ori_cache, map_location='cpu', weights_only=False)
    tran_cache, ori_cache = tran_blob['cached'], ori_blob['cached']

    merged = {'train': merge_caches(tran_cache, ori_cache, 'train'),
              'holdout': merge_caches(tran_cache, ori_cache, 'holdout')}
    print(json.dumps({'train_sequences': len(merged['train']), 'holdout_sequences': len(merged['holdout'])}))

    joint = JointAnchorFusion(hidden=args.hidden, auxiliary_quality=args.auxiliary_quality).to(device)
    opt = torch.optim.Adam(joint.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in joint.parameters())
    print(json.dumps({'joint_fusion_param_count': n_params}))

    # Load the two independent baselines for a fair re-evaluation under the SAME
    # correlated-failure noise used to train/test the joint model.
    cam_blob = torch.load(args.camera_fusion_ckpt, map_location='cpu', weights_only=False)
    cam_net = CameraFusionGRU(hidden=cam_blob['manifest']['hidden']).to(device)
    cam_net.load_state_dict(cam_blob['state_dict'])
    cam_net.eval()

    ori_blob2 = torch.load(args.orientation_fusion_ckpt, map_location='cpu', weights_only=False)
    ori_net = OrientationFusionGRU(hidden=ori_blob2['manifest']['hidden']).to(device)
    ori_net.load_state_dict(ori_blob2['state_dict'])
    ori_net.eval()

    def run_holdout(seed_offset=0):
        joint.eval()
        rows = []
        with torch.no_grad():
            for subset, source, tran_pred, tran_gt, ori_pred, ori_gt in merged['holdout']:
                seed = (hash(source) % 100000) + seed_offset
                slam_pos, slam_ori, _quality = synthesize_slam_joint(
                    tran_gt, ori_gt, fps=30.0, target_rms_m=args.slam_rms_m,
                    target_rms_deg=args.slam_rms_deg, seed=seed)

                imu_pos_b, slam_pos_b = tran_pred.unsqueeze(0).to(device), slam_pos.unsqueeze(0).to(device)
                imu_ori_b, slam_ori_b = ori_pred.unsqueeze(0).to(device), slam_ori.unsqueeze(0).to(device)

                # --- independent baselines, same correlated-noise test signals ---
                cam_fused, _, _ = cam_net(imu_pos_b, slam_pos_b)
                ori_fused, _, _ = ori_net(imu_ori_b, slam_ori_b)
                indep_pos_err = (cam_fused.squeeze(0).cpu() - tran_gt).norm(dim=-1).mean().item()
                indep_ori_err = angle_between_rotmats(ori_fused.squeeze(0).cpu(), ori_gt).mean().item()

                # --- joint model ---
                j_pos, j_ori, pw, ow, _, _ = joint(imu_pos_b, slam_pos_b, imu_ori_b, slam_ori_b)
                joint_pos_err = (j_pos.squeeze(0).cpu() - tran_gt).norm(dim=-1).mean().item()
                joint_ori_err = angle_between_rotmats(j_ori.squeeze(0).cpu(), ori_gt).mean().item()

                pure_imu_pos_err = (tran_pred - tran_gt).norm(dim=-1).mean().item()
                pure_imu_ori_err = angle_between_rotmats(ori_pred, ori_gt).mean().item()

                rows.append({
                    'source': source,
                    'pure_imu_pos_m': pure_imu_pos_err, 'pure_imu_ori_deg': pure_imu_ori_err,
                    'indep_pos_m': indep_pos_err, 'indep_ori_deg': indep_ori_err,
                    'joint_pos_m': joint_pos_err, 'joint_ori_deg': joint_ori_err,
                })
        joint.train()
        return rows

    history = []
    for epoch in range(args.epochs):
        joint.train()
        train_losses = []
        order = list(range(len(merged['train'])))
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        for idx in order:
            subset, source, tran_pred, tran_gt, ori_pred, ori_gt = merged['train'][idx]
            slam_pos, slam_ori, quality = synthesize_slam_joint(
                tran_gt, ori_gt, fps=30.0, target_rms_m=args.slam_rms_m,
                target_rms_deg=args.slam_rms_deg, seed=None)

            imu_pos_b, slam_pos_b = tran_pred.unsqueeze(0).to(device), slam_pos.unsqueeze(0).to(device)
            imu_ori_b, slam_ori_b = ori_pred.unsqueeze(0).to(device), slam_ori.unsqueeze(0).to(device)
            pos_target = tran_gt.unsqueeze(0).to(device)
            ori_target = ori_gt.unsqueeze(0).to(device)

            fused_pos, fused_ori, _, _, _, pred_log_quality = joint(imu_pos_b, slam_pos_b, imu_ori_b, slam_ori_b)
            loss = (nn.functional.mse_loss(fused_pos, pos_target)
                    + args.ori_loss_weight * nn.functional.mse_loss(fused_ori, ori_target))
            if args.auxiliary_quality:
                quality_target = torch.log(quality).unsqueeze(0).to(device)
                loss = loss + args.quality_loss_weight * nn.functional.mse_loss(pred_log_quality, quality_target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(joint.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        rows = run_holdout()
        record = {
            'epoch': epoch + 1,
            'train_loss': sum(train_losses) / max(len(train_losses), 1),
            'pure_imu_pos_m': sum(r['pure_imu_pos_m'] for r in rows) / len(rows),
            'pure_imu_ori_deg': sum(r['pure_imu_ori_deg'] for r in rows) / len(rows),
            'indep_pos_m': sum(r['indep_pos_m'] for r in rows) / len(rows),
            'indep_ori_deg': sum(r['indep_ori_deg'] for r in rows) / len(rows),
            'joint_pos_m': sum(r['joint_pos_m'] for r in rows) / len(rows),
            'joint_ori_deg': sum(r['joint_ori_deg'] for r in rows) / len(rows),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    torch.save({'state_dict': joint.state_dict(), 'history': history}, args.output / 'last.pt')
    final_rows = run_holdout()
    (args.output / 'holdout_detail.json').write_text(json.dumps(final_rows, indent=2, ensure_ascii=False))
    print(f"\nSaved: {args.output / 'holdout_detail.json'}")


if __name__ == '__main__':
    main()
