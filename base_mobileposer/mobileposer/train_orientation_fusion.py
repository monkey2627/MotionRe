"""Train OrientationFusionGRU on top of a frozen, already-trained MobilePoserNet
-- the orientation-axis counterpart of train_camera_fusion.py. Together they
complete the full 6-DOF drift-anchoring system (translation + heading).

Pipeline per sequence:
  1. Run the frozen base_model once to get both the existing IMU-only root
     orientation `ori_pred` (pose_pred[:,0]) and translation `tran_pred`
     (same forward_offline call as train_camera_fusion.py -- free, no extra
     compute cost, just also keeping the orientation output this time).
  2. Synthesize a waist-camera SLAM heading from ground-truth root
     orientation (synthetic_slam.synthesize_slam_orientation).
  3. Train only OrientationFusionGRU to combine (1) and (2) into something
     closer to ground-truth root orientation. The base network is frozen.

Run from base_mobileposer/:
    python -m mobileposer.train_orientation_fusion \
        --output runtime_orientation_fusion --cache-file /tmp/full_cache.pt
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn

from mobileposer.config import model_config
from mobileposer.utils.model_utils import load_model
from mobileposer.evaluate_no_head_5imu import load_sequences, prepare_imu
from mobileposer.synthetic_slam import synthesize_slam_orientation
from mobileposer.orientation_fusion import OrientationFusionGRU
from mobileposer.train_camera_fusion import split_subsets


def angle_between_rotmats(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    R = R1.transpose(-1, -2) @ R2
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.rad2deg(torch.acos(((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)))


@torch.no_grad()
def baseline_full(base_model, seq, device):
    imu = prepare_imu(seq['acc'], seq['ori']).to(device)
    base_model.reset()
    pose_pred, _, tran_pred, _ = base_model.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
    pose_pred = pose_pred.cpu().squeeze(0) if pose_pred.dim() == 4 else pose_pred.cpu()
    tran_pred = tran_pred.cpu().squeeze(0) if tran_pred.dim() == 3 else tran_pred.cpu()
    return pose_pred[:, 0], tran_pred   # root global rotation [T,3,3], translation [T,3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', default='checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth')
    ap.add_argument('--data-root', default='data/no_head_5imu_surface_processed/wrists_shanks_waist')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--slam-rms-deg', type=float, default=3.0,
                     help='Fixed noise level (used as-is unless --rms-deg-range is set).')
    ap.add_argument('--rms-deg-range', type=float, nargs=2, default=None,
                     help='Domain randomization: sample target_rms_deg ~ Uniform(min,max) '
                          'PER TRAINING SEQUENCE instead of a fixed value, so the network '
                          'is not overfit to one specific SLAM quality (see robustness_sweep.py '
                          'finding: fixed-3deg training actively hurts at 8deg test noise).')
    ap.add_argument('--sweep-grid', type=float, nargs='+', default=[1.0, 2.0, 3.0, 5.0, 8.0],
                     help='Test-time noise levels for the final robustness sweep report.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache-file', type=Path, default=None)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_config.device = device
    data_root = Path(args.data_root)

    if args.cache_file is not None and args.cache_file.exists():
        print(f'Loading cached (ori_pred, ori_gt, tran) from {args.cache_file}')
        blob = torch.load(args.cache_file, map_location='cpu', weights_only=False)
        cached, holdout_subsets = blob['cached'], blob['holdout_subsets']
    else:
        base_model = load_model(args.checkpoint).to(device).eval()
        for p in base_model.parameters():
            p.requires_grad_(False)

        split, holdout_subsets = split_subsets(data_root)
        print(json.dumps({'holdout_subsets': sorted(holdout_subsets)}, indent=2))

        cached = {'train': [], 'holdout': []}
        all_seqs = load_sequences(data_root, max_seq=None)
        for seq in all_seqs:
            subset = seq['source'].split('[')[0]
            fold = split.get(subset)
            if fold is None:
                continue
            ori_pred, tran_pred = baseline_full(base_model, seq, device)
            gt_pose = seq['pose'].float()
            T = min(len(gt_pose), len(ori_pred))
            ori_gt = gt_pose[:T, 0]
            ori_pred = ori_pred[:T]
            cached[fold].append((subset, seq['source'], ori_pred, ori_gt))
        if args.cache_file is not None:
            torch.save({'cached': cached, 'holdout_subsets': holdout_subsets}, args.cache_file)
            print(f'Saved cache to {args.cache_file}')

    print(json.dumps({'train_sequences': len(cached['train']), 'holdout_sequences': len(cached['holdout'])}))

    fusion = OrientationFusionGRU(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(fusion.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in fusion.parameters())
    print(json.dumps({'fusion_param_count': n_params}))

    def run_holdout(seed_offset=0):
        fusion.eval()
        baseline_errs, fused_errs, per_seq = [], [], []
        with torch.no_grad():
            for subset, source, ori_pred, ori_gt in cached['holdout']:
                slam_ori = synthesize_slam_orientation(ori_gt, fps=30.0, target_rms_deg=args.slam_rms_deg,
                                                         seed=(hash(source) % 100000) + seed_offset)
                imu_in = ori_pred.unsqueeze(0).to(device)
                slam_in = slam_ori.unsqueeze(0).to(device)
                fused, weight, _ = fusion(imu_in, slam_in)
                fused = fused.squeeze(0).cpu()
                b_err = angle_between_rotmats(ori_pred, ori_gt).mean().item()
                f_err = angle_between_rotmats(fused, ori_gt).mean().item()
                baseline_errs.append(b_err)
                fused_errs.append(f_err)
                per_seq.append({'source': source, 'baseline_deg': b_err, 'fused_deg': f_err,
                                 'mean_weight': weight.mean().item()})
        fusion.train()
        return baseline_errs, fused_errs, per_seq

    history = []
    for epoch in range(args.epochs):
        fusion.train()
        train_losses = []
        order = list(range(len(cached['train'])))
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        for idx in order:
            subset, source, ori_pred, ori_gt = cached['train'][idx]
            if args.rms_deg_range is not None:
                lo, hi = args.rms_deg_range
                rms = random.uniform(lo, hi)
            else:
                rms = args.slam_rms_deg
            slam_ori = synthesize_slam_orientation(ori_gt, fps=30.0, target_rms_deg=rms, seed=None)
            imu_in = ori_pred.unsqueeze(0).to(device)
            slam_in = slam_ori.unsqueeze(0).to(device)
            target = ori_gt.unsqueeze(0).to(device)
            fused, weight, _ = fusion(imu_in, slam_in)
            loss = nn.functional.mse_loss(fused, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        baseline_errs, fused_errs, per_seq = run_holdout()
        record = {
            'epoch': epoch + 1,
            'train_mse': sum(train_losses) / max(len(train_losses), 1),
            'holdout_baseline_deg': sum(baseline_errs) / max(len(baseline_errs), 1),
            'holdout_fused_deg': sum(fused_errs) / max(len(fused_errs), 1),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    manifest = {
        'status': 'orientation_fusion_heading_correction',
        'checkpoint': args.checkpoint,
        'data_root': str(data_root),
        'holdout_subsets': sorted(holdout_subsets),
        'slam_rms_deg': args.slam_rms_deg,
        'fusion_param_count': n_params,
        'epochs': args.epochs,
        'hidden': args.hidden,
        'seed': args.seed,
        'history': history,
    }
    (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    torch.save({'state_dict': fusion.state_dict(), 'manifest': manifest}, args.output / 'last.pt')

    _, _, per_seq_final = run_holdout()
    (args.output / 'holdout_detail.json').write_text(json.dumps(per_seq_final, indent=2, ensure_ascii=False))

    print('\n--- Final robustness sweep (test-time noise level, no retraining) ---')
    fusion.eval()
    sweep = []
    with torch.no_grad():
        for rms in args.sweep_grid:
            b_errs, f_errs = [], []
            for subset, source, ori_pred, ori_gt in cached['holdout']:
                slam_ori = synthesize_slam_orientation(ori_gt, fps=30.0, target_rms_deg=rms,
                                                          seed=hash(source) % 100000)
                imu_in, slam_in = ori_pred.unsqueeze(0).to(device), slam_ori.unsqueeze(0).to(device)
                fused, _, _ = fusion(imu_in, slam_in)
                b_errs.append(angle_between_rotmats(ori_pred, ori_gt).mean().item())
                f_errs.append(angle_between_rotmats(fused.squeeze(0).cpu(), ori_gt).mean().item())
            b, f = sum(b_errs) / len(b_errs), sum(f_errs) / len(f_errs)
            row = {'test_rms_deg': rms, 'baseline_deg': b, 'fused_deg': f, 'improvement_pct': (1 - f / b) * 100}
            sweep.append(row)
            print(json.dumps(row))
    (args.output / 'sweep_result.json').write_text(json.dumps(sweep, indent=2))


if __name__ == '__main__':
    main()
