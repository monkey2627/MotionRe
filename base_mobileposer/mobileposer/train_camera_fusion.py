"""Train CameraFusionGRU on top of a frozen, already-trained MobilePoserNet.

Pipeline per sequence:
  1. Run the frozen base_model (wrists_shanks_waist checkpoint) once to get
     the existing IMU-only translation baseline `tran_pred`.
  2. Synthesize a waist-camera SLAM position from ground-truth `tran`
     (synthetic_slam.synthesize_slam_position).
  3. Train only CameraFusionGRU to combine (1) and (2) into something
     closer to ground truth `tran`. The base network's four submodules are
     never updated.

Dataset-level (not subject-level) train/holdout split by AMASS subset --
same known limitation as today's other full-scale runs (no_head_layouts.py
does not retain per-sequence source filenames), documented rather than
hidden.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import nn

from mobileposer.config import model_config, paths
from mobileposer.utils.model_utils import load_model
from mobileposer.evaluate_no_head_5imu import load_sequences, prepare_imu
from mobileposer.synthetic_slam import synthesize_slam_position
from mobileposer.camera_fusion import CameraFusionGRU


def split_subsets(data_root: Path, holdout_fraction=0.2, salt='camera-fusion-v1'):
    subsets = sorted(p.stem for p in data_root.glob('*.pt'))
    n_holdout = max(1, round(len(subsets) * holdout_fraction))
    ordered = sorted(subsets, key=lambda s: hashlib.sha256(f'{salt}:{s}'.encode()).hexdigest())
    holdout = set(ordered[:n_holdout])
    return {s: ('holdout' if s in holdout else 'train') for s in subsets}, holdout


@torch.no_grad()
def baseline_translation(base_model, seq, device):
    imu = prepare_imu(seq['acc'], seq['ori']).to(device)
    base_model.reset()
    _, _, tran_pred, _ = base_model.forward_offline(imu.unsqueeze(0), [imu.shape[0]])
    tran_pred = tran_pred.cpu().squeeze(0) if tran_pred.dim() == 3 else tran_pred.cpu()
    return tran_pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth')
    parser.add_argument('--data-root', default='data/no_head_5imu_surface_processed/wrists_shanks_waist')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--slam-rms-m', type=float, default=0.66, help="EgoLocate's own reported number; see synthetic_slam.py")
    parser.add_argument('--max-seq-per-subset', type=int, default=None, help='For smoke tests only')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache-file', type=Path, default=None,
                         help='Save/load the expensive (tran_pred, tran_gt) cache to skip '
                              're-running the frozen backbone on every sequence.')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_config.device = device

    data_root = Path(args.data_root)

    if args.cache_file is not None and args.cache_file.exists():
        print(f'Loading cached (tran_pred, tran_gt) from {args.cache_file}')
        blob = torch.load(args.cache_file, map_location='cpu', weights_only=False)
        cached, holdout_subsets = blob['cached'], blob['holdout_subsets']
        print(json.dumps({'train_sequences': len(cached['train']), 'holdout_sequences': len(cached['holdout'])}))
        base_model = None
    else:
        base_model = load_model(args.checkpoint).to(device).eval()
        for p in base_model.parameters():
            p.requires_grad_(False)

        split, holdout_subsets = split_subsets(data_root)
        print(json.dumps({'subsets': split, 'holdout_subsets': sorted(holdout_subsets)}, indent=2))

        # Cache (tran_pred, tran_gt) once per sequence -- the frozen base model
        # never changes, so its output doesn't need recomputing every epoch. The
        # synthetic SLAM signal IS regenerated each epoch (different seed) for
        # training examples, acting as noise-realization augmentation; holdout
        # uses one fixed seed per sequence for a reproducible, fair comparison.
        cached = {'train': [], 'holdout': []}
        # load_sequences globs the whole directory and its max_seq is a *global*
        # cap (stops after the first N sequences across all files in glob order),
        # not per-subset -- so it must not be used to try to get "N per subset".
        # Load everything and apply the per-subset cap ourselves when bucketing.
        all_seqs = load_sequences(data_root, max_seq=None)
        for seq in all_seqs:
            subset = seq['source'].split('[')[0]  # load_sequences uses f"{pt_path.stem}[{local_idx}]"
            fold = split.get(subset)
            if fold is None:
                continue
            if args.max_seq_per_subset is not None:
                count_so_far = sum(1 for s in cached[fold] if s[0] == subset)
                if count_so_far >= args.max_seq_per_subset:
                    continue
            tran_pred = baseline_translation(base_model, seq, device)
            tran_gt = seq['tran'].float()
            T = min(len(tran_gt), len(tran_pred))
            tran_gt = tran_gt[:T]
            tran_pred = tran_pred[:T] - tran_pred[:1] + tran_gt[:1]
            cached[fold].append((subset, seq['source'], tran_pred, tran_gt))
        if args.cache_file is not None:
            torch.save({'cached': cached, 'holdout_subsets': holdout_subsets}, args.cache_file)
            print(f'Saved cache to {args.cache_file}')
    print(json.dumps({'train_sequences': len(cached['train']), 'holdout_sequences': len(cached['holdout'])}))

    fusion = CameraFusionGRU(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(fusion.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in fusion.parameters())
    print(json.dumps({'fusion_param_count': n_params}))

    def run_holdout(seed_offset=0):
        fusion.eval()
        baseline_errs, fused_errs, per_seq = [], [], []
        with torch.no_grad():
            for subset, source, tran_pred, tran_gt in cached['holdout']:
                slam_pos = synthesize_slam_position(tran_gt, fps=30.0, target_rms_m=args.slam_rms_m,
                                                      seed=(hash(source) % 100000) + seed_offset)
                imu_in = tran_pred.unsqueeze(0).to(device)
                slam_in = slam_pos.unsqueeze(0).to(device)
                fused, weight, _ = fusion(imu_in, slam_in)
                fused = fused.squeeze(0).cpu()
                b_err = (tran_pred - tran_gt).norm(dim=-1).mean().item()
                f_err = (fused - tran_gt).norm(dim=-1).mean().item()
                baseline_errs.append(b_err)
                fused_errs.append(f_err)
                per_seq.append({'source': source, 'baseline_m': b_err, 'fused_m': f_err,
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
            subset, source, tran_pred, tran_gt = cached['train'][idx]
            slam_pos = synthesize_slam_position(tran_gt, fps=30.0, target_rms_m=args.slam_rms_m,
                                                  seed=None)  # fresh noise draw every time: augmentation
            imu_in = tran_pred.unsqueeze(0).to(device)
            slam_in = slam_pos.unsqueeze(0).to(device)
            target = tran_gt.unsqueeze(0).to(device)
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
            'holdout_baseline_m': sum(baseline_errs) / max(len(baseline_errs), 1),
            'holdout_fused_m': sum(fused_errs) / max(len(fused_errs), 1),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    manifest = {
        'status': 'camera_fusion_translation_correction',
        'checkpoint': args.checkpoint,
        'data_root': str(data_root),
        'holdout_subsets': sorted(holdout_subsets),
        'slam_rms_m': args.slam_rms_m,
        'fusion_param_count': n_params,
        'epochs': args.epochs,
        'hidden': args.hidden,
        'seed': args.seed,
        'history': history,
        'limitations': [
            'dataset-level (AMASS subset) holdout split, not subject-level -- no_head_layouts.py does not retain per-sequence source filenames',
            'synthetic SLAM signal, not real camera/SLAM data -- calibrated to EgoLocate/EgoHDM reported error magnitudes on TotalCapture, not measured from this project\'s own hardware',
            'no DIP-IMU real-IMU validation possible -- DIP-IMU has no translation ground truth (tran is always 0 in that dataset)',
        ],
    }
    (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    torch.save({'state_dict': fusion.state_dict(), 'manifest': manifest}, args.output / 'last.pt')

    # Save per-sequence holdout detail from the final epoch for paired significance testing downstream.
    _, _, per_seq_final = run_holdout()
    (args.output / 'holdout_detail.json').write_text(json.dumps(per_seq_final, indent=2, ensure_ascii=False))

    # Per-FRAME weight + per-frame IMU error, for the trust-weight/failure-mode
    # correlation analysis (does the learned gate shift toward the camera
    # specifically when/where the IMU is currently wrong?).
    fusion.eval()
    per_frame = []
    with torch.no_grad():
        for subset, source, tran_pred, tran_gt in cached['holdout']:
            slam_pos = synthesize_slam_position(tran_gt, fps=30.0, target_rms_m=args.slam_rms_m,
                                                  seed=(hash(source) % 100000))
            imu_in = tran_pred.unsqueeze(0).to(device)
            slam_in = slam_pos.unsqueeze(0).to(device)
            fused, weight, _ = fusion(imu_in, slam_in)
            weight = weight.squeeze(0).cpu()               # [T,3], 1=trust IMU fully
            imu_err = (tran_pred - tran_gt).norm(dim=-1)    # [T]  per-frame |imu_pos - gt|
            per_frame.append({
                'source': source,
                'weight_mean_axis': weight.mean(dim=-1).numpy(),   # [T]
                'imu_err_m': imu_err.numpy(),                       # [T]
            })
    torch.save(per_frame, args.output / 'per_frame_detail.pt')
    print(f"Saved per-frame detail: {args.output / 'per_frame_detail.pt'}")


if __name__ == '__main__':
    main()
