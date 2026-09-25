"""Evaluate the sliding-mode-observer fusion (sliding_mode_fusion.py) on the
EXACT same frozen-backbone + train/holdout split as train_camera_fusion.py,
for a direct, apples-to-apples comparison against:
  - pure-IMU baseline            (no fusion)
  - CameraFusionGRU (learned)    (see runtime_camera_fusion_full/run_manifest.json)

Gains (lam, alpha) are grid-searched on the TRAIN fold only; the holdout
fold is touched exactly once, with the winning gains, mirroring the same
train/holdout discipline used for the learned model.

Run from base_mobileposer/:
    python -m mobileposer.evaluate_sliding_mode_fusion \
        --checkpoint checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth \
        --data-root data/no_head_5imu_surface_processed/wrists_shanks_waist \
        --output sliding_mode_fusion_results
"""
import argparse
import json
from pathlib import Path

import torch

from mobileposer.config import model_config
from mobileposer.utils.model_utils import load_model
from mobileposer.evaluate_no_head_5imu import load_sequences
from mobileposer.synthetic_slam import synthesize_slam_position
from mobileposer.sliding_mode_fusion import sliding_mode_fuse, sliding_mode_fuse_bias, grid_search_gains
from mobileposer.train_camera_fusion import split_subsets, baseline_translation


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', default='checkpoints/no_head_5imu_surface/wrists_shanks_waist/1/base_model.pth')
    ap.add_argument('--data-root', default='data/no_head_5imu_surface_processed/wrists_shanks_waist')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--slam-rms-m', type=float, default=0.66)
    ap.add_argument('--lam-grid', type=float, nargs='+',
                     default=[0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05])
    ap.add_argument('--alpha-grid', type=float, nargs='+',
                     default=[0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002])
    ap.add_argument('--method', choices=['recursive', 'bias'], default='bias',
                     help='recursive = original (integrator-windup-prone) design; '
                          'bias = re-derived bias-observer design matching the cited literature')
    ap.add_argument('--max-seq-per-subset', type=int, default=None)
    ap.add_argument('--cache-file', type=Path, default=None,
                     help='Save/load the expensive (tran_pred, tran_gt) cache to skip '
                          're-running the frozen backbone on ~9000 sequences every time.')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fuse_fn = sliding_mode_fuse_bias if args.method == 'bias' else sliding_mode_fuse

    data_root = Path(args.data_root)

    if args.cache_file is not None and args.cache_file.exists():
        print(f'Loading cached (tran_pred, tran_gt) from {args.cache_file}')
        blob = torch.load(args.cache_file, map_location='cpu', weights_only=False)
        cached, holdout_subsets = blob['cached'], blob['holdout_subsets']
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_config.device = device
        base_model = load_model(args.checkpoint).to(device).eval()
        for p in base_model.parameters():
            p.requires_grad_(False)

        split, holdout_subsets = split_subsets(data_root)   # identical split fn/salt as train_camera_fusion.py
        print(json.dumps({'holdout_subsets': sorted(holdout_subsets)}, indent=2))

        cached = {'train': [], 'holdout': []}
        all_seqs = load_sequences(data_root, max_seq=None)
        for seq in all_seqs:
            subset = seq['source'].split('[')[0]
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

    print(f'Grid-searching (lambda, alpha) on TRAIN fold only  [method={args.method}] ...')
    lam, alpha, train_err = grid_search_gains(
        cached['train'], args.slam_rms_m, args.lam_grid, args.alpha_grid, fuse_fn=fuse_fn)
    print(json.dumps({'chosen_lambda': lam, 'chosen_alpha': alpha, 'train_mean_err_m': train_err}))

    baseline_errs, fused_errs, per_seq = [], [], []
    for subset, source, tran_pred, tran_gt in cached['holdout']:
        slam_pos = synthesize_slam_position(
            tran_gt, fps=30.0, target_rms_m=args.slam_rms_m, seed=(hash(source) % 100000))
        fused = fuse_fn(tran_pred, slam_pos, lam=lam, alpha=alpha, fps=30.0)
        b_err = (tran_pred - tran_gt).norm(dim=-1).mean().item()
        f_err = (fused - tran_gt).norm(dim=-1).mean().item()
        baseline_errs.append(b_err)
        fused_errs.append(f_err)
        per_seq.append({'source': source, 'baseline_m': b_err, 'fused_m': f_err})

    mean_baseline = sum(baseline_errs) / len(baseline_errs)
    mean_fused = sum(fused_errs) / len(fused_errs)
    improvement = 1 - mean_fused / mean_baseline

    # paired significance test, same style as the learned-GRU result writeup
    import numpy as np
    from scipy import stats
    diffs = np.array(baseline_errs) - np.array(fused_errs)
    w_stat, p_wilcoxon = stats.wilcoxon(diffs)
    ci = stats.t.interval(0.95, len(diffs) - 1, loc=diffs.mean(),
                           scale=stats.sem(diffs)) if len(diffs) > 1 else (float('nan'),) * 2

    result = {
        'method': f'sliding_mode_observer_fusion (super-twisting, per-axis, causal, {args.method})',
        'chosen_lambda': lam,
        'chosen_alpha': alpha,
        'holdout_subsets': sorted(holdout_subsets),
        'n_holdout_sequences': len(cached['holdout']),
        'holdout_baseline_m': mean_baseline,
        'holdout_fused_m': mean_fused,
        'improvement_pct': improvement * 100,
        'paired_diff_mean_m': float(diffs.mean()),
        'paired_diff_95ci': [float(ci[0]), float(ci[1])],
        'wilcoxon_p': float(p_wilcoxon),
    }
    print(json.dumps(result, indent=2))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (args.output / 'holdout_detail.json').write_text(json.dumps(per_seq, indent=2, ensure_ascii=False))
    print(f"\nSaved: {args.output / 'result.json'}")


if __name__ == '__main__':
    main()
