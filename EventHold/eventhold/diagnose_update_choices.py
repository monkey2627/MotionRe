"""Offline keep/write opportunity diagnostic, never a deployable model ranking.

All policies reset at GT-mined candidate boundaries. Even the non-oracle policies
therefore use privileged episode segmentation. Oracle reads reference poses.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .audit import write_csv
from .diagnose_baseline import angle_error
from .dip import load_original, split_for_subject
from .holds import REGIONS


def rollouts(candidate, reference, valid, alpha=0.1):
    """SO(3) policies with identical predicted first-frame initialization.

    Invalid references cannot influence oracle choices. A greedy oracle is not
    a globally optimal sequence policy or an upper bound on interpolation.
    """
    q = np.asarray(candidate, dtype=float)
    gt = np.asarray(reference, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if q.ndim != 4 or q.shape[-2:] != (3, 3) or len(q) == 0:
        raise ValueError('Expected nonempty [T,J,3,3] rotations')
    if gt.shape != q.shape or valid.shape != (len(q),) or not 0 <= alpha <= 1:
        raise ValueError('Mismatched shapes or invalid interpolation weight')
    if not np.isfinite(q).all():
        raise ValueError('Candidate predictions must be finite')
    valid = valid & np.isfinite(gt).all(axis=(1, 2, 3))
    out = {'write': q.copy(), 'keep_entry': np.repeat(q[:1], len(q), axis=0),
           'fixed_so3_ema': q.copy(), 'gt_greedy_keep_write': q.copy()}
    decisions = np.zeros(len(q), dtype=bool)
    decisions[0] = True
    for t in range(1, len(q)):
        prev = out['fixed_so3_ema'][t-1]
        delta = Rotation.from_matrix(np.swapaxes(prev, -1, -2) @ q[t]).as_rotvec()
        out['fixed_so3_ema'][t] = prev @ Rotation.from_rotvec(alpha * delta).as_matrix()
        old = out['gt_greedy_keep_write'][t-1]
        # No GT access in this branch when the reference is invalid.
        choose = valid[t] and angle_error(q[t], gt[t]).mean() < angle_error(old, gt[t]).mean()
        decisions[t] = choose
        out['gt_greedy_keep_write'][t] = q[t] if choose else old
    return out, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, default=Path('reports/g0/hold_candidates.csv'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alpha', type=float, default=0.1)
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error('alpha must be in [0,1]')
    with args.candidates.open(encoding='utf-8-sig') as f:
        candidates = list(csv.DictReader(f))
    rows, skipped, hashes = [], [], {}
    for file in sorted(args.run.glob('validation_*.npz')):
        with np.load(file, allow_pickle=False) as data:
            source = str(data['source'].item())
            subject = Path(source).parent.name
            if split_for_subject(subject) != 'validation':
                raise ValueError('Only validation predictions permitted')
            matching = [c for c in candidates if c['source'] == source and c['split'] == 'validation']
            if not matching:
                continue
            hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            q, mask = data['prediction'], data['valid'].astype(bool)
            pose = np.asarray(load_original(source)['gt'][:len(q)]).reshape(-1, 24, 3)
            gt_valid = np.isfinite(pose).all(axis=(1, 2))
            gt = Rotation.from_rotvec(np.where(np.isfinite(pose), pose, 0).reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
            mask = mask & gt_valid
            for c in matching:
                start, end = int(c['start_frame']), int(c['end_frame_exclusive'])
                if not 0 <= start < end <= len(q) or not mask[start]:
                    skipped.append({'source': source, 'start': start, 'end': end,
                                    'reason': 'invalid_bounds_or_invalid_entry'})
                    continue
                ids = REGIONS[c['region']]
                pred, ref, valid = q[start:end][:, ids], gt[start:end][:, ids], mask[start:end]
                policies, decisions = rollouts(pred, ref, valid, args.alpha)
                for name, result in policies.items():
                    errors = angle_error(result, ref).mean(axis=1)
                    rows.append({'source': source, 'subject': subject, 'region': c['region'],
                                 'start_frame': start, 'end_frame_exclusive': end,
                                 'duration_seconds': float(c['duration_seconds']),
                                 'policy': name, 'mean_error_deg': float(errors[valid].mean()),
                                 'initial_frame_error_deg': float(errors[0]),
                                 'valid_frames': int(valid.sum()),
                                 'oracle_write_fraction_after_entry': float(decisions[1:][valid[1:]].mean()) if name == 'gt_greedy_keep_write' and valid[1:].any() else None})
    if not rows:
        raise RuntimeError('No valid candidate episodes')
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output/'episode_policies.csv', rows, list(rows[0]))
    aggregates = {}
    for region in sorted({r['region'] for r in rows}):
        aggregates[region] = {}
        for policy in sorted({r['policy'] for r in rows}):
            selected = [r for r in rows if r['region'] == region and r['policy'] == policy]
            subject_means = {s: float(np.mean([r['mean_error_deg'] for r in selected if r['subject'] == s]))
                             for s in sorted({r['subject'] for r in selected})}
            aggregates[region][policy] = {'episodes': len(selected), 'subject_means': subject_means,
                                          'subject_balanced_mean_deg': float(np.mean(list(subject_means.values())))}
    summary = {'status': 'offline_privileged_segmentation_diagnostic_not_model_performance',
               'test_accessed': False, 'alpha_not_tuned': args.alpha,
               'all_policies_use_gt_mined_episode_boundaries': True,
               'oracle_uses_reference_for_decisions': True,
               'initialization': 'same_first_candidate_prediction_never_gt_pose',
               'oracle_not_global_optimum_or_interpolation_upper_bound': True,
               'cannot_establish': ['novelty', 'learnability_of_gate', 'long_hold_benefit', 'exit_latency'],
               'candidate_episodes': len(rows)//4, 'skipped': skipped, 'regions': aggregates,
               'prediction_hashes': hashes,
               'candidates_sha256': hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
