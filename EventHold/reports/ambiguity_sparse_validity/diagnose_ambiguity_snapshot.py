"""Measurement-only pair selection; offline reference pose comparison afterwards.

Descriptors use causal histories, but galleries cover the whole original
validation sequence (including later times). This is NOT online prediction.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .adapters import eventhold_features
from .audit import write_csv
from .diagnose_baseline import angle_error
from .dip import load_original, measurements, split_for_subject


def eligible_frames(valid, history=120, stride=30):
    """A missing measurement anywhere in the full history excludes the sample."""
    good = np.asarray(valid, dtype=bool).all(axis=1)
    missing = np.r_[0, np.cumsum(~good)]
    indices = np.arange(history, len(good), stride)
    return indices[(missing[indices+1] - missing[indices-history]) == 0]


def pair_distances(rotations, accelerations):
    """Per-sensor geodesic degrees and acceleration Euclidean distances."""
    r = np.asarray(rotations, dtype=np.float64)
    a = np.asarray(accelerations, dtype=np.float64)
    trace = np.einsum('isab,jsab->ijs', r, r, optimize=True)
    angles = np.degrees(np.arccos(np.clip((trace-1)/2, -1, 1)))
    acceleration = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
    return angles, acceleration


def eligible_sampled_frames(valid, offsets, stride=30):
    """Post-hoc sensitivity: require raw observations only at descriptor times.

    This does not certify that the intervening continuous history is complete.
    """
    good = np.asarray(valid, dtype=bool).all(axis=1)
    indices = np.arange(max(offsets), len(good), stride)
    return indices[np.stack([good[indices-o] for o in offsets]).all(axis=0)]


def select_matches(indices, distances, setting, gap, top_k, permutation):
    """No pose/reference argument is accepted by this selection function.

    distances[0] is current; remaining entries are past-only descriptors.
    The same current-qualified candidate pool is used for all policies.
    """
    scale_r, scale_a = setting['max_sensor_rotation_deg'], setting['max_sensor_acc_difference_m_s2']
    costs = [(r/scale_r)**2 + (a/scale_a)**2 for r, a in distances]
    costs = [v.mean(axis=-1) for v in costs]
    r0, a0 = distances[0]
    allowed = ((r0.max(axis=-1) <= scale_r) & (a0.max(axis=-1) <= scale_a)
               & (np.abs(indices[:, None]-indices[None, :]) >= gap))
    past = np.mean(costs[1:], axis=0)
    policy_scores = {'current': costs[0], 'past_0_5s': (costs[0]+costs[1])/2,
                     'past_2s': (costs[0]+sum(costs[1:]))/len(costs),
                     'shuffled_past_2s': (costs[0]+len(costs[1:])*past[np.ix_(permutation, permutation)])/len(costs)}
    chosen = []
    for i in range(len(indices)):
        pool = np.flatnonzero(allowed[i])
        if len(pool) == 0:
            continue
        pool = pool[np.argsort(costs[0][i, pool], kind='stable')[:top_k]]
        for policy, score in policy_scores.items():
            j = int(pool[np.argmin(score[i, pool])])
            chosen.append({'query_index': i, 'neighbor_index': j, 'policy': policy,
                           'pool_size': len(pool),
                           'current_max_rotation_deg': float(r0[i, j].max()),
                           'current_max_acc_difference_m_s2': float(a0[i, j].max())})
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path('contracts/ambiguity_diagnostic_v1.json'))
    p.add_argument('--manifest', type=Path, default=Path('reports/g0/sequence_manifest.csv'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['split'] != 'validation':
        raise ValueError('This diagnostic only permits validation subjects')
    with args.manifest.open(encoding='utf-8-sig') as f:
        sources = [r for r in csv.DictReader(f) if r['split'] == 'validation']
    offsets = [0]+cfg['past_offsets_frames']
    rows, coverage, hashes = [], [], {}
    for record in sources:
        source = Path(record['source'])
        if split_for_subject(source.parent.name) != 'validation':
            raise ValueError('Refusing locked test or training data')
        raw = load_original(source)
        imu = measurements(raw, source)
        imu.require_rate(cfg['sampling_hz'])
        features = eventhold_features(imu)[:, :-1].reshape(-1, 5, 13)
        ori = features[:, :, :9].reshape(-1, 5, 3, 3)
        acc = features[:, :, 9:12]*30.0
        if cfg['require_all_measurements_valid_over_past_120_frames']:
            frames = eligible_frames(imu.valid, max(offsets), cfg['stride_frames'])
        elif cfg.get('require_all_measurements_valid_at_descriptor_offsets'):
            frames = eligible_sampled_frames(imu.valid, offsets, cfg['stride_frames'])
        else:
            raise ValueError('No explicit missingness policy')
        hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        distances = [pair_distances(ori[frames-o], acc[frames-o]) for o in offsets]
        permutation = np.random.RandomState(cfg['history_shuffle_seed']).permutation(len(frames))
        # Selection is completed before any pose reference is loaded into arrays.
        selected = [(setting, select_matches(frames, distances, setting,
                     cfg['minimum_pair_separation_frames'], cfg['rerank_top_k'], permutation))
                    for setting in cfg['settings']]
        pose = np.asarray(raw['gt']).reshape(-1, 24, 3)
        gt_valid = np.isfinite(pose).all(axis=(1, 2))
        gt = Rotation.from_rotvec(np.where(np.isfinite(pose), pose, 0).reshape(-1, 3)).as_matrix().reshape(-1, 24, 3, 3)
        for setting, choices in selected:
            bad_reference = 0
            for choice in choices:
                t, u = int(frames[choice['query_index']]), int(frames[choice['neighbor_index']])
                if not (gt_valid[t] and gt_valid[u]):
                    bad_reference += 1
                    continue
                for region, ids in cfg['regions'].items():
                    error = float(angle_error(gt[t, ids], gt[u, ids]).mean())
                    rows.append({'source': str(source), 'subject': source.parent.name,
                                 'setting': setting['name'], 'region': region,
                                 'query_frame': t, 'neighbor_frame': u,
                                 **{k:v for k,v in choice.items() if k not in ('query_index','neighbor_index')},
                                 'reference_pose_gap_deg': error,
                                 'large_pose_gap': error >= cfg['descriptive_pose_gap_deg']})
            coverage.append({'source': str(source), 'setting': setting['name'],
                             'eligible_queries': len(frames),
                             'matched_queries': len(choices)//len(cfg['policies']),
                             'policy_choices_without_reference': bad_reference})
        print(json.dumps({'source': str(source), 'eligible_queries': len(frames)}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError('No scored matches; do not silently relax thresholds')
    write_csv(args.output/'pairs.csv', rows, list(rows[0]))
    write_csv(args.output/'coverage.csv', coverage, list(coverage[0]))
    # Hierarchical summaries: queries -> sequences -> subjects, no frame-level tests.
    aggregates = []
    for setting in cfg['settings']:
        for region in cfg['regions']:
            for policy in cfg['policies']:
                group = [r for r in rows if r['setting']==setting['name'] and r['region']==region and r['policy']==policy]
                by_subject = {}
                for subject in sorted({r['subject'] for r in group}):
                    by_sequence = []
                    for source in sorted({r['source'] for r in group if r['subject']==subject}):
                        subset = [r for r in group if r['source']==source]
                        by_sequence.append({'source':source, 'queries':len(subset),
                                            'mean_gap_deg':float(np.mean([r['reference_pose_gap_deg'] for r in subset])),
                                            'large_gap_fraction':float(np.mean([r['large_pose_gap'] for r in subset]))})
                    by_subject[subject] = {'mean_gap_deg':float(np.mean([r['mean_gap_deg'] for r in by_sequence])),
                                           'large_gap_fraction':float(np.mean([r['large_gap_fraction'] for r in by_sequence])),
                                           'sequences':by_sequence}
                aggregates.append({'setting':setting['name'], 'region':region, 'policy':policy,
                                   'queries':len(group), 'subjects':by_subject,
                                   'subject_balanced_mean_gap_deg':float(np.mean([r['mean_gap_deg'] for r in by_subject.values()])) if group else None,
                                   'subject_balanced_large_gap_fraction':float(np.mean([r['large_gap_fraction'] for r in by_subject.values()])) if group else None})
    # This review queue intentionally uses GT after scoring; never a model input.
    ambiguous = sorted([r for r in rows if r['setting']=='primary' and r['region']=='hip_knee' and r['policy']=='current' and r['large_pose_gap']],
                       key=lambda r:r['reference_pose_gap_deg'], reverse=True)
    seen, queue = set(), []
    for row in ambiguous:
        key=(row['source'], *sorted([row['query_frame'],row['neighbor_frame']]))
        if key not in seen:
            seen.add(key);queue.append(row)
    write_csv(args.output/'gt_selected_review_queue.csv', queue[:30], list(rows[0]))
    summary={'status':'exploratory_same_sequence_retrospective_retrieval_not_deployed_model',
             'test_accessed':False, 'config':cfg, 'source_sha256':hashes,
             'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
             'aggregates':aggregates, 'primary_large_gap_unique_pairs':len(queue),
             'limitations':['references_are_inertial_labels_not_independent_optical_truth',
                            'same_sequence_gallery_can_include_later_times',
                            'history_matching_is_not_a_trained_transition_decoder',
                            'no_matches_does_not_prove_observability',
                            'reference_gap_is_not_sitting_standing_semantics',
                            'queries_and_pairs_are_correlated_not_independent_trials']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'primary_large_gap_unique_pairs':len(queue),'scored_pair_region_rows':len(rows)}))


if __name__=='__main__':
    main()
