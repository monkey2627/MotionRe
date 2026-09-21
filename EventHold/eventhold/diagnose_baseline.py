"""Evaluate saved validation predictions on previously mined candidate holds.

No test-set access, no semantic 'standing' labels inferred from angular error,
and no claim that unreviewed low-speed candidates constitute a final benchmark.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .audit import write_csv
from .dip import load_original,split_for_subject
from .holds import REGIONS
from .metrics import retention


def angle_error(pred,gt):
    d=np.swapaxes(pred,-1,-2)@gt
    cosine=np.clip((np.trace(d,axis1=-2,axis2=-1)-1)/2,-1,1)
    return np.degrees(np.arccos(cosine))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--candidates',type=Path,default=Path('reports/g0/hold_candidates.csv'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with args.candidates.open(encoding='utf-8-sig') as f:candidates=list(csv.DictReader(f))
    rows=[];coverage=[];subject_errors={}
    for file in sorted(args.run.glob('validation_*.npz')):
        saved=np.load(file,allow_pickle=False);source=str(saved['source'].item())
        subject=Path(source).parent.name
        if split_for_subject(subject)!='validation':raise ValueError('Refusing predictions outside locked validation split')
        data=load_original(source);t=len(saved['prediction'])
        pose=np.asarray(data['gt'][:t]).reshape(t,24,3)
        gt_valid=np.isfinite(pose).all((1,2))
        rot=Rotation.from_rotvec(np.where(np.isfinite(pose),pose,0).reshape(-1,3)).as_matrix().reshape(t,24,3,3)
        error=angle_error(saved['prediction'],rot)
        valid=saved['valid'] & gt_valid
        body=[i for i in range(1,22) if i not in [12,15]]
        if valid.any():subject_errors.setdefault(subject,[]).append(float(error[valid][:,body].mean()))
        for c in candidates:
            if c['source']!=source or c['split']!='validation':continue
            start,end=int(c['start_frame']),int(c['end_frame_exclusive'])
            if end>t:
                coverage.append({'source':source,'start_frame':start,'end_frame':end,'status':'prediction_truncated_episode_not_scored'})
                continue
            ids=REGIONS[c['region']]
            metric=retention(error[start:end,ids].mean(-1),valid[start:end],float(saved['fps']))
            rows.append({'source':source,'subject_id':subject,'region':c['region'],
                         'setting':c.get('setting','raw_5'),
                         'start_frame':start,'end_frame_exclusive':end,
                         'duration_seconds':float(c['duration_seconds']),
                         'annotation_status':c['annotation_status'],**metric})
    if not subject_errors:raise RuntimeError('No scored validation predictions')
    fields=['source','subject_id','region','setting','start_frame','end_frame_exclusive','duration_seconds','annotation_status',
            'status','mean_error_deg','valid_frames','total_frames','entry_error_deg','late_error_deg','retention_delta_deg',
            'correct_entry','sustained_failure','failure_all','retention_failure_given_correct_entry','false_standing']
    write_csv(args.output/'candidate_metrics.csv',rows,fields)
    summary={'status':'exploratory_diagnosis_unreviewed_candidates','test_accessed':False,
             'method':'DIP_only_initial_B0_not_external_strong_baseline','candidates_scored':len(rows),
             'candidates_ge_5s':sum(r['duration_seconds']>=5 for r in rows),
             'candidates_ge_20s':sum(r['duration_seconds']>=20 for r in rows),
             'subjects':{s:float(np.mean(e)) for s,e in subject_errors.items()},
             'subject_balanced_mean_body_deg':float(np.mean([np.mean(e) for e in subject_errors.values()])),
             'scoring_note':'subject_balanced_mean_of_sequence_means;root_neck_head_excluded',
             'retention_gate':'not_established;need_reviewed_long_validation_holds_and_strong_baselines',
             'censoring':coverage,'false_standing_claim':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
