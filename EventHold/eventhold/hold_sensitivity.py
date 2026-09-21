"""Exploratory annotation sensitivity, independent of all model predictions.

Centered reference smoothing is OFFLINE LABEL EXPLORATION ONLY. It must never
be applied to an online input stream or silently replace the locked raw rule.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

from .audit import write_csv
from .dip import load_original,measurements
from .holds import mine,REGIONS


def smooth_reference(pose,hz=60,seconds=1.):
    pose=np.asarray(pose).reshape(-1,24,3).copy()
    ids=sorted(set(j for group in REGIONS.values() for j in group))
    selected=pose[:,ids]
    finite=np.isfinite(selected).all(-1)
    r=Rotation.from_rotvec(np.where(np.isfinite(selected),selected,0).reshape(-1,3)).as_matrix().reshape(len(pose),len(ids),3,3)
    width=int(round(hz*seconds))|1
    mean=uniform_filter1d(r,size=width,axis=0,mode='nearest')
    u,_,vt=np.linalg.svd(mean)
    det=np.linalg.det(u@vt)
    u[...,:,-1]*=det[...,None]
    smooth=Rotation.from_matrix((u@vt).reshape(-1,3,3)).as_rotvec().reshape(selected.shape)
    # Mark every window contaminated by a missing reference as invalid.
    trustworthy=uniform_filter1d(finite.astype(float),size=width,axis=0,mode='nearest')>=1-1e-12
    smooth[~trustworthy]=np.nan
    pose[:,ids]=smooth
    return pose.reshape(-1,72)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=Path('reports/g0/sequence_manifest.csv'))
    p.add_argument('--output',type=Path,default=Path('reports/g0/annotation_sensitivity'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with args.manifest.open(encoding='utf-8-sig') as f:records=list(csv.DictReader(f))
    settings=[('raw_5',False,5.),('raw_10',False,10.),('raw_15',False,15.),('reference_smoothed_1s_5',True,5.)]
    all_rows=[]
    for row in records:
        if row['split']=='test_locked':continue
        d=load_original(row['source']);rec=measurements(d,row['source'])
        smooth=smooth_reference(d['gt'])
        for name,use_smooth,threshold in settings:
            for h in mine(smooth if use_smooth else d['gt'],rec.valid,threshold_deg_s=threshold):
                all_rows.append({'source':row['source'],'subject_id':row['subject_id'],'split':row['split'],
                                 'setting':name,'primary_protocol_changed':False,**h})
    fields=['source','subject_id','split','setting','primary_protocol_changed','region','start_frame',
            'end_frame_exclusive','duration_seconds','max_speed_deg_s','annotation_status','pose_semantics']
    write_csv(args.output/'candidates.csv',all_rows,fields)
    summary={'status':'exploratory_annotation_sensitivity_not_model_evaluation','test_inspected':False,
             'main_rule_unchanged':'raw_5deg_s','future_reference_used_only_for_offline_sensitivity':True,'settings':{}}
    for name,_,_ in settings:
        rows=[r for r in all_rows if r['setting']==name]
        summary['settings'][name]={'candidates':len(rows),'ge_5s':sum(r['duration_seconds']>=5 for r in rows),
                                  'ge_20s':sum(r['duration_seconds']>=20 for r in rows),
                                  'ge_60s':sum(r['duration_seconds']>=60 for r in rows),
                                  'max_duration_seconds':max([r['duration_seconds'] for r in rows],default=0)}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
