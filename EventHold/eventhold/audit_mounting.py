"""Reference-assisted fixed-offset diagnostic; not a deployed pose evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .audit import write_csv
from .diagnose_baseline import angle_error
from .mounting import fit_mounting,apply_mounting


def matrices(q):
    q=np.asarray(q)
    if not np.isfinite(q).all() or np.any(np.abs(np.linalg.norm(q,axis=-1)-1)>.01):
        raise ValueError('Invalid native quaternion')
    return Rotation.from_quat(q[..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(q.shape[:-1]+(3,3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix-seconds',type=float,default=2.5)
    args=p.parse_args()
    if args.prefix_seconds<=0:raise ValueError('Positive calibration duration required')
    meta=json.loads(args.cache.with_suffix('.json').read_text());names=meta['selected_sensor_names']
    with np.load(args.cache,allow_pickle=False) as data:
        rs=matrices(data['sensor_orientation_wxyz'])
        ids=[meta['segment_names'].index(name) for name in names]
        rb=matrices(data['segment_orientation_wxyz'][:,ids])
        times=data['time_ms']/1000
    prefix=int(round(args.prefix_seconds*meta['frame_rate']))
    if prefix<2 or prefix>=len(rs):raise ValueError('Need prefix and independent remainder')
    calibration=fit_mounting(rs[:prefix],rb[:prefix],names,provenance='reference_assisted_diagnostic',
                             source_note='Xsens reference segment orientations from first %.3f s only; not independent physical calibration'%args.prefix_seconds)
    aligned=apply_mounting(rs,calibration,names,allow_reference_diagnostic=True)
    errors=angle_error(aligned,rb)
    rows=[]
    for i,name in enumerate(names):
        row={'sensor':name,'prefix_mean_deg':float(errors[:prefix,i].mean()),
             'remainder_median_deg':float(np.median(errors[prefix:,i])),
             'remainder_p95_deg':float(np.quantile(errors[prefix:,i],.95)),
             'remainder_max_deg':float(errors[prefix:,i].max())}
        rows.append(row)
    bins=[]
    for start in np.arange(args.prefix_seconds,times[-1],10):
        mask=(times>=start)&(times<start+10)
        for i,name in enumerate(names):
            if mask.any():bins.append({'start_seconds':float(start),'sensor':name,'frames':int(mask.sum()),
                                      'mean_alignment_error_deg':float(errors[mask,i].mean()),
                                      'p95_alignment_error_deg':float(np.quantile(errors[mask,i],.95))})
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'sensor_alignment.csv',rows,list(rows[0]))
    write_csv(args.output/'alignment_over_time.csv',bins,list(bins[0]))
    calibration_record={'provenance':calibration.provenance,'source_note':calibration.source_note,
                        'sensor_names':names,'bone_to_sensor':calibration.bone_to_sensor.tolist(),
                        'prefix_frames':prefix,'prefix_seconds':args.prefix_seconds,
                        'deployable_profile_eligible':False}
    (args.output/'reference_assisted_offset.json').write_text(json.dumps(calibration_record,indent=2)+'\n')
    summary={'status':'reference_assisted_calibration_diagnostic_not_model_performance',
             'test_pose_set_accessed':False,'reference_prefix_used':True,
             'reference_beyond_prefix_used_for_fitting':False,
             'evaluation_reference_after_prefix_used_only_for_error':True,
             'reference_pose_used_for_network_initialization':False,'model_run':False,
             'offset_is_effective_alignment_not_proven_physical_mounting':True,
             'interpretation_limit':'Residual includes full-body estimator corrections, sensor filter disagreement, calibration error and motion effects; it is not an isolated drift measurement.',
             'native_fps':meta['frame_rate'],'prefix_frames':prefix,'remainder_frames':len(rs)-prefix,
             'sensors':rows,'cache':str(args.cache),
             'cache_sha256':hashlib.sha256(args.cache.read_bytes()).hexdigest(),
             'source_hashes':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['audit_mounting.py','mounting.py']}}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
