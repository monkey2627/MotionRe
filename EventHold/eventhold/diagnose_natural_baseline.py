"""P5 reference-prefix-assisted, DIP-only B0 exploration; no method ranking.

No full-body SMPL retargeting. Compare four local lower-body rotations under
canonical axis correspondence, with separate reference geometry consistency.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import freqz

from .audit import sha256, write_csv
from .audit_mounting import matrices
from .mounting import MountingCalibration, apply_mounting, change_basis
from .natural_diagnostic_input import (BASIS, LOWER_NAMES, LOWER_SMPL,
                                        TAPS, lower_reference, causal_decimate)
from .natural_motion import TARGET
from .records import FIVE, ImuRecord
from .adapters import eventhold_features
from .baseline import CausalBaseline
from .diagnose_baseline import angle_error


def stats(values):
    return {'mean':float(np.mean(values)), 'median':float(np.median(values)),
            'p95':float(np.quantile(values,.95)), 'max':float(np.max(values))}


def geometry_audit(global_r, position, names):
    """Check native segment -Z against observed reference limb directions.

    Shared-estimator consistency only, not optical validation or exact SMPL
    anatomical rest calibration. Diagnostic aborts on large convention mismatch.
    """
    result={}
    for side in ['Left','Right']:
        for proximal,distal in [('UpperLeg','LowerLeg'),('LowerLeg','Foot')]:
            i,j=names.index(side+proximal),names.index(side+distal)
            v=position[:,j]-position[:,i];n=np.linalg.norm(v,axis=-1)
            if np.any(n<.1):raise ValueError('Degenerate reference limb')
            direction=-global_r[:,i,:,2]
            error=np.degrees(np.arccos(np.clip(np.sum(direction*v,axis=-1)/n,-1,1)))
            result[side+proximal]=stats(error)
    return result


def infer(features, checkpoint, chunk=240):
    # Trusted local checkpoint only. Model receives measurements, never labels.
    model=CausalBaseline(hidden=int(checkpoint['manifest']['config']['hidden']))
    model.load_state_dict(checkpoint['state_dict']);model.eval();state=None;outputs=[]
    with torch.no_grad():
        for start in range(0,len(features),chunk):
            pred,state=model(torch.from_numpy(features[start:start+chunk]).float()[None],state)
            outputs.append(pred[0].numpy())
    return np.concatenate(outputs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--calibration-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use new diagnostic output')
    torch.set_num_threads(2)
    meta=json.loads(args.cache.with_suffix('.json').read_text())
    if meta['frame_rate']!=240 or meta['selected_sensor_names']!=TARGET:
        raise ValueError('Only named native five-sensor 240 Hz profile supported')
    previous=json.loads((args.calibration_dir/'summary.json').read_text())
    cache_hash=sha256(args.cache)
    if cache_hash!=previous['cache_sha256']:raise ValueError('Calibration/cache mismatch')
    c=json.loads((args.calibration_dir/'reference_assisted_offset.json').read_text())
    if c['provenance']!='reference_assisted_diagnostic':raise ValueError('Explicit privileged diagnostic required')
    calibration=MountingCalibration(np.asarray(c['bone_to_sensor']),tuple(c['sensor_names']),c['provenance'],c['prefix_frames'],c['source_note'])
    prefix=c['prefix_frames']
    with np.load(args.cache,allow_pickle=False) as data:
        rs=matrices(data['sensor_orientation_wxyz'][prefix:])
        acc=data['sensor_free_acceleration'][prefix:].copy()
        frame=data['frame_index'].copy();time=data['time_ms'].copy()
        rb=matrices(data['segment_orientation_wxyz'])
        position=data['segment_position'].copy()
    geometry=geometry_audit(rb,position,meta['segment_names'])
    if any(row['p95']>10 for row in geometry.values()):
        raise ValueError('Reference limb axes disagree with geometry by >10 deg P95')
    aligned=apply_mounting(rs,calibration,TARGET,allow_reference_diagnostic=True)
    r,a=change_basis(aligned,acc,BASIS)
    sampled=causal_decimate(r,a,frame[prefix:],time[prefix:])
    ids=sampled['source_indices']+prefix
    record=ImuRecord(sampled['nominal_availability_seconds'],FIVE,
                     sampled['orientation'].astype(np.float32),sampled['acceleration'].astype(np.float32),
                     np.ones((len(ids),5),dtype=bool),str(args.cache)).validate()
    record.require_rate(60)
    features=eventhold_features(record)
    checkpoint=torch.load(args.checkpoint,map_location='cpu')
    prediction=infer(features,checkpoint)
    # Independent block-boundary check without recurrent resets.
    n=min(1800,len(features))
    check=infer(features[:n],checkpoint,chunk=137)
    parity=float(np.max(np.abs(prediction[:n]-check)))
    if parity>2e-5:raise ValueError('Continuous state chunk parity failed')
    reference=lower_reference(rb,meta['segment_names'])
    errors=angle_error(prediction[:,LOWER_SMPL],reference[ids])
    delayed_errors=angle_error(prediction[:,LOWER_SMPL],reference[ids-32])
    # Diagnostic bending proxy, not SMPL FK or semantic standing classification.
    down=np.array([0.,-1.,0.])
    bends=np.degrees(np.arccos(np.clip(np.einsum('i,tnij,j->tn',down,prediction[:,[4,5]],down),-1,1)))
    ref_bends=np.degrees(np.arccos(np.clip(np.einsum('i,tnij,j->tn',down,reference[ids,2:],down),-1,1)))
    candidate_rows=[]
    with args.candidates.open(encoding='utf-8-sig') as f:candidates=list(csv.DictReader(f))
    for candidate in candidates:
        start,end=int(candidate['start_frame']),int(candidate['end_frame_exclusive'])
        mask=(ids>=start)&(ids<end)
        if not mask.any():continue
        selected=ids[mask];v=errors[mask].mean(axis=1)
        first=selected<start+2400;last=selected>=end-2400
        row={'kind':candidate['kind'],'start_seconds':candidate['start_seconds'],
             'duration_seconds':candidate['duration_seconds'],'frames':int(mask.sum()),
             'mean_lower_error_deg':float(v.mean()),'first10s_error_deg':float(v[first].mean()),
             'last10s_error_deg':float(v[last].mean()),
             'predicted_knee_bend_mean_deg':float(bends[mask].mean()),
             'reference_knee_bend_mean_deg':float(ref_bends[mask].mean())}
        candidate_rows.append(row)
    bins=[]
    for start in np.arange(0,time[-1]/1000.,10):
        mask=(record.timestamps>=start)&(record.timestamps<start+10)
        if not mask.any():continue
        row={'start_seconds':float(start),'frames':int(mask.sum()),
             'lower_mean_error_deg':float(errors[mask].mean()),
             'delay_aligned_lower_error_deg':float(delayed_errors[mask].mean()),
             'predicted_knee_bend_mean_deg':float(bends[mask].mean()),
             'reference_knee_bend_mean_deg':float(ref_bends[mask].mean())}
        row.update({name+'_error_deg':float(errors[mask,j].mean()) for j,name in enumerate(LOWER_NAMES)})
        bins.append(row)
    f,h=freqz(TAPS,worN=8192,fs=240)
    summary={'status':'reference_assisted_cross_domain_weak_B0_exploration_not_main_ranking',
             'model_run':True,'head_sensor_input':False,'reference_prefix_frames':prefix,
             'reference_after_prefix_used_for_input':False,'gt_network_initialization':False,
             'test_set_accessed':False,'recurrent_state':'zero once after calibration; continuous thereafter',
             'frames_60hz':len(ids),'first_availability_seconds':float(record.timestamps[0]),
             'last_availability_seconds':float(record.timestamps[-1]),
             'future_samples':0,'filter_signal_delay_seconds':32/240.,
             'filter_startup_discard_seconds':64/240.,'fir_taps':TAPS.tolist(),
             'linear_fir_stopband_max_db_above30hz':float(20*np.log10(np.max(np.abs(h[f>=30])))),
             'rotation_filter_limit':'matrix FIR plus SO3 projection; nonlinear projection has no strict stopband guarantee',
             'nominal_time_max_discrepancy_seconds':float(np.max(np.abs(record.timestamps-sampled['recorded_availability_seconds']))),
             'target_scope':'four local hip/knee canonical-axis rotation proxies, NOT complete SMPL retargeted ground truth',
             'reference_geometry_consistency_deg':geometry,
             'chunk_parity_first1800_max_abs':parity,
             'lower_rotation_error_current_time_deg':stats(errors.mean(axis=1)),
             'lower_rotation_error_delay_aligned_deg':stats(delayed_errors.mean(axis=1)),
             'candidate_results':candidate_rows,
             'limitations':['DIP-only small B0 cross-domain','reference-assisted mounting',
                            'anatomical rest frame differences not eliminated','single exploration recording',
                            'no complete exit event','Xsens reference not independently validated'],
             'input_hashes':{'cache':cache_hash,'checkpoint':sha256(args.checkpoint),
                             'calibration':sha256(args.calibration_dir/'reference_assisted_offset.json'),
                             'candidates':sha256(args.candidates)},
             'code_hashes':{name:sha256(Path(__file__).with_name(name)) for name in
                            ['diagnose_natural_baseline.py','natural_diagnostic_input.py','mounting.py','baseline.py','adapters.py']},
             'torch':torch.__version__}
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/'candidates.csv',candidate_rows,list(candidate_rows[0]))
    write_csv(args.output/'time_bins.csv',bins,list(bins[0]))
    np.savez_compressed(args.output/'lower_predictions.npz',source_indices=ids,
                        availability_seconds=record.timestamps,prediction=prediction[:,LOWER_SMPL],
                        reference=reference[ids],error_deg=errors,predicted_knee_bend_deg=bends,
                        reference_knee_bend_deg=ref_bends)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
