"""Generate a bounded five-IMU AMASS pilot set from raw SMPL-H poses.

This is an offline reference synthesis. It is deliberately separate from the
legacy MobilePoser cache and records finite-difference support/boundaries.
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch
from .audit import sha256

SITE_NAMES=('pelvis','left_forearm','right_forearm','left_shank','right_shank')
SITE_JOINTS=(0,18,19,4,5)
SITE_VERTICES=(3021,1961,5424,1176,4662)

def synthesize(path, model, target_hz=60., max_seconds=60.):
    with np.load(path,allow_pickle=False) as d:
        fps=float(d['mocap_framerate']); pose=np.asarray(d['poses'],np.float32)
        trans=np.asarray(d['trans'],np.float32); beta=np.asarray(d['betas'],np.float32).reshape(-1)[:10]
    if pose.ndim!=2 or pose.shape[1]<72 or fps<target_hz or abs(fps/target_hz-round(fps/target_hz))>1e-5:
        raise ValueError('Only integer downsampling from >=60 Hz is supported')
    step=int(round(fps/target_hz)); ids=np.arange(0,len(pose),step)
    if max_seconds: ids=ids[:int(max_seconds*target_hz)]
    if len(ids)<120: raise ValueError('Sequence too short')
    p=torch.from_numpy(pose[ids,:72].reshape(-1,24,3)); t=torch.from_numpy(trans[ids]); b=torch.from_numpy(beta).float()
    from mobileposer.articulate import math as articulate_math
    p=articulate_math.axis_angle_to_rotation_matrix(p).view(-1,24,3,3)
    with torch.no_grad(): grot,joint,vert=model.forward_kinematics(p,b,t,calc_mesh=True)
    g=grot[:,SITE_JOINTS].numpy(); v=vert[:,SITE_VERTICES].numpy()
    # Causal second difference: frame 0/1 are invalid because the stencil is
    # centered. The last frame is also invalid at the right boundary.
    acc=np.zeros_like(v); valid=np.ones(len(v),bool); valid[:1]=False; valid[-1:]=False
    acc[1:-1]=(v[2:]-2*v[1:-1]+v[:-2])*(target_hz**2)
    return {'orientation':g.astype(np.float32),'acceleration':acc.astype(np.float32),
            'target':grot.numpy().astype(np.float32),'valid':valid,'source_indices':ids,
            'native_fps':fps,'target_fps':target_hz,'site_names':SITE_NAMES}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate-csv',type=Path,required=True)
    p.add_argument('--mobileposer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-files',type=int,default=8);p.add_argument('--max-seconds',type=float,default=60)
    args=p.parse_args();
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use new output')
    sys.path.insert(0,str(args.mobileposer_root))
    from mobileposer.articulate.model import ParametricModel
    torch.set_num_threads(2); model=ParametricModel(str(args.mobileposer_root/'mobileposer/smpl/basicmodel_m.pkl'))
    with args.candidate_csv.open(encoding='utf-8-sig') as f: all_rows=list(csv.DictReader(f))
    chosen=[]; seen=set()
    for row in all_rows:
        if row['kind']!='seated_like' or float(row['duration_seconds'])<20: continue
        if row['source'] in seen: continue
        seen.add(row['source']);chosen.append(row)
        if len(chosen)>=args.max_files:break
    args.output.mkdir(parents=True)
    manifest=[];errors=[]
    for i,row in enumerate(chosen):
        path=Path(row['source']);
        try:
            out=synthesize(path,model,max_seconds=args.max_seconds)
            name='seq_%02d.npz'%i;np.savez_compressed(args.output/name,orientation=out['orientation'],acceleration=out['acceleration'],target=out['target'],valid=out['valid'],source_indices=out['source_indices'])
            manifest.append({'id':name,'source':str(path),'source_sha256':sha256(path),'frames':len(out['valid']),'fps':60.,'native_fps':out['native_fps'],'invalid_boundary_frames':int((~out['valid']).sum()),'site_names':list(SITE_NAMES),'site_joints':list(SITE_JOINTS),'site_vertices':list(SITE_VERTICES),'synthetic_acceleration':'centered_second_difference_m_per_s2','candidate_duration_seconds':float(row['duration_seconds'])})
        except Exception as e: errors.append({'source':str(path),'error':type(e).__name__+': '+str(e)})
    summary={'status':'bounded_five_imu_synthetic_pilot_not_real_imu','files':len(manifest),'errors':errors,'manifest':manifest,'layout':list(SITE_NAMES),'head_input':False,'legacy_cache_reused':False,'model_file':str(args.mobileposer_root/'mobileposer/smpl/basicmodel_m.pkl'),'model_sha256':sha256(args.mobileposer_root/'mobileposer/smpl/basicmodel_m.pkl'),'code_sha256':sha256(Path(__file__)),'limitations':['centered acceleration labels use future reference frames offline','boundary frames invalid and must be masked','AMASS fitted motion is not measured IMU','no sensor noise/mounting variability yet','source candidates are explored, not locked test']}
    (args.output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
