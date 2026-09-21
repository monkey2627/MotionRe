"""Six-source controlled mean-male-shape five-IMU pilot, no model training."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from .audit import sha256
from .audit_amass_coverage import ALIGN
from .synthetic_imu import JOINTS,VERTICES,body_pose,synthesize_at_60hz
from .records import FIVE,ImuRecord
from .adapters import eventhold_features


def read_csv(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def select_sources(manifest,candidates):
    by_path={Path(x['source']).resolve():x for x in manifest}
    chosen=[]
    for split in ['train','development']:
        used=set()
        for c in sorted(candidates,key=lambda x:(x['source'],int(x['start_frame']))):
            m=by_path[Path(c['source']).resolve()]
            if c['kind']!='seated_like' or c['complete_transition_proxy']!='True':continue
            if m['split']!=split or m['subject_key'] in used or float(m['fps']) not in [60,120]:continue
            chosen.append({**m,'role':'transition_candidate','candidate':c});used.add(m['subject_key'])
            if len(used)==2:break
        if len(used)!=2:raise ValueError('Need two distinct transition subjects per split')
        generic=next(m for m in sorted(manifest,key=lambda x:x['source']) if m['split']==split
                     and m['subject_key'] not in used and float(m['max_seated_seconds'])==0
                     and float(m['fps']) in [60,120] and 10<=float(m['duration_seconds'])<=60)
        chosen.append({**generic,'role':'generic_screen_negative','candidate':None})
    return chosen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--mobileposer-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use new pilot output')
    split=json.loads((args.split_dir/'summary.json').read_text())
    if sha256(args.split_dir/'manifest.csv')!=split['manifest_sha256']:raise ValueError('Frozen manifest changed')
    selected=select_sources(read_csv(args.split_dir/'manifest.csv'),read_csv(args.candidates))
    sys.path.insert(0,str(args.mobileposer_root))
    import torch
    from mobileposer.articulate.model import ParametricModel
    torch.set_num_threads(2)
    model_path=args.mobileposer_root/'mobileposer/smpl/basicmodel_m.pkl'
    model=ParametricModel(str(model_path),use_pose_blendshape=False)
    args.output.mkdir(parents=True);results=[]
    for i,item in enumerate(selected):
        path=Path(item['source'])
        if sha256(path)!=item['sha256']:raise ValueError('Frozen source changed')
        with np.load(path,allow_pickle=False) as data:
            p=body_pose(data['poses']);fps=float(data['mocap_framerate'])
            trans=np.asarray(data['trans'],dtype=float)@ALIGN.T
            gender=str(data['gender'].item()) if 'gender' in data else 'unknown'
        r=Rotation.from_rotvec(p.reshape(-1,3)).as_matrix().reshape(-1,24,3,3)
        r[:,0]=ALIGN@r[:,0]
        orientations=[];vertices=[];joints=[]
        with torch.no_grad():
            for start in range(0,len(r),128):
                gr,j,v=model.forward_kinematics(torch.tensor(r[start:start+128],dtype=torch.float32),
                    torch.zeros(10),torch.tensor(trans[start:start+128],dtype=torch.float32),calc_mesh=True)
                orientations.append(gr[:,JOINTS].numpy());vertices.append(v[:,VERTICES].numpy());joints.append(j.numpy())
        raw_ori=np.concatenate(orientations);raw_v=np.concatenate(vertices);raw_j=np.concatenate(joints)
        out=synthesize_at_60hz(raw_ori,raw_v,fps);ids=out['native_center_indices']
        rec=ImuRecord(out['timestamps'],FIVE,out['orientation'],out['acceleration'],
                      np.ones((len(ids),5),bool),str(path)).validate();rec.require_rate(60)
        features=eventhold_features(rec)
        if features.shape!=(len(ids),66) or not np.isfinite(features).all():raise ValueError('Bad feature interface')
        filename='%02d_%s.npz'%(i,item['split'])
        np.savez_compressed(args.output/filename,features=features.astype(np.float32),
            orientation=rec.orientation,acceleration=rec.acceleration,timestamps=rec.timestamps,
            target=r[ids].astype(np.float32),valid=rec.valid,reference_joints=raw_j[ids],
            native_center_indices=ids,source=str(path))
        entry={**item,'source_gender':gender,'shape_policy':'controlled male template beta0; source betas deliberately unused',
               'frames_60hz':len(ids),'artifact':filename,'artifact_sha256':sha256(args.output/filename),
               'native_hz':fps,'reference_support_each_side_seconds':out['offline_reference_future_support_seconds'],
               'acceleration_norm_p99_m_s2':float(np.quantile(np.linalg.norm(rec.acceleration,axis=-1),.99)),
               'orientation_orthogonality_max_abs':float(np.max(np.abs(rec.orientation.swapaxes(-1,-2)@rec.orientation-np.eye(3))))}
        results.append(entry);print(json.dumps({'completed':i+1,'source':str(path),'frames':len(ids)}),flush=True)
    report={'status':'synthetic_input_pilot_complete_not_trained_or_real_sensor_validation',
            'layout':list(FIVE),'orientation_joints':JOINTS,'vertices':VERTICES,'target_hz':60,
            'smplh_hand_indices':[22,37],'pose_blendshapes':False,
            'smpl_sha256':sha256(model_path),'split_manifest_sha256':split['manifest_sha256'],
            'sequence_results':results,'model_trained':False,'test_set_accessed':False,'P5_training':False,
            'synthesis_causal':False,'explanation':'Offline reference-derived FIR-center-aligned and centered-difference synthetic inputs; not online measurement processing',
            'code_hashes':{name:sha256(Path(__file__).with_name(name)) for name in ['synthetic_imu.py','synthesize_amass_pilot.py','freeze_amass_split.py']},
            'limitations':['controlled beta0 male morphology, not gender-matched subject reconstruction',
                           'skin deformation and real placement differ','orientation FIR projection not strictly bandlimited',
                           'transition candidates geometry-verified only; no human action annotation',
                           'synthetic acceleration bandwidth differs from real IMU','six sequences are pipeline checks, not sufficient training dataset']}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('Pilot completed',flush=True)


if __name__=='__main__':main()
