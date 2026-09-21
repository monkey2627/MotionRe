"""Read-only raw AMASS coverage screen, not action labels or training admission.

Sample at approximately 10 Hz to bound work; all candidates require native-rate
confirmation. Independently implemented SMPL joint FK, checked against local
ParametricModel on a synthetic pose fixture. No old IMU cache is consumed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from .audit import sha256, write_csv
from .holds import intervals
from .natural_motion import posture_masks

ALIGN=np.array([[1.,0,0],[0,0,1],[0,-1,0]])


def fk(local, rest, parents):
    global_r=np.empty_like(local);position=np.zeros(local.shape[:2]+(3,))
    global_r[:,0]=local[:,0]
    for i,p in enumerate(parents[1:],1):
        global_r[:,i]=global_r[:,p]@local[:,i]
        position[:,i]=position[:,p]+np.einsum('tij,j->ti',global_r[:,p],rest[i]-rest[p])
    return global_r,position


def screen(pose, beta, fps, j0, jshape, parents, screen_hz=10):
    step=max(1,int(round(fps/screen_hz)));ids=np.arange(0,len(pose),step);hz=fps/step
    body=np.asarray(pose[ids,:72]).reshape(-1,24,3)
    if not np.isfinite(body).all():raise ValueError('Nonfinite sampled poses')
    r=Rotation.from_rotvec(body.reshape(-1,3)).as_matrix().reshape(-1,24,3,3)
    r[:,0]=ALIGN@r[:,0]
    rest=j0+np.einsum('ijk,k->ij',jshape,beta[:10])
    _,position=fk(r,rest,parents)
    # Reuse the geometry definition only, via explicit named joint positions.
    # SMPL spine3 is a proxy for Xsens T8, with differing anatomical definitions.
    names=['Pelvis','T8','LeftUpperLeg','LeftLowerLeg','LeftFoot',
           'RightUpperLeg','RightLowerLeg','RightFoot']
    joints=[0,9,1,4,7,2,5,8]
    zup=position[:,joints][...,[0,2,1]]
    seated,standing,_=posture_masks(zup,names)
    lower=r[:,[1,2,4,5]]
    delta=lower[:-1].swapaxes(-1,-2)@lower[1:]
    speed=np.full(len(r),np.inf)
    if len(r)>1:
        speed[1:]=np.max(Rotation.from_matrix(delta.reshape(-1,3,3)).magnitude().reshape(-1,4),axis=1)*hz*180/np.pi
    stands=[(s,e) for s,e in intervals(standing) if (e-s)/hz>=1]
    candidates=[]
    for kind,mask in [('seated_like',seated),('seated_like_low_speed',seated&(speed<5))]:
        for s,e in intervals(mask):
            if (e-s)/hz<5:continue
            before=any(end<=s and s-end<=15*hz for start,end in stands)
            after=any(start>=e and start-e<=15*hz for start,end in stands)
            candidates.append({'kind':kind,'start_frame':int(ids[s]),
                               'end_frame_exclusive':int(min(len(pose),ids[e-1]+step)),
                               'start_seconds':float(ids[s]/fps),'duration_seconds':float((e-s)/hz),
                               'standing_before_15s':before,'standing_after_15s':after,
                               'complete_transition_proxy':before and after,
                               'status':'sampled_geometry_only_native_confirmation_required'})
    return candidates,hz


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mobileposer-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use new output')
    args.output.mkdir(parents=True)
    sys.path.insert(0,str(args.mobileposer_root))
    import torch
    from mobileposer.articulate.model import ParametricModel
    torch.set_num_threads(2)
    model_file=args.mobileposer_root/'mobileposer/smpl/basicmodel_m.pkl'
    model=ParametricModel(str(model_file))
    j0=(model._J_regressor@model._v_template).numpy()
    jshape=np.einsum('jv,vck->jck',model._J_regressor.numpy(),model._shapedirs.numpy())
    parents=model.parent
    beta=np.linspace(-.2,.2,10)
    fixture=Rotation.random(4*24,random_state=1).as_matrix().reshape(4,24,3,3)
    g,pos=fk(fixture,j0+np.einsum('ijk,k->ij',jshape,beta),parents)
    with torch.no_grad():
        ng,np_=model.forward_kinematics(torch.tensor(fixture,dtype=torch.float32),torch.tensor(beta,dtype=torch.float32),calc_mesh=False)
    parity={'global_rotation_max_abs':float(np.max(np.abs(g-ng.numpy()))),
            'joint_position_max_abs_m':float(np.max(np.abs(pos-np_.numpy())))}
    if max(parity.values())>1e-5:raise ValueError('SMPL FK fixture mismatch')
    files=sorted(args.root.rglob('*_poses.npz'));manifest=[];candidates=[];errors=[]
    for i,path in enumerate(files):
        rel=path.relative_to(args.root);base={'source':str(path),'dataset':rel.parts[0],
                                           'subject_key':'/'.join(rel.parts[:-1]),'bytes':path.stat().st_size}
        try:
            with np.load(path,allow_pickle=False) as data:
                fps=float(data['mocap_framerate']);pose=data['poses'];beta=np.asarray(data['betas']).ravel()
                if not np.isfinite(fps) or fps<=0 or pose.ndim!=2 or pose.shape[1]<72 or len(beta)<10 or not np.isfinite(beta[:10]).all():
                    raise ValueError('Unsupported metadata/pose shape')
                rows,hz=screen(pose,beta,fps,j0,jshape,parents)
                entry={**base,'fps':fps,'frames':len(pose),'duration_seconds':len(pose)/fps,
                       'pose_dimensions':pose.shape[1],'screen_hz':hz,
                       'max_seated_seconds':max((r['duration_seconds'] for r in rows if r['kind']=='seated_like'),default=0),
                       'max_low_speed_seated_seconds':max((r['duration_seconds'] for r in rows if r['kind']=='seated_like_low_speed'),default=0),
                       'complete_transition_proxies':sum(r['complete_transition_proxy'] for r in rows if r['kind']=='seated_like')}
                manifest.append(entry);candidates.extend({**base,**r} for r in rows)
        except Exception as error:errors.append({**base,'error':type(error).__name__+': '+str(error)})
        if (i+1)%500==0:print(json.dumps({'scanned':i+1,'total':len(files),'candidates':len(candidates),'errors':len(errors)}),flush=True)
    datasets={}
    for name in sorted(set(x['dataset'] for x in manifest)):
        m=[x for x in manifest if x['dataset']==name];c=[x for x in candidates if x['dataset']==name]
        datasets[name]={'sequences':len(m),'subject_keys':len(set(x['subject_key'] for x in m)),
                        'hours':sum(x['duration_seconds'] for x in m)/3600,
                        'fps_counts':dict(Counter(x['fps'] for x in m)),
                        'seated_ge20s':sum(x['kind']=='seated_like' and x['duration_seconds']>=20 for x in c),
                        'low_speed_seated_ge20s':sum(x['kind']=='seated_like_low_speed' and x['duration_seconds']>=20 for x in c),
                        'complete_transition_proxies_ge5s':sum(x['kind']=='seated_like' and x['complete_transition_proxy'] for x in c)}
    summary={'status':'sampled_raw_AMASS_exploration_not_training_admission','files_discovered':len(files),
             'files_scanned':len(manifest),'errors':errors,'datasets':datasets,'fk_fixture_parity':parity,
             'sensor_layout_generated':False,'smpl_file':str(model_file),'smpl_sha256':sha256(model_file),
             'code_sha256':sha256(Path(__file__)),'sampling':'approximately10Hz, original rate retained, no 59Hz-to60Hz relabel',
             'limitations':['sampled runs may hide brief transitions; native-rate confirmation required',
                            'geometry is not verified action labels','subject keys do not guarantee cross-dataset identity independence',
                            'no validation/test split chosen; sources now explored','SMPL model male only; model/AMASS shape provenance not independently certified']}
    write_csv(args.output/'sequence_manifest.csv',manifest,list(manifest[0]))
    if candidates:write_csv(args.output/'candidates.csv',candidates,list(candidates[0]))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
