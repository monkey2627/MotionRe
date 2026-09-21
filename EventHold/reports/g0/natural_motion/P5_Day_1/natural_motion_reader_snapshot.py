"""Stream raw MVNX archives for offline auditing, without GT-based calibration.

Sensor measurements remain in native export conventions. This is not yet an
EventHold input adapter or a conversion from the Xsens skeleton to SMPL.
"""
import argparse
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .audit import write_csv
from .holds import intervals

TARGET = ['Pelvis','LeftForeArm','RightForeArm','LeftLowerLeg','RightLowerLeg']


def tag(element):
    return element.tag.rsplit('}',1)[-1]


def read_stream(stream):
    """Named fields, native sampling, no interpolation, and bounded XML memory."""
    meta={'sensor_names':[], 'segment_names':[], 'calibration_frame_types':[]}
    arrays={k:[] for k in ['time_ms','frame_index','sensor_orientation_wxyz',
                          'sensor_free_acceleration','segment_orientation_wxyz','segment_position']}
    stack=[]
    for event,e in ET.iterparse(stream,events=('start','end')):
        if event=='start':
            stack.append(e)
            if tag(e)=='subject':
                meta['frame_rate']=float(e.attrib['frameRate'])
                meta['configuration']=e.get('configuration')
            continue
        kind=tag(e)
        if kind in ('sensors','segments'):
            names=[c.attrib['label'] for c in e]
            if len(names)!=len(set(names)):
                raise ValueError('Duplicate exported names')
            meta['sensor_names' if kind=='sensors' else 'segment_names']=names
        if kind=='frame':
            if e.get('type')!='normal':
                meta['calibration_frame_types'].append(e.get('type'))
            else:
                ns,nj=len(meta['sensor_names']),len(meta['segment_names'])
                ids=[meta['sensor_names'].index(n) for n in TARGET]
                fields={tag(c):c.text for c in e}
                def parse(name,n,width):
                    value=np.fromstring(fields[name],sep=' ',dtype=np.float32)
                    if value.size!=n*width:
                        raise ValueError('Wrong '+name+' field length')
                    return value.reshape(n,width)
                arrays['time_ms'].append(float(e.attrib['time']))
                arrays['frame_index'].append(int(e.attrib['index']))
                arrays['sensor_orientation_wxyz'].append(parse('sensorOrientation',ns,4)[ids])
                arrays['sensor_free_acceleration'].append(parse('sensorFreeAcceleration',ns,3)[ids])
                arrays['segment_orientation_wxyz'].append(parse('orientation',nj,4))
                arrays['segment_position'].append(parse('position',nj,3))
            if len(stack)>1:
                stack[-2].remove(e)
            e.clear()
        stack.pop()
    if not arrays['time_ms']:
        raise ValueError('No normal frames')
    meta['selected_sensor_names']=TARGET
    meta['selected_sensor_indices']=[meta['sensor_names'].index(n) for n in TARGET]
    return {k:np.asarray(v) for k,v in arrays.items()},meta


def posture_masks(position,names):
    """Geometric proxies in native Z-up export; not verified action semantics."""
    ids={n:names.index(n) for n in ['Pelvis','T8','LeftUpperLeg','LeftLowerLeg','LeftFoot',
                                  'RightUpperLeg','RightLowerLeg','RightFoot']}
    p=np.asarray(position)
    def tilt(v):
        return np.degrees(np.arccos(np.clip(np.abs(v[...,2])/np.maximum(np.linalg.norm(v,axis=-1),1e-8),0,1)))
    bends=[];thighs=[];shanks=[];valid=np.isfinite(p).all(axis=(1,2))
    for side in ['Left','Right']:
        thigh=p[:,ids[side+'LowerLeg']]-p[:,ids[side+'UpperLeg']]
        shank=p[:,ids[side+'Foot']]-p[:,ids[side+'LowerLeg']]
        nt=np.linalg.norm(thigh,axis=-1);ns=np.linalg.norm(shank,axis=-1)
        valid &= (nt>0.1)&(ns>0.1)
        bends.append(np.degrees(np.arccos(np.clip(np.sum(thigh*shank,axis=-1)/np.maximum(nt*ns,1e-8),-1,1))))
        thighs.append(tilt(thigh));shanks.append(tilt(shank))
    bend=np.stack(bends,axis=1);thigh=np.stack(thighs,axis=1);shank=np.stack(shanks,axis=1)
    trunk=p[:,ids['T8']]-p[:,ids['Pelvis']]
    valid &= (np.linalg.norm(trunk,axis=-1)>0.1)&(trunk[:,2]>0)
    trunk_tilt=tilt(trunk)
    seated=valid & ((bend>=45)&(bend<=130)&(thigh>=45)&(shank<=35)).all(axis=1)&(trunk_tilt<=40)
    standing=valid & (bend<20).all(axis=1)&(thigh<25).all(axis=1)&(trunk_tilt<30)
    return seated,standing,{'knee_bend_deg':bend,'thigh_tilt_deg':thigh,'shank_tilt_deg':shank,'trunk_tilt_deg':trunk_tilt}


def audit_arrays(a,meta):
    fps=meta['frame_rate'];n=len(a['time_ms'])
    if fps<=0 or n<2:
        raise ValueError('Need valid fps and multiple frames')
    timestamp_delta=np.diff(a['time_ms'])
    index_delta=np.diff(a['frame_index'])
    continuous=np.r_[False,(index_delta==1)&(timestamp_delta>0)&(timestamp_delta<2*1000/fps)]
    qs=a['sensor_orientation_wxyz'];qg=a['segment_orientation_wxyz']
    valid_sensor=np.isfinite(qs).all(axis=(1,2)) & np.isfinite(a['sensor_free_acceleration']).all(axis=(1,2)) & (np.abs(np.linalg.norm(qs,axis=-1)-1)<0.01).all(axis=1)
    valid_ref=np.isfinite(qg).all(axis=(1,2)) & (np.abs(np.linalg.norm(qg,axis=-1)-1)<0.01).all(axis=1)
    safe=np.where(valid_ref[:,None,None],qg,np.array([1,0,0,0]))
    rot=Rotation.from_quat(safe[..., [1,2,3,0]].reshape(-1,4)).as_matrix().reshape(n,-1,3,3)
    names=meta['segment_names'];parents=['Pelvis','LeftUpperLeg','Pelvis','RightUpperLeg']
    children=['LeftUpperLeg','LeftLowerLeg','RightUpperLeg','RightLowerLeg']
    local=np.stack([rot[:,names.index(p)].transpose(0,2,1)@rot[:,names.index(c)] for p,c in zip(parents,children)],axis=1)
    delta=local[:-1].swapaxes(-1,-2)@local[1:]
    speed=np.full((n,4),np.inf)
    speed[1:]=Rotation.from_matrix(delta.reshape(-1,3,3)).magnitude().reshape(n-1,4)*fps*180/np.pi
    seated,standing,geometry=posture_masks(a['segment_position'],names)
    valid=valid_sensor & valid_ref & continuous
    seated &= valid;standing &= valid
    strict=(speed.max(axis=1)<5)&valid&np.r_[False,valid_ref[:-1]]
    stand_runs=[(s,e) for s,e in intervals(standing) if (e-s)/fps>=1]
    rows=[]
    for kind,mask,minimum in [('seated_like',seated,20),('strict_low_angular_speed',strict,5),('seated_like_and_strict',seated&strict,5)]:
        for s,e in intervals(mask):
            if (e-s)/fps<minimum:continue
            before=[r for r in stand_runs if r[1]<=s and s-r[1]<=15*fps]
            after=[r for r in stand_runs if r[0]>=e and r[0]-e<=15*fps]
            rows.append({'kind':kind,'start_frame':int(s),'end_frame_exclusive':int(e),
                         'start_seconds':float(a['time_ms'][s]/1000),
                         'duration_seconds':float((e-s)/fps),
                         'median_knee_bend_deg':float(np.median(geometry['knee_bend_deg'][s:e])),
                         'max_local_joint_speed_deg_s':float(np.max(speed[s:e])),
                         'standing_before_within_15s':bool(before),'standing_after_within_15s':bool(after),
                         'complete_stand_seated_stand_proxy':bool(before and after) if kind=='seated_like' else False,
                         'annotation_status':'geometry_candidate_not_manually_verified'})
    stats={**meta,'normal_frames':n,'duration_seconds':float((a['time_ms'][-1]-a['time_ms'][0])/1000+1/fps),
           'timestamp_delta_ms_min_max':[float(timestamp_delta.min()),float(timestamp_delta.max())],
           'nonunit_frame_index_steps':int(np.count_nonzero(index_delta!=1)),
           'invalid_sensor_frames':int(np.count_nonzero(~valid_sensor)),
           'invalid_reference_quaternion_frames':int(np.count_nonzero(~valid_ref)),
           'candidate_counts':{kind:sum(r['kind']==kind for r in rows) for kind in ['seated_like','strict_low_angular_speed','seated_like_and_strict']},
           'max_seated_like_seconds':max((r['duration_seconds'] for r in rows if r['kind']=='seated_like'),default=0),
           'complete_stand_seated_stand_proxies':sum(r['complete_stand_seated_stand_proxy'] for r in rows),
           'sensor_calibration_from_reference_performed':False,
           'eligible_for_model_evaluation':False,
           'reference_type':'Xsens_fullbody_inertial_solution_not_independent_optical_ground_truth',
           'open_contract_items':['sensor_free_acceleration_frame','sensor_to_bone_installation_calibration_without_reference_leakage','Xsens_to_SMPL_mapping','reference_drift_manual_review','training_overlap_and_subject_split']}
    return stats,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--member',default=None)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);args.cache.mkdir(parents=True,exist_ok=True)
    summaries=[];candidates=[]
    with zipfile.ZipFile(args.archive) as z:
        members=[x for x in z.namelist() if x.endswith('.mvnx') and (args.member is None or x==args.member)]
        if not members:raise ValueError('No matching MVNX member')
        for member in members:
            print('Reading '+member,flush=True)
            stem=Path(member).stem
            with z.open(member) as f:a,meta=read_stream(f)
            # Save raw fields separately from any converted model input.
            np.savez_compressed(args.cache/(stem+'.npz'),**a)
            (args.cache/(stem+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
            stats,rows=audit_arrays(a,meta)
            stats.update(archive=str(args.archive),member=member,zip_crc32=z.getinfo(member).CRC)
            summaries.append(stats)
            candidates.extend({'member':member,**r} for r in rows)
            (args.output/(stem+'_audit.json')).write_text(json.dumps(stats,indent=2)+'\n')
            print(json.dumps({'member':member,'frames':stats['normal_frames'],'candidate_counts':stats['candidate_counts'],'max_seated_seconds':stats['max_seated_like_seconds']}),flush=True)
    if candidates:write_csv(args.output/'candidates.csv',candidates,list(candidates[0]))
    summary={'status':'offline_data_audit_not_model_result','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'subject_role':'exploration_only_not_locked_test','sequences':summaries}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
