"""Train/smoke a causal B0 from original five-sensor DIP, with locked test set.

This entry is a pipeline pilot, not a strong-baseline reproduction. It uses
rotation supervision only; AMASS pretraining, FK loss, and final G1 comparisons
are deliberately not claimed. A capped run cannot establish the research gap.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from .adapters import eventhold_features
from .audit import sha256
from .baseline import CausalBaseline,geodesic
from .dip import load_original,measurements,split_for_subject,DIP_HZ

BODY=[i for i in range(1,22) if i not in [12,15]]


def load_split(root,split,max_per_subject,max_frames):
    records=[]
    for subject in sorted(root.glob('s_*')):
        if split_for_subject(subject.name)!=split:continue
        files=sorted(subject.glob('*.pkl'))
        if max_per_subject:files=files[:max_per_subject]
        for path in files:
            d=load_original(path)
            if max_frames:d={k:v[:max_frames] if hasattr(v,'shape') and len(v.shape)>0 else v for k,v in d.items()}
            rec=measurements(d,path)
            pose=np.asarray(d['gt']).reshape(-1,24,3)
            label_valid=np.isfinite(pose).all(axis=(1,2))
            safe=np.where(np.isfinite(pose),pose,0)
            target=Rotation.from_rotvec(safe.reshape(-1,3)).as_matrix().reshape(-1,24,3,3).astype(np.float32)
            records.append({'source':str(path),'subject':subject.name,'sha256':sha256(path),
                            'features':torch.from_numpy(eventhold_features(rec)).float(),
                            'target':torch.from_numpy(target),
                            'valid':torch.from_numpy(label_valid & rec.valid.all(axis=1))})
    if not records:raise RuntimeError('No records in '+split)
    return records


def pass_sequence(model,record,device,chunk,opt=None):
    state=None;err_sum=0.;count=0;outputs=[]
    for start in range(0,len(record['features']),chunk):
        x=record['features'][start:start+chunk].to(device)[None]
        y=record['target'][start:start+chunk].to(device)[None]
        mask=record['valid'][start:start+chunk].to(device)
        if opt is not None:opt.zero_grad(set_to_none=True)
        pred,state=model(x,state)
        error=geodesic(pred[0,:,BODY],y[0,:,BODY]).mean(-1)
        if mask.any():
            loss=error[mask].mean()
            if opt is not None:
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            err_sum+=float(error[mask].detach().sum());count+=int(mask.sum())
        state=state.detach() # retain values across TBPTT boundaries
        if opt is None:outputs.append(pred[0].cpu())
    return err_sum/max(count,1)*180/np.pi,count,torch.cat(outputs) if outputs else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'base_mobileposer/data/raw/DIP_IMU')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--max-per-subject',type=int,default=1)
    p.add_argument('--max-frames',type=int,default=3600)
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--chunk',type=int,default=240)
    p.add_argument('--device',default='cpu')
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    if args.epochs<1 or args.chunk<1:raise ValueError('Positive epochs/chunk required')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use new output directory to preserve previous run')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    start=time.time()
    train=load_split(args.root,'train',args.max_per_subject,args.max_frames)
    val=load_split(args.root,'validation',args.max_per_subject,args.max_frames)
    model=CausalBaseline(args.hidden).to(args.device)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    manifest={'status':'pipeline_pilot_not_research_result','input_hz':DIP_HZ,'future_frames':0,
              'root_from_measured_pelvis':True,'gt_initial_pose':False,'root_translation_estimated':False,
              'event_memory_enabled':False,'training_loss':'body_geodesic_only',
              'AMASS_pretraining':False,'sampling_note':'retain original 60Hz, no 30Hz legacy cache',
              'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'sources':{s:[{k:r[k] for k in ['source','subject','sha256']} for r in data]
                         for s,data in [('train',train),('validation',val)]},
              'code_hashes':{f.name:sha256(f) for f in Path(__file__).parent.glob('*.py')},
              'torch':torch.__version__,'test_accessed':False}
    (args.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    history=[]
    for epoch in range(args.epochs):
        model.train();order=list(range(len(train)));random.shuffle(order)
        training=[pass_sequence(model,train[i],args.device,args.chunk,opt)[0] for i in order]
        model.eval();validation=[]
        with torch.no_grad():
            for record in val:
                e,n,_=pass_sequence(model,record,args.device,args.chunk)
                validation.append({'source':record['source'],'subject':record['subject'],'mean_angle_deg':e,'valid_frames':n})
        line={'epoch':epoch+1,'train_sequence_mean_deg':float(np.mean(training)),
              'validation_sequence_mean_deg':float(np.mean([x['mean_angle_deg'] for x in validation])),
              'validation':validation,'elapsed_seconds':time.time()-start}
        history.append(line);print(json.dumps(line),flush=True)
        (args.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        torch.save({'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'manifest':manifest},args.output/'last.pt')
    with torch.no_grad():
        for i,record in enumerate(val):
            _,_,pred=pass_sequence(model,record,args.device,args.chunk)
            np.savez_compressed(args.output/('validation_%02d.npz'%i),prediction=pred.numpy(),
                                valid=record['valid'].numpy(),source=record['source'],fps=DIP_HZ)
    print('Pipeline pilot completed. Not a trained strong baseline or evidence of forgetting.',flush=True)


if __name__=='__main__':main()
