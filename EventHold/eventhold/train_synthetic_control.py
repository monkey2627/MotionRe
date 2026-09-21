"""Tiny matched synthetic controls for initial-state evidence.

Uniform and candidate-weighted modes share architecture, files, epochs and
optimizer. Candidate weighting is a training ablation, not an event-memory
method. Holdout files remain outside optimization.
"""
import argparse,csv,json,random
from pathlib import Path
import numpy as np, torch
from .adapters import eventhold_features
from .baseline import CausalBaseline, geodesic
from .region_event_model import RegionEventWrite
from .records import ImuRecord,FIVE
from .audit import sha256

BODY=[i for i in range(1,22) if i not in [12,15]]

def load(path):
 d=np.load(path,allow_pickle=False);valid=np.asarray(d['valid'],bool)[:,None].repeat(5,1)
 r=ImuRecord(np.arange(len(valid),dtype=float)/60.,FIVE,d['orientation'].astype(np.float32),d['acceleration'].astype(np.float32),valid,str(path)).validate()
 return {'x':torch.from_numpy(eventhold_features(r)).float(),'y':torch.from_numpy(d['target'].astype(np.float32)),'valid':torch.from_numpy(valid.all(1))}

def pass_seq(model,item,weight,chunk,opt=None):
 state=None;total=0.;den=0.;
 for start in range(0,len(item['x']),chunk):
  x=item['x'][start:start+chunk][None];y=item['y'][start:start+chunk][None];m=item['valid'][start:start+chunk]
  pred,state=model(x,state);e=geodesic(pred[0,:,BODY],y[0,:,BODY]).mean(-1)
  w=weight[start:start+chunk];mask=m & (w>0)
  if mask.any():
   loss=(e[mask]*w[mask]).sum()/w[mask].sum()
   if opt is not None:opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
   total+=float(e[mask].detach().sum());den+=int(mask.sum())
  state=state.detach()
 return total/max(den,1)*180/np.pi

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--candidate-csv',type=Path,required=True);p.add_argument('--mode',choices=['uniform','candidate_weighted'],required=True);p.add_argument('--model',choices=['b0','m1'],default='b0');p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=6);p.add_argument('--hidden',type=int,default=96);p.add_argument('--seed',type=int,default=0)
 a=p.parse_args();
 if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Use new output')
 torch.set_num_threads(2);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 split=json.loads((a.split/'split.json').read_text());assign={x['id']:x['split'] for x in split['rows']}
 items={};
 for name,part in assign.items():items[name]=load(a.data/name)
 with a.candidate_csv.open(encoding='utf-8-sig') as f: candidates=list(csv.DictReader(f))
 by_name={}
 for row in split['rows']:
  by_name.setdefault(row['id'],[])
  for c in candidates:
   if c['source']==row['source'] and c['kind']=='seated_like':
    # candidate rows were measured at source fps; generated pilot uses 60 Hz.
    with np.load(a.data/row['id'],allow_pickle=False) as d:native=len(d['source_indices'])
    source_fps=native/((native-1)/60.) if native>1 else 60.
    s=int(round(float(c['start_seconds'])*60));e=min(native,int(round(float(c['start_seconds'])+float(c['duration_seconds']))*60))
    by_name[row['id']].append((max(0,s),e))
 train=[k for k,v in assign.items() if v=='train_exploration'];hold=[k for k,v in assign.items() if v=='holdout_exploration']
 model=(CausalBaseline(a.hidden) if a.model=='b0' else RegionEventWrite(a.hidden));opt=torch.optim.Adam(model.parameters(),lr=3e-4);history=[]
 for ep in range(a.epochs):
  order=train[:];random.shuffle(order);train_err=[]
  for name in order:
   n=len(items[name]['x']);w=np.ones(n,np.float32)
   if a.mode=='candidate_weighted':
    for s,e in by_name.get(name,[]):w[s:e]=3.
   train_err.append(pass_seq(model,items[name],torch.from_numpy(w),240,opt))
  val=[]
  with torch.no_grad():
   for name in hold:
    n=len(items[name]['x']);val.append({'file':name,'mean_deg':pass_seq(model,items[name],torch.ones(n),240)})
  history.append({'epoch':ep+1,'train_mean_deg':float(np.mean(train_err)),'holdout_mean_deg':float(np.mean([x['mean_deg'] for x in val])),'holdout':val})
  print(json.dumps(history[-1]),flush=True)
 a.output.mkdir(parents=True);manifest={'status':'matched_synthetic_control_not_real_performance','model':a.model,'mode':a.mode,'seed':a.seed,'epochs':a.epochs,'hidden':a.hidden,'train_files':train,'holdout_files':hold,'split_sha256':sha256(a.split/'split.json'),'code_sha256':sha256(Path(__file__)),'history':history,'candidate_weighting':a.mode=='candidate_weighted','head_input':False,'test_accessed':False}
 (a.output/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');torch.save({'state_dict':model.state_dict(),'manifest':manifest},a.output/'last.pt')
if __name__=='__main__':main()
