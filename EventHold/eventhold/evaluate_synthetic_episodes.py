"""Evaluate initial-versus-late errors on held-out synthetic candidates."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from .adapters import eventhold_features
from .baseline import CausalBaseline,geodesic
from .region_event_model import RegionEventWrite
from .records import ImuRecord,FIVE
from .audit import sha256
BODY=[i for i in range(1,22) if i not in [12,15]]

def load(path):
 d=np.load(path,allow_pickle=False);v=np.asarray(d['valid'],bool)[:,None].repeat(5,1)
 r=ImuRecord(np.arange(len(v),dtype=float)/60,FIVE,d['orientation'].astype(np.float32),d['acceleration'].astype(np.float32),v,str(path)).validate()
 return eventhold_features(r),torch.from_numpy(d['target'].astype(np.float32)),torch.from_numpy(v.all(1))
def infer(model,x):
 state=None;out=[]
 with torch.no_grad():
  for i in range(0,len(x),240):y,state=model(torch.from_numpy(x[i:i+240]).float()[None],state);out.append(y[0])
 return torch.cat(out)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--split',type=Path,required=True);p.add_argument('--candidate-csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();
 if a.output.exists():raise FileExistsError('Use new output')
 m=json.loads((a.run/'run_manifest.json').read_text());ck=torch.load(a.run/'last.pt',map_location='cpu');model=(CausalBaseline(m['hidden']) if m['model']=='b0' else RegionEventWrite(m['hidden']));model.load_state_dict(ck['state_dict']);model.eval()
 split=json.loads((a.split/'split.json').read_text());rows=[]
 with a.candidate_csv.open(encoding='utf-8-sig') as f:cands=list(csv.DictReader(f))
 for entry in split['rows']:
  if entry['split']!='holdout_exploration':continue
  x,y,valid=load(a.data/entry['id']);pred=infer(model,x);e=geodesic(pred[:,BODY],y[:,BODY]).mean(-1).numpy()*180/np.pi
  for c in cands:
   if c['source']!=entry['source'] or c['kind']!='seated_like':continue
   s=max(0,int(round(float(c['start_seconds'])*60)));end=min(len(e),int(round((float(c['start_seconds'])+float(c['duration_seconds']))*60)))
   if end<=s:continue
   early=e[s:min(end,s+600)];late=e[max(s,end-600):end]
   rows.append({'file':entry['id'],'duration_seconds':float(c['duration_seconds']),'frames':end-s,'early10s_deg':float(np.mean(early)),'late10s_deg':float(np.mean(late)),'late_minus_early_deg':float(np.mean(late)-np.mean(early)),'full_deg':float(np.mean(e[s:end]))})
 a.output.mkdir(parents=True);summary={'status':'episode_initial_vs_late_synthetic_exploration','model':m['model'],'rows':rows,'mean_early_deg':float(np.mean([r['early10s_deg'] for r in rows])),'mean_late_deg':float(np.mean([r['late10s_deg'] for r in rows])),'mean_change_deg':float(np.mean([r['late_minus_early_deg'] for r in rows])),'run_sha256':sha256(a.run/'run_manifest.json'),'test_accessed':False}
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
