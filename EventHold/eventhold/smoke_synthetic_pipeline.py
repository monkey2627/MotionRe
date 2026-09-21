"""Smoke-test the generated five-IMU archive through the causal B0 contract.

No optimization and no performance claim. This catches layout, mask, target
rotation and recurrent state errors before a training comparison.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from .adapters import eventhold_features
from .baseline import CausalBaseline
from .records import ImuRecord, FIVE
from .audit import sha256


def load_one(path):
    d=np.load(path,allow_pickle=False)
    valid=np.asarray(d['valid'],bool)[:,None].repeat(5,axis=1)
    record=ImuRecord(np.arange(len(valid),dtype=float)/60.,FIVE,
                     d['orientation'].astype(np.float32),d['acceleration'].astype(np.float32),valid,str(path)).validate()
    target=d['target'].astype(np.float32)
    return record,target


def infer(model,x,chunk):
    state=None;out=[]
    with torch.no_grad():
        for i in range(0,len(x),chunk):
            y,state=model(torch.from_numpy(x[i:i+chunk]).float()[None],state)
            out.append(y[0].numpy())
    return np.concatenate(out),state


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use new output')
    torch.set_num_threads(2);model=CausalBaseline(hidden=32).eval();rows=[]
    files=sorted(args.input.glob('seq_*.npz'))
    if not files:raise ValueError('No generated sequences')
    for path in files:
        record,target=load_one(path);features=eventhold_features(record)
        if features.shape!=(len(record.timestamps),66):raise ValueError('Unexpected feature shape')
        pred_a,_=infer(model,features,240);pred_b,_=infer(model,features,137)
        if not np.isfinite(features).all() or not np.isfinite(pred_a).all():raise ValueError('Nonfinite smoke output')
        parity=float(np.max(np.abs(pred_a-pred_b)))
        if parity>2e-5:raise ValueError('Continuous state parity failed')
        root_error=float(np.max(np.abs(pred_a[:,0]-target[:,0])))
        rows.append({'file':path.name,'frames':len(features),'valid_frames':int(record.valid[:,0].sum()),'feature_dim':features.shape[1],'chunk_parity_max_abs':parity,'root_copy_max_abs':root_error,'boundary_valid_first':bool(record.valid[0].any()),'boundary_valid_last':bool(record.valid[-1].any()),'target_shape':list(target.shape)})
    args.output.mkdir(parents=True);summary={'status':'synthetic_five_imu_b0_contract_smoke_only','files':len(rows),'rows':rows,'model':'random CausalBaseline hidden32 no optimization','head_input':False,'input_manifest_sha256':sha256(args.input/'manifest.json'),'code_sha256':sha256(Path(__file__)),'performance_claim':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
