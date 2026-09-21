"""Native six-IMU PNP forward parity in the isolated compatible environment.

This verifies conversion only. It does not certify a five-IMU adaptation,
upstream provenance, general accuracy, or initialization equivalence with the
published ground-truth-initialized test protocol.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .adapters import pnp_from_native
from .audit import sha256
from .verify_native import native_pnp_reference


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,default=Path('reports/pnp_forward_parity.json'))
    p.add_argument('--frames',type=int,default=60)
    args=p.parse_args();root=args.root.resolve();output=args.output.resolve()
    native=root/'PNP'
    sys.path.insert(0,str(native));os.chdir(str(native))
    from net import PNP
    torch.set_num_threads(2)
    data=torch.load(native/'data/test_datasets/totalcapture_officalib.pt',map_location='cpu')
    report={'scope':'native_six_IMU_input_adapter_forward_parity','five_imu_benchmark_eligible':False,
            'initialization':'default_T_pose_both_paths_no_test_pose_input','frames_per_sequence':args.frames,
            'sequences':[0,2],'tolerance_rotation_matrix_absolute':1e-3,'tolerance_translation_m':1e-3,
            'cpu':True,'torch':torch.__version__,'numpy':np.__version__,'results':[],
            'source_hashes':{str(f):sha256(f) for f in [native/'test.py',native/'net.py',native/'dynamics.py',
                                                     native/'data/weights/PNP/weights.pt']}}
    model=PNP().eval()
    for index in report['sequences']:
        v={k:data[k][index][:args.frames].clone() if k in ['aS','wS','RIS'] else data[k][index].clone()
           for k in ['aS','wS','RIS','RIM','RSB']}
        v['g']=torch.tensor([0.,-9.8,0.])
        expected=native_pnp_reference(native/'test.py',v)
        actual=pnp_from_native(*(v[k].numpy() for k in ['aS','wS','RIS','RIM','RSB','g']))
        trajectories=[]
        for path,inputs in [('native',expected),('adapter',actual)]:
            model.rnn_initialize()
            pose,tran=[],[]
            for a,w,r in zip(*inputs):
                q,t=model.forward_frame(torch.from_numpy(a).float(),torch.from_numpy(w).float(),torch.from_numpy(r).float())
                pose.append(q.numpy());tran.append(t.numpy())
            trajectories.append((np.stack(pose),np.stack(tran)))
            print('%s %s completed %d frames'%(data['name'][index],path,len(pose)),flush=True)
        a,b=trajectories
        dr=float(np.max(np.abs(a[0]-b[0])));dt=float(np.max(np.abs(a[1]-b[1])))
        passed=bool(np.isfinite(dr) and np.isfinite(dt) and dr<=1e-3 and dt<=1e-3)
        report['results'].append({'sequence':str(data['name'][index]),'frames':len(a[0]),
                                  'max_rotation_matrix_abs_diff':dr,'max_translation_abs_diff_m':dt,'passed':passed})
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+'\n')
        if not passed:raise AssertionError('Native inference parity failed; see report')
    print('PNP native six-IMU parity passed; five-IMU adaptation still required.',flush=True)


if __name__=='__main__':main()
