"""Compare independent conversions with actual local native source equations.

Does not assert network/physics parity or five-sensor support. PNP reference
assignments are extracted unchanged from test.py; DynaIP's normalize_imu is
executed unchanged from utils/data.py. Source SHA256 binds this report.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch

from .adapters import pnp_from_native, dynaip_features
from .audit import sha256
from .dip import load_original, measurements
from .records import DYNA_SIX


def native_pnp_reference(source, values):
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compare_realimu")
    wanted = {'RMB', 'aM', 'wM'}
    assignments = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in wanted]
    if len(assignments) != 3:
        raise RuntimeError("Native PNP source structure changed; re-audit required")
    module = ast.fix_missing_locations(ast.Module(body=sorted(assignments, key=lambda n:n.lineno), type_ignores=[]))
    scope = dict(values, torch=torch, device=torch.device('cpu'))
    exec(compile(module, str(source), 'exec'), scope)
    return [scope[name].numpy() for name in ['aM','wM','RMB']]


def native_dyna_reference(source, acc, ori):
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'normalize_imu')
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    scope = {'torch':torch}
    exec(compile(module, str(source), 'exec'), scope)
    return scope['normalize_imu'](torch.from_numpy(acc), torch.from_numpy(ori)).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,default=Path('reports/native_preprocess_parity.json'))
    args=p.parse_args(); root=args.root.resolve()
    report={'scope':'preprocessing_only_not_end_to_end_inference','checks':[],'benchmark_eligible':False}
    source=root/'PNP/test.py'
    for filename in ['totalcapture_officalib.pt','totalcapture_dipcalib.pt']:
        file=root/'PNP/data/test_datasets'/filename
        data=torch.load(file,map_location='cpu')
        for i in range(min(3,len(data['pose']))):
            v={k:data[k][i][:120].clone() if k in ['aS','wS','RIS'] else data[k][i].clone()
               for k in ['aS','wS','RIS','RIM','RSB']}
            v['g']=torch.tensor([0.,-9.8,0.])
            expected=native_pnp_reference(source,v)
            actual=pnp_from_native(*(v[k].numpy() for k in ['aS','wS','RIS','RIM','RSB','g']))
            for name,a,b in zip(['world_linear_a','world_angular_w','bone_R'],actual,expected):
                np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-4)
                report['checks'].append({'method':'PNP','record':str(file),'sequence':str(data['name'][i]),
                                          'feature':name,'max_abs_difference':float(np.max(abs(a-b))),'passed':True})
    dyna_source=root/'NoUse/DynaIP/utils/data.py'
    dipfile=root/'base_mobileposer/data/raw/DIP_IMU/s_01/01.pkl'
    rec=measurements(load_original(dipfile),dipfile,DYNA_SIX)
    # Native normalizer is defined on finite samples. Select a contiguous valid
    # block; do not pretend filled observations were part of a native stream.
    valid=rec.valid.all(axis=1)
    from .holds import intervals
    start,end=next((s,e) for s,e in intervals(valid) if e-s>=120)
    from .records import ImuRecord
    sample=ImuRecord(rec.timestamps[start:start+120],rec.names,rec.orientation[start:start+120],
                     rec.acceleration[start:start+120],rec.valid[start:start+120],rec.source)
    actual=dynaip_features(sample)
    expected=native_dyna_reference(dyna_source,sample.acceleration,sample.orientation)
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-4)
    report['checks'].append({'method':'DynaIP','record':str(dipfile),'start_frame':int(start),
                            'feature':'native_normalize_imu_Tx6x12','passed':True,
                            'max_abs_difference':float(np.max(abs(actual-expected)))})
    report['source_hashes']={str(s):sha256(s) for s in [source,dyna_source]}
    report['remaining']=['upstream_version_audit','checkpoint_forward_parity',
                         'PNP_rbdl_dependency','DynaIP_weights_and_body_model',
                         'five_sensor_architecture_and_retraining','deployment_initialization']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Passed %d native preprocessing comparisons; no model performance claimed.'%len(report['checks']))


if __name__=='__main__':
    main()
