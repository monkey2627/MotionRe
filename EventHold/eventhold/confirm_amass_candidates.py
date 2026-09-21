"""Native-rate confirmation of files nominated by the coarse AMASS screen.

Negative files are not rechecked; this cannot estimate screening sensitivity.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from .audit import sha256, write_csv
from .audit_amass_coverage import screen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen-dir',type=Path,required=True)
    p.add_argument('--mobileposer-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use new output')
    sys.path.insert(0,str(args.mobileposer_root))
    import torch
    from mobileposer.articulate.model import ParametricModel
    torch.set_num_threads(2)
    model=ParametricModel(str(args.mobileposer_root/'mobileposer/smpl/basicmodel_m.pkl'))
    j0=(model._J_regressor@model._v_template).numpy()
    jshape=np.einsum('jv,vck->jck',model._J_regressor.numpy(),model._shapedirs.numpy())
    with (args.screen_dir/'candidates.csv').open(encoding='utf-8-sig') as f:coarse=list(csv.DictReader(f))
    sources={x['source']:{k:x[k] for k in ['source','dataset','subject_key']} for x in coarse}
    rows=[];manifest=[]
    for source,info in sorted(sources.items()):
        with np.load(source,allow_pickle=False) as data:
            fps=float(data['mocap_framerate']);pose=data['poses'];beta=np.asarray(data['betas']).ravel()
            candidates,hz=screen(pose,beta,fps,j0,jshape,model.parent,screen_hz=fps)
            entry={**info,'sha256':sha256(Path(source)),'fps':fps,'frames':len(pose),
                   'confirmed_candidates':len(candidates)}
            manifest.append(entry)
            for row in candidates:
                row['status']='native_rate_geometry_confirmed_not_action_annotation'
                rows.append({**info,**row})
        print(json.dumps({'confirmed_files':len(manifest),'total_files':len(sources),'candidates':len(rows)}),flush=True)
    summary={'status':'native_rate_confirmation_of_coarse_positive_files_only',
             'files':len(manifest),'subject_keys':len(set(x['subject_key'] for x in manifest)),
             'counts':{kind:{'ge5s':sum(x['kind']==kind for x in rows),
                            'ge20s':sum(x['kind']==kind and x['duration_seconds']>=20 for x in rows),
                            'max_seconds':max((x['duration_seconds'] for x in rows if x['kind']==kind),default=0),
                            'complete_transition_proxies':sum(x['kind']==kind and x['complete_transition_proxy'] for x in rows)}
                       for kind in ['seated_like','seated_like_low_speed']},
             'files_manifest':manifest,'code_sha256':sha256(Path(__file__)),
             'screen_code_sha256':sha256(Path(__file__).with_name('audit_amass_coverage.py')),
             'coarse_candidates_sha256':sha256(args.screen_dir/'candidates.csv'),
             'limitations':['sample-negative files not rechecked; missed candidates possible',
                            'geometry is not verified sitting action or independent events',
                            'long seated geometry is not necessarily motionless holding',
                            'all inspected sources are explored, not untouched test']}
    args.output.mkdir(parents=True)
    if rows:write_csv(args.output/'candidates.csv',rows,list(rows[0]))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='files_manifest'}),flush=True)


if __name__=='__main__':main()
