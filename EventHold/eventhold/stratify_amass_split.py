"""Pre-training coverage split, with ACCAD identity aliases kept train-only."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from .audit import sha256,write_csv


def stratify(rows,transition_subjects):
    rows=[dict(r) for r in rows]
    # ACCAD Female1General/Female1Gestures etc are not separate people.
    # Conservatively group the whole dataset until all aliases are resolved.
    accad_groups={r['group_key'] for r in rows if r['dataset']=='ACCAD'}
    for r in rows:
        if r['group_key'] in accad_groups:r['group_key']='ACCAD_all_identity_aliases_train_only'
    groups=defaultdict(list)
    for r in rows:groups[r['group_key']].append(r)
    strata=defaultdict(list)
    for key,g in groups.items():
        strata[('|'.join(sorted({r['dataset'] for r in g})),
                any(r['subject_key'] in transition_subjects for r in g))].append(key)
    assignment={}
    for (datasets,has),keys in strata.items():
        ordered=sorted(keys,key=lambda k:hashlib.sha256(('eventhold-amass-v2:'+k).encode()).hexdigest())
        n=min(len(keys)-1,max(1,math.ceil(len(keys)*.2))) if len(keys)>1 else 0
        for i,k in enumerate(ordered):assignment[k]='development' if i<n else 'train'
    for r in rows:r['split']=assignment[r['group_key']]
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-dir',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use new frozen version')
    with (args.base_dir/'manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    with args.candidates.open(encoding='utf-8-sig') as f:cs=list(csv.DictReader(f))
    transition={r['subject_key'] for r in cs if r['kind']=='seated_like' and r['complete_transition_proxy']=='True'}
    rows=stratify(rows,transition);args.output.mkdir(parents=True)
    write_csv(args.output/'manifest.csv',rows,list(rows[0]))
    summary={'status':'frozen_explored_train_development_not_test','version':'v3',
             'rule':'dataset and transition-presence stratified hash ordering, ceil20percent dev; singleton train; all ACCAD aliases train-only',
             'changes_before_training':['v1 lacks development transition candidates','v2 wrongly separates ACCAD same-person action folders; v3 conservatively groups all ACCAD'],
             'counts':{s:{'sequences':sum(r['split']==s for r in rows),'groups':len({r['group_key'] for r in rows if r['split']==s}),
                         'transition_subject_keys':sorted({r['subject_key'] for r in rows if r['split']==s and r['subject_key'] in transition})} for s in ['train','development']},
             'manifest_sha256':sha256(args.output/'manifest.csv'),'base_manifest_sha256':sha256(args.base_dir/'manifest.csv'),
             'candidate_sha256':sha256(args.candidates),'code_sha256':sha256(Path(__file__)),
             'limitations':['all sources explored; not untouched test','unknown cross-dataset identities and semantic duplicates unresolved'],
             'DIP_split_changed':False,'P5_training':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
