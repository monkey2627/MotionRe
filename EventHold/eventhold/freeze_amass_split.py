"""Freeze explored AMASS train/dev groups before any synthetic training.

Subject groups sharing byte-identical files are merged. Semantic duplicates
and cross-dataset identities are not thereby guaranteed absent. No test split.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from .audit import sha256,write_csv


def assign_groups(rows):
    parent={r['subject_key']:r['subject_key'] for r in rows}
    def root(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    seen={}
    for r in rows:
        subject=r['subject_key'];digest=r['sha256']
        if digest in seen:
            a,b=root(subject),root(seen[digest]);parent[max(a,b)]=min(a,b)
        else:seen[digest]=subject
    groups={}
    for subject in parent:groups.setdefault(root(subject),[]).append(subject)
    splits={}
    for subjects in groups.values():
        key='|'.join(sorted(subjects))
        value=int(hashlib.sha256(('eventhold-amass-v1:'+key).encode()).hexdigest()[:16],16)%100
        for subject in subjects:splits[subject]=('development' if value<20 else 'train',key)
    return [{**r,'split':splits[r['subject_key']][0],'group_key':splits[r['subject_key']][1]} for r in rows]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Frozen split cannot be overwritten')
    with args.manifest.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for i,r in enumerate(rows):
        path=Path(r['source']).resolve();r['source']=str(path);r['sha256']=sha256(path)
        if (i+1)%1000==0:print('Hashed',i+1,flush=True)
    rows=assign_groups(rows);args.output.mkdir(parents=True)
    write_csv(args.output/'manifest.csv',rows,list(rows[0]))
    summary={'status':'frozen_explored_train_development_not_test',
             'rule':'SHA256 eventhold-amass-v1 subject-group bucket <20 development, otherwise train',
             'sources':len(rows),'unique_byte_hashes':len(set(r['sha256'] for r in rows)),
             'counts':{s:{'sequences':sum(r['split']==s for r in rows),
                           'groups':len(set(r['group_key'] for r in rows if r['split']==s))} for s in ['train','development']},
             'manifest_sha256':sha256(args.output/'manifest.csv'),
             'source_inventory_sha256':sha256(args.manifest),'code_sha256':sha256(Path(__file__)),
             'DIP_split_changed':False,'P5_training':False,
             'limitations':['all sources previously screened','byte hash cannot detect alternate encodings of same motion',
                            'cross-dataset identity unresolved; not certified independent test']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
