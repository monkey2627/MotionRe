"""Freeze a source/subject-level split for the bounded synthetic pilot."""
import argparse, csv, hashlib, json
from pathlib import Path
from .audit import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--candidate-csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use new split directory')
    meta=json.loads(args.manifest.read_text()); rows=meta['manifest']
    # Group by source subject key, not filename. The pilot contains no repeated
    # subject across datasets, but the grouping rule remains explicit.
    groups={}
    with args.candidate_csv.open(encoding='utf-8-sig') as f:candidates=list(csv.DictReader(f))
    for row in rows:
        source=row['source'];match=next(x for x in candidates if x['source']==source)
        key=match['subject_key'];groups.setdefault(key,[]).append(row)
    keys=sorted(groups);holdout={k for k in keys if int(hashlib.sha256(k.encode()).hexdigest()[:8],16)%4==0}
    if not holdout:holdout={keys[-1]}
    out=[]
    for key in keys:
        split='holdout_exploration' if key in holdout else 'train_exploration'
        for row in groups[key]:out.append({**row,'subject_key':key,'split':split})
    args.output.mkdir(parents=True)
    (args.output/'split.json').write_text(json.dumps({'status':'frozen_exploration_split_not_final_benchmark','manifest_sha256':sha256(args.manifest),'candidate_csv_sha256':sha256(args.candidate_csv),'group_rule':'subject_key','groups':len(keys),'holdout_groups':sorted(holdout),'rows':out,'no_test_claim':True},indent=2)+'\n')
    print(json.dumps({'groups':len(keys),'holdout_groups':sorted(holdout),'train_files':sum(x['split']=='train_exploration' for x in out),'holdout_files':sum(x['split']=='holdout_exploration' for x in out)}),flush=True)
if __name__=='__main__':main()
