"""Export reference-only long-posture QC figures, not model predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .natural_motion import posture_masks
from .audit import write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.cache,allow_pickle=False) as a:
        pos=a['segment_position'];time=a['time_ms']/1000
    meta=json.loads(args.cache.with_suffix('.json').read_text());names=meta['segment_names']
    _,_,g=posture_masks(pos,names)
    with args.candidates.open(encoding='utf-8-sig') as f:
        candidates=[r for r in csv.DictReader(f) if r['kind']=='seated_like' and Path(r['member']).stem==args.cache.stem]
    if not candidates:raise ValueError('No seated-like candidates to review')
    values={'left_knee_deg':g['knee_bend_deg'][:,0],'right_knee_deg':g['knee_bend_deg'][:,1],
            'trunk_tilt_deg':g['trunk_tilt_deg'],'pelvis_z_m':pos[:,names.index('Pelvis'),2]}
    rows=[]
    for second in np.unique(np.floor(time).astype(int)):
        mask=(time>=second)&(time<second+1)
        row={'second':int(second),'native_frames':int(mask.sum())}
        for key,value in values.items():
            v=value[mask];good=v[np.isfinite(v)]
            row[key+'_valid_count']=len(good)
            for stat,fn in [('min',np.min),('median',np.median),('max',np.max)]:
                row[key+'_'+stat]=float(fn(good)) if len(good) else np.nan
        rows.append(row)
    write_csv(args.output/'one_second_reference_summary.csv',rows,list(rows[0]))
    colors=['#0072B2','#D55E00','#009E73']
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True,constrained_layout=True)
        t=np.array([r['second']+.5 for r in rows])/60
        for ax,keys,ylabel in zip(axes,[['left_knee_deg','right_knee_deg'],['trunk_tilt_deg'],['pelvis_z_m']],
                                  ['Knee bend (degrees)','Trunk tilt from vertical (degrees)','Native reference pelvis Z (m)']):
            for i,key in enumerate(keys):
                lo=[r[key+'_min'] for r in rows];hi=[r[key+'_max'] for r in rows];mid=[r[key+'_median'] for r in rows]
                ax.fill_between(t,lo,hi,color=colors[i],alpha=.18)
                ax.plot(t,mid,color=colors[i],ls='-' if i==0 else '--',lw=1,label=key.replace('_',' '))
            for j,r in enumerate(candidates):
                start=float(r['start_seconds'])/60;end=start+float(r['duration_seconds'])/60
                ax.axvspan(start,end,facecolor='none',edgecolor='#777777',hatch='//',lw=.7)
                if ax is axes[0]:ax.text((start+end)/2,165,'C'+str(j+1),ha='center',fontsize=9)
            ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=8)
        axes[0].set_ylim(0,180);axes[-1].set_xlabel('Native sequence time (minutes)')
        fig.suptitle(args.cache.stem+' | Xsens reference only, not EventHold predictions\n1 s median and min–max; hatched spans are geometric candidates, not verified sitting labels',fontsize=12)
        for ext in ['png','svg']:fig.savefig(args.output/('reference_timeline.'+ext),dpi=180,facecolor='white')
        plt.close(fig)
        first=candidates[0];last=candidates[-1];fps=meta['frame_rate']
        frames=[max(0,int(first['start_frame'])-int(3*fps)),(int(first['start_frame'])+int(first['end_frame_exclusive']))//2,
                min(len(pos)-1,int(first['end_frame_exclusive'])+int(3*fps)),(int(last['start_frame'])+int(last['end_frame_exclusive']))//2]
        paths=[['Pelvis','L5','L3','T12','T8','Neck','Head']]
        for side in ['Left','Right']:
            paths.extend([['T8',side+'Shoulder',side+'UpperArm',side+'ForeArm',side+'Hand'],
                          ['Pelvis',side+'UpperLeg',side+'LowerLeg',side+'Foot',side+'Toe']])
        fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
        snapshot_rows=[]
        for col,frame in enumerate(frames):
            xyz=pos[frame].copy();xyz[:,:2]-=xyz[names.index('Pelvis'),:2]
            for j,name in enumerate(names):snapshot_rows.append({'frame':frame,'time_seconds':float(time[frame]),'segment':name,
                                                               'root_centered_x_m':float(xyz[j,0]),'root_centered_y_m':float(xyz[j,1]),'native_z_m':float(xyz[j,2])})
            for row,xaxis in enumerate([0,1]):
                ax=axes[row,col]
                for path in paths:
                    ids=[names.index(n) for n in path]
                    ax.plot(xyz[ids,xaxis],xyz[ids,2],'-o',color='#333333',lw=1.2,ms=3)
                ids=[names.index(n) for n in meta['selected_sensor_names']]
                ax.scatter(xyz[ids,xaxis],xyz[ids,2],marker='s',s=28,facecolor='none',edgecolor='#0072B2',label='5 sensor segments')
                ax.set(xlim=(-1.1,1.1),ylim=(-.1,2.1),aspect='equal',xlabel=('X' if xaxis==0 else 'Y')+' relative to pelvis (m)',ylabel='Native Z (m)',title=f'frame {frame} | {time[frame]:.2f} s')
                ax.grid(alpha=.2)
        fig.suptitle('Xsens reference skeleton snapshots | horizontal root centering only\nColumns: 3 s before C1, middle C1, 3 s after C1, middle C4; rows: XZ / YZ projections',fontsize=12)
        for ext in ['png','svg']:fig.savefig(args.output/('reference_snapshots.'+ext),dpi=180,facecolor='white')
        plt.close(fig)
    write_csv(args.output/'snapshot_positions.csv',snapshot_rows,list(snapshot_rows[0]))
    manifest={'purpose':'internal_reference_quality_review_not_publication_claim','cache':str(args.cache),
              'cache_sha256':hashlib.sha256(args.cache.read_bytes()).hexdigest(),
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'matplotlib_version':matplotlib.__version__,
              'transformations':['native 240Hz reference aggregated in half-open 1s bins, median and min/max',
                                 'skeleton horizontal pelvis centering, no Z centering, fixed axes across snapshots'],
              'missing':'nonfinite values excluded within bins and valid counts retained; empty bins NaN',
              'alt_text':'Reference knee bend, trunk tilt and pelvis height over the complete P5 sequence; four hatched seated-like intervals. Separate XZ and YZ skeleton projections show selected reference frames, not model predictions.',
              'journal_requirements':'not_applicable_internal_review',
              'snapshot_frames':frames}
    (args.output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
