"""CSV-only deterministic Matplotlib publication; no raw tensor/model reads."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS=('CAP1','CAP10','CAP100','NORM_ONLY')
COLORS=('#777777','#0072B2','#D55E00','#009E73')

def run(package,output):
    root,out=Path(package),Path(output);out.mkdir(parents=True,exist_ok=False)
    used=[]
    def rows(name):
        used.append(name)
        with (root/name).open() as f:return list(csv.DictReader(f))
    def nums(rr,key):return [float(r[key]) for r in rr]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':110,'savefig.dpi':150,
                         'axes.spines.top':False,'axes.spines.right':False})
    def finish(fig,name):
        fig.tight_layout();fig.savefig(out/name,metadata={'Software':'EP alpha cap sweep CSV-only'});plt.close(fig)
    final=rows('first-final-table.csv');fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,metric in zip(axes,('RS','PS','NS')):
        rr=[next(r for r in final if r['arm']==a and r['metric']==metric) for a in ARMS]
        vv=nums(rr,'percent');ax.bar(range(4),vv,color=COLORS)
        for i,r in enumerate(rr):ax.text(i,vv[i],r['numerator']+'/'+r['denominator'],ha='center',va='bottom',fontsize=8)
        ax.set(title=metric,ylabel='Canonical success (%)',ylim=(0,105),xticks=range(4),xticklabels=ARMS)
        ax.tick_params(axis='x',rotation=35)
    fig.suptitle('Actual W10, same first1000; CAP1 reused; diagnostics not established')
    finish(fig,'final-first1000.png')
    aa=rows('per-batch-actions.csv');fig,axes=plt.subplots(2,1,figsize=(11,7))
    for a,col in zip(ARMS,COLORS):
        rr=[r for r in aa if r['arm']==a];x=nums(rr,'batch')
        axes[0].plot(x,nums(rr,'C1_native_percent'),'-o',markersize=3,color=col,label=a)
        axes[1].plot(x,nums(rr,'selected_native_percent'),'-o',markersize=3,color=col,label=a)
    axes[0].set(yscale='log',title='Maximum C1 candidate action',ylabel='Correction / native (%)')
    axes[1].set_yscale('symlog',linthresh=.003)
    axes[1].set(title='Actually selected action, RAW = 0 included',ylabel='Correction / native (%)')
    for ax in axes:ax.set(xlabel='Batch',xticks=range(1,11));ax.grid(alpha=.2);ax.legend()
    finish(fig,'candidate-versus-selected-action.png')
    batches=rows('batch-current-metrics.csv');fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,m in zip(axes,('RS','PS','NS')):
        for a,col in zip(ARMS,COLORS):
            rr=[r for r in batches if r['arm']==a and r['population']=='CURRENT' and r['state']=='SELECTED' and r['metric']==m]
            ax.plot(nums(rr,'batch'),nums(rr,'percent'),'-o',markersize=3,color=col,label=a)
        ax.set(title='Own-batch '+m,xlabel='Batch',ylabel='Canonical success (%)',xticks=range(1,11));ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Current100 at different endpoints, not cumulative retention')
    finish(fig,'current-curves.png')
    cc=rows('candidate-details.csv');fig,axes=plt.subplots(4,2,figsize=(12,10))
    for i,a in enumerate(ARMS):
        for j,key in enumerate(('E_minus_RAW','D64_minus_RAW')):
            data=np.full((3,10),np.nan)
            for k,c in enumerate(('C1','C05','C025')):
                for b in range(1,11):
                    rr=next(r for r in cc if r['arm']==a and r['candidate']==c and r['batch']==str(b))
                    if rr.get(key):data[k,b-1]=float(rr[key])
            lim=np.nanmax(np.abs(data));lim=max(lim,1e-15)
            ax=axes[i,j];im=ax.imshow(data,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
            for k,c in enumerate(('C1','C05','C025')):
                for b in range(1,11):
                    rr=next(r for r in cc if r['arm']==a and r['candidate']==c and r['batch']==str(b))
                    if rr['selected']=='True':ax.text(b-1,k,'S',ha='center',va='center')
                    elif rr.get('feasible')=='False':ax.text(b-1,k,'x',ha='center',va='center')
            ax.set(title=a+': '+key,xticks=range(10),xticklabels=range(1,11),yticks=range(3),yticklabels=('C1','C05','C025'))
            fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('All candidates: S selected, x ineligible; color scale per arm/objective')
    finish(fig,'candidate-finite-screen.png')
    pair=rows('paired-transitions.csv');fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,m in zip(axes,('RS','PS','NS')):
        rr=[next(r for r in pair if r['arm']==a and r['metric']==m and r['comparison']=='ATWRITE_TO_W10' and r['population']=='ALL_REQUESTED') for a in ARMS]
        x=np.arange(4);ax.bar(x-.2,nums(rr,'lost'),.4,label='Lost',color='#D55E00');ax.bar(x+.2,nums(rr,'gained'),.4,label='Gained',color='#0072B2')
        ax.set(title=m,xlabel='Arm',ylabel='Prompt transitions',xticks=x,xticklabels=ARMS);ax.tick_params(axis='x',rotation=35)
    axes[0].legend();fig.suptitle('Same identities: own at-write to actual W10')
    finish(fig,'atwrite-terminal-transitions.png')
    policy=rows('batch-policy.csv');dev=rows('general-observer.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for a,col in zip(ARMS,COLORS):
        rr=[r for r in policy if r['arm']==a];axes[0].plot(nums(rr,'batch'),nums(rr,'selected_D64'),'-o',color=col,label=a,markersize=3)
        dd=[r for r in dev if r['arm']==a];axes[1].plot(nums(dd,'batch'),nums(dd,'D'),'-o',color=col,label=a)
    axes[0].set(title='S64 controller panel',xlabel='Batch',ylabel='KL(p0 || pW)');axes[1].set(title='Dev128 observer panel',xlabel='Batch',ylabel='KL(p0 || pW)',xticks=(5,10))
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    finish(fig,'generic-controller-observer.png')
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(generator_sha256=sha(Path(__file__)),matplotlib=matplotlib.__version__,numpy=np.__version__,
        inputs=[dict(path=n,sha256=sha(root/n)) for n in used],outputs=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.glob('*.png'))],
        image_generation='PYTHON_MATPLOTLIB_ONLY',raw_access=False)
    with (out/'plot-manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(run(a.package,a.output)))
