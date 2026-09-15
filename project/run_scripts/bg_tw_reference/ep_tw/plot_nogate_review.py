"""Deterministic scientific PNGs from published CSVs only; no raw/model import."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def rows(root,name):return list(csv.DictReader((root/name).open()))
def nums(rr,key):return [float(r[key]) for r in rr]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(package,output):
    p,o=Path(package),Path(output);o.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':110,'savefig.dpi':150,
                         'axes.spines.top':False,'axes.spines.right':False})
    def finish(fig,name):
        fig.tight_layout();fig.savefig(o/name,metadata={'Software':'EP47962 CSV-only review'});plt.close(fig)
    current=rows(p,'batch-current-metrics.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,m in zip(axes,('RS','PS','NS')):
        for state,color,style in [('ENTRY','#777777','--'),('RAW','#E69F00','-'),('SELECTED','#0072B2',':')]:
            rr=[r for r in current if r['population']=='CURRENT' and r['metric']==m and r['state']==state]
            ax.plot(nums(rr,'batch'),nums(rr,'percent'),style,marker='o',markersize=3,color=color,label=state)
        ax.set(title=f'Current {m}',xlabel='Batch',ylabel='Canonical success (%)',xticks=range(1,11));ax.grid(alpha=.2)
    axes[0].legend();fig.suptitle('Own-batch performance; RAW and SELECTED counts overlap')
    finish(fig,'current-state-curves.png')
    cand=rows(p,'candidate-details.csv');fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,key,title in zip(axes,['E_minus_RAW','D64_minus_RAW'],['Current E change','S64 KL change']):
        data=np.array([[float(next(r[key] for r in cand if r['batch']==str(b) and r['candidate']==c))
                        for b in range(1,11)] for c in ('C1','C05','C025')])
        lim=max(abs(data.min()),abs(data.max()));im=ax.imshow(data,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
        for i,c in enumerate(('C1','C05','C025')):
            for j in range(10):
                r=next(r for r in cand if r['batch']==str(j+1) and r['candidate']==c)
                if r['selected']=='True':ax.text(j,i,'S',ha='center',va='center',weight='bold')
                elif r['feasible']=='False':ax.text(j,i,'×',ha='center',va='center')
        ax.set(title=title,xlabel='Batch',xticks=range(10),xticklabels=range(1,11),yticks=range(3),yticklabels=['C1','C05','C025'])
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('All correction candidates: S selected; × quality-ineligible; RAW at B5/B10')
    finish(fig,'candidate-screen.png')
    mech=rows(p,'per-batch-mechanism.csv');fig,axes=plt.subplots(2,2,figsize=(11,7))
    x=nums(mech,'batch')
    axes[0,0].bar(x,nums(mech,'gradient_cos'),color='#0072B2');axes[0,0].axhline(0,color='black',lw=.7)
    axes[0,0].set(title='gE / gD alignment',ylabel='Cosine')
    axes[0,1].bar(x,[100*float(r['correction_native_ratio']) for r in mech],color='#009E73')
    axes[0,1].set(title='Layer-wise Update Magnitude',ylabel='Actual correction / native (%)')
    axes[1,0].bar(x,nums(mech,'ball_clamped_requests'),color='#E69F00');axes[1,0].set(title='Target-ball projected requests',ylabel='Requests / 100')
    axes[1,1].bar(x,nums(mech,'ge_dot_C_recorded'),color='#CC79A7');axes[1,1].axhline(0,color='black',lw=.7)
    axes[1,1].set(title='After ball/trust: <gE, C>',ylabel='Stored inner product')
    for ax in axes.flat:ax.set(xlabel='Batch',xticks=range(1,11));ax.grid(axis='y',alpha=.2)
    finish(fig,'mechanism-actions.png')
    cohort=rows(p,'cohort-retention.csv');fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,m in zip(axes,('RS','PS','NS')):
        rr=[r for r in cohort if r['metric']==m];xx=nums(rr,'cohort')
        ax.bar(np.array(xx)-.18,nums(rr,'lost'),.36,label='Lost',color='#D55E00')
        ax.bar(np.array(xx)+.18,nums(rr,'gained'),.36,label='Gained',color='#0072B2')
        ax.set(title=f'{m}: at-write → W10',xlabel='Request cohort (batch)',ylabel='Prompt pairs',xticks=range(1,11))
    axes[0].legend();fig.suptitle('Canonical transitions; B10 has zero future batch exposure')
    finish(fig,'cohort-transitions.png')
    base=rows(p,'baseline-first1000.csv');final=rows(p,'first-final-table.csv')
    policies=list(dict.fromkeys(r['policy'] for r in base));labels=['Alpha BLUE L4','Alpha BLUE L4+8','MEMIT BLUE L4+8','Base Alpha','Base MEMIT','W0']
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    for ax,m in zip(axes,('RS','PS','NS')):
        values=[float(next(r['percent'] for r in base if r['policy']==z and r['metric']==m)) for z in policies]
        values.append(float(next(r['percent'] for r in final if r['metric']==m)))
        ax.bar(range(7),values,color=['#999999']*6+['#0072B2'])
        ax.set(title=m,ylabel='Canonical success (%)',ylim=(0,105),xticks=range(7),xticklabels=labels+['EP-TW-1'])
        ax.tick_params(axis='x',rotation=65)
    fig.suptitle('W10 first1000 / W0 references: seed, hparams and trajectory are not controlled')
    finish(fig,'baseline-reference.png')
    fig,axes=plt.subplots(1,3,figsize=(11,3.8))
    for ax,m in zip(axes,('RS','PS','NS')):
        r=next(r for r in final if r['metric']==m);xx=np.arange(4)
        for off,key,label,color in [(-.18,'new_nll_','New','#0072B2'),(.18,'true_nll_','True','#E69F00')]:
            ax.bar(xx+off,[float(r[key+q]) for q in ('mean','median','p95','p99')],.36,label=label,color=color)
        ax.set(title=f'W10 {m}',ylabel='Sequence mean NLL',xticks=xx,xticklabels=['Mean','Median','p95','p99'])
    axes[0].legend();finish(fig,'final-nll-tails.png')
    inputs=['batch-current-metrics.csv','candidate-details.csv','per-batch-mechanism.csv','cohort-retention.csv','baseline-first1000.csv','first-final-table.csv']
    manifest=dict(generator_sha256=digest(Path(__file__)),matplotlib=matplotlib.__version__,numpy=np.__version__,
                  inputs=[dict(path=z,sha256=digest(p/z)) for z in inputs],
                  outputs=[dict(path=f.name,bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(o.glob('*.png'))],
                  image_generation='PYTHON_MATPLOTLIB_ONLY',raw_access=False)
    with (o/'plot-manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps({'png_count':len(manifest['outputs']),'output':str(o)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.package,a.output)
