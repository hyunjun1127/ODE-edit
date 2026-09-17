"""Deterministic static scientific PNGs from compact CPU-review CSVs."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .metrics import ARMS

def rows(path):return list(csv.DictReader(Path(path).open()))
def plots(data,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);data=Path(data)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':120,'savefig.dpi':150})
    colors=dict(zip(ARMS,plt.get_cmap('tab10').colors))
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/name,metadata={'Software':'local-z CPU review matplotlib'});plt.close(fig)
    pop=rows(data/'population-metrics.csv');batch=rows(data/'batch-metrics.csv');selection=rows(data/'selection.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7))
    for ax,tag in zip(axes,('RS','PS','NS')):
        rr={r['arm']:r for r in pop if r['population']=='W10_ALL1000' and r['status']=='ALL' and r['metric']==tag}
        values=[float(rr[a]['percent']) for a in ARMS];ax.bar(ARMS,values,color=[colors[a] for a in ARMS])
        ax.set(title=tag+' at W10',ylabel='Canonical success (%)',ylim=(0,105));ax.tick_params(axis='x',rotation=45)
        for i,v in enumerate(values):ax.text(i,v+1,f'{v:.2f}',ha='center',fontsize=8)
    fig.suptitle('Local-z: matched cold first1000 (one fixed order)',y=.99);save(fig,'final-seven-arm.png')
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for ax,tag in zip(axes,('RS','PS','NS')):
        for a in ARMS:
            rr=[r for r in batch if r['arm']==a and r['state']=='selected' and r['metric']==tag]
            ax.plot([int(r['batch']) for r in rr],[float(r['percent']) for r in rr],'.-',label=a,color=colors[a])
        ax.set(ylabel=tag+' current (%)');ax.grid(alpha=.2)
    axes[0].legend(ncol=7,fontsize=8);axes[-1].set(xlabel='Batch (different current100 each point)',xticks=range(1,11));save(fig,'current-batch-curves.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for a in ARMS:
        rr=[r for r in selection if r['arm']==a];axes[0].plot([int(r['batch']) for r in rr],[float(r['D']) for r in rr],'.-',color=colors[a],label=a)
    axes[0].set(xlabel='Batch',ylabel='Selected S64 KL',title='Controller panel (fixed policies: scored only)');axes[0].legend(ncol=2)
    compare=rows(data/'paired-policy-comparisons.csv')
    rr=[r for r in compare if r['treatment']=='LD' and r['reference']=='N4'];x=np.arange(3)
    axes[1].bar(x-.18,[int(r['lost']) for r in rr],.36,label='Lost');axes[1].bar(x+.18,[int(r['gained']) for r in rr],.36,label='Gained')
    axes[1].set(xticks=x,xticklabels=[r['metric'] for r in rr],ylabel='Prompt comparisons',title='N4 → LD, same W10 population');axes[1].legend();save(fig,'selector-and-paired.png')
    fig,ax=plt.subplots(figsize=(11,3));matrix=[]
    labels=[]
    for a in ('L4D','LD','TD'):
        rr=[r for r in selection if r['arm']==a];matrix.append([0 if r['selected']=='N4' else 1 if r['selected']=='L0.75-0' else 2 for r in rr]);labels.append([r['selected'] for r in rr])
    from matplotlib.colors import ListedColormap
    ax.imshow(matrix,aspect='auto',vmin=0,vmax=2,cmap=ListedColormap(['#dde3e8','#a9d6cf','#679bcc']))
    for i,rr in enumerate(labels):
        for j,label in enumerate(rr):ax.text(j,i,label,ha='center',va='center',fontsize=8)
    ax.set(yticks=range(3),yticklabels=['L4D','LD','TD'],xticks=range(10),xticklabels=range(1,11),xlabel='Batch',title='All dynamic selections (own-entry N4 is not the independent N4 chain)');save(fig,'dynamic-selections.png')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);a=p.parse_args();plots(a.data,a.out)
