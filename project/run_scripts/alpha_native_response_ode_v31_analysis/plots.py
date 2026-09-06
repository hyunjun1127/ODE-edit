"""Deterministic matplotlib figures; input tables only, no model imports."""
from pathlib import Path
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODELS=('llama3-8b-inst','qwen2.5-7b-inst')
def plot_all(directory,batches,layers,nodes):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':110,
        'savefig.dpi':110,'axes.unicode_minus':False,'path.simplify':False})
    outputs=[]
    def finish(fig,name):
        fig.tight_layout();p=directory/name
        fig.savefig(p,metadata={'Software':'alpha-jv-sequential-analysis-v1'})
        plt.close(fig);outputs.append(dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    def heat(ax,values,title,labels):
        v=np.array(values,dtype=float);masked=np.ma.masked_where(v<=0,v)
        im=ax.imshow(np.ma.log10(masked),aspect='auto',interpolation='nearest',cmap='viridis')
        ax.set_title(title);ax.set_xticks(range(5),['L4','L5','L6','L7','L8'])
        ax.set_yticks(range(len(labels)),labels);fig=ax.figure
        fig.colorbar(im,ax=ax,label='log10 Frobenius norm')
    fig,axs=plt.subplots(2,2,figsize=(12,9));fig.suptitle('Layer-wise Update Magnitude')
    for i,m in enumerate(MODELS):
        for j,arm in enumerate(('O_NATIVE','JV_NATIVE')):
            x=[r for r in layers if r['alias']==m and r['arm']==arm]
            vals=[[next(r['batch_net_norm'] for r in x if r['batch']==b and r['layer']==l) for l in range(4,9)] for b in range(1,11)]
            heat(axs[i,j],vals,f'{m} / {arm}',[f'B{b}' for b in range(1,11)])
    finish(fig,'layer-update-magnitude.png')
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for i,m in enumerate(MODELS):
        x=[r for r in batches if r['alias']==m and r['arm']=='JV_NATIVE'];t=[r['batch'] for r in x]
        axs[i,0].plot(t,[100*r['L8_energy_share'] if r['L8_energy_share'] is not None else np.nan for r in x],marker='o',label='L8 energy %')
        axs[i,0].set_ylim(0,102);axs[i,0].set_title(m);axs[i,0].legend()
        for key,label in [('total_batch_net_energy','total'),('L4_7_batch_net_energy','L4-7')]:
            axs[i,1].plot(t,[r[key] for r in x],marker='o',label=label)
        axs[i,1].set_yscale('log');axs[i,1].set_title('Actual batch-net squared Frobenius');axs[i,1].legend()
        for k in ('RS','PS','NS'):axs[i,2].plot(t,[r[k+'_rate']*100 for r in x],marker='o',label=k)
        axs[i,2].set_title('Current B100 preference (%)');axs[i,2].legend()
        for ax in axs[i]:ax.set_xlabel('Sequential batch');ax.set_xticks(t)
    finish(fig,'concentration-absolute-action-endpoints.png')
    fig,axs=plt.subplots(2,2,figsize=(13,10));fig.suptitle('Layer-wise Update Magnitude')
    for i,m in enumerate(MODELS):
        selected=[r for r in nodes if r['alias']==m]
        v=[[r[f'L{l}_step_norm'] for l in range(4,9)] for r in selected]
        heat(axs[i,0],v,m,[f'B{r["batch"]}/n{r["node"]}' for r in selected])
        for b in range(1,11):
            a=[r for r in selected if r['batch']==b]
            axs[i,1].plot([r['node'] for r in a],[r['V_ratio'] for r in a],marker='.',label=f'B{b}')
        axs[i,1].set_title(m+' / V after node / V entry');axs[i,1].set_xticks(range(4));axs[i,1].legend(ncol=2)
    finish(fig,'four-node-progression.png')
    return outputs
