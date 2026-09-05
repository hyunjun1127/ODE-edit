"""Deterministic headless figures from sealed sequential analysis tables only."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .sequential_analysis import ARMS, CELLS

ORDER=('LM','LA','QM','QA')
LABELS={'LM':'Llama / MEMIT','LA':'Llama / AlphaEdit','QM':'Qwen / MEMIT','QA':'Qwen / AlphaEdit'}
COLORS=dict(zip(ARMS,('#444444','#0072B2','#D55E00','#009E73','#CC79A7')))
STYLE={'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.spines.top':False,
       'axes.spines.right':False,'savefig.dpi':160,'figure.dpi':160,'path.simplify':False}


def panels(title, ylabel, *, ylim=None):
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,layout='constrained')
    fig.suptitle(title)
    for cell,ax in zip(ORDER,axes.flat):
        ax.set_title(LABELS[cell]);ax.set_ylabel(ylabel)
        if ylim is not None:ax.set_ylim(*ylim)
    return fig,axes


def weight_figure(frame):
    fig,axes=panels('Layer-wise Update Magnitude','Mean batch update Frobenius norm')
    for c,ax in zip(ORDER,axes.flat):
        for j,a in enumerate(ARMS):
            g=frame[(frame.cell==c)&(frame.arm==a)].groupby('layer').update_magnitude.mean().reindex(range(4,9))
            ax.bar(np.arange(5)+(j-2)*.16,g.to_numpy(),width=.15,color=COLORS[a],label=a)
        ax.set_xticks(range(5),range(4,9));ax.set_xlabel('Editable layer');ax.legend(fontsize=8,ncol=3)
    return fig


def render(tables:Path, output:Path) -> list[str]:
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update(STYLE);np.random.seed(20260906)
    files=[]
    def save(fig,name):
        path=output/name
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,metadata={'Software':'ODE-edit sequential_plots v1'});plt.close(fig);files.append(name)
    p=pd.read_csv(tables/'batch-performance.csv');post=p[p.stage=='IMMEDIATE_POST']
    for metric in ('RS','PS','NS'):
        fig,axes=panels(f'Immediate batch endpoint: {metric} (not cumulative retention)',f'{metric} (%)',ylim=(0,102))
        for c,ax in zip(ORDER,axes.flat):
            for a in ARMS:
                g=post[(post.cell==c)&(post.arm==a)].sort_values('batch')
                ax.plot(g.batch,100*g[metric+'_rate'],marker='o',ms=3,color=COLORS[a],label=a)
            ax.set_xticks(range(1,11));ax.set_xlabel('Sequential batch (100 new requests)');ax.legend(fontsize=8,ncol=3)
        save(fig,f'immediate-{metric.lower()}-trajectory.png')
    online=pd.read_csv(tables/'online-prefix-not-cumulative-W.csv')
    fig,axes=panels('Online-at-edit-time RS prefix aggregate — NOT W_t on all seen requests','RS (%)',ylim=(0,102))
    for c,ax in zip(ORDER,axes.flat):
        for a in ARMS:
            g=online[(online.cell==c)&(online.arm==a)].sort_values('through_batch')
            ax.plot(g.through_batch*100,100*g.RS_rate,color=COLORS[a],label=a)
        ax.set_xlabel('Requests edited (each scored at its own edit time)');ax.legend(fontsize=8,ncol=3)
    save(fig,'online-prefix-not-cumulative-W.png')
    nll=pd.read_csv(tables/'batch-distributions.csv')
    for category in ('rewrite','rephrase'):
        fig,axes=panels(f'Immediate {category} target-new NLL: median / p90','Request-cluster NLL (lower better)',ylim=(0,None))
        for c,ax in zip(ORDER,axes.flat):
            for a in ARMS:
                g=nll[(nll.cell==c)&(nll.arm==a)&(nll.stage=='IMMEDIATE_POST')&(nll.category==category)&(nll.target=='new')&(nll.unit=='request_cluster')].sort_values('batch')
                ax.plot(g.batch,g.nll_median,color=COLORS[a],label=a)
                ax.plot(g.batch,g.nll_p90,color=COLORS[a],ls='--',alpha=.7)
            ax.set_xlabel('Batch; solid=median, dashed=p90');ax.legend(fontsize=8,ncol=3)
        save(fig,f'immediate-{category}-nll.png')
    save(weight_figure(pd.read_csv(tables/'layer-updates.csv')),'layer-wise-update-magnitude.png')
    m=pd.read_csv(tables/'batch-mechanism-compute.csv')
    fig,axes=panels('Post-horizon normalized residual (dynamic arms only)','Mean request q = ||z* - z|| / ||R_entry||',ylim=(0,None))
    for c,ax in zip(ORDER,axes.flat):
        for a in ARMS[1:]:
            g=m[(m.cell==c)&(m.arm==a)].sort_values('batch')
            ax.plot(g.batch,g.q_terminal_mean,color=COLORS[a],label=a)
        ax.set_xlabel('Sequential batch');ax.legend(fontsize=8,ncol=2)
    save(fig,'dynamic-terminal-residual.png')
    fig,axes=panels('Measured endpoint-arm time / Official O time','Ratio (10-batch summed endpoint-call time)',ylim=(0,None))
    for c,ax in zip(ORDER,axes.flat):
        g=m[m.cell==c].groupby('arm').wall_seconds.sum().reindex(ARMS)
        ax.bar(range(5),g/g.loc['O'],color=[COLORS[a] for a in ARMS]);ax.set_xticks(range(5),ARMS);ax.set_xlabel('Includes endpoint evaluator; excludes shared compute_z')
    save(fig,'endpoint-call-compute-ratio.png')
    return files


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tables',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();render(a.tables,a.output)
