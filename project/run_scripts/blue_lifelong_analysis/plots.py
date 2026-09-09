"""Deterministic Agg PNGs from sealed aggregate CSVs only."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

COLORS={'ORIGINAL':'#2563a6','L4_ONLY':'#d56b28','L8_ONLY':'#238466'}
VARIANTS=list(COLORS);METHODS=['MEMIT','AlphaEdit']
PANEL_ARMS=[m+'_'+v for m in METHODS for v in VARIANTS]
DPI=160
def setup():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'figure.dpi':DPI,'savefig.dpi':DPI,'axes.grid':False,'path.simplify':False})

def selected(rows,arm,metric=None):
    return [r for r in rows if r['arm']==arm and (metric is None or r.get('metric')==metric)]

def update_figure(rows):
    fig,axes=plt.subplots(1,2,figsize=(11,4),squeeze=False)
    for ax,method in zip(axes[0],METHODS):
        for i,var in enumerate(VARIANTS):
            rr=selected(rows,method+'_'+var)
            vals=[np.mean([float(r['update_norm']) for r in rr if int(r['layer'])==l]) if any(int(r['layer'])==l for r in rr) else 0 for l in [4,8]]
            ax.bar(np.arange(2)+(i-1)*.22,vals,.22,label=var,color=COLORS[var])
        ax.set_xticks([0,1],['L4','L8']);ax.set_title(method);ax.set_ylabel('Mean batch-net Frobenius norm');ax.set_ylim(bottom=0);ax.legend(fontsize=7)
    fig.suptitle('Layer-wise Update Magnitude');fig.tight_layout();return fig

def generate(out,dest):
    setup();dest.mkdir(parents=True,exist_ok=False)
    inputs={n:csvread(out/(n+'.csv')) for n in ['final-summary','cumulative-metrics','current-metrics','age-strata-metrics','cohort-retention','layer-updates','batch-cost']}
    manifest=[]
    def write(fig,name,used,caption):
        if fig.get_layout_engine() is None:fig.tight_layout()
        axes_spec=[dict(title=ax.get_title(),xlabel=ax.get_xlabel(),ylabel=ax.get_ylabel(),xlim=list(ax.get_xlim()),ylim=list(ax.get_ylim())) for ax in fig.axes]
        size=list(fig.get_size_inches());path=dest/(name+'.png');fig.savefig(path,metadata={'Software':'ODE-edit BLUE lifelong deterministic Agg'});plt.close(fig)
        manifest.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size,source_sha256=sha(Path(__file__)),inputs=[dict(path=n+'.csv',sha256=sha(out/(n+'.csv'))) for n in used],
                             command='python -m project.run_scripts.blue_lifelong_analysis.plots --out PACKAGE --dest NEW_FIGURE_DIRECTORY',backend='Agg',DPI=DPI,figsize_inches=size,axes=axes_spec,colors=COLORS,seed='NO_RANDOMNESS',ordering='methods MEMIT/AlphaEdit; variants ORIGINAL/L4_ONLY/L8_ONLY',caption=caption))
    fig,ax=plt.subplots(figsize=(11,4.7));xs=np.arange(6)
    for j,tag in enumerate(MULT):ax.bar(xs+(j-1)*.24,[100*float(r[tag+'_rate']) for r in inputs['final-summary']],.24,label=tag)
    ax.set_xticks(xs,ARMS,rotation=22,ha='right');ax.set_ylim(0,100);ax.set_ylabel('Canonical preference success (%)');ax.legend();ax.set_title('Final W100: all 10,000 edited requests')
    write(fig,'final-full10000',['final-summary'],'6 arms; RS 10000/PS 20000/NS 100000 prompts per arm, strict inequality; no missing/imputation; final endpoint only.')
    cum=inputs['cumulative-metrics'];cur=inputs['current-metrics']
    fig,axes=plt.subplots(2,3,figsize=(13,7),squeeze=False)
    for j,method in enumerate(METHODS):
        for k,tag in enumerate(MULT):
            ax=axes[j,k]
            for v in VARIANTS:
                rr=selected(cum,method+'_'+v,tag);ax.plot([100*int(r['batch']) for r in rr],[100*float(r['rate']) for r in rr],'-o',ms=3,label=v,color=COLORS[v])
            ax.set_title(method+' '+tag);ax.set_ylim(0,100);ax.set_xlim(0,10000);ax.set_xlabel('Seen requests');ax.set_ylabel('Success (%)');ax.legend(fontsize=7)
    write(fig,'cumulative-seen-prefix',['cumulative-metrics'],'12 exact stored states per arm (100,500,1000..10000); denominators t/2t/10t. Connecting lines are guides, not evaluated missing states.')
    fig,axes=plt.subplots(2,3,figsize=(13,7),squeeze=False)
    for ax,arm in zip(axes.flat,PANEL_ARMS):
        for scope,rows,style in [('Current B100',cur,'-'),('All seen',cum,'--o')]:
            rr=selected(rows,arm,'RS');ax.plot([int(r['batch']) for r in rr],[100*float(r['rate']) for r in rr],style,ms=3,label=scope)
        ax.set_title(arm);ax.set_ylim(0,100);ax.set_xlim(1,100);ax.set_xlabel('Sequential batch');ax.set_ylabel('RS (%)');ax.legend(fontsize=7)
    write(fig,'current-vs-cumulative',['current-metrics','cumulative-metrics'],'Current denominator100 at each of100 batches versus all-seen denominator100*b at12 checkpoints. Distinct states/cohorts; no online pooling as final.')
    fig,axes=plt.subplots(2,3,figsize=(13,7),squeeze=False,layout='constrained')
    for ax,arm in zip(axes.flat,PANEL_ARMS):
        a=np.full((len(SCHEDULE),100),np.nan)
        for r in selected(inputs['cohort-retention'],arm,'RS'):a[SCHEDULE.index(int(r['batch'])),int(r['cohort'])-1]=100*float(r['rate'])
        im=ax.imshow(a,vmin=0,vmax=100,aspect='auto',interpolation='nearest',cmap='viridis');ax.set_title(arm);ax.set_yticks(range(len(SCHEDULE)),SCHEDULE);ax.set_xticks([0,24,49,74,99],[1,25,50,75,100]);ax.set_xlabel('Write cohort (100 requests)');ax.set_ylabel('Checkpoint batch')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='RS (%)',shrink=.7)
    write(fig,'cohort-retention-heatmap',['cohort-retention'],'Each occupied cell100 requests; rows12 frozen checkpoints; future/unseen cohorts masked, no interpolation/imputation. Panel rows MEMIT/AlphaEdit, columns original/L4/L8.')
    for side in ['new','true']:
        fig,axes=plt.subplots(2,3,figsize=(13,7),squeeze=False)
        for j,method in enumerate(METHODS):
            for k,tag in enumerate(MULT):
                ax=axes[j,k]
                for v in VARIANTS:
                    rr=selected(cum,method+'_'+v,tag);x=[100*int(r['batch']) for r in rr]
                    for st,style in [('median','-'),('p90','--')]:ax.plot(x,[float(r[f'{side}_nll_request_{st}']) for r in rr],style,label=v+' '+st,color=COLORS[v])
                ax.set_title(method+' '+tag+' target-'+side);ax.set_ylim(bottom=0);ax.set_xlim(0,10000);ax.set_xlabel('Seen requests');ax.set_ylabel('Request-cluster mean-token NLL (nats)');ax.legend(fontsize=6)
        write(fig,'cumulative-nll-'+side,['cumulative-metrics'],'Each request first averages own prompts; then median/p90 across t requests. Rewrite1/rephrase2/neighborhood10 prompts; target-new/true kept separate; no imputation.')
    for tag in ['RS','NS']:
        fig,axes=plt.subplots(2,3,figsize=(13,7),squeeze=False)
        for ax,arm in zip(axes.flat,PANEL_ARMS):
            for st in ['early','middle','recent']:
                rr=[r for r in selected(inputs['age-strata-metrics'],arm,tag) if r['stratum']==st]
                ax.plot([100*int(r['batch']) for r in rr],[100*float(r['rate']) for r in rr],'-o',ms=3,label=st)
            ax.set_ylim(0,100);ax.set_xlim(0,10000);ax.set_title(arm+' '+tag);ax.set_xlabel('Seen requests');ax.set_ylabel('Success (%)');ax.legend(fontsize=7)
        write(fig,'age-retention-'+tag,['age-strata-metrics'],'Relative seen-prefix first20%, middle60%, last20%; prompt denominators per CSV; bins change with checkpoint, not longitudinal fixed individuals.')
    write(update_figure(inputs['layer-updates']),'layer-wise-update-magnitude',['layer-updates'],'Each value averages100 observed batch-net update norms; unselected layer zero by support definition. Magnitude not squared norm/native action; no uniform reference line.')
    fig,ax=plt.subplots(figsize=(11,4.7));edit=[];ev=[]
    for arm in ARMS:
        rr=selected(inputs['batch-cost'],arm);edit.append(sum(float(r['edit_seconds']) for r in rr)/3600);ev.append(sum(float(r['evaluation_seconds']) for r in rr)/3600)
    ax.bar(np.arange(6),edit,label='Native writer (inclusive)');ax.bar(np.arange(6),ev,bottom=edit,label='Endpoint evaluation');ax.set_xticks(np.arange(6),ARMS,rotation=22,ha='right');ax.set_ylabel('Recorded wall time (hours)');ax.set_title('Writer and evaluator accounting');ax.legend()
    write(fig,'compute-accounting',['batch-cost'],'100 batches/arm. Writer includes targets/keys/solve; evaluation includescurrent+12 seen-prefix scopes. Snapshot/hash/restore residual not attributed to writer/evaluator; not controlled speedup.')
    save(dest/'plot-manifest.json',manifest);return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();generate(a.out,a.dest)
