"""Deterministic headless PNGs from sealed aggregate CSV only."""
import argparse,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import OUT,LOCAL,ARMS,family,sha,save,csvread,np

VARIANTS=['Native','BLUE','L4','L5','L6','L7','L8']
COLORS=['#777777','#111111','#0072B2','#009E73','#E69F00','#CC79A7','#D55E00']
STYLE=dict(dpi=150,seed=0,font='DejaVu Sans',families=['MEMIT','AlphaEdit'],variants=VARIANTS,colors=COLORS,missing='NA masked; no interpolation/imputation')
def variant(a):return 'Native' if a.startswith('BASE') else 'BLUE' if 'ORIGINAL' in a else a[-7:-5]
def ordered(method):return sorted([a for a in ARMS if family(a)==method],key=lambda a:VARIANTS.index(variant(a)))
def weight_panel(ax,rows,method):
    names=ordered(method);x=np.arange(7);width=.15
    for j,l in enumerate(range(4,9)):
        y=[]
        for a in names:
            r=next((r for r in rows if r['arm']==a and int(r['layer'])==l),None)
            y.append(float(r['update_norm_mean']) if r else np.nan)
        ax.bar(x+(j-2)*width,y,width,label='L'+str(l),color=COLORS[j+2])
    ax.set_xticks(x,VARIANTS);ax.set_title(method);ax.set_ylabel('Mean batch-net Frobenius magnitude');ax.legend(fontsize=8)

def main(destination):
    np.random.seed(0);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':150,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    destination.mkdir(parents=True,exist_ok=True);receipts=[]
    summary=csvread(OUT/'final-summary.csv');cum=csvread(OUT/'cumulative-metrics.csv');cur=csvread(OUT/'current-metrics.csv');cohort=csvread(OUT/'cohort-retention.csv');layer=csvread(OUT/'layer-summary.csv');cost=csvread(OUT/'compute-summary.csv')
    inputs=[OUT/(n+'.csv') for n in ['final-summary','cumulative-metrics','current-metrics','cohort-retention','layer-summary','compute-summary']]
    def emit(fig,name,caption):
        p=destination/name;assert not p.exists()
        fig.tight_layout();fig.savefig(p,metadata={'Software':'ODE-edit CPU plotting'});plt.close(fig)
        receipts.append(dict(path=name,sha256=sha(p),bytes=p.stat().st_size,caption=caption))
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for i,method in enumerate(STYLE['families']):
        names=ordered(method)
        for j,tag in enumerate(['RS','PS','NS']):
            values=[100*float(next(r for r in summary if r['arm']==a)[tag+'_rate']) for a in names]
            values=[{'RS':7.91,'PS':9.985,'NS':89.212}[tag]]+values
            axs[i,j].barh(['W0']+VARIANTS,values,color=['#cccccc']+COLORS);axs[i,j].invert_yaxis();axs[i,j].set_xlim(0,100);axs[i,j].set_title(method+' '+tag);axs[i,j].set_xlabel('Canonical NLL-pair success (%)')
    emit(fig,'final-family-rspsns.png','Final W100 full10000: per arm RS10000/PS20000/NS100000. W0 shared once, displayed twice. No imputation.')
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for i,method in enumerate(STYLE['families']):
        for j,tag in enumerate(['RS','PS','NS']):
            names=[next(a for a in ordered(method) if variant(a)=='L'+str(l)) for l in range(4,9)]
            axs[i,j].plot(range(4,9),[100*float(next(r for r in summary if r['arm']==a)[tag+'_rate']) for a in names],marker='o',label='Single-layer')
            for v,style in [('BLUE','--'),('Native',':')]:
                a=next(a for a in ordered(method) if variant(a)==v);value=100*float(next(r for r in summary if r['arm']==a)[tag+'_rate'])
                axs[i,j].plot([4,8],[value,value],style,label=v)
            axs[i,j].set(xticks=range(4,9),ylim=(0,100),title=method+' '+tag,xlabel='Editable AND target layer',ylabel='Success (%)');axs[i,j].legend(fontsize=8)
    emit(fig,'single-layer-vs-blue-native.png','Same final10000 requests. Layer-local target changes with layer; BLUE/native references not causal controls. RS/PS/NS denominators10000/20000/100000.')
    for compare in [False,True]:
        fig,axs=plt.subplots(2,3,figsize=(14,8))
        for i,method in enumerate(STYLE['families']):
            for j,tag in enumerate(['RS','PS','NS']):
                ax=axs[i,j]
                for a in ordered(method):
                    color=COLORS[VARIANTS.index(variant(a))];r=sorted([r for r in cum if r['arm']==a and r['metric']==tag],key=lambda r:int(r['batch']))
                    ax.plot([int(x['batch'])*100 for x in r],[float(x['rate'])*100 for x in r],'-o',ms=3,color=color,label=variant(a))
                    if compare:
                        v=sorted([r for r in cur if r['arm']==a and r['metric']==tag],key=lambda r:int(r['batch']))
                        ax.plot([int(x['batch'])*100 for x in v],[float(x['rate'])*100 for x in v],'--',lw=.7,alpha=.4,color=color)
                ax.set(xlim=(0,10000),ylim=(0,100),title=method+' '+tag,xlabel='Accepted edits',ylabel='Success (%)');ax.legend(fontsize=7,ncol=2)
        emit(fig,'current-vs-cumulative.png' if compare else 'cumulative-rspsns.png','Solid: actual Wk all-seen prefix at12 stored checkpoints, denominator t/2t/10t; dashed if present: current B100 denominator100/200/1000. Lines connect observations, no fabricated intermediate values.')
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for i,method in enumerate(STYLE['families']):
        for j,tag in enumerate(['RS','PS','NS']):
            ax=axs[i,j]
            for a in ordered(method):
                r=sorted([r for r in cum if r['arm']==a and r['metric']==tag],key=lambda x:int(x['batch']));color=COLORS[VARIANTS.index(variant(a))]
                ax.plot([int(x['batch'])*100 for x in r],[float(x['new_nll_prompt_median']) for x in r],color=color,label=variant(a))
                ax.plot([int(x['batch'])*100 for x in r],[float(x['new_nll_prompt_p90']) for x in r],'--',color=color,alpha=.6)
            ax.set(xlim=(0,10000),title=method+' '+tag,xlabel='Accepted edits',ylabel='Target-new NLL (nats/token)');ax.legend(fontsize=7,ncol=2)
    emit(fig,'nll-median-tail.png','Solid median, dashed p90 of prompt-level mean-token NLL; all-seen t/2t/10t prompts, not request-cluster quantiles. p99 and true/margin in tables.')
    fig,axs=plt.subplots(2,7,figsize=(20,7),sharey=True)
    for i,method in enumerate(STYLE['families']):
        for j,a in enumerate(ordered(method)):
            rr=[r for r in cohort if r['arm']==a and r['metric']=='RS'];batches=sorted({int(r['batch']) for r in rr});matrix=np.full((100,len(batches)),np.nan)
            for r in rr:matrix[int(r['cohort'])-1,batches.index(int(r['batch']))]=float(r['rate'])*100
            ax=axs[i,j];ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',vmin=0,vmax=100,cmap='viridis',origin='lower');ax.set_title(method+' '+variant(a));ax.set_xticks([0,5,11],[str(batches[k]*100) for k in [0,5,11]],rotation=60);ax.set_xlabel('State edits')
            if j==0:ax.set_ylabel('Cohort B1–B100')
    emit(fig,'cohort-retention-heatmap.png','Each pixel RS success/100 of a fixed edit cohort at recorded Wk; color0–100%. Future cohorts masked (not evaluated), no imputation.12 columns,100 rows/arm.')
    fig,axs=plt.subplots(2,1,figsize=(13,8))
    for ax,method in zip(axs,STYLE['families']):weight_panel(ax,layer,method)
    fig.suptitle('Layer-wise Update Magnitude')
    emit(fig,'layer-wise-update-magnitude.png','100 batch-net stored-weight updates per selected layer, arithmetic mean Frobenius norm. Missing layers unedited, no uniform/ideal line. Magnitude is not native metric action or post-write activation realization.')
    fig,axs=plt.subplots(2,1,figsize=(13,8))
    for ax,method in zip(axs,STYLE['families']):
        names=ordered(method);x=np.arange(7);bottom=np.zeros(7)
        for key,color in [('target_seconds','#0072B2'),('key_seconds','#009E73'),('solve_seconds','#E69F00'),('evaluation_seconds','#CC79A7')]:
            y=np.array([float(next(r for r in cost if r['arm']==a)[key])/3600 for a in names]);ax.bar(x,y,bottom=bottom,label=key.replace('_seconds',''),color=color);bottom+=y
        ax.plot(x,[float(next(r for r in cost if r['arm']==a)['gpu_hours']) for a in names],'ko',label='Allocated GPUh');ax.set(xticks=x,xticklabels=VARIANTS,title=method,ylabel='Hours');ax.legend(fontsize=8,ncol=5)
    emit(fig,'cost-breakdown.png','Source-recorded target/key/solve/evaluation wall subsets vs Slurm allocated GPUh. Unseparated load/snapshot/guards excluded from stack; not controlled speedup.')
    with (destination/'plot-receipt.json').open('x') as f:
        import json
        json.dump(dict(style=STYLE,matplotlib=matplotlib.__version__,source_sha256=sha(__file__),inputs=[dict(path=str(p),sha256=sha(p)) for p in inputs],figures=receipts,command='OPENBLAS_NUM_THREADS=1 /data/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.server4_experiments_review.plots --out <new-output-directory>'),f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT/'figures');main(p.parse_args().out)
