"""Deterministic code-generated PNGs from numeric CSV, no image tools."""
import argparse
import csv
import io
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS=['N','OS','BF1','BF8','Frozen-BF8']
COLORS=['#777777','#2878b5','#df8c2e','#2b9366','#9b59b6']

def rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))

def render(root,kind):
    summary=rows(root/'endpoint-summary.csv');trajectory=rows(root/'trajectory.csv')
    harms=rows(root/'base-preservation-versus-recovery.csv');compute=rows(root/'compute-ledger.csv')
    entries=[e for e in ('Early','Middle','Late') if any(r['entry']==e and int(r['batch_raw'])==100 for r in summary)]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':120,'savefig.dpi':120}):
        if kind=='endpoint':
            fig,axes=plt.subplots(3,len(entries),figsize=(5*len(entries),9),squeeze=False)
            for col,e in enumerate(entries):
                for row,metric in enumerate(('RS','PS','NS')):
                    ax=axes[row,col];rr=[r for r in summary if r['entry']==e and int(r['batch_raw'])==100 and r['panel']=='Current100' and r['reference']=='N' and r['metric']==metric]
                    by={r['arm']:r for r in rr};arms=[a for a in ARMS if a in by]
                    ax.bar(arms,[float(by[a]['rate']) for a in arms],color=[COLORS[ARMS.index(a)] for a in arms])
                    ax.set_ylim(0,105);ax.set_ylabel(metric+' (%)');ax.set_title(e+' — Current100')
                    ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
            fig.suptitle('Observed fixed endpoints — no success-based selection')
        elif kind=='paired':
            fig,axes=plt.subplots(2,len(entries),figsize=(5*len(entries),6),squeeze=False)
            for j,e in enumerate(entries):
                rr=[r for r in summary if r['entry']==e and r['batch_raw']=='100' and r['panel']=='Current100' and r['reference']=='N']
                by={(r['arm'],r['metric']):r for r in rr};arms=[a for a in ARMS[1:] if (a,'RS') in by]
                x=np.arange(len(arms));width=.24
                for k,m in enumerate(('RS','PS','NS')):
                    axes[0,j].bar(x+(k-1)*width,[float(by[a,m]['delta_pp']) for a in arms],width,label=m)
                for m,marker in (('RS','o'),('PS','s')):
                    axes[1,j].plot(x,[float(by[a,m]['paired_new_nll_delta_mean']) for a in arms],marker=marker,label=m)
                for i in range(2):
                    axes[i,j].axhline(0,color='black',lw=.7);axes[i,j].set_xticks(x,arms,rotation=25)
                    axes[i,j].set_title(e);axes[i,j].grid(axis='y',alpha=.2);axes[i,j].legend(fontsize=8)
                axes[0,j].set_ylabel('Success difference from N (pp)')
                axes[1,j].set_ylabel('Mean target-new NLL difference from N')
            fig.suptitle('Paired Current differences — success and NLL are distinct')
        elif kind=='harm':
            fig,axes=plt.subplots(1,len(entries),figsize=(5*len(entries),4),squeeze=False)
            for ax,e in zip(axes[0],entries):
                rr=[r for r in harms if r['entry']==e and int(r['batch_raw'])==100 and r.get('controller_value','')!='']
                values={(r['arm'],r['bank']):float(r['controller_value']) for r in rr}
                for a,c,marker in zip(ARMS,COLORS,('o','s','^','D','x')):
                    if (a,'Past') not in values:continue
                    ax.scatter(values[(a,'Past')],values[(a,'Base')],c=c,s=45,label=a,marker=marker)
                ax.set_xlabel('Past context harm');ax.set_ylabel('Base KL from We');ax.set_title(e);ax.grid(alpha=.2);ax.legend(fontsize=8)
            fig.suptitle('Controller-bank harm, distinct from held-out locality')
        elif kind=='trajectory':
            import json
            fig,axes=plt.subplots(2,len(entries),figsize=(5*len(entries),6),squeeze=False)
            for j,e in enumerate(entries):
                for a,c in zip(ARMS,COLORS):
                    rr=sorted([r for r in trajectory if r['entry']==e and int(r['batch_raw'])==100 and r['arm']==a and r.get('normalized_actual')],key=lambda r:int(r['node']))
                    if not rr:continue
                    xx=[float(r['s_next']) for r in rr];yy=[json.loads(r['normalized_actual']) for r in rr]
                    for i,bank in enumerate(('Past','Base')):
                        axes[i,j].plot(xx,[v[i] for v in yy],'-o',markersize=3,color=c,label=a)
                        axes[i,j].set_title(e+' '+bank);axes[i,j].set_xlabel('s');axes[i,j].set_ylabel('Observed normalized harm');axes[i,j].grid(alpha=.2)
                axes[0,j].legend(fontsize=8)
            fig.suptitle('Saved nodes only; connecting lines are visual guides')
        elif kind=='compute':
            fig,axes=plt.subplots(1,2,figsize=(13,4))
            names=sorted({r['entry']+' B'+r['batch_raw'] for r in compute})
            seconds=[];backward=[]
            for name in names:
                rr=[r for r in compute if r['entry']+' B'+r['batch_raw']==name]
                seconds.append(sum(float(r['value']) for r in rr if r['category']=='seconds'))
                backward.append(sum(float(r['value']) for r in rr if r['category']=='counts' and r['component']=='functional_backward'))
            axes[0].bar(names,seconds);axes[1].bar(names,backward,color='#2b9366')
            axes[0].set_ylabel('Summed scoped components (s), not GPU allocation');axes[1].set_ylabel('Functional backward calls')
            for ax in axes:ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
            fig.suptitle('Measured compute; supplemental evaluation and scheduler ledger separate')
        else:raise ValueError(kind)
        fig.tight_layout();buf=io.BytesIO();fig.savefig(buf,format='png',metadata={'Software':'ODE-edit L4 two-memory v2 plotting'})
        plt.close(fig);return buf.getvalue()

def main(args):
    root=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for kind in ('endpoint','paired','harm','trajectory','compute'):
        with (out/f'{kind}.png').open('xb') as f:f.write(render(root,kind))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);main(p.parse_args())
