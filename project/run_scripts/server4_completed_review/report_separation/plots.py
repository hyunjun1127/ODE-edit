"""Two family-only figures; existing scientific observations are not recomputed."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import CAKE,readcsv,save,sha,ref

def run(output):
    output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    inputs=['family-final-comparison.csv','w0-reference.csv','baseline-cumulative.csv','seen-prefix.csv']
    family=readcsv(CAKE/inputs[0]);w0={x['metric']:x for x in readcsv(CAKE/inputs[1])}
    fig,axes=plt.subplots(2,3,figsize=(18,10))
    for i,name in enumerate(('AlphaEdit','MEMIT')):
        rr=[r for r in family if r['method']==name]
        for j,m in enumerate(('RS','PS','NS')):
            ax=axes[i,j];xx=[100*float(r[m+'_rate']) for r in rr]
            ax.barh(range(len(rr)),xx,color=['#D55E00' if r['arm']=='CAKE_NATIVE' else '#0072B2' for r in rr])
            ax.axvline(100*float(w0[m]['rate']),color='#555',linestyle='--',label='Pre-edit W0, same 10k')
            ax.set(yticks=range(len(rr)),yticklabels=[r['arm'] for r in rr],xlim=(0,104),xlabel='Canonical success (%)',title=name+' / '+m)
            ax.invert_yaxis();ax.legend(loc='lower right',fontsize=8)
    fig.suptitle('Actual W100 / fixed10k: CAKE and native / BLUE-style baselines\nCAKE repeated as one reference run, never pooled twice')
    fig.tight_layout();fig.savefig(output/'cake-family-comparison.png',dpi=150,metadata={'Software':'report-separation CSV-only v2'});plt.close(fig)
    curve=readcsv(CAKE/inputs[2]);cake=readcsv(CAKE/inputs[3])
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for i,family_name in enumerate(('AlphaEdit','MEMIT')):
        names=list(dict.fromkeys(r['arm'] for r in family if r['method']==family_name and r['arm']!='CAKE_NATIVE'))
        for j,m in enumerate(('RS','PS','NS')):
            ax=axes[i,j]
            for name in names:
                rr=sorted([r for r in curve if r['arm']==name and r['metric']==m],key=lambda x:int(x['batch']))
                ax.plot([int(r['batch']) for r in rr],[100*float(r['rate']) for r in rr],label=name,linewidth=1.2)
            rr=[r for r in cake if r['population']=='ACTUAL_FULL_SEEN' and r['metric']==m]
            ax.plot([int(r['batch']) for r in rr],[float(r['percent']) for r in rr],color='black',label='CAKE_NATIVE',linewidth=2,marker='o',markersize=3)
            ax.set(title=family_name+' / '+m,xlabel='Batch (B100)',ylabel='Actual full-seen success (%)');ax.grid(alpha=.2)
            if j==0:ax.legend(fontsize=7,loc='lower left')
    fig.suptitle('Existing actual checkpoint observations only; increasing seen-prefix population')
    fig.tight_layout();fig.savefig(output/'family-cumulative.png',dpi=150,metadata={'Software':'report-separation CSV-only v2'});plt.close(fig)
    result=dict(generator=ref(Path(__file__)),matplotlib=matplotlib.__version__,
        inputs=[ref(CAKE/name) for name in inputs],outputs=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(output.glob('*.png'))],
        model_forwards=0,raw_access=False)
    save(output/'generated-plot-manifest.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    print(run(p.parse_args().output))
