"""Reproducible CSV-only CAKE figures; no raw/source/model access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def run(package,output):
    p,o=Path(package),Path(output);o.mkdir(parents=True,exist_ok=False);inputs=[]
    def rows(name):
        inputs.append(name)
        with (p/name).open() as f:return list(csv.DictReader(f))
    def nums(x,k):return [float(v[k]) for v in x]
    def finish(fig,name):
        fig.tight_layout();fig.savefig(o/name,dpi=150,metadata={'Software':'CAKE CPU CSV review'});plt.close(fig)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    seen=rows('seen-prefix.csv');current=rows('batch-current.csv');fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,m in zip(axs,('RS','PS','NS')):
        full=[x for x in seen if x['population']=='ACTUAL_FULL_SEEN' and x['metric']==m]
        cur=[x for x in current if x['metric']==m]
        ax.plot(nums(cur,'batch'),nums(cur,'percent'),color='#999999',alpha=.65,label='Current100')
        ax.plot(nums(full,'batch'),nums(full,'percent'),'-o',color='#0072B2',markersize=4,label='Actual full-seen')
        ax.set(title=m,xlabel='B100 batch',ylabel='Canonical success (%)');ax.grid(alpha=.2)
    axs[0].legend();fig.suptitle('CAKE: current vs actual all-seen, distinct populations')
    finish(fig,'cake-fullseen.png')
    cohort=rows('cohort-retention.csv');fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,m in zip(axs,('RS','PS','NS')):
        rr=[x for x in cohort if x['metric']==m];x=nums(rr,'cohort');d=np.array(nums(rr,'denominator'))
        ax.plot(x,100*np.array(nums(rr,'before_num'))/d,color='#777777',label='Own at-write')
        ax.plot(x,100*np.array(nums(rr,'after_num'))/d,color='#D55E00',label='Actual W100')
        ax.set(title=m,xlabel='Write cohort B100',ylabel='Canonical success (%)');ax.grid(alpha=.2)
    axs[0].legend();fig.suptitle('Cohorts at own write and at final W100; B100 has zero future batches')
    finish(fig,'cake-cohort-retention.png')
    fam=rows('family-final-comparison.csv');fig,axs=plt.subplots(2,3,figsize=(14,9))
    for i,method in enumerate(('AlphaEdit','MEMIT')):
        rr=[x for x in fam if x['method']==method];labels=[x['arm'].replace('AlphaEdit_','AE_').replace('MEMIT_','ME_') for x in rr]
        for j,m in enumerate(('RS','PS','NS')):
            v=100*np.array(nums(rr,m+'_rate'));ax=axs[i,j];ax.barh(range(len(rr)),v,color=['#D55E00' if x['arm']=='CAKE_NATIVE' else '#0072B2' for x in rr])
            ax.set(yticks=range(len(rr)),yticklabels=labels,xlabel='Success (%)',xlim=(0,104),title=method+' / '+m)
            ax.invert_yaxis()
    fig.suptitle('Same fixed10k final populations; source/hparam differences retained')
    finish(fig,'cake-family-comparison.png')
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    m=dict(generator_sha256=sha(Path(__file__)),matplotlib=matplotlib.__version__,numpy=np.__version__,
      inputs=[dict(path=n,sha256=sha(p/n)) for n in inputs],outputs=[dict(path=q.name,sha256=sha(q),bytes=q.stat().st_size) for q in sorted(o.glob('*.png'))],
      raw_access=False,method='CSV_PYTHON_MATPLOTLIB_ONLY')
    with (o/'plot-manifest.json').open('x') as f:json.dump(m,f,indent=2);f.write('\n')
    return m

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(run(a.package,a.output)))
