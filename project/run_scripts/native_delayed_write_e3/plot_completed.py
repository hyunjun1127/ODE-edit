"""Deterministic code-only figures from completed compact CSV; no interpolation."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .common import sha, save


def csv_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def plots(report, dest):
    report=Path(report);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.hashsalt':'native-delayed-e3-20260924'})
    rows=csv_rows(report/'endpoint-summary.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(panel,kind,label) in zip(axes.flat,[('N_diag1000','N','Neighborhood preference'),
        ('H_diag_B1_R100_P200','R','B1 rewrite preference'),('H_diag_B1_R100_P200','P','B1 rephrase preference'),
        ('BaseEval256','BASE','Base facts preference')]):
        for fam,color in [('BASE_ALPHAEDIT','#0072B2'),('BASE_MEMIT','#D55E00')]:
            rr=[r for r in rows if r['panel']==panel and r['kind']==kind and r['endpoint'].startswith(fam+'_W')]
            rr.sort(key=lambda r:int(r['endpoint'].rsplit('_W',1)[1]))
            # Scatter only: unobserved checkpoints are not rendered as observations.
            ax.scatter([int(r['endpoint'].rsplit('_W',1)[1]) for r in rr],
                       [100*int(r['success'])/int(r['count']) for r in rr],label=fam,color=color,s=28)
        w0=next(r for r in rows if r['endpoint']=='W0' and r['panel']==panel and r['kind']==kind)
        ax.axhline(100*int(w0['success'])/int(w0['count']),color='gray',linestyle=':',label='W0 fixed reference')
        ax.set(title=label,xlabel='Stored endpoint batch',ylabel='Success (%)',ylim=(-2,102));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(dest/'endpoint-preferences.png',dpi=160,metadata={'Software':'native_delayed_write_e3.plot_completed'})
    plt.close(fig)
    patch=csv_rows(report/'path-patch-paired.csv')
    rr=[r for r in patch if r['panel']=='N_diag1000' and r['kind']=='N' and 'LOST_ONLY' not in r['contrast']]
    keys=sorted({r['contrast'].split('/')[0] for r in rr})
    variants=['dose_0p5','dose_1','dose_minus1','rotation_2026092401','rotation_2026092402','rotation_2026092403']
    values=np.array([[float(next(r for r in rr if r['contrast'].startswith(key+'/'+v+' '))['desired_nll_delta']) for v in variants] for key in keys])
    lim=max(float(np.abs(values).max()),1e-9)
    fig,ax=plt.subplots(figsize=(10,6),constrained_layout=True)
    im=ax.imshow(values,cmap='coolwarm',vmin=-lim,vmax=lim,aspect='auto')
    ax.set_xticks(range(6),['dose .5','dose 1','dose -1','rotation 1','rotation 2','rotation 3'])
    ax.set_yticks(range(len(keys)),[x.replace('BASE_ALPHAEDIT','Alpha').replace('BASE_MEMIT','MEMIT') for x in keys])
    ax.set_title('N_diag1000 true-target NLL: patch minus actual11\nMeasured variants only; negative = lower NLL')
    fig.colorbar(im,ax=ax,label='Mean NLL delta')
    fig.savefig(dest/'patch-neighborhood-nll.png',dpi=160,metadata={'Software':'native_delayed_write_e3.plot_completed'})
    plt.close(fig)
    save(dest/'figure-manifest.json',dict(source_sha256=sha(__file__),
        inputs=[dict(path=str(report/n),sha256=sha(report/n)) for n in ('endpoint-summary.csv','path-patch-paired.csv')],
        files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(dest.glob('*.png'))],
        interpolation_of_unmeasured_results=False,model_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();plots(a.report,a.destination)
