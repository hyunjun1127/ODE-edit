"""Deterministic Python-only publication PNG, no model or image-generation tools."""
import argparse
import csv
from pathlib import Path


def generate(package, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':120,'savefig.dpi':160})
    fig, axes=plt.subplots(2,3,figsize=(13,7),sharey=True)
    colors=['#717171','#4477aa','#228833','#cc6677']
    for row,family in enumerate(('MEMIT','AlphaEdit')):
        data=list(csv.DictReader((package/f'final-{family}-with-W0.csv').open()))
        for col,metric in enumerate(('RS','PS','NS')):
            ax=axes[row,col];ys=[100*float(r[metric+'_rate']) for r in data]
            ax.bar(range(4),ys,color=colors)
            ax.axhline(ys[0],color='#555555',linestyle='--',linewidth=.8)
            ax.set_xticks(range(4),['W0','BLUE L4+L8','BLUE L4','BLUE L8'],rotation=15)
            ax.set_ylim(0,110);ax.set_title(f'{family}: {metric}');ax.set_ylabel('Canonical NLL preference (%)')
            for i,y in enumerate(ys):ax.text(i,y+1.5,f'{y:.3f}',ha='center',fontsize=9)
    fig.suptitle('Full fixed 10,000 requests: shared W0 vs final W100\nW0 evaluated once; same prompt identities; cross-GPU numeric parity not tested')
    fig.tight_layout(rect=(0,0,1,.92))
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    fig.savefig(destination,metadata={'Software':'ODE-edit fixed10k_preedit_blue_report.plots'})
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();generate(a.package,a.destination)
