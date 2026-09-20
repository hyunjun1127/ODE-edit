"""Deterministic plots of observed batch endpoints only (no new evaluation)."""
import argparse
import csv
import hashlib
import tempfile
from pathlib import Path


def plot(package):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    package=Path(package)
    rows=list(csv.DictReader((package/'independent-metrics.csv').open()))
    colors={'N4':'#333333','EN_EXACT':'#31688e','EN_ADAPT':'#b24b25','EN_NUM':'#55964e'}
    figures=[]
    for field,label,filename in [('percent','NLL preference (%)','observed-preference.png'),
                                 ('tf_token_micro','Desired-token TF accuracy (%)','observed-tf-accuracy.png')]:
        fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharex=True)
        for ax,family in zip(axes,['RS','PS','NS']):
            for arm,color in colors.items():
                points=sorted((int(r['batch']),float(r[field])*(100 if field=='tf_token_micro' else 1))
                    for r in rows if r['family']==family and r['arm']==arm and r['scope']=='all_seen' and r[field])
                if points:
                    x,y=zip(*points)
                    ax.plot(x,y,marker='o',label=arm,color=color,linestyle='none' if arm=='EN_NUM' else '--',linewidth=1)
            ax.set_title(family);ax.set_xticks([1,2,3]);ax.set_ylim(0,101);ax.grid(alpha=.2);ax.set_xlabel('Actual completed batch')
        axes[0].set_ylabel(label);axes[-1].legend(fontsize=8)
        fig.suptitle('Fixed-order B100 × 3: observed all-seen endpoints; EN_NUM is B1 only',fontsize=10)
        fig.tight_layout()
        target=package/filename;fig.savefig(target,dpi=160,metadata={'Software':'EN adaptive S4 CPU plot'})
        with tempfile.TemporaryDirectory() as temporary:
            repeated=Path(temporary)/filename;fig.savefig(repeated,dpi=160,metadata={'Software':'EN adaptive S4 CPU plot'})
            if hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(repeated.read_bytes()).digest():raise ValueError('PNG_REPRODUCTION_MISMATCH')
        plt.close(fig);figures.append(filename)
    return figures


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('package');print(plot(p.parse_args().package))
