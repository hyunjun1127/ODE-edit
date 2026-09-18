"""Deterministic static scientific PNGs from published sealed CSV only."""
import argparse
import csv
import io
import platform
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from review import sha, dump


def render(root):
    rows=list(csv.DictReader((root/'final-eight-arm-table.csv').open()))
    labels=[r['arm'] for r in rows];x=np.arange(len(rows));images={}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':120,'savefig.dpi':120})
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    for m in ['RS','PS','NS']:
        axes[0].plot(x,[100*float(r[m+'_n'])/float(r[m+'_d']) for r in rows],marker='o',label=m)
    axes[0].set(xticks=x,xticklabels=labels,ylabel='Success (%)',ylim=(84,101),title='B1: same 100 requests; RS/PS/NS denominators 100/200/1000')
    axes[0].legend(ncol=3);axes[0].grid(alpha=.2)
    for m in ['S64_KL','Dev128_KL']:
        base=float(rows[0][m]);axes[1].plot(x,[100*(float(r[m])/base-1) for r in rows],marker='o',label=m)
    axes[1].set(xticks=x,xticklabels=labels,ylabel='Change vs N4 (%)',title='Observed output KL; S64 is not EN-COV optimization loss')
    axes[1].legend();axes[1].grid(alpha=.2)
    b=io.BytesIO();fig.savefig(b,format='png',metadata={'Software':'ENFC CPU review code'});plt.close(fig);images['endpoint-observations.png']=b.getvalue()
    t=list(csv.DictReader((root/'trials.csv').open()))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for a in ['EN-F','EN-F4','EN-COV']:
        rr=[r for r in t if r['arm']==a]; start=float(rr[0]['current_loss'])
        axes[0].plot(range(1,len(rr)+1),[float(r['loss'])/start for r in rr],'-o',label=a)
        for i,r in enumerate(rr):
            if r['accepted']=='True':axes[0].scatter(i+1,float(r['loss'])/start,s=85,facecolors='none',edgecolors='black')
    axes[0].set(xlabel='Attempted trial index',ylabel='Arm objective / initial objective',title='Stored trials (outlined = accepted)');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].bar(x,[float(r['correction_norm']) for r in rows]);axes[1].set(xticks=x,xticklabels=labels,ylabel='Frobenius norm of W - WN',yscale='symlog',title='Actual FP32 correction magnitude')
    axes[1].tick_params(axis='x',rotation=35)
    b=io.BytesIO();fig.savefig(b,format='png',metadata={'Software':'ENFC CPU review code'});plt.close(fig);images['correction-trials.png']=b.getvalue()
    return images


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    one,two=render(a.root),render(a.root)
    assert one==two,'PNG_BYTE_REPRODUCTION_FAILURE'
    for name,data in one.items():(a.root/name).write_bytes(data)
    dump(a.root/'plot-reproduction.json',dict(command=' '.join(sys.argv),python=platform.python_version(),
        matplotlib=matplotlib.__version__,numpy=np.__version__,backend='Agg',font='DejaVu Sans',
        input_sha={n:sha(a.root/n) for n in ['final-eight-arm-table.csv','trials.csv']},
        output_sha={n:sha(a.root/n) for n in one},independent_render_passes=2,byte_identical=True))


if __name__=='__main__':main()
