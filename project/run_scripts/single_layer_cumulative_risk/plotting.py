"""Deterministic code-generated PNG from observed aggregate CSV, never image tools."""
import argparse
import csv
import hashlib
import io
import json
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .records import save

def render(rows,kind):
    entries=[x for x in ['Early','Middle','Late'] if any(r['entry']==x for r in rows)]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':120,'savefig.dpi':120}):
        fig,axes=plt.subplots(1,len(entries),figsize=(5*len(entries),4),squeeze=False)
        for ax,entry in zip(axes[0],entries):
            per={}
            for r in rows:
                if r['entry']==entry:per[(r['endpoint'],r['panel'],r['metric'])]=r
            for panel,color in [('Current100','#2878b5'),('Fixed100','#c95c25'),('Past100','#3c9166')]:
                xs=[];ys=[]
                for endpoint in sorted({r['endpoint'] for r in rows if r['entry']==entry}):
                    edit=per.get((endpoint,'Current100','RS'));ns=per.get((endpoint,panel,'NS'))
                    if not edit or not ns:continue
                    xs.append(float(edit['new_nll_mean']) if kind=='nll' else 100*float(edit['rate']))
                    ys.append(float(ns['additional_margin_mean']) if kind=='nll' else 100*float(ns['rate']))
                ax.scatter(xs,ys,s=25,alpha=.8,color=color,label=panel)
            ax.set_title(entry+' — observed endpoints')
            ax.set_xlabel('Current rewrite target-new NLL' if kind=='nll' else 'Current RS (%)')
            ax.set_ylabel('NS margin change from entry' if kind=='nll' else 'NS (%)')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Cumulative-risk A diagnosis — no interpolated endpoints')
        fig.tight_layout();buffer=io.BytesIO()
        fig.savefig(buffer,format='png',metadata={'Software':'ODE-edit cumulative-risk plotting'})
        plt.close(fig);return buffer.getvalue()

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=a.input.read_bytes();rows=list(csv.DictReader(io.StringIO(data.decode())))
    # Separate resolutions: curve NS is 2/request and is never labeled full NS.
    a.output.mkdir(parents=True,exist_ok=False);members=[]
    for resolution in ['curve','full']:
        selected=[r for r in rows if r['resolution']==resolution]
        if not selected:continue
        for kind in ['nll','rate']:
            first=render(selected,kind);second=render(selected,kind)
            assert first==second,'NONDETERMINISTIC_PLOT_BYTES'
            path=a.output/f'{resolution}-{kind}-locality.png'
            with path.open('xb') as f:f.write(first)
            members.append(dict(path=path.name,sha256=hashlib.sha256(first).hexdigest(),bytes=len(first),byte_reproduction=True))
    save(a.output/'plot-reproduction.json',dict(input_path=str(a.input),input_sha256=hashlib.sha256(data).hexdigest(),
         command=f'python -m project.run_scripts.single_layer_cumulative_risk.plotting --input {a.input} --output {a.output}',
         python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=np.__version__,outputs=members,
         image_tools=0,manual_edit=0))

if __name__=='__main__':main()
