"""Recorded risk/margin and correction/retention companions, CPU plots only."""
import argparse
import csv
import hashlib
import io
from pathlib import Path
import platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from .records import save

def render(summary,structures,trajectory,kind):
    summary=[r for r in summary if r.get('reference','') in ['', 'ENTRY']]
    entries=[e for e in ['Early','Middle','Late'] if any(r['entry']==e for r in summary)]
    physical={(r['entry'],r['endpoint']):r for r in structures}
    steps={(r['entry'],r['endpoint']+f"/eval-{int(r['step']):03d}"):r for r in trajectory if r.get('step','') not in ['', '-1']}
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':120,'savefig.dpi':120}):
        fig,axes=plt.subplots(1,len(entries),figsize=(5*len(entries),4),squeeze=False)
        for ax,entry in zip(axes[0],entries):
            if kind=='refresh':
                for arm in ['Continue','FrozenGlobal','RefreshedGlobal','RefreshedLocal','SoftGlobal']:
                    group=sorted([r for r in trajectory if r['entry']==entry and r['endpoint']==arm],key=lambda r:int(r['step']))
                    if group:ax.plot([int(r['step']) for r in group],[float(r['edit_nll']) for r in group],marker='o',label=arm)
                ax.set_xlabel('Observed optimizer step');ax.set_ylabel('Six-context current NLL')
                if ax.lines:ax.legend(fontsize=7)
            else:
                for panel,color in [('Current100','#2878b5'),('Fixed100','#c95c25'),('Past100','#3c9166')]:
                    points=[]
                    for r in summary:
                        if r['entry']!=entry or r['panel']!=panel or r['metric']!=('NS' if kind=='risk' else 'RS'):continue
                        endpoint=(entry,r['endpoint'])
                        if kind=='risk':
                            post=physical.get(endpoint);base=physical.get((entry,'ENTRY-full'))
                            if post is None or base is None:continue
                            points.append((.5*(float(post['global_frobenius_sq'])-float(base['global_frobenius_sq'])),float(r['additional_margin_mean'])))
                        else:
                            step=steps.get(endpoint)
                            if step and step.get('correction_norm','')!='':points.append((float(step['correction_norm']),float(r['new_nll_delta_mean'])))
                    if points:ax.scatter(*zip(*points),label=panel,color=color,s=25)
                ax.set_xlabel('Global Frobenius risk change from entry' if kind=='risk' else 'Actual physical correction norm at observed step')
                ax.set_ylabel('Neighborhood margin change from entry' if kind=='risk' else 'Rewrite target-new NLL change from entry')
                if ax.collections:ax.legend(fontsize=8)
            ax.set_title(entry);ax.grid(alpha=.2)
        fig.suptitle('Observed mechanism diagnostics — no causal or safety claim');fig.tight_layout()
        data=io.BytesIO();fig.savefig(data,format='png',metadata={'Software':'ODE-edit cumulative-risk plotting'})
        plt.close(fig);return data.getvalue()

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--structures',type=Path,required=True)
    p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();inputs=[];tables=[]
    for path in [a.summary,a.structures,a.trajectory]:
        raw=path.read_bytes();inputs.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
        tables.append(list(csv.DictReader(io.StringIO(raw.decode()))))
    a.output.mkdir(parents=True,exist_ok=False);outputs=[]
    has_c=any(r.get('correction_norm','')!='' for r in tables[2])
    plans=[(resolution,kind) for resolution in ['curve','full'] for kind in ['risk']+(['correction'] if has_c else [])]
    if has_c:plans.append(('all','refresh'))
    for resolution,kind in plans:
        selected=[r for r in tables[0] if resolution=='all' or r['resolution']==resolution]
        if not selected:continue
        first=render(selected,*tables[1:],kind);second=render(selected,*tables[1:],kind)
        assert first==second,'PNG_BYTE_REPRODUCTION_FAILURE'
        path=a.output/f'{resolution}-{kind}.png'
        with path.open('xb') as f:f.write(first)
        outputs.append(dict(path=path.name,sha256=hashlib.sha256(first).hexdigest(),bytes=len(first)))
    save(a.output/'plot-reproduction.json',dict(inputs=inputs,outputs=outputs,byte_reproduction=True,
         command=f'python -m project.run_scripts.single_layer_cumulative_risk.mechanism_plotting --summary {a.summary} --structures {a.structures} --trajectory {a.trajectory} --output {a.output}',
         python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,image_tools=0))

if __name__=='__main__':main()
