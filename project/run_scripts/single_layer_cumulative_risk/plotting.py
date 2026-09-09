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

def marker(endpoint):
    if endpoint.startswith('B-alpha'):return 'o'
    if endpoint.startswith('C-alpha'):return '^'
    if endpoint.startswith('native-scale'):return 's'
    if endpoint.startswith('N-') or endpoint=='N_REUSED':return 'X'
    if endpoint.startswith('W0'):return 'D'
    if endpoint.startswith('ENTRY'):return 'P'
    return 'v'

def render(rows,kind):
    entries=[x for x in ['Early','Middle','Late'] if any(r['entry']==x for r in rows)]
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':120,'savefig.dpi':120}):
        fig,axes=plt.subplots(1,len(entries),figsize=(5*len(entries),4),squeeze=False)
        for ax,entry in zip(axes[0],entries):
            per={}
            for r in rows:
                if r['entry']==entry:per[(r['endpoint'],r['panel'],r['metric'])]=r
            for panel,color in [('Current100','#2878b5'),('Fixed100','#c95c25'),('Past100','#3c9166')]:
                labeled=False
                for endpoint in sorted({r['endpoint'] for r in rows if r['entry']==entry}):
                    edit=per.get((endpoint,'Current100','RS'));ns=per.get((endpoint,panel,'NS'))
                    if not edit or not ns:continue
                    x=float(edit['new_nll_mean']) if kind=='nll' else 100*float(edit['rate'])
                    y=float(ns['additional_margin_mean']) if kind=='nll' else 100*float(ns['rate'])
                    ax.scatter(x,y,s=25,alpha=.8,color=color,marker=marker(endpoint),label=panel if not labeled else None)
                    labeled=True
            ax.set_title(entry+' — observed endpoints')
            ax.set_xlabel('Current rewrite target-new NLL' if kind=='nll' else 'Current RS (%)')
            ax.set_ylabel('NS margin change from entry' if kind=='nll' else 'NS (%)')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('Cumulative-risk diagnosis — no interpolated endpoints')
        fig.tight_layout();buffer=io.BytesIO()
        fig.savefig(buffer,format='png',metadata={'Software':'ODE-edit cumulative-risk plotting'})
        plt.close(fig);return buffer.getvalue()

def companion(rows,trajectories,kind):
    entries=[e for e in ['Early','Middle','Late'] if any(r['entry']==e for r in rows)]
    train={(r['entry'],r.get('observation_endpoint') or r['endpoint']+f"/eval-{int(r['step']):03d}"):r
           for r in trajectories if r.get('observation_endpoint') or r.get('step','')!=''}
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'figure.dpi':120,'savefig.dpi':120}):
        fig,axes=plt.subplots(1,len(entries),figsize=(5*len(entries),4),squeeze=False)
        for ax,entry in zip(axes[0],entries):
            per={(r['endpoint'],r['panel'],r['metric']):r for r in rows if r['entry']==entry}
            for endpoint in sorted({r['endpoint'] for r in rows if r['entry']==entry}):
                ps=per.get((endpoint,'Current100','PS'));rs=per.get((endpoint,'Current100','RS'))
                if kind=='rs_ps_ns':
                    ns=per.get((endpoint,'Current100','NS'))
                    if ps and rs and ns:
                        ax.scatter(100*float(rs['rate']),100*float(ns['rate']),c=[100*float(ps['rate'])],
                             vmin=0,vmax=100,cmap='viridis',s=26,marker=marker(endpoint))
                elif kind=='ps_retention':
                    if not ps:continue
                    for metric,point_marker in [('RS','o'),('PS','x')]:
                        past=per.get((endpoint,'Past100',metric))
                        if past:ax.scatter(100*float(ps['rate']),100*float(past['loss'])/float(past['denominator']),
                                          marker=point_marker,c='#2878b5' if metric=='RS' else '#c95c25',s=26)
                else:
                    point=train.get((entry,endpoint))
                    if not point:continue # Native has no recorded common train NLL; no substitution.
                    for panel,color in [('Current100','#2878b5'),('Fixed100','#c95c25'),('Past100','#3c9166')]:
                        ns=per.get((endpoint,panel,'NS'))
                        if ns:ax.scatter(float(point['edit_nll']),float(ns['additional_margin_mean']),color=color,s=26)
            ax.set_title(entry);ax.grid(alpha=.2)
            labels={'rs_ps_ns':('Current RS (%)','Current NS (%); color = Current PS (%)'),
                    'ps_retention':('Current PS (%)','Past entry-success→failure / all prompts (%)'),
                    'train_ns':('Observed six-context train NLL','NS margin change from entry')}
            ax.set_xlabel(labels[kind][0]);ax.set_ylabel(labels[kind][1])
        if kind=='rs_ps_ns':
            fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0,100),cmap='viridis'),ax=axes[0].tolist(),label='Current PS (%)',fraction=.025)
        fig.suptitle('Observed companion comparisons; no interpolation or success filtering')
        if kind!='rs_ps_ns':fig.tight_layout()
        buffer=io.BytesIO();fig.savefig(buffer,format='png',metadata={'Software':'ODE-edit cumulative-risk plotting'})
        plt.close(fig);return buffer.getvalue()

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trajectory',type=Path);a=p.parse_args()
    data=a.input.read_bytes();rows=list(csv.DictReader(io.StringIO(data.decode())))
    rows=[r for r in rows if r.get('reference','') in ['', 'ENTRY']]
    # Separate resolutions: curve NS is 2/request and is never labeled full NS.
    a.output.mkdir(parents=True,exist_ok=False);members=[]
    trajectory_data=a.trajectory.read_bytes() if a.trajectory else b''
    trajectories=list(csv.DictReader(io.StringIO(trajectory_data.decode()))) if trajectory_data else []
    for resolution in ['curve','full']:
        selected=[r for r in rows if r['resolution']==resolution]
        if not selected:continue
        for kind in ['nll','rate']:
            first=render(selected,kind);second=render(selected,kind)
            assert first==second,'NONDETERMINISTIC_PLOT_BYTES'
            path=a.output/f'{resolution}-{kind}-locality.png'
            with path.open('xb') as f:f.write(first)
            members.append(dict(path=path.name,sha256=hashlib.sha256(first).hexdigest(),bytes=len(first),byte_reproduction=True))
        has_train=any(r.get('edit_nll','')!='' and (r.get('observation_endpoint') or r.get('step','')!='') for r in trajectories)
        for kind in ['rs_ps_ns','ps_retention']+(['train_ns'] if has_train else []):
            first=companion(selected,trajectories,kind);second=companion(selected,trajectories,kind)
            assert first==second,'NONDETERMINISTIC_PLOT_BYTES'
            path=a.output/f'{resolution}-{kind}.png'
            with path.open('xb') as f:f.write(first)
            members.append(dict(path=path.name,sha256=hashlib.sha256(first).hexdigest(),bytes=len(first),byte_reproduction=True))
    save(a.output/'plot-reproduction.json',dict(input_path=str(a.input),input_sha256=hashlib.sha256(data).hexdigest(),
         command=f'python -m project.run_scripts.single_layer_cumulative_risk.plotting --input {a.input} --output {a.output}'+(f' --trajectory {a.trajectory}' if a.trajectory else ''),
         trajectory_path=str(a.trajectory) if a.trajectory else None,trajectory_sha256=hashlib.sha256(trajectory_data).hexdigest() if trajectory_data else None,
         comparison_reference='ENTRY',marker_key={'W0':'diamond','ENTRY':'filled plus','Native':'X','native scaling':'square','Direct-B':'circle','Direct-C':'triangle','other probes':'down triangle'},
         python=platform.python_version(),matplotlib=matplotlib.__version__,numpy=np.__version__,outputs=members,
         image_tools=0,manual_edit=0))

if __name__=='__main__':main()
