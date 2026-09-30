"""CPU-only supplemental arithmetic, reading completed B010 JSON only."""
from pathlib import Path
from statistics import mean
from .review_b010 import ROOT, PUB, ARMS, MILESTONES, csvsave, plot
from .common import read, record, save
import tempfile
import shutil

def run():
    rows=[];layer=[];guard=[];members=[]
    def load(p):
        members.append(record(p));return read(p)
    for arm in ARMS:
        d=ROOT/'output'/('B010-'+arm)
        for b in (0,)+MILESTONES:
            rr=load(d/'entry-metrics.json' if b==0 else d/f'B{b:03d}/fixed-post.json')
            for role in ('base_observer','history_observer'):
                a=[r for r in rr if r['role']==role and r['label']==('true' if role=='base_observer' else 'new')]
                rows.append(dict(arm=arm,step=b,role=role,rows=len(a),mean_desired_nll=mean(r['nll'] for r in a),mean_w0_kl=mean(r['w0_kl'] for r in a) if role=='base_observer' else 'NOT_APPLICABLE'))
        for b in range(1,101):
            c=load(d/f'B{b:03d}/commit.json')
            layer.extend(dict(arm=arm,step=b,**v) for v in c['layer_delta'])
            if arm=='NATIVE':continue
            s=load(d/f'B{b:03d}/selection.json');sel=s['selected'];res=sel['residuals'];bounds=s['bounds']
            entry=load(d/f'B{b:03d}/solver/entry-objective.json')
            guard.append(dict(arm=arm,step=b,selected_round=sel['round'],selected_energy=sel['energy'],
                base_bound=bounds['base'],base_kl=res['base']+bounds['base'],base_residual=res['base'],
                max_edit_residual=max(v for k,v in res.items() if k.startswith('edit:')),
                max_preference_residual=max(v for k,v in res.items() if k.startswith('preference:')),
                worst_history_residual=max(v for k,v in res.items() if k.startswith('history:')),
                history_constraints=len(bounds['history']),basis_ranks=','.join(str(v['rank']) for v in entry['geometry']),
                max_raw_solve_residual=max(v['raw_solve_relative_residual'] for v in entry['geometry'])))
    for name,r in [('observer_kl',rows),('layer_actions',layer),('selected_constraints',guard)]:csvsave(PUB/(name+'.csv'),r)
    with tempfile.TemporaryDirectory(prefix='b010-plot-repro-') as td:
        t=Path(td);shutil.copyfile(PUB/'metrics.csv',t/'metrics.csv');plot(t)
        exact=(t/'b010-curves.png').read_bytes()==(PUB/'b010-curves.png').read_bytes()
    assert exact
    save(PUB/'supplement-verification.json',dict(input_members=members,plot_byte_exact_reproduction=exact,
        model_forward=0,source=record(__file__),boundary='선정 제약은 실제 accepted 물리 상태의 기존 기록. 행렬/모델 재실행 없음.'))
    print('observer_final',[r for r in rows if r['step']==100])
    print('selected_final',[r for r in guard if r['step']==100])

if __name__=='__main__':run()
