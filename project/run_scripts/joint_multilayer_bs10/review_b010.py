"""CPU-only B010 review. No model import, scheduler mutation, or live-arm reads."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from .common import read, record, save, require, digest
from .reduce import summary as old_summary

ARMS = ('NATIVE', 'JOINT_STEP', 'JOINT_CUM')
MILESTONES = (1, 5, 10, 25, 50, 75, 100)
ROOT = Path('/mnt/raid5/janghj/ODE-edit/local/joint-multilayer-bs1/20260929-v1/attempt-s2-r1')
PUB = Path('experiment-reports/servers/server2/joint-multilayer-bs1-20260929-v1/b010-review-20260930-v1')

def pairs(rows):
    grouped = {}
    for r in rows:
        assert math.isfinite(r['nll']) and r['target_count'] > 0
        assert len(r['token_nll']) == len(r['token_predictions']) == r['target_count']
        assert all(math.isfinite(v) for v in r['token_nll'])
        assert math.isclose(mean(r['token_nll']), r['nll'], abs_tol=2e-5, rel_tol=1e-6)
        assert r['strict'] == (r['token_correct'] == r['target_count'])
        g = grouped.setdefault(r['pair_id'], {})
        assert r['label'] not in g
        g[r['label']] = r
    ans = {}
    for key, g in grouped.items():
        assert set(g) == {'true', 'new'}
        t, n = g['true'], g['new']
        assert all(t[k] == n[k] for k in ('pair_id', 'case_id', 'role', 'kind', 'prompt_index', 'checkpoint'))
        desired_true = t['kind'] == 'N' or t['role'].startswith('base_')
        d = t if desired_true else n
        margin = (n['nll']-t['nll']) if desired_true else (t['nll']-n['nll'])
        ident = digest({label: {k:r[k] for k in ('row_id','input_sha','target_sha','position_sha','case_id','kind','role','prompt_index')} for label,r in g.items()})
        ans[key] = dict(identity=ident, case_id=t['case_id'], panel=t['role']+':'+t['kind'],
            success=margin > 0, margin=margin, true_nll=t['nll'], new_nll=n['nll'],
            strict=d['strict'], token_correct=d['token_correct'], tokens=d['target_count'])
    return ans

def summarize(p):
    grouped = collections.defaultdict(list)
    for r in p.values(): grouped[r['panel']].append(r)
    out=[]
    for panel, rows in sorted(grouped.items()):
        margins=sorted(x['margin'] for x in rows)
        out.append(dict(panel=panel, numerator=sum(x['success'] for x in rows), denominator=len(rows),
            percent=100*sum(x['success'] for x in rows)/len(rows), strict=sum(x['strict'] for x in rows),
            token_correct=sum(x['token_correct'] for x in rows), tokens=sum(x['tokens'] for x in rows),
            mean_true_nll=mean(x['true_nll'] for x in rows), mean_new_nll=mean(x['new_nll'] for x in rows),
            mean_margin=mean(margins), min_margin=min(margins), p10_margin=margins[int(.1*(len(rows)-1))],
            median_margin=margins[len(rows)//2]))
    return out

def transitions(a,b):
    assert set(a)==set(b)
    for k in a: assert a[k]['identity']==b[k]['identity']
    out=[]
    for panel in sorted({r['panel'] for r in a.values()}):
        keys=[k for k in a if a[k]['panel']==panel]
        out.append(dict(panel=panel,denominator=len(keys),before=sum(a[k]['success'] for k in keys),
            after=sum(b[k]['success'] for k in keys),lost=sum(a[k]['success'] and not b[k]['success'] for k in keys),
            gained=sum(not a[k]['success'] and b[k]['success'] for k in keys),
            strict_lost=sum(a[k]['strict'] and not b[k]['strict'] for k in keys),
            strict_gained=sum(not a[k]['strict'] and b[k]['strict'] for k in keys),
            margin_worsened=sum(b[k]['margin'] < a[k]['margin'] for k in keys),
            mean_margin_change=mean(b[k]['margin']-a[k]['margin'] for k in keys)))
    return out

def csvsave(path, rows):
    with Path(path).open('x', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(repo, scratch):
    repo=Path(repo);scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    pub=repo/PUB;pub.mkdir(parents=True,exist_ok=False)
    inventory={};metrics=[];trans=[];steps=[];costs=[];calls=[];snapshots=[];guards=[];traces=[];greedy=[]
    final={};atwrite={};entry={};w0={};terminal={};state0={};catalog0=None
    def load(p):
        r=record(p);inventory[str(p)]=r;obj=read(p);assert record(p)==r;return obj
    lock=load(ROOT/'execution.lock.json');cfg=load(ROOT/'configuration.json')
    assert record(ROOT/'configuration.json')['sha256']==lock['configuration']['sha256']
    expected_ids=cfg['execution_ids'];assert len(set(expected_ids))==len(expected_ids)==100
    def metric(arm, step, scope, rows):
        pp=pairs(rows); ss=summarize(pp)
        old={r['panel']:r for r in old_summary(rows)}
        for s in ss:
            assert (s['numerator'],s['denominator'],s['strict'],s['token_correct'],s['tokens']) == tuple(old[s['panel']][k] for k in ('success','prompts','strict','token_correct','tokens'))
            metrics.append(dict(arm=arm,step=step,scope=scope,**s))
        return pp
    for arm in ARMS:
        d=ROOT/'output'/('B010-'+arm);t=load(d/'terminal.json');terminal[arm]=t
        assert t['status']=='COMPLETED' and t['batches']==t['offered']==100 and t['snapshots']==4
        assert t['source']==lock['source'] and t['config_sha']==lock['configuration']['sha256']
        rt=load(d/'runtime.json'); assert rt['source']==lock['source']
        catalog=load(d/'token-catalog.json')
        cat={r['row_id']:r for r in catalog}
        if catalog0 is None:catalog0=cat
        else:assert cat==catalog0
        state0[arm]=load(d/'entry-state.json'); prev=state0[arm]
        entry[arm]=metric(arm,0,'parent_entry',load(d/'entry-metrics.json'))
        w0[arm]=metric(arm,0,'pretrained_w0',load(d/'w0-metrics.json'))
        written=[];accepted=0;rounderrors=[]
        for b in range(1,101):
            p=d/f'B{b:03d}';e=load(p/'entry.json');c=load(p/'commit.json');s=load(p/'selection.json')
            assert e['state']==prev==c['before'] and c['offered_ids']==[expected_ids[b-1]]==e['offered_ids']
            assert c['selection']['sha256']==inventory[str(p/'selection.json')]['sha256']
            assert c['history_appends']==(5 if s['accepted'] else 0)
            assert c['accepted']==s['accepted']
            if not c['accepted']:
                assert all(c['before'][k]==c['after'][k] for k in ('weight','history','anchors','rng','context','accepted_ids'))
            accepted+=int(c['accepted']);prev=c['after'];assert len(prev['accepted_ids'])==accepted
            rows=load(p/'current.json');assert len(rows)==6
            for r in rows:
                target=cat[r['row_id']]['target_ids']; assert r['token_correct']==sum(a==z for a,z in zip(r['token_predictions'],target))
            written+=rows; metric(arm,b,'current',rows)
            steps.append(dict(arm=arm,step=b,accepted=c['accepted'],outcome=s['outcome'],history_appends=c['history_appends'],
                method_seconds=s['seconds'],geometry_seconds=s.get('geometry_seconds',0),proposals=s.get('proposals',0),
                trials=s.get('trials',0),full_guards=s.get('full_guard_passes',0),
                selected_energy=(s.get('selected') or {}).get('energy','NOT_RECORDED'),
                actual_normalized_energy=sum(x['normalized_energy'] for x in c['layer_delta'])))
            if arm!='NATIVE':
                for f in sorted((p/'solver').glob('round-*.json')):
                    r=load(f);rounderrors.append(r['materialization_max_row_difference'])
                    guards.append(dict(arm=arm,step=b,round=r['round'],feasible=r['feasible'],energy=r['energy'],
                        max_residual=max(r['full_residuals'].values()),base_residual=r['full_residuals']['base'],
                        worst_history=max(v for k,v in r['full_residuals'].items() if k.startswith('history:')),
                        row_parity_error=r['materialization_max_row_difference'],margin_parity_error=r['materialization_max_margin_difference']))
                selected=s.get('selected')
                if selected:assert max(selected['residuals'].values())<=1e-5
            if b in MILESTONES:
                a=metric(arm,b,'all_offered',load(p/'all-offered.json'));assert len(a)==3*b
                trans.extend(dict(arm=arm,step=b,comparison='atwrite_to_same_state',**v) for v in transitions(pairs(written),a))
                fixedrows=load(p/'fixed-post.json');f=metric(arm,b,'fixed_post',fixedrows)
                old={k:v for k,v in entry[arm].items() if '_observer:' in v['panel']}
                trans.extend(dict(arm=arm,step=b,comparison='parent_to_fixed_observer',**v) for v in transitions(old,f))
                metric(arm,b,'fixed_pre',load(p/'fixed-pre.json'))
                metric(arm,b,'own_request_neighborhood',load(p/'neighborhood.json'))
                for role in ('base_observer','history_observer'):
                    for l in range(4,9):
                        vals=[z for r in fixedrows if r['role']==role for z in r['layer_traces'] if z['layer']==l]
                        traces.append(dict(arm=arm,step=b,role=role,layer=l,rows=len(vals),**{k:mean(v[k] for v in vals) for k in ('key_norm','entry_key_drift','entry_readout_change','E_deltaK_norm','actual_delta_norm')}))
                if b==100:final[arm]={**a,**f}
        atwrite[arm]=metric(arm,100,'online_current_pool',written)
        assert accepted==t['accepted']==sum(r['accepted'] for r in t['batch_records'])
        for n in (25,50,75,100):
            f=d/'weights'/('B010-'+arm)/f'T{n:03d}.pt';r=load(f.with_suffix('.receipt.json'));actual=record(f)
            assert actual['bytes']==r['file']['bytes'] and actual['sha256']==r['file']['sha256']
            assert r['exact_logp'] and not r['early_replacement'] and r['tensor_hashes']==read(d/f'B{n:03d}'/'commit.json')['after']['weight']
            inventory[str(f)]=actual;snapshots.append(dict(arm=arm,step=n,**actual,recorded_reload_logp_exact=True,current_gpu_reload=False))
        grow=load(d/'final-greedy.json');assert len(grow)==300
        for kind in ('R','P'):
            rr=[r for r in grow if r['kind']==kind];greedy.append(dict(arm=arm,kind=kind,denominator=len(rr),exact_match=sum(r['exact_match'] for r in rr),censored=sum(r['censored'] for r in rr)))
        ss=[r for r in steps if r['arm']==arm]
        costs.append(dict(arm=arm,program_seconds=t['program_seconds'],method_seconds=sum(r['method_seconds'] for r in ss),geometry_seconds=sum(r['geometry_seconds'] for r in ss),
            proposals=sum(r['proposals'] for r in ss),trials=sum(r['trials'] for r in ss),full_guards=sum(r['full_guards'] for r in ss),
            native_fits=t['scientific_native_fits'],native_loss_evaluations=t['loss_evaluations'],native_adam_updates=t['adam_updates'],
            scorer_backwards=t['backwards'],scorer_backward_seconds=t['backward_seconds'],cuda_peak_bytes=t['cuda_peak_bytes'],host_maxrss_kib=t['host_maxrss_kib']))
        for catname,count in t['calls'].items():calls.append(dict(arm=arm,category=catname,calls=count,tokens=t['tokens'][catname],forward_seconds=t['timers'].get(catname,'NOT_RECORDED')))
        print(arm,'100 commits verified',flush=True)
    assert all(state0[a]==state0['NATIVE'] for a in ARMS)
    for arm in ARMS[1:]:
        trans.extend(dict(arm=arm,step=100,comparison='NATIVE_to_arm_final',**r) for r in transitions(final['NATIVE'],final[arm]))
        assert {k:v['identity'] for k,v in entry[arm].items()}=={k:v['identity'] for k,v in entry['NATIVE'].items()}
    entry_spread=max(abs(entry[a][k]['true_nll']-entry['NATIVE'][k]['true_nll']) for a in ARMS for k in entry[a])
    outputs=dict(metrics=metrics,transitions=trans,steps=steps,compute=costs,calls=calls,snapshots=snapshots,guards=guards,layer_traces=traces,greedy=greedy)
    for name,rows in outputs.items():csvsave(pub/(name+'.csv'),rows)
    save(scratch/'input-members.json',list(inventory.values()))
    save(scratch/'paired-metrics.json',dict(final=final,atwrite=atwrite,entry=entry,w0=w0))
    save(pub/'verification.json',dict(status='CPU_REDUCED',execution_source=lock['source'],config_sha=lock['configuration']['sha256'],
        lock=record(ROOT/'execution.lock.json'),input_inventory=record(scratch/'input-members.json'),input_members=len(inventory),
        ordered_inventory_root=digest(list(inventory.values())),execution_ids_sha=digest(expected_ids),
        arms=list(ARMS),commits=300,snapshots_full_sha_verified=12,identical_parent_state=True,
        parent_true_nll_max_abs_spread=entry_spread,independent_reducer_matches_original=True,
        new_model_forward=0,new_gpu=0,independent_red_agent=False,scientific_promotion=False,
        boundary='원 snapshot GPU reload receipt 재사용. 이번 검산은 CPU 파일 SHA/상태 연결/분모/독립 reducer. NS는 milestone 현재 요청 각10문항이며 최종 전체100요청 NS1000이 아니다.'))
    plot(pub)
    print(json.dumps({'final':[r for r in metrics if r['step']==100 and r['scope'] in ('all_offered','fixed_post')],'compute':costs,'transitions':[r for r in trans if r['step']==100]},ensure_ascii=False),flush=True)

def plot(pub):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rr=list(csv.DictReader((pub/'metrics.csv').open()))
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,(scope,panel) in zip(axes.flat,[('all_offered','continuation:R'),('all_offered','continuation:P'),('fixed_post','base_observer:R'),('fixed_post','history_observer:R')]):
        for arm in ARMS:
            points=[r for r in rr if r['arm']==arm and r['scope']==scope and r['panel']==panel]
            ax.plot([int(r['step']) for r in points],[float(r['percent']) for r in points],marker='o',label=arm)
        ax.set(title=panel,xlabel='Offered edits',ylabel='Strict NLL-pair success (%)',ylim=(-2,102));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(pub/'b010-curves.png',dpi=160,metadata={'Software':'review_b010.py'});plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--scratch',required=True);a=p.parse_args();run(a.repo,a.scratch)
