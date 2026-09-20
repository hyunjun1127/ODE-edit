"""Independent stored-NLL reduction and S4 completed/partial review supplement.

No model/torch/evaluator import. Runtime raw is read-only. CSV carries compact
case identities and scalars, not source prompts or teacher/model tensors.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS=('N4','EN_EXACT','EN_NUM','EN_ADAPT')
CHAINS=('N4','EN_EXACT','EN_ADAPT')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csv_dump(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def validate_reduce(obs,case_ids=None):
    ids=obs['request_ids'] if case_ids is None else list(case_ids)
    if len(ids)!=len(set(ids)) or not set(ids).issubset(obs['request_ids']):raise ValueError('REQUEST_IDENTITY')
    selected=set(ids);by={x:[] for x in ('RS','PS','NS')};seen=set()
    for row in obs['rows']:
        if row['case_id'] not in selected:continue
        key=(row['family'],row['case_id'],row['prompt_index'])
        if key in seen:raise ValueError('DUPLICATE_PROMPT')
        seen.add(key)
        a,b=row['new_nll'],row['true_nll']
        if not math.isfinite(a) or not math.isfinite(b):raise ValueError('NONFINITE_NLL')
        family=row['family'];desired='true' if family=='NS' else 'new'
        success=b<a if family=='NS' else a<b
        flags=row[desired+'_token_correct']
        if not flags or any(type(f) is not bool for f in flags):raise ValueError('TOKEN_FLAGS')
        if row[desired+'_token_count']!=len(flags) or row[desired+'_strict']!=all(flags):raise ValueError('TOKEN_DENOMINATOR')
        if row['success']!=success:raise ValueError('STORED_PREFERENCE_MISMATCH')
        by[family].append(dict(row,independent_success=success,independent_strict=all(flags),desired_flags=flags))
    out=[]
    for family,factor in [('RS',1),('PS',2),('NS',10)]:
        rows=by[family];n=len(rows)
        if n!=factor*len(ids):raise ValueError('PROMPT_CARDINALITY:'+family)
        for case in ids:
            if sum(r['case_id']==case for r in rows)!=factor:raise ValueError('PER_CASE_CARDINALITY')
        successes=sum(r['independent_success'] for r in rows);strict=sum(r['independent_strict'] for r in rows)
        flags=[f for r in rows for f in r['desired_flags']]
        out.append(dict(family=family,requests=len(ids),numerator=successes,denominator=n,percent=100*successes/n if n else None,
            strict_numerator=strict,strict_denominator=n,tf_token_correct=sum(flags),tf_token_total=len(flags),
            tf_token_micro=sum(flags)/len(flags) if flags else None,
            tf_prompt_macro=statistics.mean(sum(r['desired_flags'])/len(r['desired_flags']) for r in rows) if rows else None,
            new_nll=statistics.mean(r['new_nll'] for r in rows) if rows else None,
            true_nll=statistics.mean(r['true_nll'] for r in rows) if rows else None))
    return out,by


def transitions(a,b,case_ids):
    _,before=validate_reduce(a,case_ids);_,after=validate_reduce(b,case_ids)
    rows=[]
    for family in before:
        key=lambda r:(r['case_id'],r['prompt_index'],r['identity'],r['token_identity'])
        left={key(r):r for r in before[family]};right={key(r):r for r in after[family]}
        if left.keys()!=right.keys():raise ValueError('PAIRED_IDENTITY')
        for k,x in left.items():
            y=right[k];old=x['independent_success'];new=y['independent_success']
            rows.append(dict(family=family,case_id=k[0],prompt_index=k[1],identity=k[2],
                lost=old and not new,gained=not old and new,
                strict_lost=x['independent_strict'] and not y['independent_strict'],
                strict_gained=not x['independent_strict'] and y['independent_strict'],
                new_nll_delta=y['new_nll']-x['new_nll'],true_nll_delta=y['true_nll']-x['true_nll']))
    return rows


def replay_controller(controller):
    accepted=[];checks=[]
    for row in controller['ledger']:
        j=row['objective']['J'];base=row['native_objective']['J'];slope=row['actual_gradient_inner_product']
        expected=j<base and j<=base+1e-4*slope and slope<0
        checks.append(dict(trial=row['trial'],accepted_saved=row['accepted'],accepted_independent=expected,
            acceptance_equal=expected==row['accepted'],geometry=all(row['geometry_checks'].values())))
        if expected:accepted.append((j,row['geometry']['actual_norm'],row['trial']))
    selected=min(accepted)[2] if accepted else None
    actual=controller.get('selected_trial')
    if selected!=actual or any(not r['acceptance_equal'] or not r['geometry'] for r in checks):raise ValueError('CONTROLLER_REPLAY')
    return dict(selected_trial=selected,status=controller['status'],candidates=len(checks),checks=checks)


def replay_frontier(payload):
    """Scalar-only independent replay; no production selector/tensor imports."""
    s=payload['spectrum'];selection=payload['selection'];results=[]
    cumulative_g=[s['exact_energy']];cumulative_a=[0.]
    added=0.;action=0.
    for eigen,energy in zip(s['eigenvalues'],s['mode_energies']):
        added+=energy;action+=eigen*energy
        cumulative_g.append(s['exact_energy']+added);cumulative_a.append(action)
    for eps,stored in selection['frontiers'].items():
        reconstructed=[]
        for k in [0,*s['group_ends']]:
            g2=cumulative_g[k];a2=cumulative_a[k];g=math.sqrt(g2);a=math.sqrt(a2)
            eta=0. if s['loss']<=selection['loss_floor'] or g==0 or s['native_norm']==0 else min(
                s['loss']/g2,s['native_norm']/g,float(eps)*s['native_action']/a if a else math.inf)
            reconstructed.append(dict(released_modes=k,gradient_energy=g2,eta=eta,
                predicted_decrease=eta*g2,correction_norm=eta*g,response_norm=eta*a))
        if len(stored)!=len(reconstructed):raise ValueError('FRONTIER_CARDINALITY')
        max_error=0.
        for actual,expected in zip(stored,reconstructed):
            if actual['released_modes']!=expected['released_modes']:raise ValueError('FRONTIER_ORDER')
            for key in ('gradient_energy','eta','predicted_decrease','correction_norm','response_norm'):
                max_error=max(max_error,abs(actual[key]-expected[key]))
                if not math.isclose(actual[key],expected[key],rel_tol=1e-12,abs_tol=1e-15):
                    raise ValueError('FRONTIER_ARITHMETIC:'+key)
        remaining=reconstructed
        for key,maximize in [('predicted_decrease',True),('correction_norm',False),('response_norm',False)]:
            values=[r[key] for r in remaining];best=(max if maximize else min)(values)
            tolerance=64*2.220446049250313e-16*max(1.,max(abs(x) for x in values))
            remaining=[r for r in remaining if abs(r[key]-best)<=tolerance]
        chosen=min(remaining,key=lambda r:r['released_modes'])['released_modes']
        if float(eps)==selection['epsilon_primary']:
            expected={'EN_EXACT':0,'EN_NUM':s['numerical_released'],'EN_ADAPT':chosen}
            if any(selection['selected'][arm]['released_modes']!=k for arm,k in expected.items()):
                raise ValueError('FRONTIER_SELECTION')
        results.append(dict(epsilon=float(eps),rows=len(stored),selected_adaptive_modes=chosen,
            scalar_max_abs_difference=max_error,arithmetic_comparison_tolerance='relative1e-12/absolute1e-15; not scientific acceptance'))
    return results


def review(output,destination,first_only=False):
    output=Path(output);dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    tables=[];pairs=[];selection=[];history=[];missing=[];inputs={};observed={};frontiers=[]
    def read(path):
        p=output/path
        if not p.is_file():return None
        inputs[path]=dict(bytes=p.stat().st_size,sha256=digest(p))
        return json.loads(p.read_text())
    for batch in (1,) if first_only else (1,2,3):
        for group in ('SHARED',) if batch==1 else CHAINS:
            spectrum=read(f'B{batch}/{group}-spectrum.json')
            if spectrum:frontiers.append(dict(batch=batch,group=group,checks=replay_frontier(spectrum)))
        w0=read(f'B{batch}/W0-current.json');current=w0['request_ids'] if w0 else None
        for arm in ARMS if batch==1 else CHAINS:
            obs=read(f'B{batch}/{arm}-metrics.json')
            if obs is None:missing.append(f'B{batch}/{arm}');continue
            observed[batch,arm]=obs
            expected=batch*100
            if len(obs['request_ids'])!=expected:raise ValueError('ALL_SEEN_REQUESTS')
            scopes={'all_seen':obs['request_ids'],'current':current,'first100':obs['request_ids'][:100]}
            commit=read(f'B{batch}/{arm}-commit.json')
            if commit:
                scopes['active_past']=commit['active_past_ids']
                history.append(dict(batch=batch,arm=arm,append=commit['history_append'],
                    native_receipt=commit['receipt']['native'],active_past_ids=commit['active_past_ids']))
            for scope,ids in scopes.items():
                if ids is None:continue
                reduced,_=validate_reduce(obs,ids)
                tables.extend(dict(batch=batch,arm=arm,scope=scope,**r) for r in reduced)
            ctrl=read(f'B{batch}/{arm}-controller.json')
            if ctrl:selection.append(dict(batch=batch,arm=arm,**replay_controller(ctrl)))
        native=observed.get((batch,'N4'))
        if native:
            for arm in ARMS if batch==1 else CHAINS:
                obs=observed.get((batch,arm))
                if arm=='N4' or obs is None:continue
                commit=read(f'B{batch}/{arm}-commit.json')
                scopes={'all_seen':obs['request_ids'],'current':current,'first100':obs['request_ids'][:100],
                    'active_past':commit['active_past_ids'] if commit else None}
                for scope,ids in scopes.items():
                    if ids is not None:
                        pairs.extend(dict(batch=batch,arm=arm,scope=scope,comparison='minus_N4',**r) for r in transitions(native,obs,ids))
    first=[r for r in tables if r['batch']==1 and r['scope']=='all_seen']
    csv_dump(dest/'first-final-table.csv',first)
    csv_dump(dest/'independent-metrics.csv',tables)
    csv_dump(dest/'independent-paired.csv',pairs)
    dump(dest/'selector-replay.json',selection);dump(dest/'history-evidence.json',history)
    dump(dest/'frontier-replay.json',frontiers)
    completeness=dict(complete_endpoints=len(observed),expected_endpoints=4 if first_only else 10,
        missing_endpoints=missing,history_appends=sum(r['append'] for r in history),
        numeric_pass=False,precision_status='NOT_ESTABLISHED',checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    dump(dest/'independent-reducer.json',dict(completeness=completeness,inputs=inputs,
        source_sha256=digest(__file__),validation='fresh true/new NLL reduction; token flags identity/cardinality; CPU selector replay',
        first_table_sha256=digest(dest/'first-final-table.csv')))
    print(json.dumps(dict(first_table=first,completeness=completeness),ensure_ascii=False))
    return completeness


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--destination',required=True);p.add_argument('--first-only',action='store_true');a=p.parse_args()
    review(a.output,a.destination,a.first_only)
