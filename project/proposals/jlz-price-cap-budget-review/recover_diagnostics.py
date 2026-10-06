#!/usr/bin/env python3
"""Read-only remote CPU recovery of SHA-sealed PRICE B1..15 diagnostics.

Run from this review worktree with Python's standard library. Only the six
allowlisted scalar files per committed batch are read remotely. No model,
scheduler, source mutation, B16+, or counterfactual execution is involved.
The local output is restricted to the five named compact audit files.
"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

PUBLICATION = '677dad89'
SOURCE = '0415aba3c160170d306be8196792f198dad4d122'
MANIFEST = 'audits/servers/server4/jlz-interference-priced-l1-2k/w15-review/input-manifest.json'
BASE = '/data/janghj/ODE-edit/local/jlz-interference-priced-l1-2k/repair-59721/PRICE'
NAMES = ('entry-price.json', 'fit.json', 'events.jsonl', 'commit.json',
         'writer/realization.json', 'writer/subject-alltoken-gap.json')

REMOTE = r'''
import collections, hashlib, json, pathlib
sealed=__SEALED__
base=__BASE__
source=__SOURCE__
verified=[]
def read(batch,name,events=False):
    path=f'{base}/batch-{batch:02d}/{name}'
    data=pathlib.Path(path).read_bytes(); record=sealed[path]
    assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'],path
    verified.append(record)
    return [json.loads(line)['payload'] for line in data.splitlines()] if events else json.loads(data)
def stats(values):
    v=sorted(x for x in values if x is not None)
    if not v:return dict(n=0,min=None,p50=None,p90=None,p95=None,max=None,mean=None)
    def quantile(p):
        z=(len(v)-1)*p;i=int(z)
        return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(z-i)
    return dict(n=len(v),min=v[0],p50=quantile(.5),p90=quantile(.9),p95=quantile(.95),max=v[-1],mean=sum(v)/len(v))
def counter(values):return dict(sorted(collections.Counter(values).items()))
def category(x):return 'zero' if x==0 else 'tiny' if x<=1e-12 else 'meaningful'
def select(rows,batch='ALL',layer='ALL',stage='ALL'):
    return [x for x in rows if (batch=='ALL' or x['batch']==batch)
            and (layer=='ALL' or x['layer']==layer) and (stage=='ALL' or x['stage']==stage)]
requests=[];projections=[];terminal=[];roles=[];realizations=[];gaps=[]
for batch in range(1,16):
    price=read(batch,'entry-price.json')['payload'];fit=read(batch,'fit.json')
    events=read(batch,'events.jsonl',True);commit=read(batch,'commit.json')
    real=read(batch,'writer/realization.json');gap=read(batch,'writer/subject-alltoken-gap.json')
    assert commit['source']==source and commit['batch']==batch and commit['seen_requests']==batch*100
    assert len(events)==25 and len(price['effective_pi'])==5
    assert fit['candidate_stream']['sha256']==sealed[f'{base}/batch-{batch:02d}/events.jsonl']['sha256']
    assert fit['entry_price_sha256']==price['record_sha256'] and events[-1]['states']==fit['terminal_states']
    term=events[-1];ctrl=term['controller'];realizations.append(dict(batch=batch,record=real))
    gaps.append(dict(batch=batch,subject_F=stats(gap['subject_F']),actual_alltoken_F=stats(gap['actual_alltoken_F'])))
    for r in range(100):
        pi=[price['effective_pi'][i][r] for i in range(5)]
        minimum=min(pi);minlayers=[l for i,l in enumerate(price['layers']) if pi[i]==minimum]
        beta=ctrl['beta'][r];spend=term['telemetry']['weighted_spend'][r]
        requests.append(dict(batch=batch,owner=r,minlayer=minlayers[0],minlayers=minlayers,pi=pi,
            raw=[price['raw_kappa'][i][r] for i in range(5)],floor=[price['floor_mask'][i][r] for i in range(5)],
            pimax=max(pi),beta=beta,stage=ctrl['expansion'][r],state=term['states'][r],F=term['F'][r],
            owner_mean_rewrite_nll=sum(term['nll'][r])/len(term['nll'][r]),KL=term['KL'][r],
            spend_ratio=spend/beta,unused_budget=beta-spend,unused_budget_fraction=(beta-spend)/beta,
            ceiling_fraction_used=(beta-.75)/(.75*max(pi)-.75)))
    for i,l in enumerate(price['layers']):
        anchors=price['anchors'][i];t=term['telemetry']['layers'][str(l)]
        for r in range(100):
            terminal.append(dict(batch=batch,layer=l,owner=r,normalized=t['normalized_norm'][r],
                **{k:t[k][r] for k in ('requested_energy_share','raw_relative_norm_share','priced_spend_share','realized_canonical_energy_share')},
                canonical_cosine=t['canonical']['cosine'][r],canonical_normratio=t['canonical']['normratio'][r]))
        for role in ('mean','canonical','rewrite','KL'):
            obj=real['layers'][str(l)][role]
            for r in range(100):
                v={k:vv[r] for k,vv in obj.items()} if isinstance(obj,dict) else obj[r]
                target=v['target_norm']/anchors[r]
                roles.append(dict(batch=batch,layer=l,role=role,owner=r,category=category(target),target=target,
                    actual=v['actual_norm']/anchors[r],ratio=v['normratio'],cosine=v['cosine']))
        for event in events[:-1]:
            p=event['projection'];after=event['post_update_controller']
            for r in range(100):
                cap=price['local_caps'][i][r];weight=price['effective_pi'][i][r]/anchors[r]
                norm=p['post_norm'][i][r]/anchors[r]
                projections.append(dict(batch=batch,candidate=event['candidate'],layer=l,owner=r,
                    stage=after['expansion'][r],capped=p['capped_mask'][i][r],zero=p['zero_mask'][i][r],free=p['free_mask'][i][r],
                    strict_cap_pressure=p['pre_norm'][i][r]-p['tau'][r]*weight-cap>1e-10*max(1,cap,p['pre_norm'][i][r]),
                    shared=p['shared_active'][r],norm=norm,category=category(norm)))
price_rows=[];controller_rows=[];cap_rows=[]
for batch in list(range(1,16))+['ALL']:
    req=[x for x in requests if batch=='ALL' or x['batch']==batch]
    for i,l in enumerate([4,5,6,7,8]):
        row=dict(batch=batch,layer=l,requests=len(req),min_price_owner_count=sum(x['minlayer']==l for x in req),
            min_price_tied_membership_count=sum(l in x['minlayers'] for x in req),floored_count=sum(x['floor'][i] for x in req))
        for name,values in [('pi',[x['pi'][i] for x in req]),('raw_kappa',[x['raw'][i] for x in req])]:
            row.update({name+'_'+k:v for k,v in stats(values).items()})
        price_rows.append(row)
    row=dict(batch=batch,requests=len(req),expanded=sum(x['stage']>0 for x in req),maxstage=sum(x['stage']==4 for x in req))
    for state in ('SATISFIED_BASE','SATISFIED_EXPANDED','UNSATISFIED','UNSATISFIED_MAX','ZERO_STEP'):row[state]=sum(x['state']==state for x in req)
    for stage in range(5):row['stage_'+str(stage)]=sum(x['stage']==stage for x in req)
    for name,values in [('pi_max',[x['pimax'] for x in req]),('terminal_beta',[x['beta'] for x in req]),('F',[x['F'] for x in req])]:
        row.update({name+'_'+k:v for k,v in stats(values).items()})
    controller_rows.append(row)
    for stage in list(range(5))+['ALL']:
        for layer in [4,5,6,7,8,'ALL']:
            points=select(projections,batch,layer,stage)
            if not points:continue
            def update_ids(pred):return {(x['batch'],x['candidate'],x['owner']) for x in points if pred(x)}
            cap_rows.append(dict(batch=batch,stage=stage,layer=layer,block_updates=len(points),request_updates=len(update_ids(lambda x:True)),
                capped_blocks=sum(x['capped'] for x in points),zero_blocks=sum(x['zero'] for x in points),free_blocks=sum(x['free'] for x in points),
                strict_cap_pressure_blocks=sum(x['strict_cap_pressure'] for x in points),any_cap_request_updates=len(update_ids(lambda x:x['capped'])),
                any_strict_cap_pressure_request_updates=len(update_ids(lambda x:x['strict_cap_pressure'])),shared_active_request_updates=len(update_ids(lambda x:x['shared'])),
                distinct_requests_with_cap=len({(x['batch'],x['owner']) for x in points if x['capped']})))
terminal_rows=[];real_rows=[];tiny_rows=[]
for batch in [1,5,10,15,'ALL']:
    for layer in [4,5,6,7,8]:
        ts=select(terminal,batch,layer)
        terminal_rows.append(dict(batch=batch,layer=layer,requests=len(ts),zero_normalized_count=sum(x['normalized']==0 for x in ts),
            at_local_cap_with_fp32_tolerance_count=sum(abs(x['normalized']-.75)<=1e-6 for x in ts),
            **{k:stats([x[k] for x in ts]) for k in ('normalized','requested_energy_share','raw_relative_norm_share','priced_spend_share','realized_canonical_energy_share','canonical_cosine','canonical_normratio')}))
        items=[x['record']['layers'][str(layer)] for x in realizations if batch=='ALL' or x['batch']==batch]
        row=dict(batch=batch,layer=layer,batches=len(items),roles={})
        for k in ('ideal_Q','effective_Q','effective_update_norm','masked_pre_minus_actual_pre_RMS'):row[k]=stats([x.get(k) for x in items])
        for role in ('mean','canonical','rewrite','KL'):
            row['roles'][role]={}
            for k in ('normratio','directionalratio','cosine','relative_error','zero_target_leakage'):
                values=[]
                for x in items:
                    g=x[role];values.extend(g[k] if isinstance(g,dict) else [v[k] for v in g])
                row['roles'][role][k]=stats(values)
        real_rows.append(row)
    for layer in [4,5,6,7,8,'ALL']:
        for name,source_rows in [('post_projection',projections)]+[('terminal_'+role,[x for x in roles if x['role']==role]) for role in ('mean','canonical','rewrite','KL')]:
            xs=select(source_rows,batch,layer);post=name=='post_projection'
            row=dict(batch=batch,layer=layer,source=name,denominator=len(xs),exact_zero=sum(x['category']=='zero' for x in xs),
                positive_le_1e_12=sum(x['category']=='tiny' for x in xs),gt_1e_12=sum(x['category']=='meaningful' for x in xs),
                zero_mask_count=sum(x['zero'] for x in xs) if post else None,
                zero_mask_vs_numeric_zero_mismatch=sum(x['zero']!=(x['norm']==0) for x in xs) if post else None)
            for key,values in [('tiny_normalized',[x['norm' if post else 'target'] for x in xs if x['category']=='tiny']),
                ('normratio_all_positive',[] if post else [x['ratio'] for x in xs]),
                ('normratio_target_gt_1e_12',[] if post else [x['ratio'] for x in xs if x['category']=='meaningful'])]:
                row.update({key+'_'+k:v for k,v in stats(values).items()})
            tiny_rows.append(row)
controller_groups=[]
for field in ('stage','state'):
    for value in sorted(set(x[field] for x in requests),key=str):
        xs=[x for x in requests if x[field]==value]
        controller_groups.append(dict(group_by=field,value=value,requests=len(xs),
            **{k:stats([x[k] for x in xs]) for k in ('F','owner_mean_rewrite_nll','KL','beta','spend_ratio','unused_budget','unused_budget_fraction','ceiling_fraction_used')}))
capowners={(x['batch'],x['owner']) for x in projections if x['stage']>0 and x['capped']}
unsat={(x['batch'],x['owner']) for x in requests if x['state']=='UNSATISFIED_MAX'}
by_norm=[]
for l in (4,5,6,7,8):
    xs=[x for x in roles if x['layer']==l and x['role']=='canonical']
    by_norm.append(dict(layer=l,categories={cat:dict(count=sum(x['category']==cat for x in xs),
        actual_normalized=stats([x['actual'] for x in xs if x['category']==cat]),cosine=stats([x['cosine'] for x in xs if x['category']==cat])) for cat in ('zero','tiny','meaningful')}))
summary=dict(scope='SEALED_PRICE_B1_B15_ONLY',source_commit=source,publication_commit='677dad89',raw_files_verified=len(verified),
    raw_bytes_verified=sum(x['bytes'] for x in verified),provenance=verified,price_summary=controller_rows[-1],
    min_price_layer_counts=counter([x['minlayer'] for x in requests]),min_price_tie_requests=sum(len(x['minlayers'])>1 for x in requests),
    floored_blocks=sum(sum(x['floor']) for x in requests),
    terminal_by_stage={str(stage):dict(count=sum(x['stage']==stage for x in requests),states=counter([x['state'] for x in requests if x['stage']==stage]),F=stats([x['F'] for x in requests if x['stage']==stage])) for stage in range(5)},
    terminal_layer_allocation=terminal_rows,realization=real_rows,subject_actual_gap=gaps,cap_summary=[x for x in cap_rows if x['batch']=='ALL'],
    definitions=dict(quantile='linear interpolation on sorted scalar observations',capped='saved exact FP64 projected_norm == local cap, excludes zero',
        strict_cap_pressure='proposal_norm - tau*weight - cap > 1e-10*max(1,cap,proposal_norm); stage0 dual is nonunique so this is not an independent-cap causal test',
        stage='post-update expansion stage used by projection',terminal='actual candidate24; FP32 near-cap abs(normalized-.75)<=1e-6',
        prices='fixed own batch c0; pooled across distinct batches',absence='no B16+, live status, model import/forward, tensor replay, scheduler query, remote write'),
    limitations=['Counterfactual uncapped or beta0=1 trajectories are unobserved.','Stage0 beta=.75 with pi>=1 already implies every .75 local cap.','Scalar source/price verification does not establish semantic causality.'],
    tiny_block_audit=dict(scope='same 90 SHA-verified sealed files, no additional remote reads',threshold=1e-12,
        threshold_units='Euclidean norm divided by own local a_lr',threshold_is_diagnostic_only=True,expanded_cap_distinct_requests=len(capowners),
        unsatisfied_max_requests=len(unsat),unsatisfied_max_with_expandedcap=len(capowners&unsat),terminal_controller_groups=controller_groups,
        canonical_absolute_action_by_target_magnitude=by_norm,
        role_denominators='1500 owners x 5 layers per role; canonical/mean one per owner, rewrite/KL saved owner means; do not sum roles',
        nll_definition='Owner mean across saved rewrite NLL rows; unweighted KL separate',
        normratio_interpretation='Raw means retained. Thresholded statistics are conditional and do not redefine all-owner realization.',
        initial_summary_failure='Earlier CPU-only draft treated six-rewrite NLL arrays as scalar; corrected owner_mean_rewrite_nll. No production change.'))
print(json.dumps(dict(summary=summary,prices=price_rows,controllers=controller_rows,caps=cap_rows,tiny=tiny_rows),allow_nan=False))
'''


def main():
    output = Path(__file__).resolve().parent
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=output, text=True).strip())
    manifest_bytes = subprocess.check_output(['git', 'show', PUBLICATION + ':' + MANIFEST], cwd=root)
    manifest = json.loads(manifest_bytes)
    allow = {f'{BASE}/batch-{batch:02d}/{name}' for batch in range(1, 16) for name in NAMES}
    sealed = {x['path']: x for x in manifest['files'] if x['path'] in allow}
    assert len(sealed) == 90
    program = REMOTE.replace('__SEALED__', repr(sealed)).replace('__BASE__', repr(BASE)).replace('__SOURCE__', repr(SOURCE))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'rke-server4', 'python3', '-'],
                            input=program, text=True, capture_output=True, timeout=60, check=True)
    data = json.loads(result.stdout)
    summary = data['summary']
    assert summary['raw_files_verified'] == 90 and summary['price_summary']['requests'] == 1500
    summary['manifest_reference'] = dict(path=MANIFEST, git_commit=PUBLICATION, sha256=hashlib.sha256(manifest_bytes).hexdigest())
    summary['reducer'] = dict(path='project/proposals/jlz-price-cap-budget-review/recover_diagnostics.py',
                            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                            remote_program_sha256=hashlib.sha256(program.encode()).hexdigest(),
                            status='EXECUTED_STDLIB_ONLY_SHA_VERIFIED_NO_SCIENTIFIC_RUN')
    for key, name in [('prices', 'price-distribution.csv'), ('controllers', 'controller-summary.csv'),
                      ('caps', 'cap-activity.csv'), ('tiny', 'tiny-block-audit.csv')]:
        with (output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[key][0]))
            writer.writeheader()
            writer.writerows(data[key])
    (output / 'evidence-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(status=summary['reducer']['status'], raw_files=summary['raw_files_verified'],
                          raw_bytes=summary['raw_bytes_verified'], reducer_sha256=summary['reducer']['sha256'])))


if __name__ == '__main__':
    main()
