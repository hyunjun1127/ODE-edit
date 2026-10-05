"""CPU-only independent raw reducer. Partial endpoints are never imputed."""
import argparse,csv,json,math,statistics,subprocess
from pathlib import Path
from project.run_scripts.jlz_realization.observe import reduce_rows
from .common import *

def validate_fit(folder):
    fit=json.loads((folder/'fit.json').read_text())
    events=[json.loads(p.read_text()) for p in sorted(folder.glob('candidate-*.json'))]
    require(1<=len(events)<=25 and [r['candidate'] for r in events]==list(range(len(events))),'CANDIDATE_SEQUENCE')
    require(fit['candidates']==len(events) and fit['updates']==len(events)-1<=24,'FIT_BUDGET')
    require(sum(r['terminal'] for r in events)==1 and events[-1]['terminal'],'ONE_COMMON_TERMINAL')
    require(events[-1]['gradient'] is None and not events[-1]['gradient_measured'],'TERMINAL_NO_BACKWARD')
    for row in events[:-1]:
        require(row['gradient_measured'] and row['reverse']['bridge_count']==1,'GRADIENT_ONCE')
        require(len(row['projection'])==row['B'] and all(p['update']==row['candidate']+1 for p in row['projection']),'SYNCHRONOUS_UPDATES')
        require(all(p['stored_budget']<=1.500001 and max(p['stored_norm'])<=.750001 for p in row['projection']),'REQUESTED_BUDGET')
        require(not all(j<.05 for j in row['J']),'STOP_BEFORE_BACKWARD')
    require(fit['request_evaluations']==events[0]['B']*len(events),'REQUEST_COUNT')
    return fit

def rows_from(folder):
    return [r for p in sorted(folder.glob('chunk-*.json')) for r in json.loads(p.read_text())['rows']]

def paired(before,after):
    b={r['identity']:r for r in before};a={r['identity']:r for r in after};require(a.keys()==b.keys(),'PAIRED_IDENTITY')
    result={}
    for kind in ('R','P','N'):
        keys=[k for k in b if b[k]['kind']==kind];label='true' if kind=='N' else 'new'
        result[kind]={}
        for metric in ('preference','strict'):
            def success(r):return r[label+'_strict'] if metric=='strict' else r['true_nll']<r['new_nll'] if kind=='N' else r['new_nll']<r['true_nll']
            for k in keys:
                require(all(a[k][t+'_token_identity']==b[k][t+'_token_identity'] for t in ('new','true')),'PAIR_TOKEN_IDENTITY')
            result[kind][metric]=dict(denominator=len(keys),lost=sum(success(b[k]) and not success(a[k]) for k in keys),
                gained=sum(not success(b[k]) and success(a[k]) for k in keys),retained=sum(success(b[k]) and success(a[k]) for k in keys))
    return result

def telemetry_summary(actions):
    groups={}
    for row in actions:
        for scope in ([row['kind'],'canonical'] if row['canonical'] else [row['kind']]):
            groups.setdefault((row['layer'],scope),[]).append(row)
    result=[]
    for (layer,scope),rows in sorted(groups.items()):
        record=dict(layer=layer,scope=scope,rows=len(rows))
        for field in ('directional_ratio','norm_ratio','cosine','relative_error','error_norm'):
            values=[r['actual'][field] for r in rows if r['actual'][field] is not None]
            record[field]=dict(count=len(values),undefined=len(rows)-len(values),mean=statistics.mean(values) if values else None,
                median=statistics.median(values) if values else None,max=max(values) if values else None)
        perowner={}
        for row in rows:perowner.setdefault(row['owner'],[]).append(row['actual']['error_norm'])
        record['owner_error']=[dict(owner=k,weighted_RMS=math.sqrt(sum(x*x for x in v)/len(v)),max=max(v)) for k,v in sorted(perowner.items())]
        result.append(record)
    return result

def augmented(rows):
    s=reduce_rows(rows)
    for kind,r in s.items():
        desired='true' if kind=='N' else 'new';group=[x for x in rows if x['kind']==kind]
        r['strict_accuracy']=r['strict_numerator']/r['strict_denominator']
        r['desired_nll_mean']=r[desired+'_nll_mean']
        for field in ('true_nll','new_nll','margin_true_minus_new'):
            values=sorted(x[field] for x in group)
            r[field+'_p50']=values[len(values)//2];r[field+'_p95']=values[int((len(values)-1)*.95)];r[field+'_max']=max(values)
        r['ties']=sum(x['new_nll']==x['true_nll'] for x in group)
    return s

def harmonic(s):
    v=[s[k]['rate'] for k in ('R','P','N')]
    return 0. if min(v)==0 else 3/sum(1/x for x in v)

def collect(attempt):
    out=attempt/'cpu-report';out.mkdir(exist_ok=False);root=attempt/'main'
    config=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
    expected=json.loads(verify(config['observer_identity']).read_text())['rows'];identities={r['identity']:r for r in expected}
    if config['W0']['mode']=='EXACT_REUSE':
        for m in config['W0']['members']:verify(m)
        raw0=rows_from(Path(config['W0']['summary']['path']).parent)
    else:raw0=rows_from(root/'W0')
    w0full=len(raw0)==26000
    endpoint={};coverage={};pair={};cohorts={};telemetry={};fit_summary=[];costs=[];previous=None;atwrite=[];commits=0
    if raw0:endpoint['W0']=augmented(raw0)
    for batch in range(1,21):
        folder=root/f'batch-{batch:02d}';complete=folder/'commit.json';summary=folder/'evaluation/summary.json'
        if not complete.exists() or not summary.exists():coverage[f'W{batch}']='NOT_COMMITTED';continue
        commit=json.loads(complete.read_text());entry=json.loads((folder/'entry.json').read_text());raw=rows_from(folder/'evaluation')
        require(entry['batch']==commit['batch']==batch and batch==commits+1,'COMMIT_PREFIX')
        if previous:require(all(entry[k]==previous[k] for k in ('state','rng','context','ledger')),'COMMIT_ENTRY_LINK')
        require(len(raw)==batch*100*13,'ALLSEEN_ROWS')
        for r in raw:
            require(r['identity'] in identities,'UNKNOWN_EVAL_ROW')
            require(all(r[t+'_token_identity']==identities[r['identity']][t+'_token_identity'] for t in ('new','true')),'TOKEN_IDENTITY')
            require(abs(r['margin_true_minus_new']-(r['true_nll']-r['new_nll']))<1e-12,'MARGIN_SIGN')
        rawsummary=reduce_rows(raw);stored=json.loads(summary.read_text())
        require(rawsummary==stored['summary']==commit['metrics'] and stored['no_mutation'] and stored['state']==commit['state'],'RAW_REDUCER_OBSERVER')
        require({k:r['denominator'] for k,r in rawsummary.items()}==dict(R=batch*100,P=batch*200,N=batch*1000),'ENDPOINT_DENOMINATORS')
        wr=json.loads((folder/'writer/writer.json').read_text())
        require(wr['accepted_weight_copy_exact'] and wr['before']==entry['state'] and wr['after']==commit['state'],'WEIGHT_STATE')
        require(wr['history_appends']==5 and all(r['append_count']==1 and r['columns']==100 and r['rewrite_only'] for r in wr['history']),'H_ONCE')
        fit=validate_fit(folder/'fit');fit_summary.append(dict(batch=batch,**fit));commits+=1;previous=commit
        name=f'W{batch}';coverage[name]='COMMITTED_OBSERVED';endpoint[name]=augmented(raw)
        ids=set(config['packing'][batch-1]['ids']);current=[r for r in raw if r['case_id'] in ids];atwrite.extend(current)
        seenids={r['identity'] for r in raw}
        pair[name]=dict(atwrite=paired(atwrite,raw))
        if w0full:pair[name]['W0']=paired([r for r in raw0 if r['identity'] in seenids],raw)
        panels=dict(current=current,past=[r for r in raw if r['case_id'] not in ids],
            first100=[r for r in raw if r['case_id'] in set(config['packing'][0]['ids'])],
            first500=[r for r in raw if r['case_id'] in set(x for p in config['packing'][:5] for x in p['ids'])],
            active=[r for r in raw if r['active_at_endpoint']],superseded=[r for r in raw if not r['active_at_endpoint']])
        panels.update({f'birth{b}':[r for r in raw if r['case_id'] in set(config['packing'][b-1]['ids'])] for b in range(1,batch+1)})
        cohorts[name]={panel:augmented(rows) for panel,rows in panels.items() if rows}
        actions=json.loads((folder/'writer/actions.json').read_text())
        telemetry[name]=dict(native_context=telemetry_summary(actions['rows']),native_mean=actions['mean'],layers=wr['layers'],
            shares=json.loads((folder/'writer/shares.json').read_text()))
        events=[json.loads(p.read_text()) for p in sorted((folder/'fit').glob('candidate-*.json'))]
        exclusive={key:sum(float(e.get(key,0)) for e in events) for key in ('builder_seconds','masked_seconds','replay_seconds','optimizer_seconds')}
        exclusive['reverse_seconds']=sum(e.get('reverse',{}).get('seconds',0) for e in events)
        exclusive['probe_seconds']=sum(e.get('probe',{}).get('seconds',0) for e in events)
        exclusive['writer_seconds']=wr['seconds'];exclusive['observer_seconds']=stored['seconds']
        costs.append(dict(batch=batch,exclusive=exclusive,fit_inclusive_seconds=fit['seconds'],
            native_subject_physical=[e.get('physical') for e in events],routes=fit.get('routes'),
            note='inclusive fit not added to nested phase totals; activation checkpoint recomputation tokens NOT_SEPARATELY_MEASURED'))
    terminal=json.loads((root/'terminal.json').read_text()) if (root/'terminal.json').exists() else dict(status='NOT_RECORDED')
    complete=commits==20 and w0full and terminal['status']=='W20_COMPLETE'
    status='COMPLETED' if complete else 'PARTIAL_OR_TECHNICAL_BLOCKED'
    if complete:require(sum(f['candidates'] for f in fit_summary)<=500 and sum(f['updates'] for f in fit_summary)<=480,'TOTAL_BUDGET')
    write(out/'metrics.json',dict(status=status,coverage=coverage,endpoints=endpoint,harmonic={k:harmonic(v) for k,v in endpoint.items()},paired=pair,cohorts=cohorts))
    write(out/'realization.json',dict(endpoints=telemetry));write(out/'fits.json',dict(batches=fit_summary))
    write(out/'cost.json',dict(measured=terminal,batches=costs,committed_prefix=commits,history_appends=commits*5,joins=max(0,commits-1)))
    table=[dict(endpoint=name,kind=kind,harmonic_score=harmonic(s),**r) for name,s in endpoint.items() for kind,r in s.items()]
    if table:
        with (out/'endpoints.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# SH3 V14 budget1.5/local.75 2k 사실 보고','',f'상태: {status}; 완료 commit {commits}/20',
        f'실행 source `{lock["source_commit"]}`','Cold W0/H0, fixed first2000 BS100×20. NoCP/exact resume NOT_AVAILABLE.',
        'CPU reducer/owner audit. GPU qualification과 완료는 별도 receipt. 과학 승격/새 baseline 없음.','',
        '| endpoint | RS | PS | NS | 조화평균 |','|---|---:|---:|---:|---:|']
    for name in ('W0','W1','W5','W10','W15','W20'):
        if name in endpoint:
            s=endpoint[name];lines.append('| '+name+' | '+' | '.join(f"{100*s[k]['rate']:.3f}" for k in ('R','P','N'))+f' | {100*harmonic(s):.3f} |')
        else:lines.append('| '+name+' | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | NOT_RECORDED |')
    lines+=['','TF 정확도는 자유생성 정확도가 아니다. R/P desired=new, N desired=true. Ties failure.',
        'TF strict/token-micro/prompt-macro, NLL/tails 및 paired/cohort는 metrics.json/endpoints.csv.',
        '실현/Q/층별 배분은 realization.json. 시간은 inclusive/exclusive를 구분하며 타 GPU 속도비는 산출하지 않는다.',
        '기존 baseline/V14.75 비교는 동일 cohort/runtime 결속 후 별도 review; 결측은 NOT_AVAILABLE.',
        '원 tensor/prompt/전체 로그는 Git 미게시. NO_BROADCAST_NOT_REQUIRED.','원자료: '+str(root)]
    (out/'report-ko.md').write_text('\n'.join(lines)+'\n')
    write(out/'inventory.json',dict(files=[member(p) for p in sorted(root.rglob('*')) if p.is_file()],no_checkpoint=True,raw_local_KEEP=True))
    write(out/'terminal.json',dict(status=status,report=member(out/'report-ko.md'),inventory=member(out/'inventory.json'),coverage=coverage))
    return dict(status=status,out=str(out))

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);a=p.parse_args();print(json.dumps(collect(a.attempt)))

if __name__=='__main__':main()
