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
        require(all(p['stored_budget']<=.750001 for p in row['projection']),'REQUESTED_BUDGET')
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

def collect(attempt):
    out=attempt/'cpu-report';out.mkdir(exist_ok=False);root=attempt/'B1'
    config=json.loads((attempt/'config.json').read_text());lock=json.loads((attempt/'execution.lock.json').read_text())
    expected=json.loads(verify(config['observer_identity']).read_text())['rows'];identities={r['identity']:r for r in expected}
    summaries={};raw={};warnings=[];coverage={};writer_summaries={};telemetry={}
    for name in ('W0',)+BRANCHES:
        folder=root/'W0' if name=='W0' else root/name/'evaluation';rows=rows_from(folder)
        if not rows:coverage[name]='NOT_MEASURED';continue
        summary=reduce_rows(rows)
        for r in rows:
            require(r['identity'] in identities,'UNKNOWN_EVAL_ROW')
            require(all(r[x+'_token_identity']==identities[r['identity']][x+'_token_identity'] for x in ('new','true')),'TOKEN_ROW_BINDING')
            require(abs(r['margin_true_minus_new']-(r['true_nll']-r['new_nll']))<1e-12,'MARGIN_SIGN')
        full={k:v['denominator'] for k,v in summary.items()}==dict(R=100,P=200,N=1000)
        coverage[name]='COMPLETE' if full else 'PARTIAL';summaries[name]=summary;raw[name]=rows
        if (folder/'summary.json').exists():require(summary==json.loads((folder/'summary.json').read_text())['summary'],'INDEPENDENT_REDUCER')
        if name!='W0' and (root/name/'complete.json').exists():
            wr=json.loads((root/name/'writer.json').read_text())
            require(wr['accepted_weight_copy_exact'] and wr['history_appends']==5 and all(x['append_count']==1 and x['columns']==100 and x['rewrite_only'] for x in wr['history']),'COMMIT_HISTORY')
            require(all(x['solve']['relative_residual']<=1e-8 for x in wr['layers'].values()),'NATIVE_SOLVER_INTEGRITY')
            writer_summaries[name]=wr
            actions=json.loads((root/name/'actions.json').read_text())
            telemetry[name]=dict(native_context=telemetry_summary(actions['rows']),native_mean=actions['mean'],
                shares=json.loads((root/name/'shares.json').read_text()))
    pair={name:paired(raw['W0'],raw[name]) for name in BRANCHES if coverage.get(name)=='COMPLETE' and coverage.get('W0')=='COMPLETE'}
    fit=None
    if (root/'fit'/'fit.json').exists():
        fit=validate_fit(root/'fit')
        require(fit['request_evaluations']<=2500 and fit['request_update_participations']<=2400,'FIT_BUDGET')
    complete=all(coverage.get(name)=='COMPLETE' for name in ('W0',)+BRANCHES) and len(writer_summaries)==1 and fit is not None
    terminal=json.loads((root/'terminal.json').read_text()) if (root/'terminal.json').exists() else dict(status='NOT_RECORDED')
    status='COMPLETED' if complete and terminal['status']=='B1_COMPLETE' else 'PARTIAL_OR_TECHNICAL_BLOCKED'
    write(out/'metrics.json',dict(status=status,coverage=coverage,endpoints=summaries,paired=pair,fit=fit))
    write(out/'realization.json',dict(branches=telemetry))
    write(out/'writers.json',dict(branches=writer_summaries))
    submission=json.loads((attempt/'submission.json').read_text()) if (attempt/'submission.json').exists() else {}
    jobs=list(submission.get('jobs',{}).values());cost='NOT_RECORDED'
    if jobs:
        response=subprocess.run(['sacct','-X','-P','-n','-j',','.join(jobs),'--format=JobIDRaw,JobName,State,ElapsedRaw,AllocTRES'],capture_output=True,text=True)
        cost=dict(returncode=response.returncode,parent_rows=response.stdout,allocated_GPU_seconds_rule='ElapsedRaw * AllocTRES gres/gpu, parent once; CPU0GPU',requested_wall_not_ETA=True)
    write(out/'cost.json',dict(scheduler=cost,measured=terminal))
    table=[]
    for name,summary in summaries.items():
        for kind,r in summary.items():table.append(dict(endpoint=name,kind=kind,**r))
    if table:
        with (out/'comparison-B1.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    lines=['# V14 B1 사실 보고','',f'상태: {status}',f'실행 source: `{lock["source_commit"]}`',
        'cold W0/H0, 고정 first100, V14_RD fit 1회·commit 1회. B2/추가 fit/CP 없음.',
        'CPU 검산과 실제 모델 검산은 별도 기록. 독립 reviewer 미사용(owner audit).',
        '', '| endpoint | R | P | N |','|---|---:|---:|---:|']
    for name in ('W0',)+BRANCHES:
        s=summaries.get(name)
        values=[f'{s[k]["numerator"]}/{s[k]["denominator"]}' if s and k in s else 'NOT_MEASURED' for k in ('R','P','N')]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines+=['','정확도/실현 telemetry는 comparison-B1.csv, realization.json, writers.json에 분리했다.',
        'margin_true_minus_new = true NLL − new NLL; 반대 부호는 부호를 바꾼 별도 필드만 사용한다.',
        'Raw와 tensor는 Git 미게시. NO_BROADCAST_NOT_REQUIRED: 동일 서버 B1 관측, compact 보고/manifest만 공유.',
        f'원자료: `{root}`','V13 비교: PENDING_COMPARISON (기존 paused monitor 재개 및 결과 조회 없음).',
        '과학적 해석·후속 승격 없음.']
    (out/'report-ko.md').write_text('\n'.join(lines)+'\n')
    inventory=[member(p) for p in sorted(root.rglob('*')) if p.is_file()]
    write(out/'inventory.json',dict(files=inventory,no_checkpoint=True,raw_local_KEEP=True))
    write(out/'terminal.json',dict(status=status,report=member(out/'report-ko.md'),inventory=member(out/'inventory.json'),coverage=coverage))
    return dict(status=status,out=str(out))

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);a=p.parse_args();print(json.dumps(collect(a.attempt)))

if __name__=='__main__':main()
