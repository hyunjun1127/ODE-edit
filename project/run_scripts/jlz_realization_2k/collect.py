"""CPU-only completed/partial reducer; no evaluator, fit, retry or polling."""
import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from project.run_scripts.jlz_realization.observe import reduce_rows, active_flags
from project.run_scripts.jlz_realization.collect import paired
from .common import INSTRUCTION, require, member, sha, write, verify, digest
from .schedule import BATCHES, MILESTONES, denominators, selection, verify_commit

def read(path):
    return json.loads(Path(path).read_text())

def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        w.writeheader(); w.writerows(rows)

def validate_rows(rows, selected, seen, expected):
    flags = active_flags(seen)
    for kind,n in (('R',1),('P',2),('N',10)):
        actual = [(r['case_id'],r['prompt_index']) for r in rows if r['kind']==kind]
        require(actual == [(r['case_id'],i) for r in selected for i in range(n)], 'OBSERVER_ROW_ORDER_COVERAGE')
    for row in rows:
        key = row['identity']; require(key in expected, 'UNKNOWN_PROMPT_ROW')
        for label in ('true','new'):
            require(row[label+'_token_identity'] == expected[key][label+'_token_identity']
                    and row[label+'_token_count'] == expected[key][label+'_token_count'], 'OBSERVER_TOKEN_IDENTITY')
        require(row['active_at_endpoint'] == flags[row['case_id']], 'ACTIVE_VERSION')
    return reduce_rows(rows)

def tails(rows):
    result = {}
    for kind in ('R','P','N'):
        group = [r for r in rows if r['kind']==kind]
        result[kind] = {}
        for name in ('true_nll','new_nll','margin_true_minus_new'):
            x = sorted(r[name] for r in group)
            def q(p):
                z=(len(x)-1)*p; lo=int(z); hi=min(lo+1,len(x)-1)
                return x[lo]+(x[hi]-x[lo])*(z-lo)
            result[kind][name]=dict(n=len(x),min=x[0],q05=q(.05),q50=q(.5),q95=q(.95),q99=q(.99),max=x[-1])
    return result

def complete(term, count, joins, errors, lock):
    return (term.get('status')=='COMPLETED' and term.get('instruction')==INSTRUCTION
        and term.get('source')==lock['source_commit'] and term.get('config_sha256')==lock['config_sha256']
        and term.get('main_commits')==20 and count==20 and joins==19 and not errors)

def collect(attempt, out):
    config, lock = read(attempt/'config.json'), read(attempt/'execution.lock.json')
    require(sha(attempt/'config.json') == lock['config_sha256'] and config['instruction_id']==INSTRUCTION, 'COLLECTOR_LOCK')
    data = read(config['stream'])[:2000]
    expected = {r['identity']:r for r in read(verify(config['observer_identity']))['rows']}
    w0bridge = read(verify(config['w0_reuse']['receipt']))
    w0 = read(verify(w0bridge['observations']))['rows']
    summary, cohorts, inventory, metrics, costs, telemetry, tail_rows, final = {},{},[],[],[],[],{},{}
    candidate_summary, terminal_context = [], {}
    config_digest = digest(config)
    for arm in ('A','B'):
        root=attempt/('main-'+arm); errors=[]; count=0; joins=0; previous=None; birth=[]; milestones={}
        term=read(root/'terminal.json') if (root/'terminal.json').exists() else dict(status='NOT_RUN_OR_NO_TERMINAL')
        for number in BATCHES:
            directory=root/f'batch-{number:02d}'
            try:
                c=read(directory/'commit.json')
                current,seen,selected=selection(data,number)
                verify_commit(c,number,[r['case_id'] for r in current],lock['source_commit'],config_digest)
                if previous is not None:
                    require(c['before']==previous,'OWN_STATE_JOIN');joins+=1
                previous=c['after']; count+=1
                inventory.append(member(directory/'commit.json'))
                candidate_paths=sorted((directory/'fit').glob('candidate-*.json'))
                require(len(candidate_paths)==25,'CANDIDATE_COMPLETENESS')
                for k,path in enumerate(candidate_paths,1):
                    r=read(path)
                    require(r['candidate']==k and r['Adam_updates_after']==min(k,24)
                            and r['gradient_measured']==(k<25) and r['replay'] is False,'CANDIDATE_BUDGET')
                    require(r['past_rows']==r['past_forward_count']==r['past_loss']==0,'NO_REPLAY')
                    inventory.append(member(path))
                    candidate_summary.append(dict(arm=arm,batch=number,candidate=k,
                        **{key:r[key] for key in ('Adam_updates_after','gradient_measured','native_nll_mean',
                           'native_kl_mean','native_norm_mean','policy_mean','native_seconds','builder_seconds','seconds','lr')},
                        auxiliary_loss_sum=r['auxiliary']['loss_sum'] if r['auxiliary'] else None,
                        clipped_groups=sum(v['clipped'] for v in r.get('post_update_clamp',{}).values()),
                        gradient_components='NOT_RECORDED; total D/q norms remain in original candidate receipt'))
                    if k==25:
                        for layer,v in r['layer'].items():
                            item=dict(arm=arm,batch=number,layer=layer,component_gradient_norm='NOT_RECORDED')
                            for key in ('gamma','norm_ratio','rho','realized_rho','planned_layer_share','realized_layer_share','M_diagonal'):
                                valid=[x for x in v[key] if x is not None]
                                item[key+'_mean']=sum(valid)/len(valid) if valid else None
                                item[key+'_valid']=len(valid)
                            telemetry.append(item)
                diagnostic=read(directory/'fit/diagnostic-manifest.json')
                require(len(diagnostic)==10 and all(set(x['tensors'])=={'M','planned_D','realized_Y'} for x in diagnostic),'DIAGNOSTIC_SCOPE')
                for row in diagnostic: verify(row)
                inventory.extend(diagnostic)
                for path in (directory/'entry.json',directory/'input.json',directory/'fit/terminal-actual.json',directory/'fit/diagnostic-manifest.json'):
                    inventory.append(member(path))
                terminal_context[arm+'-W'+str(number)]=read(directory/'fit/terminal-actual.json')
                require(not (directory/'Q2').exists(),'NEW_EXACT_PROBE_FORBIDDEN')
                observer=root/f'observe-W{number:02d}'; rows=[]
                for path in sorted(observer.glob('chunk-*.json')):
                    raw=read(path);require(raw['state']==c['after'],'OBSERVER_STATE')
                    rows.extend(raw['rows']);inventory.append(member(path))
                reduced=validate_rows(rows,selected,seen,expected)
                require({k:v['denominator'] for k,v in reduced.items()}==denominators(number),'EVAL_DENOMINATOR')
                declared=read(observer/'summary.json')
                require(declared['summary']==reduced and declared['no_mutation']
                        and declared['state']==c['after'],'OBSERVER_REDUCER')
                current_ids={r['case_id'] for r in current}
                current_rows=[r for r in rows if r['case_id'] in current_ids]
                require(declared['current']==reduce_rows(current_rows),'MILESTONE_CURRENT_RAW_REUSE')
                birth.extend(current_rows)
                scopes={'current':current_rows}
                if number in MILESTONES:
                    milestones[number]=rows;scopes['all_seen']=rows
                    tail_rows[arm+'-W'+str(number)]=tails(rows)
                    for name,limit in (('first100',100),('first500',500)):
                        ids={r['case_id'] for r in data[:limit]}
                        scopes[name]=[r for r in rows if r['case_id'] in ids]
                    scopes['active_all_seen']=[r for r in rows if r['active_at_endpoint']]
                for scope,subset in scopes.items():
                    for kind,v in reduce_rows(subset).items():
                        metrics.append(dict(arm=arm,batch=number,scope=scope,kind=kind,**v))
                costs.extend([dict(arm=arm,batch=number,phase='batch_inclusive',seconds=c['seconds']),
                              dict(arm=arm,batch=number,phase='observer',seconds=declared['seconds'])])
                inventory.append(member(observer/'summary.json'))
            except Exception as exc:
                errors.append(dict(batch=number,type=type(exc).__name__,error=str(exc)))
        for number,rows in milestones.items():
            ids={r['case_id'] for r in data[:number*100]}
            b=[r for r in birth if r['case_id'] in ids]
            by_birth={}
            for k in range(1,number+1):
                cohort_ids={x['case_id'] for x in data[(k-1)*100:k*100]}
                by_birth[str(k)]=paired([r for r in b if r['case_id'] in cohort_ids],
                    [r for r in rows if r['case_id'] in cohort_ids])
            cohorts[arm+'-W'+str(number)]=dict(at_write=paired(b,rows),W0=paired(w0,rows),
                by_birth=by_birth)
        for earlier,later in ((5,10),(5,20),(10,20)):
            if earlier in milestones and later in milestones:
                cohorts[arm+f'-W{earlier}-W{later}']=paired(milestones[earlier],milestones[later])
        if 20 in milestones: final[arm]=milestones[20]
        if (root/'batch-21').exists() or (root/'observe-W21').exists():
            errors.append(dict(batch=21,error='B21_FORBIDDEN'))
        good=complete(term,count,joins,errors,lock)
        summary[arm]=dict(complete2000=good,commits=count,own_state_joins=joins,terminal=term,errors=errors)
        if (root/'terminal.json').exists():inventory.append(member(root/'terminal.json'))
    comparison=[];base_report=[]
    for arm,rows in final.items():
        for kind,v in reduce_rows(rows).items():comparison.append(dict(method='V9_'+arm,endpoint=20,kind=kind,scope='CURRENT_MATCHED_A_B',**v))
    for baseline in config['baselines']:
        item=dict(baseline)
        try:
            if baseline['status']!='VERIFIED_HISTORICAL_REUSE':
                comparison.append(dict(method=baseline['name'],endpoint=20,status='NOT_AVAILABLE'));base_report.append(item);continue
            verify(baseline['raw']);verify(baseline['commit'])
            rows=read(verify(baseline['normalized']))['rows']
            for kind,v in reduce_rows(rows).items():comparison.append(dict(method=baseline['name'],endpoint=20,kind=kind,scope=baseline['comparison_scope'],**v))
            for arm,after in final.items():cohorts[baseline['name']+'-to-'+arm]=paired(rows,after)
        except Exception as exc:item.update(status='NOT_AVAILABLE',reason=str(exc))
        base_report.append(item)
    accounting={'status':'NOT_AVAILABLE','parent_allocated_GPU_seconds':None}
    if (attempt/'submission.json').exists():
        jobs=read(attempt/'submission.json')['jobs']
        parent_ids=[jobs[k] for k in ('main-A','main-B')]
        try:
            p=subprocess.run(['sacct','-X','-n','-P','-j',','.join(parent_ids),'-o','JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'],capture_output=True,text=True,timeout=30)
            accounting=dict(status='ACCOUNTING_SINGLE_READ' if p.returncode==0 else 'NOT_AVAILABLE',job_ids=parent_ids,output=p.stdout,error=p.stderr)
        except Exception as exc:accounting['reason']=str(exc)
    out.mkdir(parents=True,exist_ok=False)
    for name,value in [('summary',summary),('paired-cohorts',cohorts),('tails',tail_rows),('baselines',base_report),('accounting',accounting),
                       ('terminal-context-decomposition',terminal_context),
                       ('cost',dict(timers=costs,parent_not_added_to_nested_timers=True,new_Q1=0,new_Q2=0,new_baseline=0))]:write(out/(name+'.json'),value)
    for name,rows in [('metrics',metrics),('comparison-W20',comparison),('terminal-realization',telemetry),
                      ('candidate-summary',candidate_summary)]:csv_write(out/(name+'.csv'),rows)
    report=['# JLZ v9 ridge 2000 edit 실행 보고','',f'실행 source `{lock["source_commit"]}`. 기존500과 별도 cold W0/H0 두 경로다.',
        '새 exact probe, baseline fit, checkpoint는 0이다. 기존 Q1 재사용은 config bridge를 따른다.','',
        '| Arm | commits | state joins | W20 완료 |','|---|---:|---:|---|']
    for arm,r in summary.items():report.append(f'| {arm} | {r["commits"]} | {r["own_state_joins"]} | {r["complete2000"]} |')
    report+=['','W5/W10/W20 및 current는 metrics.csv에 구분했다. TF는 자유생성 지표가 아니다.',
        'comparison-W20.csv의 historical reference는 같은2000문항이나 runtime/layers/조건이 다른 비교다. 없는 baseline은 NOT_AVAILABLE이다.',
        '오류/부분실행은 실패 점수 0 또는 전체완료로 표시하지 않았다. 성분별 gradient는 NOT_RECORDED다.',
        'CPU scalar reducer이며 새 모델 평가가 아니다. 원 raw/tensor는 local KEEP/Git0. NO_BROADCAST_NOT_REQUIRED.']
    (out/'report-ko.md').write_text('\n'.join(report)+'\n')
    inventory += [member(p) for p in sorted(out.iterdir()) if p.is_file()]
    write(out/'artifact-index.json',inventory)
    all_complete=all(r['complete2000'] for r in summary.values())
    write(out/'terminal.json',dict(status='COMPLETED_2000_BOTH' if all_complete else 'PARTIAL_OR_TECHNICAL_FAILED',
        instruction=INSTRUCTION,source=lock['source_commit'],config_sha256=lock['config_sha256'],
        main_commits=sum(r['commits'] for r in summary.values()),own_state_joins=sum(r['own_state_joins'] for r in summary.values()),
        report=member(out/'report-ko.md'),artifacts=member(out/'artifact-index.json'),no_new_evaluation=True,no_checkpoint=True))

def dispatch(attempt,out):
    try:
        collect(attempt,out)
    except Exception as exc:
        # Preserve any already generated report and expose early identity/I/O failure.
        import traceback
        failure=out.parent/(out.name+'-failure')
        write(failure/'first-error.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        (failure/'report-ko.md').write_text('# CPU collector 기술 실패\n\n원 partial 산출물을 보존했습니다. 전체 성공으로 집계하지 않았습니다.\n')
        write(failure/'artifact-index.json',[member(p) for p in sorted(failure.iterdir()) if p.is_file()])
        write(failure/'terminal.json',dict(status='COLLECTOR_TECHNICAL_FAILED',instruction=INSTRUCTION,
            report=member(failure/'report-ko.md'),artifacts=member(failure/'artifact-index.json')))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();dispatch(a.attempt,a.report)
