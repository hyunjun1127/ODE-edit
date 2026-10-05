"""Independent CPU scalar/row reducer; no model/Slurm writes or backfill."""
import argparse,csv,json,math,os,pwd,subprocess,traceback
from pathlib import Path
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader,validate_rows as independent_rows,reduce_rows,compare_summary,paired,harmonic,active_flags)
from . import *
from .profile import ARMS,history_expected

def finite(value):
    if isinstance(value,float):require(math.isfinite(value),'NONFINITE_RAW_SCALAR')
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(tuple,list)):
        for v in value:finite(v)

def validate_fit(fit,profile,B=100):
    finite(fit);events=fit['events'];n=fit['candidates']
    require(1<=n<=25 and len(events)==n and fit['updates']==n-1<=24
        and fit['logical_builds']==fit['logical_subject_forwards']==n
        and fit['logical_subject_backwards']<=24,'25_24_BUILD_SUBJECT_BUDGET')
    require(fit['terminal_candidate']==n-1 and fit['terminal_no_backward'] and fit['terminal_extra_forward']==0
        and fit['norm_analytic_once'] and fit['production_builder_reverse']==fit['production_solve_VJP']==0,
        'PRODUCTION_APPROXIMATION_AND_TERMINAL_POLICY')
    require(fit['requested_gradient']==('same_owner_adjoint_sum' if profile['blind'] else 'FP64_Lambda_M_T'),
        'ARM_PULLBACK_POLICY')
    oldactive=[False]*B;oldt=[0]*B;olde=[0]*B;oldradius=None;backwards=0
    for k,row in enumerate(events):
        require(row['candidate']==k and row['ordinal']==k+1 and len(row['F'])==B,'CANDIDATE_ORDER_CARDINALITY')
        mask=[oldactive[r] or row['F'][r]>=.05 for r in range(B)]
        require(row['active_mask']==mask and row['controller']['active']==mask,'IRREVERSIBLE_CURRENT_NATIVE_LOSS_ACTIVE_MASK')
        ctrl=row['controller'];require(ctrl['update_counts']==oldt and ctrl['expansion']==olde,'OWN_REQUEST_COUNTER_JOIN')
        require(ctrl['n_exp']==profile['n_exp'] and len(ctrl['radius'])==B,'PROFILE_RADIUS')
        anchors=ctrl['anchor_values'];star=ctrl['anchor_star'];caps=ctrl['local_caps']
        require(ctrl['eligible_layers']==profile['eligible_layers'] and len(anchors)==len(caps)==len(profile['eligible_layers'])
            and all(len(a)==B for a in anchors+caps) and len(star)==B,'NATIVE_ANCHOR_CAP_CARDINALITY')
        require(all(a>0 for values in anchors for a in values) and all(a>0 for a in star),'NATIVE_POSITIVE_ANCHORS')
        require(all(math.isclose(cap,.75*a,rel_tol=1e-12,abs_tol=1e-12) for values,cs in zip(anchors,caps) for a,cap in zip(values,cs)),
            'LOCAL_CAP_NATIVE_ANCHOR')
        for r in range(B):
            maximum=.75*star[r];mainbase=min(.75*anchors[0][r],maximum);requested=profile['base_multiplier']*mainbase;base=min(requested,maximum)
            require(math.isclose(ctrl['base_requested'][r],requested,rel_tol=1e-12,abs_tol=1e-12)
                and math.isclose(ctrl['base_effective'][r],base,rel_tol=1e-12,abs_tol=1e-12)
                and math.isclose(ctrl['radius_max'][r],maximum,rel_tol=1e-12,abs_tol=1e-12)
                and ctrl['base2x_clipped'][r]==(requested>maximum),'BASE_AND_ORIGINAL_MAXIMUM')
            expected_radius=base if profile['n_exp']==0 else base*(maximum/base)**(olde[r]/profile['n_exp'])
            require(math.isclose(ctrl['radius'][r],expected_radius,rel_tol=1e-12,abs_tol=1e-12),'INDEPENDENT_EXPANSION_RADIUS')
        terminal=k==24 or not any(mask)
        require(row['terminal']==terminal and (k==n-1)==terminal,'LAST_EVALUATED_NOT_BEST_SELECTION')
        require(row['backward']==(not terminal and any(mask)),'NO_BACKWARD_TERMINAL_OR_INACTIVE')
        backwards+=row['backward']
        require(math.isclose(row['full_task_sum'],sum(row['F']),rel_tol=1e-12,abs_tol=1e-10)
            and math.isclose(row['masked_backward_sum'],sum(f for f,m in zip(row['F'],mask) if m),rel_tol=1e-10,abs_tol=1e-8),
            'FORWARD_ALL_MASKED_SUM_GRADIENT')
        if oldradius is not None:require(all(x>=y for x,y in zip(ctrl['radius'],oldradius)),'RADIUS_NONDECREASING')
        if terminal:
            require(row['gradient_status']=='NO_BACKWARD_TERMINAL' and 'projection' not in row,'TERMINAL_GRADIENT_NULL_POLICY')
            oldactive=mask;break
        after=row['post_update_controller'];projection=row['projection']
        require(projection['local_excess_max']<=1e-6 and projection['radius_excess_max']<=1e-6
            and not projection['postcast_shrink'] and projection['lr']==.1 and projection['eps']==1e-8
            and not projection['moment_reset'],'FP32_ENERGY_FEASIBILITY_AND_ABSOLUTE_ADAM')
        expected_t=[t+int(active) for t,active in zip(oldt,mask)]
        expected_e=[e+int(active and t>=12 and F>=.05 and e<profile['n_exp'] and base<maximum)
            for e,active,t,F,base,maximum in zip(olde,mask,oldt,row['F'],ctrl['base_effective'],ctrl['radius_max'])]
        require(after['update_counts']==expected_t and after['expansion']==expected_e
            and projection['adam_updates']==expected_t,'OWN_UPDATE_GRACE12_EXPANSION_BEFORE13')
        norms=projection['post_norm'];require(len(norms)==len(caps) and all(len(a)==B for a in norms),'POSTCAST_NORM_CARDINALITY')
        local_excess=max(0.,max(n-cap for ns,cs in zip(norms,caps) for n,cap in zip(ns,cs)))
        radius_excess=max(0.,max(math.sqrt(sum(ns[r]**2 for ns in norms))-after['radius'][r] for r in range(B)))
        require(local_excess<=1e-6 and radius_excess<=1e-6
            and math.isclose(local_excess,projection['local_excess_max'],rel_tol=1e-8,abs_tol=1e-12)
            and math.isclose(radius_excess,projection['radius_excess_max'],rel_tol=1e-8,abs_tol=1e-12),'INDEPENDENT_POSTCAST_BUDGET_REDUCTION')
        require(all(0<=x<=1 for x in projection['theta']),'EXACT_PROJECTION_THETA')
        oldactive,oldt,olde,oldradius=mask,expected_t,expected_e,after['radius']
    require(backwards==fit['logical_subject_backwards'] and sum(oldt)==fit['request_updates']<=B*24
        and fit['controller']['update_counts']==oldt and fit['controller']['expansion']==olde,'FIT_TERMINAL_COUNTERS')
    require(len(fit['terminal_states'])==B and all(s in ('ZERO_STEP','SATISFIED_BASE','SATISFIED_EXPANDED','UNSATISFIED_MAX','UNSATISFIED')
        for s in fit['terminal_states']),'TERMINAL_STATE_POPULATION')
    return dict(builds=n,subject_forwards=n,subject_backwards=backwards,request_updates=sum(oldt),
        expanded_requests=sum(e>0 for e in olde),statuses={s:fit['terminal_states'].count(s) for s in set(fit['terminal_states'])})

def endpoint(reader,folder,identities,ids,name,state_value,seen):
    folder=Path(folder)
    if not (folder/'summary.json').exists():return None
    rows=[]
    for path in sorted(folder.glob('chunk-*.json')):
        chunk=reader.json(path);require(chunk['state']==state_value and chunk['optimizer_feedback'] is False,'RAW_ENDPOINT_STATE')
        rows.extend(chunk['rows'])
    independent_rows(rows,expected_rows(identities,ids),name)
    result=reduce_rows(rows);saved=reader.json(folder/'summary.json')
    require(saved['endpoint']==name and saved['state']==state_value and saved['requests']==len(ids)
        and saved['no_mutation'] and saved['optimizer_feedback'] is False,'ENDPOINT_RECEIPT')
    compare_summary(result,saved['summary']);flags=active_flags(seen)
    require(all(r['active_at_endpoint']==flags[r['case_id']] for r in rows),'SEEN_PREFIX_METADATA')
    return dict(rows=rows,metrics=result,seconds=saved['seconds'])

def arm_review(reader,attempt,c,lock,arm,records,identities):
    out=attempt/arm;ids=[r['case_id'] for r in records];profile=c['arm_profiles'][arm];layers=list(map(str,profile['eligible_layers']))
    cold={key:{l:c['cold_W0_H0'][key][l] for l in layers} for key in ('W','H')}
    commits=[];metrics=[];pairs=[];cost=[];realizations=[];prefix={};atwrite=[]
    counts=dict(builds=0,subject_forwards=0,subject_backwards=0,request_updates=0)
    w0=endpoint(reader,out/'W0',identities,ids,'W0',cold,records)
    if w0:metrics.append(dict(endpoint='W0',requests=2000,metrics=w0['metrics']))
    for number in range(1,21):
        folder=out/f'batch-{number:02d}'
        if not (folder/'commit.json').exists():break
        commit=reader.json(folder/'commit.json');entry=reader.json(folder/'entry.json');pack=c['packs'][number-1]
        require(commit['task']==entry['task']==TASK and commit['arm']==entry['arm']==arm
            and commit['batch']==entry['batch']==number and commit['source']==entry['source']==lock['source_commit']
            and commit['config']==entry['config']==digest(c) and commit['profile']==entry['profile']==digest(profile),'ARM_SOURCE_CONFIG_BATCH')
        require(commit['ids']==entry['ids']==pack['ids'] and commit['native_pack']==entry['native_pack']==pack['identity'],'NATIVE_PACK_ORDER')
        require(commit['before']==entry['state'] and commit['RNG_before']==entry['RNG'],'ENTRY_STATE_RNG')
        if commits:
            previous=commits[-1];require(commit['before']==previous['after'] and commit['RNG_before']==previous['RNG_after']
                and entry['ledger']==previous['ledger'] and entry['context']==previous['context'],'19_OWN_STATE_JOINS')
        else:require(commit['before']==cold,'OWN_COLD_W0_H0')
        require(commit['observer_no_mutation'] and commit['RNG_after']==commit['RNG_before'],'OBSERVER_RNG_NONMUTATION')
        require(commit['history_appends']==len(layers) and set(commit['history'])==set(layers),'ELIGIBLE_HISTORY_COUNT')
        for l,h in commit['history'].items():
            require(h['appends']==1 and h['requests']==100 and h['rows']=='native_rewrite_only_nestedmean'
                and h['zero_and_unsatisfied_requests_included'] and h['before']==commit['before']['H'][l]
                and h['after']==commit['after']['H'][l],'HISTORY_ALL_REQUESTS_ONCE')
        fit=reader.json(verify(commit['fit']));cnt=validate_fit(fit,profile)
        for k in counts:counts[k]+=cnt[k]
        writer=reader.json(verify(commit['writer']))
        require(writer['accepted_weight_copy_exact'] and writer['terminal_last_evaluated_not_best'] and writer['no_resolve']
            and writer['no_double_add'] and writer['candidate']==fit['terminal_candidate']==commit['accepted_candidate'],'LAST_EVALUATED_EXACT_COMMIT')
        require(writer['weight_hashes']==commit['after']['W'] and writer['history']==commit['history'],'PAYLOAD_AND_H_IDENTITIES')
        capture=reader.json(folder/'entry-capture.json');require(capture['fresh_capture'] and capture['H_entry']==entry['state']['H']
            and capture['anchor_layer']==8 and capture['eligible_layers']==profile['eligible_layers'],'FRESH_TEACHER_ANCHOR_FACTOR')
        realization=reader.json(verify(writer['realization']));finite(realization)
        require(realization['candidate']==commit['accepted_candidate'] and realization['B']==100 and set(realization['layers'])==set(layers)
            and realization['terminal_extra_planner_forward']==realization['terminal_extra_planner_backward']==0,'TERMINAL_REALIZATION_BINDING')
        realizations.append(dict(batch=number,realization=realization))
        gap=reader.json(verify(writer['actual_gap']));finite(gap)
        require(gap['record_only'] and len(gap['subject_F'])==len(gap['actual_alltoken_F'])==100,'SUBJECT_ACTUAL_GAP_NOT_EXACTNESS')
        current=records[(number-1)*100:number*100];seen=records[:number*100];birth=pack['ids']
        pre=endpoint(reader,folder/'pre',identities,birth,f'B{number}_PRE',commit['before'],seen)
        selected=seen if number in MILESTONES else current;selected_ids=[r['case_id'] for r in selected]
        post=endpoint(reader,folder/'post',identities,selected_ids,f'W{number}',commit['after'],seen)
        require(pre is not None and post is not None,'COMMITTED_OBSERVATION_MISSING');compare_summary(pre['metrics'],commit['pre']);compare_summary(post['metrics'],commit['post'])
        birthrows=[r for r in post['rows'] if r['case_id'] in set(birth)];compare_summary(reduce_rows(birthrows),commit['post_current'])
        metrics.extend([dict(endpoint=f'B{number}_PRE',requests=100,metrics=pre['metrics']),
            dict(endpoint=f'W{number}_CURRENT',requests=100,metrics=reduce_rows(birthrows))])
        pairs.append(dict(from_endpoint=f'B{number}_PRE',to_endpoint=f'W{number}_CURRENT',paired=paired(pre['rows'],birthrows)));atwrite.extend(birthrows)
        if number in MILESTONES:
            prefix[number]=post['rows'];metrics.append(dict(endpoint=f'W{number}_ALL_SEEN',requests=len(seen),metrics=post['metrics']))
            metrics.append(dict(endpoint=f'W{number}_FIRST500',requests=500,metrics=reduce_rows([r for r in post['rows'] if r['case_id'] in set(ids[:500])])))
            pairs.append(dict(from_endpoint='AT_WRITE',to_endpoint=f'W{number}',paired=paired(atwrite,post['rows'])))
            if w0:pairs.append(dict(from_endpoint='W0',to_endpoint=f'W{number}',paired=paired([r for r in w0['rows'] if r['case_id'] in set(ids[:number*100])],post['rows'])))
        diag=reader.json(folder/'diagnostic/diagnostic.json') if (folder/'diagnostic/diagnostic.json').exists() else None
        if number in (5,10,20):require(diag is not None and diag['same_terminal_candidate']==commit['accepted_candidate'],'SCHEDULED_TERMINAL_DIAGNOSTIC')
        cost.append(dict(batch=number,batch_inclusive_seconds=commit['seconds'],entry_capture_seconds=capture['seconds'],
            fit_inclusive_seconds=fit['seconds'],build_seconds=fit['build_seconds'],subject_seconds=fit['subject_seconds'],
            pullback_seconds=fit['pullback_seconds'],optimizer_projection_seconds=fit['optimizer_projection_seconds'],
            candidate_scalar_telemetry_seconds=fit['candidate_scalar_telemetry_seconds'],
            writer_inclusive_seconds=writer['seconds'],scalar_terminal_seconds=realization['seconds'],
            observer_pre_seconds=pre['seconds'],observer_post_seconds=post['seconds'],logical_and_physical_fit_calls=fit['call_counts'],
            additional_actual_native_groups=gap['extra_actual_native_forward_groups'],
            diagnostic=diag,timing_policy='exclusive fit stages; fit/writer/batch inclusive timers not additive again'))
        commits.append(commit)
    terminal=reader.json(out/'terminal.json') if (out/'terminal.json').exists() else None
    firsterror=reader.json(out/'first-error.json') if (out/'first-error.json').exists() else None
    complete=len(commits)==20 and 20 in prefix and terminal is not None and terminal['status']=='W20_COMPLETE'
    if len(commits)==20:require({k:v['denominator'] for k,v in reduce_rows(prefix[20]).items()}==dict(R=2000,P=4000,N=20000),'W20_FULL_DENOMINATORS')
    result=dict(arm=arm,status='W20_COMPLETE' if complete else 'PARTIAL_OR_TECHNICAL_BLOCKED',commits=len(commits),requests=len(commits)*100,
        expected=dict(commits=20,requests=2000,joins=19,history_appends=history_expected(arm)),
        actual=dict(joins=max(0,len(commits)-1),history_appends=sum(r['history_appends'] for r in commits)),
        counters=counts,metrics=metrics,paired=pairs,realization=realizations,cost=cost,terminal=terminal,first_error=firsterror,
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE',no_missing_as_zero=True)
    return result,prefix

def accounting_snapshot(attempt):
    receipt=attempt/'submission.json'
    if not receipt.exists():return dict(status='NOT_AVAILABLE',reason='NO_EXACT_LOCAL_SUBMISSION')
    data=json.loads(receipt.read_text());ids=[str(v) for v in data['jobs'].values()]
    require(all(x.isdigit() for x in ids),'EXACT_JOB_IDS')
    response=subprocess.run(['sacct','-n','-P','-j',','.join(ids),
        '--format=JobIDRaw,User,JobName%120,State,ElapsedRaw,AllocTRES,TotalCPU,ExitCode'],check=True,capture_output=True,text=True,timeout=30)
    roles={str(job):role for role,job in data['jobs'].items()}
    parents=[]
    for line in response.stdout.splitlines():
        v=line.split('|')
        if len(v)<8 or v[0] not in ids:continue
        require(v[1]==pwd.getpwuid(os.getuid()).pw_name and v[2]==TASK+'-'+roles[v[0]],'EXACT_PARENT_OWNER_NAME')
        tres=dict(x.split('=',1) for x in v[5].split(',') if '=' in x);gpu=int(tres.get('gres/gpu',0));elapsed=int(v[4] or 0)
        require(gpu==0 if roles[v[0]]=='collector' else gpu in (0,1),'EXACT_PARENT_ALLOCATED_GPU')
        parents.append(dict(job=int(v[0]),role=roles[v[0]],state=v[3],elapsed_seconds=elapsed,GPUs=gpu,allocated_GPU_seconds=gpu*elapsed,TotalCPU=v[6],exit=v[7]))
    observed={str(v['job']) for v in parents};missing=sorted(set(ids)-observed)
    return dict(status='ONE_BOUNDED_EXACT_PARENT_QUERY' if not missing else 'PARTIAL_ACCOUNTING_NOT_AVAILABLE',parents=parents,
        expected_job_ids=ids,observed_job_ids=sorted(observed),missing_job_ids=missing,cost_complete=not missing,
        observed_allocated_GPU_seconds=sum(v['allocated_GPU_seconds'] for v in parents),
        allocated_GPU_seconds=sum(v['allocated_GPU_seconds'] for v in parents) if not missing else None,
        pending_not_allocated=True,steps_not_double_counted=True,collector_GPU0=True)

def _collect(attempt,out,accounting):
    out.mkdir(parents=True,exist_ok=False);reader=Reader();c=reader.json(attempt/'config.json');lock=reader.json(attempt/'execution.lock.json')
    require(c['task_id']==lock['task_id']==TASK and c['instruction_id']==lock['instruction_id']==NONCE
        and sha(attempt/'config.json')==lock['config_sha256'],'COLLECTOR_SOURCE_CONFIG_AUTHORITY')
    records=load_prefix(Path(c['stream']).parent,2000);require(digest([r['case_id'] for r in records])==ORDERED_SHA,'ORDERED_FIXED_COHORT')
    identities=reader.json(verify(c['observer_identity']))['rows'];arms={};prefix={}
    for arm in ARMS:arms[arm],prefix[arm]=arm_review(reader,attempt,c,lock,arm,records,identities)
    cross=[]
    for arm in ARMS[1:]:
        for k in MILESTONES:
            if k in prefix['MAIN'] and k in prefix[arm]:cross.append(dict(reference='MAIN',arm=arm,endpoint=f'W{k}',paired=paired(prefix['MAIN'][k],prefix[arm][k])))
    complete=all(v['status']=='W20_COMPLETE' for v in arms.values());parents=accounting_snapshot(attempt) if accounting else dict(status='NOT_REQUESTED_CPU_FIXTURE')
    qfolder=attempt/'qualification-runtime'
    qualification=dict(terminal=reader.json(qfolder/'terminal.json') if (qfolder/'terminal.json').exists() else None,
        first_error=reader.json(qfolder/'first-error.json') if (qfolder/'first-error.json').exists() else None,
        READY=reader.json(attempt/'qualification/READY.json') if (attempt/'qualification/READY.json').exists() else None,
        actual_candidates_total_max=3,profile='MAIN',new_fit_or_updates=0)
    result=dict(task=TASK,status='FIVE_ARMS_W20_COMPLETE' if complete else 'PARTIAL_OR_TECHNICAL_BLOCKED',source=lock['source_commit'],config_sha256=lock['config_sha256'],
        arms=arms,qualification=qualification,cross_arm_paired=cross,actual=dict(commits=sum(v['commits'] for v in arms.values()),
        history_appends=sum(v['actual']['history_appends'] for v in arms.values())),expected=dict(commits=100,history_appends=420),accounting=parents,
        CPU_review='Independent immutable-row arithmetic/token identity/state linkage, no target-model replay',
        original_document_CPU_table='UNVERIFIED_NOT_SUPPLIED;new_source_CPU_receipt_separate',baseline_new_fits=0,
        no_missing_as_zero=True,checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    write(out/'reduction.json',result);write(out/'paired.json',cross)
    rows=[]
    for arm,v in arms.items():
        for point in v['metrics']:
            m=point['metrics']
            for kind,item in m.items():rows.append(dict(arm=arm,endpoint=point['endpoint'],kind=kind,
                **{k:item[k] for k in ('numerator','denominator','rate','token_micro','prompt_macro','strict_numerator','strict_denominator')},
                harmonic_RSPSNS=harmonic(m)))
    with (out/'comparison.csv').open('x',newline='') as f:
        names=list(rows[0]) if rows else ['arm','endpoint','kind','numerator','denominator']
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
    lines=['# v12-R 실현 응답 2k 사실 보고','',f'- 상태: {result["status"]}',f'- 실행 source: `{lock["source_commit"]}`',
        f'- Commit {result["actual"]["commits"]}/100, H append {result["actual"]["history_appends"]}/420.',
        '- 5 arm은 각각 cold W0/H0에서 자기 W/H trajectory를 따른다. L4-ONLY의 anchor는 L8이며 H append는 batch당1회다.',
        '- 원 문서 CPU 완료표/500초/4.7GPUh는 미검증 역사·추정이다. 이번 source-bound CPU, GPU qualification, 제출, 완료를 구분한다.',
        '', '| Arm/endpoint | RS | PS | NS |', '|---|---:|---:|---:|']
    for arm,v in arms.items():
        for point in v['metrics']:
            if point['endpoint']=='W0' or point['endpoint'].endswith('ALL_SEEN'):
                m=point['metrics'];lines.append('| '+arm+'/'+point['endpoint']+' | '+' | '.join(f'{100*m[k]["rate"]:.3f} ({m[k]["numerator"]}/{m[k]["denominator"]})' for k in ('R','P','N'))+' |')
    lines.extend(['','## Coverage·기제·비용',''])
    for arm,v in arms.items():
        lines.append(f'- {arm}: {v["status"]}; commit {v["commits"]}/20, join {v["actual"]["joins"]}/19, H {v["actual"]["history_appends"]}/{history_expected(arm)}; '+
            '/'.join(str(v['counters'][k]) for k in ('builds','subject_forwards','subject_backwards','request_updates'))+' BUILD/subjectF/subjectB/request-update.')
        if v['first_error']:lines.append(f'  최초 기술 오류: {v["first_error"]["type"]}: {v["first_error"]["error"]}')
    lines.extend(['','- Terminal requested/realized norm·direction·cosine·error/share, zero-owner leakage, ideal/effective Q·capacity·rawSPD applicability는 reduction.json arm realization에 있다.',
        '- Subject loss와 실제 all-token loss gap은 별도로 관측했으며 동일payload가 두 경로의 hidden/loss/전체gradient 동일성을 뜻하지 않는다.',
        '- MAIN vs 대조의 paired는 같은 실제 endpoint/case/token 분모가 있을 때만 재집계했다. 미측정 endpoint는 0점/완료로 대체하지 않았다.',
        f'- 부모 allocated GPU seconds(단일계상): {parents.get("allocated_GPU_seconds","NOT_AVAILABLE")}. Pending·jobsteps 중복 가산 없음.',
        '- Fit exclusive BUILD/subject/pullback/optimizer time과 inclusive fit/writer/batch time을 구분했다. Terminal 실제 observer·고정 diagnostic 비용을 별도 보존했다.',
        '- 새 baseline fit 0; 조건 검산 없는 기존 baseline은 HISTORICAL_REFERENCE/NOT_AVAILABLE. noCP, exact resume NOT_AVAILABLE.',
        '- 원 source/raw/log KEEP. Source·compact report/countCSV/manifest만 Git; NO_BROADCAST_NOT_REQUIRED.'])
    report=out/'report-ko.md';report.write_text('\n'.join(lines)+'\n')
    files=[member(p) for p in sorted(out.iterdir()) if p.is_file()]
    write(out/'inventory.json',dict(task=TASK,files=files,reader_files=reader.files,raw_local_KEEP=True))
    write(out/'terminal.json',dict(task=TASK,status=result['status'],scientific_coverage_complete=complete,
        source=lock['source_commit'],report=member(report),inventory=member(out/'inventory.json')))
    return result

def collect(attempt,out=None,accounting=True):
    attempt=Path(attempt).resolve();out=Path(out).resolve() if out else attempt/'collector';require(not out.exists(),'COLLECTOR_CREATE_ONCE')
    try:return _collect(attempt,out,accounting)
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True);r=dict(task=TASK,status='CPU_REVIEW_TECHNICAL_BLOCKED',error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc(),scientific_coverage_complete=False,original_KEEP=True,automatic_retry=False)
        write(out/'reducer-first-error.json',r)
        report=out/'report-ko.md';report.write_text('# v12-R CPU 검산 기술 차단\n\n'+str(error)+'\n\n원 source/raw KEEP; 새 GPU/평가/자동 retry 0. W20 완료를 인증하지 않는다.\n')
        files=[member(p) for p in out.iterdir() if p.is_file() and p.name not in ('inventory.json','terminal.json')]
        inv=out/'failure-inventory.json';write(inv,dict(task=TASK,files=files,raw_local_KEEP=True))
        write(out/'terminal.json',dict(task=TASK,status=r['status'],report=member(report),inventory=member(inv),scientific_coverage_complete=False))
        return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();r=collect(a.attempt,a.out);print(json.dumps(dict(status=r['status'],actual=r.get('actual'))))
    if r['status']=='CPU_REVIEW_TECHNICAL_BLOCKED':raise SystemExit(1)
