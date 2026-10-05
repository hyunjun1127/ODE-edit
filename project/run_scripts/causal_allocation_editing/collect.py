"""Independent CPU raw reducer: exact rows/state/counters, never a model run."""
import argparse
import csv
import json
import math
import os
import pwd
import subprocess
import traceback
from pathlib import Path
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader,validate_rows as independent_rows,reduce_rows,compare_summary,paired,harmonic,quantiles,active_flags)
from scripts.fixed_counterfact import load_prefix
from . import *
from .profile import execution

def finite(value,label='NONFINITE_REVIEW_SCALAR'):
    if isinstance(value,float):require(math.isfinite(value),label)
    elif isinstance(value,dict):
        for v in value.values():finite(v,label)
    elif isinstance(value,(tuple,list)):
        for v in value:finite(v,label)

def validate_solver(receipt):
    """Stored finite relations, not a post-hoc gradient/model reconstruction."""
    if receipt['status']=='GENUINE_NATIVE_ALL_NOOP_PENDING_PRICE':
        require(receipt['logical_candidate_evaluations']==receipt['accepted_updates']==receipt['full_gradient_evaluations']==0,'NOOP_COUNTS')
        return dict(evaluations=0,accepted_updates=0,full_gradients=0,rejected_trials=0)
    finite(receipt)
    e,u,g=receipt['logical_evaluations'],receipt['accepted_updates'],receipt['full_gradients']
    require(type(e) is int and type(u) is int and type(g) is int and 1<=e<=50 and 0<=u<=24 and g==u+1<=25,'50_24_25_BUDGETS')
    require(receipt['terminal_gradient_reused'] and receipt['terminal_extra_gradient']==0
            and receipt['terminal_extra_evaluation']==0,'TERMINAL_MAPPING_REUSES_GRADIENT')
    require(receipt['logical_gradient_reduction']=='REQUEST_SUM' and not receipt['norm_gradient_in_smooth']
            and receipt['norm_prox_once'] and receipt['shared_sum_cap'] is None and receipt['per_block_radius']==.75,'OBJECTIVE_POLICY')
    events=receipt['events'];require(len(events)==e,'CANDIDATE_TRACE_CARDINALITY')
    require([r['candidate_id'] for r in events]==list(range(e)),'CANDIDATE_ORDER')
    accepted=[r['candidate_id'] for r in events if r['accepted']]
    require(len(accepted)==u+1 and accepted[-1]==receipt['accepted_candidate_id'],'LAST_ACCEPTED_NOT_LAST_REJECTED')
    require(sum(r['full_gradient'] for r in events)==g,'ACCEPTED_ONLY_BACKWARD')
    require(all(r['accepted'] or not r['full_gradient'] for r in events),'REJECTED_BACKWARD_FORBIDDEN')
    require(receipt['tau']==receipt['eta0'],'FIXED_TERMINAL_TAU')
    require(math.isclose(receipt['J_mean']*receipt['B'],receipt['J_sum'],rel_tol=1e-12,abs_tol=1e-10),'MEAN_REPORT_SUM_GRADIENT')
    return dict(evaluations=e,accepted_updates=u,full_gradients=g,rejected_trials=e-u-1,
                accepted_candidate=accepted[-1],terminal_gradient_reused=True)

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
    compare_summary(result,saved['summary'])
    flags=active_flags(seen)
    require(all(r['active_at_endpoint']==flags[r['case_id']] for r in rows),'SEEN_PREFIX_ACTIVE_METADATA')
    return dict(rows=rows,metrics=result)

def source_cost(attempt,task=TASK):
    """One exact task parent accounting lookup, GPU seconds counted once."""
    receipt=attempt/'submission.json'
    if not receipt.exists():return dict(status='NOT_AVAILABLE',reason='NO_LOCAL_SUBMISSION_RECEIPT')
    data=json.loads(receipt.read_text());jobmap=data.get('jobs',{})
    ids=[int(v) for v in jobmap.values() if type(v) in (int,str) and str(v).isdigit()]
    if not ids:return dict(status='NOT_AVAILABLE',reason='NO_EXACT_PARENT_IDS')
    command=['sacct','-n','-P','-j',','.join(map(str,ids)),
             '--format=JobIDRaw,User,JobName,State,ElapsedRaw,AllocTRES,TotalCPU,ExitCode']
    result=subprocess.run(command,check=True,capture_output=True,text=True,timeout=30)
    parents=[]
    for line in result.stdout.splitlines():
        fields=line.split('|')
        if len(fields)<8 or fields[0] not in set(map(str,ids)):continue
        require(fields[1]==pwd.getpwuid(os.getuid()).pw_name and fields[2]==task,'ACCOUNTING_TASK_OWNER_IDENTITY')
        tres=dict(item.split('=',1) for item in fields[5].split(',') if '=' in item)
        gpu=int(tres.get('gres/gpu',0));elapsed=int(fields[4] or 0)
        parents.append(dict(job=int(fields[0]),state=fields[3],elapsed_seconds=elapsed,GPUs=gpu,
                            allocated_GPU_seconds=gpu*elapsed,TotalCPU=fields[6],exit=fields[7]))
    return dict(status='ONE_BOUNDED_EXACT_PARENT_SNAPSHOT',parents=parents,
                allocated_GPU_seconds=sum(p['allocated_GPU_seconds'] for p in parents),
                steps_not_double_counted=True,pending_not_allocated=True)

def batch_diagnostics(reader,folder,commit):
    """Validate saved scalar telemetry, not a tensor/gradient reconstruction."""
    realization=reader.json(folder/'realization.json');finite(realization)
    layers=set(commit['after']['W'])
    require(realization['candidate']==commit['accepted_candidate'] and realization['terminal']
            and set(realization['layers'])==layers and realization['B']==100,'TERMINAL_REALIZATION_IDENTITY')
    require(realization['no_extra_model_forward'] and realization['no_extra_backward']
            and realization['no_durable_tensors'],'SCALAR_TELEMETRY_NO_EXTRA_COMPUTE')
    for layer,row in realization['layers'].items():
        require(row['effective_Q_status']=='MEASURED' and row['per_layer_cap']==.75
                and row['actual_update']=='double(W_candidate_FP32)-double(W_entry_FP32)',
                'REALIZATION_EFFECTIVE_APPLIED_WEIGHT')
        for kind in ('mean','canonical','rewrite','KL'):
            view=row[kind]
            require('status' not in view and view['columns']>0,'REALIZATION_COVERAGE:'+kind)
        for prefix in ('ideal','effective'):
            require(math.isclose(row[prefix+'_Q'],row[prefix+'_Q_C0']+row[prefix+'_Q_H'],
                    rel_tol=1e-8,abs_tol=1e-8*max(1.,abs(row[prefix+'_Q']))),'RAW_Q_COMPONENT_SUM')
    capture=reader.json(folder/'entry-capture.json')
    require(capture['fresh_capture'] and capture['pack']==commit['native_pack']
            and capture['H_entry']==commit['before']['H'],'FRESH_ENTRY_CAPTURE_RECEIPT')
    pre=reader.json(folder/'pre/summary.json');post=reader.json(folder/'post/summary.json')
    writer=reader.json(verify(commit['writer']))
    native_path=folder/'calibration/native-reference.json'
    native=reader.json(native_path) if native_path.exists() else None
    if native:
        require(native['reference_commits']==native['history_appends']==0 and native['requests']==100,
                'REFERENCE_NOT_AN_ADDITIONAL_COMMIT')
    calls=commit['fit'].get('engine_calls')
    trace=folder/'solver-events.jsonl';event_counts={};gradient_channels={};candidate_ids=[]
    if trace.exists():
        for line in reader.bytes(trace).decode().splitlines():
            event=json.loads(line);finite(event)
            require(event['experiment']==commit.get('experiment',TASK) and event['batch']==commit['batch'],'BATCH_EVENT_NAMESPACE')
            kind=event['event'];event_counts[kind]=event_counts.get(kind,0)+1
            payload=event['payload']
            if kind=='candidate':candidate_ids.append(payload['candidate'])
            if kind=='gradient':
                label='+'.join(payload['channels']);gradient_channels[label]=gradient_channels.get(label,0)+1
                require(not payload['norm_in_smooth_gradient'] and not payload['logical_candidate_added'],
                        'GRADIENT_CHANNEL_PROX_NORM_ONCE')
    cost=dict(batch=commit['batch'],batch_seconds=commit['seconds'],entry_capture_seconds=capture['seconds'],
        main_solver_seconds=commit['fit'].get('seconds'),terminal_scalar_telemetry_seconds=realization['seconds'],
        exact_copy_final_native_capture_H_seconds=writer['seconds'],pre_observer_seconds=pre['seconds'],
        post_observer_seconds=post['seconds'],main_engine_calls=calls,events=event_counts,
        gradient_channels=gradient_channels,native_reference=None if native is None else dict(
            seconds=native['seconds'],forwards=native['forwards'],Adam_updates=native['updates'],
            requests=native['requests'],history_appends=0),
        time_policy='batch is inclusive; listed stage timers are nested/partially overlapping, NOT additive',
        exclusive_reverse_solve_transfer_seconds='NOT_MEASURED',
        physical_F_B_count='engine group/call units retained; not inferred from logical candidates')
    return dict(batch=commit['batch'],accepted_candidate=commit['accepted_candidate'],
        layers=realization['layers'],ideal_Q_sum=realization['ideal_Q_sum'],native_NLL_sum=realization['native_NLL_sum'],
        native_KL_sum=realization['native_KL_sum'],norm_sum=realization['norm_sum'],
        measurement_scope=realization['scope'],no_extra_model_forward=True,no_extra_backward=True),cost

def _collect(attempt,out=None,accounting=True):
    attempt=Path(attempt).resolve();out=Path(out).resolve() if out else attempt/'collector'
    out.mkdir(parents=True,exist_ok=False);reader=Reader();warnings=[];metrics=[];pairs=[]
    c=reader.json(attempt/'config.json');lock=reader.json(attempt/'execution.lock.json')
    profile=execution(c);count_batches=profile['batches'];count_requests=profile['requests'];task=profile['task']
    require(c['task_id']==task and c['instruction_id']==profile['nonce']==lock['instruction_id'],'COLLECTOR_TASK_AUTHORITY')
    require(sha(attempt/'config.json')==lock['config_sha256'],'COLLECTOR_CONFIG_LOCK')
    records=load_prefix(Path(c['stream']).parent,count_requests);ids=[r['case_id'] for r in records]
    require(digest(ids)==c['ordered_ids_sha256'],'COLLECTOR_INPUT_ORDER')
    if count_requests==2000:require(c['ordered_ids_sha256']==ORDERED_SHA,'PRODUCTION_ORDER')
    identities=reader.json(verify(c['observer_identity']))['rows'];main=attempt/'main';commits=[]
    if count_batches==1:require(not (main/'batch-02').exists(),'B1_NO_SECOND_ENTRY')
    w0=endpoint(reader,main/'W0',identities,ids,'W0',c['cold_W0_H0'],records)
    if w0:metrics.append(dict(endpoint='W0',requests=count_requests,metrics=w0['metrics']))
    atwrite=[];prefix_rows={};pricehash=None;price_receipt=None;realizations=[];costs=[]
    counts=dict(evaluations=0,accepted_updates=0,full_gradients=0,rejected_trials=0)
    for number in range(1,count_batches+1):
        folder=main/f'batch-{number:02d}'
        if not (folder/'commit.json').exists():break
        commit=reader.json(folder/'commit.json');entry=reader.json(folder/'entry.json');pack=c['packs'][number-1]
        current=records[(number-1)*100:number*100];seen=records[:number*100];birth=pack['ids']
        require(commit['batch']==entry['batch']==number and commit['source']==entry['source']==lock['source_commit']
                and commit['config']==entry['config']==digest(c),'COMMIT_SOURCE_BATCH')
        require(commit['ids']==entry['ids']==birth and commit['native_pack']==entry['native_pack']==pack['identity'],'COMMIT_PACK_IDENTITY')
        require(commit['before']==entry['state'] and commit['RNG_before']==entry['RNG'],'ENTRY_STATE_RNG')
        if commits:
            previous=commits[-1]
            require(commit['before']==previous['after'] and commit['RNG_before']==previous['RNG_after']
                    and entry['ledger']==previous['ledger'] and entry['context']==previous['context'],'19_OWN_CHAIN_JOINS')
        else:require(commit['before']==c['cold_W0_H0'],'FIRST_BATCH_COLD')
        require(commit['RNG_after']==commit['RNG_before'] and commit['observer_no_mutation'],'RNG_OBSERVER_NONMUTATION')
        layers=list(map(str,c['profile']['eligible_layers']))
        require(commit['history_appends']==len(layers) and set(commit['history'])==set(layers),'HISTORY_ONCE_LAYER_COVERAGE')
        for layer in layers:
            h=commit['history'][layer]
            require(h['before']==commit['before']['H'][layer] and h['after']==commit['after']['H'][layer]
                    and h['appends']==1 and h['requests']==100 and h['rows']=='native_rewrite_only_nestedmean','NATIVE_H_APPEND_RELATION')
        writer=reader.json(verify(commit['writer']))
        require(writer['accepted_weight_copy_exact'] and writer['candidate']==commit['accepted_candidate'],'EXACT_ACCEPTED_COMMIT')
        for layer in layers:require(writer['weight_hashes'][layer]==commit['after']['W'][layer],'ACCEPTED_WEIGHT_HASH')
        realization,cost=batch_diagnostics(reader,folder,commit);realizations.append(realization);costs.append(cost)
        count=validate_solver(commit['fit'])
        for key in counts:counts[key]+=count[key]
        if commit['price'] is not None:
            require(math.isfinite(commit['price']) and commit['price']>0,'POSITIVE_FINITE_PRICE')
            if pricehash is None:
                pricehash=commit['price_sha256'];price_receipt=reader.json(attempt/'price.json');finite(price_receipt)
                raw=dict(price_receipt);receipt_hash=raw.pop('receipt_hash')
                require(digest(raw)==receipt_hash and price_receipt['status']=='VALID_POSITIVE_RADIAL_PRICE'
                        and price_receipt['identity']['source']==lock['source_commit']
                        and price_receipt['identity']['config_sha256']==lock['config_sha256'],
                        'RADIAL_PRICE_SOURCE_RECEIPT_HASH')
                require(price_receipt['d']>0 and price_receipt['t']>price_receipt['n']
                        and price_receipt['lambda_Q']==(price_receipt['t']-price_receipt['n'])/price_receipt['d'],
                        'POSITIVE_PRICE_FROM_UNMODIFIED_OPERANDS')
                require(abs(price_receipt['d']-2*price_receipt['Q'])<=1e-7+1e-5*abs(2*price_receipt['Q']),
                        'ONE_LAYER_RADIAL_D_TWO_Q')
            require(commit['price_sha256']==pricehash==sha(attempt/'price.json'),'IMMUTABLE_PRICE_20BATCH')
            require(commit['price']==price_receipt['lambda_Q'],'ONE_IMMUTABLE_PRICE_VALUE')
        pre=endpoint(reader,folder/'pre',identities,birth,f'B{number}_PRE',commit['before'],seen)
        selected=seen if number in profile['milestones'] else current;selected_ids=[r['case_id'] for r in selected]
        post=endpoint(reader,folder/'post',identities,selected_ids,f'W{number}',commit['after'],seen)
        require(pre is not None and post is not None,'COMMITTED_OBSERVATION_MISSING')
        compare_summary(pre['metrics'],commit['pre']);compare_summary(post['metrics'],commit['post'])
        birthrows=[r for r in post['rows'] if r['case_id'] in set(birth)];compare_summary(reduce_rows(birthrows),commit['post_current'])
        metrics.extend([dict(endpoint=f'B{number}_PRE',requests=100,metrics=pre['metrics']),
                        dict(endpoint=f'W{number}_CURRENT',requests=100,metrics=reduce_rows(birthrows))])
        atwrite.extend(birthrows);pairs.append(dict(from_endpoint=f'B{number}_PRE',to_endpoint=f'W{number}_CURRENT',paired=paired(pre['rows'],birthrows)))
        if number in profile['milestones']:
            prefix_rows[number]=post['rows'];metrics.append(dict(endpoint=f'W{number}_ALL_SEEN',requests=len(seen),metrics=post['metrics']))
            if count_requests>=500:
                metrics.append(dict(endpoint=f'W{number}_FIRST500',requests=500,metrics=reduce_rows(
                    [r for r in post['rows'] if r['case_id'] in set(ids[:500])])) )
            pairs.append(dict(from_endpoint='AT_WRITE',to_endpoint=f'W{number}',paired=paired(atwrite,post['rows'])))
            if w0:pairs.append(dict(from_endpoint='W0',to_endpoint=f'W{number}',paired=paired([r for r in w0['rows'] if r['case_id'] in set(ids[:number*100])],post['rows'])))
        commits.append(commit)
    terminal=reader.json(main/'terminal.json') if (main/'terminal.json').exists() else None
    complete=len(commits)==count_batches and count_batches in prefix_rows and terminal is not None and terminal['status']==profile['status']
    if len(commits)==count_batches:
        require({kind:v['denominator'] for kind,v in reduce_rows(prefix_rows[count_batches]).items()}==dict(R=count_requests,P=2*count_requests,N=10*count_requests),'PROFILE_EXACT_DENOMINATORS')
    if not complete:warnings.append('PARTIAL_OR_TECHNICAL_BLOCKED: no absent endpoint is imputed or called complete.')
    accounting_record=source_cost(attempt,task) if accounting else dict(status='NOT_REQUESTED_CPU_FIXTURE')
    first_error=reader.json(main/'first-error.json') if (main/'first-error.json').exists() else None
    qualification_terminal=reader.json(attempt/'qualification-runtime/terminal.json') if (attempt/'qualification-runtime/terminal.json').exists() else None
    qualification_error=reader.json(attempt/'qualification-runtime/first-error.json') if (attempt/'qualification-runtime/first-error.json').exists() else None
    result=dict(experiment=task,status=profile['status'] if complete else 'PARTIAL_OR_TECHNICAL_BLOCKED',
        source=lock['source_commit'],config_sha256=lock['config_sha256'],commits=len(commits),requests=100*len(commits),
        expected=dict(commits=count_batches,requests=count_requests,joins=count_batches-1,history_appends=5*count_batches),
        actual=dict(joins=max(0,len(commits)-1),history_appends=sum(r['history_appends'] for r in commits)),
        counters=counts,price_sha256=pricehash,calibration=price_receipt,metrics=metrics,paired=pairs,warnings=warnings,
        terminal=terminal,first_technical_error=first_error,qualification_terminal=qualification_terminal,
        qualification_first_technical_error=qualification_error,accounting=accounting_record,
        realization=realizations,cost=dict(per_batch=costs,time_policy='Nested timers are not summed with parent allocation',
            main_peak_RSS_KiB=None if terminal is None else terminal.get('peak_RSS_KiB'),
            main_peak_VRAM_bytes=None if terminal is None else terminal.get('peak_VRAM_bytes'),
            qualification_cost_separate=True),raw_reducer='Independent stdlib arithmetic/token identity validation',
        reviewer='CPU reducer; not model replay, not independent GPU/gradient proof',checkpoint_saved=False,exact_resume='NOT_AVAILABLE')
    write(out/'reduction.json',result);write(out/'metrics.json',metrics);write(out/'paired.json',pairs)
    write(out/'realization.json',realizations);write(out/'cost.json',result['cost'])
    table=[]
    for endpoint_data in metrics:
        m=endpoint_data['metrics']
        for kind,value in m.items():
            table.append(dict(endpoint=endpoint_data['endpoint'],kind=kind,**{k:value[k] for k in ('numerator','denominator','rate','token_micro','prompt_macro','strict_numerator','strict_denominator')},
                              harmonic_RSPSNS=harmonic(m)))
    with (out/'comparison.csv').open('x',newline='') as f:
        if table:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        else:f.write('endpoint,kind,numerator,denominator\n')
    lines=['# Causal Allocation Editing 사실 보고','',f'- 상태: {result["status"]}',
        f'- 실행 source: `{lock["source_commit"]}`',f'- 검산된 commit: {len(commits)}/{count_batches}, 요청: {100*len(commits)}/{count_requests}',
        f'- H append: {result["actual"]["history_appends"]}/{5*count_batches}; own state join: {result["actual"]["joins"]}/{count_batches-1}',
        f'- 후보/accepted update/full gradient: {counts["evaluations"]}/{counts["accepted_updates"]}/{counts["full_gradients"]}',
        '- CPU 원행 재집계이며 새 모델/GPU 평가가 아니다. noCP, exact resume 불가.',
        '- Reference/calibration 및 main/rejected/replay cost는 parent allocated GPU 시간과 중복 합산하지 않는다.',
        '', '| Endpoint | RS | PS | NS |','|---|---:|---:|---:|']
    for point in metrics:
        if point['endpoint']=='W0' or point['endpoint'].endswith('ALL_SEEN'):
            mm=point['metrics'];lines.append('| '+point['endpoint']+' | '+' | '.join(f'{100*mm[k]["rate"]:.3f} ({mm[k]["numerator"]}/{mm[k]["denominator"]})' for k in ('R','P','N'))+' |')
    lines.extend(['','## 실현 및 비용','',
        '- 실현은 actual FP32 weight 차분의 local own-write action이며 requested R 대비 norm/direction/cosine/error를 별도로 기록한다. norm ratio만으로 실현 성공을 판정하지 않는다.',
        '- mean/canonical/rewrite/KL, 층별 share·zero-owner action·ideal/effective Q(C0/H) 원수치는 realization.json에 있다. net/inherited/task 효과를 local action과 혼용하지 않는다.',
        '| Batch/layer | Mean norm ratio | Directional ratio | Cosine | Relative error | Effective Q |',
        '|---|---:|---:|---:|---:|---:|'])
    def shown(value):return 'NOT_AVAILABLE' if value is None else f'{value:.6g}'
    for rr in realizations:
        for layer,value in rr['layers'].items():
            view=value['mean'];lines.append('| '+str(rr['batch'])+'/'+layer+' | '+
                ' | '.join(shown(view[k]) for k in ('norm_ratio','directional_ratio','cosine','relative_error'))+
                ' | '+shown(value['effective_Q'])+' |')
    lines.extend(['', '- Calibration: '+('NOT_AVAILABLE' if price_receipt is None else
        ', '.join(k+'='+shown(price_receipt[k]) for k in ('t','n','d','t_minus_n','lambda_Q')))+
        '; 원 native L8 반환 delta의 radial 가격을 한 번 고정하며 heldout 성적으로 선택하지 않는다.',
        f'- 부모 allocated GPU seconds(한 번 계상): {accounting_record.get("allocated_GPU_seconds","NOT_AVAILABLE")}. Pending은 GPU 할당시간이 아니다.',
        '- Batch/entry/solver/writer/observer/telemetry timer는 inclusive·nested 또는 부분 중첩이다. 이를 합계 GPU 시간으로 중복 가산하지 않았다. 역전파/solve/transfer exclusive 시간 미분리 항목은 NOT_MEASURED.',
        '- Native L8 reference Adam fit·두 calibration adjoint channel 비용은 cost.json에 별도 보존하고 총 parent allocation에 포함한다. logical 후보 수를 물리 F/B 또는 FLOP 수로 간주하지 않는다.',
        '', '## 한계','',*warnings,'','- 새 baseline 실행 0. 조건 결속 없는 기존 baseline은 NOT_AVAILABLE/HISTORICAL_REFERENCE이며 matched 성능으로 표기하지 않는다.',
        '- CPU reducer가 저장된 state hash 연결을 검산했다. RAM tensor 재생 또는 미측정 gradient 성분을 추정하지 않았다.',
        '- 원 raw/source/log는 local KEEP, Git에는 compact 보고·표·manifest만. NO_BROADCAST_NOT_REQUIRED.'])
    report=out/'report-ko.md';report.write_text('\n'.join(lines)+'\n')
    inventory=[member(p) for p in sorted(out.iterdir()) if p.is_file()]
    write(out/'inventory.json',dict(experiment=task,files=inventory,raw_local_KEEP=True,reader_files=reader.files))
    # Report and inventory precede terminal publication.
    write(out/'terminal.json',dict(experiment=task,status=result['status'],report=member(report),
        inventory=member(out/'inventory.json'),source=lock['source_commit'],scientific_coverage_complete=complete))
    return result

def collect(attempt,out=None,accounting=True):
    """A reducer fault publishes a typed failure package, never a fake PASS."""
    attempt=Path(attempt).resolve();out=Path(out).resolve() if out else attempt/'collector'
    require(not out.exists(),'COLLECTOR_OUTPUT_CREATE_ONCE')
    try:return _collect(attempt,out,accounting)
    except Exception as error:
        out.mkdir(parents=True,exist_ok=True)
        task=TASK
        try:task=json.loads((attempt/'config.json').read_text())['task_id']
        except (OSError,ValueError,KeyError):pass
        result=dict(experiment=task,status='CPU_REVIEW_TECHNICAL_BLOCKED',scientific_coverage_complete=False,
            original_KEEP=True,error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),no_model_or_GPU_replay=True,automatic_retry=False,
            no_missing_endpoint_imputation=True)
        write(out/'reducer-first-error.json',result)
        report=out/'report-ko.md'
        report.write_text('# Causal Allocation Editing CPU 검산 기술 차단\n\n'
            '저장 산출물의 독립 CPU 검산이 기술 오류로 차단됐다. 실험 완료 또는 해당 horizon coverage를 인증하지 않는다.\n\n'
            f'- 최초 reducer 오류: {type(error).__name__}: {error}\n'
            '- 원 source/raw/log와 이미 작성된 검산 산출물은 KEEP. 새 GPU/모델 평가·자동 retry 없음.\n')
        inventory=[member(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('inventory.json','terminal.json')]
        write(out/'inventory.json',dict(experiment=task,files=inventory,raw_local_KEEP=True))
        write(out/'terminal.json',dict(experiment=task,status=result['status'],report=member(report),
            inventory=member(out/'inventory.json'),scientific_coverage_complete=False))
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--out',type=Path)
    args=p.parse_args();r=collect(args.attempt,args.out);print(json.dumps(dict(status=r['status'],commits=r.get('commits'))))
    if r['status']=='CPU_REVIEW_TECHNICAL_BLOCKED':raise SystemExit(1)
