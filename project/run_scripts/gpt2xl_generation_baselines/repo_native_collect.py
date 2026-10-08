"""MEMIT/AlphaEdit/AlphaEdit-BLUE saved-raw independent CPU reducer at W20.

Preserves independent RPN/native/paired checks and distinguishes interrupted
prefixes, generation completion, remote delivery and scheduler status. No model
load/forward/native apply/upload is performed by this collector.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import subprocess

from .repo_native_common import TASK, NONCE, ARMS, MILESTONES, digest, member, require, sha, write, validate_config
from .collect import (COUNTERS, HISTORY_LAYERS, native_guard, reduce_generation,
    generation_metric_row)
from .run import generation_assets
from project.run_scripts.experiment_generation_eval.native_profile import PROFILE, ROUTE, runtime_identity
from project.run_scripts.experiment_generation_eval.native_observer import read_observed, verify_native_raw
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader, active_flags, compare_summary, reduce_rows)
from project.run_scripts.gpt2xl_native_baselines.collect import _metric_rows, _paired_rows, _csv, _atomic_text
from project.run_scripts.gpt2xl_cake_blue.collect import endpoint, zero_history_hash

SCHEDULE='W20_ONLY_FIRST2000'


def compact_error_code(error):
    """Only controlled technical tags enter compact reports, never raw errors."""
    message=str(error)
    return message if re.fullmatch('[A-Z][A-Z0-9_]{0,159}',message) else 'UNCLASSIFIED_REVIEW_ERROR'


def native_generation_endpoint(reader,receipt,c,records,physical_state):
    """Separate native raw validation plus the pre-existing independent reducer."""
    runtime=digest(runtime_identity(c['generation'],c['generation']['reference_assets_sha256']))
    saved=reader.bound(receipt['rows']);identity=saved['identity']
    state_identity=dict(W=physical_state['W'],H={})
    expected_records=[dict(ordered_occurrence=ordinal,case_id=record['case_id'],
        generation_prompts=record.get('generation_prompts',[]),
        relation_id=record['requested_rewrite'].get('relation_id'),
        target_new_id=record['requested_rewrite']['target_new'].get('id'))
        for ordinal,record in enumerate(records,1)]
    expected_stream=digest(dict(runtime=runtime,eval_seed=20261007,
        ordered_record_identities=expected_records))
    require(identity['sampling_stream_sha256']==expected_stream,
        'NATIVE_COLLECT_EXPECTED_GLOBAL_STREAM')
    require(identity['runtime']==runtime and identity['endpoint']=='W20'
        and identity['cohort']=='ALL_SEEN' and identity['state_sha256']==digest(state_identity)
        and identity['ordered_occurrences']==list(range(1,len(records)+1))
        and receipt['identity']==identity and receipt['identity_sha256']==saved['identity_sha256']==digest(identity)
        and receipt['RNG_restored'] is saved['RNG_restored'] is True
        and receipt['observer_no_mutation'] is saved['observer_no_mutation'] is True,
        'NATIVE_COLLECT_ENDPOINT_STATE_STREAM')
    require(saved['native_execution_member']==receipt['native_execution_member'],
        'NATIVE_COLLECT_EXECUTION_MEMBER_BINDING')
    execution=reader.bound(saved['native_execution_member'])
    require(execution['identity']==identity and execution['profile']==PROFILE
        and execution['route']==ROUTE and execution['native_execution_complete'] is True
        and execution['qualification_performed'] is False and execution['no_fallback'] is True,
        'NATIVE_COLLECT_EXECUTION_NOT_QUALIFICATION')
    assets=generation_assets(c)
    loaded=read_observed(receipt['rows']['path'],expected_runtime=runtime,assets=assets)
    require(loaded['identity']==identity and loaded['rows']==saved['rows'],
        'NATIVE_COLLECT_VERIFIED_ROWS')
    counter_names=('generation_forwards','full_prefix_token_work','physical_forward_calls',
        'prefill_query_tokens','decode_query_tokens','completed_prompts','generated_tokens')
    counters={key:0 for key in counter_names}
    for ordinal,(row,record,expected) in enumerate(zip(saved['rows'],records,expected_records),1):
        raw=reader.json(row['observation_path'])
        require(raw['identity']['record_identity']==expected and raw['occurrence']==ordinal
            and raw['case_id']==record['case_id'] and row['metrics']==raw['metrics']
            and raw['identity']['state_identity']==state_identity,
            'NATIVE_COLLECT_RECORD_TOKEN_COHORT')
        verify_native_raw(raw,expected_runtime=runtime,assets=assets,
            expected_stream=identity['sampling_stream_sha256'])
        reader.bound(row['provenance']['raw_member'])
        observations=raw['observations']
        counters['generation_forwards']+=sum(observation['model_forwards'] for observation in observations)
        counters['full_prefix_token_work']+=sum(observation['full_prefix_token_work'] for observation in observations)
        counters['completed_prompts']+=len(observations)
        counters['generated_tokens']+=sum(observation['continuation_token_count'] for observation in observations)
        for key in ('physical_forward_calls','prefill_query_tokens','decode_query_tokens'):
            counters[key]+=sum(observation[key] for observation in observations)
    require(len(saved['rows'])==len(records),'NATIVE_COLLECT_COMPLETE_ROW_COUNT')
    reduced=reduce_generation(saved['rows'])
    require(reduced==saved['summary']==receipt['summary'],'NATIVE_COLLECT_INDEPENDENT_SUM_COUNT')
    work=receipt['work']
    require(all(type(work[key]) is int and work[key]>=0 for key in
        counter_names+('new_case_observations','cached_case_observations'))
        and type(work['seconds']) in (int,float) and math.isfinite(work['seconds']) and work['seconds']>=0
        and work['new_case_observations']==len(records) and work['cached_case_observations']==0
        and all(work[key]==value for key,value in counters.items())
        and all(execution[key]==counters[key] for key in
            ('physical_forward_calls','prefill_query_tokens','decode_query_tokens'))
        and counters['completed_prompts']==reduced['generation_prompt_count']
        and counters['generated_tokens']==reduced['generated_token_count'],
        'NATIVE_COLLECT_FRESH_WORK_AND_EXECUTION_COUNTERS')
    return dict(summary=reduced,rows=saved['rows'],work=work)


def require_only_terminal_generation(out, commits):
    """Rejected earlier keys stay missing, never fabricated zero measurements."""
    require(not (out/'generation-W0.json').exists() and not (out/'generation').exists(),
        'W20_COLLECT_NO_W0_GENERATION')
    phases=list((out/'generation-work').glob('*.json'))
    require(all(path.name == 'W20_POST.json' for path in phases), 'W20_COLLECT_NO_EARLY_GENERATION_PHASE')
    for commit in commits:
        require(commit['generation_schedule'] == SCHEDULE
            and commit['generation_measured'] is False and commit['generation'] is None
            and commit['terminal_generation_pending'] is (commit['batch']==20),
            'W20_COLLECT_GENERATION_SCHEDULE')


def review_arm(reader, attempt, c, lock, arm, identities, records, progress=None):
    out=Path(attempt)/arm;packs=c['packs'];all_ids=[case for pack in packs for case in pack['ids']]
    require(arm in ARMS and len(packs) == 20 and all(len(p['ids']) == 100 for p in packs)
        and len(set(all_ids)) == 2000 and all_ids == [r['case_id'] for r in records], 'W20_COLLECT_20X100_ORDER')
    metric_rows=[];paired_rows=[];compute_rows=[];counter_rows=[];generation_rows=[];masks=[];commits=[]
    at_write=[];prefixes={}
    result=dict(arm=arm,scientific_status='PARTIAL_OR_NOT_VERIFIED',commits=0,requests=0,state_links=0,
        W0_RPN_available=False,generation_W20_available=False,generation_schedule=SCHEDULE,
        earlier_generation_measured=False,metric_rows=metric_rows,paired_rows=paired_rows,
        compute_rows=compute_rows,counter_rows=counter_rows,generation_rows=generation_rows,
        rect_mask_rows=masks,missing=[],new_model_forwards=0)
    if progress is not None:progress.update(result)
    if not (out/'runtime.json').exists():
        result['missing'].append('RUNTIME_NOT_RECORDED');return result
    runtime=reader.json(out/'runtime.json');cold=runtime['cold_state']
    require(cold['W'] == c['cold_W'] and runtime['source'] == lock['source_commit']
        and runtime['config'] == digest(c) and runtime['arm'] == arm
        and runtime['cold_history_zero_verified'] is True and runtime['generation_schedule'] == SCHEDULE,
        'W20_COLLECT_RUNTIME_COLD_BINDING')
    initial=HISTORY_LAYERS[arm] if arm in ('CAKE','ALPHAEDIT_BLUE') else ()
    require(cold['H'] == {str(layer):zero_history_hash() for layer in initial},
        'W20_COLLECT_COLD_HISTORY_ZERO')
    w0=endpoint(reader,out/'W0',identities,all_ids,'W0',cold,records)
    if w0:
        result['W0_RPN_available']=True
        metric_rows.extend(_metric_rows(arm,'W0_FIRST2000',0,w0['summary']))
        compute_rows.append(dict(arm=arm,phase='W0_RPN',batch=0,seconds=w0['seconds'],reference_only=w0['reference_only']))
    for number in range(1,21):
        folder=out/f'batch-{number:02d}'
        if not (folder/'commit.json').exists():break
        commit=reader.json(folder/'commit.json');ids=packs[number-1]['ids']
        before=cold if not commits else commits[-1]['after']
        require(commit['task'] == TASK and commit['arm'] == arm and commit['batch'] == number
            and commit['case_ids'] == ids and commit['source'] == lock['source_commit']
            and commit['config'] == digest(c) and commit['before'] == before
            and set(commit['after']['W']) == set(cold['W']) and commit['observer_no_mutation'] is True
            and set(commit['after']['H']) == {str(layer) for layer in HISTORY_LAYERS[arm]},
            'W20_COLLECT_COMMIT_EXACT_LINK')
        if arm == 'ALPHAEDIT_BLUE':
            require(all(commit['after']['W'][str(layer)] == before['W'][str(layer)] for layer in (14,15,16)),
                    'W20_COLLECT_BLUE_TWO_LAYERS')
        seen=records[:number*100];seen_ids=all_ids[:number*100]
        pre=endpoint(reader,folder/'pre',identities,ids,f'B{number}_PRE',before,seen)
        post=endpoint(reader,folder/'post',identities,seen_ids if number in MILESTONES else ids,
            f'W{number}',commit['after'],seen)
        require(pre is not None and post is not None,'W20_COLLECT_COMMITTED_RPN_MISSING')
        compare_summary(pre['summary'],commit['pre']);compare_summary(post['summary'],commit['post'])
        current_rows=[row for row in post['rows'] if row['case_id'] in set(ids)]
        post_current=reduce_rows(current_rows);compare_summary(post_current,commit['post_current'])
        require({kind:post_current[kind]['denominator'] for kind in 'RPN'} == dict(R=100,P=200,N=1000),
                'W20_COLLECT_CURRENT_ALWAYS100')
        metric_rows.extend(_metric_rows(arm,f'B{number}_PRE',number*100,pre['summary']))
        metric_rows.extend(_metric_rows(arm,f'W{number}_CURRENT',number*100,post_current))
        paired_rows.extend(_paired_rows(arm,f'B{number}_PRE',f'W{number}_CURRENT',pre['rows'],current_rows))
        at_write.extend(current_rows)
        if number in MILESTONES:
            prefixes[number]=post['rows'];metric_rows.extend(_metric_rows(arm,f'W{number}_ALL_SEEN',number*100,post['summary']))
            metric_rows.extend(_metric_rows(arm,f'W{number}_FIRST500',number*100,
                reduce_rows([row for row in post['rows'] if row['case_id'] in set(all_ids[:500])])))
            paired_rows.extend(_paired_rows(arm,'AT_WRITE',f'W{number}_ALL_SEEN',at_write,post['rows']))
            for born in range(1,number+1):
                cohort=set(packs[born-1]['ids'])
                paired_rows.extend(_paired_rows(arm,f'B{born}_AT_WRITE',f'W{number}',
                    [row for row in at_write if row['case_id'] in cohort],
                    [row for row in post['rows'] if row['case_id'] in cohort],born))
            if w0:
                paired_rows.extend(_paired_rows(arm,'W0',f'W{number}',
                    [row for row in w0['rows'] if row['case_id'] in set(seen_ids)],post['rows']))
            flags=active_flags(seen)
            for active in (True,False):
                selected=[row for row in post['rows'] if flags[row['case_id']] is active]
                if selected:metric_rows.extend(_metric_rows(arm,f'W{number}_'+('ACTIVE' if active else 'SUPERSEDED'),number*100,reduce_rows(selected)))
        masks.extend(native_guard(commit,arm,number,cold))
        counter_rows.append(dict(arm=arm,batch=number,**commit['native_counts']))
        compute_rows.extend([dict(arm=arm,phase='RPN_PRE',batch=number,seconds=pre['seconds']),
            dict(arm=arm,phase='RPN_POST',batch=number,seconds=post['seconds']),
            dict(arm=arm,phase='BATCH_INCLUSIVE',batch=number,seconds=commit['seconds'],nested_cost_not_added_again=True)])
        commits.append(commit);require_only_terminal_generation(out,commits)
        result.update(commits=len(commits),requests=100*len(commits),state_links=max(0,len(commits)-1))
        if progress is not None:progress.update(result)
    require_only_terminal_generation(out,commits)
    terminal=reader.json(out/'terminal.json') if (out/'terminal.json').exists() else None
    totals={key:sum(row[key] for row in counter_rows) for key in COUNTERS}
    if terminal:
        require(terminal['source'] == lock['source_commit'] and terminal['config'] == digest(c)
            and terminal['commits'] == len(commits) and terminal['generation_schedule'] == SCHEDULE
            and all(terminal['native_counts'][key] >= totals[key] for key in COUNTERS),
            'W20_COLLECT_TERMINAL_AND_FAILED_COST')
        compute_rows.append(dict(arm=arm,phase='PROGRAM_WALL',seconds=terminal['program_seconds'],not_added_to_allocation=True))
    if 5 in prefixes and 20 in prefixes:
        paired_rows.extend(_paired_rows(arm,'W5_FIRST500','W20_FIRST500',prefixes[5],
            [row for row in prefixes[20] if row['case_id'] in set(all_ids[:500])]))
    if (out/'generation-W20.json').exists():
        require(len(commits)==20,'NATIVE_COLLECT_GENERATION_REQUIRES_COMMITTED20')
        value=reader.json(out/'generation-W20.json');last=commits[-1]
        require(value['generation_schedule']==SCHEDULE and value['completion_verified'] is True
            and value['generation_measured_at_edits']==2000
            and value['physical_state']==dict(W=last['after']['W'],H={})
            and value['committed_batch20_sha256']==sha(out/'batch-20/commit.json')
            and value['native_execution_member']==value['observation']['native_execution_member'],
            'NATIVE_COLLECT_TERMINAL_COMMITTED_STATE')
        observed=native_generation_endpoint(reader,value['observation'],c,records,last['after'])
        require(observed['summary']['planned_count']==2000,'NATIVE_COLLECT_GENERATION_PLANNED2000')
        generation_rows.append(generation_metric_row(arm,'W20_ALL_SEEN',2000,observed))
        compute_rows.append(dict(arm=arm,phase='W20_GENERATION',batch=20,**observed['work']))
        result['generation_W20_available']=True
        result['native_execution_member']=value['native_execution_member']
    else:
        result['missing'].append('W20_GENERATION_NOT_COMPLETED')
    phase_path=out/'generation-work'/'W20_POST.json'
    if phase_path.exists() and not result['generation_W20_available']:
        value=reader.json(phase_path)
        require(value['phase'] == 'W20_POST' and value['batch'] == 20
            and value['scientific_commit_not_asserted'] is True,'W20_COLLECT_FAILED_RETURNED_WORK')
        work=value['observation']['work']
        compute_rows.append(dict(arm=arm,phase='RETURNED_W20_GENERATION_WITH_COMMITTED_EDITS',**work,
            cost_not_additive_to_program_or_allocation=True,completed_generation_not_asserted=True))
    completed=(len(commits) == 20 and w0 is not None and result['generation_W20_available']
               and terminal is not None and terminal['status'] == 'COMPLETED')
    if completed:
        require(terminal['completed_batches'] == 20 and terminal['edits'] == 2000
            and terminal['state'] == commits[-1]['after']
            and terminal['generation_W20_complete'] is True
            and terminal['terminal_transforms'] == (1 if arm == 'PRUNE' else 0)
            and terminal['prune_applied'] is (arm == 'PRUNE')
            and all(terminal['native_counts'][key] == totals[key] for key in COUNTERS),
            'W20_COLLECT_COMPLETE20_NATIVE_COUNTS')
    result.update(scientific_status='COMPLETED_VALIDATED' if completed else 'PARTIAL_OR_NOT_VERIFIED',
        measured_native_counts=totals,terminal_status=terminal.get('status') if terminal else None,
        failed_or_uncommitted_native_counts={key:terminal['native_counts'][key]-totals[key] for key in COUNTERS} if terminal else None,
        no_PRICE_KKT=True,earlier_generation_measured=False)
    if (out/'tracking-identity.json').exists():
        tracking=reader.json(out/'tracking-identity.json');cfg=tracking['config'];generation=c['generation']
        require(cfg['task_id'] == TASK and cfg['arm'] == arm and cfg['model'] == 'gpt2xl'
            and cfg['model_family'] == 'gpt2' and cfg['role'] == 'scientific'
            and cfg['generation_schedule'] == SCHEDULE
            and cfg['generation_metric_schema'] == generation['schema']
            and cfg['generation_profile'] == generation['profile']
            and cfg['generation_source_sha'] == generation['generation_source_sha']
            and cfg['generation_repair_instruction'] == NONCE and cfg['baseline'] == arm
            and tracking['source_sha'] == lock['source_commit'] and tracking['config_sha'] == lock['config_sha256'],
            'W20_COLLECT_TRACKING_IDENTITY')
        finish=reader.json(out/'tracking-finish.json') if (out/'tracking-finish.json').exists() else {}
        result['tracking']=dict(run_id=tracking['run_id'],url=tracking.get('url'),job_id=cfg['job_id'],
            startup_status=tracking.get('startup_readback',{}).get('status'),finish_status=finish.get('status','NOT_RECORDED'),
            SDK_acceptance_not_remote_readback=True,scientific_completion_is_separate=True)
    return result


def allocation_once(reader, attempt, lock, runner=None, owner=None):
    path=Path(attempt)/'submission.json'
    if not path.exists():return dict(status='NOT_RECORDED',queries=0,reason='OWN_SUBMISSION_NOT_FOUND')
    queries=0
    try:
        submission=reader.json(path)
        require(submission['instruction_id'] == NONCE and submission['task_id'] == TASK
            and submission['source_commit'] == lock['source_commit'],'W20_OWN_ACCOUNTING_BINDING')
        ids={arm:str(submission['jobs'][arm]) for arm in ARMS}
        require(len(set(ids.values())) == 3 and all(re.fullmatch('[1-9][0-9]*',job) for job in ids.values()),
                'W20_OWN_ACCOUNTING_THREE_IDS')
        expected_owner=pwd.getpwuid(os.getuid()).pw_name if owner is None else owner
        require(lock['owner'] == expected_owner,'W20_OWN_ACCOUNTING_OWNER');queries=1
        result=(subprocess.run if runner is None else runner)(['sacct','-X','-n','-P','-j',','.join(ids.values()),
            '--format=JobIDRaw,JobName%120,User,State,ExitCode,ElapsedRaw,AllocTRES'],text=True,capture_output=True,timeout=20,check=False)
        require(result.returncode == 0 and len(result.stdout) <= 1024**2,'W20_OWN_ACCOUNTING_BOUNDED')
        records={}
        for line in result.stdout.splitlines():
            fields=line.strip().split('|')
            if fields and fields[-1] == '':fields.pop()
            require(len(fields) == 7,'W20_OWN_ACCOUNTING_COLUMNS')
            job,name,user,status,exit_code,elapsed,tres=fields
            require(job in ids.values() and job not in records,'W20_OWN_ACCOUNTING_EXACT_PARENT')
            arm=next(arm for arm,value in ids.items() if value == job)
            require(name == TASK+'-'+arm and user == expected_owner and elapsed.isdigit(),
                    'W20_OWN_ACCOUNTING_OWNER_NAME')
            resources=dict(item.split('=',1) for item in tres.split(',') if '=' in item)
            gpu=resources.get('gres/gpu')
            if gpu is None:
                typed=[value for key,value in resources.items() if key.startswith('gres/gpu:')]
                require(len(typed) <= 1,'W20_OWN_ACCOUNTING_GPU_SCHEMA');gpu=typed[0] if typed else '0'
            require(gpu in ('0','1'),'W20_OWN_ACCOUNTING_SINGLE_GPU')
            records[job]=dict(arm=arm,job_id=job,owner=user,scheduler_state=status,exit_code=exit_code,
                parent_elapsed_seconds=int(elapsed),allocated_GPUs=int(gpu),allocated_GPU_seconds=int(elapsed)*int(gpu),
                AllocTRES=tres,child_steps_excluded=True,allocation_not_program_timer_sum=True)
        require(set(records) == set(ids.values()),'W20_OWN_ACCOUNTING_THREE_ROWS')
        return dict(status='RECORDED',queries=queries,records=[records[ids[arm]] for arm in ARMS],
            scheduler_completion_not_scientific_completion=True)
    except Exception as error:
        return dict(status='NOT_RECORDED',queries=queries,error_type=type(error).__name__,
            reason='OWN_ACCOUNTING_UNAVAILABLE_OR_IDENTITY_ERROR',no_retry=True)


def collect(attempt):
    attempt=Path(attempt);reader=Reader();c=reader.json(attempt/'config.json');lock=reader.json(attempt/'execution.lock.json')
    validate_config(c)
    require(c['task_id'] == lock['task_id'] == TASK and c['instruction_id'] == lock['instruction_id'] == NONCE
        and sha(attempt/'config.json') == lock['config_sha256'],'W20_COLLECT_CONFIG_SOURCE')
    for field in ('source_members','runtime_sources','launchers','native_closure','source_config_members'):
        for item in lock.get(field,[]):
            data=reader.bytes(item['path'])
            require(len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'],
                    'W20_COLLECT_SOURCE_MEMBER_BYTES')
    identities=reader.bound(c['observer_identity'])['rows']
    stream=next(item for item in c['assets'] if item['path'] == c['stream']);records=reader.bound(stream)[:2000]
    require([record['case_id'] for record in records] == [case for pack in c['packs'] for case in pack['ids']],
            'W20_COLLECT_ORDERED_FIRST2000')
    out=attempt/'collector';require(not out.exists(),'W20_COLLECT_CREATE_ONCE');out.mkdir();reviews=[]
    for arm in ARMS:
        progress={}
        try:reviews.append(review_arm(reader,attempt,c,lock,arm,identities,records,progress))
        except Exception as error:
            reviews.append(dict(progress,arm=arm,scientific_status='TECHNICAL_BLOCKED_REVIEW',
                error_type=type(error).__name__,error_code=compact_error_code(error),valid_prefix_preserved=True))
    allocation=allocation_once(reader,attempt,lock);write(out/'allocation.json',allocation)
    for record in allocation.get('records',[]):
        next(review for review in reviews if review['arm'] == record['arm']).setdefault('compute_rows',[]).append(dict(record,phase='PARENT_ALLOCATION'))
    tables=(('metric_rows','metrics.csv'),('paired_rows','paired.csv'),('compute_rows','compute.csv'),
        ('counter_rows','native-counts.csv'),('generation_rows','generation.csv'),('rect_mask_rows','rect-support.csv'))
    for key,name in tables:_csv(out/name,[row for review in reviews for row in review.get(key,[])])
    compact=[{key:value for key,value in review.items() if key not in {name for name,_ in tables}} for review in reviews]
    complete=all(review['scientific_status'] == 'COMPLETED_VALIDATED' for review in compact)
    report=['# GPT2-XL MEMIT/AlphaEdit/AlphaEdit-BLUE native generation CPU 검산','',
        '저장 RPN/native/generation scalar·raw를 독립 CPU 재집계했습니다. 모델/forward/fit/W&B upload는 없습니다.','',
        '| Arm | 상태 | 완료 batch | 요청 | W20 generation |','| --- | --- | ---: | ---: | --- |']
    for review in compact:
        report.append('| %s | %s | %s | %s | %s |'%(review['arm'],review['scientific_status'],
            review.get('commits','NA'),review.get('requests','NA'),review.get('generation_W20_available',False)))
    report.extend(['','Generation은 실제 W20 first2000에서만 all_seen/post fluency/consistency로 관측합니다.',
        'W0/current pre/post/W5/W10/W15/first500 generation은 미측정이며 0점으로 채우지 않았습니다.',
        'RPN의 W0/current100 pre/post와 W5/W10/W15/W20 all-seen 및 paired retention은 기존 정의 그대로입니다.',
        'Stock MEMIT noH, AlphaEdit cold 미초기화 H→첫 apply 이후 5층 H, AlphaEdit-BLUE cold 및 누적 H는 L13/L17만 검산합니다.',
        '원 CAKE padded case-batch KV profile을 별도 identity로 검산합니다. 기존 qualification 및 no-cache fallback은 이 profile에 사용하지 않습니다.',
        'SDK 접수, 원격 readback, scheduler 완료, 과학 완료는 서로 다른 사실입니다. Raw text/token은 local-only입니다.',''])
    _atomic_text(out/'report-ko.md','\n'.join(report))
    write(out/'review.json',dict(task=TASK,source=lock['source_commit'],generation_schedule=SCHEDULE,
        reviews=compact,scientific_complete=complete,new_model_forwards=0,GPU_validation_by_collector=False))
    outputs=[member(path) for path in sorted(out.iterdir()) if path.is_file()]
    write(out/'manifest.json',dict(task=TASK,source=lock['source_commit'],inputs=list(reader.files.values()),outputs=outputs,
        raw_copied=False,raw_free_report=True,no_checkpoints=True,automatic_retry=False))
    write(out/'terminal.json',dict(status='COMPLETED',scientific_complete=complete,
        scientific_status='COMPLETED_VALIDATED' if complete else 'PARTIAL_OR_TECHNICAL_BLOCKED',source=lock['source_commit'],
        report=member(out/'report-ko.md'),manifest=member(out/'manifest.json'),new_model_forwards=0))
    return dict(collector_status='COMPLETED',scientific_complete=complete,output=str(out))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=Path,required=True)
    print(json.dumps(collect(parser.parse_args().attempt)))
