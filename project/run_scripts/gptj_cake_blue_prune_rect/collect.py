"""Independent stored-row/count audit for four native GPT-J arms; GPU0."""
import argparse
import json
import os
import pwd
import re
import subprocess
from pathlib import Path
from .common import *
from .logic import expected_counts
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader, active_flags, compare_summary, paired, reduce_rows, validate_rows)
from project.run_scripts.gptj_native_baselines.collect import _metric_rows,_paired_rows,_csv,_atomic_text,_expected

COUNT_KEYS=('native_z','write_keys','history_keys','solves','history_appends')

def compact_terminal(terminal):
    allowed=('status','stage','error_type','completed_batches','commits','edits','native_counts',
        'rollback_verified','checkpoint_saved','exact_resume','source','config','program_seconds',
        'peak_host_RSS_bytes','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes')
    return {key:terminal[key] for key in allowed if key in terminal}

def endpoint(reader,folder,identities,ids,name,expected_state,records):
    folder=Path(folder)
    if not (folder/'summary.json').exists():return None
    saved=reader.json(folder/'summary.json')
    reused=(folder/'reuse.json').exists()
    if reused:
        reuse=reader.json(folder/'reuse.json')
        require(name=='W0' and reuse['scalar_bridge_only'] and not reuse['history_or_editor_resume'],'W0_SCALAR_ONLY')
        require(reuse['actual_cold_weights']==expected_state['W'],'W0_ACTUAL_WEIGHTS')
        paths=[reader.bound(row) for row in reuse['chunks']]
    else:
        paths=[reader.json(p) for p in sorted(folder.glob('chunk-*.json'))]
    raw=[]
    for chunk in paths:
        require(chunk['optimizer_feedback'] is False,'OBSERVER_ONLY')
        if not reused:require(chunk['state']==expected_state,'OBSERVED_STATE')
        raw.extend(chunk['rows'])
    validate_rows(raw,_expected(identities,ids),name)
    flags=active_flags(records)
    require(all(row['active_at_endpoint']==flags[row['case_id']] for row in raw),'ENDPOINT_ACTIVE_IDENTITY')
    reduced=reduce_rows(raw);compare_summary(reduced,saved['summary'])
    require(saved['endpoint']==name and saved['state']==expected_state and saved['row_count']==len(raw)
        and saved['row_order']==digest([r['identity'] for r in raw]) and saved['no_mutation'],'ENDPOINT_RECEIPT')
    if reused:require(saved['seconds']==0 and saved['new_forwards']==0,'OLD_COST_NOT_RECHARGED')
    return dict(rows=raw,summary=reduced,seconds=saved['seconds'],reference_only=reused)

def accounting(reader,attempt,lock):
    try:
        submitted=reader.json(attempt/'submission.json')
        require(submitted['task_id']==TASK and submitted['source_commit']==lock['source_commit']
            and submitted['lock']['sha256']==sha(attempt/'execution.lock.json'),'ACCOUNTING_SCOPE')
        ids={arm:submitted['jobs'][arm] for arm in ARMS}
        require(len(set(ids.values()))==4 and all(re.fullmatch('[1-9][0-9]*',j) for j in ids.values()),'EXACT_FOUR_IDS')
        require(lock['owner']==pwd.getpwuid(os.getuid()).pw_name,'ACCOUNTING_OWNER')
        cmd=['sacct','-n','-P','-j',','.join(ids.values()),
            '--format=JobIDRaw,JobName%120,User,State,ExitCode,ElapsedRaw,AllocTRES']
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
        require(result.returncode==0 and len(result.stdout)<1024**2,'ACCOUNTING_BOUNDED')
        values={}
        for line in result.stdout.splitlines():
            parts=line.split('|')
            if len(parts)==8 and not parts[-1]:parts.pop()
            require(len(parts)==7,'ACCOUNTING_SCHEMA')
            j,name,owner,status,exitcode,elapsed,tres=parts
            if j not in ids.values():continue
            arm=next(a for a,v in ids.items() if v==j)
            require(name==TASK+'-'+arm and owner==lock['owner'] and elapsed.isdigit() and j not in values,'ACCOUNTING_IDENTITY')
            match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
            if not match:match=re.search(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?:,|$)',tres)
            gpu=int(match[1]) if match else 0
            require(gpu in (0,1),'ACCOUNTING_GPU')
            values[j]=dict(arm=arm,job_id=j,scheduler_state=status,exit_code=exitcode,elapsed_seconds=int(elapsed),
                allocated_GPUs=gpu,allocated_GPU_seconds=int(elapsed)*gpu,child_steps_excluded=True)
        require(set(values)==set(ids.values()),'ACCOUNTING_FOUR_PARENTS')
        return dict(status='RECORDED',queries=1,records=list(values.values()))
    except Exception as error:
        return dict(status='NOT_RECORDED',error_type=type(error).__name__,no_retry=True)

def review(reader,attempt,c,lock,arm,identities,records,progress=None):
    out=attempt/arm;result=progress if progress is not None else {}
    result.update(arm=arm,commits=0,requests=0,state_links=0,
        metrics=[],paired=[],counts=[],compute=[],issues=[])
    if not (out/'runtime.json').exists():
        result['scientific_status']='NOT_STARTED_OR_STARTUP_FAILED';return result
    runtime=reader.json(out/'runtime.json')
    require(runtime['source']==lock['source_commit'] and runtime['arm']==arm and runtime['initial_history_zero'],'RUNTIME_COLD_SCOPE')
    initial=runtime['initial_state'];layers={str(l) for l in c['arm_layers'][arm]}
    require(initial['W']=={l:c['cold_W'][l] for l in layers},'COLD_WEIGHT_SCOPE')
    require(set(initial['H'])==(layers if arm in ('CAKE','ALPHAEDIT_BLUE') else set()),'COLD_H_SCOPE')
    all_ids=[r['case_id'] for r in records]
    w0=endpoint(reader,out/'W0',identities,all_ids,'W0',initial,records)
    if w0:
        result['metrics'].extend(_metric_rows(arm,'W0_FIRST2000',0,w0['summary']))
        result['compute'].append(dict(arm=arm,phase='W0',seconds=w0['seconds'],reference_only=w0['reference_only']))
    previous=initial;atwrite=[];totals={k:0 for k in COUNT_KEYS}
    for number,current,seen in batches(records):
        folder=out/f'batch-{number:02d}'
        if not (folder/'commit.json').exists():break
        receipt=reader.json(folder/'commit.json');ids=[r['case_id'] for r in current];seen_ids=[r['case_id'] for r in seen]
        require(receipt['task']==TASK and receipt['arm']==arm and receipt['batch']==number
            and receipt['source']==lock['source_commit'] and receipt['config']==digest(c),'COMMIT_SOURCE')
        require(receipt['case_ids']==ids and receipt['before']==previous and set(receipt['after']['W'])==layers,'OWN_STATE_LINK')
        require(set(receipt['after']['H'])==(layers if arm in ('CAKE','ALPHAEDIT_BLUE') else set()),'OWN_H_LINK')
        measured=receipt['native_counts'];require(measured==expected_counts(arm),'MEASURED_NATIVE_COUNTS')
        require(all(receipt['native']['delta'][k]==measured[k] for k in COUNT_KEYS),'NATIVE_COUNTER_RECEIPT')
        require(receipt['observer_no_mutation'] and not receipt['checkpoint_saved'] and receipt['exact_resume']=='NOT_AVAILABLE','NOCP_OBSERVER')
        pre=endpoint(reader,folder/'pre',identities,ids,f'B{number}_PRE',previous,seen)
        postids=seen_ids if number in MILESTONES else ids
        post=endpoint(reader,folder/'post',identities,postids,f'W{number}',receipt['after'],seen)
        require(pre is not None and post is not None,'COMMITTED_OBSERVATIONS')
        compare_summary(pre['summary'],receipt['pre']);compare_summary(post['summary'],receipt['post'])
        current_rows=[r for r in post['rows'] if r['case_id'] in set(ids)]
        summary=reduce_rows(current_rows);compare_summary(summary,receipt['post_current'])
        require({k:summary[k]['denominator'] for k in ('R','P','N')}==dict(R=100,P=200,N=1000),'CURRENT100_NOT_PREFIX')
        result['metrics'].extend(_metric_rows(arm,f'B{number}_PRE',number*100,pre['summary']))
        result['metrics'].extend(_metric_rows(arm,f'W{number}_CURRENT',number*100,summary))
        result['paired'].extend(_paired_rows(arm,f'B{number}_PRE',f'W{number}_CURRENT',pre['rows'],current_rows))
        atwrite.extend(current_rows)
        if number in MILESTONES:
            result['metrics'].extend(_metric_rows(arm,f'W{number}_ALL_SEEN',number*100,post['summary']))
            first500=[r for r in post['rows'] if r['case_id'] in set(all_ids[:500])]
            result['metrics'].extend(_metric_rows(arm,f'W{number}_FIRST500',number*100,reduce_rows(first500)))
            result['paired'].extend(_paired_rows(arm,'AT_WRITE',f'W{number}',atwrite,post['rows']))
            if w0:result['paired'].extend(_paired_rows(arm,'W0',f'W{number}',[r for r in w0['rows'] if r['case_id'] in set(seen_ids)],post['rows']))
            for birth in range(1,number+1):
                cohort=set(all_ids[(birth-1)*100:birth*100])
                result['paired'].extend(_paired_rows(arm,f'B{birth}_AT_WRITE',f'W{number}',
                    [r for r in atwrite if r['case_id'] in cohort],[r for r in post['rows'] if r['case_id'] in cohort],birth))
        if arm=='PRUNE':
            applied='terminal_prune' in receipt['native']
            require(applied==(number==20),'PRUNE_ONE_TERMINAL')
            if applied:require(receipt['native']['terminal_prune']['repair_label']=='PRUNE_TERMINAL_BASE_FIX','PRUNE_LABEL')
        for k in COUNT_KEYS:totals[k]+=measured[k]
        result['counts'].append(dict(arm=arm,batch=number,**measured))
        result['compute'].append(dict(arm=arm,batch=number,phase='whole_batch',seconds=receipt['seconds'],
            pre_eval_seconds=pre['seconds'],post_eval_seconds=post['seconds'],nested_timers_not_summed=True))
        result.update(commits=number,requests=number*100,state_links=max(0,number-1))
        previous=receipt['after']
    terminal=reader.json(out/'terminal.json') if (out/'terminal.json').exists() else {}
    if terminal:
        for k in COUNT_KEYS:
            require(type(terminal.get('native_counts',{}).get(k)) is int and terminal['native_counts'][k]>=totals[k],'TOTAL_COUNTER_LOWER_BOUND')
    complete=terminal.get('status')=='COMPLETED' and result['commits']==20
    result['scientific_status']='COMPLETED_VALIDATED_ROWS_COUNTS' if complete else 'PARTIAL_OR_FAILED'
    result['terminal']=compact_terminal(terminal)
    result['measured_counts']=totals
    return result

def collect(attempt):
    attempt=Path(attempt);reader=Reader();c=reader.json(attempt/'config.json');lock=reader.json(attempt/'execution.lock.json')
    require(c['task_id']==TASK and c['instruction_id']==NONCE and sha(attempt/'config.json')==lock['config_sha256'],'COLLECT_SOURCE')
    identities=reader.bound(c['observer_identity']);records=reader.bound(next(r for r in c['assets'] if r['path']==c['stream']))[:2000];list(batches(records))
    out=attempt/'collector';require(not out.exists(),'CREATE_ONCE_COLLECTOR');out.mkdir();reviews=[]
    for arm in ARMS:
        progress={}
        try:reviews.append(review(reader,attempt,c,lock,arm,identities,records,progress=progress))
        except Exception as error:
            progress.update(arm=arm,scientific_status='TECHNICAL_BLOCKED_REDUCER',
                error_type=type(error).__name__,original_raw_preserved=True)
            progress.setdefault('issues',[]).append(dict(error_type=type(error).__name__,code='STORED_DATA_INCONSISTENCY'))
            reviews.append(progress)
    alloc=accounting(reader,attempt,lock);write(out/'allocation.json',alloc)
    for kind in ('metrics','paired','counts','compute'):
        _csv(out/(kind+'.csv'),[row for result in reviews for row in result.get(kind,[])])
    compact=[{k:v for k,v in row.items() if k not in ('metrics','paired','counts','compute')} for row in reviews]
    complete=all(r['scientific_status']=='COMPLETED_VALIDATED_ROWS_COUNTS' for r in reviews)
    lines=['# GPT-J CAKE / BLUE / PRUNE / RECT 저장 결과 CPU 검산','',
        '새 모델 평가·native fit·Slurm 변경·W&B 업로드 없이 저장 row/분모/상태 연결/계수를 검산했다.','',
        '| Arm | 상태 | 완료 batch | 요청 | 상태 연결 |','| --- | --- | ---: | ---: | ---: |']
    for r in compact:lines.append(f"| {r['arm']} | {r['scientific_status']} | {r.get('commits','NA')} | {r.get('requests','NA')} | {r.get('state_links','NA')} |")
    lines.extend(['','R/P desired=new, N desired=true. NLL preference tie는 실패이며 TF와 자유생성은 별개다.',
        'PRUNE W5/10/15는 dense MEMIT, W20만 PRUNE_TERMINAL_BASE_FIX 후 관측이다.',
        '각 arm은 독립 cold W0/H0이며 CPU collector 성공이 과학 완료를 뜻하지 않는다.',
        'NoCP/exact_resume=NOT_AVAILABLE. 원 raw·실패·미측정은 그대로 보존한다.',
        'Parent allocation과 nested program timers는 합산하지 않는다. 이전 재사용 W0 비용은 신규 청구하지 않는다.',
        '','[지표](metrics.csv) · [paired](paired.csv) · [native 계수](counts.csv) · [계산비용](compute.csv)',''])
    _atomic_text(out/'report-ko.md','\n'.join(lines))
    write(out/'review.json',dict(task=TASK,source=lock['source_commit'],reviews=compact,scientific_complete=complete,new_forwards=0))
    write(out/'manifest.json',dict(inputs=list(reader.files.values()),outputs=[member(p) for p in sorted(out.iterdir()) if p.is_file()],raw_copied=False))
    write(out/'terminal.json',dict(status='COMPLETED',scientific_complete=complete,report=member(out/'report-ko.md'),source=lock['source_commit']))
    return dict(status='CPU_REPORT_WRITTEN',scientific_complete=complete,output=str(out))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    print(json.dumps(collect(p.parse_args().attempt)))
