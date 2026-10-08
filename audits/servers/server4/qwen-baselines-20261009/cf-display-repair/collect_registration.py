"""Bounded read-only registration audit, no scientific polling or job mutation."""
import json
from pathlib import Path
from project.run_scripts.server4_qwen_submit import read, cmd, job_fields, check, now, dependency_matches, verify
from official.experiments.prepare import write_new, file_sha

root=Path('/data/janghj/ODE-edit/local/qwen-baselines-12-20261009/execution-cf-display-r1')
old=Path('/data/janghj/ODE-edit/local/qwen-baselines-12-20261009/execution-noqual-r2')
verify(root);verify(old)
released=read(root/'released.json');repair=read(root/'cf-repair.json')
entries=[]
for item in released['jobs']:
    inspected=read(root/'inspections'/f"{item['job_id']}.json")['raw']
    for text in ('QOS=lab_gpu_s4 ', 'Partition=gpu ', 'Requeue=0 ',
                 'TimeLimit='+('2-00:00:00 ' if item['kind']=='gpu' else '04:00:00 ')):
        check(text in inspected,'HELD_RESOURCE_MISMATCH')
    _,f=job_fields(item['job_id'])
    check(f['JobState']=='PENDING' and f['Reason']=='Dependency','EXPECTED_RELEASED_RESOURCE_PENDING')
    check(dependency_matches(f['Dependency'],item['dependency']),'RELEASED_DEPENDENCY_CHANGED')
    config=root/'configs'/f"{item['logical_main_row']}.json"
    check(file_sha(config)==file_sha(old/'configs'/config.name),'SCIENCE_CONFIG_CHANGED')
    fields={k:item[k] for k in ('logical_main_row','method','dataset','kind','job_id','name','dependency','source','config_sha256','submitted_at')}
    fields.update(state=f['JobState'],reason=f['Reason'],rerun_attempt=root.name,model='qwen25',
                  cold_identity='INDEPENDENT_BASE_W0_NATIVE_INITIAL_STATE_NO_OLD_CP_RESUME',
                  config_file_sha256=file_sha(config),WandB='NOT_STARTED_NOT_REMOTE_VERIFIED')
    if item['kind']=='gpu':
        submission=read(root/'archive'/item['logical_main_row']/'submission.json')
        check(submission['actual_registered_job_id']==item['job_id'],'ARCHIVE_JOB_BINDING')
        check(submission['checkpoint_identity']['code_commit']==item['source'],'ARCHIVE_SOURCE_BINDING')
        fields['checkpoint_identity']=submission['checkpoint_identity']
    entries.append(fields)
retained=[]
for item in repair['retained']:
    _,f=job_fields(item['job_id'])
    dependency='afterok:'+released['jobs'][-1]['job_id'] if item['job_id']=='61755' else item['dependency']
    check(f['JobState']=='PENDING' and f['Reason']=='Dependency','RETAINED_RELEASED_STATE')
    check(dependency_matches(f['Dependency'],dependency),'RETAINED_DEPENDENCY')
    check(f['Command']==str(old/'scripts'/f"{item['logical_main_row']}-{item['kind']}.sh"),'RETAINED_SOURCE_MUTATION')
    retained.append(dict(job_id=item['job_id'],name=item['name'],dataset='zsre',source=item['source'],
                         config_sha256=item['config_sha256'],dependency=dependency,state=f['JobState']))
cancelled=[]
for item in repair['affected']:
    evidence=read(root/'repair-cancellations'/f"{item['job_id']}.json")
    check('JobState=CANCELLED ' in evidence['after'],'CANCELLATION_EVIDENCE')
    cancelled.append(dict(job_id=item['job_id'],name=item['name'],state='CANCELLED',never_started=True,
                          source_preserved=item['source'],reason=repair['reason']))
stream=read(root/'streams/cf-stream.lock.json')
receipt=dict(instruction_id='USER-SH1-GH-CF-DISPLAY-REPAIR-KEEP-HEALTHY-20261009-R1',
             ack_nonce='GH-SH4-CF-DISPLAY-REPAIR-KEEP-HEALTHY-20261009-R1',observed_at=now(),
             server='server4',status='REPLACEMENTS_RELEASED_RESOURCE_PENDING',attempt_root=str(root),
             execution_source=read(root/'source-lock.json')['code_commit'],
             source_lock_sha256=file_sha(root/'source-lock.json'),input_lock_sha256=file_sha(root/'input-lock.json'),
             ordered_sample_lock_sha256=file_sha(root/'streams/cf-stream.lock.json'),
             original_actual_failure='NOT_OBSERVED',actual_raw='NOT_OBSERVED_UNSTARTED',
             CPU_tests=52,source_files_verified=157,external_task_imports=0,
             qualification='NOT_RUN_USER_DISABLED',GPU_resume_equivalence='NOT_RUN_USER_DISABLED',
             actual_GPU_main='NOT_STARTED',actual_WandB_online='NOT_STARTED',
             cancelled=cancelled,new_jobs=entries,retained_zsre=retained,
             protected_resource_frontier=read(root/'resource-frontier.json'),
             archive_actual_transfers=0,archive_actual_deletions=0,
             report='experiment-reports/servers/server4/qwen-baselines-20261009/cf-display-repair-ko.md',
             raw_provenance_keep=True,recurring_monitor=False)
write_new(Path(__file__).parent/'submission.json',receipt)
print(json.dumps(dict(status=receipt['status'],jobs=[e['job_id'] for e in entries],retained=[e['job_id'] for e in retained])))
