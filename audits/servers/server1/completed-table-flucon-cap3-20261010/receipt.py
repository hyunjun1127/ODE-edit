"""Compact source/admission/registration receipt; no scheduler query."""
from pathlib import Path
from official.runners.server1.common import read,member,verify
from official.experiments.prepare import write_new
from official.runners.server1.flucon_submit import CAP3_AUTHORITY,CAP3_LOCAL
O=Path(__file__).parent
root=Path('/mnt/raid5/janghj/ODE-edit')
s=read(CAP3_LOCAL/'registration-r1/submission.json')
a=read(CAP3_LOCAL/'registration-r1/admission.json')
jobs=[]
for key,j in s['jobs'].items():
    c=read(verify(j['config'])) if j['config'] else None
    row=dict(job_id=j['job_id'],job_name=j['name'],dependency=j['dependencies'],GPU=j['gpus'],
             config=j['config'],script=j['script'],initial_state=s['initial_snapshot'][key]['JobState'])
    if c:row.update(model=c['model'],method=c['original']['method'],dataset='cf',original_job=c['original']['job_id'],
        original_checkpoint=c['original']['checkpoint'],original_identity=c['original']['identity'],
        ordered_case_ids_sha256=c['original']['ordered_case_ids_sha256'],output=c['output'],
        config_sha256=c['config_sha256'],reference_identity=c['reference_identity'],protocol_instruction=c['instruction'])
    jobs.append(row)
write_new(O/'submission.json',dict(nonce=CAP3_AUTHORITY,accepted_turn='01a1236d-9293-73b0-aff5-7e2789e84602',
    session='01a04939-f93a-7b50-bca0-65438eab2062',cwd=str(root),host='devbox',origin='https://github.com/hyunjun1127/ODE-edit.git',
    registry=member('servers/active/server1.md'),source=s['source'],official_tree=s['official_tree'],execution_lock=s['execution_lock'],
    jobs=jobs,held_inspected=s['held_inspected'],released=s['released'],local_submission=member(CAP3_LOCAL/'registration-r1/submission.json'),
    resources=a['resources'],effective_cap=a['effective_cap'],admitted_width=a['DAG_width'],
    cap_file=member(root/'servers/local/gpu-caps.tsv'),previous_own_local_cap=4,other_rows_unchanged=True,
    current_allocated_at_admission=a['existing']['allocated_gpus'],pending_existing_adjustments=[],existing_jobs_unchanged=True,
    prior_cancelled_eval_ids_preserved=['62262','62263'],dedup_native_completed=['62259','62260','62261'],
    CPU_tests=44,source_files_verified=166,actual_GPU_qualification='NOT_RUN_USER_DISABLED',
    online_readback='NOT_OBSERVED_BEFORE_STARTUP',new_generation_complete=False,README_owner='GH',
    broadcast='NO_BROADCAST_NOT_REQUIRED'))
print('COMPACT_REGISTRATION_READY')
