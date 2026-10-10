"""Compact exact completed submission receipt; no scheduler or model calls."""
from pathlib import Path
from official.experiments.prepare import read,write_new
from official.runners.server1.common import member
OUT=Path(__file__).resolve().parent
BASE=Path('/mnt/raid5/janghj/ODE-edit/local/baseline-refresh-s2-flucon-20261010')
registration=BASE/'registration-r1';result=read(registration/'submission.json')
prep=read(BASE/'historical-r1/preparation.json');admission=read(registration/'admission.json')
assert result['released'] and result['held_inspected'] and admission['DAG_width']==3
rows=[]
for key,j in result['jobs'].items():
    snapshot=result['initial_snapshot'][key]
    row=dict(key=key,job_id=j['job_id'],job_name=j['name'],state=snapshot['JobState'],reason=snapshot['Reason'],
        dependencies=j['dependencies'],source=result['source'],official_tree=result['official_tree'],
        output=j['output'],GPU=j['gpus'],argv=j['argv'],script=j['script'],held_inspected=True,released=True,
        online_readback='NOT_OBSERVED_BEFORE_STARTUP',GPU_complete=False)
    if j['config']:
        c=read(j['config']['path']);o=c['original']
        row.update(model=c['model'],method=o['method'],dataset='cf',endpoint='W20_FIRST2000',original_job_id=o['job_id'],
            config=j['config'],config_sha256=c['config_sha256'],checkpoint=o['checkpoint'],
            original_identity=o['identity'],stream=c['stream'],reference_identity=c['reference_identity'],
            generation_schedule='W20_ONLY_FIRST2000',checkpoint_schema=o.get('checkpoint_schema','official-baseline-checkpoint-v1'),
            registration_authority=c['registration_authority'],original_artifacts_changed=False)
    rows.append(row)
write_new(OUT/'submission.json',dict(instruction=result['instruction'],source=result['source'],official_tree=result['official_tree'],
    jobs=rows,execution_lock=result['execution_lock'],submission_member=member(registration/'submission.json'),
    held_inspection_member=member(registration/'held-inspection.json'),admission_member=member(registration/'admission.json'),
    effective_cap=admission['effective_cap'],DAG_width=admission['DAG_width'],resources=admission['resources'],
    disk_available_bytes=admission['disk_available_bytes'],required_reserve_and_output_bytes=admission['required_reserve_and_output_bytes'],
    CPU_fixture_tests=15,official_sources=166,external_task_imports=0,GPU_qualification='NOT_RUN_USER_DISABLED',
    existing_jobs_changed=False,CP_transfers=0,CP_deletions=0,recurring_monitor=False,
    broadcast='NO_BROADCAST_NOT_REQUIRED',independent_reviewer='NOT_USED_OWNER_CPU_AUDIT'))
write_new(OUT/'inventory.json',dict(instruction=result['instruction'],eligible=prep['inventory'],dedup=prep['dedup_excluded'],
    historical_reference=prep['historical_reference'],own_live_admission=admission['existing'],
    SH1_received_publication='d6733abb1afe9022584699b36410bd7e685b9d47',
    SH1_history_generations=dict(GPTJ='62581 COMPLETE_CPU_VERIFIED',Llama='62582 COMPLETE_CPU_VERIFIED',Qwen='62583 ALREADY_REGISTERED'),
    native_Llama_completed=['62259','62260','62261'],author_W20_missing=['62529','62531']))
print({r['job_id']:r['state'] for r in rows})
