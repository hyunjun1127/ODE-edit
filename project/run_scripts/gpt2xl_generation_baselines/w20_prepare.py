"""Reuse exact native assets/qualification PLAN; no generation W0 preparation."""
import argparse
import copy
import csv
import shutil
import subprocess
from .w20_common import *
from .prepare import dependency_members
from project.run_scripts.base_model_eval.gpt2xl_server1_prepare import runtime
from project.run_scripts.experiment_generation_eval.assets import member as asset_member
from project.run_scripts.experiment_tracking.schema import load_env
from scripts.fixed_counterfact import load_prefix

def prepared_config(prior,attempt,source_attempt,plan_member,transition_member):
    """Pure schedule/scope transform; retained native configs are byte contracts."""
    value=copy.deepcopy(prior)
    value.update(instruction_id=NONCE,task_id=TASK,parent_task_id=PARENT_TASK,attempt=str(attempt),
        arms=list(ARMS),run_instance=dict(attempt=attempt.name),
        source_configs={arm:copy.deepcopy(prior['source_configs'][arm]) for arm in ARMS},
        source_reference=member(source_attempt/'config.json'),source_reference_lock=member(source_attempt/'execution.lock.json'),
        authority=member(ROOT/ENVELOPE),generation_policy=member(ROOT/GENERATION_POLICY),
        policy_members=[member(ROOT/path) for path in POLICY_SHA],
        transition_receipt=transition_member,W0_generation_required=False,
        generation_endpoints=[dict(endpoint='all_seen/post',state='W20',edits=2000,requests=2000)],
        noCP=True,z_disk_cache=False,exact_resume='NOT_AVAILABLE')
    for key in ('cpu_preflight','manual_retry_authority_member','manual_recall_id','cancellation_receipt',
        'registration_recall','parent_instruction_id'):value.pop(key,None)
    generation=value['generation']
    for key in ('shared_W0_root','old_w0_reuse','primary_arm','qualification_receipt','qualification_receipt_member'):
        generation.pop(key,None)
    generation.update(schedule=SCHEDULE,generation_schedule=SCHEDULE,phase='W20_generation',
        qualification_plan_member=plan_member,qualification_plan_sha256=digest(read(verify(plan_member))),
        declared_route='QUALIFICATION_REQUIRED',generation_route='QUALIFICATION_REQUIRED',
        qualification_receipts={arm:str(attempt/arm/'qualification/qualification-actual.json') for arm in ARMS},
        plan=dict(new_generation_case_observations_per_arm_max=2000,
            three_arm_case_observations_max=6000,W0_generation=0,intermediate_generation=0,
            quality_gate=False,ETA='NOT_MEASURED; per-arm route qualification and actual W20 only'))
    return validate_config(value)

def prepare(out,attempt,source_attempt,transition):
    authority();require(attempt.parent==LOCAL and not out.exists() and not attempt.exists(),
        'W20_CREATE_ONCE_NAMESPACE')
    require(not registered_attempts(),'W20_NONCE_ALREADY_REGISTERED')
    prior=read(source_attempt/'config.json');lock=read(source_attempt/'execution.lock.json')
    require(prior['task_id']=='gpt2xl-baselines-generation-cache-repair'
        and lock['config_sha256']==sha(source_attempt/'config.json')
        and set(ARMS)<=set(prior['source_configs']),'W20_NATIVE_SOURCE_REFERENCE')
    transition_member=member(transition);transition_receipt(transition_member,source_attempt)
    for row in prior['assets']:
        stat=Path(row['path']).stat()
        require((stat.st_size,stat.st_ino,stat.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),
            'W20_PROTECTED_ASSET_STAT')
    records=load_prefix(Path(prior['stream']).parent,2000);chunks=list(batches(records))
    contract=read(ROOT/CONTRACT)
    require(sha(prior['stream'])==contract['science']['dataset_sha256'],'W20_EXACT_DATASET')
    schedule=ROOT/contract['science']['schedule']
    require(sha(schedule)==contract['science']['schedule_sha256'],'W20_EXACT_SCHEDULE')
    with schedule.open(newline='') as handle:rows=list(csv.DictReader(handle))
    require([int(row['case_id']) for row in rows]==[r['case_id'] for r in records]
        and [int(row['stream_index0']) for row in rows]==list(range(2000)),
        'W20_ORDERED_OCCURRENCES')
    revision=subprocess.check_output(['git','-C',prior['model'],'rev-parse','HEAD'],text=True).strip()
    require(revision=='15ea56dee5df4983c59b2538573817e1667135e2','W20_GPT2_REVISION')
    rt=runtime();require((rt['torch'],rt['transformers'])==
        (prior['runtime']['torch'],prior['runtime']['transformers']),'W20_RUNTIME_PIN_PRESERVED')
    load_env(prior['tracking']['env_file'])
    oldgen=prior['generation'];reference=read(verify(oldgen['assets_manifest_member']))
    require(reference['status']=='READY' and reference['identity_sha256']==oldgen['reference_assets_sha256'],
        'W20_REFERENCE_READY_IDENTITY')
    for row in reference['files'].values():
        actual=asset_member(row['path'])
        require(actual['bytes']==row['bytes'] and actual['sha256']==row['sha256'],'W20_REFERENCE_BYTES')
    plan=read(verify(oldgen['qualification_plan_member']))
    require(digest(plan)==oldgen['qualification_plan_sha256'] and plan['model_identity']==oldgen['model_identity']
        and len(plan['requests'])<=8,'W20_EXACT_FIXED_QUALIFICATION_PLAN')
    require(shutil.disk_usage(LOCAL).free>=16*1024**3,'W20_OUTPUT_RESERVE')
    value=prepared_config(prior,attempt,source_attempt,oldgen['qualification_plan_member'],transition_member)
    value['runtime']=rt;value['model_revision']=revision
    value['session_boundary_member'],value['session_boundary_identity']=session_boundary()
    value['packs']=[dict(batch=batch,ids=[r['case_id'] for r in current]) for batch,current,_ in chunks]
    value['dependency_sources']=dependency_members(value['source_configs'])
    source=[member(path) for path in sorted((ROOT/'project/run_scripts/experiment_generation_eval').glob('*.py'))
        if not path.name.startswith('test_')]
    value['generation'].update(source_members=source,generation_source_sha=digest(source),
        source_identity=dict(generation_sources=source,policy_sha256=POLICY_SHA[GENERATION_POLICY]))
    value['resources']=dict(project_cap=2,task_cap=2,gpu=1,cpu=8,host_mib=65536,hard_host_mib=183296,
        wall='2-00:00:00',collector_cpu=8,collector_host_mib=24576,collector_wall='04:00:00',
        reserve_bytes=16*1024**3,ETA='NOT_MEASURED; request ceilings only')
    value['broadcast']='NO_BROADCAST_NOT_REQUIRED; same-host protected assets/raw KEEP; compact Git publication'
    write(out/'config.json',value)
    write(out/'preparation.json',dict(status='CPU_BOUND_NOT_MODEL_GPU_PASS',instruction_id=NONCE,
        source_reference=value['source_reference'],transition=transition_member,qualification_plan=oldgen['qualification_plan_member'],
        qualification_actual='NOT_RUN; each new independent cold arm inside approved job',
        W0_generation=0,intermediate_generation=0,generation_schedule=SCHEDULE,model_loads=0,
        native_apply=0,stats_P_recomputed=False,source_raw_preserved=True))
    return out/'config.json'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--attempt',type=Path,required=True);parser.add_argument('--source-attempt',type=Path,required=True)
    parser.add_argument('--transition-receipt',type=Path,required=True)
    args=parser.parse_args();print(prepare(args.out.resolve(),args.attempt.resolve(),
        args.source_attempt.resolve(),args.transition_receipt.resolve()))
