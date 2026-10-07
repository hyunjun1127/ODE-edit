"""Explicit failed 60619 single-arm cold retry; preparation is CPU/source-only.

The owner calls register_held, binds the existing pending OURS/baseline DAG to
the actual returned retry ID, then calls release with its cap proof.  There is
no periodic retry, cancellation, polling, all-six registration or checkpoint
resume in this module.
"""
import argparse
import ast
import copy
import getpass
import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from .common import ROOT, LOCAL, TASK, NONCE, require, member, sha, verify, write
from .submit import command, freeze, verify_frozen, arguments, inspect, resource_inventory, job_name

RETRY_NONCE='USER-SH4-GPTJ-ALPHA-CAP075-CAST-REPAIR-20261008'
CELL='ALPHA_CAP075'
OLD=LOCAL/'checkpoint-repair-20261007'
NEW=LOCAL/'cap075-cast-repair-20261008'
AUDIT=LOCAL/'cast-repair-audit-20261008'
SOURCE='298be5da189c3a5f4ffb212e4954ac73583e2be7'
ACK='messages/acks/server4/jlz-price-gptj-cast-repair-20261008.json'
CAST_MODE='cap_endpoint_toward_zero_v1'
SESSION='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'


def boundary():
    require(socket.gethostname()=='server4' and os.environ.get('CODEX_THREAD_ID')==SESSION,'SH4_BOUNDARY')
    branch=command(['git','branch','--show-current'],ROOT)
    require(branch.startswith('codex/') and branch!='main','DEDICATED_NON_MAIN_BRANCH')
    require('hyunjun1127/ODE-edit' in command(['git','remote','get-url','origin'],ROOT),'ORIGIN')


def narrow_preflight(path):
    """New source checks, not inherited toy/model/actual-GPU qualification."""
    paths=sorted((ROOT/'project/run_scripts/jlz_price_gptj').glob('*.py'))
    for source in paths:
        ast.parse(source.read_text(),filename=str(source))
        importlib.import_module('project.run_scripts.jlz_price_gptj.'+source.stem)
    for module in ('run','collect','submit','single_retry'):
        result=subprocess.run([sys.executable,'-m','project.run_scripts.jlz_price_gptj.'+module,'--help'],
            cwd=ROOT,capture_output=True,text=True,timeout=60)
        require(result.returncode==0,'CLI_IMPORT:'+module+':'+result.stderr)
    from .tracking import contract_ready
    contract_ready()
    import torch
    require(not torch.cuda.is_initialized(),'CPU_ONLY_SOURCE_CHECK')
    result=dict(status='PASS',passed=True,tracking_ready=True,numeric_tests=0,toy_runs=0,
        model_load=False,GPU=False,actual_B1='NOT_OBSERVED',
        source=[member(p) for p in paths],
        helper_sources=[member(p) for p in sorted((ROOT/'project/run_scripts/experiment_tracking').glob('*.py'))],
        tracking_contracts=[member(ROOT/p) for p in ('control/wandb-policy.json',
            'control/wandb-method-metric-schema.json','messages/head/2026-10-07-wandb-method-metrics-all-sh.json')],
        source_review_level='OWNER_SOURCE_AUDIT; no independent target-model qualification claim')
    write(path,result)
    return member(path)


def prepare(failure_path):
    boundary();AUDIT.mkdir(exist_ok=True)
    require(not NEW.exists() and not (AUDIT/'prepared-config.json').exists(),'CREATE_ONCE_SINGLE_RETRY_PREPARATION')
    failure_member=member(Path(failure_path).resolve());failure=json.loads(verify(failure_member).read_text())
    require(str(failure.get('failed_job'))=='60619' and failure.get('state')=='FAILED'
        and failure.get('commits')==11 and failure.get('source')==SOURCE,'EXACT_SINGLE_FAILED_JOB_PROOF')
    oldlock=json.loads((OLD/'execution.lock.json').read_text());oldsub=json.loads((OLD/'submission.json').read_text())
    require(oldlock['source_commit']==oldsub['source']==SOURCE and oldsub['jobs'][CELL]=='60619'
        and sha(OLD/'config.json')==oldlock['config_sha256'],'EXACT_OLD_ATTEMPT')
    terminal=json.loads((OLD/CELL/'terminal.json').read_text())
    require(terminal['commits']==11 and terminal['status']=='TECHNICAL_BLOCKED'
        and terminal['job']=='60619' and terminal['source']==SOURCE,'FAILED_PREFIX_NOT_ZERO_NOT_RESUMED')
    c=copy.deepcopy(json.loads((OLD/'config.json').read_text()))
    for row in c['assets']:
        st=Path(row['path']).stat()
        require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_FRESH_STAT')
    for row in c['runtime']['source_members']+c['native_reference']+c['dependency_sources']:verify(row)
    ready=json.loads(verify(c['native_input_reuse']).read_text())
    for key in ('contexts_member','native_input_alignment','native_full_input_binding','observer_identity'):verify(ready[key])
    alpha=c['models']['ALPHA'];reuse=alpha['W0_reuse']
    require(reuse['status']=='QUALIFIED_EXACT_REUSE' and reuse['cold_state']==alpha['cold_W0_H0']
        and reuse['observation_identity']==ready['observation_identity'],'EXACT_COLD_W0_RAW_REUSE')
    for row in reuse['chunks']+[reuse['summary'],reuse['runtime']]:verify(row)
    require(len(ready['packs'])==20 and sum(len(p['ids']) for p in ready['packs'])==2000,'EXACT_NATIVE_INPUT_HORIZON')
    for model in c['models'].values():
        for profile in model['profiles'].values():
            require(profile['eligible_layers']==list(range(3,9)) and profile['lr']==.5
                and profile['lambda_alpha']==10. and profile['c']==.75,'NATIVE_HPARAMS_UNCHANGED')
    alpha['profiles']['CAP075']['endpoint_cast_mode']=CAST_MODE
    c['selected_cells']=[CELL];c['attempt']=str(NEW)
    c['run_instance'].update(attempt=NEW.name,date='2026-10-08')
    parent=json.loads((OLD/CELL/'tracking-identity.json').read_text())
    c['parent_WandB_runs']={CELL:parent['run_id']}
    ack=member(ROOT/ACK);ack_value=json.loads(verify(ack).read_text())
    require(ack_value.get('nonce',ack_value.get('instruction_id'))==RETRY_NONCE,'EXACT_NEW_RETRY_USER_ACK')
    c['authority_members'].append(ack)
    c['execution_override']=dict(instruction_id=RETRY_NONCE,failed_job='60619',selected_cells=[CELL],
        parent_failure=failure_member,old_source=SOURCE,old_successful_prefix=11,
        repair='first FP64-to-FP32 local-CAP endpoint realization toward zero; interior nearest unchanged',
        endpoint_cast_mode=CAST_MODE,same_hparams_and_method=True,
        raw_W0_and_input_only_reuse=True,no_checkpoint_resume=True,cold_W0_H0=True,
        additional_full_B_fit=0,automatic_retry=False,priority='OURS_BEFORE_PENDING_BASELINES')
    free=shutil.disk_usage(LOCAL).free;stats=os.statvfs(LOCAL)
    require(free>=c['resources']['combined_reserve_bytes'] and stats.f_favail>10000,'RESOURCE_BLOCKED_STORAGE')
    c['resources'].update(project_cap=2,task_cap=1,free_bytes=free,free_inodes=stats.f_favail)
    c['cpu_preflight']=narrow_preflight(AUDIT/'static-source.json')
    review=json.loads(verify(c['tracking']['cpu_review']).read_text())
    require(review['helper_ready'] and review['integration']=='FAKE_SDK_PAYLOAD_AXES_IDENTITY_PASS','EXISTING_TRACKING_CPU_EVIDENCE')
    for row in review['helper_sources']:verify(row)
    write(AUDIT/'prepared-config.json',c)
    write(AUDIT/'reuse.json',dict(native_ready=c['native_input_reuse'],W0=reuse,raw_copy=False,
        old_successful_prefix=11,old_prefix_resume=False,old_selected_weights_reused=False,
        new_W0_forward=0,context_generation=0,target_model_qualification='NOT_OBSERVED',
        source_raw_KEEP=True,noCP=True,exact_resume='NOT_AVAILABLE'))
    return dict(status='SINGLE_ARM_COLD_RETRY_PREPARED_NOT_SUBMITTED',config=member(AUDIT/'prepared-config.json'),
        selected_cells=[CELL],GPU_qualification='NOT_OBSERVED',jobs=[])


def effective_cap():
    rows=Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines()
    local=next(x for x in rows if x.startswith('server4\t')).split('\t')
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server4\t')).split('\t')[1])
    cap=min(2,int(local[2]),tracked)
    require(cap>=1 and int(local[3])<=60416,'CURRENT_USER_CAP_AND_MEMORY_POLICY')
    return cap,int(local[3])


def register_held(config_path,gpu_dependency=None):
    """Owner-only one deliberate registration; never invoked by preparation."""
    boundary();cap,local_memory=effective_cap()
    queue=command(['squeue','-h','-u',getpass.getuser(),'--name='+job_name(CELL),'-o','%i|%j|%T'])
    require(not queue,'NO_DUPLICATE_SINGLE_ALPHA_CAP075_REGISTRATION')
    before=resource_inventory();lock,c=freeze(Path(config_path).resolve(),NEW);r=c['resources']
    require(r['project_cap']==2 and r['task_cap']==1 and r['gpu']==1 and r['host_mib']==59392
        and r['hard_host_mib']==60416 and r['collector_host_mib']==24576
        and 1<=r['cpu']<=8 and r['wall']=='2-00:00:00' and r['collector_wall']=='04:00:00','SINGLE_RETRY_RESOURCES')
    policy=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server4',
        '--gpus','1','--mem',str(r['host_mib'])+'M','--local-limit-mib-per-gpu',str(local_memory)])
    require('ALLOW_MEMORY_POLICY' in policy,'MEMORY_POLICY')
    if gpu_dependency:
        require(gpu_dependency.startswith('afterany:') and all(j.isdigit() for j in gpu_dependency.split(':')[1:]),'EXACT_RESOURCE_DEPENDENCY')
    ids={};mapping={};held=[]
    for role in (CELL,'collector'):
        dep=gpu_dependency if role==CELL else 'afterany:'+ids[CELL]
        argv=arguments(role,dep,NEW,r);job=command(argv).split(';')[0]
        require(job.isdigit(),'ACTUAL_JOB_ID')
        ids[role]=job;mapping[role]=dict(job=job,dependency=dep,argv=argv)
        write(NEW/('submitted-'+role+'.json'),dict(nonce=RETRY_NONCE,role=role,status='HELD',**mapping[role]))
        held.append(inspect(job,role,dep,argv,NEW,r))
    verify_frozen(NEW)
    write(NEW/'held-inspection.json',dict(jobs=held,before=before,effective_project_cap=cap,
        selected_cells=[CELL],task_cap=1,new_GPUs=1,collector_GPUs=0,memory_policy=policy,
        source_inspected=True,resource_graph_release_gate='OWNER_MUST_REBIND_PENDING_OURS_AND_BASELINES_TO_ACTUAL_ID',
        old_job_mutations=0,automatic_retry=False))
    write(NEW/'submission.json',dict(nonce=RETRY_NONCE,task_id=TASK,status='HELD_NOT_RELEASED',jobs=ids,mapping=mapping,
        selected_cells=[CELL],source=lock['source_commit'],lock=member(NEW/'execution.lock.json'),
        held=member(NEW/'held-inspection.json'),GPU_qualification='INTEGRATED_ACTUAL_B1_NOT_OBSERVED',
        W20='NOT_OBSERVED',monitoring_active=False,automatic_retry=False))
    return dict(status='SINGLE_ARM_AND_COLLECTOR_HELD',jobs=ids,attempt=str(NEW))


def release(resource_proof):
    """Release only after owner publishes the actual retry-ID cap2/priority DAG."""
    boundary();cap,_=effective_cap();verify_frozen(NEW)
    proof_member=member(Path(resource_proof).resolve());proof=json.loads(verify(proof_member).read_text())
    sub=json.loads((NEW/'submission.json').read_text());ids=sub['jobs']
    require(sub['status']=='HELD_NOT_RELEASED' and proof['retry_job']==ids[CELL]
        and proof['maximum_possible_GPU_concurrency']<=cap and proof['running_mutations']==0
        and proof['baseline_after_ours'] is True,'OWNER_ACTUAL_CAP_PRIORITY_PROOF')
    c=json.loads((NEW/'config.json').read_text())
    for role in (CELL,'collector'):
        row=sub['mapping'][role];inspect(ids[role],role,row['dependency'],row['argv'],NEW,c['resources'])
    for role in ('collector',CELL):
        result=command(['scontrol','release',ids[role]])
        write(NEW/('released-'+role+'.json'),dict(job=ids[role],command_succeeded=True,result=result))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%b|%N|%r'])
    sub.update(status='RELEASED',bounded_initial_snapshot=snapshot,resource_priority_proof=proof_member)
    # The held submission is immutable provenance, not a mutable status file.
    # This separate receipt must not conflict after successful scheduler release.
    write(NEW/'release.json',sub)
    return dict(status='RELEASED',jobs=ids,initial_snapshot=snapshot)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-failure',type=Path)
    parser.add_argument('--register-held-config',type=Path)
    parser.add_argument('--gpu-dependency')
    parser.add_argument('--release-resource-proof',type=Path)
    args=parser.parse_args()
    require(sum(x is not None for x in (args.prepare_failure,args.register_held_config,args.release_resource_proof))==1,'ONE_EXPLICIT_PHASE')
    result=(prepare(args.prepare_failure) if args.prepare_failure else
        register_held(args.register_held_config,args.gpu_dependency) if args.register_held_config else
        release(args.release_resource_proof))
    print(json.dumps(result))


if __name__=='__main__':main()
