"""Exactly three new native jobs; afterany resource DAG and target-only reducer."""
import argparse
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from .native_w20_common import *
from .submit import command,resource_inventory,width,dependencies,held_dependency_conditions
from project.run_scripts.gpt2xl_prune_rect.submit import SOURCES as NATIVE_SOURCES

ROLES=(*ARMS,'collector')
SOURCES=list(dict.fromkeys(NATIVE_SOURCES+[
    'project/run_scripts/gpt2xl_generation_baselines','project/run_scripts/experiment_generation_eval',
    ENVELOPE,CONTRACT,*POLICY_SHA]))

def order(cap):
    require(cap in (1,2),'W20_STRICTER_TASK_CAP')
    return {role:[] if index==0 else [ARMS[index-1]] for index,role in enumerate(ARMS)} if cap==1 else {
        'BASE_MEMIT':[],'BASE_ALPHAEDIT':[],'CAKE':['BASE_MEMIT']}

def role_dependencies(role,frontier,parents,ids):
    return list(ids.values()) if role=='collector' else list(dict.fromkeys(frontier+[ids[p] for p in parents[role]]))

def validate_frontier(existing,frontier):
    """Preserve both admitted protected GPU leaves, never the CPU collector."""
    require(set(frontier)==set(dependencies(existing)),'NATIVE_W20_EXACT_CURRENT_GPU_FRONTIER')
    require(PROTECTED_IDS['collector'] not in frontier
        and not set(frontier)&set(OLD_TARGET_IDS.values()),'NATIVE_W20_CPU_OR_OLD_TARGET_NOT_FRONTIER')
    active={row['job'] for row in existing}
    if {PROTECTED_IDS[arm] for arm in ('ALPHAEDIT_BLUE','PRUNE','RECT')}<=active:
        require({PROTECTED_IDS['PRUNE'],PROTECTED_IDS['RECT']}<=set(frontier)
            and PROTECTED_IDS['ALPHAEDIT_BLUE'] not in frontier,
            'NATIVE_W20_BOTH_PROTECTED_GPU_LEAVES_REQUIRED')
    return frontier

def launcher(source,commit,role,attempt,cpu=8):
    require(role in ROLES,'W20_TARGET_ROLE_ONLY')
    argv=[PYTHON,'-u','-m','project.run_scripts.gpt2xl_generation_baselines.'+
        ('native_w20_collect' if role=='collector' else 'native_w20_run'),'--attempt',str(attempt)]
    if role!='collector':argv+=['--arm',role]
    env=dict(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(cpu),
        MKL_NUM_THREADS=str(cpu),OPENBLAS_NUM_THREADS=str(cpu),TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(attempt/'tmp'/role))
    env[SOURCE_ENV]=commit
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    return '#!/bin/bash\nset -euo pipefail\n'+''.join('export '+key+'='+shlex.quote(value)+'\n'
        for key,value in env.items())+'cd '+shlex.quote(str(source))+'\nexec '+shlex.join(argv)+'\n'

def verify_inputs(value):
    validate_config(value)
    for row in value['dependency_sources']+value['source_config_members']+value['generation']['source_members']:
        verify(row)
    for key in ('source_reference','source_reference_lock','authority','generation_policy','observer_identity',
        'transition_receipt','session_boundary_member'):verify(value[key])
    boundary,boundary_identity=session_boundary()
    require(value['session_boundary_member']==boundary and value['session_boundary_identity']==boundary_identity,
        'W20_BOUNDARY_INPUT_LOCK')
    for row in value['policy_members']:verify(row)
    transition_receipt(value['transition_receipt'],Path(value['source_reference']['path']).parent)
    verify(value['generation']['assets_manifest_member'])
    plan=read(verify(value['generation']['qualification_plan_member']))
    require(digest(plan)==value['generation']['qualification_plan_sha256']
        and plan['model_identity']==value['generation']['model_identity']
        and len(plan['requests'])<=8,'W20_PRELOCKED_QUALIFICATION_PLAN')
    require(all(not Path(path).exists() for path in value['generation']['qualification_receipts'].values()),
        'W20_ACTUAL_QUALIFICATION_NOT_YET_RUN')

def freeze(config,attempt):
    authority();require(attempt.parent==LOCAL and not attempt.exists() and not registered_attempts(),
        'W20_CREATE_ONCE_NO_DUPLICATE')
    require(not command(['git','status','--porcelain','--',*SOURCES],ROOT),'W20_COMMIT_SOURCE_BEFORE_FREEZE')
    value=read(config);require(value['attempt']==str(attempt),'W20_CONFIG_ATTEMPT')
    verify_inputs(value)
    tests=read(verify(value['cpu_preflight']));require(tests['status']=='PASS','W20_CPU_PREFLIGHT')
    for row in tests['source']+tests['helper_source']:verify(row)
    commit=command(['git','rev-parse','HEAD'],ROOT);tree=command(['git','rev-parse','HEAD^{tree}'],ROOT)
    attempt.mkdir();source=attempt/'source';source.mkdir();archive=attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCES],ROOT)
    with tarfile.open(archive) as bundle:
        items=bundle.getmembers();require(len(items)==len({item.name for item in items})
            and all((item.isfile() or item.isdir()) and not Path(item.name).is_absolute()
                and '..' not in Path(item.name).parts for item in items),'W20_SAFE_SOURCE_ARCHIVE')
        bundle.extractall(source,filter='data')
    for row in tests['source']+tests['helper_source']:
        require(sha(source/Path(row['path']).relative_to(ROOT))==row['sha256'],'W20_TESTED_ARCHIVE_BYTES')
    write(attempt/'config.json',value)
    for role in ROLES:
        (attempt/'tmp'/role).mkdir(parents=True,mode=0o700)
        script=attempt/(role+'.sh');write_bytes(script,launcher(source,commit,role,attempt).encode());script.chmod(0o755)
    lock=dict(instruction_id=NONCE,task_id=TASK,source_commit=commit,source_tree=tree,
        config_sha256=sha(attempt/'config.json'),archive=member(archive),
        source_members=[member(path) for path in sorted(source.rglob('*')) if path.is_file()],
        runtime_sources=value['runtime']['source_members'],native_closure=value['dependency_sources'],
        source_config_members=value['source_config_members'],generation_reference=value['generation']['assets_manifest_member'],
        qualification_plan=value['generation']['qualification_plan_member'],
        qualification_plan_sha256=value['generation']['qualification_plan_sha256'],
        qualification_actual='NOT_RUN; new independent cold arm runtime only; no actual PASS claimed',
        actual_qualification_PASS_claim=False,qualification_actual_paths=value['generation']['qualification_receipts'],
        generation_schedule=SCHEDULE,W0_generation_required=False,arms=list(ARMS),
        launchers=[member(attempt/(role+'.sh')) for role in ROLES],tracking_env=member(value['tracking']['env_file']),
        owner=getpass.getuser(),session=SESSION,host='server1',resources=value['resources'],
        noCP=True,checkpoint_saved=False,z_disk_cache=False,exact_resume='NOT_AVAILABLE',
        transition_receipt=value['transition_receipt'],source_reference=value['source_reference'],
        session_boundary_member=value['session_boundary_member'],session_boundary_identity=value['session_boundary_identity'],
        protected_jobs_mutated=False,parent_task_id=PARENT_TASK,automatic_retry=False,
        dependency_policy='afterany exact current resource frontier and two target lanes; no generation W0 prerequisite; no score gate',
        collector_coverage=list(ARMS))
    write(attempt/'execution.lock.json',lock);return value,lock

def arguments(role,dep,attempt,resources):
    require(role in ROLES,'W20_TARGET_ROLE_ONLY');cpu=role=='collector'
    require(all(type(job) is str and job.isdigit() and int(job)>0 for job in dep),'W20_ACTUAL_DEP_IDS')
    argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s1','--nodelist=devbox',
        '--nodes=1','--ntasks=1','--cpus-per-task='+str(resources['collector_cpu'] if cpu else resources['cpu']),
        '--export=NONE','--no-requeue','--job-name='+TASK+'-'+role,'--chdir='+str(attempt/'source'),
        '--mem='+str(resources['collector_host_mib'] if cpu else resources['host_mib'])+'M',
        '--time='+(resources['collector_wall'] if cpu else resources['wall']),
        '--output='+str(attempt/(role+'-%j.out')),'--error='+str(attempt/(role+'-%j.err'))]
    if not cpu:argv+=['--gres=gpu:1']
    if dep:argv+=['--dependency=afterany:'+':'.join(dep)]
    return argv+[str(attempt/(role+'.sh'))]

def inspect(job,role,dep,argv,attempt,resources):
    detail=command(['scontrol','show','job',job,'--oneliner']);cpu=role=='collector'
    cpus=resources['collector_cpu'] if cpu else resources['cpu']
    memory=resources['collector_host_mib'] if cpu else resources['host_mib']
    for field in (f'JobId={job} ',f'JobName={TASK}-{role} ','UserId='+getpass.getuser()+'(',
        'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ',f'CPUs/Task={cpus} ',
        'ReqNodeList=devbox ','Partition=gpu ','QOS=lab_gpu_s1 ',
        'Command='+str(attempt/(role+'.sh'))+' ','WorkDir='+str(attempt/'source')+' '):
        require(field in detail,'W20_HELD_FIELD:'+field)
    require('gres/gpu' not in detail if cpu else 'TresPerNode=gres/gpu:1' in detail,'W20_HELD_GPU')
    require(f'mem={memory}M' in detail or f'mem={memory//1024}G' in detail,'W20_HELD_MEMORY')
    require('TimeLimit='+(resources['collector_wall'] if cpu else resources['wall'])+' ' in detail,'W20_HELD_WALL')
    observed=re.search(r'\bDependency=([^ ]+)',detail)[1]
    own={read(attempt/('submitted-'+name+'.json'))['job'] for name in ROLES
        if (attempt/('submitted-'+name+'.json')).exists()}
    parsed=held_dependency_conditions(observed,dict(afterok=[],afterany=dep),own)
    for missing in set(dep)-parsed['afterany']:
        states=command(['sacct','-n','-X','-j',missing,'--format=JobIDRaw,State','-P'])
        require(any(line.split('|')[0]==missing and line.split('|')[1].split()[0].split('+')[0]
            in TERMINAL_STATES for line in states.splitlines()),'W20_REMOVED_DEP_NOT_TERMINAL')
    submitted=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(submitted and shlex.split(submitted[1])==argv,'W20_HELD_FULL_ARGV')
    script=attempt/(role+'.sh')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==script.read_text().strip(),
        'W20_HELD_SCRIPT_BYTES')
    return dict(job=job,role=role,dependency=dep,argv=argv,resource_detail=detail,launcher=member(script))

def submit(config,attempt):
    authority();require(not registered_attempts(),'W20_NO_DUPLICATE_NONCE')
    names=','.join(TASK+'-'+role for role in ROLES)
    require(not command(['squeue','-h','-u',getpass.getuser(),'--name='+names,'-o','%i|%j']),
        'W20_EXISTING_REGISTERED_KEEP_NO_DUPLICATE')
    before=resource_inventory();existing=list(before['jobs'])
    local=next(row for row in (ROOT/'servers/local/gpu-caps.tsv').read_text().splitlines()
        if row.startswith('server1\t')).split('\t')
    canonical=int(next(row for row in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines()
        if row.startswith('server1\t')).split('\t')[1])
    cap=min(2,int(local[2]),canonical);require(cap>=1 and width(existing)<=cap,'W20_EXISTING_COMBINED_CAP')
    frontier=validate_frontier(existing,dependencies(existing));parents=order(cap);virtual=list(existing);fake={}
    require(not set(frontier)&set(OLD_TARGET_IDS.values()),'W20_CANCELLED_ID_NOT_FRONTIER')
    for index,role in enumerate(ARMS):
        fake[role]=str(999999980+index);dep=role_dependencies(role,frontier,parents,fake)
        virtual.append(dict(job=fake[role],gpus=1,resource_detail='Dependency='+
            ('afterany:'+':'.join(dep) if dep else '(null)')+' '))
    projected=width(virtual);require(projected<=cap,'W20_PROJECTED_COMBINED_CAP')
    node=command(['scontrol','show','node','devbox']);partition=command(['scontrol','show','partition','gpu'])
    qos=command(['sacctmgr','-n','-P','show','qos','lab_gpu_s1','format=Name,MaxWall,MaxTRESPJ,MaxTRESPU'])
    require('Gres=gpu' in node and 'PartitionName=gpu' in partition and qos,'W20_CANONICAL_RESOURCES')
    value,lock=freeze(config,attempt);resources=value['resources']
    memory=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server1',
        '--gpus','1','--mem',str(resources['host_mib'])+'M','--local-limit-mib-per-gpu',local[3]])
    fs=os.statvfs(attempt);raid=shutil.disk_usage(attempt);root=shutil.disk_usage('/')
    require('ALLOW_MEMORY_POLICY' in memory and raid.free>=resources['reserve_bytes']
        and fs.f_favail>32 and root.free>0,'W20_MEMORY_STORAGE_INODES')
    write(attempt/'resource-preflight.json',dict(before=before,node=node,partition=partition,qos=qos,
        effective_cap=cap,projected_GPU_width=projected,frontier=frontier,memory_policy=memory,
        root_available_bytes=root.free,RAID_available_bytes=raid.free,available_inodes=fs.f_favail,
        canonical_policy=member(ROOT/'control/gpu-concurrency-policy.tsv'),protected_jobs_mutated=False,
        generation_schedule=SCHEDULE,W0_generation_prerequisite=False))
    write(attempt/'registration-pass-started.json',dict(instruction_id=NONCE,one_deliberate_pass=True,automatic_retry=False))
    ids={};held=[];role=None
    try:
        for role in ROLES:
            dep=role_dependencies(role,frontier,parents,ids);argv=arguments(role,dep,attempt,resources)
            result=subprocess.run(argv,text=True,capture_output=True)
            write(attempt/('sbatch-'+role+'.json'),dict(role=role,argv=argv,returncode=result.returncode,
                stdout=result.stdout,stderr=result.stderr))
            require(result.returncode==0,'W20_SBATCH_FAILED:'+result.stderr.strip())
            job=result.stdout.strip().split(';')[0];require(job.isdigit(),'W20_ACTUAL_JOB_ID')
            ids[role]=job;write(attempt/('submitted-'+role+'.json'),dict(role=role,job=job,dependency=dep,argv=argv))
            held.append(inspect(job,role,dep,argv,attempt,resources))
        write(attempt/'held-inspection.json',dict(source_commit=lock['source_commit'],config_sha256=lock['config_sha256'],
            jobs=held,passed=True,actual_GPU_qualification_PASS_claim=False,collector_coverage=list(ARMS)))
        fresh=resource_inventory(exclude=tuple(ids.values()))
        final=width(fresh['jobs']+[dict(job=row['job'],gpus=1,resource_detail=row['resource_detail'])
            for row in held if row['role']!='collector'])
        require(final<=cap,'W20_PRE_RELEASE_COMBINED_CAP')
        write(attempt/'pre-release-admission.json',dict(existing=fresh,projected_GPU_width=final,
            effective_cap=cap,passed=True))
        for field in ('source_members','launchers','runtime_sources','native_closure','source_config_members'):
            for row in lock[field]:verify(row)
        for key in ('archive','tracking_env','generation_reference','transition_receipt','qualification_plan','session_boundary_member'):verify(lock[key])
        release=[dict(role=name,job=ids[name],result=command(['scontrol','release',ids[name]])) for name in reversed(ROLES)]
        write(attempt/'release.json',dict(jobs=release))
        snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%R'])
        receipt=dict(instruction_id=NONCE,task_id=TASK,jobs=ids,source_commit=lock['source_commit'],source_tree=lock['source_tree'],
            lock=member(attempt/'execution.lock.json'),config=member(attempt/'config.json'),
            held=member(attempt/'held-inspection.json'),release=member(attempt/'release.json'),
            dependencies={row['role']:row['dependency'] for row in held},snapshot=snapshot,initial='NOT_OBSERVED',
            W_B='NOT_YET_RUN; startup/finish readback in actual jobs',generation_schedule=SCHEDULE,
            qualification_actual='NOT_RUN; plan only, not GPU PASS',collector_coverage=list(ARMS),
            checkpoint_saved=False,exact_resume='NOT_AVAILABLE',combined_GPU_cap=cap,projected_GPU_width=final,
            broadcast='NO_BROADCAST_NOT_REQUIRED',monitoring=False,automatic_retry=False,protected_jobs_mutated=False)
        write(attempt/'submission.json',receipt);return receipt
    except Exception as error:
        write(attempt/'registration-failure.json',dict(instruction_id=NONCE,role=role,jobs_returned=ids,
            error=str(error),automatic_retry=False,protected_jobs_mutated=False,
            release_complete=(attempt/'release.json').exists()))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--attempt',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(submit(args.config.resolve(),args.attempt.resolve()),ensure_ascii=False))
