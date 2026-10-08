"""One explicit native-profile registration pass; old RUNNING source untouched."""
import getpass
import json
import shlex
import subprocess
import tarfile
from pathlib import Path
from .generation_native_common import *
from .generation_submit import SOURCES as BASE_SOURCES, admission, command, field, inspect_held, sbatch_argv
from .generation_cache_submit import inventory

SOURCES = list(dict.fromkeys([*BASE_SOURCES, AUTHORITY]))
PROTECTED_ATTEMPT = LOCAL / 'final-generation-v1/attempt-r1'
PROTECTED_SOURCE = '573e25c58e0a351e7cf88998e9916ff7ef32686f'
PROTECTED_LOCK_SHA = '03d703230492ed2f85c5ba373c0e916aa17aca97aebca3562bf7b56070766125'

def protected_identity():
    require(sha(PROTECTED_ATTEMPT/'execution.lock.json') == PROTECTED_LOCK_SHA,
        'PROTECTED_OLD_LOCK_UNCHANGED')
    old = read(PROTECTED_ATTEMPT/'submission.json')
    require(old['jobs']['BASE_MEMIT'] == '61428' and old['source_commit'] == PROTECTED_SOURCE,
        'PROTECTED_OLD_MEMIT_SOURCE_ID')
    detail = command(['scontrol','show','job','61428','--oneliner'])
    require(field(detail,'JobId') == '61428'
        and (field(detail,'UserId') or '').startswith(getpass.getuser()+'(')
        and field(detail,'ReqNodeList') == 'server2'
        and field(detail,'Command') == str(PROTECTED_ATTEMPT/'BASE_MEMIT.sh')
        and field(detail,'WorkDir') == str(PROTECTED_ATTEMPT/'source'), 'PROTECTED_MEMIT_CURRENT_IDENTITY')
    return dict(job='61428', source_commit=PROTECTED_SOURCE, lock_sha256=PROTECTED_LOCK_SHA,
        detail=detail,state=field(detail,'JobState'),mutation=False)

def dependencies(role,frontier,jobs,cap):
    """Two independent native lanes; MEMIT serializes the protected old cell."""
    require(role in (*ARMS,'collector') and 1 <= cap <= 2,'NATIVE_EXACT_ROLE_CAP')
    require(len(set(frontier)) == len(frontier) and all(j.isdigit() for j in frontier),
        'NATIVE_EXACT_FRONTIER_IDS')
    require(set(jobs)==set(ARMS[:len(jobs)]) and all(j.isdigit() for j in jobs.values()),
        'NATIVE_DETERMINISTIC_REGISTRATION')
    if role=='collector':
        require(set(jobs)==set(ARMS),'NATIVE_ALL_SIX_BEFORE_COLLECTOR')
        return [jobs[arm] for arm in ARMS]
    index=ARMS.index(role)
    require(len(jobs)==index,'NATIVE_REGISTER_ROLE_ORDER')
    if index==0:return list(frontier)
    if cap==1:return [jobs[ARMS[index-1]]]
    if role=='BASE_ALPHAEDIT':
        # A single protected running GPU occupies lane A; lane B can start.
        # Any additional admitted frontier makes both new heads wait safely.
        return [] if not frontier or frontier==['61428'] else list(frontier)
    parent={'CAKE':'BASE_MEMIT','ALPHAEDIT_BLUE':'BASE_ALPHAEDIT','PRUNE':'CAKE','RECT':'ALPHAEDIT_BLUE'}[role]
    return [jobs[parent]]

def launcher(attempt,role,config,source):
    r=config['resources'];cpu=r['collector_cpu'] if role=='collector' else r['cpu']
    env=dict(PYTHONPATH=str(attempt/'source'),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS=str(cpu),MKL_NUM_THREADS=str(cpu),OPENBLAS_NUM_THREADS=str(cpu),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONHASHSEED='20261002')
    env[SOURCE_ENV]=source
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    module='project.run_scripts.gptj_native_baselines.generation_native_'+('collect' if role=='collector' else 'run')
    argv=[config['runtime']['python'],'-u','-m',module,'--attempt',str(attempt)]
    if role!='collector':argv += ['--arm',role]
    return '#!/bin/bash\n# ODEEDIT_SLURM_SERVER=server2\nset -euo pipefail\n'+''.join(
        'export '+key+'='+shlex.quote(value)+'\n' for key,value in env.items())+'cd '+shlex.quote(
        str(attempt/'source'))+'\nexec '+shlex.join(argv)+'\n'

def freeze_lock(config,source,attempt):
    gen=config['generation']
    return dict(instruction_id=NONCE,task_id=TASK,parent_task_id=config['parent_task_id'],
        source_commit=source,source_tree=command(['git','rev-parse','HEAD^{tree}']),
        config_sha256=sha(attempt/'config.json'),
        source_members=[member(p) for p in sorted((attempt/'source').rglob('*')) if p.is_file()],
        runtime_sources=config['runtime']['members'],archive=member(attempt/'source.tar'),
        launchers=[member(attempt/(role+'.sh')) for role in (*ARMS,'collector')],
        tracking_env=member(config['tracking']['env_file']),owner=getpass.getuser(),session=SESSION,
        shared_generation_source=gen['source_sha'],shared_generation_tree=gen['package_tree'],
        reference=gen['reference_manifest'],reference_identity=gen['reference_assets_sha256'],
        native_profile=PROFILE,native_route=ROUTE,registration_profile='native-repo-r1',
        tracking_attempt=config['tracking_attempt'],manual_native_authority=config['manual_native_authority'],
        generation_schedule=TRACKING_SCHEDULE,final_generation_calls_per_arm=1,
        final_generation_requests_per_arm=2000,W0_generation_calls=0,intermediate_generation_calls=0,
        qualification_performed=False,qualification_prerequisite=False,native_execution='NOT_OBSERVED',
        noCP=True,exact_resume='NOT_AVAILABLE',resources=config['resources'],count_plan=counts(),
        raw_generation_local_only=True,source_freeze_distinct_from_report=True,
        no_new_monitor_or_automatic_retry=True)

def submit():
    authority();path=PREPARATION/'config.json';config=read(path);ready(config)
    require(not ATTEMPT.exists() and not list(NATIVE_LOCAL.glob('attempt-*/submitted-*.json')),
        'NATIVE_NO_DUPLICATE_REGISTRATION')
    require(not command(['git','status','--porcelain','--',*SOURCES]),'NATIVE_SOURCE_COMMITTED')
    check=read(NATIVE_LOCAL/'cpu-integration-r1.json')
    require(check['status']=='PASS_CPU_INTEGRATION' and check['CUDA_initialized'] is False,
        'NATIVE_NARROW_CPU_NOT_ACTUAL_GPU')
    require(check['config']['sha256']==sha(path),'NATIVE_CPU_CONFIG_BOUND')
    for item in check['source_members']:verify(item)
    for item in config['assets']+config['runtime']['members']+[config['observer_identity']]:stat_seal(item)
    for arm in ARMS:
        closure=config['arm_configs'][arm]['native']['closure']
        for row in closure if isinstance(closure,list) else [r for group in closure.values() for r in group]:verify(row)
    for item in config['generation']['shared_source_members']:verify(item)
    require(command(['git','rev-parse','HEAD:project/run_scripts/experiment_generation_eval'])==PACKAGE_TREE,
        'NATIVE_SHARED_PACKAGE_TREE')
    protected=protected_identity()
    resources=admission(TASK,inventory_fn=inventory)
    if protected['state'] in ('RUNNING','COMPLETING','CONFIGURING','PENDING','SUSPENDED'):
        require('61428' in resources['frontier'],'PROTECTED_MEMIT_COUNTED_IN_FRONTIER')
    if resources['frontier']==['61428']:
        require(resources['allocated_GPUs']==1 and protected['state']=='RUNNING',
            'NATIVE_FREE_SECOND_LANE_EXACT_PROOF')
    source=command(['git','rev-parse','HEAD']);ATTEMPT.mkdir();(ATTEMPT/'source').mkdir()
    archive=ATTEMPT/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),source,*SOURCES])
    with tarfile.open(archive) as stream:
        require(all((item.isfile() or item.isdir()) and not Path(item.name).is_absolute()
            and '..' not in Path(item.name).parts for item in stream.getmembers()),'SAFE_NATIVE_ARCHIVE')
        stream.extractall(ATTEMPT/'source',filter='data')
    write(ATTEMPT/'config.json',config)
    for role in (*ARMS,'collector'):
        with (ATTEMPT/(role+'.sh')).open('x') as stream:stream.write(launcher(ATTEMPT,role,config,source))
    lock=freeze_lock(config,source,ATTEMPT);write(ATTEMPT/'execution.lock.json',lock)
    jobs,inspections,depmap={},{},{}
    try:
        for role in (*ARMS,'collector'):
            deps=dependencies(role,resources['frontier'],jobs,resources['cap']);depmap[role]=deps
            argv=sbatch_argv(ATTEMPT,role,config,deps)
            write(ATTEMPT/('submission-command-'+role+'.json'),dict(argv=argv,role=role))
            response=subprocess.run(argv,text=True,capture_output=True,timeout=45)
            write(ATTEMPT/('sbatch-result-'+role+'.json'),dict(returncode=response.returncode,
                stdout=response.stdout.strip(),stderr=response.stderr.strip(),role=role,argv=argv))
            require(response.returncode==0,'SBATCH_REJECTED:'+role+':'+response.stderr.strip()[:800])
            job=response.stdout.strip().split(';')[0];require(job.isdigit(),'ACTUAL_NATIVE_JOB_ID')
            jobs[role]=job
            write(ATTEMPT/('submitted-'+role+'.json'),dict(job=job,argv=argv,dependencies=deps))
            inspections[role]=inspect_held(job,role,ATTEMPT,argv,deps,config)
        require(len(set(jobs.values()))==7,'SEVEN_DISTINCT_NATIVE_JOBS')
        fresh=inventory(exclude=jobs.values())
        require({row['job'] for row in fresh['project']}<=set(resources['frontier']),
            'ADMISSION_RACE_ALL_NEW_JOBS_HELD')
        write(ATTEMPT/'held-inspection.json',dict(jobs=list(inspections.values()),**resources,fresh=fresh,
            protected_MEMIT=protected,concurrency_bound=resources['cap'],
            DAG='protected old MEMIT->newMEMIT->CAKE->PRUNE; independent AlphaEdit->BLUE->RECT; CPU afterany all6',
            qualification_performed=False,actual_native_execution='NOT_OBSERVED',
            source_config_runtime_verified=True,noCP=True,resources_request_not_ETA=True))
        for role in ('collector','RECT','PRUNE','ALPHAEDIT_BLUE','CAKE','BASE_ALPHAEDIT','BASE_MEMIT'):
            response=command(['scontrol','release',jobs[role]])
            write(ATTEMPT/('released-'+role+'.json'),dict(job=jobs[role],response=response,released=True))
    except BaseException as error:
        write(ATTEMPT/'submission-failure.json',dict(status='NATIVE_SUBMISSION_OR_RELEASE_BLOCKED',
            jobs=jobs,error_type=type(error).__name__,error=str(error)[:1200],source=source,
            lock=member(ATTEMPT/'execution.lock.json'),automatic_retry=False,
            all_unrelated_jobs_unchanged=True,protected_MEMIT_unchanged=True,scientific_complete=False))
        raise
    snapshot=command(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%j|%T|%b|%N|%r'])
    receipt=dict(instruction_id=NONCE,task_id=TASK,status='SUBMISSION_HANDOFF',jobs=jobs,dependencies=depmap,
        source_commit=source,source_tree=lock['source_tree'],lock=member(ATTEMPT/'execution.lock.json'),
        config_sha256=lock['config_sha256'],cap=resources['cap'],frontier=resources['frontier'],
        initial_snapshot=snapshot,resources=config['resources'],shared_generation_source=SHARED_SOURCE,
        shared_generation_tree=PACKAGE_TREE,reference_identity=config['generation']['reference_assets_sha256'],
        generation_profile=PROFILE,generation_schedule=TRACKING_SCHEDULE,
        final_generation_calls_per_arm=1,final_generation_requests_per_arm=2000,
        W0_generation_calls=0,intermediate_generation_calls=0,qualification_performed=False,
        native_execution='NOT_OBSERVED',W_B='NEW_ACTUAL_ONLINE_STARTUP_NOT_OBSERVED',
        protected_MEMIT=protected,all_prior_jobs_unchanged=True,scientific_complete=False,
        monitoring_active=False,automatic_retry=False)
    write(ATTEMPT/'submission.json',receipt);print(json.dumps(receipt));return receipt

if __name__=='__main__':submit()
