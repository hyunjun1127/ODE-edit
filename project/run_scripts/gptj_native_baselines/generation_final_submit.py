"""One immutable final-only registration pass; no retry or monitor loop.

Native GPU launchers/admission/held inspection are reused unchanged. The
collector entry is explicit and source-frozen, never runtime monkeypatched.
Qualification PLAN is bound before submission; actual PASS is not invented.
"""
import getpass
import json
import shlex
import subprocess
import tarfile
from pathlib import Path

from .generation_common import (
    ARMS,SESSION,SOURCE_ENV,member,read,require,sha,stat_seal,verify,write)
from .generation_cache_common import TASK,NONCE
from .generation_final_common import (
    FINAL_LOCAL,PREPARATION,ATTEMPT,AUTHORITY,SCHEDULE,authority,ready,counts)
from .generation_cache_submit import SOURCES as CACHE_SOURCES, inventory
from .generation_plan import dependencies
from .generation_submit import (
    admission,command,inspect_held,launcher as gpu_launcher,sbatch_argv)


SOURCES=list(dict.fromkeys([*CACHE_SOURCES,AUTHORITY]))


def launcher(attempt,role,config,source):
    """Reuse exact scientific launcher; collector gets explicit final reducer."""
    require(role in (*ARMS,'collector'),'FINAL_EXACT_ROLE')
    if role!='collector':return gpu_launcher(attempt,role,config,source)
    cpu=config['resources']['collector_cpu']
    env=dict(PYTHONPATH=str(attempt/'source'),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS=str(cpu),MKL_NUM_THREADS=str(cpu),OPENBLAS_NUM_THREADS=str(cpu),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONHASHSEED='20261002',CUDA_VISIBLE_DEVICES='')
    env[SOURCE_ENV]=source
    argv=[config['runtime']['python'],'-u','-m',
        'project.run_scripts.gptj_native_baselines.generation_final_collect',
        '--attempt',str(attempt)]
    return '#!/bin/bash\n# ODEEDIT_SLURM_SERVER=server2\nset -euo pipefail\n'+''.join(
        'export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+'cd '+shlex.quote(
        str(attempt/'source'))+'\nexec '+shlex.join(argv)+'\n'


def freeze_lock(config,source,attempt):
    generation,repair=config['generation'],config['generation']['repair']
    return dict(instruction_id=NONCE,task_id=TASK,parent_task_id=config['parent_task_id'],
        source_commit=source,source_tree=command(['git','rev-parse','HEAD^{tree}']),
        config_sha256=sha(attempt/'config.json'),
        source_members=[member(p) for p in sorted((attempt/'source').rglob('*')) if p.is_file()],
        runtime_sources=config['runtime']['members'],archive=member(attempt/'source.tar'),
        launchers=[member(attempt/(role+'.sh')) for role in (*ARMS,'collector')],
        tracking_env=member(config['tracking']['env_file']),owner=getpass.getuser(),session=SESSION,
        shared_generation_source=generation['source_sha'],shared_generation_tree=generation['package_tree'],
        reference=generation['reference_manifest'],reference_identity=generation['reference_assets_sha256'],
        qualification_plan=repair['qualification_plan'],qualification_cohort=repair['qualification_cohort'],
        qualification_plan_sha256=repair['qualification_plan_sha256'],
        shared_qualification_plan=repair['shared_qualification_plan'],
        shared_qualification_plan_sha256=repair['shared_qualification_plan_sha256'],
        qualification_status='PLAN_BOUND_NOT_ACTUAL_PASS',actual_GPU_qualification='NOT_RUN',
        actual_qualification_in_first_replacement_job=True,
        noCP=True,exact_resume='NOT_AVAILABLE',resources=config['resources'],
        registration_profile='final-v1',tracking_attempt=config['tracking_attempt'],
        manual_schedule_authority=config['manual_schedule_authority'],
        generation_schedule=SCHEDULE,final_generation_calls_per_arm=1,
        final_generation_requests_per_arm=2000,W0_generation_calls=0,
        intermediate_generation_calls=0,count_plan=counts(),raw_generation_local_only=True,
        source_freeze_distinct_from_report=True,no_new_monitor_or_automatic_retry=True)


def submit():
    authority()
    config_path=PREPARATION/'config.json';config=read(config_path);ready(config)
    attempt=Path(config['attempt'])
    require(attempt==ATTEMPT and not attempt.exists(),'FINAL_ATTEMPT_CREATE_ONCE')
    require(not list(FINAL_LOCAL.glob('attempt-*/submitted-*.json')),'FINAL_NO_DUPLICATE_REGISTRATION')
    require(not command(['git','status','--porcelain','--',*SOURCES]),'FINAL_SOURCE_COMMITTED')
    checks=read(FINAL_LOCAL/'cpu-integration-final.json')
    require(checks['status']=='PASS_CPU_INTEGRATION' and checks['CUDA_initialized'] is False,
            'FINAL_NARROW_CPU_NOT_GPU')
    require(checks['config']['sha256']==sha(config_path),'FINAL_CPU_CONFIG_BOUND')
    for item in checks['source']:verify(item)
    for item in config['assets']+config['runtime']['members']+[config['observer_identity']]:stat_seal(item)
    for arm in ARMS:
        closure=config['arm_configs'][arm]['native']['closure']
        for item in closure if isinstance(closure,list) else [r for g in closure.values() for r in g]:verify(item)
    for item in config['generation']['shared_source_members']:verify(item)
    require(command(['git','rev-parse','HEAD:project/run_scripts/experiment_generation_eval'])
        ==config['generation']['package_tree'],'FINAL_EXACT_SHARED_PACKAGE_TREE')
    resources=admission(TASK,inventory_fn=inventory)
    source=command(['git','rev-parse','HEAD']);attempt.mkdir();(attempt/'source').mkdir()
    archive=attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),source,*SOURCES])
    with tarfile.open(archive) as stream:
        require(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute()
            and '..' not in Path(m.name).parts for m in stream.getmembers()),'SAFE_SOURCE_ARCHIVE')
        stream.extractall(attempt/'source',filter='data')
    write(attempt/'config.json',config)
    for role in (*ARMS,'collector'):
        with (attempt/(role+'.sh')).open('x') as stream:
            stream.write(launcher(attempt,role,config,source))
    lock=freeze_lock(config,source,attempt);write(attempt/'execution.lock.json',lock)
    jobs,inspections,depmap={},{},{}
    try:
        for role in (*ARMS,'collector'):
            deps=dependencies(role,resources['frontier'],jobs,resources['cap']);depmap[role]=deps
            argv=sbatch_argv(attempt,role,config,deps)
            write(attempt/('submission-command-'+role+'.json'),dict(argv=argv,role=role))
            result=subprocess.run(argv,text=True,capture_output=True,timeout=45)
            write(attempt/('sbatch-result-'+role+'.json'),dict(returncode=result.returncode,
                stdout=result.stdout.strip(),stderr=result.stderr.strip(),role=role,argv=argv))
            require(result.returncode==0,'SBATCH_REJECTED:'+role+':'+result.stderr.strip()[:800])
            job=result.stdout.strip().split(';')[0];require(job.isdigit(),'ACTUAL_SBATCH_JOB_ID')
            jobs[role]=job
            write(attempt/('submitted-'+role+'.json'),dict(job=job,argv=argv,dependencies=deps))
            inspections[role]=inspect_held(job,role,attempt,argv,deps,config)
        require(len(set(jobs.values()))==7,'SEVEN_DISTINCT_ACTUAL_JOBS')
        fresh=inventory(exclude=jobs.values())
        require({r['job'] for r in fresh['project']}<=set(resources['frontier']),
                'ADMISSION_RACE_ALL_NEW_JOBS_HELD')
        write(attempt/'held-inspection.json',dict(jobs=list(inspections.values()),**resources,fresh=fresh,
            qualification_status='PLAN_BOUND_NOT_ACTUAL_PASS',actual_qualification=False,
            qualification_plan=config['generation']['repair']['qualification_plan'],
            generation_schedule=SCHEDULE,final_generation_calls_per_arm=1,W0_generation_calls=0,
            intermediate_generation_calls=0,concurrency_bound=resources['cap'],
            DAG='BASE_MEMIT cold qualification -> lanes AlphaEdit->BLUE->RECT; CAKE->PRUNE; CPU afterany all6',
            source_config_runtime_verified=True,resources_request_not_ETA=True,
            peak_RAM_VRAM='NOT_MEASURED',no_extra_fit_or_edit_pilot=True))
        for role in ('collector','RECT','PRUNE','ALPHAEDIT_BLUE','CAKE','BASE_ALPHAEDIT','BASE_MEMIT'):
            result=command(['scontrol','release',jobs[role]])
            write(attempt/('released-'+role+'.json'),dict(job=jobs[role],result=result,released=True))
    except BaseException as error:
        write(attempt/'submission-failure.json',dict(status='SUBMISSION_OR_RELEASE_BLOCKED',jobs=jobs,
            error_type=type(error).__name__,error=str(error)[:1200],source=source,
            lock=member(attempt/'execution.lock.json'),automatic_retry=False,
            all_unrelated_jobs_unchanged=True,scientific_complete=False,generation_schedule=SCHEDULE))
        raise
    snapshot=command(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%j|%T|%b|%N|%r'])
    result=dict(instruction_id=NONCE,task_id=TASK,status='SUBMISSION_HANDOFF',
        jobs=jobs,dependencies=depmap,source_commit=source,source_tree=lock['source_tree'],
        lock=member(attempt/'execution.lock.json'),config_sha256=lock['config_sha256'],
        cap=resources['cap'],frontier=resources['frontier'],initial_snapshot=snapshot,
        resources=config['resources'],shared_generation_source=config['generation']['source_sha'],
        qualification_status='PLAN_BOUND_NOT_ACTUAL_PASS',actual_qualification='NOT_OBSERVED',
        reference_identity=config['generation']['reference_assets_sha256'],
        generation_schedule=SCHEDULE,final_generation_calls_per_arm=1,
        final_generation_requests_per_arm=2000,W0_generation_calls=0,intermediate_generation_calls=0,
        W_B='ACTUAL_ONLINE_STARTUP_IN_SEALED_RUNNER_NOT_OBSERVED',
        scientific_complete=False,monitoring_active=False,automatic_retry=False)
    write(attempt/'submission.json',result);print(json.dumps(result));return result


if __name__=='__main__':submit()
