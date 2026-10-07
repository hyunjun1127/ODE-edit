"""Seven exact jobs, source-bound shared generation, held inspect then release."""
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from .generation_common import *
from .generation_plan import dependencies, ready, counts
from project.run_scripts.gptj_cake_blue_prune_rect.submit import command, field, gpu_count, inventory
from scripts.slurm_memory_policy import load_policy, check_request

SOURCES=['project/run_scripts/'+n for n in ('gptj_cake_blue_prune_rect','gptj_native_baselines',
    'base_model_eval','jlz_price_gptj','jlz_interference_l1','jlz_realization',
    'jlz_realized_writer_sequential','jlz_shared_budget','jlz_pilot','experiment_tracking',
    'experiment_generation_eval')]
SOURCES+=['scripts/fixed_counterfact.py','scripts/slurm_memory_policy.py',ENVELOPE,CONTRACT,POLICY,
    'plans/global/2026-10-04-jlz-v12-marginal-allocation/experiment-2k/case-schedule-first2000.csv',
    'plans/global/2026-10-07-gptj-cake-blue-prune-rect-2k/contract.json',
    'control/wandb-policy.json','control/wandb-method-metric-schema.json',
    'control/gpu-concurrency-policy.tsv','servers/slurm-memory-policy.tsv']
LOCAL_CAP=Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')

def launcher(attempt,role,c,source):
    cpu=c['resources']['collector_cpu'] if role=='collector' else c['resources']['cpu']
    env=dict(PYTHONPATH=str(attempt/'source'),PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS=str(cpu),MKL_NUM_THREADS=str(cpu),OPENBLAS_NUM_THREADS=str(cpu),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONHASHSEED='20261002')
    env[SOURCE_ENV]=source
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    module='project.run_scripts.gptj_native_baselines.generation_'+('collect' if role=='collector' else 'run')
    argv=[c['runtime']['python'],'-u','-m',module,'--attempt',str(attempt)]
    if role!='collector':argv+=['--arm',role]
    return '#!/bin/bash\n# ODEEDIT_SLURM_SERVER=server2\nset -euo pipefail\n'+''.join(
        'export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+'cd '+shlex.quote(str(attempt/'source'))+'\nexec '+shlex.join(argv)+'\n'

def sbatch_argv(attempt,role,c,dep):
    r=c['resources'];collector=role=='collector'
    mem=r['collector_host_mib'] if collector else r['host_mib']
    cpu=r['collector_cpu'] if collector else r['cpu']
    wall=r['collector_wall'] if collector else r['wall']
    argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s2','--nodelist=server2',
        '--nodes=1','--ntasks=1','--cpus-per-task='+str(cpu),'--mem='+str(mem)+'M','--time='+wall,
        '--export=NONE','--no-requeue','--job-name='+c.get('task_id', TASK)+'-'+role,'--chdir='+str(attempt/'source'),
        '--output='+str(attempt/(role+'-%j.out')),'--error='+str(attempt/(role+'-%j.err'))]
    if not collector:argv+=['--gres=gpu:1']
    if dep:argv+=['--dependency=afterany:'+':'.join(dep)]
    return argv+[str(attempt/(role+'.sh'))]

def inspect_held(job,role,attempt,argv,deps,c):
    detail=command(['scontrol','show','job',job,'--oneliner'])
    collector=role=='collector';r=c['resources']
    cpu=r['collector_cpu'] if collector else r['cpu']
    mem=r['collector_host_mib'] if collector else r['host_mib']
    wall=r['collector_wall'] if collector else r['wall']
    required=dict(JobId=job,JobName=c.get('task_id', TASK)+'-'+role,JobState='PENDING',Reason='JobHeldUser',
        Requeue='0',ReqNodeList='server2',Partition='gpu',QOS='lab_gpu_s2',
        TimeLimit=wall,Command=str(attempt/(role+'.sh')),WorkDir=str(attempt/'source'))
    required['CPUs/Task']=str(cpu)
    for k,v in required.items():require(field(detail,k)==v,'HELD_FIELD:'+k)
    require((field(detail,'UserId') or '').startswith(getpass.getuser()+'('),'HELD_OWNER')
    tres=field(detail,'ReqTRES')
    require(tres.startswith('cpu='+str(cpu)+',')
        and (f'mem={mem}M' in tres or f'mem={mem//1024}G' in tres),'HELD_CPU_MEMORY')
    require(gpu_count(tres)==(0 if collector else 1),'HELD_GPU')
    observed=field(detail,'Dependency') or ''
    require(set(re.findall(r'(?:afterany:|:)([0-9]+)',observed))==set(deps),'HELD_DEPENDENCY')
    require(not deps or observed.startswith('afterany:'),'HELD_AFTERANY')
    matched=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(matched and shlex.split(matched[1])==argv,'HELD_FULL_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()
        ==(attempt/(role+'.sh')).read_text().strip(),'HELD_SCRIPT_BYTES')
    return dict(role=role,job=job,detail=detail,argv=argv,dependencies=deps)

def admission(task=TASK, inventory_fn=inventory):
    before=inventory_fn()
    require(not any(row['name'].startswith(task) for row in before['project']),'DUPLICATE_TASK_QUEUE')
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines()
        if x.startswith('server2\t')).split('\t')[1])
    local=next(x for x in LOCAL_CAP.read_text().splitlines() if x.startswith('server2\t')).split('\t')
    cap=min(2,tracked,int(local[2]))
    require(1<=cap<=2,'CAP_DISABLED')
    requested,maximum=check_request(load_policy(ROOT/'servers/slurm-memory-policy.tsv')['server2'],
        1,'59392M',int(local[3]))
    require(requested<=maximum,'HOST_REQUEST_CEILING')
    allocated=sum(row['allocated_GPUs'] for row in before['project'])
    require(allocated<=cap,'LEGACY_OVERCAP_NO_NEW_ADMISSION')
    frontier=[row['job'] for row in before['project']]
    require(all(j.isdigit() for j in frontier),'ARRAY_FRONTIER_REQUIRES_EXACT_PROOF')
    require(all(row['gpus']==1 for row in before['project']),'MULTIGPU_FRONTIER_REQUIRES_EXACT_PROOF')
    node=command(['scontrol','show','node','server2'])
    partition=command(['scontrol','show','partition','gpu'])
    qos=command(['sacctmgr','-n','-P','show','qos','lab_gpu_s2','format=Name,MaxWall,MaxTRESPerUser,MaxTRESPerJob'])
    hardware=command(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'])
    require('lab_gpu_s2' in partition and 'Gres=gpu:' in node,'NODE_QOS_GPU')
    require('lab_gpu_s2|' in qos,'QOS_CURRENT')
    storage=shutil.disk_usage(LOCAL)
    require(storage.free>=2*16*1024**3,'TWO_LANE_OUTPUT_RESERVE')
    return dict(before=before,cap=cap,frontier=frontier,allocated_GPUs=allocated,node=node,
        partition=partition,qos=qos,hardware=hardware,
        storage=dict(total_bytes=storage.total,free_bytes=storage.free,inode_free=os.statvfs(LOCAL).f_favail),
        memory_policy=dict(requested_mib=requested,maximum_mib=maximum),
        generic_pattern_helper_not_full_scope='Name pattern omits FE/GPT; source/owner/node exact inventory used')

def submit():
    authority();config_path=LOCAL/'preparation-r2/config.json';c=read(config_path);ready(c)
    attempt=Path(c['attempt'])
    require(attempt==LOCAL/'attempt-r1' and not attempt.exists()
        and not list(LOCAL.glob('*/submission.json')),'NONCE_NOT_ALREADY_SUBMITTED')
    require(c['resources']['cpu']==6 and c['resources']['host_mib']==59392
        and c['resources']['gpu']==1 and c['noCP'] and not c['z_disk_cache'],'RESOURCE_SCIENCE_NOCP')
    require(not command(['git','status','--porcelain','--',*SOURCES]),'SOURCE_COMMITTED')
    checks=read(LOCAL/'cpu-integration-r2.json')
    require(checks['status']=='PASS_CPU_INTEGRATION' and not checks['CUDA_initialized'],'NARROW_CPU_INTEGRATION')
    for row in checks['source']:verify(row)
    require(checks['config']['sha256']==sha(config_path),'CPU_CONFIG_BOUND')
    for row in c['assets']+c['runtime']['members']+[c['observer_identity']]:stat_seal(row)
    for arm in ARMS:
        closure=c['arm_configs'][arm]['native']['closure']
        for row in closure if isinstance(closure,list) else [r for group in closure.values() for r in group]:verify(row)
    for row in c['generation']['shared_source_members']:verify(row)
    require(command(['git','rev-parse','HEAD:project/run_scripts/experiment_generation_eval'])
        ==c['generation']['package_tree'],'SHARED_PACKAGE_TREE')
    resources=admission()
    source=command(['git','rev-parse','HEAD']);attempt.mkdir();(attempt/'source').mkdir()
    archive=attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),source,*SOURCES])
    with tarfile.open(archive) as stream:
        require(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute()
            and '..' not in Path(m.name).parts for m in stream.getmembers()),'SAFE_SOURCE_ARCHIVE')
        stream.extractall(attempt/'source',filter='data')
    write(attempt/'config.json',c)
    for role in (*ARMS,'collector'):
        with (attempt/(role+'.sh')).open('x') as f:f.write(launcher(attempt,role,c,source))
    lock=dict(instruction_id=NONCE,task_id=TASK,source_commit=source,
        source_tree=command(['git','rev-parse','HEAD^{tree}']),config_sha256=sha(attempt/'config.json'),
        source_members=[member(p) for p in sorted((attempt/'source').rglob('*')) if p.is_file()],
        runtime_sources=c['runtime']['members'],archive=member(archive),
        launchers=[member(attempt/(role+'.sh')) for role in (*ARMS,'collector')],
        tracking_env=member(c['tracking']['env_file']),shared_generation_source=c['generation']['source_sha'],
        shared_generation_tree=c['generation']['package_tree'],reference=c['generation']['reference_manifest'],
        reference_identity=c['generation']['reference_assets_sha256'],owner=getpass.getuser(),session=SESSION,
        noCP=True,exact_resume='NOT_AVAILABLE',resources=c['resources'],
        raw_generation_local_only=True,source_freeze_distinct_from_report=True,
        count_plan=counts(),no_new_monitor_or_automatic_retry=True)
    write(attempt/'execution.lock.json',lock);jobs={};inspections=[];depmap={}
    try:
        for role in (*ARMS,'collector'):
            dep=dependencies(role,resources['frontier'],jobs,resources['cap'])
            depmap[role]=dep;argv=sbatch_argv(attempt,role,c,dep)
            # Persist exact command before sbatch, including scheduler rejection evidence.
            write(attempt/('submission-command-'+role+'.json'),dict(argv=argv,role=role))
            result=subprocess.run(argv,text=True,capture_output=True,timeout=45)
            write(attempt/('sbatch-result-'+role+'.json'),dict(returncode=result.returncode,
                stdout=result.stdout.strip(),stderr=result.stderr.strip(),role=role,argv=argv))
            require(result.returncode==0,'SBATCH_REJECTED:'+role+':'+result.stderr.strip()[:800])
            job=result.stdout.strip().split(';')[0]
            require(job.isdigit(),'ACTUAL_SBATCH_JOB_ID')
            jobs[role]=job
            write(attempt/('submitted-'+role+'.json'),dict(job=job,argv=argv,dependencies=dep))
            inspections.append(inspect_held(job,role,attempt,argv,dep,c))
        require(len(set(jobs.values()))==7,'SEVEN_DISTINCT_ACTUAL_JOBS')
        fresh=inventory(exclude=jobs.values())
        require({r['job'] for r in fresh['project']}<=set(resources['frontier']),
            'ADMISSION_RACE_ALL_JOBS_REMAIN_HELD')
        write(attempt/'held-inspection.json',dict(jobs=inspections,**resources,fresh=fresh,
            count_plan=counts(),concurrency_bound=resources['cap'],
            DAG='BASE_MEMIT W0 owner -> lanes BASE_ALPHAEDIT->ALPHAEDIT_BLUE->RECT; CAKE->PRUNE',
            resources_request_not_ETA=True,peak_RAM_VRAM='NOT_MEASURED',no_extra_generation_forward=True))
        for role in ('collector','RECT','PRUNE','ALPHAEDIT_BLUE','CAKE','BASE_ALPHAEDIT','BASE_MEMIT'):
            result=command(['scontrol','release',jobs[role]])
            write(attempt/('released-'+role+'.json'),dict(job=jobs[role],result=result,released=True))
    except BaseException as error:
        write(attempt/'submission-failure.json',dict(status='SUBMISSION_OR_RELEASE_BLOCKED',
            jobs=jobs,error_type=type(error).__name__,error=str(error)[:1200],source=source,
            lock=member(attempt/'execution.lock.json'),all_prior_jobs_unchanged=True,
            automatic_retry=False,new_jobs_not_relabelled_complete=True,science_complete=False))
        raise
    snapshot=command(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%j|%T|%b|%N|%r'])
    result=dict(instruction_id=NONCE,task_id=TASK,status='SUBMISSION_HANDOFF',jobs=jobs,
        dependencies=depmap,source_commit=source,source_tree=lock['source_tree'],
        lock=member(attempt/'execution.lock.json'),config_sha256=lock['config_sha256'],
        cap=resources['cap'],frontier=resources['frontier'],initial_snapshot=snapshot,
        resources=c['resources'],shared_generation_source=c['generation']['source_sha'],
        reference_identity=c['generation']['reference_assets_sha256'],
        W_B='ACTUAL_ONLINE_STARTUP_IN_SEALED_RUNNER_NOT_OBSERVED',
        actual_initial_write='NOT_OBSERVED',scientific_complete=False,
        monitoring_active=False,automatic_retry=False)
    write(attempt/'submission.json',result);print(json.dumps(result))
    return result
if __name__=='__main__':submit()
