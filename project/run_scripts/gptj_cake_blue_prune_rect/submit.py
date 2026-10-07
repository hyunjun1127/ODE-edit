"""Exact five-job registration with two dependency lanes, held inspection first."""
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from .common import *
from .logic import lane_dependencies

ENVELOPE='messages/head/2026-10-07-gptj-cake-blue-prune-rect-sh2.json'
ENVELOPE_SHA='c7d2d2c4932a2cc4a40b3eec9771d75628c1c94190db93b5de6890f2a7d90791'
SOURCES=['project/run_scripts/'+n for n in ('gptj_cake_blue_prune_rect','gptj_native_baselines',
    'base_model_eval','jlz_price_gptj','jlz_interference_l1','jlz_realization',
    'jlz_realized_writer_sequential','jlz_shared_budget','jlz_pilot','experiment_tracking')]
SOURCES+=['scripts/fixed_counterfact.py',ENVELOPE,CONTRACT,'control/wandb-policy.json',
    'control/wandb-method-metric-schema.json','control/gpu-concurrency-policy.tsv',
    'servers/slurm-memory-policy.tsv']

def command(argv):
    result=subprocess.run(argv,text=True,capture_output=True,timeout=45)
    require(result.returncode==0,'COMMAND_FAILED:'+shlex.join(argv)+':'+result.stderr[:800])
    require(len(result.stdout)<16*1024**2,'BOUNDED_COMMAND_OUTPUT')
    return result.stdout.strip()

def field(detail,key):
    match=re.search(r'(?:^| )'+re.escape(key)+r'=([^ ]+)',detail)
    return match[1] if match else None

def gpu_count(tres):
    if not tres:return 0
    match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
    if not match:match=re.search(r'(?:^|,)gres/gpu:[^=,]+=(\d+)(?:,|$)',tres)
    return int(match[1]) if match else 0

def project_job(detail):
    # Names alone cannot classify FE/causal/GPT jobs. Use actual source roots.
    paths=[field(detail,key) or '' for key in ('Command','WorkDir')]
    return any(path.startswith('/mnt/raid5/janghj/ODE-edit/')
        or (path.startswith('/mnt/raid5/janghj/.codex/worktrees/') and '/ODE-edit/' in path)
        or (path.startswith('/mnt/raid5/janghj/.codex/worktrees/odeedit') and '/project/' in path)
        for path in paths)

def inventory(exclude=()):
    """Whole owner queue, including PENDING jobs whose NodeList is empty."""
    rows=[];excluded=[];owner=getpass.getuser()
    raw=command(['squeue','-h','-r','-u',owner,'-o','%i|%j|%T|%b|%N|%R'])
    for line in raw.splitlines():
        job,name,status,gres,nodes,reason=line.split('|',5)
        if job in exclude:continue
        detail=command(['scontrol','show','job',job,'--oneliner'])
        require((field(detail,'UserId') or '').startswith(owner+'('),'QUEUE_OWNER')
        require(not name.startswith(TASK),'DUPLICATE_TASK_QUEUE')
        requested=gpu_count(field(detail,'ReqTRES'))
        if not requested:continue
        node=field(detail,'NodeList');requested_node=field(detail,'ReqNodeList')
        on_s2=node=='server2' or requested_node=='server2'
        if not on_s2:
            if requested_node not in (None,'(null)','') or node not in (None,'(null)',''):
                excluded.append(dict(job=job,reason='EXPLICIT_OTHER_NODE',node=node,requested_node=requested_node))
                continue
            # Explicit node-specific QoS is sufficient metadata for unallocated jobs.
            if field(detail,'QOS')=='lab_gpu_s2':on_s2=True
            else:
                require(not project_job(detail),'UNRESOLVED_PROJECT_NODE_SCOPE')
                excluded.append(dict(job=job,reason='OTHER_SOURCE'));continue
        if not project_job(detail):
            excluded.append(dict(job=job,reason='OTHER_PROJECT_SOURCE',requested_node=requested_node));continue
        require(re.fullmatch(r'[1-9][0-9]*(?:_[0-9]+)?',job),'EXACT_QUEUE_JOB_ID')
        allocated=gpu_count(field(detail,'AllocTRES'))
        rows.append(dict(job=job,name=name,state=status,gpus=requested,allocated_GPUs=allocated,
            reason=reason,node=node,requested_node=requested_node,detail=detail))
    return dict(project=rows,excluded=excluded,queries='one owner squeue plus exact metadata per queued job')

def launcher(attempt,role,config,source):
    cpu=str(config['resources']['collector_cpu'] if role=='collector' else config['resources']['cpu'])
    env=dict(PYTHONPATH=str(attempt/'source'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=cpu,
        MKL_NUM_THREADS=cpu,OPENBLAS_NUM_THREADS=cpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='20261002')
    env[SOURCE_ENV]=source
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    module='project.run_scripts.gptj_cake_blue_prune_rect.'+('collect' if role=='collector' else 'run')
    argv=[config['runtime']['python'],'-u','-m',module,'--attempt',str(attempt)]
    if role!='collector':argv+=['--arm',role]
    return '#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())\
        +'cd '+shlex.quote(str(attempt/'source'))+'\nexec '+shlex.join(argv)+'\n'

def inspect_held(job,role,attempt,argv,dependencies,config):
    detail=command(['scontrol','show','job',job,'--oneliner'])
    mem=config['resources']['collector_host_mib'] if role=='collector' else config['resources']['host_mib']
    cpu=config['resources']['collector_cpu'] if role=='collector' else config['resources']['cpu']
    wall=config['resources']['collector_wall'] if role=='collector' else config['resources']['wall']
    for key,value in dict(JobId=job,JobName=TASK+'-'+role,JobState='PENDING',Reason='JobHeldUser',
        Requeue='0',ReqNodeList='server2',Partition='gpu',QOS='lab_gpu_s2',TimeLimit=wall,
        Command=str(attempt/(role+'.sh')),WorkDir=str(attempt/'source'),**{'CPUs/Task':str(cpu)}).items():
        require(field(detail,key)==value,'HELD_FIELD:'+key)
    require((field(detail,'UserId') or '').startswith(getpass.getuser()+'('),'HELD_OWNER')
    requested=field(detail,'ReqTRES')
    require(requested.startswith('cpu='+str(cpu)+',')
        and (f'mem={mem}M' in requested or f'mem={mem//1024}G' in requested),'HELD_CPU_MEMORY')
    require(gpu_count(field(detail,'ReqTRES'))==(0 if role=='collector' else 1),'HELD_GPU')
    observed=field(detail,'Dependency')
    require(set(re.findall(r'(?:afterany:|:)(\d+)',observed))==set(dependencies),'HELD_DEPENDENCIES')
    require(not dependencies or observed.startswith('afterany:'),'AFTERANY_NOT_PERFORMANCE')
    match=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(match and shlex.split(match[1])==argv,'HELD_FULL_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==(attempt/(role+'.sh')).read_text().strip(),'HELD_SCRIPT_BYTES')
    return dict(role=role,job=job,detail=detail,argv=argv,dependencies=dependencies)

def submit():
    authority();require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA,'ENVELOPE_BYTES')
    config_path=LOCAL/'preparation-r1/config.json';config=read(config_path);attempt=LOCAL/'attempt-r1'
    require(not attempt.exists() and not list(LOCAL.glob('*/submission.json')),'NONCE_NOT_SUBMITTED')
    require(config['resources']['cpu']==6 and config['resources']['host_mib']==59392
        and config['resources']['gpu']==1 and config['noCP'] and not config['z_disk_cache'],'RESOURCE_NOCP_PROFILE')
    require(not command(['git','status','--porcelain','--',*SOURCES]),'SOURCE_COMMITTED')
    tests=read(LOCAL/'cpu-tests-r2.json');require(tests['passed'],'NARROW_CPU_CHECKS')
    for path,value in tests['source_sha256'].items():require(sha(ROOT/path)==value,'CPU_SOURCE_BOUND')
    require(sha(config_path)==tests['config']['sha256'],'CPU_CONFIG_BOUND')
    for row in config['assets']+config['runtime']['members']:stat_seal(row)
    for closure in config['native']['closure'].values():
        for row in closure:verify(row)
    before=inventory()
    tracked=int(next(line for line in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if line.startswith('server2\t')).split('\t')[1])
    local=int(next(line for line in Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if line.startswith('server2\t')).split('\t')[2])
    cap=min(2,tracked,local);require(cap>=1,'CAP_DISABLED')
    allocated=sum(row['allocated_GPUs'] for row in before['project'])
    require(allocated<=cap,'LEGACY_OVERCAP_NO_NEW_ADMISSION')
    frontier=[row['job'] for row in before['project']]
    require(all(job.isdigit() for job in frontier),'ARRAY_FRONTIER_REQUIRES_EXACT_PARENT_PROOF')
    node=command(['scontrol','show','node','server2']);partition=command(['scontrol','show','partition','gpu'])
    qos=command(['sacctmgr','-n','-P','show','qos','lab_gpu_s2','format=Name,MaxWall,MaxTRESPerUser,MaxTRESPerJob'])
    hardware=command(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'])
    require('lab_gpu_s2' in partition and 'Gres=gpu:' in node,'CURRENT_NODE_QOS_GPU')
    storage=shutil.disk_usage(LOCAL)
    require(storage.free>=2*config['resources']['reserve_bytes'],'TWO_LANE_DISK_RESERVE')
    source=command(['git','rev-parse','HEAD']);attempt.mkdir();(attempt/'source').mkdir()
    archive=attempt/'source.tar';command(['git','archive','--format=tar','--output='+str(archive),source,*SOURCES])
    with tarfile.open(archive) as stream:
        require(all((item.isfile() or item.isdir()) and not Path(item.name).is_absolute()
            and '..' not in Path(item.name).parts for item in stream.getmembers()),'SAFE_SOURCE_ARCHIVE')
        stream.extractall(attempt/'source',filter='data')
    write(attempt/'config.json',config)
    for role in (*ARMS,'collector'):
        with (attempt/(role+'.sh')).open('x') as stream:stream.write(launcher(attempt,role,config,source))
    lock=dict(instruction_id=NONCE,source_commit=source,source_tree=command(['git','rev-parse','HEAD^{tree}']),
        config_sha256=sha(attempt/'config.json'),source_members=[member(path) for path in sorted((attempt/'source').rglob('*')) if path.is_file()],
        runtime_sources=config['runtime']['members'],launchers=[member(attempt/(role+'.sh')) for role in (*ARMS,'collector')],
        archive=member(archive),tracking_env=member(config['tracking']['env_file']),owner=getpass.getuser(),
        noCP=True,exact_resume='NOT_AVAILABLE',resources=config['resources'],session=SESSION,
        original_native_readonly=True,source_freeze_distinct_from_report=True)
    write(attempt/'execution.lock.json',lock);jobs={};inspections=[];depmap={}
    try:
        for role in (*ARMS,'collector'):
            dependencies=list(jobs.values()) if role=='collector' else lane_dependencies(role,frontier,jobs,cap)
            dep='afterany:'+':'.join(dependencies) if dependencies else None;depmap[role]=dependencies
            mem=config['resources']['collector_host_mib'] if role=='collector' else config['resources']['host_mib']
            wall=config['resources']['collector_wall'] if role=='collector' else config['resources']['wall']
            argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s2','--nodelist=server2',
                '--nodes=1','--ntasks=1','--cpus-per-task=6','--mem='+str(mem)+'M','--time='+wall,
                '--export=NONE','--no-requeue','--job-name='+TASK+'-'+role,'--chdir='+str(attempt/'source'),
                '--output='+str(attempt/(role+'-%j.out')),'--error='+str(attempt/(role+'-%j.err'))]
            if role!='collector':argv+=['--gres=gpu:1']
            if dep:argv+=['--dependency='+dep]
            argv+=[str(attempt/(role+'.sh'))]
            job=command(argv).split(';')[0];require(job.isdigit(),'ACTUAL_SBATCH_JOB_ID');jobs[role]=job
            write(attempt/('submitted-'+role+'.json'),dict(job=job,argv=argv,dependencies=dependencies))
            inspections.append(inspect_held(job,role,attempt,argv,dependencies,config))
        require(len(set(jobs.values()))==5,'FIVE_DISTINCT_ACTUAL_JOBS')
        fresh=inventory(exclude=jobs.values())
        require({row['job'] for row in fresh['project']}<=set(frontier),'ADMISSION_RACE_ALL_NEW_JOBS_REMAIN_HELD')
        write(attempt/'held-inspection.json',dict(jobs=inspections,cap=cap,before=before,fresh=fresh,
            frontier=frontier,allocated_GPUs=allocated,node=node,partition=partition,qos=qos,hardware=hardware,
            storage=dict(total_bytes=storage.total,free_bytes=storage.free),
            schedule='CAKE->PRUNE; ALPHAEDIT_BLUE->RECT, first lanes after all surviving prior GPU frontier; afterany only',
            concurrency_bound=cap,source=source,independent_model_processes=True,
            resource_request_not_ETA=True,simultaneous_requested_host_mib=2*59392,
            actual_peak_RAM_VRAM='NOT_MEASURED_BEFORE_RUN'))
        # Collector registered first in release order; it can only run after all parents.
        for role in ('collector','PRUNE','RECT','CAKE','ALPHAEDIT_BLUE'):
            result=command(['scontrol','release',jobs[role]])
            write(attempt/('released-'+role+'.json'),dict(job=jobs[role],released=True,result=result))
    except BaseException as error:
        write(attempt/'submission-failure.json',dict(status='SUBMISSION_INCOMPLETE_OR_RELEASE_FAILURE',
            jobs=jobs,error_type=type(error).__name__,source=source,original_jobs_unchanged=True,
            no_automatic_retry=True,new_jobs_not_relabelled_complete=True))
        raise
    snapshot=command(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%j|%T|%b|%N|%r'])
    receipt=dict(instruction_id=NONCE,task_id=TASK,status='SUBMISSION_HANDOFF',jobs=jobs,
        source_commit=source,config_sha256=lock['config_sha256'],lock=member(attempt/'execution.lock.json'),
        dependencies=depmap,frontier=frontier,cap=cap,initial_snapshot=snapshot,resources=config['resources'],
        WandB='STARTUP_IN_SEALED_RUNNER_NOT_OBSERVED',actual_first_write='NOT_OBSERVED',
        monitoring_active=False,automatic_retry=False,scientific_complete=False)
    write(attempt/'submission.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':submit()
