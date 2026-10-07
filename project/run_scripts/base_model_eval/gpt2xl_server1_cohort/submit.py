"""Exact prior W0 lane admission and held registration; no old job mutation."""
import getpass
import re
import shlex
import subprocess
from .common import *

def command(argv):
    result=subprocess.run(argv,capture_output=True,text=True,timeout=30)
    require(result.returncode==0,'COHORT_COMMAND_FAILED:'+shlex.join(argv)+':'+result.stderr)
    return result.stdout.strip()

def prior_binding():
    """One exact scheduler accounting read, only to serialize the old W0 lane."""
    from ..gpt2xl_server1_common import verify_config_lock
    old,lock=verify_config_lock(OLD_ATTEMPT/'config.json',OLD_ATTEMPT/'execution.lock.json')
    receipt=read(OLD_ATTEMPT/'submission.json')
    require(receipt['source_commit']==lock['source_commit']==OLD_SOURCE and
        receipt['jobs']==dict(GPU='60156',collector='60157'),'COHORT_PRIOR_EXACT_BINDING')
    for key in ('config','lock','held','release'):verify(receipt[key])
    held=read(receipt['held']['path']);require(held['passed'] is True and held['source']==OLD_SOURCE,'COHORT_PRIOR_HELD_SOURCE')
    rows=command(['sacct','-X','-n','-P','-j','60156,60157','--format=JobIDRaw,User,JobName%90,State,ExitCode,ElapsedRaw,Start,End,AllocTRES%100'])
    account={}
    for line in rows.splitlines():
        fields=line.split('|')
        if fields[0] in ('60156','60157'):
            require(fields[0] not in account and len(fields)>=9,'COHORT_PRIOR_ACCOUNTING_UNIQUE')
            account[fields[0]]=fields[:9]
    require(set(account)=={'60156','60157'},'COHORT_PRIOR_ACCOUNTING_SCOPE')
    active={'PENDING','RUNNING','COMPLETING','CONFIGURING','SUSPENDED','STAGE_OUT','RESIZING','SIGNALING','REQUEUED','REQUEUE_FED','REQUEUE_HOLD'}
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED','SPECIAL_EXIT'}
    states={};live=[];current_source_checks=[]
    for role,job in receipt['jobs'].items():
        fields=account[job];state=fields[3].split()[0].split('+')[0]
        require(fields[1]==getpass.getuser() and fields[2]=='base-model-gpt2xl-w0-'+role,'COHORT_PRIOR_OWNER_NAME')
        require(state in active|terminal,'COHORT_PRIOR_TYPED_STATE')
        states[job]=dict(role=role,state=state,exit_code=fields[4],elapsed_seconds=fields[5],
            start=fields[6],end=fields[7],allocation=fields[8])
        if state in active:
            detail=command(['scontrol','show','job',job,'--oneliner'])
            require('UserId='+getpass.getuser()+'(' in detail and 'JobName=base-model-gpt2xl-w0-'+role+' ' in detail and
                'Command='+str(OLD_ATTEMPT/(role+'.sh'))+' ' in detail,'COHORT_PRIOR_ACTIVE_SOURCE')
            script=OLD_ATTEMPT/(role+'.sh')
            require(command(['scontrol','write','batch_script',job,'-']).strip()==script.read_text().strip(),'COHORT_PRIOR_ACTIVE_SCRIPT_BYTES')
            require('ODEEDIT_W0_SOURCE_COMMIT='+OLD_SOURCE in script.read_text(),'COHORT_PRIOR_ACTUAL_SOURCE')
            live.append(job);current_source_checks.append(dict(job=job,script=member(script)))
    return dict(prior_jobs=receipt['jobs'],states=states,active_job_ids=live,
        source=OLD_SOURCE,prior_submission=member(OLD_ATTEMPT/'submission.json'),prior_lock=receipt['lock'],
        historical_held_source=receipt['held'],current_active_script_checks=current_source_checks,
        rule='afterany all active old W0 GPU/collector; terminal old jobs KEEP, independent fresh rerun',
        scheduler_accounting_calls=1,old_jobs_mutated=False,scientific_results_queried=False)

def argv_for(role,attempt,r,new_gpu=None):
    cpu=role=='collector'
    args=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s1','--nodelist=devbox','--nodes=1','--ntasks=1',
        '--cpus-per-task='+str(r['collector_cpu'] if cpu else r['cpu']),
        '--mem='+str(r['collector_host_mib'] if cpu else r['host_mib'])+'M',
        '--time='+(r['collector_wall'] if cpu else r['wall']),'--export=NONE','--no-requeue',
        '--job-name='+TASK+'-'+role,'--chdir='+str(attempt/'source'),
        '--output='+str(attempt/(role+'-%j.out')),'--error='+str(attempt/(role+'-%j.err'))]
    deps=[str(new_gpu)] if cpu else r['dependency']
    require(all(str(job).isdigit() for job in deps),'COHORT_ACTUAL_DEPENDENCY_IDS')
    if cpu:require(new_gpu is not None,'COHORT_NEW_GPU_COLLECTOR')
    if deps:args+=['--dependency=afterany:'+':'.join(deps)]
    if not cpu:args+=['--gres=gpu:1']
    return args+[str(attempt/(role+'.sh'))]

def inspect(job,role,argv,attempt,r,deps):
    detail=command(['scontrol','show','job',job,'--oneliner']);cpu=role=='collector'
    mem=r['collector_host_mib'] if cpu else r['host_mib'];cpus=r['collector_cpu'] if cpu else r['cpu']
    for value in (f'JobId={job} ',f'JobName={TASK}-{role} ','UserId='+getpass.getuser()+'(',
        'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ',f'CPUs/Task={cpus} ','ReqNodeList=devbox ',
        'Partition=gpu ','QOS=lab_gpu_s1 ','Command='+str(attempt/(role+'.sh'))+' ','WorkDir='+str(attempt/'source')+' '):
        require(value in detail,'COHORT_HELD_FIELD:'+value)
    require('gres/gpu' not in detail if cpu else 'TresPerNode=gres/gpu:1' in detail,'COHORT_HELD_GPU')
    require(f'mem={mem}M' in detail or f'mem={mem//1024}G' in detail,'COHORT_HELD_MEMORY')
    require('TimeLimit='+(r['collector_wall'] if cpu else r['wall'])+' ' in detail,'COHORT_HELD_WALL')
    observed=re.search(r'\bDependency=([^ ]+)',detail)[1]
    if not deps:require(observed=='(null)','COHORT_HELD_NO_DEPENDENCY')
    else:
        # Slurm may omit already-terminal afterany dependencies; retained live ones
        # must be an exact subset of the sealed IDs, while SubmitLine stays exact.
        found=set(re.findall(r'(?:^|:)(\d+)(?=\(|:|\?|$)',observed))
        require(observed=='(null)' or observed.startswith('afterany:') and found<=set(deps),'COHORT_HELD_DEPENDENCY')
        if cpu:
            # The new upstream GPU is still user-held and cannot already have
            # satisfied afterany. Unlike old W0 dependencies, no omission is valid.
            require(observed.startswith('afterany:') and found==set(deps),'COHORT_HELD_NEW_COLLECTOR_DEPENDENCY')
    submit=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(submit and shlex.split(submit[1])==argv,'COHORT_HELD_FULL_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==(attempt/(role+'.sh')).read_text().strip(),'COHORT_HELD_SCRIPT_BYTES')
    return dict(job=job,role=role,argv=argv,sealed_dependencies=deps,observed_dependency=observed,detail=detail,
        script=member(attempt/(role+'.sh')))

def submit(config_path,lock_path):
    authority();c,lock=verify_lock(config_path,lock_path);attempt=Path(c['attempt']);r=c['resources']
    require(not list(LOCAL.glob('*/submission.json')) and not list(LOCAL.glob('*/submitted-*.json')),'COHORT_NONCE_DUPLICATE_REGISTERED')
    require(not command(['squeue','-h','-u',getpass.getuser(),'--name='+TASK+'-GPU,'+TASK+'-collector','-o','%i|%j']),'COHORT_DUPLICATE_QUEUE')
    prior=read(verify(c['prior_binding']))
    require(r['dependency']==prior['active_job_ids'] and r['scoped_project_cap_exception'],'COHORT_W0_SERIAL_LANE')
    require(r['gpu']==1 and r['cpu']==8 and r['host_mib']==65536 and r['wall']=='04:00:00','COHORT_RESOURCE_PROFILE')
    local=next(line for line in Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if line.startswith('server1\t')).split('\t')
    memory=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server1','--gpus','1',
        '--mem','65536M','--local-limit-mib-per-gpu',local[3]])
    require('ALLOW_MEMORY_POLICY' in memory,'COHORT_MEMORY_CEILING')
    from project.run_scripts.jlz_price_gpt2xl.submit import resource_inventory
    inventory=resource_inventory();capacity(attempt)
    write(attempt/'resource-preflight.json',dict(owner_resource_metadata=inventory,node=command(['scontrol','show','node','devbox']),
        partition=command(['scontrol','show','partition','gpu']),memory_policy=memory,USER_SCOPED_CAP_EXCEPTION=True,
        old_method_cap_changed=False,global_cap_files_changed=False,old_jobs_mutated=False,
        old_W0_serial_dependencies=r['dependency'],physical_shortage_action='scheduler PENDING',no_other_science_query=True))
    scripts=read(attempt/'launchers.json')['scripts']
    for item in scripts.values():verify(item)
    jobs={};held=[]
    for role in ('GPU','collector'):
        argv=argv_for(role,attempt,r,jobs.get('GPU'));job=command(argv).split(';')[0]
        require(job.isdigit(),'COHORT_ACTUAL_JOB_ID');jobs[role]=job
        deps=[jobs['GPU']] if role=='collector' else r['dependency']
        write(attempt/('submitted-'+role+'.json'),dict(job=job,argv=argv,dependencies=deps))
        held.append(inspect(job,role,argv,attempt,r,deps))
    write(attempt/'held-inspection.json',dict(passed=True,jobs=held,source=lock['source_commit'],config=lock['config_sha256']))
    verify_lock(config_path,lock_path)
    for item in scripts.values():verify(item)
    releases=[dict(role=role,job=jobs[role],result=command(['scontrol','release',jobs[role]])) for role in ('collector','GPU')]
    write(attempt/'release.json',dict(jobs=releases))
    value=dict(nonce=NONCE,task_id=TASK,jobs=jobs,dependencies=dict(GPU=r['dependency'],collector=[jobs['GPU']]),
        source_commit=lock['source_commit'],source_tree=lock['source_tree'],config=member(config_path),lock=member(lock_path),
        held=member(attempt/'held-inspection.json'),release=member(attempt/'release.json'),prior_binding=c['prior_binding'],
        snapshot=command(['squeue','-h','-j',','.join(jobs.values()),'-o','%i|%j|%T|%R']),
        new_W_B_identity=dict(run_id=c['run_instance']['run_id'],URL='NOT_YET_OBSERVED',stage='NEW_ID_SEALED_NOT_ONLINE_VERIFIED'),
        initial='NOT_OBSERVED',USER_SCOPED_CAP_EXCEPTION=True,actual_model_edits=0,
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE',monitoring=False,broadcast='NO_BROADCAST_NOT_REQUIRED')
    write(attempt/'submission.json',value);return value
