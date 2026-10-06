"""Owner-only held registration of one USER-scoped cap-exempt W0 allocation."""
import argparse
import getpass
import json
import re
import shlex
import subprocess
from pathlib import Path
from .gpt2xl_server1_common import *

def command(argv):
    r=subprocess.run(argv,text=True,capture_output=True)
    require(r.returncode==0,'W0_COMMAND_FAILED:'+shlex.join(argv)+':'+r.stderr)
    return r.stdout.strip()

def argv_for(role,attempt,resources,job=None):
    cpu=role=='collector';name=TASK+'-'+role
    args=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s1','--nodelist=devbox',
        '--nodes=1','--ntasks=1','--cpus-per-task='+str(resources['collector_cpu'] if cpu else resources['cpu']),
        '--mem='+str(resources['collector_host_mib'] if cpu else resources['host_mib'])+'M',
        '--time='+(resources['collector_wall'] if cpu else resources['wall']),
        '--export=NONE','--no-requeue','--job-name='+name,'--chdir='+str(attempt/'source'),
        '--output='+str(attempt/(role+'-%j.out')),'--error='+str(attempt/(role+'-%j.err'))]
    if cpu:
        require(job and str(job).isdigit(),'W0_ACTUAL_COLLECTOR_DEPENDENCY');args+=['--dependency=afterany:'+str(job)]
    else:args+=['--gres=gpu:1']
    return args+[str(attempt/(role+'.sh'))]

def inspect(job,role,argv,attempt,r,dep=None):
    raw=command(['scontrol','show','job',job,'--oneliner']);cpu=role=='collector'
    cpus=r['collector_cpu'] if cpu else r['cpu'];mem=r['collector_host_mib'] if cpu else r['host_mib']
    for term in (f'JobId={job} ',f'JobName={TASK}-{role} ','UserId='+getpass.getuser()+'(',
        'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ',f'CPUs/Task={cpus} ',
        'ReqNodeList=devbox ','Partition=gpu ','QOS=lab_gpu_s1 ',
        'Command='+str(attempt/(role+'.sh'))+' ','WorkDir='+str(attempt/'source')+' '):require(term in raw,'W0_HELD_FIELD:'+term)
    require('gres/gpu' not in raw if cpu else 'TresPerNode=gres/gpu:1' in raw,'W0_HELD_GPU')
    require(f'mem={mem}M' in raw or f'mem={mem//1024}G' in raw,'W0_HELD_MEMORY')
    require('TimeLimit='+(r['collector_wall'] if cpu else r['wall'])+' ' in raw,'W0_HELD_WALL')
    observed=re.search(r'\bDependency=([^ ]+)',raw)[1]
    require(observed=='(null)' if not dep else observed.startswith('afterany:') and set(re.findall(r'(?:^|:)(\d+)(?=\(|:|\?|$)',observed))=={dep},'W0_HELD_DEPENDENCY')
    submit=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',raw)
    require(submit and shlex.split(submit[1])==argv,'W0_HELD_FULL_ARGV')
    script=attempt/(role+'.sh')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==script.read_text().strip(),'W0_HELD_SCRIPT_BYTES')
    return dict(job=job,role=role,argv=argv,dependency=dep,detail=raw,script=member(script))

def submit(config_path,lock_path):
    c,lock=verify_config_lock(config_path,lock_path);attempt=Path(c['attempt']);r=c['resources']
    require(sha(ROOT/ENVELOPE)==ENVELOPE_SHA,'W0_AUTHORITY_SHA')
    auth=read(ROOT/ENVELOPE)
    require(auth['instruction_id']==NONCE and auth['resources']['server1_scoped_GPU_cap_exception'] and
        r['scoped_project_cap_exception'] and r['dependency'] is None,'USER_SCOPED_CAP_EXCEPTION')
    require(not list(LOCAL.glob('*/submission.json')) and not list(LOCAL.glob('*/submitted-*.json')),'W0_NONCE_DUPLICATE_REGISTRATION')
    require(not command(['squeue','-h','-u',getpass.getuser(),'--name='+TASK+'-GPU,'+TASK+'-collector','-o','%i|%j']),'W0_EXISTING_JOB')
    require(r['gpu']==1 and r['cpu']==8 and r['host_mib']==65536 and r['wall']=='04:00:00','W0_RESOURCE_PROFILE')
    local=next(line for line in Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if line.startswith('server1\t')).split('\t')
    memory=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server1','--gpus','1','--mem','65536M','--local-limit-mib-per-gpu',local[3]])
    require('ALLOW_MEMORY_POLICY' in memory,'W0_HARD_HOST_MEMORY')
    from project.run_scripts.jlz_price_gpt2xl.submit import resource_inventory
    before=resource_inventory()
    node=command(['scontrol','show','node','devbox']);partition=command(['scontrol','show','partition','gpu'])
    capacity(attempt)
    write(attempt/'resource-preflight.json',dict(owned_resource_metadata=before,node=node,partition=partition,
        memory_policy=memory,effective_action='USER_SCOPED_CAP_EXCEPTION',W0_extra_GPU=1,
        old_method_cap_unchanged=True,global_cap_files_changed=False,old_jobs_mutated=False,
        physical_resource_shortage_action='SCHEDULER_PENDING; no dependency to old arms',
        project_cap_helper='NOT_USED_AS_ADMISSION_GATE_FOR_USER_SCOPED_W0',proof=member(ROOT/ENVELOPE)))
    scripts=read(attempt/'launchers.json')['scripts']
    for role in ('GPU','collector'):verify(scripts[role])
    ids={};held=[]
    for role in ('GPU','collector'):
        dep=ids.get('GPU') if role=='collector' else None
        argv=argv_for(role,attempt,r,dep);job=command(argv).split(';')[0]
        require(job.isdigit(),'W0_REAL_JOB_ID');ids[role]=job
        write(attempt/('submitted-'+role+'.json'),dict(job=job,role=role,argv=argv,dependency=dep))
        held.append(inspect(job,role,argv,attempt,r,dep))
    write(attempt/'held-inspection.json',dict(passed=True,jobs=held,source=lock['source_commit'],config=lock['config_sha256']))
    verify_config_lock(config_path,lock_path)
    for row in scripts.values():verify(row)
    release=[]
    for role in ('collector','GPU'):release.append(dict(role=role,job=ids[role],result=command(['scontrol','release',ids[role]])))
    write(attempt/'release.json',dict(jobs=release))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%R'])
    value=dict(nonce=NONCE,task_id=TASK,jobs=ids,dependencies=dict(GPU=None,collector='afterany:'+ids['GPU']),
        source_commit=lock['source_commit'],source_tree=lock['source_tree'],lock=member(lock_path),config=member(config_path),
        held=member(attempt/'held-inspection.json'),release=member(attempt/'release.json'),snapshot=snapshot,
        initial='NOT_OBSERVED',W_B='NOT_YET_OBSERVED',fresh_W0_REQUIRED=True,USER_SCOPED_CAP_EXCEPTION=True,
        checkpoint_saved=False,exact_resume='NOT_AVAILABLE',monitoring=False,broadcast='NO_BROADCAST_NOT_REQUIRED')
    write(attempt/'submission.json',value);print(json.dumps(value,ensure_ascii=False));return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);a=p.parse_args();submit(a.config.resolve(),a.lock.resolve())
