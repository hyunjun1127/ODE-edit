"""Create-once source freeze; conservative owned resource frontier; held release."""
import argparse
import getpass
import json
import re
import shlex
import shutil
import subprocess
import tarfile
from .common import *

ROLES=(*ARMS,'collector')
SOURCES=['project/run_scripts/gpt2xl_prune_rect','project/run_scripts/gpt2xl_cake_blue','project/run_scripts/gpt2xl_native_baselines',
 'project/run_scripts/base_model_eval/gpt2xl_server1_common.py',
 'project/run_scripts/base_model_eval/gpt2xl_server1_prepare.py',ENVELOPE,CONTRACT,
 'project/run_scripts/jlz_price_gpt2xl','project/run_scripts/jlz_interference_l1',
 'project/run_scripts/jlz_realization','project/run_scripts/jlz_pilot','project/run_scripts/jlz_shared_budget',
 'project/run_scripts/jlz_realized_writer_sequential/review_completed.py',
 'project/run_scripts/jlz_realized_writer_sequential/__init__.py','project/run_scripts/jlz_v12r',
 'project/run_scripts/jlz_native_writer_aware','project/run_scripts/jlz_realized_subject',
 'project/run_scripts/jlz_writer_coupled','project/run_scripts/jlz_realized_writer',
 'project/run_scripts/experiment_tracking','scripts/fixed_counterfact.py',
 'scripts/slurm_memory_policy.py','control/gpu-concurrency-policy.tsv',
 'control/wandb-policy.json','control/wandb-method-metric-schema.json','servers/slurm-memory-policy.tsv',
 'messages/head/2026-10-07-all-servers-gpu-cap2.json']

def command(argv,cwd=None):
    r=subprocess.run(argv,cwd=cwd,text=True,capture_output=True)
    require(r.returncode==0,'COMMAND_FAILED:'+shlex.join(argv)+':'+r.stderr)
    return r.stdout.strip()

def launcher(source,commit,role,attempt):
    args=[PYTHON,'-u','-m','project.run_scripts.gpt2xl_prune_rect.'+('collect' if role=='collector' else 'run'),
        '--attempt',str(attempt)]
    if role!='collector':args+=['--arm',role]
    env=dict(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env[SOURCE_ENV]=commit
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    return '#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+'cd '+shlex.quote(str(source))+'\nexec '+shlex.join(args)+'\n'

def native_members(c):
    return [x[key] for b in c['native_bundle'].values() for x in b['files'] for key in ('original','effective')]

def freeze(configpath,attempt):
    authority();require(attempt.parent==LOCAL and not attempt.exists(),'CREATE_ONCE_ATTEMPT')
    require(not command(['git','status','--porcelain','--',*SOURCES],ROOT),'COMMIT_SOURCE_BEFORE_FREEZE')
    c=read(configpath);require(c['instruction_id']==NONCE and c['attempt']==str(attempt),'CONFIG_SCOPE')
    tests=read(verify(c['cpu_preflight']));require(tests['status']=='PASS','CPU_PREFLIGHT')
    for row in tests['source']+tests['helper_source']+native_members(c):verify(row)
    commit=command(['git','rev-parse','HEAD'],ROOT);tree=command(['git','rev-parse','HEAD^{tree}'],ROOT)
    attempt.mkdir();source=attempt/'source';source.mkdir();archive=attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCES],ROOT)
    with tarfile.open(archive) as tf:
        items=tf.getmembers();require(len(items)==len({x.name for x in items}) and all((x.isfile() or x.isdir())
            and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in items),'SAFE_SOURCE_ARCHIVE')
        tf.extractall(source,filter='data')
    for row in tests['source']+tests['helper_source']:
        require(sha(source/Path(row['path']).relative_to(ROOT))==row['sha256'],'TESTED_SOURCE_ARCHIVE')
    write(attempt/'config.json',c)
    for role in ROLES:
        script=attempt/(role+'.sh');write_bytes(script,launcher(source,commit,role,attempt).encode());script.chmod(0o755)
    lock=dict(instruction_id=NONCE,task_id=TASK,source_commit=commit,source_tree=tree,
        config_sha256=sha(attempt/'config.json'),archive=member(archive),
        source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        runtime_sources=c['runtime']['source_members'],native_closure=native_members(c),
        launchers=[member(attempt/(r+'.sh')) for r in ROLES],tracking_env=member(c['tracking']['env_file']),
        owner=getpass.getuser(),session=SESSION,host='server1',resources=c['resources'],
        checkpoint_saved=False,z_disk_cache=False,exact_resume='NOT_AVAILABLE',old_jobs_mutated=False)
    write(attempt/'execution.lock.json',lock)
    return c,lock

def arguments(role,dep,attempt,r):
    cpu=role=='collector'
    argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s1','--nodelist=devbox',
        '--nodes=1','--ntasks=1','--cpus-per-task='+str(r['collector_cpu'] if cpu else r['cpu']),
        '--export=NONE','--no-requeue','--job-name='+TASK+'-'+role,'--chdir='+str(attempt/'source'),
        '--mem='+str(r['collector_host_mib'] if cpu else r['host_mib'])+'M',
        '--time='+(r['collector_wall'] if cpu else r['wall']),
        '--output='+str(attempt/(role+'-%j.out')),'--error='+str(attempt/(role+'-%j.err'))]
    if not cpu:argv+=['--gres=gpu:1']
    if dep:argv+=['--dependency=afterany:'+':'.join(dep)]
    return argv+[str(attempt/(role+'.sh'))]

def inspect(job,role,dep,argv,attempt,r):
    detail=command(['scontrol','show','job',job,'--oneliner']);cpu=role=='collector'
    cpus=r['collector_cpu'] if cpu else r['cpu'];mem=r['collector_host_mib'] if cpu else r['host_mib']
    for value in (f'JobId={job} ',f'JobName={TASK}-{role} ','UserId='+getpass.getuser()+'(',
        'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ',f'CPUs/Task={cpus} ',
        'ReqNodeList=devbox ','Partition=gpu ','QOS=lab_gpu_s1 ',
        'Command='+str(attempt/(role+'.sh'))+' ','WorkDir='+str(attempt/'source')+' '):require(value in detail,'HELD_FIELD:'+value)
    require('gres/gpu' not in detail if cpu else 'TresPerNode=gres/gpu:1' in detail,'HELD_GPU')
    require(f'mem={mem}M' in detail or f'mem={mem//1024}G' in detail,'HELD_MEMORY')
    require('TimeLimit='+(r['collector_wall'] if cpu else r['wall'])+' ' in detail,'HELD_WALL')
    observed=re.search(r'\bDependency=([^ ]+)',detail)[1]
    got=set(re.findall(r'(?:^|:)(\d+)(?=\(|:|\?|$)',observed)) if observed!='(null)' else set()
    own=set(dep)&set(read(attempt/('submitted-'+x+'.json'))['job'] for x in ROLES if (attempt/('submitted-'+x+'.json')).exists())
    require(got<=set(dep) and own<=got and (not got or observed.startswith('afterany:')),'HELD_DEPENDENCY')
    # Completed old resource dependencies may be removed by Slurm; own held IDs never are.
    missing=set(dep)-got
    for old in missing:
        state=command(['sacct','-n','-X','-j',old,'--format=JobIDRaw,State','-P'])
        require(any(line.split('|')[0]==old and line.split('|')[1].split()[0].split('+')[0] in
            ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE') for line in state.splitlines()),'REMOVED_DEP_NOT_TERMINAL')
    submit=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(submit and shlex.split(submit[1])==argv,'HELD_FULL_ARGV')
    path=attempt/(role+'.sh')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==path.read_text().strip(),'HELD_SCRIPT_BYTES')
    return dict(job=job,role=role,argv=argv,dependency=dep,resource_detail=detail,launcher=member(path))

def dependencies(jobs):
    ids={j['job'] for j in jobs};parents=set()
    for j in jobs:
        raw=re.search(r'\bDependency=([^ ]+)',j['resource_detail'])
        require(raw is not None,'DEPENDENCY_REQUIRED')
        if raw[1].startswith('afterany:') and '?' not in raw[1]:
            parents|=set(re.findall(r'(?:^|:)(\d+)(?=\(|:|\?|$)',raw[1]))&ids
    return sorted(ids-parents,key=int)  # all leaves of a verified acyclic graph

def split_inventory(before):
    # Latest USER combined2 supersedes the historical W0 exception. Count
    # every own GPU allocation/admitted job, regardless of its name or role.
    return list(before['jobs']),[]

def submit(configpath,attempt):
    authority();require(not list(LOCAL.glob('*/submission.json')) and not list(LOCAL.glob('*/submitted-*.json')),'NO_DUPLICATE_NONCE')
    require(not command(['squeue','-h','-u',getpass.getuser(),'--name='+','.join(TASK+'-'+r for r in ROLES),'-o','%i|%j']),'DUPLICATE_JOB_NAME')
    from project.run_scripts.jlz_price_gpt2xl.submit import resource_inventory
    from project.run_scripts.jlz_price_gpt2xl.admission import width
    before=resource_inventory();science,w0=split_inventory(before)
    row=next(x for x in Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if x.startswith('server1\t')).split('\t')
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server1\t')).split('\t')[1])
    policy=ROOT/'control/gpu-concurrency-policy.tsv'
    require(sha(policy)=='fb36c85243cf6e1166a431f366efe4a0229942d880a1211a3c6d6bca0593f3a2','LATEST_COMBINED_CAP_POLICY')
    cap=min(2,int(row[2]),tracked);require(cap>=1 and width(science)<=cap,'EXISTING_COMBINED_CAP')
    frontier=dependencies(science);deps={role:list(frontier) for role in ARMS}
    virtual=list(science)
    for i,role in enumerate(ARMS):
        dep=frontier+(['999999991'] if cap==1 and i else [])
        virtual.append(dict(job=str(999999991+i),gpus=1,resource_detail='Dependency='+('afterany:'+':'.join(dep) if dep else '(null)')+' '))
    projected=width(virtual);require(projected<=cap,'PROJECTED_SCIENCE_CAP')
    node=command(['scontrol','show','node','devbox']);partition=command(['scontrol','show','partition','gpu'])
    require('Gres=gpu' in node and 'PartitionName=gpu' in partition,'CANONICAL_GPU_NODE_PARTITION')
    c,lock=freeze(configpath,attempt);r=c['resources']
    memory=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server1','--gpus','1','--mem',str(r['host_mib'])+'M','--local-limit-mib-per-gpu',row[3]])
    require('ALLOW_MEMORY_POLICY' in memory and shutil.disk_usage(attempt).free>=r['reserve_bytes'],'MEMORY_DISK_PREFLIGHT')
    write(attempt/'resource-preflight.json',dict(before=before,science=science,excluded_W0_scoped=w0,node=node,partition=partition,
        effective_cap=cap,projected_GPU_width=projected,frontier=frontier,memory_policy=memory,old_jobs_mutated=False))
    ids={};held=[]
    for role in ROLES:
        dep=list(ids.values()) if role=='collector' else deps[role]+([ids[ARMS[0]]] if cap==1 and role==ARMS[1] else [])
        argv=arguments(role,dep,attempt,r);job=command(argv).split(';')[0];require(job.isdigit(),'ACTUAL_JOB_ID');ids[role]=job
        write(attempt/('submitted-'+role+'.json'),dict(role=role,job=job,dependency=dep,argv=argv))
        held.append(inspect(job,role,dep,argv,attempt,r))
    write(attempt/'held-inspection.json',dict(source_commit=lock['source_commit'],config_sha256=lock['config_sha256'],jobs=held,passed=True))
    fresh=resource_inventory(exclude=tuple(ids.values()));current,_=split_inventory(fresh)
    final_width=width(current+[dict(job=x['job'],gpus=1,resource_detail=x['resource_detail']) for x in held if x['role']!='collector'])
    require(final_width<=cap,'PRE_RELEASE_SCIENCE_CAP')
    write(attempt/'pre-release-admission.json',dict(existing=fresh,projected_GPU_width=final_width,effective_cap=cap,passed=True))
    for item in lock['source_members']+lock['launchers']+lock['runtime_sources']+lock['native_closure']+[lock['archive'],lock['tracking_env']]:verify(item)
    release=[dict(role=role,job=ids[role],result=command(['scontrol','release',ids[role]])) for role in reversed(ROLES)]
    write(attempt/'release.json',dict(jobs=release))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%R'])
    receipt=dict(instruction_id=NONCE,task_id=TASK,jobs=ids,source_commit=lock['source_commit'],source_tree=lock['source_tree'],
        lock=member(attempt/'execution.lock.json'),config=member(attempt/'config.json'),held=member(attempt/'held-inspection.json'),release=member(attempt/'release.json'),
        dependencies={x['role']:x['dependency'] for x in held},snapshot=snapshot,initial='NOT_OBSERVED',
        W_B='NOT_YET_RUN; startup/readback inside actual jobs',checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
        combined_GPU_cap=cap,projected_GPU_width=projected,broadcast='NO_BROADCAST_NOT_REQUIRED',monitoring=False)
    write(attempt/'submission.json',receipt);print(json.dumps(receipt,ensure_ascii=False));return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    a=p.parse_args();submit(a.config.resolve(),a.attempt.resolve())
if __name__=='__main__':main()
