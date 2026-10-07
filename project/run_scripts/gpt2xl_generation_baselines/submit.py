"""Exact six native attempts, source seal and conservative combined-cap DAG."""
import argparse
import getpass
import json
import re
import shlex
import shutil
import tarfile
from .common import *
from project.run_scripts.gpt2xl_prune_rect.submit import command,dependencies
from project.run_scripts.gpt2xl_prune_rect.submit import SOURCES as OLD_SOURCES
from project.run_scripts.jlz_price_gpt2xl.admission import width

ROLES=(*ARMS,'collector')
SOURCES=list(dict.fromkeys(OLD_SOURCES+[
 'project/run_scripts/gpt2xl_generation_baselines','project/run_scripts/experiment_generation_eval',
 ENVELOPE,CONTRACT,GENERATION_POLICY,
 'audits/servers/server1/gpt2xl-baselines-fluency-consistency-2k/cancellation.json']))

def order(cap):
    require(cap in (1,2),'TASK_CAP_SUPPORTED')
    if cap==1:return {r:[] if i==0 else [ARMS[i-1]] for i,r in enumerate(ARMS)}
    # Both heads depend on primary fresh W0 READY; subsequent independent
    # native cold arms remain at most two physical GPU lanes.
    return dict(BASE_MEMIT=[],BASE_ALPHAEDIT=['BASE_MEMIT'],CAKE=['BASE_MEMIT'],
        ALPHAEDIT_BLUE=['BASE_ALPHAEDIT'],PRUNE=['CAKE'],RECT=['ALPHAEDIT_BLUE'])

def launcher(source,commit,role,attempt,cpu=8):
    args=[PYTHON,'-u','-m','project.run_scripts.gpt2xl_generation_baselines.'+
          ('collect' if role=='collector' else 'run'),'--attempt',str(attempt)]
    if role!='collector':args+=['--arm',role]
    env=dict(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(cpu),
        MKL_NUM_THREADS=str(cpu),OPENBLAS_NUM_THREADS=str(cpu),TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(attempt/'tmp'/role))
    env[SOURCE_ENV]=commit
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    return '#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+'cd '+shlex.quote(str(source))+'\nexec '+shlex.join(args)+'\n'

def freeze(configpath,attempt):
    authority();require(attempt.parent==LOCAL and not attempt.exists(),'CREATE_ONCE_ATTEMPT')
    require(not command(['git','status','--porcelain','--',*SOURCES],ROOT),'COMMIT_SOURCE_BEFORE_FREEZE')
    c=read(configpath);require(c['instruction_id']==NONCE and c['attempt']==str(attempt),'CONFIG_SCOPE')
    tests=read(verify(c['cpu_preflight']));require(tests['status']=='PASS','CPU_PREFLIGHT')
    for row in tests['source']+tests['helper_source']+c['dependency_sources']+c['source_config_members']:verify(row)
    for row in c['generation']['source_members']:verify(row)
    verify(c['generation']['assets_manifest_member'])
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
        (attempt/'tmp'/role).mkdir(parents=True,mode=0o700)
        script=attempt/(role+'.sh');write_bytes(script,launcher(source,commit,role,attempt).encode());script.chmod(0o755)
    lock=dict(instruction_id=NONCE,task_id=TASK,source_commit=commit,source_tree=tree,
        config_sha256=sha(attempt/'config.json'),archive=member(archive),
        source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        runtime_sources=c['runtime']['source_members'],native_closure=c['dependency_sources'],
        source_config_members=c['source_config_members'],generation_reference=c['generation']['assets_manifest_member'],
        launchers=[member(attempt/(r+'.sh')) for r in ROLES],tracking_env=member(c['tracking']['env_file']),
        owner=getpass.getuser(),session=SESSION,host='server1',resources=c['resources'],
        noCP=True,checkpoint_saved=False,z_disk_cache=False,exact_resume='NOT_AVAILABLE',
        old_protected_jobs_mutated=False,generator_route='UNPADDED_FULL_PREFIX_NO_CACHE',
        baseline_cancel_receipt=c['cancellation_receipt'])
    write(attempt/'execution.lock.json',lock);return c,lock

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
    for old in set(dep)-got:
        state=command(['sacct','-n','-X','-j',old,'--format=JobIDRaw,State','-P'])
        require(any(line.split('|')[0]==old and line.split('|')[1].split()[0].split('+')[0] in
            ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE') for line in state.splitlines()),'REMOVED_DEP_NOT_TERMINAL')
    submit=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(submit and shlex.split(submit[1])==argv,'HELD_FULL_ARGV')
    path=attempt/(role+'.sh');require(command(['scontrol','write','batch_script',job,'-']).strip()==path.read_text().strip(),'HELD_SCRIPT_BYTES')
    return dict(job=job,role=role,argv=argv,dependency=dep,resource_detail=detail,launcher=member(path))

def submit(configpath,attempt):
    authority();require(not list(LOCAL.glob('*/submission.json')) and not list(LOCAL.glob('*/submitted-*.json')),'NO_DUPLICATE_NONCE')
    require(not command(['squeue','-h','-u',getpass.getuser(),'--name='+','.join(TASK+'-'+r for r in ROLES),'-o','%i|%j']),'DUPLICATE_JOB_NAME')
    from project.run_scripts.jlz_price_gpt2xl.submit import resource_inventory
    before=resource_inventory();existing=list(before['jobs'])
    row=next(x for x in (ROOT/'servers/local/gpu-caps.tsv').read_text().splitlines() if x.startswith('server1\t')).split('\t')
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server1\t')).split('\t')[1])
    cap=min(2,int(row[2]),tracked);require(cap>=1 and width(existing)<=cap,'EXISTING_COMBINED_CAP')
    frontier=dependencies(existing);roles=order(cap);virtual=list(existing);virtual_ids={}
    for i,role in enumerate(ARMS):
        fake=str(999999990+i);virtual_ids[role]=fake
        dep=frontier+[virtual_ids[x] for x in roles[role]]
        virtual.append(dict(job=fake,gpus=1,resource_detail='Dependency='+('afterany:'+':'.join(dep) if dep else '(null)')+' '))
    projected=width(virtual);require(projected<=cap,'PROJECTED_COMBINED_CAP')
    node=command(['scontrol','show','node','devbox']);partition=command(['scontrol','show','partition','gpu'])
    qos=command(['sacctmgr','-n','-P','show','qos','lab_gpu_s1','format=Name,MaxWall,MaxTRESPerJob,MaxTRESPerUser'])
    require('Gres=gpu' in node and 'PartitionName=gpu' in partition and qos,'CANONICAL_NODE_PARTITION_QOS')
    c,lock=freeze(configpath,attempt);r=c['resources']
    memory=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server1','--gpus','1','--mem',str(r['host_mib'])+'M','--local-limit-mib-per-gpu',row[3]])
    require('ALLOW_MEMORY_POLICY' in memory and shutil.disk_usage(attempt).free>=r['reserve_bytes'],'MEMORY_DISK_PREFLIGHT')
    write(attempt/'resource-preflight.json',dict(before=before,node=node,partition=partition,qos=qos,
        effective_cap=cap,projected_GPU_width=projected,frontier=frontier,memory_policy=memory,
        canonical_cap_policy=member(ROOT/'control/gpu-concurrency-policy.tsv'),old_jobs_mutated=False))
    ids={};held=[]
    for role in ROLES:
        dep=list(ids.values()) if role=='collector' else list(dict.fromkeys(frontier+[ids[x] for x in roles[role]]))
        argv=arguments(role,dep,attempt,r);job=command(argv).split(';')[0];require(job.isdigit(),'ACTUAL_JOB_ID');ids[role]=job
        write(attempt/('submitted-'+role+'.json'),dict(role=role,job=job,dependency=dep,argv=argv))
        held.append(inspect(job,role,dep,argv,attempt,r))
    write(attempt/'held-inspection.json',dict(source_commit=lock['source_commit'],config_sha256=lock['config_sha256'],jobs=held,passed=True))
    fresh=resource_inventory(exclude=tuple(ids.values()))
    final_width=width(fresh['jobs']+[dict(job=x['job'],gpus=1,resource_detail=x['resource_detail']) for x in held if x['role']!='collector'])
    require(final_width<=cap,'PRE_RELEASE_COMBINED_CAP')
    write(attempt/'pre-release-admission.json',dict(existing=fresh,projected_GPU_width=final_width,effective_cap=cap,passed=True))
    for item in lock['source_members']+lock['launchers']+lock['runtime_sources']+lock['native_closure']+lock['source_config_members']+[lock['archive'],lock['tracking_env'],lock['generation_reference']]:verify(item)
    release=[dict(role=role,job=ids[role],result=command(['scontrol','release',ids[role]])) for role in reversed(ROLES)]
    write(attempt/'release.json',dict(jobs=release))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%R'])
    receipt=dict(instruction_id=NONCE,task_id=TASK,jobs=ids,source_commit=lock['source_commit'],source_tree=lock['source_tree'],
        lock=member(attempt/'execution.lock.json'),config=member(attempt/'config.json'),held=member(attempt/'held-inspection.json'),release=member(attempt/'release.json'),
        dependencies={x['role']:x['dependency'] for x in held},snapshot=snapshot,initial='NOT_OBSERVED',
        W_B='NOT_YET_RUN; startup/readback inside actual jobs',checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
        combined_GPU_cap=cap,projected_GPU_width=projected,broadcast='NO_BROADCAST_NOT_REQUIRED',monitoring=False)
    write(attempt/'submission.json',receipt);print(json.dumps(receipt,ensure_ascii=False));return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    a=p.parse_args();submit(a.config.resolve(),a.attempt.resolve())
