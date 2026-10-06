"""Exact cap-safe held GPU chain + afterany CPU collector, no polling loop."""
import argparse
import getpass
import json
import re
import shlex
import shutil
import subprocess
import tarfile
from . import *
from .profile import profile
SESSION='01a0493a-074c-7f91-9a13-769116326fef'
SOURCES=['project/run_scripts/fe_baseline','project/run_scripts/jlz_realization/__init__.py',
    'project/run_scripts/jlz_realization/common.py','project/run_scripts/jlz_realization/inputs.py',
    'project/run_scripts/jlz_realization/observe.py','project/run_scripts/jlz_shared_budget/__init__.py',
    'project/run_scripts/jlz_shared_budget/common.py','project/run_scripts/jlz_pilot/__init__.py',
    'project/run_scripts/jlz_pilot/prompts.py','scripts/fixed_counterfact.py',
    'plans/global/fe-sequential-2k','messages/head/2026-10-06-fe-sequential-2k-sh2.json',
    'control/experiment-exceptions/fe-sequential-2k-sh2-20261006.json','control/gpu-concurrency-policy.tsv','servers/slurm-memory-policy.tsv',
    'project/run_scripts/experiment_tracking','control/wandb-policy.json',
    'messages/head/2026-10-06-fe-wandb-login-resume-sh2.json','plans/updates/server2/fe-sequential-2k/user-10k.md']
def command(argv,cwd=None):
    r=subprocess.run(argv,cwd=cwd,text=True,capture_output=True);require(r.returncode==0,'COMMAND_FAILED:'+shlex.join(argv)+':'+r.stderr);return r.stdout.strip()
def launcher(role,a,commit,c):
    env=dict(PYTHONPATH=str(a/'source'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONHASHSEED='0',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',FE_SOURCE_COMMIT=commit)
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    args=[c['runtime']['python'],'-u','-m','project.run_scripts.fe_baseline.'+('collect' if role=='collector' else 'run'),'--attempt',str(a)]
    return '#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+'cd '+shlex.quote(str(a/'source'))+'\nexec '+shlex.join(args)+'\n'
def argv(role,a,dependency=None,task=TASK):
    x=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s2','--nodelist=server2','--nodes=1','--ntasks=1',
        '--cpus-per-task=6','--export=NONE','--no-requeue','--job-name='+task,'--chdir='+str(a/'source'),
        '--mem='+('24576M' if role=='collector' else '59392M'),'--time='+('04:00:00' if role=='collector' else '48:00:00'),
        '--output='+str(a/(role+'-%j.out')),'--error='+str(a/(role+'-%j.err'))]
    if role=='main':x+=['--gres=gpu:1']
    if dependency:x+=['--dependency='+dependency]
    return x+[str(a/(role+'.sh'))]
def inventory(exclude=()):
    raw=command(['squeue','-h','-r','-w','server2','-u',getpass.getuser(),'-o','%i|%j|%T|%b|%R']);jobs=[]
    for line in raw.splitlines():
        job,name,status,gres,reason=line.split('|',4)
        if job in exclude:continue
        detail=command(['scontrol','show','job',job,'--oneliner']);req=re.search(r'\bReqTRES=([^ ]+)',detail);gpu=re.search(r'gres/gpu=(\d+)',req[1]) if req else None
        if gpu:jobs.append(dict(job=job,name=name,state=status,GPUs=int(gpu[1]),reason=reason,detail=detail))
    return jobs
def inspect(job,role,dep,args,a,task=TASK):
    d=command(['scontrol','show','job',job,'--oneliner'])
    for s in [f'JobId={job} ',f'JobName={task} ','UserId='+getpass.getuser()+'(','JobState=PENDING ','Reason=JobHeldUser ',
        'Requeue=0 ','CPUs/Task=6 ','ReqTRES=cpu=6,','ReqNodeList=server2 ','Partition=gpu ','QOS=lab_gpu_s2 ']:require(s in d,'HELD:'+s)
    require(re.search(r'\bNumCPUs=6(?:-\d+)? ',d),'HELD_CPUS')
    mem=24576 if role=='collector' else 59392
    require(f'mem={mem}M' in d or f'mem={mem//1024}G' in d,'HELD_MEMORY')
    require(('gres/gpu' not in d) if role=='collector' else 'TresPerNode=gres/gpu:1' in d,'HELD_GPU')
    require('TimeLimit='+('04:00:00' if role=='collector' else '2-00:00:00')+' ' in d,'HELD_WALL')
    observed=re.search(r'\bDependency=([^ ]+)',d)[1]
    dep_ids=set(re.findall(r'(?:afterany:|:)(\d+)',observed))
    require(dep_ids==(set(dep.split(':')[1:]) if dep else set()) and (not dep or observed.startswith('afterany:')),'HELD_DEPENDENCY')
    require('Command='+str(a/(role+'.sh'))+' ' in d and 'WorkDir='+str(a/'source')+' ' in d,'HELD_PATH')
    line=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',d)
    require(line and shlex.split(line[1])==args,'HELD_FULL_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==(a/(role+'.sh')).read_text().strip(),'HELD_BATCH_SCRIPT')
    return dict(job=job,role=role,argv=args,detail=d)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--attempt-name',default='attempt');args=p.parse_args()
    require(args.attempt_name.startswith('attempt') and '/' not in args.attempt_name,'ATTEMPT_NAME');a=LOCAL/args.attempt_name
    require(not a.exists(),'CREATE_ONCE_ATTEMPT')
    require(not command(['squeue','-h','-w','server2','-u',getpass.getuser(),'--name=fe-sequential-2k,fe-sequential-10k','-o','%i']),'DUPLICATE_TASK')
    # Latest USER policy arrived before FE freeze/submission. No grandfathering.
    # Shared experiment_tracking is SH1-owned; do not substitute a private logger.
    c=json.loads(args.config.read_text())
    require(c['profile']==profile(c['settings']['requests']) and c['task_id']==c['profile']['task_id'],'PROFILE')
    require(c.get('tracking',{}).get('integration')=='SH1_SHARED_HELPER_BOUND','LOGGING_BLOCKED_SHARED_HELPER_NOT_BOUND')
    online=json.loads(verify(c['tracking']['online_receipt']).read_text())
    verify(c['tracking']['env_member'])
    for row in c['tracking']['helper_members']:verify(row)
    require(online['status']=='READY_ONLINE_VERIFIED' and online['entity']=='wkdguswns2256' and online['project']=='layer allocation','NEEDS_USER_LOGIN_OR_ONLINE_READBACK')
    before=inventory();tracked=int(next(r for r in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if r.startswith('server2\t')).split('\t')[1])
    local=int(next(r for r in Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if r.startswith('server2\t')).split('\t')[2]);cap=min(2,tracked,local)
    require(cap>=1,'CAP_DISABLED');c=json.loads(args.config.read_text());r=c['resources']
    require(c['instruction_id']==NONCE and r['cpu']==6 and r['host_mib']==59392 and r['wall']=='48:00:00','AUTHORITY_RESOURCES')
    require(not command(['git','status','--porcelain','--',*SOURCES],ROOT),'COMMIT_SOURCE')
    tests=json.loads(verify(c['cpu_preflight']).read_text());require(tests['passed'],'CPU')
    for row in tests['source']+c['upstream_members']+c['runtime']['source_members']+c['authority_members']:verify(row)
    for row in c['assets']:
        st=Path(row['path']).stat();require((st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']),'ASSET_CHANGED')
    require(shutil.disk_usage(LOCAL).free>=r['reserve_bytes'],'DISK_RESERVE')
    node=command(['scontrol','show','node','server2']);partition=command(['scontrol','show','partition','gpu'])
    qos=command(['sacctmgr','-n','-P','show','qos','lab_gpu_s2','format=Name,MaxWall,MaxTRESPerUser,MaxTRESPerJob'])
    commit=command(['git','rev-parse','HEAD'],ROOT);tree=command(['git','rev-parse','HEAD^{tree}'],ROOT)
    a.mkdir();source=a/'source';source.mkdir();archive=a/'source.tar';command(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCES],ROOT)
    with tarfile.open(archive) as t:
        require(all((x.isfile() or x.isdir()) and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in t.getmembers()),'SAFE_ARCHIVE');t.extractall(source,filter='data')
    for row in tests['source']:require(sha(source/Path(row['path']).relative_to(ROOT))==row['sha256'],'TESTED_SOURCE_ARCHIVE')
    write(a/'config.json',c)
    for role in ('main','collector'):
        script=a/(role+'.sh');script.write_text(launcher(role,a,commit,c));script.chmod(0o755)
    lock=dict(instruction_id=NONCE,source_commit=commit,source_tree=tree,config_sha256=sha(a/'config.json'),archive=member(archive),
        source_members=[member(x) for x in sorted(source.rglob('*')) if x.is_file()],launchers=[member(a/(role+'.sh')) for role in ('main','collector')],
        resources=r,owner=getpass.getuser(),host='server2',session=SESSION,noCP=True,exact_resume='NOT_AVAILABLE')
    write(a/'execution.lock.json',lock);barrier=[j['job'] for j in before] if sum(j['GPUs'] for j in before)+1>cap else []
    ids={};held=[]
    for role in ('main','collector'):
        dep='afterany:'+ids['main'] if role=='collector' else ('afterany:'+':'.join(barrier) if barrier else None)
        args=argv(role,a,dep,c['task_id']);job=command(args).split(';')[0];require(job.isdigit(),'JOB_ID');ids[role]=job
        write(a/('submitted-'+role+'.json'),dict(job=job,argv=args,dependency=dep));held.append(inspect(job,role,dep,args,a,c['task_id']))
    pre=inventory(tuple(ids.values()));require({j['job'] for j in pre}<={j['job'] for j in before},'ADMISSION_RACE_KEEP_HELD')
    require(barrier or sum(j['GPUs'] for j in pre)+1<=cap,'CAP_KEEP_HELD')
    write(a/'held-inspection.json',dict(jobs=held,before=before,prerelease=pre,cap=cap,external_dependencies=barrier,node=node,partition=partition,qos=qos))
    for role in ('collector','main'):write(a/('released-'+role+'.json'),dict(job=ids[role],result=command(['scontrol','release',ids[role]]),success=True))
    snap=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%b|%N|%r'])
    receipt=dict(nonce=NONCE,status='SUBMISSION_HANDOFF',jobs=ids,source=commit,lock=member(a/'execution.lock.json'),initial_snapshot=snap,
        resources=r,profile=c['profile'],actual_qualification='NOT_OBSERVED',final_endpoint='NOT_OBSERVED',monitoring_active=False,automatic_resume=False,automatic_retry=False)
    write(a/'submission.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
