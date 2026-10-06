"""Held direct-native jobs at exact existing writer frontiers; no auto retry."""
import argparse
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from .common import *

ROLES=(*ARMS,'collector')
SESSION='01a04939-f93a-7b50-bca0-65438eab2062'
# Read-only source/import closure, not copies of assets or the EasyEdit repository.
SOURCES=['project/run_scripts/gpt2xl_native_baselines','messages/head/2026-10-07-gpt2xl-easyedit-native-baselines-sh1.json',
 'project/run_scripts/jlz_price_gpt2xl','project/run_scripts/jlz_interference_l1',
 'project/run_scripts/jlz_realization','project/run_scripts/jlz_pilot','project/run_scripts/jlz_shared_budget',
 'project/run_scripts/jlz_realized_writer_sequential/review_completed.py',
 'project/run_scripts/jlz_realized_writer_sequential/__init__.py','project/run_scripts/jlz_v12r',
 'project/run_scripts/jlz_native_writer_aware','project/run_scripts/jlz_realized_subject',
 'project/run_scripts/jlz_writer_coupled','project/run_scripts/jlz_realized_writer',
 'project/run_scripts/experiment_tracking','scripts/fixed_counterfact.py',
 'scripts/slurm_memory_policy.py','control/gpu-concurrency-policy.tsv',
 'control/wandb-policy.json','control/wandb-method-metric-schema.json','servers/slurm-memory-policy.tsv']

def command(argv,cwd=None):
    r=subprocess.run(argv,cwd=cwd,text=True,capture_output=True)
    require(r.returncode==0,'COMMAND_FAILED:'+shlex.join(argv)+':'+r.stderr)
    return r.stdout.strip()

def launcher(source,commit,role,attempt):
    module='collect' if role=='collector' else 'run'
    args=[PYTHON,'-u','-m','project.run_scripts.gpt2xl_native_baselines.'+module,'--attempt',str(attempt)]
    if role!='collector':args+=['--arm',role]
    env=dict(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GPT2_NATIVE_BASELINE_SOURCE_COMMIT=commit)
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    return '#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+'cd '+shlex.quote(str(source))+'\nexec '+shlex.join(args)+'\n'

def freeze(configpath,attempt):
    authority();require(attempt.parent==LOCAL and not attempt.exists(),'CREATE_ONCE_ATTEMPT')
    require(not command(['git','status','--porcelain','--',*SOURCES],ROOT),'COMMIT_SOURCE_BEFORE_FREEZE')
    c=json.loads(configpath.read_text());require(c['instruction_id']==NONCE and c['attempt']==str(attempt),'CONFIG_SCOPE')
    verify(c['cpu_preflight']);tests=json.loads(Path(c['cpu_preflight']['path']).read_text());require(tests['status']=='PASS','CPU_PREFLIGHT')
    for row in tests['source']+tests['helper_source']:verify(row)
    commit=command(['git','rev-parse','HEAD'],ROOT);tree=command(['git','rev-parse','HEAD^{tree}'],ROOT)
    attempt.mkdir();source=attempt/'source';source.mkdir();archive=attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCES],ROOT)
    with tarfile.open(archive) as tf:
        entries=tf.getmembers();require(len(entries)==len({x.name for x in entries}) and all((x.isfile() or x.isdir()) and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in entries),'SAFE_SOURCE_ARCHIVE')
        tf.extractall(source,filter='data')
    write(attempt/'config.json',c)
    for role in ROLES:
        path=attempt/(role+'.sh');path.write_text(launcher(source,commit,role,attempt));path.chmod(0o755)
    lock=dict(instruction_id=NONCE,task_id=TASK,source_commit=commit,source_tree=tree,
        config_sha256=sha(attempt/'config.json'),archive=member(archive),
        source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        runtime_sources=c['runtime']['source_members'],native_closure=c['native']['closure'],
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
    for text in (f'JobId={job} ',f'JobName={TASK}-{role} ','UserId='+getpass.getuser()+'(',
        'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ',f'CPUs/Task={cpus} ',
        'ReqNodeList=devbox ','Partition=gpu ','QOS=lab_gpu_s1 ',
        'Command='+str(attempt/(role+'.sh'))+' ','WorkDir='+str(attempt/'source')+' '):require(text in detail,'HELD_FIELD:'+text)
    require('gres/gpu' not in detail if cpu else 'TresPerNode=gres/gpu:1' in detail,'HELD_GPU')
    require(f'mem={mem}M' in detail or f'mem={mem//1024}G' in detail,'HELD_MEMORY')
    require('TimeLimit='+(r['collector_wall'] if cpu else r['wall'])+' ' in detail,'HELD_WALL')
    observed=re.search(r'\bDependency=([^ ]+)',detail)[1]
    got=set(re.findall(r'(?:^|:)(\d+)(?=\(|:|\?|$)',observed)) if observed!='(null)' else set()
    require(got==set(dep) and (not dep or observed.startswith('afterany:')),'HELD_DEPENDENCY')
    submit=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(submit and shlex.split(submit[1])==argv,'HELD_FULL_ARGV')
    path=attempt/(role+'.sh')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==path.read_text().strip(),'HELD_SCRIPT_BYTES')
    return dict(job=job,role=role,argv=argv,dependency=dep,resource_detail=detail,launcher=member(path))

def resource_frontiers(inventory,ours):
    expected=ours['jobs'] if isinstance(ours.get('jobs'),dict) else ours['mapping']
    # Submission receipt may contain a mapping under cells; normalize explicitly.
    if isinstance(expected,list):expected={x['role']:x['job'] for x in expected}
    expected={k:(str(v['job']) if isinstance(v,dict) else str(v)) for k,v in expected.items()}
    roots={'BASE_MEMIT':['MEMIT_CAP075','MEMIT_CAP100','MEMIT_FREE100'],
           'BASE_ALPHAEDIT':['ALPHAEDIT_CAP075','ALPHAEDIT_CAP100','ALPHAEDIT_FREE100']}
    current={j['job']:j for j in inventory['jobs']}
    known={expected[x] for names in roots.values() for x in names}
    external=[j['job'] for j in inventory['jobs'] if j['job'] not in known]
    result={}
    for role,names in roots.items():
        active=[expected[n] for n in names if expected[n] in current]
        if active:
            tail=expected[names[-1]]
            result[role]=([tail] if tail in current else active)+external
        else:result[role]=external.copy()
    return result,expected

def submit(configpath,attempt):
    authority();require(not list(LOCAL.glob('*/submission.json')) and not list(LOCAL.glob('*/submitted-*.json')),'NO_DUPLICATE_NONCE_SUBMISSION')
    require(not command(['squeue','-h','-u',getpass.getuser(),'--name='+','.join(TASK+'-'+r for r in ROLES),'-o','%i|%j']),'EXACT_NAME_DUPLICATE')
    from project.run_scripts.jlz_price_gpt2xl.submit import resource_inventory
    from project.run_scripts.jlz_price_gpt2xl.admission import width
    before=resource_inventory()
    old=Path('/mnt/raid5/janghj/ODE-edit/local/jlz-price-gpt2xl-2k/attempt-checkpoint-repair-r1')
    ours=json.loads((old/'submission.json').read_text());oldlock=json.loads((old/'execution.lock.json').read_text())
    deps,expected=resource_frontiers(before,ours)
    for j in before['jobs']:
        matched=[role for role,job in expected.items() if job==j['job']]
        if matched:
            role=matched[0];require('UserId='+getpass.getuser()+'(' in j['resource_detail'] and
                'Command='+str(old/(role+'.sh'))+' ' in j['resource_detail'] and
                j['name']=='jlz-price-gpt2xl-2k-'+role,'OURS_FRONTIER_OWNER_SOURCE')
    row=next(x for x in Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if x.startswith('server1\t')).split('\t')
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server1\t')).split('\t')[1]);cap=min(2,int(row[2]),tracked)
    require(cap>=1,'NO_PROJECT_GPU_CAP')
    if cap==1:deps['BASE_ALPHAEDIT']=sorted(set(deps['BASE_ALPHAEDIT']+deps['BASE_MEMIT']))
    node=command(['scontrol','show','node','devbox']);partition=command(['scontrol','show','partition','gpu'])
    c,lock=freeze(configpath,attempt);r=c['resources']
    memory=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server1','--gpus','1','--mem',str(r['host_mib'])+'M','--local-limit-mib-per-gpu',row[3]])
    require('ALLOW_MEMORY_POLICY' in memory,'HARD_MEMORY_POLICY')
    require(shutil.disk_usage(attempt).free>=r['reserve_bytes'],'DISK_RESERVE')
    # Simulate appended lanes in the existing DAG; no physical resource claiming by inspection.
    virtual=list(before['jobs'])
    for i,role in enumerate(ARMS):
        if cap==1 and i==1:deps[role]+=['NEW_BASE_MEMIT']
        virtual.append(dict(job='NEW_'+role,gpus=1,resource_detail='Dependency='+('afterany:'+':'.join(deps[role]) if deps[role] else '(null)')+' '))
    # width parser is numeric; use numeric sentinels exclusively for the CPU graph (not submitted IDs).
    for j in virtual:
        j['job']=j['job'].replace('NEW_BASE_MEMIT','999999991').replace('NEW_BASE_ALPHAEDIT','999999992')
        j['resource_detail']=j['resource_detail'].replace('NEW_BASE_MEMIT','999999991').replace('NEW_BASE_ALPHAEDIT','999999992')
    projected=width(virtual);require(projected<=cap,'COMBINED_GPU_DAG_WIDTH')
    write(attempt/'resource-preflight.json',dict(before=before,node=node,partition=partition,effective_cap=cap,
        projected_GPU_width=projected,memory_policy=memory,old_source=oldlock['source_commit'],old_submission=member(old/'submission.json'),
        frontier_dependencies=deps,checkpoint_saved=False,existing_jobs_mutated=False))
    ids={};held=[]
    for role in ROLES:
        dep=list(ids.values()) if role=='collector' else [ids['BASE_MEMIT'] if x=='NEW_BASE_MEMIT' else x for x in deps[role]]
        argv=arguments(role,dep,attempt,r)
        job=command(argv).split(';')[0];require(job.isdigit(),'ACTUAL_SLURM_JOB_ID');ids[role]=job
        write(attempt/('submitted-'+role+'.json'),dict(role=role,job=job,dependency=dep,argv=argv))
        held.append(inspect(job,role,dep,argv,attempt,r))
    write(attempt/'held-inspection.json',dict(source_commit=lock['source_commit'],config_sha256=lock['config_sha256'],jobs=held,passed=True))
    fresh=resource_inventory(exclude=tuple(ids.values()))
    final_graph=list(fresh['jobs'])
    for item in held:
        if item['role']!='collector':final_graph.append(dict(job=item['job'],gpus=1,resource_detail=item['resource_detail']))
    final_width=width(final_graph);require(final_width<=cap,'PRE_RELEASE_COMBINED_GPU_DAG_WIDTH')
    write(attempt/'pre-release-admission.json',dict(existing=fresh,projected_GPU_width=final_width,effective_cap=cap,own_held_excluded=list(ids.values()),passed=True))
    for item in lock['source_members']+lock['launchers']+lock['runtime_sources']+lock['native_closure']+[lock['archive'],lock['tracking_env']]:verify(item)
    release=[]
    for role in reversed(ROLES):release.append(dict(role=role,job=ids[role],result=command(['scontrol','release',ids[role]])))
    write(attempt/'release.json',dict(jobs=release))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%R'])
    receipt=dict(instruction_id=NONCE,task_id=TASK,jobs=ids,source_commit=lock['source_commit'],source_tree=lock['source_tree'],
        lock=member(attempt/'execution.lock.json'),config=member(attempt/'config.json'),held=member(attempt/'held-inspection.json'),release=member(attempt/'release.json'),
        dependencies={x['role']:x['dependency'] for x in held},snapshot=snapshot,initial='NOT_OBSERVED',
        W_B='NOT_YET_RUN; startup/readback inside actual jobs',checkpoint_saved=False,exact_resume='NOT_AVAILABLE',
        combined_GPU_cap=cap,projected_GPU_width=projected,broadcast='NO_BROADCAST_NOT_REQUIRED',monitoring=False)
    write(attempt/'submission.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
    return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True);a=p.parse_args();submit(a.config.resolve(),a.attempt.resolve())
if __name__=='__main__':main()
