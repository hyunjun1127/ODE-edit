"""One deliberate SH2 Qwen12 held-registration pass, cap4, no GPU qualification."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from official.experiments.prepare import file_sha, write_new
from official.runners.server2.qwen_plan import rows
from project.run_scripts import server2_qwen_archive as archive
from official.runners.server3.submit import AGENT_SEALS, official_tree_sha256, check_wandb

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
AUDIT=ROOT/'audits/servers/server2/qwen-migration-results-20261009'
def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def check(value,code):
    if not value:raise RuntimeError(code)
def cmd(argv,**kwargs):return subprocess.check_output(list(map(str,argv)),text=True,timeout=60,**kwargs).strip()

def verify(root,gpu_space=False):
    for m in read(root/'source-lock.json')['members']:
        check(file_sha(root/'source'/m['relative'])==m['sha256'],'FROZEN_SOURCE_CHANGED')
    for m in read(root/'input-lock.json')['members']:
        check(file_sha(m['path'])==m['sha256'],'FROZEN_INPUT_CHANGED')
    if gpu_space:check(shutil.disk_usage(root).free>=read(root/'prepared.json')['storage_reserve_bytes'],'STORAGE_KEEP_SOURCE')

def prepare(root,preparation,receiver=None):
    check(not root.exists(),'ATTEMPT_ALREADY_EXISTS')
    check(not cmd(['git','status','--porcelain'],cwd=ROOT),'SOURCE_DIRTY')
    commit=cmd(['git','rev-parse','HEAD'],cwd=ROOT)
    check(subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/main'],cwd=ROOT).returncode==0,'SOURCE_NOT_PUBLISHED')
    source=root/'source';source.mkdir(parents=True)
    for name in ('logs','scripts','processes'): (root/name).mkdir()
    closure=['official',*AGENT_SEALS,'project/run_scripts/checkpoint_archive',
        'project/run_scripts/server2_qwen_submit.py','project/run_scripts/server2_qwen_archive.py','project/run_scripts/server2_qwen_finish.py']
    subprocess.run(['git','archive','--format=tar','--output='+str(root/'source.tar'),commit,*closure],cwd=ROOT,check=True)
    with tarfile.open(root/'source.tar') as tar:
        for entry in tar.getmembers():
            check(not Path(entry.name).is_absolute() and '..' not in Path(entry.name).parts and (entry.isfile() or entry.isdir()),'UNSAFE_SOURCE_ARCHIVE')
            p=source/entry.name
            if entry.isdir():p.mkdir(parents=True,exist_ok=True)
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(entry) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst)
    lock=dict(code_commit=commit,official_tree_sha256=official_tree_sha256(source),members=[
        dict(relative=str(p.relative_to(source)),sha256=file_sha(p)) for p in sorted(source.rglob('*')) if p.is_file()])
    write_new(root/'source-lock.json',lock)
    for d in ('configs','streams'):shutil.copytree(preparation/d,root/d)
    assets=read(preparation/'assets.candidate.json');assets['output_root']=str(root)
    write_new(root/'assets.json',assets)
    shutil.copyfile(AUDIT/'cutover.json',root/'cutover.json')
    shutil.copyfile(ROOT/'control/final-checkpoint-archive-policy.json',root/'archive-policy.json')
    if receiver:shutil.copyfile(receiver,root/'receiver.json')
    os.environ.update(CUDA_VISIBLE_DEVICES='',NLTK_DATA=assets['nltk_data'])
    write_new(root/'wandb-project-precheck.json',check_wandb(source,assets['wandb_env']))
    os.environ['ODEEDIT_WANDB_PROJECT_VERIFIED']='1'
    from official.runners.server2.qwen_assets import preflight
    from official.runners.server2.qwen_run import checkpoint_identity,_tokenizer_receipt
    asset=preflight(root/'assets.json',hash_large=True)
    write_new(root/'asset-preflight.json',asset)
    check(asset['ready_to_submit'],'ASSET_PREFLIGHT_BLOCKED:'+str(asset['blockers']))
    for row in rows():
        logical=row['logical_main_row'];dataset=row['config']['dataset']
        stream=read(root/'streams'/f'{dataset}-stream.lock.json')
        token=_tokenizer_receipt(asset['assets']['model_snapshot']['path'],None,stream)
        archive.adopt(root,logical,checkpoint_identity(row['config'],stream,lock,asset,token),adapter_source=source/'project/run_scripts/server2_qwen_archive.py')
    roles=[(r['logical_main_row'],k) for r in rows() for k in ('gpu','archive')]+[('collector','collector')]
    for logical,kind in roles:
        env=dict(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
            HOME=pwd.getpwuid(os.getuid()).pw_dir,USER='janghj',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='6' if kind=='gpu' else '2',
            MKL_NUM_THREADS='6' if kind=='gpu' else '2',NLTK_DATA=assets['nltk_data'],
            ODEEDIT_SOURCE_LOCK=str(root/'source-lock.json'),ODEEDIT_CODE_COMMIT=commit,
            ODEEDIT_OFFICIAL_TREE_SHA256=lock['official_tree_sha256'],ODEEDIT_WANDB_PROJECT_VERIFIED='1',
            ODEEDIT_WANDB_ENV_FILE=assets['wandb_env'],WANDB_MODE='online',WANDB_DISABLED='false')
        if kind!='gpu':env['CUDA_VISIBLE_DEVICES']=''
        module='official.runners.server2.qwen_pipeline' if kind=='gpu' else 'project.run_scripts.server2_qwen_finish'
        argv=[PYTHON,'-B','-m',module,'--root',str(root)]
        argv+=['--cell',logical] if kind!='collector' else ['--collector']
        text='#!/bin/bash\nset -euo pipefail\n'+'\n'.join('export '+k+'='+shlex.quote(v) for k,v in env.items())+'\n'
        text+=shlex.join([PYTHON,'-B','-m','project.run_scripts.server2_qwen_submit','verify','--root',str(root)]+(['--gpu-space'] if kind=='gpu' else []))+'\n'
        text+='exec '+shlex.join(argv)+'\n'
        (root/'scripts'/f'{logical}-{kind}.sh').write_text(text)
    # All twelve retained CP plus four concurrent atomic replacements and raw reserve.
    sizes=[r['checkpoint_bytes'] for r in rows()]
    required=sum(sizes)+4*max(sizes)+32*1024**3
    check(shutil.disk_usage(root).free>=required,'FULL_RETAINED_CHECKPOINT_BUDGET')
    write_new(root/'prepared.json',dict(source=commit,cells=12,cap=4,job_ids=[],qualification='NOT_RUN_USER_DISABLED',
        retained_CP_bytes=sum(sizes),atomic_four_lane_bytes=4*max(sizes),initial_storage_required_bytes=required,
        storage_reserve_bytes=32*1024**3,archive_without_ready='ARCHIVE_PENDING_KEEP_SOURCE',
        CF_generation='W0_AND_W20_FIRST2000',zsre_generation='NOT_APPLICABLE'))
    paths=[p for d in ('configs','streams','scripts') for p in (root/d).iterdir() if p.is_file()]
    paths += [root/n for n in ('assets.json','cutover.json','archive-policy.json','prepared.json')]
    paths.append(Path(assets['generation_reference_manifest']))
    if receiver:paths.append(root/'receiver.json')
    write_new(root/'input-lock.json',dict(members=[dict(path=str(p),sha256=file_sha(p)) for p in paths]))
    print(json.dumps({'stage':'PREPARED_NOT_SUBMITTED','source':commit,'required_bytes':required}))

def fields(job):
    text=cmd(['scontrol','show','job',job,'-o'])
    return text,dict(x.split('=',1) for x in text.split() if '=' in x)

def admission():
    observed=[]
    for raw in cmd(['scontrol','show','job','-o']).splitlines():
        f=dict(x.split('=',1) for x in raw.split() if '=' in x)
        if f.get('UserId','').startswith('janghj(') and ('server2' in (f.get('NodeList'),f.get('ReqNodeList'))):
            if f.get('JobState') in ('RUNNING','COMPLETING','CONFIGURING','PENDING') and 'gres/gpu' in f.get('ReqTRES',''):
                observed.append(f)
    # No guessing at an unfamiliar owner's frontier: caller must rebind explicitly.
    check(not observed,'OWN_GPU_FRONTIER_REQUIRES_EXACT_BINDING')
    return dict(at=now(),owned_active_GPU_jobs=observed,cap=4,
        cap_authority='Latest direct USER server2 cap4; supersedes older TSV cap2',
        local_caps=cmd(['sed','-n','1,8p','/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv']),
        node=cmd(['scontrol','show','node','server2','-o']),
        qos=cmd(['sacctmgr','-n','-P','show','qos','lab_gpu_s2','format=Name,MaxTRESPU']),
        no_unrelated_mutations=True)

def inspect(root,item):
    raw,f=fields(item['job_id']);gpu=item['kind']=='gpu'
    expected=dict(JobId=item['job_id'],JobName=item['name'],JobState='PENDING',Requeue='0',
        Command=str(root/'scripts'/f"{item['cell']}-{item['kind']}.sh"),WorkDir=str(root),ReqNodeList='server2',
        **{'CPUs/Task':'6' if gpu else '2'},MinMemoryNode='58G' if gpu else '4G')
    for k,v in expected.items():check(f.get(k)==v,'HELD_'+k)
    check(f.get('UserId','').startswith('janghj(') and f.get('Reason')=='JobHeldUser','HELD_OWNER_OR_REASON')
    actual=set(re.findall(r'(afterany|afterok):(\d+)',f.get('Dependency','')))
    check(actual=={('afterany',j) for j in item['dependencies']},'HELD_DEPENDENCY')
    check(('gres/gpu=1' in f.get('ReqTRES','')) if gpu else ('gres/gpu' not in f.get('ReqTRES','')),'HELD_GPU')
    return raw

def submit(root):
    verify(root,True);check(not (root/'registration-started.json').exists(),'NO_DUPLICATE_OR_AUTOMATIC_RETRY')
    write_new(root/'registration-started.json',admission())
    jobs=[];lanes=[None]*4;producer={};previous_archive=None
    configs=rows();ordered=[r for method in ('FT','MEMIT','ALPHAEDIT','ALPHAEDIT_BLUE','MEMIT_FE','SPHERE') for r in configs if r['config']['method']==method]
    for i,row in enumerate(ordered):
        cell=row['logical_main_row'];dataset=row['config']['dataset'];lane=i%4
        deps=sorted({x for x in (lanes[lane],producer.get(dataset)) if x})
        for kind in ('gpu','archive'):
            gpu=kind=='gpu'
            if not gpu:deps=[jobs[-1]['job_id']]+([previous_archive] if previous_archive else [])
            item=register(root,cell,kind,deps,row['config']);jobs.append(item)
            if gpu:
                lanes[lane]=item['job_id'];producer.setdefault(dataset,item['job_id'])
                archive.registered(root,cell,item['job_id'],item['submitted_at'])
            else:previous_archive=item['job_id']
    jobs.append(register(root,'collector','collector',[x['job_id'] for x in jobs],None))
    write_new(root/'submission.json',dict(stage='ALL_HELD',jobs=jobs,qualification='NOT_RUN_USER_DISABLED'))
    verify(root)
    for item in jobs:inspect(root,item)
    for item in reversed(jobs):
        cmd(['scontrol','release',item['job_id']])
        write_new(root/'releases'/f"{item['job_id']}.json",dict(job_id=item['job_id'],at=now()))
    snapshot=cmd(['squeue','-j',','.join(j['job_id'] for j in jobs),'-h','-o','%i|%j|%T|%E'])
    write_new(root/'released.json',dict(at=now(),stage='RELEASED',jobs=jobs,snapshot=snapshot))
    print(json.dumps(dict(jobs=jobs,snapshot=snapshot)))

def register(root,cell,kind,deps,config):
    gpu=kind=='gpu';name='s2-'+cell+'-'+kind
    argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s2','--nodelist=server2',
        '--nodes=1','--ntasks=1','--cpus-per-task='+('6' if gpu else '2'),'--mem='+('59392M' if gpu else '4096M'),
        '--time='+('48:00:00' if gpu else '04:00:00'),'--export=NONE','--no-requeue','--job-name='+name,
        '--chdir='+str(root),'--output='+str(root/'logs'/f'{cell}-{kind}-%j.out'),'--error='+str(root/'logs'/f'{cell}-{kind}-%j.err')]
    if gpu:argv+=['--gres=gpu:a6000:1']
    if deps:argv+=['--dependency=afterany:'+':'.join(deps)]
    argv+=[str(root/'scripts'/f'{cell}-{kind}.sh')]
    at=now();job=cmd(argv).split(';')[0];check(job.isdigit(),'SBATCH_NO_ID')
    item=dict(job_id=job,cell=cell,kind=kind,name=name,dependencies=deps,argv=argv,submitted_at=at,
        source=read(root/'source-lock.json')['code_commit'],config_sha256=config['config_sha256'] if config else None)
    write_new(root/'registrations'/f'{job}.json',item)
    write_new(root/'inspections'/f'{job}.json',dict(job_id=job,raw=inspect(root,item)))
    return item

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify','submit']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--preparation',type=Path);p.add_argument('--receiver',type=Path);p.add_argument('--gpu-space',action='store_true');a=p.parse_args()
    if a.command=='prepare':prepare(a.root,a.preparation,a.receiver)
    elif a.command=='verify':verify(a.root,a.gpu_space)
    else:submit(a.root)
