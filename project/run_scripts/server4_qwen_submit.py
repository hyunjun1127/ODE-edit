"""One deliberate held-registration pass for twelve serial Qwen official cells."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from official.runners.server4.qwen_plan import rows
from official.runners.server3.submit import official_tree_sha256, AGENT_SEALS, check_wandb
from official.experiments.prepare import write_new, file_sha
from project.run_scripts import server4_qwen_archive as archive

ROOT=Path(__file__).resolve().parents[2]
PYTHON='/data/janghj/EasyEdit/.venv/bin/python'

def cmd(argv,**kwargs):return subprocess.check_output(list(map(str,argv)),text=True,timeout=60,**kwargs).strip()
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def check(ok,reason):
    if not ok:raise RuntimeError(reason)

def verify(root,gpu_space=False):
    lock=read(root/'source-lock.json')
    for m in lock['members']:
        check(file_sha(root/'source'/m['relative'])==m['sha256'],'FROZEN_SOURCE_CHANGED')
    for m in read(root/'input-lock.json')['members']:
        check(file_sha(Path(m['path']))==m['sha256'],'FROZEN_INPUT_CHANGED')
    if gpu_space:check(shutil.disk_usage(root).free>=32*1024**3,'RESOURCE_BLOCKED_STORAGE_KEEP_SOURCE')

def prepare(root,preparation):
    check(not root.exists(),'ATTEMPT_ALREADY_EXISTS')
    check(not cmd(['git','status','--porcelain'],cwd=ROOT),'SOURCE_DIRTY')
    commit=cmd(['git','rev-parse','HEAD'],cwd=ROOT)
    check(subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/main'],cwd=ROOT).returncode==0,'SOURCE_NOT_PUBLISHED')
    root.mkdir(parents=True);(root/'logs').mkdir();(root/'scripts').mkdir();(root/'processes').mkdir()
    source=root/'source';source.mkdir()
    closure=['official',*AGENT_SEALS,'project/run_scripts/checkpoint_archive',
             'project/run_scripts/server4_qwen_archive.py','project/run_scripts/server4_qwen_submit.py']
    subprocess.run(['git','archive','--format=tar','--output='+str(root/'source.tar'),commit,*closure],cwd=ROOT,check=True)
    with tarfile.open(root/'source.tar') as tar:
        for entry in tar.getmembers():
            check(not Path(entry.name).is_absolute() and '..' not in Path(entry.name).parts and (entry.isfile() or entry.isdir()),'UNSAFE_ARCHIVE')
            p=source/entry.name
            if entry.isdir():p.mkdir(parents=True,exist_ok=True)
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(entry) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst)
    lock=dict(code_commit=commit,official_tree_sha256=official_tree_sha256(source),
              members=[dict(relative=str(p.relative_to(source)),sha256=file_sha(p)) for p in sorted(source.rglob('*')) if p.is_file()])
    write_new(root/'source-lock.json',lock)
    for directory in ('configs','streams'):shutil.copytree(preparation/directory,root/directory)
    assets=read(preparation/'assets.candidate.json');assets['output_root']=str(root)
    write_new(root/'assets.json',assets)
    for name,src in [('cutover.json',ROOT/'audits/servers/server4/qwen-baselines-20261009/cutover.json'),
                     ('archive-policy.json',ROOT/'control/final-checkpoint-archive-policy.json')]:
        shutil.copyfile(src,root/name)
    received=read(ROOT/'audits/global/qwen12-archive-connection-20261009/receiver-ready.json')['receiver_binding']
    write_new(root/'receiver.json',dict(host='codex-server1',source=received['checkout'],policy=received['policy'],
        python=received['python'],policy_sha256=received['policy_sha256'],
        cutovers=received['cutovers'],helper_members=received['helper_members']))
    os.environ['NLTK_DATA']=assets['nltk_data'];os.environ['CUDA_VISIBLE_DEVICES']=''
    write_new(root/'wandb-project-precheck.json',check_wandb(source,assets['wandb_env']))
    os.environ['ODEEDIT_WANDB_PROJECT_VERIFIED']='1'
    from official.runners.server4.qwen_assets import preflight
    from official.runners.server4.qwen_run import checkpoint_identity,_tokenizer_receipt
    asset=preflight(root/'assets.json',hash_large=True)
    write_new(root/'asset-preflight.json',asset)
    check(asset['ready_to_submit'],'ASSET_PREFLIGHT_BLOCKED:'+str(asset['blockers']))
    for row in rows():
        logical=row['logical_main_row'];dataset=row['config']['dataset']
        stream=read(root/'streams'/f'{dataset}-stream.lock.json')
        token=_tokenizer_receipt(asset['assets']['model_snapshot']['path'],None,stream)
        archive.adopt(root,logical,checkpoint_identity(row['config'],stream,lock,asset,token),
                      adapter_source=source/'project/run_scripts/server4_qwen_archive.py')
        for kind in ('gpu','archive'):
            env=dict(PYTHONPATH=str(source),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                     HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
                     OMP_NUM_THREADS='8' if kind=='gpu' else '2',MKL_NUM_THREADS='8' if kind=='gpu' else '2',
                     NLTK_DATA=assets['nltk_data'],ODEEDIT_SOURCE_LOCK=str(root/'source-lock.json'),
                     ODEEDIT_CODE_COMMIT=commit,ODEEDIT_OFFICIAL_TREE_SHA256=lock['official_tree_sha256'],
                     ODEEDIT_WANDB_PROJECT_VERIFIED='1',WANDB_MODE='online',WANDB_DISABLED='false')
            if kind=='archive':env['CUDA_VISIBLE_DEVICES']=''
            module='official.runners.server4.qwen_pipeline' if kind=='gpu' else 'project.run_scripts.server4_qwen_archive'
            text='#!/bin/bash\nset -euo pipefail\n'+ '\n'.join('export '+k+'='+shlex.quote(v) for k,v in env.items())+'\n'
            text+=shlex.join([PYTHON,'-B','-m','project.run_scripts.server4_qwen_submit','verify','--root',str(root)]+(['--gpu-space'] if kind=='gpu' else []))+'\n'
            text+='exec '+shlex.join([PYTHON,'-B','-m',module,'--root',str(root),'--cell',logical])+'\n'
            with (root/'scripts'/f'{logical}-{kind}.sh').open('x') as f:f.write(text)
    paths=[p for d in ('configs','streams','scripts') for p in (root/d).iterdir() if p.is_file()]
    paths += [root/n for n in ('assets.json','receiver.json','cutover.json','archive-policy.json')]
    write_new(root/'input-lock.json',dict(members=[dict(path=str(p),sha256=file_sha(p)) for p in paths]))
    write_new(root/'prepared.json',dict(source=lock['code_commit'],main_cells=12,qualification='NOT_RUN_USER_DISABLED',
        project_GPU_cap=2,serial_GPU_lane=1,storage_min_free_bytes=32*1024**3,
        storage_plan='Per-cell W20 final-only archive verification and reclaim before next cell; no qualification CP',
        total_final_bytes=47045410816,max_single_payload_bytes=8543797248,
        concurrent_atomic_bytes=17087594496,raw_and_error_reserve_bytes=17179869184,
        planned_checkpoint_consumers=['writer/evaluation in same GPU job'],
        archive_stage='CPU-only storage consumer, no model/resume/evaluation',job_ids=[]))

def inspect(job,name,script,gpus,dependency):
    text=cmd(['scontrol','show','job',job,'-o']);fields=dict(x.split('=',1) for x in text.split() if '=' in x)
    expected=dict(JobId=job,JobName=name,JobState='PENDING',Requeue='0',Command=str(script),ReqNodeList='server4',
                  **{'CPUs/Task':'8' if gpus else '2'},MinMemoryNode='58G' if gpus else '4G')
    for k,v in expected.items():check(fields.get(k)==v,'HELD_INSPECT_'+k)
    check(fields['UserId'].startswith('janghj('),'OWNER_MISMATCH')
    check(fields.get('Reason')=='JobHeldUser','NOT_USER_HELD')
    check(dependency in fields.get('Dependency',''),'DEPENDENCY_MISMATCH')
    check(('gres/gpu=1' in fields.get('ReqTRES','')) if gpus else ('gres/gpu' not in fields.get('ReqTRES','')),'GPU_RESOURCE_MISMATCH')
    check(fields.get('ReqTRES','').split(',')[0]==('cpu=8' if gpus else 'cpu=2'),'REQUESTED_CPU_MISMATCH')
    return text

def submit(root,frontier,continue_held=False):
    verify(root,gpu_space=True);check((root/'prepared.json').exists(),'PREPARED_MISSING')
    existing=[read(p) for p in (root/'registrations').glob('*.json')]
    check(not (root/'registration-started.json').exists() or continue_held,'NO_DUPLICATE_OR_AUTOMATIC_SUBMISSION_RETRY')
    check(not (root/'released.json').exists(),'ALREADY_RELEASED')
    existing_by_key={(x['logical_main_row'],x['kind']):x for x in existing}
    check(len(existing_by_key)==len(existing),'DUPLICATE_REGISTRATION_RECEIPTS')
    queue=cmd(['squeue','-u','janghj','-h','-o','%i|%j|%T|%N|%b|%E'])
    owned=[]
    for line in cmd(['scontrol','show','job','-o']).splitlines():
        fields=dict(x.split('=',1) for x in line.split() if '=' in x)
        if fields.get('UserId','').startswith('janghj(') and (fields.get('ReqNodeList')=='server4' or fields.get('NodeList')=='server4'):
            if fields.get('JobState') in ('RUNNING','COMPLETING','CONFIGURING','PENDING'):owned.append(fields)
    check(all(x['JobId'] in {frontier,*[j['job_id'] for j in existing]} for x in owned),'FRESH_QUEUE_RECONCILIATION_REQUIRED')
    tuning=cmd(['scontrol','show','job',frontier,'-o'])
    check('JobName=qwen-price-tuning-sweep ' in tuning and 'UserId=janghj(' in tuning,'FRONTIER_IDENTITY')
    check('gres/gpu=2' in tuning,'FRONTIER_RESOURCE_RECHECK')
    record=dict(at=now(),queue=queue,own_server4=owned,frontier=tuning,cap=2)
    if continue_held:write_new(root/'registration-control-reconciliation.json',record)
    else:write_new(root/'registration-started.json',record)
    jobs=[];dependency='afterany:'+frontier
    for row in rows():
        logical=row['logical_main_row']
        for kind in ('gpu','archive'):
            gpu=kind=='gpu';name='qwen-'+row['config']['dataset']+'-'+row['config']['method'].lower()+('-archive' if not gpu else '')
            script=root/'scripts'/f'{logical}-{kind}.sh'
            argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4',
                  '--nodes=1','--ntasks=1','--cpus-per-task='+('8' if gpu else '2'),
                  '--mem='+('59392M' if gpu else '4096M'),'--time='+('48:00:00' if gpu else '04:00:00'),
                  '--export=NONE','--no-requeue','--job-name='+name,'--chdir='+str(root),
                  '--output='+str(root/'logs'/f'{logical}-{kind}-%j.out'),
                  '--error='+str(root/'logs'/f'{logical}-{kind}-%j.err'),'--dependency='+dependency]
            if gpu:argv+=['--gres=gpu:rtx_pro_6000:1']
            argv+=['/bin/bash',str(script)]
            # sbatch requires the script itself; its shebang provides bash.
            argv=argv[:-2]+[str(script)]
            item=existing_by_key.get((logical,kind))
            if item:
                check(item['argv']==argv and item['dependency']==dependency,'EXISTING_JOB_CONTRACT_CHANGED')
                job=item['job_id']
            else:
                submitted=now();response=cmd(argv);job=response.split(';')[0];check(job.isdigit(),'SBATCH_NO_REAL_ID')
                item=dict(logical_main_row=logical,method=row['config']['method'],dataset=row['config']['dataset'],
                      kind=kind,job_id=job,name=name,dependency=dependency,argv=argv,submitted_at=submitted,
                      source=read(root/'source-lock.json')['code_commit'],config_sha256=row['config']['config_sha256'])
                write_new(root/'registrations'/f'{job}.json',item)
                if gpu:archive.registered(root,logical,job,submitted)
            jobs.append(item)
            receipt=inspect(job,name,script,1 if gpu else 0,dependency)
            write_new(root/'inspections'/f'{job}.json',dict(job_id=job,raw=receipt))
            dependency='afterok:'+job
    write_new(root/'submission.json',dict(status='ALL_HELD_INSPECTED',jobs=jobs,qualification='NOT_RUN_USER_DISABLED'))
    for item in jobs:
        inspect(item['job_id'],item['name'],root/'scripts'/f"{item['logical_main_row']}-{item['kind']}.sh",1 if item['kind']=='gpu' else 0,item['dependency'])
    for item in reversed(jobs):
        cmd(['scontrol','release',item['job_id']])
        write_new(root/'releases'/f"{item['job_id']}.json",dict(job_id=item['job_id'],released_at=now()))
    snapshot=cmd(['squeue','-j',','.join(x['job_id'] for x in jobs),'-h','-o','%i|%j|%T|%E'])
    write_new(root/'released.json',dict(status='RELEASED',at=now(),jobs=jobs,snapshot=snapshot))
    print(json.dumps(dict(jobs=jobs,snapshot=snapshot)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','verify','submit']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--gpu-space',action='store_true')
    p.add_argument('--continue-held',action='store_true',help='explicit operator reconciliation, never automatic')
    p.add_argument('--preparation',type=Path);p.add_argument('--frontier');args=p.parse_args()
    if args.command=='prepare':prepare(args.root,args.preparation)
    elif args.command=='verify':verify(args.root,args.gpu_space)
    else:submit(args.root,args.frontier,args.continue_held)
