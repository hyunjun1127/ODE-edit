"""CPU input binding and exact held submission; no scheduler polling loop."""
import argparse
import getpass
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from .gptj_server2 import NONCE, TASK, REVISION, specifications, read
from project.run_scripts.jlz_realization.common import require, sha, member, digest, write

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local/base-model-gptj-w0')
MODEL = Path('/mnt/raid5/janghj/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots')/REVISION
DATA = Path('/mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1')
ENVELOPE = 'messages/head/2026-10-07-base-model-w0-server2.json'
SOURCES = ['project/run_scripts/'+d for d in ('base_model_eval','jlz_price_gptj','jlz_interference_l1',
    'jlz_realization','jlz_shared_budget','jlz_pilot','experiment_tracking')]
SOURCES += ['scripts/fixed_counterfact.py', ENVELOPE, 'control/wandb-policy.json',
            'control/wandb-method-metric-schema.json', 'control/gpu-concurrency-policy.tsv', 'servers/slurm-memory-policy.tsv']


def command(argv):
    p = subprocess.run(argv, text=True, capture_output=True)
    require(p.returncode==0, 'COMMAND_FAILED:'+shlex.join(argv)+':'+p.stderr[:1000])
    return p.stdout.strip()


def prepare():
    import torch, transformers
    from transformers import AutoTokenizer
    from scripts.fixed_counterfact import load_prefix
    from project.run_scripts.jlz_realization.inputs import CounterFactAdapter
    from project.run_scripts.experiment_tracking.schema import load_env
    out=LOCAL/'preparation-r1'
    require(not out.exists(), 'CREATE_ONCE_PREPARATION')
    require(sha(ROOT/ENVELOPE)=='59dc6f104e9333f8e7058fbcb9ed6d70a46e389ac5921ade004f52a85f773c56', 'AUTHORITY')
    require(not torch.cuda.is_initialized(), 'CPU_ONLY')
    require(torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.57.1', 'RUNTIME')
    require(MODEL.is_dir() and (MODEL/'pytorch_model.bin').is_file(), 'MODEL_CACHE_MISSING_NO_DOWNLOAD')
    assets=[]
    for path in sorted(MODEL.iterdir()):
        require(path.is_file() and path.resolve().is_relative_to(MODEL.parents[1]/'blobs'), 'HF_ASSET_PATH')
        row=member(path); row['snapshot_path']=str(path); assets.append(row)
    weight=next(r for r in assets if r['snapshot_path'].endswith('/pytorch_model.bin'))
    require(weight['sha256']==Path(weight['path']).name, 'HF_LFS_CONTENT_HASH')
    modules=['torch','transformers','tokenizers','transformers.models.gptj.modeling_gptj',
             'transformers.models.gpt2.tokenization_gpt2','transformers.models.gpt2.tokenization_gpt2_fast',
             'transformers.modeling_utils','transformers.masking_utils']
    runtime=dict(python=sys.executable,torch=torch.__version__,transformers=transformers.__version__,
                 members=[dict(module=n,**member(importlib.import_module(n).__file__)) for n in modules])
    records=load_prefix(DATA,2000)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    tok.pad_token=tok.eos_token;tok.padding_side='right'
    refs=specifications(CounterFactAdapter(tok,[]),records)
    require(max(len(p)+len(t)-1 for r in refs for p,t in r['tokens'].values())<=2048, 'NO_TRUNCATION')
    env='/mnt/raid5/janghj/ODE-edit/servers/local/wandb.env';load_env(env)
    require(shutil.disk_usage(LOCAL).free>8*1024**3, 'DISK_RESERVE')
    out.mkdir();write(out/'observer-identity.json',refs)
    config=dict(instruction_id=NONCE,task_id=TASK,model=str(MODEL),revision=REVISION,
        dataset=str(DATA),ordered_ids=digest([r['case_id'] for r in records]),
        record_root=digest([digest(r) for r in records]),observer_manifest=member(out/'observer-identity.json'),
        assets=assets,runtime=runtime,seed=20261002,microbatch=2,tracking_env=env,
        tracking_env_identity=member(env),reserve_bytes=2*1024**3,
        resources=dict(cpu=6,gpu=1,host_mib=60416,wall='04:00:00',collector_cpu=6,
            collector_host_mib=24576,collector_wall='04:00:00',project_cap=2,task_cap=1,
            default_cpu8_adjustment='server2 job_submit maximum6 CPUs per1GPU; preserved scientific scope',
            host_plan='FP32 model ~22.55GiB; low_cpu_mem_usage loader, host ceiling59GiB; no C0/P/H; row/token metadata <1GiB',
            gpu_plan='FP32 model ~22.55GiB; physicalMB2 full sequence hidden + selected full-vocab head; actual peak recorded',
            wall_is_eta=False),save_checkpoints=False,exact_resume='NOT_AVAILABLE',
        native_contexts='NOT_USED_BY_RPN_EVALUATOR; no generated training contexts',
        input_source=member(DATA/'counterfact.json'))
    write(out/'config.json',config)
    print(json.dumps(dict(config=str(out/'config.json'),rows=len(refs),model_sha=weight['sha256'],new_GPU=0)))


def inventory():
    raw=command(['squeue','-h','-r','-w','server2','-u',getpass.getuser(),'-o','%i|%j|%T|%b|%R'])
    rows=[]
    for line in raw.splitlines():
        job,name,status,gres,reason=line.split('|',4)
        d=command(['scontrol','show','job',job,'--oneliner'])
        req=re.search(r'\bReqTRES=([^ ]+)',d);gpu=re.search(r'gres/gpu=(\d+)',req[1]) if req else None
        if gpu:rows.append(dict(job=job,name=name,state=status,gpus=int(gpu[1]),reason=reason,detail=d))
        require(name!=TASK, 'DUPLICATE_TASK_JOB')
    return rows


def launcher(a,role,c):
    env=dict(PYTHONPATH=str(a/'source'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
        OPENBLAS_NUM_THREADS='6',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONHASHSEED='20261002')
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    return '#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+\
        'cd '+shlex.quote(str(a/'source'))+'\nexec '+shlex.join([c['runtime']['python'],'-u','-m',
        'project.run_scripts.base_model_eval.gptj_server2','collect' if role=='collector' else 'run','--attempt',str(a)])+'\n'


def submit():
    a=LOCAL/'attempt-r1';c=read(LOCAL/'preparation-r1/config.json')
    require(not a.exists(),'CREATE_ONCE_ATTEMPT')
    require(not command(['git','status','--porcelain','--',*SOURCES]),'UNCOMMITTED_SOURCE')
    tests=read(LOCAL/'cpu-tests.json');require(tests['passed'],'CPU_CHECK')
    for p,h in tests['source_sha256'].items():require(sha(ROOT/p)==h,'CPU_TEST_SOURCE_CHANGED')
    require(not command(['squeue','-h','-u',getpass.getuser(),'--name='+TASK,'-o','%i']), 'DUPLICATE_TASK')
    before=inventory()
    tracked=int(next(l for l in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if l.startswith('server2\t')).split('\t')[1])
    local=int(next(l for l in Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if l.startswith('server2\t')).split('\t')[2])
    cap=min(2,tracked,local);require(cap>=1,'CAP_DISABLED')
    barrier=[r['job'] for r in before] if sum(r['gpus'] for r in before)+1>cap else []
    node=command(['scontrol','show','node','server2']);partition=command(['scontrol','show','partition','gpu'])
    qos=command(['sacctmgr','-n','-P','show','qos','lab_gpu_s2','format=Name,MaxWall,MaxTRESPerUser,MaxTRESPerJob'])
    hardware=command(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'])
    for item in c['assets']+c['runtime']['members']:
        s=Path(item['path']).stat();require((s.st_size,s.st_ino,s.st_mtime_ns)==(item['bytes'],item['inode'],item['mtime_ns']),'INPUT_CHANGED')
    require(shutil.disk_usage(LOCAL).free>8*1024**3,'DISK_RESERVE')
    commit=command(['git','rev-parse','HEAD']);a.mkdir();(a/'source').mkdir()
    archive=a/'source.tar';command(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCES])
    with tarfile.open(archive) as tar:
        require(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in tar.getmembers()),'SAFE_SOURCE')
        tar.extractall(a/'source',filter='data')
    write(a/'config.json',c)
    for role in ('main','collector'):
        with (a/(role+'.sh')).open('x') as f:f.write(launcher(a,role,c))
    lock=dict(instruction_id=NONCE,source_commit=commit,source_tree=command(['git','rev-parse','HEAD^{tree}']),
        config_sha256=sha(a/'config.json'),archive=member(archive),source_members=[dict(relative=str(p.relative_to(a/'source')),sha256=sha(p)) for p in sorted((a/'source').rglob('*')) if p.is_file()],
        resources=c['resources'],launchers=[member(a/(r+'.sh')) for r in ('main','collector')],noCP=True,owner=getpass.getuser(),session='01a0493a-074c-7f91-9a13-769116326fef')
    write(a/'execution.lock.json',lock);ids={};inspections=[]
    for role in ('main','collector'):
        dep='afterany:'+ids['main'] if role=='collector' else ('afterany:'+':'.join(barrier) if barrier else None)
        mem=24576 if role=='collector' else 60416
        argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s2','--nodelist=server2','--nodes=1','--ntasks=1',
            '--cpus-per-task=6','--mem='+str(mem)+'M','--time=04:00:00','--export=NONE','--no-requeue',
            '--job-name='+TASK,'--chdir='+str(a/'source'),'--output='+str(a/(role+'-%j.out')),'--error='+str(a/(role+'-%j.err'))]
        if role=='main':argv+=['--gres=gpu:1']
        if dep:argv+=['--dependency='+dep]
        argv+=[str(a/(role+'.sh'))]
        job=command(argv).split(';')[0];require(job.isdigit(),'JOB_ID');ids[role]=job
        write(a/('submitted-'+role+'.json'),dict(job=job,argv=argv,dependency=dep))
        d=command(['scontrol','show','job',job,'--oneliner'])
        for value in [f'JobId={job} ',f'JobName={TASK} ','UserId='+getpass.getuser()+'(','JobState=PENDING ','Reason=JobHeldUser ',
            'Requeue=0 ','CPUs/Task=6 ','ReqNodeList=server2 ','Partition=gpu ','QOS=lab_gpu_s2 ','TimeLimit=04:00:00 ',
            'Command='+str(a/(role+'.sh'))+' ','WorkDir='+str(a/'source')+' ']:require(value in d,'HELD:'+value)
        require(re.search(r'\bNumCPUs=6(?:-\d+)? ',d) and 'ReqTRES=cpu=6,' in d,'HELD_CPU')
        require(f'mem={mem}M' in d or f'mem={mem//1024}G' in d,'HELD_MEM')
        require(('gres/gpu' not in d) if role=='collector' else 'TresPerNode=gres/gpu:1' in d,'HELD_GPU')
        observed=re.search(r'\bDependency=([^ ]+)',d)[1]
        require(set(re.findall(r'(?:afterany:|:)(\d+)',observed))==(set(dep.split(':')[1:]) if dep else set()),'HELD_DEPENDENCY')
        require(not dep or observed.startswith('afterany:'),'HELD_DEPENDENCY_TYPE')
        line=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',d)
        require(line and shlex.split(line[1])==argv,'HELD_ARGV')
        require(command(['scontrol','write','batch_script',job,'-']).strip()==(a/(role+'.sh')).read_text().strip(),'HELD_SCRIPT')
        inspections.append(dict(job=job,role=role,detail=d,argv=argv))
    # Recheck resource admission only; exclude these exact newly held IDs.
    queue=command(['squeue','-h','-r','-w','server2','-u',getpass.getuser(),'-o','%i|%b'])
    other={l.split('|')[0] for l in queue.splitlines() if 'gpu' in l and l.split('|')[0] not in ids.values()}
    require(other<={r['job'] for r in before},'ADMISSION_RACE_KEEP_HELD')
    require(barrier or sum(r['gpus'] for r in before if r['job'] in other)+1<=cap,'CAP_KEEP_HELD')
    write(a/'held-inspection.json',dict(jobs=inspections,before=before,cap=cap,external_dependencies=barrier,node=node,partition=partition,qos=qos,hardware=hardware))
    for role in ('collector','main'):
        write(a/('released-'+role+'.json'),dict(job=ids[role],result=command(['scontrol','release',ids[role]]),released=True))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%b|%N|%r'])
    receipt=dict(nonce=NONCE,status='SUBMISSION_HANDOFF',jobs=ids,source=commit,config_sha256=lock['config_sha256'],lock=member(a/'execution.lock.json'),
        dependencies=barrier,initial_snapshot=snapshot,resources=c['resources'],WandB='STARTUP_IN_RUNNER_NOT_OBSERVED',
        actual_evaluation='NOT_OBSERVED',monitoring_active=False,automatic_retry=False)
    write(a/'submission.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','submit'));args=p.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    (prepare if args.mode=='prepare' else submit)()
