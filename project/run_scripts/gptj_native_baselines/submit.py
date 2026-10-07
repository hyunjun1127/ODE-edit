"""Two native GPU jobs plus afterany CPU collector, held-inspect-release once."""
import getpass
import json
import os
import re
import shlex
import shutil
import tarfile
from .common import *
from .prepare import stat_seal
from project.run_scripts.base_model_eval.gptj_server2_cohort_control import command

SOURCES=['project/run_scripts/'+n for n in ('gptj_native_baselines','base_model_eval','jlz_price_gptj',
    'jlz_interference_l1','jlz_realization','jlz_realized_writer_sequential','jlz_shared_budget','jlz_pilot','experiment_tracking')]
SOURCES+=['scripts/fixed_counterfact.py',ENVELOPE,'control/wandb-policy.json','control/wandb-method-metric-schema.json',
    'control/gpu-concurrency-policy.tsv','servers/slurm-memory-policy.tsv']

def launcher(a,role,c,source):
    env=dict(PYTHONPATH=str(a/'source'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',
        OPENBLAS_NUM_THREADS='6',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONHASHSEED='20261002',GPTJ_NATIVE_BASELINE_SOURCE_COMMIT=source)
    if role=='collector':env['CUDA_VISIBLE_DEVICES']=''
    module='project.run_scripts.gptj_native_baselines.'+('collect' if role=='collector' else 'run')
    argv=[c['runtime']['python'],'-u','-m',module,'--attempt',str(a)]
    if role!='collector':argv+=['--arm',role]
    return '#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())+'cd '+shlex.quote(str(a/'source'))+'\nexec '+shlex.join(argv)+'\n'

def inventory():
    rows=[]
    raw=command(['squeue','-h','-r','-w','server2','-u',getpass.getuser(),'-o','%i|%j|%T|%b|%R'])
    for line in raw.splitlines():
        job,name,status,gres,reason=line.split('|',4)
        require(not name.startswith(TASK),'DUPLICATE_TASK')
        d=command(['scontrol','show','job',job,'--oneliner'])
        req=re.search(r'\bReqTRES=([^ ]+)',d);gpu=re.search(r'gres/gpu=(\d+)',req[1]) if req else None
        if gpu:rows.append(dict(job=job,name=name,state=status,gpus=int(gpu[1]),reason=reason,detail=d))
    return rows

def submit():
    authority();c=json.loads((LOCAL/'preparation-r1/config-submission.json').read_text());a=LOCAL/'attempt-r1'
    require(not a.exists(),'CREATE_ONCE_ATTEMPT')
    require(not command(['git','status','--porcelain','--',*SOURCES]),'SOURCE_COMMITTED')
    tests=json.loads((LOCAL/'cpu-tests-r2.json').read_text());require(tests['passed'],'CPU')
    for p,h in tests['source_sha256'].items():require(sha(ROOT/p)==h,'CPU_SOURCE')
    require(sha(LOCAL/'preparation-r1/config-submission.json')==tests['config']['sha256'],'CPU_CONFIG')
    for row in c['assets']+c['runtime']['members']:stat_seal(row)
    for row in c['native']['closure']:verify(row)
    before=inventory()
    tracked=int(next(l for l in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if l.startswith('server2\t')).split('\t')[1])
    local=int(next(l for l in Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if l.startswith('server2\t')).split('\t')[2])
    cap=min(2,tracked,local);require(cap>=1,'CAP_DISABLED')
    frontier=[r['job'] for r in before]
    node=command(['scontrol','show','node','server2']);partition=command(['scontrol','show','partition','gpu'])
    qos=command(['sacctmgr','-n','-P','show','qos','lab_gpu_s2','format=Name,MaxWall,MaxTRESPerUser,MaxTRESPerJob'])
    hardware=command(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'])
    require(shutil.disk_usage(LOCAL).free>16*1024**3,'DISK_RESERVE_TWO_JOBS')
    source=command(['git','rev-parse','HEAD']);a.mkdir();(a/'source').mkdir()
    archive=a/'source.tar';command(['git','archive','--format=tar','--output='+str(archive),source,*SOURCES])
    with tarfile.open(archive) as t:
        require(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in t.getmembers()),'SAFE_ARCHIVE')
        t.extractall(a/'source',filter='data')
    write(a/'config.json',c)
    for role in (*ARMS,'collector'):
        with (a/(role+'.sh')).open('x') as f:f.write(launcher(a,role,c,source))
    lock=dict(instruction_id=NONCE,source_commit=source,source_tree=command(['git','rev-parse','HEAD^{tree}']),
        config_sha256=sha(a/'config.json'),source_members=[member(p) for p in sorted((a/'source').rglob('*')) if p.is_file()],
        runtime_sources=c['runtime']['members'],launchers=[member(a/(r+'.sh')) for r in (*ARMS,'collector')],
        archive=member(archive),tracking_env=member(c['tracking']['env_file']),owner=getpass.getuser(),
        noCP=True,resources=c['resources'],session='01a0493a-074c-7f91-9a13-769116326fef')
    write(a/'execution.lock.json',lock);ids={};inspections=[]
    for role in (*ARMS,'collector'):
        dependencies=list(ids.values()) if role=='collector' else list(frontier)
        if cap==1 and role==ARMS[1]:dependencies.append(ids[ARMS[0]])
        dep='afterany:'+':'.join(dependencies) if dependencies else None
        mem=24576 if role=='collector' else 59392;wall='04:00:00' if role=='collector' else '2-00:00:00'
        name=TASK+'-'+role
        argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s2','--nodelist=server2','--nodes=1','--ntasks=1',
            '--cpus-per-task=6','--mem='+str(mem)+'M','--time='+wall,'--export=NONE','--no-requeue','--job-name='+name,
            '--chdir='+str(a/'source'),'--output='+str(a/(role+'-%j.out')),'--error='+str(a/(role+'-%j.err'))]
        if role!='collector':argv+=['--gres=gpu:1']
        if dep:argv+=['--dependency='+dep]
        argv+=[str(a/(role+'.sh'))]
        job=command(argv).split(';')[0];require(job.isdigit(),'JOB_ID');ids[role]=job
        write(a/('submitted-'+role+'.json'),dict(job=job,argv=argv,dependency=dep))
        d=command(['scontrol','show','job',job,'--oneliner'])
        for value in [f'JobId={job} ',f'JobName={name} ','UserId='+getpass.getuser()+'(','JobState=PENDING ',
            'Reason=JobHeldUser ','Requeue=0 ','CPUs/Task=6 ','ReqNodeList=server2 ','Partition=gpu ','QOS=lab_gpu_s2 ',
            'TimeLimit='+wall+' ','Command='+str(a/(role+'.sh'))+' ','WorkDir='+str(a/'source')+' ']:require(value in d,'HELD:'+value)
        require('ReqTRES=cpu=6,' in d and (f'mem={mem}M' in d or f'mem={mem//1024}G' in d),'HELD_RESOURCE')
        require(('gres/gpu' not in d) if role=='collector' else 'TresPerNode=gres/gpu:1' in d,'HELD_GPU')
        observed=re.search(r'\bDependency=([^ ]+)',d)[1]
        require(set(re.findall(r'(?:afterany:|:)(\d+)',observed))==set(dependencies),'HELD_DEPENDENCY')
        require(not dep or observed.startswith('afterany:'),'DEPENDENCY_TYPE')
        line=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',d)
        require(line and shlex.split(line[1])==argv,'HELD_ARGV')
        require(command(['scontrol','write','batch_script',job,'-']).strip()==(a/(role+'.sh')).read_text().strip(),'HELD_SCRIPT')
        inspections.append(dict(role=role,job=job,detail=d,argv=argv))
    queue=command(['squeue','-h','-r','-w','server2','-u',getpass.getuser(),'-o','%i|%b'])
    other={l.split('|')[0] for l in queue.splitlines() if 'gpu' in l and l.split('|')[0] not in ids.values()}
    require(other<=set(frontier),'ADMISSION_RACE_KEEP_HELD')
    write(a/'held-inspection.json',dict(jobs=inspections,cap=cap,before=before,frontier=frontier,node=node,partition=partition,qos=qos,hardware=hardware,
        schedule='Both native lanes follow complete admitted GPU frontier; at most2 afterwards (cap1 serial)',source=source))
    for role in ('collector',*ARMS):
        write(a/('released-'+role+'.json'),dict(job=ids[role],released=True,result=command(['scontrol','release',ids[role]])))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%b|%N|%r'])
    receipt=dict(instruction_id=NONCE,task_id=TASK,status='SUBMISSION_HANDOFF',jobs=ids,source_commit=source,
        config_sha256=lock['config_sha256'],lock=member(a/'execution.lock.json'),dependencies=frontier,
        initial_snapshot=snapshot,resources=c['resources'],WandB='STARTUP_IN_RUNNER_NOT_OBSERVED',
        actual_first_write='NOT_OBSERVED',monitoring_active=False,automatic_retry=False)
    write(a/'submission.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':submit()
