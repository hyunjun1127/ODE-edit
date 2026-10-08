"""Single explicit-user registration of frozen Qwen science + CPU operations."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

TASK='qwen-ours-m1-2k-20261008'
NONCE='USER-SH4-QWEN-OURS-M1-2K-LIVE-WANDB-20261008-R1'
ROOT=Path('/data/janghj/ODE-edit/local')/TASK
ATTEMPT=ROOT/'preparation-v1'
OPS=ROOT/'execution-v1'
EXPECTED={'config.json':'3062c9fca73388673b25b9f9a377ead373ac8fe63472feecac44cf84ea883361',
    'execution.lock.json':'83afecec61e9d51b6cdecee6d2ac7b34bdb12eab96dfbee176eefc0aa96c6d58',
    'run.sh':'f4aabef4d163cc15aa86386084f3ae4608a381f6bd79b7fa4e7ca5139dc80490',
    'source.tar.gz':'11614d4ce52f888eafc3b01bc92945dd4216fa64bfe6ea1a4d802e203c87161b'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def job(j):
    text=subprocess.check_output(['scontrol','show','job',str(j),'-o'],text=True)
    keys=('JobId','JobName','UserId','JobState','Priority','Dependency','ReqNodeList',
          'NodeList','Command','WorkDir','NumCPUs','MinMemoryNode','TimeLimit','Requeue','TresPerNode')
    r={k:(m.group(1) if (m:=re.search(r'(?:^| )'+k+r'=(\S*)',text)) else None) for k in keys}
    r['raw']=text;return r

def main():
    target=OPS/'submission.json'
    if target.exists() or (ATTEMPT/'QWEN_M1_CAP075').exists():raise RuntimeError('DUPLICATE_ATTEMPT_OR_SUBMISSION')
    for name,expected in EXPECTED.items():
        if sha(ATTEMPT/name)!=expected:raise RuntimeError('FROZEN_SHA_'+name)
    env=dict(os.environ,QWEN_OURS_SOURCE='9ecdf8342c52ee7e6a85a6ed741fc4f5468bc1ef',
        PYTHONPATH=str(ATTEMPT/'source'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    subprocess.run(['/data/janghj/EasyEdit/.venv/bin/python','-c',
        'from pathlib import Path; from project.run_scripts.qwen_ours_m1.run import validate; '
        'from project.run_scripts.jlz_interference_l1.cap_tracking import contract_ready; '
        'validate(Path('+repr(str(ATTEMPT))+')); contract_ready()'],env=env,cwd=ATTEMPT/'source',check=True)
    free=shutil.disk_usage(ROOT).free
    if free<24536416256:raise RuntimeError('STORAGE_RESERVE_BLOCK')
    caps=Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines()
    rows=[r.split('\t') for r in caps if r.startswith('server4\t')]
    if len(rows)!=1 or int(rows[0][2])!=3 or int(rows[0][3])!=60416:raise RuntimeError('LOCAL_CAP_BINDING')
    before=[];active=0
    for j in subprocess.check_output(['squeue','-h','-u','janghj','-o','%i'],text=True).split():
        r=job(j)
        if r['ReqNodeList']!='server4' and r['NodeList']!='server4':continue
        if r['UserId']!='janghj(1025)':raise RuntimeError('OWNER')
        before.append(r)
        if TASK in r['JobName'] or str(ROOT) in (r['Command'] or ''):raise RuntimeError('DUPLICATE_QWEN_JOB')
        if j in [str(x) for x in range(60917,60924)]:
            if r['JobState']!='PENDING' or r['Priority']!='0':raise RuntimeError('BASELINE_NOT_HELD')
            continue
        if j=='61421' and r['TresPerNode'] is None:continue
        if j in ('61418','61420') and r['Command']==str(Path('/data/janghj/ODE-edit/local/price-ridge-m1-m3-2k-20261008/attempt-generation-w20-only-r1')/({'61418':'LLAMA_REPRO','61420':'GPT2XL_M1_M2'}[j]+'.sh')):
            if r['TresPerNode']!='gres/gpu:1':raise RuntimeError('PREDECESSOR_GPU_IDENTITY')
            active+=1;continue
        raise RuntimeError('UNKNOWN_OWN_SERVER4_ADMISSION_'+j)
    if active+1>3:raise RuntimeError('PROJECT_CAP_BLOCK')
    OPS.mkdir(exist_ok=False)
    for name in ('supervise.py','monitor.py'):
        shutil.copyfile(Path(__file__).with_name(name),OPS/name)
    launcher=OPS/'launch.sh'
    launcher.write_text('#!/bin/bash\nset -euo pipefail\nexport PYTHONDONTWRITEBYTECODE=1\nexec /usr/bin/python3 '+
        str(OPS/'supervise.py')+' --attempt '+str(ATTEMPT)+' --ops '+str(OPS)+'\n')
    subprocess.run(['bash','-n',str(launcher)],check=True)
    data=dict(nonce=NONCE,task_id=TASK,frozen_source='9ecdf8342c52ee7e6a85a6ed741fc4f5468bc1ef',
        operations_source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        frozen_sha=EXPECTED,operations_sha={p.name:sha(p) for p in (OPS/'supervise.py',OPS/'monitor.py',launcher)},
        before=before,project_cap=3,existing_lanes=active,free_bytes=free,reserve_bytes=24536416256,
        stage='REGISTERING_HELD',job_id=None,other_jobs_mutation=False)
    def persist():target.write_text(json.dumps(data,indent=2)+'\n')
    persist()
    argv=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s4','--nodelist=server4',
        '--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=59392M','--gres=gpu:rtx_pro_6000:1',
        '--time=24:00:00','--export=NONE','--no-requeue','--job-name='+TASK,
        '--chdir='+str(ATTEMPT),'--output='+str(OPS/'qwen-%j.out'),'--error='+str(OPS/'qwen-%j.err'),str(launcher)]
    p=subprocess.run(argv,capture_output=True,text=True);data['submit_exit']=p.returncode
    if p.returncode or not re.fullmatch(r'[0-9]+(?:;[^\s]+)?\s*',p.stdout):
        data.update(stage='SUBMISSION_BLOCKED',stdout=p.stdout,stderr=p.stderr);persist()
        raise RuntimeError('SUBMISSION_BLOCKED_NO_AUTORETRY')
    j=p.stdout.strip().split(';')[0];data['job_id']=j;persist();r=job(j);data['held']=r;persist()
    assert r['UserId']=='janghj(1025)' and r['JobName']==TASK and r['ReqNodeList']=='server4'
    assert r['JobState']=='PENDING' and r['Priority']=='0' and r['Command']==str(launcher)
    assert r['WorkDir']==str(ATTEMPT) and r['NumCPUs'] in ('8','8-14')
    assert r['MinMemoryNode']=='58G' and r['TimeLimit'] in ('1-00:00:00','24:00:00')
    assert r['TresPerNode']=='gres/gpu:rtx_pro_6000:1' and r['Requeue']=='0'
    assert '--export=NONE' in r['raw'] and '--no-requeue' in r['raw'] and r['Dependency']=='(null)'
    for name,expected in EXPECTED.items():assert sha(ATTEMPT/name)==expected
    subprocess.run(['scontrol','release',j],check=True)
    data.update(stage='RELEASED',initial=job(j));persist()
    print(json.dumps({'job_id':j,'stage':data['stage'],'initial_state':data['initial']['JobState']}))

if __name__=='__main__':main()
