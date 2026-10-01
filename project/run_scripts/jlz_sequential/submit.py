"""One create-once held/inspect/release submission; no monitor or rerun loop."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

def command(argv):
    r=subprocess.run(argv,text=True,capture_output=True)
    if r.returncode:raise RuntimeError(str(argv)+': '+r.stderr)
    return r.stdout.strip()

def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')

def capacity(exclude=()):
    # Admission-only resource view, including pending/configuring/completing.
    raw=command(['squeue','-h','-w','devbox','-o','%i|%u|%j|%T|%b'])
    counts=[]
    for line in raw.splitlines():
        job,user,name,state,gres=line.split('|')
        if job in exclude:continue
        if user!='janghj' and not any(fnmatch.fnmatch(name,p) for p in ('odeedit_*','motivation_*','session01_*')):continue
        if gres in ('N/A','(null)',''):
            details=command(['scontrol','show','job',job,'--oneliner'])
            match=re.search(r'\bReqTRES=([^ ]+)',details)
            if not match:raise RuntimeError('REQUEST_TRES_UNRESOLVED '+job)
            gpu_match=re.search(r'gres/gpu=(\d+)',match[1])
            n=int(gpu_match[1]) if gpu_match else 0
        else:
            match=re.search(r'gpu(?::[^:,]+)?:(\d+)',gres)
            if not match:raise RuntimeError('GPU_TRES_UNRESOLVED '+job)
            n=int(match[1])
        counts.append({'job':job,'state':state,'gpus':n})
    total=sum(r['gpus'] for r in counts)
    if total+1>2:raise RuntimeError('WAITING_FOR_ISOLATED_RESOURCE '+str(counts))
    return {'active_plus_admitted':total,'new':1,'cap':2,'records':counts}

def inspect(job,run,name,gpu,dependency=None):
    text=command(['scontrol','show','job',job,'--oneliner'])
    assert f'JobId={job} ' in text and f'JobName={name} ' in text and 'UserId=janghj(' in text
    assert 'JobState=PENDING ' in text and 'Reason=JobHeldUser ' in text
    assert 'Requeue=0 ' in text and 'NumCPUs=8 ' in text and 'ReqNodeList=devbox ' in text
    assert str(run/('gpu.sh' if gpu else 'collector.sh')) in text
    if gpu:
        assert 'TresPerNode=gres/gpu:1' in text and ('mem=128G' in text or 'mem=131072M' in text)
    else:
        assert f'afterany:{dependency}' in text and ('mem=8G' in text or 'mem=8192M' in text)
    return text

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    assert not (run/'submission.json').exists() and not (run/'submitted-gpu.json').exists()
    lock=json.loads((run/'execution.lock.json').read_text());wt=Path(lock['worktree'])
    assert lock['resource']['mem_MiB']==131072 and lock['resource']['project_cap']==2
    env=dict(os.environ,AGENT_GPU_CAPS_FILE='/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    c=subprocess.run(['bash',str(wt/'scripts/check-slurm-resource-cap.sh'),'server1','1','131072M'],env=env,text=True,capture_output=True)
    if c.returncode:raise RuntimeError(c.stdout+c.stderr)
    admission=capacity()
    common=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s1','--nodelist=devbox',
            '--nodes=1','--ntasks=1','--cpus-per-task=8','--export=NONE','--no-requeue',f'--chdir={wt}']
    gpu_args=common+['--job-name=odeedit_jlz_sequential_s1','--gres=gpu:1','--mem=131072M','--time=48:00:00',
                     f'--output={run}/gpu-%j.out',f'--error={run}/gpu-%j.err',str(run/'gpu.sh')]
    gpu=command(gpu_args).split(';')[0];assert gpu.isdigit()
    save(run/'submitted-gpu.json',{'job':gpu,'args':gpu_args,'status':'HELD_NOT_RELEASED'})
    cpu_args=common+['--job-name=odeedit_jlz_collect_s1','--mem=8192M','--time=02:00:00',f'--dependency=afterany:{gpu}',
                    f'--output={run}/collector-%j.out',f'--error={run}/collector-%j.err',str(run/'collector.sh'),gpu]
    cpu=command(cpu_args).split(';')[0];assert cpu.isdigit()
    receipt={'gpu_job':gpu,'collector_job':cpu,'gpu_args':gpu_args,'collector_args':cpu_args,
             'admission':admission,'helper':c.stdout,'gpu_held':inspect(gpu,run,'odeedit_jlz_sequential_s1',True),
             'collector_held':inspect(cpu,run,'odeedit_jlz_collect_s1',False,gpu),
             'lock_sha256':hashlib.sha256((run/'execution.lock.json').read_bytes()).hexdigest(),
             'launcher_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'gpu.sh',run/'collector.sh')}}
    receipt['prerelease_admission']=capacity(exclude=(gpu,cpu))
    save(run/'held-inspection.json',receipt)
    command(['scontrol','release',cpu]);command(['scontrol','release',gpu])
    receipt.update(status='RELEASED',actual_initial='NOT_YET_OBSERVED',dependency='afterany:'+gpu)
    save(run/'submission.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','gpu_job','collector_job','dependency','lock_sha256')}))

if __name__=='__main__':main()
