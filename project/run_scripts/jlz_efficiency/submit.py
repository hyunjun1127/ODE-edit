"""Single admission + held exact inspection + release. No monitor loop."""
import argparse
import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess
from .bind import once
from .measurement import sha

def command(args):
    r=subprocess.run(args,text=True,capture_output=True)
    if r.returncode:raise RuntimeError(str(args)+' '+r.stderr)
    return r.stdout.strip()

def capacity(exclude=()):
    raw=command(['squeue','--array','-h','-w','devbox','-o','%i|%u|%j|%T'])
    rows=[]
    for line in raw.splitlines():
        job,user,name,state=line.split('|')
        if job in exclude:continue
        if user!='janghj' and not any(fnmatch.fnmatch(name,p) for p in ('odeedit_*','motivation_*','session01_*')):continue
        details=command(['scontrol','show','job',job,'--oneliner']);m=re.search(r'\bReqTRES=([^ ]+)',details)
        if not m:raise RuntimeError('UNRESOLVED_REQUEST '+job)
        entries=dict(x.split('=',1) for x in m[1].split(','));aggregate=entries.get('gres/gpu')
        typed=[int(v) for k,v in entries.items() if k.startswith('gres/gpu:')]
        n=int(aggregate) if aggregate is not None else sum(typed)
        if aggregate is not None and typed and sum(typed)!=n:raise RuntimeError('GPU_TRES_CONFLICT '+job)
        rows.append(dict(job=job,state=state,gpus=n))
    total=sum(r['gpus'] for r in rows)
    if total+1>2:raise RuntimeError('WAITING_FOR_ISOLATED_RESOURCE '+str(rows))
    return dict(active_plus_admitted=total,new=1,cap=2,records=rows)

def inspect(job,run,wt,name,gpu,parent=None):
    txt=command(['scontrol','show','job',job,'--oneliner'])
    for text in (f'JobId={job} ',f'JobName={name} ','UserId=janghj(','JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ','ReqNodeList=devbox ',f'WorkDir={wt} ',f'NumCPUs={8 if gpu else 4} '):assert text in txt,text
    assert f'Command={run/("gpu.sh" if gpu else "collector.sh")}' in txt
    assert ('TimeLimit=12:00:00' if gpu else 'TimeLimit=02:00:00') in txt
    if gpu:assert 'Dependency=(null)' in txt and 'TresPerNode=gres/gpu:1' in txt and ('mem=128G' in txt or 'mem=131072M' in txt)
    else:assert f'afterany:{parent}' in txt and ('mem=16G' in txt or 'mem=16384M' in txt)
    return txt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    assert not (run/'submitted-gpu.json').exists()
    lock=json.loads((run/'execution.lock.json').read_text());wt=Path(lock['worktree'])
    assert command(['git','-C',str(wt),'rev-parse','HEAD'])==lock['source'],'SOURCE_HEAD_CHANGED'
    assert not command(['git','-C',str(wt),'status','--porcelain']),'SOURCE_DIRTY'
    env=dict(os.environ,AGENT_GPU_CAPS_FILE='/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    check=subprocess.run(['bash',str(wt/'scripts/check-slurm-resource-cap.sh'),'server1','1','131072M'],env=env,text=True,capture_output=True)
    if check.returncode:raise RuntimeError(check.stdout+check.stderr)
    admission=capacity();common=['sbatch','--parsable','--hold','--partition=gpu','--qos=lab_gpu_s1','--nodelist=devbox','--nodes=1','--ntasks=1','--export=NONE','--no-requeue',f'--chdir={wt}']
    ga=common+['--job-name=odeedit_jlz_efficiency_s1','--gres=gpu:1','--cpus-per-task=8','--mem=131072M','--time=12:00:00',f'--output={run}/gpu-%j.out',f'--error={run}/gpu-%j.err',str(run/'gpu.sh')]
    gpu=command(ga).split(';')[0];assert gpu.isdigit();once(run/'submitted-gpu.json',dict(job=gpu,args=ga,status='HELD'))
    ca=common+['--job-name=odeedit_jlz_eff_collect_s1','--cpus-per-task=4','--mem=16384M','--time=02:00:00',f'--dependency=afterany:{gpu}',f'--output={run}/collector-%j.out',f'--error={run}/collector-%j.err',str(run/'collector.sh'),gpu]
    cpu=command(ca).split(';')[0];assert cpu.isdigit();once(run/'submitted-collector.json',dict(job=cpu,args=ca,status='HELD'))
    receipt=dict(gpu_job=gpu,collector_job=cpu,gpu_args=ga,collector_args=ca,admission=admission,resource_helper=check.stdout,
        gpu_held=inspect(gpu,run,wt,'odeedit_jlz_efficiency_s1',True),collector_held=inspect(cpu,run,wt,'odeedit_jlz_eff_collect_s1',False,gpu),
        lock_sha256=sha(run/'execution.lock.json'),launcher_sha256={p.name:sha(p) for p in (run/'gpu.sh',run/'collector.sh')},export='NONE')
    receipt['prerelease_admission']=capacity((gpu,cpu));once(run/'held-inspection.json',receipt)
    command(['scontrol','release',cpu]);once(run/'collector-release.json',dict(job=cpu,released=True))
    command(['scontrol','release',gpu]);receipt.update(status='RELEASED',actual_initial='NOT_YET_OBSERVED',gpu_dependency=None,collector_dependency='afterany:'+gpu)
    once(run/'submission.json',receipt);print(json.dumps({k:receipt[k] for k in ('status','gpu_job','collector_job','gpu_dependency','collector_dependency','lock_sha256')}))

if __name__=='__main__':main()
