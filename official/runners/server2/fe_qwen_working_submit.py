"""One deliberate four-arm registration pass; task-only USER cap exception."""
import argparse,getpass,os,re,shlex
from pathlib import Path
from official.experiments.prepare import read,write_new,digest
from official.runners.server1.common import member,verify
from official.runners.server1 import submit as shared
from official.runners.server2 import submit as own
ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/fe-qwen-working-20261012')
NONCE='USER-FE-QWEN-WORKING-CURRENT-Z-20261012-R1'
def register(attempt):
    attempt=Path(attempt).resolve();assert attempt.is_relative_to(ROOT)
    lockpath=attempt/'lock.json';lock=read(lockpath)
    assert lock['server']=='server2' and lock['cpus']==8
    assert not (attempt/'submission.json').exists(),'EXISTING_SUBMISSION_RECONCILE_NO_DUPLICATE'
    assert not list(attempt.glob('submitted-*.json')),'PARTIAL_SUBMISSION_RECONCILE_NO_RETRY'
    configs=[read(verify(m)) for m in lock['configs']]
    assert sorted(c['L2'] for c in configs)==[0,.1,1,10]
    assert all(c['instruction']==NONCE and c['server']=='server2' for c in configs)
    source=lock['source_commit'];own.command(['git','merge-base','--is-ancestor',source,'origin/main'])
    for m in lock['source_members']+lock['author_members']:verify(m)
    before=own.inventory();assert not before['ambiguous']
    # This initial plan was made against zero owned admissions. Never guess new
    # resource edges if another owner action changes that inventory concurrently.
    assert not before['project'],'FRONTIER_CHANGED_REQUIRES_EXACT_RESOURCE_PLAN'
    node=shared.metadata(own.command(['scontrol','show','node','server2','-o']))
    qos=own.command(['sacctmgr','-nP','show','qos','lab_gpu_s2','format=Name,MaxTRESPU,MaxTRESPJ,MaxWall'])
    assert 'gres/gpu=4' in qos
    cap=Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    caprows=[s.split('\t') for s in cap.read_text().splitlines() if s.startswith('server2\t')]
    assert len(caprows)==1 and int(caprows[0][3])>=60416
    policy=own.command(['python3','scripts/slurm_memory_policy.py','request','--server','server2',
        '--gpus','1','--mem','60416M','--local-limit-mib-per-gpu',caprows[0][3]])
    assert int(node['RealMemory'])>=4*60416 and int(node['CPUTot'])>=4*8
    stat=os.statvfs(attempt);required=max(c['storage_required_bytes'] for c in configs)
    assert stat.f_bavail*stat.f_frsize>=required and stat.f_favail>=10000,'STORAGE_ADMISSION'
    write_new(attempt/'server2-admission.json',dict(instruction=NONCE,task_only_cap_exception=4,
        global_cap_unchanged=member(cap),inventory=before,node=node,qos=qos,memory_policy=policy,
        free_bytes=stat.f_bavail*stat.f_frsize,required_bytes=required,resources=dict(GPU=1,CPU=8,memory_MiB=60416,wall_hours=48)))
    (attempt/'scripts').mkdir();(attempt/'logs').mkdir()
    jobs={};held={}
    for cm,c in zip(lock['configs'],configs):
        key=Path(cm['path']).stem;name='s2-fe-qwen-current-z-'+key
        argv=[lock['python'],'-B','-u','-m','official.diagnostics.fe_qwen_current.launch',cm['path'],str(lockpath)]
        env=dict(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',WANDB_CONSOLE='off',WANDB_SAVE_CODE='false')
        text='#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
        text+='cd '+shlex.quote(lock['source_directory'])+'\nexec '+shlex.join(argv)+'\n'
        script=attempt/'scripts'/f'{key}.sh'
        with script.open('x') as f:f.write(text)
        cmd=['sbatch','--parsable','--hold','--export=NONE','--no-requeue','--partition=gpu',
            '--qos=lab_gpu_s2','--nodelist=server2','--nodes=1','--ntasks=1','--cpus-per-task=8',
            '--gres=gpu:a6000:1','--mem=60416M','--time=2-00:00:00','--job-name='+name,
            '--chdir='+lock['source_directory'],'--output='+str(attempt/'logs'/f'{key}-%j.out'),
            '--error='+str(attempt/'logs'/f'{key}-%j.err'),str(script)]
        response=own.command(cmd);jid=response.split(';')[0];assert re.fullmatch(r'\d+',jid)
        job=dict(job_id=jid,name=name,config=cm,config_sha256=c['config_sha256'],source=source,
            argv=argv,script=member(script),dependency=[],output=c['output'],L2=c['L2'])
        write_new(attempt/f'submitted-{key}.json',dict(job,command=cmd,response=response));jobs[key]=job
        d=shared.metadata(own.command(['scontrol','show','job',jid,'-o']))
        assert d['UserId'].split('(')[0]==getpass.getuser() and d['JobState']=='PENDING' and d['Reason']=='JobHeldUser'
        assert d['Command']==str(script) and d['WorkDir']==lock['source_directory'] and d['JobName']==name
        assert d['ReqNodeList']=='server2' and d['QOS']=='lab_gpu_s2' and d['Requeue']=='0'
        assert shared.gpu_count(d['ReqTRES'])==1 and shared.gpu_count(d.get('AllocTRES',''))==0
        assert shared.requested_cpu_matches(d,8) and shared.memory_MiB(d['MinMemoryNode'])==60416
        assert shared.seconds(d['TimeLimit'])==48*3600 and not shared.dependencies(d.get('Dependency'))
        assert own.command(['scontrol','write','batch_script',jid,'-']).strip()==text.strip()
        held[key]=d
    fresh=own.inventory();assert {r['job'] for r in fresh['project']}=={j['job_id'] for j in jobs.values()}
    write_new(attempt/'held-inspection.json',dict(jobs=held,lock=member(lockpath),actual_GPU_PASS=False,
        qualification='NOT_RUN_USER_DISABLED',task_only_DAG_width=4,other_tasks_unchanged=True))
    for j in jobs.values():own.command(['scontrol','release',j['job_id']])
    snapshot={k:shared.metadata(own.command(['scontrol','show','job',j['job_id'],'-o'])) for k,j in jobs.items()}
    result=dict(instruction=NONCE,jobs=jobs,initial=snapshot,source=source,released=True,
        online='NOT_OBSERVED_BEFORE_STARTUP',GPU_completion=False)
    write_new(attempt/'submission.json',result);print({k:(j['job_id'],snapshot[k]['JobState']) for k,j in jobs.items()})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();register(a.attempt)
