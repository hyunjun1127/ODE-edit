"""One deliberate three-job held registration; no old-job mutations or retries."""
import argparse
import getpass
import os
import re
import shlex
from pathlib import Path
from official.experiments.prepare import write_new,digest
from .common import read,member,verify
from . import submit as control
from .fe_history_prepare import INSTRUCTION,TASK,MODELS,LOCAL
from .fe_history_run import validate_config

def scheduling(existing,models,cap):
    rows=[dict(row) for row in existing['jobs']]
    answer={}
    for model in models:
        candidates=[[]]+[[x] for x in control.frontier(rows)]
        # Minimal lane edges when sufficient; conservative full frontier only if required.
        candidates.append(control.frontier(rows))
        for parents in candidates:
            item=dict(key=model,gpus=1,parents=parents)
            if control.graph_width(rows+[item])<=cap:
                answer[model]=parents;rows.append(item);break
        else:raise ValueError('CAP_NO_VALID_DAG')
    return answer,control.graph_width(rows)

def register(preparation,attempt):
    prep=read(preparation);assert prep['instruction']==INSTRUCTION and prep['models']==list(MODELS)
    attempt=Path(attempt).absolute();assert attempt.is_relative_to(LOCAL)
    if (attempt/'submission.json').exists():
        print(read(attempt/'submission.json'));return
    assert not attempt.exists(),'INCOMPLETE_ATTEMPT_RECONCILE_NO_BLIND_RETRY'
    # Search own attempt receipts before any sbatch; no duplicate nonce submission.
    for old in LOCAL.rglob('submission.json'):
        if read(old).get('instruction')==INSTRUCTION:
            raise ValueError('EXISTING_REGISTERED_ATTEMPT:'+str(old))
    for m in prep['configs']:validate_config(read(verify(m)))
    local_cap=Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    lines=[x.split('\t') for x in local_cap.read_text().splitlines() if x.startswith('server1\t')]
    assert len(lines)==1 and lines[0][1]=='devbox'
    cap=min(4,int(lines[0][2]));assert cap>0
    resources=dict(control.DEFAULT_RESOURCES,memory_MiB=98304)
    assert resources['memory_MiB']<=int(lines[0][3])<=183296
    existing=control.inventory()
    assert existing['allocated_gpus']<=cap and existing['admitted_DAG_width']<=cap
    assert not any(TASK in row['original'] or 'fe-history' in row['original'] for row in existing['jobs']), 'MATCHING_LIVE_TASK_RECONCILE'
    graph,width=scheduling(existing,list(MODELS),cap)
    source=control.command(['git','rev-parse','HEAD']); tree=control.command(['git','rev-parse','HEAD:official'])
    control.command(['git','merge-base','--is-ancestor',source,'origin/main'])
    assert not control.command(['git','status','--porcelain','--','official'])
    plan=dict(cap=cap,source=dict(main_commit=source,official_tree=tree),resources=resources)
    physical=control.resource_preflight(plan)
    disk=os.statvfs(LOCAL);assert disk.f_bavail*disk.f_frsize>256*(1<<30) and disk.f_favail>10000
    attempt.mkdir(parents=True);(attempt/'logs').mkdir();(attempt/'scripts').mkdir()
    write_new(attempt/'admission.json',dict(instruction=INSTRUCTION,existing=existing,cap=cap,local_cap=member(local_cap),
        direct_cap_authority='USER_DIRECT_SERVER1_CAP4',effective_graph_width=width,graph=graph,
        resources=resources,physical=physical,disk_available_bytes=disk.f_bavail*disk.f_frsize,
        disk_reserve_bytes=256*(1<<30),no_GPU_wait=True,no_old_job_mutation=True))
    frozen=control.freeze_source(plan,attempt)
    lock=dict(instruction=INSTRUCTION,source_commit=source,official_tree=tree,source_directory=frozen['directory'],
        source_members=frozen['members'],configs=prep['configs'],preparation=member(preparation),
        qualification='NOT_RUN_USER_DISABLED',main_actual_H0_matrix_guard=True)
    lock_path=attempt/'execution-lock.json';write_new(lock_path,lock)
    ids={};jobs={};held={};argvs={}
    for cm in prep['configs']:
        cfg=read(verify(cm));model=cfg['model'];key='cf-'+model+'-memit-fe-history'
        parents=[ids.get(p,p) for p in graph[model]]
        deps=[('afterany',p) for p in parents]
        argv=[control.DEFAULT_PYTHON,'-B','-u','-m','official.runners.server1.fe_history_run','--config',cm['path'],'--lock',str(lock_path)]
        env=dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
                 OFFICIAL_CODE_COMMIT=source,OFFICIAL_TREE_SHA256=tree,WANDB_CONSOLE='off',WANDB_SAVE_CODE='false')
        script=attempt/'scripts'/f'{model}.sh'
        text='#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
        text+='cd '+shlex.quote(frozen['directory'])+'\nexec '+shlex.join(argv)+'\n'
        with script.open('x') as stream:stream.write(text)
        job=dict(key=key,gpus=1,mode='chain',method='MEMIT_FE_HISTORY',dataset='cf',config=cm,output=cfg['output'],parents=[],external_resource_parents=parents)
        cmd,expected=control.sbatch_argv(plan,job,script,attempt,{})
        response=control.command(cmd);jobid=response.split(';')[0];assert re.fullmatch(r'\d+',jobid)
        ids[model]=jobid;jobs[model]=dict(job_id=jobid,name='official-s1-'+key,config=cm,script=member(script),dependency=expected,argv=argv)
        write_new(attempt/f'submitted-{model}.json',dict(response=response,command=cmd,**jobs[model]))
        detail=control.metadata(control.command(['scontrol','show','job',jobid,'--oneliner']))
        assert detail['UserId'].split('(')[0]==getpass.getuser() and detail['JobState']=='PENDING' and detail['Reason']=='JobHeldUser'
        assert detail['Command']==str(script) and detail['WorkDir']==frozen['directory'] and detail['JobName']=='official-s1-'+key
        assert detail['ReqNodeList']=='devbox' and detail['Requeue']=='0' and detail['QOS']==resources['qos'] and detail['Partition']==resources['partition']
        assert control.gpu_count(detail['ReqTRES'])==1 and control.gpu_count(detail.get('AllocTRES',''))==0
        assert control.requested_cpu_matches(detail,8) and control.memory_MiB(detail['MinMemoryNode'])==98304
        assert control.seconds(detail['TimeLimit'])==48*3600 and control.dependencies(detail.get('Dependency',''))==sorted(expected)
        assert control.command(['scontrol','write','batch_script',jobid,'-']).strip()==text.strip()
        held[model]=detail
    # Fresh queue proof after all held registration and before releasing any job.
    fresh=control.inventory();assert fresh['admitted_DAG_width']<=cap and fresh['allocated_gpus']<=cap
    write_new(attempt/'held-inspection.json',dict(jobs=held,fresh=fresh,source_lock=member(lock_path),GPU_qualification=False))
    for jobid in ids.values():control.command(['scontrol','release',jobid])
    snapshot={m:control.metadata(control.command(['scontrol','show','job',i,'--oneliner'])) for m,i in ids.items()}
    result=dict(instruction=INSTRUCTION,task_id=TASK,source=source,official_tree=tree,jobs=jobs,
        execution_lock=member(lock_path),held_inspected=True,released=True,initial_snapshot=snapshot,
        qualification='NOT_RUN_USER_DISABLED',W20_not_observed=True,monitoring_active=False)
    write_new(attempt/'submission.json',result)
    print({m:dict(job_id=ids[m],state=snapshot[m]['JobState'],dependency=jobs[m]['dependency']) for m in ids})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preparation',required=True);p.add_argument('--attempt',required=True)
    a=p.parse_args();register(a.preparation,a.attempt)
