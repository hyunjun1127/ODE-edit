"""Saved CF checkpoint generation evaluations plus CPU reducer; held inspection, no retry."""
import argparse
import getpass
import os
import re
import shlex
from pathlib import Path
from official.experiments.prepare import write_new,digest
from .common import read,member,verify
from . import submit as control
from .fe_history_submit import scheduling
from .flucon_eval import tracking_values,INSTRUCTION,TASK,LOCAL

def register(preparation,attempt):
    preparation=Path(preparation).absolute();prep=read(preparation)
    assert prep['instruction']==INSTRUCTION and 0 < len(prep['configs']) <= 5
    attempt=Path(attempt).absolute();assert attempt.is_relative_to(LOCAL)
    if (attempt/'submission.json').exists():print(read(attempt/'submission.json'));return
    assert not attempt.exists(),'EXISTING_ATTEMPT_RECONCILE'
    assert not list(LOCAL.rglob('submission.json')),'EXISTING_REGISTRATION_NO_DUPLICATE'
    source=control.command(['git','rev-parse','HEAD']);tree=control.command(['git','rev-parse','HEAD:official'])
    control.command(['git','merge-base','--is-ancestor',source,'origin/main'])
    assert not control.command(['git','status','--porcelain','--','official'])
    configs=[read(verify(m)) for m in prep['configs']]
    for c in configs:
        assert c['instruction']==INSTRUCTION and digest({k:v for k,v in c.items() if k!='config_sha256'})==c['config_sha256']
        verify(c['evaluator']);verify(c['reference']);verify(c['original']['checkpoint']);verify(c['original']['pointer'])
        tracking_values(c,source)
    capfile=Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    caprows=[r.split('\t') for r in capfile.read_text().splitlines() if r.startswith('server1\t')]
    assert len(caprows)==1;cap=min(4,int(caprows[0][2]));assert cap>0
    resources=dict(control.DEFAULT_RESOURCES,wall_seconds=48*3600)
    assert resources['memory_MiB']<=int(caprows[0][3])<=183296
    existing=control.inventory();assert existing['allocated_gpus']<=cap and existing['admitted_DAG_width']<=cap
    assert not any('flucon-eval' in r['original'] for r in existing['jobs']),'EXISTING_LIVE_REEVAL'
    methods=[c['key'] for c in configs];graph,width=scheduling(existing,methods,cap)
    plan=dict(cap=cap,source=dict(main_commit=source,official_tree=tree),resources=resources)
    physical=control.resource_preflight(plan);disk=os.statvfs(LOCAL)
    assert disk.f_bavail*disk.f_frsize>=64*(1<<30) and disk.f_favail>10000
    attempt.mkdir(parents=True);(attempt/'logs').mkdir();(attempt/'scripts').mkdir()
    write_new(attempt/'admission.json',dict(existing=existing,effective_cap=cap,DAG_width=width,graph=graph,
        resources=resources,physical=physical,cap_member=member(capfile),direct_cap_authority='USER_DIRECT_SERVER1_CAP4',
        disk_available_bytes=disk.f_bavail*disk.f_frsize,unchanged_old_jobs=True))
    frozen=control.freeze_source(plan,attempt)
    lock=dict(instruction=INSTRUCTION,source_commit=source,official_tree=tree,source_directory=frozen['directory'],
              source_members=frozen['members'],configs=prep['configs'],preparation=member(preparation),GPU_qualification='NOT_RUN_USER_DISABLED')
    lp=attempt/'execution-lock.json';write_new(lp,lock)
    ids={};submitted={};held={}
    entries=list(zip(methods,prep['configs']))+[('collector',None)]
    for method,cm in entries:
        gpu=cm is not None;key='flucon-eval-'+method.lower()
        parents=[ids.get(p,p) for p in graph[method]] if gpu else list(ids.values())
        argv=[control.DEFAULT_PYTHON,'-B','-u','-m','official.runners.server1.flucon_eval']
        if gpu:argv+=['run','--config',cm['path'],'--lock',str(lp)]
        else:argv+=['collect','--preparation',str(preparation),'--output',str(attempt/'collector-results.json'),'--lock',str(lp)]
        env=dict(PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',OFFICIAL_CODE_COMMIT=source,OFFICIAL_TREE_SHA256=tree,
            WANDB_CONSOLE='off',WANDB_SAVE_CODE='false')
        if not gpu:env['CUDA_VISIBLE_DEVICES']=''
        text='#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
        text+='cd '+shlex.quote(frozen['directory'])+'\nexec '+shlex.join(argv)+'\n'
        script=attempt/'scripts'/f'{method}.sh'
        with script.open('x') as f:f.write(text)
        job=dict(key=key,gpus=int(gpu),mode='chain' if gpu else 'collect',parents=[],external_resource_parents=parents)
        cmd,deps=control.sbatch_argv(plan,job,script,attempt,{})
        response=control.command(cmd);jid=response.split(';')[0];assert re.fullmatch(r'\d+',jid)
        ids[method]=jid;submitted[method]=dict(job_id=jid,name='official-s1-'+key,config=cm,dependencies=deps,argv=argv,script=member(script),gpus=int(gpu))
        write_new(attempt/f'submitted-{method}.json',dict(response=response,command=cmd,**submitted[method]))
        d=control.metadata(control.command(['scontrol','show','job',jid,'--oneliner']))
        assert d['JobState']=='PENDING' and d['Reason']=='JobHeldUser' and d['UserId'].split('(')[0]==getpass.getuser()
        assert d['Command']==str(script) and d['WorkDir']==frozen['directory'] and d['JobName']=='official-s1-'+key
        assert d['ReqNodeList']=='devbox' and d['QOS']==resources['qos'] and d['Partition']==resources['partition'] and d['Requeue']=='0'
        assert control.gpu_count(d['ReqTRES'])==int(gpu) and control.gpu_count(d.get('AllocTRES',''))==0
        assert control.requested_cpu_matches(d,8) and control.memory_MiB(d['MinMemoryNode'])==(65536 if gpu else 24576)
        assert control.seconds(d['TimeLimit'])==(48 if gpu else 4)*3600 and control.dependencies(d.get('Dependency',''))==sorted(deps)
        assert control.command(['scontrol','write','batch_script',jid,'-']).strip()==text.strip()
        held[method]=d
    fresh=control.inventory();assert fresh['admitted_DAG_width']<=cap and fresh['allocated_gpus']<=cap
    write_new(attempt/'held-inspection.json',dict(jobs=held,inventory=fresh,source_lock=member(lp),GPU_qualification=False))
    for jid in ids.values():control.command(['scontrol','release',jid])
    initial={m:control.metadata(control.command(['scontrol','show','job',jid,'--oneliner'])) for m,jid in ids.items()}
    result=dict(instruction=INSTRUCTION,source=source,official_tree=tree,jobs=submitted,initial_snapshot=initial,
                execution_lock=member(lp),released=True,held_inspected=True,eval_completed=False,online_readback='NOT_OBSERVED_BEFORE_STARTUP')
    write_new(attempt/'submission.json',result);print({m:dict(job_id=ids[m],state=initial[m]['JobState'],dependency=submitted[m]['dependencies']) for m in ids})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preparation',required=True);p.add_argument('--attempt',required=True);a=p.parse_args();register(a.preparation,a.attempt)
