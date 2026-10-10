"""Four native author cold runs, two bounded Slurm resource lanes."""
import argparse,getpass,os,re,shlex
from pathlib import Path
from official.experiments.prepare import write_new,digest
from official.runners.fe_original import INSTRUCTION,validate
from .common import read,member,verify
from . import submit as control
ROOT=Path('/mnt/raid5/janghj/ODE-edit/local/fe-original-w0-2k-20261011')
PYTHON=str(ROOT/'venv/bin/python')

def lane_graph(rows):
    keys=['llama3-cf','llama3-zsre','gptj-cf','gptj-zsre']
    frontier=control.frontier(rows)
    # One existing GPU lane permits immediate admission into the other lane.
    graph={keys[0]:[] if control.graph_width(rows)<=1 else frontier,
           keys[1]:[keys[0]],keys[2]:frontier,keys[3]:[keys[2]]}
    combined=rows+[dict(key=k,gpus=1,parents=graph[k]) for k in keys]
    assert control.graph_width(combined)<=2,'CAP2_DAG'
    return graph

def register(preparation,attempt):
    preparation=Path(preparation).absolute();prep=read(preparation)
    attempt=Path(attempt).absolute();assert attempt.is_relative_to(ROOT)
    if (attempt/'submission.json').exists():print(read(attempt/'submission.json'));return
    assert not attempt.exists(),'EXISTING_ATTEMPT_RECONCILE'
    assert not list(ROOT.rglob('submitted-*.json')),'EXISTING_REGISTRATION_NO_DUPLICATE'
    source=control.command(['git','rev-parse','HEAD']);tree=control.command(['git','rev-parse','HEAD:official'])
    control.command(['git','merge-base','--is-ancestor',source,'origin/main'])
    assert not control.command(['git','status','--porcelain','--','official'])
    configs=[read(verify(m)) for m in prep['configs']]
    keys=[c['model']+'-'+c['dataset'] for c in configs]
    assert keys==['llama3-cf','llama3-zsre','gptj-cf','gptj-zsre']
    for c in configs:
        validate(c);assert c['server']=='server1'
        assert c['checkpoint_lock']==str(ROOT/'checkpoint-serialization.lock')
        verify(c['assets']);verify(c['stream'])
        for m in c['author_members']:verify(m)
        if c['dataset']=='zsre':verify(c['query_proof'])
    capfile=Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    caprows=[r.split('\t') for r in capfile.read_text().splitlines() if r.startswith('server1\t')]
    assert len(caprows)==1 and int(caprows[0][2])==2
    resources=dict(control.DEFAULT_RESOURCES,cpus=8,wall_seconds=48*3600,memory_MiB=98304)
    assert resources['memory_MiB']<=int(caprows[0][3])<=183296
    existing=control.inventory();assert existing['allocated_gpus']<=2 and existing['admitted_DAG_width']<=2
    assert not any('fe-original' in r['original'] for r in existing['jobs']),'LIVE_DUPLICATE'
    graph=lane_graph(existing['jobs']);plan=dict(cap=2,source=dict(main_commit=source,official_tree=tree),resources=resources)
    physical=control.resource_preflight(plan);disk=os.statvfs(ROOT)
    assert disk.f_bavail*disk.f_frsize>=64*1024**3 and disk.f_favail>10000
    attempt.mkdir(parents=True);(attempt/'logs').mkdir();(attempt/'scripts').mkdir()
    write_new(attempt/'admission.json',dict(existing=existing,effective_cap=2,graph=graph,resources=resources,
        physical=physical,cap_member=member(capfile),disk_available_bytes=disk.f_bavail*disk.f_frsize,old_jobs_unchanged=True))
    frozen=control.freeze_source(plan,attempt)
    lock=dict(instruction=INSTRUCTION,source_commit=source,official_tree=tree,source_directory=frozen['directory'],
        source_members=frozen['members'],configs=prep['configs'],preparation=member(preparation),GPU_qualification='NOT_RUN_USER_DISABLED')
    lp=attempt/'execution-lock.json';write_new(lp,lock)
    ids={};submitted={};held={}
    for key,cm in zip(keys,prep['configs']):
        parents=[ids.get(p,p) for p in graph[key]]
        argv=[PYTHON,'-B','-u','-m','official.runners.fe_original','--config',cm['path'],'--lock',str(lp)]
        env=dict(PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',OFFICIAL_CODE_COMMIT=source,OFFICIAL_TREE_SHA256=tree,
            WANDB_CONSOLE='off',WANDB_SAVE_CODE='false')
        script_text='#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
        script_text+='cd '+shlex.quote(frozen['directory'])+'\nexec '+shlex.join(argv)+'\n'
        script=attempt/'scripts'/f'{key}.sh'
        with script.open('x') as f:f.write(script_text)
        job=dict(key=key+'-fe-original',gpus=1,mode='chain',parents=[],external_resource_parents=parents)
        cmd,deps=control.sbatch_argv(plan,job,script,attempt,{})
        response=control.command(cmd);jid=response.split(';')[0];assert re.fullmatch(r'\d+',jid)
        ids[key]=jid;submitted[key]=dict(job_id=jid,name='official-s1-'+job['key'],config=cm,dependencies=deps,
            argv=argv,script=member(script),gpus=1,output=read(cm['path'])['output'])
        write_new(attempt/f'submitted-{key}.json',dict(response=response,command=cmd,**submitted[key]))
        d=control.metadata(control.command(['scontrol','show','job',jid,'--oneliner']))
        assert d['JobState']=='PENDING' and d['Reason']=='JobHeldUser' and d['UserId'].split('(')[0]==getpass.getuser()
        assert d['Command']==str(script) and d['WorkDir']==frozen['directory'] and d['JobName']==submitted[key]['name']
        assert d['ReqNodeList']=='devbox' and d['QOS']==resources['qos'] and d['Partition']==resources['partition'] and d['Requeue']=='0'
        assert control.gpu_count(d['ReqTRES'])==1 and control.gpu_count(d.get('AllocTRES',''))==0
        assert control.requested_cpu_matches(d,8) and control.memory_MiB(d['MinMemoryNode'])==98304
        assert control.seconds(d['TimeLimit'])==48*3600 and control.dependencies(d.get('Dependency',''))==sorted(deps)
        assert control.command(['scontrol','write','batch_script',jid,'-']).strip()==script_text.strip()
        held[key]=d
    fresh=control.inventory();assert fresh['admitted_DAG_width']<=2 and fresh['allocated_gpus']<=2
    write_new(attempt/'held-inspection.json',dict(jobs=held,inventory=fresh,source_lock=member(lp),GPU_qualification=False))
    for jid in ids.values():control.command(['scontrol','release',jid])
    initial={k:control.metadata(control.command(['scontrol','show','job',jid,'--oneliner'])) for k,jid in ids.items()}
    result=dict(instruction=INSTRUCTION,source=source,official_tree=tree,jobs=submitted,initial_snapshot=initial,
        execution_lock=member(lp),released=True,held_inspected=True,completed=False,online_readback='NOT_OBSERVED_BEFORE_STARTUP')
    write_new(attempt/'submission.json',result)
    print({k:dict(job_id=ids[k],state=initial[k]['JobState'],dependencies=submitted[k]['dependencies']) for k in ids})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preparation',required=True);p.add_argument('--attempt',required=True)
    a=p.parse_args();register(a.preparation,a.attempt)
