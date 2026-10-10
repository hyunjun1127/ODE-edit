"""One deliberate cap3 held/inspect/release pass, no scientific retry."""
import argparse,getpass,re,shlex,shutil
from pathlib import Path
from official.experiments.prepare import write_new,digest
from official.runners.server1.common import read,member,verify
from official.runners.server1 import submit as shared
from official.runners.server1.flucon_submit import cap3_scheduling
from . import submit as own
from .flucon_eval import AUTHORITY,LOCAL,tracking_values

PYTHON='/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
RESOURCES=dict(gpus=1,CPUs=6,memory_MiB=59392,wall_hours=48,node='server2',qos='lab_gpu_s2',partition='gpu')

def graph_inventory(existing):
    assert not existing['ambiguous']
    return dict(jobs=[dict(key=r['job'],gpus=r['gpus'],parents=own.dependency_ids(r['dependency']))
                      for r in existing['project']])

def register(preparation,attempt):
    prep=read(preparation);assert prep['registration_authority']==AUTHORITY and prep['configs']
    configs=[read(verify(m)) for m in prep['configs']]
    assert len({c['original']['checkpoint']['sha256'] for c in configs})==len(configs)
    attempt=Path(attempt).absolute();assert attempt.is_relative_to(LOCAL) and not attempt.exists()
    assert not list(LOCAL.rglob('submitted-*.json')),'EXISTING_REGISTRATION_RECONCILE'
    capfile=Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    caps=[r.split('\t') for r in capfile.read_text().splitlines() if r.startswith('server2\t')]
    assert len(caps)==1 and caps[0][1]=='server2'
    cap=min(3,int(caps[0][2]));assert cap>0 and RESOURCES['memory_MiB']<=int(caps[0][3])
    existing=own.inventory();base=graph_inventory(existing)
    assert sum(r['allocated_GPUs'] for r in existing['project'])<=cap
    assert not any('s2-flucon-' in r['name'] for r in existing['project']),'LIVE_TASK_RECONCILE'
    graph,width=cap3_scheduling(base,[c['key'] for c in configs],cap)
    node=shared.metadata(own.command(['scontrol','show','node','server2','--oneliner']))
    partition=shared.metadata(own.command(['scontrol','show','partition','gpu','--oneliner']))
    qos=own.command(['sacctmgr','-nP','show','qos','lab_gpu_s2','format=Name,MaxWall,MaxTRESPU,MaxTRESPJ,GrpTRES'])
    assert int(node['RealMemory'])>=59392 and int(node['CPUTot'])>=6
    assert 'lab_gpu_s2' in partition['AllowQos'] and 'gres/gpu=4' in qos
    free=shutil.disk_usage(LOCAL).free
    # No CP/model duplication. Reserve existing checkpoint/temp growth plus
    # at most 2GiB local observation/spool allowance per evaluation.
    needed=(64+2*len(configs))*(1<<30);assert free>=needed,'STORAGE_PENDING_KEEP_SOURCE'
    repo=Path(__file__).resolve().parents[3]
    source=own.command(['git','rev-parse','HEAD'],cwd=repo)
    tree=own.command(['git','rev-parse','HEAD:official'],cwd=repo)
    own.command(['git','merge-base','--is-ancestor',source,'origin/main'],cwd=repo)
    assert not own.command(['git','status','--porcelain','--','official'],cwd=repo)
    for c in configs:
        assert c['registration_authority']==AUTHORITY
        assert c['config_sha256']==digest({k:v for k,v in c.items() if k!='config_sha256'})
        for m in (c['original']['checkpoint'],c['original']['pointer'],c['reference'],c['evaluator']):verify(m)
        tracking_values(c,source)
    attempt.mkdir(parents=True);(attempt/'scripts').mkdir();(attempt/'logs').mkdir()
    write_new(attempt/'admission.json',dict(instruction=AUTHORITY,existing=existing,effective_cap=cap,DAG_width=width,
        graph=graph,node=node,partition=partition,qos=qos,resources=RESOURCES,cap_member=member(capfile),
        disk_available_bytes=free,required_reserve_and_output_bytes=needed,old_jobs_changed=False))
    frozen=shared.freeze_source(dict(source=dict(main_commit=source,official_tree=tree)),attempt,repository=repo,runner=own.command)
    lock=dict(instruction=AUTHORITY,source_commit=source,official_tree=tree,source_directory=frozen['directory'],
        source_members=frozen['members'],configs=prep['configs'],preparation=member(preparation),qualification='NOT_RUN_USER_DISABLED')
    lp=attempt/'execution-lock.json';write_new(lp,lock)
    jobs={};held={}
    entries=list(zip(configs,prep['configs']))+[(None,None)]
    for c,cm in entries:
        gpu=c is not None;key=c['key'] if gpu else 'collector'
        parents=[jobs[p]['job_id'] if p in jobs else p for p in graph[key]] if gpu else [j['job_id'] for j in jobs.values()]
        deps=[('afterany',p) for p in parents];name='s2-flucon-'+key
        argv=[PYTHON,'-B','-u','-m','official.runners.server2.flucon_eval']
        argv+=['run','--config',cm['path'],'--lock',str(lp)] if gpu else ['collect','--preparation',str(preparation),'--output',str(attempt/'collector-results.json'),'--lock',str(lp)]
        env=dict(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',OFFICIAL_CODE_COMMIT=source,
            OFFICIAL_TREE_SHA256=tree,WANDB_CONSOLE='off',WANDB_SAVE_CODE='false')
        if not gpu:env['CUDA_VISIBLE_DEVICES']=''
        text='#!/bin/bash\nset -euo pipefail\n'+''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
        text+='cd '+shlex.quote(frozen['directory'])+'\nexec '+shlex.join(argv)+'\n'
        script=attempt/'scripts'/f'{key}.sh'
        with script.open('x') as f:f.write(text)
        mem=59392 if gpu else 24576;hours=48 if gpu else 4
        cmd=['sbatch','--parsable','--hold','--export=NONE','--no-requeue','--partition=gpu','--qos=lab_gpu_s2',
            '--nodelist=server2','--nodes=1','--ntasks=1','--cpus-per-task=6',f'--mem={mem}M',
            '--time='+('2-00:00:00' if gpu else '04:00:00'),'--job-name='+name,'--chdir='+frozen['directory'],
            '--output='+str(attempt/'logs'/f'{key}-%j.out'),'--error='+str(attempt/'logs'/f'{key}-%j.err')]
        if gpu:cmd+=['--gres=gpu:a6000:1']
        if parents:cmd+=['--dependency=afterany:'+':'.join(parents)]
        cmd+=[str(script)];response=own.command(cmd);jid=response.split(';')[0];assert re.fullmatch(r'\d+',jid)
        jobs[key]=dict(job_id=jid,name=name,config=cm,script=member(script),argv=argv,dependencies=deps,
            output=c['output'] if gpu else str(attempt/'collector-results.json'),gpus=int(gpu))
        write_new(attempt/f'submitted-{key}.json',dict(response=response,command=cmd,**jobs[key]))
        d=shared.metadata(own.command(['scontrol','show','job',jid,'--oneliner']))
        assert d['UserId'].split('(')[0]==getpass.getuser() and d['JobState']=='PENDING' and d['Reason']=='JobHeldUser'
        assert d['Command']==str(script) and d['WorkDir']==frozen['directory'] and d['JobName']==name
        assert d['ReqNodeList']=='server2' and d['QOS']=='lab_gpu_s2' and d['Partition']=='gpu' and d['Requeue']=='0'
        assert shared.gpu_count(d['ReqTRES'])==int(gpu) and shared.gpu_count(d.get('AllocTRES',''))==0
        assert shared.requested_cpu_matches(d,6) and shared.memory_MiB(d['MinMemoryNode'])==mem
        assert shared.seconds(d['TimeLimit'])==hours*3600 and shared.dependencies(d['Dependency'])==sorted(deps)
        assert own.command(['scontrol','write','batch_script',jid,'-']).strip()==text.strip()
        held[key]=d
    fresh=own.inventory();assert shared.graph_width(graph_inventory(fresh)['jobs'])<=cap
    assert sum(r['allocated_GPUs'] for r in fresh['project'])<=cap
    write_new(attempt/'held-inspection.json',dict(jobs=held,inventory=fresh,lock=member(lp),actual_GPU_PASS=False))
    for j in jobs.values():own.command(['scontrol','release',j['job_id']])
    snapshot={k:shared.metadata(own.command(['scontrol','show','job',j['job_id'],'--oneliner'])) for k,j in jobs.items()}
    result=dict(instruction=AUTHORITY,source=source,official_tree=tree,jobs=jobs,execution_lock=member(lp),
        released=True,held_inspected=True,initial_snapshot=snapshot,online_readback='NOT_OBSERVED_BEFORE_STARTUP',
        actual_GPU_completion=False,existing_jobs_changed=False)
    write_new(attempt/'submission.json',result)
    print({k:dict(job_id=j['job_id'],state=snapshot[k]['JobState'],dependencies=j['dependencies']) for k,j in jobs.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preparation',required=True);p.add_argument('--attempt',required=True)
    a=p.parse_args();register(a.preparation,a.attempt)
