"""One-shot SH2 CF -> zsRE held registration; existing allocations are untouched."""
import argparse
import getpass
import re
import shlex
import shutil
from pathlib import Path

from official.experiments.prepare import write_new
from official.runners.server1.common import read, member, verify
from official.runners.server1 import submit as shared
from official.runners.server2 import submit as own
from .fe_author_prepare import INSTRUCTION, LOCAL, storage_plan

PYTHON = '/mnt/raid5/janghj/EasyEdit/.venv/bin/python'
RESOURCES = dict(GPUs=1, CPUs=6, memory_MiB=59392, wall='2-00:00:00',
                 partition='gpu', qos='lab_gpu_s2', node='server2')

def graph(existing, first=None, second=None):
    rows = [dict(key=r['job'], gpus=r['gpus'], parents=own.dependency_ids(r['dependency']))
            for r in existing['project']]
    frontier = own.frontier(existing['project'])
    if first: rows.append(dict(key=first, gpus=1, parents=frontier))
    if second: rows.append(dict(key=second, gpus=1, parents=[first]))
    return frontier, shared.graph_width(rows)

def register(preparation, attempt):
    # Shared scientific runner is selected only by the own prepared and reviewed
    # binding. No science implementation is copied into this control adapter.
    prep = read(preparation)
    assert prep['instruction'] == INSTRUCTION and prep['server'] == 'server2'
    assert prep['datasets'] == ['cf', 'zsre'] and len(prep['configs']) == 2
    from official.runners.fe_author_history import validate_config
    configs = [validate_config(read(verify(m))) for m in prep['configs']]
    assert [c['dataset'] for c in configs] == ['cf', 'zsre']
    assert all(c['model'] == 'qwen25' and c['method'] == 'MEMIT_FE_HISTORY' for c in configs)
    attempt = Path(attempt).absolute()
    assert attempt.is_relative_to(LOCAL) and not attempt.exists(), 'ATTEMPT_RECONCILE_NO_RETRY'
    for path in LOCAL.rglob('submitted-*.json'):
        raise ValueError('EXISTING_REGISTRATION_RECONCILE:' + str(path))
    cap_path = Path('/mnt/raid5/janghj/ODE-edit/servers/local/gpu-caps.tsv')
    caps = [r.split('\t') for r in cap_path.read_text().splitlines() if r.startswith('server2\t')]
    assert len(caps) == 1 and caps[0][1] == 'server2'
    cap = min(4, int(caps[0][2])); assert cap > 0 and RESOURCES['memory_MiB'] <= int(caps[0][3])
    existing = own.inventory(); assert not existing['ambiguous']
    assert not any('fe-author' in r['name'] for r in existing['project']), 'DUPLICATE_LIVE_TASK'
    frontier, width = graph(existing, 'NEW_CF', 'NEW_ZSRE')
    assert width <= cap and sum(r['allocated_GPUs'] for r in existing['project']) <= cap
    storage = storage_plan(shutil.disk_usage(LOCAL).free)
    assert storage['sufficient'], 'STORAGE_PENDING_KEEP_SOURCE'
    node = shared.metadata(own.command(['scontrol','show','node','server2','--oneliner']))
    partition = shared.metadata(own.command(['scontrol','show','partition','gpu','--oneliner']))
    qos = own.command(['sacctmgr','-nP','show','qos','lab_gpu_s2','format=Name,MaxWall,MaxTRESPU,MaxTRESPJ,GrpTRES'])
    assert 'gres/gpu=4' in qos and 'lab_gpu_s2' in partition['AllowQos']
    assert int(node['RealMemory']) >= RESOURCES['memory_MiB'] and int(node['CPUTot']) >= 6
    repo = Path(__file__).resolve().parents[3]
    source = own.command(['git','rev-parse','HEAD'], cwd=repo)
    tree = own.command(['git','rev-parse','HEAD:official'], cwd=repo)
    own.command(['git','merge-base','--is-ancestor',source,'origin/main'], cwd=repo)
    assert not own.command(['git','status','--porcelain','--','official'], cwd=repo)
    attempt.mkdir(parents=True); (attempt/'logs').mkdir(); (attempt/'scripts').mkdir()
    write_new(attempt/'admission.json', dict(instruction=INSTRUCTION, existing=existing,
        frontier=frontier, effective_cap=cap, DAG_width=width, storage=storage,
        local_cap=member(cap_path), node=node, partition=partition, qos=qos, resources=RESOURCES))
    frozen = shared.freeze_source(dict(source=dict(main_commit=source, official_tree=tree)), attempt,
                                  repository=repo, runner=own.command)
    lock = dict(instruction=INSTRUCTION, source_commit=source, official_tree=tree,
        source_directory=frozen['directory'], source_members=frozen['members'], configs=prep['configs'],
        preparation=member(preparation), qualification='NOT_RUN_USER_DISABLED')
    lock_path = attempt/'execution-lock.json'; write_new(lock_path, lock)
    jobs = {}; held = {}
    for dataset, cm, cfg in zip(prep['datasets'], prep['configs'], configs):
        parents = frontier if dataset == 'cf' else [jobs['cf']['job_id']]
        dependency = [('afterany', p) for p in parents]
        name = f's2-qwen25-{dataset}-fe-author-history'
        argv = [PYTHON,'-B','-u','-m','official.runners.fe_author_history',
                '--config',cm['path'],'--lock',str(lock_path)]
        env = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', OPENBLAS_NUM_THREADS='6',
            OFFICIAL_CODE_COMMIT=source, OFFICIAL_TREE_SHA256=tree, WANDB_CONSOLE='off', WANDB_SAVE_CODE='false')
        text = '#!/bin/bash\nset -euo pipefail\n' + ''.join('export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items())
        text += 'cd '+shlex.quote(frozen['directory'])+'\nexec '+shlex.join(argv)+'\n'
        script = attempt/'scripts'/f'{dataset}.sh'
        with script.open('x') as f: f.write(text)
        cmd = ['sbatch','--parsable','--hold','--export=NONE','--no-requeue','--partition=gpu',
            '--qos=lab_gpu_s2','--nodelist=server2','--nodes=1','--ntasks=1','--cpus-per-task=6',
            '--gres=gpu:a6000:1','--mem=59392M','--time=2-00:00:00','--job-name='+name,
            '--chdir='+frozen['directory'],'--output='+str(attempt/'logs'/f'{dataset}-%j.out'),
            '--error='+str(attempt/'logs'/f'{dataset}-%j.err')]
        if parents: cmd.append('--dependency=afterany:'+':'.join(parents))
        cmd.append(str(script)); response = own.command(cmd)
        jid = response.split(';')[0]; assert re.fullmatch(r'\d+',jid)
        jobs[dataset] = dict(job_id=jid,name=name,config=cm,script=member(script),argv=argv,
                            dependency=dependency,output=cfg['output'])
        write_new(attempt/f'submitted-{dataset}.json',dict(response=response,command=cmd,**jobs[dataset]))
        d = shared.metadata(own.command(['scontrol','show','job',jid,'--oneliner']))
        assert d['UserId'].split('(')[0] == getpass.getuser() and d['JobState']=='PENDING' and d['Reason']=='JobHeldUser'
        assert d['Command']==str(script) and d['WorkDir']==frozen['directory'] and d['JobName']==name
        assert d['ReqNodeList']=='server2' and d['QOS']=='lab_gpu_s2' and d['Partition']=='gpu' and d['Requeue']=='0'
        assert shared.gpu_count(d['ReqTRES'])==1 and shared.gpu_count(d.get('AllocTRES',''))==0
        assert shared.requested_cpu_matches(d,6) and shared.memory_MiB(d['MinMemoryNode'])==59392
        assert shared.seconds(d['TimeLimit'])==48*3600 and shared.dependencies(d['Dependency'])==sorted(dependency)
        assert own.command(['scontrol','write','batch_script',jid,'-']).strip()==text.strip()
        held[dataset]=d
    fresh = own.inventory(); assert not fresh['ambiguous'] and graph(fresh)[1] <= cap
    write_new(attempt/'held-inspection.json',dict(jobs=held,inventory=fresh,lock=member(lock_path),
        GPU_qualification='NOT_RUN_USER_DISABLED', actual_GPU_PASS=False))
    for job in jobs.values(): own.command(['scontrol','release',job['job_id']])
    snapshot = {k: shared.metadata(own.command(['scontrol','show','job',v['job_id'],'--oneliner'])) for k,v in jobs.items()}
    result = dict(instruction=INSTRUCTION,source=source,official_tree=tree,jobs=jobs,
        execution_lock=member(lock_path),held_inspected=True,released=True,initial_snapshot=snapshot,
        existing_jobs_changed=False,monitoring_active=False)
    write_new(attempt/'submission.json',result)
    print({k:dict(job_id=v['job_id'],state=snapshot[k]['JobState']) for k,v in jobs.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preparation',required=True);p.add_argument('--attempt',required=True)
    a=p.parse_args();register(a.preparation,a.attempt)
