"""Create-once frozen native six-baseline DAG; no retries or other job writes.

Scheduler inspection/admission primitives are reused read-only from the prior
SH4 PRICE launcher, not its scientific runner or two-writer resource graph.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile

from .common import ROOT, LOCAL, TASK, NONCE, SESSION, METHODS, authority, member, require, sha, stat_seal, verify, write
from .native_binding import verify as verify_native, asset_stat
from project.run_scripts.jlz_price_gptj.submit import command, resource_inventory, dependency_ids
from project.run_scripts.jlz_price_gptj.admission import width

PYTHON = '/data/janghj/EasyEdit/.venv/bin/python'
ROLES = (*METHODS, 'collector')
SOURCES = ['project/run_scripts/llama3_native_baselines',
    'project/run_scripts/experiment_generation_eval', 'project/run_scripts/experiment_tracking',
    'project/run_scripts/jlz_interference_l1', 'project/run_scripts/jlz_v12r',
    'project/run_scripts/jlz_native_writer_aware', 'project/run_scripts/jlz_realized_subject',
    'project/run_scripts/jlz_shared_budget', 'project/run_scripts/jlz_realization',
    'project/run_scripts/jlz_writer_coupled', 'project/run_scripts/jlz_realized_writer',
    'project/run_scripts/jlz_price_gptj',
    'project/run_scripts/jlz_realized_writer_sequential/review_completed.py',
    'project/run_scripts/jlz_pilot/prompts.py', 'project/run_scripts/jlz_pilot/__init__.py',
    'scripts/fixed_counterfact.py', 'scripts/check-slurm-resource-cap.sh',
    'scripts/check-slurm-gpu-cap.sh', 'scripts/slurm_memory_policy.py',
    'servers/slurm-memory-policy.tsv', 'control/gpu-concurrency-policy.tsv',
    'control/wandb-policy.json', 'control/wandb-method-metric-schema.json',
    'control/generation-metric-policy.json',
    'plans/global/2026-10-07-baseline-fluency-consistency-rerun/contract.json',
    'messages/head/2026-10-07-baseline-fluency-consistency-rerun-server4.json',
    'plans/global/2026-10-04-jlz-v12-marginal-allocation/experiment-2k/case-schedule-first2000.csv']


def order(parallel):
    require(1 <= parallel <= 3, 'TASK_CONCURRENCY')
    graph = {'MEMIT': []}
    rest = METHODS[1:]
    for index, method in enumerate(rest):
        graph[method] = ['MEMIT'] if index < parallel else [rest[index - parallel]]
    graph['collector'] = list(METHODS)
    return graph


def launcher(source, commit, role, attempt, resources, generation=None):
    collector = role == 'collector'
    cpu = resources['collector_cpu'] if collector else resources['cpu']
    argv = [PYTHON, '-u', '-m', 'project.run_scripts.llama3_native_baselines.' +
            ('collect' if collector else 'run'), '--attempt', str(attempt)]
    if not collector: argv += ['--method', role]
    env = dict(PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS=str(cpu), MKL_NUM_THREADS=str(cpu), OPENBLAS_NUM_THREADS=str(cpu),
        TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        LLAMA3_NATIVE_BASELINES_SOURCE_COMMIT=commit)
    if generation:
        env['NLTK_DATA'] = generation['nltk_data']
    if collector: env['CUDA_VISIBLE_DEVICES'] = ''
    return '#!/bin/bash\nset -euo pipefail\n' + ''.join(
        'export '+k+'='+shlex.quote(v)+'\n' for k,v in env.items()) + \
        'cd '+shlex.quote(str(source))+'\nexec '+shlex.join(argv)+'\n'


def freeze(configpath, attempt):
    require(attempt.parent == LOCAL and not attempt.exists(), 'CREATE_ONCE_ATTEMPT')
    authority()
    c = json.loads(configpath.read_text())
    require(c['task_id'] == TASK and c['instruction_id'] == NONCE
        and c['attempt'] == str(attempt), 'EXACT_AUTHORIZED_ATTEMPT')
    require(not command(['git','status','--porcelain','--',*SOURCES], ROOT), 'COMMIT_BEFORE_FREEZE')
    proof = json.loads(verify(c['cpu_preflight']).read_text())
    require(proof['status'] == 'CPU_SOURCE_READY' and proof['actual_GPU'] == 'NOT_OBSERVED',
            'HONEST_CPU_SOURCE_RECEIPT')
    for row in proof['source_members']: verify(row)
    generation = c['generation']
    require(generation['status'] == 'SOURCE_REFERENCE_BOUND' and generation['source_members'],
            'SH1_SOURCE_REFERENCE_NOT_BOUND')
    verify(generation['READY']); verify(generation['reference_manifest'])
    for row in generation['source_members']: verify(row)
    commit = command(['git','rev-parse','HEAD'],ROOT)
    tree = command(['git','rev-parse','HEAD^{tree}'],ROOT)
    attempt.mkdir()
    source = attempt/'source'; source.mkdir()
    archive = attempt/'source.tar'
    command(['git','archive','--format=tar','--output='+str(archive),commit,*SOURCES],ROOT)
    with tarfile.open(archive) as tf:
        rows = tf.getmembers()
        require(len(rows) == len({r.name for r in rows}) and all(
            (r.isfile() or r.isdir()) and not Path(r.name).is_absolute()
            and '..' not in Path(r.name).parts for r in rows), 'SAFE_SOURCE_ARCHIVE')
        tf.extractall(source, filter='data')
    for row in proof['source_members'] + generation['source_members']:
        rel = Path(row['path']).relative_to(ROOT)
        require(sha(source/rel) == row['sha256'], 'TESTED_ARCHIVE_BYTES')
    native_sources = []
    for binding in c['native'].values():
        native_sources += binding['files'] + [binding['hparams'], binding['context_reuse']['contexts']]
        if binding.get('terminal_source'): native_sources.append(binding['terminal_source'])
    write(attempt/'config.json',c)
    for role in ROLES:
        script = attempt/(role+'.sh')
        script.write_text(launcher(source,commit,role,attempt,c['resources'],c['generation']))
        script.chmod(0o755)
    write(attempt/'execution.lock.json',dict(instruction_id=NONCE,task_id=TASK,
        source_commit=commit,source_tree=tree,archive=member(archive),
        source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        runtime_sources=c['runtime']['source_members'],generation_sources=generation['source_members'],
        native_sources=native_sources,config_sha256=sha(attempt/'config.json'),
        launchers=[member(attempt/(role+'.sh')) for role in ROLES],
        tracking_env=member(c['tracking']['env_file']), owner=getpass.getuser(),host='server4',
        session=SESSION,resources=c['resources'],run_instance=c['run_instance'],
        noCP=True,exact_resume='NOT_AVAILABLE',native_source_in_RAM_only=False,
        source_copy_only='small Python/hparams; no W/H/z/payload checkpoints'))
    return verify_frozen(attempt)


def verify_frozen(attempt):
    lock=json.loads((attempt/'execution.lock.json').read_text())
    c=json.loads((attempt/'config.json').read_text())
    require(lock['owner']==getpass.getuser() and lock['host']=='server4'
            and c['instruction_id']==lock['instruction_id']==NONCE
            and c['task_id']==lock['task_id']==TASK, 'FROZEN_OWNER_TASK')
    require(sha(attempt/'config.json')==lock['config_sha256'], 'CONFIG_BYTES')
    for row in lock['source_members']+lock['runtime_sources']+lock['native_sources']+lock['launchers']+[lock['archive'],lock['tracking_env']]:
        verify(row)
    for row in c['assets']: stat_seal(row)
    for binding in c['native'].values():
        for row in binding['stats']: asset_stat(row)
        if binding.get('projector'): asset_stat(binding['projector'])
    for row in c['W0_evaluator']:
        verify(row['historical'])
        require(sha(attempt/'source'/row['relative'])==row['historical']['sha256'], 'W0_ARCHIVE_EVALUATOR')
    require(shutil.disk_usage(attempt).free >= c['resources']['combined_reserve_bytes'],
            'RESOURCE_BLOCKED_STORAGE')
    return lock,c


def arguments(role, dep, attempt, r):
    collector=role=='collector';cpus=r['collector_cpu'] if collector else r['cpu']
    argv=['sbatch','--parsable','--hold','--partition=gpu','--qos='+r['qos'],
        '--nodelist=server4','--nodes=1','--ntasks=1','--cpus-per-task='+str(cpus),
        '--export=NONE','--no-requeue','--job-name='+TASK+'-'+role,
        '--chdir='+str(attempt/'source'),
        '--mem='+str(r['collector_host_mib'] if collector else r['host_mib'])+'M',
        '--time='+(r['collector_wall'] if collector else r['wall']),
        '--output='+str(attempt/(role+'-%j.out')),'--error='+str(attempt/(role+'-%j.err'))]
    if not collector: argv+=['--gres=gpu:1']
    if dep: argv+=['--dependency='+dep]
    return argv+[str(attempt/(role+'.sh'))]


def inspect(job,role,dep,argv,attempt,r):
    detail=command(['scontrol','show','job',job,'--oneliner'])
    collector=role=='collector';cpu=r['collector_cpu'] if collector else r['cpu']
    for term in (f'JobId={job} ',f'JobName={TASK}-{role} ','UserId='+getpass.getuser()+'(',
        'JobState=PENDING ','Reason=JobHeldUser ','Requeue=0 ',f'CPUs/Task={cpu} ',
        'ReqNodeList=server4 ','Partition=gpu ','QOS='+r['qos']+' '):
        require(term in detail,'HELD:'+term)
    require(re.search(rf'\bNumCPUs={cpu}(?:-[0-9]+)? ',detail),'HELD_CPU')
    require('gres/gpu' not in detail if collector else 'TresPerNode=gres/gpu:1' in detail,'HELD_GPU')
    mem=r['collector_host_mib'] if collector else r['host_mib']
    require(f'mem={mem}M' in detail or (mem%1024==0 and f'mem={mem//1024}G' in detail),'HELD_MEMORY')
    require('TimeLimit='+(r['collector_wall'] if collector else r['wall'])+' ' in detail,'HELD_TIME')
    got=re.search(r'\bDependency=([^ ]+)',detail)
    require(got and dependency_ids(got[1])==(set(dep.split(':')[1:]) if dep else set())
        and (not dep or got[1].startswith('afterany:')),'HELD_DEPENDENCY')
    script=attempt/(role+'.sh')
    require('Command='+str(script)+' ' in detail and 'WorkDir='+str(attempt/'source')+' ' in detail,'HELD_PATH')
    line=re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)',detail)
    require(line and shlex.split(line[1])==argv,'HELD_FULL_ARGV')
    require(command(['scontrol','write','batch_script',job,'-']).strip()==script.read_text().strip(),'HELD_SCRIPT')
    return dict(job=job,method=role,argv=argv,resource_detail=detail,launcher=member(script))


def submit(config,attempt):
    require(not list(LOCAL.glob('attempt-*/submitted-*.json')),'NO_DUPLICATE_REGISTRATION')
    before=resource_inventory()
    require(not any(j['name'].startswith(TASK+'-') for j in before['jobs']), 'EXACT_TASK_ALREADY_ADMITTED')
    local=next(row.split('\t') for row in (Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv')).read_text().splitlines() if row.startswith('server4\t'))
    tracked=int(next(row.split('\t')[1] for row in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if row.startswith('server4\t')))
    cap=min(3,int(local[2]),tracked);require(cap>=1,'NO_ADMISSION_CAP')
    lock,c=freeze(Path(config).resolve(),attempt);r=c['resources'];parallel=min(cap,r['task_cap'])
    require(1<=r['cpu']<=8 and r['host_mib']<=min(60416,int(local[3]))
            and r['task_cap']==3 and r['project_cap']==3,'LEGAL_RESOURCES')
    policy=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request','--server','server4',
        '--gpus','1','--mem',str(r['host_mib'])+'M','--local-limit-mib-per-gpu',local[3]])
    require('ALLOW_MEMORY_POLICY' in policy,'MEMORY_POLICY')
    node=command(['scontrol','show','node','server4']);partition=command(['scontrol','show','partition','gpu'])
    prior_width=width(before['jobs'])
    # A complete exact resource barrier avoids scheduling new lanes alongside
    # existing project lanes. It imposes no science PASS or method gate.
    barrier=[j['job'] for j in before['jobs']] if prior_width+parallel>cap else []
    ids={};mapping={};held=[];graph=order(parallel)
    for role in ROLES:
        parents=[ids[p] for p in graph[role]] if graph[role] else barrier
        dep='afterany:'+':'.join(parents) if parents else None
        argv=arguments(role,dep,attempt,r)
        job=command(argv).split(';')[0];require(job.isdigit(),'ACTUAL_JOB_ID')
        ids[role]=job;mapping[role]=dict(job=job,dependency=dep,argv=argv)
        write(attempt/('submitted-'+role+'.json'),dict(nonce=NONCE,role=role,status='HELD',**mapping[role]))
        held.append(inspect(job,role,dep,argv,attempt,r))
    current=resource_inventory(tuple(ids.values()))
    require({j['job'] for j in current['jobs']} <= {j['job'] for j in before['jobs']},'ADMISSION_RACE_KEEP_HELD')
    require(bool(barrier) or width(current['jobs'])+parallel<=cap,'CAP_RACE_KEEP_HELD')
    verify_frozen(attempt)
    write(attempt/'held-inspection.json',dict(jobs=held,before=before,prerelease=current,
        effective_project_cap=cap,task_cap=3,maximum_new_GPU_concurrency=parallel,
        resource_barrier=barrier,prior_DAG_width=prior_width,node=node,partition=partition,memory_policy=policy,
        source_lock=member(attempt/'execution.lock.json'),all_held_inspected_before_release=True,
        unrelated_job_mutations=0,protected_ours_mutations=0,
        generic_helper='NOT_PASS_PROJECT_PREFIX: exact own GPU/DAG inventory used; helper unchanged'))
    write(attempt/'held-submission.json',dict(nonce=NONCE,task_id=TASK,status='INSPECTED_HELD',jobs=ids,
        mapping=mapping,source=lock['source_commit'],lock=member(attempt/'execution.lock.json'),
        held=member(attempt/'held-inspection.json')))
    for role in reversed(ROLES):
        command(['scontrol','release',ids[role]])
        write(attempt/('released-'+role+'.json'),dict(job=ids[role],command_succeeded=True))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%b|%N|%r'])
    result=dict(nonce=NONCE,task_id=TASK,status='RELEASED',jobs=ids,mapping=mapping,
        source=lock['source_commit'],lock=member(attempt/'execution.lock.json'),held=member(attempt/'held-inspection.json'),
        bounded_initial_snapshot=snapshot,actual_B1='NOT_OBSERVED',agent_monitoring_active=False,
        automatic_resume=False,automatic_retry=False,noCP=True)
    write(attempt/'submission.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--attempt',type=Path,required=True);args=parser.parse_args()
    result=submit(args.config,args.attempt.resolve())
    print(json.dumps({k:result[k] for k in ('status','jobs','source','bounded_initial_snapshot')}))


if __name__=='__main__': main()
