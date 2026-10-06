"""Create-once six-cell freeze and two model resource lanes, combined cap2."""
import argparse
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from .common import ROOT, LOCAL, TASK, NONCE, CELLS, require, write, member, sha, verify

SOURCES=['project/run_scripts/jlz_price_alpha_writer','plans/global/jlz-price-alpha-writer-2k','project/proposals/jlz-alpha-writer-review','messages/head/2026-10-07-price-alpha-writer-2k-sh4.json','project/run_scripts/jlz_interference_l1','project/run_scripts/jlz_v12r','project/run_scripts/jlz_native_writer_aware',
    'project/run_scripts/jlz_realized_subject','project/run_scripts/jlz_shared_budget',
    'project/run_scripts/jlz_realization','project/run_scripts/jlz_writer_coupled',
    'project/run_scripts/jlz_realized_writer','project/run_scripts/jlz_realized_writer_sequential/review_completed.py',
    'project/run_scripts/jlz_pilot/prompts.py','project/run_scripts/jlz_pilot/__init__.py',
    'project/run_scripts/memit_history_lifelong/hparams.json',
    'scripts/fixed_counterfact.py','scripts/check-slurm-resource-cap.sh','scripts/check-slurm-gpu-cap.sh',
    'scripts/slurm_memory_policy.py','servers/slurm-memory-policy.tsv','control/gpu-concurrency-policy.tsv',
    'project/proposals/jlz-interference-budget-v1',
    'plans/global/2026-10-04-jlz-v12-marginal-allocation/experiment-2k/case-schedule-first2000.csv',
    'messages/head/2026-10-06-jlz-interference-l1-sh4.json',
    'messages/head/2026-10-06-price-cap-base-repair-2k-sh4.json','project/proposals/jlz-price-cap-budget-review',
    'project/run_scripts/experiment_tracking','control/wandb-policy.json']
ROLES=(*CELLS,'collector')
def resource_order(parallel):
    if parallel==1:return {r:([] if i==0 else [CELLS[i-1]]) for i,r in enumerate(CELLS)} | {'collector':list(CELLS)}
    return {r:([] if r.endswith('CAP075') else [r.rsplit('_',1)[0]+('_CAP075' if r.endswith('CAP100') else '_CAP100')]) for r in CELLS} | {'collector':list(CELLS)}
SESSION='01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
PYTHON='/data/janghj/EasyEdit/.venv/bin/python'
def job_name(role):return TASK+'-'+role


def command(argv, cwd=None):
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    require(result.returncode == 0, 'COMMAND_FAILED:' + shlex.join(argv) + ':' + result.stderr)
    return result.stdout.strip()


def launcher(source, commit, role, attempt,cpu=8):
    module = 'collect' if role == 'collector' else 'run'
    args = [PYTHON, '-u', '-m', 'project.run_scripts.jlz_price_alpha_writer.' + module,
            '--attempt', str(attempt)]
    if role != 'collector':
        args += ['--cell',role]
    threads=str(cpu)
    env = dict(PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=threads,
               MKL_NUM_THREADS=threads, TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', JLZ_ALPHA_WRITER_SOURCE_COMMIT=commit)
    if role == 'collector':
        env['CUDA_VISIBLE_DEVICES'] = ''
    script = '#!/bin/bash\nset -euo pipefail\n'
    script += ''.join('export ' + k + '=' + shlex.quote(v) + '\n' for k, v in env.items())
    script += 'cd ' + shlex.quote(str(source)) + '\nexec ' + shlex.join(args) + '\n'
    return script


def freeze(configpath, attempt, roles=ROLES):
    require(attempt.parent == LOCAL and not attempt.exists(), 'CREATE_ONCE_ATTEMPT')
    c = json.loads(configpath.read_text())
    require(c['instruction_id'] == NONCE and c['task_id'] == TASK, 'AUTHORITY')
    require(Path(c['attempt']) == attempt, 'PREPARATION_ATTEMPT')
    require(not command(['git', 'status', '--porcelain', '--', *SOURCES], ROOT), 'COMMIT_BEFORE_FREEZE')
    tested = json.loads(verify(c['cpu_preflight']).read_text())
    require(tested.get('passed') is True or tested.get('status') == 'PASS', 'CPU_PREFLIGHT_REQUIRED')
    for row in tested.get('source', []):
        verify(row)
    commit = command(['git', 'rev-parse', 'HEAD'], ROOT)
    tree = command(['git', 'rev-parse', 'HEAD^{tree}'], ROOT)
    attempt.mkdir()
    source = attempt / 'source'; source.mkdir()
    archive = attempt / 'source.tar'
    command(['git', 'archive', '--format=tar', '--output=' + str(archive), commit, *SOURCES], ROOT)
    with tarfile.open(archive) as tf:
        rows = tf.getmembers()
        require(len(rows) == len({r.name for r in rows}), 'ARCHIVE_DUPLICATE')
        require(all((r.isfile() or r.isdir()) and not Path(r.name).is_absolute()
                    and '..' not in Path(r.name).parts for r in rows), 'ARCHIVE_SAFE_REGULAR')
        tf.extractall(source, filter='data')
    for row in tested.get('source', []):
        rel = Path(row['path']).relative_to(ROOT)
        require(sha(source / rel) == row['sha256'], 'TESTED_ARCHIVE_CLOSURE')
    write(attempt / 'config.json', c)
    require(tuple(roles)==ROLES,'AUTHORIZED_SIX_CELLS')
    for role in roles:
        script = attempt / (role + '.sh')
        script.write_text(launcher(source, commit, role, attempt,c['resources']['collector_cpu'] if role=='collector' else c['resources']['cpu'])); script.chmod(0o755)
    write(attempt / 'execution.lock.json', dict(
        instruction_id=NONCE, task_id=TASK, source_commit=commit, source_tree=tree,
        archive=member(archive), source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        config_sha256=sha(attempt / 'config.json'), runtime_sources=c['runtime']['source_members'],tracking_env=member(c['tracking']['env_file']),
        dependency_sources=c.get('dependency_sources', []), native_reference=c['native_reference'],
        native_hparams=member(c['native_hparams']), launchers=[member(attempt / (r + '.sh')) for r in roles],
        owner=getpass.getuser(), host='server4', session=SESSION, resources=c['resources'],
        noCP=True, exact_resume='NOT_AVAILABLE', run_instance=c['run_instance'],
        profiles_sha256=__import__('project.run_scripts.jlz_interference_l1',fromlist=['digest']).digest({m:c['models'][m]['profiles'] for m in c['models']}),
        flow='Two model lanes, each CAP075 -> afterany CAP100 -> afterany FREE100; CPU afterany exact6; task/user cap2'))
    return verify_frozen(attempt)


def verify_frozen(attempt):
    lock = json.loads((attempt / 'execution.lock.json').read_text())
    c = json.loads((attempt / 'config.json').read_text())
    require(lock['owner'] == getpass.getuser() and lock['host'] == 'server4'
            and lock['instruction_id'] == c['instruction_id'] == NONCE, 'FROZEN_OWNER_AUTHORITY')
    require(sha(attempt / 'config.json') == lock['config_sha256'], 'CONFIG_SHA')
    for row in (lock['source_members'] + lock['runtime_sources'] + lock['native_reference']
                + lock['dependency_sources'] + lock['launchers'] + [lock['archive'], lock['native_hparams']]
                + c['authority_members'] + [lock['tracking_env']]):
        verify(row)
    for row in c['assets']:
        s = Path(row['path']).stat()
        require((s.st_size, s.st_ino, s.st_mtime_ns) == (row['bytes'], row['inode'], row['mtime_ns']), 'ASSET_STAT')
    require(shutil.disk_usage(attempt).free >= c['resources']['reserve_bytes'], 'RESOURCE_BLOCKED_STORAGE')
    return lock, c


def resource_inventory(exclude=()):
    # Pending jobs with implicit nodes are omitted by squeue -w. Read only the
    # own resource queue, then resolve node binding from scheduler metadata.
    raw = command(['squeue', '-h', '-r', '-u', getpass.getuser(), '-o', '%i|%u|%j|%T|%b|%R'])
    jobs = []
    for line in raw.splitlines():
        job, user, name, state, gres, reason = line.split('|', 5)
        if user != getpass.getuser() or job in exclude:
            continue
        detail = command(['scontrol', 'show', 'job', job, '--oneliner'])
        reqnode=re.search(r'\bReqNodeList=([^ ]+)',detail)
        allocation=re.search(r'\bNodeList=([^ ]+)',detail)
        bound=reqnode[1] if reqnode else '(null)';allocated=allocation[1] if allocation else '(null)'
        if bound not in ('(null)','ALL') and 'server4' not in bound:continue
        if state!='PENDING' and allocated not in ('(null)','ALL') and 'server4' not in allocated:continue
        requested = re.search(r'\bReqTRES=([^ ]+)', detail)
        gpu = re.search(r'gres/gpu=(\d+)', requested[1]) if requested else None
        if gpu:
            jobs.append(dict(job=job, user=user, name=name, state=state, gpus=int(gpu[1]),
                             reason=reason, resource_detail=detail,node_binding=bound,allocated_nodes=allocated,
                             implicit_pending_counted=(state=='PENDING' and bound in ('(null)','ALL'))))
    return dict(jobs=jobs, scope='현재 own GPU allocation/admitted pending의 보수적 자원 metadata만; 과학결과/로그/타job변경0')


def arguments(role, dep, attempt, r):
    cpu = role == 'collector'
    cpus=r['collector_cpu'] if cpu else r['cpu']
    wall = r['collector_wall'] if cpu else r['wall']
    argv = ['sbatch', '--parsable', '--hold', '--partition=gpu', '--qos=lab_gpu_s4',
            '--nodelist=server4', '--nodes=1', '--ntasks=1', '--cpus-per-task='+str(cpus), '--export=NONE',
            '--no-requeue', '--job-name=' + job_name(role), '--chdir=' + str(attempt / 'source'),
            '--mem=' + str(r['collector_host_mib'] if cpu else r['host_mib']) + 'M', '--time=' + wall,
            '--output=' + str(attempt / (role + '-%j.out')), '--error=' + str(attempt / (role + '-%j.err'))]
    if not cpu:
        argv += ['--gres=gpu:1']
    if dep:
        argv += ['--dependency=' + dep]
    return argv + [str(attempt / (role + '.sh'))]


def dependency_ids(text):
    """Slurm may print bare IDs or each ID followed by its fulfillment state."""
    return set(re.findall(r'(?:^|:)(\d+)(?=\(|:|\?|$)', text))


def inspect(job, role, dep, argv, attempt, r):
    detail = command(['scontrol', 'show', 'job', job, '--oneliner'])
    cpus=r['collector_cpu'] if role=='collector' else r['cpu']
    for term in [f'JobId={job} ', f'JobName={job_name(role)} ', 'UserId=' + getpass.getuser() + '(',
                 'JobState=PENDING ', 'Reason=JobHeldUser ', 'Requeue=0 ', f'CPUs/Task={cpus} ',
                 f'ReqTRES=cpu={cpus},', 'ReqNodeList=server4 ', 'Partition=gpu ', 'QOS=lab_gpu_s4 ']:
        require(term in detail, 'HELD:' + term)
    require(re.search(rf'\bNumCPUs={cpus}(?:-[0-9]+)? ', detail), 'HELD_CPU')
    require('gres/gpu' not in detail if role == 'collector' else 'TresPerNode=gres/gpu:1' in detail, 'HELD_GPU')
    mem = r['collector_host_mib'] if role == 'collector' else r['host_mib']
    require(f'mem={mem}M' in detail or f'mem={mem // 1024}G' in detail, 'HELD_MEMORY')
    wall = r['collector_wall'] if role == 'collector' else r['wall']
    require('TimeLimit=' + wall + ' ' in detail, 'HELD_WALL')
    observed = re.search(r'\bDependency=([^ ]+)', detail)
    expected = set(dep.split(':')[1:]) if dep else set()
    require(observed is not None, 'HELD_DEPENDENCY_FIELD')
    got = set() if observed[1] == '(null)' else dependency_ids(observed[1])
    require(got == expected and (not dep or observed[1].startswith(dep.split(':')[0] + ':')), 'HELD_DEPENDENCY')
    script = attempt / (role + '.sh')
    require('Command=' + str(script) + ' ' in detail and 'WorkDir=' + str(attempt / 'source') + ' ' in detail, 'HELD_SOURCE')
    submit = re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)', detail)
    require(submit and shlex.split(submit[1]) == argv, 'HELD_FULL_ARGV')
    require(command(['scontrol', 'write', 'batch_script', job, '-']).strip() == script.read_text().strip(), 'HELD_SCRIPT_BYTES')
    return dict(job=job, role=role, argv=argv, resource_detail=detail, launcher=member(script))


def submit(config,attempt):
    # This function is invoked by the owning root only. Tests never call a
    # scheduler mutation; no automatic retry/cancel is implemented.
    require(not list(LOCAL.glob('*/submission.json')) and not list(LOCAL.glob('*/submitted-*.json')),'NO_DUPLICATE_REGISTRATION')
    queue=command(['squeue','-h','-u',getpass.getuser(),'--name='+','.join(job_name(r) for r in ROLES),'-o','%i|%j|%T'])
    require(not queue,'EXACT_TASK_ALREADY_REGISTERED')
    before=resource_inventory()
    from .predecessor import bind
    predecessor=bind()
    caps=Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines()
    localrow=next(x for x in caps if x.startswith('server4\t')).split('\t')
    local=int(localrow[2]);local_memory=int(localrow[3])
    tracked=int(next(x for x in (ROOT/'control/gpu-concurrency-policy.tsv').read_text().splitlines() if x.startswith('server4\t')).split('\t')[1])
    cap=min(2,local,tracked);require(cap>=1,'NO_CAP')
    parallel=min(2,cap)
    node=command(['scontrol','show','node','server4']);partition=command(['scontrol','show','partition','gpu'])
    lock,c=freeze(Path(config).resolve(),attempt);r=c['resources']
    require(r['host_mib']==59392 and r['hard_host_mib']==60416 and r['collector_host_mib']==24576
        and 1<=r['cpu']<=8 and r['task_cap']==2 and r['wall']=='2-00:00:00'
        and r['collector_wall']=='04:00:00','SEALED_RESOURCES')
    memory_policy=command(['python3',str(ROOT/'scripts/slurm_memory_policy.py'),'request',
        '--server','server4','--gpus','1','--mem',str(r['host_mib'])+'M',
        '--local-limit-mib-per-gpu',str(local_memory)])
    require('ALLOW_MEMORY_POLICY' in memory_policy,'HARD_MEMORY_POLICY')
    helper=subprocess.run(['bash',str(ROOT/'scripts/check-slurm-resource-cap.sh'),'server4','1','59392M'],
        env=dict(os.environ,AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'),capture_output=True,text=True)
    require(helper.returncode in (0,4) and 'DENY_MEMORY' not in helper.stdout+helper.stderr,'RESOURCE_HELPER_BLOCK')
    external=[j['job'] for j in before['jobs']]
    previous=set(predecessor['jobs'])
    others=[j for j in external if j not in previous]
    barrier=list(dict.fromkeys(predecessor['frontier']+others))
    order=resource_order(parallel);ids={};mapping={};held=[]
    scheduler_dry_checks=[]
    for role in ('LLAMA_AE_CAP075','QWEN_AE_CAP075','collector'):
        checkargv=[x for x in arguments(role,None,attempt,r) if x not in ('--parsable','--hold')]
        checkargv.insert(1,'--test-only')
        checked=subprocess.run(checkargv,text=True,capture_output=True)
        scheduler_dry_checks.append(dict(role=role,argv=checkargv,returncode=checked.returncode,
            stdout=checked.stdout,stderr=checked.stderr,allocation_created=False))
        require(checked.returncode==0,'SCHEDULER_RESOURCE_BINDING_REJECTED:'+checked.stderr)
    write(attempt/'scheduler-resource-check.json',dict(checks=scheduler_dry_checks,allocation_created=False))
    for role in ROLES:
        parents=[ids[r] for r in order[role]] if order[role] else barrier
        dep='afterany:'+':'.join(parents) if parents else None
        argv=arguments(role,dep,attempt,r);job=command(argv).split(';')[0];require(job.isdigit(),'JOB_ID')
        ids[role]=job;mapping[role]=dict(job=job,dependency=dep,argv=argv,dependency_role='afterany resource ordering only; each arm own integrated B1 checks; no quality gate')
        write(attempt/('submitted-'+role+'.json'),dict(nonce=NONCE,role=role,status='HELD',**mapping[role]))
        held.append(inspect(job,role,dep,argv,attempt,r))
    current=resource_inventory(tuple(ids.values()))
    require({j['job'] for j in current['jobs']}<=set(external),'ADMISSION_RACE_KEEP_HELD')
    require(bool(barrier) or sum(j['gpus'] for j in current['jobs'])+parallel<=cap,'CAP_RACE_KEEP_HELD')
    verify_frozen(attempt)
    write(attempt/'held-inspection.json',dict(jobs=held,before=before,prerelease=current,effective_project_cap=cap,
        predecessor=predecessor,all_predecessors_before_both_Alpha_lanes=True,
        task_cap=2,maximum_new_GPU_concurrency=parallel,external_resource_afterany=barrier,node=node,partition=partition,
        helper=dict(code=helper.returncode,output=helper.stdout+helper.stderr),memory_policy=memory_policy,
        helper_prefix_exception='jlz-price-alpha-writer-2k outside generic prefix: explicitauthority + all owned GPU/admitted inventory; helperproject-count not PASS',
        scheduler_dry_checks=scheduler_dry_checks,all_held_inspected_before_release=True,
        unrelated_job_mutations=0,CPU_binding=r['cpu']))
    for role in reversed(ROLES):
        output=command(['scontrol','release',ids[role]])
        write(attempt/('released-'+role+'.json'),dict(job=ids[role],command_succeeded=True,result=output))
    snapshot=command(['squeue','-h','-j',','.join(ids.values()),'-o','%i|%j|%T|%b|%N|%r'])
    write(attempt/'submission.json',dict(nonce=NONCE,task_id=TASK,status='RELEASED',jobs=ids,mapping=mapping,
        source=lock['source_commit'],lock=member(attempt/'execution.lock.json'),held=member(attempt/'held-inspection.json'),
        bounded_initial_snapshot=snapshot,GPU_qualification='INTEGRATED_ACTUAL_B1_NOT_OBSERVED',six_cell_B1_B2='NOT_OBSERVED',
        W20='NOT_OBSERVED',monitoring_active=False,automatic_resume=False,automatic_retry=False))
    return dict(status='RELEASED',jobs=ids,initial_snapshot=snapshot)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--freeze-only',action='store_true')
    a=p.parse_args();print(json.dumps(freeze(a.config.resolve(),a.attempt.resolve())[0] if a.freeze_only else submit(a.config,a.attempt.resolve())))
