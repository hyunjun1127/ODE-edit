"""Create-once source freeze and cap1 held-inspected qualification/main DAG."""
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
from . import ROOT, LOCAL, TASK, NONCE, require, write, member, sha, verify

SOURCES = [
    'project/run_scripts/causal_allocation_editing',
    'project/run_scripts/jlz_native_writer_aware',
    'project/run_scripts/jlz_realized_subject',
    'project/run_scripts/jlz_realized_writer',
    'project/run_scripts/jlz_realized_writer_sequential/review_completed.py',
    'project/run_scripts/jlz_shared_budget',
    'project/run_scripts/jlz_realization',
    'project/run_scripts/jlz_writer_coupled',
    'project/run_scripts/jlz_pilot/prompts.py', 'project/run_scripts/jlz_pilot/__init__.py',
    'project/run_scripts/jlz_two_arm/baseline_pilot.py',
    'project/run_scripts/jlz_two_arm/common.py', 'project/run_scripts/jlz_two_arm/__init__.py',
    'project/run_scripts/memit_history_lifelong/hparams.json',
    'scripts/fixed_counterfact.py', 'scripts/check-slurm-resource-cap.sh',
    'scripts/check-slurm-gpu-cap.sh', 'scripts/slurm_memory_policy.py',
    'servers/slurm-memory-policy.tsv', 'control/gpu-concurrency-policy.tsv',
    'plans/global/2026-10-06-jlz-causal-joint-writer',
    'plans/global/causal-allocation-editing',
    'plans/global/2026-10-05-jlz-v13-realized-writer/method.tex',
    'plans/global/2026-10-04-jlz-v12-marginal-allocation/experiment-2k/case-schedule-first2000.csv',
    'messages/head/causal-allocation-editing.json',
    'control/experiment-exceptions/causal-allocation-editing.json',
]
ROLES = ('qualification', 'main', 'collector')
SESSION = '01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
PYTHON = '/data/janghj/EasyEdit/.venv/bin/python'


def command(argv, cwd=None):
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    require(result.returncode == 0, 'COMMAND_FAILED:' + shlex.join(argv) + ':' + result.stderr)
    return result.stdout.strip()


def launcher(source, commit, role, attempt):
    module = 'collect' if role == 'collector' else 'run'
    args = [PYTHON, '-u', '-m', 'project.run_scripts.causal_allocation_editing.' + module,
            '--attempt', str(attempt)]
    if role != 'collector':
        args += ['--qualification-only' if role == 'qualification' else '--main-only']
    env = dict(PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8',
               MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', CAUSAL_ALLOCATION_EDITING_SOURCE_COMMIT=commit)
    if role == 'collector':
        env['CUDA_VISIBLE_DEVICES'] = ''
    script = '#!/bin/bash\nset -euo pipefail\n'
    script += ''.join('export ' + k + '=' + shlex.quote(v) + '\n' for k, v in env.items())
    script += 'cd ' + shlex.quote(str(source)) + '\nexec ' + shlex.join(args) + '\n'
    return script


def freeze(configpath, attempt):
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
    for role in ROLES:
        script = attempt / (role + '.sh')
        script.write_text(launcher(source, commit, role, attempt)); script.chmod(0o755)
    write(attempt / 'execution.lock.json', dict(
        instruction_id=NONCE, task_id=TASK, source_commit=commit, source_tree=tree,
        archive=member(archive), source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        config_sha256=sha(attempt / 'config.json'), runtime_sources=c['runtime']['source_members'],
        dependency_sources=c.get('dependency_sources', []), native_reference=c['native_reference'],
        native_hparams=member(c['native_hparams']), launchers=[member(attempt / (r + '.sh')) for r in ROLES],
        owner=getpass.getuser(), host='server4', session=SESSION, resources=c['resources'],
        noCP=True, exact_resume='NOT_AVAILABLE', run_instance=c['run_instance'],
        flow='qualification -> afterany READY-before-model-load fresh cold main calibration/20 commits -> afterany CPU collector'))
    return verify_frozen(attempt)


def verify_frozen(attempt):
    lock = json.loads((attempt / 'execution.lock.json').read_text())
    c = json.loads((attempt / 'config.json').read_text())
    require(lock['owner'] == getpass.getuser() and lock['host'] == 'server4'
            and lock['instruction_id'] == c['instruction_id'] == NONCE, 'FROZEN_OWNER_AUTHORITY')
    require(sha(attempt / 'config.json') == lock['config_sha256'], 'CONFIG_SHA')
    for row in (lock['source_members'] + lock['runtime_sources'] + lock['native_reference']
                + lock['dependency_sources'] + lock['launchers'] + [lock['archive'], lock['native_hparams']]
                + c['authority_members']):
        verify(row)
    for row in c['assets']:
        s = Path(row['path']).stat()
        require((s.st_size, s.st_ino, s.st_mtime_ns) == (row['bytes'], row['inode'], row['mtime_ns']), 'ASSET_STAT')
    require(shutil.disk_usage(attempt).free >= c['resources']['reserve_bytes'], 'RESOURCE_BLOCKED_STORAGE')
    return lock, c


def resource_inventory(exclude=()):
    raw = command(['squeue', '-h', '-r', '-w', 'server4', '-o', '%i|%u|%j|%T|%b|%R'])
    jobs = []
    for line in raw.splitlines():
        job, user, name, state, gres, reason = line.split('|', 5)
        if user != getpass.getuser() or job in exclude:
            continue
        detail = command(['scontrol', 'show', 'job', job, '--oneliner'])
        requested = re.search(r'\bReqTRES=([^ ]+)', detail)
        gpu = re.search(r'gres/gpu=(\d+)', requested[1]) if requested else None
        if gpu:
            jobs.append(dict(job=job, user=user, name=name, state=state, gpus=int(gpu[1]),
                             reason=reason, resource_detail=detail))
    return dict(jobs=jobs, scope='현재 own GPU allocation/admitted pending의 보수적 자원 metadata만; 과학결과/로그/타job변경0')


def arguments(role, dep, attempt, r):
    cpu = role == 'collector'
    wall = r['collector_wall'] if cpu else r['qualification_wall'] if role == 'qualification' else r['wall']
    argv = ['sbatch', '--parsable', '--hold', '--partition=gpu', '--qos=lab_gpu_s4',
            '--nodelist=server4', '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--export=NONE',
            '--no-requeue', '--job-name=' + TASK, '--chdir=' + str(attempt / 'source'),
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
    for term in [f'JobId={job} ', f'JobName={TASK} ', 'UserId=' + getpass.getuser() + '(',
                 'JobState=PENDING ', 'Reason=JobHeldUser ', 'Requeue=0 ', 'CPUs/Task=8 ',
                 'ReqTRES=cpu=8,', 'ReqNodeList=server4 ', 'Partition=gpu ', 'QOS=lab_gpu_s4 ']:
        require(term in detail, 'HELD:' + term)
    require(re.search(r'\bNumCPUs=8(?:-[0-9]+)? ', detail), 'HELD_CPU')
    require('gres/gpu' not in detail if role == 'collector' else 'TresPerNode=gres/gpu:1' in detail, 'HELD_GPU')
    mem = r['collector_host_mib'] if role == 'collector' else r['host_mib']
    require(f'mem={mem}M' in detail or f'mem={mem // 1024}G' in detail, 'HELD_MEMORY')
    wall = r['collector_wall'] if role == 'collector' else r['qualification_wall'] if role == 'qualification' else r['wall']
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


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--attempt', type=Path, required=True); args = p.parse_args()
    attempt = args.attempt.resolve()
    require(not list(LOCAL.glob('*/submitted-*.json')) and not list(LOCAL.glob('*/submission.json')), 'NO_DUPLICATE_REGISTRATION')
    queue = command(['squeue', '-h', '-u', getpass.getuser(), '--name=' + TASK, '-o', '%i|%j|%T'])
    require(not queue, 'EXISTING_EXACT_TASK_JOB')
    before = resource_inventory()
    caps = Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines()
    local = int(next(r for r in caps if r.startswith('server4\t')).split('\t')[2])
    tracked = int(next(r for r in (ROOT / 'control/gpu-concurrency-policy.tsv').read_text().splitlines()
                       if r.startswith('server4\t')).split('\t')[1])
    cap = min(3, local, tracked); require(cap >= 1, 'NO_CAP')
    helper = subprocess.run(['bash', str(ROOT / 'scripts/check-slurm-resource-cap.sh'), 'server4', '1', '59392M'],
                            env=dict(os.environ, AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'),
                            capture_output=True, text=True)
    require(helper.returncode in (0, 4), 'RESOURCE_HELPER:' + helper.stdout + helper.stderr)
    node = command(['scontrol', 'show', 'node', 'server4'])
    partition = command(['scontrol', 'show', 'partition', 'gpu'])
    require('MaxTime=30-00:00:00' in partition, 'PARTITION_CEILING')
    lock, c = freeze(args.config.resolve(), attempt); r = c['resources']
    require(r['host_mib'] == 59392 and r['hard_host_mib'] == 60416 and r['task_cap'] == 1
            and r['collector_host_mib'] == 24576
            and r['wall'] == '2-00:00:00' and r['qualification_wall'] == '04:00:00'
            and r['collector_wall'] == '04:00:00', 'SEALED_RESOURCE')
    external = [j['job'] for j in before['jobs']]
    barrier = external if sum(j['gpus'] for j in before['jobs']) + 1 > cap else []
    ids, mapping, held = {}, {}, []
    for role in ROLES:
        dep = ('afterany:' + ':'.join(barrier)) if role == 'qualification' and barrier else None
        # READY is checked before any model load.  afterany makes a failed
        # qualification produce a typed main failure and lets the collector
        # finish, instead of leaving an afterok main pending indefinitely.
        if role == 'main': dep = 'afterany:' + ids['qualification']
        if role == 'collector': dep = 'afterany:' + ids['qualification'] + ':' + ids['main']
        argv = arguments(role, dep, attempt, r)
        job = command(argv).split(';')[0]; require(job.isdigit(), 'JOB_ID')
        ids[role] = job; mapping[role] = dict(job=job, dependency=dep, argv=argv)
        write(attempt / ('submitted-' + role + '.json'), dict(nonce=NONCE, role=role, status='HELD', **mapping[role]))
        held.append(inspect(job, role, dep, argv, attempt, r))
    prerelease = resource_inventory(tuple(ids.values()))
    require({j['job'] for j in prerelease['jobs']} <= set(external), 'ADMISSION_RACE_KEEP_HELD')
    require(bool(barrier) or sum(j['gpus'] for j in prerelease['jobs']) + 1 <= cap, 'CAP_RACE_KEEP_HELD')
    verify_frozen(attempt)
    write(attempt / 'held-inspection.json', dict(jobs=held, before=before, prerelease=prerelease,
        effective_project_cap=cap, task_cap=1, maximum_new_GPU_concurrency=1, external_resource_afterany=barrier,
        node=node, partition=partition, helper=dict(code=helper.returncode, output=helper.stdout + helper.stderr),
        helper_naming_exception='기존 helper projectprefix가 새 이름을 포괄하지 않아 모든 ownGPU자원/admittedpending를 별도 보수적 집계; 공유helper편집0',
        all_held_inspected_before_release=True, other_job_mutations=0))
    for role in reversed(ROLES):
        output = command(['scontrol', 'release', ids[role]])
        write(attempt / ('released-' + role + '.json'), dict(job=ids[role], command_succeeded=True, result=output))
    snapshot = command(['squeue', '-h', '-j', ','.join(ids.values()), '-o', '%i|%j|%T|%b|%N|%r'])
    write(attempt / 'submission.json', dict(nonce=NONCE, task_id=TASK, status='RELEASED', jobs=ids, mapping=mapping,
        source=lock['source_commit'], lock=member(attempt / 'execution.lock.json'),
        held=member(attempt / 'held-inspection.json'), bounded_initial_snapshot=snapshot,
        GPU_qualification='NOT_OBSERVED', main_B1_B2='NOT_OBSERVED', W20='NOT_OBSERVED',
        monitoring_active=False, automatic_resume=False, automatic_retry=False))
    print(json.dumps(dict(status='RELEASED', jobs=ids, initial_snapshot=snapshot)))


if __name__ == '__main__':
    main()
