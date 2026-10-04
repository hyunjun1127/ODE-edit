"""Two parallel, create-once GPU lanes and CPU afterany collector; no polling."""
import argparse, getpass, json, os, re, shlex, shutil, subprocess, tarfile
from pathlib import Path
from project.run_scripts.jlz_writer_coupled.submit import command, launcher, _copy_once, _write_script, dependencies, dependency_members, expected_dependencies
from project.run_scripts.jlz_shared_budget.submit import resource_inventory
from project.run_scripts.jlz_realized_writer.submit import SOURCES as B1_SOURCES
from .common import *

SOURCES = list(B1_SOURCES) + ['project/run_scripts/jlz_realized_writer_sequential', CONTRACT, ENVELOPE, EXCEPTION]
STAGES = ARMS + ('collector',)

def freeze(configpath, attempt, cpu_receipt):
    require(attempt.parent == LOCAL and attempt.name.startswith('attempt-') and not attempt.exists(), 'CREATE_ONCE_ATTEMPT')
    require(configpath.is_relative_to(LOCAL), 'TASK_LOCAL_CONFIG')
    require(not command(['git', 'status', '--porcelain', '--', *SOURCES], cwd=ROOT), 'COMMIT_BEFORE_FREEZE')
    c = json.loads(configpath.read_text()); require(c['instruction_id'] == NONCE and c['task_id'] == TASK, 'NONCE_TASK')
    require(cpu_receipt.is_relative_to(LOCAL), 'CPU_RECEIPT_TASK_SCOPE')
    tested = json.loads(cpu_receipt.read_text()); require(tested['passed'], 'FINAL_CPU_CHECKED')
    for row in tested['source']:
        verify(row)
    c['cpu_preflight_original'] = c['cpu_preflight']
    c['cpu_preflight'] = member(cpu_receipt)
    commit = command(['git', 'rev-parse', 'HEAD'], cwd=ROOT); tree = command(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT)
    attempt.mkdir(); source = attempt / 'source'; source.mkdir(); archive = attempt / 'source.tar'
    command(['git', 'archive', '--format=tar', '--output=' + str(archive), commit, *SOURCES], cwd=ROOT)
    with tarfile.open(archive) as t:
        members = t.getmembers()
        require(len({m.name for m in members}) == len(members), 'DUPLICATE_SOURCE_MEMBER')
        require(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members), 'SAFE_SOURCE_ARCHIVE')
        t.extractall(source, filter='data')
    write(attempt / 'config.json', c)
    for stage in STAGES:
        args = ['--attempt', str(attempt)] + ([] if stage == 'collector' else ['--arm', stage])
        _write_script(attempt / (stage + '.sh'), launcher(source, commit,
            'project.run_scripts.jlz_realized_writer_sequential.' + ('collect' if stage == 'collector' else 'run'), args, cpu_only=stage == 'collector'))
    oldlock = json.loads((PRIOR / 'execution.lock.json').read_text())
    write(attempt / 'execution.lock.json', dict(instruction_id=NONCE, task_id=TASK, source_commit=commit, source_tree=tree,
        archive=member(archive), source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        config_sha256=sha(attempt / 'config.json'), original_preparation_config=member(configpath),
        CPU_expanded_revision=member(cpu_receipt), runtime_sources=c['runtime']['source_members'], native_reference=c['native_reference'],
        native_hparams=member(c['native_hparams']), dependency_sources=oldlock['dependency_sources'],
        launchers=[member(attempt / (n + '.sh')) for n in STAGES], owner=getpass.getuser(), host='server4',
        session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd', resources=c['resources'], noCP=True,
        fits_each=20, total_fits=40, arms=list(ARMS), flow='parallel cold MD20 and CD20; afterany both CPU collector'))

def verify_frozen(attempt):
    lock = json.loads((attempt / 'execution.lock.json').read_text()); c = json.loads((attempt / 'config.json').read_text())
    require(lock['owner'] == getpass.getuser() and lock['instruction_id'] == c['instruction_id'] == NONCE, 'OWNER_NONCE')
    require(sha(attempt / 'config.json') == lock['config_sha256'], 'FROZEN_CONFIG')
    for row in lock['source_members'] + lock['runtime_sources'] + lock['dependency_sources'] + lock['native_reference'] + lock['launchers'] + [lock['archive'], lock['native_hparams']]:
        verify(row)
    for row in c['authority_members'] + c['qualification_reuse']['receipts'] + [c['observer_identity'], c['native_input_alignment'], c['cpu_preflight']]:
        verify(row)
    for row in c['assets']:
        s = Path(row['path']).stat()
        require((s.st_size, s.st_ino, s.st_mtime_ns) == (row['bytes'], row['inode'], row['mtime_ns']), 'ASSET_STAT')
    require(shutil.disk_usage(attempt).free >= c['resources']['reserve_bytes'], 'STORAGE_RESERVE')
    return lock, c

def jobname(stage):
    return 'odeedit_jlz_v13_mdcd_s4_' + stage

def argv(stage, dep, attempt, r):
    cpu = stage == 'collector'
    result = ['sbatch', '--parsable', '--hold', '--partition=gpu', '--qos=lab_gpu_s4', '--nodelist=server4', '--nodes=1', '--ntasks=1',
        '--cpus-per-task=8', '--export=NONE', '--no-requeue', '--job-name=' + jobname(stage), '--chdir=' + str(attempt / 'source'),
        '--mem=' + str(r['collector_host_mib'] if cpu else r['host_mib']) + 'M', '--time=' + r['collector_wall' if cpu else 'wall'],
        '--output=' + str(attempt / (stage + '-%j.out')), '--error=' + str(attempt / (stage + '-%j.err'))]
    if not cpu:
        result += ['--gres=gpu:1']
    if dep:
        result += ['--dependency=' + dep]
    return result + [str(attempt / (stage + '.sh'))]

def inspect(job, stage, dep, args, attempt, r):
    detail = command(['scontrol', 'show', 'job', job, '--oneliner']); cpu = stage == 'collector'
    for term in [f'JobId={job} ', f'JobName={jobname(stage)} ', 'UserId=' + getpass.getuser() + '(',
        'JobState=PENDING ', 'Reason=JobHeldUser ', 'Requeue=0 ', 'CPUs/Task=8 ', 'ReqTRES=cpu=8,',
        'ReqNodeList=server4 ', 'Partition=gpu ', 'QOS=lab_gpu_s4 ']:
        require(term in detail, 'HELD:' + term)
    require(re.search(r'\bNumCPUs=8(?:-[0-9]+)? ', detail), 'HELD_CPU')
    require('gres/gpu' not in detail if cpu else 'TresPerNode=gres/gpu:1' in detail, 'HELD_GPU')
    mem = r['collector_host_mib' if cpu else 'host_mib']; wall = r['collector_wall' if cpu else 'wall']
    require(f'mem={mem}M' in detail or f'mem={mem // 1024}G' in detail, 'HELD_MEMORY')
    require('TimeLimit=' + wall + ' ' in detail, 'HELD_WALL')
    field = re.search(r'\bDependency=([^ ]+)', detail)
    actual = set() if field and field[1] == '(null)' else dependency_members(field[1]) if field else None
    require(actual == expected_dependencies(dep), 'HELD_DEPENDENCY')
    script = attempt / (stage + '.sh')
    require('Command=' + str(script) + ' ' in detail and 'WorkDir=' + str(attempt / 'source') + ' ' in detail, 'HELD_SOURCE')
    submit = re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)', detail)
    require(submit and shlex.split(submit[1]) == args, 'HELD_FULL_ARGV')
    require(command(['scontrol', 'write', 'batch_script', job, '-']).strip() == script.read_text().strip(), 'HELD_SCRIPT_BYTES')
    return dict(job=job, stage=stage, argv=args, detail=detail, launcher=member(script))

def admission_plan(inventory, cap):
    require(cap >= 2, 'RESOURCE_CAP_BELOW_APPROVED_PARALLEL_TWO_LANES')
    occupied = sum(j['gpus'] for j in inventory['jobs'])
    # If project-admitted capacity is occupied, BOTH arms wait on the SAME
    # resource barrier. Neither depends on the other's scientific result.
    return [j['job'] for j in inventory['jobs']] if occupied + 2 > cap else []

def verify_parallel_dependencies(ids, mapping):
    # dependencies() returns None for an empty parent list.
    md = expected_dependencies(mapping['MD']['dependency'])
    cd = expected_dependencies(mapping['CD']['dependency'])
    require(not any(job == ids['CD'] for _, job in md) and not any(job == ids['MD'] for _, job in cd), 'NO_ARM_SERIALIZATION')
    require(md == cd, 'IDENTICAL_EXTERNAL_RESOURCE_BARRIER')
    require(expected_dependencies(mapping['collector']['dependency']) == {('afterany', ids[arm]) for arm in ARMS}, 'COLLECTOR_AFTERANY_BOTH')
    return md

def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True); p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--cpu-receipt', type=Path, required=True)
    args = p.parse_args()
    require(not list(LOCAL.glob('attempt-*/submitted-*.json')) and not list(LOCAL.glob('attempt-*/submission.json')), 'PRIOR_REGISTRATION_NO_DUPLICATE')
    queue = command(['squeue', '-h', '-r', '-u', getpass.getuser(), '-o', '%i|%j|%T'])
    require('odeedit_jlz_v13_mdcd_s4_' not in queue, 'DUPLICATE_MD_CD_JOB')
    before = resource_inventory()
    caps = Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text()
    local = int(next(s for s in caps.splitlines() if s.startswith('server4\t')).split('\t')[2])
    tracked = int(next(s for s in (ROOT / 'control/gpu-concurrency-policy.tsv').read_text().splitlines() if s.startswith('server4\t')).split('\t')[1])
    cap = min(3, local, tracked); barrier = admission_plan(before, cap)
    helper = subprocess.run(['bash', str(ROOT / 'scripts/check-slurm-resource-cap.sh'), 'server4', '2', '59392M'],
        env=dict(os.environ, AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'), capture_output=True, text=True)
    require(helper.returncode in (0, 4), 'RESOURCE_HELPER:' + helper.stdout + helper.stderr)
    require(helper.returncode == 0 or barrier, 'PROJECT_HELPER_UNRESOLVED_PENDING_CAPACITY')
    partition = command(['scontrol', 'show', 'partition', 'gpu']); node = command(['scontrol', 'show', 'node', 'server4'])
    require('MaxTime=30-00:00:00' in partition, 'PARTITION_LIMIT_CHANGED')
    attempt = args.attempt.resolve(); freeze(args.config.resolve(), attempt, args.cpu_receipt.resolve()); lock, c = verify_frozen(attempt); r = c['resources']
    require(r['host_mib'] == 59392 and r['task_cap'] == 2 and r['wall'] == '1-00:00:00', 'SEALED_RESOURCES')
    ids = {}; mapping = {}; checks = []
    for stage in STAGES:
        parents = barrier if stage in ARMS else [ids[arm] for arm in ARMS]
        dep = dependencies(('afterany', parents)); argsv = argv(stage, dep, attempt, r)
        job = command(argsv).split(';')[0]; require(job.isdigit(), 'JOB_ID'); ids[stage] = job
        mapping[stage] = dict(job=job, dependency=dep, argv=argsv)
        write(attempt / ('submitted-' + stage + '.json'), dict(nonce=NONCE, status='HELD', **mapping[stage]))
        checks.append(inspect(job, stage, dep, argsv, attempt, r))
    after = resource_inventory(tuple(ids.values())); external = {j['job'] for j in before['jobs']}
    require({j['job'] for j in after['jobs']} <= external, 'ADMISSION_RACE_KEEP_HELD')
    require(barrier or sum(j['gpus'] for j in after['jobs']) + 2 <= cap, 'PROJECT_CAP_KEEP_HELD')
    verify_parallel_dependencies(ids, mapping)
    verify_frozen(attempt)
    write(attempt / 'held-inspection.json', dict(checks=checks, before=before, prerelease=after, project_cap=cap, task_cap=2,
        aggregate_new_GPU=2, aggregate_new_CPU=16, aggregate_new_host_mib=118784, identical_resource_barrier=barrier,
        partition=partition, node=node, helper=dict(rc=helper.returncode, output=helper.stdout + helper.stderr),
        all_held_before_release=True, no_other_job_mutation=True, arm_science_dependency=False))
    for stage in reversed(STAGES):
        result = command(['scontrol', 'release', ids[stage]])
        write(attempt / ('released-' + stage + '.json'), dict(job=ids[stage], success=True, result=result))
    write(attempt / 'submission.json', dict(nonce=NONCE, status='RELEASED', jobs=ids, mapping=mapping, source=lock['source_commit'],
        lock=member(attempt / 'execution.lock.json'), held=member(attempt / 'held-inspection.json'),
        main_initial='NOT_OBSERVED', actual_new_GPU='NOT_OBSERVED', monitoring_active=False, automatic_resume=False))
    print(json.dumps(dict(status='RELEASED', jobs=ids, parallel_arms=True)))

if __name__ == '__main__':
    main()
