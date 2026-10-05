"""Create-once scoped registration, frozen source, held audit, then release."""
import argparse, getpass, json, os, re, shutil, subprocess, tarfile
from pathlib import Path
from project.run_scripts.jlz_writer_coupled.submit import command, launcher, _write_script, dependencies
from project.run_scripts.jlz_shared_budget.submit import resource_inventory
from project.run_scripts.jlz_realized_writer.submit import SOURCES as BASE_SOURCES
from project.run_scripts.jlz_realized_writer_sequential.submit import inspect as old_inspect
from .common import *

SOURCES = list(BASE_SOURCES) + ['project/run_scripts/jlz_realized_writer_sequential',
    'project/run_scripts/jlz_cd_cumulative', DESIGN, ENVELOPE, EXCEPTION]
STAGES = ARMS + ('collector',)

def jobname(stage):
    return 'odeedit_jlz_cd_cumulative_s4_' + stage

def arguments(stage, dep, attempt, r):
    cpu = stage == 'collector'
    argv = ['sbatch', '--parsable', '--hold', '--partition=gpu', '--qos=lab_gpu_s4',
        '--nodelist=server4', '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--export=NONE',
        '--no-requeue', '--job-name=' + jobname(stage), '--chdir=' + str(attempt / 'source'),
        '--mem=' + str(r['collector_host_mib'] if cpu else r['host_mib']) + 'M',
        '--time=' + r['collector_wall' if cpu else 'wall'],
        '--output=' + str(attempt / (stage + '-%j.out')), '--error=' + str(attempt / (stage + '-%j.err'))]
    if not cpu:
        argv.append('--gres=gpu:1')
    if dep:
        argv.append('--dependency=' + dep)
    return argv + [str(attempt / (stage + '.sh'))]

def inspect(job, stage, dep, argv, attempt, resources):
    # The existing helper is a read-only general scheduler parser. Its globals
    # are changed only in this NEW submit process, never archived old jobs.
    import project.run_scripts.jlz_realized_writer_sequential.submit as helper
    original = helper.jobname
    try:
        helper.jobname = jobname
        return old_inspect(job, stage, dep, argv, attempt, resources)
    finally:
        helper.jobname = original

def freeze(configpath, attempt):
    require(attempt.parent == LOCAL and attempt.name.startswith('attempt-') and not attempt.exists(), 'CREATE_ONCE_ATTEMPT')
    require(not command(['git', 'status', '--porcelain', '--', *SOURCES], cwd=ROOT), 'COMMIT_BEFORE_FREEZE')
    c = json.loads(configpath.read_text())
    require(c['instruction_id'] == NONCE and c['task_id'] == TASK, 'NONCE')
    tested = json.loads(verify(c['cpu_preflight']).read_text())
    require(tested['passed'], 'FINAL_CPU_FAILED')
    for row in tested['source']:
        verify(row)
    commit = command(['git', 'rev-parse', 'HEAD'], cwd=ROOT); tree = command(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT)
    attempt.mkdir(); source = attempt / 'source'; source.mkdir(); archive = attempt / 'source.tar'
    command(['git', 'archive', '--format=tar', '--output=' + str(archive), commit, *SOURCES], cwd=ROOT)
    with tarfile.open(archive) as f:
        rows = f.getmembers()
        require(len(rows) == len({r.name for r in rows}), 'SOURCE_DUPLICATE')
        require(all((r.isfile() or r.isdir()) and not Path(r.name).is_absolute() and '..' not in Path(r.name).parts for r in rows), 'SAFE_SOURCE')
        f.extractall(source, filter='data')
    for row in tested['source']:
        relative = Path(row['path']).relative_to(ROOT)
        require(sha(source / relative) == row['sha256'], 'CPU_TESTED_SOURCE_ARCHIVE_CLOSURE')
    write(attempt / 'config.json', c)
    for stage in STAGES:
        args = ['--attempt', str(attempt)] + ([] if stage == 'collector' else ['--arm', stage])
        _write_script(attempt / (stage + '.sh'), launcher(source, commit,
            'project.run_scripts.jlz_cd_cumulative.' + ('collect' if stage == 'collector' else 'run'),
            args, cpu_only=stage == 'collector'))
    oldlock = json.loads(verify(c['prior_execution']).read_text())
    write(attempt / 'execution.lock.json', dict(instruction_id=NONCE, task_id=TASK, source_commit=commit,
        source_tree=tree, archive=member(archive), source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        config_sha256=sha(attempt / 'config.json'), preparation=member(configpath), runtime_sources=c['runtime']['source_members'],
        dependency_sources=oldlock['dependency_sources'], native_reference=c['native_reference'],
        native_hparams=member(c['native_hparams']), launchers=[member(attempt / (n + '.sh')) for n in STAGES],
        owner=getpass.getuser(), host='server4', session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd', resources=c['resources'],
        effective_resource_override='canonical121856MiB -> default59392/hard60416 S4 without canonical-byte edits',
        noCP=True, arms=list(ARMS), fits_each=20, qualification='2 native requests fixed-candidate; no fit/update',
        lambda_lock='FIRST_NONZERO_MAIN_B1_SHARED_ONCE_NO_EXTRA_FIT'))

def verify_frozen(attempt):
    from .run import locked
    lock = json.loads((attempt / 'execution.lock.json').read_text())
    previous = os.environ.get('ODEEDIT_SOURCE_COMMIT')
    try:
        os.environ['ODEEDIT_SOURCE_COMMIT'] = lock['source_commit']
        c, lock = locked(attempt)
    finally:
        if previous is None:
            os.environ.pop('ODEEDIT_SOURCE_COMMIT', None)
        else:
            os.environ['ODEEDIT_SOURCE_COMMIT'] = previous
    require(lock['owner'] == getpass.getuser(), 'OWNER')
    require(shutil.disk_usage(attempt).free >= c['resources']['reserve_bytes'], 'STORAGE_RESERVE')
    return lock, c

def admission(before, cap, node):
    require(cap >= 1, 'CURRENT_CAP_UNKNOWN')
    occupied = sum(row['gpus'] for row in before['jobs'])
    total = int(re.search(r'CfgTRES=.*?gres/gpu=(\d+)', node)[1])
    used = int(re.search(r'AllocTRES=.*?gres/gpu=(\d+)', node)[1])
    barrier = [r['job'] for r in before['jobs']] if occupied + 1 > cap else []
    # Capacity1 schedules producer first, so a consumer never holds the only
    # GPU while waiting for a producer queued behind it. Both jobs still
    # register/release upfront. A scientific predecessor PASS is not required.
    serial = cap - occupied < 2 or total - used < 2
    return barrier, serial

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', type=Path, required=True); args = parser.parse_args()
    require(not list(LOCAL.glob('attempt-*/submitted-*.json')) and not list(LOCAL.glob('attempt-*/submission.json')), 'NO_DUPLICATE_REGISTRATION')
    queue = command(['squeue', '-h', '-r', '-u', getpass.getuser(), '-o', '%i|%j|%T'])
    require('odeedit_jlz_cd_cumulative_s4_' not in queue, 'NO_DUPLICATE_JOB')
    before = resource_inventory()
    local = int(next(r for r in Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if r.startswith('server4\t')).split('\t')[2])
    tracked = int(next(r for r in (ROOT / 'control/gpu-concurrency-policy.tsv').read_text().splitlines() if r.startswith('server4\t')).split('\t')[1])
    cap = min(3, local, tracked)
    node = command(['scontrol', 'show', 'node', 'server4']); partition = command(['scontrol', 'show', 'partition', 'gpu'])
    require('MaxTime=30-00:00:00' in partition, 'PARTITION_CEILING')
    barrier, serial = admission(before, cap, node)
    require(serial, 'PARALLEL_SCALAR_READY_NOT_YET_QUALIFIED; register producer-first only when fresh capacity requires it')
    helper = subprocess.run(['bash', str(ROOT / 'scripts/check-slurm-resource-cap.sh'), 'server4', '1', '59392M'],
        env=dict(os.environ, AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'), capture_output=True, text=True)
    require(helper.returncode in (0, 4), 'RESOURCE_HELPER')
    require(helper.returncode == 0 or barrier, 'UNRESOLVED_RESOURCE_CAP')
    attempt = args.attempt.resolve(); freeze(args.config.resolve(), attempt); lock, c = verify_frozen(attempt)
    r = c['resources']; require(r['host_mib'] == 59392 and r['hard_host_mib'] == 60416 and r['wall'] == '2-00:00:00', 'SEALED_MEMORY_WALL')
    ids = {}; checks = []; mapping = {}
    for stage in STAGES:
        parents = barrier if stage == 'CD_Q' else [ids['CD_Q']] if stage == 'CD_C' and serial else barrier if stage == 'CD_C' else [ids[a] for a in ARMS]
        dep = dependencies(('afterany', parents)); argv = arguments(stage, dep, attempt, r)
        job = command(argv).split(';')[0]; require(job.isdigit(), 'JOB_ID'); ids[stage] = job
        mapping[stage] = dict(job=job, dependency=dep, argv=argv)
        write(attempt / ('submitted-' + stage + '.json'), dict(nonce=NONCE, status='HELD', **mapping[stage]))
        checks.append(inspect(job, stage, dep, argv, attempt, r))
    after = resource_inventory(tuple(ids.values()))
    require({j['job'] for j in after['jobs']} <= {j['job'] for j in before['jobs']}, 'ADMISSION_RACE_KEEP_HELD')
    require(barrier or sum(j['gpus'] for j in after['jobs']) + 1 <= cap, 'CAP_RACE_KEEP_HELD')
    verify_frozen(attempt)
    write(attempt / 'held-inspection.json', dict(checks=checks, before=before, prerelease=after,
        project_cap=cap, task_cap=2, max_new_GPU_concurrent=1, registered_GPU_lanes=2,
        scalar_resource_producer_first_serialization=True, external_barrier=barrier, node=node, partition=partition,
        helper=dict(rc=helper.returncode, output=helper.stdout + helper.stderr), all_held_before_release=True,
        no_other_job_mutation=True, no_scientific_quality_dependency=True))
    for stage in reversed(STAGES):
        result = command(['scontrol', 'release', ids[stage]])
        write(attempt / ('released-' + stage + '.json'), dict(job=ids[stage], success=True, result=result))
    write(attempt / 'submission.json', dict(nonce=NONCE, status='RELEASED', jobs=ids, mapping=mapping,
        source=lock['source_commit'], lock=member(attempt / 'execution.lock.json'), held=member(attempt / 'held-inspection.json'),
        main_initial='NOT_OBSERVED', new_GPU_qualification='NOT_OBSERVED', monitoring_active=False, automatic_resume=False))
    print(json.dumps(dict(status='RELEASED', jobs=ids, calibration_resource_serialized=True)))

if __name__ == '__main__':
    main()
