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
REPAIR_SCHEMA = 'jlz-cd-cumulative-cold-repair-v1'
REPAIR_PREDECESSOR_SOURCE = 'abead2333c30cb57ea10ca9756a21f765f8dbc29'

def jobname(stage):
    return 'odeedit_jlz_cd_cumulative_s4_' + stage

def _bound_member(row, expected):
    path = Path(row['path'])
    require(path == expected and path.is_file() and not path.is_symlink(), 'REPAIR_MEMBER_PATH')
    return verify(row)

def validate_repair_receipt(path, attempt, preparation=None):
    """Local immutable identity checks only; scheduler admission is separate."""
    path = Path(path).resolve(); attempt = Path(attempt).resolve()
    require(path.is_file() and not path.is_symlink(), 'REPAIR_RECEIPT_REQUIRED')
    receipt = json.loads(path.read_text())
    require(receipt['schema'] == REPAIR_SCHEMA and receipt['instruction_id'] == NONCE
            and receipt['task_id'] == TASK, 'REPAIR_AUTHORITY')
    require(receipt['user_quote'] == 'fail되었으니 repair올려' and receipt['original_KEEP'] is True
            and receipt['repair_kind'] == 'COLD_ZERO_COMMIT', 'REPAIR_USER_COLD_KEEP')
    predecessor = receipt['predecessor']; successor = receipt['successor']
    prior = LOCAL / 'attempt-r1'
    require(Path(predecessor['attempt']) == prior and attempt == LOCAL / 'attempt-r2'
            and Path(successor['attempt']) == attempt
            and Path(successor['preparation']) == LOCAL / 'preparation-r2', 'REPAIR_ATTEMPT_PATH')
    if preparation is not None:
        require(Path(preparation).resolve() == Path(successor['preparation']), 'REPAIR_PREPARATION_PATH')
    require(predecessor['owner'] == getpass.getuser(), 'REPAIR_OWNER')
    require(predecessor['source_commit'] == REPAIR_PREDECESSOR_SOURCE, 'REPAIR_PREDECESSOR_SOURCE')
    require(set(predecessor['jobs']) == set(STAGES) and len(set(predecessor['jobs'].values())) == 3
            and all(type(job) is str and job.isdigit() for job in predecessor['jobs'].values()), 'REPAIR_JOB_SET')
    require(receipt['zero_commits'] == {arm: 0 for arm in ARMS}, 'REPAIR_ZERO_COMMIT_RECEIPT')
    submission = json.loads(_bound_member(predecessor['submission'], prior / 'submission.json').read_text())
    config = json.loads(_bound_member(predecessor['config'], prior / 'config.json').read_text())
    lock = json.loads(_bound_member(predecessor['lock'], prior / 'execution.lock.json').read_text())
    require(submission['nonce'] == NONCE and submission['status'] == 'RELEASED'
            and submission['jobs'] == predecessor['jobs'] and set(submission['mapping']) == set(STAGES), 'REPAIR_SUBMISSION')
    require(config['instruction_id'] == lock['instruction_id'] == NONCE
            and config['task_id'] == lock['task_id'] == TASK, 'REPAIR_PREDECESSOR_AUTHORITY')
    require(submission['source'] == lock['source_commit'] == predecessor['source_commit']
            and lock['owner'] == predecessor['owner'] and lock['host'] == 'server4', 'REPAIR_SOURCE_OWNER')
    require(lock['config_sha256'] == predecessor['config']['sha256']
            and submission['lock']['path'] == predecessor['lock']['path']
            and submission['lock']['sha256'] == predecessor['lock']['sha256'], 'REPAIR_LOCK_CONFIG_BINDING')
    for row in lock['source_members']:
        require(Path(row['path']).is_relative_to(prior / 'source'), 'REPAIR_FROZEN_SOURCE_PATH')
        verify(row)
    require(set(predecessor['terminals']) == set(STAGES)
            and set(receipt['failure_qualification']) == set(ARMS), 'REPAIR_TERMINAL_SET')
    for stage in STAGES:
        mapping = submission['mapping'][stage]
        require(mapping['job'] == predecessor['jobs'][stage]
                and '--job-name=' + jobname(stage) in mapping['argv']
                and '--chdir=' + str(prior / 'source') in mapping['argv']
                and mapping['argv'][-1] == str(prior / (stage + '.sh')), 'REPAIR_LAUNCH_IDENTITY')
        registered = json.loads((prior / ('submitted-' + stage + '.json')).read_text())
        require(registered['nonce'] == NONCE and registered['job'] == mapping['job']
                and registered['argv'] == mapping['argv'], 'REPAIR_REGISTERED_IDENTITY')
        terminal_path = prior / ('main-' + stage + '/terminal.json' if stage in ARMS else 'cpu-report/terminal.json')
        terminal = json.loads(_bound_member(predecessor['terminals'][stage], terminal_path).read_text())
        require(terminal['source'] == predecessor['source_commit'], 'REPAIR_TERMINAL_SOURCE')
        if stage in ARMS:
            require(terminal['arm'] == stage and terminal['job'] == mapping['job']
                    and type(terminal['commits']) is int and terminal['commits'] == 0
                    and terminal['status'] == 'TECHNICAL_BLOCKED'
                    and terminal['checkpoint_saved'] is False, 'REPAIR_COLD_TERMINAL')
            require(not list((prior / ('main-' + stage)).glob('batch-*/commit.json'))
                    and not (prior / ('main-' + stage + '/initial.json')).exists(), 'REPAIR_PRIOR_COMMIT_PRESENT')
            qualification = json.loads(_bound_member(receipt['failure_qualification'][stage],
                prior / ('main-' + stage + '/qualification/qualification.json')).read_text())
            require(qualification['pass_'] is False and qualification['GPU_qualified'] is False
                    and qualification['common_candidate_no_fit'] is True
                    and qualification['sealed_budget']['fit_calls'] == 0, 'REPAIR_FAILURE_QUALIFICATION')
        else:
            require(terminal['status'] == 'PARTIAL_OR_TECHNICAL_BLOCKED', 'REPAIR_COLLECTOR_TERMINAL')
    policy = receipt['repair_policy']
    require(policy['production_physical_grouping'] == 'ORIGINAL_COMPLETE_OWNER_GROUPS_1'
            and policy['rejected_physical_regrouping'] == 'NOT_QUALIFIED_NOT_USED'
            and policy['new_GPU_READY_required'] is True
            and policy['gradient_tolerance'] == {'atol': 1e-6, 'rtol': 2e-4, 'reduction': 'elementwise'}
            and policy['tolerance_widening'] is False and policy['method_coefficients_changed'] is False
            and policy['extra_fullB_fit'] == 0 and policy['automatic_retry'] is False, 'REPAIR_POLICY')
    return receipt

def registration_guard(attempt, receipt=None):
    """Only the exact bound predecessor may already have registrations."""
    require(not Path(attempt).exists(), 'CREATE_ONCE_ATTEMPT')
    registered = set(LOCAL.glob('attempt-*/submitted-*.json')) | set(LOCAL.glob('attempt-*/submission.json'))
    allowed = set()
    if receipt is not None:
        prior = Path(receipt['predecessor']['attempt'])
        allowed = {prior / 'submission.json'} | {prior / ('submitted-' + stage + '.json') for stage in STAGES}
    require(registered == allowed, 'NO_DUPLICATE_REGISTRATION')

def terminal_scheduler_rows(raw, receipt):
    """Reject missing/extra jobs, active jobs, or changed owner/name/source."""
    predecessor = receipt['predecessor']; expected = {job: stage for stage, job in predecessor['jobs'].items()}
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        values = [v.strip() for v in line.split('|')]
        if values and values[-1] == '':
            values.pop()
        require(len(values) == 6, 'REPAIR_SACCT_FIELDS')
        rows.append(dict(zip(('job', 'user', 'name', 'state', 'exitcode', 'workdir'), values)))
    require(len(rows) == len(expected) and {r['job'] for r in rows} == set(expected), 'REPAIR_SACCT_EXACT_JOBS')
    for row in rows:
        stage = expected[row['job']]
        require(row['user'] == predecessor['owner'] and row['name'] == jobname(stage)
                and row['workdir'] == str(Path(predecessor['attempt']) / 'source'), 'REPAIR_SACCT_IDENTITY')
        require(row['state'] == ('FAILED' if stage in ARMS else 'COMPLETED'), 'REPAIR_SACCT_TERMINAL')
        require(re.fullmatch(r'\d+:\d+', row['exitcode']) is not None
                and (row['exitcode'] != '0:0' if stage in ARMS else row['exitcode'] == '0:0'), 'REPAIR_SACCT_EXITCODE')
    return rows

def fresh_repair_terminal(receipt):
    jobs = ','.join(receipt['predecessor']['jobs'][stage] for stage in STAGES)
    raw = command(['sacct', '-n', '-P', '-X', '-j', jobs,
        '--format=JobIDRaw,User%64,JobName%128,State%32,ExitCode,WorkDir%1024'])
    return terminal_scheduler_rows(raw, receipt)

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
    c = json.loads(configpath.read_text())
    require(c['instruction_id'] == NONCE and c['task_id'] == TASK, 'NONCE')
    require(Path(c.get('attempt', LOCAL / 'attempt-r1')) == attempt
            and c['calibration']['path'] == str(attempt / 'calibration.json'), 'PREPARATION_ATTEMPT_BINDING')
    sources = list(SOURCES)
    if c.get('repair_receipt') is not None:
        receipt_path = verify(c['repair_receipt'])
        validate_repair_receipt(receipt_path, attempt, configpath.parent)
        require(receipt_path.is_relative_to(ROOT), 'TRACKED_REPAIR_RECEIPT_REQUIRED')
        sources.append(str(receipt_path.relative_to(ROOT)))
    require(not command(['git', 'status', '--porcelain', '--', *sources], cwd=ROOT), 'COMMIT_BEFORE_FREEZE')
    tested = json.loads(verify(c['cpu_preflight']).read_text())
    require(tested['passed'], 'FINAL_CPU_FAILED')
    for row in tested['source']:
        verify(row)
    commit = command(['git', 'rev-parse', 'HEAD'], cwd=ROOT); tree = command(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT)
    attempt.mkdir(); source = attempt / 'source'; source.mkdir(); archive = attempt / 'source.tar'
    command(['git', 'archive', '--format=tar', '--output=' + str(archive), commit, *sources], cwd=ROOT)
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
        lambda_lock='FIRST_NONZERO_MAIN_B1_SHARED_ONCE_NO_EXTRA_FIT', repair_receipt=c.get('repair_receipt')))

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
    require(lock.get('repair_receipt') == c.get('repair_receipt'), 'REPAIR_LOCK_RECEIPT_BINDING')
    if c.get('repair_receipt') is not None:
        validate_repair_receipt(verify(c['repair_receipt']), attempt)
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
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--repair-receipt', type=Path); args = parser.parse_args()
    attempt = args.attempt.resolve(); configpath = args.config.resolve()
    config = json.loads(configpath.read_text()); receipt = None; repair_terminal = None
    if args.repair_receipt is not None:
        repairpath = args.repair_receipt.resolve()
        require(config.get('repair_receipt') == member(repairpath), 'CONFIG_REPAIR_RECEIPT_BINDING')
        receipt = validate_repair_receipt(repairpath, attempt, configpath.parent)
    else:
        require(config.get('repair_receipt') is None and attempt == LOCAL / 'attempt-r1', 'EXPLICIT_REPAIR_RECEIPT_REQUIRED')
    registration_guard(attempt, receipt)
    if receipt is not None:
        repair_terminal = fresh_repair_terminal(receipt)
    queue = command(['squeue', '-h', '-r', '--name=' + ','.join(jobname(stage) for stage in STAGES), '-o', '%i|%128j|%T|%u'])
    require('odeedit_jlz_cd_cumulative_s4_' not in queue, 'NO_DUPLICATE_JOB')
    before = resource_inventory()
    local = int(next(r for r in Path('/data/janghj/ODE-edit/servers/local/gpu-caps.tsv').read_text().splitlines() if r.startswith('server4\t')).split('\t')[2])
    tracked = int(next(r for r in (ROOT / 'control/gpu-concurrency-policy.tsv').read_text().splitlines() if r.startswith('server4\t')).split('\t')[1])
    cap = min(3, local, tracked)
    node = command(['scontrol', 'show', 'node', 'server4']); partition = command(['scontrol', 'show', 'partition', 'gpu'])
    require('MaxTime=30-00:00:00' in partition, 'PARTITION_CEILING')
    barrier, capacity_serial = admission(before, cap, node)
    # Parallel scalar readiness is not qualified. Repair keeps the already
    # proven resource-only afterany producer ordering even with spare GPUs.
    serial = capacity_serial or receipt is not None
    require(serial, 'PARALLEL_SCALAR_READY_NOT_YET_QUALIFIED; register producer-first only when fresh capacity requires it')
    serialization_reason = ('IMMUTABLE_SHARED_SCALAR_REFERENCE_PRODUCER_FIRST_FALLBACK'
        if receipt is not None else 'FRESH_RESOURCE_CAPACITY_REQUIRES_PRODUCER_FIRST')
    helper = subprocess.run(['bash', str(ROOT / 'scripts/check-slurm-resource-cap.sh'), 'server4', '1', '59392M'],
        env=dict(os.environ, AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'), capture_output=True, text=True)
    require(helper.returncode in (0, 4), 'RESOURCE_HELPER')
    require(helper.returncode == 0 or barrier, 'UNRESOLVED_RESOURCE_CAP')
    freeze(configpath, attempt); lock, c = verify_frozen(attempt)
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
        capacity_serial=capacity_serial, serialization_reason=serialization_reason,
        repair_predecessor_terminal=repair_terminal, repair_receipt=c.get('repair_receipt'),
        helper=dict(rc=helper.returncode, output=helper.stdout + helper.stderr), all_held_before_release=True,
        no_other_job_mutation=True, no_scientific_quality_dependency=True))
    for stage in reversed(STAGES):
        result = command(['scontrol', 'release', ids[stage]])
        write(attempt / ('released-' + stage + '.json'), dict(job=ids[stage], success=True, result=result))
    write(attempt / 'submission.json', dict(nonce=NONCE, status='RELEASED', jobs=ids, mapping=mapping,
        source=lock['source_commit'], lock=member(attempt / 'execution.lock.json'), held=member(attempt / 'held-inspection.json'),
        repair_receipt=c.get('repair_receipt'),
        max_new_GPU_concurrent=1, capacity_serial=capacity_serial, serialization_reason=serialization_reason,
        main_initial='NOT_OBSERVED', new_GPU_qualification='NOT_OBSERVED', monitoring_active=False, automatic_resume=False))
    print(json.dumps(dict(status='RELEASED', jobs=ids, calibration_resource_serialized=True)))

if __name__ == '__main__':
    main()
