"""One deliberate replacement registration pass, never a retry/monitor loop.

Reuse the parent native launcher, exact held inspection, resource admission and
cap-safe six-arm graph. The new attempt binds a qualification PLAN before any
GPU execution. Actual qualification is created inside BASE_MEMIT, not invented
as an sbatch prerequisite or claimed in held inspection.
"""
import getpass
import argparse
import json
import re
import subprocess
import tarfile
from pathlib import Path

from .generation_common import (ARMS, SESSION, digest, member, read, require,
    sha, stat_seal, verify, write)
from .generation_cache_common import (ATTEMPT, ENVELOPE, NONCE, REPAIR_LOCAL,
    ROOT, TASK, RERUN_AUTHORITY, authority, ready, layout, recall_authority)
from .generation_plan import counts, dependencies
from .generation_submit import (SOURCES as PARENT_SOURCES, admission, command,
    field, gpu_count, inspect_held, launcher, sbatch_argv)
from project.run_scripts.gptj_cake_blue_prune_rect.submit import project_job

SOURCES = list(dict.fromkeys([*PARENT_SOURCES, ENVELOPE, RERUN_AUTHORITY]))


def inventory(exclude=()):
    """Server2 admission only; no detailed other-server job queries.

    Union of actual server2 node allocations/requests and its permitted QoS
    pending queue. Ambiguous cross-node/QoS bindings fail closed instead of
    being counted as zero or expanding this task's query authority.
    """
    owner, candidates = getpass.getuser(), {}
    for selector in (['--nodelist=server2'], ['--qos=lab_gpu_s2']):
        raw = command(['squeue', '-h', '-r', '-u', owner, *selector,
                       '-o', '%i|%j|%T|%b|%N|%R'])
        for line in raw.splitlines():
            parts = line.split('|', 5)
            require(len(parts) == 6, 'CACHE_REPAIR_LOCAL_QUEUE_FORMAT')
            require(parts[0] not in candidates or candidates[parts[0]][1] == parts[1],
                'CACHE_REPAIR_LOCAL_QUEUE_IDENTITY_CHANGED')
            candidates[parts[0]] = parts
    rows, excluded = [], []
    for job, name, status, gres, nodes, reason in candidates.values():
        if job in exclude:
            continue
        if nodes not in ('server2', '', '(null)', 'None'):
            excluded.append(dict(job=job, reason='EXPLICIT_OTHER_NODE_NO_DETAIL_QUERY'))
            continue
        detail = command(['scontrol', 'show', 'job', job, '--oneliner'])
        require((field(detail, 'UserId') or '').startswith(owner + '('), 'QUEUE_OWNER')
        node, requested_node = field(detail, 'NodeList'), field(detail, 'ReqNodeList')
        require(node == 'server2' or requested_node == 'server2',
            'CACHE_REPAIR_UNRESOLVED_SERVER2_SCOPE_NO_BROADER_QUERY')
        requested = gpu_count(field(detail, 'ReqTRES'))
        if not requested:
            continue
        if not project_job(detail):
            excluded.append(dict(job=job, reason='OTHER_PROJECT_SOURCE_PROTECTED'))
            continue
        require(re.fullmatch(r'[1-9][0-9]*(?:_[0-9]+)?', job), 'EXACT_QUEUE_JOB_ID')
        rows.append(dict(job=job, name=name, state=status, gpus=requested,
            allocated_GPUs=gpu_count(field(detail, 'AllocTRES')), reason=reason,
            node=node, requested_node=requested_node, detail=detail))
    return dict(project=rows, excluded=excluded,
        queries='server2 node plus lab_gpu_s2 queue; exact local details only',
        other_server_detailed_queries=0)


def submit(profile='r1'):
    authority()
    repair_local, expected_attempt = layout(profile)
    if profile == 'r2':
        recall_authority()
    config_path = repair_local / 'preparation-r1/config.json'
    config = read(config_path)
    ready(config)
    attempt = Path(config['attempt'])
    require(attempt == expected_attempt and not attempt.exists(), 'CACHE_REPAIR_ATTEMPT_CREATE_ONCE')
    require(not list(repair_local.glob('attempt-*/submitted-*.json')),
        'CACHE_REPAIR_NO_DUPLICATE_REGISTRATION')
    require(not command(['git', 'status', '--porcelain', '--', *SOURCES]),
        'CACHE_REPAIR_SOURCE_COMMITTED')
    # r1 is preserved FAILED: its mutated-raw negative fixture expected only
    # RuntimeError while the unchanged bound-file guard emits ValueError.
    checks_path = repair_local / 'cpu-integration-r2.json'
    checks = read(checks_path)
    require(checks['status'] == 'PASS_CPU_INTEGRATION' and checks['CUDA_initialized'] is False,
        'CACHE_REPAIR_NARROW_CPU_NOT_GPU')
    require(checks['config']['sha256'] == sha(config_path), 'CACHE_REPAIR_CPU_CONFIG_BOUND')
    for item in checks['source']:
        verify(item)
    for item in config['assets'] + config['runtime']['members'] + [config['observer_identity']]:
        stat_seal(item)
    for arm in ARMS:
        closure = config['arm_configs'][arm]['native']['closure']
        for item in closure if isinstance(closure, list) else [r for group in closure.values() for r in group]:
            verify(item)
    generation = config['generation']
    for item in generation['shared_source_members']:
        verify(item)
    require(command(['git', 'rev-parse', 'HEAD:project/run_scripts/experiment_generation_eval'])
        == generation['package_tree'], 'CACHE_REPAIR_EXACT_SHARED_PACKAGE_TREE')
    resources = admission(TASK, inventory_fn=inventory)
    source = command(['git', 'rev-parse', 'HEAD'])
    attempt.mkdir()
    (attempt / 'source').mkdir()
    archive = attempt / 'source.tar'
    command(['git', 'archive', '--format=tar', '--output=' + str(archive), source, *SOURCES])
    with tarfile.open(archive) as stream:
        require(all((m.isfile() or m.isdir()) and not Path(m.name).is_absolute()
            and '..' not in Path(m.name).parts for m in stream.getmembers()), 'SAFE_SOURCE_ARCHIVE')
        stream.extractall(attempt / 'source', filter='data')
    write(attempt / 'config.json', config)
    for role in (*ARMS, 'collector'):
        with (attempt / (role + '.sh')).open('x') as stream:
            stream.write(launcher(attempt, role, config, source))
    repair = generation['repair']
    lock = dict(instruction_id=NONCE, task_id=TASK, parent_task_id=config['parent_task_id'],
        source_commit=source, source_tree=command(['git', 'rev-parse', 'HEAD^{tree}']),
        config_sha256=sha(attempt / 'config.json'),
        source_members=[member(p) for p in sorted((attempt / 'source').rglob('*')) if p.is_file()],
        runtime_sources=config['runtime']['members'], archive=member(archive),
        launchers=[member(attempt / (role + '.sh')) for role in (*ARMS, 'collector')],
        tracking_env=member(config['tracking']['env_file']), owner=getpass.getuser(), session=SESSION,
        shared_generation_source=generation['source_sha'], shared_generation_tree=generation['package_tree'],
        reference=generation['reference_manifest'], reference_identity=generation['reference_assets_sha256'],
        qualification_plan=repair['qualification_plan'],
        qualification_plan_sha256=repair['qualification_plan_sha256'],
        shared_qualification_plan=repair['shared_qualification_plan'],
        shared_qualification_plan_sha256=repair['shared_qualification_plan_sha256'],
        qualification_status='PLAN_BOUND_NOT_ACTUAL_PASS', actual_GPU_qualification='NOT_RUN',
        actual_qualification_in_first_replacement_job=True,
        old_complete_case_inventory=repair['old_complete_case_inventory'],
        noCP=True, exact_resume='NOT_AVAILABLE', resources=config['resources'],
        registration_profile=profile, tracking_attempt=config.get('tracking_attempt', 'cache-repair-r1'),
        manual_recall_authority=config.get('manual_recall_authority'),
        count_plan=counts(), raw_generation_local_only=True,
        source_freeze_distinct_from_report=True, no_new_monitor_or_automatic_retry=True)
    write(attempt / 'execution.lock.json', lock)
    jobs, inspections, depmap = {}, [], {}
    try:
        for role in (*ARMS, 'collector'):
            deps = dependencies(role, resources['frontier'], jobs, resources['cap'])
            depmap[role] = deps
            argv = sbatch_argv(attempt, role, config, deps)
            write(attempt / ('submission-command-' + role + '.json'), dict(argv=argv, role=role))
            result = subprocess.run(argv, text=True, capture_output=True, timeout=45)
            write(attempt / ('sbatch-result-' + role + '.json'), dict(returncode=result.returncode,
                stdout=result.stdout.strip(), stderr=result.stderr.strip(), role=role, argv=argv))
            require(result.returncode == 0, 'SBATCH_REJECTED:' + role + ':' + result.stderr.strip()[:800])
            job = result.stdout.strip().split(';')[0]
            require(job.isdigit(), 'ACTUAL_SBATCH_JOB_ID')
            jobs[role] = job
            write(attempt / ('submitted-' + role + '.json'), dict(job=job, argv=argv, dependencies=deps))
            inspections.append(inspect_held(job, role, attempt, argv, deps, config))
        require(len(set(jobs.values())) == 7, 'SEVEN_DISTINCT_ACTUAL_JOBS')
        fresh = inventory(exclude=jobs.values())
        require({r['job'] for r in fresh['project']} <= set(resources['frontier']),
            'ADMISSION_RACE_ALL_NEW_JOBS_HELD')
        write(attempt / 'held-inspection.json', dict(jobs=inspections, **resources, fresh=fresh,
            qualification_status='PLAN_BOUND_NOT_ACTUAL_PASS', actual_qualification=False,
            qualification_plan=repair['qualification_plan'], concurrency_bound=resources['cap'],
            DAG='BASE_MEMIT qualification/W0 producer -> two native lanes -> CPU afterany all six',
            source_config_runtime_verified=True, resources_request_not_ETA=True,
            peak_RAM_VRAM='NOT_MEASURED', no_extra_fit_or_edit_pilot=True))
        for role in ('collector', 'RECT', 'PRUNE', 'ALPHAEDIT_BLUE', 'CAKE', 'BASE_ALPHAEDIT', 'BASE_MEMIT'):
            result = command(['scontrol', 'release', jobs[role]])
            write(attempt / ('released-' + role + '.json'), dict(job=jobs[role], result=result, released=True))
    except BaseException as error:
        write(attempt / 'submission-failure.json', dict(status='SUBMISSION_OR_RELEASE_BLOCKED', jobs=jobs,
            error_type=type(error).__name__, error=str(error)[:1200], source=source,
            lock=member(attempt / 'execution.lock.json'), automatic_retry=False,
            all_unrelated_jobs_unchanged=True, scientific_complete=False))
        raise
    snapshot = command(['squeue', '-h', '-j', ','.join(jobs.values()), '-o', '%i|%j|%T|%b|%N|%r'])
    result = dict(instruction_id=NONCE, task_id=TASK, status='SUBMISSION_HANDOFF',
        jobs=jobs, dependencies=depmap, source_commit=source, source_tree=lock['source_tree'],
        lock=member(attempt / 'execution.lock.json'), config_sha256=lock['config_sha256'],
        cap=resources['cap'], frontier=resources['frontier'], initial_snapshot=snapshot,
        resources=config['resources'], shared_generation_source=generation['source_sha'],
        qualification_status='PLAN_BOUND_NOT_ACTUAL_PASS', actual_qualification='NOT_OBSERVED',
        reference_identity=generation['reference_assets_sha256'],
        W_B='ACTUAL_ONLINE_STARTUP_IN_SEALED_RUNNER_NOT_OBSERVED',
        scientific_complete=False, monitoring_active=False, automatic_retry=False)
    write(attempt / 'submission.json', result)
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('r1', 'r2'), default='r1')
    submit(parser.parse_args().profile)
