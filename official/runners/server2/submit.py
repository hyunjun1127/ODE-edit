"""One deliberate source-bound official Server2 registration/resume pass.

Science is exclusively the published official runner. This controller only
seals bytes, inspects resources/owned project DAG, submits held jobs, inspects
their complete commands, and releases this attempt. No cancellation, polling,
automatic retry or scientific-quality selection is implemented.
"""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile

from official.experiments.prepare import METHODS, digest, file_sha, read, write_new
from official.runners.server2.checkpoint_profile import deferred
from official.runners.server2 import no_gpu_qualification as noqual

REPO = Path(__file__).resolve().parents[3]
OUTPUT = Path('/mnt/raid5/janghj/ODE-edit/local/official-baselines-server2/20261008-r1')
SESSION = '01a0493a-074c-7f91-9a13-769116326fef'
INSTRUCTION = 'USER-OFFICIAL-BASELINES-20261008-R1'
TASK = 'official-baselines-server2-20261008-r1'
ACTIVE = ('RUNNING', 'PENDING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED')
CONTROL_PATHS = ('official/runners/server2/submit.py', 'official/runners/server2/test_submit.py')
CONTROL_IMPORT_PATHS = (*CONTROL_PATHS, 'official/__init__.py',
    'official/experiments/__init__.py', 'official/experiments/prepare.py',
    'official/runners/__init__.py', 'official/runners/server2/__init__.py')


def require(value, code):
    if not value:
        raise ValueError(code)


def command(argv, *, cwd=REPO):
    value = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, timeout=45)
    require(value.returncode == 0, 'COMMAND_FAILED:'+argv[0]+':'+value.stderr.strip()[:800])
    return value.stdout.strip()


def field(detail, name):
    value = re.search(r'(?:^|\s)'+re.escape(name)+r'=([^\s]+)', detail)
    return value.group(1) if value else None


def gpu_count(tres):
    value = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', tres or '')
    if value:
        return int(value.group(1))
    return sum(int(n) for n in re.findall(r'(?:^|,)gres/gpu:[^,=]+=(\d+)(?:,|$)', tres or ''))


def member(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'REGULAR_FILE_REQUIRED:'+str(path))
    path = path.resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path))


def verify_member(row):
    actual = member(row['path'])
    require(actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'], 'SEALED_MEMBER_CHANGED')
    return Path(actual['path'])


def sealed_source(source, official_tree, *, run=command):
    require(re.fullmatch(r'[0-9a-f]{40}', source or '')
        and re.fullmatch(r'[0-9a-f]{40}', official_tree or ''), 'SOURCE_TREE_EXACT_SHA_REQUIRED')
    require(run(['git', 'remote', 'get-url', 'origin']).removesuffix('.git').endswith('hyunjun1127/ODE-edit'),
        'WRONG_REPOSITORY_ORIGIN')
    require(run(['git', 'merge-base', '--is-ancestor', source, 'origin/main']) == '',
        'SOURCE_NOT_PUBLISHED_MAIN')
    require(run(['git', 'rev-parse', source+':official']) == official_tree, 'OFFICIAL_TREE_MISMATCH')
    require(not run(['git', 'status', '--porcelain', '--', 'official']), 'OFFICIAL_SOURCE_UNCOMMITTED')
    require(run(['git', 'rev-parse', 'HEAD:official']) == official_tree,
            'IMPORTED_OFFICIAL_TREE_NOT_SELECTED_SOURCE')
    return dict(code_commit=source, official_tree_sha256=official_tree,
        observed_main=run(['git', 'rev-parse', 'origin/main']), only_published_official=True)


def tracking_binding(manifest):
    """Cheap CPU source binding, not SDK/auth/remote-delivery certification."""
    value = manifest.get('tracking')
    require(isinstance(value, dict) and value.get('namespace') == 'official.tracking'
        and re.fullmatch(r'[0-9a-f]{64}', value.get('source_sha256', ''))
        and value.get('env_file') and value.get('metric_schema') == 'official-baselines-scalar-v1',
        'OFFICIAL_TRACKING_API_NOT_READY')
    path = value['namespace'][len('official.'):].replace('.', '/')
    matches = [item for item in (path+'.py', path+'/__init__.py') if item in manifest['source_members']]
    require(len(matches) == 1 and manifest['source_members'][matches[0]] == value['source_sha256'],
        'OFFICIAL_TRACKING_IMPORTED_SOURCE_NOT_BOUND')
    require(Path(value['env_file']).is_file(), 'OFFICIAL_TRACKING_CONFIG_MISSING')
    return dict(namespace=value['namespace'], source_sha256=value['source_sha256'],
        SDK_auth_remote_status='NOT_CERTIFIED_BY_CPU_BINDING')


def _project(detail, roots):
    return any(value == root.rstrip('/') or value.startswith(root)
        for value in (field(detail, 'Command') or '', field(detail, 'WorkDir') or '') for root in roots)


def inventory(*, exclude=(), run=command, owner=None):
    """Exact Server2 owned admission metadata; no science/result readback."""
    owner = getpass.getuser() if owner is None else owner
    candidates = {}
    roots = ['/mnt/raid5/janghj/ODE-edit/']
    for line in run(['git', 'worktree', 'list', '--porcelain']).splitlines():
        if line.startswith('worktree '):
            roots.append(line[len('worktree '):].rstrip('/')+'/')
    roots = tuple(dict.fromkeys(roots))
    for selector in ('--nodelist=server2', '--qos=lab_gpu_s2'):
        raw = run(['squeue', '-h', '-r', '-u', owner, selector, '-o', '%i|%j|%T|%b|%N|%R'])
        for line in raw.splitlines():
            parts = line.split('|', 5)
            require(len(parts) == 6, 'LOCAL_QUEUE_FORMAT')
            candidates[parts[0]] = parts
    rows, excluded, ambiguous = [], [], []
    for job, name, status, gres, node, reason in candidates.values():
        if job in exclude:
            continue
        require(re.fullmatch(r'[1-9][0-9]*(?:_[0-9]+)?', job), 'EXACT_LOCAL_JOB_ID_REQUIRED')
        if node not in ('server2', '', '(null)', 'None'):
            excluded.append(dict(job=job, reason='OTHER_NODE_NO_DETAILED_QUERY'))
            continue
        detail = run(['scontrol', 'show', 'job', job, '--oneliner'])
        require((field(detail, 'UserId') or '').startswith(owner+'('), 'EXACT_JOB_OWNER_REQUIRED')
        require(field(detail, 'NodeList') == 'server2' or field(detail, 'ReqNodeList') == 'server2',
            'AMBIGUOUS_LOCAL_NODE_IDENTITY')
        requested = gpu_count(field(detail, 'ReqTRES'))
        if not requested:
            continue
        if not _project(detail, roots):
            source_fields = [field(detail, key) for key in ('Command', 'WorkDir')]
            if any(value in (None, '', '(null)', 'None') for value in source_fields) \
                    or any(value.startswith('/mnt/raid5/janghj/.codex/worktrees/') for value in source_fields):
                ambiguous.append(dict(job=job, gpus=requested,
                    reason='OWN_GPU_PROJECT_SOURCE_UNRESOLVED_NO_ZERO_ASSUMPTION'))
                continue
            excluded.append(dict(job=job, reason='NONPROJECT_COMMAND_WORKDIR_PROTECTED'))
            continue
        require(status in ACTIVE, 'UNEXPECTED_ACTIVE_QUEUE_STATE')
        rows.append(dict(job=job, name=name, state=status, gpus=requested,
            allocated_GPUs=gpu_count(field(detail, 'AllocTRES')), reason=reason,
            dependency=field(detail, 'Dependency'), command=field(detail, 'Command'),
            workdir=field(detail, 'WorkDir'), detail=detail))
    return dict(project=rows, excluded=excluded, ambiguous=ambiguous, owner=owner,
        project_scope='Command/WorkDir inside exact repository registered worktrees or root local',
        other_server_detailed_queries=0, science_results_queried=False)


def dependency_ids(value):
    if value in (None, '', '(null)', 'None'):
        return []
    require('?' not in value, 'UNKNOWN_DAG_DEPENDENCY_SEMANTICS')
    result = []
    for group in value.split(','):
        require(group.startswith(('afterany:', 'afterok:')), 'UNKNOWN_DAG_DEPENDENCY_SEMANTICS')
        ids = [part.split('(', 1)[0] for part in group.split(':')[1:]]
        require(all(re.fullmatch(r'[1-9][0-9]*(?:_[0-9]+)?', job) for job in ids),
            'INVALID_DAG_DEPENDENCY_ID')
        result.extend(ids)
    return list(dict.fromkeys(result))


def typed_dependencies(value):
    if isinstance(value, list):
        value = 'afterany:'+':'.join(value) if value else ''
    if value in (None, '', '(null)', 'None'):
        return ()
    dependency_ids(value)
    # Slurm may expand afterany:a:b into afterany:a,afterany:b. Commas
    # are conjunctions; normalize only equal dependency kinds, never OR/? or
    # afterok versus afterany. Array task zero remains part of the actual ID.
    groups = {}
    for group in value.split(','):
        kind, *jobs = group.split(':')
        groups.setdefault(kind, set()).update(job.split('(', 1)[0] for job in jobs)
    return tuple(sorted((kind, tuple(sorted(jobs))) for kind, jobs in groups.items()))


def frontier(rows):
    """Leaves of the already admitted GPU DAG, not merely RUNNING jobs."""
    jobs = {row['job'] for row in rows}
    parents = {job for row in rows for job in dependency_ids(row['dependency']) if job in jobs}
    leaves = jobs-parents
    require(not jobs or leaves, 'UNKNOWN_CYCLIC_FRONTIER_NO_NEW_ADMISSION')
    return sorted(leaves, key=lambda value:tuple(int(v) for v in value.split('_')))


def resources(manifest):
    r = manifest['resources']
    require(r['gpu'] == 1 and 1 <= r['cpu'] <= 6 and 1 <= r['host_mib'] <= 59392,
        'SERVER2_STRICTER_GPU_CPU_MEMORY_LIMIT')
    require(r['wall'] == '2-00:00:00' and 1 <= r['collector_cpu'] <= 6
        and 1 <= r['collector_host_mib'] <= 24576 and r['collector_wall'] == '04:00:00',
        'SERVER2_WALL_COLLECTOR_LIMIT')
    require(r['partition'] and r['qos'] == 'lab_gpu_s2' and r['node'] == 'server2',
        'SERVER2_QUEUE_NODE_BINDING')
    require(type(r['reserve_bytes']) is int and r['reserve_bytes'] > 0, 'CHECKPOINT_DISK_RESERVE_REQUIRED')
    return r


def admission(manifest, out, *, run=command, inspect_inventory=inventory):
    r = resources(manifest)
    tracked = [line.split('\t')[1] for line in (REPO/'control/gpu-concurrency-policy.tsv').read_text().splitlines()
        if line.startswith('server2\t')]
    local = [line.split('\t') for line in Path(manifest['local_caps_file']).read_text().splitlines()
        if line.startswith('server2\t')]
    require(len(tracked) == len(local) == 1, 'UNIQUE_LOCAL_TRACKED_SERVER2_CAP')
    # Exact direct USER cap3 overrides the historical tracked cap2 only for
    # this new profile. Other tasks retain their previous admission semantics.
    from official.runners.server2.zsre_profile import enabled as zsre_enabled
    cap = min(4, int(local[0][2])) if noqual.enabled(manifest) or zsre_enabled(manifest) else min(3, int(local[0][2])) if deferred(manifest) else min(2, int(tracked[0]), int(local[0][2]))
    require(1 <= cap <= (4 if noqual.enabled(manifest) or zsre_enabled(manifest) else 3 if deferred(manifest) else 2) and local[0][1] == 'server2' and r['host_mib'] <= int(local[0][3]),
        'CAP_DISABLED_OR_MEMORY_CEILING')
    before = inspect_inventory()
    require(not before.get('ambiguous'), 'PROJECT_GPU_CLASSIFICATION_UNRESOLVED')
    require(sum(row['allocated_GPUs'] for row in before['project']) <= cap, 'LEGACY_OVERCAP_NO_NEW_ADMISSION')
    require(all(row['gpus'] == 1 and row['job'].isdigit() for row in before['project']),
        'MULTIGPU_ARRAY_FRONTIER_REQUIRES_EXACT_PROOF')
    require(not any(row['name'].startswith(TASK+'-'+manifest['registration_stage']) for row in before['project']),
        'DUPLICATE_STAGE_REGISTERED_QUEUE')
    node = run(['scontrol', 'show', 'node', 'server2', '--oneliner'])
    partition = run(['scontrol', 'show', 'partition', r['partition'], '--oneliner'])
    qos = run(['sacctmgr', '-n', '-P', 'show', 'qos', r['qos'], 'format=Name,MaxWall,MaxTRESPerUser,MaxTRESPerJob'])
    require(field(node, 'NodeName') == 'server2' and 'gpu:' in (field(node, 'Gres') or ''), 'ACTUAL_NODE_GPU')
    require(r['qos'] in partition and r['qos']+'|' in qos, 'ACTUAL_PARTITION_QOS')
    memory_policy = run(['python3', str(REPO/'scripts/slurm_memory_policy.py'), 'request',
        '--server', 'server2', '--gpus', '1', '--mem', str(r['host_mib'])+'M',
        '--local-limit-mib-per-gpu', local[0][3]])
    hardware = run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])
    usage = shutil.disk_usage(out)
    require(usage.free >= r['reserve_bytes'], 'CHECKPOINT_TWO_LANE_DISK_RESERVE')
    return dict(cap=cap, before=before, frontier=frontier(before['project']), node=node,
        partition=partition, qos=qos, hardware=hardware, memory_policy=memory_policy,
        allocated_GPUs=sum(row['allocated_GPUs'] for row in before['project']),
        disk_free_bytes=usage.free, inode_free=os.statvfs(out).f_favail,
        configured_reserve_bytes=r['reserve_bytes'], requested_wall_is_not_ETA=True)


def w0_binding(path, *, manifest, dataset):
    bound = member(path)
    ready = read(path)
    require(ready.get('status') == 'READY_COLD_W0_COMPLETE' and ready.get('actual_GPU') is True
        and ready.get('model') == 'gptj' and ready.get('dataset') == dataset
        and ready.get('code_commit') == manifest['code_commit']
        and ready.get('official_tree_sha256') == manifest['official_tree_sha256']
        and ready.get('model_revision') == manifest['model_revision']
        and ready.get('tokenizer_sha256') == manifest['tokenizer_sha256']
        and ready.get('stream_sha256') == manifest['streams'][dataset]['lock']['stream_sha256'],
        'ACTUAL_W0_REFERENCE_IDENTITY')
    if dataset == 'cf':
        from official.runners.server2.run import verify_cf_native_oracle
        verify_cf_native_oracle(manifest, ready.get('original_native_reference'))
    return bound


def gate(path, *, manifest, kind):
    """Only actual model execution evidence is a scientific-stage prerequisite."""
    if kind == 'qualification':
        from official.runners.server2 import oracle
        frozen = manifest.get('cf_native_reference_plan')
        require(frozen is not None and frozen == oracle.plan(manifest,
            read(manifest['streams']['cf']['path'])[:4]), 'FUTURE_W0_ORIGINAL_ORACLE_PLAN_REQUIRED')
        # Original scorer evidence is produced by the new W0 job and gates its
        # READY, not a circular prerequisite for registering that W0 producer.
        # Resume evidence retains its own old producer source/identities.
        if manifest.get('qualification_input_plan') is not None:
            from official.runners.server2 import qualification_input
            binding = qualification_input.verify(manifest['qualification_input_plan'], manifest, path)
            require(binding.get('status') == 'VERIFIED_PRODUCER_QUALIFICATION_INPUT'
                and binding.get('actual_GPU') is True,
                'ACTUAL_PRODUCER_QUALIFICATION_INPUT_NOT_READY')
            return dict(member(path), native_resume_consumer_binding=binding)
    receipt = read(path)
    require(receipt['status'] == 'PASS_ACTUAL_'+kind.upper() and receipt['actual_GPU'] is True
        and receipt['code_commit'] == manifest['code_commit']
        and receipt['official_tree_sha256'] == manifest['official_tree_sha256']
        and receipt['manifest_sha256'] == manifest['base_manifest_sha256'], 'ACTUAL_STAGE_RECEIPT_IDENTITY')
    if kind == 'qualification':
        from official.runners.server2.run import checkpoint_identity
        require(set(receipt['methods']) == set(METHODS), 'SIX_ACTUAL_QUALIFICATIONS_REQUIRED')
        for method, value in receipt['methods'].items():
            require(value['method'] == method and value['dataset'] == 'cf' and value['model'] == 'gptj'
                and value['actual_GPU'] is True and value['continuous_batches'] == 3
                and value['resume_after_batch'] == 2 and value['resumed_batches'] == [3]
                and value.get('status') == 'PASS_ACTUAL_QUALIFICATION'
                and value.get('code_commit') == manifest['code_commit']
                and value.get('official_tree_sha256') == manifest['official_tree_sha256']
                and value.get('manifest_sha256') == manifest['base_manifest_sha256']
                and value.get('checkpoint_identity') == checkpoint_identity(manifest, method, 'cf')
                and all(value[key] is True for key in ('weights_equal', 'history_equal', 'rng_equal',
                    'metrics_equal', 'contexts_equal')),
                'NATIVE_ACTUAL_RESUME_PROOF_REQUIRED:'+method)
    elif kind == 'smoke':
        require(receipt['model'] == 'gptj' and receipt['dataset'] == 'zsre' and receipt['batches'] == 1,
            'ACTUAL_ZSRE_SMOKE_REQUIRED')
        w0_binding(verify_member(receipt['W0member']), manifest=manifest, dataset='zsre')
    else:
        require(False, 'UNKNOWN_ACTUAL_STAGE_GATE')
    return member(path)


def roles(stage, resume_method=None, smoke_only=False):
    require(stage in ('qualification', 'cf', 'zsre', 'cf_checkpoint', 'zsre_pipeline', 'no_gpu_qual'), 'UNKNOWN_REGISTRATION_STAGE')
    if stage == 'no_gpu_qual':
        require(not resume_method and not smoke_only, 'USER_DISABLED_COLD_ONLY')
        return list(noqual.ROLES)
    if stage == 'zsre_pipeline':
        from official.runners.server2.zsre_profile import METHODS as ZSRE_METHODS
        require(not resume_method and not smoke_only, 'ZSRE_PIPELINE_COLD_ONLY')
        return list(ZSRE_METHODS)
    if resume_method:
        require(resume_method in METHODS and stage in ('cf', 'zsre'), 'RESUME_EXACT_MAIN_METHOD')
        return [resume_method]
    if smoke_only:
        require(stage == 'zsre', 'SMOKE_ONLY_ZSRE_STAGE')
        return ['W0_ZSRE', 'ZSRE_SMOKE']
    return ['W0_CF', *METHODS] if stage == 'cf' else list(METHODS)


def dependencies(role, ordered, jobs, external, cap):
    require(1 <= cap <= 4 and len(set(external)) == len(external)
        and all(job.isdigit() for job in external), 'EXACT_FRONTIER_CAP')
    if role == 'collector':
        require(set(jobs) == set(ordered), 'COLLECTOR_AFTER_ALL_OWN_GPU')
        return list(jobs.values())
    index = ordered.index(role)
    require(list(jobs) == ordered[:index], 'DETERMINISTIC_ROLE_REGISTRATION_ORDER')
    return list(external) if index < cap else [jobs[ordered[index-cap]]]


def stage_dependencies(role, stage, ordered, jobs, external, cap):
    """W0 is an actual technical READY, not a scientific-quality afterok gate."""
    if stage == 'no_gpu_qual' and role != 'collector':
        edges = dependencies(role, ordered, jobs, external, cap)
        dataset, method, mode = noqual.cell(role)
        if mode == 'w0':
            return edges
        required = 'afterok:'+jobs['W0_'+dataset.upper()]
        resource_edges = [job for job in edges if job != jobs['W0_'+dataset.upper()]]
        return required + (',afterany:'+':'.join(resource_edges) if resource_edges else '')
    if stage in ('cf_checkpoint', 'zsre_pipeline') and role != 'collector':
        # FT produces the one shared cold factual W0 before its own native
        # qualification/chain. No GPU file polling: consumers run after FT,
        # verify READY, and fail closed if the producer did not persist it.
        index = ordered.index(role)
        require(list(jobs) == ordered[:index], 'CHECKPOINT_PIPELINE_REGISTRATION_ORDER')
        if index == 0:
            return list(external)
        return [jobs[ordered[0]]] if index <= cap else [jobs[ordered[index-cap]]]
    if stage == 'zsre' and 'W0_ZSRE' in ordered and role != 'collector':
        if role == 'W0_ZSRE':
            require(not jobs, 'ZSRE_W0_FIRST_REGISTRATION')
            return list(external)
        require(role == 'ZSRE_SMOKE' and list(jobs) == ['W0_ZSRE'], 'ZSRE_MODEL_SMOKE_ORDER')
        return 'afterok:'+jobs['W0_ZSRE']
    if stage != 'cf' or 'W0_CF' not in ordered or role == 'collector':
        return dependencies(role, ordered, jobs, external, cap)
    if role == 'W0_CF':
        require(not jobs, 'CF_W0_FIRST_REGISTRATION')
        return list(external)
    methods = [value for value in ordered if value != 'W0_CF']
    index = methods.index(role)
    require(list(jobs) == ['W0_CF', *methods[:index]], 'CF_METHOD_REGISTRATION_ORDER')
    required = 'afterok:'+jobs['W0_CF']
    if index >= cap:
        required += ',afterany:'+jobs[methods[index-cap]]
    return required


def runner_argv(manifest_path, role, stage, out, manifest, *, resume=None):
    if stage == 'no_gpu_qual':
        require(noqual.enabled(manifest) and resume is None, 'EXPLICIT_USER_DISABLED_COLD_MAIN')
        dataset, method, mode = noqual.cell(role)
        return [manifest['runtime']['python'], '-u', '-m', 'official.runners.server2.run',
            '--manifest', str(manifest_path), '--method', method, '--dataset', dataset,
            '--mode', mode, '--out', str(out/role)]
    require(role in (*METHODS, 'ZSRE_SMOKE', 'W0_CF', 'W0_ZSRE'), 'EXACT_RUNNER_ROLE')
    method = manifest.get('smoke_method', 'MEMIT') if role == 'ZSRE_SMOKE' else 'MEMIT' if role.startswith('W0_') else role
    if stage == 'zsre_pipeline':
        from official.runners.server2.zsre_profile import enabled
        require(enabled(manifest) and role in METHODS and resume is None,'ZSRE_PIPELINE_PROFILE')
        return [manifest['runtime']['python'], '-u', '-m', 'official.runners.server2.zsre_pipeline',
                '--manifest',str(manifest_path),'--method',role,'--out',str(out/role)]
    if stage == 'cf_checkpoint':
        require(deferred(manifest) and role in METHODS and resume is None, 'CHECKPOINT_PIPELINE_PROFILE')
        return [manifest['runtime']['python'], '-u', '-m', 'official.runners.server2.checkpoint_pipeline',
                '--manifest', str(manifest_path), '--method', role, '--out', str(out/role)]
    mode = 'w0' if role.startswith('W0_') else 'smoke' if role == 'ZSRE_SMOKE' else 'qualification' if stage == 'qualification' else 'chain'
    argv = [manifest['runtime']['python'], '-u', '-m', 'official.runners.server2.run',
        '--manifest', str(manifest_path), '--method', method,
        '--dataset', 'cf' if stage in ('qualification', 'cf') else 'zsre',
        '--mode', mode, '--out', str(out/role)]
    if resume is not None:
        argv += ['--resume', str(resume)]
    return argv


def launcher(attempt, role, manifest, *, resume=None):
    cpu = manifest['resources']['collector_cpu'] if role == 'collector' else manifest['resources']['cpu']
    env = dict(PYTHONPATH=str(attempt/'source'), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS=str(cpu),
        MKL_NUM_THREADS=str(cpu), OPENBLAS_NUM_THREADS=str(cpu), HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED='0',
        OFFICIAL_CODE_COMMIT=manifest['code_commit'], OFFICIAL_TREE_SHA256=manifest['official_tree_sha256'])
    if role == 'collector':
        env['CUDA_VISIBLE_DEVICES'] = ''
        argv = [manifest['runtime']['python'], '-u', '-m',
            'official.runners.server2.checkpoint_pipeline' if manifest['registration_stage'] == 'cf_checkpoint'
            else 'official.runners.server2.zsre_pipeline' if manifest['registration_stage'] == 'zsre_pipeline'
            else 'official.runners.server2.collect',
            '--attempt', str(attempt)]
    else:
        argv = runner_argv(attempt/'manifest.json', role, manifest['registration_stage'], attempt,
            manifest, resume=resume)
    return '#!/bin/bash\n# ODEEDIT_SLURM_SERVER=server2\nset -euo pipefail\n'+''.join(
        'export '+key+'='+shlex.quote(value)+'\n' for key,value in env.items())+\
        'cd '+shlex.quote(str(attempt/'source'))+'\nexec '+shlex.join(argv)+'\n'


def sbatch_argv(attempt, role, manifest, dep):
    r = resources(manifest)
    collector = role == 'collector'
    cpu, memory, wall = (r['collector_cpu'], r['collector_host_mib'], r['collector_wall']) if collector \
        else (r['cpu'], r['host_mib'], r['wall'])
    argv = ['sbatch', '--parsable', '--hold', '--partition='+r['partition'], '--qos='+r['qos'],
        '--nodelist=server2', '--nodes=1', '--ntasks=1', '--cpus-per-task='+str(cpu),
        '--mem='+str(memory)+'M', '--time='+wall, '--export=NONE', '--no-requeue',
        '--job-name='+TASK+'-'+manifest['registration_stage']+'-'+role,
        '--chdir='+str(attempt/'source'), '--output='+str(attempt/(role+'-%j.out')),
        '--error='+str(attempt/(role+'-%j.err'))]
    if not collector:
        argv += ['--gres=gpu:1']
    if dep:
        typed_dependencies(dep)
        argv += ['--dependency='+('afterany:'+':'.join(dep) if isinstance(dep, list) else dep),
                 '--kill-on-invalid-dep=yes']
    return argv+[str(attempt/(role+'.sh'))]


def inspect_held(job, role, attempt, manifest, argv, deps, *, run=command, owner=None):
    owner = getpass.getuser() if owner is None else owner
    require(job.isdigit(), 'ACTUAL_SCHEDULER_JOB_ID')
    detail = run(['scontrol', 'show', 'job', job, '--oneliner'])
    r = manifest['resources']
    collector = role == 'collector'
    cpu, memory, wall = (r['collector_cpu'], r['collector_host_mib'], r['collector_wall']) if collector \
        else (r['cpu'], r['host_mib'], r['wall'])
    for key, expected in dict(JobId=job, JobName=TASK+'-'+manifest['registration_stage']+'-'+role,
            JobState='PENDING', Reason='JobHeldUser', Requeue='0', ReqNodeList='server2',
            Partition=r['partition'], QOS=r['qos'], TimeLimit=wall, Command=str(attempt/(role+'.sh')),
            WorkDir=str(attempt/'source'), **{'CPUs/Task':str(cpu)}).items():
        require(field(detail, key) == expected, 'HELD_FIELD:'+key)
    require((field(detail, 'UserId') or '').startswith(owner+'('), 'HELD_OWNER')
    tres = field(detail, 'ReqTRES') or ''
    require(re.search(r'(?:^|,)cpu='+str(cpu)+r'(?:,|$)', tres)
        and (f'mem={memory}M' in tres or (memory%1024 == 0 and f'mem={memory//1024}G' in tres))
        and gpu_count(tres) == (0 if collector else 1), 'HELD_CPU_MEMORY_GPU')
    require(typed_dependencies(field(detail, 'Dependency')) == typed_dependencies(deps), 'HELD_DEPENDENCY')
    match = re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)', detail)
    require(match and shlex.split(match.group(1)) == argv, 'HELD_FULL_ARGV')
    script = (attempt/(role+'.sh')).read_text()
    require(run(['scontrol', 'write', 'batch_script', job, '-']).strip() == script.strip(), 'HELD_SCRIPT_BYTES')
    return dict(job=job, role=role, owner=owner, detail=detail, argv=argv, dependencies=deps,
        script=member(attempt/(role+'.sh')), actual_GPU_qualification=False, held=True)


def owned_checkpoint_folder(folder):
    """Resolve no aliases into the explicitly authorized latest-one run scope."""
    folder = Path(os.path.abspath(os.fspath(folder)))
    require(OUTPUT in folder.parents and folder.name == 'checkpoints', 'RESUME_TASK_CHECKPOINT_SCOPE')
    require(all(not path.is_symlink() for path in (folder, *folder.parents)), 'RESUME_SYMLINK_COMPONENT')
    require(folder.is_dir(), 'RESUME_ORIGINAL_CHECKPOINT_FOLDER_MISSING')
    return folder


def resume_binding(folder, expected_identity, *, method, dataset):
    """Read metadata only; actual runner verifies/deserializes its checkpoint."""
    from official.experiments.checkpoint import validate_identity
    validate_identity(expected_identity)
    require(method in METHODS and dataset in ('cf', 'zsre'), 'RESUME_EXACT_ORIGINAL_CELL')
    folder = owned_checkpoint_folder(folder)
    original_out, attempt = folder.parent, folder.parent.parent
    require(original_out.name == method and re.fullmatch(r'registration-[A-Za-z0-9_-]+', attempt.name),
        'RESUME_OWN_REGISTERED_METHOD_FOLDER')
    original_manifest = attempt/'manifest.json'
    require(original_manifest.is_file() and not original_manifest.is_symlink(),
        'RESUME_ORIGINAL_MANIFEST_REGULAR_FILE_REQUIRED')
    original = read(original_manifest)
    require(original.get('instruction_id') == INSTRUCTION and original.get('model') == 'gptj'
        and original.get('owner', {}).get('server') == 'server2'
        and original.get('owner', {}).get('session') == SESSION
        and original.get('registration_stage') == dataset and method in original.get('registration_roles', []),
        'RESUME_ORIGINAL_CALLER_IDENTITY')
    from official.runners.server2.run import checkpoint_identity
    require(checkpoint_identity(original, method, dataset) == expected_identity,
        'RESUME_ORIGINAL_MANIFEST_IDENTITY_MISMATCH')
    pointer = folder/'latest.json'
    require(pointer.is_file() and not pointer.is_symlink(), 'RESUME_POINTER_REGULAR_FILE_REQUIRED')
    value = read(pointer)
    require(value['identity_sha256'] == digest(expected_identity), 'RESUME_CHECKPOINT_IDENTITY_MISMATCH')
    filename = value['file']
    require(type(filename) is str and re.fullmatch(r'batch-[0-9]{2}-[0-9a-f]{16}\.pt', filename)
        and filename == f"batch-{value['batch']:02d}-{value['sha256'][:16]}.pt",
        'RESUME_UNSAFE_CHECKPOINT_MEMBER')
    path = folder/filename
    require(path.is_file() and not path.is_symlink() and type(value['batch']) is int
        and 0 <= value['batch'] <= 20 and re.fullmatch(r'[0-9a-f]{64}', value['sha256']),
        'RESUME_CHECKPOINT_POINTER_SCHEMA')
    return dict(folder=str(folder), pointer=member(pointer), checkpoint_path=str(path),
        checkpoint_sha256=value['sha256'], checkpoint_bytes=path.stat().st_size,
        batch=value['batch'], identity=expected_identity, method=method, dataset=dataset,
        original_manifest=member(original_manifest), controller_tensor_load=False,
        actual_hash_and_payload_check='REQUIRED_IN_RUNNER')


def resume_writer_guard(binding, rows):
    """Pending writers count too; never give a latest-one folder two owners."""
    folder = Path(binding['folder'])
    for row in rows:
        script = Path(row.get('command') or '')
        if not script.is_absolute() or OUTPUT not in script.parents:
            continue
        require(script.is_file() and not script.is_symlink(), 'ACTIVE_OFFICIAL_CALLER_SOURCE_UNKNOWN')
        caller = script.parent
        require((caller/'manifest.json').is_file() and not (caller/'manifest.json').is_symlink(),
            'ACTIVE_OFFICIAL_CALLER_MANIFEST_UNKNOWN')
        manifest = read(caller/'manifest.json')
        require(manifest.get('owner', {}).get('server') == 'server2'
            and manifest.get('owner', {}).get('session') == SESSION
            and manifest.get('instruction_id') == INSTRUCTION, 'ACTIVE_OFFICIAL_CALLER_IDENTITY_UNKNOWN')
        same_original = script.stem == binding['method'] and caller == folder.parent.parent
        same_resume = (manifest.get('resume') or {}).get('folder') == str(folder)
        require(not (same_original or same_resume), 'RESUME_CHECKPOINT_ACTIVE_OR_ADMITTED_WRITER:'+row['job'])
    return dict(single_writer=True, active_and_admitted_rows_checked=len(rows), tensor_load=False)


def _existing(out, manifest_sha, stage, ordered=None, resume=None):
    for path in sorted(out.glob('registration-*/submission.json')):
        value = read(path)
        if value.get('stage') == stage and value.get('base_manifest_sha256') == manifest_sha \
                and (ordered is None or value.get('registration_roles') == ordered) \
                and value.get('resume_binding') == resume:
            require(value.get('status') == 'SUBMISSION_HANDOFF' and value.get('jobs'), 'EXISTING_PARTIAL_SUBMISSION_NO_DUPLICATE')
            if ordered is not None:
                require(set(value['jobs']) == set([*ordered, 'collector'])
                    and len(set(value['jobs'].values())) == len(value['jobs']),
                    'EXISTING_FULL_DAG_REQUIRED_NO_DUPLICATE')
            return dict(value, duplicate_prevented=True)
    for path in sorted(out.glob('registration-*/submission-failure.json')):
        manifest_path = path.parent/'manifest.json'
        if not manifest_path.is_file():
            continue
        old = read(manifest_path)
        if old.get('registration_stage') == stage and old.get('base_manifest_sha256') == manifest_sha \
                and (ordered is None or old.get('registration_roles') == ordered) \
                and old.get('resume') == resume:
            failure = read(path)
            require(not failure.get('jobs'), 'EXISTING_PARTIAL_REGISTERED_DAG_NO_RETRY')
    return None


def existing_snapshot(receipt, *, run=command, owner=None):
    """Receipt is not current queue evidence; reconcile only its exact own IDs."""
    owner = getpass.getuser() if owner is None else owner
    manifest_path = verify_member(receipt['manifest'])
    manifest = read(manifest_path)
    attempt = manifest_path.parent
    verify_member(receipt['lock'])
    require(manifest.get('instruction_id') == INSTRUCTION and manifest.get('model') == 'gptj'
        and manifest.get('owner', {}).get('server') == 'server2'
        and manifest.get('owner', {}).get('session') == SESSION, 'EXISTING_RECEIPT_OWNER_SOURCE')
    rows = []
    for role, job in receipt['jobs'].items():
        require(type(job) is str and job.isdigit(), 'EXISTING_EXACT_JOB_ID')
        expected = read(attempt/('command-'+role+'.json'))['argv']
        expected_name = TASK+'-'+manifest['registration_stage']+'-'+role
        try:
            detail = run(['scontrol', 'show', 'job', job, '--oneliner'])
        except ValueError:
            raw = run(['sacct', '-n', '-P', '-j', job,
                '--format=JobIDRaw%64,JobName%128,User%64,State%64,NodeList%64,SubmitLine%2048,WorkDir%1024'])
            found = [line.split('|') for line in raw.splitlines() if line.split('|')[0] == job]
            require(len(found) == 1 and len(found[0]) >= 7, 'EXISTING_ACCOUNTING_NOT_OBSERVED')
            actual, name, user, state, node, submit_line, workdir = found[0][:7]
            require(actual == job and name == expected_name and user == owner
                and node in ('server2', '', 'None', '(null)') and shlex.split(submit_line) == expected
                and workdir == str(attempt/'source'), 'EXISTING_ACCOUNTING_IDENTITY_MISMATCH')
            rows.append(dict(job=job, role=role, state=state, source='SACCT_EXACT_SUBMIT_LINE',
                scientific_result_queried=False))
            continue
        require(field(detail, 'JobId') == job and field(detail, 'JobName') == expected_name
            and (field(detail, 'UserId') or '').startswith(owner+'(')
            and field(detail, 'ReqNodeList') == 'server2'
            and field(detail, 'Command') == str(attempt/(role+'.sh'))
            and field(detail, 'WorkDir') == str(attempt/'source'), 'EXISTING_CURRENT_JOB_IDENTITY_MISMATCH')
        match = re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)', detail)
        require(match and shlex.split(match.group(1)) == expected, 'EXISTING_FULL_ARGV_MISMATCH')
        rows.append(dict(job=job, role=role, state=field(detail, 'JobState'),
            reason=field(detail, 'Reason'), source='SCONTROL_EXACT_COMMAND', scientific_result_queried=False))
    return dict(jobs=rows, source=manifest['code_commit'], official_tree_sha256=manifest['official_tree_sha256'],
        no_new_registration=True, source_members_retained=True, monitoring_active=False)


def safe_archive_members(rows):
    return all((row.isfile() or row.isdir()) and not Path(row.name).is_absolute()
        and '..' not in Path(row.name).parts
        and (row.name.startswith('official/') or row.name == 'official' and row.isdir()) for row in rows)


def git_bytes(argv):
    """Read exact Git bytes without a checkout/archive write or remote call."""
    require(argv and argv[0] == 'git', 'CONTROL_GIT_READ_ONLY')
    result = subprocess.run(argv, cwd=REPO, capture_output=True, timeout=45)
    require(result.returncode == 0, 'CONTROL_GIT_EVIDENCE_FAILED')
    return result.stdout


def published_control_binding(manifest, source, tree, *, run=command, raw=git_bytes):
    require(re.fullmatch(r'[0-9a-f]{40}', source or '')
        and re.fullmatch(r'[0-9a-f]{40}', tree or ''), 'CONTROL_EXACT_SOURCE_TREE_REQUIRED')
    require(run(['git', 'remote', 'get-url', 'origin']).removesuffix('.git').endswith('hyunjun1127/ODE-edit'),
            'CONTROL_WRONG_ORIGIN')
    for commit in (manifest['code_commit'], source):
        require(run(['git', 'merge-base', '--is-ancestor', commit, 'origin/main']) == '',
                'CONTROL_EXECUTION_AND_CONTROLLER_MUST_BE_PUBLISHED')
    require(run(['git', 'rev-parse', manifest['code_commit']+':official'])
            == manifest['official_tree_sha256'], 'CONTROL_EXECUTION_TREE_CHANGED')
    require(run(['git', 'rev-parse', source+':official']) == tree, 'CONTROL_CONTROLLER_TREE_MISMATCH')
    changes = run(['git', 'diff', '--name-status', manifest['code_commit'], source, '--', 'official'])
    rows = [line.split('\t') for line in changes.splitlines()]
    require(rows and all(len(row) == 2 and row[0] == 'M' and row[1] in CONTROL_PATHS for row in rows)
        and CONTROL_PATHS[0] in {row[1] for row in rows}, 'CONTROL_ONLY_SUBMIT_AND_TEST_DIFF_ALLOWED')
    require(not run(['git', 'status', '--porcelain', '--', *CONTROL_IMPORT_PATHS]), 'CONTROL_UNCOMMITTED_SOURCE')
    members = {}
    # Concurrent other-server merges may change unrelated official profiles.
    # Bind every byte actually imported by this controller rather than falsely
    # certifying that the whole live checkout equals the selected old branch.
    for path in CONTROL_IMPORT_PATHS:
        expected = hashlib.sha256(raw(['git', 'show', source+':'+path])).hexdigest()
        require(file_sha(REPO/path) == expected, 'CONTROL_IMPORTED_BYTES_NOT_PUBLISHED:'+path)
        members[path] = expected
    difference = raw(['git', 'diff', '--no-ext-diff', '--binary', manifest['code_commit'], source, '--', 'official'])
    return dict(code_commit=source, official_tree_sha256=tree, source_members=members,
        execution_commit=manifest['code_commit'], execution_tree=manifest['official_tree_sha256'],
        changed_paths=[row[1] for row in rows], exact_official_diff_sha256=hashlib.sha256(difference).hexdigest(),
        observed_main=run(['git', 'rev-parse', 'origin/main']), science_or_archive_hotpatch=False)


def verify_held_attempt(manifest_path, attempt, *, run=command):
    """Narrow first-FT dependency formatting failure; no arbitrary retry path."""
    attempt = Path(os.path.abspath(os.fspath(attempt)))
    require(OUTPUT in attempt.parents and re.fullmatch(r'registration-[A-Za-z0-9_-]+', attempt.name)
        and all(not path.is_symlink() for path in (attempt, *attempt.parents)), 'CONTROL_EXACT_OWN_ATTEMPT_SCOPE')
    require(attempt.is_dir() and not (attempt/'submission.json').exists()
        and not (attempt/'control-repair.lock.json').exists()
        and not list(attempt.glob('released-*.json')), 'CONTROL_ALREADY_STARTED_OR_RELEASED_NO_RETRY')
    failure_member = member(attempt/'submission-failure.json')
    failure = read(failure_member['path'])
    require(failure.get('status') == 'REGISTRATION_OR_RELEASE_BLOCKED'
        and failure.get('error') == 'HELD_DEPENDENCY'
        and set(failure.get('jobs', {})) == {'FT'}, 'CONTROL_EXACT_FIRST_FT_HELD_DEPENDENCY_FAILURE_ONLY')
    lock_path = verify_member(failure['lock'])
    require(lock_path == attempt/'execution.lock.json', 'CONTROL_ORIGINAL_LOCK_PATH')
    lock = read(lock_path)
    require(verify_member(lock['base_manifest']) == Path(manifest_path).resolve()
        and verify_member(lock['manifest']) == attempt/'manifest.json'
        and verify_member(lock['archive']) == attempt/'source.tar', 'CONTROL_ORIGINAL_IMMUTABLE_MEMBERS')
    manifest = read(attempt/'manifest.json')
    require(manifest.get('model') == 'gptj' and manifest.get('instruction_id') == INSTRUCTION
        and manifest.get('owner') == {'server':'server2', 'session':SESSION}
        and manifest.get('registration_stage') == 'qualification'
        and manifest.get('registration_roles') == list(METHODS)
        and not manifest.get('resume') and lock.get('stage') == 'qualification'
        and lock.get('code_commit') == manifest['code_commit']
        and lock.get('official_tree_sha256') == manifest['official_tree_sha256']
        and lock.get('resources') == manifest['resources'], 'CONTROL_FROZEN_QUALIFICATION_IDENTITY')
    require(file_sha(manifest_path) == manifest['base_manifest_sha256'], 'CONTROL_BASE_MANIFEST_CHANGED')
    source = attempt/'source'
    expected = {'official/'+name:checksum for name, checksum in manifest['source_members'].items()}
    actual = {str(path.relative_to(source)):file_sha(path) for path in source.rglob('*') if path.is_file()}
    require(actual == expected, 'CONTROL_ARCHIVED_SOURCE_MEMBERS_CHANGED')
    locked = {str(verify_member(row).relative_to(source)):row['sha256'] for row in lock['source_members']}
    require(locked == actual, 'CONTROL_SOURCE_LOCK_MEMBERS_CHANGED')
    # Also bind the actual extracted files to the old Git objects. The lock's
    # original archive/manifest hash is not permission to relabel a new source.
    git_rows = run(['git', 'ls-tree', '-r', manifest['code_commit']+':official'])
    blobs = {}
    for line in git_rows.splitlines():
        info, path = line.split('\t', 1)
        mode, kind, checksum = info.split()
        require(mode in ('100644', '100755') and kind == 'blob', 'CONTROL_GIT_SOURCE_REGULAR_FILES_ONLY')
        blobs['official/'+path] = checksum
    actual_blobs = {}
    for name in actual:
        data = (source/name).read_bytes()
        actual_blobs[name] = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    require(actual_blobs == blobs, 'CONTROL_ARCHIVE_NOT_ORIGINAL_PUBLISHED_GIT_SOURCE')
    with tarfile.open(attempt/'source.tar') as archive:
        require(safe_archive_members(archive.getmembers()), 'CONTROL_UNSAFE_ORIGINAL_ARCHIVE')
        archive_files = {row.name:hashlib.sha256(archive.extractfile(row).read()).hexdigest()
                         for row in archive.getmembers() if row.isfile()}
    require(archive_files == actual, 'CONTROL_ARCHIVE_AND_EXTRACTED_BYTES_DIFFER')
    launchers = {str(verify_member(row)):row['sha256'] for row in lock['launchers']}
    require(set(launchers) == {str(attempt/(role+'.sh')) for role in (*METHODS, 'collector')},
            'CONTROL_ALL_ORIGINAL_LAUNCHERS_REQUIRED')
    for role in (*METHODS, 'collector'):
        require((attempt/(role+'.sh')).read_text() == launcher(attempt, role, manifest),
                'CONTROL_LAUNCHER_OR_SCIENCE_ARGV_CHANGED')
    require({path.name for path in attempt.glob('submitted-*.json')} == {'submitted-FT.json'}
        and {path.name for path in attempt.glob('sbatch-*.json')} == {'sbatch-FT.json'}
        and {path.name for path in attempt.glob('command-*.json')} == {'command-FT.json'},
        'CONTROL_UNKNOWN_PARTIAL_SCHEDULER_RECEIPT_NO_DUPLICATE')
    submitted, sbatch = read(attempt/'submitted-FT.json'), read(attempt/'sbatch-FT.json')
    external = submitted['dependencies']
    require(isinstance(external, list) and external and len(set(external)) == len(external)
        and all(re.fullmatch(r'[1-9][0-9]*', job) for job in external), 'CONTROL_ORIGINAL_EXACT_EXTERNAL_FRONTIER')
    argv = sbatch_argv(attempt, 'FT', manifest, external)
    require(submitted['argv'] == read(attempt/'command-FT.json')['argv'] == sbatch['argv'] == argv
        and sbatch['returncode'] == 0 and sbatch['stdout'].split(';')[0] == submitted['job']
        and failure['jobs'] == {'FT':submitted['job']} and submitted['job'].isdigit(),
        'CONTROL_KNOWN_FT_SUBMISSION_BINDING')
    return dict(attempt=attempt, manifest=manifest, lock=lock, failure= failure_member,
        jobs={'FT':submitted['job']}, external=external, FT_argv=argv,
        immutable_members=dict(manifest=member(attempt/'manifest.json'), lock=member(lock_path),
                              archive=member(attempt/'source.tar'), base_manifest=member(manifest_path)))


def continuation_frontier(bound, observed):
    rows = observed['project']
    attempt = str(bound['attempt'])
    require(not any(row.get('command', '').startswith(attempt+'/')
                    or row.get('workdir', '').startswith(attempt+'/') for row in rows),
            'CONTROL_UNKNOWN_SAME_ATTEMPT_JOB_NO_DUPLICATE')
    current = {row['job'] for row in rows}
    require(set(bound['external']) <= current, 'CONTROL_ORIGINAL_FRONTIER_TRANSITION_REMAIN_HELD')
    surviving = sorted(bound['external'])
    require(frontier(rows) == surviving, 'CONTROL_CHANGED_OR_NEW_EXTERNAL_FRONTIER_REMAIN_HELD')
    return dict(surviving=surviving, original_expected_frontier=bound['external'])


def control_lock_once(path, value):
    """An exclusive side-effect claim, not idempotent preparation/retry."""
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n').encode()
    try:
        with Path(path).open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ValueError('CONTROL_ALREADY_STARTED_NO_RETRY') from error


def reverify_frozen_execution(bound, controller):
    for row in bound['immutable_members'].values():
        verify_member(row)
    verify_member(bound['failure'])
    source = bound['attempt']/'source'
    actual = {str(path.relative_to(source)):file_sha(path) for path in source.rglob('*') if path.is_file()}
    locked = {str(verify_member(row).relative_to(source)):row['sha256'] for row in bound['lock']['source_members']}
    require(actual == locked, 'CONTROL_PRE_RELEASE_SOURCE_SET_CHANGED')
    for row in bound['lock']['launchers']:
        verify_member(row)
    for path, checksum in controller['source_members'].items():
        require(file_sha(REPO/path) == checksum, 'CONTROL_PRE_RELEASE_IMPORTED_BYTES_CHANGED:'+path)


def continue_held_registration(manifest_path, attempt, controller_source, controller_tree):
    """One explicit control-only pass; old FT/source/archive are never replaced."""
    bound = verify_held_attempt(manifest_path, attempt, run=command)
    attempt, manifest, jobs = bound['attempt'], bound['manifest'], dict(bound['jobs'])
    controller = published_control_binding(manifest, controller_source, controller_tree, run=command)
    bound_tracking = tracking_binding(manifest)
    # Never exclude the known FT ID from cap/admission before establishing its
    # current exact owner, held state, source script, full argv and dependencies.
    first = inspect_held(jobs['FT'], 'FT', attempt, manifest, bound['FT_argv'], bound['external'], run=command)
    observed = inventory(exclude=jobs.values())
    external = continuation_frontier(bound, observed)
    actual_admission = admission(manifest, attempt, inspect_inventory=lambda:observed)
    repair = dict(instruction_id=INSTRUCTION, task_id=TASK, repair='HELD_DEPENDENCY_CONJUNCTION_CONTROL_ONLY',
        execution=bound['immutable_members'], old_failure=bound['failure'], controller=controller,
        original_jobs=bound['jobs'], original_frontier=bound['external'], admission=actual_admission,
        surviving_frontier=external, known_held_inspection=first,
        old_source_manifest_archive_launchers_unchanged=True, no_new_scientific_attempt=True,
        tracking=bound_tracking, create_once_no_automatic_retry=True)
    control_lock_once(attempt/'control-repair.lock.json', repair)
    deps, inspections = {'FT':bound['external']}, [first]
    ordered = list(METHODS)
    try:
        for role in [*ordered[1:], 'collector']:
            dep = stage_dependencies(role, 'qualification', ordered, jobs,
                external['surviving'], actual_admission['cap'])
            deps[role] = dep
            argv = sbatch_argv(attempt, role, manifest, dep)
            require(not any((attempt/(prefix+role+'.json')).exists()
                    for prefix in ('submitted-', 'sbatch-', 'command-')), 'CONTROL_MISSING_ROLE_ALREADY_ATTEMPTED')
            write_new(attempt/('command-'+role+'.json'), dict(role=role, argv=argv))
            result = subprocess.run(argv, text=True, capture_output=True, timeout=45)
            write_new(attempt/('sbatch-'+role+'.json'), dict(returncode=result.returncode,
                stdout=result.stdout.strip(), stderr=result.stderr.strip(), argv=argv))
            require(result.returncode == 0, 'SBATCH_REJECTED:'+role+':'+result.stderr.strip()[:800])
            job = result.stdout.strip().split(';')[0]
            require(job.isdigit() and job not in jobs.values(), 'ACTUAL_DISTINCT_JOB_ID_REQUIRED')
            jobs[role] = job
            write_new(attempt/('submitted-'+role+'.json'), dict(job=job, dependencies=dep, argv=argv))
            inspections.append(inspect_held(job, role, attempt, manifest, argv, dep, run=command))
        fresh = inventory(exclude=jobs.values())
        require(not fresh.get('ambiguous') and {row['job'] for row in fresh['project']}
            <= {row['job'] for row in observed['project']}, 'CONTROL_ADMISSION_RACE_REMAIN_HELD')
        final_frontier = continuation_frontier(bound, fresh)
        inspections = [inspect_held(jobs[role], role, attempt, manifest,
            read(attempt/('command-'+role+'.json'))['argv'], deps[role], run=command)
            for role in [*ordered, 'collector']]
        reverify_frozen_execution(bound, controller)
        write_new(attempt/'held-inspection.json', dict(jobs=inspections, admission=actual_admission,
            fresh=fresh, cap_bound=actual_admission['cap'], actual_GPU_PASS=False,
            source_manifest_verified=True, control_repair=member(attempt/'control-repair.lock.json')))
        for role in reversed([*ordered, 'collector']):
            answer = command(['scontrol', 'release', jobs[role]])
            write_new(attempt/('released-'+role+'.json'), dict(job=jobs[role], released=True, response=answer))
    except BaseException as error:
        write_new(attempt/'control-repair-failure.json', dict(status='CONTROL_CONTINUATION_BLOCKED', jobs=jobs,
            error_type=type(error).__name__, error=str(error)[:1200],
            control_repair=member(attempt/'control-repair.lock.json'), original_failure_preserved=True,
            no_automatic_retry=True, no_cancellation=True))
        raise
    result = dict(instruction_id=INSTRUCTION, task_id=TASK, status='SUBMISSION_HANDOFF', stage='qualification',
        jobs=jobs, dependencies=deps, source=bound['lock']['source'],
        base_manifest_sha256=manifest['base_manifest_sha256'], registration_roles=ordered, resume_binding=None,
        manifest=member(attempt/'manifest.json'), lock=member(attempt/'execution.lock.json'),
        collector=jobs['collector'], resources=manifest['resources'], cap=actual_admission['cap'],
        frontier=external['surviving'], original_execution_source=manifest['code_commit'],
        control_repair=member(attempt/'control-repair.lock.json'), original_failure=bound['failure'],
        existing_FT_reused_not_resubmitted=True, controller=controller,
        initial_snapshot=command(['squeue', '-h', '-j', ','.join(jobs.values()), '-o', '%i|%j|%T|%b|%N|%r']),
        W_B='NEW_RUN_ACTUAL_STARTUP_NOT_OBSERVED', actual_GPU_qualification='NOT_OBSERVED',
        scientific_complete=False, monitoring_active=False, automatic_retry=False)
    write_new(attempt/'submission.json', result)
    return result


def submit(manifest_path, out, stage, *, qualification_receipt=None, smoke_receipt=None,
           smoke_only=False, resume=None, method=None, attempt_name='registration-r1'):
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    require(out == OUTPUT or OUTPUT in out.parents, 'TASK_LOCAL_OUTPUT_SCOPE')
    require(re.fullmatch(r'registration-[A-Za-z0-9_-]+', attempt_name), 'EXACT_IMMUTABLE_ATTEMPT_NAME')
    require(manifest_path.is_file() and not manifest_path.is_symlink(), 'REGULAR_MANIFEST_REQUIRED')
    base = read(manifest_path)
    base_sha = file_sha(manifest_path)
    require(base['model'] == 'gptj' and base['owner']['server'] == 'server2'
        and base['owner']['session'] == SESSION and base['instruction_id'] == INSTRUCTION,
        'OFFICIAL_SERVER2_USER_IDENTITY')
    source = sealed_source(base['code_commit'], base['official_tree_sha256'])
    bound_tracking = tracking_binding(base)
    manifest = dict(base, registration_stage=stage, base_manifest_sha256=base_sha)
    require(method is None or resume is not None, 'METHOD_ONLY_WITH_EXPLICIT_RESUME')
    ordered = roles(stage, method if resume else None, smoke_only)
    if stage == 'no_gpu_qual':
        require(noqual.enabled(manifest) and not any((resume, method, smoke_only, qualification_receipt, smoke_receipt)),
            'EXACT_USER_DISABLED_NO_OLD_PROOF_OR_RESUME')
    if stage == 'zsre_pipeline':
        from official.runners.server2.zsre_profile import enabled
        require(enabled(manifest) and not any((resume,method,smoke_only,qualification_receipt,smoke_receipt)),
                'EXACT_ZSRE_PIPELINE_PLAN_NOT_ACTUAL_PASS')
    if stage == 'cf_checkpoint':
        require(deferred(manifest) and not any((resume, method, smoke_only, qualification_receipt)),
                'EXACT_CF_CHECKPOINT_PIPELINE_ONLY')
    if stage == 'cf':
        require(qualification_receipt is not None, 'ACTUAL_QUALIFICATION_NOT_AVAILABLE')
        manifest['qualification_receipt'] = gate(qualification_receipt, manifest=manifest, kind='qualification')
    if stage == 'zsre' and not smoke_only:
        require(smoke_receipt is not None, 'ACTUAL_ZSRE_SMOKE_NOT_AVAILABLE')
        manifest['zsre_smoke_receipt'] = gate(smoke_receipt, manifest=manifest, kind='smoke')
        manifest['W0_zsre_ready_path'] = str(verify_member(read(smoke_receipt)['W0member']))
    if resume:
        require(method is not None and not smoke_only, 'EXACT_RESUME_METHOD_REQUIRED')
        from official.runners.server2.run import checkpoint_identity
        identity = checkpoint_identity(base, method, stage)
        require(identity['code_commit'] == base['code_commit']
            and identity['official_tree_sha256'] == base['official_tree_sha256'], 'RESUME_SOURCE_BINDING')
        manifest['resume'] = resume_binding(resume, identity, method=method, dataset=stage)
    out.mkdir(parents=True, exist_ok=True)
    existing = _existing(out, base_sha, stage, ordered, manifest.get('resume'))
    if existing:
        return dict(existing, current_registration_snapshot=existing_snapshot(existing))
    attempt = out/attempt_name
    require(not attempt.exists(), 'ATTEMPT_ALREADY_EXISTS_NO_AUTORETRY')
    if stage == 'no_gpu_qual':
        for dataset in ('cf', 'zsre'):
            manifest['W0_'+dataset+'_ready_path'] = str(attempt/('W0_'+dataset.upper())/'READY.json')
            manifest['w0_'+dataset+'_output'] = str(attempt/('W0_'+dataset.upper()))
    if stage == 'zsre_pipeline':
        manifest['W0_zsre_ready_path'] = str(attempt/'W0_ZSRE'/'READY.json')
        manifest['w0_zsre_output'] = str(attempt/'W0_ZSRE')
    if stage == 'cf' and not resume:
        manifest['W0_cf_ready_path'] = str(attempt/'W0_CF'/'READY.json')
        manifest['w0_cf_output'] = str(attempt/'W0_CF')
    elif stage == 'cf_checkpoint':
        manifest['W0_cf_ready_path'] = str(attempt/'W0_CF'/'READY.json')
        manifest['w0_cf_output'] = str(attempt/'W0_CF')
    elif stage == 'cf':
        original_manifest = read(manifest['resume']['original_manifest']['path'])
        original_w0 = w0_binding(original_manifest['W0_cf_ready_path'], manifest=manifest, dataset='cf')
        manifest['W0_cf_ready_path'] = original_w0['path']
        manifest['resume_original_W0_member'] = original_w0
        manifest['w0_cf_output'] = str(Path(original_w0['path']).parent)
    if stage == 'zsre' and smoke_only:
        manifest['W0_zsre_ready_path'] = str(attempt/'W0_ZSRE'/'READY.json')
        manifest['w0_zsre_output'] = str(attempt/'W0_ZSRE')
    manifest['registration_roles'] = ordered
    manifest['collector_mode'] = stage
    actual_admission = admission(manifest, out)
    if resume:
        manifest['resume_writer_guard'] = resume_writer_guard(manifest['resume'], actual_admission['before']['project'])
    attempt.mkdir()
    (attempt/'source').mkdir()
    archive = attempt/'source.tar'
    command(['git', 'archive', '--format=tar', '--output='+str(archive), base['code_commit'], 'official'])
    with tarfile.open(archive) as stream:
        require(safe_archive_members(stream.getmembers()), 'OFFICIAL_ONLY_SAFE_SOURCE_ARCHIVE')
        stream.extractall(attempt/'source', filter='data')
    write_new(attempt/'manifest.json', manifest)
    for role in [*ordered, 'collector']:
        (attempt/(role+'.sh')).write_text(launcher(attempt, role, manifest, resume=resume))
    lock = dict(instruction_id=INSTRUCTION, task_id=TASK, source=source,
        official_tree_sha256=base['official_tree_sha256'], code_commit=base['code_commit'],
        base_manifest=member(manifest_path), manifest=member(attempt/'manifest.json'),
        archive=member(archive), source_members=[member(path) for path in sorted((attempt/'source').rglob('*')) if path.is_file()],
        launchers=[member(attempt/(role+'.sh')) for role in [*ordered, 'collector']],
        resources=manifest['resources'], stage=stage, checkpoint_latest1=True, tracking=bound_tracking,
        actual_qualification=noqual.DISABLED if stage == 'no_gpu_qual' else 'NOT_OBSERVED_AT_REGISTRATION' if stage in ('qualification','cf_checkpoint','zsre_pipeline') else 'ACTUAL_RECEIPT_BOUND',
        all_existing_jobs_preserved=True, source_freeze_distinct_from_publication=True)
    write_new(attempt/'execution.lock.json', lock)
    jobs, deps, inspections = {}, {}, []
    try:
        for role in [*ordered, 'collector']:
            dep = stage_dependencies(role, stage, ordered, jobs, actual_admission['frontier'], actual_admission['cap'])
            deps[role] = dep
            argv = sbatch_argv(attempt, role, manifest, dep)
            write_new(attempt/('command-'+role+'.json'), dict(role=role, argv=argv))
            result = subprocess.run(argv, text=True, capture_output=True, timeout=45)
            write_new(attempt/('sbatch-'+role+'.json'), dict(returncode=result.returncode,
                stdout=result.stdout.strip(), stderr=result.stderr.strip(), argv=argv))
            require(result.returncode == 0, 'SBATCH_REJECTED:'+role+':'+result.stderr.strip()[:800])
            job = result.stdout.strip().split(';')[0]
            require(job.isdigit() and job not in jobs.values(), 'ACTUAL_DISTINCT_JOB_ID_REQUIRED')
            jobs[role] = job
            write_new(attempt/('submitted-'+role+'.json'), dict(job=job, dependencies=dep, argv=argv))
            inspections.append(inspect_held(job, role, attempt, manifest, argv, dep))
        fresh = inventory(exclude=jobs.values())
        require(not fresh.get('ambiguous'), 'ADMISSION_RACE_PROJECT_SOURCE_UNRESOLVED_REMAIN_HELD')
        require({row['job'] for row in fresh['project']} <= {row['job'] for row in actual_admission['before']['project']},
            'ADMISSION_RACE_NEW_JOBS_REMAIN_HELD')
        write_new(attempt/'held-inspection.json', dict(jobs=inspections, admission=actual_admission,
            fresh=fresh, cap_bound=actual_admission['cap'], actual_GPU_PASS=False, source_manifest_verified=True))
        for role in reversed([*ordered, 'collector']):
            answer = command(['scontrol', 'release', jobs[role]])
            write_new(attempt/('released-'+role+'.json'), dict(job=jobs[role], released=True, response=answer))
    except BaseException as error:
        write_new(attempt/'submission-failure.json', dict(status='REGISTRATION_OR_RELEASE_BLOCKED', jobs=jobs,
            error_type=type(error).__name__, error=str(error)[:1200], lock=member(attempt/'execution.lock.json'),
            no_automatic_retry=True, no_cancellation=True, original_jobs_unchanged=True))
        raise
    result = dict(instruction_id=INSTRUCTION, task_id=TASK, status='SUBMISSION_HANDOFF', stage=stage,
        jobs=jobs, dependencies=deps, source=source, base_manifest_sha256=base_sha,
        registration_roles=ordered, resume_binding=manifest.get('resume'),
        manifest=member(attempt/'manifest.json'), lock=member(attempt/'execution.lock.json'),
        collector=jobs['collector'], resources=manifest['resources'], cap=actual_admission['cap'],
        frontier=actual_admission['frontier'], initial_snapshot=command(['squeue', '-h', '-j',
            ','.join(jobs.values()), '-o', '%i|%j|%T|%b|%N|%r']),
        W_B='NEW_RUN_ACTUAL_STARTUP_NOT_OBSERVED', actual_GPU_qualification=noqual.DISABLED if stage == 'no_gpu_qual' else 'NOT_OBSERVED',
        scientific_complete=False, monitoring_active=False, automatic_retry=False)
    write_new(attempt/'submission.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--stage', choices=('qualification', 'cf', 'zsre', 'cf_checkpoint', 'zsre_pipeline', 'no_gpu_qual'), required=True)
    parser.add_argument('--qualification-receipt', type=Path)
    parser.add_argument('--smoke-receipt', type=Path)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--attempt-name', default='registration-r1')
    parser.add_argument('--continue-held', type=Path)
    parser.add_argument('--controller-source')
    parser.add_argument('--controller-official-tree')
    args = parser.parse_args()
    if args.continue_held is not None:
        require(args.stage == 'qualification' and not any((args.qualification_receipt,
            args.smoke_receipt, args.smoke_only, args.resume, args.method)), 'CONTROL_QUALIFICATION_ONLY_NO_NEW_SCIENCE')
        print(json.dumps(continue_held_registration(args.manifest, args.continue_held,
            args.controller_source, args.controller_official_tree), sort_keys=True))
        return
    require(args.controller_source is None and args.controller_official_tree is None,
            'CONTROL_EVIDENCE_ONLY_WITH_EXPLICIT_CONTINUE_HELD')
    print(json.dumps(submit(args.manifest, args.out, args.stage,
        qualification_receipt=args.qualification_receipt, smoke_receipt=args.smoke_receipt,
        smoke_only=args.smoke_only, resume=args.resume, method=args.method, attempt_name=args.attempt_name),
        sort_keys=True))


if __name__ == '__main__':
    main()
