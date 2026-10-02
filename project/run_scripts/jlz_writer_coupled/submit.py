"""Create-once frozen v5 jobs: held inspection before release, no polling.

Only invoking main performs scheduler I/O. Qualification and science are
separate explicit submissions; no daemon, retry, baseline, or v4 resume.
"""
from __future__ import annotations

import argparse
import getpass
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile

from .common import ROOT, LOCAL, INSTRUCTION, TASK, digest, member, require, sha, write
from .prepare import AUTHORITY, ENVELOPE, EXCEPTION, DESIGN, CASE, EVALUATION, verify_member


PYTHON = '/data/janghj/EasyEdit/.venv/bin/python'
SOURCE_PATHS = [
    'project/run_scripts/jlz_writer_coupled', 'project/run_scripts/jlz_pilot/prompts.py',
    'project/run_scripts/jlz_pilot/__init__.py', DESIGN, CASE, EVALUATION,
    'docs/methods/jlz-writer-coupled-v5.tex',
    'experiment-reports/global/2026-10-02-jlz-v4-method-structure-audit/report-ko.md',
    ENVELOPE, EXCEPTION, 'audits/global/2026-10-02-jlz-v5-sh4-dispatch/v4-cancellation-receipt.json',
    'scripts/check-slurm-resource-cap.sh',
    'scripts/check-slurm-gpu-cap.sh', 'scripts/slurm_memory_policy.py',
    'servers/slurm-memory-policy.tsv', 'control/gpu-concurrency-policy.tsv',
]


def command(argv, *, cwd=None, env=None):
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    require(result.returncode == 0, repr(argv) + ':' + result.stderr + result.stdout)
    return result.stdout.strip()


def _copy_once(source, target):
    with Path(source).open('rb') as origin, Path(target).open('xb') as destination:
        shutil.copyfileobj(origin, destination)
        destination.flush()
        os.fsync(destination.fileno())
    require(sha(source) == sha(target), 'COPY_BYTES_CHANGED')


def _write_script(path, content):
    with Path(path).open('x') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o700)


def launcher(source, source_commit, module, arguments, *, cpu_only=False):
    lines = ['#!/usr/bin/env bash', 'set -euo pipefail', 'umask 077',
             'export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8',
             'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false',
             'export PYTHONDONTWRITEBYTECODE=1 SLURM_EXPORT_ENV=ALL',
             'export ODEEDIT_SOURCE_COMMIT=' + shlex.quote(source_commit),
             'export PYTHONPATH=' + shlex.quote(str(source))]
    if cpu_only:
        lines.append("export CUDA_VISIBLE_DEVICES=''")
    lines += ['cd ' + shlex.quote(str(source)), 'exec ' + shlex.join([PYTHON, '-m', module, *arguments]), '']
    return '\n'.join(lines)


def freeze(config_path, attempt, report):
    """Archive only clean committed source into this task's new namespace."""
    config_path, attempt = Path(config_path).resolve(), Path(attempt).resolve()
    require(attempt.parent == LOCAL and attempt.name.startswith('attempt-') and not attempt.exists(), 'NEW_TASK_ATTEMPT_ONLY')
    require(config_path.is_relative_to(LOCAL), 'TASK_LOCAL_CONFIG_ONLY')
    config = json.loads(config_path.read_text())
    require(config['instruction_id'] == INSTRUCTION and config['task_id'] == TASK
            and config['authority'] == AUTHORITY, 'CONFIG_AUTHORITY')
    require(not command(['git', 'status', '--porcelain', '--', *SOURCE_PATHS], cwd=ROOT), 'COMMIT_SOURCE_BEFORE_FREEZE')
    commit = command(['git', 'rev-parse', 'HEAD'], cwd=ROOT)
    tree = command(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT)
    require(config['resources']['cap'] == 2 and config['resources']['gpu'] == 1
            and config['resources']['host_mib'] == 60416, 'RESOURCE_SCOPE')
    require(shutil.disk_usage(LOCAL).free >= config['resources']['reserve_bytes'], 'STORAGE_RESERVE')
    for row in config['runtime']['source_members'] + config['native_reference']:
        verify_member(row)
    for key in ('envelope', 'exception'):
        verify_member(config['authority_binding'][key])
    require(config['authority_binding']['envelope']['sha256'] == sha(ROOT / ENVELOPE), 'SOURCE_ENVELOPE')
    require(config['authority_binding']['exception']['sha256'] == sha(ROOT / EXCEPTION), 'SOURCE_EXCEPTION')
    attempt.mkdir()
    source = attempt / 'source'
    source.mkdir()
    archive = attempt / 'source.tar'
    command(['git', 'archive', '--format=tar', '--output=' + str(archive), commit, *SOURCE_PATHS], cwd=ROOT)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        require(len({row.name for row in members}) == len(members), 'ARCHIVE_DUPLICATE')
        for row in members:
            path = Path(row.name)
            require((row.isfile() or row.isdir()) and not path.is_absolute() and '..' not in path.parts,
                    'ARCHIVE_REGULAR_SCOPED_PATH')
        bundle.extractall(source, filter='data')
    for name in ('run.py', 'collect.py'):
        require((source / 'project/run_scripts/jlz_writer_coupled' / name).is_file(), 'MISSING_FROZEN_RUNNER:' + name)
    _copy_once(config_path, attempt / 'config.json')
    bridge_path = verify_member(config['w0_reuse']['receipt'])
    _copy_once(bridge_path, attempt / 'W0-reuse.json')
    bridge = json.loads(bridge_path.read_text())
    verify_member(bridge['observations'])
    write(attempt / 'W0-raw-inventory.json', dict(bridge=member(attempt / 'W0-reuse.json'),
                                                observations=bridge['observations'], new_W0_forward=0,
                                                raw_copied=False, original_preserved=True))
    qualification_path = attempt / 'qualification/qualification.json'
    common = ['--config', str(attempt / 'config.json'), '--attempt', str(attempt)]
    _write_script(attempt / 'qualification.sh', launcher(source, commit, 'project.run_scripts.jlz_writer_coupled.run',
                                                        ['--phase', 'qualification', *common]))
    for arm in ('A', 'B'):
        _write_script(attempt / ('arm-' + arm + '.sh'), launcher(source, commit, 'project.run_scripts.jlz_writer_coupled.run',
                      ['--phase', 'arm', '--arm', arm, *common, '--qualification', str(qualification_path)]))
    for name, destination in [('qualification-collector', attempt / 'qualification-report'), ('collector', report)]:
        _write_script(attempt / (name + '.sh'), launcher(source, commit, 'project.run_scripts.jlz_writer_coupled.collect',
                      ['--attempt', str(attempt), '--report', str(destination)], cpu_only=True))
    sources = [member(path) for path in sorted(source.rglob('*')) if path.is_file()]
    write(attempt / 'execution.lock.json', dict(instruction_id=INSTRUCTION, task_id=TASK,
          authority_commit=AUTHORITY, source_commit=commit, source_tree=tree, archive=member(archive),
          execution_path=str(source), worktree=str(ROOT), config_sha256=sha(attempt / 'config.json'),
          original_config=member(config_path), source_members=sources, source_root_sha256=digest(sources),
          launchers=[member(path) for path in sorted(attempt.glob('*.sh'))], resources=config['resources'],
          runtime_sources=config['runtime']['source_members'], native_reference=config['native_reference'],
          qualification_path=str(qualification_path), report=str(report),
          session='01a04939-b5c7-7a03-ba2d-ef3343d62cfd', owner=getpass.getuser(), actor='SH4', hostname='server4',
          checkpoint_saved=False, exact_resume='NOT_AVAILABLE', no_other_task_resume=True,
          flow='one common qualification, then independent A/B persistent pilot2/timing3/coldmain20; afterany CPU collection'))
    return config


def verify_frozen(attempt, original_config):
    lock = json.loads((attempt / 'execution.lock.json').read_text())
    config = json.loads((attempt / 'config.json').read_text())
    require(lock['instruction_id'] == config['instruction_id'] == INSTRUCTION
            and lock['task_id'] == config['task_id'] == TASK, 'FROZEN_TASK')
    require(sha(original_config) == sha(attempt / 'config.json') == lock['config_sha256'], 'FROZEN_CONFIG_CHANGED')
    require(lock['owner'] == getpass.getuser() and lock['hostname'] == 'server4', 'FROZEN_OWNER')
    verify_member(lock['archive'])
    for row in lock['source_members'] + lock['launchers'] + lock['runtime_sources'] + lock['native_reference']:
        verify_member(row)
    for row in config['assets']:
        stat = Path(row['path']).stat()
        require((stat.st_size, stat.st_ino, stat.st_mtime_ns) == (row['bytes'], row['inode'], row['mtime_ns']), 'ASSET_STAT_CHANGED')
    verify_member(config['w0_reuse']['receipt'])
    require(shutil.disk_usage(attempt).free >= lock['resources']['reserve_bytes'], 'STORAGE_RESERVE')
    return lock, config


def qualification_binding(path, attempt, lock):
    path = Path(path).resolve()
    require(str(path) == lock['qualification_path'], 'EXACT_QUALIFICATION_PATH')
    value = json.loads(path.read_text())
    require(value['status'] == 'QUALIFIED' and value['instruction'] == INSTRUCTION, 'QUALIFICATION_STATUS')
    require(value['source_commit'] == lock['source_commit']
            and value['config_sha256'] == lock['config_sha256'], 'QUALIFICATION_SOURCE_CONFIG')
    epsilon = value['epsilon_num_per_request']
    require(isinstance(epsilon, (float, int)) and math.isfinite(epsilon) and epsilon >= 0, 'QUALIFIED_EPSILON')
    require(value['epsilon_num_units'] == 'request_mean; solver uses actual_B multiplier', 'EPSILON_UNITS')
    require(value['route'] in ('direct', 'dense') and type(value['cached']) is bool, 'QUALIFIED_ROUTE')
    require(1 <= value['qualification_pair_count'] <= 6 and value['actual_probe_limit_total'] == 6
            and 0 <= value['physical_forward'] <= 12 and 0 <= value['physical_backward'] <= 12,
            'QUALIFICATION_BUDGET')
    return dict(receipt=member(path), epsilon_num_per_request=epsilon,
                epsilon_num_units=value['epsilon_num_units'], route=value['route'], cached=value['cached'],
                source_commit=value['source_commit'], config_sha256=value['config_sha256'],
                no_result_based_tolerance_change=True)


def admission(exclude=()):
    raw = command(['squeue', '-h', '-r', '-w', 'server4', '-o', '%i|%u|%j|%T|%b|%R'])
    jobs = []
    for line in raw.splitlines():
        job, user, name, state, gres, reason = line.split('|', 5)
        if job in exclude:
            continue
        if user != getpass.getuser() and not name.startswith(('odeedit_', 'bfode_', 'motivation_', 'session01_')):
            continue
        detail = command(['scontrol', 'show', 'job', job, '--oneliner'])
        match = re.search(r'\bReqTRES=([^ ]+)', detail)
        require(match, 'UNKNOWN_RESOURCE')
        gpu = re.search(r'gres/gpu=(\d+)', match[1])
        if gpu and int(gpu[1]) > 0:
            jobs.append(dict(job=job, user=user, name=name, state=state, gpus=int(gpu[1]), reason=reason))
    return dict(jobs=jobs, raw=raw, scope='resource-only current owner/project server4; no mutation')


def dependencies(*groups):
    parts = [kind + ':' + ':'.join(dict.fromkeys(ids)) for kind, ids in groups if ids]
    return ','.join(parts) if parts else None


def dependency_members(value):
    # Slurm versions use either colon lists or repeated comma-separated
    # dependency types, with fulfillment markers after individual IDs.
    clean = re.sub(r'\([^)]*\)', '', value)
    result = set()
    for group in clean.split(','):
        if not group:
            continue
        match = re.fullmatch(r'(afterok|afterany):([0-9]+(?:_[0-9]+)?(?::[0-9]+(?:_[0-9]+)?)*)', group)
        require(match, 'UNSUPPORTED_DEPENDENCY_REPRESENTATION:' + group)
        result.update((match[1], job) for job in match[2].split(':'))
    return result


def job_name(name):
    return 'odeedit_jlz_v5_s4_' + name.replace('-', '_')


def expected_dependencies(value):
    result = set()
    for group in (value or '').split(','):
        if group:
            kind, *ids = group.split(':')
            result.update((kind, job) for job in ids)
    return result


def inspect(job, name, gpu, dependency, attempt, argv, resources):
    detail = command(['scontrol', 'show', 'job', job, '--oneliner'])
    script = attempt / (name + '.sh')
    for term in [f'JobId={job} ', f'JobName={job_name(name)} ', 'UserId=' + getpass.getuser() + '(',
                 'JobState=PENDING ', 'Reason=JobHeldUser ', 'Requeue=0 ', 'NumCPUs=8 ',
                 'ReqNodeList=server4 ', 'Partition=gpu ', 'QOS=lab_gpu_s4 ']:
        require(term in detail, 'HELD_INSPECTION:' + term)
    require(('TresPerNode=gres/gpu:1' in detail) if gpu else 'gres/gpu' not in detail, 'HELD_GPU')
    memory = resources['host_mib'] if gpu else resources['collector_host_mib']
    require(f'mem={memory}M' in detail or f'mem={memory // 1024}G' in detail, 'HELD_MEMORY')
    wall = resources['qualification_wall'] if name == 'qualification' else resources['wall'] if gpu else resources['collector_wall']
    allowed_wall = {wall, '1-00:00:00'} if wall == '24:00:00' else {wall}
    require(any('TimeLimit=' + value + ' ' in detail for value in allowed_wall), 'HELD_WALL')
    command_field = re.search(r'\bCommand=(.*?)(?= [A-Z][A-Za-z]+=|$)', detail)
    require(command_field and command_field[1] == str(script), 'HELD_COMMAND_PATH')
    require('WorkDir=' + str(attempt / 'source') + ' ' in detail, 'HELD_SOURCE_DIRECTORY')
    field = re.search(r'\bDependency=([^ ]+)', detail)
    require(field and dependency_members(field[1]) == expected_dependencies(dependency), 'HELD_DEPENDENCY_EXACT')
    submit = re.search(r'\bSubmitLine=(.*?)(?= WorkDir=|$)', detail)
    require(submit and shlex.split(submit[1]) == argv, 'HELD_FULL_ARGV')
    require(command(['scontrol', 'write', 'batch_script', job, '-']).strip() == script.read_text().strip(), 'HELD_SCRIPT_BYTES')
    require('--export=NONE' in argv and '--no-requeue' in argv and '--hold' in argv, 'HELD_EXPORT_REQUEUE')
    return dict(job=job, name=name, requested_argv=argv, scontrol=detail, script_sha256=sha(script))


def sbatch_argv(name, gpu, dependency, attempt, resources):
    wall = resources['qualification_wall'] if name == 'qualification' else resources['wall'] if gpu else resources['collector_wall']
    memory = resources['host_mib'] if gpu else resources['collector_host_mib']
    argv = ['sbatch', '--parsable', '--hold', '--partition=gpu', '--qos=lab_gpu_s4', '--nodelist=server4',
            '--nodes=1', '--ntasks=1', '--cpus-per-task=8', '--export=NONE', '--no-requeue',
            '--job-name=' + job_name(name), '--chdir=' + str(attempt / 'source'), '--mem=' + str(memory) + 'M',
            '--time=' + wall, '--output=' + str(attempt / (name + '-%j.out')),
            '--error=' + str(attempt / (name + '-%j.err'))]
    if gpu:
        argv.append('--gres=gpu:1')
    if dependency:
        argv.append('--dependency=' + dependency)
        if 'afterok:' in dependency:
            argv.append('--kill-on-invalid-dep=yes')
    argv.append(str(attempt / (name + '.sh')))
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('qualification', 'arms'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    require(attempt.parent == LOCAL and attempt.name.startswith('attempt-'), 'TASK_LOCAL_ATTEMPT')
    report = args.report.resolve() if args.report else attempt / 'report'
    require(report.is_relative_to(LOCAL), 'TASK_LOCAL_RAW_REPORT')
    if args.mode == 'qualification':
        require(args.qualification is None, 'NO_PREEXISTING_QUALIFICATION_FOR_NEW_ATTEMPT')
        freeze(args.config, attempt, report)
    lock, config = verify_frozen(attempt, args.config)
    require(not (attempt / (args.mode + '-submission.json')).exists(), 'NO_DUPLICATE_STAGE_SUBMISSION')
    names = ['qualification', 'qualification-collector'] if args.mode == 'qualification' else ['arm-A', 'arm-B', 'collector']
    require(not any((attempt / ('submitted-' + name + '.json')).exists() for name in names), 'NO_IMPLICIT_PARTIAL_RESUBMISSION')
    qualifier = None
    qualified = None
    if args.mode == 'arms':
        require(args.qualification is not None, 'SCIENCE_REQUIRES_SEALED_QUALIFICATION')
        qualified = qualification_binding(args.qualification, attempt, lock)
        previous = json.loads((attempt / 'qualification-submission.json').read_text())
        require(previous['status'] == 'RELEASED', 'QUALIFICATION_NOT_RELEASED')
        qualifier = previous['jobs']['qualification']
        write(attempt / 'qualification-seal.json', dict(instruction_id=INSTRUCTION, **qualified,
                                                       execution_lock=member(attempt / 'execution.lock.json')))
    check = subprocess.run(['bash', str(attempt / 'source/scripts/check-slurm-resource-cap.sh'), 'server4', '1', '60416M'],
                           env=dict(os.environ, AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'),
                           capture_output=True, text=True)
    require(check.returncode in (0, 4), 'RESOURCE_HELPER_ERROR:' + check.stdout + check.stderr)
    helper = dict(exit_code=check.returncode, output=check.stdout + check.stderr,
                  limitation='exit4 is dependency-pending admission, not a helper PASS')
    before = admission()
    external = [row['job'] for row in before['jobs'] if row['job'] != qualifier]
    ids, inspected = {}, []
    for name in names:
        gpu = name in ('qualification', 'arm-A', 'arm-B')
        if name == 'qualification':
            dependency = dependencies(('afterany', external))
        elif name in ('arm-A', 'arm-B'):
            dependency = dependencies(('afterok', [qualifier]), ('afterany', external))
        elif name == 'qualification-collector':
            dependency = dependencies(('afterany', [ids['qualification']]))
        else:
            dependency = dependencies(('afterany', [qualifier, ids['arm-A'], ids['arm-B']]))
        argv = sbatch_argv(name, gpu, dependency, attempt, config['resources'])
        job = command(argv).split(';')[0]
        require(job.isdigit(), 'INVALID_JOB_ID')
        ids[name] = job
        write(attempt / ('submitted-' + name + '.json'), dict(job=job, argv=argv, status='HELD',
              instruction_id=INSTRUCTION, task_id=TASK, execution_lock_sha256=sha(attempt / 'execution.lock.json')))
        inspected.append(inspect(job, name, gpu, dependency, attempt, argv, config['resources']))
    after = admission(exclude=tuple(ids.values()))
    require({row['job'] for row in after['jobs']} <= set(external) | ({qualifier} if qualifier else set()), 'ADMISSION_RACE_KEEP_OWN_JOBS_HELD')
    # Recheck all immutable sources and selected qualification after the held
    # graph inspection and before releasing any member of this stage.
    verify_frozen(attempt, args.config)
    if qualified is not None:
        require(member(args.qualification)['sha256'] == qualified['receipt']['sha256'], 'QUALIFICATION_CHANGED_BEFORE_RELEASE')
    inspection_path = attempt / (args.mode + '-held-inspection.json')
    write(inspection_path, dict(jobs=inspected, before=before, prerelease=after, helper=helper,
          external_afterany=external, max_concurrent_gpu=1 if args.mode == 'qualification' else 2,
          execution_lock_sha256=sha(attempt / 'execution.lock.json'), qualification=qualified,
          all_stage_jobs_held_before_any_release=True, no_other_job_mutation=True))
    for name in reversed(names):
        output = command(['scontrol', 'release', ids[name]])
        write(attempt / ('released-' + name + '.json'), dict(job=ids[name], command_succeeded=True, output=output))
    submission = dict(status='RELEASED', mode=args.mode, jobs=ids, instruction_id=INSTRUCTION, task_id=TASK,
                      held_inspection=member(inspection_path), execution_lock=member(attempt / 'execution.lock.json'),
                      qualification=qualified, initial='NOT_OBSERVED', no_other_job_mutation=True,
                      monitoring_active=False, automatic_resume=False)
    write(attempt / (args.mode + '-submission.json'), submission)
    if args.mode == 'arms':
        submission['jobs'] = dict(previous['jobs'], **ids)
        write(attempt / 'submission.json', submission)
    print(json.dumps(dict(status='RELEASED', mode=args.mode, jobs=ids, initial='NOT_OBSERVED')))


if __name__ == '__main__':
    main()
