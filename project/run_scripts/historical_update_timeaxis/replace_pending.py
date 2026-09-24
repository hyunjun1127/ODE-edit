"""One-shot, exact-ID pending replacement authorized by the user's repair recall.

No execution monitoring, model loading, result reads, or automatic retry.
The old immutable attempt is never modified.
"""
import argparse
import datetime
import subprocess

from .common import *
from .launch import freeze, submit

ATTEMPT = 'attempt-r2-routing'
OLD_GPU = ('53176', '53177')
OLD_CPU = '53179'
OLD = ROOT / 'attempt-v1'


def command(argv):
    return subprocess.check_output(argv, text=True).strip()


def inspect_old(job):
    info = command(['scontrol', 'show', 'job', '-o', job])
    expected = OLD / ('collector-repair-r1/collector.sbatch' if job == OLD_CPU else 'gpu.sbatch')
    assert f'JobId={job} ' in info and 'UserId=janghj(' in info
    assert f'Command={expected} ' in info and 'ReqNodeList=server4 ' in info
    assert 'JobState=PENDING ' in info and 'RunTime=00:00:00 ' in info, ('NOT_UNSTARTED_PENDING', job, info)
    assert 'AllocTRES=(null)' in info, ('ALLOCATION_EXISTS', job)
    if job == OLD_CPU:
        assert 'afterany:53176' in info and 'afterany:53177' in info
    else:
        family = FAMILIES[OLD_GPU.index(job)]
        assert f'gpu.sbatch {family} ' in info and 'Dependency=(null)' in info
    return info


def prepare():
    old_lock = read(OLD / 'execution.lock.json')
    assert sha(OLD / 'execution.lock.json') == 'bc8ae75e983c68a1eb4489e9c3be98e0f77f8a567cf32e9eb88d6b9e1c994d9f'
    assert read(OLD / 'submission.json')['family_jobs'] == dict(zip(FAMILIES, OLD_GPU))
    assert read(OLD / 'collector-repair-r1/receipt.json')['new_collector'] == int(OLD_CPU)
    t0 = old_lock['T0']; assert sha(t0['path']) == t0['sha256']
    binding = read(t0['path'])
    checked = []
    for family, batches in binding['checkpoints'].items():
        for batch, r in batches.items():
            s = Path(r['path']).stat()
            assert (s.st_size, s.st_ino, s.st_mtime_ns) == (r['bytes'], r['inode'], r['mtime_ns'])
            checked.append(dict(family=family, batch=batch, path=r['path'], sha256=r['sha256'], verification='PRIOR_FULL_SHA_PLUS_CURRENT_UNCHANGED_STAT'))
    assert len(checked) == 24
    assert sha(binding['token_manifest']['path']) == binding['token_manifest']['sha256']
    assert sha(DESIGN / 'experiment-contract.json') == old_lock['contract_sha256']
    dest = freeze(ATTEMPT)
    new_lock = read(dest / 'execution.lock.json')
    for field in ('contract_sha256', 'T0_sha256', 'token_sha256', 'evaluator_signature', 'save_checkpoints', 'scope'):
        assert new_lock[field] == old_lock[field], ('UNINTENDED_CONTRACT_CHANGE', field)
    for member in old_lock['source_members']:
        name = Path(member['path']).relative_to(OLD / 'source')
        if name.name == 'backend.py':
            continue
        if name.name == 'reduce.py':
            baseline = OLD / 'collector-repair-r1/source' / name
        else:
            baseline = Path(member['path'])
        assert sha(dest / 'source' / name) == sha(baseline), ('UNRELATED_SOURCE_CHANGE', str(name))
    for script in ('gpu.sbatch', 'collector.sbatch'):
        subprocess.run(['bash', '-n', str(dest / script)], check=True)
    audit = subprocess.run(['python3', 'scripts/slurm_memory_policy.py', 'audit',
                            str(dest / 'gpu.sbatch'), str(dest / 'collector.sbatch')],
                           capture_output=True, text=True)
    save(dest / 'script-memory-audit.json', dict(exit_code=audit.returncode, stdout=audit.stdout, stderr=audit.stderr))
    assert audit.returncode == 0, ('SCRIPT_MEMORY_AUDIT_FAILED', audit.stdout, audit.stderr)
    save(dest / 'repair-plan.json', dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        latest_user_instruction='오류 사항 있으면 수리 재제출; job 모니터링만 하지 말라는 뜻',
        old_lock=record(OLD / 'execution.lock.json'), new_lock=record(dest / 'execution.lock.json'),
        exact_old_jobs=[*OLD_GPU, OLD_CPU], repair='Resolve force_removal before counterfactual-only recipe fields',
        scientific_contract_unchanged=True, CPU_tests=22, actual_GPU_validation='NOT_OBSERVED',
        reused_T0=t0, readonly_checkpoint_bindings=checked, reuse='Prior verified inputs only; old jobs not started at inspection',
        monitoring_active=False, automatic_resume=False, cancel_other_jobs=False,
        source_preserved=True, checkpoint_saved=False))
    print(json.dumps(dict(status='FROZEN_NOT_SUBMITTED', attempt=str(dest), lock=record(dest / 'execution.lock.json'))))


def replace():
    dest = ROOT / ATTEMPT
    assert (dest / 'repair-plan.json').exists()
    assert not (dest / 'cancel-request.json').exists(), 'ONE_SHOT_REPLACEMENT_ALREADY_STARTED'
    assert not (dest / 'submission.json').exists(), 'DUPLICATE_SUBMISSION'
    lock = read(dest / 'execution.lock.json')
    for member in lock['source_members']:
        assert sha(member['path']) == member['sha256']
    before = {j: inspect_old(j) for j in (*OLD_GPU, OLD_CPU)}
    # Admission-only owner inventory, not progress monitoring. Do not displace unrelated work.
    queue = command(['squeue', '-u', 'janghj', '-w', 'server4', '-h', '-o', '%i|%j|%T|%b|%R'])
    assert {line.split('|')[0] for line in queue.splitlines()} <= {*OLD_GPU, OLD_CPU}, ('UNRELATED_ADMITTED_JOB', queue)
    save(dest / 'pre-replacement-inspection.json', dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=before, queue=queue))
    # Stop the afterany collector first so cancelling its upstream does not start it.
    subprocess.run(['scancel', OLD_CPU], check=True)
    collector = command(['scontrol', 'show', 'job', '-o', OLD_CPU])
    assert 'JobState=CANCELLED' in collector, ('COLLECTOR_NOT_TERMINAL', collector)
    subprocess.run(['scancel', *OLD_GPU], check=True)
    terminal = {j: command(['scontrol', 'show', 'job', '-o', j]) for j in OLD_GPU}
    assert all('JobState=CANCELLED' in info for info in terminal.values()), ('OLD_GPU_NOT_TERMINAL', terminal)
    accounting = command(['sacct', '-n', '-X', '-j', ','.join([*OLD_GPU, OLD_CPU]), '--format=JobIDRaw,JobName%32,User,State,ExitCode,ElapsedRaw,AllocTRES%100,Start,End', '--parsable2'])
    save(dest / 'cancel-request.json', dict(exact_jobs=[*OLD_GPU, OLD_CPU], reason='User-authorized replacement of unstarted jobs containing reproduced T1 routing defect',
        collector_terminal=collector, gpu_terminal=terminal, accounting=accounting, original_data_untouched=True))
    cap = subprocess.run(['bash', 'scripts/check-slurm-resource-cap.sh', 'server4', '2', '120832M'],
                         env=dict(os.environ, AGENT_GPU_CAPS_FILE='/data/janghj/ODE-edit/servers/local/gpu-caps.tsv'),
                         capture_output=True, text=True)
    save(dest / 'resource-cap.json', dict(exit_code=cap.returncode, stdout=cap.stdout, stderr=cap.stderr))
    assert cap.returncode == 0, ('RESOURCE_CAP_NOT_PASSED', cap.stdout, cap.stderr)
    submit(dest)  # Creates complete T0→T4 DAG, held-inspects, releases, one final submission snapshot.
    print('REPLACEMENT_REGISTERED_MONITORING_STOPPED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'replace'))
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else replace()
