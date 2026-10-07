"""One USER-recalled registration pass over an unchanged scientific archive.

Only registration paths/attempt identity are rebound. No model/evaluator/editor
is executed here, no parent files are rewritten and no scheduler retry loop.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import tarfile

from . import submit as original
from .common import LOCAL, ROOT, NONCE, TASK, SESSION, member, require, sha, verify, write
from .generation import require_binding

RESUME = 'USER-GH-SH1-SH2-SH4-BASELINE-GENERATION-REGISTER-RESUME-20261007-R1'
ENVELOPE = 'messages/head/2026-10-07-baseline-generation-register-resume.json'
ENVELOPE_SHA = '43099367d76e1f8a2b57b539c4a18aa17351e2194885db70e3516af19464fc16'
SCIENTIFIC_SOURCE = 'e3019677e5b17edf98401e381972c272a711ecc7'
SCIENTIFIC_ARCHIVE_SHA = 'f0339e37f0dfa5672347fd1cd286fe69787d25aa7add1fd7c0920ec7ddaa8034'
CONTROL_FILES = ('project/run_scripts/llama3_native_baselines/register_resume.py',
                 'project/run_scripts/llama3_native_baselines/test_register_resume.py')


def authority():
    path = ROOT / ENVELOPE
    require(sha(path) == ENVELOPE_SHA, 'EXACT_RESUME_AUTHORITY_SHA')
    value = json.loads(path.read_text())
    owner = value['targets']['server4']
    require(value['instruction_id'] == RESUME and value['parent_instruction'] == NONCE
            and owner['session'] == SESSION and owner['task_id'] == TASK
            and owner['source'] == SCIENTIFIC_SOURCE and owner['project_GPU_cap'] == 3,
            'EXACT_RESUME_OWNER_SCOPE')
    return member(path)


def rebound_config(parent, attempt):
    """These are the only three changed fields; science and parent nonce stay."""
    require(parent['instruction_id'] == NONCE and parent['task_id'] == TASK,
            'PARENT_INSTRUCTION_TASK')
    require(attempt.parent == LOCAL and attempt.name == 'attempt-r2', 'NEW_REGISTRATION_SCOPE')
    result = copy.deepcopy(parent)
    result['attempt'] = str(attempt)
    result['generation']['W0_ready_path'] = str(attempt / 'shared-generation-W0-ready.json')
    result['run_instance']['attempt'] = '20261007-r2'
    return result


def assert_rebinding(parent, child, attempt):
    require(child == rebound_config(parent, attempt), 'REGISTRATION_ONLY_THREE_FIELD_REBIND')


def freeze_parent(parent, configpath, attempt):
    """Exact byte-copy/extraction of the verified e301 archive, not HEAD."""
    require(parent == LOCAL / 'attempt-r1' and attempt == LOCAL / 'attempt-r2'
            and not attempt.exists(), 'PRESERVE_PARENT_CREATE_ONCE_CHILD')
    resume = authority()
    old, previous = original.verify_frozen(parent)
    require(old['source_commit'] == SCIENTIFIC_SOURCE
            and old['archive']['sha256'] == SCIENTIFIC_ARCHIVE_SHA,
            'EXACT_UNCHANGED_SCIENTIFIC_SOURCE')
    require(not list(parent.glob('submitted-*.json')), 'PARENT_NO_SUCCESSFUL_REGISTRATIONS')
    c = json.loads(configpath.read_text())
    assert_rebinding(previous, c, attempt)
    proof = json.loads(verify(c['cpu_preflight']).read_text())
    require(proof['status'] == 'CPU_SOURCE_READY' and proof['tests'] == 61
            and proof['actual_GPU'] == 'NOT_OBSERVED', 'REUSED_CPU61_NOT_GPU_PASS')
    for row in proof['source_members']:
        verify(row)
    require_binding(c)
    require(not original.command(['git', 'status', '--porcelain', '--', *CONTROL_FILES], ROOT),
            'COMMIT_REGISTRATION_CONTROL_BEFORE_USE')
    control_commit = original.command(['git', 'rev-parse', 'HEAD'], ROOT)
    attempt.mkdir()
    source = attempt / 'source'
    source.mkdir()
    archive = attempt / 'source.tar'
    shutil.copyfile(verify(old['archive']), archive)
    require(sha(archive) == SCIENTIFIC_ARCHIVE_SHA, 'IDENTICAL_CHILD_SOURCE_ARCHIVE')
    with tarfile.open(archive) as tf:
        rows = tf.getmembers()
        require(len(rows) == len({r.name for r in rows}) and all(
            (r.isfile() or r.isdir()) and not Path(r.name).is_absolute()
            and '..' not in Path(r.name).parts for r in rows), 'SAFE_REUSED_ARCHIVE')
        tf.extractall(source, filter='data')
    old_source = parent / 'source'
    expected = {str(Path(row['path']).relative_to(old_source)): row['sha256']
                for row in old['source_members']}
    actual = {str(p.relative_to(source)): sha(p) for p in source.rglob('*') if p.is_file()}
    require(actual == expected, 'ENTIRE_RUNTIME_SOURCE_BYTES_UNCHANGED')
    write(attempt / 'config.json', c)
    for role in original.ROLES:
        script = attempt / (role + '.sh')
        script.write_text(original.launcher(source, SCIENTIFIC_SOURCE, role, attempt,
                                            c['resources'], c['generation']))
        script.chmod(0o755)
    child_lock = copy.deepcopy(old)
    child_lock.update(archive=member(archive), config_sha256=sha(attempt / 'config.json'),
        source_members=[member(p) for p in sorted(source.rglob('*')) if p.is_file()],
        launchers=[member(attempt / (role + '.sh')) for role in original.ROLES],
        run_instance=c['run_instance'])
    write(attempt / 'execution.lock.json', child_lock)
    write(attempt / 'registration.lock.json', dict(schema='native-baseline-registration-recall-v1',
        nonce=RESUME, parent_instruction=NONCE, task_id=TASK, owner_session=SESSION,
        authority=resume, parent_lock=member(parent / 'execution.lock.json'),
        parent_error=member(parent / 'submission-failed.json'), parent_bytes_KEEP=True,
        scientific_source=SCIENTIFIC_SOURCE, runtime_members_identical=True,
        copied_archive_sha256=SCIENTIFIC_ARCHIVE_SHA, registration_control_commit=control_commit,
        registration_control_sources=[member(ROOT / p) for p in CONTROL_FILES],
        config_changes=['attempt', 'generation.W0_ready_path', 'run_instance.attempt'],
        generation_source=c['generation']['source_sha'],
        reference_identity=c['generation']['assets_sha256'],
        original_cpu_receipt=c['cpu_preflight'], actual_GPU='NOT_OBSERVED',
        new_scientific_fits=0, model_loads=0, automatic_retry=False))
    return original.verify_frozen(attempt)


def register(parent, attempt):
    authority()
    require(parent == LOCAL / 'attempt-r1' and attempt == LOCAL / 'attempt-r2',
            'EXACT_PARENT_CHILD_PATHS')
    require(not attempt.exists(), 'CREATE_ONCE_REGISTRATION_PASS')
    # Accounting/nonce reconciliation is externally recorded before this one
    # explicit call. Original submit also rejects live/receipt duplicates.
    previous = json.loads((parent / 'config.json').read_text())
    configpath = LOCAL / 'registration-r2-config.json'
    write(configpath, rebound_config(previous, attempt))
    freezer = original.freeze
    original.freeze = lambda config, target: freeze_parent(parent, config, target)
    try:
        result = original.submit(configpath, attempt)
    finally:
        original.freeze = freezer
    write(attempt / 'resume-submission.json', dict(nonce=RESUME, parent_nonce=NONCE,
        registration=member(attempt / 'registration.lock.json'),
        submission=member(attempt / 'submission.json'), status=result['status'],
        jobs=result['jobs'], scientific_source=SCIENTIFIC_SOURCE,
        actual_B1='NOT_OBSERVED', monitoring_active=False, automatic_retry=False))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    result = register(args.parent.resolve(), args.attempt.resolve())
    print(json.dumps({k: result[k] for k in ('status', 'jobs', 'source', 'bounded_initial_snapshot')}))


if __name__ == '__main__':
    main()
