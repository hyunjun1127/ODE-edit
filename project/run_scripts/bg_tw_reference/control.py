"""Create-once control records, immutable input binding and fail-closed scope."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from datetime import datetime, timezone

TASK = 'ODEEDIT-S06-BG1-C4-OURS-FIRST-SH4-V1'
SESSION = '01a04939-b5c7-7a03-ba2d-ef3343d62cfd'
REPO = Path('/data/janghj/ODE-edit')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path).absolute()
    s = path.stat()
    return dict(path=str(path), bytes=s.st_size, sha256=sha(path))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    return identity(path)


def verify_dispatch(path):
    d = json.loads(Path(path).read_text())
    assert d['new_scientific_policies'] == ['BG-1'], 'ONE_BG_POLICY_ONLY'
    assert d['new_scientific_chains'] == 1 and d['batch_size'] == 100
    assert d['batches_per_chain'] == 10 and d['unique_requests'] == 1000
    assert d['initial_model'] == 'pre-edit W0', 'WARM_ENTRY_FORBIDDEN'
    assert d['execution_server'] == 'server4'
    assert d['new_editing_logical_batches'] == 10
    assert not d['baseline_editing_reruns_allowed'], 'BASELINE_RERUN_FORBIDDEN'
    assert not d['calibration']['native_target_or_writer_reexecution_allowed']
    assert d['calibration']['missing_data_blocks_scientific_execution']
    assert not d['calibration']['silent_threshold_fallback_allowed']
    assert d['after_gate']['agent_state'] == 'WAITING_USER_RESUME'
    assert not d['after_gate']['automatic_resume']
    assert set(d['reference_policies_reuse_only']) == {
        'AlphaEdit', 'MEMIT', 'AlphaEdit-BLUE', 'MEMIT-BLUE', 'AlphaEdit-L4_only', 'REFIT4'}
    return d


def boundary(worktree):
    worktree = Path(worktree).resolve()
    assert socket.gethostname().lower() == 'server4'
    remote = subprocess.check_output(['git', '-C', str(worktree), 'remote', 'get-url', 'origin'], text=True).strip()
    assert remote == 'https://github.com/hyunjun1127/ODE-edit.git'
    common = subprocess.check_output(['git', '-C', str(worktree), 'rev-parse', '--path-format=absolute', '--git-common-dir'], text=True).strip()
    assert Path(common).resolve() == REPO / '.git'
    registry = (worktree / 'servers/connection-inventory.md').read_text()
    assert SESSION in registry and str(REPO) in registry
    return dict(host='server4', session=SESSION, repository='hyunjun1127/ODE-edit',
                registered_cwd=str(REPO), analysis_or_execution_worktree=str(worktree),
                registry_sha256=sha(worktree / 'servers/connection-inventory.md'),
                shared_session_file='STALE_PRESERVED_NOT_USED',
                binding='EXACT_USER_ENVELOPE_AND_REGISTERED_SESSION_TASK_LOCAL')


def initialize(worktree, attempt):
    worktree, attempt = Path(worktree).resolve(), Path(attempt).absolute()
    bind = boundary(worktree)
    attempt.mkdir(parents=True, exist_ok=False, mode=0o700)
    inv = json.loads((worktree / 'audits/global/2026-09-15-bg1-c4-ours-first-dispatch-source-inventory.json').read_text())
    files = [r['path'] for r in inv['files']] + ['PROTOCOL.md',
        'messages/head/2026-09-15-sh4-bg1-c4-ours-first.md',
        'audits/global/2026-09-15-bg1-c4-ours-first-dispatch-source-inventory.json',
        'project/run_scripts/low_cost_write_donor_pilot/fitting.py']
    expected = {r['path']: r for r in inv['files']}
    refs = []
    for rel in files:
        src, dst = worktree / rel, attempt / 'authoritative' / rel
        ref = identity(src)
        if rel in expected:
            assert all(ref[k] == expected[rel][k] for k in ('bytes', 'sha256')), rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open('rb') as f, dst.open('xb') as out:
            shutil.copyfileobj(f, out)
        os.chmod(dst, 0o600)
        assert sha(dst) == ref['sha256']
        refs.append(dict(**ref, copy=str(dst)))
    native = REPO / 'local/blue-alphaedit-sequential-comparison/attempt-v1/blue-source/AlphaEdit'
    refs.extend(identity(native / name) for name in ('AlphaEdit_main.py', 'compute_z.py'))
    verify_dispatch(worktree / 'plans/global/2026-09-15-bg1-c4-ours-first-dispatch-contract.json')
    return save(attempt / 'full-read-receipt.json', dict(instruction_id=TASK,
        timestamp=datetime.now(timezone.utc).isoformat(), boundary=bind,
        source_main='e56af00b16ecaed38ecb4ea890ce71f0c57aa4ff',
        source_tree='f4f770b3a06684bf5fb7017e8640dd9bdfb172f8',
        full_read_by_parent=True, files=refs, policy_scope='BG-1_ONLY_W0_FIRST1000',
        no_broadcast='NO_BROADCAST_NOT_REQUIRED', gpu_hour_cap=None, gpu_cap=2,
        initial_gate='NOT_RUN', scientific_submissions=[]))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--worktree', required=True)
    ap.add_argument('--attempt', required=True)
    args = ap.parse_args()
    print(json.dumps(initialize(args.worktree, args.attempt), ensure_ascii=False))
