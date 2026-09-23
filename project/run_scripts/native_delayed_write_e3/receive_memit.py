"""Necessary ordinary repo-local input pull under envelope §2 / PROTOCOL.

Not an extension of the separate fifteen-small-files approval. No broadcast,
source writes, overwrite, delete, or input checkpoint regeneration.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from .common import ROOT, BASE, BATCHES, INSTRUCTION, read, save, sha

REMOTE = '/mnt/raid5/janghj/ODE-edit/local/checkpoint-archives/server4-migration-20260911-v1/new177-v1/payload/local/fixed10k-native-baselines/attempt-v1/output/main-cell-2'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    start = time.time()
    receipt = ROOT / 'receipts/memit-transfer.json'
    if receipt.exists():
        raise RuntimeError('TRANSFER_ALREADY_RECORDED')
    original = read(BASE / 'output/main-cell-2/terminal.json')
    members = {m['path']: m for m in original['manifest_members']}
    rows = []
    for batch in BATCHES:
        rel = f'B{batch:03d}/W-method-state.pt'
        old = members[rel]
        source = f'{REMOTE}/{rel}'
        dest = ROOT / 'inputs/checkpoints/BASE_MEMIT' / rel
        result = subprocess.check_output(['ssh', '-oBatchMode=yes', '-oConnectTimeout=10', 'rke-server2', 'stat -c ' + shlex.quote('%s %U %F') + ' -- ' + shlex.quote(source)], text=True).strip()
        size, owner, kind = result.split(' ', 2)
        assert owner == 'janghj' and kind == 'regular file' and int(size) == old['bytes'], result
        rows.append(dict(source=source, destination=str(dest), owner=owner, bytes=old['bytes'], sha256=old['sha256'], original_terminal_sha256=sha(BASE / 'output/main-cell-2/terminal.json')))
    total = sum(r['bytes'] for r in rows)
    free = shutil.disk_usage(ROOT).free
    # Compact rows + bounded temporary key chunks + one atomic CP partial + safety.
    reserve = 8 * 2**30
    assert free >= total + reserve, ('BLOCKED_STORAGE', free, total, reserve)
    plan = dict(instruction_id=INSTRUCTION, authority='envelope §2 necessary ordinary repo-local missing inputs; PROTOCOL ordinary artifacts', source_keep=True, large_broadcast=False, full_state_outputs=False, files=rows, bytes=total, free_before=free, post_pull_output_temp_safety_reserve=reserve)
    manifest = ROOT / 'receipts/memit-transfer-plan.json'
    if manifest.exists():
        prior = read(manifest)
        assert prior['files'] == rows
    else:
        save(manifest, plan)
    if not args.execute:
        print('EXACT_PLAN_SEALED', total, free, flush=True)
        return
    for row in rows:
        dest = Path(row['destination'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.stat().st_size == row['bytes'] and sha(dest) == row['sha256']
            row['status'] = 'REUSED_VERIFIED'
        else:
            partial = dest.with_name(dest.name + '.partial')
            with partial.open('xb') as out:
                child = subprocess.Popen(['ssh', '-oBatchMode=yes', 'rke-server2', 'cat -- ' + shlex.quote(row['source'])], stdout=out)
                rc = child.wait()
                out.flush(); os.fsync(out.fileno())
            if rc:
                raise RuntimeError(('SOURCE_READ_FAILED', row['source'], rc))
            assert partial.stat().st_size == row['bytes'] and sha(partial) == row['sha256'], ('TRANSFER_MISMATCH', str(partial))
            os.link(partial, dest)
            partial.unlink()
            row['status'] = 'RECEIVED_FULL_SHA_SIZE_VERIFIED'
        print('MEMIT_RECEIVED', dest.parent.name, row['status'], flush=True)
    result = save(receipt, dict(status='PASS', instruction_id=INSTRUCTION, files=rows, source_keep=True, count=len(rows), bytes=total, elapsed_seconds=time.time()-start, free_after=shutil.disk_usage(ROOT).free))
    print(result, flush=True)


if __name__ == '__main__':
    main()
