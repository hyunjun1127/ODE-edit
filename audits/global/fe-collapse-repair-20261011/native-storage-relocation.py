"""Preserve completed diagnostic checkpoints on local SSD with original-path aliases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run(config_path, jobs, destination, report):
    # No active training/evaluation readers may be involved in this operation.
    states = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', ','.join(jobs),
        '--format=JobIDRaw,State,ExitCode'], text=True)
    observed = {parts[0]: parts[1:3] for line in states.splitlines()
                if (parts := line.strip().split('|'))[0] in jobs}
    assert set(observed) == set(jobs)
    assert all(state == ['COMPLETED', '0:0'] for state in observed.values()), observed
    config = json.loads(config_path.read_text())
    folder = Path(config['output']) / 'checkpoints'
    receipts = [json.loads((folder / f'B{b:02d}.json').read_text()) for b in (0, 5, 10, 15, 20)]
    assert all(r['identity_sha256'] == config['identity_sha256'] for r in receipts)
    assert not report.exists() and not destination.exists()
    destination.mkdir(mode=0o700, parents=True)
    assert os.stat(folder).st_dev != os.stat(destination).st_dev
    free = os.statvfs(destination).f_bavail * os.statvfs(destination).f_frsize
    assert free > 32 * 2**30 + sum(r['bytes'] for r in receipts)
    completed = []
    try:
        for receipt in receipts:
            source = Path(receipt['path'])
            assert source.parent == folder and source.is_file() and not source.is_symlink()
            assert source.stat().st_size == receipt['bytes'] and sha(source) == receipt['sha256']
            before = source.stat()
            final = destination / source.name
            partial = destination / (source.name + '.partial')
            assert not final.exists() and not partial.exists()
            subprocess.run(['cp', '--sparse=always', '--preserve=mode,timestamps', '--',
                            str(source), str(partial)], check=True)
            assert partial.stat().st_size == receipt['bytes'] and sha(partial) == receipt['sha256']
            with partial.open('rb') as stream:
                os.fsync(stream.fileno())
            os.link(partial, final)  # Exclusive, validated publication on the SSD.
            partial.unlink()
            fsync_dir(destination)
            assert source.stat().st_ino == before.st_ino and sha(source) == receipt['sha256']
            alias = source.with_suffix('.pt.ssd-link')
            os.symlink(final, alias)
            os.replace(alias, source)  # Original bytes already fsynced and verified elsewhere.
            fsync_dir(folder)
            assert source.resolve() == final and sha(source) == receipt['sha256']
            row = {'original_path': str(source), 'storage_path': str(final),
                   'sha256': receipt['sha256'], 'bytes': receipt['bytes'],
                   'original_allocated_bytes': before.st_blocks * 512,
                   'storage_allocated_bytes': final.stat().st_blocks * 512,
                   'same_bytes_at_original_path': True}
            completed.append(row)
            print(json.dumps(row), flush=True)
    finally:
        with report.open('x') as stream:
            json.dump({'scope': 'Storage relocation only; original paths resolve to identical bytes',
                       'config': str(config_path), 'native_identity': config['identity_sha256'],
                       'required_terminal_jobs': observed, 'checkpoints': completed,
                       'complete': len(completed) == len(receipts)}, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--jobs', nargs=2, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.jobs, args.destination, args.report)
