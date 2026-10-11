"""Preserve bytes and SHA while releasing allocated zero runs in a W0 snapshot."""
import argparse
import datetime
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


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text())
path = Path(config['output']) / 'checkpoints/B00.pt'
assert path.is_relative_to(Path(__file__).resolve().parent) and not path.is_symlink()
assert not args.report.exists()
receipt = json.loads(path.with_suffix('.json').read_text())
assert receipt['completed_batches'] == 0 and receipt['identity_sha256'] == config['identity_sha256']
assert sha(path) == receipt['sha256'] and path.stat().st_size == receipt['bytes']
before = path.stat()
result = subprocess.run(['fallocate', '--dig-holes', '--verbose', str(path)],
                        capture_output=True, text=True, check=True)
with path.open('rb') as stream:
    os.fsync(stream.fileno())
after = path.stat()
assert before.st_size == after.st_size and sha(path) == receipt['sha256']
output = {'checkpoint': str(path), 'sha256': receipt['sha256'], 'bytes': after.st_size,
          'allocated_before': before.st_blocks * 512, 'allocated_after': after.st_blocks * 512,
          'bytes_unchanged': True, 'method': 'fallocate --dig-holes; zero byte runs only',
          'stdout': result.stdout, 'at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
with args.report.open('x') as stream:
    json.dump(output, stream, indent=2)
    stream.write('\n')
print(json.dumps(output), flush=True)
