"""Task-local byte-exact input seals and create-once receipts."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess

ROOT = Path('/data/janghj/ODE-edit/local/bpcw512/20260918-v2')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def create_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'CREATE_ONCE_MISMATCH: {path}')
        return
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

def create_json(path, obj):
    create_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())

def seal_inputs(repo):
    repo = Path(repo)
    source = repo / 'audits/global/2026-09-18-sh4-bpcw512-b1-dispatch/source-manifest.json'
    members = json.loads(source.read_text())['members']
    for rel in ('messages/head/2026-09-18-sh4-bpcw512-cold-b1-v2.md',
                'tasks/pending/bpcw512-cold-b1-sh4-20260918-v2.json',
                'PROTOCOL.md', 'control/gpu-concurrency-policy.tsv', 'servers/active/server4.md'):
        members.append({'path': rel, 'sha256': sha(repo / rel)})
    result = []
    for row in members:
        p = repo / row['path']
        actual = sha(p)
        if actual != row['sha256']:
            raise ValueError(f'AUTHORITATIVE_SHA_MISMATCH:{p}')
        target = ROOT / 'authoritative' / row['path']
        create_bytes(target, p.read_bytes())
        result.append({**row, 'bytes': p.stat().st_size, 'copy': str(target)})
    prior = Path('/data/janghj/ODE-edit/local/single-layer-edit-preserving-correction/20260918-v1/receipts/full-read-m0.json')
    receipt = dict(time_utc=now(), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                   members=result, full_read='Canonical new instruction/design/contract/cells/math/envelope read in full; unchanged PROTOCOL exact prior FULL_READ reused',
                   prior_full_read={'path': str(prior), 'sha256': sha(prior)},
                   max_batches=1, sequential_authorized=False, scientific_gpu_status='NOT_SUBMITTED',
                   R512_additional128_status='NOT_BUILT', answer_capsules_status='NOT_BUILT',
                   model_numerical_validation='NOT_ESTABLISHED', en_cleanup='NOT_YET_PERFORMED')
    create_json(ROOT / 'receipts/full-read-m0.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal-inputs', type=Path, required=True)
    seal_inputs(parser.parse_args().seal_inputs)
