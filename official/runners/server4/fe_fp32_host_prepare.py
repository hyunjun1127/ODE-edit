"""Read-only SH4 asset/storage preflight; no scientific adapter or submission.

The scientific FP32 core is owned by SH1. This module never loads tensors.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024 ** 3
STREAM_SHA = '66edc483a8d4bcadedd479e4c36759a686ad61a38741d8870a9052b795710e37'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def budget(hidden, intermediate, layers, margin, concurrent):
    # Actual selected W and native history shapes, both FP32. Serialization
    # overhead is separate from the actual tensor payload, not an author rule.
    w = layers * hidden * intermediate * 4
    h = layers * intermediate * intermediate * 4
    payload_bound = w + h + 32 * 1024 ** 2
    return dict(W_bytes=w, H_bytes=h, tensor_bytes=w+h,
                serialization_overhead_bytes=32*1024**2,
                payload_bound_bytes=payload_bound,
                latest_and_atomic_tmp_bytes=2*payload_bound,
                raw_log_runtime_margin_bytes=margin,
                concurrent_growth_reservation_bytes=concurrent,
                operating_reserve_bytes=8*GIB,
                required_free_bytes=2*payload_bound+margin+concurrent+8*GIB)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--stream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--storage-root', type=Path, required=True)
    p.add_argument('--raw-margin-bytes', type=int, required=True)
    p.add_argument('--concurrent-growth-bytes', type=int, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if min(a.raw_margin_bytes, a.concurrent_growth_bytes) < 0:
        raise ValueError('negative reservation')
    assets = json.loads(a.assets.read_text())
    members = assets['model']['members'] + [v['member'] for v in assets['C0'].values()]
    verified = []
    for m in members:
        path = Path(m['path'])
        before = path.stat()
        digest = sha(path)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError('asset changed during verification: '+str(path))
        if digest != m['sha256'] or after.st_size != m['bytes']:
            raise ValueError('asset SHA/bytes mismatch: '+str(path))
        verified.append(dict(path=str(path), sha256=digest, bytes=after.st_size,
                             device=after.st_dev, inode=after.st_ino, mtime_ns=after.st_mtime_ns))
    if a.stream.stat().st_size != 4145654 or sha(a.stream) != STREAM_SHA:
        raise ValueError('main CF stream identity mismatch')
    records = json.loads(a.stream.read_text())
    if len(records) != 2000:
        raise ValueError('expected exact 2000 records')
    config = assets['model']['config']
    identity = assets['model']['identity']
    if identity['layers'] != [4,5,6,7,8]:
        raise ValueError('unexpected edit layers')
    b = budget(config['hidden_size'], config['intermediate_size'], 5,
               a.raw_margin_bytes, a.concurrent_growth_bytes)
    st = os.statvfs(a.storage_root)
    free = st.f_bavail * st.f_frsize
    result = dict(schema=1, task='fe-qwen-cf-fp32-20261012',
                  observed_at=datetime.now(timezone.utc).isoformat(),
                  scientific_core_owner='SH1', scientific_core_status='SOURCE_INPUT_PENDING',
                  asset_manifest_sha256=sha(a.assets), assets=verified,
                  stream=dict(path=str(a.stream), sha256=STREAM_SHA, bytes=4145654, count=2000),
                  model_identity=identity, tokenizer_sha256=assets['tokenizer_sha256'],
                  checkpoint_budget=b, free_bytes=free, free_inodes=st.f_favail,
                  storage_formula_fits=free >= b['required_free_bytes'],
                  remaining_after_budget_bytes=free-b['required_free_bytes'],
                  quota_status='NOT_VERIFIED_QUOTA_TOOL_UNAVAILABLE',
                  admission='NOT_FINAL_RECHECK_AT_SUBMIT_AND_STARTUP',
                  operating_reserve_authority='USER server4 operational 8GiB; not FE author science',
                  new_job_ids=[], GPU_forward_count=0, asset_copies=0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ('checkpoint_budget','free_bytes','storage_formula_fits','remaining_after_budget_bytes','admission')}))


if __name__ == '__main__':
    main()
