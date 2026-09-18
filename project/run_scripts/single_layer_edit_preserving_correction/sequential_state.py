"""S-only received ledger and durable selected-state publication; no model load."""
import copy
import hashlib
import os
import json
from pathlib import Path
import torch
from .common import member, tensor_sha, digest

S_ARMS = ('EN-S', 'EN-F', 'EN-COV', 'EN-F4')

def fact(record):
    r = record['requested_rewrite']
    return r['subject'], r['relation_id']

def receive(ledger, records):
    out = copy.deepcopy(ledger)
    seen = {r['case_id'] for r in out}
    for record in records:
        if record['case_id'] in seen:
            raise ValueError('DUPLICATE_RECEIVED_EVENT')
        seen.add(record['case_id'])
        out.append(dict(case_id=record['case_id'], requested_rewrite=copy.deepcopy(record['requested_rewrite']),
                        ordinal=len(out)))
    return out

def past64(ledger, current):
    latest = {}
    for r in ledger:
        latest[fact(r)] = r
    overwritten = {fact(r) for r in current}
    eligible = [r for k, r in latest.items() if k not in overwritten]
    priority = lambda r: (hashlib.sha256(f'ENFC-v1|past|{r["case_id"]}'.encode('utf-8')).hexdigest(), str(r['case_id']))
    return copy.deepcopy(sorted(eligible, key=priority)[:64])

def registry(ledger):
    latest = {}
    for r in ledger:
        latest[fact(r)] = r['case_id']
    return [dict(case_id=r['case_id'], ordinal=r['ordinal'],
                 status='ACTIVE' if latest[fact(r)] == r['case_id'] else 'SUPERSEDED') for r in ledger]

def atomic_tensor(path, payload):
    """Create-once, fsync, safe reload, and no-overwrite atomic link.

    Only this invocation's owned temporary inode is unlinked after publication;
    pre-existing scientific artifacts are never deleted or overwritten.
    """
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    tmp = p.with_name(p.name + f'.partial-{os.getpid()}')
    with tmp.open('xb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    loaded = torch.load(tmp, map_location='cpu', weights_only=True, mmap=True)
    for key in ('weight', 'M4'):
        if key not in payload:
            continue
        t = loaded[key]
        if t.dtype != torch.float32 or t.shape != payload[key].shape or not torch.isfinite(t).all():
            raise ValueError('SNAPSHOT_SCHEMA_' + key)
        if tensor_sha(t) != tensor_sha(payload[key]):
            raise ValueError('SNAPSHOT_BYTES_' + key)
    del loaded
    os.link(tmp, p)  # fails rather than replaces an existing final name
    tmp.unlink()
    fd = os.open(p.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return member(p)

def state_identity(weight, memory, rng, ledger, context):
    return dict(W=tensor_sha(weight), M=tensor_sha(memory), rng=digest(rng),
                ledger=digest(ledger), context=digest(context))

def atomic_json(path, payload):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.partial-{os.getpid()}')
    with tmp.open('x') as f:
        json.dump(payload,f,ensure_ascii=False,sort_keys=True,allow_nan=False,indent=2)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(tmp,p);tmp.unlink()
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return member(p)

def require_next(parent, entry):
    if parent != entry:
        raise ValueError('SEQUENTIAL_PARENT_ENTRY_LINK_MISMATCH')

def require_finalizer(receipts, before, selected, after):
    if len(receipts) != 1:
        raise ValueError('HISTORY_ONCE_CARDINALITY')
    r = receipts[0]
    if (r['layer'] != 4 or r['history_append'] != 1 or r['compute_ks'] != 1 or
        r.get('compute_z', 0) or r.get('solve', 0) or
        r['before_sha256'] != before or r['after_sha256'] != after or r['weight_sha256'] != selected):
        raise ValueError('HISTORY_ONCE_BINDING')
