"""Shared constants and checks of the server1 resubmission of the cancelled Qwen SPHERE jobs 62873 and 63303.

User 2026-10-11, "server1 cap 남으니 그냥 올려". Both jobs keep their original science; only host
bindings change. A field-level diff against the cancelled job's own config is computed and every
changed leaf must match an allowlisted host-binding pattern.
"""
from pathlib import Path

from official.runners.server1.common import member

TASK = 'qwen-sphere-s1-resubmit-20261011'
AUTHORITY = 'USER-QWEN-SPHERE-S1-RESUBMIT-20261011'
LOCAL = Path('/mnt/raid5/janghj/ODE-edit/local') / TASK
RECEIPT_FIELDS = ('key', 'host', 'source', 'source_sha256', 'source_bytes', 'local', 'local_sha256', 'status', 'seconds')
MISSING = object()


def require(ok, code):
    if not ok:
        raise ValueError(code)


def leaves(value, prefix=''):
    if isinstance(value, dict):
        for key in value:
            yield from leaves(value[key], f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f'{prefix}[{index}]')
    else:
        yield prefix, value


def field_diff(old, new):
    """Dotted leaf paths whose values differ (a missing leaf counts as different)."""
    a, b = dict(leaves(old)), dict(leaves(new))
    return sorted(k for k in a.keys() | b.keys() if a.get(k, MISSING) != b.get(k, MISSING))


def host_only(changed, patterns):
    science = [k for k in changed if not any(p.fullmatch(k) for p in patterns)]
    require(not science, 'SCIENTIFIC_FIELD_CHANGED:' + ','.join(science))
    return changed


def differences(old, new, changed):
    a, b = dict(leaves(old)), dict(leaves(new))
    return {k: dict(original=a.get(k), server1=b.get(k)) for k in changed}


def transfer_rows(path):
    """Read-only pull receipt rows (transfer.sh): SHA-256 and size on the source host and on devbox."""
    rows = [line.split('\t') for line in Path(path).read_text().splitlines() if line]
    require(rows and all(len(r) == len(RECEIPT_FIELDS) for r in rows), 'TRANSFER_RECEIPT_SCHEMA')
    rows = [dict(zip(RECEIPT_FIELDS, r)) for r in rows]
    require(len({r['local'] for r in rows}) == len(rows), 'TRANSFER_RECEIPT_DUPLICATE_LOCAL')
    return {r['local']: r for r in rows}


def received(rows, local, source=None):
    """A copied file is usable only with an OK receipt whose two SHA-256 and sizes equal the file now."""
    row = rows.get(str(local))
    found = member(local)
    require(row is not None and row['status'] == 'OK' and (source is None or row['source'] == str(source))
            and row['source_sha256'] == row['local_sha256'] == found['sha256']
            and int(row['source_bytes']) == found['bytes'], 'TRANSFER_RECEIPT_MISMATCH:' + str(local))
    return row, found
