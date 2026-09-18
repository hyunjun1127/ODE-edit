"""Read-only remote checkpoint inventory; writes reports only on GH.

Never imports torch or unpickles objects. Payload hashes are not recomputed.
Remote inspection is metadata plus bounded ZIP/pickle *opcode* schema scanning.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import pickletools
import re
import socket
import stat
import subprocess
import sys
import time
import zipfile

VERSION = 'checkpoint-location-index-v1'
ROOTS = ['/mnt/raid5/janghj', '/data/janghj']
EXCLUDE = {'.git', '.svn', '__pycache__', 'node_modules', 'site-packages',
           '.venv', 'venv', 'env', 'envs', '.tox', '.pytest_cache', '.mypy_cache',
           '.ruff_cache', 'wandb', 'tensorboard', '.ipynb_checkpoints'}
EXT = ('.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.npz', '.npy')
SCHEMA_KEYS = {'weights', 'selected_weights', 'weight', 'W', 'W0', 'We', 'WN',
               'W4', 'W8', 'M', 'M4', 'M8', 'history', 'state_dict',
               'model_state_dict', 'optimizer_state_dict', 'optimizer',
               'rng', 'rng_state', 'rng_states', 'context_templates',
               'contexts', 'ledger', 'active_registry', 'next_batch',
               'batch_index', 'batch', 'step', 'selected', 'delta',
               'delta_W', 'target', 'targets', 'keys', 'P', 'C0',
               'prepared', 'snapshot', 'tensors', 'model', 'commit_id',
               'params', 'parameters', 'weight_before', 'weight_after',
               'weights_before', 'weights_after'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def within(path):
    return any(path == r or path.startswith(r + '/') for r in ROOTS)


def candidate(name):
    lower = name.lower()
    return lower.endswith(EXT) or bool(re.search(r'\.(?:pt|pth|ckpt|safetensors)\.', lower))


def classify(path, keys=None):
    p, n = path.lower(), os.path.basename(path).lower()
    keys = set(keys or [])
    if '/.cache/huggingface/' in p or n.startswith(('pytorch_model', 'model-000')):
        return 'PRETRAINED_OR_FULL_MODEL', 'path_or_shard_name'
    if any(x in n for x in ('.partial', '.tmp', '.incomplete', '.part.')) or '/.rsync-partial/' in p:
        return 'PARTIAL_OR_TEMP', 'partial_filename'
    if any(x in p for x in ('/teacher', '/stats/', '/mom2/', '/cov_cache/')) or 'mom2' in n:
        return 'AUXILIARY_TEACHER_STATS', 'path_or_name'
    if n.startswith(('native-target', 'target-', 'target_', 'key-', 'key_', 'gradient', 'grad-', 'grad_', 'teacher')) or n in {'p.pt', 'p4.pt', 'p8.pt', 'c0.pt', 'keys.pt', 'targets.pt', 'z.pt'}:
        return 'AUXILIARY_TARGET_KEY_GRADIENT', 'name'
    if re.search(r'checkpoint|snapshot|endpoint|selected[-_]?(?:state|weight)|final[-_](?:state|weight)|state[-_](?:final|dict)|(?:^|[-_])wn(?:[-_.]|$)|own[-_]n4', n) or re.search(r'/(checkpoints|snapshots)/', p):
        return 'CHECKPOINT_CANDIDATE', 'checkpoint_endpoint_name'
    if n.startswith('prepared') or n in {'comparison.pt', 'native.pt', 'entry.pt', 'state.pt'}:
        return 'PREPARED_OR_STATE_CAPSULE', 'capsule_name'
    if keys & {'weights', 'selected_weights', 'state_dict', 'model_state_dict', 'W4', 'W8', 'WN', 'weights_after', 'weight_after'}:
        return 'CHECKPOINT_CANDIDATE', 'bounded_schema_weight_key'
    if keys & {'W', 'weight', 'delta_W', 'delta', 'weights_before', 'weight_before'}:
        return 'WEIGHT_OR_DELTA_CANDIDATE', 'bounded_schema_weight_or_delta_key'
    if keys & {'targets', 'target', 'keys', 'P', 'C0'}:
        return 'AUXILIARY_TARGET_KEY_GRADIENT', 'bounded_schema_auxiliary_key'
    if any(x in n for x in ('weight', 'delta', 'history', 'model', 'adapter')):
        return 'WEIGHT_OR_DELTA_CANDIDATE', 'name'
    return 'UNCLASSIFIED_TENSOR', 'requires_owner_schema_confirmation'


def bounded_schema(path, size):
    result = {'schema_status': 'NOT_INSPECTED', 'schema_keys': '', 'zip_tensor_members': '', 'header_bytes_read': 0}
    if not path.lower().endswith(('.pt', '.pth', '.ckpt', '.bin')) or size < 8:
        return result
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            result['header_bytes_read'] = 4
        if magic != b'PK\x03\x04':
            result['schema_status'] = 'NONZIP_NOT_DESERIALIZED'
            return result
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            result['zip_tensor_members'] = sum('/data/' in i.filename for i in infos)
            data = next((i for i in infos if i.filename.endswith('/data.pkl') or i.filename == 'data.pkl'), None)
            if data is None or data.file_size > 1024 * 1024:
                result['schema_status'] = 'NO_SMALL_DATA_PICKLE'
                return result
            buf = z.read(data)
            result['header_bytes_read'] += len(buf)
            keys = set()
            for op, val, _ in pickletools.genops(buf):
                if isinstance(val, str) and val in SCHEMA_KEYS:
                    keys.add(val)
            result['schema_keys'] = '|'.join(sorted(keys))
            result['schema_status'] = 'ZIP_PICKLE_OPCODES_ONLY_NOT_DESERIALIZED'
    except Exception as e:
        result['schema_status'] = type(e).__name__ + ':' + str(e)[:150]
    return result


def scan(server):
    started = now()
    rows, errors, aliases, scope, metadata_files = [], [], [], [], []
    visited = set()
    counts = {'directories': 0, 'files_seen': 0, 'excluded_directories': 0}
    roots = []
    for root in ROOTS:
        if not os.path.isdir(root):
            scope.append({'path': root, 'status': 'ABSENT'})
            continue
        stv = os.statvfs(root)
        scope.append({'path': root, 'status': 'PRESENT', 'free_bytes': stv.f_bavail * stv.f_frsize})
        for item in os.scandir(root):
            if item.name.startswith('.'):
                if item.name == '.codex':
                    target = item.path + '/worktrees'
                    if os.path.isdir(target):
                        roots.append(target)
                elif item.name == '.cache':
                    target = item.path + '/huggingface'
                    if os.path.isdir(target):
                        roots.append(target)
                continue
            if item.is_dir() or item.is_symlink():
                roots.append(item.path)
            elif item.is_file() and candidate(item.name):
                roots.append(item.path)

    stack = list(reversed(sorted(roots)))
    def error(path, exc):
        errors.append({'path': path, 'error': type(exc).__name__, 'detail': str(exc)[:160]})

    def add_file(path, lst):
        counts['files_seen'] += 1
        name = os.path.basename(path)
        low = name.lower()
        if low.endswith(('.json', '.csv')) and re.search(r'(checkpoint|migration|preserv|transfer|retention|source-seal|manifest|receipt)', low):
            if lst.st_size <= 12 * 1024 * 1024:
                metadata_files.append({'path': path, 'bytes': lst.st_size, 'mtime_ns': lst.st_mtime_ns})
        if not candidate(name):
            return
        resolved = os.path.realpath(path)
        row = {'server': server, 'hostname': socket.gethostname(), 'path': path,
               'realpath': resolved, 'is_symlink': stat.S_ISLNK(lst.st_mode),
               'link_target': os.readlink(path) if stat.S_ISLNK(lst.st_mode) else '',
               'observed_at': now()}
        try:
            if not within(resolved):
                row.update({'availability': 'LINK_OUTSIDE_SCOPE_NOT_FOLLOWED', 'bytes': '', 'category': classify(path)[0], 'classification_basis': 'name_only'})
                rows.append(row)
                return
            s = os.stat(path)
            row.update({'availability': 'PRESENT', 'bytes': s.st_size,
                        'allocated_bytes': getattr(s, 'st_blocks', 0) * 512,
                        'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns,
                        'device': s.st_dev, 'inode': s.st_ino, 'nlink': s.st_nlink})
            cat, basis = classify(path)
            if cat not in {'PRETRAINED_OR_FULL_MODEL', 'AUXILIARY_TEACHER_STATS', 'AUXILIARY_TARGET_KEY_GRADIENT', 'PARTIAL_OR_TEMP'}:
                schema = bounded_schema(path, s.st_size)
                row.update(schema)
                cat, basis = classify(path, schema['schema_keys'].split('|'))
            row.update({'category': cat, 'classification_basis': basis})
            after = os.stat(path)
            row['stable_during_inspection'] = (s.st_size, s.st_mtime_ns, s.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
        except FileNotFoundError:
            row.update({'availability': 'DISAPPEARED_OR_BROKEN_LINK', 'category': classify(path)[0], 'classification_basis': 'name_only', 'bytes': ''})
        except Exception as e:
            row.update({'availability': 'STAT_ERROR', 'category': classify(path)[0], 'classification_basis': 'name_only', 'bytes': ''})
            error(path, e)
        rows.append(row)

    while stack:
        path = stack.pop()
        try:
            ls = os.lstat(path)
            if stat.S_ISLNK(ls.st_mode):
                target = os.path.realpath(path)
                if not within(target):
                    aliases.append({'path': path, 'target': target, 'status': 'OUTSIDE_USER_ROOT_NOT_FOLLOWED'})
                    continue
                if os.path.isdir(path):
                    aliases.append({'path': path, 'target': target, 'status': 'IN_SCOPE_DIRECTORY_ALIAS'})
                    stack.append(target)
                else:
                    add_file(path, ls)
                continue
            if not stat.S_ISDIR(ls.st_mode):
                if stat.S_ISREG(ls.st_mode):
                    add_file(path, ls)
                continue
            ident = (ls.st_dev, ls.st_ino)
            if ident in visited:
                continue
            visited.add(ident)
            counts['directories'] += 1
            with os.scandir(path) as entries:
                for item in entries:
                    if item.name in EXCLUDE or (item.name.startswith('.') and item.is_dir(follow_symlinks=False) and item.name != '.rsync-partial'):
                        counts['excluded_directories'] += 1
                        continue
                    stack.append(item.path)
        except Exception as e:
            error(path, e)

    return {'version': VERSION, 'server': server, 'hostname': socket.gethostname(),
            'started_at': started, 'finished_at': now(), 'scope': scope,
            'scanned_roots': roots, 'excluded_directory_names': sorted(EXCLUDE),
            'hidden_scope': 'Only .codex/worktrees and .cache/huggingface at user-root; nested hidden directories excluded except .rsync-partial.',
            'counts': counts, 'errors': errors, 'directory_aliases': aliases,
            'files': rows, 'metadata_candidates': metadata_files,
            'payload_hashes_recomputed': 0, 'tensor_deserialization': 0,
            'remote_writes': 0, 'GPU_or_scheduler_actions': 0}


def collect(out):
    out.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).read_text()
    hosts = [('server1', None), ('server2', 'rke-server2'), ('server3', 'rke-server3'), ('server4', 'rke-server4')]
    def run(item):
        server, host = item
        command = ['nice', '-n', '10', 'python3', '-', '--scan', server]
        if host:
            command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, 'nice -n 10 python3 - --scan ' + server]
        proc = subprocess.run(command, input=src, text=True, capture_output=True, timeout=900)
        if proc.returncode:
            result = {'server': server, 'error': proc.stderr[-3000:], 'exit_code': proc.returncode}
        else:
            result = json.loads(proc.stdout)
        target = out / (server + '-scan.json')
        if target.exists():
            raise RuntimeError('create-once refusal: ' + str(target))
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'server': server, 'files': len(result.get('files', [])), 'counts': result.get('counts'), 'errors': len(result.get('errors', [])), 'exit_code': proc.returncode}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, hosts))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan')
    ap.add_argument('--collect', type=Path)
    args = ap.parse_args()
    if args.scan:
        print(json.dumps(scan(args.scan), ensure_ascii=False))
    elif args.collect:
        collect(args.collect)
    else:
        ap.error('choose --scan or --collect')
