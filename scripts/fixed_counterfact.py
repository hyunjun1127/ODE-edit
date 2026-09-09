#!/usr/bin/env python3
"""Materialize and consume the user-frozen CounterFact stream; never resample."""
import argparse
import hashlib
import json
from pathlib import Path

SAMPLE_SHA = 'a8d22c230611b6c14740d1d00658a53856bbde26d060189d5ddf30f2ffdbac92'
SOURCE_SHA = 'd017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f'
ORDER_ROOT = '5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729'
DATASET_SHA = '3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1'


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_sample(path):
    raw = Path(path).read_bytes()
    require(sha(raw) == SAMPLE_SHA, 'authoritative sample bytes mismatch')
    sample = json.loads(raw)
    require(sha(encoded(sample['records'])) == ORDER_ROOT == sample['ordered_root'], 'order root mismatch')
    require(len(sample['records']) == 10000, 'expected 10000 sample records')
    return raw, sample


def validate_records(records, sample):
    require(len(records) == 10000, 'expected 10000 extracted records')
    require(len({r['case_id'] for r in records}) == 10000, 'duplicate case ID')
    for i, (record, sealed) in enumerate(zip(records, sample['records'])):
        require(sealed['ordinal'] == i and sealed['batch_index'] == i // 100 + 1
                and sealed['batch_ordinal'] == i % 100, 'sealed batch/ordinal mismatch')
        require(record['case_id'] == sealed['case_id'], 'case order mismatch')
        require(sha(encoded(record)) == sealed['raw_record_sha256'], 'record content mismatch')
    counts = {'rewrite': len(records),
              'rephrase': sum(len(r['paraphrase_prompts']) for r in records),
              'neighborhood': sum(len(r['neighborhood_prompts']) for r in records)}
    require(counts == {'rewrite': 10000, 'rephrase': 20000, 'neighborhood': 100000}, 'prompt inventory mismatch')
    return counts


def create_once(path, data):
    if path.exists():
        require(path.is_file() and not path.is_symlink() and path.read_bytes() == data,
                'existing artifact differs: ' + str(path))
        return
    with path.open('xb') as stream:
        stream.write(data)


def build(source, sample_path, output):
    source_bytes = Path(source).read_bytes()
    require(sha(source_bytes) == SOURCE_SHA, 'full source dataset mismatch')
    sample_bytes, sample = load_sample(sample_path)
    full = json.loads(source_bytes)
    by_id = {r['case_id']: r for r in full}
    require(len(by_id) == len(full), 'duplicate IDs in source')
    records = [by_id[r['case_id']] for r in sample['records']]
    counts = validate_records(records, sample)
    data = encoded(records) + b'\n'
    require(sha(data) == DATASET_SHA, 'canonical extracted bytes mismatch')
    receipt = {'dataset_id': 'counterfact-fixed-10k-v1', 'records': 10000,
               'source_dataset_sha256': SOURCE_SHA, 'source_sample_sha256': SAMPLE_SHA,
               'ordered_root': ORDER_ROOT, 'dataset_sha256': sha(data),
               'dataset_bytes': len(data), 'inventory': counts,
               'prefix_policy': 'records[:n], 1 <= n <= 10000; no shuffle/reselection',
               'case_order_sha256': sha(encoded([r['case_id'] for r in records]))}
    output = Path(output)
    require(not output.is_symlink(), 'output directory cannot be a symlink')
    output.mkdir(parents=True, exist_ok=True)
    create_once(output / 'source-sample.lock.json', sample_bytes)
    create_once(output / 'counterfact.json', data)
    create_once(output / 'receipt.json', encoded(receipt) + b'\n')
    return verify(output)


def verify(root):
    root = Path(root)
    _, sample = load_sample(root / 'source-sample.lock.json')
    data = (root / 'counterfact.json').read_bytes()
    require(sha(data) == DATASET_SHA, 'frozen extracted bytes mismatch')
    records = json.loads(data)
    counts = validate_records(records, sample)
    receipt = json.loads((root / 'receipt.json').read_bytes())
    require(receipt['dataset_sha256'] == sha(data) and receipt['dataset_bytes'] == len(data), 'dataset receipt mismatch')
    require(receipt['ordered_root'] == ORDER_ROOT and receipt['source_sample_sha256'] == SAMPLE_SHA
            and receipt['source_dataset_sha256'] == SOURCE_SHA, 'receipt source mismatch')
    require(receipt['inventory'] == counts and receipt['records'] == 10000, 'receipt count mismatch')
    require(receipt['case_order_sha256'] == sha(encoded([r['case_id'] for r in records])), 'case order receipt mismatch')
    return receipt


def load_prefix(root, count):
    """Use from an experiment runner before model loading; returns original records."""
    require(isinstance(count, int) and not isinstance(count, bool) and 1 <= count <= 10000,
            'count must be 1..10000 requests, not thousands-of-requests')
    verify(root)
    return json.loads((Path(root) / 'counterfact.json').read_bytes())[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    builder = sub.add_parser('build')
    builder.add_argument('--source', required=True)
    builder.add_argument('--sample-lock', required=True)
    builder.add_argument('--output', required=True)
    verifier = sub.add_parser('verify')
    verifier.add_argument('--root', required=True)
    prefix = sub.add_parser('prefix')
    prefix.add_argument('--root', required=True)
    prefix.add_argument('--count', type=int, required=True)
    prefix.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        receipt = build(args.source, args.sample_lock, args.output)
    elif args.command == 'verify':
        receipt = verify(args.root)
    else:
        records = load_prefix(args.root, args.count)
        path = Path(args.output)
        require(path.parent.is_dir(), 'create prefix output directory explicitly first')
        data = encoded(records) + b'\n'
        create_once(path, data)
        receipt = {'count': len(records), 'sha256': sha(data), 'ordered_root': ORDER_ROOT}
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
