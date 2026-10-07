"""Bounded saved-JSON evidence reader. No model import, load, or forward."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


def member(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read_price(path):
    value = json.loads(Path(path).read_text())
    d = value.get('payload', value)
    if d['B'] != 100 or len(d['denominator']) != len(d['layers']):
        raise ValueError('PHASE0_B_LAYER_IDENTITY')
    for row in d['denominator']:
        if len(row) != 100 or not all(math.isfinite(x) and x > 1e-8 for x in row):
            raise ValueError('PHASE0_DENOMINATOR_IDENTITY_FINITE')
    return d


def inspect(attempt, cell, model):
    p = Path(attempt)
    lock = json.loads((p/'execution.lock.json').read_text())
    cfg = json.loads((p/'config.json').read_text())
    if member(p/'config.json')['sha256'] != lock['config_sha256']:
        raise ValueError('PHASE0_CONFIG_LOCK_MISMATCH')
    f = p/cell/'batch-01/entry-price.json'
    d = read_price(f)
    lowest = {'llama3': 4, 'gptj': 3, 'gpt2xl': 13}[model]
    if d['layers'][0] != lowest:
        raise ValueError('PHASE0_LOWEST_LAYER')
    native = 20000. if model == 'gpt2xl' else 15000.
    key = {'llama3': 'LLAMA', 'gptj': 'MEMIT', 'gpt2xl': 'MEMIT'}[model]
    mc = cfg['models'][key]
    profile = mc['profiles']['CAP075']
    if profile['lambda_C'] != native:
        raise ValueError('PHASE0_NATIVE_LAMBDA')
    v = [1-x for x in d['denominator'][0]]
    median = statistics.median(v)
    saved_tensors = []
    for root, dirs, files in os.walk(p, followlinks=False):
        dirs[:] = [x for x in dirs if x not in ('source', 'wandb', '.git')
                   and not Path(root, x).is_symlink()]
        for name in files:
            if Path(name).suffix in ('.pt', '.pth', '.npy', '.npz', '.safetensors'):
                saved_tensors.append(str(Path(root, name)))
    result = dict(model=model, cell=cell, source=lock['source_commit'],
        config=member(p/'config.json'), lock=member(p/'execution.lock.json'),
        B1_price=member(f), layer=lowest, requests=100, native_lambda=native,
        median_realization=median, min_realization=min(v), max_realization=max(v),
        definition='M=P.T@K; realization=1-denominator, no clipping',
        denominator_strict_min=1e-8, median_policy='middle pair arithmetic mean for B100',
        key_sha256=d['mean_key_hash'][str(lowest)],
        no_durable_matrices=d.get('no_durable_matrices'), saved_tensor_paths=saved_tensors,
        tensor_scan_scope='exact attempt excluding source/wandb/.git; symlinks not followed',
        M3='NATIVE_UNCHANGED' if median >= .5 else 'NEEDS_SAVED_B1_K',
        calibration='NOT_REQUIRED' if median >= .5 else 'NOT_RECORDED',
        stats_paths=mc['stats'], ordered_ids_sha256=cfg['ordered_ids_sha256'], seed=cfg['seed'])
    if model == 'gptj':
        rows=[]
        for batch,slot in ((3,47),(5,36),(7,26),(10,25)):
            f=p/cell/f'batch-{batch:02d}/entry-price.json';r=read_price(f)
            for layer,anchors in zip(r['layers'],r['anchors']):
                med=statistics.median(anchors)
                rows.append(dict(batch=batch,slot_zero_based=slot,layer=layer,
                    anchor=anchors[slot],batch_median=med,ratio=anchors[slot]/med,
                    slot_one_based_alternative_ratio=anchors[slot-1]/med,
                    artifact=member(f),prefix_hidden_norm_mean='NOT_RECORDED'))
        result['anchor_rows']=rows
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True)
    p.add_argument('--cell',required=True);p.add_argument('--model',required=True)
    args=p.parse_args()
    print(json.dumps(inspect(args.attempt,args.cell,args.model),ensure_ascii=False,indent=2))
