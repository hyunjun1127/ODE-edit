"""Final read-only rehash of bound input/source bytes, raw-free copied locks."""
import argparse
from pathlib import Path
from .identity import ROOT,save,member,sha,digest
from .analysis import read

def main(args):
    output=Path(args.output);original=ROOT/'control/input.lock.json';lock=read(original)
    original_members=lock['assets']+lock['source']+read(ROOT/'control/m0.json')['members']
    checked=[];seen={}
    for expected in original_members:
        path=expected['path']
        if path in seen:
            if seen[path]!=expected['sha256']:raise ValueError('CONFLICTING_INPUT_BINDINGS')
            continue
        seen[path]=expected['sha256'];actual=member(path)
        if actual['sha256']!=expected['sha256']:raise ValueError(f'INPUT_BYTE_CHANGED: {path}')
        checked.append(dict(**actual,before_sha256=expected['sha256'],after_unchanged=True))
    for name in ('input.lock.json','m0.json','published-reference-check.json'):
        source=ROOT/'control'/name
        with (output/name).open('xb') as f:f.write(source.read_bytes())
    save(output/'input-preservation-receipt.json',dict(status='BOUND_INPUT_BYTES_UNCHANGED',members=checked,
        members_root=digest(checked),input_lock_sha256=sha(original),
        model_shard_scope='pinned immutable HF revision and initial shard size/presence; no claim of redundant full multi-GB shard rehash',
        excludes_unrelated_artifacts=True,scientific_promotion=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args())
