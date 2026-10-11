"""Compare on-arrival W0 targets with archived author-precompute weight hashes.

The historical run is only an author-method reference, not the current baseline.
No old targets or weights are supplied to the running native experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--native-config', type=Path, required=True)
parser.add_argument('--reference-run', type=Path, required=True)
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
config = json.loads(args.native_config.read_text())
assert config['z_reference'] == 'initial_model_on_arrival'
cp = Path(config['output']) / 'checkpoints' / f'B{args.batch:02d}.pt'
receipt = json.loads(cp.with_suffix('.json').read_text())
assert sha(cp) == receipt['sha256']
payload = torch.load(cp, mmap=True, map_location='cpu', weights_only=False)
assert payload['identity_sha256'] == config['identity_sha256'] == receipt['identity_sha256']
assert payload['completed_batches'] == args.batch
assert payload['z_policy'] == 'CURRENT_BATCH_ONLY_INITIAL_MODEL'
old_commit = args.reference_run / 'commits' / f'batch-{args.batch:02d}.json'
archived = json.loads(old_commit.read_text())
assert archived['identity']['author'] == config['author_commit']
assert archived['cursor']['batch'] == args.batch
raw_path = Path(archived['cursor']['all_seen']['path'])
assert sha(raw_path) == archived['cursor']['all_seen']['sha256']
raw = json.loads(raw_path.read_text())
assert [case['case_id'] for case in raw['cases']] == payload['seen_case_ids']
assert set(archived['cursor']['selected_after']) == set(payload['weights'])
result = dict(scope='Historical all-future W0 precompute is an author-method reference, not the current baseline',
              native_checkpoint=receipt, reference_commit_path=str(old_commit),
              reference_commit_sha256=sha(old_commit), reference_raw_sha256=archived['cursor']['all_seen']['sha256'],
              same_ordered_cases=True, weights={}, CPU_only=True)
for name, weight in payload['weights'].items():
    digest = hashlib.sha256(weight.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    expected = archived['cursor']['selected_after'][name]
    result['weights'][name] = dict(native_sha256=digest, reference_sha256=expected, identical_bytes=digest == expected)
    print(name, digest == expected, flush=True)
if args.batch == 20:
    final = json.loads((args.reference_run / 'COMPLETE.json').read_text())
    old_path = Path(final['checkpoint']['path'])
    assert sha(old_path) == final['checkpoint']['sha256']
    original = torch.load(old_path, mmap=True, map_location='cpu', weights_only=False)
    assert original['batch'] == args.batch
    result['actual_final_weights_bit_identical'] = all(torch.equal(weight, original['weights'][name])
                                                       for name, weight in payload['weights'].items())
    result['actual_final_history_bit_identical'] = torch.equal(payload['history'], original['cache_c'])
    result['same_contexts'] = payload['contexts'] == original['contexts']
    assert sha(old_path) == final['checkpoint']['sha256']
    result['reference_checkpoint'] = final['checkpoint']
assert sha(cp) == receipt['sha256']
result['checkpoint_bytes_unchanged'] = True
with args.output.open('x') as destination:
    json.dump(result, destination, indent=2)
    destination.write('\n')
print(json.dumps({k: v for k, v in result.items() if k.startswith('actual_')}), flush=True)
