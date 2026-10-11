"""Compare committed paired native checkpoints without changing their tensors."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def load(config_path, batch):
    config = json.loads(config_path.read_text())
    path = Path(config['output']) / 'checkpoints' / f'B{batch:02d}.pt'
    receipt = json.loads(path.with_suffix('.json').read_text())
    assert receipt['identity_sha256'] == config['identity_sha256']
    assert sha(path) == receipt['sha256']
    payload = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
    assert payload['identity_sha256'] == config['identity_sha256']
    assert payload['completed_edits'] == batch * 100
    assert len(payload['seen_case_ids']) == batch * 100
    return config, path, receipt, payload


parser = argparse.ArgumentParser()
parser.add_argument('--control', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(8)
left, left_path, lr, lp = load(args.control, args.batch)
right, right_path, rr, rp = load(args.candidate, args.batch)
assert left['original_stream']['sha256'] == right['original_stream']['sha256']
assert left['model_assets']['sha256'] == right['model_assets']['sha256']
assert lp['seen_case_ids'] == rp['seen_case_ids']
assert lp['contexts'] == rp['contexts']
assert lp['config']['llms'] == rp['config']['llms']
assert lp['history'].shape == rp['history'].shape
assert lp['history'].dtype == rp['history'].dtype == torch.float32
layers = lp['config']['llms']['layers']
result = dict(control_checkpoint=lr, candidate_checkpoint=rr,
              same_case_order=True, same_contexts=True, same_hparams=True,
              control_z_policy=lp['z_policy'], candidate_z_policy=rp['z_policy'],
              scope='Saved native cumulative key Gram, without recomputing any edit',
              layers={}, CPU_only=True)
for index, layer in enumerate(layers):
    a, b = lp['history'][index], rp['history'][index]
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert torch.equal(a, a.T) and torch.equal(b, b.T)
    assert a.diagonal().double().sum() > 0 and b.diagonal().double().sum() > 0
    difference = a.double() - b.double()
    result['layers'][str(layer)] = dict(
        bit_identical=torch.equal(a, b),
        control_trace=float(a.diagonal().double().sum()),
        candidate_trace=float(b.diagonal().double().sum()),
        max_abs_difference=float(difference.abs().max()),
        relative_frobenius_difference=float(difference.norm() / a.double().norm()),
        both_finite=True, both_symmetric=True, both_nonzero=True)
    print(layer, json.dumps(result['layers'][str(layer)]), flush=True)
    del difference
assert sha(left_path) == lr['sha256'] and sha(right_path) == rr['sha256']
result['checkpoint_bytes_unchanged'] = True
with args.output.open('x') as destination:
    json.dump(result, destination, indent=2)
    destination.write('\n')
