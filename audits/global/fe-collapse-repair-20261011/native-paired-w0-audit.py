"""Verify paired native FE controls start from identical weights, hparams, and context."""
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


def load(config_path):
    config = json.loads(config_path.read_text())
    cp = Path(config['output']) / 'checkpoints/B00.pt'
    receipt = json.loads(cp.with_suffix('.json').read_text())
    assert receipt['identity_sha256'] == config['identity_sha256']
    assert sha(cp) == receipt['sha256']
    payload = torch.load(cp, mmap=True, map_location='cpu', weights_only=False)
    assert payload['completed_edits'] == 0 and payload['seen_case_ids'] == []
    assert payload['identity_sha256'] == config['identity_sha256']
    assert not torch.count_nonzero(payload['history'])
    return config, receipt, payload


parser = argparse.ArgumentParser()
parser.add_argument('--control', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(2)
left, lr, lp = load(args.control)
right, rr, rp = load(args.candidate)
assert left['original_stream']['sha256'] == right['original_stream']['sha256']
assert left['model_assets']['sha256'] == right['model_assets']['sha256']
same = ('llms', 'algs', 'bs', 'num_edits', 'model_dtype', 'seed', 'data')
for field in same:
    assert lp['config'][field] == rp['config'][field], field
assert lp['contexts'] == rp['contexts']
assert set(lp['weights']) == set(rp['weights'])
for name in lp['weights']:
    assert lp['weights'][name].dtype == rp['weights'][name].dtype
    assert torch.equal(lp['weights'][name], rp['weights'][name]), name
assert torch.equal(lp['rng']['torch'], rp['rng']['torch'])
assert len(lp['rng']['cuda']) == len(rp['rng']['cuda'])
assert all(torch.equal(a, b) for a, b in zip(lp['rng']['cuda'], rp['rng']['cuda']))
assert sha(Path(lr['path'])) == lr['sha256'] and sha(Path(rr['path'])) == rr['sha256']
result = {'control': str(args.control), 'candidate': str(args.candidate),
          'control_checkpoint': lr, 'candidate_checkpoint': rr,
          'native_hparams_and_settings_identical': list(same),
          'contexts_identical': True, 'weights_bit_identical': True,
          'both_history_zero': True, 'torch_and_cuda_rng_identical': True,
          'stream_and_model_assets_identical': True,
          'control_z_policy': lp['z_policy'], 'candidate_z_policy': rp['z_policy'],
          'checkpoint_bytes_unchanged': True, 'CPU_only': True}
with args.output.open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result), flush=True)
