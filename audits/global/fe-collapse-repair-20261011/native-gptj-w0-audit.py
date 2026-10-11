"""Validate the first native GPT-J snapshot against its pinned pretrained asset."""
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
root = Path(__file__).resolve().parent
config = json.loads((root / 'native-r2/configs/gptj.json').read_text())
assets = json.loads(Path(config['model_assets']['path']).read_text())
source = next(m for m in assets['model']['members'] if Path(m['path']).name == 'pytorch_model.bin')
checkpoint = Path(config['output']) / 'checkpoints/B00.pt'
receipt = json.loads(checkpoint.with_suffix('.json').read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


assert sha(checkpoint) == receipt['sha256']
assert sha(source['path']) == source['sha256']
original = torch.load(source['path'], map_location='cpu', mmap=True, weights_only=True)
payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
assert payload['identity_sha256'] == config['identity_sha256']
assert payload['completed_edits'] == 0 and payload['seen_case_ids'] == []
assert payload['z_policy'] == 'CURRENT_BATCH_ENTRY_NO_FUTURE_TARGETS'
cfg = payload['config']
expected = {cfg['llms']['rewrite_module_tmp'].format(layer) + '.weight'
            for layer in cfg['llms']['layers']}
assert set(payload['weights']) == expected
results = {}
for name, value in payload['weights'].items():
    assert value.dtype == torch.bfloat16 and torch.equal(value, original[name].to(value.dtype))
    results[name] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'pretrained_bit_identical': True}
assert not torch.count_nonzero(payload['history'])
assert sha(checkpoint) == receipt['sha256']
result = {'job': 63386, 'checkpoint': receipt, 'source': source, 'identity_match': True,
          'weights': results, 'history_zero': True, 'checkpoint_unchanged': True, 'CPU_only': True}
with (root / 'native-r2/gptj-w0-checkpoint-validation.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result), flush=True)
