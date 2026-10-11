"""Read-only comparison of native FE's final save and the W20 observer snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def run(config_path, output):
    torch.set_num_threads(2)
    config = json.loads(config_path.read_text())
    cp = Path(config['output']) / 'checkpoints/B20.pt'
    receipt = json.loads(cp.with_suffix('.json').read_text())
    assert sha(cp) == receipt['sha256']
    payload = torch.load(cp, mmap=True, map_location='cpu', weights_only=False)
    assert payload['identity_sha256'] == config['identity_sha256']
    assert payload['completed_edits'] == 2000
    stream = Path(config['original_stream']['path'])
    assert sha(stream) == config['original_stream']['sha256']
    rows = json.loads(stream.read_text())
    assert payload['seen_case_ids'] == [r['case_id'] for r in rows]
    native = payload['config']
    final = (Path(native['cache_dir']) / 'saved_weights' / native['algs']['name'] /
             (f"{native['data']}-{native['save_name']}-{native['llms']['name'].replace('/', '-')}.pt"))
    checksum = sha(final)
    saved = torch.load(final, mmap=True, map_location='cpu', weights_only=False)
    assert set(saved) == set(payload['weights'])
    comparisons = {}
    for name, value in saved.items():
        expected = payload['weights'][name]
        assert value.dtype == expected.dtype and value.shape == expected.shape
        assert torch.isfinite(value).all() and torch.equal(value, expected)
        comparisons[name] = {'dtype': str(value.dtype), 'shape': list(value.shape),
                             'bit_identical': True}
    assert sha(cp) == receipt['sha256'] and sha(final) == checksum
    result = {'config': str(config_path), 'identity': config['identity_sha256'],
              'checkpoint': receipt, 'native_final': {'path': str(final), 'sha256': checksum,
                'bytes': final.stat().st_size}, 'weights': comparisons,
              'files_unchanged': True, 'complete_original_stream': True,
              'scope': 'CPU artifact identity and final selected weights, not accuracy'}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
