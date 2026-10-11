"""Independently reaggregate CF preferences from native-checkpoint raw NLLs."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--evaluation', type=Path, required=True)
parser.add_argument('--native-config', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
config, config_sha = read(args.native_config)
stream, stream_sha = read(Path(config['original_stream']['path']))
assert stream_sha == config['original_stream']['sha256']
complete, complete_sha = read(args.evaluation / 'COMPLETE.json')
assert complete['W20_available'] and complete['missing_milestones'] == []
assert complete['native_config'] == config['identity_sha256']
assert set(complete['results']) == {'0', '5', '10', '15', '20'}
result = dict(native_config=str(args.native_config), native_config_sha256=config_sha,
              identity_sha256=config['identity_sha256'], stream_sha256=stream_sha,
              complete_path=str(args.evaluation / 'COMPLETE.json'), complete_sha256=complete_sha,
              definition='Per-request mean preference of token-mean NLL; then mean across requests; harmonic Score',
              unit='percent, including Score', checkpoints={})
for key in ('0', '5', '10', '15', '20'):
    receipt = complete['results'][key]
    observed, observed_sha = read(Path(receipt['path']))
    assert observed_sha == receipt['sha256']
    batch = int(key)
    expected = stream if batch == 0 else stream[:batch * 100]
    assert [row['case_id'] for row in observed['cases']] == [row['case_id'] for row in expected]
    external = observed['identity']['external_identity']
    assert external['native_config'] == config['identity_sha256']
    assert external['checkpoint_sha256'] == receipt['checkpoint']['sha256']
    assert external['completed_edits'] == batch * 100
    assert observed['model_no_mutation'] and observed['RNG_restored']
    recomputed = {}
    prompts = {}
    for metric, field, desired in (
        ('Efficacy', 'rewrite_observations', 'new'),
        ('Generalization', 'paraphrase_observations', 'new'),
        ('Specificity', 'neighborhood_observations', 'true'),
    ):
        per_case = []
        prompt_count = correct_tokens = token_count = correct_prompts = 0
        for case in observed['cases']:
            entries = case[field]
            assert entries
            wins = []
            for row in entries:
                assert row['desired_target'] == desired
                a = row['target_' + desired]
                b = row['target_' + ('true' if desired == 'new' else 'new')]
                assert math.isfinite(a['mean_nll']) and math.isfinite(b['mean_nll'])
                wins.append(a['mean_nll'] < b['mean_nll'])
                correct_tokens += sum(a['token_correct'])
                token_count += len(a['token_correct'])
                correct_prompts += all(a['token_correct'])
                prompt_count += 1
            per_case.append(sum(wins) / len(wins))
        recomputed[metric] = 100 * math.fsum(per_case) / len(per_case)
        assert math.isclose(recomputed[metric], receipt['summary'][metric], abs_tol=1e-10)
        prompts[metric] = dict(prompts=prompt_count, correct_prompts=correct_prompts,
                               correct_tokens=correct_tokens, target_tokens=token_count)
    recomputed['Score'] = (3 / math.fsum(1 / recomputed[x] for x in ('Efficacy', 'Generalization', 'Specificity'))
                           if min(recomputed.values()) > 0 else 0.0)
    assert math.isclose(recomputed['Score'], receipt['summary']['Score'], abs_tol=1e-10)
    result['checkpoints'][key] = dict(requests=len(expected), summary=recomputed,
                                      raw_path=receipt['path'], raw_sha256=observed_sha,
                                      accuracy_counts=prompts, independent_reaggregation='PASS')
    print(key, json.dumps(recomputed), flush=True)
    del observed
with args.output.open('x') as destination:
    json.dump(result, destination, indent=2)
    destination.write('\n')
