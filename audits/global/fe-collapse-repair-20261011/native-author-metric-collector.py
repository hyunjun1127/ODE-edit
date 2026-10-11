"""Collect completed native author's CF evaluation without changing its rounding."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--job', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert args.job.isdigit()
config = json.loads(args.config.read_text())
options = dict(item.lstrip('+').split('=', 1) for item in config['overrides'])
assert options['data'] == 'counterfact_stream_2000'
model = yaml.safe_load((Path(config['root']) / 'configs/llms' / (options['llms'] + '.yaml')).read_text())
accounting = subprocess.check_output(['sacct', '-n', '-X', '-P', '-j', args.job,
                                      '-o', 'JobID,State,ExitCode,Start,End,Elapsed'], text=True)
row = next(line.split('|') for line in accounting.splitlines() if line.startswith(args.job + '|'))
assert row[1:3] == ['COMPLETED', '0:0'], row
root = Path(options['results_dir']) / options['data'] / model['name'].replace('/', '-')
result = dict(job=args.job, native_config=config['identity_sha256'], sacct=accounting,
              metric_definition='Mean per-request preference of token-mean NLL; components rounded to 3 decimals by native author code',
              score_definition='Harmonic mean computed here from the already rounded native components',
              native_results={})
for phase, filename in [('before', options['save_name']),
                        ('after', options['algs'] + '-' + options['save_name'])]:
    path = root / filename
    data = path.read_bytes()
    marker = re.search(r'The Evaluation Results ' + phase + r' Editing:\s*(\{.*\})\s*$',
                       data.decode(), re.S)
    assert marker is not None, path
    raw = json.loads(marker[1])
    percentages = {metric: 100 * raw[field] for metric, field in (
        ('Efficacy', 'rewrite_prompts_probs'), ('Generalization', 'paraphrase_prompts_probs'),
        ('Specificity', 'neighborhood_prompts_probs'))}
    assert all(0 <= value <= 100 for value in percentages.values())
    percentages['Score_from_rounded_native_components'] = (
        3 / sum(1 / value for value in percentages.values()) if min(percentages.values()) > 0 else 0)
    result['native_results'][phase] = dict(raw=raw, percentages=percentages,
                                          path=str(path), sha256=hashlib.sha256(data).hexdigest())
with args.output.open('x') as destination:
    json.dump(result, destination, indent=2)
    destination.write('\n')
print(json.dumps({phase: row['percentages'] for phase, row in result['native_results'].items()}))
