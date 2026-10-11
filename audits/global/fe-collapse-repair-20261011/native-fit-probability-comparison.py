"""Observe paired native target fitting, separately from post-edit CF metrics."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def collect(path):
    if not path.exists():
        return dict(path=str(path), prefix_bytes=0, prefix_sha256=None, batches={})
    data = path.read_bytes()
    data = data[:data.rfind(b'\n') + 1]
    batches = {}
    batch = last = None
    active = False
    for line in data.decode(errors='replace').splitlines():
        match = re.match(r'NATIVE_FE_BATCH_START (\d+) case_ids=(.+)', line)
        if match:
            batch = int(match[1])
            assert not active and batch not in batches
            assert batch == len(batches) + 1
            active = True
            last = None
            batches[batch] = dict(case_ids=ast.literal_eval(match[2]), probabilities=[], complete=False)
        elif line.startswith('NATIVE_FE_BATCH_COMPLETE '):
            assert active and int(line.split()[-1]) == batch
            batches[int(line.split()[-1])]['complete'] = True
            active = False
        elif 'avg prob of [' in line or line.startswith('Init norm '):
            assert active, 'TARGET_FIT_OUTSIDE_CURRENT_BATCH'
            match = re.search(r'avg prob of \[.*\] (\S+)', line)
            if match:
                last = float(match[1])
            if line.startswith('Init norm '):
                assert last is not None and math.isfinite(last) and 0 <= last <= 1
                batches[batch]['probabilities'].append(last)
                last = None
    result = dict(path=str(path), prefix_bytes=len(data), prefix_sha256=hashlib.sha256(data).hexdigest(), batches={})
    for batch, row in batches.items():
        if not row['complete']:
            continue
        probabilities = row['probabilities']
        assert len(probabilities) == len(row['case_ids']) == 100
        result['batches'][str(batch)] = dict(
            count=len(probabilities), mean=statistics.mean(probabilities),
            median=statistics.median(probabilities), minimum=min(probabilities),
            below_point5=sum(value < .5 for value in probabilities),
            below_point01=sum(value < .01 for value in probabilities),
            cases=[dict(case_id=case, last_reported_probability=value)
                   for case, value in zip(row['case_ids'], probabilities)])
    return result


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--summary-only', action='store_true',
                    help='Keep batch summaries and per-case content hashes instead of every fit value')
args = parser.parse_args()
specs = dict(gptj_original='native-r2/gptj-63386.out',
             gptj_repaired='native-causal-r1/gptj_repaired-63389.out',
             gptj_online_w0_repaired='native-causal-r2/gptj_online_w0_repaired-63425.out',
             qwen_current='native-r2/qwen25-63385.out',
             qwen_online_w0='native-causal-r1/qwen_online_w0-63391.out')
result = dict(scope='Last reported target-fit probability over fitting contexts; not CF post-edit efficacy or accuracy',
              audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              sample_order_changed=False, runs={name: collect(args.root / file) for name, file in specs.items()})
config = json.loads((args.root / 'native-r2/configs/gptj.json').read_text())
data = Path(config['original_stream']['path']).read_bytes()
assert hashlib.sha256(data).hexdigest() == config['original_stream']['sha256']
expected_ids = [row['case_id'] for row in json.loads(data)]
assert len(expected_ids) == 2000
for arm in result['runs'].values():
    for batch, row in arm['batches'].items():
        start = (int(batch) - 1) * 100
        assert [case['case_id'] for case in row['cases']] == expected_ids[start:start + 100]
result['original_stream_sha256'] = config['original_stream']['sha256']
result['all_observed_fits_inside_current_batch_markers'] = True
result['completed_batch_case_ids_match_original_stream'] = True
for first, second in (('gptj_original', 'gptj_repaired'),
                      ('gptj_repaired', 'gptj_online_w0_repaired'),
                      ('qwen_current', 'qwen_online_w0')):
    left, right = result['runs'][first]['batches'], result['runs'][second]['batches']
    for batch in left.keys() & right.keys():
        assert [r['case_id'] for r in left[batch]['cases']] == [r['case_id'] for r in right[batch]['cases']]
result['per_case_details_included'] = not args.summary_only
if args.summary_only:
    for arm in result['runs'].values():
        for row in arm['batches'].values():
            details = row.pop('cases')
            content = json.dumps(details, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            row['per_case_details_sha256'] = hashlib.sha256(content).hexdigest()
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({name: {batch: {k: v for k, v in row.items() if k != 'cases'}
                         for batch, row in arm['batches'].items()}
                  for name, arm in result['runs'].items()}))
