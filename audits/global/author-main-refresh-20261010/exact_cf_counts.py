"""저장 raw만 읽는 CF author 정확 유리수 집계. GPU/파일쓰기 없음."""
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

path, expected_sha = sys.argv[1:]
payload = Path(path).read_bytes()
assert hashlib.sha256(payload).hexdigest() == expected_sha
data = json.loads(payload)
assert len(data['cases']) == 2000
result = dict(raw_path=path, raw_sha256=expected_sha, requests=2000, metrics={})
for metric, group, desired, width in [
    ('Efficacy', 'rewrite', 'new', 1),
    ('Generalization', 'paraphrase', 'new', 2),
    ('Specificity', 'neighborhood', 'true', 10),
]:
    means = []
    numerator = 0
    for case in data['cases']:
        observations = case[group + '_observations']
        assert len(observations) == width
        bits = [o['target_' + desired]['mean_nll'] < o['target_' + ('true' if desired == 'new' else 'new')]['mean_nll'] for o in observations]
        numerator += sum(bits)
        means.append(Fraction(sum(bits), len(bits)))
    exact = 100 * sum(means) / len(means)
    assert abs(float(exact) - data['summary'][metric]) < 1e-10
    result['metrics'][metric] = dict(success_count=numerator, prompt_count=width*2000,
        prompts_per_request=width, exact_request_macro_pct=str(exact), decimal_pct=float(exact))
score = 3 / sum(1 / Fraction(v['exact_request_macro_pct']) for v in result['metrics'].values())
assert abs(float(score) - data['summary']['Score']) < 1e-10
result['Score_exact'] = str(score)
result['Score'] = float(score)
print(json.dumps(result, ensure_ascii=False))
