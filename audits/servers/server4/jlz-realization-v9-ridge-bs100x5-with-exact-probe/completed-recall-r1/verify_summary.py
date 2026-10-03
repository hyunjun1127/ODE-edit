"""사용자 완료 recall의 소형 CPU 검산. 모델/torch/Slurm 호출과 원자료 수정 없음."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def record(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def save(path, data):
    with path.open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def reduce(rows):
    assert len({r['identity'] for r in rows}) == len(rows)
    result = {}
    for kind in ('R', 'P', 'N'):
        group = [r for r in rows if r['kind'] == kind]
        desired = 'true' if kind == 'N' else 'new'
        for r in group:
            for label in ('true', 'new'):
                assert math.isfinite(r[label + '_nll'])
                count, correct = r[label + '_token_count'], r[label + '_token_correct']
                assert type(count) is int and type(correct) is int and 0 <= correct <= count and count > 0
                assert r[label + '_strict'] == (correct == count)
        success = sum((r['true_nll'] < r['new_nll']) if kind == 'N'
                      else (r['new_nll'] < r['true_nll']) for r in group)
        tokens = sum(r[desired + '_token_count'] for r in group)
        correct = sum(r[desired + '_token_correct'] for r in group)
        result[kind] = dict(denominator=len(group), numerator=success, rate=success/len(group),
            true_nll_mean=sum(r['true_nll'] for r in group)/len(group),
            new_nll_mean=sum(r['new_nll'] for r in group)/len(group),
            desired_token_count=tokens, desired_token_correct=correct, token_micro=correct/tokens,
            prompt_macro=sum(r[desired + '_token_correct']/r[desired + '_token_count'] for r in group)/len(group),
            strict_numerator=sum(r[desired + '_strict'] for r in group), strict_denominator=len(group),
            new_strict_numerator=sum(r['new_strict'] for r in group))
    return result


def observer(path, expected_state=None):
    summary, rows = read(path/'summary.json'), []
    for chunk in sorted(path.glob('chunk-*.json')):
        data = read(chunk)
        assert data['state'] == summary['state']
        rows.extend(data['rows'])
    assert summary['no_mutation'] and len(rows) == summary['row_count']
    if expected_state is not None:
        assert summary['state'] == expected_state
    reduced = reduce(rows)
    assert reduced == summary['summary']
    return reduced, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.attempt.resolve(), args.out.resolve()
    assert not out.exists(), '출력은 create-once 경로여야 한다'
    lock = read(root/'execution.lock.json')
    terminal = read(root/'report/terminal.json')
    assert terminal['status'] == 'COMPLETED_500_BOTH' and terminal['main_commits'] == 10
    assert terminal['source'] == lock['source_commit']
    table = {(r['arm'], int(r['batch']), r['kind']): r
             for r in csv.DictReader((root/'report/metrics.csv').open())}
    assert len(table) == 30
    evidence = [record(root/'execution.lock.json'), record(root/'config.json')]
    checks, row_count = {}, 0
    for arm in ('A', 'B'):
        lane, previous, ids = root/('main-'+arm), None, []
        term = read(lane/'terminal.json')
        assert term['status'] == 'COMPLETED' and term['main_commits'] == 5
        assert term['no_B6'] and not term['checkpoint_saved'] and not (lane/'batch-06').exists()
        for batch in range(1, 6):
            stage = lane/f'batch-{batch:02d}'
            commit = read(stage/'commit.json')
            assert commit['source'] == lock['source_commit'] and commit['actual_B'] == 100
            assert commit['candidate_count'] == 25 and commit['Adam_updates'] == 24
            assert commit['history_appends'] == 5 and commit['no_checkpoint'] and not commit['replay']
            assert commit['exact_commit_sha'] == commit['after']['W']
            assert not commit['terminal_gradient_measured']
            if previous is not None:
                assert previous == commit['before']
            previous = commit['after']
            ids.extend(commit['current_ids'])
            candidates = sorted((stage/'fit').glob('candidate-*.json'))
            assert len(candidates) == 25
            for k, p in enumerate(candidates, 1):
                c = read(p)
                assert c['candidate'] == k and c['Adam_updates_after'] == min(k, 24)
                assert c['gradient_measured'] == (k < 25)
            metrics, count = observer(lane/f'observe-W{batch:02d}', previous)
            row_count += count
            n = 500 if batch == 5 else 100
            assert {k:v['denominator'] for k,v in metrics.items()} == dict(R=n, P=2*n, N=10*n)
            for kind, reduced in metrics.items():
                assert all(float(table[arm,batch,kind][k]) == v for k,v in reduced.items())
            evidence += [record(stage/'commit.json'), record(lane/f'observe-W{batch:02d}'/'summary.json')]
        assert len(ids) == len(set(ids)) == 500
        checks[arm] = dict(commits=5, candidates=125, Adam_updates=120, history_appends=25,
                           unique_cases=500, no_B6=True, final_summary=metrics)
        evidence.append(record(lane/'terminal.json'))
    q = root/'main-A/batch-01/Q2'
    qr = read(q/'receipt.json')
    assert qr['restore_verified'] and qr['ridge_payload_unchanged'] and qr['exact_qualified']
    assert qr['additional_fits'] == qr['exact_backward'] == qr['exact_updates'] == qr['history_appends'] == 0
    exact, n = observer(q/'observer')
    row_count += n
    cq = read(q/'causal-qualification.json')
    fq = read(q/'frozen-ridge-K-operators.json')
    assert cq['qualified'] and not cq['frozen_upper_verdicts_reused']
    fields = ('status','rank','rcond','FP64_relative','FP32_RMS','FP32_limit')
    q2 = dict(receipt=qr, causal={l:{k:v[k] for k in fields} for l,v in cq['layers'].items()},
              frozen={l:{k:v[k] for k in fields} for l,v in fq.items()}, exact_observer=exact)
    for v in cq['layers'].values():
        assert v['rank'] == 100 and v['FP64_relative'] <= 1e-8 and v['FP32_RMS'] <= v['FP32_limit']
    out.mkdir(parents=True)
    for name in ('metrics.csv','summary.json','paired-cohorts.json','cost.json','terminal.json'):
        p = root/'report'/name
        (out/('collector-'+name)).write_bytes(p.read_bytes())
        evidence.append(record(p))
    for name in ('receipt.json','causal-qualification.json','frozen-ridge-K-operators.json','observer/summary.json'):
        evidence.append(record(q/name))
    index = root/'report/artifact-index.json'
    items = read(index)
    assert record(index)['sha256'] == terminal['artifacts']['sha256']
    evidence.append(record(index))
    save(out/'q2-summary.json', q2)
    save(out/'verification.json', dict(status='PASS_BOUNDED_CPU_READBACK', checked_rows=row_count,
        checked=checks, implementation='표 집계 독립 재작성, 동일 owner 검산; 독립 reviewer 아님',
        reused_collector_inventory=dict(entries=len(items), indexed_bytes=sum(x['bytes'] for x in items),
            extensions=dict(Counter(Path(x['path']).suffix for x in items)),
            tensor_rehash_on_recall=False, source='sealed collector artifact-index.json'),
        evidence=evidence, verifier=record(Path(__file__).resolve()),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        new_model_forward=0, new_jobs=0, monitoring_active=False, automatic_resume=False))
    print(json.dumps(dict(status='PASS', checked_rows=row_count, main_commits=10, output=str(out))))


if __name__ == '__main__':
    main()
