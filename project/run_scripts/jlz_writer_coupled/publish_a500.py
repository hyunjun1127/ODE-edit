"""CPU-only publication reducer for the user-selected A/W05 endpoint.

No model imports, scheduler calls, evaluation, runtime edits, or raw-row export.
Reproduce: python -m project.run_scripts.jlz_writer_coupled.publish_a500 --attempt PATH
The JSON on stdout contains compact metrics and exact local evidence identities.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def reduce(rows):
    assert len({r['identity'] for r in rows}) == len(rows)
    answer = {}
    for r in rows:
        assert r['kind'] in ('R', 'P', 'N')
        for side in ('true', 'new'):
            assert math.isfinite(r[side+'_nll'])
            n, c = r[side+'_token_count'], r[side+'_token_correct']
            assert type(n) is int and type(c) is int and n > 0 and 0 <= c <= n
            assert type(r[side+'_strict']) is bool
            assert r[side+'_strict'] == (c == n)
        assert abs(r['margin_true_minus_new']-(r['true_nll']-r['new_nll'])) < 1e-12
    for kind in ('R', 'P', 'N'):
        group = [r for r in rows if r['kind'] == kind]
        wanted = 'true' if kind == 'N' else 'new'
        wins = sum(r[wanted+'_nll'] < r[('new' if wanted == 'true' else 'true')+'_nll'] for r in group)
        nc = sum(r[wanted+'_token_count'] for r in group)
        correct = sum(r[wanted+'_token_correct'] for r in group)
        answer[kind] = dict(denominator=len(group), numerator=wins, rate=wins/len(group),
            true_nll_mean=sum(r['true_nll'] for r in group)/len(group),
            new_nll_mean=sum(r['new_nll'] for r in group)/len(group),
            desired_token_count=nc, desired_token_correct=correct, token_micro=correct/nc,
            prompt_macro=sum(r[wanted+'_token_correct']/r[wanted+'_token_count'] for r in group)/len(group),
            strict_numerator=sum(r[wanted+'_strict'] for r in group),
            strict_denominator=len(group), new_strict_numerator=sum(r['new_strict'] for r in group))
    return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=Path, required=True)
    args = ap.parse_args()
    root = args.attempt.resolve()
    inventory = []

    def read(relative):
        p = root/relative
        b = p.read_bytes()
        inventory.append(dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest(),
                              mtime_ns=p.stat().st_mtime_ns, disposition='LOCAL_KEEP'))
        return json.loads(b)

    lock = read('execution.lock.json')
    assert lock['source_commit'] == 'fd2082e4720aa971dc64b00a133fce1d8e5e7d93'
    config = read('config.json')
    assert inventory[-1]['sha256'] == lock['config_sha256']
    cold = read('arm-A/main/cold.json')
    previous = cold['state']
    commits, ids, batch_stats = [], [], []
    for batch in range(1, 6):
        prefix = f'arm-A/main/batch-{batch:02d}'
        entry = read(prefix+'/entry.json')
        inputs = read(prefix+'/input.json')
        c = read(prefix+'/commit.json')
        assert c['source'] == lock['source_commit'] and c['config'] == digest(config)
        assert c['before'] == previous == entry['state']
        assert c['batch'] == batch and c['actual_B'] == 100 and c['history_appends'] == 5
        assert c['current_ids'] == inputs['ids'] and len(c['current_ids']) == 100
        assert c['no_checkpoint'] and c['candidate_count'] <= 25 and c['backward_count'] <= 24
        assert c['memory_after']['commits'] == batch and c['memory_after']['events'] == batch*100
        assert len(c['after']['W']) == len(c['after']['H']) == 5
        ids.extend(c['current_ids']); previous = c['after']; commits.append(c)
        batch_stats.append({k:c[k] for k in ('batch','actual_B','candidate_count','backward_count',
            'accepted_updates','rejected_trials','history_appends','seconds')})
    assert len(ids) == len(set(ids)) == 500
    summary = read('arm-A/main/observe-W05/summary.json')
    assert summary['endpoint'] == 5 and summary['requests'] == 500
    assert summary['state'] == previous
    assert summary['no_mutation'] and summary['memory_no_mutation']
    assert summary['memory'] == commits[-1]['memory_after']
    rows = []
    for offset in range(0, 500, 50):
        chunk = read(f'arm-A/main/observe-W05/chunk-{offset:04d}.json')
        assert chunk['state'] == previous and not chunk['optimizer_feedback']
        rows.extend(chunk['rows'])
    expected = [(case,kind,index) for case in ids for kind,count in (('R',1),('P',2),('N',10)) for index in range(count)]
    assert [(r['case_id'],r['kind'],r['prompt_index']) for r in rows] == expected
    assert all(r['endpoint'] == 5 for r in rows)
    assert len(rows) == summary['row_count'] == 6500
    assert digest([r['identity'] for r in rows]) == summary['row_order']
    result = reduce(rows)
    assert result == summary['summary'], 'INDEPENDENT_REDUCER_MISMATCH'
    assert {k:v['denominator'] for k,v in result.items()} == {'R':500,'P':1000,'N':5000}
    current = reduce([r for r in rows if r['case_id'] in set(ids[-100:])])
    assert current == summary['current']
    cohorts = {str(b):reduce([r for r in rows if r['case_id'] in set(ids[(b-1)*100:b*100])]) for b in range(1,6)}
    tail = root/'arm-A/main/batch-06'
    report = dict(scope='ARM_A_500_EDITS_W05_CUMULATIVE_ONLY',execution_source=lock['source_commit'],
        summary=result,current=current,cohorts_at_W05=cohorts,commits=batch_stats,
        rows=6500,unique_cases=500,state_sha256=digest(previous),row_order_sha256=summary['row_order'],
        observer_seconds=summary['seconds'],observer_mutation=False,
        memory={k:summary['memory'][k] for k in ('resident_count','events','admissions','evictions','commits')},
        checks='independent stdlib raw reducer, finite, TF counts, exact case/prompt order, denominators, state/commit chain',
        A_batch6=dict(entered=(tail/'entry.json').exists(),committed=(tail/'commit.json').exists(),
                     saved_candidate_receipts=len(list((tail/'fit').glob('candidate-*.json'))),
                     result_included=False,raw_preserved=True,rollback='NOT_VERIFIED_AFTER_SIGTERM'),
        other_arm_metrics='NOT_READ_NOT_INCLUDED',new_forward=0,new_checkpoint=False,
        artifacts=inventory)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
