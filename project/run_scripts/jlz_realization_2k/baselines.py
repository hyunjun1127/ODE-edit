"""Read-only historical W20 scalar reuse. No baseline fit or model import."""
import hashlib
import json
from pathlib import Path
from project.run_scripts.jlz_realization.observe import reduce_rows
from .common import require, digest, member, write, verify

TAG = {'RS':'R','PS':'P','NS':'N'}

def normalize(document, records, bench):
    require(document['requests'] == 2000, 'BASELINE_NOT_2K')
    rows = []
    for tag, kind in TAG.items():
        group = document['metrics'][tag]['rows']
        expected = [(r, i, p) for r in records for i,p in enumerate(bench.panels(r)[kind])]
        require(len(group) == len(expected), 'BASELINE_DENOMINATOR')
        for row, (record, index, prompt) in zip(group, expected):
            rw = record['requested_rewrite']
            values = [record['case_id'], index, prompt, rw['target_new']['str'], rw['target_true']['str']]
            old_id = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',',':'), ensure_ascii=True).encode()).hexdigest()
            require(row['case_id'] == record['case_id'] and row['prompt_index'] == index
                    and row['identity'] == old_id, 'BASELINE_PROMPT_TARGET_ORDER')
            for label in ('true','new'):
                _, target = bench.evaluation_ids(prompt, rw['target_' + label]['str'])
                require(len(target) == row[label + '_token_count'], 'BASELINE_TOKEN_LENGTH')
            # Explicit convention conversion after full prompt/target identity verification.
            rows.append(dict(row, kind=kind, identity=digest([record['case_id'],kind,index,prompt,
                rw['target_new']['str'],rw['target_true']['str']]), original_identity=row['identity']))
        reduced = reduce_rows([r for r in rows if r['kind'] == kind])[kind]
        require(reduced['numerator'] == document['metrics'][tag]['numerator'], 'BASELINE_RAW_NUMERATOR')
    reduce_rows(rows)
    return rows

def bind(records, bench, out):
    base = Path('/data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1')
    memit = Path('/data/janghj/ODE-edit/local/jlz-twoarm/20261002-bs100x20-v1/inputs/memit-h-baseline')
    cases = [('BASE_ALPHAEDIT',base/'output/main-cell-1/B020',base/'configs/AlphaEdit-native.json'),
             ('BASE_MEMIT',base/'output/main-cell-2/B020',base/'configs/MEMIT-native.json'),
             ('MEMIT-H',memit/'output/B020',memit/'execution.lock.json')]
    result = []
    for name, folder, meta in cases:
        item = dict(name=name, status='NOT_AVAILABLE', endpoint=20, new_fit=0,
            comparison_scope='HISTORICAL_REFERENCE_SAME_FIRST2000_NOT_MATCHED_RUNTIME',
            runtime_difference='Historical transformers4.44.2 vs current4.57.1; evaluator layout differs; no matched speed claim',
            hardware='S3 H200' if name == 'MEMIT-H' else 'historical S4; device binding in original receipt',
            layers=[4,5,6,7,8], past_actual_token_ids='NOT_RECORDED; prompt/target SHA and token counts rebound')
        try:
            path = folder/'seen-full.json'
            document = json.loads(path.read_text())
            commit = json.loads((folder/'commit.json').read_text())
            require(commit['batch'] == 20 and commit['seen_requests'] == 2000, 'BASELINE_ENDPOINT')
            if 'state' in document:
                require(document['state'] == commit.get('endpoint',commit.get('post')), 'BASELINE_STATE')
                binding = 'RAW_STATE_EQUALS_B020_COMMIT'
            else:
                require(name == 'MEMIT-H' and document['evaluation_type'] == 'ALL_SEEN_RPN'
                    and commit['nonfinite'] == commit['observer_mutation'] == 0
                    and commit['observer_context_rng_ledger_unchanged'], 'BASELINE_PRODUCER_GUARD')
                binding = 'SOURCE_B020_GUARD; RAW_STATE_FIELD_NOT_RECORDED'
            rows = normalize(document, records, bench)
            target = out/(name + '-normalized.json')
            write(target, dict(rows=rows, original=member(path), state_binding=binding, new_evaluation=0))
            item.update(status='VERIFIED_HISTORICAL_REUSE', raw=member(path), commit=member(folder/'commit.json'),
                        metadata=member(meta), normalized=member(target), summary=reduce_rows(rows), state_binding=binding)
        except Exception as exc:
            item.update(reason=str(exc), error_type=type(exc).__name__)
        result.append(item)
    result.append(dict(name='AlphaEdit-BLUE', status='NOT_AVAILABLE', endpoint=20, new_fit=0,
        reason='기존 local execution-tech-r2/main-llama inventory에 B020 same-first2000 full R/P/N endpoint 없음. 500 결과/최종10k로 대체하지 않음.',
        comparison_scope='NOT_AVAILABLE_2K', layers=[4,8], L2=1))
    write(out/'baseline-matrix.json',result)
    return result
