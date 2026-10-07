"""Observation-only GPT2 metrics; native fit contexts are never transplanted.

The canonical scorer is read-only. Its grouping, learned-position/left-padding,
target means, and final-LN/head semantics are inherited unchanged.
"""
import json
import time
from pathlib import Path

from project.run_scripts.gpt2xl_native_baselines.metrics import (
    ObservationView, RUNTIME_FIELDS, _expected)
from project.run_scripts.jlz_price_gpt2xl.scores import scores
from project.run_scripts.jlz_realization.observe import active_flags, reduce_rows as production_reduce
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    compare_summary, reduce_rows, validate_rows)
from .common import digest, require, verify, write, tensor_sha


def state(view, history):
    require(isinstance(history, dict) and set(history) <= {13,14,15,16,17},
            'NATIVE_PHYSICAL_HISTORY_LAYERS')
    return dict(W={str(l):tensor_sha(w) for l,w in view.weights.items()},
                H={str(l):tensor_sha(h) for l,h in history.items()})


def observe(view, bench, all_records, selected, history, endpoint, out,
            current_ids=None, microbatch=2):
    """Unchanged canonical scorer, generalized physical native H slots only."""
    out = Path(out)
    before = state(view, history)
    before_guard, hooks = view.guard(), view.hook_signature()
    started, result_rows = time.monotonic(), []
    flags = active_flags(all_records)
    try:
        for start in range(0, len(selected), 50):
            specs, pairs = [], []
            for record in selected[start:start+50]:
                rewrite = record['requested_rewrite']
                for kind,prompts in bench.panels(record).items():
                    require(kind in ('R','P','N'), 'OBSERVER_KIND')
                    for index,prompt in enumerate(prompts):
                        specs.append(dict(case_id=record['case_id'],kind=kind,prompt_index=index,
                            identity=digest([record['case_id'],kind,index,prompt,
                                rewrite['target_new']['str'],rewrite['target_true']['str']]),
                            endpoint=endpoint,active_at_endpoint=flags[record['case_id']]))
                        pairs.extend(((prompt,rewrite['target_new']['str']),
                                      (prompt,rewrite['target_true']['str'])))
            values = scores(view,bench,pairs,microbatch)
            require(len(values) == 2*len(specs), 'OBSERVER_PAIR_CARDINALITY')
            for index,row in enumerate(specs):
                for label,value in zip(('new','true'),values[2*index:2*index+2]):
                    row.update({label+'_'+key:item for key,item in value.items()})
                row['margin_true_minus_new'] = row['true_nll']-row['new_nll']
                row['margin_new_minus_true'] = row['new_nll']-row['true_nll']
            production_reduce(specs)
            write(out/f'chunk-{start:04d}.json',dict(schema='jlz-observer-rows-v1',
                state=before,rows=specs,optimizer_feedback=False))
            result_rows.extend(specs)
    finally:
        require(state(view,history) == before and view.guard() == before_guard
                and view.hook_signature() == hooks, 'NATIVE_OBSERVER_MUTATION')
    current = set(current_ids if current_ids is not None else (r['case_id'] for r in selected))
    result = dict(schema='jlz-observer-summary-v1',endpoint=endpoint,state=before,
        requests=len(selected),summary=production_reduce(result_rows),
        current=production_reduce([r for r in result_rows if r['case_id'] in current]),
        row_count=len(result_rows),row_order=digest([r['identity'] for r in result_rows]),
        seconds=time.monotonic()-started,no_mutation=True,optimizer_feedback=False,replay=False)
    write(out/'summary.json',result)
    return result


def rows(folder, expected_state=None):
    folder = Path(folder)
    if (folder / 'reuse.json').exists():
        receipt = json.loads((folder / 'reuse.json').read_text())
        value = receipt['manifest']
        require(receipt['manifest_sha256'] == digest(value)
                and receipt['no_forward'] and receipt['no_raw_copy']
                and receipt['projection'] == 'MODEL_WEIGHTS_ONLY'
                and receipt['method_state_reused'] is False, 'W0_REFERENCE_SCOPE')
        require(receipt['source_state']['W'] == receipt['actual_observer_state']['W'],
                'W0_WEIGHT_PROJECTION')
        if expected_state is not None:
            require(receipt['actual_observer_state'] == expected_state, 'W0_ACTUAL_STATE')
        paths = [verify(item) for item in value['chunks']]
        source_state = receipt['source_state']
    else:
        paths = sorted(folder.glob('chunk-*.json'))
        source_state = expected_state
    result = []
    for path in paths:
        chunk = json.loads(path.read_text())
        require(chunk['optimizer_feedback'] is False, 'OBSERVER_NO_FEEDBACK')
        if source_state is not None:
            require(chunk['state'] == source_state, 'OBSERVER_EXACT_STATE')
        result.extend(chunk['rows'])
    return result


def install_W0(c, view, history, out, bench, records):
    """Exact model-only old raw reuse, or one fresh cold first2k observation.

CAKE's explicit zero RAM H is not copied from an old method. The evaluator does
not consume H, so only the model W may be projected into this observation; the
current native history remains separately hash-bound and untouched.
"""
    actual = state(view, history)
    require(actual['W'] == c['cold_W'], 'ACTUAL_COLD_W0')
    value = c.get('W0_reuse') or {}
    if value.get('status') != 'QUALIFIED_EXACT_REUSE':
        result = observe(view, bench, records, records, history, 'W0', Path(out) / 'W0')
        require(result['row_count'] == 26000, 'W0_FIRST2K_COUNT')
        return result
    require(value['observation_identity'] == c['observation_identity']
            and actual['W'] == value['cold_state']['W'], 'W0_REUSE_IDENTITY')
    original_runtime = json.loads(verify(value['runtime']).read_text())
    current_runtime = json.loads((Path(out) / 'runtime.json').read_text())
    require(all(current_runtime.get(k) == original_runtime.get(k) for k in RUNTIME_FIELDS),
            'W0_REUSE_RUNTIME')
    identities = json.loads(verify(c['observer_identity']).read_text())['rows']
    ids = [record['case_id'] for record in records]
    require(len(ids) == 2000 and len(value['chunks']) == 40, 'W0_REUSE_FULL_FIRST2K')
    raw = []
    for item in value['chunks']:
        chunk = json.loads(verify(item).read_text())
        require(chunk['state'] == value['cold_state'] and chunk['optimizer_feedback'] is False,
                'W0_REUSE_RAW_STATE')
        raw.extend(chunk['rows'])
    validate_rows(raw, _expected(identities, ids), 'W0')
    original = json.loads(verify(value['summary']).read_text())
    require(original['endpoint'] == 'W0' and original['requests'] == 2000
            and original['row_count'] == len(raw) == 26000
            and original['row_order'] == digest([r['identity'] for r in raw])
            and original['no_mutation'] and original['optimizer_feedback'] is False,
            'W0_REUSE_SUMMARY_IDENTITY')
    reduced = reduce_rows(raw)
    compare_summary(reduced, original['summary'])
    require({k: reduced[k]['denominator'] for k in ('R', 'P', 'N')}
            == dict(R=2000, P=4000, N=20000), 'W0_REUSE_DENOMINATORS')
    receipt = dict(endpoint='W0', manifest=value, manifest_sha256=digest(value),
        source_state=value['cold_state'], actual_observer_state=actual,
        projection='MODEL_WEIGHTS_ONLY', method_state_reused=False,
        no_forward=True, no_raw_copy=True, no_checkpoint=True)
    write(Path(out) / 'W0/reuse.json', receipt)
    result = dict(original, state=actual, seconds=0., new_forwards=0, reference_only=True,
        source_observed_state=value['cold_state'],
        original_evaluation_seconds=original.get('original_evaluation_seconds', original['seconds']))
    write(Path(out) / 'W0/summary.json', result)
    return result
