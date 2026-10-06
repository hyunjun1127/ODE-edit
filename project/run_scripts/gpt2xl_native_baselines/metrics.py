"""Observation-only GPT2 view and immutable scalar W0 references.

The scorer retains the frozen GPT2 grouping/token/left-padding semantics.  No
PRICE writer, optimizer, controller, or materialization is invoked here.
MEMIT has no edit-history state; only AlphaEdit's actual cache is observed.
"""
import json
import time
from pathlib import Path

from project.run_scripts.jlz_realization.common import tensor_sha
from project.run_scripts.jlz_realization.observe import active_flags, reduce_rows
from project.run_scripts.jlz_price_gpt2xl.scores import scores
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    compare_summary, reduce_rows as independent_reduce, validate_rows)
from .common import digest, member, require, verify, write

RUNTIME_FIELDS = ('device', 'torch', 'transformers', 'model', 'FP32', 'eager',
                  'TF32', 'autocast', 'CPU_threads')
SITES = (13, 14, 15, 16, 17)


class ObservationView:
    """Read-only physical-model interface; no replacement forward or fit."""
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.sites = SITES
        self.weights = {layer: model.transformer.h[layer].mlp.c_proj.weight
                        for layer in self.sites}
        require(model.config.model_type == 'gpt2', 'OBSERVER_GPT2_MODEL')
        require(all(tuple(w.shape) == (6400, 1600) and str(w.dtype) == 'torch.float32'
                    for w in self.weights.values()), 'OBSERVER_CONV1D_LAYOUT')

    def observer_hidden(self, **tokens):
        # GPT2Model already applies ln_f. The scorer applies lm_head once.
        return self.model.transformer(**tokens, use_cache=False).last_hidden_state

    def guard(self):
        return tuple((name, id(value), value.data_ptr(), value._version,
                      tuple(value.shape), str(value.dtype))
                     for name, value in self.model.named_parameters())

    def hook_signature(self):
        fields = ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks',
                  '_backward_hooks')
        return tuple((name, field, tuple((key, id(hook)) for key, hook in
                      getattr(module, field, {}).items()))
                     for name, module in self.model.named_modules() for field in fields)


def state(view, history):
    require(isinstance(history, dict), 'NATIVE_HISTORY_MAPPING')
    require(not history or set(map(int, history)) == set(SITES), 'ALPHA_HISTORY_LAYERS')
    return dict(W={str(layer): tensor_sha(weight) for layer, weight in view.weights.items()},
                H={str(layer): tensor_sha(value) for layer, value in history.items()})


def observe(view, bench, all_records, selected, history, endpoint, out,
            current_ids=None, microbatch=2):
    """Canonical scalar rows; the same physical state is guarded before/after."""
    out = Path(out)
    before = state(view, history)
    before_guard, hooks = view.guard(), view.hook_signature()
    started, result_rows = time.monotonic(), []
    flags = active_flags(all_records)
    try:
        for start in range(0, len(selected), 50):
            specs, pairs = [], []
            for record in selected[start:start + 50]:
                rewrite = record['requested_rewrite']
                for kind, prompts in bench.panels(record).items():
                    require(kind in ('R', 'P', 'N'), 'OBSERVER_KIND')
                    for index, prompt in enumerate(prompts):
                        specs.append(dict(case_id=record['case_id'], kind=kind,
                            prompt_index=index, identity=digest([record['case_id'], kind,
                                index, prompt, rewrite['target_new']['str'],
                                rewrite['target_true']['str']]), endpoint=endpoint,
                            active_at_endpoint=flags[record['case_id']]))
                        pairs.extend(((prompt, rewrite['target_new']['str']),
                                      (prompt, rewrite['target_true']['str'])))
            values = scores(view, bench, pairs, microbatch)
            require(len(values) == 2 * len(specs), 'OBSERVER_PAIR_CARDINALITY')
            for index, row in enumerate(specs):
                for label, value in zip(('new', 'true'), values[2 * index:2 * index + 2]):
                    row.update({label + '_' + key: item for key, item in value.items()})
                row['margin_true_minus_new'] = row['true_nll'] - row['new_nll']
                row['margin_new_minus_true'] = row['new_nll'] - row['true_nll']
            reduce_rows(specs)
            write(out / f'chunk-{start:04d}.json', dict(schema='jlz-observer-rows-v1',
                state=before, rows=specs, optimizer_feedback=False))
            result_rows.extend(specs)
    finally:
        require(state(view, history) == before and view.guard() == before_guard
                and view.hook_signature() == hooks, 'NATIVE_OBSERVER_MUTATION')
    current = set(current_ids if current_ids is not None else
                  (record['case_id'] for record in selected))
    value = dict(schema='jlz-observer-summary-v1', endpoint=endpoint, state=before,
        requests=len(selected), summary=reduce_rows(result_rows),
        current=reduce_rows([row for row in result_rows if row['case_id'] in current]),
        row_count=len(result_rows), row_order=digest([row['identity'] for row in result_rows]),
        seconds=time.monotonic() - started, no_mutation=True, optimizer_feedback=False,
        replay=False)
    write(out / 'summary.json', value)
    return value


def _safe_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'UNSAFE_OBSERVATION_MEMBER')
    return json.loads(path.read_text())


def rows(folder, expected_state=None):
    """Read original bytes; reused method state is never relabelled in raw."""
    folder = Path(folder)
    reference = folder / 'reuse.json'
    if reference.exists():
        receipt = _safe_json(reference)
        value = receipt['manifest']
        require(receipt['manifest_sha256'] == digest(value), 'W0_REFERENCE_BINDING')
        require(receipt['endpoint'] == 'W0' and receipt['no_forward']
                and receipt['no_raw_copy'] and receipt['no_checkpoint'], 'W0_REFERENCE_SCOPE')
        actual, source = receipt['actual_observer_state'], receipt['source_state']
        require(actual['W'] == source['W'] and receipt['projection'] == 'MODEL_WEIGHTS_ONLY',
                'W0_MODEL_STATE_PROJECTION')
        require(not actual['H'] or actual['H'] == source['H'], 'W0_HISTORY_NOT_RESUMED')
        if expected_state is not None:
            require(actual == expected_state, 'W0_ACTUAL_STATE')
        require(value['cold_state'] == source and len(value['chunks']) == 40, 'W0_SOURCE_STATE')
        paths = [verify(item) for item in value['chunks']]
        chunk_state = source
    else:
        paths = sorted(folder.glob('chunk-*.json'))
        chunk_state = expected_state
    result = []
    for path in paths:
        chunk = _safe_json(path)
        require(chunk['optimizer_feedback'] is False, 'OBSERVER_FEEDBACK')
        if chunk_state is not None:
            require(chunk['state'] == chunk_state, 'OBSERVER_CHUNK_STATE')
        result.extend(chunk['rows'])
    return result


def _expected(identities, ids):
    require(len(ids) == len(set(ids)), 'DUPLICATE_REQUEST_OCCURRENCE')
    indexed = {}
    for row in identities:
        indexed.setdefault(row['case_id'], []).append(row)
    require(all(case in indexed for case in ids), 'MISSING_OBSERVER_IDENTITY')
    return [row for case in ids for row in indexed[case]]


def install_W0(c, view, history, out, bench):
    """Reference exact cold observations, not an edited-state resume.

    Runtime/model/token/evaluator qualification is bound in the owner config;
    this function rechecks physical W, runtime fields, bound bytes and counts.
    """
    value = c['W0_reuse']
    require(value['status'] == 'QUALIFIED_EXACT_REUSE', 'W0_REUSE_NOT_QUALIFIED')
    require(value['observation_identity'] == c['observation_identity'], 'W0_EVALUATOR_IDENTITY')
    actual = state(view, history)
    require(actual['W'] == c['cold_W'] == value['cold_state']['W'], 'W0_COLD_WEIGHT_IDENTITY')
    require(not actual['H'] or actual['H'] == value['cold_state']['H'], 'W0_ALPHA_H0_IDENTITY')
    original_runtime = json.loads(verify(value['runtime']).read_text())
    runtime = _safe_json(Path(out) / 'runtime.json')
    require(all(key in runtime and runtime[key] == original_runtime[key]
                for key in RUNTIME_FIELDS), 'W0_RUNTIME_IDENTITY')
    identities = json.loads(verify(c['observer_identity']).read_text())['rows']
    ids = [case for pack in c['packs'] for case in pack['ids']]
    require(len(ids) == 2000 and len(value['chunks']) == 40, 'W0_EXACT_FIRST2000_CHUNKS')
    raw = []
    for item in value['chunks']:
        chunk = json.loads(verify(item).read_text())
        require(chunk['state'] == value['cold_state'] and chunk['optimizer_feedback'] is False,
                'W0_SOURCE_RAW_STATE')
        raw.extend(chunk['rows'])
    validate_rows(raw, _expected(identities, ids), 'W0')
    require(len(raw) == 26000, 'W0_FULL_ROW_COUNT')
    original = json.loads(verify(value['summary']).read_text())
    require(original['endpoint'] == 'W0' and original['state'] == value['cold_state']
            and original['requests'] == 2000 and original['row_count'] == 26000
            and original['no_mutation'] is True and original['optimizer_feedback'] is False
            and original['row_order'] == digest([row['identity'] for row in raw]), 'W0_SOURCE_SUMMARY')
    reduced = independent_reduce(raw)
    compare_summary(reduced, original['summary'])
    require({kind: reduced[kind]['denominator'] for kind in ('R', 'P', 'N')}
            == dict(R=2000, P=4000, N=20000), 'W0_EXACT_DENOMINATORS')
    folder = Path(out) / 'W0'
    receipt = dict(schema='native-observation-reuse-v1', endpoint='W0', manifest=value,
        manifest_sha256=digest(value), source_state=value['cold_state'],
        actual_observer_state=actual, projection='MODEL_WEIGHTS_ONLY',
        method_state_reused=False, no_forward=True, no_raw_copy=True, no_checkpoint=True)
    write(folder / 'reuse.json', receipt)
    summary = dict(original, state=actual, seconds=0., new_forwards=0, reference_only=True,
        original_evaluation_seconds=original.get('original_evaluation_seconds', original['seconds']),
        reused_from=value['source_folder'], source_observed_state=value['cold_state'])
    write(folder / 'summary.json', summary)
    return summary
