"""Read-only Llama native R/P/N observer; scalar W0 reuse is not editor resume.

Scoring and arithmetic reuse jlz_realization.observe unchanged.  Physical sites
may differ by native method (BLUE edits L4/L8), while scalar cold-W0 qualification
always checks every L4--L8 projection.  Generation is owned by the shared SH1
evaluator and deliberately is not implemented here.
"""
import copy
import json
import time
from pathlib import Path

from project.run_scripts.jlz_realization.common import (
    digest, member, require, sha, tensor_sha, write)
from project.run_scripts.jlz_realization.observe import active_flags, reduce_rows, scores
from project.run_scripts.jlz_realization.writer import rng_equal, rng_restore, rng_snapshot
from project.run_scripts.jlz_shared_budget.common import verify

SITES = (4, 5, 6, 7, 8)
HOOK_FIELDS = ('_forward_pre_hooks', '_forward_hooks', '_backward_pre_hooks',
               '_backward_hooks', '_forward_pre_hooks_with_kwargs',
               '_forward_hooks_with_kwargs', '_forward_hooks_always_called')
RUNTIME_FIELDS = ('device', 'torch', 'transformers', 'model', 'FP32', 'eager',
                  'TF32', 'autocast', 'CPU_threads')


class ObservationView:
    """The actual Llama model, with no replacement forward or injected loss."""
    def __init__(self, model, sites=SITES):
        self.model = model
        self.device = next(model.parameters()).device
        self.sites = tuple(sites)
        require(bool(self.sites) and len(set(self.sites)) == len(self.sites)
                and all(type(layer) is int and layer in SITES for layer in self.sites),
                'LLAMA_NATIVE_PHYSICAL_SITES')
        require(model.config.model_type == 'llama', 'OBSERVER_LLAMA_MODEL')
        self.weights = {layer: model.model.layers[layer].mlp.down_proj.weight
                        for layer in self.sites}
        require(all(tuple(weight.shape) == (4096, 14336)
                    and str(weight.dtype) == 'torch.float32'
                    for weight in self.weights.values()), 'OBSERVER_LLAMA_LINEAR_LAYOUT')

    def observer_hidden(self, **tokens):
        return self.model.model(**tokens, use_cache=False).last_hidden_state

    def guard(self):
        return tuple((name, id(value), value.data_ptr(), value._version,
                      tuple(value.shape), str(value.dtype))
                     for name, value in self.model.named_parameters())

    def hook_signature(self):
        return tuple((name, field, tuple((key, id(hook)) for key, hook in
                      getattr(module, field, {}).items()))
                     for name, module in self.model.named_modules() for field in HOOK_FIELDS)


def state(view, history):
    require(isinstance(history, dict), 'NATIVE_HISTORY_MAPPING')
    require(not history or set(history) == set(view.sites), 'NATIVE_HISTORY_PHYSICAL_SITES')
    return dict(W={str(layer): tensor_sha(weight) for layer, weight in view.weights.items()},
                H={str(layer): tensor_sha(value) for layer, value in history.items()})


def cold_weight_identity(model):
    """Hash actual full-five-site W0 even for a two-site native method."""
    require(model.config.model_type == 'llama', 'W0_LLAMA_MODEL')
    return {str(layer): tensor_sha(model.model.layers[layer].mlp.down_proj.weight)
            for layer in SITES}


def observe(view, bench, all_records, selected, history, endpoint, out,
            current_ids=None, microbatch=2):
    """Immutable scalar rows and exact current/seen reductions; preserve RNG.

    Completed chunks remain immutable if a later chunk fails.  The caller may
    roll back its RAM transaction, but an observation is never an editor commit.
    """
    out = Path(out)
    require(bool(selected), 'EMPTY_NATIVE_OBSERVATION')
    before = state(view, history)
    before_guard, hooks = view.guard(), view.hook_signature()
    rng, context = rng_snapshot(), copy.deepcopy(bench.contexts)
    started, result_rows = time.monotonic(), []
    flags = active_flags(all_records)
    selected_ids = [record['case_id'] for record in selected]
    require(len(set(selected_ids)) == len(selected_ids)
            and all(case in flags for case in selected_ids), 'OBSERVER_SELECTED_OCCURRENCES')
    current = list(selected_ids if current_ids is None else current_ids)
    require(len(set(current)) == len(current) and set(current) <= set(selected_ids),
            'OBSERVER_CURRENT_OCCURRENCES')
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
        rng_restore(rng)
        require(state(view, history) == before and view.guard() == before_guard
                and view.hook_signature() == hooks and bench.contexts == context
                and rng_equal(rng), 'NATIVE_OBSERVER_MUTATION')
    current_set = set(current)
    value = dict(schema='jlz-observer-summary-v1', endpoint=endpoint, state=before,
        requests=len(selected), summary=reduce_rows(result_rows),
        current=reduce_rows([row for row in result_rows if row['case_id'] in current_set]),
        row_count=len(result_rows), row_order=digest([row['identity'] for row in result_rows]),
        seconds=time.monotonic() - started, no_mutation=True, RNG_restored=True,
        optimizer_feedback=False, replay=False)
    write(out / 'summary.json', value)
    return value


def _safe_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'UNSAFE_OBSERVATION_MEMBER')
    return json.loads(path.read_text())


def rows(folder, expected_state=None):
    folder = Path(folder)
    reuse_path = folder / 'reuse.json'
    source_state = expected_state
    if reuse_path.exists():
        receipt = _safe_json(reuse_path)
        require(receipt['scalar_bridge_only'] and not receipt['history_or_editor_resume'],
                'W0_SCALAR_ONLY')
        require(digest(receipt['manifest']) == receipt['manifest_sha256'], 'W0_REFERENCE_HASH')
        if expected_state is not None:
            require(receipt['actual_native_state'] == expected_state, 'W0_NATIVE_STATE')
        source_state = receipt['manifest']['cold_state']
        paths = [verify(item) for item in receipt['manifest']['chunks']]
    else:
        paths = sorted(folder.glob('chunk-*.json'))
    require(bool(paths), 'MISSING_OBSERVER_CHUNKS')
    result = []
    for path in paths:
        chunk = _safe_json(path)
        require(chunk['optimizer_feedback'] is False, 'OBSERVER_NO_FEEDBACK')
        if source_state is not None:
            require(chunk['state'] == source_state, 'CHUNK_STATE')
        result.extend(chunk['rows'])
    reduce_rows(result)
    return result


def install_W0(c, view, history, out, bench, records):
    """Install only an explicitly qualified, hash-bound exact first2k manifest.

    PRICE's all-zero H provenance and the native editor's empty/two/five-site H
    are distinct: only scalar observations of identical cold weights are reused.
    A supplied invalid manifest blocks, never silently falls back to old scores.
    If no reuse is requested, the original necessary cold observation is run.
    """
    value = c.get('W0_reuse')
    if value is None:
        return observe(view, bench, records, records, history, 'W0', Path(out) / 'W0')
    from project.run_scripts.jlz_interference_l1 import validate_rows
    require(len(c.get('W0_evaluator', [])) == 3, 'W0_CURRENT_EVALUATOR_NOT_BOUND')
    from .common import ROOT, sha
    for row in c['W0_evaluator']:
        historical = verify(row['historical'])
        require(sha(ROOT / row['relative']) == row['current']['sha256']
                == row['historical']['sha256'], 'W0_CURRENT_EVALUATOR_BYTES')
    require(value['status'] == 'QUALIFIED_EXACT_REUSE', 'W0_REUSE_NOT_QUALIFIED')
    require(value['observation_identity'] == c['observation_identity'], 'W0_OBSERVER_IDENTITY')
    require(len(records) == 2000 and digest([row['case_id'] for row in records])
            == c['ordered_ids_sha256'], 'W0_FIRST2000_ORDER')
    actual = state(view, history)
    require(cold_weight_identity(view.model) == c['cold_W'] == value['cold_state']['W'],
            'W0_FULL_FIVE_COLD_WEIGHTS')
    require(all(bool((h == 0).all()) for h in history.values()), 'W0_NATIVE_HISTORY_NOT_COLD')
    runtime = _safe_json(verify(value['runtime']))
    current_runtime = _safe_json(Path(out) / 'runtime.json')
    require(all(runtime[key] == current_runtime[key] for key in RUNTIME_FIELDS),
            'W0_LIVE_RUNTIME')
    require(runtime['cold_W0_H0'] == value['cold_state'], 'W0_SOURCE_COLD_PROVENANCE')
    require(value['cold_state']['H'] == c['W0_source_zero_H'], 'W0_SOURCE_ZERO_HISTORY_PROVENANCE')
    refs = _safe_json(verify(c['observer_identity']))['rows']
    require(len(value['chunks']) == 40, 'W0_FULL_CHUNK_COUNT')
    raw = []
    for item in value['chunks']:
        chunk = _safe_json(verify(item))
        require(chunk['state'] == value['cold_state'] and chunk['optimizer_feedback'] is False,
                'W0_SOURCE_CHUNK_STATE')
        raw.extend(chunk['rows'])
    summary = validate_rows(raw, refs, [row['case_id'] for row in records], 'W0')
    old = _safe_json(verify(value['summary']))
    require(old['endpoint'] == 'W0' and old['state'] == value['cold_state']
            and old['summary'] == summary and old['no_mutation']
            and old['optimizer_feedback'] is False and len(raw) == 26000,
            'W0_FULL_REDUCTION_IDENTITY')
    folder = Path(out) / 'W0'
    write(folder / 'reuse.json', dict(manifest=value, manifest_sha256=digest(value),
        actual_native_state=actual, source_runtime=member(verify(value['runtime'])),
        scalar_bridge_only=True, history_or_editor_resume=False, new_forwards=0,
        source_PRICE_zero_history_is_not_native_history=True, raw_copied=False))
    result = dict(old, state=actual, seconds=0., new_forwards=0, reference_only=True,
        scalar_bridge_only=True, native_history_separately_recorded=True,
        original_evaluation_seconds=old.get('original_evaluation_seconds', old['seconds']),
        reused_from=value['source_folder'], row_order=digest([row['identity'] for row in raw]))
    write(folder / 'summary.json', result)
    return result
