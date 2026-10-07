"""Observation-only GPTJ view and immutable scalar W0 references.

The scorer retains the frozen GPTJ grouping/token/left-padding semantics.  No
PRICE writer, optimizer, controller, or materialization is invoked here.
CAKE/BLUE retain their actual per-arm history; PRUNE/RECT have no history.
"""
import json
import time
from pathlib import Path

from project.run_scripts.jlz_realization.common import tensor_sha
from project.run_scripts.jlz_realization.observe import active_flags, reduce_rows
from project.run_scripts.jlz_price_gptj.scores import scores
from .common import digest, member, require, verify, write, sha, ROOT

RUNTIME_FIELDS = ('device', 'torch', 'transformers', 'model', 'FP32', 'eager',
                  'TF32', 'autocast', 'CPU_threads')
SITES = (3, 4, 5, 6, 7, 8)


class ObservationView:
    """Read-only physical-model interface; no replacement forward or fit."""
    def __init__(self, model, sites=SITES):
        self.model = model
        self.device = next(model.parameters()).device
        self.sites = tuple(sites)
        require(self.sites in (SITES, (3, 8)), 'GPTJ_ARM_PHYSICAL_SITES')
        self.weights = {layer: model.transformer.h[layer].mlp.fc_out.weight
                        for layer in self.sites}
        require(model.config.model_type == 'gptj', 'OBSERVER_GPTJ_MODEL')
        require(all(tuple(w.shape) == (4096, 16384) and str(w.dtype) == 'torch.float32'
                    for w in self.weights.values()), 'OBSERVER_GPTJ_LINEAR_LAYOUT')

    def observer_hidden(self, **tokens):
        # GPTJModel already applies ln_f. The scorer applies lm_head once.
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
    require(not history or set(map(int, history)) == set(view.sites), 'ALPHA_HISTORY_LAYERS')
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
    folder=Path(folder)
    if (folder/'reuse.json').exists():
        receipt=_safe_json(folder/'reuse.json')
        require(receipt['scalar_bridge_only'] and not receipt['history_or_editor_resume'],'SCALAR_ONLY')
        paths=[verify(item) for item in receipt['chunks']]
        expected_state=None # Original W0 chunk schema has no state field.
    else:paths=sorted(folder.glob('chunk-*.json'))
    result=[]
    for path in paths:
        chunk=_safe_json(path);require(chunk['optimizer_feedback'] is False,'NO_FEEDBACK')
        if expected_state is not None:require(chunk['state']==expected_state,'CHUNK_STATE')
        result.extend(chunk['rows'])
    return result

def install_W0(c, view, history, out, bench, records):
    """Use a ready, exact fresh W0 scalar bridge once, never wait/poll for it."""
    from project.run_scripts.jlz_interference_l1 import validate_rows as validate_w0
    source=Path(c['W0_reference'])
    if not (source/'result.json').exists() or not (source/'terminal.json').exists():
        return observe(view,bench,records,records,history,'W0',Path(out)/'W0')
    original=_safe_json(source/'config.json');lock=_safe_json(source/'execution.lock.json')
    terminal=_safe_json(source/'terminal.json');result=_safe_json(source/'result.json')
    if terminal.get('status')!='COMPLETED':
        return observe(view,bench,records,records,history,'W0',Path(out)/'W0')
    require(sha(source/'config.json')==lock['config_sha256'] and sha(source/'result.json')==terminal['result_sha256'],'W0_SEAL')
    require(original['revision']==c['model_revision'] and original['model']==c['model'] and original['seed']==c['seed'],'W0_MODEL_REVISION')
    require(original['assets']==c['model_assets'] and original['runtime']==c['runtime'] and original['microbatch']==2,'W0_MODEL_RUNTIME')
    require(original['ordered_ids']==c['ordered_ids_sha256'] and original['observer_manifest']==c['observer_identity'],'W0_ROW_INPUT')
    require(result['new_actual_evaluation'] and result['no_mutation'] and result['edit_calls']==result['fits']==result['solves']==result['history_appends']==0,'W0_COLD_OBSERVATION')
    for name in c['evaluator_sources']:
        require(sha(source/'source'/name)==sha(ROOT/name),'W0_SCORER_SOURCE')
    raw=[]
    for item in result['raw']:
        chunk=_safe_json(verify(item));require(chunk['optimizer_feedback'] is False,'NO_FEEDBACK');raw.extend(chunk['rows'])
    refs=json.loads(verify(c['observer_identity']).read_text())
    summary=validate_w0(raw,refs,[r['case_id'] for r in records],'W0')
    require(summary==result['summary'],'W0_REDUCTION')
    folder=Path(out)/'W0'
    write(folder/'reuse.json',dict(chunks=result['raw'],source_result=member(source/'result.json'),
        scalar_bridge_only=True,history_or_editor_resume=False,original_chunk_has_state=False,
        actual_cold_weights=state(view,history)['W'],source_config=member(source/'config.json')))
    value=dict(endpoint='W0',summary=summary,current=summary,requests=2000,row_count=26000,
        state=state(view,history),seconds=0.,new_forwards=0,no_mutation=True,scalar_bridge_only=True,
        reference_only=True,optimizer_feedback=False,row_order=digest([r['identity'] for r in raw]))
    write(folder/'summary.json',value);return value
