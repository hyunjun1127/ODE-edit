"""Frozen GPT-J cache qualification caller; no generation implementation.

The SH1 transport is deliberately an injected ``measure`` callable, not an
invented shared import. It receives ``route, cohort, microbatch, reference``
and returns output rows, transient forced-prefix traces, metric rows/summary,
coverage, work and cost. The caller adapter must bind the eventual immutable
shared API. Traces/logits never enter durable receipts.
"""
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np


PLAN_SCHEMA = 'gptj-generation-cache-qualification-plan-v1'
RECEIPT_SCHEMA = 'gptj-generation-cache-qualification-v1'
PROFILE = 'cf-cake-prompt-inclusive-total100-eos-corrected-v1'
REFERENCE = 'UNPADDED_FULL_PREFIX_NO_CACHE'
SINGLETON = 'UNPADDED_SINGLETON_KV_CACHE'
BATCH = 'EQUAL_TOKEN_LENGTH_KV_BATCH'
ROUTES = (REFERENCE, SINGLETON, BATCH)
SHARED_SOURCE = '199cfe5664355f6f1c9069c72396ec759bb25cec'
SHARED_ROUTES = {REFERENCE:REFERENCE, SINGLETON:'UNPADDED_KV_SINGLETON',
                 BATCH:'EQUAL_LENGTH_KV_BATCH'}
TOLERANCES = dict(logits=dict(atol=2e-4, rtol=2e-4),
    topk_probabilities=dict(atol=2e-5, rtol=2e-4),
    metrics=dict(atol=1e-6, rtol=0), tokens='exact', topk_ids='exact',
    EOS='exact', row_mapping='exact', seed_stream='exact',
    validity_counts_and_missing_reasons='exact')
REQUIRED_COVERAGE = ('forced_prefix', 'full_vocab_logits', 'topk_probabilities',
                     'tokens', 'EOS', 'row_mapping', 'seed_stream', 'metrics')
WORK_FIELDS = ('physical_forward_calls', 'prefill_query_tokens', 'decode_query_tokens',
               'logical_row_token_decisions')
COST_FIELDS = ('elapsed_seconds', 'synchronized_GPU_seconds', 'peak_gpu_allocated_bytes',
               'peak_gpu_reserved_bytes', 'peak_host_RSS_bytes')


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def member(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'QUALIFICATION_REGULAR_FILE')
    path = path.resolve()
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def installed_native_binding():
    """Small installed source/version metadata; no model or CUDA import/load."""
    import torch
    import transformers
    versions=dict(torch=str(torch.__version__),transformers=str(transformers.__version__))
    distribution_versions={name:importlib.metadata.version(name) for name in ('torch','transformers')}
    require(versions == dict(torch='2.9.1+cu128', transformers='4.57.1'),
            'QUALIFICATION_NATIVE_RUNTIME_PIN')
    require(distribution_versions==dict(torch='2.9.1',transformers='4.57.1'),
            'QUALIFICATION_NATIVE_DISTRIBUTION_PIN')
    distribution = importlib.metadata.distribution('transformers')
    names = ('transformers/models/gptj/modeling_gptj.py', 'transformers/cache_utils.py')
    return dict(versions=versions,distribution_versions=distribution_versions,files=[dict(relative=name,
        **member(distribution.locate_file(name))) for name in names],
        native_path='DynamicCache(config); get_seq_length; cache_position; position_ids',
        caller_config_use_cache_unchanged=True, KV_lifetime='CALL_LOCAL_RAM_ONLY')


def _length(tokenizer, prompt):
    encoded = tokenizer(prompt, padding=False, truncation=False)
    ids = encoded['input_ids']
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        require(len(ids) == 1, 'QUALIFICATION_SINGLE_PROMPT_TOKENS')
        ids = ids[0]
    require(isinstance(ids, list) and ids and all(type(x) is int for x in ids),
            'QUALIFICATION_TOKEN_LENGTH_SCHEMA')
    return len(ids)


def build_plan(records, tokenizer, *, model_identity, tokenizer_identity,
               reference_identity, shared_source_sha, native_source_binding,
               batch_microbatch=8, admission_reason='DEFAULT_MB8', fixture_only=False):
    """Freeze the real SH1 token-only selected requests; raw IDs stay local."""
    from project.run_scripts.experiment_generation_eval.kv_qualification import build_qualification_plan
    require(batch_microbatch in (4, 8), 'QUALIFICATION_PREDECLARED_MB4_OR_MB8')
    require(admission_reason in ('DEFAULT_MB8', 'PREDECLARED_MEMORY_MB4')
            and (batch_microbatch == 4) == (admission_reason == 'PREDECLARED_MEMORY_MB4'),
            'QUALIFICATION_ADMISSION_CHOICE_BEFORE_MEASUREMENT')
    records = list(records)
    require(len(records) == 2000 and len({r['case_id'] for r in records}) == 2000,
            'QUALIFICATION_FIXED_FIRST2000')
    if not fixture_only:
        from .generation_common import ORDERED_SHA
        require(digest([r['case_id'] for r in records]) == ORDERED_SHA,
                'QUALIFICATION_CANONICAL_FIRST2000_ORDER')
    ordered=[]
    for ordinal, record in enumerate(records, 1):
        require(all(type(record[k]) is int and record[k]==ordinal for k in
            ('ordered_occurrence','occurrence_index','occurrence','ordinal') if k in record),
            'QUALIFICATION_RECORD_OCCURRENCE')
        ordered.append(dict(copy.deepcopy(record),occurrence_index=ordinal))
    shared=build_qualification_plan(tokenizer,ordered,model_identity=model_identity,
        microbatch=batch_microbatch,memory_alternative=admission_reason=='PREDECLARED_MEMORY_MB4')
    selected=[dict(occurrence=r['occurrence'],case_id=records[r['record_index']]['case_id'],
        prompt_index=r['prompt_index'],prompt=r['prompt'],input_token_count=r['token_length'])
        for r in shared['requests']]
    lengths = [r['input_token_count'] for r in selected]
    coverage = dict(same_length_width=shared['coverage']['tested_max_batch_width'],
        shortest_input=shared['coverage']['shortest'],longest_input=shared['coverage']['longest'],
        length100_boundary=any(99 <= length <= 101 for length in lengths),
        has_decode=any(length < 99 for length in lengths),
        has_length_cap_no_continuation=any(length >= 100 for length in lengths))
    local = dict(schema=PLAN_SCHEMA, raw_local_only=True, selection='SH1_TOKEN_LENGTH_ONLY',
                 prompts=selected,shared_plan=shared)
    plan = dict(schema=PLAN_SCHEMA, status='CPU_PLAN_FROZEN_NOT_GPU_PASS',
        model='gptj', model_identity=model_identity, tokenizer_identity=tokenizer_identity,
        fixture_only=fixture_only, vocab_size=50400,
        reference_identity=reference_identity, shared_source_sha=shared_source_sha,
        native_source_binding=copy.deepcopy(native_source_binding), profile=PROFILE,
        eval_seed=20261007, max_prompts=8, cohort_count=len(selected),
        cohort_sha256=digest(local),shared_plan_sha256=digest(shared),
        shared_route_map=copy.deepcopy(SHARED_ROUTES),
        selection='SH1_TOKEN_LENGTH_THEN_ORIGINAL_ORDINAL_ONLY', coverage=coverage,
        tolerances=copy.deepcopy(TOLERANCES), routes=list(ROUTES),
        batch_microbatch=batch_microbatch, admission_reason=admission_reason,
        alternate_microbatch=4, automatic_OOM_retry=False,
        measure_each_route_once=True, fallback_order=[BATCH, SINGLETON, REFERENCE],
        no_fit=True, checkpoint_saved=False, KV_lifetime='CALL_LOCAL_RAM_ONLY')
    return plan, local


def freeze_plan(plan, cohort, out):
    from project.run_scripts.experiment_generation_eval.common import immutable_write
    validate_plan(plan, cohort)
    out = Path(out)
    immutable_write(out / 'plan.json', plan)
    immutable_write(out / 'cohort-local.json', cohort)
    immutable_write(out / 'shared-plan-local.json', cohort['shared_plan'])
    return dict(qualification_plan=member(out / 'plan.json'),
        qualification_plan_sha256=digest(plan), qualification_cohort=member(out / 'cohort-local.json'),
        shared_qualification_plan=member(out / 'shared-plan-local.json'),
        shared_qualification_plan_sha256=plan['shared_plan_sha256'],
        status='CPU_PLAN_FROZEN_NOT_GPU_PASS', actual_GPU=False)


def validate_plan(plan, cohort):
    require(plan['schema'] == PLAN_SCHEMA and plan['status'] == 'CPU_PLAN_FROZEN_NOT_GPU_PASS'
            and plan['profile'] == PROFILE and plan['eval_seed'] == 20261007
            and plan['tolerances'] == TOLERANCES and plan['routes'] == list(ROUTES),
            'QUALIFICATION_FROZEN_PLAN_CHANGED')
    require(plan['cohort_sha256'] == digest(cohort) and cohort['raw_local_only'] is True
            and 1 <= plan['cohort_count'] == len(cohort['prompts']) <= 8,
            'QUALIFICATION_FROZEN_COHORT_CHANGED')
    shared=cohort['shared_plan']
    require(plan['shared_plan_sha256']==digest(shared)
            and plan['shared_route_map']==SHARED_ROUTES
            and shared['model_identity']==plan['model_identity']
            and shared['microbatch']==plan['batch_microbatch']
            and shared['allowed_routes']==list(SHARED_ROUTES.values())
            and shared['tolerances']=={k:plan['tolerances'][k]
                for k in ('logits','topk_probabilities','metrics')}
            and [(r['occurrence'],r['prompt_index'],r['prompt'],r['token_length'])
                 for r in shared['requests']]
                ==[(r['occurrence'],r['prompt_index'],r['prompt'],r['input_token_count'])
                    for r in cohort['prompts']], 'QUALIFICATION_SHARED_PLAN_CHANGED')
    require(plan['batch_microbatch'] in (4, 8) and plan['no_fit'] is True
            and plan['checkpoint_saved'] is False and plan['automatic_OOM_retry'] is False,
            'QUALIFICATION_FROZEN_NO_RETRY_PROFILE')
    require(plan['model'] == 'gptj' and plan['vocab_size'] == 50400
            and plan['measure_each_route_once'] is True
            and plan['fallback_order'] == [BATCH, SINGLETON, REFERENCE]
            and plan['KV_lifetime'] == 'CALL_LOCAL_RAM_ONLY'
            and plan['admission_reason'] in ('DEFAULT_MB8', 'PREDECLARED_MEMORY_MB4')
            and (plan['batch_microbatch'] == 4)
                == (plan['admission_reason'] == 'PREDECLARED_MEMORY_MB4'),
            'QUALIFICATION_FROZEN_ROUTE_ADMISSION')


def _key(row):
    return row['occurrence'], row['prompt_index']


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    require(value.ndim == 1 and np.isfinite(value).all(), 'QUALIFICATION_TRACE_FINITE_VECTOR')
    return value


def _metric_equal(before, after):
    if type(before) is dict and type(after) is dict:
        return before.keys() == after.keys() and all(_metric_equal(before[k], after[k]) for k in before)
    if type(before) is list and type(after) is list:
        return len(before) == len(after) and all(_metric_equal(a, b) for a, b in zip(before, after))
    if type(before) is float or type(after) is float:
        return type(before) is float and type(after) is float and math.isfinite(before) \
            and math.isfinite(after) and math.isclose(before, after, rel_tol=0, abs_tol=1e-6)
    return type(before) is type(after) and before == after


def compare_route(plan, cohort, reference, candidate, route):
    """Compare transient proof, emitting compact booleans/max errors only."""
    gates = dict(tokens=True, EOS=True, row_mapping=True, seed_stream=True,
                 logits=True, topk_ids=True, topk_probabilities=True, metrics=True,
                 positions=True, coverage=True)
    expected = [_key(row) for row in cohort['prompts']]
    gates['row_mapping'] = [_key(r) for r in reference['rows']] == expected \
        and [_key(r) for r in candidate['rows']] == expected
    output_fields = ('input_token_ids', 'continuation_token_ids', 'full_token_ids',
                     'input_token_count', 'continuation_token_count', 'stop_reason')
    if len(reference['rows']) != len(candidate['rows']):
        gates['tokens'] = False
    for before, after in zip(reference['rows'], candidate['rows']):
        gates['tokens'] &= all(before[k] == after[k] for k in output_fields)
        gates['EOS'] &= all(before[k] == after[k] for k in ('eos_ids', 'eos_binding'))
        gates['seed_stream'] &= before['seed'] == after['seed']
    by_key = {_key(row):row for row in reference['rows']}
    for selected in cohort['prompts']:
        row=by_key.get(_key(selected))
        if row is None:
            gates['tokens']=False
            continue
        gates['tokens'] &= row['input_token_count']==selected['input_token_count'] \
            and len(row['input_token_ids'])==row['input_token_count'] \
            and len(row['continuation_token_ids'])==row['continuation_token_count'] \
            and row['full_token_ids']==row['input_token_ids']+row['continuation_token_ids'] \
            and (not row['continuation_token_ids'] or len(row['full_token_ids'])<=100)
    gates['metrics'] = _metric_equal(reference['metric_rows'], candidate['metric_rows']) \
        and _metric_equal(reference['metric_summary'], candidate['metric_summary'])
    before_trace = reference['trace']; after_trace = candidate['trace']
    expected_trace=[(key,step) for key in expected if key in by_key
                    for step in range(by_key[key]['continuation_token_count'])]
    gates['row_mapping'] &= [(_key(row),row['step']) for row in before_trace]==expected_trace
    gates['logits'] = len(before_trace) == len(after_trace) \
        and (not plan['coverage']['has_decode'] or bool(before_trace))
    max_logits = max_probabilities = 0.
    for before, after in zip(before_trace, after_trace):
        gates['row_mapping'] &= (_key(before), before['step']) == (_key(after), after['step'])
        prefix = before['prefix_token_ids']
        gates['tokens'] &= prefix == after['prefix_token_ids']
        row=by_key.get(_key(before))
        gates['tokens'] &= row is not None and prefix==row['input_token_ids'] \
            +row['continuation_token_ids'][:before['step']]
        gates['seed_stream'] &= all(before[k] == after[k] for k in
                                  ('generator_state_before', 'generator_state_after'))
        gates['seed_stream'] &= before.get('replayed_sample_exact',True) is True \
            and after.get('replayed_sample_exact',True) is True
        gates['topk_ids'] &= before['topk_ids'] == after['topk_ids']
        for field, gate in (('logits', 'logits'), ('topk_probabilities', 'topk_probabilities')):
            left, right = _array(before[field]), _array(after[field])
            tolerance = plan['tolerances'][field]
            width = plan['vocab_size'] if field == 'logits' else 5
            equal = left.shape == right.shape == (width,) \
                and left.dtype == right.dtype == np.dtype('float32') \
                and np.allclose(left, right, **tolerance)
            gates[gate] &= bool(equal)
            error = float(np.max(np.abs(left-right))) if left.shape == right.shape and left.size else 0.
            if field == 'logits':
                max_logits = max(max_logits, error)
            else:
                max_probabilities = max(max_probabilities, error)
        query = prefix if route == REFERENCE or after['step'] == 0 else prefix[-1:]
        positions = list(range(len(prefix)-len(query), len(prefix)))
        gates['positions'] &= after['query_token_ids'] == query \
            and after['attention_mask'] == [1]*len(prefix) \
            and after['position_ids'] == positions and after['cache_position'] == positions
    coverage = {k:candidate['coverage'].get(k) is True for k in REQUIRED_COVERAGE}
    gates['coverage'] = all(coverage.values())
    if route == BATCH:
        coverage.update(actual_max_microbatch=candidate['coverage'].get('actual_max_microbatch'),
                        active_row_removal=candidate['coverage'].get('active_row_removal') is True)
        gates['coverage'] &= plan['coverage']['same_length_width'] >= plan['batch_microbatch'] \
            and candidate['coverage'].get('actual_max_microbatch') == plan['batch_microbatch'] \
            and candidate['coverage'].get('active_row_removal') is True
    else:
        gates['coverage'] &= plan['coverage']['has_decode']
    if route != REFERENCE:
        gates['coverage'] &= any(r['step'] > 0 for r in after_trace)
    return dict(status='PASS' if all(gates.values()) else 'FAILED_GATE', gates=gates,
                max_logit_abs_error=max_logits, max_topk_probability_abs_error=max_probabilities,
                coverage=coverage)


def compact_work_cost(bundle):
    """Only measured fixed-name scalars can enter qualification evidence."""
    work = {key:bundle['work'][key] for key in WORK_FIELDS}
    cost = {key:bundle['cost'][key] for key in COST_FIELDS}
    require(all(type(value) is int and value >= 0 for value in work.values())
            and work['physical_forward_calls'] > 0 and work['prefill_query_tokens'] > 0,
            'QUALIFICATION_MEASURED_WORK_REQUIRED')
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in cost.values()) and cost['peak_gpu_allocated_bytes'] > 0
            and cost['peak_gpu_reserved_bytes'] >= cost['peak_gpu_allocated_bytes']
            and cost['peak_host_RSS_bytes'] > 0, 'QUALIFICATION_MEASURED_COST_REQUIRED')
    require(all(type(cost[key]) is int for key in COST_FIELDS if key.endswith('_bytes')),
            'QUALIFICATION_MEASURED_MEMORY_INTEGER')
    return work, cost


def select_route(results):
    """Only completed fixed-route gates participate; absent proof is not PASS."""
    if set(results) != set(ROUTES):
        return None
    if results[SINGLETON].get('status')=='PASS':
        return BATCH if results[BATCH].get('status')=='PASS' else SINGLETON
    return REFERENCE if results[REFERENCE].get('status')=='PASS' else None


def validate_actual_receipt(receipt, plan):
    """Consumer seam for ROOT bridge/reuse/collector, with no GPU prerequisite."""
    require(receipt['schema'] == RECEIPT_SCHEMA
            and receipt['status'] == 'QUALIFIED_ACTUAL_GPU_ROUTE'
            and receipt['actual_GPU'] is True and plan.get('fixture_only') is False,
            'QUALIFICATION_ACTUAL_RECEIPT_REQUIRED')
    require(receipt['plan_sha256'] == digest(plan)
            and receipt['cohort_sha256'] == plan['cohort_sha256']
            and receipt['model_identity'] == plan['model_identity']
            and receipt['shared_source_sha'] == plan['shared_source_sha']
        and receipt['native_source_binding'] == plan['native_source_binding']
            and receipt.get('shared_plan_sha256') == plan['shared_plan_sha256']
            and receipt['tolerances'] == plan['tolerances'] == TOLERANCES,
            'QUALIFICATION_ACTUAL_RECEIPT_IDENTITY')
    route = receipt['selected_route']
    require(route == select_route(receipt['route_results']) and route in ROUTES
            and receipt['selected_route_passed'] is True
            and receipt['state_unchanged'] is True and receipt['RNG_restored'] is True
            and receipt['no_fit'] is True and receipt['checkpoint_saved'] is False
            and receipt['fixed_microbatch'] == (plan['batch_microbatch'] if route == BATCH else 1),
            'QUALIFICATION_ACTUAL_SELECTED_ROUTE_GUARDS')
    selected = receipt['route_results'][route]
    expected_gates = {'tokens', 'EOS', 'row_mapping', 'seed_stream', 'logits',
                      'topk_ids', 'topk_probabilities', 'metrics', 'positions', 'coverage'}
    require(set(selected['gates']) == expected_gates
            and all(value is True for value in selected['gates'].values())
            and all(type(selected[k]) in (int,float) and math.isfinite(selected[k])
                    and selected[k] >= 0 for k in
                    ('max_logit_abs_error', 'max_topk_probability_abs_error'))
            and all(selected['coverage'].get(k) is True for k in REQUIRED_COVERAGE),
            'QUALIFICATION_ACTUAL_MEASURED_GATES')
    if route == BATCH:
        require(selected['coverage'].get('actual_max_microbatch') == receipt['fixed_microbatch']
                and selected['coverage'].get('active_row_removal') is True,
                'QUALIFICATION_ACTUAL_BATCH_COVERAGE')
    require(set(receipt['work']) == set(receipt['cost']) == set(ROUTES),
            'QUALIFICATION_ACTUAL_THREE_ROUTE_COSTS')
    for measured_route in ROUTES:
        compact_work_cost(dict(work=receipt['work'][measured_route], cost=receipt['cost'][measured_route]))
    return True


def qualify_runtime(plan, cohort, model, measure, state_callback, receipt_path):
    """First-job GPU-only execution; exactly three calls and no retry/refit.

    ``measure`` is the root adapter to received SH1 code, NOT a shared API
    claimed by this module. It must evaluate actual GPU rows and return the
    transient proof contract documented above, including forced-prefix traces.
    """
    import torch
    from project.run_scripts.experiment_generation_eval.generator import rng_snapshot, rng_equal, rng_restore
    from project.run_scripts.experiment_generation_eval.observer import model_signature
    from project.run_scripts.experiment_generation_eval.common import immutable_write
    validate_plan(plan, cohort)
    require(plan.get('fixture_only') is False, 'QUALIFICATION_CPU_FIXTURE_NOT_GPU_PROOF')
    require(plan['shared_source_sha']==SHARED_SOURCE,'QUALIFICATION_SHARED_SOURCE_PIN')
    require(model.config.model_type == 'gptj' and not model.training
            and torch.cuda.is_initialized()
            and all(p.device.type == 'cuda' and p.dtype == torch.float32 for p in model.parameters()),
            'QUALIFICATION_ACTUAL_GPU_GPTJ_FP32_REQUIRED')
    require(installed_native_binding() == plan['native_source_binding'],
            'QUALIFICATION_INSTALLED_NATIVE_SOURCE_CHANGED')
    require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
            'QUALIFICATION_TF32_DISABLED_REQUIRED')
    before, external = model_signature(model), copy.deepcopy(state_callback())
    saved_rng = rng_snapshot()
    results, bundles, work, cost, error = {}, {}, {}, {}, None
    try:
        for route in ROUTES:
            microbatch = plan['batch_microbatch'] if route == BATCH else 1
            bundles[route] = measure(route=route, cohort=copy.deepcopy(cohort['prompts']),
                microbatch=microbatch, reference=bundles.get(REFERENCE))
            work[route], cost[route] = compact_work_cost(bundles[route])
            require(model_signature(model) == before and state_callback() == external,
                    'QUALIFICATION_MODEL_NATIVE_STATE_MUTATION')
            require(rng_equal(saved_rng), 'QUALIFICATION_OBSERVER_RNG_MUTATION')
            results[route] = compare_route(plan, cohort, bundles[REFERENCE], bundles[route], route)
    except Exception as caught:
        # A third-party exception can embed raw inputs. Preserve only public
        # stage/type, not its string or arbitrary exception arguments.
        error = dict(error_type=type(caught).__name__, stage=route,
                     reason='QUALIFICATION_MEASUREMENT_OR_GATE_EXCEPTION')
    finally:
        rng_restore(saved_rng)
    state_ok = model_signature(model) == before and state_callback() == external
    rng_ok = rng_equal(saved_rng)
    chosen = select_route(results)
    # A callback exception/state mutation is terminal; no fallback after OOM or
    # a partially measured route, and no additional invocation of any route.
    success = error is None and state_ok and rng_ok and chosen is not None
    receipt = dict(schema=RECEIPT_SCHEMA,
        status='QUALIFIED_ACTUAL_GPU_ROUTE' if success else 'FAILED_ACTUAL_GPU_QUALIFICATION',
        actual_GPU=True, plan_sha256=digest(plan), cohort_sha256=plan['cohort_sha256'],
        model_identity=plan['model_identity'], shared_source_sha=plan['shared_source_sha'],
        shared_plan_sha256=plan['shared_plan_sha256'],shared_route_map=copy.deepcopy(SHARED_ROUTES),
        native_source_binding=copy.deepcopy(plan['native_source_binding']),
        selected_route=chosen if success else None,
        fixed_microbatch=(plan['batch_microbatch'] if chosen == BATCH else 1) if success else None,
        route_results=results, selected_route_passed=success,
        state_unchanged=state_ok, RNG_restored=rng_ok, no_fit=True,
        checkpoint_saved=False, tolerances=copy.deepcopy(TOLERANCES),
        work=work, cost=cost, error=error,
        seed_stream_evidence='DEVICE_LOCAL_GENERATOR_REPLAY_NOT_INTERNAL_STATE_MEASUREMENT')
    # Qualification tensors are transient diagnostics only. Release reference
    # and cached/batch traces before durable scalar receipts or W0 generation.
    bundles.clear()
    if success:
        validate_actual_receipt(receipt, plan)
    immutable_write(receipt_path, receipt)
    require(success, 'QUALIFICATION_FAILED_NO_AUTOMATIC_RETRY')
    return receipt
