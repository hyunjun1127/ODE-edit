"""Prelocked local plan and one-shot native GPT2/GPT-J KV qualification.

CPU fixtures certify software plumbing only. The first replacement allocation
must produce its own immutable actual receipt; a plan is never actual PASS.
No prompt, token, case identity or trace tensor is a public tracking payload.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import time

import torch

from .common import EVAL_SEED, PROFILE, digest, immutable_write, require
from .generator import (REFERENCE_ROUTE, SINGLETON_ROUTE, BATCH_ROUTE, ALLOWED_ROUTES,
    _encoded_tokens, generate_rows, isolated_rng, model_device, rng_snapshot, rng_equal)
from .metrics import score_case

QUALIFICATION_SCHEMA = 'gpt2-gptj-kv-fixed8-v1'
TOLERANCES = dict(logits=dict(atol=2e-4, rtol=2e-4),
    topk_probabilities=dict(atol=2e-5, rtol=2e-4),
    metrics=dict(atol=1e-6, rtol=0.0))


def _ordinal(record, index):
    fields = [record[key] for key in ('ordered_occurrence', 'occurrence_index', 'occurrence', 'ordinal')
              if key in record]
    require(fields and all(isinstance(x, int) and not isinstance(x, bool) for x in fields)
        and len(set(fields)) == 1, 'GENERATION_QUALIFICATION_OCCURRENCE_REQUIRED')
    return fields[0]


def build_qualification_plan(tokenizer, records, *, model_identity, microbatch=8,
                             memory_alternative=False):
    """CPU token-only selection; fixed tolerances and up to eight local prompts.

    Diverse length/boundary rows are selected first. A full MB8 group does not
    fit beside diverse rows in eight slots: such a plan honestly cannot qualify
    production MB8 and selects singleton unless its actual cohort has width8.
    MB4 is allowed only as a predeclared memory alternative, never OOM retry.
    """
    require(microbatch in (4, 8), 'GENERATION_QUALIFICATION_MB')
    require(microbatch != 4 or memory_alternative is True,
            'GENERATION_MB4_MEMORY_ALTERNATIVE_REQUIRED')
    requests = []
    for index, record in enumerate(records):
        ordinal = _ordinal(record, index)
        rewrite = record.get('requested_rewrite', {})
        target = rewrite.get('target_new', {})
        require(isinstance(target, dict), 'GENERATION_QUALIFICATION_TARGET_SCHEMA')
        prompts = record.get('generation_prompts', [])
        require(isinstance(prompts, list) and all(isinstance(x, str) for x in prompts),
                'GENERATION_QUALIFICATION_PROMPT_SCHEMA')
        for prompt_index, prompt in enumerate(prompts):
            tokens = _encoded_tokens(tokenizer, prompt)
            requests.append(dict(prompt=prompt, occurrence=ordinal, prompt_index=prompt_index,
                model_identity=copy.deepcopy(model_identity), input_token_ids=tokens,
                token_length=len(tokens), record_index=index, relation_id=rewrite.get('relation_id'),
                target_new_id=target.get('id')))
    require(requests, 'GENERATION_QUALIFICATION_NO_PROMPTS')
    ordered = sorted(range(len(requests)), key=lambda i: (requests[i]['token_length'], i))
    below = [i for i in ordered if requests[i]['token_length'] < 100]
    above = [i for i in ordered if requests[i]['token_length'] >= 100]
    require(below, 'GENERATION_QUALIFICATION_NO_GENERATING_PROMPTS')
    roles = dict(shortest=ordered[0], longest=ordered[-1], closest_below100=below[-1])
    if above:
        roles['closest_at_or_above100'] = above[0]
    buckets = {}
    for index in below:
        buckets.setdefault(requests[index]['token_length'], []).append(index)
    group_length, group = min(buckets.items(), key=lambda item: (-len(item[1]), item[0]))
    selection = []
    for index in list(roles.values()) + group[:microbatch]:
        if index not in selection and len(selection) < 8:
            selection.append(index)
    for index in ordered:
        if index not in selection and len(selection) < 8:
            selection.append(index)
    chosen = [requests[i] for i in selection]
    widths = {length: sum(row['token_length'] == length for row in chosen)
              for length in {row['token_length'] for row in chosen}}
    generating_width = max((count for length, count in widths.items() if length < 100), default=0)
    coverage = dict(selection_rule='token_length_then_original_ordinal_only',
        selected_count=len(chosen), shortest=any(i == ordered[0] for i in selection),
        longest=any(i == ordered[-1] for i in selection),
        below100=any(i == below[-1] for i in selection),
        at_or_above100_available=bool(above),
        at_or_above100=any(i == above[0] for i in selection) if above else False,
        selected_equal_length_group=group_length, tested_max_batch_width=generating_width,
        required_batch_width=microbatch, batch_width_covered=generating_width >= microbatch,
        unavailable_native_boundary='CPU_fixture_only' if not above else None)
    return dict(schema=QUALIFICATION_SCHEMA, profile=PROFILE, eval_seed=EVAL_SEED,
        model_identity=copy.deepcopy(model_identity), requests=chosen, coverage=coverage,
        microbatch=microbatch, memory_alternative=memory_alternative,
        allowed_routes=list(ALLOWED_ROUTES), tolerances=copy.deepcopy(TOLERANCES),
        max_prompts=8, fit_calls=0, edit_commits=0, actual_qualification=False,
        raw_local_only=True, sampling='full_vocab_FP32_softmax_topk5_row_multinomial',
        fallback_order=[BATCH_ROUTE, SINGLETON_ROUTE, REFERENCE_ROUTE],
        no_oom_retry=True)


def _assert_plan(plan):
    require(plan.get('schema') == QUALIFICATION_SCHEMA and plan.get('profile') == PROFILE
        and plan.get('eval_seed') == EVAL_SEED and plan.get('tolerances') == TOLERANCES
        and plan.get('allowed_routes') == list(ALLOWED_ROUTES)
        and 0 < len(plan.get('requests', [])) <= 8 and plan.get('actual_qualification') is False,
        'GENERATION_QUALIFICATION_PLAN_CHANGED')
    require(plan.get('microbatch') in (4, 8)
        and (plan['microbatch'] != 4 or plan.get('memory_alternative') is True),
        'GENERATION_QUALIFICATION_MB_PLAN')


def _metric_equal(a, b):
    if set(a) != set(b):
        return False
    for key in a:
        if isinstance(a[key], float) and isinstance(b[key], (int, float)):
            if not math.isclose(a[key], b[key], abs_tol=1e-6, rel_tol=0):
                return False
        elif a[key] != b[key]:
            return False
    return True


def _compare(reference, candidate, reference_trace, candidate_trace, metrics_a, metrics_b):
    trace_keys = ('input_ids', 'attention_mask', 'topk_ids', 'sampled_token')
    exact = ('seed', 'occurrence', 'prompt_index', 'input_token_ids', 'continuation_token_ids',
             'full_token_ids', 'stop_reason', 'eos_ids', 'input_token_count',
             'continuation_token_count', 'text')
    token_equal = len(reference) == len(candidate) and all(
        all(a[key] == b[key] for key in exact) for a, b in zip(reference, candidate))
    keys = set(reference_trace) | set(candidate_trace)
    trace_equal = set(reference_trace) == set(candidate_trace)
    logits_close = probabilities_close = trace_equal
    max_logit = max_probability = 0.0
    for key in keys:
        aa, bb = reference_trace.get(key, []), candidate_trace.get(key, [])
        if len(aa) != len(bb):
            trace_equal = logits_close = probabilities_close = False
            continue
        for a, b in zip(aa, bb):
            for field in trace_keys:
                va, vb = a[field], b[field]
                trace_equal = trace_equal and (torch.equal(va, vb) if torch.is_tensor(va) else va == vb)
            # Effective full attention positions are native arange for both routes;
            # cached query positions are the suffix of the forced full prefix.
            full_positions = a['position_ids']
            query_positions = b['position_ids']
            trace_equal = trace_equal and torch.equal(full_positions[:, -query_positions.shape[1]:],
                                                       query_positions)
            for field, tol, which in [('logits', TOLERANCES['logits'], 'logits'),
                                     ('topk_probabilities', TOLERANCES['topk_probabilities'], 'probs')]:
                va, vb = a[field], b[field]
                close = torch.allclose(va, vb, **tol)
                if va.shape == vb.shape:
                    error = float((va-vb).abs().max())
                else:
                    error = float('inf')
                if which == 'logits':
                    logits_close = logits_close and close
                    max_logit = max(max_logit, error)
                else:
                    probabilities_close = probabilities_close and close
                    max_probability = max(max_probability, error)
    metrics_equal = len(metrics_a) == len(metrics_b) and all(
        _metric_equal(a, b) for a, b in zip(metrics_a, metrics_b))
    return dict(token_sequence_exact=bool(token_equal), topk_mask_position_exact=bool(trace_equal),
        logits_close=bool(logits_close), topk_probabilities_close=bool(probabilities_close),
        metric_values_close_and_validity_reasons_exact=bool(metrics_equal),
        max_abs_logit_error=max_logit, max_abs_topk_probability_error=max_probability,
        passed=bool(token_equal and trace_equal and logits_close and probabilities_close and metrics_equal))


def run_qualification(model, tokenizer, assets, plan, *, out, state_callback=None,
                      source_identity=None):
    """One reference, one singleton, then one fixed batch pass (no retry).

    This function loads no model, writes no tensor and performs zero fits. The
    caller supplies its already cold model and writes/freeze the plan beforehand.
    """
    from .observer import model_signature
    _assert_plan(plan)
    before = model_signature(model)
    external = copy.deepcopy(state_callback()) if state_callback else None
    saved_rng = rng_snapshot()
    requests = plan['requests']
    for request in requests:
        require(_encoded_tokens(tokenizer, request['prompt']) == request['input_token_ids'],
                'GENERATION_QUALIFICATION_TOKEN_BINDING')
        require(request['model_identity'] == plan['model_identity'],
                'GENERATION_QUALIFICATION_MODEL_BINDING')
    device = model_device(model)
    traces, results, costs = {}, {}, {}
    def metrics(rows):
        output = []
        for request, row in zip(requests, rows):
            refs = None if assets is None else assets.snippets_for(request['relation_id'], request['target_new_id'])
            output.append(score_case([row], refs, getattr(assets, 'vectorizer', None),
                                     getattr(assets, 'word_tokenize', None)))
        return output
    def execute(route, microbatch, forced=None):
        captured = {}
        def trace(value):
            captured.setdefault((value['occurrence'], value['prompt_index']), []).append(value)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
        start = time.perf_counter()
        rows = generate_rows(model, tokenizer, requests, route=route, microbatch=microbatch,
                             trace=trace, forced_prefixes=forced)
        if device.type == 'cuda':
            stop_event.record()
            torch.cuda.synchronize(device)
            gpu_seconds = start_event.elapsed_time(stop_event)/1000
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        else:
            gpu_seconds = peak_allocated = peak_reserved = None
        costs[route] = dict(elapsed_sec=time.perf_counter()-start, GPU_seconds=gpu_seconds,
            peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
            host_max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            prompt_count=len(rows), generated_tokens=sum(x['continuation_token_count'] for x in rows),
            logical_row_forward_decisions=sum(x['model_forwards'] for x in rows),
            physical_forward_calls=sum(x['physical_forward_calls'] for x in rows),
            prefill_query_tokens=sum(x['prefill_query_tokens'] for x in rows),
            decode_query_tokens=sum(x['decode_query_tokens'] for x in rows),
            full_prefix_query_tokens=sum(x['full_prefix_token_work'] for x in rows))
        costs[route]['native_cache_classes'] = sorted({x.get('cache_class') for x in rows
                                                     if x.get('cache_class') is not None})
        costs[route]['cache_position_explicit'] = sorted({x.get('cache_position_explicit', False) for x in rows})
        require(model_signature(model) == before, 'GENERATION_QUALIFICATION_MODEL_MUTATION')
        require((state_callback() if state_callback else None) == external,
                'GENERATION_QUALIFICATION_NATIVE_STATE_MUTATION')
        require(rng_equal(saved_rng), 'GENERATION_QUALIFICATION_RNG_MUTATION')
        traces[route] = captured
        return rows, metrics(rows)
    # An OOM/IO/native exception is not silently retried on a different route.
    with isolated_rng():
        reference, ref_metrics = execute(REFERENCE_ROUTE, 1)
        results[REFERENCE_ROUTE] = dict(passed=True, actual_prompt_count=len(reference),
            forced_prefix=False, no_mutation=True)
        forced = [row['continuation_token_ids'] for row in reference]
        singleton, singleton_metrics = execute(SINGLETON_ROUTE, 1, forced)
        results[SINGLETON_ROUTE] = _compare(reference, singleton, traces[REFERENCE_ROUTE],
            traces[SINGLETON_ROUTE], ref_metrics, singleton_metrics)
        selected_route, selected_mb = REFERENCE_ROUTE, 1
        if results[SINGLETON_ROUTE]['passed']:
            selected_route = SINGLETON_ROUTE
            batch, batch_metrics = execute(BATCH_ROUTE, plan['microbatch'], forced)
            result = _compare(reference, batch, traces[REFERENCE_ROUTE], traces[BATCH_ROUTE],
                              ref_metrics, batch_metrics)
            result.update(actual_max_batch_width=plan['coverage']['tested_max_batch_width'],
                required_batch_width=plan['microbatch'],
                batch_width_covered=plan['coverage']['batch_width_covered'])
            result['passed'] = result['passed'] and result['batch_width_covered']
            if not result['batch_width_covered']:
                result['failure_reason'] = 'ACTUAL_BATCH_WIDTH_UNVERIFIED'
            results[BATCH_ROUTE] = result
            if result['passed']:
                selected_route, selected_mb = BATCH_ROUTE, plan['microbatch']
        else:
            results[BATCH_ROUTE] = dict(passed=False, executed=False,
                failure_reason='CACHED_SINGLETON_GATE_FAILED_FIRST')
    require(model_signature(model) == before and rng_equal(saved_rng),
            'GENERATION_QUALIFICATION_FINAL_STATE')
    receipt = dict(schema=QUALIFICATION_SCHEMA, actual_qualification=True,
        qualification_pass=True, selected_route=selected_route, microbatch=selected_mb,
        fixed_microbatch=selected_mb, plan_sha256=digest(plan), model_identity=plan['model_identity'],
        source_identity=source_identity, profile=PROFILE, eval_seed=EVAL_SEED,
        tolerances=TOLERANCES, coverage=plan['coverage'], route_results=results, cost=costs,
        model_no_mutation=True, RNG_restored=True, native_state_no_mutation=True,
        fit_calls=0, edit_commits=0, retry_count=0, raw_local_only=True,
        selection_reason='FIXED_GATE_ORDER_NOT_SCIENTIFIC_QUALITY',
        native_models_scope=['gpt2', 'gptj'], pretrained_GPU_PASS=device.type == 'cuda')
    receipt['identity_sha256'] = digest(receipt)
    path = Path(out)/'qualification-actual.json'
    immutable_write(path, receipt)
    data = path.read_bytes()
    return dict(receipt, member=dict(path=str(path.resolve()), bytes=len(data),
                                    sha256=hashlib.sha256(data).hexdigest()))


def verify_actual_receipt(member_or_path, *, expected_plan_sha256=None,
                          expected_model_identity=None, allow_cpu_fixture=False):
    member = member_or_path if isinstance(member_or_path, dict) else dict(path=str(member_or_path))
    path = Path(member['path'])
    data = path.read_bytes()
    if 'bytes' in member:
        require(len(data) == member['bytes'], 'GENERATION_QUALIFICATION_RECEIPT_SIZE')
    if 'sha256' in member:
        require(hashlib.sha256(data).hexdigest() == member['sha256'],
                'GENERATION_QUALIFICATION_RECEIPT_SHA')
    receipt = json.loads(data)
    require(receipt.get('schema') == QUALIFICATION_SCHEMA
        and receipt.get('identity_sha256') == digest({k:v for k,v in receipt.items() if k != 'identity_sha256'})
        and receipt.get('actual_qualification') is True and receipt.get('qualification_pass') is True
        and receipt.get('selected_route') in ALLOWED_ROUTES
        and receipt.get('fixed_microbatch') == receipt.get('microbatch')
        and all(receipt.get(key) is True for key in ('model_no_mutation', 'RNG_restored', 'native_state_no_mutation'))
        and receipt.get('tolerances') == TOLERANCES, 'GENERATION_ACTUAL_QUALIFICATION_REQUIRED')
    require(receipt.get('pretrained_GPU_PASS') is True or allow_cpu_fixture is True,
            'GENERATION_NATIVE_GPU_QUALIFICATION_REQUIRED')
    route = receipt['selected_route']
    require(receipt['route_results'][route]['passed'] is True,
            'GENERATION_SELECTED_ROUTE_NOT_QUALIFIED')
    require(receipt['microbatch'] in (4, 8) if route == BATCH_ROUTE else receipt['microbatch'] == 1,
            'GENERATION_SELECTED_MB_NOT_QUALIFIED')
    if expected_plan_sha256 is not None:
        require(receipt['plan_sha256'] == expected_plan_sha256, 'GENERATION_QUALIFICATION_PLAN_SHA')
    if expected_model_identity is not None:
        require(receipt['model_identity'] == expected_model_identity, 'GENERATION_QUALIFICATION_MODEL_IDENTITY')
    return receipt
