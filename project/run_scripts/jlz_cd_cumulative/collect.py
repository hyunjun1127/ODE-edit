"""CPU-only independent scalar/raw reduction; no model/tokenizer/GPU imports.

The stdlib review primitives are independent of the production observer/fit.
Stored hashes verify receipts, not a replay of missing W/H/activation tensors.
"""
import argparse
import csv
import json
import math
import os
import pwd
import subprocess
import time
import traceback
from collections import Counter
from pathlib import Path
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader, validate_rows as validate_scalar_rows, reduce_rows, harmonic,
    compare_summary, paired, active_flags, select_rows, quantiles)
from .common import *


def number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label)
    return value


def aggregate(values):
    defined = [number(v, 'NONFINITE_TELEMETRY') for v in values if v is not None]
    return dict(count=len(defined), undefined=len(values) - len(defined),
        mean=math.fsum(defined) / len(defined) if defined else None, quantiles=quantiles(defined))


def endpoint(reader, folder, identities, ids, name, state_value, seen):
    paths = sorted(Path(folder).glob('chunk-*.json'))
    if not paths:
        return None, None
    rows = []
    for path in paths:
        chunk = reader.json(path)
        require(chunk['state'] == state_value and chunk['optimizer_feedback'] is False, 'OBSERVER_CHUNK_STATE')
        rows.extend(chunk['rows'])
    validate_scalar_rows(rows, expected_rows(identities, ids), name)
    result = reduce_rows(rows)
    saved = reader.json(Path(folder) / 'summary.json')
    require(saved['state'] == state_value and saved['endpoint'] == name and saved['requests'] == len(ids), 'ENDPOINT_STATE_DENOMINATOR')
    require(saved['no_mutation'] and saved['optimizer_feedback'] is False, 'OBSERVER_RECEIPT')
    compare_summary(result, saved['summary'])
    require(saved.get('row_count', len(rows)) == len(rows), 'ENDPOINT_ROW_COUNT')
    require(saved.get('row_order', digest([r['identity'] for r in rows])) == digest([r['identity'] for r in rows]), 'ENDPOINT_ROW_ORDER')
    flags = active_flags(seen)
    require(all(r['active_at_endpoint'] == flags[r['case_id']] for r in rows), 'SEEN_PREFIX_ACTIVE')
    return rows, result


def validate_fit(fit, events, ids, layers, source, arm, batch, lambda_identity):
    B = len(ids); c = fit['terminal_candidate']; alpha = ALPHA[arm]
    require(type(c) is int and 0 <= c <= 24 and fit['schema'] == 'JLZ_CD_CUMULATIVE_FIT_V1', 'FIT_SCHEMA_CANDIDATE')
    require(fit['requests'] == B and fit['logical_evaluations'] == c + 1
            and fit['request_evaluations'] == B * (c + 1) and fit['optimizer_updates'] == c
            and fit['request_updates'] == B * c, 'FIT_COUNT_UNITS')
    require(fit['terminal_extra_forward'] == fit['terminal_extra_backward'] == 0
            and fit['norm_gradient_additions'] == fit['allocation_gradient_additions'] == c,
            'TERMINAL_OR_GRADIENT_ADDITION_COUNT')
    require(fit['synchronous_candidate'] and fit['independent_request_freeze'] is False
            and not fit['checkpoint_saved'] and fit['alpha'] == alpha, 'FIT_SYNC_POLICY')
    require(fit['lambda_identity'] == lambda_identity, 'FIT_SHARED_LAMBDA')
    require(set(fit['candidates']) == set(map(str, ids)) and all(p['candidate'] == c for p in fit['candidates'].values()), 'COMMON_TERMINAL')
    trace = fit['candidate_trace']
    require([p['candidate'] for p in trace] == list(range(c + 1)), 'COMMON_CANDIDATE_SEQUENCE')
    for i, p in enumerate(trace):
        finite_tree(p)
        require(p['logical_evaluation'] == i + 1 and p['updates_completed'] == i and p['alpha'] == alpha
                and len(p['native_J']) == B and all(type(v) in (int, float) for v in p['native_J']), 'CANDIDATE_IDENTITY')
        require(math.isclose(p['native_sum'], math.fsum(p['native_J']), rel_tol=1e-12, abs_tol=1e-10), 'CANDIDATE_SUM_MEAN')
        require(set(p['costs']) == set(map(str, layers)), 'CANDIDATE_COST_LAYER_SET')
        for cost in p['costs'].values():
            for name in ('Q', 'c', 'cumulative', 'Pi', 'weighted_cost'):
                number(cost[name], 'NONFINITE_ALLOCATION_COST')
        risk = math.fsum(v['weighted_cost'] for v in p['costs'].values())
        require(math.isclose(p['objective_mean'], (p['native_sum'] + risk) / B, rel_tol=1e-12, abs_tol=1e-10), 'REPORT_MEAN_NOT_GRADIENT_SCALE')
        reason = p['common_terminal_reason']
        require((reason is not None) == (i == c) and p['norm_added_once'] == p['allocation_added_once'] == (i < c), 'COMMON_TERMINAL_BACKWARD')
        threshold = all(v < .05 for v in p['native_J'])
        if i < c:
            require(not threshold, 'MISSING_COMMON_STOP')
        else:
            expected_reason = ('ZERO_STEP' if i == 0 else 'OBJECTIVE_THRESHOLD') if threshold else 'EVALUATION_BUDGET'
            require(reason == expected_reason and (threshold or i == 24), 'NATIVE_ONLY_COMMON_STOP')
    counts = Counter(); requests = {}; layer_events = Counter(); updates = Counter(); terminals = {}; coupled = []
    for event in events:
        require(event['arm'] == arm and event['batch'] == batch, 'EVENT_ARM_BATCH')
        finite_tree(event['payload'])
        k, i, l, p = event['event'], event.get('candidate'), event.get('layer'), event['payload']
        require(type(i) is int and 0 <= i <= c, 'EVENT_CANDIDATE_RANGE')
        key = (str(event.get('request')), i)
        if k == 'coupled_candidate':
            coupled.append(i); require(p == trace[i], 'TRACE_EVENT_IDENTITY')
        elif k == 'candidate_request':
            require(key not in requests and key[0] in set(map(str, ids)), 'REQUEST_EVENT_UNIQUE_IDENTITY')
            require(p['evaluation_ordinal'] == i + 1 and p['updates_completed'] == i
                    and p['will_backward'] == (i < c) and p['common_candidate']
                    and p['permanent_request_freeze'] is False, 'REQUEST_EVENT_SYNC')
            require(all(type(p[name]) in (int, float) for name in ('nll', 'kl', 'F', 'J', 'price', 'budget'))
                    and p['price'] > 0 and 0 <= p['budget'] <= .75 + 1e-6, 'REQUEST_NUMERIC_BUDGET')
            require(len(p['norms']) == len(layers) and math.isclose(math.fsum(p['norms']), p['budget'], rel_tol=1e-12, abs_tol=1e-10), 'REQUEST_NORM_SUM')
            require(math.isclose(p['F'], p['nll'] + .0625 * p['kl'], rel_tol=1e-12, abs_tol=1e-10)
                    and math.isclose(p['J'], p['F'] + p['price'] * p['budget'], rel_tol=1e-12, abs_tol=1e-10)
                    and math.isclose(p['J'], trace[i]['native_J'][ids.index(int(key[0]))], rel_tol=1e-12, abs_tol=1e-10), 'REQUEST_NATIVE_OBJECTIVE')
            requests[key] = p
        elif k == 'candidate_layer':
            require(str(l) in set(map(str, layers)), 'EVENT_LAYER_IDENTITY'); layer_events[(key, str(l))] += 1
            require((p['gradients'] is None) == (i == c), 'TERMINAL_GRADIENT_NULL')
        elif k == 'optimizer_layer':
            require(str(l) in set(map(str, layers)) and i < c, 'TERMINAL_OPTIMIZER_FORBIDDEN'); updates[(key, str(l))] += 1
            require(p['update'] == i + 1 and p['stored_budget'] <= .75 + 1e-6 and p['violation'] <= 1e-6
                    and p['moments_preserved_after_projection'] and p['eligible_for_reentry'], 'OPTIMIZER_COUNTER_BUDGET')
        elif k == 'terminal_request':
            require(key[0] not in terminals and i == c and p['accepted_candidate_index'] == c
                    and p['logical_evaluations'] == c + 1 and p['optimizer_updates'] == p['backward_calls'] == c
                    and p['gradient_kkt_status'] == 'NO_BACKWARD_TERMINAL', 'TERMINAL_EVENT_COUNT')
            terminals[key[0]] = p
        else:
            raise RuntimeError('UNKNOWN_FIT_EVENT:' + k)
        counts[k] += 1
    require(coupled == list(range(c + 1)) and set(terminals) == set(map(str, ids)), 'EVENT_CANDIDATE_TERMINAL_COVERAGE')
    require(set(requests) == {(str(r), i) for r in ids for i in range(c + 1)}, 'REQUEST_EVENT_COVERAGE')
    expected_layers = {((str(r), i), str(l)) for r in ids for i in range(c + 1) for l in layers}
    require(set(layer_events) == expected_layers and all(v == 1 for v in layer_events.values()), 'CANDIDATE_LAYER_EXACT_COVERAGE')
    expected_updates = {((str(r), i), str(l)) for r in ids for i in range(c) for l in layers}
    require(set(updates) == expected_updates and all(v == 1 for v in updates.values()), 'OPTIMIZER_LAYER_EXACT_COVERAGE')
    return dict(requests=B, candidates=c + 1, request_evaluations=B * (c + 1),
        updates=c, request_updates=B * c, events=dict(counts), terminal_no_backward=True,
        validation_scope='stored scalar/JSONL relations; not a recomputation of model gradients')


def validate_commit(commit, writer, entry, expected, previous, source, config_hash, arm, batch, layers, lambda_identity):
    require(commit['source'] == entry['source'] == source and commit['config'] == entry['config'] == config_hash, 'COMMIT_SOURCE_CONFIG')
    require(commit['arm'] == entry['arm'] == arm and commit['batch'] == entry['batch'] == batch, 'COMMIT_ARM_BATCH')
    require(commit['ids'] == entry['ids'] == expected['ids'] and commit['native_pack'] == entry['native_pack'] == expected['identity'], 'COMMIT_INPUT_ORDER')
    require(commit['before'] == entry['state'] == previous, 'OWN_W_H_JOIN')
    require(commit['RNG_before'] == entry['RNG'] == commit['RNG_after'], 'OWN_RNG_JOIN')
    require(commit['lambda_identity'] == lambda_identity and commit['fit_count'] == 1
            and commit['observer_no_mutation'] and not commit['checkpoint_saved'], 'COMMIT_POLICY')
    history = writer['history']
    require(commit['history_appends'] == writer['history_appends'] == len(history) == len(layers)
            and len({r['layer'] for r in history}) == len(layers) and {r['layer'] for r in history} == set(layers), 'HISTORY_EXACT_ONCE')
    for row in history:
        l = str(row['layer']); saved = writer['layers'][l]
        require(row['append_count'] == 1 and row['columns'] == len(expected['ids']) and row['rewrite_only']
                and row['KL_in_history'] is False and row['CPU_FP32'], 'NATIVE_REWRITE_HISTORY_SEMANTICS')
        require(row['before'] == previous['H'][l] and row['after'] == commit['after']['H'][l], 'HISTORY_STATE_LINK')
        require(saved['weight_after'] == commit['after']['W'][l] and saved['solver']['numerical_projection_verified']
                and saved['ideal_effective_parity']['pass_'], 'WRITER_EXACT_COMMIT_PARITY_RECEIPT')
        if 'row_map' in expected:
            require(saved['solver']['row_order'] == digest([(r['global_row'], r['owner'], r['kind'], r['lookup'])
                    for r in expected['row_map']]), 'SOLVER_ROW_ORDER_OWNER_LOOKUP')
    return commit['after']


def validate_actions(actions, pack, layers, arm):
    rows = actions['rows']; q = len(pack['row_kind']) if 'row_kind' in pack else pack['native_rows']
    require(len(rows) == q * len(layers), 'ACTION_EXACT_COUNT')
    require({(r['layer'], r['global_row']) for r in rows} == {(l, j) for l in layers for j in range(q)}, 'ACTION_EXACT_LAYER_ROW_SET')
    for row in rows:
        require(row.get('branch', arm) == arm and 0 <= row['owner'] < len(pack['ids'])
                and row['case_id'] == pack['ids'][row['owner']], 'ACTION_OWNER_CASE_IDENTITY')
        require(row.get('reference') == 'ACTUAL_FIT_PROJECTED_Y', 'ACTION_FIT_Y_REFERENCE')
        if 'row_map' in pack:
            expected = pack['row_map'][row['global_row']]
            require(all(row[field] == expected[field] for field in ('owner', 'kind', 'lookup', 'canonical')), 'ACTION_EXACT_ROLE_LOOKUP_CANONICAL')
        require(row['kind'] in ('rewrite', 'kl'), 'ACTION_ROLE'); finite_tree(row)
    result = []
    for l in layers:
        for scope in ('rewrite', 'kl', 'canonical'):
            subset = [r for r in rows if r['layer'] == l and (r['canonical'] if scope == 'canonical' else r['kind'] == scope)]
            item = dict(layer=l, scope=scope, rows=len(subset))
            for field in ('directional_ratio', 'norm_ratio', 'cosine', 'relative_error', 'error_norm'):
                item[field] = aggregate([r['actual'].get(field) for r in subset])
            for field in ('inherited_gap_norm', 'direct_fit_error_norm', 'direct_error_norm', 'final_virtual_gap_norm'):
                if any(field in r for r in subset): item[field] = aggregate([r.get(field) for r in subset])
            result.append(item)
    return result


def read_events(reader, path):
    return [json.loads(line) for line in reader.bytes(path).decode().splitlines() if line]


def reduce_arm(reader, attempt, arm, c, lock, identities, records, calibration):
    root = attempt / ('main-' + arm); layers = c['profile']['eligible_layers']; cold = c.get('cold_W0_H0', c['qualification_reuse']['cold_W0_H0'])
    result = dict(status='PARTIAL_OR_TECHNICAL_BLOCKED', endpoints={}, coverage={}, warnings=[], commits=0,
        next_entry_links=0, history_appends=0, fit_count=0, request_evaluations=0, request_updates=0,
        paired={}, telemetry=[], source=lock['source_commit'], arm=arm)
    raw_endpoints = {}; previous = cold; previous_commit = None; atwrite = []
    raw_w0, summary_w0 = endpoint(reader, root / 'W0', identities, [r['case_id'] for r in records], 'W0', cold, records)
    if raw_w0 is None:
        result['warnings'].append(dict(type='MISSING_W0', error='Cold first2000 W0 rows absent'))
        return result, raw_endpoints
    result['endpoints']['W0'] = summary_w0; result['coverage']['W0'] = 'COMPLETE'; raw_endpoints['W0'] = raw_w0
    require(not (root / 'batch-21').exists(), 'FORBIDDEN_B21')
    lambda_identity = None if calibration is None else calibration['receipt_hash']
    for number, current, seen in batches(records, c['settings']['B']):
        folder = root / f'batch-{number:02d}'; label = f'W{number}'
        if not (folder / 'commit.json').is_file() or (folder / 'rollback.json').exists():
            result['coverage'][label] = 'NOT_COMMITTED'
            if any((root / f'batch-{n:02d}/commit.json').exists() for n in range(number + 1, 21)):
                result['warnings'].append(dict(batch=number, type='NONCONTIGUOUS_COMMIT_PREFIX'))
            break
        try:
            commit = reader.json(folder / 'commit.json'); entry = reader.json(folder / 'entry.json'); writer = reader.bound(commit['writer'])
            if previous_commit is not None:
                require(entry['RNG'] == previous_commit['RNG_after'] and entry['context_hash'] == previous_commit['context_hash'], 'NEXT_ENTRY_RNG_CONTEXT')
            after = validate_commit(commit, writer, entry, c['packs'][number - 1], previous, lock['source_commit'], digest(c), arm, number, layers, lambda_identity)
            selected = selected_for_post(current, seen, number); ids = [r['case_id'] for r in current]
            pre, pre_summary = endpoint(reader, folder / 'pre', identities, ids, f'B{number}_PRE', previous, seen)
            post, post_summary = endpoint(reader, folder / 'post', identities, [r['case_id'] for r in selected], label, after, seen)
            require(pre is not None and post is not None, 'COMMITTED_EVALUATION_COVERAGE')
            compare_summary(pre_summary, commit['pre']); compare_summary(post_summary, commit['post'])
            post_current = select_rows(post, ids); compare_summary(reduce_rows(post_current), commit['post_current'])
            fit = reader.json(folder / 'fit/fit.json')
            counts = validate_fit(fit, read_events(reader, folder / 'fit-events.jsonl'), ids, layers, lock['source_commit'], arm, number, lambda_identity)
            actions = reader.json(folder / 'writer/actions.json'); telemetry = validate_actions(actions, c['packs'][number - 1], layers, arm)
            local = reader.json(folder / 'writer/local-additivity.json')
            require(set(local['checks']) == set(map(str, layers)) and all(r['pass_'] for r in local['checks'].values()), 'LOCAL_ADDITIVITY_RECEIPT')
            planned_path = folder / 'writer/planned-cost.json'
            planned = reader.json(planned_path) if planned_path.exists() else 'NOT_RECORDED'
            if planned != 'NOT_RECORDED': finite_tree(planned)
            result['telemetry'].append(dict(batch=number, actions=telemetry, mean=actions['mean'],
                writer_layers=writer['layers'], planned_cost=planned, shares=reader.json(folder / 'writer/shares.json'),
                fit_candidate_trace=fit['candidate_trace'], fit_seconds=fit.get('seconds'), writer_seconds=writer.get('seconds'),
                batch_seconds=commit.get('seconds'), nested_times_not_added=True))
            previous = after; previous_commit = commit; result['commits'] += 1
            result['history_appends'] += len(layers); result['fit_count'] += 1
            result['request_evaluations'] += counts['request_evaluations']; result['request_updates'] += counts['request_updates']
            result['next_entry_links'] = max(0, result['commits'] - 1)
            result['endpoints'][label] = post_summary; result['coverage'][label] = 'COMPLETE'
            result['paired'][f'B{number}_pre_post'] = paired(pre, post_current); atwrite.extend(post_current)
            if number in MILESTONES:
                raw_endpoints[label] = post; seen_ids = [r['case_id'] for r in seen]
                record = dict(atwrite_to_endpoint=paired(select_rows(atwrite, seen_ids), post),
                    W0_to_endpoint=paired(select_rows(raw_w0, seen_ids), post), cohorts={}, first_prefix={})
                for cohort, born, _ in batches(seen, c['settings']['B']):
                    born_ids = [r['case_id'] for r in born]
                    record['cohorts'][str(cohort)] = paired(select_rows(atwrite, born_ids), select_rows(post, born_ids))
                for size in (100, 500, 1000, 1500):
                    if size <= len(seen):
                        subset_ids = [r['case_id'] for r in seen[:size]]; subset = select_rows(post, subset_ids)
                        record['first_prefix'][str(size)] = dict(summary=reduce_rows(subset), atwrite=paired(select_rows(atwrite, subset_ids), subset))
                record['active_summary'] = reduce_rows([r for r in post if r['active_at_endpoint']])
                superseded = [r for r in post if not r['active_at_endpoint']]
                record['superseded_summary'] = reduce_rows(superseded) if superseded else 'EMPTY_POPULATION'
                result['paired'][label] = record
        except Exception as error:
            result['warnings'].append(dict(batch=number, type=type(error).__name__, error=str(error)))
            result['coverage'][label] = 'TECHNICAL_REDUCER_MISMATCH'; break
    for n in range(1, 21): result['coverage'].setdefault(f'W{n}', 'NOT_COMMITTED')
    terminal_path = root / 'terminal.json'; terminal = reader.json(terminal_path) if terminal_path.exists() else dict(status='NOT_RECORDED')
    result['terminal'] = terminal
    if terminal_path.exists():
        require(terminal['source'] == lock['source_commit'] and terminal['arm'] == arm and terminal['commits'] == result['commits'], 'TERMINAL_SOURCE_COUNTS')
    complete = result['commits'] == 20 and result['history_appends'] == 100 and not result['warnings']
    if complete and terminal['status'] == 'W20_COMPLETE': result['status'] = 'W20_COMPLETE'
    result['harmonic'] = {label: harmonic(summary) for label, summary in result['endpoints'].items()}
    result['noCP'] = True; result['exact_resume'] = 'NOT_AVAILABLE'
    result['verification_scope'] = 'Independent scalar/raw arithmetic and stored identities/relations, not tensor/model replay'
    return result, raw_endpoints


def calibration_record(reader, attempt, c, lock):
    path = attempt / 'calibration.json'
    if not path.exists(): return None
    data = reader.json(path); core = {k: v for k, v in data.items() if k != 'receipt_hash'}
    require(data['schema'] == 'JLZ_CD_CUMULATIVE_LAMBDA_LOCK_V1' and data['receipt_hash'] == digest(core), 'SHARED_LAMBDA_RECEIPT_HASH')
    require(data['shared_arms'] == list(ARMS) and data['immutable'] and data['persisted_tensors'] is False
            and data['actual_delta_zero'] and data['projected_action_nonzero'], 'SHARED_LAMBDA_POLICY')
    value = number(data['lambda_alloc'], 'LAMBDA_NONFINITE'); require(value > 0, 'LAMBDA_NONPOSITIVE')
    require(math.isclose(value, data['norm_gradient_norm'] / data['Q_gradient_norm'], rel_tol=1e-12, abs_tol=0), 'LAMBDA_GRADIENT_RATIO')
    identity = data['identity']
    require(identity.get('source') == lock['source_commit'], 'LAMBDA_SOURCE')
    require(identity.get('config', identity.get('config_digest')) == digest(c), 'LAMBDA_CONFIG')
    require(identity.get('ordered_case_ids') == c['ordered_ids_sha256']
            and identity.get('B1_pack') == c['packs'][0]['identity']
            and identity.get('cold') == c['qualification_reuse']['cold_W0_H0'], 'LAMBDA_ORDER_PACK_COLD_STATE')
    return data


def parse_accounting(text, jobs, owner):
    """Exact parent records only; steps cannot duplicate allocated GPU time."""
    wanted = {str(v) for v in jobs.values()}; result = {}; ignored = []
    for line in text.splitlines():
        fields = line.split('|')
        if len(fields) < 6: continue
        job, user, name, state_value, elapsed, tres = fields[:6]
        if job not in wanted:
            ignored.append(job); continue
        require(job not in result and user == owner, 'ACCOUNTING_EXACT_PARENT_OWNER')
        require(elapsed.isdigit(), 'ACCOUNTING_ELAPSED')
        resources = dict(part.split('=', 1) for part in tres.split(',') if '=' in part)
        gpu = int(resources.get('gres/gpu', 0))
        role = next(k for k, v in jobs.items() if str(v) == job)
        if role in ARMS:
            require(gpu == 1, 'ACCOUNTING_ONE_GPU_ARM')
        elif role.lower() in ('collector', 'cpu', 'cpu_collector'):
            require(gpu == 0, 'ACCOUNTING_CPU_COLLECTOR_GPU_ZERO')
        else:
            raise RuntimeError('UNKNOWN_REGISTERED_JOB_ROLE:' + role)
        result[job] = dict(job=job, owner=user, name=name, state=state_value,
            elapsed_seconds=int(elapsed), allocated_GPU=gpu, allocated_GPU_seconds=gpu * int(elapsed),
            AllocTRES=tres, parent_once=True)
    return dict(status='COMPLETE_PARENT_RECORDS' if set(result) == wanted else 'PARTIAL_ACCOUNTING',
        parents=list(result.values()), missing_parents=sorted(wanted - set(result)), ignored_step_or_other_rows=ignored,
        allocated_GPU_seconds=math.fsum(r['allocated_GPU_seconds'] for r in result.values()),
        missing_rows_are_not_zero=True)


def sealed_accounting(reader, attempt, c, lock):
    """At most one bounded query, only from the authorized afterany CPU collector."""
    path = attempt / 'accounting.json'
    if path.exists(): return reader.json(path)
    submission_path = attempt / 'submission.json'
    if not submission_path.exists(): return dict(status='NOT_RECORDED', reason='SUBMISSION_RECEIPT_MISSING')
    submission = reader.json(submission_path)
    require(submission.get('source', lock['source_commit']) == lock['source_commit'], 'ACCOUNTING_SUBMISSION_SOURCE')
    jobs = submission.get('jobs', {})
    require(jobs and all(str(v).isdigit() for v in jobs.values()) and len(set(map(str, jobs.values()))) == len(jobs), 'ACCOUNTING_JOB_ALLOWLIST')
    cpu_ids = {str(v) for k, v in jobs.items() if k.lower() in ('collector', 'cpu', 'cpu_collector')}
    if os.environ.get('SLURM_JOB_ID') not in cpu_ids:
        return dict(status='NOT_MEASURED', reason='NOT_RUNNING_AS_EXACT_REGISTERED_CPU_COLLECTOR', queries=0)
    # GPU parents and this CPU collector are queried once.  No loop or polling.
    owner = pwd.getpwuid(os.getuid()).pw_name
    response = subprocess.run(['sacct', '-X', '-P', '-n', '-j', ','.join(sorted(map(str, jobs.values()))),
        '--format=JobIDRaw,User,JobName,State,ElapsedRaw,AllocTRES'], capture_output=True, text=True, timeout=30)
    if response.returncode:
        return dict(status='NOT_MEASURED', queries=1, returncode=response.returncode, reason='ACCOUNTING_COMMAND_FAILED')
    result = parse_accounting(response.stdout, jobs, owner)
    result.update(queries=1, request_wall_is_not_ETA=True, pending_elapsed_not_GPU_time=True)
    return result


def write_csv(path, rows):
    if not rows: return
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def collect(attempt):
    started = time.monotonic(); attempt = Path(attempt).resolve(); out = attempt / 'cpu-report'; out.mkdir(exist_ok=False)
    reader = Reader(); c = reader.json(attempt / 'config.json'); lock = reader.json(attempt / 'execution.lock.json')
    require(c['instruction_id'] == lock['instruction_id'] == NONCE and c['task_id'] == TASK
            and sha(attempt / 'config.json') == lock['config_sha256'], 'COLLECTOR_AUTHORITY_SOURCE_CONFIG')
    require(tuple(c['settings']['arms']) == ARMS and c['settings']['requests'] == 2000
            and c['settings']['B'] == 100 and c['settings']['batches'] == 20, 'COLLECTOR_SCOPE')
    identities = reader.bound(c['observer_identity'])['rows']; records = load_prefix(Path(c['stream']).parent, 2000)
    require(digest([r['case_id'] for r in records]) == c['ordered_ids_sha256'], 'COLLECTOR_ORDERED_INPUT')
    calibration = calibration_record(reader, attempt, c, lock); arms = {}; raw = {}; errors = []
    for arm in ARMS:
        try:
            arms[arm], raw[arm] = reduce_arm(reader, attempt, arm, c, lock, identities, records, calibration)
        except Exception as error:
            errors.append(dict(arm=arm, type=type(error).__name__, error=str(error), traceback=traceback.format_exc()))
            arms[arm] = dict(status='TECHNICAL_REDUCER_BLOCKED', endpoints={}, coverage={}, warnings=[str(error)])
    cross = {}
    for label in ('W0',) + tuple(f'W{n}' for n in MILESTONES):
        if all(label in raw.get(arm, {}) for arm in ARMS): cross[label] = paired(raw['CD_Q'][label], raw['CD_C'][label])
    status = 'COMPLETED' if not errors and all(arm['status'] == 'W20_COMPLETE' for arm in arms.values()) else 'PARTIAL_OR_TECHNICAL_BLOCKED'
    write(out / 'metrics.json', dict(status=status, arms=arms, errors=errors, source=lock['source_commit'], config=digest(c), shared_lambda=calibration))
    write(out / 'paired-CD_Q-CD_C.json', dict(CD_Q_to_CD_C=cross, same_occurrence_identity=True, trajectories_independent=True))
    table = []
    fields = ('numerator', 'denominator', 'rate', 'strict_numerator', 'strict_denominator', 'strict_rate',
        'desired_token_correct', 'desired_token_count', 'token_micro', 'prompt_macro', 'true_nll_mean', 'new_nll_mean',
        'desired_nll_mean', 'true_minus_new_mean', 'new_minus_true_mean')
    for arm, result in arms.items():
        for label, summary in result['endpoints'].items():
            for kind, row in summary.items():
                table.append(dict(arm=arm, endpoint=label, kind=kind, **{k: row[k] for k in fields}, harmonic_RS_PS_NS=harmonic(summary)))
    write_csv(out / 'comparison-endpoints.csv', table)
    write_csv(out / 'comparison-W20.csv', [row for row in table if row['endpoint'] == 'W20'])
    accounting = sealed_accounting(reader, attempt, c, lock)
    write(out / 'cost.json', dict(accounting=accounting, terminals={a: r.get('terminal', 'NOT_MEASURED') for a, r in arms.items()},
        rule='Exact parent allocations counted once; fit/writer/eval nested times never added to parent allocated time',
        missing_cost_is_not_zero=True, one_exact_parent_query_only_from_registered_CPU_collector=True))
    lines = ['# CD cumulative allocation 2k 사실 보고', '', f'상태: {status}', f'실행 source: `{lock["source_commit"]}`',
        'CD_Q/CD_C는 각자 cold W0/H0에서 같은 first2000을 BS100×20으로 처리한다. 새 baseline/추가 full-B fit 없음.',
        'CPU independent scalar/raw reducer는 생산 observer와 별도 산술 경로다. 저장 hash/관계 검산이지 모델·tensor 재실행 증명은 아니다.', '',
        '| arm | 검산 commit | W20 RS | W20 PS | W20 NS |', '|---|---:|---:|---:|---:|']
    for arm, result in arms.items():
        summary = result['endpoints'].get('W20', {})
        values = [f'{summary[k]["numerator"]}/{summary[k]["denominator"]}' if k in summary else 'NOT_MEASURED' for k in ('R', 'P', 'N')]
        lines.append('| ' + arm + ' | ' + str(result.get('commits', 'NOT_VERIFIED')) + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'W20 완전 측정 분모는 arm별 R2000/P4000/N20000. 누락·부분·기술실패는 0점으로 대체하지 않았다.',
        'TF strict/token micro/prompt macro는 teacher-forced 지표이며 자유생성 정확도가 아니다.',
        'true_minus_new=true NLL−new NLL, new_minus_true는 그 반대 부호로 별도 저장했다.',
        'shared lambda/후보·update/own-state join/Honce/paired/cohort/actual-Y telemetry는 metrics.json에 기록했다.',
        '낮은 성능, finite range-LS, 집중, 미수렴은 사실 관측이며 추가 실행이나 method 변경의 조건으로 쓰지 않았다.',
        'noCP, exact_resume=NOT_AVAILABLE. 원 raw/tensor/model/prompt는 local KEEP; compact 보고만 Git 대상.',
        'NO_BROADCAST_NOT_REQUIRED: 같은 서버에서 생성한 compact 보고/manifest 외 대형 원자료 전송 불필요.',
        f'원자료 경로: `{attempt}`', '과거 baseline은 자동 matched 비교하지 않았다. 과학적 우월성/후속 선정 없음.']
    (out / 'report-ko.md').write_text('\n'.join(lines) + '\n')
    write(out / 'inventory.json', dict(files=list(reader.files.values()), raw_local_KEEP=True,
        scope='all source/config/metric/receipt bytes actually read by this CPU reducer', no_model_or_tensor_replay=True))
    write(out / 'terminal.json', dict(status=status, report=member(out / 'report-ko.md'), inventory=member(out / 'inventory.json'),
        metrics=member(out / 'metrics.json'), source=lock['source_commit'], seconds=time.monotonic() - started))
    return dict(status=status, report=str(out / 'report-ko.md'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(collect(args.attempt)))
