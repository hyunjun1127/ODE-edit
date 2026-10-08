"""Saved native GPT-J W20 generation/RPN independent CPU reducer.

Historical qualified-route collectors stay read-only. This fresh profile binds
SH1's native case-batched global RNG receipt, not a qualification or W0 READY.
Only existing local raw arithmetic and source/state/count identity are checked.
"""
import argparse
import json
import math
import re
import subprocess
from pathlib import Path

from . import generation_collect as prior
from .generation_common import (ARMS, ARM_LAYERS, MILESTONES, batches, digest,
    expected_counts, member, require, write, writer_identity)
from .collect import _atomic_text, _csv, _metric_rows, _paired_rows

from .generation_native_common import TASK, NONCE, ready
from project.run_scripts.experiment_generation_eval.native_profile import PROFILE, ROUTE, runtime_identity

SCHEDULE = 'FINAL_W20_ONLY'
TABLE_KEYS = prior.TABLE_KEYS
COUNT_KEYS = prior.COUNT_KEYS
PUBLIC_REASONS = prior.REASONS[:6]
FORBIDDEN_COMMIT_KEYS = ('gen_before', 'gen_current', 'gen_prefix', 'gen_W0')


def schedule(config, lock):
    """This exact new registration carries native profile and unchanged science."""
    gen = config['generation']
    require(config['task_id'] == lock['task_id'] == TASK
        and config['instruction_id'] == lock['instruction_id'] == NONCE,
        'NATIVE_COLLECT_TASK_AUTHORITY')
    require(config.get('registration_profile') == 'native-repo-r1'
        and gen.get('evaluation_schedule') == SCHEDULE
        and gen.get('final_generation_requests') == 2000
        and gen.get('W0_generation_enabled') is False
        and gen.get('intermediate_generation_enabled') is False
        and gen['schema'] == prior.SCHEMA and gen['profile'] == PROFILE
        and gen['eval_seed'] == prior.EVAL_SEED
        and gen.get('generation_route') == ROUTE
        and config['noCP'] is True and config['z_disk_cache'] is False,
        'NATIVE_COLLECT_SCHEDULE_IDENTITY')
    require(not any(key in gen for key in ('repair', 'qualification_owner',
        'qualification_plan_member', 'qualification_receipt_member',
        'generation_microbatch', 'W0_cache')),
        'NATIVE_COLLECT_NO_OLD_QUALIFICATION_OR_W0_GATE')
    require(lock.get('shared_generation_source') == gen['source_sha']
        and lock.get('shared_generation_tree') == gen['package_tree']
        and lock.get('reference_identity') == gen['reference_assets_sha256'],
        'NATIVE_COLLECT_SHARED_SOURCE_REFERENCE')

def unscheduled_commit(receipt):
    """Unperformed generation is typed missing, never a fictitious guard PASS."""
    require(not any(key in receipt for key in FORBIDDEN_COMMIT_KEYS)
        and receipt.get('generation_schedule') == SCHEDULE
        and receipt.get('generation_available') is False
        and receipt.get('generation_unavailable_reason') == 'FINAL_W20_ONLY_SCHEDULE'
        and receipt.get('generation_observer_status') == 'NOT_SCHEDULED_INTERMEDIATE',
        'INTERMEDIATE_GENERATION_NOT_SCHEDULED')
    require('generation_observer_no_mutation' not in receipt,
            'UNRUN_GENERATION_NOT_NONMUTATION_PASS')


def generation_scalars(summary):
    """Independent mapping of the sealed shared scalar protocol, not scoring."""
    prefix = 'all_seen/post'
    values = {}
    for source, target in (('ngram_entropy', 'fluency/ngram_entropy'),
                           ('reference_score', 'consistency/reference_score')):
        if source in summary:
            values[prefix + '/' + target] = summary[source]
    for key in ('planned_count', 'fluency_count', 'consistency_count',
                'generation_prompt_count', 'generated_token_count'):
        values[prefix + '/generation/' + key] = summary[key]
    for reason in PUBLIC_REASONS:
        values[prefix + '/generation/missing_' + reason + '_count'] = summary['missing_reason_counts'][reason]
    return values


def tracking_evidence(reader, out, config, lock, arm, summary):
    """Parent-accepted scalar journal is separate from SDK/remote delivery."""
    identity = reader.json(out / 'tracking' / 'identity.json')
    cfg = identity['config']
    require(cfg['server'] == 'server2' and cfg['task_id'] == config['task_id']
        and cfg['arm'] == arm and cfg['attempt'] == config['tracking_attempt']
        and cfg['source_sha'] == lock['source_commit']
        and cfg['config_sha'] == lock['config_sha256']
        and cfg['writer'] == writer_identity(arm) and cfg['model'] == cfg['model_family'] == 'gptj'
        and cfg.get('generation_schedule') == 'W20_ONLY_FIRST2000'
        and cfg.get('generation_profile') == PROFILE
        and cfg.get('generation_repair_instruction') == 'USER-DIRECT-NATIVE-FLUCON-REPAIR-20261008-R1'
        and cfg['role'] == 'scientific' and cfg['execution_backend'] == 'slurm'
        and re.fullmatch(r'[1-9][0-9]*', cfg['job_id'])
        and 'job' + cfg['job_display_id'] in identity['run_name']
        and identity['startup_remote_identity_verified'] is True
        and identity['scientific_completion_claim'] is False,
        'FINAL_TRACKING_IMMUTABLE_JOB_SOURCE')
    lines = reader.bytes(out / 'tracking' / 'accepted-scalars.jsonl')
    require(len(lines) <= 128 * 1024**2, 'FINAL_TRACKING_BOUNDED_SCALAR_JOURNAL')
    expected = generation_scalars(summary)
    measured = []
    for line in lines.splitlines():
        command = json.loads(line)
        if command.get('op') != 'log':
            continue
        values = command['values']
        genkeys = {key for key in values if '/fluency/' in key or '/consistency/' in key
                   or '/generation/' in key}
        if genkeys:
            require(genkeys == set(expected) and values.get('edits') == 2000
                and values.get('post_state_edits') == 2000,
                'FINAL_GENERATION_ONLY_ONE_W20_SCALAR_PREFIX')
            prior.compare_value(expected, {key: values[key] for key in genkeys},
                                'FINAL_GENERATION_RAW_TO_WANDB')
            measured.append(command)
    require(len(measured) == 1, 'FINAL_GENERATION_SCALAR_LOG_EXACTLY_ONCE')
    delivery = reader.json(out / 'tracking' / 'receipt.json')
    require(delivery['run_id'] == identity['run_id'], 'FINAL_TRACKING_RECEIPT_RUN_ID')
    result = delivery.get('result', {})
    return dict(status='LOCAL_ACCEPTED_SCALAR_JOURNAL_VERIFIED_NOT_REMOTE_ACK',
        run_id=identity['run_id'], url=identity['url'], job_id=cfg['job_id'],
        scalar_payloads=1, source_config_identity=True,
        acceptance_scope='PARENT_LOCAL_JOURNAL_NOT_SDK_OR_REMOTE_DELIVERY_CERTIFICATION',
        transport_status=delivery.get('status', 'NOT_RECORDED'),
        SDK_finish_status=result.get('status', 'NOT_RECORDED'),
        SDK_flush_observed=result.get('status') == 'FINISHED_SDK_FLUSHED',
        # These are the production sidecar's exact protocol fields. SDK finish
        # is not remote delivery, and method/progress readback have different
        # bounded scopes; do not collapse either into a generic PASS.
        method_readback=result.get('method_readback', {'status': 'NOT_RECORDED'}),
        generation_progress_readback=result.get('generation_progress_readback', {'status': 'NOT_RECORDED'}),
        scientific_completion_not_implied=True)


def compact_error_code(error):
    message = str(error)
    return message if re.fullmatch(r'[A-Z][A-Z0-9_]{0,159}', message) else 'UNCLASSIFIED_REVIEW_ERROR'


def native_runtime(gen):
    """Only actual native generation inputs, never legacy route PLAN fields."""
    return runtime_identity(dict(model_identity=gen['model_identity'],
        generation_source_sha=gen['source_sha'], profile=gen['profile'],
        generation_route=gen['generation_route'], eval_seed=gen['eval_seed']),
        gen['reference_assets_sha256'])


def native_generation_endpoint(reader, receipt, records, state, config, attempt):
    """Bind exact native global stream/rows/work without regenerating or rescoring."""
    from project.run_scripts.experiment_generation_eval.native_observer import read_observed, verify_native_raw
    runtime = native_runtime(config['generation'])
    runtime_sha = digest(runtime)
    path = prior.local_path(receipt['rows_path'], attempt)
    require(receipt['raw_endpoint_member']['path'] == str(path), 'NATIVE_GEN_ENDPOINT_MEMBER_PATH')
    saved = reader.bound(receipt['raw_endpoint_member'])
    identity = saved['identity']
    expected_records = [dict(ordered_occurrence=ordinal, case_id=record['case_id'],
        generation_prompts=record.get('generation_prompts', []),
        relation_id=record['requested_rewrite'].get('relation_id'),
        target_new_id=record['requested_rewrite']['target_new'].get('id'))
        for ordinal, record in enumerate(records, 1)]
    stream = digest(dict(runtime=runtime_sha, eval_seed=prior.EVAL_SEED,
                         ordered_record_identities=expected_records))
    require(identity['runtime'] == runtime_sha and identity['sampling_stream_sha256'] == stream
        and identity['state_sha256'] == digest(state) and identity['endpoint'] == 'W20'
        and identity['cohort'] == 'ALL_SEEN'
        and identity['ordered_occurrences'] == list(range(1, len(records) + 1))
        and receipt['identity'] == identity
        and receipt['identity_sha256'] == saved['identity_sha256'] == digest(identity)
        and receipt['shared_state_identity'] == state
        and receipt['shared_runtime_identity'] == runtime,
        'NATIVE_GEN_EXPECTED_RUNTIME_GLOBAL_STREAM_STATE')
    require(receipt['requests'] == len(records)
        and receipt['cohort_identity'] == digest([record['case_id'] for record in records])
        and all(holder[key] is True for holder in (saved, receipt)
                for key in ('RNG_restored', 'observer_no_mutation'))
        and saved['raw_local_only'] is True,
        'NATIVE_GEN_FULL_COHORT_NONMUTATION')
    require(not any(key in saved or key in receipt for key in
        ('qualification_receipt_member', 'compatibility_member', 'parent_endpoint_member'))
        and saved['native_execution_member'] == receipt['native_execution_member'],
        'NATIVE_GEN_ACTUAL_EXECUTION_NOT_QUALIFICATION_OR_SUBSET')
    observer = reader.json(path.parent.parent / 'observer-identity.json')
    require(observer['identity'] == runtime and observer['identity_sha256'] == runtime_sha
        and observer['raw_local_only'] is True and observer['checkpoint_saved'] is False,
        'NATIVE_GEN_OBSERVER_RUNTIME')
    execution = reader.bound(saved['native_execution_member'])
    require(execution['identity'] == identity and execution['identity_sha256'] == digest(identity)
        and execution['profile'] == PROFILE and execution['route'] == ROUTE
        and execution['native_execution_complete'] is True
        and execution['qualification_performed'] is False and execution['no_fallback'] is True
        and execution['RNG_restored'] is True and execution['observer_no_mutation'] is True
        and execution['raw_local_only'] is True,
        'NATIVE_GEN_ACTUAL_EXECUTION_RECEIPT')
    loaded = read_observed(path, expected_runtime=runtime_sha)
    require(loaded['identity'] == identity and loaded['rows'] == saved['rows'],
            'NATIVE_GEN_SHARED_READER_ROWS')
    rows = saved['rows']
    require(len(rows) == len(records)
        and [row['case_id'] for row in rows] == [record['case_id'] for record in records]
        and [row['occurrence'] for row in rows] == list(range(1, len(records) + 1))
        and identity['observation_identities'] == [row['identity_sha256'] for row in rows],
        'NATIVE_GEN_COMPLETE_ORDERED_CASES')
    counter_names = ('generation_forwards', 'full_prefix_token_work', 'physical_forward_calls',
                     'prefill_query_tokens', 'decode_query_tokens', 'completed_prompts', 'generated_tokens')
    counters = {key: 0 for key in counter_names}
    for ordinal, (row, record, record_identity) in enumerate(zip(rows, records, expected_records), 1):
        raw_path = prior.local_path(row['observation_path'], attempt)
        raw = reader.json(raw_path)
        provenance = row['provenance']
        require(provenance['raw_member']['path'] == str(raw_path)
            and provenance['origin'] == 'NEW_CURRENT_RUNTIME'
            and provenance['runtime_sha256'] == runtime_sha
            and provenance['generation_source_sha'] == config['generation']['source_sha']
            and provenance['route'] == ROUTE,
            'NATIVE_GEN_RAW_PROVENANCE_NO_OLD_RELABEL')
        require(reader.bound(provenance['raw_member']) == raw, 'NATIVE_GEN_RAW_MEMBER_BYTES')
        require(raw['identity'] == dict(runtime=runtime_sha, state_identity=state,
            record_identity=record_identity, sampling_stream_sha256=stream)
            and raw['identity_sha256'] == row['identity_sha256'] == digest(raw['identity'])
            and raw['payload_sha256'] == row['payload_sha256']
            and raw['metrics'] == row['metrics'] and raw['case_id'] == record['case_id']
            and raw['occurrence'] == ordinal,
            'NATIVE_GEN_RECORD_TOKEN_STATE_IDENTITY')
        verify_native_raw(raw, expected_runtime=runtime_sha, expected_stream=stream)
        observations = raw['observations']
        counters['generation_forwards'] += sum(value['model_forwards'] for value in observations)
        counters['full_prefix_token_work'] += sum(value['full_prefix_token_work'] for value in observations)
        counters['completed_prompts'] += len(observations)
        counters['generated_tokens'] += sum(value['continuation_token_count'] for value in observations)
        for key in ('physical_forward_calls', 'prefill_query_tokens', 'decode_query_tokens'):
            counters[key] += sum(value[key] for value in observations)
        require(row['metrics']['generation_prompt_count'] == len(observations)
            and row['metrics']['generated_token_count']
                == sum(value['continuation_token_count'] for value in observations)
            and row['metrics']['length_cap_no_continuation_count']
                == sum(value['stop_reason'] == 'length_cap_no_continuation' for value in observations),
            'NATIVE_GEN_CASE_METRIC_TOKEN_COUNTS')
    reduced = prior.reduce_generation(rows)
    prior.compare_value(reduced, saved['summary'], 'NATIVE_GEN_INDEPENDENT_SUMMARY')
    prior.compare_value(reduced, receipt['summary'], 'NATIVE_GEN_COMPACT_SUMMARY')
    prior.compare_value(reduced, receipt['shared_summary'], 'NATIVE_GEN_SHARED_SUMMARY')
    work = receipt['work']
    require(all(type(work[key]) is int and work[key] >= 0 for key in
        counter_names + ('new_case_observations', 'cached_case_observations'))
        and type(work['seconds']) in (int, float) and math.isfinite(work['seconds']) and work['seconds'] >= 0
        and work['new_case_observations'] == len(records) and work['cached_case_observations'] == 0
        and all(work[key] == count for key, count in counters.items())
        and all(execution[key] == counters[key] for key in
                ('physical_forward_calls', 'prefill_query_tokens', 'decode_query_tokens'))
        and counters['completed_prompts'] == reduced['generation_prompt_count']
        and counters['generated_tokens'] == reduced['generated_token_count'],
        'NATIVE_GEN_LOGICAL_PHYSICAL_EXECUTION_COUNTERS')
    return dict(rows=rows, summary=reduced, work=work, identity=identity,
                identity_sha256=saved['identity_sha256'], native_execution_member=saved['native_execution_member'])


def final_generation(reader, attempt, config, lock, arm, records, state):
    out = Path(attempt) / arm
    receipt = reader.json(out / 'generation-final.json')
    require(receipt['task'] == config['task_id'] and receipt['arm'] == arm
        and receipt['source'] == lock['source_commit'] and receipt['config'] == digest(config)
        and receipt['generation_schedule'] == SCHEDULE and receipt['endpoint'] == 'W20'
        and receipt['edits'] == receipt['actual_model_edits'] == 2000
        and receipt['model_state'] == state and receipt['requests'] == 2000
        and receipt.get('derived_subset') is False
        and receipt['checkpoint_saved'] is False and receipt['exact_resume'] == 'NOT_AVAILABLE'
        and receipt['generation_observer_no_mutation'] is True,
        'NATIVE_FINAL_PHYSICAL_W20_SOURCE')
    committed = receipt['committed_batch20_member']
    require(committed['path'] == str((out / 'batch-20' / 'commit.json').resolve()),
            'NATIVE_FINAL_PERSISTED_BATCH20_PATH')
    last = reader.bound(committed)
    require(last['batch'] == 20 and last['after'] == state
        and last['source'] == lock['source_commit'] and last['config'] == digest(config),
        'NATIVE_FINAL_AFTER_PERSISTED_TWENTY_COMMITS')
    require(not any((out / name).exists() for name in ('generation-W0-reference.json',
        'generation-W0', 'generation-W0.json')), 'NATIVE_FINAL_NO_W0_ENDPOINT')
    endpoint_dir = out / 'generation-raw' / 'endpoints'
    endpoint_files = sorted(endpoint_dir.glob('*.json'))
    require(len(endpoint_files) == 1
        and str(endpoint_files[0]) == receipt['raw_endpoint_member']['path'],
        'NATIVE_FINAL_EXACTLY_ONE_SAVED_ENDPOINT')
    observed = native_generation_endpoint(reader, receipt, records, state, config, attempt)
    require(observed['summary']['planned_count'] == len(observed['rows']) == 2000,
            'NATIVE_FINAL_FULLFIRST2000')
    tracking = tracking_evidence(reader, out, config, lock, arm, observed['summary'])
    return observed, tracking

def terminal_status(terminal, commits, final_available, issues):
    if terminal.get('status') in ('FAILED', 'BLOCKED'):
        return terminal['status']
    if terminal.get('status') == 'COMPLETED':
        return ('COMPLETED_VALIDATED_ROWS_COUNTS' if commits == 20 and final_available and not issues
                else 'TECHNICAL_BLOCKED_INCOMPLETE_EVIDENCE')
    return 'PARTIAL' if commits else 'NOT_STARTED_OR_STARTUP_FAILED'


def history_layout(state, arm, *, initial=False):
    """Stock AlphaEdit starts lazy/empty; its first write creates six planes.

    CAKE and BLUE have their caller-owned cold preallocated history. This
    distinction is an initialization identity, not a change to native history.
    Every completed stock AlphaEdit batch must retain all six native planes.
    """
    owners = ('CAKE', 'ALPHAEDIT_BLUE') if initial else prior.HISTORY_ARMS
    expected = {str(layer) for layer in ARM_LAYERS[arm]} if arm in owners else set()
    require(set(state['H']) == expected,
            'NATIVE_INITIAL_HISTORY_LAYOUT' if initial else 'NATIVE_COMMITTED_HISTORY_LAYOUT')
    return expected


def review_arm(reader, attempt, config, lock, arm, identities, records, progress=None):
    out = Path(attempt) / arm
    result = progress if progress is not None else {}
    totals = {key: 0 for key in COUNT_KEYS}
    result.update(arm=arm, writer=writer_identity(arm), commits=0, requests=0, state_links=0,
        measured_counts=totals, W0_RPN_available=False, generation_final_available=False,
        generation_endpoint_calls=0, generation_W0_endpoint_calls=0,
        generation_intermediate_endpoint_calls=0, issues=[], **{key: [] for key in TABLE_KEYS})
    terminal = reader.json(out / 'terminal.json') if (out / 'terminal.json').exists() else {}
    result['terminal'] = prior.compact_terminal(terminal)
    if not (out / 'runtime.json').exists():
        result['scientific_status'] = terminal_status(terminal, 0, False, result['issues'])
        return result
    runtime = reader.json(out / 'runtime.json')
    require(runtime['source'] == lock['source_commit'] and runtime['config'] == digest(config)
        and runtime['arm'] == arm and runtime['initial_history_zero'] is True
        and runtime['checkpoint_saved'] is False, 'FINAL_NATIVE_RUNTIME_SOURCE_COLD')
    layers = {str(layer) for layer in ARM_LAYERS[arm]}
    initial = runtime['initial_state']
    require(initial['W'] == {layer: config['cold_W'][layer] for layer in layers},
        'FINAL_NATIVE_INITIAL_COLD_LAYOUT')
    history_layout(initial, arm, initial=True)
    all_ids = [record['case_id'] for record in records]
    w0 = prior.rpn_endpoint(reader, out / 'W0', identities, all_ids, 'W0', initial, records)
    if w0 is not None:
        require({key: w0['summary'][key]['denominator'] for key in ('R', 'P', 'N')}
            == dict(R=2000, P=4000, N=20000), 'FINAL_W0_RPN_DENOMINATORS')
        result['W0_RPN_available'] = True
        result['metrics'].extend(_metric_rows(arm, 'W0_FIRST2000', 0, w0['summary']))
        result['compute'].append(dict(arm=arm, phase='W0_RPN', seconds=w0['seconds'],
                                    reference_only=w0['reference_only']))
    previous, at_write = initial, []
    for number, current, seen in batches(records):
        folder = out / f'batch-{number:02d}'
        if not (folder / 'commit.json').exists():
            require(not any((out / f'batch-{later:02d}' / 'commit.json').exists()
                for later in range(number + 1, 22)), 'FINAL_NONCONSECUTIVE_COMMIT')
            break
        receipt = reader.json(folder / 'commit.json')
        ids, seen_ids = [r['case_id'] for r in current], [r['case_id'] for r in seen]
        require(receipt['task'] == config['task_id'] and receipt['arm'] == arm
            and receipt['batch'] == number and receipt['writer'] == writer_identity(arm)
            and receipt['source'] == lock['source_commit'] and receipt['config'] == digest(config)
            and receipt['case_ids'] == ids, 'FINAL_NATIVE_COMMIT_SOURCE_ORDER')
        require(receipt['before'] == previous and set(receipt['after']['W']) == layers,
            'FINAL_NATIVE_PHYSICAL_STATE_LINK')
        history_layout(receipt['after'], arm)
        require(receipt['ledger'] == digest(seen_ids) and receipt['seen_requests'] == number * 100
            and receipt['post_scope'] == ('ALL_SEEN' if number in MILESTONES else 'CURRENT'),
            'FINAL_NATIVE_CURSOR_PREFIX')
        unscheduled_commit(receipt)
        measured = receipt['native_counts']
        require(measured == expected_counts(arm) and all(type(measured[k]) is int for k in COUNT_KEYS),
                'FINAL_NATIVE_MEASURED_COUNTS')
        fitting = prior.native_receipt(receipt['native'], arm, number, measured, totals)
        require(receipt['observer_no_mutation'] is True and receipt['checkpoint_saved'] is False
            and receipt['exact_resume'] == 'NOT_AVAILABLE', 'FINAL_NATIVE_NOCP_OBSERVER')
        if arm == 'PRUNE':
            prior.prune_terminal(receipt['native'], number)
        pre = prior.rpn_endpoint(reader, folder / 'pre', identities, ids,
                                 f'B{number}_PRE', previous, seen)
        selected = seen if number in MILESTONES else current
        post = prior.rpn_endpoint(reader, folder / 'post', identities,
            seen_ids if number in MILESTONES else ids, f'W{number}', receipt['after'], seen)
        require(pre is not None and post is not None, 'FINAL_COMMITTED_RPN_OBSERVATIONS_REQUIRED')
        prior.compare_summary(pre['summary'], receipt['pre'])
        prior.compare_summary(post['summary'], receipt['post'])
        require({k: pre['summary'][k]['denominator'] for k in ('R', 'P', 'N')}
            == dict(R=100, P=200, N=1000), 'FINAL_PRE_CURRENT100_DENOMINATORS')
        require({k: post['summary'][k]['denominator'] for k in ('R', 'P', 'N')}
            == dict(R=len(selected), P=2 * len(selected), N=10 * len(selected)),
            'FINAL_POST_PREFIX_DENOMINATORS')
        current_rows = [row for row in post['rows'] if row['case_id'] in set(ids)]
        current_summary = prior.reduce_rows(current_rows)
        prior.compare_summary(current_summary, receipt['post_current'])
        require({k: current_summary[k]['denominator'] for k in ('R', 'P', 'N')}
            == dict(R=100, P=200, N=1000), 'FINAL_POST_CURRENT100_NOT_PREFIX')
        result['metrics'].extend(_metric_rows(arm, f'B{number}_PRE', number * 100, pre['summary']))
        result['metrics'].extend(_metric_rows(arm, f'W{number}_CURRENT', number * 100, current_summary))
        result['paired'].extend(_paired_rows(arm, f'B{number}_PRE', f'W{number}_CURRENT',
                                            pre['rows'], current_rows))
        at_write.extend(current_rows)
        if number in MILESTONES:
            result['metrics'].extend(_metric_rows(arm, f'W{number}_ALL_SEEN', number * 100, post['summary']))
            first500 = [row for row in post['rows'] if row['case_id'] in set(all_ids[:500])]
            result['metrics'].extend(_metric_rows(arm, f'W{number}_FIRST500', number * 100,
                                                 prior.reduce_rows(first500)))
            result['paired'].extend(_paired_rows(arm, 'AT_WRITE', f'W{number}_ALL_SEEN',
                                                 at_write, post['rows']))
            if w0 is not None:
                result['paired'].extend(_paired_rows(arm, 'W0', f'W{number}_ALL_SEEN',
                    [r for r in w0['rows'] if r['case_id'] in set(seen_ids)], post['rows']))
            for birth in range(1, number + 1):
                cohort = set(all_ids[(birth - 1) * 100:birth * 100])
                result['paired'].extend(_paired_rows(arm, f'B{birth}_AT_WRITE', f'W{number}_ALL_SEEN',
                    [r for r in at_write if r['case_id'] in cohort],
                    [r for r in post['rows'] if r['case_id'] in cohort], birth))
        for key in COUNT_KEYS:
            totals[key] += measured[key]
        seconds = prior.finite(receipt['seconds'], 'FINAL_BATCH_SECONDS')
        require(seconds >= 0, 'FINAL_BATCH_SECONDS_NONNEGATIVE')
        result['counts'].append(dict(arm=arm, batch=number, **measured, **fitting,
                                    terminal_svd_calls=12 if arm == 'PRUNE' and number == 20 else 0))
        result['compute'].extend((dict(arm=arm, batch=number, phase='batch_inclusive', seconds=seconds,
            nested_timers_not_added_again=True), dict(arm=arm, batch=number, phase='pre_RPN', seconds=pre['seconds']),
            dict(arm=arm, batch=number, phase='post_RPN', seconds=post['seconds'])))
        result.update(commits=number, requests=number * 100, state_links=max(0, number - 1))
        previous = receipt['after']
    require(not (out / 'batch-21' / 'commit.json').exists(), 'FINAL_NO_BATCH21')
    final_path = out / 'generation-final.json'
    if final_path.exists():
        require(result['commits'] == 20, 'FINAL_GENERATION_REQUIRES_TWENTY_COMMITS')
        observed, tracked = final_generation(reader, attempt, config, lock, arm, records,
                                             previous)
        result.update(generation_final_available=True, generation_endpoint_calls=1,
            final_generation_identity_sha256=observed['identity_sha256'], generation_tracking=tracked)
        result['generation'].append(prior.generation_table(arm, 'W20_ALL_SEEN', 2000, observed))
        result['compute'].append(dict(arm=arm, phase='final_W20_generation', **observed['work']))
    if terminal:
        require(terminal.get('source') == lock['source_commit'] and terminal.get('config') == digest(config)
            and terminal.get('commits') == terminal.get('completed_batches') == result['commits']
            and terminal.get('checkpoint_saved') is False and terminal.get('exact_resume') == 'NOT_AVAILABLE',
            'FINAL_TERMINAL_SOURCE_PROGRESS_NOCP')
        claimed = terminal.get('native_counts')
        if claimed is not None:
            require(all(type(claimed.get(k)) is int and claimed[k] >= totals[k] for k in COUNT_KEYS),
                    'FINAL_TERMINAL_NATIVE_LOWER_BOUND')
            result['failed_or_uncommitted_claimed_counts'] = {k: claimed[k] - totals[k] for k in COUNT_KEYS}
        if terminal.get('status') == 'COMPLETED':
            require(claimed == totals and terminal.get('edits') == 2000
                and terminal.get('generation_endpoints') == 1 and terminal.get('generation_W0_endpoints') == 0
                and terminal.get('generation_intermediate_endpoints') == 0
                and terminal.get('generation_final_requests') == 2000
                and terminal.get('native_scope_completed') is True
                and terminal.get('final_generation_completed') is True
                and terminal.get('generation_schedule') == SCHEDULE, 'FINAL_COMPLETE_NATIVE_GENERATION_COUNTS')
    if not result['W0_RPN_available']:
        result['issues'].append('W0_RPN_NOT_AVAILABLE')
    if not result['generation_final_available']:
        result['issues'].append('FINAL_W20_GENERATION_NOT_AVAILABLE')
    result['scientific_status'] = terminal_status(terminal, result['commits'],
                                                result['generation_final_available'], result['issues'])
    return result


def collect(attempt):
    attempt, reader = Path(attempt).resolve(), prior.Reader()
    config, lock = reader.json(attempt / 'config.json'), reader.json(attempt / 'execution.lock.json')
    schedule(config, lock)
    require(reader.files[str(attempt / 'config.json')]['sha256'] == lock['config_sha256'],
            'FINAL_COLLECT_CONFIG_BYTES')
    # Native profile ready() binds source/input/noCP. No old route qualification,
    # shared W0 compatibility or scientific performance gate is inherited.
    ready(config)
    for bound in config['generation'].get('shared_source_members', []):
        prior.bound_bytes(reader, bound, 'FINAL_SHARED_SOURCE_BYTES')
    identities = reader.bound(config['observer_identity'])
    stream = next(row for row in config['assets'] if row['path'] == config['stream'])
    records = reader.bound(stream)[:2000]
    require(len(records) == len({r['case_id'] for r in records}) == 2000
        and [r['case_id'] for r in records] == [case for pack in config['packs'] for case in pack['ids']],
        'FINAL_COLLECT_FIRST2000_ORDER')
    list(batches(records))
    out = attempt / 'collector'
    require(not out.exists(), 'FINAL_COLLECT_CREATE_ONCE_OUTPUT')
    out.mkdir()
    reviews = []
    for arm in ARMS:
        progress = {}
        try:
            reviews.append(review_arm(reader, attempt, config, lock, arm, identities, records,
                                      progress))
        except Exception as error:
            progress.update(arm=arm, scientific_status='TECHNICAL_BLOCKED_REDUCER',
                error_type=type(error).__name__, error_code=compact_error_code(error),
                valid_prefix_preserved=True, original_raw_preserved=True)
            progress.setdefault('issues', []).append('STORED_DATA_INCONSISTENCY')
            reviews.append(progress)
    for key in TABLE_KEYS:
        _csv(out / (key + '.csv'), [row for review in reviews for row in review.get(key, [])])
    expected = {key: sum(expected_counts(arm)[key] * 20 for arm in ARMS) for key in COUNT_KEYS}
    actual = {key: sum(review.get('measured_counts', {}).get(key, 0) for review in reviews) for key in COUNT_KEYS}
    complete = all(r['scientific_status'] == 'COMPLETED_VALIDATED_ROWS_COUNTS' for r in reviews)
    complete = complete and actual == expected
    accounting = prior.allocation_once(reader, attempt, lock)
    write(out / 'allocation.json', accounting)
    write(out / 'counts-plan-actual.json', dict(expected_native=expected, verified_committed_native=actual,
        generation_schedule=SCHEDULE, expected_generation_endpoints_per_arm=1,
        verified_generation_endpoints=sum(r.get('generation_endpoint_calls', 0) for r in reviews),
        expected_final_case_observations=12000, generation_W0_endpoints=0,
        generation_intermediate_endpoints=0, generation_profile=PROFILE,
        generation_route=ROUTE, qualification_performed=False,
        prior_attempt_cost_not_added=True, scientific_complete=complete))
    compact = [{k: v for k, v in r.items() if k not in TABLE_KEYS} for r in reviews]
    lines = ['# GPT-J baseline 최종 W20 fluency·consistency CPU 검산', '',
        '생성은 arm당 최종 W20 first2000에서 한 번만 수행한다. W0 및 중간 generation은 미측정이며 0점으로 채우지 않는다.', '',
        '| Arm | 검산 상태 | commit | 요청 | 연결 | 최종 generation |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in compact:
        lines.append(f"| {row['arm']} | {row['scientific_status']} | {row.get('commits', 0)} | "
            f"{row.get('requests', 0)} | {row.get('state_links', 0)} | {row.get('generation_endpoint_calls', 0)} |")
    lines += ['', 'R/P/N pre/post와 W5/10/15/20 실제 누적·first500·birth cohort 평가는 유지한다.',
        'Fluency는 bits, consistency는 TF-IDF cosine이며 두 지표는 동일 생성문을 사용한다. 결측 평균은 생략한다.',
        'Native case-padded KV/topk5/global endpoint RNG/noEOS 생성은 별도 MB qualification·fallback 없이 수행한다.',
        'Scalar journal 접수, remote readback, 과학 계산 완결은 서로 다른 상태다.',
        'NoCP / exact_resume=NOT_AVAILABLE. 실패·부분 prefix·원 raw는 보존한다.',
        '신규 six parent allocation만 별도 기록하며 이전 attempt 비용이나 nested timer를 중복 합산하지 않는다.', '',
        '[RPN](metrics.csv) · [paired](paired.csv) · [최종 생성](generation.csv) · [계수](counts.csv) · [비용](compute.csv)', '']
    _atomic_text(out / 'report-ko.md', '\n'.join(lines))
    write(out / 'review.json', dict(task=config['task_id'], instruction_id=config['instruction_id'],
        source=lock['source_commit'], generation_schedule=SCHEDULE, reviews=compact,
        generation_profile=PROFILE, generation_route=ROUTE, qualification_performed=False,
        scientific_complete=complete,
        new_model_forwards=0, raw_copied=False, raw_publication=False))
    write(out / 'raw-input-manifest.json', dict(local_only=True, inputs=list(reader.files.values())))
    write(out / 'manifest.json', dict(task=config['task_id'], source=lock['source_commit'],
        input_files=len(reader.files), input_manifest=member(out / 'raw-input-manifest.json'),
        outputs=[member(path) for path in sorted(out.iterdir())
                 if path.is_file() and path.name != 'raw-input-manifest.json'],
        raw_copied=False, raw_input_manifest_local_only=True))
    # The atomic collector terminal is deliberately last, after report/inventory.
    write(out / 'terminal.json', dict(status='COMPLETED', scientific_complete=complete,
        generation_schedule=SCHEDULE, report=member(out / 'report-ko.md'),
        source=lock['source_commit'], new_model_forwards=0))
    return dict(status='CPU_REPORT_WRITTEN', scientific_complete=complete, output=str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', required=True, type=Path)
    print(json.dumps(collect(parser.parse_args().attempt)))
