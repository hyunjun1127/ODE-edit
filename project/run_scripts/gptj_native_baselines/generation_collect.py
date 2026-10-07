"""CPU-only independent review of six native chains and saved generation rows.

This module never imports a model, tokenizer, fitter or generation implementation.
It checks the published SH1 receipt protocol, independently reduces saved scalar
metrics, and preserves valid prefixes when later data are missing or inconsistent.
Stored token relations are checked; tokenization and scientific scoring are not
reexecuted. Optional scheduler accounting is one exact six-parent query, never
a monitor; its absence cannot erase valid scientific rows.
"""
import argparse
import json
import math
import re
import subprocess
from pathlib import Path

from .generation_common import (
    ARMS, ARM_LAYERS, MILESTONES, NONCE, TASK, batches, digest,
    expected_counts, member, require, write, writer_identity,
)
from .generation_plan import counts as planned_counts
from .collect import _atomic_text, _csv, _metric_rows, _paired_rows
from project.run_scripts.gptj_cake_blue_prune_rect.collect import endpoint as rpn_endpoint
from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader, compare_summary, reduce_rows,
)

COUNT_KEYS = ('native_z', 'write_keys', 'history_keys', 'solves', 'history_appends')
HISTORY_ARMS = ('BASE_ALPHAEDIT', 'CAKE', 'ALPHAEDIT_BLUE')
SCHEMA = 'counterfact-cake-generation-metrics-v1'
PROFILE = 'cf-cake-prompt-inclusive-total100-eos-corrected-v1'
EVAL_SEED = 20261007
# Closed vocabulary from SH1 metrics.py at 83535c6a; not a new scoring rule.
REASONS = ('missing_generation_prompts', 'missing_reference', 'zero_generated_vector',
           'zero_reference_vector', 'nonfinite_score', 'length_cap_no_continuation',
           'asset_not_available', 'tokenizer_not_available')
TABLE_KEYS = ('metrics', 'paired', 'generation', 'counts', 'compute')


def integer(value, label):
    require(type(value) is int and value >= 0, label)
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label)
    return value


def compare_value(actual, saved, label):
    """No loose key matching, missing mean substitution, or scientific gate."""
    if isinstance(actual, dict):
        require(type(saved) is dict and set(actual) == set(saved), label + '_KEYS')
        for key in actual:
            compare_value(actual[key], saved[key], label)
    elif type(actual) is float:
        finite(saved, label)
        require(math.isclose(actual, saved, rel_tol=1e-12, abs_tol=1e-12), label)
    else:
        require(type(actual) is type(saved) and actual == saved, label)


def reduce_generation(rows):
    """Independent stdlib arithmetic over per-occurrence stored SH1 metrics."""
    entropy, cosine = [], []
    prompt_count = token_count = capped_count = 0
    reasons = {key: 0 for key in REASONS}
    for row in rows:
        metric = row['metrics']
        require(type(metric['fluency_valid']) is bool
                and type(metric['consistency_valid']) is bool, 'GEN_VALIDITY_BOOL')
        for flag, scalar, values in (('fluency_valid', 'ngram_entropy', entropy),
                                     ('consistency_valid', 'reference_score', cosine)):
            if metric[flag]:
                values.append(finite(metric[scalar], 'GEN_VALID_FINITE_SCALAR'))
            else:
                require(metric[scalar] is None, 'GEN_MISSING_NOT_ZERO')
        why = metric['reasons']
        require(type(why) is list and all(type(item) is str for item in why)
                and why == sorted(set(why)) and set(why) <= set(REASONS), 'GEN_REASON_VOCABULARY')
        for reason in why:
            reasons[reason] += 1
        prompts = integer(metric['generation_prompt_count'], 'GEN_PROMPT_COUNT')
        tokens = integer(metric['generated_token_count'], 'GEN_TOKEN_COUNT')
        capped = integer(metric['length_cap_no_continuation_count'], 'GEN_CAPPED_COUNT')
        require(capped <= prompts, 'GEN_CAPPED_PROMPT_BOUND')
        if not prompts:
            require(not metric['fluency_valid'] and not metric['consistency_valid']
                    and why == ['missing_generation_prompts'] and tokens == capped == 0,
                    'GEN_EMPTY_PROMPTS_TYPED_MISSING')
        prompt_count += prompts
        token_count += tokens
        capped_count += capped
    result = dict(planned_count=len(rows), fluency_count=len(entropy),
        consistency_count=len(cosine), fluency_sum=math.fsum(entropy),
        consistency_sum=math.fsum(cosine), missing_reason_counts=reasons,
        generation_prompt_count=prompt_count, generated_token_count=token_count,
        length_cap_no_continuation_prompt_count=capped_count,
        reason_count_unit='request_occurrences_nonexclusive', fluency_unit='bits',
        consistency_unit='cosine_0_to_1')
    if entropy:
        result['ngram_entropy'] = result['fluency_sum'] / len(entropy)
    if cosine:
        result['reference_score'] = result['consistency_sum'] / len(cosine)
    return result


def local_path(path, attempt):
    """Only explicitly referenced files within this attempt, never a scan."""
    path, attempt = Path(path), Path(attempt).resolve()
    require(path.is_absolute() and not path.is_symlink(), 'GEN_RAW_LOCAL_PATH')
    resolved = path.resolve()
    require(resolved.is_relative_to(attempt), 'GEN_RAW_OUTSIDE_ATTEMPT')
    return path


def runtime_identity(config):
    generation = config['generation']
    return dict(schema=SCHEMA, profile=PROFILE, eval_seed=EVAL_SEED,
        model_identity=generation['model_identity'], generation_source_sha=generation['source_sha'],
        reference_assets_sha256=generation['reference_assets_sha256'],
        route='UNPADDED_FULL_PREFIX_NO_CACHE')


def validate_prompt(observation, prompt, occurrence, index, model_identity):
    """Check recorded exact route/EOS/token/work relations without a tokenizer."""
    require(observation['profile'] == PROFILE and observation['prompt'] == prompt
            and observation['occurrence'] == occurrence and observation['prompt_index'] == index
            and observation['route'] == 'UNPADDED_FULL_PREFIX_NO_CACHE'
            and observation['RNG_restored'] is True, 'GEN_PROMPT_IDENTITY_ROUTE')
    seed = int(digest(dict(model_identity=model_identity, ordered_occurrence=occurrence,
                          prompt_index=index, eval_seed=EVAL_SEED))[:16], 16) % (2**63 - 1)
    require(type(observation['seed']) is int and observation['seed'] == seed, 'GEN_CASE_SEED')
    require(observation['sampling'] == dict(top_k=5, temperature=1, top_p=1, max_total_tokens=100),
            'GEN_NATIVE_SAMPLING_PROFILE')
    original, generated, full = (observation[key] for key in
        ('input_token_ids', 'continuation_token_ids', 'full_token_ids'))
    require(all(type(values) is list and all(type(token) is int and token >= 0 for token in values)
                for values in (original, generated, full)) and original and full == original + generated,
            'GEN_TOKEN_SEQUENCE')
    n, m = len(original), len(generated)
    require(type(observation['input_token_count']) is int and type(observation['continuation_token_count']) is int
            and observation['input_token_count'] == n and observation['continuation_token_count'] == m
            and type(observation['text']) is str, 'GEN_RECORDED_TOKEN_LENGTH')
    binding = observation['eos_binding']
    require(type(binding) is dict and set(binding) <= {'generation_config', 'model_config', 'tokenizer'}
            and all(type(values) is list and all(type(token) is int and token >= 0 for token in values)
                    for values in binding.values()), 'GEN_EOS_BINDING')
    eos = sorted({token for values in binding.values() for token in values})
    require(observation['eos_ids'] == eos, 'GEN_EOS_UNION')
    stop = observation['stop_reason']
    if n >= 100:
        require(m == 0 and stop == 'length_cap_no_continuation', 'GEN_LONG_PROMPT_NO_CONTINUATION')
    else:
        require(0 < m <= 100 - n and not any(token in eos for token in generated[:-1]),
                'GEN_EOS_FIRST_TERMINATION')
        if stop == 'eos':
            require(generated[-1] in eos, 'GEN_EOS_LAST_TOKEN')
        else:
            require(stop == 'length_cap' and n + m == 100 and generated[-1] not in eos,
                    'GEN_TOTAL100_CAP')
    forwards, work = m, n * m + m * (m - 1) // 2
    require(type(observation['model_forwards']) is int and observation['model_forwards'] == forwards
            and type(observation['full_prefix_token_work']) is int
            and observation['full_prefix_token_work'] == work, 'GEN_PROMPT_WORK_RELATION')
    return forwards, work, stop == 'length_cap_no_continuation'


def validate_work(work, count, total_forwards, total_tokens, *, cached_only=False):
    require(type(work) is dict, 'GEN_WORK_REQUIRED')
    for key in ('new_case_observations', 'cached_case_observations', 'generation_forwards',
                'full_prefix_token_work'):
        integer(work[key], 'GEN_WORK_INTEGER')
    require(finite(work['seconds'], 'GEN_WORK_SECONDS') >= 0, 'GEN_WORK_SECONDS_NONNEGATIVE')
    fresh, cached = work['new_case_observations'], work['cached_case_observations']
    require(fresh + cached == count, 'GEN_WORK_CASE_COVERAGE')
    if cached_only:
        require(fresh == 0, 'GEN_SUBSET_NO_NEW_GENERATION')
    if not fresh:
        require(work['generation_forwards'] == work['full_prefix_token_work'] == 0,
                'GEN_CACHE_NO_RECHARGED_WORK')
    elif not cached:
        require(work['generation_forwards'] == total_forwards
                and work['full_prefix_token_work'] == total_tokens, 'GEN_FRESH_WORK_EXACT')
    else:
        # SH1 does not persist a per-row new/cache flag. Do not invent attribution.
        require(work['generation_forwards'] <= total_forwards
                and work['full_prefix_token_work'] <= total_tokens, 'GEN_MIXED_WORK_BOUND')
    return dict(work, work_attribution='EXACT' if not fresh or not cached else 'RECORDED_MIXED_BOUND')


def generation_endpoint(reader, receipt, selected, occurrences, expected_state, endpoint,
                        cohort, config, attempt, *, cached_only=False):
    """Independent read_observed protocol verification, including original bytes."""
    require(type(receipt) is dict and receipt['requests'] == len(selected)
            and receipt['cohort_identity'] == digest([r['case_id'] for r in selected]),
            'GEN_COMPACT_COHORT')
    require(receipt['shared_state_identity'] == expected_state, 'GEN_SHARED_PHYSICAL_STATE')
    path = local_path(receipt['rows_path'], attempt)
    require(receipt['raw_endpoint_member']['path'] == str(path), 'GEN_ENDPOINT_MEMBER_PATH')
    value = reader.bound(receipt['raw_endpoint_member'])
    identity = value['identity']
    runtime = runtime_identity(config)
    runtime_sha = digest(runtime)
    observer = reader.json(path.parent.parent / 'observer-identity.json')
    require(observer['identity'] == runtime and observer['identity_sha256'] == runtime_sha
            and observer['raw_local_only'] is True and observer['checkpoint_saved'] is False,
            'GEN_RUNTIME_SOURCE_REFERENCE')
    require(identity['runtime'] == runtime_sha and identity['state_sha256'] == digest(expected_state)
            and identity['endpoint'] == endpoint and identity['cohort'] == cohort
            and value['identity_sha256'] == digest(identity)
            and receipt['identity'] == identity and receipt['identity_sha256'] == value['identity_sha256'],
            'GEN_ENDPOINT_IDENTITY')
    require(value['RNG_restored'] is True and value['observer_no_mutation'] is True
            and value['raw_local_only'] is True and receipt['RNG_restored'] is True
            and receipt['observer_no_mutation'] is True, 'GEN_ENDPOINT_NONMUTATION')
    rows = value['rows']
    require(identity['ordered_occurrences'] == occurrences
            and [r['occurrence'] for r in rows] == occurrences
            and all(type(r['occurrence']) is int and type(r['case_id']) is int for r in rows)
            and len(set(occurrences)) == len(occurrences)
            and [r['case_id'] for r in rows] == [r['case_id'] for r in selected]
            and identity['observation_identities'] == [r['identity_sha256'] for r in rows],
            'GEN_ORDERED_OCCURRENCE_ROWS')
    forwards = token_work = 0
    for row, record, occurrence in zip(rows, selected, occurrences):
        raw = reader.json(local_path(row['observation_path'], attempt))
        rewrite = record['requested_rewrite']
        record_identity = dict(ordered_occurrence=occurrence, case_id=record['case_id'],
            generation_prompts=record.get('generation_prompts', []), relation_id=rewrite.get('relation_id'),
            target_new_id=rewrite.get('target_new', {}).get('id'))
        require(raw['identity_sha256'] == row['identity_sha256'] == digest(raw['identity'])
                and raw['identity'] == dict(runtime=runtime_sha, state_identity=expected_state,
                                           record_identity=record_identity)
                and raw['payload_sha256'] == row['payload_sha256']
                and raw['payload_sha256'] == digest({k: v for k, v in raw.items() if k != 'payload_sha256'})
                and raw['occurrence'] == occurrence and raw['case_id'] == record['case_id']
                and raw['metrics'] == row['metrics'] and raw['raw_local_only'] is True
                and raw['checkpoint_saved'] is False, 'GEN_RAW_IDENTITY_PAYLOAD')
        prompts = record_identity['generation_prompts']
        require(type(prompts) is list and all(type(prompt) is str for prompt in prompts)
                and len(raw['observations']) == len(prompts), 'GEN_ORIGINAL_PROMPT_COVERAGE')
        tokens = capped = 0
        for index, (observed, prompt) in enumerate(zip(raw['observations'], prompts)):
            n, work, cap = validate_prompt(observed, prompt, occurrence, index, runtime['model_identity'])
            forwards += n
            token_work += work
            tokens += n
            capped += cap
        require(row['metrics']['generation_prompt_count'] == len(prompts)
                and row['metrics']['generated_token_count'] == tokens
                and row['metrics']['length_cap_no_continuation_count'] == capped,
                'GEN_CASE_TOKEN_COUNTS')
    reduced = reduce_generation(rows)
    compare_value(reduced, value['summary'], 'GEN_RAW_SUMMARY')
    compare_value(reduced, receipt['summary'], 'GEN_COMPACT_SUMMARY')
    compare_value(reduced, receipt['shared_summary'], 'GEN_SHARED_SUMMARY')
    work = validate_work(receipt['work'], len(rows), forwards, token_work, cached_only=cached_only)
    return dict(rows=rows, summary=reduced, work=work, identity=identity,
                identity_sha256=value['identity_sha256'])


def generation_table(arm, label, edits, observed):
    summary = observed['summary']
    result = dict(arm=arm, endpoint=label, edits=edits,
        **{key: value for key, value in summary.items() if key != 'missing_reason_counts'})
    result.update({'missing_' + key + '_count': value
                   for key, value in summary['missing_reason_counts'].items()})
    return result


def prune_terminal(native, number):
    applied = 'terminal_prune' in native
    require(applied == (number == 20), 'PRUNE_EXACTLY_ONCE_TERMINAL')
    if applied:
        terminal = native['terminal_prune']
        require(native['prune_applied'] is True and native['explicit_repair'] == 'PRUNE_TERMINAL_BASE_FIX'
                and terminal['prune_applied'] is True
                and terminal['explicit_repair'] == 'PRUNE_TERMINAL_BASE_FIX'
                and terminal['native_svd_calls'] == 12, 'PRUNE_TERMINAL_SAVED_W0_BASE_FIX')
        require(terminal['status'] == 'PRUNE_TERMINAL_APPLIED' and terminal['arm'] == 'PRUNE'
                and terminal['batch'] == 20 and terminal['state_edits'] == 2000
                and terminal['native_spectral_formula_unchanged'] is True
                and terminal['final_weight_base'] == 'saved_cold_W0'
                and terminal['W0_RAM_only'] is True and terminal['checkpoint_saved'] is False
                and terminal['exact_resume'] == 'NOT_AVAILABLE'
                and [row['layer'] for row in terminal['layers']] == list(ARM_LAYERS['PRUNE']),
                'PRUNE_TERMINAL_PHYSICAL_PROVENANCE')
        for row in terminal['layers']:
            require(row['native_svd_calls'] == 2
                    and integer(row['singular_values_compressed'], 'PRUNE_COMPRESSED_COUNT')
                    <= integer(row['singular_values'], 'PRUNE_SINGULAR_COUNT'), 'PRUNE_LAYER_SVD_COUNTS')
            for key in ('max_sigma_cold', 'max_sigma_update', 'max_sigma_compressed',
                        'dense_delta_norm', 'compressed_delta_norm', 'terminal_change_norm',
                        'cold_weight_norm', 'final_weight_norm'):
                require(finite(row[key], 'PRUNE_FINITE_SPECTRAL_SCALAR') >= 0, 'PRUNE_NORM_NONNEGATIVE')
        require(terminal['compressed_singular_values'] == sum(row['singular_values_compressed']
                    for row in terminal['layers'])
                and finite(terminal['seconds'], 'PRUNE_TERMINAL_SECONDS') >= 0, 'PRUNE_SPECTRAL_TOTALS')


def native_receipt(native, arm, number, measured, prior):
    require(native['status'] == 'NATIVE_APPLY_RETURNED' and native['arm'] == arm
            and native['writer'] == writer_identity(arm) and native['batch'] == number
            and native['requests'] == 100 and native['same_model_returned'] is True
            and native['native_has_history'] == (arm in HISTORY_ARMS)
            and native['caller_history_appends'] == 0 and native['native_z_disk_cache'] is False
            and native['cache_template'] is None and native['checkpoint_saved'] is False
            and native['exact_resume'] == 'NOT_AVAILABLE', 'NATIVE_DIRECT_APPLY_RECEIPT')
    require(all(native['counts'][key] == native['delta'][key] == measured[key]
                and native['cumulative'][key] == prior[key] + measured[key] for key in COUNT_KEYS),
            'NATIVE_CUMULATIVE_COUNTS')
    trace = native['counts']['fit_trace']
    require(len(trace) == measured['native_z']
            and [row['request_index'] for row in trace] == list(range(1, len(trace) + 1)),
            'NATIVE_MEASURED_FIT_TRACE_COUNT')
    for row in trace:
        require(type(row['evaluations']) is int and 1 <= row['evaluations'] <= 25
                and type(row['Adam_updates']) is int and row['Adam_updates'] == row['evaluations'] - 1
                and row['stop'] in ('TOTAL_LOSS_BELOW_005', 'BUDGET_EXHAUSTED'), 'NATIVE_FIT_BUDGET_TRACE')
        for key in ('loss', 'nll_loss', 'kl_loss', 'weight_decay'):
            finite(row[key], 'NATIVE_FIT_TRACE_FINITE')
    evaluations = sum(row['evaluations'] for row in trace)
    updates = sum(row['Adam_updates'] for row in trace)
    if arm not in ('BASE_MEMIT', 'BASE_ALPHAEDIT'):
        require(native['counts']['fit_forwards'] == evaluations
                and native['counts']['fit_updates'] == updates, 'NATIVE_FIT_TRACE_COUNTER_MATCH')
    return dict(measured_fit_evaluations=evaluations, measured_Adam_updates=updates)


def compact_terminal(value):
    allowed = ('status', 'stage', 'error_type', 'completed_batches', 'commits', 'edits',
        'native_counts', 'rollback_verified', 'checkpoint_saved', 'exact_resume', 'source',
        'config', 'program_seconds', 'peak_host_RSS_bytes', 'peak_gpu_allocated_bytes',
        'peak_gpu_reserved_bytes', 'generation_endpoints')
    return {key: value[key] for key in allowed if key in value}


def terminal_status(terminal, commits, w0_ready, issues):
    status = terminal.get('status')
    if status in ('FAILED', 'BLOCKED'):
        return status
    if status == 'COMPLETED' and commits == 20 and w0_ready and not issues:
        return 'COMPLETED_VALIDATED_ROWS_COUNTS'
    if status == 'COMPLETED':
        return 'TECHNICAL_BLOCKED_INCOMPLETE_EVIDENCE'
    return 'PARTIAL' if commits else 'NOT_STARTED_OR_STARTUP_FAILED'


def allocation_once(reader, attempt, lock, *, runner=None):
    """One bounded read of only this task's six submitted GPU parent IDs."""
    submission_path = Path(attempt) / 'submission.json'
    if not submission_path.exists():
        return dict(status='NOT_RECORDED', reason='OWN_SUBMISSION_NOT_FOUND', queries=0)
    queries = 0
    try:
        submission = reader.json(submission_path)
        require(submission['task_id'] == TASK and submission['instruction_id'] == NONCE
                and submission['source_commit'] == lock['source_commit']
                and submission['lock']['path'] == str(Path(attempt) / 'execution.lock.json'),
                'ALLOCATION_OWN_SUBMISSION_SOURCE')
        reader.bound(submission['lock'])
        jobs = {arm: submission['jobs'][arm] for arm in ARMS}
        require(len(set(jobs.values())) == 6 and all(type(job) is str
                and re.fullmatch(r'[1-9][0-9]*', job) for job in jobs.values()), 'ALLOCATION_EXACT_SIX_IDS')
        argv = ['sacct', '-n', '-P', '-X', '-j', ','.join(jobs.values()),
                '--format=JobIDRaw,State,ElapsedRaw,AllocTRES']
        queries = 1
        value = (subprocess.run if runner is None else runner)(argv, text=True,
            capture_output=True, timeout=20, check=False)
        require(value.returncode == 0 and len(value.stdout) <= 1024**2, 'ALLOCATION_BOUNDED_RESPONSE')
        records = {}
        for line in value.stdout.splitlines():
            if not line.strip():
                continue
            fields = line.strip().split('|')
            if len(fields) == 5 and not fields[-1]:
                fields.pop()
            require(len(fields) == 4, 'ALLOCATION_COLUMN_SCHEMA')
            job, state, elapsed, tres = fields
            if job not in jobs.values():
                require(any(job.startswith(parent + '.') for parent in jobs.values()),
                        'ALLOCATION_UNREQUESTED_JOB')
                continue  # Never double count child steps, even a malformed -X response.
            require(job not in records and elapsed.isdigit(), 'ALLOCATION_EXACT_PARENT_ROW')
            resources = dict(item.split('=', 1) for item in tres.split(',') if '=' in item)
            gpu = resources.get('gres/gpu')
            if gpu is None:
                typed = [count for key, count in resources.items() if key.startswith('gres/gpu:')]
                require(len(typed) <= 1, 'ALLOCATION_TYPED_GPU_SCHEMA')
                gpu = typed[0] if typed else '0'
            require(gpu in ('0', '1'), 'ALLOCATION_ONE_GPU_PER_PARENT')
            arm = next(arm for arm, item in jobs.items() if item == job)
            records[job] = dict(arm=arm, job_id=job, scheduler_state=state,
                elapsed_seconds=int(elapsed), allocated_GPUs=int(gpu),
                allocated_GPU_seconds=int(elapsed) * int(gpu), AllocTRES=tres,
                child_steps_excluded=True, allocation_not_program_timer_sum=True)
        require(set(records) == set(jobs.values()), 'ALLOCATION_ALL_SIX_PARENTS_REQUIRED')
        ordered = [records[jobs[arm]] for arm in ARMS]
        return dict(status='RECORDED', queries=queries, scope='OWN_SIX_GPU_PARENTS_ONLY',
            records=ordered, allocated_GPU_seconds=sum(row['allocated_GPU_seconds'] for row in ordered),
            scheduler_success_not_scientific_success=True)
    except Exception as error:
        return dict(status='NOT_RECORDED', queries=queries, error_type=type(error).__name__,
            reason='OWN_ALLOCATION_UNAVAILABLE_OR_IDENTITY_MISMATCH', no_retry=True,
            scheduler_success_not_scientific_success=True)


def review_arm(reader, attempt, config, lock, arm, identities, records, progress=None):
    out, result = Path(attempt) / arm, progress if progress is not None else {}
    totals = {key: 0 for key in COUNT_KEYS}
    result.update(arm=arm, writer=writer_identity(arm), commits=0, requests=0, state_links=0,
        measured_counts=totals, W0_RPN_available=False, W0_generation_available=False,
        generation_endpoint_calls=0, generation_current_subset_reductions=0,
        generation_edit_state_case_observations=0, issues=[], **{key: [] for key in TABLE_KEYS})
    terminal = reader.json(out / 'terminal.json') if (out / 'terminal.json').exists() else {}
    result['terminal'] = compact_terminal(terminal)
    if not (out / 'runtime.json').exists():
        result['scientific_status'] = terminal_status(terminal, 0, False, result['issues'])
        return result
    runtime = reader.json(out / 'runtime.json')
    require(runtime['source'] == lock['source_commit'] and runtime['config'] == digest(config)
            and runtime['arm'] == arm and runtime['initial_history_zero'] is True
            and runtime['checkpoint_saved'] is False, 'NATIVE_RUNTIME_SOURCE_COLD')
    layers = {str(layer) for layer in ARM_LAYERS[arm]}
    initial = runtime['initial_state']
    require(initial['W'] == {layer: config['cold_W'][layer] for layer in layers}
            and set(initial['H']) == (layers if arm in ('CAKE', 'ALPHAEDIT_BLUE') else set()),
            'NATIVE_INITIAL_COLD_LAYOUT')
    all_ids = [record['case_id'] for record in records]
    occurrence = {record['case_id']: index for index, record in enumerate(records, 1)}
    w0 = rpn_endpoint(reader, out / 'W0', identities, all_ids, 'W0', initial, records)
    if w0 is not None:
        require({key: w0['summary'][key]['denominator'] for key in ('R', 'P', 'N')}
                == dict(R=2000, P=4000, N=20000), 'W0_RPN_DENOMINATORS')
        result['W0_RPN_available'] = True
        result['metrics'].extend(_metric_rows(arm, 'W0_FIRST2000', 0, w0['summary']))
        result['compute'].append(dict(arm=arm, phase='W0_RPN', seconds=w0['seconds'],
                                      reference_only=w0['reference_only']))
    w0gen = None
    if (out / 'generation-W0-reference.json').exists():
        compact = reader.json(out / 'generation-W0-reference.json')
        require(compact['model_state'] == initial, 'W0_GEN_ARM_PHYSICAL_STATE')
        w0gen = generation_endpoint(reader, compact, records, list(range(1, 2001)),
            config['generation']['W0_state_identity'], 'W0', 'FIRST2000', config, attempt,
            cached_only=arm != 'BASE_MEMIT')
        result['W0_generation_available'] = True
        result['W0_generation_identity'] = w0gen['identity_sha256']
        result['generation'].append(generation_table(arm, 'W0_FIRST2000', 0, w0gen))
        result['compute'].append(dict(arm=arm, phase='W0_generation', **w0gen['work']))
    previous, at_write = initial, []
    for number, current, seen in batches(records):
        folder = out / f'batch-{number:02d}'
        if not (folder / 'commit.json').exists():
            require(not any((out / f'batch-{later:02d}' / 'commit.json').exists()
                            for later in range(number + 1, 22)), 'NONCONSECUTIVE_OR_BATCH21_COMMIT')
            break
        receipt = reader.json(folder / 'commit.json')
        ids = [record['case_id'] for record in current]
        seen_ids = [record['case_id'] for record in seen]
        require(receipt['task'] == TASK and receipt['arm'] == arm and receipt['batch'] == number
                and receipt['writer'] == writer_identity(arm) and receipt['source'] == lock['source_commit']
                and receipt['config'] == digest(config) and receipt['case_ids'] == ids,
                'NATIVE_COMMIT_SOURCE_REQUEST_ORDER')
        require(receipt['before'] == previous and set(receipt['after']['W']) == layers
                and set(receipt['after']['H']) == (layers if arm in HISTORY_ARMS else set()),
                'NATIVE_PHYSICAL_STATE_LINK')
        require(receipt['ledger'] == digest(seen_ids) and receipt['seen_requests'] == number * 100
                and receipt['post_scope'] == ('ALL_SEEN' if number in MILESTONES else 'CURRENT'),
                'NATIVE_CURSOR_PREFIX')
        measured = receipt['native_counts']
        require(measured == expected_counts(arm)
                and all(type(measured[key]) is int and receipt['native']['delta'][key] == measured[key]
                        for key in COUNT_KEYS), 'NATIVE_MEASURED_COUNTS')
        fitting = native_receipt(receipt['native'], arm, number, measured, totals)
        seconds = finite(receipt['seconds'], 'NATIVE_BATCH_SECONDS')
        require(seconds >= 0, 'NATIVE_BATCH_SECONDS_NONNEGATIVE')
        require(receipt['observer_no_mutation'] is True
                and receipt['generation_observer_no_mutation'] is True
                and receipt['checkpoint_saved'] is False and receipt['exact_resume'] == 'NOT_AVAILABLE',
                'NATIVE_NOCP_OBSERVER')
        if arm == 'PRUNE':
            prune_terminal(receipt['native'], number)
        pre = rpn_endpoint(reader, folder / 'pre', identities, ids, f'B{number}_PRE', previous, seen)
        selected, post_ids = (seen, seen_ids) if number in MILESTONES else (current, ids)
        post = rpn_endpoint(reader, folder / 'post', identities, post_ids, f'W{number}', receipt['after'], seen)
        require(pre is not None and post is not None, 'COMMITTED_RPN_OBSERVATIONS_MISSING')
        compare_summary(pre['summary'], receipt['pre'])
        compare_summary(post['summary'], receipt['post'])
        require({key: pre['summary'][key]['denominator'] for key in ('R', 'P', 'N')}
                == dict(R=100, P=200, N=1000), 'PRE_CURRENT100_DENOMINATORS')
        require({key: post['summary'][key]['denominator'] for key in ('R', 'P', 'N')}
                == dict(R=len(selected), P=2 * len(selected), N=10 * len(selected)),
                'POST_SELECTED_DENOMINATORS')
        current_rows = [row for row in post['rows'] if row['case_id'] in set(ids)]
        current_summary = reduce_rows(current_rows)
        compare_summary(current_summary, receipt['post_current'])
        require({key: current_summary[key]['denominator'] for key in ('R', 'P', 'N')}
                == dict(R=100, P=200, N=1000), 'POST_CURRENT100_NOT_PREFIX')
        prestate = config['generation']['W0_state_identity'] if number == 1 else previous
        require(receipt['gen_before']['model_state'] == previous
                and receipt['gen_current']['model_state'] == receipt['after'], 'GEN_ARM_STATE_LINK')
        pregen = generation_endpoint(reader, receipt['gen_before'], current,
            [occurrence[case] for case in ids], prestate, f'B{number}_PRE', 'CURRENT',
            config, attempt, cached_only=number == 1)
        currentgen = generation_endpoint(reader, receipt['gen_current'], current,
            [occurrence[case] for case in ids], receipt['after'],
            f'W{number}_CURRENT' if number in MILESTONES else f'W{number}', 'CURRENT',
            config, attempt, cached_only=number in MILESTONES)
        postgen = currentgen
        if number in MILESTONES:
            require(receipt['gen_current']['derived_subset'] is True
                    and receipt['gen_prefix']['model_state'] == receipt['after'], 'GEN_MILESTONE_SUBSET')
            postgen = generation_endpoint(reader, receipt['gen_prefix'], seen,
                list(range(1, number * 100 + 1)), receipt['after'], f'W{number}', 'ALL_SEEN', config, attempt)
            subset = [row for row in postgen['rows'] if row['case_id'] in set(ids)]
            require(subset == currentgen['rows'], 'GEN_SUBSET_SAME_OBSERVATIONS_NO_REGENERATION')
        else:
            require(receipt['gen_prefix'] is None and receipt['gen_current']['derived_subset'] is False,
                    'GEN_NONMILESTONE_CURRENT_ONLY')
        require(w0gen is not None and receipt['gen_W0']['identity'] == w0gen['identity']
                and receipt['gen_W0']['reference'] == str(out / 'generation-W0-reference.json'),
                'GEN_W0_REFERENCE_LINK')
        # Append report data only after every observation of this commit verifies.
        result['metrics'].extend(_metric_rows(arm, f'B{number}_PRE', number * 100, pre['summary']))
        result['metrics'].extend(_metric_rows(arm, f'W{number}_CURRENT', number * 100, current_summary))
        result['paired'].extend(_paired_rows(arm, f'B{number}_PRE', f'W{number}_CURRENT', pre['rows'], current_rows))
        result['generation'].append(generation_table(arm, f'B{number}_PRE', number * 100, pregen))
        result['generation'].append(generation_table(arm, f'W{number}_CURRENT', number * 100, currentgen))
        at_write.extend(current_rows)
        if number in MILESTONES:
            result['metrics'].extend(_metric_rows(arm, f'W{number}_ALL_SEEN', number * 100, post['summary']))
            result['paired'].extend(_paired_rows(arm, 'AT_WRITE', f'W{number}_ALL_SEEN', at_write, post['rows']))
            if w0 is not None:
                result['paired'].extend(_paired_rows(arm, 'W0', f'W{number}_ALL_SEEN',
                    [row for row in w0['rows'] if row['case_id'] in set(seen_ids)], post['rows']))
            result['generation'].append(generation_table(arm, f'W{number}_ALL_SEEN', number * 100, postgen))
            result['compute'].append(dict(arm=arm, batch=number, phase='generation_current_subset', **currentgen['work']))
            result['generation_current_subset_reductions'] += 1
        for key in COUNT_KEYS:
            totals[key] += measured[key]
        result['counts'].append(dict(arm=arm, batch=number, **measured, **fitting,
            terminal_svd_calls=12 if arm == 'PRUNE' and number == 20 else 0))
        result['compute'].extend((dict(arm=arm, batch=number, phase='batch_inclusive', seconds=seconds,
            nested_timers_not_added_again=True),
            dict(arm=arm, batch=number, phase='pre_RPN', seconds=pre['seconds']),
            dict(arm=arm, batch=number, phase='post_RPN', seconds=post['seconds']),
            dict(arm=arm, batch=number, phase='pre_generation', **pregen['work']),
            dict(arm=arm, batch=number, phase='post_generation', **postgen['work'])))
        result.update(commits=number, requests=number * 100, state_links=max(0, number - 1))
        result['generation_endpoint_calls'] += 2
        result['generation_edit_state_case_observations'] += 100 + len(selected)
        previous = receipt['after']
    require(not (out / 'batch-21' / 'commit.json').exists(), 'NO_BATCH21')
    if terminal:
        require(terminal.get('source') == lock['source_commit'] and terminal.get('config') == digest(config)
                and terminal.get('commits') == terminal.get('completed_batches') == result['commits']
                and terminal.get('checkpoint_saved') is False
                and terminal.get('exact_resume') == 'NOT_AVAILABLE', 'NATIVE_TERMINAL_PROGRESS_SOURCE')
        claimed = terminal.get('native_counts')
        if claimed is not None:
            require(all(type(claimed.get(key)) is int and claimed[key] >= totals[key] for key in COUNT_KEYS),
                    'NATIVE_TERMINAL_COUNTER_LOWER_BOUND')
            result['failed_or_uncommitted_claimed_counts'] = {key: claimed[key] - totals[key] for key in COUNT_KEYS}
        if terminal.get('status') == 'COMPLETED':
            require(claimed is not None and all(claimed[key] == totals[key] for key in COUNT_KEYS)
                    and terminal.get('edits') == 2000 and terminal.get('generation_endpoints') == 40,
                    'NATIVE_COMPLETE_COUNTERS_ENDPOINTS')
    if not result['W0_RPN_available']:
        result['issues'].append('W0_RPN_NOT_AVAILABLE')
    if not result['W0_generation_available']:
        result['issues'].append('W0_GENERATION_NOT_AVAILABLE')
    result['scientific_status'] = terminal_status(terminal, result['commits'],
        result['W0_RPN_available'] and result['W0_generation_available'], result['issues'])
    return result


def collect(attempt):
    attempt, reader = Path(attempt).resolve(), Reader()
    config, lock = reader.json(attempt / 'config.json'), reader.json(attempt / 'execution.lock.json')
    require(config['task_id'] == TASK and config['instruction_id'] == lock['instruction_id'] == NONCE
            and reader.files[str(attempt / 'config.json')]['sha256'] == lock['config_sha256']
            and type(lock['source_commit']) is str and len(lock['source_commit']) == 40,
            'COLLECT_TASK_CONFIG_SOURCE')
    generation = config['generation']
    require(generation['common_source_status'] == 'READY_BOUND'
            and generation['reference_status'] == 'READY_VERIFIED'
            and generation['schema'] == SCHEMA and generation['profile'] == PROFILE
            and generation['eval_seed'] == EVAL_SEED and generation['W0_owner'] == 'BASE_MEMIT'
            and config['noCP'] is True and config['z_disk_cache'] is False,
            'COLLECT_GENERATION_SOURCE_REFERENCE_PROFILE')
    require(lock.get('shared_generation_source') == generation['source_sha']
            and lock.get('shared_generation_tree') == generation['package_tree']
            and lock.get('reference_identity') == generation['reference_assets_sha256'],
            'COLLECT_LOCK_SHARED_SOURCE_REFERENCE')
    # Small sealed source/reference manifests, never model/stat/projector tensors.
    for row in generation.get('shared_source_members', []):
        data = reader.bytes(row['path'])
        require(len(data) == row['bytes'] and reader.files[str(Path(row['path']))]['sha256'] == row['sha256'],
                'COLLECT_SHARED_SOURCE_BYTES')
    identities = reader.bound(config['observer_identity'])
    stream = next(row for row in config['assets'] if row['path'] == config['stream'])
    records = reader.bound(stream)[:2000]
    require(len(records) == 2000 and len({r['case_id'] for r in records}) == 2000
            and [r['case_id'] for r in records] == [case for pack in config['packs'] for case in pack['ids']],
            'COLLECT_FIRST2000_ORDER')
    list(batches(records))
    out = attempt / 'collector'
    require(not out.exists(), 'COLLECT_CREATE_ONCE_OUTPUT')
    out.mkdir()
    reviews = []
    for arm in ARMS:
        progress = {}
        try:
            reviews.append(review_arm(reader, attempt, config, lock, arm, identities, records, progress))
        except Exception as error:
            # Do not publish exception text containing a raw case ID or prompt.
            progress.update(arm=arm, scientific_status='TECHNICAL_BLOCKED_REDUCER',
                error_type=type(error).__name__, valid_prefix_preserved=True, original_raw_preserved=True)
            progress.setdefault('issues', []).append('STORED_DATA_INCONSISTENCY')
            reviews.append(progress)
    ready_path = local_path(generation['W0_cache'], attempt) / 'READY.json'
    w0_ready = None
    if ready_path.exists():
        try:
            ready = reader.json(ready_path)
            require(ready['status'] == 'READY' and ready['producer_arm'] == 'BASE_MEMIT'
                    and ready['identity_sha256'] == digest(ready['identity'])
                    and ready['identity']['runtime'] == digest(runtime_identity(config))
                    and ready['identity']['cold_state_sha256'] == digest(generation['W0_state_identity'])
                    and ready['identity']['ordered_occurrences'] == list(range(1, 2001)),
                    'COLLECT_SINGLE_COLD_W0_READY')
            endpoint = reader.bound(ready['endpoint'])
            require(endpoint['identity_sha256'] == digest(endpoint['identity'])
                    and endpoint['identity']['runtime'] == digest(runtime_identity(config))
                    and endpoint['identity']['state_sha256'] == digest(generation['W0_state_identity'])
                    and endpoint['identity']['endpoint'] == 'W0'
                    and endpoint['identity']['cohort'] == 'FIRST2000'
                    and endpoint['identity']['ordered_occurrences'] == list(range(1, 2001)),
                    'COLLECT_READY_ENDPOINT_COLD_IDENTITY')
            producer_work = ready['producer_work']
            for key in ('new_case_observations', 'cached_case_observations',
                        'generation_forwards', 'full_prefix_token_work'):
                integer(producer_work[key], 'READY_WORK_INTEGER')
            require(producer_work['new_case_observations'] + producer_work['cached_case_observations'] == 2000
                    and finite(producer_work['seconds'], 'READY_WORK_SECONDS') >= 0,
                    'READY_WORK_COLD_COHORT')
            w0_ready = dict(status='READY_VERIFIED', producer_arm='BASE_MEMIT',
                endpoint_identity_sha256=endpoint['identity_sha256'], producer_work=ready['producer_work'])
            for review in reviews:
                if review.get('W0_generation_available'):
                    require(review['W0_generation_identity'] == endpoint['identity_sha256'], 'COLLECT_SAME_W0_ALL_ARMS')
        except Exception as error:
            w0_ready = dict(status='TECHNICAL_BLOCKED_READY', error_type=type(error).__name__)
    for key in TABLE_KEYS:
        _csv(out / (key + '.csv'), [row for review in reviews for row in review.get(key, [])])
    compact = [{key: value for key, value in review.items() if key not in TABLE_KEYS} for review in reviews]
    actual = {key: sum(review.get('measured_counts', {}).get(key, 0) for review in reviews) for key in COUNT_KEYS}
    plan = planned_counts()
    expected = {key: sum(values[key] for values in plan['native_per_arm'].values()) for key in COUNT_KEYS}
    complete = all(review['scientific_status'] == 'COMPLETED_VALIDATED_ROWS_COUNTS' for review in reviews)
    complete = complete and w0_ready is not None and w0_ready['status'] == 'READY_VERIFIED' and actual == expected
    accounting = allocation_once(reader, attempt, lock)
    write(out / 'allocation.json', accounting)
    write(out / 'counts-plan-actual.json', dict(plan=plan, expected_native=expected,
        verified_committed_native=actual, native_z_unit='measured_native_target_fit_calls_not_Adam_updates',
        verified_committed_edit_generation_case_observations=sum(row.get('new_case_observations', 0)
            for review in reviews for row in review.get('compute', [])
            if row['phase'] in ('pre_generation', 'post_generation')),
        recorded_cold_W0_producer_work=(w0_ready['producer_work']
            if w0_ready and w0_ready['status'] == 'READY_VERIFIED' else None),
        failed_or_uncommitted_generation_work='LOCAL_RECEIPTS_PRESERVED_NOT_IN_COMMITTED_TOTAL',
        plan_is_not_actual=not complete, scientific_complete=complete))
    lines = ['# GPT-J 6개 native arm 생성·R/P/N 저장 결과 CPU 검산', '',
        '저장된 raw occurrence·SHA·분모·상태 연결·계수를 검산했다. 모델/토크나이저/fit 호출은 없다.', '',
        '| Arm | 과학 상태 | 검산 batch | 요청 | native fit | solves | H append |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for review in compact:
        count = review.get('measured_counts', {})
        lines.append(f"| {review['arm']} | {review['scientific_status']} | {review.get('commits', 0)} | "
                     f"{review.get('requests', 0)} | {count.get('native_z', 'NA')} | "
                     f"{count.get('solves', 'NA')} | {count.get('history_appends', 'NA')} |")
    lines.extend(['', f"계획: native fit {expected['native_z']}, solves {expected['solves']}, H append {expected['history_appends']}.",
        f"검산 완료 commit 실제: native fit {actual['native_z']}, solves {actual['solves']}, H append {actual['history_appends']}.",
        'native fit은 Adam update가 아니다. 실패/미완료 비용 및 미측정값은 0으로 대체하지 않는다.',
        'Fluency는 bits, consistency는 native cosine이다. 유효 분모가 0이면 평균을 생략한다.',
        'W0 생성은 BASE_MEMIT 한 번만 새로 관측하고, B1 pre와 milestone current는 동일 raw의 CPU subset이다.',
        'PRUNE W5/10/15는 dense, W20만 saved W0 terminal basefix 후 관측이다.',
        'NoCP / exact_resume=NOT_AVAILABLE. collector 완료는 모든 arm 과학 완료와 별개다.',
        'Raw 생성문·토큰·case ID는 local 원본에만 남긴다. Nested timer와 allocation은 합산하지 않는다.', '',
        'Allocation은 자신의 six parent ID만 sacct 1회로 조회하며, 미확보 시 NOT_RECORDED로 남긴다.', '',
        '[R/P/N](metrics.csv) · [paired](paired.csv) · [생성 지표](generation.csv) · [native 계수](counts.csv) · [계산비용](compute.csv)', ''])
    _atomic_text(out / 'report-ko.md', '\n'.join(lines))
    write(out / 'review.json', dict(task=TASK, instruction_id=NONCE, source=lock['source_commit'],
        generation_source=generation['source_sha'], reference_identity=generation['reference_assets_sha256'],
        reviews=compact, W0_generation_READY=w0_ready, scientific_complete=complete,
        new_model_forwards=0, raw_copied=False, raw_publication=False))
    # This potentially large input listing is local only; the compact manifest
    # binds its bytes rather than embedding paths/row identities in the report.
    write(out / 'raw-input-manifest.json', dict(local_only=True, inputs=list(reader.files.values())))
    write(out / 'manifest.json', dict(task=TASK, source=lock['source_commit'],
        input_files=len(reader.files), input_manifest=member(out / 'raw-input-manifest.json'),
        outputs=[member(path) for path in sorted(out.iterdir())
                 if path.is_file() and path.name != 'raw-input-manifest.json'],
        raw_copied=False, raw_input_manifest_local_only=True))
    write(out / 'terminal.json', dict(status='COMPLETED', scientific_complete=complete,
        report=member(out / 'report-ko.md'), source=lock['source_commit'], new_model_forwards=0))
    return dict(status='CPU_REPORT_WRITTEN', scientific_complete=complete, output=str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    print(json.dumps(collect(parser.parse_args().attempt)))
