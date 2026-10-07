"""Independent CPU audit of six sealed native Llama baseline trajectories.

Only stored scalar/text observations are consumed. No model, generator, tensor,
native fit, network upload, scheduler mutation, retry or automatic Git publish.
Accounting is one bounded snapshot of the exact six parents and own collector.
"""
import argparse
import csv
import io
import json
import math
import os
import pwd
import re
import subprocess
import time
from pathlib import Path

from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader, active_flags, compare_summary, harmonic, paired, reduce_rows, validate_rows)
from .common import METHODS, MILESTONES, NONCE, PROFILE, TASK, digest, member, require, sha, write

COUNTERS = ('native_z', 'write_keys', 'history_keys', 'solves', 'history_appends')
HISTORY_SITES = dict(MEMIT=0, PRUNE=0, RECT=0, ALPHAEDIT=5, ALPHAEDIT_BLUE=2, CAKE=5)
KINDS = ('R', 'P', 'N')
REASONS = ('missing_generation_prompts', 'missing_reference', 'zero_generated_vector',
           'zero_reference_vector', 'nonfinite_score', 'length_cap_no_continuation')
FIELDS = ('count', 'success_count', 'success_pct', 'token_acc_pct', 'prompt_acc_pct',
          'strict_acc_pct', 'true_nll', 'new_nll', 'margin_true_minus_new')
_ASSETS = {}
_RESCORED = {}


def finite(value, label, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0), label)
    return value


def expected_rows(identities, ids):
    require(len(ids) == len(set(ids)), 'COLLECT_DUPLICATE_OCCURRENCE')
    indexed = {}
    for row in identities:
        indexed.setdefault(row['case_id'], []).append(row)
    require(all(case in indexed for case in ids), 'COLLECT_MISSING_CASE_IDENTITY')
    return [row for case in ids for row in indexed[case]]


def endpoint(reader, folder, identities, ids, name, state, seen=None):
    """Re-reduce exact ordered raw rows; endpoint filenames prove nothing."""
    folder = Path(folder)
    if not (folder / 'summary.json').is_file():
        return None
    source_state, reused = state, (folder / 'reuse.json').exists()
    if reused:
        receipt = reader.json(folder / 'reuse.json')
        require(name == 'W0' and receipt['scalar_bridge_only'] is True
                and receipt['history_or_editor_resume'] is False
                and receipt['actual_native_state'] == state,
                'COLLECT_W0_SCALAR_STATE_SCOPE')
        manifest = receipt['manifest']
        require(digest(manifest) == receipt['manifest_sha256'], 'COLLECT_W0_REFERENCE_HASH')
        source_state = manifest['cold_state']
        chunks = [reader.bound(row) for row in manifest['chunks']]
    else:
        chunks = [reader.json(path) for path in sorted(folder.glob('chunk-*.json'))]
    require(bool(chunks), 'COLLECT_ENDPOINT_RAW_MISSING')
    raw = []
    for chunk in chunks:
        require(chunk['state'] == source_state and chunk['optimizer_feedback'] is False
                and 0 < len(chunk['rows']) <= 650, 'COLLECT_RAW_STATE_OR_SIZE')
        raw.extend(chunk['rows'])
    validate_rows(raw, expected_rows(identities, ids), name)
    reduced = reduce_rows(raw)
    require({kind: reduced[kind]['denominator'] for kind in KINDS}
            == dict(R=len(ids), P=2 * len(ids), N=10 * len(ids)), 'COLLECT_RPN_DENOMINATORS')
    saved = reader.json(folder / 'summary.json')
    require(saved['endpoint'] == name and saved['state'] == state
            and saved['requests'] == len(ids) and saved['row_count'] == len(raw)
            and saved['row_order'] == digest([row['identity'] for row in raw])
            and saved['no_mutation'] is True and saved['optimizer_feedback'] is False,
            'COLLECT_ENDPOINT_SCOPE')
    compare_summary(reduced, saved['summary'])
    finite(saved['seconds'], 'COLLECT_OBSERVER_SECONDS', True)
    if seen is not None and not reused:
        flags = active_flags(seen)
        require(all(type(row['active_at_endpoint']) is bool
                    and row['active_at_endpoint'] == flags[row['case_id']] for row in raw),
                'COLLECT_ACTIVE_SEEN_PREFIX')
    if reused:
        require(saved['seconds'] == 0 and saved['new_forwards'] == 0
                and saved['reference_only'] is True, 'COLLECT_W0_REUSED_COST')
    return dict(rows=raw, summary=reduced, state=state, seconds=saved['seconds'],
                reference_only=reused,
                original_evaluation_seconds=saved.get('original_evaluation_seconds'))


def scalar_rpn(prefix, summary, count):
    """Independent scalar mapping, N desired=true and harmonic in percent."""
    require(set(summary) == set(KINDS), 'COLLECT_RPN_KINDS')
    result = {}
    for kind, factor in (('R', 1), ('P', 2), ('N', 10)):
        value = summary[kind]
        require(value['denominator'] == count * factor, 'COLLECT_PAYLOAD_DENOMINATOR')
        fields = (value['denominator'], value['numerator'], 100 * value['rate'],
                  100 * value['token_micro'], 100 * value['prompt_macro'],
                  100 * value['strict_rate'], value['true_nll_mean'],
                  value['new_nll_mean'], value['true_minus_new_mean'])
        result.update({prefix + '/' + kind + '/' + field: item
                       for field, item in zip(FIELDS, fields)})
    result[prefix + '/success_harmonic_pct'] = 100 * harmonic(summary)
    return result


def generation_scalar(prefix, aggregate):
    require(type(aggregate['planned_count']) is int and aggregate['planned_count'] > 0,
            'COLLECT_GENERATION_PLANNED')
    result = {prefix + '/generation/planned_count': aggregate['planned_count']}
    for kind, field in (('fluency', 'ngram_entropy'), ('consistency', 'reference_score')):
        count, total = aggregate[kind + '_count'], aggregate[kind + '_sum']
        require(type(count) is int and 0 <= count <= aggregate['planned_count'],
                'COLLECT_GENERATION_VALID_COUNT')
        finite(total, 'COLLECT_GENERATION_SUM', True)
        require(count > 0 or total == 0, 'COLLECT_MISSING_HAS_SUM')
        if kind == 'consistency':
            # Match the frozen shared scalar roundoff policy. Preserve raw
            # values; this is not clipping or a result-driven method tolerance.
            require(total <= count * (1 + 4 * math.ulp(1.)), 'COLLECT_COSINE_SUM_RANGE')
        result[prefix + '/generation/' + kind + '_count'] = count
        if count:
            result[prefix + '/' + kind + '/' + field] = total / count
    for field in ('generation_prompt_count', 'generated_token_count'):
        value = aggregate[field]
        require(type(value) is int and value >= 0, 'COLLECT_GENERATION_COUNT')
        result[prefix + '/generation/' + field] = value
    require(set(aggregate['reason_counts']) <= set(REASONS), 'COLLECT_GENERATION_REASON')
    for reason, count in aggregate['reason_counts'].items():
        require(type(count) is int and count >= 0, 'COLLECT_GENERATION_REASON_COUNT')
        result[prefix + '/generation/missing_' + reason + '_count'] = count
    return result


def exact_scalars(actual, stored):
    require(set(actual) == set(stored), 'COLLECT_SCALAR_KEYS')
    for key, value in actual.items():
        got = stored[key]
        require(type(value) is type(got) or (type(value) in (int, float)
                and type(got) in (int, float)), 'COLLECT_SCALAR_TYPE:' + key)
        if type(value) in (int, float):
            require(math.isfinite(got) and math.isclose(got, value, rel_tol=1e-12, abs_tol=1e-12),
                    'COLLECT_SCALAR_VALUE:' + key)
        else:
            require(got == value, 'COLLECT_SCALAR_VALUE:' + key)


def native_counts(native, method, number, running):
    sites, history = (2 if method == 'ALPHAEDIT_BLUE' else 5), HISTORY_SITES[method]
    expected = dict(native_z=100 * sites if method == 'ALPHAEDIT_BLUE' else 100,
                    write_keys=sites, history_keys=history, solves=sites, history_appends=history)
    require(native['method'] == method and native['batch'] == number
            and native['requests'] == 100 and native['same_model_returned'] is True
            and native['caller_history_appends'] == 0 and native['checkpoint_saved'] is False
            and native['cache_template'] is None, 'COLLECT_NATIVE_APPLY_IDENTITY')
    require({key: native['counts'][key] for key in COUNTERS} == expected,
            'COLLECT_NATIVE_BATCH_COUNTS')
    for key in COUNTERS:
        require(type(native['counts'][key]) is int, 'COLLECT_NATIVE_INTEGER_COUNTER')
        require(native['cumulative'][key] == running[key] + expected[key],
                'COLLECT_NATIVE_CUMULATIVE_COUNTER')
    return expected


def fit_events(reader, receipt, native, number, prior_forwards, prior_updates):
    ref = receipt['member']
    data = reader.bytes(ref['path'])
    require(len(data) == ref['bytes'] and sha(ref['path']) == ref['sha256']
            and len(data) <= 32 * 1024**2 and receipt['authoritative_stream'] is True,
            'COLLECT_FIT_AUTHORITATIVE_MEMBER')
    events = [json.loads(line) for line in data.splitlines() if line]
    terminal = [row for row in events if 'evaluations' in row]
    trace = native['counts']['fit_trace']
    require(receipt['line_count'] == len(events) == native['counts']['fit_forwards'] + native['counts']['native_z']
            and len(terminal) == len(trace) == native['counts']['native_z'], 'COLLECT_FIT_EVENT_COUNTS')
    require([row['request_index'] for row in trace] == list(range(1, len(trace) + 1)),
            'COLLECT_NATIVE_FIT_REQUEST_ORDER')
    for event, stored in zip(terminal, trace):
        require(event['batch'] == number and all(event[key] == value for key, value in stored.items()),
                'COLLECT_FIT_TERMINAL_TRACE')
        evaluations, updates = event['evaluations'], event['Adam_updates']
        require(type(evaluations) is type(updates) is int and 1 <= evaluations <= 25
                and updates == evaluations - 1 and updates <= 24
                and event['stop'] == ('TOTAL_LOSS_BELOW_005' if event['loss'] < .05 else 'BUDGET_EXHAUSTED'),
                'COLLECT_ORIGINAL_NATIVE_FIT_BUDGET')
        for field in ('loss', 'nll_loss', 'kl_loss', 'weight_decay'):
            finite(event[field], 'COLLECT_FIT_FINITE')
    require(sum(row['evaluations'] for row in terminal) == native['counts']['fit_forwards']
            and sum(row['Adam_updates'] for row in terminal) == native['counts']['fit_updates'],
            'COLLECT_FIT_TOTALS')
    previous = prior_forwards
    for event in events:
        require(event['batch'] == number and type(event['fit_global_candidate']) is int
                and previous <= event['fit_global_candidate'] <= prior_forwards + native['counts']['fit_forwards'],
                'COLLECT_FIT_MONOTONIC_AXIS')
        previous = event['fit_global_candidate']
    require(previous == prior_forwards + native['counts']['fit_forwards']
            and native['cumulative']['fit_forwards'] == previous
            and native['cumulative']['fit_updates'] == prior_updates + native['counts']['fit_updates'],
            'COLLECT_FIT_GLOBAL_COUNTERS')
    return dict(native_fit_evaluations=native['counts']['fit_forwards'],
                native_Adam_updates=native['counts']['fit_updates'], event_lines=len(events))


def generation_reduce(case_rows, ids):
    """Reaggregate normalized per-case scalar records, never score missing as0.

    The generation adapter supplies original per-case values with explicit
    validity. An optional pure-metric rescore runs before this independent sum.
    No generated text or case identifiers are returned in compact reports.
    """
    require([row['case_id'] for row in case_rows] == ids, 'COLLECT_GENERATION_CASE_ORDER')
    result = dict(planned_count=len(ids), fluency_count=0, consistency_count=0,
                  fluency_sum=0., consistency_sum=0., generation_prompt_count=0,
                  generated_token_count=0, reason_counts={})
    for row in case_rows:
        for kind, field in (('fluency', 'ngram_entropy'), ('consistency', 'reference_score')):
            value = row[field]
            require(value is None or type(value) in (int, float), 'COLLECT_GENERATION_VALUE_TYPE')
            if value is not None:
                finite(value, 'COLLECT_GENERATION_VALUE', True)
                require(kind != 'consistency' or value <= 1 + 4 * math.ulp(1.),
                        'COLLECT_GENERATION_COSINE')
                result[kind + '_count'] += 1
        for field in ('generation_prompt_count', 'generated_token_count'):
            require(type(row[field]) is int and row[field] >= 0, 'COLLECT_GENERATION_CASE_COUNT')
            result[field] += row[field]
        reasons = row['reason_counts']
        require(type(reasons) is dict and set(reasons) <= set(REASONS), 'COLLECT_GENERATION_CASE_REASON')
        for reason, count in reasons.items():
            require(type(count) is int and count >= 0, 'COLLECT_GENERATION_CASE_REASON_COUNT')
            result['reason_counts'][reason] = result['reason_counts'].get(reason, 0) + count
    result['fluency_sum'] = math.fsum(row['ngram_entropy'] for row in case_rows
                                    if row['ngram_entropy'] is not None)
    result['consistency_sum'] = math.fsum(row['reference_score'] for row in case_rows
                                        if row['reference_score'] is not None)
    return result


def _metric_rows(method, endpoint_name, edits, groups):
    for kind, value in groups.items():
        yield dict(method=method, endpoint=endpoint_name, edits=edits, kind=kind,
            count=value['denominator'], success_count=value['numerator'],
            success_pct=100 * value['rate'], token_count=value['desired_token_count'],
            token_correct=value['desired_token_correct'], token_acc_pct=100 * value['token_micro'],
            prompt_acc_pct=100 * value['prompt_macro'], strict_count=value['strict_numerator'],
            strict_acc_pct=100 * value['strict_rate'], true_nll=value['true_nll_mean'],
            new_nll=value['new_nll_mean'], desired_nll=value['desired_nll_mean'],
            margin_true_minus_new=value['true_minus_new_mean'],
            success_harmonic_pct=100 * harmonic(groups) if harmonic(groups) is not None else None)


def _paired_rows(method, before_name, after_name, before, after, cohort=None):
    for kind, values in paired(before, after).items():
        for metric, value in values.items():
            yield dict(method=method, before_endpoint=before_name, after_endpoint=after_name,
                       cohort=cohort, kind=kind, metric=metric, **value)


def _identity(value, method, lock, config, job, label):
    require(value['source'] == lock['source_commit'] and value['config'] == lock['config_sha256'], label)
    identity = value.get('identity', value)
    require(str(identity.get('job', identity.get('job_id'))) == job, label + '_JOB')
    # The runner records public method/model/task identity, never relies on name.
    require(identity.get('method') == method and identity.get('model') == 'llama3'
            and identity.get('task_id') == TASK, label + '_MODEL_METHOD')


def read_generation(reader, ref, c, ids, state, endpoint_name, rescore=True):
    """Lazy CPU-only adapter; it must not instantiate a model or generator.

    The production runner owns the exact shared package binding. This reader is
    imported only when that binding exists; its contract is checked explicitly.
    """
    from .generation import model_identity, normalize_summary, read_observed, runtime_identity
    require(reader.bound(ref['member'])['identity'] == ref['identity'],
            'COLLECT_GENERATION_ENDPOINT_MEMBER')
    result = read_observed(ref, c)
    identity = result['identity']
    full_weights = dict(c['cold_W'], **state['W'])
    expected_state = dict(W=full_weights, model_identity=model_identity(c))
    require(identity['state_sha256'] == digest(expected_state)
            and identity['runtime'] == digest(runtime_identity(c))
            and identity['endpoint'] == endpoint_name
            and identity['ordered_occurrences'] == [c['_collector_occurrences'][str(case)] for case in ids]
            and [row['case_id'] for row in result['rows']] == ids
            and result['RNG_restored'] is True and result['observer_no_mutation'] is True,
            'COLLECT_GENERATION_ENDPOINT_SCOPE')
    normalized, rescored_rows = [], []
    assets = None
    if rescore:
        from project.run_scripts.experiment_generation_eval.assets import load_assets
        from project.run_scripts.experiment_generation_eval.metrics import score_case, reduce_cases
        key = c['generation']['assets_sha256']
        if key not in _ASSETS:
            _ASSETS[key] = load_assets(dict(reference_assets=c['generation']['reference_manifest'],
                                           asset_paths=c['generation']['asset_paths']))
        assets = _ASSETS[key]
        require(assets.sha == key, 'COLLECT_RESCORE_FROZEN_REFERENCE_IDENTITY')
    for row in result['rows']:
        raw = reader.json(row['observation_path'])
        require(raw['payload_sha256'] == row['payload_sha256']
                and raw['case_id'] == row['case_id'] and raw['occurrence'] == row['occurrence']
                and raw['identity_sha256'] == row['identity_sha256']
                and raw['identity']['state_identity'] == expected_state
                and raw['metrics'] == row['metrics'], 'COLLECT_GENERATION_RAW_IDENTITY')
        metric = row['metrics']
        if assets is not None:
            key = (assets.sha, raw['payload_sha256'])
            if key not in _RESCORED:
                record = raw['identity']['record_identity']
                _RESCORED[key] = score_case(raw['observations'],
                    assets.snippets_for(record['relation_id'], record['target_new_id']),
                    assets.vectorizer, word_tokenize=assets.word_tokenize)
            rescored = _RESCORED[key]
            require(set(rescored) == set(metric), 'COLLECT_RESCORE_METRIC_SCHEMA')
            for field, value in metric.items():
                if type(value) in (float, int) and type(value) is not bool:
                    require(math.isclose(rescored[field], value, rel_tol=1e-12, abs_tol=1e-12),
                            'COLLECT_RESCORE_VALUE:' + field)
                else:
                    require(rescored[field] == value, 'COLLECT_RESCORE_VALUE:' + field)
            rescored_rows.append(dict(metrics=rescored))
        require(metric['fluency_valid'] == (metric['ngram_entropy'] is not None)
                and metric['consistency_valid'] == (metric['reference_score'] is not None),
                'COLLECT_GENERATION_VALIDITY')
        reasons = {reason: metric['reasons'].count(reason) for reason in REASONS}
        require(set(metric['reasons']) <= set(REASONS), 'COLLECT_GENERATION_UNREADY_ASSET_REASON')
        normalized.append(dict(case_id=row['case_id'], ngram_entropy=metric['ngram_entropy'],
            reference_score=metric['reference_score'], generation_prompt_count=metric['generation_prompt_count'],
            generated_token_count=metric['generated_token_count'], reason_counts=reasons))
    reduced = generation_reduce(normalized, ids)
    exact_scalars(reduced, normalize_summary(result['summary']))
    if assets is not None:
        exact_scalars(reduced, normalize_summary(reduce_cases(rescored_rows)))
    return dict(summary=reduced, rows=normalized, work=ref.get('work', {}),
                pure_metric_rescore='PERFORMED_CACHED_FROZEN_REFERENCE' if assets is not None
                else 'NOT_PERFORMED; immutable per-case metrics independently summed')


def review_arm(reader, attempt, c, lock, method, job, identities, records,
               generation_reader=None, progress=None):
    """Preserve the last fully audited atomic commit prefix on later defects."""
    out = Path(attempt) / method
    ids_all = [row['case_id'] for row in records]
    selected_sites = [str(layer) for layer in c['native'][method]['layers']]
    metrics_table, generation_table, paired_table, costs, counters = [], [], [], [], []
    result = dict(method=method, job_id=job, scientific_status='NOT_AVAILABLE', commits=0,
        requests=0, state_links=0, measured_native_counts={key: 0 for key in COUNTERS},
        expected_commits=20, expected_requests=2000, expected_state_links=19,
        expected_history_appends=20 * HISTORY_SITES[method], W0_available=False,
        W0_generation_available=False,
        missing=[], metric_rows=metrics_table, generation_rows=generation_table,
        paired_rows=paired_table, compute_rows=costs, counter_rows=counters,
        new_model_forwards=0, exact_resume='NOT_AVAILABLE')
    if progress is not None:
        progress.update(result)
    if not out.is_dir():
        result['missing'].append('ARM_OUTPUT_NOT_AVAILABLE')
        return result
    require(not (out / 'batch-21').exists(), 'COLLECT_FORBIDDEN_B21')
    w0_saved = reader.json(out / 'W0/summary.json') if (out / 'W0/summary.json').exists() else None
    cold = None if w0_saved is None else w0_saved['state']
    if cold is not None:
        require(cold['W'] == {key: c['cold_W'][key] for key in selected_sites}, 'COLLECT_COLD_SELECTED_W')
        require(not cold['H'] if not HISTORY_SITES[method] else
                (not cold['H'] and method == 'ALPHAEDIT') or
                set(cold['H']) == set(selected_sites), 'COLLECT_COLD_NATIVE_H_SCOPE')
        w0 = endpoint(reader, out / 'W0', identities, ids_all, 'W0', cold, records)
        result['W0_available'] = True
        metrics_table.extend(_metric_rows(method, 'W0_FIRST2000', 0, w0['summary']))
        costs.append(dict(method=method, phase='W0_RPN', batch=0, seconds=w0['seconds'],
                          reference_only=w0['reference_only'], old_reference_cost_excluded=True))
    else:
        w0 = None
        result['missing'].append('W0_RPN_NOT_AVAILABLE')
    gen_reader = read_generation if generation_reader is None else generation_reader
    if (out / 'generation-W0.json').exists() and cold is not None:
        gen_w0 = gen_reader(reader, reader.json(out / 'generation-W0.json'), c,
                            ids_all, cold, 'W0_first2000')
        generation_table.append(dict(method=method, batch=0, endpoint='W0_first2000',
            **gen_w0['summary'], pure_metric_rescore=gen_w0.get('pure_metric_rescore')))
        result['W0_generation_available'] = True
    else:
        result['missing'].append('W0_GENERATION_NOT_AVAILABLE')
    commits, at_write, prefixes = [], [], {}
    fit_forwards, fit_updates = 0, 0
    for number in range(1, 21):
        folder = out / ('batch-%02d' % number)
        if not (folder / 'commit.json').exists():
            result['missing'].append('UNCOMMITTED_PREFIX_END_B' + str(number))
            require(not any((out / ('batch-%02d' % later) / 'commit.json').exists()
                            for later in range(number + 1, 21)), 'COLLECT_NONCONTIGUOUS_COMMIT_PREFIX')
            break
        require(not (folder / 'rollback.json').exists(), 'COLLECT_COMMIT_AND_ROLLBACK_CONFLICT')
        commit, entry = reader.json(folder / 'commit.json'), reader.json(folder / 'entry.json')
        _identity(commit, method, lock, c, job, 'COLLECT_COMMIT_IDENTITY')
        _identity(entry, method, lock, c, job, 'COLLECT_ENTRY_IDENTITY')
        ids = ids_all[(number - 1) * 100:number * 100]
        seen_ids, seen = ids_all[:number * 100], records[:number * 100]
        before, after = commit['state_before'], commit['state_after']
        expected_before = cold if not commits else commits[-1]['state_after']
        require(expected_before is not None and before == entry['state'] == expected_before
                and entry['ids'] == ids and entry['seen_ids'] == seen_ids
                and commit['batch'] == entry['batch'] == number
                and commit['requests'] == entry['requests'] == 100
                and commit['transaction_finished'] is True, 'COLLECT_COMMIT_ENTRY_PREFIX')
        require(set(after['W']) == set(selected_sites) and
                (set(after['H']) == set(selected_sites) if HISTORY_SITES[method] else after['H'] == {}),
                'COLLECT_NATIVE_AFTER_STATE_SCOPE')
        require(commit['RNG_before'] == entry['RNG'] and commit['entry_context'] == entry['context']
                and commit['ledger_before'] == entry['ledger'], 'COLLECT_ENTRY_LOGICAL_STATE')
        if commits:
            previous = commits[-1]
            require(entry['RNG'] == previous['RNG_after']
                    and entry['context'] == previous['exit_context']
                    and entry['ledger'] == previous['ledger_after'], 'COLLECT_19_OWN_STATE_JOINS')
        native = reader.bound(commit['native'])
        binding = c['native'][method]
        require(native['layers'] == binding['layers']
                and native['native_source_commit'] == binding['source_commit']
                and native['native_has_history'] is binding['native_has_history']
                and all(native['hparams'][key] == value for key, value in binding['scientific_fields'].items())
                and native['context']['contexts_sha256'] == binding['context_reuse']['contexts']['sha256'],
                'COLLECT_NATIVE_SCIENCE_SOURCE_PARAMETERS')
        delta = native_counts(native, method, number, result['measured_native_counts'])
        measured_fit = fit_events(reader, commit['events'], native, number, fit_forwards, fit_updates)
        require({key: commit['native_counts'][key] for key in COUNTERS} == delta
                and commit['history_appends'] == delta['history_appends'],
                'COLLECT_NATIVE_COMMIT_COUNTS')
        if method == 'PRUNE' and number == 20:
            prune = reader.json(folder / 'prune-terminal.json')
            require(prune['status'] == 'PRUNE_TERMINAL_BASE_FIX_APPLIED'
                    and len(prune['rows']) == 5
                    and all(row['explicit_repair'] == 'PRUNE_TERMINAL_BASE_FIX'
                            and row['base'] == 'RAM_COLD_W0' and row['double_add'] is False
                            and row['checkpoint_saved'] is False for row in prune['rows']),
                    'COLLECT_PRUNE_COLD_BASE_FIX')
        pre = endpoint(reader, folder / 'pre', identities, ids, 'B%d_PRE' % number, before, seen)
        observed_ids = seen_ids if number in MILESTONES else ids
        post_label = ('W%d_ALLSEEN' if number in MILESTONES else 'B%d_POST') % number
        post = endpoint(reader, folder / 'post', identities, observed_ids, post_label, after, seen)
        require(pre is not None and post is not None, 'COLLECT_COMMITTED_OBSERVATION_MISSING')
        compare_summary(pre['summary'], commit['pre'])
        compare_summary(post['summary'], commit['post'])
        current_rows = [row for row in post['rows'] if row['case_id'] in set(ids)]
        current = reduce_rows(current_rows)
        compare_summary(current, commit['post_current'])
        scalars = dict(edits=number * 100, batch=number,
                       pre_state_edits=(number - 1) * 100, post_state_edits=number * 100)
        scalars.update(scalar_rpn('current/pre', pre['summary'], 100))
        scalars.update(scalar_rpn('current/post', current, 100))
        if number in MILESTONES:
            scalars.update(scalar_rpn('all_seen/post', post['summary'], number * 100))
        generation_receipt = reader.bound(commit['generation'])
        require(generation_receipt['task_id'] == TASK and generation_receipt['method'] == method
                and generation_receipt['model'] == 'llama3' and generation_receipt['batch'] == number
                and generation_receipt['native_history_identity_separate'] == after['H']
                and generation_receipt['raw_local_only'] is True,
                'COLLECT_GENERATION_LINKS_SCOPE')
        generation = generation_receipt['endpoints']
        gen_groups = [('pre', 'current/pre', ids, before, 'B%d_PRE' % number),
                      ('current', 'current/post', ids, after, 'B%d_POST' % number),
                      ('w0_current', 'w0/current', ids, cold, 'W0_B%d_CURRENT' % number)]
        if number in MILESTONES:
            gen_groups += [('all_seen', 'all_seen/post', seen_ids, after, 'W%d_ALLSEEN' % number),
                           ('w0_all_seen', 'w0/all_seen', seen_ids, cold, 'W0_W%d_ALLSEEN' % number)]
        else:
            require(generation.get('all_seen') is None and generation.get('w0_all_seen') is None,
                    'COLLECT_UNMEASURED_GENERATION_MILESTONE')
        batch_gen = []
        for key, prefix, cohort, gen_state, label in gen_groups:
            observed = gen_reader(reader, generation[key], c, cohort, gen_state, label)
            scalars.update(generation_scalar(prefix, observed['summary']))
            batch_gen.append(dict(method=method, batch=number, endpoint=prefix,
                                  **observed['summary'], pure_metric_rescore=observed.get('pure_metric_rescore')))
            work = observed.get('work', {})
            require(all(type(value) in (str, int, float, bool) or value is None for value in work.values()),
                    'COLLECT_GENERATION_COMPACT_WORK_ONLY')
            costs.append(dict(method=method, batch=number, phase='generation_' + prefix,
                              **{'work_' + key: value for key, value in work.items()},
                              nested_timers_not_added=True))
        if number in MILESTONES:
            first500 = gen_reader(reader, generation['first500'], c, ids_all[:500], after,
                                  'W%d_FIRST500' % number)
            batch_gen.append(dict(method=method, batch=number, endpoint='first500',
                **first500['summary'], pure_metric_rescore=first500.get('pure_metric_rescore')))
        if w0 is not None:
            for prefix, cohort in [('w0/current/N', ids)] + (
                    [('w0/all_seen/N', seen_ids)] if number in MILESTONES else []):
                groups = reduce_rows([row for row in w0['rows'] if row['case_id'] in set(cohort)])
                mapped = scalar_rpn('_', groups, len(cohort))
                scalars.update({prefix + key[len('_/N'):]: value for key, value in mapped.items()
                                if key.startswith('_/N/')})
        exact_scalars(scalars, reader.bound(commit['metrics']))
        # Advance verified prefix only after every raw/identity/producer check.
        commits.append(commit)
        metrics_table.extend(_metric_rows(method, 'B%d_PRE' % number, number * 100, pre['summary']))
        metrics_table.extend(_metric_rows(method, 'W%d_CURRENT' % number, number * 100, current))
        paired_table.extend(_paired_rows(method, 'B%d_PRE' % number, 'W%d_CURRENT' % number,
                                         pre['rows'], current_rows))
        at_write.extend(current_rows)
        if number in MILESTONES:
            prefixes[number] = post['rows']
            metrics_table.extend(_metric_rows(method, 'W%d_ALL_SEEN' % number, number * 100, post['summary']))
            fixed500 = [row for row in post['rows'] if row['case_id'] in set(ids_all[:500])]
            metrics_table.extend(_metric_rows(method, 'W%d_FIRST500' % number, number * 100, reduce_rows(fixed500)))
            paired_table.extend(_paired_rows(method, 'AT_WRITE', 'W%d_ALL_SEEN' % number, at_write, post['rows']))
            for birth in range(1, number + 1):
                cohort = set(ids_all[(birth - 1) * 100:birth * 100])
                paired_table.extend(_paired_rows(method, 'B%d_AT_WRITE' % birth, 'W%d' % number,
                    [row for row in at_write if row['case_id'] in cohort],
                    [row for row in post['rows'] if row['case_id'] in cohort], birth))
            if w0 is not None:
                paired_table.extend(_paired_rows(method, 'W0', 'W%d_ALL_SEEN' % number,
                    [row for row in w0['rows'] if row['case_id'] in set(seen_ids)], post['rows']))
        generation_table.extend(batch_gen)
        counters.append(dict(method=method, batch=number, **delta, **measured_fit))
        fit_forwards += measured_fit['native_fit_evaluations']
        fit_updates += measured_fit['native_Adam_updates']
        for key in COUNTERS:
            result['measured_native_counts'][key] += delta[key]
        costs += [dict(method=method, batch=number, phase='pre_RPN', seconds=pre['seconds']),
                  dict(method=method, batch=number, phase='post_RPN', seconds=post['seconds']),
                  dict(method=method, batch=number, phase='native_apply_inclusive', seconds=native['seconds'],
                       nested_timers_not_added=True),
                  dict(method=method, batch=number, phase='batch_inclusive', seconds=commit['seconds'],
                       nested_timers_not_added=True)]
        for phase, seconds in native['counts']['seconds'].items():
            finite(seconds, 'COLLECT_NATIVE_NESTED_SECONDS', True)
            costs.append(dict(method=method, batch=number, phase='native_nested_' + phase,
                              seconds=seconds, nested_timers_not_added=True))
        result.update(commits=len(commits), requests=len(commits) * 100,
                      state_links=max(0, len(commits) - 1), scientific_status='PARTIAL_VALIDATED_PREFIX')
        if progress is not None:
            progress.update(result)
    terminal = reader.json(out / 'terminal.json') if (out / 'terminal.json').exists() else None
    failed = reader.json(out / 'failed.json') if (out / 'failed.json').exists() else None
    if (out / 'cost.json').exists():
        program_cost = reader.json(out / 'cost.json')
        for field in ('wall_seconds', 'host_peak_rss_bytes', 'actual_GPU_peak_allocated_bytes',
                      'actual_GPU_peak_reserved_bytes'):
            if field in program_cost:
                finite(program_cost[field], 'COLLECT_PROGRAM_COST', True)
        costs.append(dict(method=method, phase='program_inclusive',
            **{key: value for key, value in program_cost.items()
               if key in ('wall_seconds', 'host_peak_rss_bytes', 'actual_GPU_peak_allocated_bytes',
                          'actual_GPU_peak_reserved_bytes')}, nested_timers_not_added=True))
    if terminal is not None:
        _identity(terminal, method, lock, c, job, 'COLLECT_TERMINAL_IDENTITY')
        require(terminal['commits'] == len(commits) and terminal['requests'] == len(commits) * 100,
                'COLLECT_TERMINAL_PREFIX')
        require(terminal.get('complete') is True and len(commits) == 20
                and terminal['state'] == commits[-1]['state_after']
                and terminal['history_appends'] == 20 * HISTORY_SITES[method]
                and all(terminal['counts'][key] == result['measured_native_counts'][key] for key in COUNTERS),
                'COLLECT_TERMINAL_COMPLETE_EVIDENCE')
        listed = terminal.get('commit_receipts')
        require(listed is not None and len(listed) == 20, 'COLLECT_TERMINAL_COMMIT_MEMBERS')
        for number, ref in enumerate(listed, 1):
            require(Path(ref['path']) == out / ('batch-%02d' % number) / 'commit.json'
                    and reader.bound(ref) == commits[number - 1], 'COLLECT_TERMINAL_ATOMIC_COMMIT_BINDING')
        require(failed is None and result['W0_available'] and result['W0_generation_available'],
                'COLLECT_TERMINAL_NO_FAILURE_OR_MISSING_W0')
        result['scientific_status'] = 'COMPLETED_VALIDATED'
    elif failed is not None:
        require(failed['source'] == lock['source_commit'] and str(failed['job']) == job
                and failed['method'] == method and failed['model'] == 'llama3'
                and failed['task_id'] == TASK, 'COLLECT_FAILURE_IDENTITY')
        require(failed['technical_block'] is True and failed['committed_prefix'] >= len(commits),
                'COLLECT_FAILURE_PREFIX')
        result['runner_failure'] = {key: failed.get(key) for key in
            ('technical_block', 'committed_prefix', 'error_type', 'rollback_verified')}
        if 'config' in failed:
            require(failed['config'] == lock['config_sha256'], 'COLLECT_FAILURE_CONFIG')
        result['runner_failure']['config_identity'] = (
            'VERIFIED' if 'config' in failed else 'NOT_RECORDED_IN_FAILURE_RECEIPT')
        result['missing'].append('RUNNER_TECHNICAL_FAILURE; unverified suffix not zero-imputed')
    else:
        result['missing'].append('TERMINAL_NOT_AVAILABLE')
    return result


def allocation_once(reader, attempt, lock, submission):
    """Exact seven parents; steps never add GPU-sec a second time."""
    queries = 0
    try:
        jobs = submission['jobs']
        require(set(jobs) == set(METHODS) | {'collector'}, 'ACCOUNTING_EXACT_SEVEN_KEYS')
        require(all(type(job) is str and re.fullmatch(r'[1-9][0-9]*', job) for job in jobs.values())
                and len(set(jobs.values())) == 7, 'ACCOUNTING_EXACT_SEVEN_IDS')
        owner = pwd.getpwuid(os.getuid()).pw_name
        require(lock['owner'] == owner, 'ACCOUNTING_EXACT_OWNER')
        argv = ['sacct', '-n', '-P', '-j', ','.join(jobs.values()),
                '--format=JobIDRaw,JobName%120,User,State,ExitCode,ElapsedRaw,AllocTRES']
        queries = 1
        response = subprocess.run(argv, capture_output=True, text=True, timeout=20, check=False)
        require(response.returncode == 0 and len(response.stdout) <= 1024**2,
                'ACCOUNTING_BOUNDED_AVAILABLE')
        records = {}
        for line in response.stdout.splitlines():
            fields = [field.strip() for field in line.split('|')]
            if not line.strip():
                continue
            if len(fields) == 8 and fields[-1] == '':
                fields.pop()
            require(len(fields) == 7, 'ACCOUNTING_COLUMNS')
            job, name, user, status, exit_code, elapsed, tres = fields
            if job not in jobs.values():
                require(any(job.startswith(parent + '.') for parent in jobs.values()),
                        'ACCOUNTING_UNEXPECTED_ROW')
                continue
            method = next(key for key, parent in jobs.items() if parent == job)
            require(job not in records and user == owner and elapsed.isdigit()
                    and name == TASK + '-' + method, 'ACCOUNTING_PARENT_IDENTITY')
            resources = dict(item.split('=', 1) for item in tres.split(',') if '=' in item)
            gpu = resources.get('gres/gpu')
            if gpu is None:
                typed = [value for key, value in resources.items() if key.startswith('gres/gpu:')]
                require(len(typed) <= 1, 'ACCOUNTING_GPU_SCHEMA')
                gpu = typed[0] if typed else '0'
            require(gpu in ('0', '1') and (gpu == '0' if method == 'collector' else True),
                    'ACCOUNTING_GPU_BOUND')
            seconds = int(elapsed)
            records[job] = dict(method=method, job_id=job, owner=user, scheduler_state=status,
                exit_code=exit_code, allocated_GPUs=int(gpu), parent_elapsed_seconds=seconds,
                allocated_GPU_seconds=int(gpu) * seconds, AllocTRES=tres,
                child_steps_excluded=True, pending_is_not_GPU_allocation=True,
                collector_elapsed_is_snapshot_not_completed_CPU_cost=method == 'collector')
        require(set(records) == set(jobs.values()), 'ACCOUNTING_SEVEN_PARENT_ROWS')
        return dict(status='RECORDED', queries=1, scope='EXACT_SIX_GPU_PLUS_OWN_CPU_PARENTS',
                    records=[records[jobs[key]] for key in (*METHODS, 'collector')],
                    program_nested_timers_not_added=True, scheduler_success_not_scientific_success=True)
    except Exception as error:
        return dict(status='NOT_RECORDED', queries=queries, error_type=type(error).__name__,
                    reason='OWN_ACCOUNTING_UNAVAILABLE_OR_IDENTITY_MISMATCH', no_retry=True)


def _text(path, content):
    path = Path(path)
    data = content.encode('utf-8')
    require(not path.exists() and len(data) <= 32 * 1024**2, 'COLLECT_CREATE_ONCE_COMPACT')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path); temporary.unlink()


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields or ['availability'], lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    _text(path, stream.getvalue())


def collect(attempt, accounting=True):
    started, cpu_started = time.monotonic(), time.process_time()
    attempt = Path(attempt).resolve()
    reader = Reader()
    lock, c = reader.json(attempt / 'execution.lock.json'), reader.json(attempt / 'config.json')
    require(lock['instruction_id'] == NONCE and lock['task_id'] == c['task_id'] == TASK
            and lock['config_sha256'] == sha(attempt / 'config.json')
            and lock['host'] == 'server4' and lock['noCP'] is True and c['noCP'] is True,
            'COLLECT_SEALED_SOURCE_CONFIG')
    require(bool(c['generation'].get('source_members')) and c['generation'].get('READY') is not None
            and c['generation'].get('reference_manifest') is not None, 'COLLECT_GENERATION_NOT_BOUND')
    submission = reader.json(attempt / 'submission.json')
    require(submission['nonce'] == NONCE and submission['task_id'] == TASK
            and submission['source'] == lock['source_commit'] and submission['status'] == 'RELEASED'
            and reader.bound(submission['lock']) == lock
            and set(submission['jobs']) == set(METHODS) | {'collector'}, 'COLLECT_OWN_SUBMISSION')
    for closure in ('source_members', 'generation_sources', 'native_sources', 'launchers'):
        for row in lock.get(closure, []):
            path = Path(row['path'])
            require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
                    and row['bytes'] <= 32 * 1024**2 and sha(path) == row['sha256'],
                    'COLLECT_FROZEN_SOURCE_BYTES')
    identity_file = reader.bound(c['observer_identity'])
    identities = identity_file['rows'] if isinstance(identity_file, dict) else identity_file
    stream_receipt = next(row for row in c['assets'] if row['path'] == c['stream'])
    records = reader.bound(stream_receipt)[:2000]
    require(len(records) == 2000 and digest([row['case_id'] for row in records]) == c['ordered_ids_sha256'],
            'COLLECT_EXACT_FIRST2000')
    c['_collector_occurrences'] = {str(row['case_id']): index for index, row in enumerate(records)}
    output = attempt / 'collector'
    require(not output.exists(), 'COLLECT_CREATE_ONCE_OUTPUT')
    output.mkdir()
    reviews = []
    for method in METHODS:
        progress = {}
        try:
            reviews.append(review_arm(reader, attempt, c, lock, method,
                                      submission['jobs'][method], identities, records, progress=progress))
        except Exception as error:
            reviews.append(dict(progress, method=method, scientific_status='TECHNICAL_BLOCKED_REVIEW',
                                error_type=type(error).__name__, error=str(error),
                                verified_prefix_preserved=True))
    allocation = allocation_once(reader, attempt, lock, submission) if accounting else dict(
        status='NOT_REQUESTED', queries=0, reason='CPU_FIXTURE_OR_READONLY_REVIEW')
    write(output / 'allocation.json', allocation)
    for record in allocation.get('records', []):
        if record['method'] != 'collector':
            next(row for row in reviews if row['method'] == record['method']).setdefault('compute_rows', []).append(
                dict(record, phase='parent_allocation', allocation_not_program_timer_sum=True))
    for key, name in (('metric_rows', 'metrics.csv'), ('generation_rows', 'generation.csv'),
                      ('paired_rows', 'paired.csv'), ('compute_rows', 'compute.csv'),
                      ('counter_rows', 'native-counts.csv')):
        _csv(output / name, [row for review in reviews for row in review.get(key, [])])
    compact = [{key: value for key, value in review.items() if key not in
                ('metric_rows', 'generation_rows', 'paired_rows', 'compute_rows', 'counter_rows')}
               for review in reviews]
    complete = all(row['scientific_status'] == 'COMPLETED_VALIDATED' for row in compact)
    report = ['# Llama3 native baseline fluency/consistency CPU 검산', '',
              '새 모델·generation·편집·GPU·W&B 업로드 없이 봉인 raw를 독립 재집계했습니다.', '',
              '| Method | 상태 | 검산 commit/20 | 요청/2000 | join/19 | H append 기대 |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in compact:
        report.append('| {method} | {scientific_status} | {commits} | {requests} | {state_links} | {history} |'.format(
            method=row['method'], scientific_status=row['scientific_status'], commits=row.get('commits', 0),
            requests=row.get('requests', 0), state_links=row.get('state_links', 0),
            history=20 * HISTORY_SITES[row['method']]))
    report += ['', 'R/P: new NLL < true NLL; N: true NLL < new NLL. Tie 실패, N TF desired=true.',
               'Current는 milestone에도 R100/P200/N1000; all-seen은 실제 W5/10/15/20만 기록합니다.',
               'Fluency는 bits, consistency는 cosine이며 유효 occurrence sum/count macro입니다. 결측은 0 대입하지 않습니다.',
               '온전한 commit raw/identity/producer/state 연결을 모두 검산한 prefix만 완료 수로 집계합니다.',
               'Parent GPU-sec 단일 계상; fit/solve/observer nested 시간은 GPU allocation에 더하지 않습니다.',
               'NoCP이며 정확 editor resume는 NOT_AVAILABLE. 원 raw와 실패 이력은 그대로 보존합니다.',
               '새 agent monitor/자동 재제출/Git 자동 push는 수행하지 않습니다. compact 산출물은 사용자 recall 시 통합할 수 있습니다.',
               '', '[RPN](metrics.csv) · [generation](generation.csv) · [paired](paired.csv) · [비용](compute.csv) · [native counts](native-counts.csv)', '']
    _text(output / 'report-ko.md', '\n'.join(report))
    write(output / 'review.json', dict(task_id=TASK, source=lock['source_commit'], reviews=compact,
        scientific_complete=complete, allocation=allocation, collector_wall_seconds=time.monotonic() - started,
        collector_process_CPU_seconds=time.process_time() - cpu_started, new_model_forwards=0,
        generator_calls=0, independent_CPU_reducer=True, raw_copied=False, auto_publish=False))
    # Full per-case path/hash inventory stays ignored local, not a compact Git
    # artifact. The publishable manifest binds its count/digest/member only.
    input_members = list(reader.files.values())
    write(output / 'input-manifest.local.json', dict(inputs=input_members, local_only=True,
                                                  raw_path_inventory_not_for_Git=True))
    write(output / 'manifest.json', dict(task_id=TASK, source=lock['source_commit'],
        input_count=len(input_members), input_inventory_sha256=digest(input_members),
        local_input_inventory=member(output / 'input-manifest.local.json'),
        outputs=[member(path) for path in sorted(output.iterdir())
                 if path.is_file() and path.name != 'input-manifest.local.json'],
        raw_copied=False, no_checkpoints=True, no_broadcast='NO_BROADCAST_NOT_REQUIRED: same-host compact review'))
    write(output / 'terminal.json', dict(status='COMPLETED', scientific_complete=complete,
        source=lock['source_commit'], report=member(output / 'report-ko.md'),
        manifest=member(output / 'manifest.json'), new_model_forwards=0, generator_calls=0))
    return dict(collector_status='COMPLETED', scientific_complete=complete, output=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.attempt)))
