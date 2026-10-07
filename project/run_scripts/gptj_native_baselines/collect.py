"""Independent stdlib scalar audit for two stock native GPT-J chains.

No model, native fitter, scheduler writes, network, or PRICE KKT is used. A
partial chain is retained honestly; collector completion is not science success.
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
from pathlib import Path

from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader, active_flags, compare_summary, harmonic, paired, reduce_rows, validate_rows)
from .common import TASK, NONCE, ARMS, digest, member, require, sha, write

MILESTONES = (5, 10, 15, 20)
COUNTERS = ('native_z', 'write_keys', 'history_keys', 'solves', 'history_appends')


def allocation_once(reader, attempt, lock, runner=None, owner=None):
    """Optional one-shot accounting of ONLY the two actually submitted GPU IDs.

    This is called by the sealed afterany collector, never a polling loop. Missing
    accounting cannot turn partial scientific results into success or erase them.
    """
    submission_path = Path(attempt) / 'submission.json'
    if not submission_path.exists():
        return dict(status='NOT_RECORDED', reason='OWN_SUBMISSION_NOT_FOUND', queries=0)
    queries = 0
    try:
        submission = reader.json(submission_path)
        require(submission['instruction_id'] == NONCE and submission['task_id'] == TASK
                and submission['source_commit'] == lock['source_commit']
                and submission['lock']['path'] == str(Path(attempt) / 'execution.lock.json')
                and submission['lock']['sha256'] == sha(Path(attempt) / 'execution.lock.json'),
                'ACCOUNTING_OWN_SOURCE_BINDING')
        ids = {arm: submission['jobs'][arm] for arm in ARMS}
        require(all(type(job) is str and re.fullmatch(r'[1-9][0-9]*', job)
                    for job in ids.values()) and len(set(ids.values())) == len(ARMS),
                'ACCOUNTING_EXACT_TWO_GPU_IDS')
        expected_owner = pwd.getpwuid(os.getuid()).pw_name if owner is None else owner
        require(lock.get('owner') == expected_owner, 'ACCOUNTING_OWN_OWNER')
        argv = ['sacct', '-n', '-P', '-j', ','.join(ids.values()),
                '--format=JobIDRaw,JobName%120,User,State,ExitCode,ElapsedRaw,AllocTRES']
        queries = 1
        result = (subprocess.run if runner is None else runner)(
            argv, text=True, capture_output=True, timeout=20, check=False)
        require(result.returncode == 0 and len(result.stdout) <= 1024**2,
                'ACCOUNTING_AVAILABLE_BOUNDED')
        by_id = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split('|')]
            if len(fields) == 8 and not fields[-1]:
                fields.pop()
            require(len(fields) == 7, 'ACCOUNTING_COLUMN_SCHEMA')
            job, name, user, status, exit_code, elapsed, tres = fields
            if job not in ids.values():
                require(any(job.startswith(item + '.') for item in ids.values()),
                        'ACCOUNTING_UNEXPECTED_JOB')
                continue
            arm = next(arm for arm, item in ids.items() if item == job)
            require(job not in by_id and name == TASK + '-' + arm
                    and user == expected_owner and elapsed.isdigit(),
                    'ACCOUNTING_PARENT_IDENTITY')
            resources = dict(item.split('=', 1) for item in tres.split(',') if '=' in item)
            gpu = resources.get('gres/gpu')
            if gpu is None:
                typed = [value for key, value in resources.items() if key.startswith('gres/gpu:')]
                require(len(typed) <= 1, 'ACCOUNTING_GPU_ALLOCATION_SCHEMA')
                gpu = typed[0] if typed else '0'
            require(gpu in ('0', '1'), 'ACCOUNTING_ONE_GPU_ALLOCATION')
            seconds = int(elapsed)
            by_id[job] = dict(arm=arm, job_id=job, owner=user, scheduler_state=status,
                exit_code=exit_code, parent_elapsed_seconds=seconds, allocated_GPUs=int(gpu),
                allocated_GPU_seconds=seconds * int(gpu), AllocTRES=tres,
                allocation_status='ALLOCATED' if gpu == '1' else 'NOT_ALLOCATED',
                child_steps_excluded=True, program_timers_not_added_to_allocation=True)
        require(set(by_id) == set(ids.values()), 'ACCOUNTING_TWO_PARENT_ROWS_REQUIRED')
        return dict(status='RECORDED', queries=queries,
                    scope='OWN_TWO_GPU_PARENTS_ONLY', records=[by_id[ids[arm]] for arm in ARMS],
                    scheduler_success_not_scientific_success=True)
    except Exception as error:
        # No stderr, command stdout, broad environment, or unrelated job data is retained.
        return dict(status='NOT_RECORDED', queries=queries,
                    reason='OWN_ACCOUNTING_UNAVAILABLE_OR_IDENTITY_MISMATCH',
                    error_type=type(error).__name__, no_retry=True)


def _expected(identities, ids):
    require(len(ids) == len(set(ids)), 'COLLECT_DUPLICATE_REQUEST')
    indexed = {}
    for row in identities:
        indexed.setdefault(row['case_id'], []).append(row)
    require(all(case in indexed for case in ids), 'COLLECT_MISSING_IDENTITY')
    return [row for case in ids for row in indexed[case]]


def endpoint(reader, folder, identities, ids, name, expected_state, seen=None):
    """Independent raw arithmetic; absent observations stay unavailable."""
    folder = Path(folder)
    if not (folder / 'summary.json').exists():
        return None
    reused = (folder / 'reuse.json').exists()
    if reused:
        receipt = reader.json(folder / 'reuse.json')
        require(name=='W0' and receipt['scalar_bridge_only'] and not receipt['history_or_editor_resume'],'COLLECT_SCALAR_W0_ONLY')
        require(receipt['actual_cold_weights']==expected_state['W'] and not expected_state['H'],'COLLECT_W0_COLD')
        chunks=[reader.bound(item) for item in receipt['chunks']]
        source_state=None
    else:
        chunks = [reader.json(path) for path in sorted(folder.glob('chunk-*.json'))]
        source_state = expected_state
    raw = []
    for chunk in chunks:
        require((source_state is None or chunk['state'] == source_state) and chunk['optimizer_feedback'] is False,
                'COLLECT_RAW_ENDPOINT_STATE')
        require(0 < len(chunk['rows']) <= 650, 'COLLECT_CHUNK_ROWS')
        raw.extend(chunk['rows'])
    validate_rows(raw, _expected(identities, ids), name)
    for row in raw:
        require(row.get('margin_new_minus_true', row['new_nll'] - row['true_nll']) == row['new_nll'] - row['true_nll'],
                'COLLECT_SECOND_MARGIN_SIGN')
    reduced = reduce_rows(raw)
    require({kind: reduced[kind]['denominator'] for kind in ('R', 'P', 'N')}
            == dict(R=len(ids), P=2 * len(ids), N=10 * len(ids)), 'COLLECT_RPN_DENOMINATORS')
    saved = reader.json(folder / 'summary.json')
    require(saved['endpoint'] == name and saved['state'] == expected_state
            and saved['requests'] == len(ids) and saved['row_count'] == len(raw)
            and saved['row_order'] == digest([row['identity'] for row in raw])
            and saved['no_mutation'] is True and saved['optimizer_feedback'] is False,
            'COLLECT_ENDPOINT_SUMMARY_SCOPE')
    compare_summary(reduced, saved['summary'])
    if seen is not None and not reused:
        flags = active_flags(seen)
        require(all(row['active_at_endpoint'] == flags[row['case_id']] for row in raw),
                'COLLECT_ACTIVE_SUPERSEDED_SCOPE')
    if reused:
        require(saved['seconds'] == 0 and saved['new_forwards'] == 0
                and saved['reference_only'], 'COLLECT_W0_NO_NEW_COST')
    return dict(rows=raw, summary=reduced, seconds=saved['seconds'], reference_only=reused,
                original_evaluation_seconds=saved.get('original_evaluation_seconds'))


def _metric_rows(arm, name, edits, selected, original_seconds=None):
    for kind, value in selected.items():
        row = dict(arm=arm, endpoint=name, edits=edits, kind=kind,
            count=value['denominator'], success_count=value['numerator'],
            success_pct=100 * value['rate'], token_count=value['desired_token_count'],
            token_correct=value['desired_token_correct'], token_acc_pct=100 * value['token_micro'],
            prompt_acc_pct=100 * value['prompt_macro'], strict_count=value['strict_numerator'],
            strict_acc_pct=100 * value['strict_rate'], true_nll=value['true_nll_mean'],
            new_nll=value['new_nll_mean'], desired_nll=value['desired_nll_mean'],
            margin_true_minus_new=value['true_minus_new_mean'],
            success_harmonic_pct=None if harmonic(selected) is None else 100 * harmonic(selected),
            original_observation_seconds=original_seconds)
        for field in ('true_nll', 'new_nll', 'desired_nll', 'margin_true_minus_new'):
            for quantile in ('p50', 'p95', 'p99'):
                row[field + '_' + quantile] = value[field + '_quantiles'][quantile]
        yield row


def _paired_rows(arm, before_name, after_name, before, after, cohort=None):
    result = paired(before, after)
    for kind, metrics in result.items():
        for metric, value in metrics.items():
            yield dict(arm=arm, before_endpoint=before_name, after_endpoint=after_name,
                       cohort=cohort, kind=kind, metric=metric, **value)


def _counter_delta(commit):
    # Only measured native counters are accepted. Never infer 20 Adam updates.
    value = commit.get('native_counts')
    if value is None:
        native = commit.get('native', {})
        value = native.get('delta') if isinstance(native, dict) else None
    if value is None:
        return None
    require(all(key in value and type(value[key]) is int and value[key] >= 0
                for key in COUNTERS), 'COLLECT_NATIVE_COUNT_SCHEMA')
    return {key: value[key] for key in COUNTERS}


def review_arm(reader, attempt, c, lock, arm, identities, progress=None, records=None):
    out = attempt / arm
    writer = 'memit' if arm == 'BASE_MEMIT' else 'alphaedit'
    packs = c['packs']
    require(len(packs) == 20 and all(len(pack['ids']) == 100 for pack in packs),
            'COLLECT_20X100_PACKS')
    all_ids = [case for pack in packs for case in pack['ids']]
    require(len(set(all_ids)) == 2000, 'COLLECT_UNIQUE_FIRST2000')
    cold = dict(W=c['cold_W'], H={})
    metric_table, paired_table, cost_table, count_table = [], [], [], []
    commits, at_write, prefixes, errors = [], [], {}, []
    if progress is not None:
        progress.update(arm=arm, writer=writer, commits=0, requests=0, state_links=0,
            metric_rows=metric_table, paired_rows=paired_table, compute_rows=cost_table,
            counter_rows=count_table, missing=errors, W0_available=False)
    w0 = endpoint(reader, out / 'W0', identities, all_ids, 'W0', cold, records)
    if w0 is not None:
        if progress is not None:
            progress['W0_available'] = True
        metric_table.extend(_metric_rows(arm, 'W0_FIRST2000', 0, w0['summary'],
                                         w0['original_evaluation_seconds']))
        cost_table.append(dict(arm=arm, phase='W0', batch=0, seconds=w0['seconds'],
            reference_only=w0['reference_only'], old_reference_cost_excluded=True,
            original_seconds=w0['original_evaluation_seconds']))
    for number in range(1, 21):
        folder = out / f'batch-{number:02d}'
        if not (folder / 'commit.json').exists():
            break
        commit = reader.json(folder / 'commit.json')
        pack = packs[number - 1]
        require(commit['task'] == TASK and commit['arm'] == arm and commit['batch'] == number
                and commit['source'] == lock['source_commit'] and commit['config'] == digest(c),
                'COLLECT_NATIVE_COMMIT_BINDING')
        require(commit['case_ids'] == pack['ids'], 'COLLECT_NATIVE_REQUEST_ORDER')
        before = cold if not commits else commits[-1]['after']
        require(commit['before'] == before and set(commit['after']['W']) == set(c['cold_W']),
                'COLLECT_NATIVE_19_STATE_LINKS')
        if writer == 'memit':
            require(commit['before']['H'] == commit['after']['H'] == {}, 'COLLECT_MEMIT_NO_H')
        else:
            require(set(commit['after']['H']) == set(c['cold_W']), 'COLLECT_ALPHA_HISTORY_SCOPE')
        require(commit['observer_no_mutation'] is True, 'COLLECT_OBSERVER_MUTATION')
        ids = pack['ids']
        seen_ids = [case for p in packs[:number] for case in p['ids']]
        seen_records = None if records is None else records[:number * 100]
        pre = endpoint(reader, folder / 'pre', identities, ids, f'B{number}_PRE', before, seen_records)
        observed_ids = seen_ids if number in MILESTONES else ids
        post = endpoint(reader, folder / 'post', identities, observed_ids, f'W{number}', commit['after'],
                        seen_records)
        require(pre is not None and post is not None, 'COLLECT_COMMITTED_OBSERVATION_MISSING')
        compare_summary(pre['summary'], commit['pre'])
        compare_summary(post['summary'], commit['post'])
        current_rows = [row for row in post['rows'] if row['case_id'] in set(ids)]
        current = reduce_rows(current_rows)
        compare_summary(current, commit['post_current'])
        require({key: current[key]['denominator'] for key in ('R', 'P', 'N')}
                == dict(R=100, P=200, N=1000), 'COLLECT_CURRENT_ALWAYS100')
        metric_table.extend(_metric_rows(arm, f'B{number}_PRE', 100 * number, pre['summary']))
        metric_table.extend(_metric_rows(arm, f'W{number}_CURRENT', 100 * number, current))
        paired_table.extend(_paired_rows(arm, f'B{number}_PRE', f'W{number}_CURRENT',
                                         pre['rows'], current_rows))
        at_write.extend(current_rows)
        if number in MILESTONES:
            prefixes[number] = post['rows']
            metric_table.extend(_metric_rows(arm, f'W{number}_ALL_SEEN', 100 * number, post['summary']))
            first500 = [row for row in post['rows'] if row['case_id'] in set(all_ids[:500])]
            metric_table.extend(_metric_rows(arm, f'W{number}_FIRST500', 100 * number, reduce_rows(first500)))
            paired_table.extend(_paired_rows(arm, 'AT_WRITE', f'W{number}_ALL_SEEN', at_write, post['rows']))
            for born in range(1, number + 1):
                cohort = set(packs[born - 1]['ids'])
                paired_table.extend(_paired_rows(arm, f'B{born}_AT_WRITE', f'W{number}',
                    [row for row in at_write if row['case_id'] in cohort],
                    [row for row in post['rows'] if row['case_id'] in cohort], born))
            if w0 is not None:
                paired_table.extend(_paired_rows(arm, 'W0', f'W{number}_ALL_SEEN',
                    [row for row in w0['rows'] if row['case_id'] in set(seen_ids)], post['rows']))
            for active in (True, False):
                subset = [row for row in post['rows'] if row['active_at_endpoint'] is active]
                if subset:
                    label = 'ACTIVE' if active else 'SUPERSEDED'
                    metric_table.extend(_metric_rows(arm, f'W{number}_{label}', 100 * number,
                                                     reduce_rows(subset)))
                    selected_set = {row['identity'] for row in subset}
                    paired_table.extend(_paired_rows(arm, f'AT_WRITE_{label}', f'W{number}_{label}',
                        [row for row in at_write if row['identity'] in selected_set], subset))
        delta = _counter_delta(commit)
        native = commit['native']
        require(native['arm'] == arm and native['writer'] == writer and native['batch'] == number
                and native['requests'] == 100 and native['same_model_returned'] is True
                and native['returned_weights_copy_is_history'] is False
                and native['native_z_disk_cache'] is False and native['cache_template'] is None
                and native['caller_history_appends'] == 0 and native['checkpoint_saved'] is False,
                'COLLECT_DIRECT_NATIVE_APPLY_SCOPE')
        require(native['Alpha_reset_cache'] == ((number == 1) if writer == 'alphaedit' else None),
                'COLLECT_ALPHA_INITIAL_RESET_ONLY')
        measured = native.get('delta', native.get('counts'))
        require(measured is not None and delta is not None
                and all(measured[key] == delta[key] for key in COUNTERS),
                'COLLECT_NATIVE_RECEIPT_DELTA')
        if delta is None:
            errors.append(dict(batch=number, reason='NATIVE_COUNTS_NOT_RECORDED'))
        else:
            expected = dict(native_z=100, write_keys=6, history_keys=6 if writer == 'alphaedit' else 0,
                            solves=6, history_appends=6 if writer == 'alphaedit' else 0)
            require(delta == expected, 'COLLECT_NATIVE_BATCH_COUNTS')
            count_table.append(dict(arm=arm, batch=number, **delta))
        cost_table.extend((dict(arm=arm, phase='pre_observer', batch=number, seconds=pre['seconds']),
                           dict(arm=arm, phase='post_observer', batch=number, seconds=post['seconds'])))
        if 'seconds' in commit:
            require(type(commit['seconds']) in (int, float) and math.isfinite(commit['seconds'])
                    and commit['seconds'] >= 0, 'COLLECT_BATCH_TIME')
            cost_table.append(dict(arm=arm, phase='batch_inclusive', batch=number, seconds=commit['seconds'],
                                   nested_observer_cost_not_added_again=True))
        commits.append(commit)
        if progress is not None:
            progress.update(commits=len(commits), requests=len(commits) * 100,
                            state_links=max(0, len(commits) - 1))
    terminal = reader.json(out / 'terminal.json') if (out / 'terminal.json').exists() else None
    if terminal is not None:
        require(terminal.get('source') == lock['source_commit']
                and terminal.get('commits') == len(commits), 'COLLECT_TERMINAL_PROGRESS')
        if 'completed_batches' in terminal:
            require(terminal['completed_batches'] == len(commits), 'COLLECT_TERMINAL_BATCHES')
        if 'native_counts' in terminal:
            for key in COUNTERS:
                measured = terminal['native_counts'].get(key)
                verified = sum(row[key] for row in count_table)
                require(type(measured) is int and measured >= verified,
                        'COLLECT_TERMINAL_NATIVE_COUNTS')
                if terminal['status'] == 'COMPLETED':
                    require(measured == verified, 'COLLECT_COMPLETED_COUNTER_EXACT')
    if 5 in prefixes and 20 in prefixes:
        same = set(all_ids[:500])
        paired_table.extend(_paired_rows(arm, 'W5_FIRST500', 'W20_FIRST500', prefixes[5],
            [row for row in prefixes[20] if row['case_id'] in same]))
    finished = len(commits) == 20 and w0 is not None and not errors
    terminal_complete = terminal is not None and terminal.get('status') in ('COMPLETED', 'COMPLETE')
    scientific = 'COMPLETED_VALIDATED' if finished and terminal_complete else 'PARTIAL_OR_NOT_VERIFIED'
    return dict(arm=arm, writer=writer, scientific_status=scientific,
        commits=len(commits), requests=len(commits) * 100, state_links=max(0, len(commits) - 1),
        measured_native_counts={key: sum(row[key] for row in count_table) for key in COUNTERS},
        failed_or_uncommitted_native_counts=None if terminal is None or 'native_counts' not in terminal else
            {key: terminal['native_counts'][key] - sum(row[key] for row in count_table) for key in COUNTERS},
        W0_available=w0 is not None,
        terminal=None if terminal is None else {key: terminal[key] for key in
            ('status', 'source', 'config', 'commits', 'completed_batches', 'native_counts',
             'error_type', 'seconds') if key in terminal}, missing=errors,
        metric_rows=metric_table, paired_rows=paired_table, compute_rows=cost_table,
        counter_rows=count_table, no_PRICE_KKT=True, new_model_forwards=0)


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode('utf-8')
    require(len(data) <= 32 * 1024**2, 'COLLECT_COMPACT_TEXT_BOUND')
    require(not path.exists(), 'COLLECT_CREATE_ONCE_TEXT')
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(tmp, path)
    tmp.unlink()


def _csv(path, values):
    fields = list(dict.fromkeys(key for row in values for key in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields or ['availability'], lineterminator='\n')
    writer.writeheader()
    writer.writerows(values)
    _atomic_text(path, stream.getvalue())


def collect(attempt):
    attempt = Path(attempt)
    reader = Reader()
    lock = reader.json(attempt / 'execution.lock.json')
    c = reader.json(attempt / 'config.json')
    require(sha(attempt / 'config.json') == lock['config_sha256'] and c['task_id'] == TASK,
            'COLLECT_SOURCE_CONFIG')
    identities = reader.bound(c['observer_identity'])
    stream_member = next(item for item in c['assets'] if item['path'] == c['stream'])
    records = reader.bound(stream_member)[:2000]
    require([record['case_id'] for record in records] ==
            [case for pack in c['packs'] for case in pack['ids']], 'COLLECT_FIRST2000_STREAM_ORDER')
    out = attempt / 'collector'
    require(not out.exists(), 'COLLECT_CREATE_ONCE_OUTPUT')
    out.mkdir()
    reviews = []
    for arm in ARMS:
        progress = {}
        try:
            reviews.append(review_arm(reader, attempt, c, lock, arm, identities, progress, records))
        except Exception as error:
            reviews.append(dict(progress, arm=arm, scientific_status='TECHNICAL_BLOCKED_REVIEW',
                                error_type=type(error).__name__, error=str(error),
                                valid_prefix_preserved=True))
    allocation = allocation_once(reader, attempt, lock)
    write(out / 'allocation.json', allocation)
    for record in allocation.get('records', []):
        review = next(value for value in reviews if value['arm'] == record['arm'])
        review.setdefault('compute_rows', []).append(dict(record, phase='parent_allocation',
            seconds=record['parent_elapsed_seconds'], allocation_not_program_timer_sum=True))
    for key, filename in (('metric_rows', 'metrics.csv'), ('paired_rows', 'paired.csv'),
                          ('compute_rows', 'compute.csv'), ('counter_rows', 'native-counts.csv')):
        _csv(out / filename, [row for review in reviews for row in review.get(key, [])])
    compact = [{key: value for key, value in review.items()
                if key not in ('metric_rows', 'paired_rows', 'compute_rows', 'counter_rows')}
               for review in reviews]
    complete = all(review['scientific_status'] == 'COMPLETED_VALIDATED' for review in compact)
    report = ['# GPT-J stock EasyEdit baseline CPU 검산', '',
              '새 모델 평가·native fit·Slurm 변경·W&B 업로드를 수행하지 않은 독립 저장 row 검산입니다.', '',
              '| Arm | 과학 상태 | 완료 batch | 요청 | 상태 연결 |',
              '| --- | --- | ---: | ---: | ---: |']
    for review in compact:
        report.append('| {arm} | {scientific_status} | {commits} | {requests} | {state_links} |'.format(
            commits=review.get('commits', 'NA'), requests=review.get('requests', 'NA'),
            state_links=review.get('state_links', 'NA'), **{key: review[key] for key in ('arm', 'scientific_status')}))
    report.extend(['', 'R/P는 new NLL<true NLL, N은 true NLL<new NLL이며 tie는 실패입니다.',
        'N desired=true; TF는 자유생성 지표가 아닙니다. Milestone current 분모는 항상100 요청입니다.',
        'Native backward/Adam update 횟수는 실측이 없으면 NOT_RECORDED이며 50,000회로 추정하지 않습니다.',
        '원 raw와 실패/부분 결과는 보존하며, 이전 W0 비용은 신규 계산비용에 중복 합산하지 않습니다.',
        '자체 두 GPU job parent allocation 단발 accounting: ' + allocation['status'] + '.',
        'Parent GPU-sec와 프로그램/observer 시간은 서로 다른 계수이며 중복 합산하지 않습니다.',
        '', '[지표](metrics.csv) · [paired](paired.csv) · [계산비용](compute.csv) · [native 계수](native-counts.csv)', ''])
    _atomic_text(out / 'report-ko.md', '\n'.join(report))
    write(out / 'review.json', dict(task=TASK, source=lock['source_commit'], reviews=compact,
        scientific_complete=complete, allocation=allocation, GPU_validation_by_collector=False,
        new_model_forwards=0))
    outputs = [member(path) for path in sorted(out.iterdir()) if path.is_file()]
    write(out / 'manifest.json', dict(task=TASK, source=lock['source_commit'],
        inputs=list(reader.files.values()), outputs=outputs, raw_copied=False,
        raw_free_report=True, no_checkpoints=True, collector_not_scientific_execution=True))
    # A successfully written report/manifest precedes the atomic terminal receipt.
    write(out / 'terminal.json', dict(status='COMPLETED', scientific_complete=complete,
        scientific_status='COMPLETED_VALIDATED' if complete else 'PARTIAL_OR_TECHNICAL_BLOCKED',
        source=lock['source_commit'], report=member(out / 'report-ko.md'),
        manifest=member(out / 'manifest.json'), new_model_forwards=0))
    return dict(collector_status='COMPLETED', scientific_complete=complete, output=str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.attempt)))
