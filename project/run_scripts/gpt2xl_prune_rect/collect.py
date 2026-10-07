"""Independent saved-scalar CPU review of native PRUNE/RECT chains.

No torch/model/tokenizer/native apply, W&B upload, trial, or repair is performed.
The collector's terminal is emitted only after its report and manifest exist.
"""
import argparse
import hashlib
import json
import math
import os
import pwd
import re
import subprocess
from pathlib import Path

from project.run_scripts.jlz_realized_writer_sequential.review_completed import (
    Reader, active_flags, compare_summary, paired, reduce_rows, validate_rows)
from project.run_scripts.gpt2xl_native_baselines.collect import (
    _expected, _metric_rows, _paired_rows, _csv, _atomic_text)
from .common import TASK, NONCE, ARMS, MILESTONES, digest, member, require, sha, write

COUNTERS = ('native_z', 'write_keys', 'history_keys', 'solves', 'history_appends')
EXPECTED = {
    'PRUNE': dict(native_z=100, write_keys=5, history_keys=0, solves=5, history_appends=0),
    'RECT': dict(native_z=100, write_keys=5, history_keys=0, solves=5, history_appends=0)}


def transform_guard(commit, arm, batch, cold=None):
    expected = arm == 'PRUNE' and batch == 20
    value = commit['terminal_transform']
    require(commit['prune_applied'] is expected and value['prune_applied'] is expected
            and commit['native_after']['H'] == {}, 'COLLECT_PRUNE_STAGE_IDENTITY')
    if not expected:
        require(commit['native_after'] == commit['after'], 'COLLECT_NONTERMINAL_DENSE_ENDPOINT')
        return None
    require(value['repair'] == 'PRUNE_TERMINAL_BASE_FIX' and value['repair_authorized'] is True
            and value['upstream_bitwise_equivalence'] is False and value['terminal_transforms'] == 1
            and value['final_base'] == 'SAVED_COLD_W0' and value['spectrum_formula_changed'] is False
            and value['original_line'] == 'adjusted_weight = original_weight + upd_matrix[k]'
            and value['repaired_line'] == 'adjusted_weight = saved_cold_weight + upd_matrix[k]'
            and value['checkpoint_saved'] is False, 'COLLECT_PRUNE_EXPLICIT_BASE_FIX')
    names = {f'transformer.h.{l}.mlp.c_proj.weight':str(l) for l in (13,14,15,16,17)}
    require(set(value['layers']) == set(names), 'COLLECT_PRUNE_EXACT_PHYSICAL_PARAMETER_NAMES')
    require(cold is not None and set(cold['W']) == set(names.values())
            and set(commit['native_after']['W']) == set(names.values())
            and set(commit['after']['W']) == set(names.values()), 'COLLECT_PRUNE_STATE_HASH_LAYOUT')
    for name,row in value['layers'].items():
        require(row['stored_weight_shape'] == [6400,1600] and row['dtype'] == 'torch.float32'
                and row['singular_count'] == 1600 and row['exact_copy_verified'] is True
                and 0 <= row['transformed_singular_count'] <= 1600,
                'COLLECT_PRUNE_NATIVE_SHAPE_SPECTRUM')
        for key in ('max_sigma','delta_norm','compressed_delta_norm','dense_weight_norm','final_weight_norm'):
            require(type(row[key]) in (int,float) and math.isfinite(row[key]) and row[key] >= 0,
                    'COLLECT_PRUNE_FINITE_SCALAR')
        for key in ('dense_weight_sha256','cold_weight_sha256','compressed_delta_sha256','final_weight_sha256'):
            require(type(row[key]) is str and re.fullmatch('[a-f0-9]{64}',row[key]),
                    'COLLECT_PRUNE_HASH_IDENTITY')
        layer = names[name]
        require(row['dense_weight_sha256'] == commit['native_after']['W'][layer]
                and row['final_weight_sha256'] == commit['after']['W'][layer]
                and row['cold_weight_sha256'] == cold['W'][layer],
                'COLLECT_PRUNE_EXACT_DENSE_COLD_FINAL_HASH_BINDING')
    return dict(phase='terminal_prune_transform',seconds=value['seconds'],batch=20,
                repair='PRUNE_TERMINAL_BASE_FIX',nested_native_batch_cost_not_added_again=True)


def rect_masks(native, arm, batch):
    values = native['rect_mask_rows']
    if arm == 'PRUNE':
        require(values == [], 'COLLECT_PRUNE_NOT_RECT')
        return []
    expected = {f'transformer.h.{l}.mlp.c_proj.weight' for l in (13,14,15,16,17)}
    require(len(values) == 5 and {r['weight'] for r in values} == expected,
            'COLLECT_RECT_FIVE_NATIVE_MASKS')
    output = []
    for row in values:
        count, support, ties, less = (row[k] for k in
            ('denominator','support_count','threshold_ties','strictly_below_threshold_count'))
        require(all(type(x) is int for x in (count,support,ties,less))
                and count == 6400*1600 and 0 <= less < count and 0 < ties <= support <= count
                and support == count-less and row['kth_index'] == int(count*.6)
                and row['native_k_percent'] == 40 and row['native_epsilon'] == 1e-8
                and row['native_comparison'] == '>=' and row['no_mask_tensor_saved'] is True
                and row['no_additional_model_forward'] is True, 'COLLECT_RECT_SUPPORT_TIE_IDENTITY')
        require(math.isclose(row['support_pct'],100*support/count,rel_tol=0,abs_tol=1e-10)
                and type(row['threshold']) in (int,float) and math.isfinite(row['threshold'])
                and row['threshold'] >= 0, 'COLLECT_RECT_FINITE_THRESHOLD_SCALAR')
        output.append(dict(arm=arm,batch=batch,**row))
    return output


def endpoint(reader, folder, identities, ids, name, expected_state, seen=None):
    folder = Path(folder)
    if not (folder / 'summary.json').exists():
        return None
    reused = (folder / 'reuse.json').exists()
    if reused:
        receipt = reader.json(folder / 'reuse.json')
        manifest = receipt['manifest']
        require(name == receipt['endpoint'] == 'W0'
                and receipt['manifest_sha256'] == digest(manifest)
                and receipt['source_state'] == manifest['cold_state']
                and receipt['actual_observer_state'] == expected_state
                and receipt['source_state']['W'] == expected_state['W']
                and receipt['projection'] == 'MODEL_WEIGHTS_ONLY'
                and receipt['method_state_reused'] is False
                and receipt['no_forward'] and receipt['no_raw_copy'] and receipt['no_checkpoint'],
                'COLLECT_W0_MODEL_ONLY_REUSE')
        require(len(manifest['chunks']) == 40, 'COLLECT_W0_40_CHUNKS')
        chunks = [reader.bound(item) for item in manifest['chunks']]
        raw_state = receipt['source_state']
    else:
        chunks = [reader.json(path) for path in sorted(folder.glob('chunk-*.json'))]
        raw_state = expected_state
    raw = []
    for chunk in chunks:
        require(chunk['state'] == raw_state and chunk['optimizer_feedback'] is False
                and 0 < len(chunk['rows']) <= 650, 'COLLECT_RAW_STATE_COUNT')
        raw.extend(chunk['rows'])
    validate_rows(raw, _expected(identities, ids), name)
    require(all(row['margin_new_minus_true'] == row['new_nll'] - row['true_nll']
                for row in raw), 'COLLECT_SECOND_MARGIN_SIGN')
    reduced = reduce_rows(raw)
    require({k: reduced[k]['denominator'] for k in ('R', 'P', 'N')}
            == dict(R=len(ids), P=2 * len(ids), N=10 * len(ids)), 'COLLECT_RPN_DENOMINATORS')
    saved = reader.json(folder / 'summary.json')
    require(saved['endpoint'] == name and saved['state'] == expected_state
            and saved['requests'] == len(ids) and saved['row_count'] == len(raw)
            and saved['row_order'] == digest([r['identity'] for r in raw])
            and saved['no_mutation'] and saved['optimizer_feedback'] is False,
            'COLLECT_ENDPOINT_SCOPE')
    compare_summary(reduced, saved['summary'])
    if seen is not None:
        flags = active_flags(seen)
        require(all(r['active_at_endpoint'] == flags[r['case_id']] for r in raw),
                'COLLECT_ACTIVE_SUPERSEDED')
    if reused:
        require(saved['seconds'] == 0 and saved['new_forwards'] == 0
                and saved['reference_only'], 'COLLECT_REUSE_NO_NEW_FORWARD')
    return dict(rows=raw, summary=reduced, seconds=saved['seconds'], reference_only=reused,
        original_evaluation_seconds=saved.get('original_evaluation_seconds'))


def allocation_once(reader, attempt, lock, runner=None, owner=None):
    """One bounded sacct call for exactly this task's two GPU parent jobs."""
    path = Path(attempt) / 'submission.json'
    if not path.exists():
        return dict(status='NOT_RECORDED', queries=0, reason='OWN_SUBMISSION_NOT_FOUND')
    queries = 0
    try:
        submission = reader.json(path)
        require(submission['instruction_id'] == NONCE and submission['task_id'] == TASK
                and submission['source_commit'] == lock['source_commit'], 'OWN_ACCOUNTING_BINDING')
        ids = {arm:str(submission['jobs'][arm]) for arm in ARMS}
        require(len(set(ids.values())) == 2 and all(re.fullmatch(r'[1-9][0-9]*', value)
                for value in ids.values()), 'OWN_ACCOUNTING_TWO_IDS')
        expected_owner = pwd.getpwuid(os.getuid()).pw_name if owner is None else owner
        require(lock['owner'] == expected_owner, 'OWN_ACCOUNTING_OWNER')
        argv = ['sacct', '-X', '-n', '-P', '-j', ','.join(ids.values()),
                '--format=JobIDRaw,JobName%120,User,State,ExitCode,ElapsedRaw,AllocTRES']
        queries = 1
        result = (subprocess.run if runner is None else runner)(argv, text=True,
            capture_output=True, timeout=20, check=False)
        require(result.returncode == 0 and len(result.stdout) <= 1024**2, 'OWN_ACCOUNTING_AVAILABLE')
        records = {}
        for line in result.stdout.splitlines():
            fields = line.strip().split('|')
            if fields and fields[-1] == '': fields.pop()
            require(len(fields) == 7, 'OWN_ACCOUNTING_COLUMNS')
            job, name, user, status, exit_code, elapsed, tres = fields
            require(job in ids.values() and job not in records, 'OWN_ACCOUNTING_EXACT_PARENT')
            arm = next(arm for arm, value in ids.items() if job == value)
            require(name == TASK + '-' + arm and user == expected_owner and elapsed.isdigit(),
                    'OWN_ACCOUNTING_SOURCE_OWNER_NAME')
            resources = dict(item.split('=', 1) for item in tres.split(',') if '=' in item)
            gpu = resources.get('gres/gpu')
            if gpu is None:
                typed = [v for k,v in resources.items() if k.startswith('gres/gpu:')]
                require(len(typed) <= 1, 'OWN_ACCOUNTING_GPU_SCHEMA')
                gpu = typed[0] if typed else '0'
            require(gpu in ('0', '1'), 'OWN_ACCOUNTING_ONE_GPU')
            records[job] = dict(arm=arm, job_id=job, owner=user, scheduler_state=status,
                exit_code=exit_code, parent_elapsed_seconds=int(elapsed), allocated_GPUs=int(gpu),
                allocated_GPU_seconds=int(elapsed)*int(gpu), AllocTRES=tres,
                child_steps_excluded=True, allocation_not_program_timer_sum=True)
        require(set(records) == set(ids.values()), 'OWN_ACCOUNTING_TWO_ROWS')
        return dict(status='RECORDED', queries=queries, records=[records[ids[a]] for a in ARMS],
                    scheduler_completion_not_scientific_completion=True)
    except Exception as error:
        return dict(status='NOT_RECORDED', queries=queries, error_type=type(error).__name__,
                    reason='OWN_ACCOUNTING_UNAVAILABLE_OR_IDENTITY_ERROR', no_retry=True)


def review_arm(reader, attempt, c, lock, arm, identities, records, progress=None):
    out = Path(attempt) / arm
    packs = c['packs']
    require(len(packs) == 20 and all(len(p['ids']) == 100 for p in packs), 'COLLECT_20X100')
    all_ids = [case for p in packs for case in p['ids']]
    require(len(set(all_ids)) == 2000, 'COLLECT_FIRST2000_UNIQUE_IDENTITY')
    metric_table, pair_table, costs, counts, masks, commits, at_write, prefixes = [], [], [], [], [], [], [], {}
    result = dict(arm=arm, scientific_status='PARTIAL_OR_NOT_VERIFIED', commits=0, requests=0,
        state_links=0, W0_available=False, metric_rows=metric_table, paired_rows=pair_table,
        compute_rows=costs, counter_rows=counts, rect_mask_rows=masks, missing=[], new_model_forwards=0)
    if progress is not None:
        progress.update(result)
    if not (out / 'runtime.json').exists():
        result['missing'].append('RUNTIME_NOT_RECORDED')
        return result
    runtime = reader.json(out / 'runtime.json')
    cold = runtime['cold_state']
    require(cold['W'] == c['cold_W'] and runtime['source'] == lock['source_commit']
            and runtime['config'] == digest(c) and runtime['arm'] == arm
            and runtime['cold_history_zero_verified'], 'COLLECT_RUNTIME_COLD_BINDING')
    layers = (13,14,15,16,17)
    require(cold['H'] == {}, 'COLLECT_NATIVE_NO_HISTORY')
    w0 = endpoint(reader, out / 'W0', identities, all_ids, 'W0', cold, records)
    if w0 is not None:
        result['W0_available'] = True
        metric_table.extend(_metric_rows(arm, 'W0_FIRST2000', 0, w0['summary'],
                                         w0['original_evaluation_seconds']))
        costs.append(dict(arm=arm, phase='W0', batch=0, seconds=w0['seconds'],
            reference_only=w0['reference_only'], old_reference_cost_excluded=True))
    for number in range(1, 21):
        folder = out / f'batch-{number:02d}'
        if not (folder / 'commit.json').exists(): break
        commit = reader.json(folder / 'commit.json')
        ids = packs[number - 1]['ids']
        before = cold if not commits else commits[-1]['after']
        require(commit['task'] == TASK and commit['arm'] == arm and commit['batch'] == number
                and commit['source'] == lock['source_commit'] and commit['config'] == digest(c)
                and commit['case_ids'] == ids and commit['before'] == before
                and set(commit['after']['W']) == set(cold['W'])
                and commit['observer_no_mutation'], 'COLLECT_COMMIT_EXACT_LINK')
        require(commit['after']['H'] == {}, 'COLLECT_NATIVE_NO_HISTORY')
        transform_cost = transform_guard(commit,arm,number,cold)
        if transform_cost is not None:
            costs.append(dict(arm=arm,**transform_cost))
        seen_ids = [case for p in packs[:number] for case in p['ids']]
        seen = records[:number*100]
        pre = endpoint(reader, folder / 'pre', identities, ids, f'B{number}_PRE', before, seen)
        observed = seen_ids if number in MILESTONES else ids
        post = endpoint(reader, folder / 'post', identities, observed, f'W{number}', commit['after'], seen)
        require(pre is not None and post is not None, 'COLLECT_COMMITTED_OBSERVER_MISSING')
        compare_summary(pre['summary'], commit['pre']); compare_summary(post['summary'], commit['post'])
        current_rows = [r for r in post['rows'] if r['case_id'] in set(ids)]
        current = reduce_rows(current_rows)
        compare_summary(current, commit['post_current'])
        require({k:current[k]['denominator'] for k in ('R','P','N')} == dict(R=100,P=200,N=1000),
                'COLLECT_CURRENT_ALWAYS100')
        metric_table.extend(_metric_rows(arm, f'B{number}_PRE', 100*number, pre['summary']))
        metric_table.extend(_metric_rows(arm, f'W{number}_CURRENT', 100*number, current))
        pair_table.extend(_paired_rows(arm, f'B{number}_PRE', f'W{number}_CURRENT', pre['rows'], current_rows))
        at_write.extend(current_rows)
        if number in MILESTONES:
            prefixes[number] = post['rows']
            metric_table.extend(_metric_rows(arm, f'W{number}_ALL_SEEN', 100*number, post['summary']))
            first500 = [r for r in post['rows'] if r['case_id'] in set(all_ids[:500])]
            metric_table.extend(_metric_rows(arm, f'W{number}_FIRST500', 100*number, reduce_rows(first500)))
            pair_table.extend(_paired_rows(arm, 'AT_WRITE', f'W{number}_ALL_SEEN', at_write, post['rows']))
            for born in range(1, number + 1):
                cohort = set(packs[born - 1]['ids'])
                pair_table.extend(_paired_rows(arm, f'B{born}_AT_WRITE', f'W{number}',
                    [r for r in at_write if r['case_id'] in cohort],
                    [r for r in post['rows'] if r['case_id'] in cohort], born))
            if w0 is not None:
                pair_table.extend(_paired_rows(arm, 'W0', f'W{number}_ALL_SEEN',
                    [r for r in w0['rows'] if r['case_id'] in set(seen_ids)], post['rows']))
            for active in (True, False):
                subset = [r for r in post['rows'] if r['active_at_endpoint'] is active]
                if subset:
                    label = 'ACTIVE' if active else 'SUPERSEDED'
                    metric_table.extend(_metric_rows(arm, f'W{number}_{label}', 100*number, reduce_rows(subset)))
                    chosen = {r['identity'] for r in subset}
                    pair_table.extend(_paired_rows(arm, 'AT_WRITE_' + label, f'W{number}_{label}',
                        [r for r in at_write if r['identity'] in chosen], subset))
        native, delta = commit['native'], commit['native_counts']
        require(delta == native['delta'] == EXPECTED[arm], 'COLLECT_NATIVE_CALL_HISTORY_COUNTS')
        require(native.get('arm', arm) == arm and native.get('batch', number) == number,
                'COLLECT_NATIVE_RECEIPT_SCOPE')
        require(native['requests'] == 100 and native['same_model_returned'] is True
                and native['native_has_history'] is False and native['caller_history_appends'] == 0
                and native['cache_template'] is None and native['native_z_disk_cache'] is False
                and native['checkpoint_saved'] is False, 'COLLECT_NATIVE_ONE_APPLY_NO_CACHE')
        hp = native['hparams']
        require(hp['layers'] == list(layers) and hp['mom2_update_weight'] == 20000
                and hp['v_lr'] == .5 and hp['v_num_grad_steps'] == 20
                and hp['v_loss_layer'] == 47 and hp['clamp_norm_factor'] == .75
                and hp['v_weight_decay'] == .5 and hp['kl_factor'] == .0625
                and hp['blue'] is False, 'COLLECT_NATIVE_EFFECTIVE_HPARAMS')
        require(native['return_orig_weights'] is (arm == 'PRUNE' and number == 1)
                and native['selected_coldW0_saved_in_RAM'] is (arm == 'PRUNE' and number == 1)
                and native['native_uses_P'] is False, 'COLLECT_RAM_COLD_BASE_CAPTURE_ONCE_NO_P')
        require(native['counts']['public_applies'] == native['counts']['executes'] == 1
                and native['counts']['provisional_restore']['exact'] is True
                and native['counts']['provisional_restore']['before'] ==
                    native['counts']['provisional_restore']['after'], 'COLLECT_NATIVE_PROVISIONAL_RESTORE')
        masks.extend(rect_masks(native,arm,number))
        extras = {k:native['counts'][k] for k in ('fit_forwards','fit_updates','public_applies','executes','rect_masks')}
        require(all(type(x) is int and x >= 0 for x in extras.values()), 'COLLECT_ACTUAL_NATIVE_WORK_SCALARS')
        counts.append(dict(arm=arm,batch=number,**delta,**extras))
        costs.extend((dict(arm=arm,phase='pre_observer',batch=number,seconds=pre['seconds']),
            dict(arm=arm,phase='post_observer',batch=number,seconds=post['seconds']),
            dict(arm=arm,phase='batch_inclusive',batch=number,seconds=commit['seconds'],
                 nested_observer_cost_not_added_again=True)))
        commits.append(commit)
        result.update(commits=len(commits),requests=100*len(commits),state_links=max(0,len(commits)-1))
        if progress is not None: progress.update(result)
    terminal = reader.json(out / 'terminal.json') if (out / 'terminal.json').exists() else None
    totals = {k:sum(row[k] for row in counts) for k in COUNTERS}
    if terminal is not None:
        require(terminal['source'] == lock['source_commit'] and terminal['config'] == digest(c)
                and terminal['commits'] == len(commits), 'COLLECT_TERMINAL_BINDING')
        measured = terminal['native_counts']
        require(all(type(measured[k]) is int and measured[k] >= totals[k] for k in COUNTERS),
                'COLLECT_FAILURE_COST_KEEP')
        if terminal['status'] == 'COMPLETED':
            require(all(measured[k] == totals[k] for k in COUNTERS)
                    and terminal['completed_batches'] == 20
                    and terminal['edits'] == 2000, 'COLLECT_COMPLETED_NATIVE_COUNTS')
            require(terminal['prune_applied'] is (arm == 'PRUNE')
                    and terminal['terminal_transforms'] == (1 if arm == 'PRUNE' else 0),
                    'COLLECT_PRUNE_TERMINAL_ONCE')
        costs.append(dict(arm=arm,phase='program_wall',seconds=terminal['program_seconds'],
                         program_wall_not_added_to_allocation=True))
    if 5 in prefixes and 20 in prefixes:
        pair_table.extend(_paired_rows(arm,'W5_FIRST500','W20_FIRST500',prefixes[5],
            [r for r in prefixes[20] if r['case_id'] in set(all_ids[:500])]))
    complete = len(commits) == 20 and w0 is not None and terminal is not None and terminal['status'] == 'COMPLETED'
    result.update(scientific_status='COMPLETED_VALIDATED' if complete else 'PARTIAL_OR_NOT_VERIFIED',
        measured_native_counts=totals, failed_or_uncommitted_native_counts=None if terminal is None else
            {k:terminal['native_counts'][k]-totals[k] for k in COUNTERS},
        terminal_status=None if terminal is None else terminal['status'], no_PRICE_KKT=True)
    result['terminal_prune_applied'] = bool(commits and commits[-1]['prune_applied'])
    result['terminal_base_fix'] = 'PRUNE_TERMINAL_BASE_FIX' if result['terminal_prune_applied'] else None
    identity_path, finish_path = out/'tracking-identity.json', out/'tracking-finish.json'
    if identity_path.exists():
        identity = reader.json(identity_path)
        cfg = identity['config']
        require(cfg['task_id'] == TASK and cfg['arm'] == arm
                and cfg['model'] == 'gpt2xl' and cfg['role'] == 'scientific'
                and identity['source_sha'] == lock['source_commit']
                and identity['config_sha'] == lock['config_sha256'], 'COLLECT_TRACKING_IDENTITY')
        finish = reader.json(finish_path) if finish_path.exists() else {}
        result['tracking'] = dict(run_id=identity['run_id'],url=identity.get('url'),
            job_id=cfg['job_id'],startup_status=identity.get('startup_readback',{}).get('status'),
            finish_status=finish.get('status','NOT_RECORDED'),
            local_finish_status=finish.get('local_status','NOT_RECORDED'),
            SDK_acceptance_not_remote_readback=True,scientific_completion_is_separate=True)
    else:
        result['tracking'] = dict(status='NOT_RECORDED', scientific_completion_is_separate=True)
    return result


def collect(attempt):
    attempt, reader = Path(attempt), Reader()
    c, lock = reader.json(attempt/'config.json'), reader.json(attempt/'execution.lock.json')
    require(c['task_id'] == TASK and c['instruction_id'] == lock['instruction_id'] == NONCE
            and sha(attempt/'config.json') == lock['config_sha256'], 'COLLECT_CONFIG_SOURCE')
    identities = reader.bound(c['observer_identity'])['rows']
    stream_member = next(item for item in c['assets'] if item['path'] == c['stream'])
    records = reader.bound(stream_member)[:2000]
    require([r['case_id'] for r in records] == [case for p in c['packs'] for case in p['ids']],
            'COLLECT_ORDERED_FIRST2000')
    out = attempt/'collector'
    require(not out.exists(), 'COLLECT_CREATE_ONCE')
    out.mkdir()
    reviews = []
    for arm in ARMS:
        progress = {}
        try:
            reviews.append(review_arm(reader,attempt,c,lock,arm,identities,records,progress))
        except Exception as error:
            reviews.append(dict(progress,arm=arm,scientific_status='TECHNICAL_BLOCKED_REVIEW',
                error_type=type(error).__name__,error=str(error),valid_prefix_preserved=True))
    allocation = allocation_once(reader,attempt,lock)
    write(out/'allocation.json',allocation)
    for rec in allocation.get('records',[]):
        review = next(r for r in reviews if r['arm'] == rec['arm'])
        review.setdefault('compute_rows',[]).append(dict(rec,phase='parent_allocation'))
    for key,name in (('metric_rows','metrics.csv'),('paired_rows','paired.csv'),
                     ('compute_rows','compute.csv'),('counter_rows','native-counts.csv'),
                     ('rect_mask_rows','rect-support.csv')):
        _csv(out/name,[row for review in reviews for row in review.get(key,[])])
    compact = [{k:v for k,v in review.items() if k not in
                ('metric_rows','paired_rows','compute_rows','counter_rows','rect_mask_rows')} for review in reviews]
    complete = all(r['scientific_status'] == 'COMPLETED_VALIDATED' for r in compact)
    report = ['# GPT2-XL PRUNE / RECT native CPU 검산','',
        '독립 저장 scalar row를 재집계했습니다. 새 모델 forward/fit/편집·W&B 업로드는 없습니다.','',
        '| Arm | 과학 상태 | 완료 batch | 요청 | 상태 연결 |',
        '| --- | --- | ---: | ---: | ---: |']
    for review in compact:
        report.append('| %s | %s | %s | %s | %s |' % (review['arm'],review['scientific_status'],
            review.get('commits','NA'),review.get('requests','NA'),review.get('state_links','NA')))
    report.extend(['','PRUNE와 RECT는 L13–17의 native no-H/no-P 경로입니다.',
        'PRUNE W5/W10/W15는 dense 경로, W20만 명시적인 PRUNE_TERMINAL_BASE_FIX 이후 endpoint입니다.',
        '원 upstream currentW+compressedD의 중복가산을 saved coldW0+compressedD로 수리했습니다; upstream bitwise라고 주장하지 않습니다.',
        'RECT 지원은 score>=threshold이므로 ties에서40%를 넘을 수 있습니다; 실제 count와 tie를 별도 표에 보존합니다.',
        'R/P new<true, N true<new; tie=failure. N desired=true. TF는 자유생성이 아닙니다.',
        'Current는 milestone에서도100 요청, all-seen은 W5/W10/W15/W20의 실제 prefix입니다.',
        'Native Adam/backward 횟수는 실측 없으면 NOT_RECORDED이며 targetfit 호출로 대체하지 않습니다.',
        '과거 W0 재사용 비용·program wall·parent allocation을 중복 합산하지 않습니다.',
        '원 raw/부분·실패 비용은 보존하며 낮은 기능 지표는 실패 gate가 아닙니다.',
        'Parent accounting 상태: '+allocation['status']+'.','',
        '[지표](metrics.csv) · [paired](paired.csv) · [비용](compute.csv) · [native 계수](native-counts.csv) · [RECT 지원](rect-support.csv)',''])
    _atomic_text(out/'report-ko.md','\n'.join(report))
    write(out/'review.json',dict(task=TASK,source=lock['source_commit'],reviews=compact,
        scientific_complete=complete,new_model_forwards=0,GPU_validation_by_collector=False))
    outputs = [member(p) for p in sorted(out.iterdir()) if p.is_file()]
    write(out/'manifest.json',dict(task=TASK,source=lock['source_commit'],inputs=list(reader.files.values()),
        outputs=outputs,raw_copied=False,raw_free_report=True,no_checkpoints=True))
    write(out/'terminal.json',dict(status='COMPLETED',scientific_complete=complete,
        scientific_status='COMPLETED_VALIDATED' if complete else 'PARTIAL_OR_TECHNICAL_BLOCKED',
        source=lock['source_commit'],report=member(out/'report-ko.md'),manifest=member(out/'manifest.json'),
        new_model_forwards=0))
    return dict(collector_status='COMPLETED',scientific_complete=complete,output=str(out))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--attempt',type=Path,required=True)
    print(json.dumps(collect(p.parse_args().attempt)))
