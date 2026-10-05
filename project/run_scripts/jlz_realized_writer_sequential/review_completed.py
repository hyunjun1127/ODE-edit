"""Read-only independent scalar/raw V13 sequential review (Python stdlib only).

No production reducers, torch, tokenizer, model, scheduler, or evaluation imports.
The reader hashes original bytes; outputs are create-once in a separate directory.
This verifies stored token identities/count relations, not tokenization reexecution.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

SOURCE = '2ff0ecc63dda2c830fbd68e6a2d76f3317a446f7'
CONFIG_SHA = 'b2745b667e011f2a27e81b5f5f275cb731b9f418342b20cf741855c2149fd429'
LOCK_SHA = 'a399d80257c38182cb67498d4aef8be12729f0e8fffe118f8d7f1e8b2701351c'
MILESTONES = (5, 10, 15, 20)
KINDS = ('R', 'P', 'N')


def ensure(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class Reader:
    def __init__(self):
        self.files = {}

    def bytes(self, path):
        path = Path(path)
        ensure(path.is_file() and not path.is_symlink(), 'UNSAFE_OR_MISSING_FILE:' + str(path))
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        ensure((before.st_size, before.st_mtime_ns, before.st_ino) ==
               (after.st_size, after.st_mtime_ns, after.st_ino), 'READ_CHANGED:' + str(path))
        item = dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        ensure(str(path) not in self.files or self.files[str(path)] == item, 'SECOND_READ_CHANGED')
        self.files[str(path)] = item
        return data

    def json(self, path):
        return json.loads(self.bytes(path))

    def bound(self, receipt):
        data = self.bytes(receipt['path'])
        ensure(len(data) == receipt['bytes'] and hashlib.sha256(data).hexdigest() ==
               receipt['sha256'], 'BOUND_FILE_SHA:' + receipt['path'])
        return json.loads(data)


def quantiles(values):
    if not values:
        return None
    values = sorted(values)
    result = {}
    for name, p in (('min', 0), ('p01', .01), ('p05', .05), ('p50', .5),
                    ('p95', .95), ('p99', .99), ('max', 1)):
        at = p * (len(values) - 1)
        lo = math.floor(at)
        hi = math.ceil(at)
        result[name] = values[lo] * (hi - at) + values[hi] * (at - lo) if hi != lo else values[lo]
    return result


def success(row, metric='preference'):
    desired = 'true' if row['kind'] == 'N' else 'new'
    return row[desired + '_strict'] if metric == 'strict' else (
        row['true_nll'] < row['new_nll'] if row['kind'] == 'N' else
        row['new_nll'] < row['true_nll'])


def validate_rows(rows, expected=None, endpoint=None):
    ensure(len({r['identity'] for r in rows}) == len(rows), 'DUPLICATE_IDENTITY')
    if expected is not None:
        ensure([r['identity'] for r in rows] == [r['identity'] for r in expected],
               'EXACT_IDENTITY_ORDER_CARDINALITY')
    for index, row in enumerate(rows):
        ensure(row['kind'] in KINDS, 'INVALID_KIND')
        if endpoint is not None:
            ensure(row['endpoint'] == endpoint, 'ENDPOINT_LABEL')
        if expected is not None:
            ref = expected[index]
            for field in ('case_id', 'kind', 'prompt_index', 'new_token_identity', 'true_token_identity'):
                ensure(row[field] == ref[field], 'ROW_OR_TOKEN_IDENTITY:' + field)
        for label in ('new', 'true'):
            nll = row[label + '_nll']
            ensure(type(nll) in (int, float) and math.isfinite(nll), 'NONFINITE_NLL')
            n, c = row[label + '_token_count'], row[label + '_token_correct']
            ensure(type(n) is int and type(c) is int and n > 0 and 0 <= c <= n, 'TOKEN_COUNT_RELATION')
            ensure(type(row[label + '_strict']) is bool and row[label + '_strict'] == (n == c),
                   'STRICT_TOKEN_RELATION')
        ensure(math.isfinite(row['margin_true_minus_new']) and
               abs(row['margin_true_minus_new'] - row['true_nll'] + row['new_nll']) <= 1e-12,
               'EXPLICIT_MARGIN_SIGN')


def reduce_rows(rows):
    validate_rows(rows)
    result = {}
    for kind in KINDS:
        group = [r for r in rows if r['kind'] == kind]
        if not group:
            continue  # no missing-as-zero
        desired = 'true' if kind == 'N' else 'new'
        n = len(group)
        correct = sum(r[desired + '_token_correct'] for r in group)
        tokens = sum(r[desired + '_token_count'] for r in group)
        result[kind] = dict(
            denominator=n, numerator=sum(success(r) for r in group),
            rate=sum(success(r) for r in group) / n,
            true_nll_mean=math.fsum(r['true_nll'] for r in group) / n,
            new_nll_mean=math.fsum(r['new_nll'] for r in group) / n,
            desired_nll_mean=math.fsum(r[desired + '_nll'] for r in group) / n,
            desired_token_count=tokens, desired_token_correct=correct, token_micro=correct / tokens,
            prompt_macro=math.fsum(r[desired + '_token_correct'] / r[desired + '_token_count']
                                    for r in group) / n,
            strict_numerator=sum(r[desired + '_strict'] for r in group), strict_denominator=n,
            strict_rate=sum(r[desired + '_strict'] for r in group) / n,
            new_strict_numerator=sum(r['new_strict'] for r in group),
            true_minus_new_mean=math.fsum(r['true_nll'] - r['new_nll'] for r in group) / n,
            new_minus_true_mean=math.fsum(r['new_nll'] - r['true_nll'] for r in group) / n,
            true_nll_quantiles=quantiles([r['true_nll'] for r in group]),
            new_nll_quantiles=quantiles([r['new_nll'] for r in group]),
            desired_nll_quantiles=quantiles([r[desired + '_nll'] for r in group]),
            margin_true_minus_new_quantiles=quantiles([r['true_nll'] - r['new_nll'] for r in group]),
        )
    return result


def harmonic(summary):
    if any(k not in summary for k in KINDS):
        return None
    values = [summary[k]['rate'] for k in KINDS]
    return 0.0 if any(v == 0 for v in values) else 3 / math.fsum(1 / v for v in values)


def compare_summary(actual, stored):
    ensure(set(actual) == set(stored), 'SUMMARY_KINDS')
    for kind, fields in stored.items():
        for field, value in fields.items():
            got = actual[kind][field]
            ensure((type(value) is float and math.isclose(got, value, rel_tol=1e-12, abs_tol=1e-12))
                   or got == value, 'STORED_AGGREGATE_MISMATCH:' + kind + ':' + field)


def paired(before, after):
    validate_rows(before)
    validate_rows(after)
    b = {r['identity']: r for r in before}
    a = {r['identity']: r for r in after}
    ensure(set(a) == set(b), 'PAIRED_IDENTITY_SET')
    for identity in b:
        for field in ('case_id', 'kind', 'prompt_index', 'new_token_identity', 'true_token_identity',
                      'new_token_count', 'true_token_count'):
            ensure(b[identity][field] == a[identity][field], 'PAIRED_TOKEN_ROLE_COUNT')
    out = {}
    for kind in KINDS:
        keys = [k for k in b if b[k]['kind'] == kind]
        if not keys:
            continue
        out[kind] = {}
        for metric in ('preference', 'strict'):
            old = sum(success(b[k], metric) for k in keys)
            new = sum(success(a[k], metric) for k in keys)
            retained = sum(success(b[k], metric) and success(a[k], metric) for k in keys)
            lost = old - retained
            gained = new - retained
            ensure(new - old == gained - lost, 'PAIRED_CONSERVATION')
            out[kind][metric] = dict(denominator=len(keys), before=old, after=new, retained=retained,
                                     lost=lost, gained=gained, before_correct_retention=retained / old if old else None)
    return out


def active_flags(records):
    flags = {r['case_id']: True for r in records}
    claims = {}
    for row in records:
        rewrite = row['requested_rewrite']
        claim = (unicodedata.normalize('NFC', ' '.join(rewrite['subject'].split())), rewrite['relation_id'])
        target = rewrite['target_new'].get('id', rewrite['target_new']['str'])
        for prior, old in claims.get(claim, []):
            if target != old:
                flags[prior] = False
        claims.setdefault(claim, []).append((row['case_id'], target))
    return flags


def select_rows(rows, ids):
    wanted = set(ids)
    return [r for r in rows if r['case_id'] in wanted]


def read_endpoint(reader, folder, identities, ids, name, state, seen, reused_rows=None):
    paths = sorted(folder.glob('chunk-*.json'))
    if not paths:
        return None, None
    refs_by_case = {}
    for ref in identities:
        refs_by_case.setdefault(ref['case_id'], []).append(ref)
    expected = [ref for case in ids for ref in refs_by_case[case]]
    rows = []
    for path in paths:
        chunk = reader.json(path)
        ensure(chunk['state'] == state and chunk['optimizer_feedback'] is False, 'CHUNK_STATE_FEEDBACK')
        rows.extend(chunk['rows'])
    validate_rows(rows, expected, name)
    flags = active_flags(seen)
    ensure(all(type(r['active_at_endpoint']) is bool and r['active_at_endpoint'] == flags[r['case_id']]
               for r in rows), 'SEEN_PREFIX_ACTIVE_FLAGS')
    stored = reader.json(folder / 'summary.json')
    ensure(stored['state'] == state and stored['endpoint'] == name and stored['requests'] == len(ids),
           'SUMMARY_STATE_CARDINALITY')
    if 'reused_from' in stored:
        ensure(name == 'B1_PRE' and stored['new_forwards'] == 0 and reused_rows is not None,
               'EXACT_W0_SUBSET_REUSE_RECEIPT')
        expected_reuse = [dict(r, endpoint=name, active_at_endpoint=flags[r['case_id']])
                          for r in select_rows(reused_rows, ids)]
        ensure(rows == expected_reuse, 'W0_SUBSET_EXACT_RAW_VALUES')
        ensure(Path(stored['reused_from']) == folder.parent.parent / 'W0', 'W0_REUSE_ORIGIN')
    else:
        ensure(stored['row_count'] == len(rows) and
               stored['row_order'] == digest([r['identity'] for r in rows]), 'SUMMARY_ROW_ORDER_HASH')
        ensure(stored['replay'] is False, 'NO_REPLAY_RECEIPT')
    ensure(stored['no_mutation'] is True and stored['optimizer_feedback'] is False,
           'OBSERVER_NO_MUTATION_RECEIPT')
    summary = reduce_rows(rows)
    compare_summary(summary, stored['summary'])
    ensure({k: summary[k]['denominator'] for k in KINDS} ==
           dict(R=len(ids), P=2 * len(ids), N=10 * len(ids)), 'EXACT_R_P_N_DENOMINATORS')
    return rows, summary


def review_events(reader, path, arm, number, ids, layers, fit):
    requests = {str(i): {} for i in ids}
    layer_events = set()
    optimizer_events = set()
    terminal = {}
    for line in reader.bytes(path).splitlines():
        row = json.loads(line)
        ensure(row['run_id'] == SOURCE + ':' + arm and row['batch_index'] == number, 'EVENT_ARM_BATCH')
        rid, index = row['request_id'], row['candidate_index']
        ensure(rid in requests and type(index) is int and 0 <= index <= 24, 'EVENT_ID_INDEX')
        event, p = row['event'], row['payload']
        if event == 'candidate_request':
            ensure(index not in requests[rid], 'DUPLICATE_CANDIDATE')
            ensure(p['evaluation_ordinal'] == index + 1 and p['updates_completed'] == index, 'CANDIDATE_COUNTER')
            ensure(math.isfinite(p['objective_J']) and math.isfinite(p['budget_used']), 'CANDIDATE_FINITE')
            ensure(p['budget_used'] <= .750001, 'CANDIDATE_BUDGET')
            ensure(p['terminal_z_capture_candidate'] == index, 'SAME_EVALUATED_Z')
            if p['will_backward']:
                ensure(index < 24 and p['terminal_reason'] is None and p['objective_J'] >= .05,
                       'STOP_BEFORE_BACKWARD')
            else:
                ensure(p['kkt']['availability'] == 'NO_BACKWARD_TERMINAL' and
                       all(p['kkt'][k] is None for k in ('mu_hat', 'stationarity_l2', 'primal_violation')),
                       'NO_TERMINAL_GRADIENT')
                ensure(index == 24 or p['objective_J'] < .05, 'LAST_EVALUATED_STOP')
            requests[rid][index] = p
        elif event in ('candidate_layer', 'optimizer_layer'):
            layer = int(row['layer_id'])
            ensure(layer in layers, 'EVENT_ELIGIBLE_LAYER')
            target = layer_events if event == 'candidate_layer' else optimizer_events
            key = (rid, index, layer)
            ensure(key not in target, 'DUPLICATE_LAYER_EVENT')
            target.add(key)
            if event == 'optimizer_layer':
                ensure(p['update_ordinal'] == index + 1 and p['moments_preserved_after_projection']
                       and p['eligible_for_reentry'] and p['postcast_primal_violation'] <= 1e-6,
                       'OPTIMIZER_PROJECTION_RECEIPT')
        elif event == 'terminal_request':
            ensure(rid not in terminal, 'DUPLICATE_TERMINAL_REQUEST')
            terminal[rid] = row
        else:
            raise ValueError('UNRECOGNIZED_FIT_EVENT:' + event)
    updates = 0
    for rid, sequence in requests.items():
        ensure(sequence and sorted(sequence) == list(range(len(sequence))), 'CANDIDATE_CONTIGUITY')
        last = len(sequence) - 1
        ensure(rid in terminal and not sequence[last]['will_backward'] and
               all(sequence[k]['will_backward'] for k in range(last)), 'TERMINAL_FREEZE')
        ensure(terminal[rid]['candidate_index'] == last, 'TERMINAL_CANDIDATE_IDENTITY')
        final = terminal[rid]['payload']
        ensure(final['accepted_candidate_index'] == last and final['logical_evaluations'] == last + 1
               and final['optimizer_updates'] == final['backward_calls'] == last
               and final['gradient_kkt_status'] == 'NO_BACKWARD_TERMINAL'
               and final['frozen_after_terminal'] and final['kept_in_full_batch_writer']
               and final['accepted_z_capture_stage'] ==
               'CANONICAL_POST_INJECTION_EXISTING_EVALUATION_FORWARD', 'TERMINAL_Z_FREEZE_COUNTERS')
        ensure(fit['candidates'][rid]['candidate'] == last and
               fit['candidates'][rid]['reason'] == sequence[last]['terminal_reason'] and
               fit['candidates'][rid]['J'] == sequence[last]['objective_J'], 'FIT_TERMINAL_AGGREGATE')
        updates += last
        ensure(all((rid, k, layer) in layer_events for k in sequence for layer in layers), 'ALL_LAYER_COVERAGE')
        ensure(all((rid, k, layer) in optimizer_events for k in range(last) for layer in layers),
               'ALL_UPDATE_LAYER_COVERAGE')
        ensure(all((rid, last, layer) not in optimizer_events for layer in layers), 'NO_TERMINAL_UPDATE')
    evaluations = sum(len(v) for v in requests.values())
    ensure(len(layer_events) == evaluations * len(layers) and len(optimizer_events) == updates * len(layers),
           'NO_EXTRA_LAYER_EVENTS')
    ensure(fit['requests'] == len(ids) and fit['request_evaluations'] == evaluations and
           fit['request_updates'] == updates and fit['terminal_extra_backward'] == 0 and
           fit['terminal_extra_forward'] == 0, 'FIT_COUNTS_INDEPENDENT')
    return dict(requests=len(ids), request_evaluations=evaluations, request_updates=updates,
                stop_reasons=dict(Counter(v[max(v)]['terminal_reason'] for v in requests.values())),
                terminal_J=quantiles([v[max(v)]['objective_J'] for v in requests.values()]))


def review_arm(reader, attempt, arm, config, lock, identities, records, collector):
    root = attempt / ('main-' + arm)
    layers = config['profile']['eligible_layers']
    cold = config['qualification_reuse']['cold_W0_H0']
    ids_all = [r['case_id'] for r in records]
    w0, summary0 = read_endpoint(reader, root / 'W0', identities, ids_all, 'W0', cold, records)
    result = dict(arm=arm, status='PARTIAL', endpoints={}, current={}, milestones={}, paired={},
                  commits=0, next_entry_links=0, history_appends=0, fits=[], warnings=[], costs=[])
    raw = {}
    if w0 is None:
        result['warnings'].append('W0_NOT_AVAILABLE')
        return result, raw
    result['endpoints']['W0'] = summary0
    raw['W0'] = w0
    previous, previous_commit = cold, None
    atwrite = []
    config_digest = digest(config)
    ensure(not (root / 'batch-21').exists(), 'FORBIDDEN_BATCH21')
    for n, pack in enumerate(config['packs'], 1):
        folder = root / ('batch-%02d' % n)
        if not (folder / 'commit.json').exists() or (folder / 'rollback.json').exists():
            result['warnings'].append('UNCOMMITTED_PREFIX_END_B' + str(n))
            break
        commit, entry = reader.json(folder / 'commit.json'), reader.json(folder / 'entry.json')
        ensure(commit['source'] == entry['source'] == lock['source_commit'] == SOURCE and
               commit['config'] == entry['config'] == config_digest, 'COMMIT_SOURCE_CONFIG')
        ensure(commit['arm'] == entry['arm'] == arm and commit['batch'] == entry['batch'] == n, 'ARM_BATCH')
        ensure(commit['ids'] == entry['ids'] == pack['ids'] and
               commit['native_pack'] == entry['native_pack'] == pack['identity'], 'COMMIT_PACK_IDENTITY')
        seen = records[:n * config['settings']['B']]
        ensure(entry['seen_ids'] == [r['case_id'] for r in seen] and commit['seen_requests'] == len(seen),
               'NO_FUTURE_SEEN_PREFIX')
        ensure(commit['before'] == entry['state'] == previous, 'OWN_ENTRY_W_H_JOIN')
        ensure(commit['RNG_before'] == entry['RNG'] == commit['RNG_after'], 'OWN_RNG_NONMUTATION')
        if previous_commit:
            ensure(entry['RNG'] == previous_commit['RNG_after'] and
                   entry['context_hash'] == previous_commit['context_hash'], 'NEXT_ENTRY_RNG_CONTEXT')
        ensure(commit['cache_invalidated_for_next_batch'] and commit['observer_no_mutation']
               and commit['checkpoint_saved'] is False and commit['fit_count'] == entry['fit_count'] == 1,
               'CACHE_TRANSACTION_NOCP_RECEIPT')
        writer = reader.bound(commit['writer'])
        ensure(writer['branch'] == arm and writer['history_appends'] == commit['history_appends'] == len(layers),
               'EXACT_HISTORY_ONCE_COUNT')
        ensure({r['layer'] for r in writer['history']} == set(layers), 'H_LAYER_SET')
        for row in writer['history']:
            layer = str(row['layer'])
            ensure(row['append_count'] == 1 and row['columns'] == len(pack['ids']) and row['rewrite_only']
                   and row['KL_in_history'] is False and row['CPU_FP32'], 'REWRITE_ONLY_CPUFP32_H')
            ensure(row['before'] == previous['H'][layer] and row['after'] == commit['after']['H'][layer],
                   'H_HASH_JOIN')
            ensure(writer['layers'][layer]['weight_after'] == commit['after']['W'][layer], 'EXACT_WEIGHT_COMMIT')
        pre, pres = read_endpoint(reader, folder / 'pre', identities, pack['ids'], 'B%d_PRE' % n,
                                 previous, seen, reused_rows=w0 if n == 1 else None)
        post_ids = [r['case_id'] for r in seen] if n in MILESTONES else pack['ids']
        post, posts = read_endpoint(reader, folder / 'post', identities, post_ids, 'W%d' % n, commit['after'], seen)
        ensure(pre is not None and post is not None, 'COMMITTED_RAW_MISSING')
        current = select_rows(post, pack['ids'])
        currents = reduce_rows(current)
        compare_summary(pres, commit['pre'])
        compare_summary(posts, commit['post'])
        compare_summary(currents, commit['post_current'])
        compare_summary(posts, collector['arms'][arm]['endpoints']['W%d' % n])
        fit = reader.json(folder / 'fit/fit.json')
        fit_counts = review_events(reader, folder / 'fit-events.jsonl', arm, n, pack['ids'], layers, fit)
        result['fits'].append(dict(batch=n, **fit_counts))
        result['commits'] += 1
        result['history_appends'] += len(layers)
        result['next_entry_links'] += int(previous_commit is not None)
        result['endpoints']['W%d' % n] = posts
        result['current']['B%d' % n] = dict(pre=pres, post=currents,
                                           paired=paired(pre, current), harmonic=harmonic(currents))
        result['costs'].append(dict(batch=n, batch_seconds=commit['seconds'], fit_seconds=fit['seconds'],
                                    writer_seconds=writer['seconds'], nested_times_not_additive=True))
        atwrite.extend(current)
        if n in MILESTONES:
            label = 'W%d' % n
            raw[label] = post
            seen_ids = [r['case_id'] for r in seen]
            result['milestones'][label] = dict(
                harmonic=harmonic(posts), atwrite_to_endpoint=paired(atwrite, post),
                W0_to_endpoint=paired(select_rows(w0, seen_ids), post), cohorts={}, first_prefix={})
            milestone = result['milestones'][label]
            for cohort, born in enumerate(config['packs'][:n], 1):
                milestone['cohorts'][str(cohort)] = dict(
                    summary=reduce_rows(select_rows(post, born['ids'])),
                    paired=paired(select_rows(atwrite, born['ids']), select_rows(post, born['ids'])))
            for size in (100, 500, 1000, 1500):
                if size <= len(seen):
                    subset = seen_ids[:size]
                    milestone['first_prefix'][str(size)] = dict(
                        summary=reduce_rows(select_rows(post, subset)),
                        atwrite_to_endpoint=paired(select_rows(atwrite, subset), select_rows(post, subset)),
                        W0_to_endpoint=paired(select_rows(w0, subset), select_rows(post, subset)))
            milestone['active'] = reduce_rows([r for r in post if r['active_at_endpoint']])
            milestone['superseded'] = reduce_rows([r for r in post if not r['active_at_endpoint']]) or None
        previous, previous_commit = commit['after'], commit
    terminal = reader.json(root / 'terminal.json') if (root / 'terminal.json').exists() else None
    result['terminal'] = terminal
    complete = result['commits'] == 20 and result['next_entry_links'] == 19 and result['history_appends'] == 100
    result['status'] = 'W20_RAW_STATE_VERIFIED' if complete and terminal and terminal['status'] == 'W20_COMPLETE' else 'PARTIAL'
    result['request_evaluations'] = sum(f['request_evaluations'] for f in result['fits'])
    result['request_updates'] = sum(f['request_updates'] for f in result['fits'])
    for key in ('commits', 'next_entry_links', 'history_appends', 'request_evaluations', 'request_updates'):
        ensure(result[key] == collector['arms'][arm][key], 'COLLECTOR_COUNT_MISMATCH:' + key)
    compare_summary(summary0, collector['arms'][arm]['endpoints']['W0'])
    return result, raw


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def review(attempt, output):
    start = time.monotonic()
    ensure(output.resolve() != attempt.resolve() and not output.exists(), 'OUTPUT_MUST_BE_NEW_SEPARATE_DIRECTORY')
    reader = Reader()
    config = reader.json(attempt / 'config.json')
    lock = reader.json(attempt / 'execution.lock.json')
    ensure(reader.files[str(attempt / 'config.json')]['sha256'] == CONFIG_SHA and
           reader.files[str(attempt / 'execution.lock.json')]['sha256'] == LOCK_SHA and
           lock['config_sha256'] == CONFIG_SHA and lock['source_commit'] == SOURCE, 'EXACT_SOURCE_LOCK_CONFIG')
    identities = reader.bound(config['observer_identity'])['rows']
    records = reader.json(config['stream'])[:config['settings']['requests']]
    ids = [r['case_id'] for r in records]
    ensure(ids == [i for p in config['packs'] for i in p['ids']] and digest(ids) == config['ordered_ids_sha256'],
           'ORDERED_FIRST2000_AND_20PACKS')
    ensure(len(records) == 2000 and len(config['packs']) == 20 and len(identities) == 26000, 'FULL_INPUT_CARDINALITY')
    collector = reader.json(attempt / 'cpu-report/metrics.json')
    results, raw = {}, {}
    for arm in ('MD', 'CD'):
        results[arm], raw[arm] = review_arm(reader, attempt, arm, config, lock, identities, records, collector)
    cross = {}
    for label in ('W0', 'W5', 'W10', 'W15', 'W20'):
        if all(label in raw[a] for a in ('MD', 'CD')):
            cross[label] = paired(raw['MD'][label], raw['CD'][label])
    stored_cross = reader.json(attempt / 'cpu-report/paired-MD-CD.json')['MD_to_CD']
    for label, kinds in stored_cross.items():
        for kind, metrics in kinds.items():
            for metric, fields in metrics.items():
                for field, value in fields.items():
                    ensure(cross[label][kind][metric][field] == value, 'PAIRED_COLLECTOR_MISMATCH')
    result = dict(status='COMPLETE_RAW_CPU_VERIFIED' if all(r['status'] == 'W20_RAW_STATE_VERIFIED' for r in results.values())
                  else 'PARTIAL', source=SOURCE, config_sha256=CONFIG_SHA, lock_sha256=LOCK_SHA,
                  arms=results, MD_to_CD=cross, collector_aggregate_agreement=True,
                  scope=dict(new_GPU=0, model_load=0, tokenizer_reexecution=0, scheduler_calls=0,
                             token_checks='sealed token hash identity and stored count relations',
                             state_checks='receipt/hash links; RAM tensors not rebuilt',
                             reviewer='independent CPU implementation, not new LM parity'),
                  seconds=time.monotonic() - start)
    output.mkdir(parents=True)
    write_json(output / 'independent-metrics.json', result)
    write_json(output / 'raw-read-manifest.json', dict(files=list(reader.files.values()), source=SOURCE, raw_KEEP=True))
    table = []
    for arm, r in results.items():
        for label, summary in r['endpoints'].items():
            for kind, row in summary.items():
                table.append(dict(arm=arm, endpoint=label, kind=kind,
                    **{k: v for k, v in row.items() if type(v) in (int, float)}, harmonic_RS_PS_NS=harmonic(summary)))
        for label, current in r['current'].items():
            for phase in ('pre', 'post'):
                for kind, row in current[phase].items():
                    table.append(dict(arm=arm, endpoint=label + '_' + phase, kind=kind,
                        **{k: v for k, v in row.items() if type(v) in (int, float)},
                        harmonic_RS_PS_NS=harmonic(current[phase])))
    with (output / 'independent-comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    return dict(status=result['status'], output=str(output), files_read=len(reader.files),
                arms={a: dict(commits=r['commits'], joins=r['next_entry_links'], H=r['history_appends'],
                              W20=r['endpoints'].get('W20')) for a, r in results.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.attempt.resolve(), args.output.resolve()), allow_nan=False))


if __name__ == '__main__':
    main()
