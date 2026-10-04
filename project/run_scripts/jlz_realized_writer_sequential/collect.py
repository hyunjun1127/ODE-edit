"""Independent CPU raw/count/state reducer. Missing coverage is never zero."""
import argparse, csv, json, math, statistics, subprocess, time, traceback
from pathlib import Path
from scripts.fixed_counterfact import load_prefix
from project.run_scripts.jlz_realization.observe import reduce_rows, active_flags
from project.run_scripts.jlz_shared_budget.telemetry import validate_relations
from project.run_scripts.jlz_realized_writer.collect import paired, telemetry_summary
from .common import *

def harmonic(summary):
    values = [summary[k]['rate'] for k in ('R', 'P', 'N')]
    return 0.0 if any(x == 0 for x in values) else 3 / sum(1 / x for x in values)

def aggregate(values):
    vals = [v for v in values if v is not None]
    return dict(count=len(vals), undefined=len(values) - len(vals), mean=statistics.mean(vals) if vals else None,
        median=statistics.median(vals) if vals else None, max=max(vals) if vals else None)

def endpoint(folder, identities, ids, name, seen):
    rows = rows_from(folder)
    if not rows:
        return None, None
    summary = validate_rows(rows, identities, ids, name)
    stored = json.loads((folder / 'summary.json').read_text())
    require(summary == stored['summary'], 'INDEPENDENT_RAW_REDUCER')
    flags = active_flags(seen)
    require(all(r['active_at_endpoint'] == flags[r['case_id']] for r in rows), 'SEEN_PREFIX_ACTIVE_REDUCER')
    require(stored['no_mutation'] and stored['optimizer_feedback'] is False, 'OBSERVER_RECEIPT')
    return rows, summary

def validate_commit(commit, writer, entry, expected, previous, source, config_hash, layers):
    require(commit['source'] == entry['source'] == source and commit['config'] == entry['config'] == config_hash, 'COMMIT_SOURCE_CONFIG')
    require(commit['ids'] == entry['ids'] == expected['ids'] and commit['native_pack'] == entry['native_pack'] == expected['identity'], 'COMMIT_INPUT_IDENTITY')
    require(commit['before'] == entry['state'] == previous, 'OWN_W_H_JOIN')
    require(commit['RNG_before'] == entry['RNG'] == commit['RNG_after'], 'OWN_RNG_JOIN')
    require(commit['fit_count'] == 1 and commit['history_appends'] == writer['history_appends'] == len(layers), 'FIT_HISTORY_ONCE')
    require(commit['observer_no_mutation'] and not commit['checkpoint_saved'], 'COMMIT_POLICY')
    require({r['layer'] for r in writer['history']} == set(layers), 'H_LAYER_SET')
    for row in writer['history']:
        l = str(row['layer'])
        require(row['append_count'] == 1 and row['columns'] == len(expected['ids']) and row['rewrite_only']
            and row['KL_in_history'] is False and row['CPU_FP32'], 'NATIVE_HISTORY_SEMANTICS')
        require(row['before'] == previous['H'][l] and row['after'] == commit['after']['H'][l], 'H_IDENTITY_JOIN')
        saved = writer['layers'][l]
        require(saved['weight_after'] == commit['after']['W'][l], 'EXACT_WRITER_WEIGHT_COMMIT')
        require(saved['solver']['numerical_projection_verified'] and saved['ideal_effective_parity']['pass_'], 'SOLVER_PARITY')
    return commit['after']

def reduce_arm(attempt, arm, c, lock, identities, records, out):
    root = attempt / ('main-' + arm); layers = c['profile']['eligible_layers']; summary = {}
    coverage = {}; warnings = []; commits = []; telemetry = []; raw_endpoints = {}; atwrite = []; pairs = {}
    raw_w0, s0 = endpoint(root / 'W0', identities, [r['case_id'] for r in records], 'W0', records)
    if s0:
        summary['W0'] = s0; coverage['W0'] = 'COMPLETE'; raw_endpoints['W0'] = raw_w0
        previous = json.loads((root / 'W0/summary.json').read_text())['state']
        require(previous == c['qualification_reuse']['cold_W0_H0'], 'ARM_COLD_STATE')
    else:
        previous = None; coverage['W0'] = 'NOT_MEASURED'
    require(not (root / 'batch-21').exists(), 'FORBIDDEN_B21')
    fits = []
    for number, current, seen in batches(records, c['settings']['B']):
        folder = root / f'batch-{number:02d}'; label = f'W{number}'
        if not (folder / 'commit.json').exists() or (folder / 'rollback.json').exists():
            coverage[label] = 'NOT_COMMITTED'; continue
        try:
            require(previous is not None and len(commits) == number - 1, 'CONTIGUOUS_SUCCESSFUL_PREFIX')
            commit = json.loads((folder / 'commit.json').read_text()); entry = json.loads((folder / 'entry.json').read_text())
            writer = json.loads(verify(commit['writer']).read_text())
            require(commit['arm'] == entry['arm'] == arm and commit['batch'] == entry['batch'] == number, 'ARM_BATCH_IDENTITY')
            if commits:
                require(entry['RNG'] == commits[-1]['RNG_after'] and entry['context_hash'] == commits[-1]['context_hash'], 'NEXT_ENTRY_RNG_CONTEXT')
            after = validate_commit(commit, writer, entry, c['packs'][number - 1], previous, lock['source_commit'], digest(c), layers)
            expected_ids = [r['case_id'] for r in selected_for_post(current, seen, number)]
            post, post_summary = endpoint(folder / 'post', identities, expected_ids, label, seen)
            pre, pre_summary = endpoint(folder / 'pre', identities, [r['case_id'] for r in current], f'B{number}_PRE', seen)
            require(post is not None and pre is not None and pre_summary == commit['pre'] and post_summary == commit['post'], 'COMMIT_EVAL_COVERAGE')
            post_current = [r for r in post if r['case_id'] in set(entry['ids'])]
            require(validate_rows(post_current, identities, entry['ids'], label) == commit['post_current'], 'CURRENT_FROM_SAME_RAW')
            require(json.loads((folder / 'post/summary.json').read_text())['state'] == after, 'EVALUATED_COMMIT_STATE')
            require(json.loads((folder / 'pre/summary.json').read_text())['state'] == previous, 'PRE_ENTRY_STATE')
            fit = validate_relations(folder / 'fit-events.jsonl', layers, entry['ids'])
            stored_fit = json.loads((folder / 'fit/fit.json').read_text())
            require(all(stored_fit[k] == fit[k] for k in ('requests', 'request_evaluations', 'request_updates')), 'FIT_COUNT_REDUCER')
            require(fit['request_evaluations'] <= 25 * len(current) and fit['request_updates'] <= 24 * len(current), 'REQUEST_BUDGET')
            with (folder / 'fit-events.jsonl').open() as f:
                for line in f:
                    event = json.loads(line)
                    require(event['run_id'] == lock['source_commit'] + ':' + arm and event['batch_index'] == number, 'EVENT_ARM_NAMESPACE')
            fits.append(fit); commits.append(commit); previous = after
            summary[label] = post_summary; coverage[label] = 'COMPLETE'; atwrite.extend(post_current)
            pairs[f'B{number}_pre_post'] = paired(pre, post_current)
            actions = json.loads((folder / 'writer/actions.json').read_text())
            require({r['case_id'] for r in actions['rows']} == set(entry['ids']) and all(r['branch'] == arm for r in actions['rows']), 'ACTION_ARM_IDENTITY')
            require(all(x['pass_'] for x in json.loads((folder / 'writer/local-additivity.json').read_text())['checks'].values()), 'LOCAL_ADDITIVITY')
            shares = json.loads((folder / 'writer/shares.json').read_text())
            telemetry.append(dict(batch=number, native_context=telemetry_summary(actions['rows']), native_mean=actions['mean'],
                share_L1={scope: aggregate([r.get('share_L1') for r in shares[scope]]) for scope in ('canonical', 'mean', 'contexts')},
                shares=shares, layer_solvers=writer['layers'], seconds=writer['seconds']))
            if number in MILESTONES:
                raw_endpoints[label] = post
                birth = [r for r in atwrite if r['case_id'] in {x['case_id'] for x in seen}]
                pairs[label] = dict(atwrite_to_endpoint=paired(birth, post), W0_to_endpoint=paired(
                    [r for r in raw_w0 if r['case_id'] in {x['case_id'] for x in seen}], post), cohorts={})
                for cohort, born, _ in batches(seen, c['settings']['B']):
                    ids = {r['case_id'] for r in born}
                    pairs[label]['cohorts'][str(cohort)] = paired([r for r in birth if r['case_id'] in ids], [r for r in post if r['case_id'] in ids])
                pairs[label]['first_prefix'] = {}
                for size in (100, 500, 1000, 1500):
                    if size <= len(seen):
                        ids = {r['case_id'] for r in seen[:size]}
                        subset = [r for r in post if r['case_id'] in ids]
                        pairs[label]['first_prefix'][str(size)] = dict(summary=reduce_rows(subset),
                            atwrite=paired([r for r in birth if r['case_id'] in ids], subset))
                active = [r for r in post if r['active_at_endpoint']]
                inactive = [r for r in post if not r['active_at_endpoint']]
                pairs[label]['active_summary'] = reduce_rows(active)
                pairs[label]['superseded_summary'] = reduce_rows(inactive) if inactive else 'EMPTY_POPULATION'
        except Exception as error:
            warnings.append(dict(batch=number, type=type(error).__name__, error=str(error)))
            coverage[label] = 'TECHNICAL_REDUCER_MISMATCH'
            break
    terminal = json.loads((root / 'terminal.json').read_text()) if (root / 'terminal.json').exists() else dict(status='NOT_RECORDED')
    complete = not warnings and len(commits) == 20 and all(coverage.get(f'W{n}') == 'COMPLETE' for n in range(1, 21))
    require(not complete or len(commits) * len(layers) == 100, 'TOTAL_HISTORY_COUNT')
    status = 'W20_COMPLETE' if complete and terminal['status'] == 'W20_COMPLETE' else 'PARTIAL_OR_TECHNICAL_BLOCKED'
    result = dict(status=status, coverage=coverage, warnings=warnings, endpoints=summary, paired=pairs,
        commits=len(commits), next_entry_links=max(0, len(commits) - 1), history_appends=sum(r['history_appends'] for r in commits),
        fit_count=len(fits), request_evaluations=sum(r['request_evaluations'] for r in fits),
        request_updates=sum(r['request_updates'] for r in fits), terminal=terminal, harmonic={k: harmonic(v) for k, v in summary.items()})
    write(out / (arm + '-metrics.json'), result); write(out / (arm + '-realization.json'), dict(batches=telemetry))
    return result, raw_endpoints

def collect(attempt):
    started = time.monotonic(); out = attempt / 'cpu-report'; out.mkdir(exist_ok=False)
    c = json.loads((attempt / 'config.json').read_text()); lock = json.loads((attempt / 'execution.lock.json').read_text())
    require(c['instruction_id'] == lock['instruction_id'] == NONCE and sha(attempt / 'config.json') == lock['config_sha256'], 'COLLECTOR_SOURCE_CONFIG')
    identities = json.loads(verify(c['observer_identity']).read_text())['rows']
    records = load_prefix(Path(c['stream']).parent, c['settings']['requests'])
    arms = {}; raw = {}; errors = []
    for arm in ARMS:
        try:
            arms[arm], raw[arm] = reduce_arm(attempt, arm, c, lock, identities, records, out)
        except Exception as error:
            errors.append(dict(arm=arm, type=type(error).__name__, error=str(error), trace=traceback.format_exc()))
            arms[arm] = dict(status='TECHNICAL_REDUCER_BLOCKED', coverage={}, warnings=[str(error)], endpoints={})
    cross = {}
    for label in ('W0',) + tuple(f'W{x}' for x in MILESTONES):
        if all(label in raw.get(arm, {}) for arm in ARMS):
            cross[label] = paired(raw['MD'][label], raw['CD'][label])
    write(out / 'paired-MD-CD.json', dict(MD_to_CD=cross, same_order=True, trajectories_independent=True))
    rows = []
    for arm, result in arms.items():
        for label, summary in result['endpoints'].items():
            for kind, s in summary.items():
                rows.append(dict(arm=arm, endpoint=label, kind=kind, **s, harmonic_RS_PS_NS=harmonic(summary)))
    if rows:
        with (out / 'comparison-W20.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    submission = json.loads((attempt / 'submission.json').read_text()) if (attempt / 'submission.json').exists() else {}
    cost = dict(status='NOT_RECORDED')
    if submission.get('jobs'):
        response = subprocess.run(['sacct', '-X', '-P', '-n', '-j', ','.join(submission['jobs'].values()),
            '--format=JobIDRaw,JobName,State,ElapsedRaw,AllocTRES'], capture_output=True, text=True)
        cost = dict(returncode=response.returncode, parent_rows=response.stdout,
            rule='each exact GPU parent ElapsedRaw*AllocTRES gres/gpu once; CPU collector GPU0; wall != ETA')
    write(out / 'cost.json', dict(scheduler=cost, measured={arm: r.get('terminal', 'NOT_MEASURED') for arm, r in arms.items()}))
    status = 'COMPLETED' if not errors and all(r['status'] == 'W20_COMPLETE' for r in arms.values()) else 'PARTIAL_OR_TECHNICAL_BLOCKED'
    write(out / 'metrics.json', dict(status=status, arms=arms, errors=errors, source=lock['source_commit'], config=digest(c)))
    lines = ['# V13 MD/CD sequential 2k 사실 보고', '', f'상태: {status}', f'실행 source: `{lock["source_commit"]}`',
        'MD/CD 독립 cold W0/H0, 같은 first2000/BS100×20. 기존 B1 source/planner/writer 재사용; 추가 baseline/fit/pilot 없음.',
        '새 sequential CPU 검산과 기존 S4 B1 actual qualification 재사용을 구분했다. Owner audit; 별도 독립 reviewer 미사용.',
        '', '| arm | 성공 commit | W20 RS | W20 PS | W20 NS |', '|---|---:|---:|---:|---:|']
    for arm, r in arms.items():
        endpoint_summary = r['endpoints'].get('W20', {})
        values = [f'{endpoint_summary[k]["numerator"]}/{endpoint_summary[k]["denominator"]}' if k in endpoint_summary else 'NOT_MEASURED' for k in ('R', 'P', 'N')]
        lines.append('| ' + arm + ' | ' + str(r.get('commits', 'NOT_VERIFIED')) + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '분모: W20 R2000/P4000/N20000 각 arm. TF strict는 자유생성이 아니다. 누락/부분은 0점으로 대체하지 않았다.',
        'margin_true_minus_new=true NLL−new NLL. MD→CD paired 및 atwrite/cohort/pre→post는 같은 row/token identity만 연결한다.',
        'Raw/plan/tensor/model 미게시; noCP, exact_resume=NOT_AVAILABLE. NO_BROADCAST_NOT_REQUIRED: 같은 서버, compact 보고와 manifest만 공유.',
        f'원자료: `{attempt}`', '다른 실행 조건 baseline은 자동 matched 비교하지 않았다. 신규 baseline 실행 0. 과학적 우월성/후속 선정 없음.']
    (out / 'report-ko.md').write_text('\n'.join(lines) + '\n')
    inventory = [member(p) for arm in ARMS for p in sorted((attempt / ('main-' + arm)).rglob('*')) if p.is_file()]
    write(out / 'inventory.json', dict(files=inventory, raw_local_KEEP=True, no_checkpoint=True, source=lock['source_commit']))
    write(out / 'terminal.json', dict(status=status, report=member(out / 'report-ko.md'), inventory=member(out / 'inventory.json'),
        summary=member(out / 'metrics.json'), seconds=time.monotonic() - started))
    return dict(status=status, report=str(out / 'report-ko.md'))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--attempt', type=Path, required=True)
    args = p.parse_args(); print(json.dumps(collect(args.attempt.resolve())))
