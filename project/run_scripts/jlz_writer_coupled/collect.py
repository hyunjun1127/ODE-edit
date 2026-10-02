"""CPU-only saved-evidence collector. No model, CUDA, Slurm or task mutation.

Schema: arm-{A,B}/main/batch-XX/commit.json; main/observe-WXX/{chunk-*.json,
summary.json}; arm root initial.json and terminal.json (main terminal accepted).
W05/W10/W20 are all-seen endpoints; all endpoints retain their current subset.
Each invocation writes an immutable report snapshot to a new --report path.
"""
import argparse
import csv
import io
import json
import math
from pathlib import Path

from .common import INSTRUCTION, digest, member, require, sha, write
from .observe import reduce_rows

NOT_RECORDED = 'NOT_RECORDED'
EXPECTED = {'R': 2000, 'P': 4000, 'N': 20000}
MILESTONES = (5, 10, 20)


def _read(path, errors):
    if not path.is_file():
        return None
    try:
        with path.open() as source:
            return json.load(source)
    except (OSError, ValueError) as error:
        errors.append(dict(path=str(path), error=type(error).__name__))
        return None


def raw(path, errors=None, expected_state=None):
    """Read every completed raw chunk; an incomplete later chunk is reported."""
    errors = [] if errors is None else errors
    rows = []
    for chunk in sorted(Path(path).glob('chunk-*.json')):
        payload = _read(chunk, errors)
        if not isinstance(payload, dict) or not isinstance(payload.get('rows'), list):
            errors.append(dict(path=str(chunk), error='INVALID_CHUNK_ROWS'))
            continue
        if payload.get('optimizer_feedback') is not False:
            errors.append(dict(path=str(chunk), error='OBSERVER_FEEDBACK_BOUNDARY_NOT_RECORDED'))
        if expected_state is not None and payload.get('state') != expected_state:
            errors.append(dict(path=str(chunk), error='OBSERVER_CHUNK_COMMITTED_STATE_MISMATCH'))
        rows.extend(payload['rows'])
    return rows


def _reduce(rows, errors, context):
    try:
        return reduce_rows(rows)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        errors.append(dict(path=context, error=str(error)))
        return None


def transition(before, after):
    """Paired available-row transition, explicit coverage and missing-row counts.

    Lost/gained use strict NLL preference (ties fail). Conditional retention is
    conditioned only on recorded successful birth rows, never missing rows.
    """
    reduce_rows(before)
    reduce_rows(after)
    left, right = {r['identity']: r for r in before}, {r['identity']: r for r in after}
    result = {}
    for kind in sorted({r['kind'] for r in before}):
        offered = [r for r in before if r['kind'] == kind]
        pairs = [(r, right[r['identity']]) for r in offered if r['identity'] in right]
        require(all(l['kind'] == r['kind'] and l['case_id'] == r['case_id']
                    for l, r in pairs), 'PAIRED_ROW_IDENTITY_MISMATCH')
        success = lambda r: (r['true_nll'] < r['new_nll'] if kind == 'N'
                             else r['new_nll'] < r['true_nll'])
        birth = sum(success(l) for l, _ in pairs)
        lost = sum(success(l) and not success(r) for l, r in pairs)
        gained = sum(not success(l) and success(r) for l, r in pairs)
        desired = 'true' if kind == 'N' else 'new'
        strict_birth = sum(l[desired + '_strict'] for l, _ in pairs)
        strict_lost = sum(l[desired + '_strict'] and not r[desired + '_strict'] for l, r in pairs)
        result[kind] = dict(
            offered_rows=len(offered), denominator=len(pairs), missing_rows=len(offered) - len(pairs),
            coverage_complete=len(pairs) == len(offered), birth_success=birth,
            endpoint_success=sum(success(r) for _, r in pairs), lost=lost, gained=gained,
            retained=birth - lost,
            birth_conditional_retention=(birth - lost) / birth if birth else NOT_RECORDED,
            strict_birth_success=strict_birth, strict_lost=strict_lost,
            strict_gained=sum(not l[desired + '_strict'] and r[desired + '_strict'] for l, r in pairs),
            strict_retained=strict_birth - strict_lost,
            strict_birth_conditional_retention=(strict_birth - strict_lost) / strict_birth
                if strict_birth else NOT_RECORDED,
            true_nll_delta=sum(r['true_nll'] - l['true_nll'] for l, r in pairs) / len(pairs)
                if pairs else NOT_RECORDED,
            new_nll_delta=sum(r['new_nll'] - l['new_nll'] for l, r in pairs) / len(pairs)
                if pairs else NOT_RECORDED)
    return result


def _denominators(summary):
    return {kind: summary.get(kind, {}).get('denominator', 0) for kind in EXPECTED}


def _ids(commit):
    value = commit.get('current_ids') if isinstance(commit, dict) else None
    return value if isinstance(value, list) and all(type(x) in {str, int} for x in value) else []


def _completion_problems(commits, endpoints, current, terminal, errors):
    problems = []
    if not isinstance(terminal, dict) or terminal.get('status') != 'COMPLETED':
        problems.append('TERMINAL_NOT_COMPLETED')
    if set(commits) != set(range(1, 21)):
        problems.append('NOT_EXACTLY_20_COMMITS')
    for batch, commit in commits.items():
        if (commit.get('batch') != batch or commit.get('actual_B') != 100
                or len(_ids(commit)) != 100 or len(set(_ids(commit))) != 100):
            problems.append(f'B{batch:02d}_COMMIT_METADATA')
        candidates, backwards = commit.get('candidate_count'), commit.get('backward_count')
        accepted, rejected = commit.get('accepted_updates'), commit.get('rejected_trials')
        if (type(candidates) is not int or not 1 <= candidates <= 25
                or type(backwards) is not int or not 1 <= backwards <= 24
                or backwards not in {candidates, candidates - 1}
                or type(accepted) is not int or type(rejected) is not int
                or min(accepted, rejected) < 0 or accepted + rejected != candidates - 1
                or commit.get('history_appends') != 5):
            problems.append(f'B{batch:02d}_TECHNICAL_RECEIPT')
    for batch in range(1, 21):
        if _denominators(current.get(batch, {})) != {'R': 100, 'P': 200, 'N': 1000}:
            problems.append(f'W{batch:02d}_CURRENT_INCOMPLETE')
        if not endpoints.get(batch, {}).get('verified', False):
            problems.append(f'W{batch:02d}_RAW_SUMMARY_NOT_VERIFIED')
    for batch in MILESTONES:
        expected = {kind: count * batch // 20 for kind, count in EXPECTED.items()}
        if _denominators(endpoints.get(batch, {}).get('summary', {})) != expected:
            problems.append(f'W{batch:02d}_ALLSEEN_INCOMPLETE')
    all_ids = [case for c in commits.values() for case in _ids(c)]
    if len(all_ids) != 2000 or len(set(all_ids)) != 2000:
        problems.append('CURRENT_OCCURRENCE_IDS_NOT_2000_UNIQUE')
    if errors:
        problems.append('EVIDENCE_INTEGRITY_ERRORS')
    return problems


def _w0(attempt, config, errors):
    receipt_path = attempt / 'w0-reuse.json'
    if not receipt_path.is_file():
        receipt_path = attempt / 'W0-reuse.json'
    receipt = _read(receipt_path, errors)
    if receipt is None and isinstance(config, dict) and config.get('w0_reuse'):
        bridge = config['w0_reuse'].get('receipt', {})
        if bridge.get('path'):
            receipt_path = Path(bridge['path'])
            if not receipt_path.is_absolute():
                receipt_path = attempt / receipt_path
            if receipt_path.is_file() and bridge.get('sha256') == sha(receipt_path):
                receipt = _read(receipt_path, errors)
            else:
                errors.append(dict(path=str(receipt_path), error='W0_BRIDGE_IDENTITY_MISMATCH'))
    if not isinstance(receipt, dict):
        return [], dict(status=NOT_RECORDED)
    reference = receipt.get('observations', receipt.get('raw', {}))
    if not reference and receipt.get('raw_path'):
        reference = dict(path=receipt['raw_path'], sha256=receipt.get('raw_sha256'))
    if not isinstance(reference, dict) or not reference.get('path'):
        errors.append(dict(path=str(receipt_path), error='W0_RAW_REFERENCE_NOT_RECORDED'))
        return [], dict(status=NOT_RECORDED)
    path = Path(reference['path'])
    if not path.is_absolute():
        path = attempt / path
    if not path.is_file() or reference.get('sha256') != sha(path):
        errors.append(dict(path=str(path), error='W0_RAW_IDENTITY_MISMATCH'))
        return [], dict(status=NOT_RECORDED)
    payload = _read(path, errors)
    rows = payload.get('rows', []) if isinstance(payload, dict) else []
    summary = _reduce(rows, errors, str(path))
    if summary is None:
        return [], dict(status=NOT_RECORDED)
    return rows, dict(status='REUSED_HISTORICAL_NO_NEW_FORWARD', receipt=member(receipt_path),
                     raw=member(path), summary=summary,
                     numerical_bitwise_equivalence='NOT_ESTABLISHED')


def _scalars(value, prefix=''):
    """Keep numeric costs only, excluding arbitrary strings/text/tensor arrays."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = f'{prefix}.{key}' if prefix else str(key)
            if isinstance(item, dict):
                result.update(_scalars(item, name))
            elif type(item) in {int, float} and math.isfinite(item):
                result[name] = item
    return result


def _text(path, text):
    require(not path.exists() or path.read_text() == text, 'IMMUTABLE_REPORT_CONFLICT:' + str(path))
    if not path.exists():
        with path.open('x') as output:
            output.write(text)


def _csv(path, rows, base_fields):
    fields = list(dict.fromkeys(base_fields + sorted({k for r in rows for k in r})))
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, restval=NOT_RECORDED)
    writer.writeheader()
    writer.writerows(rows)
    _text(path, buffer.getvalue())


def collect(attempt, report):
    attempt, report = Path(attempt).resolve(), Path(report).resolve()
    require(attempt.is_dir(), 'ATTEMPT_DIRECTORY_MISSING')
    require(report != attempt and report not in attempt.parents, 'REPORT_MUST_NOT_REPLACE_ATTEMPT')
    require(not report.exists() or not any(report.iterdir()), 'REPORT_PATH_MUST_BE_NEW_OR_EMPTY')
    report.mkdir(parents=True, exist_ok=True)
    global_errors = []
    config = _read(attempt / 'config.json', global_errors)
    if config is None:
        config = _read(attempt / 'configuration.json', global_errors)
    w0rows, w0receipt = _w0(attempt, config, global_errors)
    metrics, costs, paired, arm_reports, retention_table = [], [], {}, {}, []
    for arm in ('A', 'B'):
        base, errors = attempt / f'arm-{arm}', []
        main = base / 'main'
        commits, endpoints, current, birth, cohorts = {}, {}, {}, [], {}
        initial = _read(base / 'initial.json', errors)
        terminal_path = base / 'terminal.json'
        if not terminal_path.is_file():
            terminal_path = main / 'terminal.json'
        terminal = _read(terminal_path, errors)
        for batch in range(1, 21):
            commit_path = main / f'batch-{batch:02d}' / 'commit.json'
            commit = _read(commit_path, errors)
            if isinstance(commit, dict):
                commits[batch] = commit
                cost = {key: commit.get(key, NOT_RECORDED) for key in (
                    'actual_B', 'candidate_count', 'backward_count', 'accepted_updates',
                    'rejected_trials', 'history_appends', 'seconds')}
                for key in ('cost', 'timing', 'allocation', 'accepted_stats'):
                    cost.update(_scalars(commit.get(key), key))
                costs.append(dict(arm=arm, batch=batch, **cost))
            path = main / f'observe-W{batch:02d}'
            rows = raw(path, errors, commit.get('after') if isinstance(commit, dict) else None)
            stored = _read(path / 'summary.json', errors)
            if not rows and stored is None:
                continue
            summary = _reduce(rows, errors, str(path))
            if summary is None:
                continue
            ids = set(_ids(commit))
            current_rows = [r for r in rows if r['case_id'] in ids]
            current_summary = reduce_rows(current_rows)
            current[batch] = current_summary
            verified = False
            if isinstance(stored, dict):
                verified = (stored.get('summary') == summary
                            and stored.get('current') == current_summary
                            and stored.get('row_count') == len(rows)
                            and stored.get('row_order') == digest([r['identity'] for r in rows])
                            and stored.get('no_mutation') is True
                            and stored.get('optimizer_feedback') is False
                            and (not isinstance(commit, dict) or 'after' not in commit
                                 or stored.get('state') == commit['after']))
                if not verified:
                    errors.append(dict(path=str(path), error='INDEPENDENT_REDUCER_OR_OBSERVER_RECEIPT_MISMATCH'))
            endpoints[batch] = dict(summary=summary, verified=verified, rows=rows)
            birth.extend(current_rows)
            cohorts[batch] = current_rows
            scope = 'allseen' if batch in MILESTONES else 'current'
            for name, reduction in ((scope, summary), ('current', current_summary)):
                if name == 'current' and scope == 'current' and reduction is current_summary:
                    continue
                for kind, values in reduction.items():
                    metrics.append(dict(arm=arm, batch=batch, scope=name, kind=kind, **values))
        birth_summary = _reduce(birth, errors, str(main) + '/at-write') if birth else None
        if birth_summary is not None:
            for kind, values in birth_summary.items():
                metrics.append(dict(arm=arm, batch=max(cohorts), scope='birth', kind=kind, **values))
        problems = _completion_problems(commits, endpoints, current, terminal, errors)
        latest = max((b for b in MILESTONES if b in endpoints), default=None)
        if latest is not None:
            final = endpoints[latest]['rows']
            relevant_birth = [r for b, records in cohorts.items() if b <= latest for r in records]
            endpoint_case_ids = {case for b, c in commits.items() if b <= latest for case in _ids(c)}
            try:
                active_present = all(type(r.get('active_at_endpoint')) is bool for r in final)
                pair = dict(endpoint=f'W{latest:02d}',
                    atwrite_to_endpoint=transition(relevant_birth, final),
                    W0_to_endpoint=transition([r for r in w0rows if r['case_id'] in endpoint_case_ids], final)
                        if w0rows else NOT_RECORDED,
                    active=reduce_rows([r for r in final if r['active_at_endpoint']])
                        if active_present else NOT_RECORDED,
                    superseded=reduce_rows([r for r in final if not r['active_at_endpoint']])
                        if active_present else NOT_RECORDED,
                    cohorts={str(b): transition(rows, final) for b, rows in cohorts.items() if b <= latest},
                    W10_to_W20=transition(endpoints[10]['rows'], final)
                        if latest == 20 and 10 in endpoints else NOT_RECORDED)
                paired[arm] = pair
                for name in ('atwrite_to_endpoint', 'W0_to_endpoint', 'W10_to_W20'):
                    if isinstance(pair[name], dict):
                        for kind, values in pair[name].items():
                            retention_table.append(dict(arm=arm, endpoint=pair['endpoint'],
                                                        transition=name, kind=kind, **values))
                for cohort, reduction in pair['cohorts'].items():
                    for kind, values in reduction.items():
                        retention_table.append(dict(arm=arm, endpoint=pair['endpoint'],
                            transition=f'cohort-{int(cohort):02d}', kind=kind, **values))
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                errors.append(dict(path=str(main), error='PAIRED_REDUCTION:' + str(error)))
                problems.append('PAIRED_REDUCTION_FAILED')
                paired[arm] = NOT_RECORDED
        else:
            paired[arm] = NOT_RECORDED
        sums = {}
        for key in ('candidate_count', 'backward_count', 'accepted_updates', 'rejected_trials', 'seconds'):
            values = [c[key] for c in commits.values() if type(c.get(key)) in {int, float}
                      and math.isfinite(c[key])]
            sums[key] = sum(values) if values else NOT_RECORDED
            sums[key + '_recorded_commits'] = len(values)
        arm_reports[arm] = dict(
            status='COMPLETED' if not problems else 'PARTIAL_OR_TECHNICAL_FAILED',
            commits=len(commits), recorded_batches=sorted(commits),
            birth=birth_summary if birth_summary is not None else NOT_RECORDED,
            current_verified_batches=[b for b, s in current.items()
                                      if _denominators(s) == {'R': 100, 'P': 200, 'N': 1000}],
            endpoints={f'W{b:02d}': dict(summary=e['summary'], verified=e['verified'])
                       for b, e in endpoints.items()},
            terminal_status=terminal.get('status', NOT_RECORDED) if isinstance(terminal, dict) else NOT_RECORDED,
            initial={k: (dict(recorded=True, identity=digest(initial[k]))
                         if isinstance(initial.get(k), (dict, list)) else initial.get(k, NOT_RECORDED))
                     for k in ('main_B1_committed', 'history_appends', 'observer_no_mutation', 'main_B2_entry')}
                     if isinstance(initial, dict) else NOT_RECORDED,
            cost=sums,
            terminal_cost={k: terminal.get(k, NOT_RECORDED) for k in
                           ('seconds', 'peak_host_kib', 'peak_gpu_bytes')}
                           if isinstance(terminal, dict) else NOT_RECORDED,
            component_cost=(_scalars(terminal.get('cost')) or NOT_RECORDED)
                           if isinstance(terminal, dict) else NOT_RECORDED,
            missing_or_invalid=problems, errors=errors)
    status = ('COMPLETED' if not global_errors
              and all(r['status'] == 'COMPLETED' for r in arm_reports.values())
              else 'PARTIAL_OR_TECHNICAL_FAILED')
    result = dict(schema='jlz-writer-coupled-v5-collection-v1', instruction=INSTRUCTION,
                  status=status, expected_commits_per_arm=20, expected_final_denominators=EXPECTED,
                  arms=arm_reports, w0=w0receipt, errors=global_errors,
                  missing_value=NOT_RECORDED, cpu_saved_evidence_only=True,
                  raw_rows_independently_reduced=True, checkpoint_saved=False,
                  exact_resume='NOT_AVAILABLE', source=member(Path(__file__)))
    # No raw rows, prompt strings, teacher/tensor payloads are copied to report.
    write(report / 'summary.json', result)
    write(report / 'paired.json', paired)
    write(report / 'W0-reuse.json', w0receipt)
    _csv(report / 'metrics.csv', metrics, ['arm', 'batch', 'scope', 'kind'])
    _csv(report / 'costs.csv', costs, ['arm', 'batch'])
    _csv(report / 'retention.csv', retention_table, ['arm', 'endpoint', 'transition', 'kind'])
    lines = ['# JLZ v5 saved-evidence report', '', f'Status: {status}.', '',
             'CPU-only independent raw reduction; no new evaluation, scheduler query, or optimizer feedback.', '',
             '| Arm | Status | Commits | Last all-seen endpoint |', '|---|---|---:|---|']
    for arm, value in arm_reports.items():
        endpoints = [key for key in value['endpoints'] if int(key[1:]) in MILESTONES]
        lines.append(f"| {arm} | {value['status']} | {value['commits']}/20 | {max(endpoints, default=NOT_RECORDED)} |")
    lines.extend(['', 'Final required denominators per arm: R=2000, P=4000, N=20000.',
                  f"W0: {w0receipt['status']}. Missing metrics are {NOT_RECORDED}, never zero or PASS.",
                  'metrics.csv: strict NLL preference, teacher-forced strict/token/prompt metrics and NLL means.',
                  'retention.csv / paired.json: recorded at-write, cohort, W0 and W10 transitions; active/superseded split.',
                  'costs.csv: actual recorded candidates, backwards, accepts/rejects and costs; missing components are not inferred.',
                  'Raw stays in the attempt; this report copies no prompts or tensor payloads. noCP; exact resume unavailable.'])
    _text(report / 'report.md', '\n'.join(lines) + '\n')
    inventory = [member(p) for p in sorted(attempt.rglob('*.json'))
                 if p.is_file() and not p.is_relative_to(report)]
    write(report / 'artifact-index.json', inventory)
    # Completion marker is last, after compact report and independent inventory.
    write(report / 'terminal.json', dict(status=status, arms={a: r['status'] for a, r in arm_reports.items()},
          report=member(report / 'report.md'), artifact_index=member(report / 'artifact-index.json'),
          raw_rows_independently_reduced=True, checkpoint_saved=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.attempt, args.report)
    print(json.dumps(dict(status=result['status'], report=str(args.report)), sort_keys=True))


if __name__ == '__main__':
    main()
