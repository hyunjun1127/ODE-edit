"""CPU-only compact report for completed local observer evidence.

This module does not submit, inspect, wait for, or monitor jobs. A sealed runner
may call build/write once after B100; missing files remain NOT_OBSERVED. An
explicit later user review can compare the two independently completed arms.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

from project.run_scripts.en_adaptive_nullspace import metrics as parent_metrics
from project.run_scripts.en_adaptive_nullspace.json_io import save, scalar
from . import observers


def _load(path, provenance):
    """Read only the exact local path requested, with its complete input seal."""
    path = Path(path)
    if not path.is_file():
        return None
    raw = path.read_bytes()
    provenance[str(path.resolve())] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    decoded = gzip.decompress(raw) if path.suffix == '.gz' else raw
    return json.loads(decoded)


def _member(directory, name, provenance):
    plain, packed = Path(directory) / (name + '.json'), Path(directory) / (name + '.json.gz')
    if plain.exists() and packed.exists():
        raise ValueError(f'ambiguous plain/compressed evidence: {name}')
    return _load(packed if packed.exists() else plain, provenance)


def _endpoint_rows(summary, batch, arm, scope):
    output = []
    for family in observers.FAMILIES:
        p, a = summary['metrics'][family], summary['aggregates'][family]
        if not p['denominator']:
            continue
        output.append(dict(batch=batch, arm=arm, scope=scope, family=family,
            observed_requests=summary['requests'], request_order=summary['request_order'],
            preference_success=p['numerator'], preference_prompts=p['denominator'], preference_rate=p['rate'],
            tf_token_correct=a['tf_token_micro']['numerator'], tf_valid_tokens=a['tf_token_micro']['denominator'],
            tf_token_micro=a['tf_token_micro']['rate'], tf_prompt_macro=a['tf_prompt_macro']['value'],
            tf_strict_correct=a['tf_strict']['numerator'], tf_strict_prompts=a['tf_strict']['denominator'],
            tf_strict_rate=a['tf_strict']['rate'], new_nll=a['nll']['new_prompt_macro'],
            true_nll=a['nll']['true_prompt_macro'], desired_nll=a['nll']['desired_prompt_macro'],
            desired_margin=a['desired_margin_prompt_macro']))
    joint = summary['joint']
    if joint['eligible_requests']:
        output.append(dict(batch=batch, arm=arm, scope=scope, family='R_plus_twoP_joint',
            observed_requests=summary['requests'], request_order=summary['request_order'],
            preference_success=joint['preference']['numerator'], preference_prompts=joint['preference']['denominator'],
            preference_rate=joint['preference']['rate'], tf_strict_correct=joint['tf_strict']['numerator'],
            tf_strict_prompts=joint['tf_strict']['denominator'], tf_strict_rate=joint['tf_strict']['rate']))
    return output


def _paired_rows(analysis, batch, arm, scope, comparison):
    output = []
    if analysis.get('status') == 'NOT_OBSERVED':
        return [dict(batch=batch, arm=arm, scope=scope, comparison=comparison, status='NOT_OBSERVED')]
    for family, value in analysis.get('families', {}).items():
        if not value['denominator']:
            continue
        rows = value.get('rows', [])
        item = dict(batch=batch, arm=arm, scope=scope, comparison=comparison, family=family,
                    status='OBSERVED', prompt_denominator=value['denominator'],
                    lost_ids=value.get('lost_ids', []), gained_ids=value.get('gained_ids', []),
                    strict_lost_ids=value.get('strict_lost_ids', []), strict_gained_ids=value.get('strict_gained_ids', []))
        item.update(lost=len(item['lost_ids']), gained=len(item['gained_ids']),
                    strict_lost=len(item['strict_lost_ids']), strict_gained=len(item['strict_gained_ids']))
        for metric in ('new_nll', 'true_nll', 'desired_nll', 'desired_margin'):
            values = [r[metric + '_delta'] for r in rows]
            item[metric + '_delta'] = sum(values) / len(values) if values else None
        item['token_lost'] = sum(len(r.get('token_lost_positions', [])) for r in rows)
        item['token_gained'] = sum(len(r.get('token_gained_positions', [])) for r in rows)
        for metric, ci in (value.get('cluster_bootstrap') or {}).items():
            if ci is not None:
                item[metric + '_paired_delta'] = ci['delta']
                item[metric + '_paired_CI95'] = ci['percentile95']
                item[metric + '_defined_resamples'] = ci['defined_resamples']
        output.append(item)
    joint = analysis.get('joint')
    if joint and joint['denominator']:
        values = joint.get('rows', [])
        item = dict(batch=batch, arm=arm, scope=scope, comparison=comparison,
                    family='R_plus_twoP_joint', status='OBSERVED', request_denominator=joint['denominator'],
                    lost_ids=[r['case_id'] for r in values if r['preference_lost']],
                    gained_ids=[r['case_id'] for r in values if r['preference_gained']],
                    strict_lost_ids=[r['case_id'] for r in values if r['strict_lost']],
                    strict_gained_ids=[r['case_id'] for r in values if r['strict_gained']])
        item.update(lost=len(item['lost_ids']), gained=len(item['gained_ids']),
                    strict_lost=len(item['strict_lost_ids']), strict_gained=len(item['strict_gained_ids']))
        for metric, ci in (joint.get('cluster_bootstrap') or {}).items():
            if ci is not None:
                item[metric + '_paired_delta'] = ci['delta']
                item[metric + '_paired_CI95'] = ci['percentile95']
                item[metric + '_defined_resamples'] = ci['defined_resamples']
        output.append(item)
    return output


def _terminal_bootstrap(evidence, plan, arm):
    output = []
    if evidence is None:
        return [dict(arm=arm, batch=100, status='NOT_OBSERVED', reason='terminal scalar evidence missing')]
    selected = evidence.get('selected')
    if selected is None:
        return [dict(arm=arm, batch=100, status='NOT_OBSERVED', reason='terminal selected metrics missing')]
    for scope in ('full_latest_valid', 'cumulative_neighborhood'):
        ids = plan['cohorts'].get(scope)
        if ids is None:
            output.append(dict(arm=arm, batch=100, scope=scope, status='NOT_OBSERVED'))
            continue
        families = plan.get('cohort_families', {}).get(scope, observers.FAMILIES)
        for source, comparison in [('atwrite', 'occurrence_atwrite_to_selected'), ('native', 'native_to_selected')]:
            before = evidence.get(source)
            analysis = observers._pair(before, selected, ids, families, bootstrap=True)
            output.extend(_paired_rows(analysis, 100, arm, scope, comparison))
    return output


def _scalars(value):
    """Only immediate primitive fields: never copy a spectrum/frontier array."""
    return {key: item for key, item in (value or {}).items()
            if item is None or type(item) in (str, int, float, bool)}


def _method_receipt(directory, batch, arm, provenance):
    cost = _member(directory, 'batch-cost', provenance)
    sealed = _member(directory, 'SELECTION_SEALED', provenance)
    spectrum = _member(directory, 'spectrum', provenance)
    history = _member(directory, 'history-selection', provenance)
    controller = _member(directory, 'controller', provenance)
    native = _member(directory, 'native-objective', provenance)
    teacher = _member(directory, 'teacher-bindings', provenance)
    commit = _member(directory, 'COMMIT', provenance)
    sources = [cost, sealed, spectrum, history, controller, native, teacher, commit]
    if all(value is None for value in sources):
        return None
    for value in (cost, sealed, commit):
        if value is not None and (value.get('batch') != batch or value.get('arm') != arm):
            raise ValueError('method arm/batch identity mismatch')
    result = dict(batch=batch, arm=arm, cost=cost,
                  selection_seal=None if sealed is None else dict(
                      **_scalars(sealed), objective=_scalars(sealed.get('objective'))),
                  commit=None if commit is None else _scalars(commit),
                  native_objective=None if native is None else {
                      name: native.get(name) for name in ('J', 'L_R', 'L_H', 'seconds', 'bank_identity')},
                  teacher_technical_failures='NOT_RECORDED_IN_SUCCESS_RECEIPT',
                  missing_receipts=[name for name, value in zip(
                      ('batch-cost','SELECTION_SEALED','spectrum','history-selection','controller',
                       'native-objective','teacher-bindings','COMMIT'), sources, strict=True) if value is None])
    if spectrum is not None:
        selected = spectrum.get('selected', {})
        result['geometry'] = dict(
            spectrum_scalars=_scalars(spectrum.get('spectrum')),
            decomposition_scalars=_scalars(spectrum.get('geometry')),
            selected_frontier_row=dict(**_scalars(selected), caps=selected.get('caps'),
                                       active_caps=selected.get('active_caps')),
            primary_epsilon=spectrum.get('primary_epsilon'), seconds=spectrum.get('seconds'),
            full_frontier='LOCAL_SOURCE_ONLY; not copied into compact report')
    if history is not None:
        weights = history.get('weights', [])
        pruning = history.get('pruning', {})
        result['history'] = dict(**_scalars(history),
            selected_count=len(history.get('selected_ids', [])),
            weights_min=min(weights) if weights else None, weights_max=max(weights) if weights else None,
            weights_sum=sum(weights),
            selected_identity=observers.digest(history.get('selected_ids', [])),
            pruning=dict(removed_count=len(pruning.get('removed', [])),
                         steps=len(pruning.get('trace', [])), **_scalars(pruning)),
            timings=_scalars(history.get('timings')),
            diagnostics={name: _scalars(value) for name, value in history.get('diagnostics', {}).items()})
    if controller is not None:
        result['controller'] = dict(**_scalars(controller), trials=[])
        for trial in controller.get('ledger', []):
            geometry = trial.get('geometry', {})
            result['controller']['trials'].append(dict(
                **_scalars(trial), objective=_scalars(trial.get('objective')),
                geometry=dict(**_scalars(geometry), request_response_quantiles=geometry.get('request_response_quantiles')),
                geometry_checks=trial.get('geometry_checks'), quadratic=trial.get('quadratic')))
        result['fallback'] = controller.get('alias') == 'native' and controller.get('status') != 'ACCEPTED'
        result['scientific_fallback_not_technical_failure'] = result['fallback']
    if teacher is not None:
        metadata = list(teacher.get('bindings', {}).values())
        observed = [item for item in metadata if type(item.get('at_write_TF_strict')) is bool]
        targets = sum(item.get('at_write_TF_valid_tokens', 0) for item in observed)
        correct = sum(sum(item.get('at_write_TF_token_correct', [])) for item in observed)
        result['teacher'] = dict(capture_receipt=_scalars(teacher.get('receipt')),
            reported_bindings=len(metadata), atwrite_TF_strict_observed=len(observed),
            atwrite_TF_strict_failures=sum(not item['at_write_TF_strict'] for item in observed),
            atwrite_TF_strict_failure_rate=(sum(not item['at_write_TF_strict'] for item in observed)/len(observed)
                                           if observed else None),
            atwrite_TF_correct_tokens=correct, atwrite_TF_valid_tokens=targets,
            atwrite_TF_token_micro=correct/targets if targets else None,
            omitted_TF_metadata=len(metadata)-len(observed),
            definition='supplied-target TF failure is an observed edit outcome, not teacher I/O/finite failure')
    return result


def _method_totals(methods):
    """Sum disjoint phase durations; retain the last cumulative counter once."""
    phases = ('seconds','native','geometry','gradient','controller','commit_teacher','observer')
    with_cost = [item for item in methods if item.get('cost') is not None]
    last = with_cost[-1]['cost'] if with_cost else None
    return dict(observed_method_batches=[item['batch'] for item in methods],
                cost_receipt_batches=[item['batch'] for item in with_cost],
                phase_seconds={name: sum(item['cost'].get(name, 0.) for item in with_cost) for name in phases},
                latest_cumulative_counts=last.get('cumulative') if last else None,
                latest_history_cumulative_counts=last.get('history_counts') if last else None,
                latest_reference_cumulative_counts=last.get('reference_counts') if last else None,
                cumulative_counts_policy='last recorded snapshot only, never sum cumulative counters',
                history_selection_sweeps=sum(bool(item.get('history', {}).get('selection_exercised')) for item in methods),
                first_history_overflow=next((item['batch'] for item in methods
                    if item.get('history', {}).get('pool_size', 0)>512), None),
                NLL_fact_VJPs=sum(item.get('history', {}).get('NLL_VJPs', 0) for item in methods),
                KL_fact_VJPs=sum(item.get('history', {}).get('KL_VJPs', 0) for item in methods),
                fallback_batches=[item['batch'] for item in methods if item.get('fallback')],
                cuda_peak_bytes=max((item['cost'].get('cuda_peak_bytes', 0) for item in with_cost), default=None),
                host_maxrss_KiB=max((item['cost'].get('host_maxrss_KiB', 0) for item in with_cost), default=None),
                allocation_gpu_hours='NOT_OBSERVED; phase wall seconds are not scheduler allocation',
                teacher_technical_failure_rate='NOT_RECORDED_IN_SUCCESS_RECEIPT',
                sketch_precision_status='NOT_ESTABLISHED; see actual one-time diagnostic fields')


def build(output_directory, arm, *, bootstrap=True):
    """Consume B001..B100/observer(.json|.json.gz), no directory result polling.

    For terminal bootstrap B100/observer-evidence(.json|.json.gz) contains
    selected/native/entry/baseline/atwrite parent-format scalar observations.
    """
    if arm not in observers.ARMS:
        raise ValueError('unapproved arm')
    output = Path(output_directory)
    provenance, endpoints, pairs, strata, costs, methods = {}, [], [], [], [], []
    observed, missing, final = [], [], None
    for batch in range(1, 101):
        method = _method_receipt(output / f'B{batch:03d}', batch, arm, provenance)
        if method is not None:
            methods.append(method)
        record = _member(output / f'B{batch:03d}', 'observer', provenance)
        if record is None:
            missing.append(batch)
            continue
        if record.get('arm') != arm or record.get('batch') != batch or record.get('selection_influence') != 0:
            raise ValueError('observer arm/batch/selection boundary mismatch')
        observed.append(batch)
        for scope, summary in record['cohorts'].items():
            endpoints.extend(_endpoint_rows(summary, batch, arm, scope))
        for scope, comparisons in record['comparisons'].items():
            for comparison, analysis in comparisons.items():
                if isinstance(analysis, dict):
                    pairs.extend(_paired_rows(analysis, batch, arm, scope, comparison))
        for group, summary in record['observed_groups'].items():
            strata.extend(_endpoint_rows(summary, batch, arm, group))
        for group, analysis in record.get('group_forgetting', {}).items():
            pairs.extend(_paired_rows(analysis, batch, arm, 'stratum/' + group, 'occurrence_atwrite_to_selected'))
        if record.get('observer_cost') is not None:
            costs.append(dict(batch=batch, arm=arm, category='observer', values=record['observer_cost']))
        if batch == 100:
            final = record
    bootstrap_rows = []
    if bootstrap and final is not None:
        evidence = _member(output / 'B100', 'observer-evidence', provenance)
        bootstrap_rows = _terminal_bootstrap(evidence, final['plan'], arm)
    lock = _member(output.parent, 'execution.lock', provenance)
    terminal = _member(output, 'TERMINAL', provenance)
    # A program terminal marker is different from scheduler accounting.
    allocation = _member(output, 'accounting', provenance)
    return dict(schema='en-adapt-gss-history-twoarm-report-v1', arm=arm,
                status='B100_OBSERVATIONS_PRESENT' if observed == list(range(1, 101)) else 'INCOMPLETE_OBSERVER_EVIDENCE',
                observed_batches=observed, missing_batches=missing,
                actual_scheduler_terminal='NOT_OBSERVED' if allocation is None else 'EXPLICIT_ACCOUNTING_AVAILABLE',
                actual_initial='NOT_OBSERVED_BY_AGENT', agent_monitoring='PAUSED_AWAITING_USER',
                terminal_marker=terminal, execution_lock=lock, allocation=allocation,
                endpoints=endpoints, paired=pairs, strata=strata, observer_costs=costs, methods=methods,
                method_totals=_method_totals(methods),
                terminal_cluster_bootstrap=bootstrap_rows,
                bootstrap=dict(performed=bool(bootstrap_rows), request_repeats=10000, seed=observers.SEED,
                               scopes=['full_latest_valid', 'cumulative_neighborhood'],
                               unit='request cluster; neighbors not independent'),
                independent_chain_cost=dict(standalone_native_batches=100, total_two_arm_native_batches=200,
                                            shared_prefix_batches=0, scope='planned counts, not measured cost'),
                limitations=[
                    'TF token/macro/strict accuracy is not free generation accuracy.',
                    'Finite identity evidence does not upgrade precision NOT_ESTABLISHED.',
                    'GSS sketch fidelity is measured separately, not inferred from CPU design tests.',
                    'No superiority over N4 or R-only lifelong is established by these two arms.',
                    'RES versus GSS_REC jointly changes selection and recency; their separate effects are not identified.',
                    'No edited W/M/delta/resume bundle; exact crash-resume NOT_AVAILABLE.',
                    'Program report is not scheduler terminal observation; no post-release agent polling.'],
                inputs=provenance)


def compare_attempts(res_directory, rec_directory, *, bootstrap=True):
    """Explicit CPU review of both terminal input sets; never automatically run."""
    provenance = {}
    inventories = []
    for directory, arm in zip((res_directory, rec_directory), observers.ARMS, strict=True):
        receipt = _member(Path(directory) / 'B100', 'observer', provenance)
        evidence = _member(Path(directory) / 'B100', 'observer-evidence', provenance)
        if receipt is None or evidence is None or receipt['arm'] != arm:
            raise ValueError('both independent B100 observation inventories are required')
        inventories.append((receipt, evidence))
    left, right = inventories
    if left[0]['plan']['seen_case_ids'] != right[0]['plan']['seen_case_ids']:
        raise ValueError('two-arm fixed stream mismatch')
    rows = []
    for scope in ('current', 'first100', 'history_panel', 'full_latest_valid', 'cumulative_neighborhood'):
        ids = left[0]['plan']['cohorts'][scope]
        if ids != right[0]['plan']['cohorts'][scope]:
            raise ValueError(f'arm-independent cohort mismatch: {scope}')
        families = left[0]['plan']['cohort_families'].get(scope, observers.FAMILIES)
        # The final two principal full cohorts receive the presealed CI.
        ci = bootstrap and scope in ('full_latest_valid', 'cumulative_neighborhood')
        analysis = observers._pair(left[1]['selected'], right[1]['selected'], ids, families, bootstrap=ci)
        rows.extend(_paired_rows(analysis, 100, 'RES_vs_GSS_REC', scope, 'GSS_REC_minus_RES'))
    return dict(comparison='independent cold chains; GSS_REC minus RES', rows=rows,
                no_shared_W_M_teacher=True, inputs=provenance,
                limitation='RES versus GSS_REC jointly changes selection and recency weighting; no isolated causal attribution')


def _publish_text(path, text):
    """Create-once text artifact, same-filesystem atomic publication."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _csv_text(rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, ensure_ascii=False, separators=(',', ':'), allow_nan=False, default=scalar)
                         if isinstance(v, (list, dict, tuple)) else v for k, v in row.items()})
    return out.getvalue()


def write(report, destination):
    """Publish raw-free Korean Markdown, CSV and exact source-manifest once."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, key in [('endpoints', 'endpoints'), ('paired', 'paired'), ('strata', 'strata'),
                      ('observer-costs', 'observer_costs'), ('method-receipts', 'methods'),
                      ('terminal-bootstrap', 'terminal_cluster_bootstrap')]:
        _publish_text(destination / (name + '.csv'), _csv_text(report[key]))
    compact = {k: v for k, v in report.items() if k not in ('endpoints', 'paired', 'strata', 'terminal_cluster_bootstrap')}
    save(destination / 'report-summary.json', compact)
    final_rows = [r for r in report['endpoints'] if r['batch'] == 100 and r['scope'] in ('full_latest_valid', 'cumulative_neighborhood')]
    lines = [f"# {report['arm']} fixed10k 관측 보고", '',
             f"프로그램 관측 상태: `{report['status']}`. 관측 batch {len(report['observed_batches'])}/100.",
             f"누락 batch: {report['missing_batches'] or '없음'}. Scheduler terminal: `{report['actual_scheduler_terminal']}`.", '',
             '각 arm은 독립 cold W0/zero M4에서 B100×100을 진행한다. 두 arm 합산 계획은 native/reference 200 batch이며, cross-job prefix·W/M·at-write teacher를 공유하지 않는다.', '',
             '|최종 cohort|family|NLL preference|TF token micro|TF prompt macro|TF strict|desired NLL|',
             '|---|---|---:|---:|---:|---:|---:|']
    def fmt(value):
        return '미관측' if value is None else f'{value:.6g}'
    for row in final_rows:
        lines.append('|' + '|'.join([row['scope'], row['family']] + [fmt(row.get(k)) for k in ('preference_rate','tf_token_micro','tf_prompt_macro','tf_strict_rate','desired_nll')]) + '|')
    lines += ['', '이 두 arm 비교는 RES uniform에서 GSS bounded-recency로 selection과 loss weight를 함께 바꾼다. 따라서 차이를 GSS 선택 또는 recency 하나의 독립 효과로 해석할 수 없다.', '', 'RS/PS는 new NLL < true NLL, NS는 true NLL < new NLL이며 tie는 실패다. TF는 supplied target의 teacher-forced 정확도이며 자유생성 평가가 아니다. Token micro, prompt macro, 전체-target strict를 분리했다.', '',
              'Full latest-valid R/PS는 active version당 최신 occurrence, 누적 NS는 overwrite·실패를 포함한 모든 occurrence를 사용한다. First100은 최초 occurrence를 고정한다. Panel은 current fact를 제외한 active past version의 seed 20260920 bottom-hash128이며 bank 선택과 독립이다.', '',
              'paired.csv에는 entry→native 손상, native→selected 보정, occurrence at-write→현재 lost/gained 및 strict ID를 남긴다. strata.csv는 실제 관측 union의 bank 안/밖, age, relation, active/superseded, at-write 성공별 분모다. 미관측 strata를 전체 history로 외삽하지 않는다.', '',
              '최종 full latest-valid / 누적 neighborhood의 request-cluster bootstrap은 10,000회, seed 20260920이며 이웃 prompt를 독립 표본으로 세지 않는다. 같은 평균은 같은 lost/gained ID를 뜻하지 않는다.', '',
              '비용은 observer와 method receipt를 분리한다. Scheduler allocation 증거가 없으면 GPU-hour를 추정값으로 대체하지 않는다. 수치 precision은 NOT_ESTABLISHED를 유지하며 GSS sketch 품질은 별도 실제 진단 증거가 필요하다.', '',
              'save_checkpoints=false. Edited W/M/delta/resume disk 저장이 없으므로 exact crash-resume은 NOT_AVAILABLE이다. 이 보고 생성은 제출 후 agent 모니터링 또는 scheduler terminal 확인을 뜻하지 않는다.', '']
    _publish_text(destination / 'report-ko.md', '\n'.join(lines))
    artifacts = {}
    for path in sorted(destination.iterdir()):
        if path.is_file():
            raw = path.read_bytes()
            artifacts[path.name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    save(destination / 'package-manifest.json', dict(schema='raw-free-local-report-package-v1', files=artifacts))
    return artifacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=observers.ARMS)
    parser.add_argument('--skip-bootstrap', action='store_true', help='mark terminal CI uncomputed, not a PASS')
    args = parser.parse_args()
    write(build(args.output, args.arm, bootstrap=not args.skip_bootstrap), args.destination)


if __name__ == '__main__':
    main()
