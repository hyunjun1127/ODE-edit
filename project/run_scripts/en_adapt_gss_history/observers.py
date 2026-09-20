"""Observer-only planning and CPU reduction for the two independent 10k chains.

The readiness-sealed evaluator remains the only model forward implementation.
This module never chooses a candidate, mutates model state, or reads a scheduler.
A case is an occurrence; a version is a semantic ledger entry. Their denominators
are kept separate, including superseded occurrences in cumulative neighborhood.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from project.run_scripts.en_adaptive_nullspace import metrics as parent_metrics

ARMS = ('EN_ADAPT_H_RES', 'EN_ADAPT_H_GSS_REC')
FULL_BATCHES = (2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
PANEL_SIZE = 128
SEED = 20260920
FAMILIES = ('RS', 'PS', 'NS')
CONTRACT = Path(__file__).with_name('metrics-contract.json')
def evaluate(model, tok, records):
    """Exact parent neural stream and microbatch layout; linear CPU reduction.

    No record chunking: each complete canonical family is evaluated in the same
    microbatch16 stream as the parent. Only its quadratic summary is replaced.
    """
    records = list(records)
    if not records or len({r['case_id'] for r in records}) != len(records):
        raise ValueError('nonempty unique record inventory required')
    if tok.padding_side != 'right':
        raise ValueError('canonical tokenizer padding_side must remain right')
    binder = parent_metrics._binding()
    modules = binder._binding()
    pairs = modules['evaluator'].counterfact_pairs(records)
    pairs['locality_target_new'] = modules['locality'].counterfact_locality_target_new_pairs(records)
    raw = {kind: modules['evaluator'].evaluate_pairs(model, tok, values,
            device=binder._device(model), microbatch_size=16) for kind, values in pairs.items()}
    result = summarize(parent_metrics._compact(raw), [int(r['case_id']) for r in records])
    result.update(metrics_contract_sha256=hashlib.sha256(parent_metrics.CONTRACT.read_bytes()).hexdigest(),
                  observer_metrics_contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
                  source_binding=modules['receipt'], evaluator_controller_influence=0,
                  evaluator_layout='HISTORICAL_MICROBATCH16_MANUAL_LEFT_PADDING_NO_POSITION_OVERRIDE',
                  transaction_guard='CALLER_OWNED', forwards_for_additional_accuracy=0,
                  reduction='linear equivalent of parent summarize; neural family streams unchanged')
    return result


def digest(value):
    return parent_metrics.digest(value)


def _priority(identity):
    return (hashlib.sha256(f'{SEED}|{identity}'.encode()).hexdigest(), str(identity))


def _ratio(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'rate': numerator / denominator if denominator else None}


def _mean(values):
    return sum(values) / len(values) if values else None


def summarize(rows, request_ids):
    """Linear-time equivalent of the parent's scalar and joint definitions.

    The parent reducer scans all rows for every joint request. Avoid that CPU
    quadratic at 10k while leaving all neural evaluation bytes untouched.
    """
    rows, request_ids = list(rows), [int(x) for x in request_ids]
    wanted = set(request_ids)
    if len(wanted) != len(request_ids):
        raise ValueError('duplicate request identity')
    keys = [(r['family'], r['case_id'], r['prompt_index']) for r in rows]
    if len(set(keys)) != len(keys) or any(r['case_id'] not in wanted for r in rows):
        raise ValueError('invalid prompt inventory')
    groups = defaultdict(list)
    per_case = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['family'] not in FAMILIES:
            raise ValueError('unknown evaluator family')
        parent_metrics._validate_row(row)
        groups[row['family']].append(row)
        per_case[row['case_id']][row['family']].append(row)
    aggregates, metrics = {}, {}
    for tag in FAMILIES:
        group = groups[tag]
        flags = [r['desired_token_correct'] for r in group]
        metrics[tag] = dict(**_ratio(sum(r['success'] for r in group), len(group)), rows=group)
        aggregates[tag] = dict(
            nl_accuracy_label={'RS':'nl rewrite acc', 'PS':'nl rephrase acc', 'NS':'nl neighborhood acc'}[tag],
            tf_token_micro=_ratio(sum(sum(f) for f in flags), sum(map(len, flags))),
            tf_prompt_macro=dict(value=_mean([sum(f) / len(f) for f in flags]), denominator=len(group)),
            tf_strict=_ratio(sum(r['desired_strict'] for r in group), len(group)),
            nll={side + '_prompt_macro': _mean([r[side + '_nll'] for r in group]) for side in ('new', 'true', 'desired')},
            desired_margin_prompt_macro=_mean([r['desired_margin'] for r in group]))
    joint = []
    for case in request_ids:
        rewrite, paras = per_case[case]['RS'], per_case[case]['PS']
        if len(rewrite) == 1 and len(paras) == 2:
            joint.append(dict(case_id=case, strict=all(r['desired_strict'] for r in rewrite + paras),
                              preference=all(r['success'] for r in rewrite + paras)))
    eligible = {r['case_id'] for r in joint}
    return dict(request_ids=request_ids, request_order=digest(request_ids), rows=rows, metrics=metrics,
                aggregates=aggregates, joint=dict(
                    definition='R plus exactly two canonical P, all desired target tokens correct',
                    eligible_requests=len(joint), omitted_case_ids=[i for i in request_ids if i not in eligible],
                    tf_strict=_ratio(sum(r['strict'] for r in joint), len(joint)),
                    preference=_ratio(sum(r['preference'] for r in joint), len(joint)), rows=joint))


def subset(result, case_ids, *, families=FAMILIES):
    wanted = set(int(x) for x in case_ids)
    if not wanted.issubset(result['request_ids']):
        raise ValueError('requested subset contains unobserved cases')
    return summarize([r for r in result['rows'] if r['case_id'] in wanted and r['family'] in families],
                     [i for i in result['request_ids'] if i in wanted])


def merge_results(results, *, request_ids=None):
    """Merge disjoint case observations, rejecting silently replaced evidence."""
    by_case, seen_order = {}, []
    for result in results:
        grouped = defaultdict(list)
        for row in result['rows']:
            grouped[row['case_id']].append(row)
        for case in result['request_ids']:
            if case in by_case:
                raise ValueError('duplicate case in evidence merge')
            by_case[case] = grouped[case]
            seen_order.append(case)
    order = seen_order if request_ids is None else list(request_ids)
    if set(order) != set(by_case) or len(order) != len(by_case):
        raise ValueError('evidence merge request inventory mismatch')
    return summarize([r for tag in FAMILIES for case in order for r in by_case[case] if r['family'] == tag], order)


def compact_summary(result):
    """Keep denominators and identity roots; omit local per-prompt scalar rows."""
    return dict(requests=len(result['request_ids']), request_order=result['request_order'],
                metrics={k: {n: v for n, v in item.items() if n != 'rows'} for k, item in result['metrics'].items()},
                aggregates=result['aggregates'], joint={k: v for k, v in result['joint'].items() if k != 'rows'})


def _snapshot(ledger):
    value = ledger.snapshot() if hasattr(ledger, 'snapshot') else ledger
    if not isinstance(value, dict):
        raise TypeError('ledger must expose snapshot metadata')
    raw = value.get('versions', {})
    versions = list(raw.values()) if isinstance(raw, dict) else list(raw)
    active = set(value.get('active_version_ids', [v['version_id'] for v in versions if v.get('active')]))
    rows = []
    for source in versions:
        row = dict(source)
        if 'representative_case_id' not in row and row.get('record'):
            row['representative_case_id'] = int(row['record']['case_id'])
        if 'latest_case_id' not in row:
            record = row.get('latest_record', row.get('record'))
            row['latest_case_id'] = int(record['case_id']) if record else row.get('representative_case_id')
        row['active'] = row['version_id'] in active
        if row.get('latest_case_id') is None or row.get('representative_case_id') is None:
            raise ValueError('version missing original/latest case identity')
        rows.append(row)
    if len({r['version_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate version identity')
    return value, rows


def _fact(record):
    request = record['requested_rewrite']
    if 'relation_id' not in request or 'subject' not in request:
        raise ValueError('canonical fact identity is unavailable')
    return (request['subject'], request['relation_id'])


def plan_records(all_records, ledger, batch, *, batch_size=100, panel_size=PANEL_SIZE):
    """Plan after logical commit, before observing the selected endpoint.

    Panel eligibility is active *past* versions, excluding current facts. It is
    independent of bank policy/weights/quality and uses one latest occurrence
    per version. Full history likewise uses latest occurrences; every occurrence
    remains in the cumulative-N denominator and separately labelled strata.
    """
    if not (1 <= batch <= 100) or batch_size != 100 or panel_size != PANEL_SIZE:
        raise ValueError('observer schedule/panel differs from sealed contract')
    records = list(all_records[:batch * batch_size])
    if len(records) != batch * batch_size:
        raise ValueError('fixed-order prefix is incomplete')
    order = [int(r['case_id']) for r in records]
    if len(set(order)) != len(order):
        raise ValueError('duplicate CounterFact case identity')
    index = {case: i for i, case in enumerate(order)}
    current = records[-batch_size:]
    current_ids = [int(r['case_id']) for r in current]
    current_facts = {_fact(r) for r in current}
    snapshot, versions = _snapshot(ledger)
    active = [v for v in versions if v['active']]
    for version in versions:
        case_ids = list(version.get('occurrence_case_ids', [version['representative_case_id']]))
        if any(int(c) not in index for c in case_ids) or version['latest_case_id'] not in index:
            raise ValueError('ledger references a future/missing occurrence')
    def version_fact(v):
        return tuple(v.get('fact', _fact(records[index[v['latest_case_id']]])))
    eligible = [v for v in active if version_fact(v) not in current_facts]
    panel_versions = sorted(eligible, key=lambda v: _priority(v['version_id']))[:panel_size]
    panel_ids = [v['latest_case_id'] for v in panel_versions]
    full = batch in FULL_BATCHES
    latest_ids = [v['latest_case_id'] for v in active]
    active_occurrences = {int(c) for v in active for c in v.get('occurrence_case_ids', [v['representative_case_id']])}
    cohorts = {'current': current_ids, 'first100': order[:100], 'history_panel': panel_ids}
    if full:
        cohorts.update(full_latest_valid=latest_ids, cumulative_neighborhood=order)
    cohort_sets = {name: set(cases) for name, cases in cohorts.items()}
    union = {case for cases in cohort_sets.values() for case in cases}
    observed_ids = [case for case in order if case in union]
    metadata = {}
    for version in versions:
        for case in version.get('occurrence_case_ids', [version['representative_case_id']]):
            case = int(case)
            metadata[case] = dict(version_id=version['version_id'], fact=list(version_fact(version)),
                                  relation_id=version_fact(version)[1], created_batch=int(version['created_batch']),
                                  # Post-write age (current versions are age zero), separate from loss age.
                                  age_batches=batch - int(version['created_batch']),
                                  active=version['active'], latest_occurrence=case == version['latest_case_id'],
                                  version_original_case_id=version['representative_case_id'])
    if set(metadata) != set(order):
        raise ValueError('semantic occurrence inventory does not cover fixed prefix')
    return dict(batch=batch, records=[records[index[c]] for c in observed_ids], request_ids=observed_ids,
                request_order=digest(observed_ids), seen_case_ids=order, current_case_ids=current_ids,
                cohorts={k: [case for case in order if case in wanted] for k, wanted in cohort_sets.items()},
                cohort_families={'full_latest_valid': ['RS', 'PS'], 'cumulative_neighborhood': ['NS']},
                panel=dict(size=panel_size, seed=SEED, priority='sha256(seed|fact_version_id)',
                           eligible_versions=len(eligible), selected_version_ids=[v['version_id'] for v in panel_versions],
                           selection_uses_bank=False, selection_uses_metrics=False,
                           eligibility='latest active past versions excluding current facts'),
                full_history_observation=full, active_occurrence_case_ids=[c for c in order if c in active_occurrences],
                superseded_occurrence_case_ids=[c for c in order if c not in active_occurrences],
                case_metadata=metadata, ledger_identity=digest({
                    'versions': [(v['version_id'], v['latest_case_id'], v['active']) for v in versions],
                    'occurrences': [(c, metadata[c]['version_id']) for c in order]}),
                bank_after_commit=list(snapshot.get('bank_ids', [])),
                pending_after_commit=list(snapshot.get('pending_ids', [])))


def compact_plan(plan):
    return {key: value for key, value in plan.items() if key not in ('records', 'case_metadata')}


def _aligned(result, ids):
    value = subset(result, ids)
    order = list(ids)
    if value['request_ids'] == order:
        return value
    grouped = defaultdict(list)
    for row in value['rows']:
        grouped[row['case_id']].append(row)
    return summarize([r for tag in FAMILIES for case in order for r in grouped[case] if r['family'] == tag], order)


def _pair(before, after, ids, families=FAMILIES, bootstrap=False):
    if before is None:
        return {'status': 'NOT_OBSERVED', 'reason': 'matching before endpoint was not observed'}
    a, b = _aligned(before, ids), _aligned(after, ids)
    if tuple(families) != FAMILIES:
        a, b = subset(a, ids, families=families), subset(b, ids, families=families)
    value = parent_metrics.paired(a, b, bootstrap=bootstrap)
    value['status'] = 'OBSERVED'
    return value


def _age_bin(age):
    if age == 0:
        return '0_current'
    if age <= 5:
        return '1_5'
    if age <= 10:
        return '6_10'
    if age <= 20:
        return '11_20'
    if age <= 50:
        return '21_50'
    return '51_plus'


def _group_ids(plan, selected, atwrite, bank_ids, bank_after):
    groups = defaultdict(list)
    atwrite_success = {}
    if atwrite is not None:
        for row in atwrite['rows']:
            if row['family'] == 'RS':
                atwrite_success[row['case_id']] = row['success']
    for case in selected['request_ids']:
        m = plan['case_metadata'][case]
        groups['active' if m['active'] else 'superseded'].append(case)
        groups['latest_occurrence' if m['latest_occurrence'] and m['active'] else 'other_occurrence'].append(case)
        groups['objective_bank_in' if m['version_id'] in bank_ids else 'objective_bank_out'].append(case)
        groups['postcommit_bank_in' if m['version_id'] in bank_after else 'postcommit_bank_out'].append(case)
        groups['age/' + _age_bin(m['age_batches'])].append(case)
        groups['relation/' + str(m['relation_id'])].append(case)
        if case in atwrite_success:
            groups['atwrite_RS_success' if atwrite_success[case] else 'atwrite_RS_failure'].append(case)
        else:
            groups['atwrite_NOT_OBSERVED'].append(case)
    return dict(groups)


def build_receipt(plan, *, baseline, native, selected, atwrite=None, entry=None,
                  bank_before_current=(), bank_after_commit=None, arm=None, cost=None):
    """Build scalar observations after selection is irrevocably sealed.

    Raw-free here means no prompt/logit/tensor. Per-prompt identity/scalars remain
    local for paired analysis; compact_summary and report omit these raw rows.
    """
    if arm is not None and arm not in ARMS:
        raise ValueError('unapproved arm')
    ids = plan['request_ids']
    selected = _aligned(selected, ids)
    for result in (baseline, native, entry):
        if result is not None:
            _aligned(result, ids)  # Fail on incomplete claimed endpoint evidence.
    bank_before = set(bank_before_current)
    bank_after = set(plan['bank_after_commit'] if bank_after_commit is None else bank_after_commit)
    cohorts = {}
    comparisons = {}
    for name, case_ids in plan['cohorts'].items():
        families = plan['cohort_families'].get(name, FAMILIES)
        current = subset(selected, case_ids, families=families)
        cohorts[name] = compact_summary(current)
        comparisons[name] = {
            'native_damage_entry_to_WN': _pair(entry, native, case_ids, families) if native is not None else {'status':'NOT_OBSERVED'},
            'correction_WN_to_selected': _pair(native, selected, case_ids, families),
            'W0_to_selected': _pair(baseline, selected, case_ids, families)}
        if atwrite is not None:
            covered = [c for c in case_ids if c in set(atwrite['request_ids'])]
            comparisons[name]['occurrence_atwrite_to_selected'] = _pair(atwrite, selected, covered, families)
            comparisons[name]['occurrence_atwrite_unobserved_ids'] = [c for c in case_ids if c not in set(covered)]
    group_ids = _group_ids(plan, selected, atwrite, bank_before, bank_after)
    groups = {name: compact_summary(subset(selected, cases)) for name, cases in group_ids.items()}
    group_forgetting = {}
    if atwrite is not None:
        covered = set(atwrite['request_ids'])
        for name, cases in group_ids.items():
            ids_available = [c for c in cases if c in covered]
            group_forgetting[name] = _pair(atwrite, selected, ids_available)
    retention = None
    if baseline is not None and native is not None:
        retention = parent_metrics.retention(_aligned(baseline, ids), _aligned(native, ids), selected)
    return dict(schema='en-adapt-gss-history-observer-v1', arm=arm, batch=plan['batch'],
                metrics_contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
                plan=compact_plan(plan), cohorts=cohorts, comparisons=comparisons,
                observed_groups=groups, group_case_ids=group_ids, group_forgetting=group_forgetting,
                w0_correct_neighborhood_retention=retention,
                bank_bindings=dict(objective_selected_before_current=sorted(bank_before),
                                   bank_after_commit=sorted(bank_after),
                                   pending_after_commit=plan['pending_after_commit']),
                observer_cost=cost, selection_influence=0,
                native_damage_definition='entry endpoint to WN on identical postselection-only observer inputs',
                correction_damage_definition='WN to selected on the same observer inputs',
                atwrite_definition='this occurrence at its own final write; no version-teacher replacement',
                group_coverage='observed union only; superseded occurrences complete only at full observation batches',
                bootstrap=dict(performed=False, deferred='terminal CPU reducer', repeats=10000, seed=SEED),
                precision_status='NOT_ESTABLISHED', free_generation_accuracy=False,
                no_checkpoint=True)


class ObserverState:
    """RAM cache of scalar W0 and occurrence-at-write metrics, never weights."""
    def __init__(self, arm):
        if arm not in ARMS:
            raise ValueError('unapproved arm')
        self.arm, self.baseline_cases, self.atwrite_cases = arm, {}, {}
        self.committed_batches = set()

    @staticmethod
    def _put(store, result, case_ids):
        result = subset(result, case_ids)
        by_case = defaultdict(list)
        for row in result['rows']:
            by_case[row['case_id']].append(row)
        for case in result['request_ids']:
            if case in store:
                if digest(store[case]) != digest(by_case[case]):
                    raise ValueError('immutable scalar observation mismatch')
            else:
                store[case] = by_case[case]

    @staticmethod
    def _get(store, ids, *, require_all=True):
        ids = list(ids)
        missing = [c for c in ids if c not in store]
        if missing and require_all:
            raise ValueError(f'missing scalar observation cases: {missing[:8]}')
        available = [c for c in ids if c in store]
        return summarize([r for tag in FAMILIES for case in available for r in store[case] if r['family'] == tag], available)

    def plan(self, all_records, ledger, batch):
        plan = plan_records(all_records, ledger, batch)
        plan['missing_W0_case_ids'] = [c for c in plan['request_ids'] if c not in self.baseline_cases]
        return plan

    def remember_baseline(self, result):
        self._put(self.baseline_cases, result, result['request_ids'])

    def record(self, batch, plan, *, native, selected, entry=None, baseline=None,
               bank_before_current=(), bank_after_commit=None, cost=None):
        if batch != plan['batch'] or batch in self.committed_batches:
            raise ValueError('observer commit batch mismatch/duplicate')
        if baseline is not None:
            self.remember_baseline(baseline)
        base = self._get(self.baseline_cases, plan['request_ids'])
        # Current occurrence metrics are valid at-write even for failed edits.
        self._put(self.atwrite_cases, selected, plan['current_case_ids'])
        atwrite = self._get(self.atwrite_cases, plan['request_ids'], require_all=False)
        receipt = build_receipt(plan, baseline=base, native=native, selected=selected,
                                entry=entry, atwrite=atwrite, arm=self.arm,
                                bank_before_current=bank_before_current, bank_after_commit=bank_after_commit, cost=cost)
        self.committed_batches.add(batch)
        return receipt

    def evidence(self, plan):
        return dict(baseline=self._get(self.baseline_cases, plan['request_ids']),
                    atwrite=self._get(self.atwrite_cases, plan['request_ids'], require_all=False))
