"""Pinned BLUE observations plus TF accuracy; never used for candidate selection.

Model forwards are delegated unchanged to the readiness-sealed evaluator. Only
compact identities, per-target correctness bits and scalar observations survive.
All uncertainty resamples requests, keeping their complete prompt/token cluster.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

RUNTIME = Path('/data/janghj/ODE-edit/local/runtime/server3-experiment-ready-v1')
CONTRACT = Path(__file__).with_name('metrics-contract.json')
FAMILIES = (('rewrite', 'RS'), ('rephrase', 'PS'), ('locality', 'NS'))
_BINDER = None


def source_path(name):
    """Same exact evaluator bytes, relocated without modifying their imports."""
    if os.environ.get('EN_ADAPT_EVALUATOR_MAP'):
        mapping = json.loads(Path(os.environ['EN_ADAPT_EVALUATOR_MAP']).read_text())
        return Path(mapping[name])
    return RUNTIME / name


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def contract_receipt():
    """Validate every pinned evaluator file before the first observation."""
    contract = json.loads(CONTRACT.read_text())
    for name, expected in contract['source_files'].items():
        data = source_path(name).read_bytes()
        if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
            raise RuntimeError(f'evaluator source identity mismatch: {name}')
    return dict(sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(), contract=contract)


def _binding():
    global _BINDER
    if _BINDER is None:
        contract_receipt()
        name = '_en_adaptive_readiness_evaluation'
        spec = importlib.util.spec_from_file_location(name, source_path('contexts/evaluation.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        helper = source_path('evaluator_source/project/run_scripts/blue_alphaedit_sequential_comparison/evaluation.py').parents[1]
        module.bind_evaluation_sources(helper / 'blue_alphaedit_sequential_comparison', helper_root=helper)
        _BINDER = module
    return _BINDER


def _ratio(n, d):
    return dict(numerator=n, denominator=d, rate=n / d if d else None)


def _mean(values):
    return sum(values) / len(values) if values else None


def _validate_row(row):
    for side in ('new', 'true'):
        flags = row[side + '_token_correct']
        if not flags or any(type(x) is not bool for x in flags):
            raise ValueError('missing/nonboolean target correctness flags')
        if row[side + '_token_count'] != len(flags) or row[side + '_strict'] != all(flags):
            raise ValueError('target correctness denominator/strict mismatch')
        if not math.isfinite(row[side + '_nll']):
            raise ValueError('nonfinite evaluator NLL')


def _compact(raw):
    rows = []
    for family, tag in FAMILIES:
        for new, true in zip(raw[family + '_target_new'], raw[family + '_target_true'], strict=True):
            if (new['case_id'], new['prompt_index'], new['prompt']) != (true['case_id'], true['prompt_index'], true['prompt']):
                raise ValueError('new/true evaluator identity mismatch')
            row = dict(family=tag, case_id=int(new['case_id']), prompt_index=int(new['prompt_index']),
                       identity=digest([new['case_id'], new['prompt_index'], new['prompt'], new['target'], true['target']]),
                       token_identity=digest([new['target_token_ids'], true['target_token_ids']]))
            for side, source in [('new', new), ('true', true)]:
                row.update({side + '_nll': source['nll'], side + '_token_correct': source['token_correct'],
                            side + '_token_count': len(source['token_correct']), side + '_strict': source['all_tokens_correct']})
            desired = 'true' if tag == 'NS' else 'new'
            row.update(desired_side=desired, desired_nll=row[desired + '_nll'],
                       desired_token_correct=row[desired + '_token_correct'], desired_token_count=row[desired + '_token_count'],
                       desired_strict=row[desired + '_strict'],
                       new_favoring_margin=row['true_nll'] - row['new_nll'])
            row['desired_margin'] = (-1 if tag == 'NS' else 1) * row['new_favoring_margin']
            row['success'] = row['desired_margin'] > 0
            _validate_row(row)
            rows.append(row)
    return rows


def summarize(rows, request_ids):
    """Pure reduction. Empty families retain zero denominators and null rates."""
    rows = list(rows)
    request_ids = list(request_ids)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError('duplicate request identity')
    keys = [(r['family'], r['case_id'], r['prompt_index']) for r in rows]
    if len(set(keys)) != len(keys) or any(r['case_id'] not in request_ids for r in rows):
        raise ValueError('invalid prompt inventory')
    metrics, aggregates = {}, {}
    for _, tag in FAMILIES:
        group = [r for r in rows if r['family'] == tag]
        metrics[tag] = dict(**_ratio(sum(r['success'] for r in group), len(group)), rows=group)
        flags = [r['desired_token_correct'] for r in group]
        aggregates[tag] = dict(
            nl_accuracy_label={'RS':'nl rewrite acc', 'PS':'nl rephrase acc', 'NS':'nl neighborhood acc'}[tag],
            tf_token_micro=_ratio(sum(sum(f) for f in flags), sum(map(len, flags))),
            tf_prompt_macro=dict(value=_mean([sum(f) / len(f) for f in flags]), denominator=len(group)),
            tf_strict=_ratio(sum(r['desired_strict'] for r in group), len(group)),
            nll={side + '_prompt_macro': _mean([r[side + '_nll'] for r in group]) for side in ('new', 'true', 'desired')},
            desired_margin_prompt_macro=_mean([r['desired_margin'] for r in group]))
    joint = []
    for case in request_ids:
        rewrite = [r for r in rows if r['case_id'] == case and r['family'] == 'RS']
        paras = [r for r in rows if r['case_id'] == case and r['family'] == 'PS']
        if len(rewrite) == 1 and len(paras) == 2:
            joint.append(dict(case_id=case, strict=all(r['desired_strict'] for r in rewrite + paras),
                              preference=all(r['success'] for r in rewrite + paras)))
    return dict(request_ids=request_ids, request_order=digest(request_ids), rows=rows, metrics=metrics, aggregates=aggregates,
                joint=dict(definition='R plus exactly two canonical P, all desired target tokens correct',
                           eligible_requests=len(joint), omitted_case_ids=[i for i in request_ids if i not in {r['case_id'] for r in joint}],
                           tf_strict=_ratio(sum(r['strict'] for r in joint), len(joint)),
                           preference=_ratio(sum(r['preference'] for r in joint), len(joint)), rows=joint))


def evaluate(model, tok, records):
    """One canonical forward stream supplies BLUE preference and all TF metrics."""
    records = list(records)
    if not records or len({r['case_id'] for r in records}) != len(records):
        raise ValueError('nonempty unique record inventory required')
    if tok.padding_side != 'right':
        raise ValueError('canonical tokenizer padding_side must remain right')
    binder = _binding()
    modules = binder._binding()
    pairs = modules['evaluator'].counterfact_pairs(records)
    pairs['locality_target_new'] = modules['locality'].counterfact_locality_target_new_pairs(records)
    raw = {kind: modules['evaluator'].evaluate_pairs(model, tok, values, device=binder._device(model), microbatch_size=16)
           for kind, values in pairs.items()}
    result = summarize(_compact(raw), [int(r['case_id']) for r in records])
    result.update(metrics_contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
                  source_binding=modules['receipt'], evaluator_controller_influence=0,
                  evaluator_layout='HISTORICAL_MICROBATCH16_MANUAL_LEFT_PADDING_NO_POSITION_OVERRIDE',
                  transaction_guard='CALLER_OWNED', forwards_for_additional_accuracy=0)
    return result


def subset(result, case_ids):
    wanted = set(case_ids)
    if not wanted.issubset(result['request_ids']):
        raise ValueError('requested subset contains unobserved cases')
    return summarize([r for r in result['rows'] if r['case_id'] in wanted], [i for i in result['request_ids'] if i in wanted])


def merge_at_write(results):
    """Latest valid case inventory wins; caller excludes invalidated ledger IDs."""
    latest, order = {}, []
    for result in results:
        for case in result['request_ids']:
            if case not in latest:
                order.append(case)
            latest[case] = [r for r in result['rows'] if r['case_id'] == case]
    rows = [r for _, tag in FAMILIES for case in order for r in latest[case] if r['family'] == tag]
    return summarize(rows, order)


def _paired_rows(before, after):
    a, b = before['rows'], after['rows']
    key = lambda r: (r['family'], r['case_id'], r['prompt_index'], r['identity'], r['token_identity'])
    if before['request_ids'] != after['request_ids'] or list(map(key, a)) != list(map(key, b)):
        raise ValueError('paired inventory/order/token identity mismatch')
    return list(zip(a, b, strict=True))


def _cluster_ci(pairs, case_ids, repeats=10000, seed=20260920):
    """Paired difference CI; ratio-of-sums preserves token-micro weighting."""
    import numpy as np
    names = ('preference', 'tf_strict', 'tf_token_micro', 'tf_prompt_macro', 'new_nll', 'true_nll', 'desired_nll', 'desired_margin')
    if not pairs:
        return {name: None for name in names}
    index = {case: i for i, case in enumerate(case_ids)}
    # numerator difference and invariant denominator per request and metric.
    num = np.zeros((len(case_ids), len(names)), dtype=np.float64)
    den = np.zeros_like(num)
    for a, b in pairs:
        i = index[a['case_id']]
        af, bf = a['desired_token_correct'], b['desired_token_correct']
        if len(af) != len(bf):
            raise ValueError('paired token count mismatch')
        values = [int(b['success']) - int(a['success']), int(b['desired_strict']) - int(a['desired_strict']),
                  sum(bf) - sum(af), (sum(bf) - sum(af)) / len(af)]
        values += [b[key] - a[key] for key in ('new_nll', 'true_nll', 'desired_nll', 'desired_margin')]
        num[i] += values
        den[i] += [1, 1, len(af), 1, 1, 1, 1, 1]
    rng = np.random.default_rng(seed)
    sampled = []
    for start in range(0, repeats, 256):
        indices = rng.integers(0, len(case_ids), size=(min(256, repeats-start), len(case_ids)))
        total_den = den[indices].sum(axis=1)
        sampled.append(np.divide(num[indices].sum(axis=1), total_den,
                                 out=np.full_like(total_den, np.nan), where=total_den != 0))
    values = np.concatenate(sampled)
    return {name: dict(delta=float(num[:, j].sum() / den[:, j].sum()),
                       percentile95=[float(v) for v in np.nanquantile(values[:, j], [.025, .975])],
                       defined_resamples=int(np.isfinite(values[:, j]).sum())) for j, name in enumerate(names)}


def paired(before, after, case_ids=None, *, bootstrap=True):
    if case_ids is not None:
        before, after = subset(before, case_ids), subset(after, case_ids)
    pairs = _paired_rows(before, after)
    result = dict(direction='after minus before', request_ids=before['request_ids'], families={})
    for _, tag in FAMILIES:
        group = [(a, b) for a, b in pairs if a['family'] == tag]
        changes = []
        for a, b in group:
            changes.append(dict(case_id=a['case_id'], prompt_index=a['prompt_index'], identity=a['identity'],
                before_success=a['success'], after_success=b['success'], before_strict=a['desired_strict'], after_strict=b['desired_strict'],
                lost=a['success'] and not b['success'], gained=not a['success'] and b['success'],
                strict_lost=a['desired_strict'] and not b['desired_strict'], strict_gained=not a['desired_strict'] and b['desired_strict'],
                new_nll_delta=b['new_nll']-a['new_nll'], true_nll_delta=b['true_nll']-a['true_nll'],
                desired_nll_delta=b['desired_nll']-a['desired_nll'], desired_margin_delta=b['desired_margin']-a['desired_margin'],
                desired_token_correct_delta=sum(b['desired_token_correct'])-sum(a['desired_token_correct']),
                token_lost_positions=[i for i, (x, y) in enumerate(zip(a['desired_token_correct'], b['desired_token_correct'], strict=True)) if x and not y],
                token_gained_positions=[i for i, (x, y) in enumerate(zip(a['desired_token_correct'], b['desired_token_correct'], strict=True)) if not x and y]))
        result['families'][tag] = dict(denominator=len(group), rows=changes,
            lost_ids=[[r['case_id'], r['prompt_index']] for r in changes if r['lost']],
            gained_ids=[[r['case_id'], r['prompt_index']] for r in changes if r['gained']],
            strict_lost_ids=[[r['case_id'], r['prompt_index']] for r in changes if r['strict_lost']],
            strict_gained_ids=[[r['case_id'], r['prompt_index']] for r in changes if r['strict_gained']],
            cluster_bootstrap=_cluster_ci(group, before['request_ids']) if bootstrap else None)
    joint_before = before['joint']['rows']
    joint_after = after['joint']['rows']
    if [r['case_id'] for r in joint_before] != [r['case_id'] for r in joint_after]:
        raise ValueError('paired joint inventory mismatch')
    joint_pairs = []
    for a, b in zip(joint_before, joint_after, strict=True):
        pair = []
        for source in (a, b):
            pair.append(dict(case_id=source['case_id'], success=source['preference'], desired_strict=source['strict'],
                             desired_token_correct=[source['strict']], new_nll=0., true_nll=0., desired_nll=0., desired_margin=0.))
        joint_pairs.append(tuple(pair))
    joint_ci = _cluster_ci(joint_pairs, before['request_ids']) if bootstrap and joint_pairs else {}
    result['joint'] = dict(denominator=len(joint_pairs),
        rows=[dict(case_id=a['case_id'], preference_lost=a['preference'] and not b['preference'],
                   preference_gained=not a['preference'] and b['preference'],
                   strict_lost=a['strict'] and not b['strict'], strict_gained=not a['strict'] and b['strict'])
              for a,b in zip(joint_before,joint_after,strict=True)],
        cluster_bootstrap={k:joint_ci[k] for k in ('preference','tf_strict') if k in joint_ci})
    result['bootstrap'] = dict(unit='request cluster; all its prompts and target tokens travel together', repeats=10000, seed=20260920,
                               interval='paired percentile 95%', performed=bootstrap)
    return result


def retention(baseline, before, after):
    """W0-correct N retention and consecutive-step recovered/newly-lost IDs."""
    first = _paired_rows(baseline, before)
    second = _paired_rows(before, after)
    rows = []
    token_correct = token_total = 0
    for (base, prior), (_, now) in zip(first, second, strict=True):
        if base['family'] != 'NS':
            continue
        for a, b in zip(base['desired_token_correct'], now['desired_token_correct'], strict=True):
            token_total += int(a)
            token_correct += int(a and b)
        rows.append(dict(case_id=base['case_id'], prompt_index=base['prompt_index'], identity=base['identity'],
                         baseline_success=base['success'], before_success=prior['success'], after_success=now['success'],
                         baseline_strict=base['desired_strict'], after_strict=now['desired_strict'],
                         retained=base['success'] and now['success'],
                         recovered=base['success'] and not prior['success'] and now['success'],
                         newly_lost=base['success'] and prior['success'] and not now['success']))
    return dict(rows=rows, w0_correct_preference_retention=_ratio(sum(r['retained'] for r in rows), sum(r['baseline_success'] for r in rows)),
                w0_correct_strict_retention=_ratio(sum(r['baseline_strict'] and r['after_strict'] for r in rows), sum(r['baseline_strict'] for r in rows)),
                w0_correct_token_retention=_ratio(token_correct, token_total),
                recovered_ids=[[r['case_id'], r['prompt_index']] for r in rows if r['recovered']],
                newly_lost_ids=[[r['case_id'], r['prompt_index']] for r in rows if r['newly_lost']])
