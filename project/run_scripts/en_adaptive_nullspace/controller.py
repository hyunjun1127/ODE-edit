"""Common two-candidate EN controller, immutable native center, FP32 endpoints.

No model loading, z, current downstream forward, scientific metrics, tensor
persistence or hash sweep occurs here. ``evaluate`` sees a candidate FP32
weight and returns {J, L_R, L_H, ...}; the runner owns its cached suffix oracle.
"""
from __future__ import annotations
import math
import numpy as np
from .geometry import array
from .selector import quadratic_second_scale


def _objective(value):
    if isinstance(value, dict):
        result = dict(value)
        if 'J' not in result:
            result['J'] = result['loss']
    else:
        result = {'J': float(value)}
    result['J'] = float(result['J'])
    if not math.isfinite(result['J']):
        raise ValueError('nonfinite objective is a technical failure, not native fallback')
    return result


def _endpoint(native, ideal):
    if hasattr(native, 'detach'):
        import torch
        return (native.detach().to(torch.float64) + torch.as_tensor(ideal, device=native.device, dtype=torch.float64)).to(torch.float32)
    return (np.asarray(native, dtype=np.float64) + ideal).astype(np.float32)


def _clone(weight):
    return weight.detach().clone() if hasattr(weight, 'detach') else weight.copy()


def run_controller(native_weight, gradient, first_delta, loss0, evaluate, geometry, *,
                   native_norm, native_action, epsilon=.05, input_identity,
                   arm='EN', request_columns=None, objective_cache=None):
    """Return dict(weight, objective, ledger, status, evaluations, alias).

    Optional objective_cache is a list shared across arms at ONE native center;
    entries have weight (owned immutable RAM), input_identity, objective, label.
    Equality is exact materialized FP32 bytes/values, never approximate. All
    supplied objective terms must share the complete input_identity.
    """
    native_weight = _clone(native_weight)
    wn = np.asarray(array(native_weight))
    if wn.dtype != np.float32 or not np.isfinite(wn).all():
        raise ValueError('native FP32 endpoint required')
    g = np.asarray(array(gradient), dtype=np.float64)
    d1 = np.asarray(array(first_delta), dtype=np.float64)
    if g.shape != wn.shape or d1.shape != wn.shape or not np.isfinite(g).all() or not np.isfinite(d1).all():
        raise ValueError('nonfinite or mismatched direction/gradient')
    if not input_identity:
        raise ValueError('objective input identity must be sealed')
    j0 = _objective(loss0)
    if any(not math.isfinite(x) or x < 0 for x in (native_norm, native_action, epsilon)):
        raise ValueError('invalid native budget')
    cache = objective_cache if objective_cache is not None else []
    # Native is not evaluated again; immutable reference is sufficient here.
    local_cache = [dict(weight=native_weight, input_identity=input_identity, objective=j0, label='native'), *cache]
    ledger, accepted, evaluations = [], [], 0
    response_budget = epsilon * native_action
    eps64 = np.finfo(np.float64).eps

    def trial(scale, label):
        nonlocal evaluations
        ideal = scale * d1
        candidate = _endpoint(native_weight, ideal)
        candidate_np = np.asarray(array(candidate))
        actual = candidate_np.astype(np.float64) - wn.astype(np.float64)
        diagnostics = geometry.diagnostics(ideal, actual, request_columns=request_columns)
        slope = float(np.sum(g * actual))
        # Roundoff slack is reported separately; never increases semantic eps.
        scale64 = max(1., native_norm, response_budget, diagnostics['actual_response'])
        arithmetic_slack = 64 * eps64 * scale64
        ideal_range_slack = 1e-10 * max(1., diagnostics['ideal_norm'])
        checks = dict(
            finite=bool(diagnostics['finite']),
            ideal_norm=diagnostics['ideal_norm'] <= native_norm + arithmetic_slack,
            ideal_response=diagnostics['ideal_response'] <= response_budget + arithmetic_slack,
            actual_norm=diagnostics['actual_norm'] <= native_norm + diagnostics['rounding_norm'] + arithmetic_slack,
            actual_response=diagnostics['actual_response'] <= response_budget + diagnostics['rounding_response'] + arithmetic_slack,
            ideal_allowed_range=diagnostics['ideal_P_leakage'] <= ideal_range_slack,
            actual_allowed_range=diagnostics['actual_P_leakage'] <= diagnostics['ideal_P_leakage'] + diagnostics['rounding_P_leakage'] + arithmetic_slack)
        if not all(checks.values()):
            raise ValueError(f'invalid correction geometry: {checks}; {diagnostics}')
        alias = None
        obj = None
        for old in local_cache:
            if old['input_identity'] == input_identity and np.array_equal(np.asarray(array(old['weight'])).view(np.uint32), candidate_np.view(np.uint32)):
                alias, obj = old['label'], old['objective']
                break
        if obj is None:
            obj = _objective(evaluate(candidate))
            evaluations += 1
            entry = dict(weight=_clone(candidate), input_identity=input_identity, objective=obj, label=arm + ':' + label)
            local_cache.append(entry)
            cache.append(entry)
        decrease = obj['J'] < j0['J']
        armijo_limit = j0['J'] + 1e-4 * slope
        armijo = obj['J'] <= armijo_limit
        valid = decrease and armijo and slope < 0
        row = dict(arm=arm, trial=label, scale=float(scale), objective=obj,
                   native_objective=j0, actual_gradient_inner_product=slope,
                   ideal_gradient_inner_product=float(np.sum(g*ideal)),
                   response_budget=response_budget, semantic_epsilon=epsilon,
                   arithmetic_fp64_slack=arithmetic_slack,
                   geometry=diagnostics, geometry_checks=checks,
                   strict_decrease=decrease, armijo_limit=armijo_limit, armijo=armijo,
                   accepted=valid, alias=alias, input_identity=input_identity)
        ledger.append(row)
        if valid:
            accepted.append((obj['J'], diagnostics['actual_norm'], candidate, obj, row))
        return row

    if not np.any(d1):
        return dict(weight=_clone(native_weight), objective=j0, ledger=[], status='NO_SIGNAL_OR_NO_STEP',
                    evaluations=0, alias='native')
    first = trial(1., 'candidate1')
    slope = first['actual_gradient_inner_product']
    if slope < 0:
        quadratic = quadratic_second_scale(j0['J'], slope, first['objective']['J'])
        first['quadratic'] = quadratic
        if quadratic['scale'] is not None:
            trial(quadratic['scale'], 'candidate2')
    else:
        first['quadratic'] = dict(scale=None, reason='NO_ACTUAL_DESCENT_DIRECTION')
    if accepted:
        # Objective ties use exact equality; no unapproved KL tolerance band.
        selected = min(accepted, key=lambda item:(item[0], item[1]))
        return dict(weight=selected[2], objective=selected[3], ledger=ledger, status='ACCEPTED',
                    evaluations=evaluations, alias=selected[4]['alias'], selected_trial=selected[4]['trial'])
    return dict(weight=_clone(native_weight), objective=j0, ledger=ledger,
                status='SEARCH_LIMIT_NO_ACCEPTED_CANDIDATE', evaluations=evaluations, alias='native')
