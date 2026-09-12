"""A-OS at immutable joint A0 endpoint, pinned design §§6–9.

One full-space two-layer functional quadratic, one joint Current equality.
No A0 optimization, model binding, physical write/history, BF loop or fallback.
The caller owns WA Current teachers and We Base/Past teachers. Shared SH2
functional/PCG/equality kernels remain the sole solver implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from copy import deepcopy
from typing import Callable
import math
import time

import torch

from ..functional import FunctionalPanel
from ..linear_solve import WeightTree, Operator, add, scale, finite, dot, norm
from ..elastic_qp import solve_constrained


@dataclass
class AProblem:
    we: WeightTree
    d0: WeightTree  # Actual stored WA-We, not an independent native target.
    support: tuple[int, ...]
    base: FunctionalPanel
    past: FunctionalPanel
    current: FunctionalPanel
    native_metric: Operator              # Projected S/mean_eigenvalue.
    native_metric_inverse: Operator      # Exact projected inverse of above.
    raw_native_metric: Operator          # Projected raw S, NOT normalized H.
    project: Operator
    native_risks: tuple[float, float]
    q_balanced: tuple[float, float]       # Frozen A0 writer-image calibration.
    sigma_e: float
    validate_state: Callable[[WeightTree], dict]
    fixture_identity: dict
    balance_weight: float = .1
    wa: WeightTree | None = None          # Exact saved WA preferred over replay.


def _typed(values, reference, label):
    if len(values) != len(reference):
        raise ValueError(f'A_{label}_SUPPORT_LENGTH')
    for value, expected in zip(values, reference):
        if value.shape != expected.shape or value.dtype != torch.float32 or value.device != expected.device:
            raise TypeError(f'A_{label}_VECTOR_SHAPE_DTYPE_DEVICE')
    return finite(tuple(values))


def _state(problem, weights):
    receipt = problem.validate_state(weights)
    if not receipt.get('selected_fp32_shape_valid') or not receipt.get('nonselected_unchanged'):
        raise RuntimeError('A_ACTUAL_WEIGHT_OR_NONSELECTED_GUARD_FAILURE')
    if receipt.get('history_append_count', 0) or receipt.get('compute_z_count', 0):
        raise RuntimeError('A_INNER_STATE_CONTAMINATION')
    return receipt


def _pinned_linear(row):
    # Explicit ownership of derivative tensors: no caller/live-state reuse.
    return replace(row, gradient=tuple(g.detach().clone() for g in row.gradient),
                   nll_gradient=tuple(g.detach().clone() for g in row.nll_gradient),
                   context_rows=deepcopy(row.context_rows), ledger=deepcopy(row.ledger))


def _observed(row):
    return dict(value=row.value, mean_nll=row.mean_nll, context_rows=row.context_rows)


def run_os(problem: AProblem, *, on_stage=None, on_solution=None):
    """Return (FP32 endpoint tuple, scalar/context receipt), without commit.

    on_stage(stage_name, receipt) fires at LINEARIZATION_COMPLETE and after the
    first *natural* full K matvec. It adds no duplicate operator call and does
    not require PCG/endpoint completion for an initial execution observation.
    on_solution(receipt, endpoint, actual_correction) is caller-owned storage.
    """
    started = time.monotonic()
    if problem.support != (4, 8) or len(problem.we) != 2 or len(problem.d0) != 2:
        raise ValueError('A_OS_SUPPORT_MUST_BE_L4_L8')
    if (problem.base.role, problem.past.role, problem.current.role) != ('base', 'past', 'current'):
        raise ValueError('A_PANEL_ROLE_MISMATCH')
    if any(panel.tau != .1 for panel in (problem.base, problem.past, problem.current)):
        raise ValueError('A_TAU_LOCK')
    if problem.balance_weight != .1:
        raise ValueError('A_OS_BALANCE_LOCK')
    if (len(problem.q_balanced) != 2 or not all(math.isfinite(q) and q > 0 for q in problem.q_balanced)
            or not math.isfinite(problem.sigma_e) or problem.sigma_e < 1e-3):
        raise ValueError('A_A0_CALIBRATION_INVALID')
    if len(problem.native_risks) != 2 or not all(math.isfinite(r) for r in problem.native_risks):
        raise FloatingPointError('A_NONFINITE_NATIVE_RISK')
    we = tuple(w.detach().clone() for w in problem.we)
    _typed(we, problem.we, 'WE')
    _typed(problem.d0, we, 'D0')
    supplied_d0 = tuple(d.detach().clone() for d in problem.d0)
    anchor = (tuple(w.detach().clone() for w in problem.wa) if problem.wa is not None
              else tuple((w + d).detach() for w, d in zip(we, supplied_d0)))
    _typed(anchor, we, 'WA')
    d0 = tuple((w - e).detach() for w, e in zip(anchor, we))
    if problem.wa is not None and not all(torch.equal(d, supplied) for d, supplied in zip(d0, supplied_d0)):
        raise ValueError('A_SAVED_WA_D0_IDENTITY_MISMATCH')
    entry_validation = _state(problem, anchor)
    sigmas = tuple(max(r, 1e-3) for r in problem.native_risks)
    balance_coefficients = tuple(problem.balance_weight * q / problem.sigma_e**2 for q in problem.q_balanced)
    panels = (('base', problem.base), ('past', problem.past), ('current', problem.current))
    counts0 = {name: dict(panel.counts) for name, panel in panels}
    op_counts = dict(full_operator_matvecs=0, native_metric_calls=0,
                     raw_native_metric_calls=0, native_inverse_calls=0, projection_calls=0)

    def project(values):
        op_counts['projection_calls'] += 1
        return _typed(problem.project(values), values, 'PROJECT')

    def native(values):
        op_counts['native_metric_calls'] += 1
        return _typed(problem.native_metric(values), values, 'NATIVE')

    def raw_native(values):
        op_counts['raw_native_metric_calls'] += 1
        return _typed(problem.raw_native_metric(values), values, 'RAW_NATIVE')

    def balance_action(values):
        return tuple(value * coefficient for value, coefficient in zip(raw_native(values), balance_coefficients))

    def counts_now():
        return {name: {key: panel.counts[key] - counts0[name][key] for key in panel.counts}
                for name, panel in panels}

    bl = _pinned_linear(problem.base.linearize(anchor))
    pl = _pinned_linear(problem.past.linearize(anchor))
    cl = _pinned_linear(problem.current.linearize(anchor, need_nll_gradient=True))
    for row in (bl, pl, cl):
        if not math.isfinite(row.value) or not math.isfinite(row.mean_nll):
            raise FloatingPointError('A_NONFINITE_PANEL_LINEARIZATION')
        _typed(row.gradient, anchor, 'GRADIENT')
        _typed(row.nll_gradient, anchor, 'NLL_GRADIENT')
    # Pinned A0 endpoint, not a mutable subsequent node or native WN.
    derivative_weights = tuple(w.detach().clone() for w in anchor)
    g_base = project(scale(bl.gradient, 1 / sigmas[0]))
    g_past = project(scale(pl.gradient, 1 / sigmas[1]))
    a = project(cl.nll_gradient)
    balance_gradient = project(balance_action(d0))
    u = project(add(add(scale(g_base, 2.), g_past), balance_gradient))
    linear_stage = dict(arm='A-OS', support=[4, 8], h=1., t_e=0.,
        fixture_identity=deepcopy(problem.fixture_identity), native_risks=list(problem.native_risks),
        sigmas=list(sigmas), q_balanced=list(problem.q_balanced), sigma_e=problem.sigma_e,
        balance_coefficients=list(balance_coefficients), balance_metric='RAW_PROJECTED_S',
        balance_gradient_norm=norm(balance_gradient), u_norm=norm(u), current_gradient_norm=norm(a),
        current_profile_gradient_observed_norm=norm(cl.gradient),
        current_profile_linear_term='EXACT_ZERO_AT_WA_REFERENCE; NOT_ADDED',
        counts=counts_now(), operator_counts=dict(op_counts),
        entry_validation=entry_validation, wall_seconds=time.monotonic()-started)
    if on_stage is not None:
        on_stage('LINEARIZATION_COMPLETE', linear_stage)

    initial_positive_operator_observed = False

    def operator(direction):
        nonlocal initial_positive_operator_observed
        x = project(_typed(direction, anchor, 'K_INPUT'))
        out = scale(native(x), .01)
        out = add(out, _typed(problem.base.ggn(derivative_weights, x), x, 'BASE_GGN'), 2 / sigmas[0])
        out = add(out, _typed(problem.past.ggn(derivative_weights, x), x, 'PAST_GGN'), 1 / sigmas[1])
        out = add(out, _typed(problem.current.ggn(derivative_weights, x), x, 'CURRENT_GGN'))
        out = project(add(out, balance_action(x)))
        op_counts['full_operator_matvecs'] += 1
        # Mirror PCG's existing intrinsic p^T K p check *before* publishing
        # the initial marker. No extra K evaluation or changed tolerance.
        # A zero stationarity vector is valid for an all-zero no-action solve;
        # it does not establish a positive-direction initial marker.
        if not initial_positive_operator_observed and norm(direction) > 0:
            quadratic_form = dot(direction, out)
            if quadratic_form <= 0:
                raise FloatingPointError('PCG_NONPOSITIVE_CURVATURE')
            initial_positive_operator_observed = True
            if on_stage is not None:
                on_stage('FIRST_FULL_OPERATOR_MATVEC_COMPLETE', dict(arm='A-OS', h=1.,
                    finite=True, quadratic_form=quadratic_form,
                    input_norm=norm(direction), output_norm=norm(out),
                    counts=counts_now(), operator_counts=dict(op_counts),
                    pcg_completion_required=False, endpoint_completion_required=False,
                    functional_cross_block_dropped=0, wall_seconds=time.monotonic()-started))
        return out

    def precondition(direction):
        projected = project(direction)
        op_counts['native_inverse_calls'] += 1
        value = _typed(problem.native_metric_inverse(projected), direction, 'NATIVE_INVERSE')
        return project(scale(value, 1 / .01))

    result = solve_constrained(operator, u, a, 0., precondition=precondition,
                               h=1., rtol=1e-4, maxiter=20)
    correction = _typed(result.correction, anchor, 'CORRECTION')
    endpoint = finite(tuple((w + c).detach() for w, c in zip(anchor, correction)))
    actual_correction = tuple((w - b).detach() for w, b in zip(endpoint, anchor))
    actual_cumulative = tuple((w - e).detach() for w, e in zip(endpoint, we))
    validation = _state(problem, endpoint)
    before = [_observed(row) for row in (bl, pl, cl)]
    after = [panel.observe(endpoint) for _, panel in panels]
    raw_before = raw_native(d0)
    raw_after = raw_native(actual_cumulative)
    per_layer = []
    for index, layer in enumerate(problem.support):
        per_layer.append(dict(layer=layer,
            d0_actual_frobenius=norm((d0[index],)), correction_frobenius=norm((correction[index],)),
            actual_correction_frobenius=norm((actual_correction[index],)),
            actual_cumulative_frobenius=norm((actual_cumulative[index],)),
            raw_native_energy_before=dot((d0[index],), (raw_before[index],)),
            raw_native_energy_after=dot((actual_cumulative[index],), (raw_after[index],)),
            signed_current_linear_change=dot((a[index],), (correction[index],)),
            signed_base_linear_change=dot((g_base[index],), (correction[index],)),
            signed_past_linear_change=dot((g_past[index],), (correction[index],)),
            interpretation='LOCAL_GRADIENT_DOT_CORRECTION; NOT_REMOVAL_INTERVENTION'))
    leaked = add(correction, project(correction), -1.)
    correction_norm = norm(correction)
    row = dict(arm='A-OS', status=result.diagnostics['status'], node=0, s=0., h=1., t_e=0., support=[4, 8],
        source_fixture=deepcopy(problem.fixture_identity), before=before, after=after,
        current_reference='IMMUTABLE_SAVED_WA', base_past_reference='IMMUTABLE_We',
        native_risks=list(problem.native_risks), sigmas=list(sigmas),
        q_balanced=list(problem.q_balanced), sigma_e=problem.sigma_e,
        balance_weight=problem.balance_weight, balance_coefficients=list(balance_coefficients),
        balance_metric='RAW_PROJECTED_S_NOT_NORMALIZED_H',
        balance_objective_before=.5*sum(c*dot((d,),(sd,)) for c,d,sd in zip(balance_coefficients,d0,raw_before)),
        balance_objective_after=.5*sum(c*dot((d,),(sd,)) for c,d,sd in zip(balance_coefficients,actual_cumulative,raw_after)),
        current_profile_gradient_observed_norm=norm(cl.gradient), current_profile_gradient_in_u=False,
        predicted_current_change=dot(a, correction), actual_current_change=after[2]['mean_nll']-before[2]['mean_nll'],
        predicted_risk_change=[dot(g_base, correction), dot(g_past, correction)],
        actual_risk_change=[(after[j]['value']-before[j]['value'])/sigmas[j] for j in range(2)],
        correction_norm=correction_norm, actual_delta_norm=norm(actual_correction),
        actual_cumulative_norm=norm(actual_cumulative),
        fp32_addition_residual_norm=norm(add(actual_correction, correction, -1.)),
        d0_anchor_reconstruction_difference=norm(add(d0, supplied_d0, -1.)),
        native_action=dot(correction, native(correction)),
        actual_native_action=dot(actual_correction, native(actual_correction)),
        permitted_range_relative_residual=norm(leaked)/correction_norm if correction_norm else 0.,
        per_layer=per_layer, solver=result.diagnostics,
        entry_validation=entry_validation, state_validation=validation,
        counts=counts_now(), operator_counts=dict(op_counts), wall_seconds=time.monotonic()-started,
        compute_z=0, history_append=0, inner_physical_assignment=0, extra_h_application=0,
        layer_equality_count=0, joint_equality_count=1, functional_cross_block_dropped=0,
        initial_positive_full_operator_observed=initial_positive_operator_observed,
        actual_removal_attributions='CALLER_ENDPOINT_EVALUATION_REQUIRED',
        terminal_materialization='CALLER_REQUIRED_AFTER_RUN', endpoint_selected_by_performance=False)
    if on_solution is not None:
        on_solution(row, endpoint, actual_correction)
    return endpoint, row
