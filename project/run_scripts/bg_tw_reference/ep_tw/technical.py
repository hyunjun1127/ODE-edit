"""Bounded first-episode model checks; no new target/gradient policy sweep.

Tolerances below are technical FP32 finite-difference resolution, never an
allowance for positive current quality loss.  Store this configuration in the
execution lock before any model experiment.  Callers own actual GPU admission.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import torch

from .model_adapter import ModelBoundary, anchored_weight, tensor_sha


DEFAULT_NUMERICS = dict(
    self_kl_document_abs_max=1e-6,
    teacher_normalizer_abs_max=5e-5,
    fd_relative_weight_step=1e-3,
    fd_step_factors=[1., .5],
    fd_derivative_relative_tolerance=.15,
    fd_derivative_absolute_tolerance=1e-7,
    fd_convergence_relative_tolerance=.15,
    fd_signal_roundoff_multiple=8.,
    functional_materialized_max_abs=0.,
    current_positive_quality_tolerance=0.,
)


@dataclass(frozen=True)
class TechnicalNumerics:
    """Source-locked FP32 probes, independent of all quality observations.

    Central differences use two fixed relative *weight* perturbations so a
    weak residual map does not make the probe invisible by construction.
    Both scales must resolve FP32 signal, match the existing analytic gradient
    within 15%, and agree within 15%; otherwise this is a technical hold.
    Self-KL 1e-6 is only a cross-forward arithmetic check, never E tolerance.
    """
    self_kl_document_abs_max: float = 1e-6
    teacher_normalizer_abs_max: float = 5e-5
    fd_relative_weight_step: float = 1e-3
    fd_step_factors: tuple = (1., .5)
    fd_derivative_relative_tolerance: float = .15
    fd_derivative_absolute_tolerance: float = 1e-7
    fd_convergence_relative_tolerance: float = .15
    fd_signal_roundoff_multiple: float = 8.
    functional_materialized_max_abs: float = 0.
    current_positive_quality_tolerance: float = 0.

    def to_dict(self):
        return asdict(self)


def verify_w0_teacher(adapter, numerics=None):
    """Actual W0 forward over all S64, not SAME_LOGP self KL tautology."""
    numerical = dict(DEFAULT_NUMERICS, **(numerics or {}))
    observed = adapter.generic('S64')
    largest = max(abs(row['kl']) for row in observed['rows'])
    normalizer = max(row['teacher_normalizer_max_abs'] for row in observed['rows'])
    receipt = dict(status='PASS' if largest <= numerical['self_kl_document_abs_max']
                   and normalizer <= numerical['teacher_normalizer_abs_max'] else 'FAIL',
                   S64=observed, max_document_abs_kl=largest,
                   normalizer_max_abs=normalizer, numerical=numerical,
                   observation='INDEPENDENT_ACTUAL_W0_FORWARD_VS_STORED_FIXED_TEACHER',
                   score_input_indices=[129,257], score_logits_indices=[128,256],
                   scored_positions_per_document=128,
                   current_quality_allowance=0.)
    if receipt['status'] != 'PASS':
        raise ModelBoundary('W0_TEACHER_SELF_KL_OR_NORMALIZATION:' + repr(receipt))
    return receipt


def functional_materialized(adapter, correction, *, numerics=None, recorder=None):
    """Same full 257-token sequence and full logits under both real routes."""
    numerical = dict(DEFAULT_NUMERICS, **(numerics or {}))
    adapter._assert_episode()
    original = adapter.weight.detach().clone()
    nonselected = [(p,p.data_ptr(),p._version) for name,p in adapter.parameters.items()
                   if name != adapter.weight_name]
    before_sha = tensor_sha(original)
    inputs, teacher, source_id = adapter.teacher.document(0, adapter.device)
    del teacher
    expected_weight = anchored_weight(correction.detach(), adapter.raw, adapter.fixed_a)
    try:
        with torch.no_grad():
            functional = adapter._forward(inputs, torch.ones_like(inputs), correction.detach())
            adapter.weight.copy_(expected_weight)
            materialized = adapter._forward(inputs, torch.ones_like(inputs), None)
            maximum = float((functional-materialized).abs().max())
            exact = torch.equal(functional, materialized)
    finally:
        with torch.no_grad():
            adapter.weight.copy_(original)
    if tensor_sha(adapter.weight) != before_sha:
        raise ModelBoundary('TECHNICAL_WEIGHT_RESTORE')
    if any(p.data_ptr()!=ptr or p._version!=version for p,ptr,version in nonselected):
        raise ModelBoundary('TECHNICAL_NONSELECTED_MUTATION')
    receipt = dict(status='PASS' if maximum <= numerical['functional_materialized_max_abs'] else 'FAIL',
        all_tokens=257, all_logits=True, source_row_id=source_id,
        logits_max_abs=maximum, logits_exact=exact,
        weight_sha256=tensor_sha(expected_weight), restored_sha256=before_sha,
        nonselected_pointer_version_preserved=True, model_forwards=2,
        endpoint_restore_exact=True)
    if recorder is not None:
        recorder(receipt)  # durable evidence before a numerical FAIL is raised
    if receipt['status'] != 'PASS':
        raise ModelBoundary('FUNCTIONAL_MATERIALIZED_ALL_TOKEN_PARITY:' + repr(receipt))
    return receipt


def directional_fd(adapter, records, gradient, *, objective, numerics=None):
    """Two preregistered central differences at one actual RAW episode.

    Uses the already acquired scientific gradient; no backward is performed.
    A direction parallel to that gradient is scaled by its executable weight
    action so FP32 cancellation can be distinguished from a derivative error.
    Failure/insufficient signal is a typed technical hold, never tolerance tune.
    """
    numerical = dict(DEFAULT_NUMERICS, **(numerics or {}))
    if objective not in ('E','D'):
        raise ModelBoundary('FD_OBJECTIVE_NOT_CURRENT_OR_S64')
    if not torch.isfinite(gradient).all():
        raise ModelBoundary('FD_NONFINITE_GRADIENT')
    norm = float(gradient.double().norm())
    if norm == 0.:
        return dict(status='ZERO_GRADIENT_NO_NONZERO_DIRECTION', objective=objective,
                    gradient_norm=0., backward_calls=0, model_forward_count=0,
                    nonzero_direction_fd='NOT_TESTED')
    direction = gradient / norm
    action = direction @ adapter.fixed_a
    action_norm = float(action.double().norm())
    if action_norm == 0.:
        raise ModelBoundary('FD_ZERO_EXECUTABLE_DIRECTION')
    weight_norm = float(adapter.raw.double().norm())
    step = numerical['fd_relative_weight_step'] * max(weight_norm, 1.) / action_norm
    analytic = float((gradient.double()*direction.double()).sum())
    rows, observed_forwards = [], 0
    started = time.monotonic()
    for factor in numerical['fd_step_factors']:
        h = step * factor
        def observe(sign):
            c = sign * h * direction
            return adapter.current(records, c) if objective=='E' else adapter.generic('S64', c)
        plus, minus = observe(1.), observe(-1.)
        observed_forwards += plus['counts']['forwards'] + minus['counts']['forwards']
        fp, fm = plus[objective], minus[objective]
        numerical_slope = (fp-fm)/(2*h)
        scale = max(abs(fp), abs(fm), 1e-6)
        roundoff = numerical['fd_signal_roundoff_multiple'] * torch.finfo(torch.float32).eps * scale
        signal = abs(fp-fm)
        error = abs(numerical_slope-analytic)
        allowance = max(numerical['fd_derivative_absolute_tolerance'],
                        numerical['fd_derivative_relative_tolerance']*abs(analytic))
        rows.append(dict(step=h, factor=factor, plus=fp, minus=fm,
            analytic=analytic, central_difference=numerical_slope, absolute_error=error,
            relative_error=error/max(abs(analytic),1e-30), signal=signal,
            roundoff_threshold=roundoff, signal_resolved=signal>roundoff,
            derivative_pass=error<=allowance))
    a,b=rows[0]['central_difference'],rows[1]['central_difference']
    convergence=abs(a-b)/max(abs(analytic),numerical['fd_derivative_absolute_tolerance'])
    okay=(all(r['signal_resolved'] and r['derivative_pass'] for r in rows)
          and convergence<=numerical['fd_convergence_relative_tolerance'])
    receipt=dict(status='PASS' if okay else 'FAIL_OR_UNRESOLVED_FP32_FD', objective=objective,
        rows=rows, convergence_relative=convergence, gradient_norm=norm,
        direction_action_norm=action_norm, raw_weight_norm=weight_norm,
        one_scientific_gradient_reused=True, backward_calls=0,
        model_forward_count=observed_forwards, seconds=time.monotonic()-started,
        numerical=numerical)
    if not okay:
        raise ModelBoundary('DIRECTIONAL_FD:' + repr(receipt))
    return receipt


def initial_episode_checks(adapter, records, sweeps, *, numerics=None):
    """Executed once before first commit; returns evidence, not a G0 claim."""
    numerical = dict(DEFAULT_NUMERICS, **(numerics or {}))
    if numerical['current_positive_quality_tolerance'] != 0.:
        raise ModelBoundary('POSITIVE_QUALITY_ALLOWANCE_FORBIDDEN')
    zero = torch.zeros_like(sweeps['gE'])
    current_raw = adapter.current(records)
    generic_raw = adapter.generic('S64')
    current_exact = current_raw['rows'] == sweeps['current']['rows']
    generic_exact = generic_raw['rows'] == sweeps['generic']['rows']
    if not current_exact or not generic_exact:
        raise ModelBoundary('C0_ACTUAL_NATIVE_METRICS_PARITY')
    zero_parity = functional_materialized(adapter, zero, numerics=numerical)
    d = sweeps['gE']
    denom = float((d @ adapter.fixed_a).double().norm())
    nonzero = d * (1e-4*float(adapter.raw.double().norm())/denom) if denom else zero
    nonzero_parity = functional_materialized(adapter, nonzero, numerics=numerical)
    e_fd = directional_fd(adapter, records, sweeps['gE'], objective='E', numerics=numerical)
    d_fd = directional_fd(adapter, records, sweeps['gD'], objective='D', numerics=numerical)
    return dict(status='MODEL_TECHNICAL_CHECKS_PASS_NOT_G0', C0_current_metrics_exact=current_exact,
        C0_generic_metrics_exact=generic_exact, zero_all_token_parity=zero_parity,
        nonzero_all_token_parity=nonzero_parity, current_directional_fd=e_fd,
        generic_directional_fd=d_fd, microbatch_mass=dict(current_reduction='group_sum/N100',
            generic_reduction='doc_mean/N64', current_microbatch=16, generic_microbatch=1), numerical=numerical,
        history_and_B1_B2_state='RUNNER_MUST_VERIFY', extra_target_calls=0,
        extra_backward_sweeps=0)


self_kl_check = verify_w0_teacher
validate_episode = initial_episode_checks
