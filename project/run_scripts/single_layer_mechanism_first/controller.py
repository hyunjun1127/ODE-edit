"""One-center, four-trial controllers for the prospectively fixed v1 methods.

There is no model load, observer, history append, checkpoint, or stage expansion
here. The caller owns endpoint-bound oracles and closes every transient binding.
EN-KL retains its KL acceptance, not the decision methods' choice acceptance.
All callbacks are in-scope training/Current/active-history callbacks only.

Endpoint hashes are recorded once per materialized candidate. Private endpoint
ownership and mutation versions protect callback boundaries; they are not an
adversarial alias audit or a substitute for the separately required actual T0.
An exception (including an evidence-sink failure) is never native fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any

import numpy as np
import torch

from . import decision, solver
from .config import SCALES
from project.run_scripts.single_layer_edit_preserving_correction.optimizer import (
    ARMIJO_C1, KL_FLOOR, tensor_sha256,
)


class ControllerFailure(RuntimeError):
    """Technical failure; partial event/counter evidence must be preserved."""

    def __init__(self, reason, *, events=None, counters=None):
        super().__init__(reason)
        self.reason = reason
        self.events = list(events or [])
        self.counters = dict(counters or {})


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _matrix(value, shape=None, dtype=torch.float64):
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    _require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
             value.ndim == 2 and value.is_floating_point(), 'CPU_MATRIX_REQUIRED')
    _require(shape is None or tuple(value.shape) == tuple(shape), 'MATRIX_SHAPE')
    _require(bool(torch.isfinite(value).all()), 'NONFINITE_MATRIX')
    return value.detach().to(dtype=dtype)


def _array(value, shape):
    if isinstance(value, torch.Tensor):
        _require(value.device.type == 'cpu', 'CPU_COEFFICIENT_ARRAY_REQUIRED')
        value = value.detach().numpy()
    result = np.asarray(value, dtype=np.float64)
    _require(result.shape == tuple(shape) and np.isfinite(result).all(), 'COEFFICIENT_ARRAY')
    return result


def _history(obs):
    if obs is None:
        return dict(phi=0., passed=True, rows=[], present=False)
    _require(math.isfinite(obs.phi_history) and obs.phi_history >= 0,
             'NONFINITE_HISTORY_RISK')
    _require(obs.coverage.get('complete') is True and
             obs.coverage.get('active_requests') == len(obs.rows), 'FULL_ACTIVE_HISTORY_REQUIRED')
    _require(isinstance(obs.guard_pass, (bool, np.bool_)), 'HISTORY_GUARD_FLAG')
    _require(all(math.isfinite(float(r['slack'])) for r in obs.rows), 'FINITE_HISTORY_SLACK')
    _require(not (not obs.rows and (obs.phi_history != 0 or not obs.guard_pass)),
             'EMPTY_HISTORY_NONZERO_OR_FAILED')
    return dict(phi=float(obs.phi_history), passed=bool(obs.guard_pass),
                rows=obs.rows, present=bool(obs.rows))


def _reference(obs):
    _require(obs.role == 'R512' and bool(obs.endpoint_identity) and bool(obs.input_identity),
             'REFERENCE_IDENTITY')
    _require(obs.coverage.get('complete') is True and obs.coverage.get('documents') == 512 and
             obs.coverage.get('positions') == obs.coverage.get('expected_positions') and
             len(obs.rows) == 512, 'FULL512_REQUIRED')
    _require(math.isfinite(obs.phi_reference) and obs.phi_reference >= 0,
             'NONFINITE_REFERENCE_RISK')
    _require(isinstance(obs.mismatches, int) and obs.mismatches >= 0, 'REFERENCE_MISMATCH_COUNT')
    positions = mismatches = 0
    for index, row in enumerate(obs.rows):
        n = row['scored_positions']
        _require(row['index'] == index and isinstance(n, int) and n > 0 and
                 len(row['positions']) == len(row['labels']) == len(row['correct']) == n,
                 'REFERENCE_ROW_CARDINALITY')
        _require(math.isfinite(float(row['mu'])), 'NONFINITE_REFERENCE_MARGIN')
        positions += n
        mismatches += sum(not bool(x) for x in row['correct'])
    _require(positions == obs.coverage['positions'] and mismatches == obs.mismatches,
             'REFERENCE_COVERAGE_COUNTS')
    return obs


def _check(value):
    if hasattr(value, 'passed'):
        return dict(passed=bool(value.passed), reason=value.reason, details=value.details)
    _require(isinstance(value, dict), 'CHECK_SCHEMA')
    passed = value.get('passed', value.get('pass'))
    _require(isinstance(passed, (bool, np.bool_)), 'CHECK_FLAG')
    return dict(passed=bool(passed), reason=value.get('reason', 'PASS' if passed else 'FAIL'),
                details=value.get('details', value))


def _current(value):
    _require(isinstance(value, dict) and isinstance(value.get('quality_pass'), (bool, np.bool_)),
             'CURRENT_CHECK_SCHEMA')
    quality = bool(value['quality_pass'])
    invariant = value.get('invariant')
    _require(not quality or (isinstance(invariant, dict) and
             isinstance(invariant.get('pass'), (bool, np.bool_))), 'CURRENT_INVARIANT_REQUIRED')
    inv = bool(invariant and invariant.get('pass') is True)
    _require(bool(value.get('pass')) == (quality and inv), 'CURRENT_CHECK_INCONSISTENT')
    return quality, inv


def _objective(value, *, gradient, shape, objective_id, expected=None):
    if isinstance(value, tuple) and len(value) == 3:
        loss, grad, rows = value
        supplied_identity = None
    elif isinstance(value, dict):
        loss, grad, rows = value['loss'], value.get('gradient'), value['rows']
        supplied_identity = value.get('objective_id')
    else:
        loss, grad, rows = value.loss, value.gradient, value.rows
        supplied_identity = getattr(value, 'objective_id', None)
    _require(supplied_identity is None or supplied_identity == objective_id, 'KL_OBJECTIVE_IDENTITY')
    loss = float(loss)
    _require(math.isfinite(loss), 'NONFINITE_KL')
    _require(loss >= -KL_FLOOR, 'KL_BELOW_ROUNDOFF_FLOOR')
    _require(isinstance(rows, list) and len(rows) == 512, 'KL_FULL512_REQUIRED')
    identities = []
    for index, row in enumerate(rows):
        _require(row['index'] == index and row['role'] == 'R512' and
                 isinstance(row['scored_positions'], int) and row['scored_positions'] > 0 and
                 math.isfinite(float(row['loss'])), 'KL_ROW_COVERAGE_OR_FINITE')
        identities.append((row['index'], row['source_row_id'], row['scored_positions']))
    _require(expected is None or identities == expected, 'KL_TRIAL_INPUT_COVERAGE_CHANGED')
    grad = _matrix(grad, shape) if gradient else None
    return loss, grad, rows, identities


@dataclass
class Result:
    arm: str
    weight: torch.Tensor
    ideal_delta: torch.Tensor
    actual_delta: torch.Tensor
    loss: float | None
    stop_reason: str
    counters: dict
    trials: list
    events: list
    summary: dict
    reference: Any = None
    history: Any = None
    current: Any = None
    details: dict = field(default_factory=dict)

    def receipt(self):
        return dict(schema='sl-mechanism-controller-v1', arm=self.arm,
                    ideal_delta_sha256=tensor_sha256(self.ideal_delta),
                    actual_delta_sha256=tensor_sha256(self.actual_delta),
                    ideal_delta_norm=float(self.ideal_delta.norm()), loss=self.loss,
                    stop_reason=self.stop_reason, counters=dict(self.counters), trials=self.trials,
                    **self.summary, details=self.details,
                    numerical=dict(scales=list(SCALES), centers=1,
                                   armijo_c1=ARMIJO_C1, KL_floor=KL_FLOOR),
                    history_appends_in_controller=0, official_observer_accesses=0,
                    nonlinear_or_global_optimality_claim=False)


class _Run:
    def __init__(self, arm, WN, native_history=None, event=None):
        _require(isinstance(WN, torch.Tensor) and WN.dtype == torch.float32, 'WN_FP32_REQUIRED')
        self.native = _matrix(WN, dtype=torch.float32).clone()
        self.native_sha = tensor_sha256(self.native)
        self.arm, self.event_sink = arm, event
        self.native_history, self.hist = native_history, _history(native_history)
        self.trials, self.events = [], []
        self.counters = dict(gradient_sweeps=0, objective_value_sweeps=0,
                             materializations=0, unique_scored_candidates=0,
                             candidate_cache_hits=0, proposal_checks=0,
                             current_checks=0, history_checks=0,
                             reference_candidate_sweeps=0, solver_calls=0,
                             linear_history_skips=0, accepted_rounds=0)
        self.details = dict(native_weight_sha256=self.native_sha,
                            native_history_status='PRESENT' if self.hist['present'] else 'N/A_EMPTY',
                            native_history_pass=self.hist['passed'])

    def emit(self, kind, **fields):
        row = dict(event=kind, arm=self.arm, **fields)
        self.events.append(row)
        if self.event_sink is not None:
            try:
                self.event_sink(row)
            except Exception as exc:
                raise ControllerFailure('EVIDENCE_WRITE_FAILURE', events=self.events,
                                        counters=self.counters) from exc

    def call(self, function, *args, **kwargs):
        tensors = [self.native] + [x for x in args if isinstance(x, torch.Tensor)]
        versions = [x._version for x in tensors]
        start = time.monotonic()
        value = function(*args, **kwargs)
        _require(all(t._version == v for t, v in zip(tensors, versions)), 'CALLBACK_ENDPOINT_MUTATION')
        self.counters['callback_wall_seconds'] = self.counters.get('callback_wall_seconds', 0.) + time.monotonic() - start
        return value

    def materialize(self, ideal):
        ideal = _matrix(ideal, self.native.shape)
        candidate = (self.native.double() + ideal).float()
        _require(bool(torch.isfinite(candidate).all()), 'NONFINITE_MATERIALIZATION')
        actual = candidate.double() - self.native.double()
        self.counters['materializations'] += 1
        return candidate, ideal, actual, tensor_sha256(candidate)

    def check_proposal(self, ideal, callback):
        if callback is None:
            return dict(passed=True, reason='PARENT_FIXED_PROJECTED_SPACE', details={})
        self.counters['proposal_checks'] += 1
        result = _check(self.call(callback, ideal))
        _require(result['passed'], 'IDEAL_PROJECTION_CHECK_FAILED:' + result['reason'])
        return result

    def check_history(self, weight, trial, callback):
        if not self.hist['present']:
            return self.native_history, self.hist
        _require(callback is not None, 'ACTIVE_HISTORY_CALLBACK_REQUIRED')
        self.counters['history_checks'] += 1
        obs = self.call(callback, weight, trial)
        facts = _history(obs)
        _require(obs.history_identity == self.native_history.history_identity and
                 obs.entry_anchor_identity == self.native_history.entry_anchor_identity and
                 [r['case_id'] for r in obs.rows] == [r['case_id'] for r in self.hist['rows']],
                 'HISTORY_TRIAL_INPUT_OR_ENTRY_CHANGED')
        return obs, facts

    def finish(self, reason, *, accepted=False, weight=None, ideal=None, actual=None,
               loss=None, reference=None, history=None, current=None, summary=None):
        if not accepted:
            weight = self.native
            ideal = actual = torch.zeros_like(self.native, dtype=torch.float64)
            history = self.native_history
        facts = _history(history)
        original_reason = reason
        if not accepted and not facts['passed']:
            reason = 'FALLBACK_WITH_PAST_VIOLATION'
        self.counters['accepted_rounds'] = int(accepted)
        selected_sha = tensor_sha256(weight) if accepted else self.native_sha
        out = dict(accepted=bool(accepted), native_fallback=not accepted,
                   actual_delta_norm=float(actual.norm()), selected_weight_sha256=selected_sha,
                   current_pass=True if not accepted else bool(current and current.get('pass')),
                   history_pass=facts['passed'], reference_pass=None, full512=False,
                   repaired_choices=0, phi_reference_native=None, phi_reference_gain=None,
                   tau_risk=None, fallback_cause=None if accepted else original_reason)
        out.update(summary or {})
        self.emit('SELECTED' if accepted else 'NATIVE_FALLBACK', stop_reason=reason,
                  fallback_cause=out['fallback_cause'], selected_weight_sha256=selected_sha,
                  accepted=bool(accepted))
        return Result(self.arm, weight, ideal, actual, loss, reason, dict(self.counters),
                      list(self.trials), list(self.events), out, reference, history, current,
                      dict(self.details))

    def fail(self, exc):
        if isinstance(exc, ControllerFailure):
            raise exc
        self.emit('TECHNICAL_FAILURE', exception_type=type(exc).__name__, reason=str(exc))
        raise ControllerFailure(str(exc), events=self.events, counters=self.counters) from exc


def native_result(WN, *, native_history=None, event=None):
    run = _Run('N4', WN, native_history, event)
    return run.finish('NATIVE_BASELINE')


def optimize_kl(WN, *, objective, project, current_check, proposal_check=None,
                history_check=None, native_history=None, initial_observation=None,
                objective_id='R512', space_status='RESOLVED', event=None):
    """One full-bank KL derivative and at most four immutable-WN proposals.

An optional initial observation is reusable only with its exact WN SHA and
objective identity; reuse does not claim a new gradient sweep in this call.
"""
    run = _Run('EN_KL_Q', WN, native_history, event)
    try:
        _require(space_status in ('RESOLVED', 'RANK_UNRESOLVED', 'REPAIR_SPACE_EMPTY'), 'SPACE_STATUS')
        run.details['space_status'] = space_status
        if space_status != 'RESOLVED':
            return run.finish(space_status)
        if initial_observation is None:
            run.counters['gradient_sweeps'] += 1
            observed = run.call(objective, run.native, gradient=True)
        else:
            get = initial_observation.get if isinstance(initial_observation, dict) else lambda k: getattr(initial_observation, k, None)
            _require(get('weight_sha256') == run.native_sha and get('objective_id') == objective_id,
                     'INITIAL_KL_REUSE_IDENTITY_REQUIRED')
            observed = initial_observation
            run.details['initial_gradient_reused'] = True
        L, G, _, identities = _objective(observed, gradient=True, shape=run.native.shape,
                                         objective_id=objective_id)
        run.details.update(native_loss=L, objective_id=objective_id,
                           reference_documents=512, scored_positions=sum(i[2] for i in identities))
        run.emit('KL_NATIVE', loss=L, gradient_norm=float(G.norm()), documents=512)
        if abs(L) <= KL_FLOOR:
            return run.finish('NUMERICAL_FLOOR', loss=L, summary=dict(full512=True))
        GQ = _matrix(run.call(project, G), run.native.shape)
        chi = float(torch.sum(GQ * GQ))
        _require(math.isfinite(chi) and chi >= 0, 'NONFINITE_CHI')
        run.details.update(chi=chi, projected_gradient_norm=float(GQ.norm()))
        if chi == 0:
            return run.finish('ZERO_GRADIENT', loss=L, summary=dict(full512=True))
        eta0 = L / chi
        _require(math.isfinite(eta0) and eta0 > 0, 'NONFINITE_ETA')
        run.details['eta0'] = eta0
        cache = {}
        for trial, scale in enumerate(SCALES, 1):
            candidate, ideal, actual, sha = run.materialize(-eta0 * scale * GQ)
            p = float(torch.sum(G * actual))
            _require(math.isfinite(p), 'NONFINITE_ACTUAL_DIRECTIONAL_PRODUCT')
            row = dict(trial=trial, scale=scale, eta=eta0*scale, candidate_sha256=sha,
                       ideal_delta_norm=float(ideal.norm()), actual_delta_norm=float(actual.norm()),
                       actual_p=p, actual_armijo_rhs=L+ARMIJO_C1*p)
            row['proposal'] = run.check_proposal(ideal, proposal_check)
            if sha == run.native_sha or p >= 0:
                row.update(accepted=False, reason='NO_RESOLVED_FP32_DESCENT')
            elif sha in cache:
                run.counters['candidate_cache_hits'] += 1
                row.update(accepted=False, reason='IDENTICAL_REJECTED_ENDPOINT',
                           score_reused_from_trial=cache[sha])
            else:
                run.counters['unique_scored_candidates'] += 1
                run.counters['objective_value_sweeps'] += 1
                value = run.call(objective, candidate, gradient=False)
                candidate_L, _, _, _ = _objective(value, gradient=False, shape=run.native.shape,
                                                 objective_id=objective_id, expected=identities)
                row.update(loss=candidate_L, actual_loss_decrease=L-candidate_L,
                           armijo=bool(candidate_L <= L+ARMIJO_C1*p and L-candidate_L > KL_FLOOR))
                if not row['armijo']:
                    row.update(accepted=False, reason='ARMIJO_OR_RESOLUTION')
                else:
                    run.counters['current_checks'] += 1
                    current = run.call(current_check, candidate, ideal, trial)
                    quality, invariant = _current(current)
                    row['current'] = current
                    if not (quality and invariant):
                        row.update(accepted=False, reason='CURRENT_OR_INVARIANT')
                    else:
                        hist_obs, hist = run.check_history(candidate, trial, history_check)
                        row.update(history_pass=hist['passed'], phi_history=hist['phi'])
                        if hist['passed']:
                            row.update(accepted=True, reason='ACCEPTED_KL')
                            run.trials.append(row); run.emit('TRIAL', **row)
                            return run.finish('ACCEPTED_KL', accepted=True, weight=candidate,
                                              ideal=ideal, actual=actual, loss=candidate_L,
                                              history=hist_obs, current=current,
                                              summary=dict(full512=True, reference_pass=None))
                        row.update(accepted=False, reason='ACTIVE_HISTORY_GUARD')
                cache[sha] = trial
            run.trials.append(row); run.emit('TRIAL', **row)
        return run.finish('NO_RESOLVED_STEP', loss=L, summary=dict(full512=True))
    except Exception as exc:
        run.fail(exc)


def optimize_decision(arm, WN, *, native_reference, raw_gradient, projected_gradient,
                      basis, reference_J, evaluate_reference, current_check,
                      proposal_check=None, native_history=None, history_J=None,
                      history_check=None, current_slack=None, current_J=None,
                      space_status='RESOLVED', event=None):
    """Same full-bank decision acceptance for LINE and MODES, one center only.

The parent supplies the source-bound derivative factors/basis/J once. The
controller never differentiates a trial or reads an official observer.
"""
    _require(arm in ('DEC_LINE', 'DEC_MODES_STEP', 'DEC_MODES_CUM'), 'DECISION_ARM_ALLOWLIST')
    run = _Run(arm, WN, native_history, event)
    try:
        native = _reference(native_reference)
        phiR, phiH = float(native.phi_reference), run.hist['phi']
        tau = decision.risk_tolerance(phiR, phiH)
        common = dict(full512=True, phi_reference_native=phiR, phi_reference_gain=0.,
                      tau_risk=tau, reference_pass=False)
        run.details.update(native_reference_endpoint=native.endpoint_identity,
                           reference_input=native.input_identity,
                           phi_history_native=phiH, psi_native=phiR+phiH,
                           basis=basis.receipt, space_status=space_status,
                           supplied_derivative_same_center_required=True)
        _require(space_status in ('RESOLVED', 'RANK_UNRESOLVED', 'REPAIR_SPACE_EMPTY'), 'SPACE_STATUS')
        if space_status != 'RESOLVED':
            return run.finish(space_status, loss=phiR+phiH, reference=native, summary=common)
        G = _matrix(raw_gradient, run.native.shape)
        GQ = _matrix(projected_gradient, run.native.shape)
        raw_norm, projected_norm = float(G.norm()), float(GQ.norm())
        reason = decision.no_direction_reason(phiR, phiH, native.mismatches,
                                               raw_norm, projected_norm,
                                               required_guards_valid=run.hist['passed'])
        run.details.update(raw_gradient_norm=raw_norm, projected_gradient_norm=projected_norm)
        if reason is not None:
            if reason == 'ZERO_RISK_NO_OP':
                common['reference_pass'] = True
            return run.finish(reason, loss=phiR+phiH, reference=native, summary=common)
        _require(0 <= basis.rank <= (1 if arm == 'DEC_LINE' else 5), 'BASIS_ARM_RANK')
        if basis.rank == 0:
            return run.finish('NO_BASIS_DIRECTION', loss=phiR+phiH, reference=native, summary=common)
        radius = (phiR+phiH) / projected_norm
        _require(math.isfinite(radius) and radius > 0, 'INVALID_PROPOSAL_RADIUS')
        J = _array(reference_J, (512, basis.rank))
        h = np.asarray([r['slack'] for r in run.hist['rows']], dtype=np.float64)
        _require(not run.hist['present'] or history_check is not None, 'ACTIVE_HISTORY_CALLBACK_REQUIRED')
        if history_J is None:
            _require(not run.hist['present'], 'ACTIVE_HISTORY_J_REQUIRED')
            JH = np.empty((0, basis.rank), dtype=np.float64)
        else:
            JH = _array(history_J, (len(h), basis.rank))
        mu = np.asarray([r['mu'] for r in native.rows], dtype=np.float64)
        run.counters['solver_calls'] += 1
        solution = run.call(solver.solve_coefficients, mu, J, radius,
                            history_slack=h if len(h) else None,
                            history_J=JH if len(h) else None,
                            current_slack=current_slack, current_J=current_J)
        run.details.update(radius=radius, solver=solution.receipt, solver_status=solution.status)
        run.emit('LOCAL_SOLVER', status=solution.status, radius=radius, receipt=solution.receipt)
        if solution.status != 'LOCAL_TWO_PHASE_SOLVED':
            _require(solution.status in ('ZERO_RADIUS_OR_BASIS',
                     'LOCAL_HISTORY_CONSTRAINT_INFEASIBLE', 'FINITE_SOLVER_UNRESOLVED'),
                     'UNKNOWN_SOLVER_STATUS')
            if solution.status == 'LOCAL_HISTORY_CONSTRAINT_INFEASIBLE':
                _require(solution.receipt.get('infeasibility_certificate', {}).get('verified') is True,
                         'LOCAL_INFEASIBILITY_CERTIFICATE_REQUIRED')
            return run.finish(solution.status, loss=phiR+phiH, reference=native, summary=common)
        coefficients = _array(solution.coefficients, (basis.rank,))
        plan = solver.history_backtracking_plan(coefficients, h, JH)
        run.details['history_backtracking_plan'] = plan
        seen = {}
        for trial, item in enumerate(plan, 1):
            scale = item['scale']
            row = dict(trial=trial, scale=scale, coefficients=(scale*coefficients).tolist(),
                       linear_history=item)
            if item['skip_before_model']:
                run.counters['linear_history_skips'] += 1
                row.update(accepted=False, reason=item['reason'])
                run.trials.append(row); run.emit('TRIAL_SKIPPED', **row)
                continue
            candidate, ideal, actual, sha = run.materialize(basis.reconstruct(scale*coefficients))
            row.update(candidate_sha256=sha, ideal_delta_norm=float(ideal.norm()),
                       actual_delta_norm=float(actual.norm()))
            row['proposal'] = run.check_proposal(ideal, proposal_check)
            if sha == run.native_sha:
                row.update(accepted=False, reason='FP32_NO_MOVE')
            elif sha in seen:
                run.counters['candidate_cache_hits'] += 1
                row.update(accepted=False, reason='IDENTICAL_REJECTED_ENDPOINT',
                           score_reused_from_trial=seen[sha])
            else:
                run.counters['unique_scored_candidates'] += 1
                run.counters['reference_candidate_sweeps'] += 1
                candidate_ref = _reference(run.call(evaluate_reference, candidate, trial))
                run.counters['current_checks'] += 1
                current = run.call(current_check, candidate, ideal, trial)
                quality, invariant = _current(current)
                hist_obs, hist = run.check_history(candidate, trial, history_check)
                accept = decision.acceptance(native, candidate_ref,
                    native_phi_history=phiH, candidate_phi_history=hist['phi'],
                    current_guard_pass=quality, invariant_pass=invariant,
                    history_guard_pass=hist['passed'])
                predicted_mu = mu + J @ (scale*coefficients)
                actual_mu = np.asarray([r['mu'] for r in candidate_ref.rows], dtype=np.float64)
                row.update(accepted=accept['accepted'], reason=accept['label'], acceptance=accept,
                           phi_reference=candidate_ref.phi_reference, phi_history=hist['phi'],
                           mismatches=candidate_ref.mismatches, current=current,
                           history_pass=hist['passed'], reference_endpoint=candidate_ref.endpoint_identity,
                           linear_prediction_max_error=float(np.max(np.abs(actual_mu-predicted_mu))),
                           linear_prediction_rms_error=float(np.sqrt(np.mean((actual_mu-predicted_mu)**2))))
                if accept['accepted']:
                    run.trials.append(row); run.emit('TRIAL', **row)
                    selected = dict(common, reference_pass=True,
                                    repaired_choices=accept['repaired_choices'],
                                    phi_reference_gain=phiR-candidate_ref.phi_reference,
                                    combined_risk_gain=accept['combined_risk_gain'])
                    return run.finish(accept['label'], accepted=True, weight=candidate, ideal=ideal,
                                      actual=actual, loss=candidate_ref.phi_reference+hist['phi'],
                                      reference=candidate_ref, history=hist_obs, current=current,
                                      summary=selected)
                seen[sha] = trial
            run.trials.append(row); run.emit('TRIAL', **row)
        return run.finish('FINITE_SEARCH_UNRESOLVED', loss=phiR+phiH, reference=native, summary=common)
    except Exception as exc:
        run.fail(exc)
