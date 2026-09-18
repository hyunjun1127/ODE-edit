"""ENFC finite, observer-injected weight-space correction controller.

No native fit, model, scheduler, history append, evaluator, or persistence is
implemented here. All endpoint tensors are CPU FP32; geometry and accumulated
corrections are FP64. Callbacks may use a separately guarded physical model.

The event sink is invoked BEFORE rejecting a probe, accepting an endpoint,
returning a fallback, or raising a technical error. Production callers must
durably save required evidence in that sink. A sink failure is a hard error.
Only deliberately typed TrialNumericalOverflow may reject a finite-weight
trial's nonfinite calculation; OOM, wrong state, and other callback exceptions
remain technical failures. No exception is converted to native success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import time
from typing import Any, Callable

import torch

ARMS = ("N4", "SCALE", "CA", "KL-P", "EN-S", "EN-F", "EN-COV", "EN-F4")
INVARIANT_ARMS = ("EN-F", "EN-COV", "EN-F4")
ARMIJO_C1 = 1e-4
BACKTRACK = 0.5
KL_FLOOR = 1e-6
MAIN_ROUNDS = 1
MAIN_TRIALS = 8
EN_F4_ROUNDS = 4
EN_F4_TRIALS = 6


class ControllerFailure(RuntimeError):
    """Technical error, never a successful native fallback."""
    def __init__(self, reason, *, events=None, counters=None):
        super().__init__(reason)
        self.reason = reason
        self.events = events or []
        self.counters = counters or {}


class EvidenceWriteFailure(ControllerFailure):
    pass


class TrialNumericalOverflow(FloatingPointError):
    """Caller has established finite trial weight but nonfinite forward value.

    Do NOT wrap teacher corruption, model-state errors, OOM, or CUDA assertion
    in this type. Only the bounded candidate calculation is rejectable.
    """


@dataclass
class Observation:
    loss: float
    gradient: torch.Tensor | None = None
    rows: Any = None
    metadata: dict = field(default_factory=dict)
    weight_sha256: str | None = None
    objective_id: str | None = None


@dataclass
class Check:
    passed: bool
    reason: str = "PASS"
    details: dict = field(default_factory=dict)


@dataclass
class Result:
    arm: str
    weight: torch.Tensor
    ideal_delta: torch.Tensor
    actual_delta: torch.Tensor
    loss: float | None
    stop_reason: str
    counters: dict
    trials: list[dict]
    events: list[dict]
    scalar_a: float | None = None

    def receipt(self):
        return {"arm": self.arm, "selected_weight_sha256": tensor_sha256(self.weight),
                "ideal_delta_sha256": tensor_sha256(self.ideal_delta),
                "actual_delta_sha256": tensor_sha256(self.actual_delta),
                "ideal_delta_norm": float(self.ideal_delta.norm()),
                "actual_delta_norm": float(self.actual_delta.norm()),
                "loss": self.loss, "stop_reason": self.stop_reason,
                "scalar_a": self.scalar_a, "scalar_s": None if self.scalar_a is None else 1-self.scalar_a,
                "counters": dict(self.counters), "trials": self.trials,
                "native_fallback": self.counters["accepted_rounds"] == 0,
                "numeric_constants": {"armijo_c1": ARMIJO_C1, "backtrack": BACKTRACK,
                                      "KL_floor": KL_FLOOR, "main_rounds": MAIN_ROUNDS,
                                      "main_trials": MAIN_TRIALS, "EN_F4_rounds": EN_F4_ROUNDS,
                                      "EN_F4_trials_per_round": EN_F4_TRIALS},
                "convergence_or_global_optimum_claim": False}


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(f"{tuple(value.shape)}|{value.dtype}|".encode("ascii")
                          + value.numpy().tobytes()).hexdigest()


def _cpu_tensor(value, *, dtype=None, shape=None):
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.ndim != 2:
        raise ValueError("CPU_RANK_TWO_TENSOR_REQUIRED")
    if dtype is not None and value.dtype != dtype:
        raise ValueError("WRONG_TENSOR_DTYPE")
    if not value.is_floating_point() or (shape is not None and tuple(value.shape) != tuple(shape)):
        raise ValueError("WRONG_TENSOR_SHAPE_OR_KIND")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("NONFINITE_TENSOR")
    return value.detach()


def _observation(value):
    if isinstance(value, Observation):
        return value
    if isinstance(value, tuple) and 2 <= len(value) <= 3:
        return Observation(float(value[0]), value[1], value[2] if len(value) == 3 else None)
    if isinstance(value, (int, float)):
        return Observation(float(value))
    if isinstance(value, dict):
        return Observation(**value)
    raise ValueError("OBJECTIVE_RETURN_SCHEMA")


def _check(value):
    if isinstance(value, Check):
        if not isinstance(value.passed, bool):
            raise ValueError("CHECK_PASSED_MUST_BE_BOOL")
        return value
    if isinstance(value, bool):
        return Check(value, "PASS" if value else "REJECTED")
    if isinstance(value, dict) and "passed" in value:
        if not isinstance(value["passed"], bool):
            raise ValueError("CHECK_PASSED_MUST_BE_BOOL")
        return Check(value["passed"], str(value.get("reason", "PASS" if value["passed"] else "REJECTED")),
                     value.get("details", {}))
    raise ValueError("CHECK_RETURN_SCHEMA")


class _Run:
    def __init__(self, arm, native, event):
        self.arm, self.event = arm, event
        self.events, self.trials = [], []
        self.counts = {key: 0 for key in (
            "gradient_rounds", "gradient_sweeps", "shared_initial_gradient_reuses",
            "objective_trial_sweeps", "attempted_trial_slots", "duplicate_trials",
            "guard_calls", "invariant_calls", "proposal_checks", "accepted_rounds",
            "rejected_trials", "nonfinite_weight_trials", "nonfinite_objective_trials")}
        self.counts.update(objective_seconds=0., guard_seconds=0., invariant_seconds=0.,
                           proposal_check_seconds=0., projection_seconds=0.)
        self.payload = {"native_weight": native}

    def emit(self, event, **fields):
        record = {"event_index": len(self.events), "event": event, "arm": self.arm, **fields}
        self.events.append(record)
        if self.event is not None:
            versions = {key: value._version for key, value in self.payload.items()
                        if isinstance(value, torch.Tensor)}
            try:
                self.event(record, dict(self.payload))
                if any(self.payload[key]._version != version for key, version in versions.items()):
                    raise RuntimeError("EVENT_SINK_MUTATED_BORROWED_TENSOR")
            except Exception as exc:
                raise EvidenceWriteFailure("EVENT_SINK_FAILED", events=self.events,
                                           counters=dict(self.counts)) from exc
        return record

    def fail(self, reason, exc=None):
        self.emit("technical_failure", reason=reason,
                  exception_type=None if exc is None else type(exc).__name__,
                  exception_text=None if exc is None else str(exc), counters=dict(self.counts))
        failure = ControllerFailure(reason, events=self.events, counters=dict(self.counts))
        if exc is not None:
            raise failure from exc
        raise failure

    def callback(self, name, function, weight, *args, gradient=None):
        before = tensor_sha256(weight)
        extra_hashes = [(arg, tensor_sha256(arg)) for arg in args if isinstance(arg, torch.Tensor)]
        started = time.perf_counter()
        self.emit(name + "_start", weight_sha256=before, gradient=gradient)
        try:
            value = function(weight, *args) if gradient is None else function(weight, gradient=gradient)
        except TrialNumericalOverflow:
            if tensor_sha256(weight) != before:
                self.fail("CALLBACK_MUTATED_WEIGHT")
            raise
        except EvidenceWriteFailure:
            raise
        except Exception as exc:
            self.fail(name.upper() + "_CALLBACK_FAILED", exc)
        finally:
            self.counts[name + "_seconds"] += time.perf_counter() - started
        if tensor_sha256(weight) != before:
            self.fail("CALLBACK_MUTATED_WEIGHT")
        if any(tensor_sha256(arg) != original for arg, original in extra_hashes):
            self.fail("CALLBACK_MUTATED_IDEAL_OR_ACTUAL_DELTA")
        return value


def optimize(arm: str, native_weight: torch.Tensor, *,
             objective: Callable | None = None, space: Any = None,
             guard: Callable | None = None, invariant: Callable | None = None,
             proposal_check: Callable | None = None, entry_weight: torch.Tensor | None = None,
             event: Callable | None = None, objective_id: str = "UNBOUND_OBJECTIVE",
             initial_observation: Observation | None = None,
             cov_resolution: float | None = None) -> Result:
    """Run one arm from one immutable native capsule; all proposals CPU-only.

    ``objective(weight, gradient=True/False)`` -> Observation or (loss,G,rows).
    ``guard(weight)`` -> Check/bool. Guards are always relative to the caller's
    own immutable W_N, not the current EN-F4 iterate.
    ``proposal_check(ideal_delta64)`` -> Check; failure is technical.
    ``invariant(weight, ideal_delta64, actual_delta64)`` -> Check; a false
    actual invariant rejects a finite trial. Required for EN-F/EN-COV/EN-F4.
    ``event(record, artifacts)`` must save before returning; artifacts are
    borrowed tensors and MUST NOT be mutated or retained as mutable aliases.

    Optional initial reuse requires exact native-weight SHA and objective_id.
    EN-COV's separately prelocked resolution (10*technical roundoff) is required;
    it is not silently assigned the KL floor. No other tolerances are knobs.
    """
    run = _Run(arm, native_weight, event)
    try:
        native = _cpu_tensor(native_weight, dtype=torch.float32).clone()
        if arm not in ARMS:
            raise ValueError("ARM_NOT_ALLOWED")
        if arm != "N4" and (objective is None or guard is None):
            raise ValueError("OBJECTIVE_AND_CURRENT_GUARD_REQUIRED")
        if arm not in ("N4", "SCALE") and space is None:
            raise ValueError("SPACE_REQUIRED")
        if arm in INVARIANT_ARMS and (proposal_check is None or invariant is None):
            raise ValueError("PROPOSAL_NULL_AND_ACTUAL_INVARIANT_CHECK_REQUIRED")
        if arm == "EN-COV":
            if cov_resolution is None or not math.isfinite(cov_resolution) or cov_resolution < 0:
                raise ValueError("PRELOCKED_COVARIANCE_RESOLUTION_REQUIRED")
            floor = float(cov_resolution)
        else:
            if cov_resolution is not None:
                raise ValueError("COVARIANCE_FLOOR_FOR_WRONG_OBJECTIVE")
            floor = KL_FLOOR
        if arm == "SCALE":
            entry = _cpu_tensor(entry_weight, dtype=torch.float32, shape=native.shape)
            native_delta = native.double() - entry.double()
        else:
            native_delta = None
    except Exception as exc:
        run.fail("INVALID_CONTROLLER_INPUT", exc)
    native64 = native.double()
    current = native.clone()
    ideal = torch.zeros_like(native64)
    scalar_a = 0. if arm == "SCALE" else None
    loss = None
    rounds, trial_cap = (EN_F4_ROUNDS, EN_F4_TRIALS) if arm == "EN-F4" else (MAIN_ROUNDS, MAIN_TRIALS)
    run.payload = {"native_weight": native, "current_weight": current, "ideal_delta": ideal}
    run.emit("controller_start", native_weight_sha256=tensor_sha256(native),
             objective_id=objective_id, rounds=0 if arm == "N4" else rounds,
             trials_per_round=0 if arm == "N4" else trial_cap,
             resolution_floor=floor, checkpoint_or_history_action=False)

    def finish(reason):
        actual = current.double() - native64
        run.payload = {"native_weight": native, "selected_weight": current,
                       "ideal_delta": ideal, "actual_delta": actual}
        if not bool(torch.isfinite(current).all()):
            run.fail("NONFINITE_SELECTED_ENDPOINT")
        run.emit("selected_endpoint", stop_reason=reason, loss=loss,
                 selected_sha256=tensor_sha256(current), ideal_sha256=tensor_sha256(ideal),
                 actual_sha256=tensor_sha256(actual), counters=dict(run.counts),
                 native_fallback=run.counts["accepted_rounds"] == 0)
        return Result(arm, current.clone(), ideal.clone(), actual, loss, reason,
                      dict(run.counts), run.trials, run.events, scalar_a)

    if arm == "N4":
        return finish("NATIVE_ENDPOINT")
    if arm != "SCALE":
        status = getattr(space, "status", "RESOLVED")
        if status == "RANK_UNRESOLVED":
            return finish("RANK_UNRESOLVED")
        if status == "REPAIR_SPACE_EMPTY" or getattr(space, "dimension", None) == 0:
            return finish("REPAIR_SPACE_EMPTY")
        if status != "RESOLVED":
            run.fail("INVALID_SPACE_STATUS")
    cache = {}
    for round_index in range(rounds):
        run.counts["gradient_rounds"] += 1
        current_hash = tensor_sha256(current)
        run.payload = {"native_weight": native, "current_weight": current, "ideal_delta": ideal}
        try:
            if round_index == 0 and initial_observation is not None:
                observed = _observation(initial_observation)
                if (objective_id == "UNBOUND_OBJECTIVE" or observed.objective_id != objective_id
                        or observed.weight_sha256 != current_hash):
                    raise ValueError("SHARED_GRADIENT_IDENTITY_MISMATCH")
                run.counts["shared_initial_gradient_reuses"] += 1
                run.emit("shared_initial_observation", weight_sha256=current_hash,
                         objective_id=objective_id)
            else:
                run.counts["gradient_sweeps"] += 1
                observed = _observation(run.callback("objective", objective, current, gradient=True))
            run.payload["observed_gradient"] = observed.gradient
            run.payload["objective_rows"] = observed.rows
            run.emit("raw_gradient_observation", round=round_index,
                     loss_repr=repr(observed.loss), observation_metadata=observed.metadata)
            if observed.weight_sha256 is not None and observed.weight_sha256 != current_hash:
                raise ValueError("GRADIENT_STATE_MISMATCH")
            if observed.objective_id is not None and observed.objective_id != objective_id:
                raise ValueError("OBJECTIVE_ID_MISMATCH")
            gradient = _cpu_tensor(observed.gradient, shape=native.shape).double().clone()
            new_loss = float(observed.loss)
            if not math.isfinite(new_loss):
                raise ValueError("NONFINITE_ACCEPTED_STATE_LOSS")
            if new_loss < -floor:
                raise ValueError("NEGATIVE_OBJECTIVE_BELOW_FLOOR")
        except EvidenceWriteFailure:
            raise
        except ControllerFailure:
            raise
        except Exception as exc:
            run.fail("INVALID_CURRENT_OBJECTIVE_OR_GRADIENT", exc)
        run.payload["gradient"] = gradient
        run.emit("gradient_observed", round=round_index, loss=new_loss,
                 previous_accepted_loss=loss, repeat_loss_delta=None if loss is None else new_loss-loss,
                 gradient_sha256=tensor_sha256(gradient), gradient_norm=float(gradient.norm()),
                 weight_sha256=current_hash, metadata=observed.metadata)
        loss = new_loss
        cache[current_hash] = {"loss": loss, "guard": Check(True, "ALREADY_ACCEPTED"),
                               "invariant": Check(True, "ALREADY_ACCEPTED")}
        if abs(loss) <= floor:
            return finish("NUMERICAL_FLOOR")
        try:
            if arm == "SCALE":
                scalar_gradient = -float(torch.sum(gradient * native_delta))
                chi = scalar_gradient * scalar_gradient
                projected = None
                if not math.isfinite(scalar_gradient) or not math.isfinite(chi):
                    raise ValueError("NONFINITE_SCALAR_GRADIENT_OR_CHI")
                if chi == 0.:
                    return finish("ZERO_GRADIENT")
                if scalar_gradient > 0. and scalar_a == 0.:
                    return finish("NO_FEASIBLE_DESCENT")
            else:
                started = time.perf_counter()
                projected = space.project(gradient)
                run.payload["projected_gradient"] = projected
                projected = _cpu_tensor(projected, shape=native.shape).double()
                run.counts["projection_seconds"] += time.perf_counter() - started
                chi = float(torch.sum(projected * projected))
                scalar_gradient = None
                if not math.isfinite(chi):
                    raise ValueError("NONFINITE_PROJECTED_CHI")
        except ControllerFailure:
            raise
        except Exception as exc:
            run.fail("INVALID_PROJECTED_DIRECTION", exc)
        run.payload["projected_gradient"] = projected
        run.emit("direction", round=round_index, chi=chi, scalar_gradient=scalar_gradient,
                 projected_sha256=None if projected is None else tensor_sha256(projected))
        if chi == 0.:
            return finish("ZERO_GRADIENT")
        eta0 = loss / chi
        if not math.isfinite(eta0):
            return finish("STEP_SCALE_UNRESOLVED")
        accepted = False
        for trial_index in range(trial_cap):
            run.counts["attempted_trial_slots"] += 1
            eta = eta0 * BACKTRACK ** trial_index
            if arm == "SCALE":
                trial_a = min(1., max(0., scalar_a - eta * scalar_gradient))
                proposal = -trial_a * native_delta
                nominal = scalar_gradient * (trial_a - scalar_a)
            else:
                trial_a = None
                proposal = ideal - eta * projected
                nominal = -eta * chi
            candidate = (native64 + proposal).float()
            actual = candidate.double() - native64
            step = candidate.double() - current.double()
            p_actual = float(torch.sum(gradient * step))
            finite_weight = bool(torch.isfinite(candidate).all())
            finite_ideal = bool(torch.isfinite(proposal).all())
            record = {"round": round_index, "trial": trial_index, "eta0": eta0, "eta": eta,
                      "nominal_prediction": nominal, "p_actual": p_actual if math.isfinite(p_actual) else None,
                      "scalar_a": trial_a, "finite_weight": finite_weight, "finite_ideal": finite_ideal,
                      "weight_sha256": tensor_sha256(candidate),
                      "ideal_sha256": tensor_sha256(proposal), "actual_sha256": tensor_sha256(actual),
                      "actual_step_norm": float(step.norm()) if finite_weight else None,
                      "ideal_norm": float(proposal.norm()) if finite_ideal else None,
                      "actual_norm": float(actual.norm()) if finite_weight else None,
                      "accepted": False, "reused_weight_score": False}
            run.trials.append(record)
            run.payload.update(candidate_weight=candidate, proposed_ideal_delta=proposal,
                               proposed_actual_delta=actual, actual_step=step)
            run.emit("trial_materialized", **record)

            def reject(reason):
                record["reason"] = reason
                run.counts["rejected_trials"] += 1
                run.emit("trial_rejected", **record)

            if not finite_weight or not finite_ideal:
                run.counts["nonfinite_weight_trials"] += 1
                reject("NONFINITE_TRIAL_WEIGHT_OR_IDEAL")
                continue
            if not math.isfinite(p_actual):
                run.fail("NONFINITE_ACTUAL_PREDICTION")
            if proposal_check is not None:
                run.counts["proposal_checks"] += 1
                started = time.perf_counter()
                try:
                    proposal_hash = tensor_sha256(proposal)
                    check = _check(proposal_check(proposal))
                    if tensor_sha256(proposal) != proposal_hash:
                        run.fail("PROPOSAL_CHECK_MUTATED_IDEAL_DELTA")
                except ControllerFailure:
                    raise
                except Exception as exc:
                    run.fail("PROPOSAL_CHECK_CALLBACK_FAILED", exc)
                finally:
                    run.counts["proposal_check_seconds"] += time.perf_counter() - started
                run.emit("proposal_geometry_checked", round=round_index, trial=trial_index,
                         passed=check.passed, reason=check.reason, details=check.details)
                if not check.passed:
                    run.fail("PROPOSED_MATHEMATICAL_NULLSPACE_FAILURE")
            if torch.equal(candidate, current):
                reject("FP32_NO_MOVE")
                return finish("NO_RESOLVED_STEP")
            if p_actual >= 0.:
                reject("ACTUAL_STEP_NOT_DESCENT")
                continue
            candidate_hash = record["weight_sha256"]
            cached = cache.get(candidate_hash)
            if cached is None:
                run.counts["objective_trial_sweeps"] += 1
                try:
                    obs = _observation(run.callback("objective", objective, candidate, gradient=False))
                    run.payload["trial_objective_rows"] = obs.rows
                    trial_loss = float(obs.loss)
                    run.emit("raw_trial_observation", round=round_index, trial=trial_index,
                             weight_sha256=candidate_hash, loss_repr=repr(trial_loss),
                             observation_metadata=obs.metadata)
                    if obs.weight_sha256 is not None and obs.weight_sha256 != candidate_hash:
                        run.fail("TRIAL_OBJECTIVE_STATE_MISMATCH")
                    if obs.objective_id is not None and obs.objective_id != objective_id:
                        run.fail("TRIAL_OBJECTIVE_ID_MISMATCH")
                except TrialNumericalOverflow as exc:
                    run.counts["nonfinite_objective_trials"] += 1
                    record["exception_type"], record["exception_text"] = type(exc).__name__, str(exc)
                    reject("FINITE_WEIGHT_NONFINITE_OBJECTIVE")
                    cache[candidate_hash] = {"nonfinite": True}
                    continue
                except (ControllerFailure, EvidenceWriteFailure):
                    raise
                except Exception as exc:
                    run.fail("INVALID_TRIAL_OBJECTIVE_SCHEMA", exc)
                if not math.isfinite(trial_loss):
                    run.counts["nonfinite_objective_trials"] += 1
                    record["nonfinite_loss_repr"] = repr(trial_loss)
                    reject("FINITE_WEIGHT_NONFINITE_OBJECTIVE")
                    cache[candidate_hash] = {"nonfinite": True}
                    continue
                if trial_loss < -floor:
                    record["loss"] = trial_loss
                    run.fail("TRIAL_NEGATIVE_OBJECTIVE_BELOW_FLOOR")
                cached = {"loss": trial_loss}
                cache[candidate_hash] = cached
            else:
                run.counts["duplicate_trials"] += 1
                record["reused_weight_score"] = True
                if cached.get("nonfinite"):
                    reject("CACHED_NONFINITE_OBJECTIVE")
                    continue
                trial_loss = cached["loss"]
            record.update(loss=trial_loss, current_loss=loss,
                          decrease=loss-trial_loss, armijo_bound=loss+ARMIJO_C1*p_actual)
            run.emit("trial_objective_observed", **record)
            if not trial_loss <= loss + ARMIJO_C1 * p_actual:
                reject("ARMIJO_FAILED")
                continue
            if not loss - trial_loss > floor:
                reject("DECREASE_NOT_RESOLVED")
                continue
            if "guard" not in cached:
                run.counts["guard_calls"] += 1
                try:
                    cached["guard"] = _check(run.callback("guard", guard, candidate))
                except ControllerFailure:
                    raise
                except Exception as exc:
                    run.fail("INVALID_GUARD_RESULT", exc)
            guard_result = cached["guard"]
            run.emit("quality_guard_checked", round=round_index, trial=trial_index,
                     passed=guard_result.passed, reason=guard_result.reason, details=guard_result.details)
            if not guard_result.passed:
                record["guard_reason"] = guard_result.reason
                reject("QUALITY_GUARD_FAILED")
                continue
            if invariant is not None:
                if "invariant" not in cached:
                    run.counts["invariant_calls"] += 1
                    try:
                        cached["invariant"] = _check(run.callback("invariant", invariant, candidate,
                                                                 proposal, actual))
                    except ControllerFailure:
                        raise
                    except Exception as exc:
                        run.fail("INVALID_INVARIANT_RESULT", exc)
                inv_result = cached["invariant"]
                run.emit("actual_invariant_checked", round=round_index, trial=trial_index,
                         passed=inv_result.passed, reason=inv_result.reason, details=inv_result.details)
                if not inv_result.passed:
                    record["invariant_reason"] = inv_result.reason
                    reject("ACTUAL_INVARIANT_FAILED")
                    continue
            record.update(accepted=True, reason="FIRST_RESOLVED_ACCEPTED")
            run.emit("trial_accepted", **record)
            current, ideal, loss, scalar_a = candidate, proposal, trial_loss, trial_a
            run.counts["accepted_rounds"] += 1
            accepted = True
            break
        if not accepted:
            return finish("TRIAL_BUDGET_EXHAUSTED_NO_NEW_ACCEPT")
    return finish("GRADIENT_BUDGET_EXHAUSTED")
