"""Fixed-horizon residual transport for Session 03 attribution.

This module contains only rewrite-side state.  Held-out evaluation payloads
are intentionally not representable by its API.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .contracts import ControllerConfig, EventReading, MethodContractError, ProposalBatch
from .controller import FullTransportSolution, solve_full_transport_qp
from .events import ControllerRequest
from .instrumentation import EditInstrumentation


CT_K = 4
CT_LAMBDAS = (0.25, 1.0 / 3.0, 0.5, 1.0)
CT_CUMULATIVE = (0.25, 0.5, 0.75, 1.0)
NAIVE_QUARTER_RESIDUAL_FRACTION = (3.0 / 4.0) ** 4


class CTArm(str, Enum):
    NATIVE_MEMIT = "native-memit"
    BF_ONESHOT_FULL = "bf-oneshot-full"
    BF_FROZEN_CT_K4 = "bf-frozen-ct-k4"
    ODE_REFRESH_CT_K4 = "ode-refresh-ct-k4"
    ODE_REFRESH_CT_K4_ES = "ode-refresh-ct-k4-es"


CT_ARM_ORDER = (
    CTArm.NATIVE_MEMIT,
    CTArm.BF_ONESHOT_FULL,
    CTArm.BF_FROZEN_CT_K4,
    CTArm.ODE_REFRESH_CT_K4,
    CTArm.ODE_REFRESH_CT_K4_ES,
)


def event_reading_to_dict(value: EventReading) -> dict[str, Any]:
    return {
        "hard_phi": value.hard_phi,
        "smooth_phi": value.smooth_phi,
        "mean_margin": math.fsum(value.context_margins)
        / len(value.context_margins),
        "mean_new": math.fsum(value.target_new_log_likelihoods)
        / len(value.target_new_log_likelihoods),
        "mean_old": math.fsum(value.target_old_log_likelihoods)
        / len(value.target_old_log_likelihoods),
        "context_margins": list(value.context_margins),
        "target_new_log_likelihoods": list(value.target_new_log_likelihoods),
        "target_old_log_likelihoods": list(value.target_old_log_likelihoods),
        "decision_deficits": list(value.decision_deficits),
        "event_mode": value.event_mode,
        "nfe": value.nfe,
    }


def assert_shared_direct_z_identities(
    source_identity: Mapping[str, Any],
    arm_identities: Mapping[CTArm, Mapping[str, Any]],
    *,
    global_n_z: int,
) -> None:
    """Require one case-level computation and byte-identical arm attachment."""

    if isinstance(global_n_z, bool) or global_n_z != 1:
        raise MethodContractError("CT-K4 case/model global N_z must equal one")
    if set(arm_identities) != set(CT_ARM_ORDER):
        raise MethodContractError("CT-K4 direct-z arm identity set differs")
    locked = dict(source_identity)
    required = {"tensor_sha256", "artifact_sha256", "artifact_size", "source_state_id"}
    if set(locked) != required:
        raise MethodContractError("CT-K4 direct-z identity schema differs")
    for arm, identity in arm_identities.items():
        if dict(identity) != locked:
            raise MethodContractError(
                f"CT-K4 shared direct-z identity differs for {arm.value}"
            )


def observe_first_hit(
    previous: int | None,
    *,
    completed_step: int,
    hit: bool,
    early_stop_enabled: bool,
) -> tuple[int | None, bool]:
    """Record all first hits while authorizing stop only for the ES arm."""

    if completed_step < 0 or not isinstance(hit, bool) or not isinstance(
        early_stop_enabled, bool
    ):
        raise MethodContractError("first-hit observation contract differs")
    observed = completed_step if previous is None and hit else previous
    return observed, bool(hit and early_stop_enabled)


@dataclass(frozen=True, slots=True)
class CTStep:
    position: int
    macro_time_before: float
    macro_time_after: float
    lambda_value: float
    cumulative_fraction: float | None
    source_state_id: str
    terminal_state_id: str
    direction_ids: tuple[str, ...]
    raw_coefficients: tuple[float, ...]
    corrector_coefficients: tuple[float, ...]
    applied_coefficients: tuple[float, ...]
    full_progress: float
    equality_residual: float
    raw_radius: float
    corrector_radius: float
    zero_slope_raw_completion: bool
    event_before: EventReading
    event_after: EventReading
    first_hit: bool
    subdivision_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "macro_time_before": self.macro_time_before,
            "macro_time_after": self.macro_time_after,
            "lambda": self.lambda_value,
            "cumulative_fraction": self.cumulative_fraction,
            "source_state_id": self.source_state_id,
            "terminal_state_id": self.terminal_state_id,
            "direction_ids": list(self.direction_ids),
            "raw_coefficients": list(self.raw_coefficients),
            "corrector_coefficients": list(self.corrector_coefficients),
            "applied_coefficients": list(self.applied_coefficients),
            "full_progress": self.full_progress,
            "equality_residual": self.equality_residual,
            "raw_radius": self.raw_radius,
            "corrector_radius": self.corrector_radius,
            "zero_slope_raw_completion": self.zero_slope_raw_completion,
            "event_before": event_reading_to_dict(self.event_before),
            "event_after": event_reading_to_dict(self.event_after),
            "first_hit": self.first_hit,
            "subdivision_count": self.subdivision_count,
        }


@dataclass(frozen=True, slots=True)
class CTArmResult:
    arm: CTArm
    status: str
    entry_state_id: str
    terminal_state_id: str
    steps: tuple[CTStep, ...]
    terminal_event: EventReading
    terminal_net_energy: Mapping[int, float]
    first_hit_step: int | None
    field_build_count: int
    subdivision_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "status": self.status,
            "entry_state_id": self.entry_state_id,
            "terminal_state_id": self.terminal_state_id,
            "steps": [step.to_dict() for step in self.steps],
            "terminal_event": event_reading_to_dict(self.terminal_event),
            "terminal_net_energy": {
                str(layer): value for layer, value in self.terminal_net_energy.items()
            },
            "first_hit_step": self.first_hit_step,
            "field_build_count": self.field_build_count,
            "subdivision_count": self.subdivision_count,
        }


def _event_payload(reading: EventReading) -> tuple[Any, ...]:
    return (
        reading.hard_phi,
        reading.smooth_phi,
        reading.context_margins,
        reading.target_new_log_likelihoods,
        reading.target_old_log_likelihoods,
        reading.decision_deficits,
        reading.event_mode,
    )


def assert_event_identity(
    trial: EventReading,
    committed: EventReading,
    *,
    atol: float,
    rtol: float,
) -> None:
    left = _event_payload(trial)
    right = _event_payload(committed)
    if left[-1] != right[-1]:
        raise MethodContractError("functional/committed event modes differ")
    for first, second in zip(left[:-1], right[:-1], strict=True):
        first_values = first if isinstance(first, tuple) else (first,)
        second_values = second if isinstance(second, tuple) else (second,)
        if len(first_values) != len(second_values) or any(
            not math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)
            for a, b in zip(first_values, second_values, strict=True)
        ):
            raise MethodContractError(
                "functional trial event differs from committed write"
            )


def _measure_event(
    backend: Any,
    request: ControllerRequest,
    instrumentation: EditInstrumentation,
    *,
    timed: bool = True,
) -> EventReading:
    timer = instrumentation.component("event") if timed else nullcontext()
    with timer, instrumentation.model_forward_scope("event"):
        reading = backend.event(request)
    if not instrumentation.tracks_model_forwards:
        for _ in range(reading.nfe):
            instrumentation.record_model_forward("event")
    return reading


def _transport_solution(
    batch: ProposalBatch,
    raw_coefficients: Sequence[float],
    *,
    omega: Mapping[int, float],
    denominators: Mapping[int, float],
    config: ControllerConfig,
    instrumentation: EditInstrumentation,
) -> FullTransportSolution:
    with instrumentation.component("qp"):
        return solve_full_transport_qp(
            layers=batch.layers,
            slopes=batch.slopes,
            raw_coefficients=raw_coefficients,
            omega={layer: omega[layer] for layer in batch.layers},
            denominators={layer: denominators[layer] for layer in batch.layers},
            config=config,
        )


def _build_field(
    backend: Any,
    frozen_target: Any,
    instrumentation: EditInstrumentation,
) -> tuple[ProposalBatch, tuple[float, ...]]:
    batch, raw = backend.build_transport_field(frozen_target)
    instrumentation.increment("N_field")
    instrumentation.increment("N_proposal_build")
    return batch, tuple(raw)


def _commit_trial(
    backend: Any,
    request: ControllerRequest,
    batch: ProposalBatch,
    coefficients: Sequence[float],
    *,
    config: ControllerConfig,
    instrumentation: EditInstrumentation,
) -> EventReading:
    locked = tuple(float(value) for value in coefficients)
    instrumentation.increment("N_trial")
    state_id = backend.current_state_id()
    with instrumentation.component("trial"), backend.trial(batch, locked) as trial:
        trial_event = _measure_event(
            backend, request, instrumentation, timed=False
        )
        if tuple(trial.applied_coefficients) != locked:
            raise MethodContractError("transport trial changed coefficients")
    if backend.current_state_id() != state_id:
        raise MethodContractError("transport functional trial changed model state")
    with instrumentation.component("commit_write"):
        applied = tuple(backend.commit(batch, locked))
    instrumentation.increment("N_write")
    instrumentation.increment("K_acc")
    if applied != locked:
        raise MethodContractError("transport commit changed coefficients")
    committed = _measure_event(backend, request, instrumentation)
    assert_event_identity(
        trial_event,
        committed,
        atol=config.functional_commit_atol,
        rtol=config.functional_commit_rtol,
    )
    return committed


def run_ct_arm(
    arm: CTArm,
    *,
    request: ControllerRequest,
    backend: Any,
    frozen_target: Any,
    denominators: Mapping[int, float],
    config: ControllerConfig,
    instrumentation: EditInstrumentation,
) -> CTArmResult:
    """Run one independent atomic attribution arm from its exact W0."""

    if arm not in CT_ARM_ORDER:
        raise MethodContractError("unknown CT-K4 arm")
    if not isinstance(request, ControllerRequest):
        raise MethodContractError("CT-K4 accepts rewrite-only requests")
    omega = {int(layer): 0.0 for layer in backend.layers}
    instrumentation.start_controller()
    entry = None
    try:
        with instrumentation.component("entry_checkpoint"):
            entry = backend.checkpoint()
        backend.prepare_event_target(frozen_target)
        before = _measure_event(backend, request, instrumentation)
        steps: list[CTStep] = []
        first_hit_step: int | None = None
        field_builds = 0

        if arm is CTArm.NATIVE_MEMIT:
            with instrumentation.component("native_proposal"):
                batch = backend.build_native_terminal(frozen_target)
            instrumentation.increment("N_proposal_build")
            instrumentation.increment("N_native_sweep")
            coefficients = tuple(1.0 for _ in batch.proposals)
            with instrumentation.component("commit_write"):
                applied = tuple(backend.commit(batch, coefficients))
            instrumentation.increment("N_write")
            instrumentation.increment("K_acc")
            after = _measure_event(backend, request, instrumentation)
            first_hit_step, _stop = observe_first_hit(
                first_hit_step,
                completed_step=1,
                hit=after.is_hit(config.event_tolerance),
                early_stop_enabled=False,
            )
            steps.append(
                CTStep(
                    0, 0.0, 1.0, 1.0, 1.0, batch.snapshot_id,
                    backend.current_state_id(), batch.direction_ids, coefficients,
                    coefficients, applied, 0.0, 0.0,
                    math.sqrt(len(coefficients)), math.sqrt(len(coefficients)),
                    False, before, after,
                    after.is_hit(config.event_tolerance),
                )
            )
        elif arm in {CTArm.BF_ONESHOT_FULL, CTArm.BF_FROZEN_CT_K4}:
            batch, raw = _build_field(backend, frozen_target, instrumentation)
            field_builds = 1
            solution = _transport_solution(
                batch, raw, omega=omega, denominators=denominators,
                config=config, instrumentation=instrumentation,
            )
            schedule = (
                ((1.0, 1.0),)
                if arm is CTArm.BF_ONESHOT_FULL
                else tuple(zip(CT_CUMULATIVE, CT_CUMULATIVE, strict=True))
            )
            for position, (lambda_value, cumulative) in enumerate(schedule):
                source_state = backend.current_state_id()
                if arm is CTArm.BF_ONESHOT_FULL:
                    after = _commit_trial(
                        backend, request, batch, solution.qp.coefficients,
                        config=config, instrumentation=instrumentation,
                    )
                    applied = solution.qp.coefficients
                else:
                    instrumentation.increment("N_trial")
                    with instrumentation.component("commit_write"):
                        applied = tuple(
                            backend.commit_frozen_cumulative(
                                entry, batch, solution.qp.coefficients, cumulative
                            )
                        )
                    instrumentation.increment("N_write")
                    instrumentation.increment("K_acc")
                    after = _measure_event(backend, request, instrumentation)
                hit = after.is_hit(config.event_tolerance)
                first_hit_step, _stop = observe_first_hit(
                    first_hit_step,
                    completed_step=position + 1,
                    hit=hit,
                    early_stop_enabled=False,
                )
                steps.append(
                    CTStep(
                        position, position / len(schedule),
                        (position + 1) / len(schedule), lambda_value, cumulative,
                        source_state, backend.current_state_id(), batch.direction_ids,
                        raw, solution.qp.coefficients, tuple(applied),
                        solution.full_progress, solution.qp.equality_residual,
                        math.sqrt(math.fsum(value * value for value in raw)),
                        solution.qp.coefficient_norm,
                        solution.zero_slope_raw_completion, before, after, hit,
                    )
                )
                before = after
        else:
            for position, lambda_value in enumerate(CT_LAMBDAS):
                if (
                    arm is CTArm.ODE_REFRESH_CT_K4_ES
                    and before.is_hit(config.event_tolerance)
                ):
                    first_hit_step = position
                    instrumentation.mark_first_hit()
                    break
                batch, raw = _build_field(backend, frozen_target, instrumentation)
                field_builds += 1
                solution = _transport_solution(
                    batch, raw, omega=omega, denominators=denominators,
                    config=config, instrumentation=instrumentation,
                )
                applied_coefficients = tuple(
                    lambda_value * value for value in solution.qp.coefficients
                )
                source_state = backend.current_state_id()
                after = _commit_trial(
                    backend, request, batch, applied_coefficients,
                    config=config, instrumentation=instrumentation,
                )
                hit = after.is_hit(config.event_tolerance)
                first_hit_step, stop = observe_first_hit(
                    first_hit_step,
                    completed_step=position + 1,
                    hit=hit,
                    early_stop_enabled=arm is CTArm.ODE_REFRESH_CT_K4_ES,
                )
                steps.append(
                    CTStep(
                        position, position / CT_K, (position + 1) / CT_K,
                        lambda_value, None, source_state,
                        backend.current_state_id(), batch.direction_ids, raw,
                        solution.qp.coefficients, applied_coefficients,
                        solution.full_progress, solution.qp.equality_residual,
                        math.sqrt(math.fsum(value * value for value in raw)),
                        solution.qp.coefficient_norm,
                        solution.zero_slope_raw_completion, before, after, hit,
                    )
                )
                before = after
                if stop:
                    instrumentation.mark_first_hit()
                    break

        if arm is CTArm.ODE_REFRESH_CT_K4 and len(steps) != CT_K:
            raise MethodContractError("primary refresh did not finish all K4 intervals")
        if arm is CTArm.BF_FROZEN_CT_K4 and len(steps) != CT_K:
            raise MethodContractError("frozen transport did not finish all K4 intervals")
        terminal = before if not steps else steps[-1].event_after
        with instrumentation.component("terminal_geometry"):
            energy = backend.terminal_net_energy(entry)
        status = (
            "event_hit_early"
            if arm is CTArm.ODE_REFRESH_CT_K4_ES and first_hit_step is not None
            else "completed_fixed_horizon"
        )
        return CTArmResult(
            arm=arm,
            status=status,
            entry_state_id=entry.state_id,
            terminal_state_id=backend.current_state_id(),
            steps=tuple(steps),
            terminal_event=terminal,
            terminal_net_energy=dict(energy),
            first_hit_step=first_hit_step,
            field_build_count=field_builds,
            subdivision_count=0,
        )
    except BaseException:
        if entry is not None:
            backend.restore(entry)
            backend.assert_checkpoint(entry)
        raise
    finally:
        instrumentation.stop_controller()
