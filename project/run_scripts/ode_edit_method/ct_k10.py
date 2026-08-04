"""Fixed-horizon K10 confirmation arms for Session 03.

K4 execution is delegated to the frozen Session 03 implementation.  This
module adds only the entry-relative frozen K10 and current-state refreshed K10
paths, both with observe-only event readings.
"""

from __future__ import annotations

import math
from dataclasses import replace
from enum import Enum
from typing import Any, Mapping

from .contracts import ControllerConfig, MethodContractError
from .ct_k4 import (
    CTArm,
    CTArmResult,
    CTStep,
    _build_field,
    _commit_trial,
    _measure_event,
    _transport_solution,
    observe_first_hit,
    run_ct_arm,
)
from .events import ControllerRequest
from .instrumentation import EditInstrumentation


CT_K10 = 10
CT_K10_LAMBDAS = tuple(1.0 / (CT_K10 - position) for position in range(CT_K10))
CT_K10_CUMULATIVE = tuple(
    (position + 1) / CT_K10 for position in range(CT_K10)
)
NAIVE_TENTH_RESIDUAL_FRACTION = (1.0 - 1.0 / CT_K10) ** CT_K10


class CTK10Arm(str, Enum):
    NATIVE_MEMIT = "native-memit"
    BF_ONESHOT_FULL = "bf-oneshot-full"
    BF_FROZEN_CT_K10 = "bf-frozen-ct-k10"
    ODE_REFRESH_CT_K4 = "ode-refresh-ct-k4"
    ODE_REFRESH_CT_K10 = "ode-refresh-ct-k10"


CT_K10_ARM_ORDER = (
    CTK10Arm.NATIVE_MEMIT,
    CTK10Arm.BF_ONESHOT_FULL,
    CTK10Arm.BF_FROZEN_CT_K10,
    CTK10Arm.ODE_REFRESH_CT_K4,
    CTK10Arm.ODE_REFRESH_CT_K10,
)


_LEGACY_ARMS = {
    CTK10Arm.NATIVE_MEMIT: CTArm.NATIVE_MEMIT,
    CTK10Arm.BF_ONESHOT_FULL: CTArm.BF_ONESHOT_FULL,
    CTK10Arm.ODE_REFRESH_CT_K4: CTArm.ODE_REFRESH_CT_K4,
}


def assert_shared_direct_z_identities(
    source_identity: Mapping[str, Any],
    arm_identities: Mapping[CTK10Arm, Mapping[str, Any]],
    *,
    global_n_z: int,
) -> None:
    """Require one case/model direct-z and exact attachment to all five arms."""

    if isinstance(global_n_z, bool) or global_n_z != 1:
        raise MethodContractError("CT-K10 case/model global N_z must equal one")
    if set(arm_identities) != set(CT_K10_ARM_ORDER):
        raise MethodContractError("CT-K10 direct-z arm identity set differs")
    locked = dict(source_identity)
    required = {"tensor_sha256", "artifact_sha256", "artifact_size", "source_state_id"}
    if set(locked) != required:
        raise MethodContractError("CT-K10 direct-z identity schema differs")
    for arm, identity in arm_identities.items():
        if dict(identity) != locked:
            raise MethodContractError(
                f"CT-K10 shared direct-z identity differs for {arm.value}"
            )


def _run_k10_arm(
    arm: CTK10Arm,
    *,
    request: ControllerRequest,
    backend: Any,
    frozen_target: Any,
    denominators: Mapping[int, float],
    config: ControllerConfig,
    instrumentation: EditInstrumentation,
) -> CTArmResult:
    if arm not in {
        CTK10Arm.BF_FROZEN_CT_K10,
        CTK10Arm.ODE_REFRESH_CT_K10,
    }:
        raise MethodContractError("unknown K10 transport arm")
    if not isinstance(request, ControllerRequest):
        raise MethodContractError("CT-K10 accepts rewrite-only requests")

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

        if arm is CTK10Arm.BF_FROZEN_CT_K10:
            batch, raw = _build_field(backend, frozen_target, instrumentation)
            field_builds = 1
            solution = _transport_solution(
                batch,
                raw,
                omega=omega,
                denominators=denominators,
                config=config,
                instrumentation=instrumentation,
            )
            for position, cumulative in enumerate(CT_K10_CUMULATIVE):
                source_state = backend.current_state_id()
                instrumentation.increment("N_trial")
                with instrumentation.component("commit_write"):
                    applied = tuple(
                        backend.commit_frozen_cumulative(
                            entry,
                            batch,
                            solution.qp.coefficients,
                            cumulative,
                        )
                    )
                instrumentation.increment("N_write")
                instrumentation.increment("K_acc")
                after = _measure_event(backend, request, instrumentation)
                hit = after.is_hit(config.event_tolerance)
                first_hit_step, _ = observe_first_hit(
                    first_hit_step,
                    completed_step=position + 1,
                    hit=hit,
                    early_stop_enabled=False,
                )
                steps.append(
                    CTStep(
                        position=position,
                        macro_time_before=position / CT_K10,
                        macro_time_after=(position + 1) / CT_K10,
                        lambda_value=1.0 / CT_K10,
                        cumulative_fraction=cumulative,
                        source_state_id=source_state,
                        terminal_state_id=backend.current_state_id(),
                        direction_ids=batch.direction_ids,
                        raw_coefficients=tuple(raw),
                        corrector_coefficients=solution.qp.coefficients,
                        applied_coefficients=applied,
                        full_progress=solution.full_progress,
                        equality_residual=solution.qp.equality_residual,
                        raw_radius=math.sqrt(
                            math.fsum(value * value for value in raw)
                        ),
                        corrector_radius=solution.qp.coefficient_norm,
                        zero_slope_raw_completion=solution.zero_slope_raw_completion,
                        event_before=before,
                        event_after=after,
                        first_hit=hit,
                    )
                )
                before = after
        else:
            for position, lambda_value in enumerate(CT_K10_LAMBDAS):
                batch, raw = _build_field(backend, frozen_target, instrumentation)
                field_builds += 1
                solution = _transport_solution(
                    batch,
                    raw,
                    omega=omega,
                    denominators=denominators,
                    config=config,
                    instrumentation=instrumentation,
                )
                applied_coefficients = tuple(
                    lambda_value * value for value in solution.qp.coefficients
                )
                source_state = backend.current_state_id()
                after = _commit_trial(
                    backend,
                    request,
                    batch,
                    applied_coefficients,
                    config=config,
                    instrumentation=instrumentation,
                )
                hit = after.is_hit(config.event_tolerance)
                first_hit_step, _ = observe_first_hit(
                    first_hit_step,
                    completed_step=position + 1,
                    hit=hit,
                    early_stop_enabled=False,
                )
                steps.append(
                    CTStep(
                        position=position,
                        macro_time_before=position / CT_K10,
                        macro_time_after=(position + 1) / CT_K10,
                        lambda_value=lambda_value,
                        cumulative_fraction=None,
                        source_state_id=source_state,
                        terminal_state_id=backend.current_state_id(),
                        direction_ids=batch.direction_ids,
                        raw_coefficients=tuple(raw),
                        corrector_coefficients=solution.qp.coefficients,
                        applied_coefficients=applied_coefficients,
                        full_progress=solution.full_progress,
                        equality_residual=solution.qp.equality_residual,
                        raw_radius=math.sqrt(
                            math.fsum(value * value for value in raw)
                        ),
                        corrector_radius=solution.qp.coefficient_norm,
                        zero_slope_raw_completion=solution.zero_slope_raw_completion,
                        event_before=before,
                        event_after=after,
                        first_hit=hit,
                    )
                )
                before = after

        if len(steps) != CT_K10:
            raise MethodContractError("K10 transport did not finish all intervals")
        if steps[0].source_state_id == steps[0].terminal_state_id:
            raise MethodContractError("K10 first interval preserved target-weight state")
        with instrumentation.component("terminal_geometry"):
            energy = backend.terminal_net_energy(entry)
        return CTArmResult(
            arm=arm,
            status="completed_fixed_horizon",
            entry_state_id=entry.state_id,
            terminal_state_id=backend.current_state_id(),
            steps=tuple(steps),
            terminal_event=steps[-1].event_after,
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


def run_ct_k10_arm(
    arm: CTK10Arm,
    *,
    request: ControllerRequest,
    backend: Any,
    frozen_target: Any,
    denominators: Mapping[int, float],
    config: ControllerConfig,
    instrumentation: EditInstrumentation,
) -> CTArmResult:
    """Dispatch frozen K4 arms unchanged and add fixed-horizon K10 paths."""

    legacy = _LEGACY_ARMS.get(arm)
    if legacy is not None:
        result = run_ct_arm(
            legacy,
            request=request,
            backend=backend,
            frozen_target=frozen_target,
            denominators=denominators,
            config=config,
            instrumentation=instrumentation,
        )
        return replace(result, arm=arm)
    return _run_k10_arm(
        arm,
        request=request,
        backend=backend,
        frozen_target=frozen_target,
        denominators=denominators,
        config=config,
        instrumentation=instrumentation,
    )
