"""No-extra-forward absolute event-strength observability.

The controller remains free to define its decision event separately.  These
helpers only retain likelihoods already produced by an event forward and
derive raw-free diagnostics after the authoritative controller timer stops.
They never accept prompts, targets, rendered contexts, or evaluation payloads.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import Arm, EventReading, MethodContractError
from .oracle_absolute_event import (
    ORACLE_ABSOLUTE_MEAN_MARGIN_EVENT_MODE,
    OracleAbsoluteMeanMarginTarget,
)
from .oracle_event import ORACLE_MEAN_EVENT_MODE, OracleMeanEventTarget


OracleTarget = OracleMeanEventTarget | OracleAbsoluteMeanMarginTarget
_ORACLE_EVENT_MODES = frozenset(
    {ORACLE_MEAN_EVENT_MODE, ORACLE_ABSOLUTE_MEAN_MARGIN_EVENT_MODE}
)


_SUPPORTED_TRACE_ARMS = frozenset(
    {Arm.NATIVE_MEMIT, Arm.STATIC_SYNCHRONOUS, Arm.FULL_ODE_EDIT}
)
_FORBIDDEN_RAW_KEYS = frozenset(
    {
        "prompt",
        "subject",
        "target_new",
        "target_old",
        "raw_context",
        "context_templates",
        "templates",
        "evaluation",
        "generation",
    }
)
_ZERO_EVALUATION_ACCOUNTING_PARENTS = frozenset(
    {"component_wall_seconds", "component_gpu_seconds"}
)


def assert_raw_free(value: Any, label: str = "event-strength payload") -> None:
    """Reject raw fields, except the two exact zero-only accounting leaves."""

    _assert_raw_free(value, label=label, path=())


def _assert_raw_free(value: Any, *, label: str, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (str(key),)
            if key in _FORBIDDEN_RAW_KEYS:
                is_zero_accounting_leaf = (
                    key == "evaluation"
                    and len(child_path) >= 2
                    and child_path[-2] in _ZERO_EVALUATION_ACCOUNTING_PARENTS
                )
                if not is_zero_accounting_leaf:
                    raise MethodContractError(
                        f"{label} contains forbidden raw fields: {[key]}"
                    )
                if (
                    isinstance(child, bool)
                    or not isinstance(child, (int, float))
                    or not math.isfinite(float(child))
                    or float(child) != 0.0
                ):
                    raise MethodContractError(
                        f"{label} contains invalid zero-only evaluation accounting"
                    )
                continue
            _assert_raw_free(child, label=label, path=child_path)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_raw_free(child, label=label, path=path + ("[]",))


def event_reading_from_history(row: Mapping[str, Any]) -> EventReading:
    """Rebuild a validated numeric reading from a backend history row."""

    assert_raw_free(row, "event history")
    return EventReading(
        hard_phi=row.get("hard_phi"),
        smooth_phi=row.get("smooth_phi"),
        context_margins=tuple(row.get("context_margins", ())),
        nfe=row.get("nfe", 0),
        target_new_log_likelihoods=tuple(
            row.get("target_new_log_likelihoods", ())
        ),
        target_old_log_likelihoods=tuple(
            row.get("target_old_log_likelihoods", ())
        ),
        event_mode=str(
            row.get("event_mode", "legacy-worst-context-margin-shadow")
        ),
        decision_deficits=tuple(row.get("decision_deficits", ())),
    )


def absolute_event_metrics(
    reading: EventReading,
    entry: EventReading,
    *,
    tau: float,
    denominator_epsilon: float,
    oracle_target: OracleTarget | None = None,
) -> dict[str, Any]:
    """Derive means and entry-anchored strength from existing likelihoods."""

    if not reading.has_absolute_likelihoods or not entry.has_absolute_likelihoods:
        raise MethodContractError("absolute event likelihoods are absent")
    if not math.isfinite(tau) or tau <= 0.0:
        raise MethodContractError("absolute event tau is invalid")
    if not math.isfinite(denominator_epsilon) or denominator_epsilon <= 0.0:
        raise MethodContractError("absolute event denominator epsilon is invalid")
    new = reading.target_new_log_likelihoods
    old = reading.target_old_log_likelihoods
    margins = reading.context_margins
    entry_new = entry.target_new_log_likelihoods
    entry_old = entry.target_old_log_likelihoods
    count = len(margins)
    if (
        count <= 0
        or len(new) != count
        or len(old) != count
        or len(entry_new) != count
        or len(entry_old) != count
    ):
        raise MethodContractError("absolute event likelihood panels are misaligned")
    mean_new = math.fsum(new) / count
    mean_old = math.fsum(old) / count
    mean_margin = math.fsum(margins) / count
    entry_mean_new = math.fsum(entry_new) / count
    entry_mean_old = math.fsum(entry_old) / count
    if not math.isclose(
        mean_margin, mean_new - mean_old, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise MethodContractError("absolute event mean margin is not recomputable")
    hard = max(-value for value in margins)
    smooth = hard + tau * (
        math.log(
            math.fsum(math.exp(((-value) - hard) / tau) for value in margins)
        )
        - math.log(count)
    )
    if oracle_target is None:
        if reading.event_mode in _ORACLE_EVENT_MODES:
            raise MethodContractError("oracle event metrics require their edit target")
        if not math.isclose(
            hard, reading.hard_phi, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise MethodContractError("legacy event hard phi is not recomputable")
        if not math.isclose(
            smooth, reading.smooth_phi, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise MethodContractError("legacy event smooth phi is not recomputable")
    else:
        expected_mode = (
            ORACLE_ABSOLUTE_MEAN_MARGIN_EVENT_MODE
            if isinstance(oracle_target, OracleAbsoluteMeanMarginTarget)
            else ORACLE_MEAN_EVENT_MODE
        )
        if reading.event_mode != expected_mode:
            raise MethodContractError("oracle target received a non-oracle event")
        if oracle_target.context_count != count:
            raise MethodContractError("oracle target context count differs")
        relative_deficit = oracle_target.required_mean_margin - mean_margin
        absolute_deficit = oracle_target.required_mean_new - mean_new
        expected_deficits = (relative_deficit, absolute_deficit)
        expected_hard = max(expected_deficits)
        expected_smooth = expected_hard + tau * (
            math.log(
                math.fsum(
                    math.exp((value - expected_hard) / tau)
                    for value in expected_deficits
                )
            )
            - math.log(2)
        )
        if len(reading.decision_deficits) != len(expected_deficits) or any(
            not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
            for left, right in zip(
                reading.decision_deficits, expected_deficits, strict=True
            )
        ):
            raise MethodContractError("oracle decision deficits are not recomputable")
        if not math.isclose(
            expected_hard, reading.hard_phi, rel_tol=1e-12, abs_tol=1e-12
        ) or not math.isclose(
            expected_smooth, reading.smooth_phi, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise MethodContractError("oracle aggregate event is not recomputable")
    raw_denominator = entry_mean_old - entry_mean_new
    denominator = max(raw_denominator, denominator_epsilon)
    payload = {
        "target_new_log_likelihoods": list(new),
        "target_old_log_likelihoods": list(old),
        "uniform_context_weights": [1.0 / count for _ in range(count)],
        "context_count": count,
        "mean_new_log_likelihood": mean_new,
        "mean_old_log_likelihood": mean_old,
        "mean_margin": mean_margin,
        "min_margin": min(margins),
        "event_mode": reading.event_mode,
        "decision_hard_phi": reading.hard_phi,
        "decision_smooth_phi": reading.smooth_phi,
        "decision_deficits": list(reading.decision_deficits),
        "legacy_hard_phi": hard,
        "legacy_smooth_phi": smooth,
        "legacy_shadow_hit": None,
        "entry_anchor_q": (mean_new - entry_mean_new) / denominator,
        "entry_anchor_denominator_raw": raw_denominator,
        "entry_anchor_denominator": denominator,
        "entry_anchor_denominator_degenerate": (
            raw_denominator <= denominator_epsilon
        ),
        "entry_old_anchor_deficit": entry_mean_old - mean_new,
        "mean_old_change_from_entry": mean_old - entry_mean_old,
        "per_context_old_change_from_entry": [
            value - anchor
            for value, anchor in zip(old, entry_old, strict=True)
        ],
    }
    if oracle_target is not None:
        margin_denominator = oracle_target.oracle_mean_margin
        new_denominator = oracle_target.new_denominator
        payload.update(
            {
                "required_mean_margin": oracle_target.required_mean_margin,
                "required_mean_new_log_likelihood": oracle_target.required_mean_new,
                "relative_deficit": (
                    oracle_target.required_mean_margin - mean_margin
                ),
                "absolute_deficit": oracle_target.required_mean_new - mean_new,
                "q_margin": mean_margin / max(
                    margin_denominator, denominator_epsilon
                ),
                "q_margin_denominator": margin_denominator,
                "q_margin_denominator_degenerate": (
                    margin_denominator <= denominator_epsilon
                ),
                "q_new": (mean_new - oracle_target.entry_mean_new)
                / max(new_denominator, denominator_epsilon),
                "q_new_denominator": new_denominator,
                "q_new_denominator_degenerate": (
                    new_denominator <= denominator_epsilon
                ),
                "oracle_mean_margin": oracle_target.oracle_mean_margin,
                "oracle_mean_new_log_likelihood": oracle_target.oracle_mean_new,
            }
        )
    assert_raw_free(payload)
    return payload


def _observation(
    reading: EventReading,
    entry: EventReading,
    *,
    role: str,
    observation_index: int,
    attempt_index: int | None,
    accepted: bool | None,
    tau: float,
    denominator_epsilon: float,
    oracle_target: OracleTarget | None,
) -> dict[str, Any]:
    return {
        "role": role,
        "observation_index": observation_index,
        "attempt_index": attempt_index,
        "accepted": accepted,
        **absolute_event_metrics(
            reading,
            entry,
            tau=tau,
            denominator_epsilon=denominator_epsilon,
            oracle_target=oracle_target,
        ),
    }


def _alias_observation(
    observation: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    return {
        **dict(observation),
        "role": role,
        "measurement_reused_from_observation_index": observation[
            "observation_index"
        ],
        "measurement_reused_no_forward": True,
    }


def build_event_strength_trace(
    arm: Arm,
    result: Mapping[str, Any],
    event_history: Sequence[Mapping[str, Any]],
    *,
    tau: float,
    denominator_epsilon: float,
    event_tolerance: float,
    oracle_target: OracleTarget | None = None,
) -> dict[str, Any]:
    """Map existing decision reads to entry/trial/commit/terminal roles.

    This post-hoc projection cannot alter a verdict.  A rollback/terminal
    observation reuses an already measured exact-state reading and explicitly
    records that no forward was performed for the alias.
    """

    if arm not in _SUPPORTED_TRACE_ARMS:
        raise MethodContractError("event-strength trace received an unsupported arm")
    assert_raw_free(event_history, "absolute event history")
    decision_rows = [row for row in event_history if row.get("source") == "event"]
    field_rows = [row for row in event_history if row.get("source") == "field"]
    if not decision_rows:
        raise MethodContractError("event-strength trace has no entry event")
    entry_reading = event_reading_from_history(decision_rows[0])
    roles: list[tuple[str, int | None, bool | None]] = [("entry", None, None)]
    steps = result.get("steps")
    if not isinstance(steps, list):
        raise MethodContractError("event-strength result steps are absent")
    if arm is Arm.NATIVE_MEMIT:
        if len(steps) > 1:
            raise MethodContractError("Native event-strength trace has extra steps")
        if steps:
            roles.append(("accepted_committed", 0, True))
    else:
        for attempt_index, step in enumerate(steps):
            if not isinstance(step, Mapping) or step.get("accepted") not in {
                True,
                False,
            }:
                raise MethodContractError("event-strength step acceptance is invalid")
            accepted = bool(step["accepted"])
            roles.append(("trial", attempt_index, accepted))
            if accepted:
                roles.append(("accepted_committed", attempt_index, True))
    if len(roles) != len(decision_rows):
        raise MethodContractError(
            "event-strength decision events do not align with controller actions"
        )
    observations = [
        _observation(
            event_reading_from_history(row),
            entry_reading,
            role=role,
            observation_index=index,
            attempt_index=attempt_index,
            accepted=accepted,
            tau=tau,
            denominator_epsilon=denominator_epsilon,
            oracle_target=oracle_target,
        )
        for index, (row, (role, attempt_index, accepted)) in enumerate(
            zip(decision_rows, roles, strict=True)
        )
    ]
    field_observations = [
        _observation(
            event_reading_from_history(row),
            entry_reading,
            role="field_differentiable",
            observation_index=index,
            attempt_index=index,
            accepted=None,
            tau=tau,
            denominator_epsilon=denominator_epsilon,
            oracle_target=oracle_target,
        )
        for index, row in enumerate(field_rows)
    ]
    for row in (*observations, *field_observations):
        row["legacy_shadow_hit"] = row["legacy_hard_phi"] <= event_tolerance
    entry = observations[0]
    trials = [row for row in observations if row["role"] == "trial"]
    committed = [
        row for row in observations if row["role"] == "accepted_committed"
    ]
    omega_appended = result.get("omega_appended")
    if omega_appended is not True and omega_appended is not False:
        raise MethodContractError("event-strength result Omega flag is invalid")
    if omega_appended:
        terminal = _alias_observation(
            committed[-1] if committed else entry, role="terminal"
        )
        rollback = None
    else:
        terminal = _alias_observation(entry, role="terminal_rollback")
        rollback = _alias_observation(entry, role="rollback")
    last_trial = (
        _alias_observation(trials[-1], role="last_trial") if trials else None
    )
    status = result.get("status")
    primary_hit = status == "event_hit"
    if primary_hit != (terminal["decision_hard_phi"] <= event_tolerance):
        raise MethodContractError("event-strength terminal decision identity differs")
    legacy_terminal_hit = terminal["legacy_hard_phi"] <= event_tolerance
    terminal["legacy_shadow_hit"] = legacy_terminal_hit
    payload = {
        "schema_version": "ode-edit-event-strength-trace/v1",
        "decision_observations": observations,
        "field_observations": field_observations,
        "anchors": {
            "entry": _alias_observation(entry, role="entry_anchor"),
            "terminal": terminal,
            "last_trial": last_trial,
            "rollback": rollback,
        },
        "diagnostics": {
            "status": status,
            "primary_event_hit": primary_hit,
            "legacy_terminal_shadow_hit": legacy_terminal_hit,
            "failed_last_trial_mean_positive_worst_context_negative": bool(
                not omega_appended
                and last_trial is not None
                and last_trial["mean_margin"] >= 0.0
                and last_trial["min_margin"] < 0.0
            ),
            "relative_pass_with_mean_old_degradation_possible": bool(
                primary_hit
                and terminal["mean_old_log_likelihood"]
                < entry["mean_old_log_likelihood"]
            ),
            "native_absolute_nll_role": (
                "diagnostic-reference-only"
                if arm is Arm.NATIVE_MEMIT
                else "not-native"
            ),
            "under_write_claim_authorized": False,
            "additional_model_forwards": 0,
            "additional_backwards": 0,
        },
    }
    assert_raw_free(payload)
    return payload


__all__ = [
    "absolute_event_metrics",
    "assert_raw_free",
    "build_event_strength_trace",
    "event_reading_from_history",
]
