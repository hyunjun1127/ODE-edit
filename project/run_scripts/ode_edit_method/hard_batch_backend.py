"""Genuine joint-batch MEMIT backend for the Session 03 hard-B10 gate.

The ordinary :mod:`easyedit_backend` path remains a singleton-compatible
backend.  This module opts into its additive request-panel support and keeps a
single snapshot, direct-z matrix, proposal/QP invocation, and transaction for
all ten requests.  It deliberately does not loop over singleton backends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from .contracts import EventReading, MethodContractError
from .easyedit_backend import EasyEditMemitBackend
from .events import ControllerRequest
from .oracle_absolute_event import (
    OracleAbsoluteMeanMarginCalibration,
    OracleAbsoluteMeanMarginTarget,
    calibrate_oracle_absolute_mean_margin_target,
    measure_differentiable_oracle_absolute_mean_margin_event,
    measure_oracle_absolute_mean_margin_event,
)


JOINT_BATCH_SIZE = 10
JOINT_EVENT_MODE = "joint-b10-uniform-request-mean-v3-diagnostic"


@dataclass(frozen=True, slots=True)
class JointDifferentiableEvent:
    """Joint event graph consumed by the existing one-backward hook."""

    reading: EventReading
    smooth_phi: torch.Tensor
    request_readings: tuple[EventReading, ...]

    def __post_init__(self) -> None:
        if (
            self.smooth_phi.ndim != 0
            or not self.smooth_phi.requires_grad
            or len(self.request_readings) != JOINT_BATCH_SIZE
        ):
            raise MethodContractError("joint differentiable event contract differs")


def _aggregate_request_readings(
    readings: Sequence[EventReading],
    *,
    smooth_value: float | None = None,
) -> EventReading:
    locked = tuple(readings)
    if len(locked) != JOINT_BATCH_SIZE:
        raise MethodContractError("joint event request count differs")
    if any(not item.has_absolute_likelihoods for item in locked):
        raise MethodContractError("joint event lacks absolute likelihoods")
    smooth = (
        math.fsum(item.smooth_phi for item in locked) / len(locked)
        if smooth_value is None
        else float(smooth_value)
    )
    return EventReading(
        hard_phi=max(item.hard_phi for item in locked),
        smooth_phi=smooth,
        context_margins=tuple(
            value for item in locked for value in item.context_margins
        ),
        nfe=sum(item.nfe for item in locked),
        target_new_log_likelihoods=tuple(
            value
            for item in locked
            for value in item.target_new_log_likelihoods
        ),
        target_old_log_likelihoods=tuple(
            value
            for item in locked
            for value in item.target_old_log_likelihoods
        ),
        event_mode=JOINT_EVENT_MODE,
        # Per-request V3 deficits remain in the request readings.  The joint
        # scalar is their uniform mean, not another decision threshold.
        decision_deficits=(),
    )


class JointHardBatchEasyEditBackend(EasyEditMemitBackend):
    """One backend transaction over exactly ten CounterFact requests."""

    def __init__(
        self,
        *,
        requests: Sequence[ControllerRequest],
        oracle_epsilon: float,
        **kwargs: Any,
    ) -> None:
        locked = tuple(requests)
        if (
            len(locked) != JOINT_BATCH_SIZE
            or len({item.case_id for item in locked}) != JOINT_BATCH_SIZE
        ):
            raise MethodContractError("hard-batch backend requires ten unique requests")
        if "request" in kwargs:
            raise MethodContractError("hard-batch primary request is derived from its panel")
        super().__init__(request=locked[0], requests=locked, **kwargs)
        self.oracle_epsilon = float(oracle_epsilon)
        if not math.isfinite(self.oracle_epsilon) or self.oracle_epsilon <= 0.0:
            raise MethodContractError("hard-batch oracle epsilon is invalid")
        self._joint_calibrations: tuple[
            OracleAbsoluteMeanMarginCalibration, ...
        ] | None = None
        self._joint_targets: tuple[OracleAbsoluteMeanMarginTarget, ...] | None = None
        self._cached_joint_entry: EventReading | None = None
        self._last_request_readings: tuple[EventReading, ...] | None = None
        self._request_event_history: list[dict[str, Any]] = []

    @property
    def joint_calibrations(
        self,
    ) -> tuple[OracleAbsoluteMeanMarginCalibration, ...]:
        if self._joint_calibrations is None:
            raise MethodContractError("joint oracle calibration is absent")
        return self._joint_calibrations

    @property
    def joint_targets(self) -> tuple[OracleAbsoluteMeanMarginTarget, ...]:
        if self._joint_targets is None:
            raise MethodContractError("joint oracle targets are absent")
        return self._joint_targets

    @property
    def request_event_history(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(row) for row in self._request_event_history)

    @property
    def last_request_event_readings(self) -> tuple[EventReading, ...]:
        if self._last_request_readings is None:
            raise MethodContractError("joint request event reading is absent")
        return self._last_request_readings

    def _record_request_readings(
        self,
        source: str,
        readings: Sequence[EventReading],
    ) -> None:
        locked = tuple(readings)
        if len(locked) != JOINT_BATCH_SIZE:
            raise MethodContractError("joint request reading count differs")
        for request, reading in zip(self.requests, locked, strict=True):
            self._request_event_history.append(
                {
                    "source": source,
                    "case_id": request.case_id,
                    "hard_phi": reading.hard_phi,
                    "smooth_phi": reading.smooth_phi,
                    "target_new_log_likelihoods": list(
                        reading.target_new_log_likelihoods
                    ),
                    "target_old_log_likelihoods": list(
                        reading.target_old_log_likelihoods
                    ),
                    "context_margins": list(reading.context_margins),
                    "decision_deficits": list(reading.decision_deficits),
                    "nfe": reading.nfe,
                }
            )

    def prepare_event_target(self, frozen_target: Any) -> None:
        target = self._assert_frozen_target(frozen_target)
        if self._joint_calibrations is not None:
            raise MethodContractError("joint oracle target was calibrated twice")
        if (
            self._entry_snapshot is None
            or self.current_state_id() != self._entry_snapshot.state_id
            or target.values.shape[1] != JOINT_BATCH_SIZE
        ):
            raise MethodContractError("joint oracle target is not at the batch entry")
        calibrations = []
        with self.instrumentation.component("event"):
            for index, request in enumerate(self.requests):
                column = SimpleNamespace(
                    values=target.values[:, index : index + 1],
                    z_layer=target.z_layer,
                )
                calibrations.append(
                    calibrate_oracle_absolute_mean_margin_target(
                        self.model,
                        self.tokenizer,
                        request,
                        self.contexts.templates,
                        direct_z=column,
                        bindings=self.bindings,
                        hparams=self.hparams,
                        tau=self.tau,
                        epsilon=self.oracle_epsilon,
                        entry_forward_scope=self.instrumentation.model_forward_scope(
                            "event"
                        ),
                        oracle_forward_scope=self.instrumentation.model_forward_scope(
                            None
                        ),
                    )
                )
        self._joint_calibrations = tuple(calibrations)
        self._joint_targets = tuple(item.target for item in calibrations)
        entries = tuple(item.entry_reading for item in calibrations)
        self._cached_joint_entry = _aggregate_request_readings(entries)
        self._last_request_readings = entries
        self._record_request_readings("oracle_entry", entries)
        self._record_request_readings(
            "oracle_shadow",
            tuple(item.oracle_shadow_reading for item in calibrations),
        )

    def _measure_event(self, request: ControllerRequest) -> EventReading:
        if request != self.request:
            raise MethodContractError("joint event primary request differs")
        if self._cached_joint_entry is not None:
            reading = self._cached_joint_entry
            self._cached_joint_entry = None
            return reading
        readings = tuple(
            measure_oracle_absolute_mean_margin_event(
                self.model,
                self.tokenizer,
                item,
                self.contexts.templates,
                target=target,
                tau=self.tau,
            )
            for item, target in zip(self.requests, self.joint_targets, strict=True)
        )
        self._last_request_readings = readings
        self._record_request_readings("event", readings)
        return _aggregate_request_readings(readings)

    def _measure_differentiable_event(self) -> JointDifferentiableEvent:
        events = tuple(
            measure_differentiable_oracle_absolute_mean_margin_event(
                self.model,
                self.tokenizer,
                item,
                self.contexts.templates,
                target=target,
                tau=self.tau,
            )
            for item, target in zip(self.requests, self.joint_targets, strict=True)
        )
        smooth = torch.stack(tuple(item.smooth_phi for item in events)).mean()
        readings = tuple(item.reading for item in events)
        self._last_request_readings = readings
        aggregate = _aggregate_request_readings(
            readings,
            smooth_value=float(smooth.detach().cpu()),
        )
        self._record_request_readings("field", readings)
        return JointDifferentiableEvent(
            reading=aggregate,
            smooth_phi=smooth,
            request_readings=readings,
        )

    def event(self, request: ControllerRequest) -> EventReading:
        if request != self.request:
            raise MethodContractError("joint event request differs")
        reading = self._measure_event(request)
        self._record_event("event", reading)
        return reading

    def assert_genuine_joint_batch(self, batch: Any) -> Mapping[str, Any]:
        """Fail closed unless one proposal carries rank-10 joint geometry."""

        if (
            len(self.requests) != JOINT_BATCH_SIZE
            or self._direct_z is None
            or self._direct_z.values.shape[1] != JOINT_BATCH_SIZE
            or tuple(self._direct_z.request_ids)
            != tuple(item.request_id for item in self.motivation_requests)
            or len(batch.proposals) != len(self.layers)
        ):
            raise MethodContractError("joint batch snapshot/direct-z geometry differs")
        ranks = []
        shapes = []
        for proposal in batch.proposals:
            direction = proposal.payload
            rank = int(direction.left.shape[1])
            if rank != JOINT_BATCH_SIZE or direction.right.shape[1] != rank:
                raise MethodContractError("joint factor rank differs from batch size")
            ranks.append(rank)
            shapes.append(
                {
                    "layer": proposal.layer,
                    "left": list(direction.left.shape),
                    "right": list(direction.right.shape),
                }
            )
        return {
            "batch_size": JOINT_BATCH_SIZE,
            "single_backend_transaction": True,
            "snapshot_request_count": JOINT_BATCH_SIZE,
            "direct_z_shape": list(self._direct_z.values.shape),
            "direct_z_receipt_count": len(self.joint_direct_z_receipts),
            "factor_ranks": ranks,
            "factor_shapes": shapes,
            "singleton_decomposition": False,
        }


__all__ = [
    "JOINT_BATCH_SIZE",
    "JOINT_EVENT_MODE",
    "JointDifferentiableEvent",
    "JointHardBatchEasyEditBackend",
]
