"""Exact-state cache for rejection retries.

An exact rollback leaves the rewrite field unchanged.  This cache makes that
reuse explicit and refuses to return any field or QP value when the current
state identity differs.  A contracted trust radius changes the QP key but not
the cached field, avoiding another rewrite backward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .contracts import EventReading, MethodContractError, ProposalBatch, canonical_hash
from .instrumentation import EditInstrumentation


@dataclass(slots=True)
class RejectedRetryCache:
    state_id: str | None = None
    field_key: str | None = None
    event: EventReading | None = None
    batch: ProposalBatch | None = None
    qp_values: dict[str, Any] = field(default_factory=dict)

    def clear(self) -> None:
        self.state_id = None
        self.field_key = None
        self.event = None
        self.batch = None
        self.qp_values.clear()

    def get_or_build_field(
        self,
        *,
        current_state_id: str,
        field_payload: Mapping[str, Any],
        build: Callable[[], tuple[EventReading, ProposalBatch]],
        instrumentation: EditInstrumentation,
    ) -> tuple[EventReading, ProposalBatch, bool]:
        if not current_state_id:
            raise MethodContractError("retry cache current state identity is empty")
        key = canonical_hash(dict(field_payload))
        if (
            self.state_id == current_state_id
            and self.field_key == key
            and self.event is not None
            and self.batch is not None
        ):
            return self.event, self.batch, True
        with instrumentation.component("field"):
            event, batch = build()
        instrumentation.increment("N_proposal_build")
        instrumentation.increment("N_field")
        if batch.snapshot_id != current_state_id:
            raise MethodContractError("built field differs from current retry state")
        self.state_id = current_state_id
        self.field_key = key
        self.event = event
        self.batch = batch
        self.qp_values.clear()
        return event, batch, False

    def get_or_solve_qp(
        self,
        *,
        current_state_id: str,
        qp_payload: Mapping[str, Any],
        solve: Callable[[], Any],
        instrumentation: EditInstrumentation,
    ) -> tuple[Any, bool]:
        if self.state_id != current_state_id or self.batch is None:
            raise MethodContractError("QP retry requested without exact cached field state")
        key = canonical_hash(dict(qp_payload))
        if key in self.qp_values:
            return self.qp_values[key], True
        with instrumentation.component("qp"):
            value = solve()
        self.qp_values[key] = value
        return value, False

    def assert_exact_rollback(
        self,
        current_state_id: str,
        instrumentation: EditInstrumentation,
    ) -> None:
        if self.state_id is None or current_state_id != self.state_id:
            self.clear()
            raise MethodContractError("rejected trial did not return to cached field state")
        instrumentation.increment("N_reject")

    def accept_and_invalidate(self, instrumentation: EditInstrumentation) -> None:
        instrumentation.increment("K_acc")
        self.clear()
