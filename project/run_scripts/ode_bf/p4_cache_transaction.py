"""Fail-closed AlphaEdit cache lifecycle for one P4 K8 batch."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import ODEBFContractError, canonical_hash
from .p4_waypoint_writer import P4_OUTER_STEPS


def _sha(value: str, label: str) -> str:
    if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise ODEBFContractError(f"P4 cache {label} identity differs")
    return value


@dataclass(slots=True)
class P4CacheTransaction:
    entry_sha256: str
    entry_width: int
    request_count: int
    _outer_rows: list[dict[str, Any]] = field(default_factory=list)
    _terminal: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _sha(self.entry_sha256, "entry")
        if self.entry_width < 0 or self.request_count <= 0:
            raise ODEBFContractError("P4 cache entry geometry differs")

    def observe_outer(
        self,
        *,
        outer_step_index: int,
        consumed_snapshot_sha256: str,
        consumed_snapshot_width: int,
        current_batch_append_count: int,
    ) -> None:
        if self._terminal is not None:
            raise ODEBFContractError("P4 cache transaction is terminal")
        if (
            outer_step_index != len(self._outer_rows)
            or outer_step_index >= P4_OUTER_STEPS
            or _sha(consumed_snapshot_sha256, "consumed") != self.entry_sha256
            or consumed_snapshot_width != self.entry_width
            or current_batch_append_count != 0
        ):
            raise ODEBFContractError("P4 cache K1-K8 snapshot contract differs")
        self._outer_rows.append(
            {
                "outer_step_index": outer_step_index,
                "consumed_snapshot_sha256": consumed_snapshot_sha256,
                "consumed_snapshot_width": consumed_snapshot_width,
                "current_batch_append_count": 0,
                "snapshot_read_only": True,
            }
        )

    def commit(self, *, exit_sha256: str, exit_width: int, append_count: int) -> Mapping[str, Any]:
        if self._terminal is not None or len(self._outer_rows) != P4_OUTER_STEPS:
            raise ODEBFContractError("P4 cache commit before complete K8 differs")
        _sha(exit_sha256, "exit")
        if append_count != 1 or exit_width != self.entry_width + self.request_count:
            raise ODEBFContractError("P4 cache post-K8 append contract differs")
        return self._finish(
            status="COMMITTED",
            exit_sha256=exit_sha256,
            exit_width=exit_width,
            append_count=append_count,
            rollback_count=0,
        )

    def abort(self, *, rollback_count: int) -> Mapping[str, Any]:
        if self._terminal is not None or rollback_count != 1:
            raise ODEBFContractError("P4 cache abort contract differs")
        return self._finish(
            status="ABORTED_ROLLED_BACK",
            exit_sha256=self.entry_sha256,
            exit_width=self.entry_width,
            append_count=0,
            rollback_count=rollback_count,
        )

    def _finish(
        self,
        *,
        status: str,
        exit_sha256: str,
        exit_width: int,
        append_count: int,
        rollback_count: int,
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p4-alphaedit-cache-transaction/v1",
            "status": status,
            "entry_sha256": self.entry_sha256,
            "entry_width": self.entry_width,
            "request_count": self.request_count,
            "outer_rows": list(self._outer_rows),
            "outer_snapshot_reuse_count": len(self._outer_rows),
            "current_batch_append_count_during_k1_k8": 0,
            "post_k8_append_count": append_count,
            "exit_sha256": exit_sha256,
            "exit_width": exit_width,
            "rollback_count": rollback_count,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        self._terminal = payload
        return payload


__all__ = ["P4CacheTransaction"]
