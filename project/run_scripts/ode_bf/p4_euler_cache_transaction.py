"""K8 cache/weight transaction ledger for the Euler P4 causal arms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import ODEBFContractError, canonical_hash
from .p4_euler_integrator import P4_EULER_INSTRUCTION_ID


P4_EULER_OUTER_STEPS = 8


def _sha(value: str, label: str) -> str:
    if len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise ODEBFContractError(f"P4 Euler {label} identity differs")
    return value


def verify_k_target_bridge(
    *,
    outer_step_index: int,
    provided_target_sha256: str,
    consumed_target_sha256: str,
    cache_template_sha256: str,
    bridge_namespace: str,
) -> Mapping[str, Any]:
    if (
        isinstance(outer_step_index, bool)
        or not isinstance(outer_step_index, int)
        or outer_step_index < 0
        or outer_step_index >= P4_EULER_OUTER_STEPS
        or not bridge_namespace
        or f"K{outer_step_index + 1}" not in bridge_namespace
    ):
        raise ODEBFContractError("P4 Euler K-specific target bridge differs")
    provided = _sha(provided_target_sha256, "provided target")
    consumed = _sha(consumed_target_sha256, "consumed target")
    template = _sha(cache_template_sha256, "cache template")
    if provided != consumed:
        raise ODEBFContractError("P4 Euler provided/consumed target SHA differs")
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-k-target-bridge/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "outer_step_index": outer_step_index,
        "provided_target_sha256": provided,
        "consumed_target_sha256": consumed,
        "cache_template_sha256": template,
        "bridge_namespace": bridge_namespace,
        "stale_bridge_reuse_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


@dataclass(slots=True)
class P4EulerCacheTransaction:
    """Fail closed over C0 reuse, K1--K7 restore, K8 append, and abort."""

    entry_cache_sha256: str
    entry_cache_width: int
    entry_weight_sha256: str
    request_count: int
    _rows: list[dict[str, Any]] = field(default_factory=list)
    _terminal: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _sha(self.entry_cache_sha256, "entry cache")
        _sha(self.entry_weight_sha256, "entry weight")
        if self.entry_cache_width < 0 or self.request_count <= 0:
            raise ODEBFContractError("P4 Euler cache entry geometry differs")

    def observe_writer_step(
        self,
        *,
        outer_step_index: int,
        writer_entry_cache_sha256: str,
        writer_exit_candidate_sha256: str,
        restored_cache_sha256: str | None,
        bridge: Mapping[str, Any],
    ) -> None:
        if self._terminal is not None or outer_step_index != len(self._rows):
            raise ODEBFContractError("P4 Euler cache transaction order differs")
        if outer_step_index >= P4_EULER_OUTER_STEPS:
            raise ODEBFContractError("P4 Euler cache outer index differs")
        if _sha(writer_entry_cache_sha256, "writer entry cache") != self.entry_cache_sha256:
            raise ODEBFContractError("P4 Euler writer did not consume C0")
        candidate = _sha(writer_exit_candidate_sha256, "writer cache candidate")
        if bridge.get("outer_step_index") != outer_step_index:
            raise ODEBFContractError("P4 Euler cache bridge K differs")
        restored: str | None
        if outer_step_index < P4_EULER_OUTER_STEPS - 1:
            if restored_cache_sha256 is None:
                raise ODEBFContractError("P4 Euler K1-K7 cache restore is absent")
            restored = _sha(restored_cache_sha256, "restored cache")
            if restored != self.entry_cache_sha256:
                raise ODEBFContractError("P4 Euler K1-K7 cache restore differs")
        else:
            if restored_cache_sha256 is not None:
                raise ODEBFContractError("P4 Euler K8 cache must not restore C0")
            restored = None
        self._rows.append(
            {
                "outer_step_index": outer_step_index,
                "writer_entry_cache_sha256": self.entry_cache_sha256,
                "writer_exit_candidate_sha256": candidate,
                "post_writer_restored_cache_sha256": restored,
                "committed_current_batch_append_count": 0,
                "bridge_identity_sha256": bridge.get("identity_sha256"),
                "provided_target_sha256": bridge.get("provided_target_sha256"),
                "consumed_target_sha256": bridge.get("consumed_target_sha256"),
            }
        )

    def commit(
        self,
        *,
        final_cache_sha256: str,
        final_cache_width: int,
        final_weight_sha256: str,
        append_event_count: int,
    ) -> Mapping[str, Any]:
        if self._terminal is not None or len(self._rows) != P4_EULER_OUTER_STEPS:
            raise ODEBFContractError("P4 Euler cache commit before complete K8")
        if (
            append_event_count != 1
            or final_cache_width != self.entry_cache_width + self.request_count
        ):
            raise ODEBFContractError("P4 Euler final cache append differs")
        final_cache = _sha(final_cache_sha256, "final cache")
        if final_cache != self._rows[-1]["writer_exit_candidate_sha256"]:
            raise ODEBFContractError("P4 Euler K8 committed cache candidate differs")
        return self._finish(
            status="COMMITTED",
            final_cache_sha256=final_cache,
            final_cache_width=final_cache_width,
            final_weight_sha256=_sha(final_weight_sha256, "final weight"),
            final_append_event_count=append_event_count,
            weight_rollback_count=0,
            cache_rollback_count=0,
        )

    def abort(
        self,
        *,
        restored_cache_sha256: str,
        restored_weight_sha256: str,
    ) -> Mapping[str, Any]:
        if self._terminal is not None:
            raise ODEBFContractError("P4 Euler cache transaction is terminal")
        if (
            _sha(restored_cache_sha256, "abort cache") != self.entry_cache_sha256
            or _sha(restored_weight_sha256, "abort weight") != self.entry_weight_sha256
        ):
            raise ODEBFContractError("P4 Euler abort rollback differs")
        return self._finish(
            status="ABORTED_ROLLED_BACK",
            final_cache_sha256=self.entry_cache_sha256,
            final_cache_width=self.entry_cache_width,
            final_weight_sha256=self.entry_weight_sha256,
            final_append_event_count=0,
            weight_rollback_count=1,
            cache_rollback_count=1,
        )

    def _finish(
        self,
        *,
        status: str,
        final_cache_sha256: str,
        final_cache_width: int,
        final_weight_sha256: str,
        final_append_event_count: int,
        weight_rollback_count: int,
        cache_rollback_count: int,
    ) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p4-euler-cache-transaction/v1",
            "instruction_id": P4_EULER_INSTRUCTION_ID,
            "status": status,
            "entry_cache_sha256": self.entry_cache_sha256,
            "entry_cache_width": self.entry_cache_width,
            "entry_weight_sha256": self.entry_weight_sha256,
            "request_count": self.request_count,
            "outer_rows": list(self._rows),
            "k1_k7_committed_current_batch_append_count": 0,
            "final_append_event_count": final_append_event_count,
            "final_cache_sha256": final_cache_sha256,
            "final_cache_width": final_cache_width,
            "final_weight_sha256": final_weight_sha256,
            "failed_or_incomplete_append_count": 0,
            "weight_rollback_count": weight_rollback_count,
            "cache_rollback_count": cache_rollback_count,
            "arm_global_singleton_isolation_required": True,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        self._terminal = payload
        return payload


__all__ = [
    "P4_EULER_OUTER_STEPS",
    "P4EulerCacheTransaction",
    "verify_k_target_bridge",
]
