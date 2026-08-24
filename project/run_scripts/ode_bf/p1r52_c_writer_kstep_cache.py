"""Batch-entry Alpha-cache policy for the C-writer K-step experiments.

The policy deliberately separates one B100's eight writer calls from the
committed cache ledger.  Every K observes the same immutable batch-entry
history; the current B100 contributes exactly one key block only after K8 and
the enclosing W transaction both succeed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256
from .scalable_batched_runtime import P1R23_GRID_COUNT, P1R23_LAYER_ORDER
from .writer_cadence import WriterCadence


def snapshot_alpha_module_cache(alpha_main: Any) -> dict[str, Any]:
    return {
        "cache_c_present": hasattr(alpha_main, "cache_c"),
        "cache_c": (
            getattr(alpha_main, "cache_c").detach().clone()
            if isinstance(getattr(alpha_main, "cache_c", None), torch.Tensor)
            else copy.deepcopy(getattr(alpha_main, "cache_c", None))
        ),
        "cache_c_new_present": hasattr(alpha_main, "cache_c_new"),
        "cache_c_new": copy.deepcopy(getattr(alpha_main, "cache_c_new", None)),
    }


def restore_alpha_module_cache(alpha_main: Any, snapshot: Mapping[str, Any]) -> None:
    for name in ("cache_c", "cache_c_new"):
        if bool(snapshot[f"{name}_present"]):
            value = snapshot[name]
            setattr(
                alpha_main,
                name,
                value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value),
            )
        elif hasattr(alpha_main, name):
            delattr(alpha_main, name)


def _snapshot_identity(snapshot: Mapping[str, Any]) -> str:
    value = snapshot.get("cache_c")
    payload = {
        "cache_c_present": bool(snapshot["cache_c_present"]),
        "cache_c_sha256": tensor_sha256(value) if isinstance(value, torch.Tensor) else canonical_hash(value),
        "cache_c_new_present": bool(snapshot["cache_c_new_present"]),
        "cache_c_new_identity": canonical_hash(snapshot.get("cache_c_new")),
    }
    return canonical_hash(payload)


@dataclass(frozen=True, slots=True)
class KStepBatchCacheCommit:
    history_keys_by_layer: Mapping[int, torch.Tensor] | None
    receipt: Mapping[str, Any]


class KStepBatchEntryCachePolicy:
    """Mutable transaction helper with immutable, raw-free commit output."""

    def __init__(
        self,
        *,
        arm: str,
        batch_index: int,
        history_keys_by_layer: Mapping[int, torch.Tensor] | None,
        history_version: int,
        alpha_main: Any | None = None,
        alpha_entry_snapshot: Mapping[str, Any] | None = None,
        expected_official_entry_sha256: str | None = None,
        writer_cadence: WriterCadence = WriterCadence.KSTEP_EACH_OUTER,
    ) -> None:
        if arm not in ("C0-KSTEP-CACHE", "C1-KSTEP-CACHE", "C3-KSTEP-CACHE"):
            raise ODEBFContractError("K-step cache arm differs")
        if isinstance(batch_index, bool) or batch_index < 1:
            raise ODEBFContractError("K-step cache batch index differs")
        expected_width = (batch_index - 1) * 100
        self.arm = arm
        self.batch_index = batch_index
        self.history_version = history_version
        self.entry_width = expected_width
        self.expected_official_entry_sha256 = expected_official_entry_sha256
        if not isinstance(writer_cadence, WriterCadence):
            raise ODEBFContractError("K-step cache writer cadence differs")
        self.writer_cadence = writer_cadence
        self._entry_snapshot = copy.deepcopy(alpha_entry_snapshot)
        self._alpha_main = alpha_main
        self._candidate_snapshot: dict[str, Any] | None = None
        self._candidate_official_exit_sha256: str | None = None
        self._batch_entry_keys: dict[int, torch.Tensor] | None = None
        self._closed_steps: set[int] = set()
        self._committed = False
        if arm.startswith(("C0", "C1")):
            if history_keys_by_layer is None:
                if expected_width != 0:
                    raise ODEBFContractError("K-step cache history inventory is absent")
                self.history_keys_by_layer = None
            else:
                if set(history_keys_by_layer) != set(P1R23_LAYER_ORDER):
                    raise ODEBFContractError("K-step cache history inventory differs")
                self.history_keys_by_layer = {
                    layer: value.detach().cpu().float().contiguous().clone()
                    for layer, value in history_keys_by_layer.items()
                }
                if any(value.shape[1] != expected_width for value in self.history_keys_by_layer.values()):
                    raise ODEBFStateError("K-step cache history width differs")
            if alpha_main is not None or alpha_entry_snapshot is not None:
                raise ODEBFContractError("C0/C1 cache policy received Official cache state")
        else:
            if history_keys_by_layer is not None or alpha_main is None or alpha_entry_snapshot is None:
                raise ODEBFContractError("C3 Official cache policy boundary differs")
            self.history_keys_by_layer = None
            self._entry_snapshot_identity = _snapshot_identity(alpha_entry_snapshot)

    def close_no_write_step(self, step_index: int) -> None:
        """Seal an outer target transition that has no writer authority."""

        if (
            self.writer_cadence is not WriterCadence.FINAL_Z_ONESHOT
            or self.writer_cadence.writes_at(step_index)
            or step_index in self._closed_steps
            or self._committed
        ):
            raise ODEBFStateError("no-write cache close order differs")
        self._closed_steps.add(step_index)

    def close_c0_c1_step(
        self,
        step_index: int,
        keys_by_layer: Mapping[int, torch.Tensor],
    ) -> None:
        if not self.arm.startswith(("C0", "C1")) or step_index in self._closed_steps:
            raise ODEBFStateError("K-step C0/C1 close order differs")
        if set(keys_by_layer) != set(P1R23_LAYER_ORDER):
            raise ODEBFContractError("K-step batch-entry key inventory differs")
        if step_index == 0:
            if self._batch_entry_keys is not None:
                raise ODEBFStateError("K-step batch-entry keys already captured")
            self._batch_entry_keys = {
                layer: value.detach().cpu().float().contiguous().clone()
                for layer, value in keys_by_layer.items()
            }
        self._closed_steps.add(step_index)

    def prepare_c3_step(self, step_index: int) -> None:
        if self.arm != "C3-KSTEP-CACHE" or step_index in self._closed_steps or self._committed:
            raise ODEBFStateError("C3 cache K-step prepare order differs")
        assert self._alpha_main is not None and self._entry_snapshot is not None
        restore_alpha_module_cache(self._alpha_main, self._entry_snapshot)

    def close_c3_step(self, step_index: int, official_dynamic: Mapping[str, Any]) -> None:
        if self.arm != "C3-KSTEP-CACHE" or step_index in self._closed_steps:
            raise ODEBFStateError("C3 cache K-step close order differs")
        assert self._alpha_main is not None and self._entry_snapshot is not None
        try:
            if (
                int(official_dynamic["logical_history_width_at_entry"]) != self.entry_width
                or int(official_dynamic["logical_history_width_after_append"]) != self.entry_width + 100
                or (self.entry_width > 0 and not bool(official_dynamic["solver_consumed_entry_cache"]))
                or (
                    self.expected_official_entry_sha256 is not None
                    and official_dynamic["entry"]["sha256"] != self.expected_official_entry_sha256
                )
            ):
                raise ODEBFStateError("C3 batch-entry cache consumption differs")
            candidate = snapshot_alpha_module_cache(self._alpha_main)
            if step_index == P1R23_GRID_COUNT - 1:
                self._candidate_snapshot = candidate
                self._candidate_official_exit_sha256 = str(official_dynamic["exit"]["sha256"])
            self._closed_steps.add(step_index)
        finally:
            restore_alpha_module_cache(self._alpha_main, self._entry_snapshot)

    def abort_c3_step(self) -> None:
        if self.arm == "C3-KSTEP-CACHE":
            assert self._alpha_main is not None and self._entry_snapshot is not None
            restore_alpha_module_cache(self._alpha_main, self._entry_snapshot)

    def commit_after_k8(self) -> KStepBatchCacheCommit:
        if self._committed or self._closed_steps != set(range(P1R23_GRID_COUNT)):
            raise ODEBFStateError("K-step cache commit completeness differs")
        if self.arm.startswith(("C0", "C1")):
            if self._batch_entry_keys is None:
                raise ODEBFStateError("K-step cache append proposal is absent")
            next_history = {
                layer: (
                    self._batch_entry_keys[layer].clone()
                    if self.history_keys_by_layer is None
                    else torch.cat(
                        (self.history_keys_by_layer[layer], self._batch_entry_keys[layer]), dim=1
                    ).contiguous()
                )
                for layer in P1R23_LAYER_ORDER
            }
            entry_identity = (
                {str(layer): "EMPTY_HISTORY" for layer in P1R23_LAYER_ORDER}
                if self.history_keys_by_layer is None
                else {
                    str(layer): tensor_sha256(self.history_keys_by_layer[layer])
                    for layer in P1R23_LAYER_ORDER
                }
            )
            append_identity = {
                str(layer): tensor_sha256(self._batch_entry_keys[layer])
                for layer in P1R23_LAYER_ORDER
            }
            cache_kind = "COMMITTED_PAST_ALPHA_SOLVE_KEYS"
            official_exit = None
        else:
            if self._candidate_snapshot is None:
                raise ODEBFStateError("C3 K8 cache candidate is absent")
            assert self._alpha_main is not None
            restore_alpha_module_cache(self._alpha_main, self._candidate_snapshot)
            next_history = None
            entry_identity = self._entry_snapshot_identity
            append_identity = self._candidate_official_exit_sha256
            cache_kind = "OFFICIAL_ALPHAEDIT_DYNAMIC_CACHE_C"
            official_exit = self._candidate_official_exit_sha256
        self._committed = True
        receipt = {
            "schema": "ode-edit-s05-p1r52-c-writer-kstep-cache-commit/v1",
            "arm": self.arm,
            "batch_index": self.batch_index,
            "cache_kind": cache_kind,
            "entry_width": self.entry_width,
            "consume_width_per_k": self.entry_width,
            "consume_count": self.writer_cadence.writer_calls_per_block,
            "append_width": 100,
            "exit_width": self.entry_width + 100,
            "entry_version": self.history_version,
            "exit_version": self.history_version + 1,
            "entry_identity": entry_identity,
            "append_identity": append_identity,
            "official_exit_sha256": official_exit,
            "same_batch_entry_cache_reuse_count": (
                self.writer_cadence.writer_calls_per_block
            ),
            "current_batch_k_key_history_inclusion_count": 0,
            "current_uncommitted_prefix_key_history_inclusion_count": 0,
            "append_count": 1,
            "rollback_count": 0,
            "commit_count": 1,
        }
        if self.writer_cadence is not WriterCadence.KSTEP_EACH_OUTER:
            receipt.update(
                {
                    "writer_cadence": self.writer_cadence.value,
                    "no_write_closed_step_count": (
                        P1R23_GRID_COUNT
                        - self.writer_cadence.writer_calls_per_block
                    ),
                }
            )
        receipt["identity_sha256"] = canonical_hash(receipt)
        return KStepBatchCacheCommit(next_history, receipt)


__all__ = [
    "KStepBatchCacheCommit",
    "KStepBatchEntryCachePolicy",
    "restore_alpha_module_cache",
    "snapshot_alpha_module_cache",
]
