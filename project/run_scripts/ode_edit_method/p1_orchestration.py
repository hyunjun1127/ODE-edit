"""Fail-closed state machine for one canonical P1 sequential arm stream."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import Arm, ArmRunResult, MethodContractError


@dataclass(frozen=True, slots=True)
class _OpenEdit:
    order_position: int
    case_id: str
    pre_edit_state_id: str
    pre_edit_target_weight_state_id: str
    omega_before: tuple[tuple[int, float], ...]
    history_length_before: int
    cache_identity: str


class P1SequentialArmGuard:
    """Validate state/Omega continuity without touching model tensors."""

    def __init__(self, arm: Arm, case_ids: Sequence[str]) -> None:
        if arm not in {
            Arm.NATIVE_MEMIT,
            Arm.STATIC_SYNCHRONOUS,
            Arm.ONE_REFRESH,
            Arm.FULL_ODE_EDIT,
        }:
            raise MethodContractError("P1 stream received a non-P1 arm")
        canonical = tuple(str(value) for value in case_ids)
        if canonical != ("2022", "12498", "20964", "768"):
            raise MethodContractError("P1 stream canonical case order differs")
        self.arm = arm
        self.case_ids = canonical
        self._start_restore_count = 0
        self._end_restore_count = 0
        self._next_position = 0
        self._open: _OpenEdit | None = None
        self._previous_post_target_weight_state_id: str | None = None
        self._previous_omega_after: tuple[tuple[int, float], ...] | None = None
        self._previous_history_length = 0
        self._cache_identities: set[str] = set()
        self._direct_z_counts: list[int] = []
        self._rows: list[dict[str, Any]] = []

    @staticmethod
    def _omega(value: Mapping[int, float]) -> tuple[tuple[int, float], ...]:
        rows = tuple(sorted((int(layer), float(load)) for layer, load in value.items()))
        if not rows or any(
            not math.isfinite(load) or load < 0.0 for _layer, load in rows
        ):
            raise MethodContractError("P1 stream Omega state is empty or negative")
        return rows

    def note_arm_start_restore(self) -> None:
        if self._start_restore_count or self._next_position or self._open is not None:
            raise MethodContractError("P1 arm start baseline restore is not exactly once")
        self._start_restore_count = 1

    def begin_edit(
        self,
        *,
        order_position: int,
        case_id: str,
        pre_edit_state_id: str,
        pre_edit_target_weight_state_id: str,
        omega_before: Mapping[int, float],
        history_length_before: int,
        cache_identity: str,
    ) -> None:
        if self._start_restore_count != 1 or self._end_restore_count:
            raise MethodContractError("P1 edit began outside an active arm stream")
        if self._open is not None:
            raise MethodContractError("P1 sequential edits overlap")
        if order_position != self._next_position or order_position >= len(self.case_ids):
            raise MethodContractError("P1 order position is non-canonical")
        if str(case_id) != self.case_ids[order_position]:
            raise MethodContractError("P1 case differs at canonical order position")
        if (
            not pre_edit_state_id
            or not pre_edit_target_weight_state_id
            or not cache_identity
        ):
            raise MethodContractError("P1 sequential state/cache identity is empty")
        if cache_identity in self._cache_identities:
            raise MethodContractError("P1 direct-z cache identity was reused")
        locked_omega = self._omega(omega_before)
        if order_position == 0:
            if history_length_before != 0 or any(load != 0.0 for _layer, load in locked_omega):
                raise MethodContractError("P1 arm did not start with independent zero Omega")
        else:
            if (
                pre_edit_target_weight_state_id
                != self._previous_post_target_weight_state_id
            ):
                raise MethodContractError(
                    "P1 retained target-weight endpoint did not feed the next edit"
                )
            if (
                locked_omega != self._previous_omega_after
                or history_length_before != self._previous_history_length
            ):
                raise MethodContractError("P1 cumulative Omega/history continuity differs")
        self._cache_identities.add(cache_identity)
        self._open = _OpenEdit(
            order_position=order_position,
            case_id=str(case_id),
            pre_edit_state_id=pre_edit_state_id,
            pre_edit_target_weight_state_id=pre_edit_target_weight_state_id,
            omega_before=locked_omega,
            history_length_before=int(history_length_before),
            cache_identity=cache_identity,
        )

    def finish_edit(
        self,
        *,
        result: ArmRunResult,
        post_edit_state_id: str,
        post_edit_target_weight_state_id: str,
        omega_after: Mapping[int, float],
        history_length_after: int,
    ) -> None:
        current = self._open
        if current is None:
            raise MethodContractError("P1 edit finished without begin_edit")
        if (
            result.arm is not self.arm
            or not post_edit_state_id
            or not post_edit_target_weight_state_id
            or result.terminal_state_id != post_edit_state_id
        ):
            raise MethodContractError("P1 edit result arm/state identity differs")
        locked_after = self._omega(omega_after)
        if tuple(layer for layer, _load in locked_after) != tuple(
            layer for layer, _load in current.omega_before
        ):
            raise MethodContractError("P1 Omega layer set changed within an edit")
        if result.omega_appended:
            if history_length_after != current.history_length_before + 1:
                raise MethodContractError("P1 completed edit did not append Omega once")
            if any(
                after < before
                for (_layer, before), (_same_layer, after) in zip(
                    current.omega_before, locked_after, strict=True
                )
            ):
                raise MethodContractError("P1 cumulative Omega decreased")
        else:
            if (
                history_length_after != current.history_length_before
                or locked_after != current.omega_before
                or post_edit_state_id != current.pre_edit_state_id
                or post_edit_target_weight_state_id
                != current.pre_edit_target_weight_state_id
            ):
                raise MethodContractError(
                    "P1 failed edit did not locally rollback request/target-weight state/Omega"
                )
        if result.direct_z_compute_count not in {0, 1}:
            raise MethodContractError("P1 direct-z count escaped once/edit contract")
        if result.direct_z_compute_count == 0 and not (
            result.status == "event_hit" and not result.steps and result.omega_appended
        ):
            raise MethodContractError("P1 zero direct-z is not an entry hit")
        self._direct_z_counts.append(result.direct_z_compute_count)
        self._rows.append(
            {
                "order_position": current.order_position,
                "case_id": current.case_id,
                "pre_edit_state_id": current.pre_edit_state_id,
                "post_edit_state_id": post_edit_state_id,
                "pre_edit_target_weight_state_id": (
                    current.pre_edit_target_weight_state_id
                ),
                "post_edit_target_weight_state_id": (
                    post_edit_target_weight_state_id
                ),
                "history_length_before": current.history_length_before,
                "history_length_after": history_length_after,
                "terminal_appended": result.omega_appended,
                "direct_z_compute_count": result.direct_z_compute_count,
                "cache_identity": current.cache_identity,
            }
        )
        self._previous_post_target_weight_state_id = (
            post_edit_target_weight_state_id
        )
        self._previous_omega_after = locked_after
        self._previous_history_length = int(history_length_after)
        self._next_position += 1
        self._open = None

    def note_arm_end_restore(self, *, exact: bool) -> None:
        if (
            exact is not True
            or self._start_restore_count != 1
            or self._end_restore_count
            or self._open is not None
            or self._next_position != len(self.case_ids)
        ):
            raise MethodContractError("P1 arm end baseline restore/integrity differs")
        self._end_restore_count = 1

    def summary(self) -> dict[str, Any]:
        if self._end_restore_count != 1:
            raise MethodContractError("P1 stream summary precedes arm isolation")
        return {
            "arm": self.arm.value,
            "execution_axis": "arm-outer-canonical-edit-order-inner",
            "case_ids": list(self.case_ids),
            "arm_start_baseline_restore_count": self._start_restore_count,
            "arm_end_baseline_restore_count": self._end_restore_count,
            "case_level_w0_restore_count": 0,
            "edit_count": len(self._rows),
            "direct_z_compute_counts": list(self._direct_z_counts),
            "cache_identity_count": len(self._cache_identities),
            "p1_cross_arm_h0_over_D_native_computed": False,
            "rows": list(self._rows),
        }
