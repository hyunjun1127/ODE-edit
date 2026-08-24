"""Typed post-write controller-anchor policy for P1R54 FZ/PDZ.

This module owns no model, writer, evaluator, cache, optimizer, or target-field
implementation.  It only seals the state transition that selects the next
controller anchor after the already-existing post-write physical capture.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEBFContractError, ODEBFStateError, canonical_hash
from .functional import tensor_sha256


class RealizationPolicy(str, Enum):
    TARGET_Z_CONTINUATION = "TARGET_Z_CONTINUATION"
    POST_WRITE_W_REALIZATION_RESET = "POST_WRITE_W_REALIZATION_RESET"


def _vector_norm_by_request(value: torch.Tensor) -> list[float]:
    observed = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return [float(item) for item in torch.linalg.vector_norm(observed, dim=0)]


def _cosine_by_request(left: torch.Tensor, right: torch.Tensor) -> list[float | None]:
    lhs = left.detach().to(device="cpu", dtype=torch.float64).contiguous()
    rhs = right.detach().to(device="cpu", dtype=torch.float64).contiguous()
    numerator = torch.sum(lhs * rhs, dim=0)
    denominator = torch.linalg.vector_norm(lhs, dim=0) * torch.linalg.vector_norm(rhs, dim=0)
    values: list[float | None] = []
    for index in range(lhs.shape[1]):
        if float(denominator[index]) == 0.0:
            values.append(None)
        else:
            values.append(float(numerator[index] / denominator[index]))
    return values


class RealizationTransitionController:
    """Fail-closed K8 transition ledger with a distinct commanded endpoint."""

    def __init__(
        self,
        policy: RealizationPolicy | str,
        *,
        request_count: int,
        outer_count: int = 8,
    ) -> None:
        self.policy = policy if isinstance(policy, RealizationPolicy) else RealizationPolicy(policy)
        if (
            isinstance(request_count, bool)
            or request_count <= 0
            or isinstance(outer_count, bool)
            or outer_count != 8
        ):
            raise ODEBFContractError("P1R54 realization geometry differs")
        self.request_count = request_count
        self.outer_count = outer_count
        self._entry_rows: list[dict[str, Any]] = []
        self._transition_rows: list[dict[str, Any]] = []
        self._last_commanded_target: torch.Tensor | None = None
        self._target_origin_sha256: str | None = None
        self._teacher_sha256: str | None = None

    @property
    def is_reset(self) -> bool:
        return self.policy is RealizationPolicy.POST_WRITE_W_REALIZATION_RESET

    def observe_entry(
        self,
        *,
        step_index: int,
        controller_anchor: torch.Tensor,
        current_terminal: torch.Tensor,
        target_origin: torch.Tensor,
        teacher_sha256: str,
        target_new_nll_by_request: Sequence[float],
    ) -> Mapping[str, Any]:
        if (
            step_index != len(self._entry_rows)
            or not 0 <= step_index < self.outer_count
            or controller_anchor.shape != current_terminal.shape
            or target_origin.shape != controller_anchor.shape
            or controller_anchor.ndim != 2
            or controller_anchor.shape[1] != self.request_count
            or len(target_new_nll_by_request) != self.request_count
        ):
            raise ODEBFContractError("P1R54 realization entry geometry differs")
        origin_sha = tensor_sha256(target_origin)
        if step_index == 0:
            self._target_origin_sha256 = origin_sha
            self._teacher_sha256 = teacher_sha256
        elif (
            origin_sha != self._target_origin_sha256
            or teacher_sha256 != self._teacher_sha256
            or self._last_commanded_target is None
            or len(self._transition_rows) != step_index
        ):
            raise ODEBFStateError("P1R54 immutable origin/teacher/command state differs")
        anchor_sha = tensor_sha256(controller_anchor)
        terminal_sha = tensor_sha256(current_terminal)
        if self.is_reset and anchor_sha != terminal_sha:
            raise ODEBFStateError("P1R54 reset controller anchor is not physical terminal")
        if (
            not self.is_reset
            and step_index > 0
            and anchor_sha != tensor_sha256(self._last_commanded_target)
        ):
            raise ODEBFStateError("P1R54 continuation anchor is not last command")
        if step_index > 0:
            previous = self._transition_rows[step_index - 1]
            if previous["post_write_target_new_nll_by_request"] is not None:
                raise ODEBFStateError("P1R54 post-write NLL was already attributed")
            previous["post_write_target_new_nll_by_request"] = [
                float(item) for item in target_new_nll_by_request
            ]
            previous["post_write_nll_source"] = "NEXT_RESET_ENTRY_W_ONLY"
            previous.pop("identity_sha256", None)
            previous["identity_sha256"] = canonical_hash(previous)
        row: dict[str, Any] = {
            "step_index": step_index,
            "controller_anchor_sha256": anchor_sha,
            "current_terminal_sha256": terminal_sha,
            "target_origin_sha256": origin_sha,
            "teacher_sha256": teacher_sha256,
            "current_W_only_target_new_nll_by_request": [
                float(item) for item in target_new_nll_by_request
            ],
            "controller_carry_norm_by_request": _vector_norm_by_request(
                controller_anchor - current_terminal
            ),
            "reset_entry_carry_zero_by_contract": self.is_reset,
        }
        row["identity_sha256"] = canonical_hash(row)
        self._entry_rows.append(row)
        return row

    def advance(
        self,
        *,
        step_index: int,
        controller_anchor: torch.Tensor,
        commanded_target: torch.Tensor,
        current_terminal: torch.Tensor,
        next_physical_terminal: torch.Tensor,
        target_origin: torch.Tensor,
        teacher_sha256: str,
        commanded_target_new_nll_by_request: Sequence[float],
        terminal_post_write_nll_by_request: Sequence[float] | None = None,
    ) -> tuple[torch.Tensor, Mapping[str, Any]]:
        if (
            step_index != len(self._transition_rows)
            or step_index >= len(self._entry_rows)
            or any(
                item.shape != controller_anchor.shape
                for item in (
                    commanded_target,
                    current_terminal,
                    next_physical_terminal,
                    target_origin,
                )
            )
            or len(commanded_target_new_nll_by_request) != self.request_count
            or (terminal_post_write_nll_by_request is not None)
            != (step_index == self.outer_count - 1)
        ):
            raise ODEBFContractError("P1R54 realization transition geometry differs")
        if (
            tensor_sha256(target_origin) != self._target_origin_sha256
            or teacher_sha256 != self._teacher_sha256
            or tensor_sha256(controller_anchor)
            != self._entry_rows[step_index]["controller_anchor_sha256"]
            or tensor_sha256(current_terminal)
            != self._entry_rows[step_index]["current_terminal_sha256"]
        ):
            raise ODEBFStateError("P1R54 realization transition entry seal differs")

        c = controller_anchor - current_terminal
        d = commanded_target - controller_anchor
        r = commanded_target - current_terminal
        a = next_physical_terminal - current_terminal
        e_write = commanded_target - next_physical_terminal
        next_anchor = (
            next_physical_terminal if self.is_reset else commanded_target
        ).detach().clone().contiguous()
        row: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r54-realization-transition/v1",
            "policy": self.policy.value,
            "step_index": step_index,
            "controller_anchor_sha256": tensor_sha256(controller_anchor),
            "commanded_target_sha256": tensor_sha256(commanded_target),
            "last_commanded_target_sha256": tensor_sha256(commanded_target),
            "current_terminal_sha256": tensor_sha256(current_terminal),
            "next_physical_terminal_sha256": tensor_sha256(next_physical_terminal),
            "next_controller_anchor_sha256": tensor_sha256(next_anchor),
            "target_origin_sha256": tensor_sha256(target_origin),
            "teacher_sha256": teacher_sha256,
            "controller_carry_norm_by_request": _vector_norm_by_request(c),
            "fresh_target_motion_norm_by_request": _vector_norm_by_request(d),
            "full_command_norm_by_request": _vector_norm_by_request(r),
            "physical_movement_norm_by_request": _vector_norm_by_request(a),
            "pre_reset_writer_residual_norm_by_request": _vector_norm_by_request(e_write),
            "residual_ratio_by_request": [
                float(num / (den + 1.0e-12))
                for num, den in zip(
                    _vector_norm_by_request(e_write),
                    _vector_norm_by_request(r),
                    strict=True,
                )
            ],
            "command_physical_cosine_by_request": _cosine_by_request(r, a),
            "carry_fresh_motion_cosine_by_request": _cosine_by_request(c, d),
            "current_W_only_target_new_nll_by_request": list(
                self._entry_rows[step_index]["current_W_only_target_new_nll_by_request"]
            ),
            "commanded_target_new_nll_by_request": [
                float(item) for item in commanded_target_new_nll_by_request
            ],
            "post_write_target_new_nll_by_request": (
                None
                if terminal_post_write_nll_by_request is None
                else [float(item) for item in terminal_post_write_nll_by_request]
            ),
            "post_write_nll_source": (
                None if terminal_post_write_nll_by_request is None else "TERMINAL_W_ONLY"
            ),
            "reset_jump_target_energy_count": 0,
            "reset_jump_accepted_path_count": 0,
            "reset_added_model_forward_count": 0,
            "reset_added_backward_count": 0,
            "reset_added_materialization_count": 0,
            "reset_added_evaluator_count": 0,
            "terminal_z_oracle_source": "LAST_COMMANDED_TARGET",
            "next_controller_anchor_is_z_oracle": False,
        }
        row["identity_sha256"] = canonical_hash(row)
        self._transition_rows.append(row)
        self._last_commanded_target = commanded_target.detach().clone().contiguous()
        return next_anchor, row

    def terminal_commanded_target(self) -> torch.Tensor:
        if self._last_commanded_target is None or len(self._transition_rows) != self.outer_count:
            raise ODEBFStateError("P1R54 terminal commanded target is incomplete")
        return self._last_commanded_target.detach().clone().contiguous()

    def terminal_receipt(self) -> Mapping[str, Any]:
        if (
            len(self._entry_rows) != self.outer_count
            or len(self._transition_rows) != self.outer_count
            or self._last_commanded_target is None
            or any(row["post_write_target_new_nll_by_request"] is None for row in self._transition_rows)
        ):
            raise ODEBFStateError("P1R54 realization transition ledger is incomplete")
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r54-realization-transition-terminal/v1",
            "policy": self.policy.value,
            "request_count": self.request_count,
            "outer_count": self.outer_count,
            "entry_rows": [dict(item) for item in self._entry_rows],
            "transition_rows": [dict(item) for item in self._transition_rows],
            "terminal_commanded_target_sha256": tensor_sha256(self._last_commanded_target),
            "target_origin_sha256": self._target_origin_sha256,
            "teacher_sha256": self._teacher_sha256,
            "origin_refresh_count": 0,
            "teacher_refresh_count": 1,
            "reset_added_model_forward_count": 0,
            "reset_added_backward_count": 0,
            "reset_added_materialization_count": 0,
            "reset_added_evaluator_count": 0,
            "commanded_post_W_anchor_endpoint_separation": True,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


__all__ = ["RealizationPolicy", "RealizationTransitionController"]
