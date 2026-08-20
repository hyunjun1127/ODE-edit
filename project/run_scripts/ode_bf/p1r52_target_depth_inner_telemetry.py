"""Observation-only accepted-inner and post-writer telemetry for P1R52 depth.

The observer uses the pinned CounterFact evaluator after a target tensor has
already been selected.  Its outputs never flow back into target selection,
the writer, routing, materialization, or transaction state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .bg_soft_diagnostics import HeldoutRequestResidualActivationOverlay
from .common_cold_coordinate import common_terminal_residual_input
from .common_coldcoord_fixed_e8_runtime import _heldout_additive_lookup_geometry
from .contracts import ODEBFContractError, canonical_hash
from .functional import tensor_sha256
from .p1_evaluator import (
    EndpointActionFreeze,
    evaluate_counterfact_success_accuracy_batch,
)
from .p1_stepwise import StepwiseActionFreeze, evaluate_counterfact_stepwise_primary


SCHEMA = "ode-edit-s05-p1r52-target-depth-inner-telemetry/v1"
OUTER_SCHEMA = "ode-edit-s05-p1r52-target-depth-outer-terminal-telemetry/v1"


@dataclass(frozen=True, slots=True)
class P1R52OuterTelemetryResult:
    receipt: Mapping[str, Any]
    terminal_four_panel: Mapping[str, Any]


def _metric(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        primary = raw.get("primary", raw.get("legacy_primary"))
        value = primary["metrics"][name]
    except (KeyError, TypeError) as exc:
        raise ODEBFContractError("P1R52 inner telemetry metric schema differs") from exc
    if not isinstance(value, Mapping):
        raise ODEBFContractError("P1R52 inner telemetry metric value differs")
    return value


def _numeric(raw: Mapping[str, Any], name: str) -> Sequence[Sequence[Mapping[str, Any]]]:
    try:
        value = raw["numeric_vectors"][name]
    except (KeyError, TypeError) as exc:
        raise ODEBFContractError("P1R52 inner telemetry numeric schema differs") from exc
    if not isinstance(value, Sequence):
        raise ODEBFContractError("P1R52 inner telemetry numeric value differs")
    return value


def _strict_request_count(metric: Mapping[str, Any]) -> int:
    correct = metric.get("per_case_correct")
    required = metric.get("per_case_required")
    if not isinstance(correct, Sequence) or not isinstance(required, Sequence):
        raise ODEBFContractError("P1R52 inner telemetry strict metric differs")
    return sum(
        int(int(left) == int(right))
        for left, right in zip(correct, required, strict=True)
    )


def _nll_new_by_request(
    raw: Mapping[str, Any], name: str
) -> list[list[float]]:
    if "numeric_vectors" not in raw:
        prompt_name = {
            "efficacy": "rewrite_success",
            "generalization": "paraphrase_success",
        }.get(name)
        if prompt_name is None:
            raise ODEBFContractError("P1R52 accuracy telemetry metric differs")
        return [
            [float(item) for item in row]
            for row in raw[prompt_name]["target_new_nll_by_request"]
        ]
    groups = _numeric(raw, name)
    values: list[list[float]] = []
    for request_index, group in enumerate(groups):
        if not isinstance(group, Sequence):
            raise ODEBFContractError("P1R52 inner telemetry request group differs")
        row: list[float] = []
        for prompt in group:
            if not isinstance(prompt, Mapping) or int(prompt["request_index"]) != request_index:
                raise ODEBFContractError("P1R52 inner telemetry request binding differs")
            row.append(float(prompt["nll_new"]))
        values.append(row)
    return values


def _nll_true_by_request(raw: Mapping[str, Any], name: str) -> list[list[float]]:
    prompt_name = {
        "efficacy": "rewrite_success",
        "generalization": "paraphrase_success",
    }.get(name)
    if prompt_name is None or prompt_name not in raw:
        raise ODEBFContractError("P1R52 accuracy telemetry true-NLL differs")
    return [
        [float(item) for item in row]
        for row in raw[prompt_name]["target_true_nll_by_request"]
    ]


def _prompt_summary(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    try:
        value = raw[name]
    except (KeyError, TypeError) as exc:
        raise ODEBFContractError("P1R52 accuracy telemetry prompt metric differs") from exc
    return {
        "prompt_numerator": int(value["prompt_numerator"]),
        "prompt_denominator": int(value["prompt_denominator"]),
        "prompt_rate": float(value["prompt_rate"]),
        "strict_request_numerator": int(value["strict_request_numerator"]),
        "strict_request_denominator": int(value["strict_request_denominator"]),
        "strict_request_rate": float(value["strict_request_rate"]),
        "prompt_bit_vector_sha256": str(value["prompt_bit_vector_sha256"]),
        "strict_request_bit_vector_sha256": str(
            value["strict_request_bit_vector_sha256"]
        ),
    }


def _mean_nested(values: Sequence[Sequence[float]]) -> float:
    flat = [float(item) for group in values for item in group]
    if not flat:
        raise ODEBFContractError("P1R52 inner telemetry denominator is empty")
    return sum(flat) / len(flat)


def _metric_summary(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    metric = _metric(raw, name)
    return {
        "numerator": int(metric["numerator"]),
        "denominator": int(metric["denominator"]),
        "rate": float(metric["official_aggregate"]),
        "strict_request_count": _strict_request_count(metric),
        "bit_vector_sha256": str(metric["bit_vector_sha256"]),
    }


class P1R52TargetDepthTelemetryObserver:
    """Pinned, no-action observer instantiated once for one independent B10."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        requests: Sequence[Mapping[str, Any]],
        cases: Sequence[Any],
        *,
        alias: str,
        method: str,
        request_order_sha256: str,
        target_layer_name: str,
        fact_token_strategy: str,
        expected_request_count: int = 10,
        include_accuracy: bool = False,
    ) -> None:
        if (
            isinstance(expected_request_count, bool)
            or expected_request_count <= 0
            or len(requests) != expected_request_count
            or len(cases) != expected_request_count
        ):
            raise ODEBFContractError("P1R52 inner telemetry request count differs")
        self.model = model
        self.tokenizer = tokenizer
        self.cases = tuple(cases)
        self.alias = alias
        self.method = method
        self.request_order_sha256 = request_order_sha256
        self.target_layer_name = target_layer_name
        self.request_count = expected_request_count
        self.include_accuracy = include_accuracy
        self._last_inner_by_outer: dict[
            int, tuple[str, Mapping[str, Any], Mapping[str, Any], str]
        ] = {}
        (
            self.lookup_positions,
            self.patched_rows,
            self.lookup_receipt,
        ) = _heldout_additive_lookup_geometry(
            tokenizer,
            requests,
            cases,
            fact_token_strategy=fact_token_strategy,
            expected_batch_size=expected_request_count,
        )

    def _freeze(
        self,
        *,
        outer_step_index: int,
        inner_index: int,
        ordinal: int,
        target_sha256: str,
        role: str,
    ) -> StepwiseActionFreeze | EndpointActionFreeze:
        snapshot = canonical_hash(
            {
                "schema": SCHEMA,
                "outer_step_index": outer_step_index,
                "inner_index": inner_index,
                "ordinal": ordinal,
                "target_sha256": target_sha256,
                "role": role,
            }
        )
        if self.include_accuracy:
            return EndpointActionFreeze(
                arm=f"{self.method}-{role}",
                sequential_batch=min(outer_step_index, 9),
                request_order_sha256=self.request_order_sha256,
                selected_snapshot_sha256=snapshot,
                fixed_budget_slots_completed=0,
            )
        return StepwiseActionFreeze(
            variant=f"{self.method}-{role}",
            request_order_sha256=self.request_order_sha256,
            rollout_sha256=snapshot,
            snapshot_sha256=snapshot,
            snapshot_index=ordinal,
            accepted_snapshot_count=ordinal + 1,
            rejected_retry_count=0,
            trajectory_status="OBSERVATION_ONLY_ACCEPTED_TARGET_DEPTH",
        )

    def _evaluate_z(
        self,
        target: torch.Tensor,
        current_terminal: torch.Tensor,
        freeze: StepwiseActionFreeze | EndpointActionFreeze,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], float]:
        residual = common_terminal_residual_input(
            target,
            current_terminal,
            self.request_order_sha256,
            expected_request_count=self.request_count,
        )
        overlay = HeldoutRequestResidualActivationOverlay(
            self.model,
            self.target_layer_name,
            residual.residual,
            self.lookup_positions,
            self.patched_rows,
        )
        started = time.perf_counter()
        with overlay:
            evaluated = (
                evaluate_counterfact_success_accuracy_batch(
                    self.model,
                    self.tokenizer,
                    self.cases,
                    model_alias=self.alias,
                    freeze=freeze,
                    expected_batch_size=self.request_count,
                )
                if self.include_accuracy
                else evaluate_counterfact_stepwise_primary(
                    self.model,
                    self.tokenizer,
                    self.cases,
                    model_alias=self.alias,
                    freeze=freeze,
                )
            )
        return evaluated.raw_free_payload(), {
            "overlay": overlay.raw_free_payload(),
            "residual": residual.raw_free_payload(),
        }, time.perf_counter() - started

    def observe_inner(
        self,
        *,
        depth_policy: str,
        configured_inner_count: int,
        outer_step_index: int,
        inner_index: int,
        global_target_update_ordinal: int,
        outer_entry_target: torch.Tensor,
        previous_target: torch.Tensor,
        accepted_target: torch.Tensor,
        current_terminal: torch.Tensor,
        selected_endpoint: Any,
    ) -> Mapping[str, Any]:
        freeze = self._freeze(
            outer_step_index=outer_step_index,
            inner_index=inner_index,
            ordinal=global_target_update_ordinal,
            target_sha256=tensor_sha256(accepted_target),
            role="INNER_Z",
        )
        raw, geometry, wall = self._evaluate_z(
            accepted_target, current_terminal, freeze
        )
        objective = [float(item) for item in selected_endpoint.per_request_values]
        rewrite = _nll_new_by_request(raw, "efficacy")
        rephrase = _nll_new_by_request(raw, "generalization")
        request_count = len(objective)
        if (
            request_count != self.request_count
            or len(rewrite) != request_count
            or len(rephrase) != request_count
            or any(len(row) != 1 for row in rewrite)
        ):
            raise ODEBFContractError("P1R52 inner telemetry denominator differs")
        move = (
            accepted_target.detach().to(device="cpu", dtype=torch.float64)
            - previous_target.detach().to(device="cpu", dtype=torch.float64)
        )
        cumulative = (
            accepted_target.detach().to(device="cpu", dtype=torch.float64)
            - outer_entry_target.detach().to(device="cpu", dtype=torch.float64)
        )
        primary = raw.get("primary", raw.get("legacy_primary"))
        if not isinstance(primary, Mapping):
            raise ODEBFContractError("P1R52 inner telemetry primary schema differs")
        if self.include_accuracy:
            self._last_inner_by_outer[outer_step_index] = (
                tensor_sha256(accepted_target),
                raw,
                geometry,
                freeze.identity(),
            )
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "depth_policy": depth_policy,
            "configured_inner_count": configured_inner_count,
            "outer_step_index": outer_step_index,
            "inner_index": inner_index,
            "global_target_update_ordinal": global_target_update_ordinal,
            "request_count": request_count,
            "request_order_sha256": self.request_order_sha256,
            "accepted_target_sha256": tensor_sha256(accepted_target),
            "target_objective_nll_mean": float(selected_endpoint.loss),
            "target_objective_nll_by_request": objective,
            "target_objective_explicit_denominator": request_count,
            "accepted_z_rewrite_nll_by_request": [row[0] for row in rewrite],
            "accepted_z_rewrite_explicit_denominator": request_count,
            "accepted_z_rephrase_nll_by_request": rephrase,
            "accepted_z_rephrase_explicit_denominator": sum(map(len, rephrase)),
            "accepted_z_efficacy": _metric_summary(raw, "efficacy"),
            "accepted_z_generalization": _metric_summary(raw, "generalization"),
            "inner_step_movement_norm": float(torch.linalg.vector_norm(move)),
            "inner_step_movement_norm_by_request": [
                float(torch.linalg.vector_norm(move[:, index]))
                for index in range(request_count)
            ],
            "cumulative_outer_entry_movement_norm": float(
                torch.linalg.vector_norm(cumulative)
            ),
            "cumulative_outer_entry_movement_norm_by_request": [
                float(torch.linalg.vector_norm(cumulative[:, index]))
                for index in range(request_count)
            ],
            "observation_pass_count": 1,
            "added_model_forward_count": int(primary["model_forward_count"]),
            "added_processed_token_count": int(primary["processed_token_count"]),
            "added_backward_count": 0,
            "added_generation_count": 0,
            "controller_action_influence_count": 0,
            "duplicate_evaluation_count": 0,
            "pinned_numeric_vectors_sha256": str(primary["numeric_vectors_sha256"]),
            "action_freeze_sha256": freeze.identity(),
            "lookup_sha256": str(self.lookup_receipt["identity_sha256"]),
            "overlay_sha256": str(geometry["overlay"]["identity_sha256"]),
            "residual_sha256": canonical_hash(geometry["residual"]),
            "wall_seconds": wall,
        }
        if self.include_accuracy:
            payload.update(
                {
                    "accepted_z_rewrite_target_true_nll_by_request": [
                        row[0] for row in _nll_true_by_request(raw, "efficacy")
                    ],
                    "accepted_z_rephrase_target_true_nll_by_request": (
                        _nll_true_by_request(raw, "generalization")
                    ),
                    "accepted_z_rewrite_success": _prompt_summary(
                        raw, "rewrite_success"
                    ),
                    "accepted_z_rewrite_accuracy": _prompt_summary(
                        raw, "rewrite_acc"
                    ),
                    "accepted_z_rephrase_success": _prompt_summary(
                        raw, "paraphrase_success"
                    ),
                    "accepted_z_rephrase_accuracy": _prompt_summary(
                        raw, "paraphrase_acc"
                    ),
                    "accuracy_observation_added_forward_count": 0,
                    "accuracy_observation_added_backward_count": 0,
                    "accuracy_observation_added_generation_count": 0,
                }
            )
        payload["identity_sha256"] = canonical_hash(payload)
        return payload

    def observe_outer(
        self,
        *,
        outer_step_index: int,
        configured_inner_count: int,
        accepted_target: torch.Tensor,
        current_terminal: torch.Tensor,
    ) -> P1R52OuterTelemetryResult:
        ordinal = (outer_step_index + 1) * configured_inner_count - 1
        freeze = self._freeze(
            outer_step_index=outer_step_index,
            inner_index=configured_inner_count - 1,
            ordinal=ordinal,
            target_sha256=tensor_sha256(accepted_target),
            role="OUTER_W_Z",
        )
        started = time.perf_counter()
        weight = (
            evaluate_counterfact_success_accuracy_batch(
                self.model,
                self.tokenizer,
                self.cases,
                model_alias=self.alias,
                freeze=freeze,
                expected_batch_size=self.request_count,
            )
            if self.include_accuracy
            else evaluate_counterfact_stepwise_primary(
                self.model,
                self.tokenizer,
                self.cases,
                model_alias=self.alias,
                freeze=freeze,
            )
        )
        weight_wall = time.perf_counter() - started
        cached = self._last_inner_by_outer.get(outer_step_index)
        if self.include_accuracy:
            if cached is None or cached[0] != tensor_sha256(accepted_target):
                raise ODEBFContractError(
                    "P1R52 outer accepted-z observation cache differs"
                )
            _, z_raw, geometry, z_action_freeze_sha256 = cached
            z_wall = 0.0
        else:
            z_raw, geometry, z_wall = self._evaluate_z(
                accepted_target, current_terminal, freeze
            )
            z_action_freeze_sha256 = freeze.identity()
        w_raw = weight.raw_free_payload()
        z_rewrite = _nll_new_by_request(z_raw, "efficacy")
        w_rewrite = _nll_new_by_request(w_raw, "efficacy")
        z_rephrase = _nll_new_by_request(z_raw, "generalization")
        w_rephrase = _nll_new_by_request(w_raw, "generalization")
        rewrite_gap = [
            w[0] - z[0]
            for w, z in zip(w_rewrite, z_rewrite, strict=True)
        ]
        rephrase_gap = [
            [right - left for right, left in zip(w, z, strict=True)]
            for w, z in zip(w_rephrase, z_rephrase, strict=True)
        ]
        w_primary = w_raw.get("primary", w_raw.get("legacy_primary"))
        z_primary = z_raw.get("primary", z_raw.get("legacy_primary"))
        panel: dict[str, Any] = {
            "schema": "ode-edit-s05-p1r52-target-depth-reused-terminal-four-panel/v1",
            "action_freeze_sha256": freeze.identity(),
            "z_action_freeze_sha256": z_action_freeze_sha256,
            "weight": w_raw,
            "z_inject": z_raw,
            "eff_z_inject": dict(_metric(z_raw, "efficacy")),
            "gen_z_inject": dict(_metric(z_raw, "generalization")),
            "eff_w": dict(_metric(w_raw, "efficacy")),
            "gen_w": dict(_metric(w_raw, "generalization")),
            "heldout_lookup": self.lookup_receipt,
            "z_overlay": geometry["overlay"],
            "terminal_residual": geometry["residual"],
            "heldout_gen_controller_access_count": 0,
            "heldout_efficacy_controller_access_count": 0,
            "terminal_only_evaluator": outer_step_index == 7,
            "inner_step_heldout_evaluation_count": 0,
            "observation_only": True,
            "wall_seconds": weight_wall + z_wall,
        }
        panel["identity_sha256"] = canonical_hash(panel)
        receipt: dict[str, Any] = {
            "schema": OUTER_SCHEMA,
            "outer_step_index": outer_step_index,
            "configured_inner_count": configured_inner_count,
            "request_count": self.request_count,
            "request_order_sha256": self.request_order_sha256,
            "accepted_z_rewrite_nll_by_request": [row[0] for row in z_rewrite],
            "accepted_z_rephrase_nll_by_request": z_rephrase,
            "writer_w_rewrite_nll_by_request": [row[0] for row in w_rewrite],
            "writer_w_rephrase_nll_by_request": w_rephrase,
            "rewrite_w_minus_z_nll_gap_by_request": rewrite_gap,
            "rephrase_w_minus_z_nll_gap_by_request": rephrase_gap,
            "rewrite_w_minus_z_nll_gap_mean": sum(rewrite_gap) / len(rewrite_gap),
            "rephrase_w_minus_z_nll_gap_mean": _mean_nested(rephrase_gap),
            "accepted_z_efficacy": _metric_summary(z_raw, "efficacy"),
            "accepted_z_generalization": _metric_summary(z_raw, "generalization"),
            "writer_w_efficacy": _metric_summary(w_raw, "efficacy"),
            "writer_w_generalization": _metric_summary(w_raw, "generalization"),
            "writer_w_locality": _metric_summary(w_raw, "locality-preservation"),
            "observation_pass_count": 1 if self.include_accuracy else 2,
            "accepted_z_observation_reused_from_last_inner": self.include_accuracy,
            "added_model_forward_count": int(w_primary["model_forward_count"])
            + (0 if self.include_accuracy else int(z_primary["model_forward_count"])),
            "added_processed_token_count": int(w_primary["processed_token_count"])
            + (0 if self.include_accuracy else int(z_primary["processed_token_count"])),
            "added_backward_count": 0,
            "added_generation_count": 0,
            "controller_action_influence_count": 0,
            "duplicate_evaluation_count": 0,
            "terminal_four_panel_sha256": panel["identity_sha256"],
            "wall_seconds": weight_wall + z_wall,
        }
        if self.include_accuracy:
            receipt.update(
                {
                    "accepted_z_rewrite_target_true_nll_by_request": [
                        row[0] for row in _nll_true_by_request(z_raw, "efficacy")
                    ],
                    "accepted_z_rephrase_target_true_nll_by_request": (
                        _nll_true_by_request(z_raw, "generalization")
                    ),
                    "writer_w_rewrite_target_true_nll_by_request": [
                        row[0] for row in _nll_true_by_request(w_raw, "efficacy")
                    ],
                    "writer_w_rephrase_target_true_nll_by_request": (
                        _nll_true_by_request(w_raw, "generalization")
                    ),
                    "accepted_z_rewrite_success": _prompt_summary(
                        z_raw, "rewrite_success"
                    ),
                    "accepted_z_rewrite_accuracy": _prompt_summary(
                        z_raw, "rewrite_acc"
                    ),
                    "accepted_z_rephrase_success": _prompt_summary(
                        z_raw, "paraphrase_success"
                    ),
                    "accepted_z_rephrase_accuracy": _prompt_summary(
                        z_raw, "paraphrase_acc"
                    ),
                    "writer_w_rewrite_success": _prompt_summary(
                        w_raw, "rewrite_success"
                    ),
                    "writer_w_rewrite_accuracy": _prompt_summary(
                        w_raw, "rewrite_acc"
                    ),
                    "writer_w_rephrase_success": _prompt_summary(
                        w_raw, "paraphrase_success"
                    ),
                    "writer_w_rephrase_accuracy": _prompt_summary(
                        w_raw, "paraphrase_acc"
                    ),
                    "accuracy_observation_added_forward_count": 0,
                    "accuracy_observation_added_backward_count": 0,
                    "accuracy_observation_added_generation_count": 0,
                }
            )
        receipt["identity_sha256"] = canonical_hash(receipt)
        return P1R52OuterTelemetryResult(receipt, panel)


__all__ = [
    "OUTER_SCHEMA",
    "P1R52OuterTelemetryResult",
    "P1R52TargetDepthTelemetryObserver",
    "SCHEMA",
]
