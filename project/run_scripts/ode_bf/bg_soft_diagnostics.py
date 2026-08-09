"""Pure, raw-free observation receipts for the BG-Soft missing-cell run.

The helpers in this module deliberately do not select a route, change a
clock, or write an artifact.  They make numerical observations auditable
without giving those observations any controller influence.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .contracts import (
    BATCH_SIZE,
    FIXED_K,
    ODEBFContractError,
    canonical_hash,
    finite,
)
from .functional import tensor_sha256


BG_SOFT_LAYER_ORDER = (4, 5, 6, 7, 8)
FALSE_FIELD_COLLAPSE_OVERLAY_OBJECTIVE_SATURATION = (
    "FALSE_FIELD_COLLAPSE_OVERLAY_OBJECTIVE_SATURATION"
)
TARGET_HOLD_ACTIVE = "TARGET_HOLD_ACTIVE"
NOT_DEFINED = "NOT_DEFINED"
SINGLE_WRITE_INTERACTION_EPSILON = 1.0e-12
_NO_FALSE_FIELD_COLLAPSE_OBSERVED = "NO_FALSE_FIELD_COLLAPSE_OBSERVED"
_DEFINED = "DEFINED"
_ZERO_REALIZED_COSINE_NOT_DEFINED = "ZERO_REALIZED_DELTA_COSINE_NOT_DEFINED"


def _validated_layer_order(layer_order: Sequence[int]) -> tuple[int, ...]:
    if isinstance(layer_order, (str, bytes)):
        raise ODEBFContractError("BG-Soft layer order differs")
    try:
        observed = tuple(layer_order)
    except TypeError as exc:
        raise ODEBFContractError("BG-Soft layer order differs") from exc
    if observed != BG_SOFT_LAYER_ORDER:
        raise ODEBFContractError("BG-Soft layer order differs")
    return observed


def _validated_sequence(name: str, value: Any) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ODEBFContractError(f"{name} must be an ordered sequence")
    return tuple(value)


def _validated_vector(name: str, value: Any) -> tuple[float, ...]:
    """Return exactly five finite values in locked layer order."""

    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.ndim != 1:
            raise ODEBFContractError(f"{name} shape differs")
        if value.numel() != len(BG_SOFT_LAYER_ORDER):
            raise ODEBFContractError(f"{name} shape differs")
        if not value.is_floating_point():
            raise ODEBFContractError(f"{name} must use a floating dtype")
        try:
            all_finite = bool(torch.isfinite(value).all().detach().cpu())
        except (RuntimeError, TypeError) as exc:
            raise ODEBFContractError(f"{name} must be finite") from exc
        if not all_finite:
            raise ODEBFContractError(f"{name} must be finite")
        values = tuple(
            float(item)
            for item in value.detach().to(device="cpu", dtype=torch.float64).tolist()
        )
    else:
        values = _validated_sequence(name, value)
        if len(values) != len(BG_SOFT_LAYER_ORDER):
            raise ODEBFContractError(f"{name} shape differs")
    if len(values) != len(BG_SOFT_LAYER_ORDER):
        raise ODEBFContractError(f"{name} shape differs")
    return tuple(finite(f"{name}[{index}]", item) for index, item in enumerate(values))


def _validated_nonnegative_scalar(name: str, value: Any) -> float:
    result = finite(name, value)
    if result < 0.0:
        raise ODEBFContractError(f"{name} must be nonnegative")
    return result


def _rank_order(values: Sequence[float], layer_order: Sequence[int]) -> list[int]:
    """Rank higher signed slopes first; layer identity resolves exact ties."""

    return [
        layer
        for _, layer in sorted(
            zip(values, layer_order, strict=True), key=lambda item: (-item[0], item[1])
        )
    ]


def _vector_payload(
    values: tuple[float, ...],
    p_max: float,
    layer_order: tuple[int, ...],
) -> dict[str, Any]:
    positive_mask = [value > 0.0 for value in values]
    values_tensor = torch.tensor(values, dtype=torch.float64)
    semantic = {
        "layer_order": list(layer_order),
        "ordered_signed_slopes": list(values),
    }
    return {
        "ordered_signed_slopes": list(values),
        "signed_slopes_sha256": canonical_hash(semantic),
        "signed_slopes_tensor_sha256": tensor_sha256(values_tensor),
        "positive_mask": positive_mask,
        "positive_count": sum(positive_mask),
        "rank_order": _rank_order(values, layer_order),
        "positive_rank_order": [
            layer
            for layer in _rank_order(values, layer_order)
            if values[layer_order.index(layer)] > 0.0
        ],
        "p_max": p_max,
    }


def signed_slope_comparison_receipt(
    overlay_signed_slopes: Sequence[float] | torch.Tensor,
    no_hook_signed_slopes: Sequence[float] | torch.Tensor,
    overlay_p_max: Any,
    no_hook_p_max: Any,
    *,
    layer_order: Sequence[int] = BG_SOFT_LAYER_ORDER,
) -> dict[str, Any]:
    """Describe overlay and no-hook signed slopes without making a decision.

    ``FALSE_FIELD_COLLAPSE_OVERLAY_OBJECTIVE_SATURATION`` is an observation
    only: the overlay maximum progress is nonpositive while the matched
    no-hook maximum progress remains positive.  Slope signs are retained as
    diagnostic evidence, but never alter this classification.  No caller
    state is consulted or changed by this helper.
    """

    order = _validated_layer_order(layer_order)
    overlay_values = _validated_vector("overlay signed slopes", overlay_signed_slopes)
    no_hook_values = _validated_vector("no-hook signed slopes", no_hook_signed_slopes)
    overlay_maximum = _validated_nonnegative_scalar("overlay p_max", overlay_p_max)
    no_hook_maximum = _validated_nonnegative_scalar("no-hook p_max", no_hook_p_max)

    overlay = _vector_payload(overlay_values, overlay_maximum, order)
    no_hook = _vector_payload(no_hook_values, no_hook_maximum, order)
    false_field_collapse = bool(
        overlay_maximum <= 0.0 and no_hook_maximum > 0.0
    )
    classification = (
        FALSE_FIELD_COLLAPSE_OVERLAY_OBJECTIVE_SATURATION
        if false_field_collapse
        else _NO_FALSE_FIELD_COLLAPSE_OBSERVED
    )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-bg-soft-signed-slope-comparison/v1",
        "layer_order": list(order),
        "overlay": overlay,
        "no_hook": no_hook,
        "overlay_signed_slopes": overlay["ordered_signed_slopes"],
        "no_hook_signed_slopes": no_hook["ordered_signed_slopes"],
        "overlay_signed_slopes_sha256": overlay["signed_slopes_sha256"],
        "no_hook_signed_slopes_sha256": no_hook["signed_slopes_sha256"],
        "overlay_positive_mask": overlay["positive_mask"],
        "no_hook_positive_mask": no_hook["positive_mask"],
        "overlay_positive_count": overlay["positive_count"],
        "no_hook_positive_count": no_hook["positive_count"],
        "overlay_rank_order": overlay["rank_order"],
        "no_hook_rank_order": no_hook["rank_order"],
        "overlay_p_max": overlay_maximum,
        "no_hook_p_max": no_hook_maximum,
        "classification": classification,
        "false_field_collapse_overlay_objective_saturation": false_field_collapse,
        "classification_basis": {
            "overlay_p_max_is_nonpositive": overlay_maximum <= 0.0,
            "no_hook_p_max_is_positive": no_hook_maximum > 0.0,
        },
        "observation_only": True,
        "classification_decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _validated_matrix(name: str, value: Any) -> tuple[torch.Tensor, tuple[int, int]]:
    if not isinstance(value, torch.Tensor):
        raise ODEBFContractError(f"{name} must be a tensor matrix")
    if value.layout != torch.strided or value.ndim != 2 or value.numel() == 0:
        raise ODEBFContractError(f"{name} shape differs")
    if not value.is_floating_point():
        raise ODEBFContractError(f"{name} must use a floating dtype")
    try:
        all_finite = bool(torch.isfinite(value).all().detach().cpu())
    except (RuntimeError, TypeError) as exc:
        raise ODEBFContractError(f"{name} must be finite") from exc
    if not all_finite:
        raise ODEBFContractError(f"{name} must be finite")
    matrix = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return matrix, (int(value.shape[0]), int(value.shape[1]))


def _finite_metric(name: str, value: Any) -> float:
    return finite(name, value)


def _cosine_or_not_defined(
    dot: float,
    left_norm: float,
    right_norm: float,
    *,
    name: str,
) -> float | str:
    if left_norm == 0.0 or right_norm == 0.0:
        return NOT_DEFINED
    return _finite_metric(name, dot / (left_norm * right_norm))


def single_write_audit_summary(
    intended_delta: torch.Tensor,
    per_layer_realized_deltas: Sequence[torch.Tensor],
    all_layer_delta: torch.Tensor,
    *,
    layer_order: Sequence[int] = BG_SOFT_LAYER_ORDER,
) -> dict[str, Any]:
    """Return a numerical, raw-free audit of one matched all-layer write.

    Every matrix must be finite, nonempty, two-dimensional, and identically
    shaped.  The interaction error is ``||all_layer - sum(per_layer)||_F``;
    it is therefore an observation of cross-writer nonadditivity, not a gate.
    """

    order = _validated_layer_order(layer_order)
    realized_inputs = _validated_sequence(
        "per-layer realized deltas", per_layer_realized_deltas
    )
    if len(realized_inputs) != len(order):
        raise ODEBFContractError("per-layer realized delta count differs")

    intended, shape = _validated_matrix("intended delta", intended_delta)
    all_layer, all_layer_shape = _validated_matrix("all-layer delta", all_layer_delta)
    if all_layer_shape != shape:
        raise ODEBFContractError("all-layer delta shape differs")
    realized: list[torch.Tensor] = []
    for layer, matrix in zip(order, realized_inputs, strict=True):
        normalized, observed_shape = _validated_matrix(
            f"realized delta layer {layer}", matrix
        )
        if observed_shape != shape:
            raise ODEBFContractError("per-layer realized delta shape differs")
        realized.append(normalized)

    intended_norm = _finite_metric(
        "intended delta Frobenius norm", torch.linalg.vector_norm(intended)
    )
    all_layer_norm = _finite_metric(
        "all-layer delta Frobenius norm", torch.linalg.vector_norm(all_layer)
    )
    realized_norms = [
        _finite_metric(
            f"realized delta layer {layer} Frobenius norm",
            torch.linalg.vector_norm(matrix),
        )
        for layer, matrix in zip(order, realized, strict=True)
    ]
    flattened = torch.stack([matrix.reshape(-1) for matrix in realized], dim=0)
    raw_gram_tensor = flattened @ flattened.T
    if not bool(torch.isfinite(raw_gram_tensor).all()):
        raise ODEBFContractError("single-write raw Gram is non-finite")
    raw_gram = [
        [_finite_metric("single-write raw Gram", item) for item in row]
        for row in raw_gram_tensor.tolist()
    ]
    normalized_cosine_gram = [
        [
            _cosine_or_not_defined(
                raw_gram[row_index][column_index],
                realized_norms[row_index],
                realized_norms[column_index],
                name="single-write normalized cosine Gram",
            )
            for column_index in range(len(order))
        ]
        for row_index in range(len(order))
    ]
    additive_sum = torch.stack(realized, dim=0).sum(dim=0)
    interaction_delta = all_layer - additive_sum
    interaction_error_raw_norm = _finite_metric(
        "single-write interaction error raw norm",
        torch.linalg.vector_norm(interaction_delta),
    )
    interaction_error = _finite_metric(
        "single-write interaction error",
        interaction_error_raw_norm / (all_layer_norm + SINGLE_WRITE_INTERACTION_EPSILON),
    )

    target_metric_status = _DEFINED if intended_norm > 0.0 else NOT_DEFINED
    per_layer: list[dict[str, Any]] = []
    per_layer_cosine: list[float | str] = []
    per_layer_norm_gain: list[float | str] = []
    for layer, matrix, realized_norm, source in zip(
        order, realized, realized_norms, realized_inputs, strict=True
    ):
        dot = _finite_metric(
            f"realized delta layer {layer} dot intended",
            torch.sum(matrix * intended),
        )
        if intended_norm == 0.0:
            cosine: float | str = NOT_DEFINED
            norm_gain: float | str = NOT_DEFINED
            metric_status = NOT_DEFINED
        else:
            cosine = _cosine_or_not_defined(
                dot,
                realized_norm,
                intended_norm,
                name=f"realized delta layer {layer} cosine",
            )
            norm_gain = _finite_metric(
                f"realized delta layer {layer} norm gain",
                realized_norm / intended_norm,
            )
            metric_status = (
                _DEFINED
                if cosine != NOT_DEFINED
                else _ZERO_REALIZED_COSINE_NOT_DEFINED
            )
        per_layer_cosine.append(cosine)
        per_layer_norm_gain.append(norm_gain)
        per_layer.append(
            {
                "layer": layer,
                "realized_delta_sha256": tensor_sha256(source),
                "realized_frobenius_norm": realized_norm,
                "cosine_to_intended": cosine,
                "norm_gain_to_intended": norm_gain,
                "target_metric_status": metric_status,
            }
        )

    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-bg-soft-single-write-audit/v1",
        "layer_order": list(order),
        "matrix_shape": list(shape),
        "intended_delta_sha256": tensor_sha256(intended_delta),
        "all_layer_delta_sha256": tensor_sha256(all_layer_delta),
        "intended_frobenius_norm": intended_norm,
        "all_layer_frobenius_norm": all_layer_norm,
        "target_metric_status": target_metric_status,
        "not_defined_token": NOT_DEFINED,
        "per_layer": per_layer,
        "per_layer_cosine_to_intended": per_layer_cosine,
        "per_layer_norm_gain_to_intended": per_layer_norm_gain,
        "raw_gram": raw_gram,
        "raw_gram_sha256": canonical_hash(raw_gram),
        "normalized_cosine_gram": normalized_cosine_gram,
        "normalized_cosine_gram_sha256": canonical_hash(normalized_cosine_gram),
        "interaction_delta_sha256": tensor_sha256(interaction_delta),
        "interaction_error": interaction_error,
        "interaction_error_normalized": interaction_error,
        "interaction_error_raw_norm": interaction_error_raw_norm,
        "interaction_error_epsilon": SINGLE_WRITE_INTERACTION_EPSILON,
        "interaction_error_definition": (
            "frobenius_norm(all_layer_delta-sum(per_layer_realized_deltas))/"
            "(frobenius_norm(all_layer_delta)+interaction_error_epsilon)"
        ),
        "observation_only": True,
        "controller_decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _validated_target_tensor(name: str, value: Any) -> tuple[tuple[int, ...], torch.dtype]:
    if not isinstance(value, torch.Tensor):
        raise ODEBFContractError(f"{name} must be a tensor")
    if value.layout != torch.strided or value.ndim == 0 or value.numel() == 0:
        raise ODEBFContractError(f"{name} shape differs")
    if not value.is_floating_point():
        raise ODEBFContractError(f"{name} must use a floating dtype")
    try:
        all_finite = bool(torch.isfinite(value).all().detach().cpu())
    except (RuntimeError, TypeError) as exc:
        raise ODEBFContractError(f"{name} must be finite") from exc
    if not all_finite:
        raise ODEBFContractError(f"{name} must be finite")
    return tuple(int(item) for item in value.shape), value.dtype


def _exact_integer(name: str, value: Any, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ODEBFContractError(f"{name} differs from the fixed E8 lock")
    return value


def _exact_scalar(name: str, value: Any, expected: float) -> float:
    observed = finite(name, value)
    if observed != expected:
        raise ODEBFContractError(f"{name} differs from the fixed E8 lock")
    return observed


def target_hold_clock_receipt(
    bootstrap_target: torch.Tensor,
    target_states: Sequence[torch.Tensor],
    *,
    weight_step_count: int = FIXED_K,
    h: Any = 1.0 / FIXED_K,
    tau_w: Any = 1.0,
    target_advance_count: int = 0,
    scientific_retry_count: int = 0,
) -> dict[str, Any]:
    """Certify an invariant bootstrap target across the locked E8 weight clock.

    ``target_states`` contains the target observed after each weight step, so
    it must have exactly eight entries.  Inputs such as a changed target,
    a retry, a target advance, or a different fixed-grid clock fail closed.
    """

    _exact_integer("weight step count", weight_step_count, FIXED_K)
    fixed_h = _exact_scalar("weight h", h, 1.0 / FIXED_K)
    fixed_tau_w = _exact_scalar("weight tau_W", tau_w, 1.0)
    _exact_integer("target advance count", target_advance_count, 0)
    _exact_integer("scientific retry count", scientific_retry_count, 0)
    states = _validated_sequence("target states", target_states)
    if len(states) != FIXED_K:
        raise ODEBFContractError("target-hold state count differs from K8")

    bootstrap_shape, bootstrap_dtype = _validated_target_tensor(
        "bootstrap target", bootstrap_target
    )
    bootstrap_hash = tensor_sha256(bootstrap_target)
    target_hashes: list[str] = []
    for step_index, target in enumerate(states):
        shape, dtype = _validated_target_tensor(
            f"target state {step_index}", target
        )
        if shape != bootstrap_shape or dtype != bootstrap_dtype:
            raise ODEBFContractError("target-hold target shape or dtype differs")
        target_hash = tensor_sha256(target)
        if target_hash != bootstrap_hash:
            raise ODEBFContractError("bootstrap target hash invariant differs")
        target_hashes.append(target_hash)

    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-bg-soft-target-hold-clock/v1",
        "status": TARGET_HOLD_ACTIVE,
        "bootstrap_target_sha256": bootstrap_hash,
        "target_sha256_by_weight_step": target_hashes,
        "target_hash_invariant": True,
        "target_hash_unique_count": 1,
        "target_shape": list(bootstrap_shape),
        "target_dtype": str(bootstrap_dtype),
        "weight_step_count": FIXED_K,
        "weight_h": fixed_h,
        "weight_h_fraction": {"numerator": 1, "denominator": FIXED_K},
        "tau_W": fixed_tau_w,
        "weight_tau_final": fixed_tau_w,
        "target_advance_count": 0,
        "target_tau_final": 0.0,
        "scientific_retry_count": 0,
        "scientific_rejection_count": 0,
        "observation_only": True,
        "controller_decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _unwrap_hook_output(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor):
        raise ODEBFContractError("held-out residual overlay output is not a tensor")
    return value


def _rewrap_hook_output(output: Any, patched: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (patched, *output[1:])
    return patched


def _validated_positions_by_forward(
    value: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], ...]:
    forwards = _validated_sequence("held-out lookup positions", value)
    if len(forwards) != BATCH_SIZE:
        raise ODEBFContractError("held-out overlay forward count differs from B10")
    normalized: list[tuple[int, ...]] = []
    for forward_index, rows in enumerate(forwards):
        row_positions = _validated_sequence(
            f"held-out lookup positions forward {forward_index}", rows
        )
        if not row_positions:
            raise ODEBFContractError("held-out overlay lookup rows are empty")
        positions: list[int] = []
        for position in row_positions:
            if isinstance(position, bool) or not isinstance(position, int):
                raise ODEBFContractError(
                    "held-out overlay lookup position is not an integer"
                )
            positions.append(position)
        normalized.append(tuple(positions))
    return tuple(normalized)


def _validated_prefix_counts(
    value: Sequence[int], positions: Sequence[Sequence[int]]
) -> tuple[int, ...]:
    counts = _validated_sequence("held-out patched-prefix row counts", value)
    if len(counts) != BATCH_SIZE:
        raise ODEBFContractError("held-out overlay prefix count differs from B10")
    normalized: list[int] = []
    for count, row_positions in zip(counts, positions, strict=True):
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or count > len(row_positions)
        ):
            raise ODEBFContractError(
                "held-out overlay patched-prefix row count differs"
            )
        normalized.append(count)
    return tuple(normalized)


def _validated_activation_layout_policy(value: Any) -> str:
    if value not in ("BATCH_FIRST", "SEQUENCE_FIRST"):
        raise ODEBFContractError("held-out overlay activation layout policy differs")
    return str(value)


class HeldoutRequestResidualActivationOverlay:
    """Patch a B10 held-out evaluator with request-wise additive residuals.

    Each evaluator forward corresponds to one request ordinal.  The caller
    supplies lookup positions for every row in that forward and the leading
    number of rows that belong to rewrite plus paraphrase evaluation.  Rows
    after that prefix are locality rows and remain byte-identical to the
    unhooked activation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_name: str,
        residual: torch.Tensor,
        lookup_positions_by_forward: Sequence[Sequence[int]],
        patched_prefix_row_counts: Sequence[int],
        *,
        activation_layout_policy: str = "BATCH_FIRST",
    ) -> None:
        if (
            not isinstance(model, torch.nn.Module)
            or not isinstance(layer_name, str)
            or not layer_name
        ):
            raise ODEBFContractError("held-out residual overlay target differs")
        if (
            not isinstance(residual, torch.Tensor)
            or residual.layout != torch.strided
            or residual.ndim != 2
            or residual.shape[0] <= 0
            or residual.shape[1] != BATCH_SIZE
            or not residual.is_floating_point()
        ):
            raise ODEBFContractError("held-out residual overlay geometry differs")
        try:
            residual_finite = bool(torch.isfinite(residual).all().detach().cpu())
        except (RuntimeError, TypeError) as exc:
            raise ODEBFContractError(
                "held-out residual overlay must be finite"
            ) from exc
        if not residual_finite:
            raise ODEBFContractError("held-out residual overlay must be finite")
        positions = _validated_positions_by_forward(lookup_positions_by_forward)
        prefix_counts = _validated_prefix_counts(patched_prefix_row_counts, positions)
        self.model = model
        self.layer_name = layer_name
        self.residual = residual
        self.lookup_positions_by_forward = positions
        self.patched_prefix_row_counts = prefix_counts
        self.activation_layout_policy = _validated_activation_layout_policy(
            activation_layout_policy
        )
        self.calls = 0
        self.request_ordinals: list[int] = []
        self.layouts: list[str] = []
        self.patched_row_count = 0
        self.locality_no_hook_row_count = 0
        self.maximum_exact_delta_error = 0.0
        self.maximum_authoritative_assignment_error = 0.0
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    @property
    def hook_active(self) -> bool:
        return self._handle is not None

    def _activation_layout(
        self, activation: torch.Tensor, expected_rows: int
    ) -> tuple[str, int]:
        if (
            activation.layout != torch.strided
            or activation.ndim != 3
            or activation.shape[-1] <= 0
            or not activation.is_floating_point()
        ):
            raise ODEBFContractError(
                "held-out residual overlay activation layout differs"
            )
        try:
            all_finite = bool(torch.isfinite(activation).all().detach().cpu())
        except (RuntimeError, TypeError) as exc:
            raise ODEBFContractError(
                "held-out residual overlay activation must be finite"
            ) from exc
        if not all_finite:
            raise ODEBFContractError("held-out residual overlay activation must be finite")
        if self.activation_layout_policy == "BATCH_FIRST":
            if activation.shape[0] != expected_rows:
                raise ODEBFContractError(
                    "held-out residual overlay batch-first layout differs"
                )
            return "BATCH_FIRST", int(activation.shape[1])
        if activation.shape[1] != expected_rows:
            raise ODEBFContractError(
                "held-out residual overlay sequence-first layout differs"
            )
        return "SEQUENCE_FIRST", int(activation.shape[0])

    @staticmethod
    def _normalize_position(position: int, sequence_length: int) -> int:
        normalized = position if position >= 0 else sequence_length + position
        if normalized < 0 or normalized >= sequence_length:
            raise ODEBFContractError("held-out overlay lookup is out of range")
        return normalized

    @staticmethod
    def _assignment_errors(
        before: torch.Tensor, residual: torch.Tensor, assigned: torch.Tensor
    ) -> tuple[float, float]:
        if (
            before.shape != residual.shape
            or before.shape != assigned.shape
            or before.dtype != residual.dtype
            or before.dtype != assigned.dtype
            or before.device != residual.device
            or before.device != assigned.device
        ):
            raise ODEBFContractError("held-out overlay assignment geometry differs")
        expected = before + residual
        if not bool(torch.isfinite(expected).all()) or not bool(
            torch.isfinite(assigned).all()
        ):
            raise ODEBFContractError("held-out overlay assignment is non-finite")
        if not torch.equal(assigned, expected):
            raise ODEBFContractError("held-out overlay assignment is not additive")
        authoritative_error = _finite_metric(
            "held-out authoritative assignment error",
            torch.max(torch.abs(assigned.float() - expected.float())),
        )
        realized_delta_error = _finite_metric(
            "held-out realized additive delta error",
            torch.max(torch.abs((assigned - before).float() - residual.float())),
        )
        return authoritative_error, realized_delta_error

    def _hook(self, _module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
        if self.calls >= BATCH_SIZE:
            raise ODEBFContractError("held-out residual overlay received extra forward")
        activation = _unwrap_hook_output(output)
        forward_index = self.calls
        positions = self.lookup_positions_by_forward[forward_index]
        prefix_count = self.patched_prefix_row_counts[forward_index]
        layout, sequence_length = self._activation_layout(activation, len(positions))
        if activation.shape[-1] != self.residual.shape[0]:
            raise ODEBFContractError("held-out residual overlay hidden size differs")
        try:
            request_residual = self.residual[:, forward_index].detach().to(
                device=activation.device, dtype=activation.dtype
            )
        except (RuntimeError, TypeError) as exc:
            raise ODEBFContractError(
                "held-out residual overlay device or dtype differs"
            ) from exc
        if (
            request_residual.device != activation.device
            or request_residual.dtype != activation.dtype
            or request_residual.ndim != 1
            or request_residual.numel() != activation.shape[-1]
            or not bool(torch.isfinite(request_residual).all())
        ):
            raise ODEBFContractError("held-out residual overlay device or dtype differs")

        normalized_positions = tuple(
            self._normalize_position(position, sequence_length)
            for position in positions
        )
        patched = activation.clone()
        for row_index in range(prefix_count):
            position = normalized_positions[row_index]
            before = (
                activation[row_index, position, :]
                if layout == "BATCH_FIRST"
                else activation[position, row_index, :]
            )
            expected = before + request_residual
            if layout == "BATCH_FIRST":
                patched[row_index, position, :] = expected
                assigned = patched[row_index, position, :]
            else:
                patched[position, row_index, :] = expected
                assigned = patched[position, row_index, :]
            authoritative_error, realized_delta_error = self._assignment_errors(
                before, request_residual, assigned
            )
            self.maximum_authoritative_assignment_error = max(
                self.maximum_authoritative_assignment_error, authoritative_error
            )
            self.maximum_exact_delta_error = max(
                self.maximum_exact_delta_error, realized_delta_error
            )

        if prefix_count < len(positions):
            locality_original = (
                activation[prefix_count:, :, :]
                if layout == "BATCH_FIRST"
                else activation[:, prefix_count:, :]
            )
            locality_patched = (
                patched[prefix_count:, :, :]
                if layout == "BATCH_FIRST"
                else patched[:, prefix_count:, :]
            )
            if not torch.equal(locality_original, locality_patched):
                raise ODEBFContractError("held-out overlay touched locality rows")
        self.layouts.append(layout)
        self.request_ordinals.append(forward_index)
        self.patched_row_count += prefix_count
        self.locality_no_hook_row_count += len(positions) - prefix_count
        self.calls += 1
        return _rewrap_hook_output(output, patched)

    def __enter__(self) -> "HeldoutRequestResidualActivationOverlay":
        if self._handle is not None:
            raise ODEBFContractError("held-out residual overlay is already active")
        try:
            module = self.model.get_submodule(self.layer_name)
            self._handle = module.register_forward_hook(self._hook)
        except (AttributeError, RuntimeError) as exc:
            self._handle = None
            raise ODEBFContractError("held-out residual overlay target differs") from exc
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def assert_complete(self) -> None:
        expected = tuple(range(BATCH_SIZE))
        if (
            self._handle is not None
            or self.calls != BATCH_SIZE
            or tuple(self.request_ordinals) != expected
            or len(self.layouts) != BATCH_SIZE
        ):
            raise ODEBFContractError("held-out residual overlay call/cleanup differs")

    def raw_free_payload(self) -> dict[str, Any]:
        """Return only geometry, hashes, and assignment observations."""

        self.assert_complete()
        payload: dict[str, Any] = {
            "schema": "ode-edit-s05-bg-soft-heldout-residual-overlay/v1",
            "mode": "REQUEST_WISE_ADDITIVE_HELDOUT_PREFIX",
            "request_count": BATCH_SIZE,
            "hook_call_count": self.calls,
            "request_ordinal_sha256": canonical_hash(self.request_ordinals),
            "residual_sha256": tensor_sha256(self.residual),
            "residual_shape": list(self.residual.shape),
            "lookup_plan_sha256": canonical_hash(
                [list(item) for item in self.lookup_positions_by_forward]
            ),
            "rows_per_forward": [
                len(item) for item in self.lookup_positions_by_forward
            ],
            "patched_prefix_row_counts": list(self.patched_prefix_row_counts),
            "activation_layout_policy": self.activation_layout_policy,
            "patched_row_count": self.patched_row_count,
            "locality_no_hook_row_count": self.locality_no_hook_row_count,
            "layouts": list(self.layouts),
            "layouts_sha256": canonical_hash(self.layouts),
            "authoritative_assignment_definition": "patched=before+request_residual",
            "maximum_authoritative_assignment_error": (
                self.maximum_authoritative_assignment_error
            ),
            "maximum_exact_delta_error": self.maximum_exact_delta_error,
            "realized_delta_error_role": "ROUNDING_OBSERVATION_ONLY",
            "absolute_replacement_count": 0,
            "locality_hook_count": 0,
            "observation_only": True,
            "controller_decision_influence_count": 0,
        }
        payload["identity_sha256"] = canonical_hash(payload)
        return payload


__all__ = [
    "BG_SOFT_LAYER_ORDER",
    "FALSE_FIELD_COLLAPSE_OVERLAY_OBJECTIVE_SATURATION",
    "HeldoutRequestResidualActivationOverlay",
    "NOT_DEFINED",
    "TARGET_HOLD_ACTIVE",
    "signed_slope_comparison_receipt",
    "single_write_audit_summary",
    "target_hold_clock_receipt",
]
