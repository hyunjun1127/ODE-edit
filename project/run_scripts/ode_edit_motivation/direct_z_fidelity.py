"""Pure helpers for direct-z intervention and fidelity diagnostics.

The functions in this module deliberately do not import EasyEdit and do not
mutate model parameters.  They provide the small amount of hook plumbing and
scalar reduction needed to compare a weight edit with the corresponding
direct hidden-state intervention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import torch

from .contracts import ContractError, LowRankFactor, MemitFactorProposal


class DirectZFidelityError(ContractError):
    """A direct-z diagnostic input violates its measurement contract."""


LayerOutput = torch.Tensor | tuple[Any, ...]
PatchMode = Literal["add", "replace"]


def unwrap_layer_output(output: LayerOutput) -> torch.Tensor:
    """Return the activation tensor from a tensor or tensor-first tuple."""

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise DirectZFidelityError(
        "layer output must be a tensor or a non-empty tensor-first tuple"
    )


def rewrap_layer_output(original: LayerOutput, updated: torch.Tensor) -> LayerOutput:
    """Replace the tensor payload while preserving a supported output wrapper."""

    current = unwrap_layer_output(original)
    if not isinstance(updated, torch.Tensor):
        raise DirectZFidelityError("updated layer output must be a tensor")
    if updated.shape != current.shape:
        raise DirectZFidelityError(
            f"updated layer output shape {tuple(updated.shape)} differs from "
            f"the original {tuple(current.shape)}"
        )
    if isinstance(original, torch.Tensor):
        return updated

    values = (updated, *original[1:])
    if type(original) is tuple:
        return values
    if hasattr(original, "_fields"):
        # Named tuples use positional construction rather than tuple(iterable).
        try:
            return type(original)(*values)
        except TypeError as exc:
            raise DirectZFidelityError(
                "tensor-first named-tuple output could not be reconstructed"
            ) from exc
    try:
        return type(original)(values)
    except TypeError as exc:
        raise DirectZFidelityError(
            "tensor-first tuple subclass could not be reconstructed"
        ) from exc


def _normalize_dimension(dimension: int, ndim: int, *, name: str) -> int:
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise DirectZFidelityError(f"{name} must be an integer")
    normalized = dimension + ndim if dimension < 0 else dimension
    if normalized < 0 or normalized >= ndim:
        raise DirectZFidelityError(f"{name}={dimension} is invalid for ndim={ndim}")
    return normalized


def _normalize_index(index: int, size: int, *, name: str) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise DirectZFidelityError(f"{name} must be an integer")
    normalized = index + size if index < 0 else index
    if normalized < 0 or normalized >= size:
        raise DirectZFidelityError(f"{name}={index} is invalid for size={size}")
    return normalized


@dataclass(frozen=True, slots=True)
class BatchPositionPatch:
    """Clone-and-patch one sequence position per batch item.

    ``values`` may be one shared ``[hidden]`` vector or a ``[batch, hidden]``
    matrix.  The callable follows the ``edit_output(output, layer_name)``
    convention used by repository-local tracing hooks.  A non-matching layer
    is returned by identity, while a matching layer is cloned before patching.
    """

    layer_name: str
    positions: tuple[int, ...]
    values: torch.Tensor
    mode: PatchMode = "add"
    batch_dim: int = 0
    sequence_dim: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.layer_name, str) or not self.layer_name:
            raise DirectZFidelityError("patch layer_name must not be empty")
        try:
            positions = tuple(self.positions)
        except TypeError as exc:
            raise DirectZFidelityError("patch positions must be a sequence") from exc
        if not positions:
            raise DirectZFidelityError("patch positions must not be empty")
        for position in positions:
            if isinstance(position, bool) or not isinstance(position, int):
                raise DirectZFidelityError("patch positions must contain only integers")
        if self.mode not in ("add", "replace"):
            raise DirectZFidelityError("patch mode must be 'add' or 'replace'")
        if not isinstance(self.values, torch.Tensor):
            raise DirectZFidelityError("patch values must be a tensor")
        if self.values.ndim not in (1, 2):
            raise DirectZFidelityError("patch values must have shape [H] or [B, H]")
        if not self.values.is_floating_point():
            raise DirectZFidelityError("patch values must be floating point")
        if not bool(torch.isfinite(self.values).all()):
            raise DirectZFidelityError("patch values must be finite")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "values", self.values.detach().clone())

    def __call__(self, output: LayerOutput, layer_name: str) -> LayerOutput:
        if layer_name != self.layer_name:
            return output

        activation = unwrap_layer_output(output)
        if activation.ndim != 3:
            raise DirectZFidelityError(
                "patched activation must be rank three with batch, sequence, and hidden axes"
            )
        if not activation.is_floating_point():
            raise DirectZFidelityError("patched activation must be floating point")
        batch_dim = _normalize_dimension(self.batch_dim, 3, name="batch_dim")
        sequence_dim = _normalize_dimension(
            self.sequence_dim, 3, name="sequence_dim"
        )
        if batch_dim == sequence_dim:
            raise DirectZFidelityError("batch_dim and sequence_dim must differ")

        patched = activation.clone()
        batch_sequence_hidden = patched.movedim(
            (batch_dim, sequence_dim),
            (0, 1),
        )
        batch_size, sequence_length, hidden_size = batch_sequence_hidden.shape
        if len(self.positions) != batch_size:
            raise DirectZFidelityError(
                f"received {len(self.positions)} positions for batch size {batch_size}"
            )
        normalized_positions = tuple(
            _normalize_index(position, sequence_length, name="sequence position")
            for position in self.positions
        )
        values = self.values.to(device=activation.device, dtype=activation.dtype)
        if values.ndim == 1:
            if values.shape[0] != hidden_size:
                raise DirectZFidelityError(
                    f"shared patch width {values.shape[0]} differs from hidden size "
                    f"{hidden_size}"
                )
            values = values.unsqueeze(0).expand(batch_size, -1)
        elif values.shape not in ((1, hidden_size), (batch_size, hidden_size)):
            raise DirectZFidelityError(
                f"batched patch shape {tuple(values.shape)} is incompatible with "
                f"batch={batch_size}, hidden={hidden_size}"
            )
        elif values.shape[0] == 1:
            values = values.expand(batch_size, -1)

        batch_indices = torch.arange(batch_size, device=activation.device)
        position_indices = torch.tensor(
            normalized_positions,
            device=activation.device,
            dtype=torch.long,
        )
        if self.mode == "add":
            batch_sequence_hidden[batch_indices, position_indices] = (
                batch_sequence_hidden[batch_indices, position_indices] + values
            )
        else:
            batch_sequence_hidden[batch_indices, position_indices] = values
        return rewrap_layer_output(output, patched)


def make_batch_position_patch(
    layer_name: str,
    positions: Sequence[int],
    values: torch.Tensor,
    *,
    mode: PatchMode = "add",
    batch_dim: int = 0,
    sequence_dim: int = 1,
) -> BatchPositionPatch:
    """Construct a :class:`BatchPositionPatch` from ordinary sequences."""

    return BatchPositionPatch(
        layer_name=layer_name,
        positions=tuple(positions),
        values=values,
        mode=mode,
        batch_dim=batch_dim,
        sequence_dim=sequence_dim,
    )


def _finite_float(value: float, *, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise DirectZFidelityError(f"{name} must be finite")
    return converted


def _validate_scalar_dataclass(instance: object) -> None:
    for name in instance.__dataclass_fields__:  # type: ignore[attr-defined]
        value = getattr(instance, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DirectZFidelityError(f"{name} must be a scalar int or float")
        if isinstance(value, float) and not math.isfinite(value):
            raise DirectZFidelityError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class SharedDeltaFidelity:
    """Scalar-only summary of subject-state fidelity to one shared delta."""

    context_count: int
    generated_count: int
    delta_l2: float
    canonical_target_error_l2: float
    canonical_target_error_relative: float
    canonical_gain: float
    canonical_cosine: float
    canonical_parallel_error_l2: float
    canonical_orthogonal_error_l2: float
    generated_mean_target_error_l2: float
    generated_worst_target_error_l2: float
    generated_mean_target_error_relative: float
    generated_worst_target_error_relative: float
    generated_mean_gain: float
    generated_worst_absolute_gain_error: float
    generated_mean_cosine: float
    generated_worst_cosine: float
    generated_mean_parallel_error_l2: float
    generated_worst_parallel_error_l2: float
    generated_mean_orthogonal_error_l2: float
    generated_worst_orthogonal_error_l2: float

    def __post_init__(self) -> None:
        _validate_scalar_dataclass(self)
        if self.context_count < 2 or self.generated_count != self.context_count - 1:
            raise DirectZFidelityError("shared-delta context counts are inconsistent")

    def to_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _as_finite_float64(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise DirectZFidelityError(f"{name} must be a tensor")
    if not tensor.is_floating_point():
        raise DirectZFidelityError(f"{name} must be floating point")
    if not bool(torch.isfinite(tensor).all()):
        raise DirectZFidelityError(f"{name} must be finite")
    return tensor.detach().to(device="cpu", dtype=torch.float64)


def _positive_epsilon(epsilon: float) -> float:
    epsilon = _finite_float(epsilon, name="epsilon")
    if epsilon <= 0.0:
        raise DirectZFidelityError("epsilon must be positive")
    return epsilon


def _context_update_statistics(
    actual_updates: torch.Tensor,
    shared_delta: torch.Tensor,
    *,
    epsilon: float,
) -> Mapping[str, torch.Tensor]:
    delta_l2 = torch.linalg.vector_norm(shared_delta)
    delta_sq = torch.dot(shared_delta, shared_delta)
    dots = actual_updates @ shared_delta
    gains = dots / delta_sq
    actual_l2 = torch.linalg.vector_norm(actual_updates, dim=1)
    cosine_denominator = actual_l2 * delta_l2
    cosines = torch.where(
        cosine_denominator > epsilon,
        dots / cosine_denominator.clamp_min(epsilon),
        torch.zeros_like(dots),
    ).clamp(-1.0, 1.0)
    parallel_errors = torch.abs(gains - 1.0) * delta_l2
    orthogonal_errors = torch.linalg.vector_norm(
        actual_updates - gains.unsqueeze(1) * shared_delta,
        dim=1,
    )
    return {
        "gain": gains,
        "cosine": cosines,
        "parallel": parallel_errors,
        "orthogonal": orthogonal_errors,
    }


def aggregate_shared_delta_fidelity(
    base_subject_states: torch.Tensor,
    edited_subject_states: torch.Tensor,
    shared_delta: torch.Tensor,
    *,
    canonical_index: int = 0,
    canonical_absolute_target: torch.Tensor | None = None,
    epsilon: float = 1e-12,
) -> SharedDeltaFidelity:
    """Compare subject states with ``base[c] + shared_delta``.

    The canonical context may optionally use an explicitly stored absolute
    direct-z target.  Generated contexts always use the shared-delta target,
    which separates target transport from prompt-specific base activations.
    Gain and angular diagnostics are computed from the realized update
    ``edited - base``.  A zero realized update has cosine zero by convention.
    """

    epsilon = _positive_epsilon(epsilon)
    base = _as_finite_float64(base_subject_states, name="base_subject_states")
    edited = _as_finite_float64(
        edited_subject_states, name="edited_subject_states"
    )
    delta = _as_finite_float64(shared_delta, name="shared_delta")
    if base.ndim != 2 or edited.shape != base.shape:
        raise DirectZFidelityError(
            "base and edited subject states must have the same [contexts, hidden] shape"
        )
    if delta.ndim != 1 or delta.shape[0] != base.shape[1]:
        raise DirectZFidelityError("shared_delta must have shape [hidden]")
    if base.shape[0] < 2:
        raise DirectZFidelityError(
            "shared-delta aggregation requires one canonical and one generated context"
        )
    canonical = _normalize_index(
        canonical_index, base.shape[0], name="canonical_index"
    )
    delta_l2_tensor = torch.linalg.vector_norm(delta)
    delta_l2 = float(delta_l2_tensor)
    if delta_l2 <= epsilon:
        raise DirectZFidelityError("shared_delta norm must exceed epsilon")

    desired = base + delta.unsqueeze(0)
    if canonical_absolute_target is not None:
        canonical_target = _as_finite_float64(
            canonical_absolute_target, name="canonical_absolute_target"
        )
        if canonical_target.shape != delta.shape:
            raise DirectZFidelityError(
                "canonical_absolute_target must have shape [hidden]"
            )
        desired[canonical] = canonical_target

    target_errors = torch.linalg.vector_norm(edited - desired, dim=1)
    relative_errors = target_errors / delta_l2_tensor
    stats = _context_update_statistics(
        edited - base,
        delta,
        epsilon=epsilon,
    )
    generated_indices = [index for index in range(base.shape[0]) if index != canonical]
    generated = torch.tensor(generated_indices, dtype=torch.long)

    generated_errors = target_errors[generated]
    generated_relative = relative_errors[generated]
    generated_gains = stats["gain"][generated]
    generated_cosines = stats["cosine"][generated]
    generated_parallel = stats["parallel"][generated]
    generated_orthogonal = stats["orthogonal"][generated]

    return SharedDeltaFidelity(
        context_count=int(base.shape[0]),
        generated_count=len(generated_indices),
        delta_l2=delta_l2,
        canonical_target_error_l2=float(target_errors[canonical]),
        canonical_target_error_relative=float(relative_errors[canonical]),
        canonical_gain=float(stats["gain"][canonical]),
        canonical_cosine=float(stats["cosine"][canonical]),
        canonical_parallel_error_l2=float(stats["parallel"][canonical]),
        canonical_orthogonal_error_l2=float(stats["orthogonal"][canonical]),
        generated_mean_target_error_l2=float(generated_errors.mean()),
        generated_worst_target_error_l2=float(generated_errors.max()),
        generated_mean_target_error_relative=float(generated_relative.mean()),
        generated_worst_target_error_relative=float(generated_relative.max()),
        generated_mean_gain=float(generated_gains.mean()),
        generated_worst_absolute_gain_error=float(
            torch.abs(generated_gains - 1.0).max()
        ),
        generated_mean_cosine=float(generated_cosines.mean()),
        generated_worst_cosine=float(generated_cosines.min()),
        generated_mean_parallel_error_l2=float(generated_parallel.mean()),
        generated_worst_parallel_error_l2=float(generated_parallel.max()),
        generated_mean_orthogonal_error_l2=float(generated_orthogonal.mean()),
        generated_worst_orthogonal_error_l2=float(generated_orthogonal.max()),
    )


@dataclass(frozen=True, slots=True)
class SequenceSpillFidelity:
    """Scalar-only subject-miss and off-token-spill summary."""

    context_count: int
    generated_count: int
    valid_token_count: int
    off_token_count: int
    delta_l2: float
    canonical_subject_miss_l2: float
    canonical_subject_miss_relative: float
    generated_mean_subject_miss_l2: float
    generated_worst_subject_miss_l2: float
    generated_mean_subject_miss_relative: float
    generated_worst_subject_miss_relative: float
    canonical_off_token_spill_rms: float
    canonical_off_token_spill_relative: float
    generated_mean_off_token_spill_rms: float
    generated_worst_off_token_spill_rms: float
    generated_mean_off_token_spill_relative: float
    generated_worst_off_token_spill_relative: float
    global_subject_miss_rms: float
    global_off_token_spill_rms: float
    full_sequence_intervention_error_rms: float
    full_sequence_intervention_error_relative: float

    def __post_init__(self) -> None:
        _validate_scalar_dataclass(self)
        if self.context_count < 2 or self.generated_count != self.context_count - 1:
            raise DirectZFidelityError("sequence context counts are inconsistent")
        if self.valid_token_count < self.context_count or self.off_token_count <= 0:
            raise DirectZFidelityError("sequence token counts are inconsistent")

    def to_dict(self) -> dict[str, int | float]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def aggregate_sequence_spill(
    base_sequence_states: torch.Tensor,
    edited_sequence_states: torch.Tensor,
    shared_delta: torch.Tensor,
    subject_positions: Sequence[int],
    *,
    canonical_index: int = 0,
    canonical_absolute_target: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    epsilon: float = 1e-12,
) -> SequenceSpillFidelity:
    """Measure miss at subject tokens and spill over all other valid tokens.

    Inputs use ``[contexts, sequence, hidden]`` layout.  The oracle sequence is
    the base sequence with exactly one subject-position intervention per
    context.  RMS metrics average squared vector norms per token, rather than
    per hidden coordinate, so they remain directly comparable to ``||delta||``.
    """

    epsilon = _positive_epsilon(epsilon)
    base = _as_finite_float64(base_sequence_states, name="base_sequence_states")
    edited = _as_finite_float64(
        edited_sequence_states, name="edited_sequence_states"
    )
    delta = _as_finite_float64(shared_delta, name="shared_delta")
    if base.ndim != 3 or edited.shape != base.shape:
        raise DirectZFidelityError(
            "base and edited sequence states must have the same "
            "[contexts, sequence, hidden] shape"
        )
    context_count, sequence_length, hidden_size = base.shape
    if context_count < 2:
        raise DirectZFidelityError(
            "sequence aggregation requires one canonical and one generated context"
        )
    if delta.ndim != 1 or delta.shape[0] != hidden_size:
        raise DirectZFidelityError("shared_delta must have shape [hidden]")
    try:
        raw_positions = tuple(subject_positions)
    except TypeError as exc:
        raise DirectZFidelityError("subject_positions must be a sequence") from exc
    if len(raw_positions) != context_count:
        raise DirectZFidelityError(
            "subject_positions must contain one position per context"
        )
    positions = tuple(
        _normalize_index(position, sequence_length, name="subject position")
        for position in raw_positions
    )
    canonical = _normalize_index(
        canonical_index, context_count, name="canonical_index"
    )
    delta_l2_tensor = torch.linalg.vector_norm(delta)
    delta_l2 = float(delta_l2_tensor)
    if delta_l2 <= epsilon:
        raise DirectZFidelityError("shared_delta norm must exceed epsilon")

    if attention_mask is None:
        valid_mask = torch.ones(
            (context_count, sequence_length),
            dtype=torch.bool,
        )
    else:
        if not isinstance(attention_mask, torch.Tensor):
            raise DirectZFidelityError("attention_mask must be a tensor")
        mask = attention_mask.detach().to(device="cpu")
        if mask.shape != (context_count, sequence_length):
            raise DirectZFidelityError(
                "attention_mask must have shape [contexts, sequence]"
            )
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise DirectZFidelityError("attention_mask values must be zero or one")
        valid_mask = mask.to(dtype=torch.bool)

    row_indices = torch.arange(context_count, dtype=torch.long)
    position_indices = torch.tensor(positions, dtype=torch.long)
    if not bool(valid_mask[row_indices, position_indices].all()):
        raise DirectZFidelityError("every subject position must be a valid token")
    off_mask = valid_mask.clone()
    off_mask[row_indices, position_indices] = False
    off_counts_by_context = off_mask.sum(dim=1)
    if not bool((off_counts_by_context > 0).all()):
        raise DirectZFidelityError(
            "every context needs at least one valid non-subject token"
        )

    desired = base.clone()
    desired[row_indices, position_indices] = (
        desired[row_indices, position_indices] + delta
    )
    if canonical_absolute_target is not None:
        canonical_target = _as_finite_float64(
            canonical_absolute_target, name="canonical_absolute_target"
        )
        if canonical_target.shape != delta.shape:
            raise DirectZFidelityError(
                "canonical_absolute_target must have shape [hidden]"
            )
        desired[canonical, positions[canonical]] = canonical_target

    errors = edited - desired
    squared_l2 = errors.square().sum(dim=2)
    subject_miss = torch.sqrt(squared_l2[row_indices, position_indices])
    subject_relative = subject_miss / delta_l2_tensor
    per_context_off_rms = torch.sqrt(
        (squared_l2 * off_mask).sum(dim=1) / off_counts_by_context
    )
    per_context_off_relative = per_context_off_rms / delta_l2_tensor

    generated_indices = [index for index in range(context_count) if index != canonical]
    generated = torch.tensor(generated_indices, dtype=torch.long)
    generated_subject = subject_miss[generated]
    generated_subject_relative = subject_relative[generated]
    generated_off = per_context_off_rms[generated]
    generated_off_relative = per_context_off_relative[generated]
    valid_token_count = int(valid_mask.sum())
    off_token_count = int(off_mask.sum())
    global_subject_rms = torch.sqrt(subject_miss.square().mean())
    global_off_rms = torch.sqrt(squared_l2[off_mask].mean())
    full_rms = torch.sqrt(squared_l2[valid_mask].mean())

    return SequenceSpillFidelity(
        context_count=context_count,
        generated_count=len(generated_indices),
        valid_token_count=valid_token_count,
        off_token_count=off_token_count,
        delta_l2=delta_l2,
        canonical_subject_miss_l2=float(subject_miss[canonical]),
        canonical_subject_miss_relative=float(subject_relative[canonical]),
        generated_mean_subject_miss_l2=float(generated_subject.mean()),
        generated_worst_subject_miss_l2=float(generated_subject.max()),
        generated_mean_subject_miss_relative=float(
            generated_subject_relative.mean()
        ),
        generated_worst_subject_miss_relative=float(
            generated_subject_relative.max()
        ),
        canonical_off_token_spill_rms=float(per_context_off_rms[canonical]),
        canonical_off_token_spill_relative=float(
            per_context_off_relative[canonical]
        ),
        generated_mean_off_token_spill_rms=float(generated_off.mean()),
        generated_worst_off_token_spill_rms=float(generated_off.max()),
        generated_mean_off_token_spill_relative=float(generated_off_relative.mean()),
        generated_worst_off_token_spill_relative=float(generated_off_relative.max()),
        global_subject_miss_rms=float(global_subject_rms),
        global_off_token_spill_rms=float(global_off_rms),
        full_sequence_intervention_error_rms=float(full_rms),
        full_sequence_intervention_error_relative=float(full_rms / delta_l2_tensor),
    )


@dataclass(frozen=True, slots=True)
class ProjectedRidgeSolution:
    """Tensor-free result of a nonnegative L2-ball ridge solve."""

    coefficients: tuple[float, ...]
    objective: float
    residual_l2: float
    coefficient_l2: float
    radius: float
    ridge: float
    iterations: int
    converged: bool
    projected_gradient_l2: float
    active_count: int

    def __post_init__(self) -> None:
        if not self.coefficients:
            raise DirectZFidelityError("ridge solution coefficients must not be empty")
        if any(not math.isfinite(value) or value < 0.0 for value in self.coefficients):
            raise DirectZFidelityError(
                "ridge solution coefficients must be finite and nonnegative"
            )
        for name in (
            "objective",
            "residual_l2",
            "coefficient_l2",
            "radius",
            "ridge",
            "projected_gradient_l2",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, float):
                raise DirectZFidelityError(f"{name} must be a float")
            if not math.isfinite(value) or value < 0.0:
                raise DirectZFidelityError(f"{name} must be finite and nonnegative")
        if isinstance(self.iterations, bool) or self.iterations < 0:
            raise DirectZFidelityError("iterations must be a nonnegative integer")
        if not isinstance(self.converged, bool):
            raise DirectZFidelityError("converged must be boolean")
        if (
            isinstance(self.active_count, bool)
            or self.active_count < 0
            or self.active_count > len(self.coefficients)
        ):
            raise DirectZFidelityError("active_count is inconsistent")
        if self.coefficient_l2 > self.radius + 1e-8 * max(1.0, self.radius):
            raise DirectZFidelityError("ridge solution violates its L2 radius")

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficients": list(self.coefficients),
            "objective": self.objective,
            "residual_l2": self.residual_l2,
            "coefficient_l2": self.coefficient_l2,
            "radius": self.radius,
            "ridge": self.ridge,
            "iterations": self.iterations,
            "converged": self.converged,
            "projected_gradient_l2": self.projected_gradient_l2,
            "active_count": self.active_count,
        }


def _project_nonnegative_l2_ball(vector: torch.Tensor, radius: float) -> torch.Tensor:
    projected = vector.clamp_min(0.0)
    norm = torch.linalg.vector_norm(projected)
    if float(norm) > radius:
        projected = projected * (radius / norm)
    return projected


def solve_nonnegative_l2_ball_ridge(
    response_matrix: torch.Tensor,
    target: torch.Tensor,
    *,
    radius: float,
    ridge: float = 1e-8,
    max_iterations: int = 20_000,
    tolerance: float = 1e-10,
) -> ProjectedRidgeSolution:
    r"""Solve a small convex response fit with projected gradient descent.

    The optimized objective is

    ``0.5 * ||response_matrix @ a - target||_2^2 + 0.5 * ridge * ||a||_2^2``

    subject to ``a >= 0`` and ``||a||_2 <= radius``.  The implementation is
    dimension-generic; :func:`solve_five_direction_ridge` pins the ODE-edit
    five-layer contract.
    """

    design = _as_finite_float64(response_matrix, name="response_matrix")
    desired = _as_finite_float64(target, name="target")
    if design.ndim != 2 or design.shape[0] == 0 or design.shape[1] == 0:
        raise DirectZFidelityError(
            "response_matrix must have non-zero shape [measurements, directions]"
        )
    if desired.ndim != 1 or desired.shape[0] != design.shape[0]:
        raise DirectZFidelityError(
            "target must have shape [measurements] matching response_matrix"
        )
    radius = _finite_float(radius, name="radius")
    ridge = _finite_float(ridge, name="ridge")
    tolerance = _finite_float(tolerance, name="tolerance")
    if radius <= 0.0:
        raise DirectZFidelityError("radius must be positive")
    if ridge < 0.0:
        raise DirectZFidelityError("ridge must be nonnegative")
    if tolerance <= 0.0:
        raise DirectZFidelityError("tolerance must be positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise DirectZFidelityError("max_iterations must be a positive integer")

    coefficient_count = design.shape[1]
    coefficients = torch.zeros(coefficient_count, dtype=torch.float64)
    spectral = torch.linalg.matrix_norm(design, ord=2)
    lipschitz = float(spectral.square()) + ridge
    converged = False
    iterations = 0
    projected_gradient_l2 = 0.0
    if lipschitz == 0.0:
        converged = True
    else:
        inverse_lipschitz = 1.0 / lipschitz
        for iterations in range(1, max_iterations + 1):
            residual = design @ coefficients - desired
            gradient = design.T @ residual + ridge * coefficients
            candidate = _project_nonnegative_l2_ball(
                coefficients - inverse_lipschitz * gradient,
                radius,
            )
            projected_gradient_l2 = float(
                torch.linalg.vector_norm(candidate - coefficients) * lipschitz
            )
            coefficients = candidate
            threshold = tolerance * max(
                1.0,
                lipschitz * float(torch.linalg.vector_norm(coefficients)),
            )
            if projected_gradient_l2 <= threshold:
                converged = True
                break

    # Recompute the gradient mapping at the returned point, including the
    # degenerate zero-design case, so the reported certificate is self-contained.
    residual = design @ coefficients - desired
    if lipschitz > 0.0:
        gradient = design.T @ residual + ridge * coefficients
        next_projected = _project_nonnegative_l2_ball(
            coefficients - gradient / lipschitz,
            radius,
        )
        projected_gradient_l2 = float(
            torch.linalg.vector_norm(next_projected - coefficients) * lipschitz
        )
    else:
        projected_gradient_l2 = 0.0
    residual_l2 = float(torch.linalg.vector_norm(residual))
    coefficient_l2 = float(torch.linalg.vector_norm(coefficients))
    objective = 0.5 * (
        residual_l2 * residual_l2 + ridge * coefficient_l2 * coefficient_l2
    )
    cleaned = tuple(max(0.0, float(value)) for value in coefficients)
    active_threshold = max(tolerance, 1e-14)
    return ProjectedRidgeSolution(
        coefficients=cleaned,
        objective=float(objective),
        residual_l2=residual_l2,
        coefficient_l2=coefficient_l2,
        radius=float(radius),
        ridge=float(ridge),
        iterations=iterations,
        converged=converged,
        projected_gradient_l2=projected_gradient_l2,
        active_count=sum(value > active_threshold for value in cleaned),
    )


def solve_five_direction_ridge(
    response_matrix: torch.Tensor,
    target: torch.Tensor,
    *,
    radius: float,
    ridge: float = 1e-8,
    max_iterations: int = 20_000,
    tolerance: float = 1e-10,
) -> ProjectedRidgeSolution:
    """Run the projected ridge solver with exactly five action coefficients."""

    if not isinstance(response_matrix, torch.Tensor) or response_matrix.ndim != 2:
        raise DirectZFidelityError("response_matrix must be a rank-two tensor")
    if response_matrix.shape[1] != 5:
        raise DirectZFidelityError(
            f"five-direction solve received {response_matrix.shape[1]} columns"
        )
    return solve_nonnegative_l2_ball_ridge(
        response_matrix,
        target,
        radius=radius,
        ridge=ridge,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def combine_unit_c_single_layer_factors(
    factors: Sequence[LowRankFactor],
    coefficients: Sequence[float],
) -> tuple[LowRankFactor, ...]:
    """Scale distinct unit-C single-layer factors by nonnegative coefficients.

    Unit-C normalization is a caller-side semantic precondition because a
    factor alone does not carry the covariance metric needed to re-check it.
    Zero coefficients intentionally retain a zero factor so a five-layer
    proposal remains structurally complete.
    """

    factors = tuple(factors)
    coefficients = tuple(coefficients)
    if not factors or len(factors) != len(coefficients):
        raise DirectZFidelityError(
            "factors and coefficients must have the same non-zero length"
        )
    names = [factor.weight_name for factor in factors]
    if len(names) != len(set(names)):
        raise DirectZFidelityError("single-layer factors must target distinct weights")
    scaled = []
    for index, (factor, coefficient) in enumerate(
        zip(factors, coefficients, strict=True)
    ):
        if not isinstance(factor, LowRankFactor):
            raise DirectZFidelityError(
                f"factor {index} is not a LowRankFactor"
            )
        value = _finite_float(coefficient, name=f"coefficient[{index}]")
        if value < 0.0:
            raise DirectZFidelityError("combination coefficients must be nonnegative")
        scaled.append(factor.scaled(value))
    return tuple(scaled)


def combine_unit_c_single_layer_proposals(
    reference: MemitFactorProposal,
    single_layer_proposals: Sequence[MemitFactorProposal],
    coefficients: Sequence[float],
    *,
    solver_suffix: str = "direct-z-fidelity/nonnegative-l2-ball-ridge",
) -> MemitFactorProposal:
    """Combine same-snapshot, unit-C, one-layer proposals in reference order."""

    if not isinstance(reference, MemitFactorProposal) or not reference.is_synchronous:
        raise DirectZFidelityError(
            "reference must be a synchronous-frozen-snapshot proposal"
        )
    proposals = tuple(single_layer_proposals)
    coefficients = tuple(coefficients)
    if len(proposals) != len(coefficients) or not proposals:
        raise DirectZFidelityError(
            "single-layer proposals and coefficients must have the same non-zero length"
        )
    if not isinstance(solver_suffix, str) or not solver_suffix:
        raise DirectZFidelityError("solver_suffix must not be empty")

    by_name: dict[str, tuple[LowRankFactor, float]] = {}
    for index, (proposal, coefficient) in enumerate(
        zip(proposals, coefficients, strict=True)
    ):
        if not isinstance(proposal, MemitFactorProposal) or not proposal.is_synchronous:
            raise DirectZFidelityError(
                f"single-layer proposal {index} is not synchronous"
            )
        reference.assert_same_entry_snapshot(proposal)
        if proposal.residual_denominator != reference.residual_denominator:
            raise DirectZFidelityError(
                "single-layer proposal residual denominator differs from reference"
            )
        if len(proposal.factors) != 1:
            raise DirectZFidelityError(
                f"single-layer proposal {index} contains {len(proposal.factors)} factors"
            )
        factor = proposal.factors[0]
        if factor.weight_name in by_name:
            raise DirectZFidelityError(
                f"duplicate single-layer weight: {factor.weight_name}"
            )
        value = _finite_float(coefficient, name=f"coefficient[{index}]")
        if value < 0.0:
            raise DirectZFidelityError("combination coefficients must be nonnegative")
        by_name[factor.weight_name] = (factor, value)

    reference_names = tuple(factor.weight_name for factor in reference.factors)
    if set(by_name) != set(reference_names):
        raise DirectZFidelityError(
            "single-layer proposal weights must exactly match the reference factor set"
        )
    ordered_factors = tuple(by_name[name][0] for name in reference_names)
    ordered_coefficients = tuple(by_name[name][1] for name in reference_names)
    scaled = combine_unit_c_single_layer_factors(
        ordered_factors,
        ordered_coefficients,
    )
    return MemitFactorProposal(
        snapshot=reference.snapshot,
        factors=scaled,
        semantics=reference.semantics,
        solver_name=f"{reference.solver_name}/{solver_suffix}",
        residual_denominator=reference.residual_denominator,
    )


__all__ = [
    "BatchPositionPatch",
    "DirectZFidelityError",
    "ProjectedRidgeSolution",
    "SequenceSpillFidelity",
    "SharedDeltaFidelity",
    "aggregate_sequence_spill",
    "aggregate_shared_delta_fidelity",
    "combine_unit_c_single_layer_factors",
    "combine_unit_c_single_layer_proposals",
    "make_batch_position_patch",
    "rewrap_layer_output",
    "solve_five_direction_ridge",
    "solve_nonnegative_l2_ball_ridge",
    "unwrap_layer_output",
]
