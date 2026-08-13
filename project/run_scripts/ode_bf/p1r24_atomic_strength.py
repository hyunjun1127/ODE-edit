"""P1R24 target/write coordinates and matched-strength Atomic-P routing.

This module is deliberately additive to the accepted P1R23 implementation.
It owns the scientific changes introduced by P1R24 while reusing the frozen
streaming target-new objective, physical field, covariance action, and BF16
materialization primitives.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Callable, Mapping, Sequence
from pathlib import Path
import hashlib

import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

from .contracts import ODEBFContractError, canonical_hash
from .fixed_e8_soft_routing import FIXED_E8_SOLVER_FTOL, FIXED_E8_SOLVER_MAXITER, FixedE8Arm
from .functional import tensor_sha256
from .progress_simplex_routing import (
    SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE,
    SIMPLEX_ENERGY_RELATIVE_TOLERANCE,
    SIMPLEX_PRIMAL_TOLERANCE,
    SIMPLEX_XI_TIE_TOLERANCE,
)
from .routing import QuadraticBarrier, RoutingProblem
from .scalable_batched_field import ScalableBatchGlobalMetric, ScalableRobustSharedMetric
from .scalable_batched_model import (
    OrdinalTargetActivationOverlay,
    ScalableObjectivePlan,
    ScalableObjectiveResult,
    _normalized_lookup_position,
    _parameter_inventory_sha256,
    evaluate_scalable_target_new_objective,
)
from .target_new_nll import _input_ids, _left_padding_offsets, _temporary_left_padding, _tokenize_left


P1R24_INSTRUCTION_ID = "ODEEDIT-S05-P1R24-ATOMIC-STRENGTH-RECOVERY-V1"
P1R24_METHOD_ID = "P1R24-ALPHA-GEOMETRY-REMAINING-LAG-MATCHED-STRENGTH-ATOMIC-P-V1"
P1R24_K = 8
P1R24_H = 1.0 / P1R24_K
P1R24_FREEZE_THRESHOLD = 0.05
P1R24_KL_FACTOR = 0.0625
P1R24_NUMERICAL_EPSILON = 1.0e-8
P1R24_Q_EPSILON = 1.0e-12


def verify_p1r24_alphaedit_geometry(
    hparams: Any, lock: "P1R24AliasTargetLock", *, easyedit_root: Path
) -> dict[str, Any]:
    source = easyedit_root / "easyeditor/models/alphaedit/compute_z.py"
    if not source.is_file() or source.is_symlink():
        raise ODEBFContractError("P1R24 AlphaEdit target source differs")
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    required = (
        '], ["{} is a"]',
        "torch.nn.functional.kl_div(",
        "kl_distr_init, kl_log_probs, log_target=True, reduction=\"batchmean\"",
        "torch.norm(delta) / torch.norm(target_init) ** 2",
        "if loss < 5e-2:",
    )
    if any(item not in text for item in required):
        raise ODEBFContractError("P1R24 AlphaEdit objective source geometry differs")
    observed = (
        float(hparams.kl_factor),
        float(hparams.v_weight_decay),
        float(hparams.clamp_norm_factor),
    )
    expected = (lock.kl_factor, lock.decay_factor, lock.clamp_factor)
    if observed != expected:
        raise ODEBFContractError("P1R24 pinned AlphaEdit hparams differ")
    payload = {
        "compute_z_sha256": hashlib.sha256(raw).hexdigest(),
        "kl_prompt": "{} is a",
        "kl_direction": "kl_div(teacher_log_probs,current_log_probs,log_target=True,batchmean)",
        "hparams": lock.raw_free_payload(),
        "native_adam_reuse_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


@dataclass(frozen=True, slots=True)
class P1R24AliasTargetLock:
    alias: str
    kl_factor: float
    decay_factor: float
    clamp_factor: float

    @classmethod
    def for_alias(cls, alias: str) -> "P1R24AliasTargetLock":
        if alias == "llama3-8b-inst":
            return cls(alias, P1R24_KL_FACTOR, 0.5, 0.75)
        if alias == "qwen2.5-7b-inst":
            return cls(alias, P1R24_KL_FACTOR, 0.001, 4.0)
        raise ODEBFContractError("P1R24 target alias differs")

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "kl_prompt": "{} is a",
            "kl_factor": self.kl_factor,
            "decay_factor": self.decay_factor,
            "clamp_factor": self.clamp_factor,
            "freeze_threshold": P1R24_FREEZE_THRESHOLD,
            "native_adam_lr_access_count": 0,
            "native_optimizer_step_count": 0,
        }


@dataclass(frozen=True, slots=True)
class P1R24KLMicrobatch:
    request_ordinals: tuple[int, ...]
    encoding: Mapping[str, torch.Tensor]
    padded_lookup_positions: tuple[int, ...]
    identity_sha256: str


@dataclass(frozen=True, slots=True)
class P1R24KLPlan:
    request_count: int
    request_order_sha256: str
    batches: tuple[P1R24KLMicrobatch, ...]
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r24-kl-plan/v1",
            "request_count": self.request_count,
            "request_order_sha256": self.request_order_sha256,
            "prompt_sha256": canonical_hash(["{} is a"]),
            "batch_request_ordinals": [list(item.request_ordinals) for item in self.batches],
            "batch_identity_sha256": [item.identity_sha256 for item in self.batches],
            "identity_sha256": self.identity_sha256,
        }


def build_p1r24_kl_plan(
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    *,
    request_order_sha256: str,
    request_microbatch_size: int,
    fact_token_strategy: str,
) -> P1R24KLPlan:
    batch = tuple(requests)
    if not batch or request_microbatch_size <= 0 or request_microbatch_size > len(batch):
        raise ODEBFContractError("P1R24 KL plan batch differs")
    prepared: list[P1R24KLMicrobatch] = []
    with _temporary_left_padding(tokenizer):
        for start in range(0, len(batch), request_microbatch_size):
            stop = min(start + request_microbatch_size, len(batch))
            ordinals = tuple(range(start, stop))
            prompts = ["{} is a".format(str(batch[index]["subject"])) for index in ordinals]
            encoded_raw = _tokenize_left(tokenizer, prompts, padding=True, return_tensors="pt")
            encoding = {
                str(name): value.detach().to(device="cpu").contiguous().clone()
                for name, value in encoded_raw.items()
                if isinstance(value, torch.Tensor)
            }
            input_ids, left_padding = _left_padding_offsets(encoding, rows=len(ordinals))
            positions: list[int] = []
            for row, ordinal in enumerate(ordinals):
                raw = _normalized_lookup_position(
                    tokenizer,
                    prompt_template="{} is a",
                    subject=str(batch[ordinal]["subject"]),
                    fact_token_strategy=fact_token_strategy,
                    prefix_ids=_input_ids(
                        _tokenize_left(tokenizer, [prompts[row]]),
                        label="P1R24 KL prompt",
                        one_row=True,
                    )[0],
                )
                position = int(left_padding[row] + raw)
                if position < 0 or position >= input_ids.shape[1]:
                    raise ODEBFContractError("P1R24 KL lookup differs")
                positions.append(position)
            payload = {
                "request_ordinals": list(ordinals),
                "input_sha256": tensor_sha256(encoding["input_ids"]),
                "attention_sha256": tensor_sha256(encoding["attention_mask"]),
                "padded_lookup_positions": positions,
            }
            prepared.append(P1R24KLMicrobatch(ordinals, encoding, tuple(positions), canonical_hash(payload)))
    payload = {
        "request_count": len(batch),
        "request_order_sha256": request_order_sha256,
        "batch_identity_sha256": [item.identity_sha256 for item in prepared],
        "prompt": "{} is a",
    }
    return P1R24KLPlan(len(batch), request_order_sha256, tuple(prepared), canonical_hash(payload))


@dataclass(frozen=True, slots=True)
class P1R24KLResult:
    loss: float
    per_request_values: tuple[float, ...]
    gradient: torch.Tensor | None
    model_forward_count: int
    backward_count: int
    processed_token_count: int
    padded_token_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-s05-p1r24-outer-entry-kl/v1",
            "loss": self.loss,
            "per_request_values": list(self.per_request_values),
            "gradient_sha256": None if self.gradient is None else tensor_sha256(self.gradient),
            "model_forward_count": self.model_forward_count,
            "backward_count": self.backward_count,
            "processed_token_count": self.processed_token_count,
            "padded_token_count": self.padded_token_count,
            "teacher_refresh_count": 0,
            "identity_sha256": self.identity_sha256,
        }


def evaluate_p1r24_kl(
    model: torch.nn.Module,
    plan: P1R24KLPlan,
    *,
    teacher_log_probs: Sequence[torch.Tensor] | None,
    target_state: torch.Tensor | None = None,
    current_terminal: torch.Tensor | None = None,
    target_layer_name: str | None = None,
) -> tuple[P1R24KLResult, tuple[torch.Tensor, ...]]:
    target_mode = target_state is not None
    if target_mode != (current_terminal is not None and target_layer_name is not None):
        raise ODEBFContractError("P1R24 KL target inputs differ")
    if target_mode and (
        target_state is None
        or current_terminal is None
        or target_state.shape != current_terminal.shape
        or target_state.shape[1] != plan.request_count
        or not target_state.requires_grad
    ):
        raise ODEBFContractError("P1R24 KL target geometry differs")
    if teacher_log_probs is not None and len(teacher_log_probs) != plan.request_count:
        raise ODEBFContractError("P1R24 KL teacher inventory differs")
    device = next(model.parameters()).device
    before = _parameter_inventory_sha256(model)
    observed_teacher: list[torch.Tensor | None] = [None] * plan.request_count
    values: list[torch.Tensor | None] = [None] * plan.request_count
    gradient = (
        torch.zeros_like(target_state, device="cpu", dtype=torch.float64)
        if target_mode
        else None
    )
    processed = 0
    padded = 0
    backwards = 0
    for batch in plan.batches:
        encoding = {name: value.to(device=device) for name, value in batch.encoding.items()}
        with ExitStack() as stack:
            overlay: OrdinalTargetActivationOverlay | None = None
            if target_mode:
                assert target_state is not None and current_terminal is not None and target_layer_name is not None
                overlay = stack.enter_context(
                    OrdinalTargetActivationOverlay(
                        model,
                        target_layer_name,
                        residual=target_state - current_terminal.to(target_state.device, target_state.dtype),
                        row_request_ordinals=batch.request_ordinals,
                        padded_lookup_positions=batch.padded_lookup_positions,
                    )
                )
            logits = model(**encoding).logits
            selected = torch.stack(
                [logits[row, position, :].float() for row, position in enumerate(batch.padded_lookup_positions)]
            )
            log_probs = torch.log_softmax(selected, dim=1)
            for row, ordinal in enumerate(batch.request_ordinals):
                observed_teacher[ordinal] = log_probs[row].detach().to(device="cpu", dtype=torch.float32)
            if teacher_log_probs is None:
                batch_values = torch.zeros(len(batch.request_ordinals), device=device, dtype=torch.float32)
            else:
                teacher = torch.stack(
                    [teacher_log_probs[ordinal].to(device=device, dtype=torch.float32) for ordinal in batch.request_ordinals]
                )
                batch_values = F.kl_div(teacher, log_probs, log_target=True, reduction="none").sum(dim=1)
            for row, ordinal in enumerate(batch.request_ordinals):
                values[ordinal] = batch_values[row]
            if target_mode:
                assert target_state is not None and gradient is not None and overlay is not None
                local = torch.autograd.grad(batch_values.sum(), target_state, retain_graph=False, create_graph=False)[0]
                gradient.add_(local.detach().to(device="cpu", dtype=torch.float64))
                backwards += 1
                if overlay.fire_count != 1:
                    raise ODEBFContractError("P1R24 KL overlay firing differs")
        processed += int(encoding["attention_mask"].sum().detach().cpu())
        padded += int(encoding["attention_mask"].numel())
    if any(item is None for item in observed_teacher) or any(item is None for item in values):
        raise ODEBFContractError("P1R24 KL coverage differs")
    final_values = tuple(float(item.detach().cpu()) for item in values if item is not None)
    if gradient is not None:
        gradient.div_(plan.request_count)
        gradient = gradient.to(dtype=torch.float32).contiguous()
    after = _parameter_inventory_sha256(model)
    if after != before:
        raise ODEBFContractError("P1R24 KL objective mutated model")
    payload = {
        "loss": math.fsum(final_values) / plan.request_count,
        "per_request_values": list(final_values),
        "gradient_sha256": None if gradient is None else tensor_sha256(gradient),
        "teacher_sha256": [tensor_sha256(item) for item in observed_teacher if item is not None],
        "plan_sha256": plan.identity_sha256,
        "forward_count": len(plan.batches),
        "backward_count": backwards,
        "processed_tokens": processed,
        "padded_tokens": padded,
    }
    result = P1R24KLResult(
        payload["loss"],
        final_values,
        gradient,
        len(plan.batches),
        backwards,
        processed,
        padded,
        canonical_hash(payload),
    )
    return result, tuple(item for item in observed_teacher if item is not None)


@dataclass(frozen=True, slots=True)
class P1R24TargetStep:
    target_next: torch.Tensor
    target_displacement: torch.Tensor
    required_displacement: torch.Tensor
    write_velocity: torch.Tensor
    nll_gradient: torch.Tensor
    combined_gradient: torch.Tensor
    rho_write_signed: float
    rho_write: float
    alpha_target_signed: float
    frozen_mask: tuple[bool, ...]
    receipt: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class P1R24WriteCoordinate:
    required_analytic: torch.Tensor
    required_model: torch.Tensor
    write_velocity: torch.Tensor
    receipt: Mapping[str, Any]


P1R24CoordinateBuilder = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int],
    P1R24WriteCoordinate,
]


def p1r24_remaining_lag_coordinate(
    target_next64: torch.Tensor,
    current_target64: torch.Tensor,
    current_terminal64: torch.Tensor,
    displacement: torch.Tensor,
    step_index: int,
) -> P1R24WriteCoordinate:
    lag = (current_target64 - current_terminal64).contiguous()
    remaining = P1R24_K - step_index
    required = (displacement + lag / remaining).contiguous()
    required_model = required.to(dtype=torch.float32).contiguous()
    write_velocity = (required_model / P1R24_H).contiguous()
    identity_residual = float(
        torch.max(torch.abs(P1R24_H * write_velocity - required_model))
    )
    analytic_to_model_cast_residual = float(
        torch.max(torch.abs(required_model.to(torch.float64) - required))
    )
    if identity_residual > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R24 remaining-step identity differs")
    receipt = {
        "schema": "ode-edit-s05-p1r24-target-write-coordinate/v1",
        "coordinate_policy": "DISPLACEMENT_PLUS_LAG_OVER_REMAINING",
        "remaining_steps": remaining,
        "write_lag_sha256": tensor_sha256(lag),
        "required_displacement_sha256": tensor_sha256(required_model),
        "analytic_required_displacement_sha256": tensor_sha256(required),
        "write_velocity_sha256": tensor_sha256(write_velocity),
        "identity_max_abs_residual": identity_residual,
        "analytic_to_model_cast_max_abs": analytic_to_model_cast_residual,
        "field_coordinate": "ACTIVATION_VELOCITY",
        "residual_presplit_count": 0,
        "physical_h_application_count": 1,
        "second_remaining_division_count": 0,
    }
    receipt["coordinate_receipt_sha256"] = canonical_hash(receipt)
    return P1R24WriteCoordinate(
        required,
        required_model,
        write_velocity,
        receipt,
    )


def p1r24_target_step(
    current_target: torch.Tensor,
    current_terminal: torch.Tensor,
    z0: torch.Tensor,
    nll: ScalableObjectiveResult,
    kl: P1R24KLResult,
    metric: ScalableBatchGlobalMetric | ScalableRobustSharedMetric,
    lock: P1R24AliasTargetLock,
    *,
    step_index: int,
    frozen_mask: Sequence[bool],
    coordinate_builder: P1R24CoordinateBuilder = p1r24_remaining_lag_coordinate,
) -> P1R24TargetStep:
    if nll.target_gradient is None or kl.gradient is None:
        raise ODEBFContractError("P1R24 target gradients are absent")
    if step_index < 0 or step_index >= P1R24_K:
        raise ODEBFContractError("P1R24 target step index differs")
    shape = current_target.shape
    if current_terminal.shape != shape or z0.shape != shape or shape[1] != len(frozen_mask):
        raise ODEBFContractError("P1R24 target state geometry differs")
    current64 = current_target.detach().to(device="cpu", dtype=torch.float64)
    z064 = z0.detach().to(device="cpu", dtype=torch.float64)
    delta = current64 - z064
    z0_norm = torch.linalg.vector_norm(z064, dim=0)
    delta_norm = torch.linalg.vector_norm(delta, dim=0)
    if bool(torch.any(z0_norm <= 0.0)):
        raise ODEBFContractError("P1R24 target decay origin is degenerate")
    decay_values = lock.decay_factor * delta_norm / torch.square(z0_norm)
    decay_gradient = torch.zeros_like(delta)
    nonzero = delta_norm > 0.0
    decay_gradient[:, nonzero] = (
        lock.decay_factor
        * delta[:, nonzero]
        / delta_norm[nonzero].unsqueeze(0)
        / torch.square(z0_norm[nonzero]).unsqueeze(0)
        / shape[1]
    )
    nll_gradient = nll.target_gradient.detach().to(device="cpu", dtype=torch.float64)
    kl_gradient = kl.gradient.detach().to(device="cpu", dtype=torch.float64)
    combined = nll_gradient + lock.kl_factor * kl_gradient + decay_gradient
    phi = tuple(
        float(nll.per_request_values[index] + lock.kl_factor * kl.per_request_values[index] + decay_values[index])
        for index in range(shape[1])
    )
    latch = tuple(bool(frozen_mask[index] or phi[index] < P1R24_FREEZE_THRESHOLD) for index in range(shape[1]))
    combined[:, list(latch)] = 0.0
    velocity, scale = metric.velocity(combined.to(dtype=torch.float32))
    candidate = current64 + P1R24_H * velocity.to(dtype=torch.float64)
    clamp_ratio: list[float] = []
    for index in range(shape[1]):
        if latch[index]:
            candidate[:, index] = current64[:, index]
            clamp_ratio.append(0.0)
            continue
        proposed = candidate[:, index] - z064[:, index]
        norm = float(torch.linalg.vector_norm(proposed))
        maximum = lock.clamp_factor * float(z0_norm[index])
        ratio = 1.0 if norm <= maximum or norm == 0.0 else maximum / norm
        candidate[:, index] = z064[:, index] + ratio * proposed
        clamp_ratio.append(ratio)
    target_next = candidate.to(dtype=torch.float32).contiguous()
    displacement = (target_next.to(dtype=torch.float64) - current64).contiguous()
    terminal64 = current_terminal.detach().to(device="cpu", dtype=torch.float64)
    coordinate = coordinate_builder(
        target_next.to(dtype=torch.float64),
        current64,
        terminal64,
        displacement,
        step_index,
    )
    required = coordinate.required_analytic
    required_model = coordinate.required_model
    write_velocity = coordinate.write_velocity
    alpha_target_signed = float(-torch.sum(nll_gradient * displacement))
    rho_signed = float(-torch.sum(nll_gradient * required))
    rho = max(rho_signed, 0.0)
    payload = {
        "schema": "ode-edit-s05-p1r24-target-write-coordinate/v1",
        "k": step_index,
        "tau_before": step_index / P1R24_K,
        "tau_after": (step_index + 1) / P1R24_K,
        "target_new_loss": nll.loss,
        "kl_loss": kl.loss,
        "decay_loss_mean": float(torch.mean(decay_values)),
        "combined_loss_by_request": list(phi),
        "frozen_mask": list(latch),
        "frozen_count": sum(latch),
        "clamp_ratio": clamp_ratio,
        "target_displacement_sha256": tensor_sha256(displacement),
        "alpha_target_signed": alpha_target_signed,
        "rho_write_signed": rho_signed,
        "rho_write": rho,
        "combined_gradient_sha256": tensor_sha256(combined),
        "target_new_gradient_sha256": tensor_sha256(nll_gradient),
        "metric": scale,
        **coordinate.receipt,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return P1R24TargetStep(
        target_next,
        displacement.to(dtype=torch.float32),
        required_model,
        write_velocity,
        nll_gradient.to(dtype=torch.float32),
        combined.to(dtype=torch.float32),
        rho_signed,
        rho,
        alpha_target_signed,
        latch,
        payload,
    )


def p1r24_disable_historical(problem: RoutingProblem) -> RoutingProblem:
    dimension = problem.signed_progress.size
    zero = np.zeros(dimension, dtype=np.float64)
    return RoutingProblem(
        problem.signed_progress,
        problem.capacity_metric,
        problem.trust_metric,
        problem.trust_radius,
        problem.layer_caps,
        problem.requested_progress,
        problem.minimum_progress,
        QuadraticBarrier("historical", 0.0, zero, np.diag(zero), 0.0, "layer-local-diagonal"),
        problem.pretrained,
    )


class P1R24RoutingStatus(str, Enum):
    JOINT_WRITE = "JOINT_WRITE"
    SEMANTIC_NO_WRITE = "SEMANTIC_NO_WRITE"
    NO_POSITIVE_DIRECTION = "NO_POSITIVE_DIRECTION"
    Q_NUMERICAL_DEGENERACY = "Q_NUMERICAL_DEGENERACY"


@dataclass(frozen=True, slots=True)
class P1R24RoutingResult:
    arm: FixedE8Arm
    status: P1R24RoutingStatus
    signed_slopes: tuple[float, ...]
    active_mask: tuple[bool, ...]
    q: float
    rho_write: float
    neutral_pi: tuple[float, ...]
    soft_pi: tuple[float, ...]
    pi: tuple[float, ...]
    neutral_velocity: tuple[float, ...]
    soft_velocity: tuple[float, ...]
    velocity: tuple[float, ...]
    predicted_progress: float
    equality_residual: float
    neutral_energy: float
    selected_energy: float
    neutral_p: float
    soft_p: float
    selected_p: float
    selected_capacity: float
    certificates: tuple[Mapping[str, Any], ...]
    identity_sha256: str

    @property
    def alpha_req(self) -> float:
        return self.rho_write

    @property
    def alpha_max(self) -> float:
        return self.q

    @property
    def alpha_apply(self) -> float:
        return self.rho_write

    @property
    def coverage(self) -> float:
        return 1.0 if self.rho_write == 0.0 else self.predicted_progress / self.rho_write

    @property
    def applied_coefficient(self) -> tuple[float, ...]:
        return tuple(P1R24_H * item for item in self.velocity)

    def raw_free_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        active_pi = np.asarray(
            [self.pi[index] for index, active in enumerate(self.active_mask) if active],
            dtype=np.float64,
        )
        positive_pi = active_pi[active_pi > 0.0]
        entropy = (
            float(-np.sum(positive_pi * np.log(positive_pi)))
            if positive_pi.size
            else 0.0
        )
        payload.update(
            {
                "instruction_id": P1R24_INSTRUCTION_ID,
                "method_id": P1R24_METHOD_ID,
                "parameterization": "v_l=rho_write*pi_l/a_l",
                "progress_definition": "a^T v=rho_write in applied-step target-new NLL units",
                "per_layer_upper_cap_influence_count": 0,
                "functional_p_inner_probe_count": 0,
                "persistent_historical_count": 0,
                "historical_decision_influence_count": 0,
                "hard_p_h_budget_influence_count": 0,
                "retry_backtracking_veto_count": 0,
                "rho_write_over_q": (
                    self.rho_write / self.q if self.q > P1R24_Q_EPSILON else None
                ),
                "simplex_entropy": entropy,
                "simplex_top1_share": (
                    float(active_pi.max()) if active_pi.size else 0.0
                ),
                "positive_velocity_count": sum(item > 0.0 for item in self.velocity),
                "maximum_velocity": max(self.velocity, default=0.0),
            }
        )
        return payload


def _energy(value: np.ndarray, metric: np.ndarray) -> float:
    return float(value @ metric @ value)


def _velocity(pi: np.ndarray, active: np.ndarray, slopes: np.ndarray, rho: float) -> np.ndarray:
    result = np.zeros_like(slopes)
    result[active] = rho * pi / slopes[active]
    return result


def solve_p1r24_matched_routing(
    problem: RoutingProblem,
    *,
    arm: FixedE8Arm | str,
    rho_write: float,
) -> P1R24RoutingResult:
    selected = FixedE8Arm(arm)
    if selected not in (FixedE8Arm.NEUTRAL, FixedE8Arm.SOFT):
        raise ODEBFContractError("P1R24 routing arm differs")
    slopes = np.asarray(problem.signed_progress, dtype=np.float64)
    energy_metric = np.asarray(problem.trust_metric, dtype=np.float64)
    capacity_metric = np.asarray(problem.capacity_metric, dtype=np.float64)
    if not math.isfinite(rho_write) or rho_write < 0.0 or not np.all(np.isfinite(slopes)):
        raise ODEBFContractError("P1R24 routing demand differs")
    active = np.flatnonzero(slopes > 0.0)
    q = float(slopes[active].sum())
    zero = np.zeros_like(slopes)
    neutral_pi = zero.copy()
    soft_pi = zero.copy()
    neutral = zero.copy()
    soft = zero.copy()
    certificates: list[Mapping[str, Any]] = []
    if rho_write == 0.0:
        status = P1R24RoutingStatus.SEMANTIC_NO_WRITE
    elif active.size == 0:
        status = P1R24RoutingStatus.NO_POSITIVE_DIRECTION
    elif q <= P1R24_Q_EPSILON:
        status = P1R24RoutingStatus.Q_NUMERICAL_DEGENERACY
    else:
        status = P1R24RoutingStatus.JOINT_WRITE
        neutral_pi[active] = slopes[active] / q
        neutral = _velocity(neutral_pi[active], active, slopes, rho_write)
        neutral_energy = _energy(neutral, energy_metric)
        limit = neutral_energy * (1.0 + SIMPLEX_ENERGY_RELATIVE_TOLERANCE) + SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
        if selected is FixedE8Arm.NEUTRAL:
            # Neutral is the exact matched-strength control. It must not
            # depend on either Soft optimizer stage.
            soft_pi[:] = neutral_pi
            soft[:] = neutral
        elif active.size == 1:
            soft_pi[:] = neutral_pi
            soft[:] = neutral
        else:
            transform = rho_write / slopes[active]

            def expand(pi: np.ndarray) -> np.ndarray:
                return _velocity(pi, active, slopes, rho_write)

            def p_value(pi: np.ndarray) -> float:
                return problem.pretrained.value(expand(pi))

            def p_grad(pi: np.ndarray) -> np.ndarray:
                value = expand(pi)
                return transform * (2.0 * problem.pretrained.linear + 2.0 * problem.pretrained.gram @ value)[active]

            def e_value(pi: np.ndarray) -> float:
                return _energy(expand(pi), energy_metric)

            def e_grad(pi: np.ndarray) -> np.ndarray:
                value = expand(pi)
                return 2.0 * transform * (energy_metric @ value)[active]

            constraints = (
                {"type": "eq", "fun": lambda value: float(value.sum() - 1.0), "jac": lambda value: np.ones(active.size)},
                {"type": "ineq", "fun": lambda value: float(limit - e_value(value)), "jac": lambda value: -e_grad(value)},
            )
            stage1 = minimize(
                p_value,
                neutral_pi[active],
                jac=p_grad,
                method="SLSQP",
                bounds=tuple((0.0, None) for _ in active),
                constraints=constraints,
                options={"disp": False, "ftol": FIXED_E8_SOLVER_FTOL, "maxiter": FIXED_E8_SOLVER_MAXITER},
            )
            pi1 = np.asarray(stage1.x, dtype=np.float64)
            v1 = expand(pi1)
            first = {
                "phase": "minimum-cumulative-structural-p",
                "success": bool(stage1.success),
                "status": int(stage1.status),
                "simplex_residual": abs(float(pi1.sum()) - 1.0),
                "negative_violation": max(float(-pi1.min(initial=0.0)), 0.0),
                "strength_residual": abs(float(slopes @ v1) - rho_write),
                "energy_violation": max(_energy(v1, energy_metric) - limit, 0.0),
            }
            first["passed"] = bool(
                first["success"]
                and first["simplex_residual"] <= SIMPLEX_PRIMAL_TOLERANCE
                and first["negative_violation"] <= SIMPLEX_PRIMAL_TOLERANCE
                and first["strength_residual"] <= SIMPLEX_PRIMAL_TOLERANCE
                and first["energy_violation"] <= SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
            )
            certificates.append(first)
            if not first["passed"]:
                raise ODEBFContractError(f"P1R24 Structural-P certificate failed: {first}")
            p_star = p_value(pi1)
            p_scale = max(float(np.trace(problem.pretrained.gram)), P1R24_Q_EPSILON)
            p_tie = SIMPLEX_XI_TIE_TOLERANCE * p_scale
            constraints2 = (*constraints, {"type": "ineq", "fun": lambda value: float(p_star + p_tie - p_value(value)), "jac": lambda value: -p_grad(value)})

            def capacity(pi: np.ndarray) -> float:
                value = expand(pi)
                return float(0.5 * value @ capacity_metric @ value)

            def capacity_grad(pi: np.ndarray) -> np.ndarray:
                value = expand(pi)
                return transform * (capacity_metric @ value)[active]

            stage2 = minimize(
                capacity,
                pi1,
                jac=capacity_grad,
                method="SLSQP",
                bounds=tuple((0.0, None) for _ in active),
                constraints=constraints2,
                options={"disp": False, "ftol": FIXED_E8_SOLVER_FTOL, "maxiter": FIXED_E8_SOLVER_MAXITER},
            )
            pi2 = np.asarray(stage2.x, dtype=np.float64)
            v2 = expand(pi2)
            second = {
                "phase": "minimum-capacity-inside-structural-p-tie",
                "success": bool(stage2.success),
                "status": int(stage2.status),
                "simplex_residual": abs(float(pi2.sum()) - 1.0),
                "negative_violation": max(float(-pi2.min(initial=0.0)), 0.0),
                "strength_residual": abs(float(slopes @ v2) - rho_write),
                "energy_violation": max(_energy(v2, energy_metric) - limit, 0.0),
                "p_tie_violation": max(p_value(pi2) - (p_star + p_tie), 0.0),
            }
            second["passed"] = bool(
                second["success"]
                and second["simplex_residual"] <= SIMPLEX_PRIMAL_TOLERANCE
                and second["negative_violation"] <= SIMPLEX_PRIMAL_TOLERANCE
                and second["strength_residual"] <= SIMPLEX_PRIMAL_TOLERANCE
                and second["energy_violation"] <= SIMPLEX_ENERGY_ABSOLUTE_TOLERANCE
                and second["p_tie_violation"] <= P1R24_NUMERICAL_EPSILON
            )
            certificates.append(second)
            if not second["passed"]:
                raise ODEBFContractError(f"P1R24 capacity certificate failed: {second}")
            soft_pi[active] = pi2
            soft = v2
    chosen_pi = neutral_pi if selected is FixedE8Arm.NEUTRAL else soft_pi
    chosen = neutral if selected is FixedE8Arm.NEUTRAL else soft
    if status is not P1R24RoutingStatus.JOINT_WRITE:
        chosen_pi = zero.copy()
        chosen = zero.copy()
    predicted = float(slopes @ chosen)
    equality = abs(predicted - rho_write) if status is P1R24RoutingStatus.JOINT_WRITE else abs(predicted)
    if status is P1R24RoutingStatus.JOINT_WRITE and equality > SIMPLEX_PRIMAL_TOLERANCE:
        raise ODEBFContractError("P1R24 selected strength differs")
    payload = {
        "arm": selected.value,
        "status": status.value,
        "slopes": slopes.tolist(),
        "q": q,
        "rho_write": rho_write,
        "neutral_pi": neutral_pi.tolist(),
        "soft_pi": soft_pi.tolist(),
        "velocity": chosen.tolist(),
        "predicted": predicted,
        "equality_residual": equality,
        "problem_sha256": problem.identity(),
        "certificates": certificates,
    }
    identity = canonical_hash(payload)
    return P1R24RoutingResult(
        selected,
        status,
        tuple(float(item) for item in slopes),
        tuple(bool(item > 0.0) for item in slopes),
        q,
        rho_write,
        tuple(float(item) for item in neutral_pi),
        tuple(float(item) for item in soft_pi),
        tuple(float(item) for item in chosen_pi),
        tuple(float(item) for item in neutral),
        tuple(float(item) for item in soft),
        tuple(float(item) for item in chosen),
        predicted,
        equality,
        _energy(neutral, energy_metric),
        _energy(chosen, energy_metric),
        problem.pretrained.value(neutral),
        problem.pretrained.value(soft),
        problem.pretrained.value(chosen),
        float(0.5 * chosen @ capacity_metric @ chosen),
        tuple(certificates),
        identity,
    )


def p1r24_cumulative_p_receipt(
    problem: RoutingProblem,
    velocity: Sequence[float],
    *,
    step_index: int,
    factor_list_hashes: Mapping[int, str],
) -> dict[str, Any]:
    value = np.asarray(tuple(float(item) for item in velocity), dtype=np.float64)
    offset = float(problem.pretrained.offset)
    cross = float(2.0 * problem.pretrained.linear @ value)
    self_term = float(value @ problem.pretrained.gram @ value)
    after = float(problem.pretrained.value(value))
    residual = abs((after - offset) - (cross + self_term))
    if residual > P1R24_NUMERICAL_EPSILON:
        raise ODEBFContractError("P1R24 cumulative Structural-P identity differs")
    payload = {
        "schema": "ode-edit-s05-p1r24-cumulative-atomic-structural-p/v1",
        "step_index": step_index,
        "atomic_contribution_count": step_index,
        "persistent_historical_count": 0,
        "factor_list_hashes": {str(layer): factor_list_hashes[layer] for layer in sorted(factor_list_hashes)},
        "d_P": offset,
        "linear_cross_term": cross,
        "quadratic_self_term": self_term,
        "P_before": offset,
        "P_after": after,
        "algebra_identity_residual": residual,
        "theta": [P1R24_H * float(item) for item in value],
        "negative_cross_allowed": True,
        "P_budget_veto_retry_influence_count": 0,
        "historical_h_input_influence_cost_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "P1R24AliasTargetLock",
    "P1R24_INSTRUCTION_ID",
    "P1R24_METHOD_ID",
    "P1R24RoutingResult",
    "P1R24RoutingStatus",
    "P1R24WriteCoordinate",
    "build_p1r24_kl_plan",
    "evaluate_p1r24_kl",
    "p1r24_cumulative_p_receipt",
    "p1r24_disable_historical",
    "p1r24_remaining_lag_coordinate",
    "p1r24_target_step",
    "solve_p1r24_matched_routing",
    "verify_p1r24_alphaedit_geometry",
]
