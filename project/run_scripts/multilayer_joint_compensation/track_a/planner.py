"""A0 fixed-entry differentiable writer-image planning (design §§5–6).

The caller supplies the joint full-model objective and *logical-batch averaged*
weight gradients. This module never computes native z, keys, history or targets.
For each layer D = R @ J, where J is [B, din] and R is [dout, B].
All physical deltas/R/Adam state are FP32. Image-metric and scalar reductions
are FP64; gradients are cast once to the FP32 optimizer parameter dtype.

Telemetry losses are pre-update (steps 1..25); final is a separate observation.
The entry call is reused for step 1 and fixes q and sigma for the entire plan.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

import torch


class PlannerBoundary(RuntimeError):
    """Invalid numerical/input execution, never a performance threshold."""


ObjectiveCallback = Callable[
    [tuple[torch.Tensor, ...]],
    tuple[float | torch.Tensor, tuple[torch.Tensor, ...], dict[str, Any]],
]


@dataclass(frozen=True)
class JointPlan:
    layer_ids: tuple[int, ...]
    rhs: tuple[torch.Tensor, ...]
    deltas: tuple[torch.Tensor, ...]
    q_raw: tuple[float, ...]
    q_balanced: tuple[float, ...]
    q_exact_zero: tuple[bool, ...]
    q_floor: float
    sigma_e: float
    trajectory: tuple[dict[str, Any], ...]
    final: dict[str, Any]
    metric_receipts: tuple[dict[str, Any], ...]
    counts: dict[str, int]


def _finite(tensor: torch.Tensor, label: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise PlannerBoundary(f"nonfinite {label}")


def _metric_inverse(metric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Return metric and square-root pseudoinverse factor; no ridge/rank cap.

    Computing q as ||g_R V_ret diag(s_ret**-.5)||² is exactly the specified
    g_R T_R† g_R.T contraction, without cancellation-induced negative q.
    """
    value = metric.detach().to(dtype=torch.float64).clone()
    _finite(value, "image metric")
    eigen_scale = float(torch.linalg.matrix_norm(value, ord=2))
    tolerance = value.shape[0] * torch.finfo(torch.float64).eps * eigen_scale
    asymmetry = float(torch.linalg.matrix_norm(value - value.T, ord=2))
    if asymmetry > 8 * tolerance:
        raise PlannerBoundary("nonsymmetric image metric")
    value = (value + value.T) * .5
    eigenvalues, vectors = torch.linalg.eigh(value)
    if float(eigenvalues.min()) < -8 * tolerance:
        raise PlannerBoundary("indefinite image metric")
    keep = eigenvalues > tolerance
    inverse_factor = vectors[:, keep] / eigenvalues[keep].sqrt()
    return value, inverse_factor, {
        "dimension": value.shape[0], "retained_rank": int(keep.sum()),
        "eigenvalues": eigenvalues.cpu().tolist(),
        "pinv_cutoff": tolerance, "cutoff_rule": "B * eps_FP64 * spectral_norm",
        "symmetry_residual": asymmetry, "ridge_added": 0,
    }


def plan_joint_targets(
    writer_factors: Sequence[torch.Tensor],
    image_metrics: Sequence[torch.Tensor],
    output_dims: Sequence[int],
    loss_and_weight_grad: ObjectiveCallback,
    *,
    layer_ids: Sequence[int] = (4, 8),
    balance_weight: float = .1,
) -> JointPlan:
    """Perform exactly 25 joint Adam updates (or B=0 no-op).

    ``image_metrics[l] = J[l] S[l] J[l].T`` is supplied in FP64 by the
    fixed-entry native-geometry owner. It is not re-estimated at later steps.
    The same callback observes both layer changes simultaneously. Callback
    gradients must be derivatives with respect to physical dense D, not R.
    Logical averaging must happen exactly once in the callback, not here.
    Only approved balance values .1 and 0 (Middle ablation) are supported.
    """
    labels = tuple(layer_ids)
    if labels not in ((4,), (4, 8)):
        raise PlannerBoundary("A0 support must be (4,) or (4, 8)")
    if balance_weight not in (0., .1):
        raise PlannerBoundary("unapproved balance weight")
    if not (len(writer_factors) == len(image_metrics) == len(output_dims) == len(labels)):
        raise PlannerBoundary("layer inventory mismatch")
    if any(j.ndim != 2 or j.dtype != torch.float32 for j in writer_factors):
        raise PlannerBoundary("writer factors must be FP32 [B, din]")
    batch = writer_factors[0].shape[0]
    device = writer_factors[0].device
    for j, metric, dout in zip(writer_factors, image_metrics, output_dims):
        if j.shape[0] != batch or j.device != device or int(dout) <= 0:
            raise PlannerBoundary("writer batch/device/output shape mismatch")
        if metric.shape != (batch, batch) or metric.device != device:
            raise PlannerBoundary("image metric shape/device mismatch")
        if metric.dtype != torch.float64:
            raise PlannerBoundary("image metrics must be FP64")
        _finite(j, "writer factor")
    factors = tuple(j.detach().clone() for j in writer_factors)
    rhs = tuple(torch.zeros((int(dout), batch), dtype=torch.float32, device=device)
                for dout in output_dims)
    counts = {"optimizer_updates": 0, "joint_gradient_calls": 0,
              "final_observation_calls": 0, "callback_calls": 0,
              "callback_weight_gradient_sets": 0,
              "native_z_calls": 0, "history_appends": 0, "q_captures": 0}
    if batch == 0:
        return JointPlan(labels, rhs, tuple(r @ j for r, j in zip(rhs, factors)),
                         tuple(0. for _ in labels), tuple(0. for _ in labels),
                         tuple(True for _ in labels), 0., 0., (),
                         {"status": "B0_NOOP", "actual_updates": 0}, (), counts)
    decompositions = tuple(_metric_inverse(m) for m in image_metrics)
    metrics = tuple(row[0] for row in decompositions)
    inverses = tuple(row[1] for row in decompositions)

    def observe(deltas: tuple[torch.Tensor, ...]) -> tuple[float, tuple[torch.Tensor, ...], dict]:
        loss, gradients, telemetry = loss_and_weight_grad(deltas)
        counts["callback_calls"] += 1
        counts["callback_weight_gradient_sets"] += 1
        loss_value = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        if not math.isfinite(loss_value):
            raise PlannerBoundary("nonfinite Current objective")
        if len(gradients) != len(rhs):
            raise PlannerBoundary("callback gradient inventory mismatch")
        pulled = []
        for grad, delta, j in zip(gradients, deltas, factors):
            if grad.shape != delta.shape or grad.dtype != torch.float32 or grad.device != device:
                raise PlannerBoundary("callback physical gradient shape/dtype/device mismatch")
            _finite(grad, "physical gradient")
            # One chain-rule pullback. No second logical-batch division.
            pulled.append(grad.detach() @ j.T)
        return loss_value, tuple(pulled), dict(telemetry)

    deltas = tuple(r @ j for r, j in zip(rhs, factors))
    loss, gradients, telemetry = observe(deltas)
    q_raw = tuple(float(torch.sum((g.double() @ inv).square()))
                  for g, inv in zip(gradients, inverses))
    if any(not math.isfinite(q) or q < 0 for q in q_raw):
        raise PlannerBoundary("invalid writer-image efficiency")
    q_floor = max(sum(q / len(q_raw) for q in q_raw) * 1e-6, 1e-12)
    q_balanced = tuple(max(q, q_floor) for q in q_raw)
    sigma = max(loss, 1e-3)
    counts["q_captures"] = 1
    optimizer = torch.optim.Adam(list(rhs), lr=.1, betas=(.9, .999),
                                 eps=1e-8, weight_decay=0., foreach=False)
    trajectory = []

    def balance_terms() -> tuple[list[float], float, tuple[torch.Tensor, ...]]:
        energies = [float(torch.sum((r.double() @ metric) * r.double()))
                    for r, metric in zip(rhs, metrics)]
        balance = sum(q * energy for q, energy in zip(q_balanced, energies)) / (2 * sigma**2)
        gradient = tuple((balance_weight * q / sigma**2 * (r.double() @ metric)).float()
                         for r, metric, q in zip(rhs, metrics, q_balanced))
        return energies, balance, gradient

    for step in range(1, 26):
        if step > 1:
            deltas = tuple(r @ j for r, j in zip(rhs, factors))
            loss, gradients, telemetry = observe(deltas)
        counts["joint_gradient_calls"] += 1
        energies, balance, balance_gradients = balance_terms()
        objective = loss + balance_weight * balance
        if not math.isfinite(objective):
            raise PlannerBoundary("nonfinite joint objective")
        trajectory.append({"step": step, "position": "pre_update",
                           "current_nll": loss, "balance": balance,
                           "balance_weight": balance_weight, "objective": objective,
                           "writer_metric_energy": energies,
                           "rhs_norms": [float(torch.linalg.vector_norm(r.double())) for r in rhs],
                           "callback": telemetry})
        for r, grad, penalty in zip(rhs, gradients, balance_gradients):
            r.grad = grad + penalty
            _finite(r.grad, "R gradient")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for r in rhs:
            _finite(r, "updated R")
        counts["optimizer_updates"] += 1
    deltas = tuple(r @ j for r, j in zip(rhs, factors))
    for delta in deltas:
        _finite(delta, "final physical delta")
    final_loss, _, final_telemetry = observe(deltas)
    counts["final_observation_calls"] += 1
    energies, balance, _ = balance_terms()
    final = {"status": "FINITE_ENDPOINT", "position": "post_update_25",
             "current_nll": final_loss, "balance": balance,
             "objective": final_loss + balance_weight * balance,
             "writer_metric_energy": energies, "callback": final_telemetry}
    return JointPlan(labels, tuple(r.detach().clone() for r in rhs),
                     tuple(d.detach().clone() for d in deltas), q_raw, q_balanced,
                     tuple(q == 0 for q in q_raw), q_floor, sigma, tuple(trajectory),
                     final, tuple(row[2] for row in decompositions), counts)
