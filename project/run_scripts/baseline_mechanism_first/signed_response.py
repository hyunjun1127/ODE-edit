"""E1 signed all-position output-response observations, not update selection.

C5 activation/output-gradient contraction is adapted to the ACTUAL native
dense direction. No progress_slope, objective, clipping, or performance sweep.
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def desired_margin(true_nll: Any, new_nll: Any, category: str, *, raw_margin: Any = None) -> dict[str, Any]:
    if category not in {"R", "P", "N"}:
        raise ValueError("category must be R, P, or N; General NLL is separate")
    for value in (true_nll, new_nll):
        if not bool(torch.isfinite(torch.as_tensor(value)).all()):
            raise ValueError("NONFINITE_NLL")
    margin = true_nll - new_nll if category in {"R", "P"} else new_nll - true_nll
    if not bool(torch.isfinite(torch.as_tensor(margin)).all()):
        raise ValueError("NONFINITE_DESIRED_MARGIN")
    return {"nll_true": true_nll, "nll_new": new_nll, "raw_margin": raw_margin,
            "desired_margin": margin, "success": margin > 0,
            "true_coefficient": 1 if category in {"R", "P"} else -1,
            "new_coefficient": -1 if category in {"R", "P"} else 1}


def teacher_forced_nll(logits: torch.Tensor, token_ids: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """Mean answer-token NLL per sequence; mask is in unshifted token coordinates.

    The mask may cover every target token, never just the first answer token.
    It is independent of the activation position mask: causal prefix activations
    remain in the derivative even though their tokens are not NLL targets.
    """
    if logits.ndim != 3 or token_ids.shape != logits.shape[:2] or target_mask.shape != token_ids.shape:
        raise ValueError("teacher-forcing geometry mismatch")
    if target_mask.dtype != torch.bool or bool(target_mask[:, 0].any()):
        raise ValueError("target mask must be boolean and exclude unpredicted position zero")
    mask = target_mask[:, 1:]
    counts = mask.sum(-1)
    if bool((counts == 0).any()):
        raise ValueError("EMPTY_TARGET_MASK")
    losses = F.cross_entropy(logits[:, :-1].transpose(1, 2), token_ids[:, 1:], reduction="none")
    return (losses * mask).sum(-1) / counts


def activation_gradient_contraction(inputs: torch.Tensor, grad_output: torch.Tensor,
                                    actual_delta: torch.Tensor,
                                    *, position_mask: torch.Tensor | None = None,
                                    diagnostic_factor: tuple[torch.Tensor, torch.Tensor] | None = None) -> dict[str, Any]:
    """Full-position sum plus separately labeled exclusions; no hidden clipping."""
    if inputs.shape[:-1] != grad_output.shape[:-1] or actual_delta.shape != (grad_output.shape[-1], inputs.shape[-1]):
        raise ValueError("activation/dense direction geometry mismatch")
    dtype = torch.float64 if torch.float64 in (inputs.dtype, grad_output.dtype, actual_delta.dtype) else torch.float32
    x, g = inputs.detach().to(dtype), grad_output.detach().to(dtype)
    response = x @ actual_delta.detach().to(device=x.device, dtype=dtype).T
    contribution = (g * response).sum(-1)
    if not bool(torch.isfinite(contribution).all()):
        raise ValueError("NONFINITE_CONTRACTION")
    if position_mask is None:
        position_mask = torch.ones_like(contribution, dtype=torch.bool)
    if position_mask.shape != contribution.shape or position_mask.dtype != torch.bool:
        raise ValueError("position mask must cover every activation position")
    raw = float(contribution.sum())
    included = float(contribution[position_mask].sum())
    excluded = float(contribution[~position_mask].sum())
    result = {"event_derivative": raw, "raw_all_position_derivative": raw,
            "included_position_derivative": included, "excluded_position_derivative": excluded,
            "raw_position_count": contribution.numel(), "included_position_count": int(position_mask.sum()),
            "excluded_position_count": int((~position_mask).sum()),
            "status": "FINITE_ZERO" if raw == 0 else "FINITE",
            "direction_source": "ACTUAL_NATIVE_MATERIALIZED_DENSE"}
    if diagnostic_factor is not None:
        R, factor_F = diagnostic_factor
        if R.ndim != 2 or factor_F.ndim != 2 or R.shape[0] != actual_delta.shape[0] or factor_F.shape != (actual_delta.shape[1], R.shape[1]):
            raise ValueError("diagnostic factor geometry mismatch")
        factor_response = (x @ factor_F.detach().to(x)) @ R.detach().to(x).T
        factor_slope = float((g * factor_response).sum())
        if not math.isfinite(factor_slope):
            raise ValueError("NONFINITE_FACTOR_CONTRACTION")
        result.update({"diagnostic_factor_derivative": factor_slope,
                       "factor_minus_actual_derivative": factor_slope - raw})
    return result


class AllPositionContraction:
    """One diagnostic forward/backward capture for a single linear actuator.

    Output values, weights, flags and .grad slots are not changed. When the
    model is fully frozen the output is returned as a grad-enabled leaf with
    identical values (diagnostic graph only). Never install during compute-z
    or the native writer. autograd.grad stops at outputs, avoiding dense dW.
    """
    def __init__(self, module: torch.nn.Module, actual_delta: torch.Tensor,
                 *, position_mask: torch.Tensor | None = None,
                 diagnostic_factor: tuple[torch.Tensor, torch.Tensor] | None = None):
        if not isinstance(module, torch.nn.Linear):
            raise ValueError("explicit nn.Linear [out,in] actuator required")
        if actual_delta.shape != module.weight.shape:
            raise ValueError("actual dense direction shape differs from actuator")
        self.module, self.delta, self.position_mask = module, actual_delta.detach(), position_mask
        self.diagnostic_factor = diagnostic_factor
        self.records: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        self.handle = None
        self.computed = False

    def _capture(self, module: torch.nn.Module, args: tuple[Any, ...], output: torch.Tensor):
        if not torch.is_grad_enabled():
            raise ValueError("signed diagnostic requires grad-enabled forward")
        if not args or not isinstance(args[0], torch.Tensor) or not isinstance(output, torch.Tensor):
            raise ValueError("tensor input and output required")
        observed = output if output.requires_grad else output.detach().requires_grad_(True)
        self.records.append((args[0].detach(), observed, args[0]._version))
        return None if observed is output else observed

    def __enter__(self):
        if self.handle is not None or self.records:
            raise ValueError("capture is single-use")
        w = self.module.weight
        self.original = (w.data_ptr(), w._version, w.requires_grad, w.grad)
        self.delta_version = self.delta._version
        self.handle = self.module.register_forward_hook(self._capture)
        return self

    def compute(self, scalar: torch.Tensor, *, retain_graph: bool = False) -> dict[str, Any]:
        if self.handle is None or self.computed or not self.records:
            raise ValueError("one compute requires active nonempty capture")
        if scalar.ndim != 0 or not scalar.requires_grad:
            raise ValueError("desired margin or General NLL must be grad-enabled scalar")
        gradients = torch.autograd.grad(scalar, tuple(r[1] for r in self.records),
                                        retain_graph=retain_graph, allow_unused=True)
        rows = []
        for (x, out, version), grad in zip(self.records, gradients):
            if x._version != version or self.delta._version != self.delta_version:
                raise ValueError("captured input or dense direction mutated")
            if grad is None:
                grad = torch.zeros_like(out)
            rows.append(activation_gradient_contraction(x, grad, self.delta, position_mask=self.position_mask,
                                                        diagnostic_factor=self.diagnostic_factor))
        keys = ("event_derivative", "raw_all_position_derivative", "included_position_derivative",
                "excluded_position_derivative", "raw_position_count", "included_position_count", "excluded_position_count")
        result = {key: sum(row[key] for row in rows) for key in keys}
        if self.diagnostic_factor is not None:
            for key in ("diagnostic_factor_derivative", "factor_minus_actual_derivative"):
                result[key] = sum(row[key] for row in rows)
        result.update({"status": "FINITE_ZERO" if result["event_derivative"] == 0 else "FINITE",
                       "module_calls": len(rows), "unused_output_count": sum(g is None for g in gradients),
                       "direction_source": "ACTUAL_NATIVE_MATERIALIZED_DENSE"})
        self.computed = True
        return result

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.records.clear()
        w = self.module.weight
        ptr, version, flag, grad = self.original
        if w.data_ptr() != ptr or w._version != version or w.requires_grad != flag or w.grad is not grad:
            raise ValueError("OBSERVER_WEIGHT_INTERFERENCE")
        return False


def finite_difference_audit(derivative: float, minus: float, zero: float, plus: float, *,
                            alpha: float, forward_noise: float, absolute_tolerance: float,
                            relative_tolerance: float = 0.0) -> dict[str, Any]:
    """Audit a single PRESEALED common small ±alpha probe; does not run/select it.

    Native alpha=1 is deliberately not evaluated by this interface. A finite
    resolution failure limits this derivative validation, not the whole cell.
    """
    values = (derivative, minus, zero, plus, alpha, forward_noise, absolute_tolerance, relative_tolerance)
    if not all(math.isfinite(float(x)) for x in values):
        return {"status": "NONFINITE", "derivative_validation": "TECHNICAL_HOLD"}
    if not 0 < alpha < 1 or min(forward_noise, absolute_tolerance, relative_tolerance) < 0:
        raise ValueError("small finite probe and nonnegative presealed tolerances required")
    slope = (plus - minus) / (2 * alpha)
    error = abs(slope - derivative)
    resolved = abs(plus - minus) > 2 * forward_noise
    tolerance = absolute_tolerance + relative_tolerance * abs(derivative)
    status = "NUMERICALLY_UNRESOLVED" if not resolved else ("PASS" if error <= tolerance else "DERIVATIVE_MISMATCH")
    return {"status": status, "event_derivative": derivative, "alpha": alpha,
            "minus": minus, "zero": zero, "plus": plus, "central_slope": slope,
            "positive_one_sided_slope": (plus - zero) / alpha,
            "absolute_error": error, "relative_error": None if derivative == 0 else error / abs(derivative),
            "forward_noise": forward_noise, "slope_noise_bound": forward_noise / alpha,
            "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance,
            "performance_selection": False}
