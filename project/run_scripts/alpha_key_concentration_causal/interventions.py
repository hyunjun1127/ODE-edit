"""Task-local O-H/O-W/K-R operations; no model loading or persistent writes.

Source contract: inputs/design/source/AlphaEdit_main.py (79da927aad5ab817...
fc842e), blue=False, physical layers 4..8, native ridge 10.  Keys and residuals
have observations in COLUMNS, whereas module input keys use their LAST axis.
The residual divisor (5/4/3/2/1) belongs to the caller, before ``native_solve``.

These are diagnostic interventions, not substitute editing methods.  They never
append history, fit z, select by N/P observations, or mutate input tensors.
Timestamp reconstruction, independent branch restoration, actual physical hook
parity, and receiving-state provenance remain mandatory caller validations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping, Sequence

import torch
import torch.nn.functional as F


class InterventionError(RuntimeError):
    """Technical input/numerical error, never a normal scientific fallback."""


@dataclass(frozen=True)
class HistoryOperand:
    M: torch.Tensor
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class WeightAblation:
    status: str
    reason: str | None
    parallel: torch.Tensor | None
    perpendicular: torch.Tensor | None
    norm_control: torch.Tensor | None
    direction: torch.Tensor | None
    alpha: float | None
    receipt: Mapping[str, Any]


@dataclass(frozen=True)
class KeyBasis:
    status: str
    reason: str | None
    vectors: torch.Tensor | None
    receipt: Mapping[str, Any]


def _tensor(value: torch.Tensor, name: str, *, ndim: int | None = None,
            dtype: torch.dtype = torch.float32) -> None:
    if not isinstance(value, torch.Tensor):
        raise InterventionError(f"{name}: expected tensor")
    if value.dtype != dtype or (ndim is not None and value.ndim != ndim):
        raise InterventionError(f"{name}: expected dtype={dtype}, ndim={ndim}")
    if not bool(torch.isfinite(value).all()):
        raise InterventionError(f"{name}: nonfinite")


def _same_device(*tensors: torch.Tensor) -> None:
    if len({tensor.device for tensor in tensors}) != 1:
        raise InterventionError("operands must share one explicit device")


def _square(matrix: torch.Tensor, dimension: int, name: str) -> None:
    _tensor(matrix, name, ndim=2)
    if matrix.shape != (dimension, dimension):
        raise InterventionError(f"{name}: wrong square shape")


def native_solve(K: torch.Tensor, R: torch.Tensor, M: torch.Tensor,
                 P: torch.Tensor, *, ridge: float = 10.0) -> torch.Tensor:
    """Return native FP32 update [d_out,d_in], preserving expression order.

    No inverse, symmetrization, projector replacement, RHS gate scaling, PSD
    clipping, dtype promotion, or history update is performed.  R has ALREADY
    received the native layer divisor.  Only the task's L2=10 is accepted.
    """
    _tensor(K, "K", ndim=2)
    _tensor(R, "R", ndim=2)
    if K.shape[1] != R.shape[1] or K.shape[1] == 0:
        raise InterventionError("K/R observation cardinality mismatch or empty")
    _square(M, K.shape[0], "M")
    _square(P, K.shape[0], "P")
    _same_device(K, R, M, P)
    if ridge != 10.0:
        raise InterventionError("native L2 is fixed at 10")
    # This parenthesization is the original direct-solve implementation.
    update = torch.linalg.solve(
        P @ (K @ K.T + M)
        + 10.0 * torch.eye(K.shape[0], dtype=torch.float32, device=K.device),
        P @ K @ R.T,
    ).T
    _tensor(update, "native update", ndim=2)
    return update


def _bank(K_stamp: torch.Tensor | None, K_current: torch.Tensor | None,
          dimension: int, bank_ids: Sequence[Hashable] | None,
          expected_bank_ids: Sequence[Hashable] | None,
          timestamp_validated: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if timestamp_validated is not True:
        raise InterventionError("timestamp key parity must be established before subtraction")
    if K_stamp is None or K_current is None:
        raise InterventionError("complete timestamp/current H512 bank is required")
    _tensor(K_stamp, "K_stamp", ndim=2)
    _tensor(K_current, "K_current", ndim=2)
    if K_stamp.shape != (dimension, 512) or K_current.shape != K_stamp.shape:
        raise InterventionError("history statistics require all 512 unweighted columns")
    if bank_ids is None or expected_bank_ids is None:
        raise InterventionError("history bank identity is required")
    ids = tuple(bank_ids)
    if ids != tuple(expected_bank_ids) or len(ids) != 512 or len(set(ids)) != 512:
        raise InterventionError("H512 identity/order/cardinality mismatch")
    _same_device(K_stamp, K_current)
    return K_stamp, K_current


def history_operand(M_native: torch.Tensor, K_stamp: torch.Tensor | None = None,
                    K_current: torch.Tensor | None = None, *, branch: str,
                    layer: int, bank_ids: Sequence[Hashable] | None = None,
                    expected_bank_ids: Sequence[Hashable] | None = None,
                    timestamp_validated: bool = False) -> HistoryOperand:
    """Build a temporary solve operand, leaving persistent native M untouched.

    SHAM really subtracts/re-adds in FP32 and may not be bitwise M_native.
    Its observed differences must be retained and tested by the caller.  MASS56
    uses trace(refresh)/trace(native), not a prespecified write gate.  No history
    validity mask is accepted: superseded H512 observations still have weight 1.
    """
    allowed = {"NATIVE", "SHAM", "H5", "H6", "H56", "MASS56"}
    if branch not in allowed or layer not in (4, 5, 6, 7, 8):
        raise InterventionError("unknown branch or physical layer")
    _tensor(M_native, "M_native", ndim=2)
    _square(M_native, M_native.shape[0], "M_native")
    applies = branch == "SHAM" or (branch == "H5" and layer == 5) or (
        branch == "H6" and layer == 6) or (branch in {"H56", "MASS56"} and layer in (5, 6))
    receipt: dict[str, Any] = {
        "branch": branch, "physical_layer": layer, "applied": applies,
        "persistent_history_mutated": False, "history_appends": 0,
        "psd_clipping": False, "psd_eigendecomposition": "NOT_PERFORMED",
        "native_solve_validity": "CALLER_REQUIRED", "dtype": "float32",
    }
    if not applies:
        return HistoryOperand(M_native.clone(), receipt)
    stamp, current = _bank(K_stamp, K_current, M_native.shape[0], bank_ids,
                           expected_bank_ids, timestamp_validated)
    _same_device(M_native, stamp, current)
    stamp_gram = stamp @ stamp.T
    replacement = stamp_gram if branch == "SHAM" else current @ current.T
    refresh = M_native - stamp_gram + replacement
    _tensor(refresh, "temporary refreshed M", ndim=2)
    trace_native = torch.trace(M_native)
    trace_refresh = torch.trace(refresh)
    receipt.update({
        "bank_n": 512, "bank_weight": 1, "includes_superseded": True,
        "timestamp_validated": True, "trace_native": float(trace_native),
        "trace_refresh": float(trace_refresh), "trace_stamp": float(torch.trace(stamp_gram)),
        "trace_current": float(torch.trace(replacement)),
        "sham_subtract_readd_executed": branch == "SHAM",
    })
    if branch == "MASS56":
        if float(trace_native) <= 0 or float(trace_refresh) < 0:
            raise InterventionError("MASS56 undefined or invalid history trace; no clipping allowed")
        scale = trace_refresh / trace_native
        result = M_native * scale
        receipt["mass_scale"] = float(scale)
    else:
        result = refresh
    _tensor(result, "temporary history operand", ndim=2)
    return HistoryOperand(result, receipt)


def history_penalty(delta: torch.Tensor, K_stamp: torch.Tensor,
                    K_current: torch.Tensor) -> dict[str, Any]:
    """All-column native action penalties, with signed per-request differences.

    Functional active/superseded reporting is separate.  No selection/filtering
    is performed here, and this helper does not claim projected-trace coverage.
    """
    _tensor(delta, "delta", ndim=2)
    _tensor(K_stamp, "K_stamp", ndim=2)
    _tensor(K_current, "K_current", ndim=2)
    if K_stamp.shape != K_current.shape or K_stamp.shape != (delta.shape[1], 512):
        raise InterventionError("history penalty requires the complete matching H512 bank")
    _same_device(delta, K_stamp, K_current)
    stamped = (delta @ K_stamp).double().square().sum(dim=0)
    current = (delta @ K_current).double().square().sum(dim=0)
    return {"stamp_per_request": stamped, "current_per_request": current,
            "signed_current_minus_stamp": current - stamped,
            "stamp_total": float(stamped.sum()), "current_total": float(current.sum()),
            "bank_n": 512, "bank_weight": 1}


class PromptActionMean:
    """Streaming equal-prompt mean of all-valid-token native FP32 actions.

    Feed every calibration prompt once, including every valid token (not just
    subject tokens).  Token sums and prompt sums use FP64 diagnostic reduction;
    final mean is materialized FP32.  No N/P observer input is permitted.
    """
    def __init__(self, *, panel_role: str = "calibration512") -> None:
        if panel_role != "calibration512":
            raise InterventionError("action mean must be fitted only on locked calibration512")
        self.total: torch.Tensor | None = None
        self.prompts = 0
        self.tokens = 0

    def add(self, actions: torch.Tensor, valid_mask: torch.Tensor) -> None:
        _tensor(actions, "prompt actions", ndim=2)
        mask = _valid_mask(valid_mask, actions.shape[:-1], actions.device)
        count = int(mask.sum())
        if count == 0:
            raise InterventionError("empty calibration prompt is not silently dropped")
        mean = actions[mask].double().sum(dim=0) / count
        if self.total is None:
            self.total = torch.zeros_like(mean)
        if self.total.shape != mean.shape or self.total.device != mean.device:
            raise InterventionError("calibration action dimension/device changed")
        self.total = self.total + mean
        self.prompts += 1
        self.tokens += count

    def finish(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.total is None or self.prompts == 0:
            raise InterventionError("calibration action mean has no prompts")
        mean = (self.total / self.prompts).float()
        _tensor(mean, "calibration action mean", ndim=1)
        return mean, {"panel_role": "calibration512", "prompts": self.prompts,
                      "valid_tokens": self.tokens, "prompt_weights": "equal",
                      "reduction": "FP64 token mean then FP64 prompt mean; FP32 output"}


def _valid_mask(mask: torch.Tensor, shape: torch.Size | tuple[int, ...],
                device: torch.device) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise InterventionError("explicit attention-derived valid-token boolean mask required")
    if mask.shape != shape or mask.device != device:
        raise InterventionError("valid-token mask shape/device mismatch")
    return mask


def component_action(keys: torch.Tensor, delta: torch.Tensor,
                     mean: torch.Tensor, mode: str) -> torch.Tensor:
    """Return no/mean/centered/full *activation* action, without modifying keys."""
    if mode not in {"no", "mean", "centered", "full"}:
        raise InterventionError("unknown activation component")
    _tensor(keys, "module keys")
    _tensor(delta, "delta", ndim=2)
    _tensor(mean, "calibration action mean", ndim=1)
    if keys.ndim < 2 or keys.shape[-1] != delta.shape[1] or mean.shape != (delta.shape[0],):
        raise InterventionError("component action orientation mismatch")
    _same_device(keys, delta, mean)
    if mode == "no":
        return keys.new_zeros((*keys.shape[:-1], delta.shape[0]))
    if mode == "mean":
        return mean.expand(*keys.shape[:-1], mean.shape[0])
    action = F.linear(keys, delta)
    return action - mean if mode == "centered" else action


def _output_first(output: Any) -> tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
    if isinstance(output, torch.Tensor):
        return output, lambda value: value
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0], lambda value: (value,) + output[1:]
    raise InterventionError("hook expects tensor output or a tuple with tensor first")


def action_component_hook(stage_weight: torch.Tensor, delta: torch.Tensor,
                          mean: torch.Tensor, mode: str,
                          valid_mask: torch.Tensor | Callable[[], torch.Tensor]
                          ) -> Callable[[Any, tuple[Any, ...], Any], Any]:
    """Forward-hook factory for down_proj, using a private immutable stage W.

    The FULL route is F.linear(k,U)+F.linear(k,delta), not secretly a physical
    weight write.  The caller MUST compare this with F.linear(k,FP32(U+delta))
    using the sealed actual FP32 repeat envelope.  Those routes are not assumed
    bitwise equivalent.  This distinction also applies to component sum parity.
    All valid tokens receive the same intervention; padding receives no action.
    A callable mask permits microbatch streaming, but may only return the true
    attention-mask validity, never a subject/prompt-class/oracle selection mask.
    """
    _tensor(stage_weight, "immutable stage weight", ndim=2)
    _tensor(delta, "delta", ndim=2)
    _tensor(mean, "mean", ndim=1)
    _same_device(stage_weight, delta, mean)
    if stage_weight.shape != delta.shape or mean.shape != (delta.shape[0],):
        raise InterventionError("hook stage/delta/mean dimensions mismatch")
    if mode not in {"no", "mean", "centered", "full"}:
        raise InterventionError("unknown activation component")
    owner = stage_weight.detach().clone()
    update = delta.detach().clone()
    mu = mean.detach().clone()
    static_mask = valid_mask.detach().clone() if isinstance(valid_mask, torch.Tensor) else None

    def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise InterventionError("module input tensor is missing")
        keys = inputs[0]
        actual, wrap = _output_first(output)
        _tensor(actual, "module output")
        base = F.linear(keys, owner)
        if actual.shape != base.shape:
            raise InterventionError("hook output shape mismatch")
        mask_value = static_mask if static_mask is not None else valid_mask()
        mask = _valid_mask(mask_value, keys.shape[:-1], keys.device)
        action = component_action(keys, update, mu, mode)
        result = base + torch.where(mask.unsqueeze(-1), action, torch.zeros_like(action))
        _tensor(result, "hook result")
        return wrap(result)

    return hook


def weight_ablation(delta: torch.Tensor, mean: torch.Tensor, *,
                    panel_role: str = "calibration512") -> WeightAblation:
    """Fixed calibration mean direction: uu.T Δ, complement, norm-matched Δ.

    These are weight-realizable components, unlike constant activation means.
    A zero mean returns NOT_APPLICABLE and NEVER chooses a replacement direction.
    Norm matching does not establish matched current editing quality.
    """
    _tensor(delta, "delta", ndim=2)
    _tensor(mean, "mean", ndim=1)
    _same_device(delta, mean)
    if mean.shape != (delta.shape[0],) or panel_role != "calibration512":
        raise InterventionError("weight ablation requires the locked calibration output mean")
    norm_mean = torch.linalg.vector_norm(mean)
    norm_delta = torch.linalg.vector_norm(delta)
    base_receipt = {"panel_role": panel_role, "rank": 1, "N_informed": False,
                    "delta_norm": float(norm_delta), "mean_norm": float(norm_mean),
                    "quality_matched": "NOT_ESTABLISHED"}
    if float(norm_mean) == 0.0:
        return WeightAblation("NOT_APPLICABLE", "ZERO_CALIBRATION_ACTION_MEAN",
                              None, None, None, None, None, base_receipt)
    if float(norm_delta) == 0.0:
        return WeightAblation("NOT_APPLICABLE", "ZERO_NATIVE_DELTA", None,
                              None, None, None, None, base_receipt)
    u = mean / norm_mean
    # Avoid constructing the d_out x d_out rank-one projection matrix.
    parallel = u[:, None] @ (u[None, :] @ delta)
    perpendicular = delta - parallel
    alpha_tensor = torch.linalg.vector_norm(perpendicular) / norm_delta
    control = delta * alpha_tensor
    for value in (parallel, perpendicular, control):
        _tensor(value, "weight ablation", ndim=2)
    receipt = {**base_receipt, "alpha": float(alpha_tensor),
               "sum_max_abs_error": float((parallel + perpendicular - delta).abs().max()),
               "norm_control_norm": float(torch.linalg.vector_norm(control)),
               "perpendicular_norm": float(torch.linalg.vector_norm(perpendicular))}
    return WeightAblation("DEFINED", None, parallel, perpendicular, control, u,
                          float(alpha_tensor), receipt)


def kr_cross_solve(K_a: torch.Tensor, K_b: torch.Tensor,
                   R_a: torch.Tensor, R_b: torch.Tensor, M: torch.Tensor,
                   P: torch.Tensor, *, receiving_state_id: str
                   ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Counterfactual K/R 2x2, one receiving W_b/P/M/L2, no history append.

    ``a/b`` describe source operands, not separate physical receiving states.
    Caller applies each returned update to the SAME restored W_b.  Downstream
    physical writes, if requested, must compute fresh K/R after that application.
    """
    if not isinstance(receiving_state_id, str) or not receiving_state_id:
        raise InterventionError("fixed receiving state identity is required")
    for name, value in (("K_a", K_a), ("K_b", K_b), ("R_a", R_a), ("R_b", R_b)):
        _tensor(value, name, ndim=2)
    if K_a.shape != K_b.shape or R_a.shape != R_b.shape or K_a.shape[1] != R_a.shape[1]:
        raise InterventionError("counterfactual operands must preserve request columns and shapes")
    updates = {p + q: native_solve(K, R, M, P)
               for p, K in (("a", K_a), ("b", K_b))
               for q, R in (("a", R_a), ("b", R_b))}
    interaction = (updates["bb"] - updates["ab"]) - (updates["ba"] - updates["aa"])
    return updates, {"receiving_state_id": receiving_state_id, "order": list(updates),
                     "operand_axes": "first=K source; second=R source", "ridge": 10,
                     "M_P_fixed": True, "history_appends": 0,
                     "interaction_frobenius": float(torch.linalg.vector_norm(interaction)),
                     "physical_evaluation": "CALLER_REQUIRED_SAME_W_b"}


def calibration_key_basis(keys: torch.Tensor, rank: int, *,
                          panel_role: str = "calibration512") -> KeyBasis:
    """Uncentered dominant input-key directions, from calibration keys only.

    Uses the small sample Gram in FP64, never a dense d_in x d_in eigensystem.
    The rank threshold is a fixed machine-precision numerical definiteness check
    (max(shape)*eps64*largest eigenvalue), not N-tuned scientific truncation.
    Requested ranks are exactly 1 (primary), 2 or 4 (auxiliary).  No substitute
    direction is returned if the requested rank cannot be numerically defined.
    """
    _tensor(keys, "calibration keys", ndim=2)
    if panel_role != "calibration512" or rank not in (1, 2, 4):
        raise InterventionError("basis may use only calibration512 and declared ranks 1/2/4")
    if keys.shape[1] == 0:
        raise InterventionError("empty calibration keys")
    work = keys.double()
    eigvals, eigvecs = torch.linalg.eigh(work.T @ work)
    order = torch.arange(len(eigvals) - 1, -1, -1, device=keys.device)
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    largest = float(eigvals[0])
    cutoff = max(keys.shape) * torch.finfo(torch.float64).eps * max(largest, 0.0)
    available = int((eigvals > cutoff).sum())
    receipt = {"panel_role": panel_role, "rank_requested": rank, "numerical_rank": available,
               "numerical_eigenvalue_cutoff": cutoff, "centered": False,
               "keys_shape": list(keys.shape), "N_informed": False,
               "top_eigenvalues": eigvals[:4].tolist()}
    if available < rank:
        return KeyBasis("NOT_APPLICABLE", "INSUFFICIENT_CALIBRATION_KEY_RANK", None, receipt)
    basis = work @ eigvecs[:, :rank] / eigvals[:rank].sqrt().unsqueeze(0)
    # QR only orthonormalizes this same computed span. It does not fit new data.
    basis = torch.linalg.qr(basis, mode="reduced").Q.float()
    _tensor(basis, "key basis", ndim=2)
    receipt["orthogonality_max_abs"] = float(
        (basis.double().T @ basis.double() - torch.eye(rank, device=basis.device)).abs().max())
    return KeyBasis("DEFINED", None, basis, receipt)


def interchange_keys(base: torch.Tensor, donor: torch.Tensor,
                     basis: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """All-valid-token k_base + Pi(k_donor-k_base); no observer-based fitting."""
    _tensor(base, "base keys")
    _tensor(donor, "donor keys")
    _tensor(basis, "basis", ndim=2)
    _same_device(base, donor, basis)
    if base.shape != donor.shape or base.ndim < 2 or basis.shape[0] != base.shape[-1]:
        raise InterventionError("interchange input/basis shape mismatch")
    if basis.shape[1] not in (1, 2, 4):
        raise InterventionError("undeclared interchange rank")
    mask = _valid_mask(valid_mask, base.shape[:-1], base.device)
    move = ((donor - base) @ basis) @ basis.T
    result = base + torch.where(mask.unsqueeze(-1), move, torch.zeros_like(move))
    _tensor(result, "interchanged keys")
    return result


def interchange_key_hook(donor_keys: torch.Tensor, basis: torch.Tensor,
                         valid_mask: torch.Tensor
                         ) -> Callable[[Any, tuple[Any, ...]], tuple[Any, ...]]:
    """Pre-hook for a fixed microbatch; preserves extra positional arguments.

    Caller binds token IDs/masks/input state so donor and receiving tokens are
    identical.  This affects inference keys, NOT the writer K/R solve operands.
    """
    donor = donor_keys.detach().clone()
    directions = basis.detach().clone()
    mask = valid_mask.detach().clone()

    def hook(module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise InterventionError("interchange module input tensor missing")
        return (interchange_keys(inputs[0], donor, directions, mask),) + inputs[1:]

    return hook
