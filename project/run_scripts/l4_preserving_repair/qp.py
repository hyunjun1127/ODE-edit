"""Certified small-dimensional/many-guard QP; CPU FP64, no model execution.

This is a primal feasible active-set projection solver, not the design's toy
combinatorial enumerator. Every step scans all rows, and the working set contains
at most three independent normals. Failure to certify is a technical exception.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

from .geometry import DEFAULT_GEOMETRY_POLICY, GeometryPolicy, whiten_gn


@dataclass(frozen=True)
class QPPolicy:
    kkt_tolerance: float = 1e-8
    step_tolerance: float = 1e-12
    rank_relative: float = 1e-12
    multiplier_tolerance: float = 1e-12
    blocking_direction_tolerance: float = 1e-14
    max_iterations: int = 10000
    gain_zero_tolerance: float = 1e-6


DEFAULT_QP_POLICY = QPPolicy()


class QPError(RuntimeError):
    def __init__(self, code: str, receipt: dict | None = None):
        super().__init__(code)
        self.code = code
        self.receipt = receipt or {}


def _max(value):
    return float(np.max(value)) if np.size(value) else 0.0


def kkt_receipt(g, C, limits, u, multipliers, row_scales):
    """Keep original-unit violations and normalized row checks side-by-side.

    No global matrix-norm denominator, clipping negative duals, or ignoring a
    large row is used. Both raw and row-normalized residuals must pass. A very
    poorly scaled problem can therefore fail closed instead of hiding a defect.
    """
    g, C, limits, u, multipliers, row_scales = [np.asarray(x, dtype=np.float64)
                                               for x in (g, C, limits, u, multipliers, row_scales)]
    slack = limits - C @ u
    stationarity = u + g + C.T @ multipliers
    scaled_dual = multipliers * row_scales
    scaled_slack = slack / row_scales
    complementarity = multipliers * slack
    return {
        "raw": {"primal_max_positive_violation": max(0.0, _max(-slack)),
                "dual_max_negative_violation": max(0.0, _max(-multipliers)),
                "stationarity_inf": _max(np.abs(stationarity)),
                "complementarity_inf": _max(np.abs(complementarity))},
        "row_normalized": {"primal_max_positive_violation": max(0.0, _max(-scaled_slack)),
                           "dual_max_negative_violation": max(0.0, _max(-scaled_dual)),
                           "stationarity_inf": _max(np.abs(stationarity)),
                           "complementarity_inf": _max(np.abs(scaled_dual * scaled_slack))},
        "slack_original_units": slack.tolist(), "multipliers_original_units": multipliers.tolist(),
        "slack_row_normalized": scaled_slack.tolist(), "multipliers_row_normalized": scaled_dual.tolist(),
        "stationarity_vector": stationarity.tolist(),
        "row_scales_l2_norm": row_scales.tolist(),
        "scaling_convention": "C_normalized=C/row_scale; multiplier_raw=multiplier_normalized/row_scale",
    }


def solve_box_qp(g, A, slack, radius: float, policy: QPPolicy = DEFAULT_QP_POLICY):
    """Solve min g'u+||u||²/2, A u<=slack, |u_j|<=radius.

    The design guarantees zero feasible; a negative input slack is an error,
    including a tiny negative one. The function returns only JSON-safe values.
    """
    g = np.asarray(g, dtype=np.float64)
    if g.ndim != 1 or not 1 <= g.size <= 3:
        raise QPError("QP_DIMENSION")
    m = g.size
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0:
        A = np.empty((0, m), dtype=np.float64)
    slack = np.asarray(slack, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] != m or slack.shape != (A.shape[0],):
        raise QPError("QP_SHAPE")
    if not math.isfinite(radius) or radius <= 0:
        raise QPError("QP_INVALID_RADIUS")
    if not np.isfinite(g).all() or not np.isfinite(A).all() or not np.isfinite(slack).all():
        raise QPError("QP_NONFINITE")
    if np.any(slack < 0):
        raise QPError("QP_ZERO_ANCHOR_INFEASIBLE", {"min_slack": float(np.min(slack))})
    C = np.concatenate((A, np.eye(m), -np.eye(m)))
    limits = np.concatenate((slack, np.full(2 * m, radius)))
    norms = np.linalg.norm(C, axis=1)
    if not np.isfinite(norms).all():
        raise QPError("QP_ROW_NORM_OVERFLOW")
    row_scales = np.where(norms == 0, 1.0, norms)
    N, h = C / row_scales[:, None], limits / row_scales
    u = np.zeros(m, dtype=np.float64)
    working: list[int] = []
    trace = []
    final_dual = None
    for iteration in range(policy.max_iterations):
        gradient = u + g
        if working:
            W = N[working]
            _, singular, vh = np.linalg.svd(W, full_matrices=True)
            rank = int(np.sum(singular > policy.rank_relative * singular[0]))
            if rank != len(working):
                raise QPError("QP_DEPENDENT_WORKING_SET", {"working": working, "singular": singular.tolist()})
            null = vh[rank:].T
            direction = -(null @ (null.T @ gradient))
        else:
            direction = -gradient
        if np.linalg.norm(direction) <= policy.step_tolerance * max(1.0, np.linalg.norm(gradient)):
            dual_work = np.linalg.lstsq(N[working].T, -gradient, rcond=policy.rank_relative)[0] if working else np.empty(0)
            negative = [j for j, value in enumerate(dual_work) if value < -policy.multiplier_tolerance]
            if negative:
                # Most negative, original row ID tie-break. No row reordering.
                drop = min(negative, key=lambda j: (float(dual_work[j]), working[j]))
                trace.append({"iteration": iteration, "action": "REMOVE", "row": working[drop]})
                working.pop(drop)
                continue
            final_dual = np.zeros(C.shape[0], dtype=np.float64)
            if working:
                final_dual[working] = dual_work / row_scales[working]
            break
        rate = N @ direction
        remaining = h - N @ u
        eligible = rate > policy.blocking_direction_tolerance * max(np.linalg.norm(direction), np.finfo(np.float64).tiny)
        if working:
            eligible[working] = False
        indices = np.flatnonzero(eligible)
        alpha, blocker = 1.0, None
        if indices.size:
            ratios = remaining[indices] / rate[indices]
            position = int(np.argmin(ratios))
            proposed = float(ratios[position])
            if proposed < -policy.kkt_tolerance:
                raise QPError("QP_PRIMAL_DRIFT", {"ratio": proposed, "row": int(indices[position])})
            if proposed < 1.0:
                alpha, blocker = max(0.0, proposed), int(indices[position])
        u += alpha * direction
        if blocker is not None:
            working.append(blocker)
            trace.append({"iteration": iteration, "action": "ADD", "row": blocker, "step": alpha})
        elif alpha == 1.0:
            trace.append({"iteration": iteration, "action": "FULL_STEP"})
    if final_dual is None:
        raise QPError("QP_ITERATION_LIMIT", {"iterations": policy.max_iterations, "trace": trace})
    certificate = kkt_receipt(g, C, limits, u, final_dual, row_scales)
    certificate_pass = all(value <= policy.kkt_tolerance for group in ("raw", "row_normalized")
                           for value in certificate[group].values())
    objective = float(g @ u + 0.5 * (u @ u))
    result = {"status": "CERTIFIED", "method": "FP64_PRIMAL_FEASIBLE_ACTIVE_SET",
              "policy": asdict(policy), "u": u.tolist(), "objective_change": objective,
              "predicted_gain": -objective, "radius": float(radius), "dimension": int(m),
              "guard_rows": int(A.shape[0]), "box_rows": int(2 * m),
              "working_set": working, "iterations": iteration + 1, "trace": trace,
              "KKT": certificate, "KKT_pass": certificate_pass}
    if not certificate_pass:
        raise QPError("QP_KKT_UNRESOLVED", result)
    if objective > policy.kkt_tolerance:
        raise QPError("QP_OBJECTIVE_WORSE_THAN_ZERO", result)
    return result


def solve_repair_qp(b, H, A, s, base_loss: float, arm: str, radius: float | None = None,
                    policy: QPPolicy = DEFAULT_QP_POLICY,
                    geometry_policy: GeometryPolicy = DEFAULT_GEOMETRY_POLICY):
    """Whiten and solve same-basis/same-box safe and free proposals.

    R-GD requires m=1 and removes edit-response guards only in its *proposal*.
    This helper does not bypass actual nonlinear rewrite acceptance.
    """
    if arm not in ("R-GD", "R-QP"):
        raise QPError("UNAUTHORIZED_REPAIR_ARM")
    b = np.asarray(b, dtype=np.float64)
    if b.ndim != 1 or b.size > 3 or (arm == "R-GD" and b.size > 1):
        raise QPError("REPAIR_DIRECTION_DIMENSION")
    H = np.asarray(H, dtype=np.float64)
    A, s = np.asarray(A, dtype=np.float64), np.asarray(s, dtype=np.float64)
    if b.size == 0 and H.size == 0:
        H = np.empty((0, 0), dtype=np.float64)
    if A.size == 0:
        A = np.empty((0, b.size), dtype=np.float64)
    if H.shape != (b.size, b.size) or A.ndim != 2 or A.shape[1] != b.size or s.shape != (A.shape[0],):
        raise QPError("REPAIR_MODEL_SHAPE")
    if not all(np.isfinite(value).all() for value in (b, H, A, s)):
        raise QPError("REPAIR_MODEL_NONFINITE")
    if np.any(s < 0):
        raise QPError("REPAIR_ZERO_ANCHOR_INFEASIBLE")
    if not math.isfinite(base_loss) or base_loss < 0:
        raise QPError("REPAIR_BASE_NONFINITE_OR_NEGATIVE")
    if b.size == 0 or base_loss <= policy.gain_zero_tolerance:
        return {"status": "NORMAL_OFF", "reason": "ZERO_DIMENSION" if b.size == 0 else "BASE_SMALL",
                "u": [0.0] * b.size, "c": [0.0] * b.size, "predicted_gain": 0.0,
                "Gamma_free": 0.0, "Gamma_safe": 0.0, "safe_over_free": None}
    white = whiten_gn(H, b, geometry_policy)
    if white["status"] == "NUMERICAL_ZERO_GN_AND_GRADIENT":
        return {"status": "NORMAL_OFF", "reason": white["status"], "whitening": white,
                "u": [0.0] * b.size, "c": [0.0] * b.size, "predicted_gain": 0.0,
                "Gamma_free": 0.0, "Gamma_safe": 0.0, "safe_over_free": None}
    R = np.asarray(white["R"], dtype=np.float64)
    r = math.sqrt(2.0 * base_loss / b.size) if radius is None else radius
    g = R.T @ b
    free = solve_box_qp(g, [], [], r, policy)
    safe = solve_box_qp(g, A @ R, s, r, policy) if arm == "R-QP" else free
    gain_free, gain_safe = free["predicted_gain"], safe["predicted_gain"]
    if gain_safe > gain_free + policy.kkt_tolerance:
        raise QPError("SAFE_GAIN_EXCEEDS_SAME_BOX_FREE_GAIN", {"safe": safe, "free": free})
    u = np.asarray(safe["u"], dtype=np.float64)
    return {"status": "CERTIFIED", "arm": arm, "whitening": white,
            "u": u.tolist(), "c": (R @ u).tolist(), "radius": float(r),
            "predicted_gain": gain_safe, "Gamma_free": gain_free, "Gamma_safe": gain_safe,
            "safe_over_free": gain_safe / gain_free if gain_free > policy.gain_zero_tolerance else None,
            "proposal_guards_used": arm == "R-QP", "safe": safe, "free": free}
