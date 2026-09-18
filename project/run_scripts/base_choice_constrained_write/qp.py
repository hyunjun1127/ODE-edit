"""FP64 minimum-norm half-space QP, with full-bank certified GSS ordering.

Input G is the *full*, unregularized Gram of projected pair gradients.  This
module never receives a model, reference labels, a norm budget, or a ridge.
It solves min .5 ||D||^2, <h_i,D> >= b_i and returns original-unit multipliers
for the POSITIVE reconstruction D = sum_i alpha_i h_i.  Certificates concern
this local linear problem only, never nonlinear choice or Llama validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Sequence

import numpy as np
from scipy import linalg


@dataclass(frozen=True)
class QPPolicy:
    """Numerical/iteration constants to seal before model execution.

    Tolerances certify residuals, not a scientific preservation/norm budget.
    Singular solves are allowed only with an original-Gram KKT certificate;
    small eigenvalues are never replaced by a ridge.
    """

    max_rows: int = 1024
    block_size: int = 32
    feasibility_absolute: float = 1e-9
    feasibility_relative: float = 1e-10
    objective_relative: float = 1e-8
    symmetry_relative: float = 1e-12
    psd_relative: float = 1e-12
    numerical_rank_relative: float = 1e-13
    eigenvalue_roundoff_multiplier: float = 2.0
    recession_residual_relative: float = 2e-12
    max_pivots: int = 10000


DEFAULT_QP_POLICY = QPPolicy()


class QPError(RuntimeError):
    """Typed non-success, retaining replayable numerical diagnostics."""

    def __init__(self, code: str, receipt: dict | None = None):
        super().__init__(code)
        self.code = code
        self.receipt = {**(receipt or {}), "status": code}


class QPInputError(QPError):
    """Malformed, nonfinite, asymmetric or non-PSD input."""


class QPInfeasible(QPError):
    """Local linear infeasibility with a recorded numerical Farkas witness."""


class QPZeroRowInfeasible(QPInfeasible):
    """An exactly zero projected gradient has a strictly positive RHS."""


class QPSolverFailure(QPError):
    """Uncertified numerical/iteration failure, NOT infeasibility."""


def _maximum(values) -> float:
    return float(np.max(values, initial=0.0))


def _id_key(value: Any) -> str:
    # Pair IDs commonly are (reference_id, position, competitor_id) tuples.
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise QPInputError("QP_INVALID_ID", {"type": type(value).__name__}) from exc


def _prepare(gram, b, ids: Sequence, policy: QPPolicy):
    if np.iscomplexobj(gram) or np.iscomplexobj(b):
        raise QPInputError("QP_COMPLEX_INPUT")
    try:
        raw_g, rhs = np.asarray(gram, dtype=np.float64), np.asarray(b, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise QPInputError("QP_INVALID_ARRAY") from exc
    if rhs.ndim != 1 or raw_g.shape != (rhs.size, rhs.size):
        raise QPInputError("QP_SHAPE")
    n = rhs.size
    if n > policy.max_rows:
        raise QPInputError("QP_ROW_CAP", {"rows": int(n), "cap": policy.max_rows})
    if len(ids) != n:
        raise QPInputError("QP_ID_COUNT")
    keys = [_id_key(i) for i in ids]
    if len(set(keys)) != n:
        raise QPInputError("QP_DUPLICATE_ID")
    if not np.isfinite(raw_g).all() or not np.isfinite(rhs).all():
        raise QPInputError("QP_NONFINITE")
    diagonal = np.diag(raw_g)
    if np.any(diagonal < 0):
        raise QPInputError("QP_NEGATIVE_DIAGONAL")
    zero = diagonal == 0
    # For a PSD Gram, an exactly zero diagonal entails an exactly zero row.
    if np.any(raw_g[zero] != 0) or np.any(raw_g[:, zero] != 0):
        raise QPInputError("QP_ZERO_DIAGONAL_NONZERO_ROW")
    norms = np.sqrt(diagonal)
    scales = np.where(zero, 1.0, norms)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized = raw_g / scales[:, None] / scales[None, :]
        beta = rhs / scales
    if not np.isfinite(normalized).all() or not np.isfinite(beta).all():
        raise QPInputError("QP_NORMALIZATION_NONFINITE")
    asymmetry = _maximum(np.abs(normalized - normalized.T))
    if asymmetry > policy.symmetry_relative:
        raise QPInputError("QP_ASYMMETRIC", {"normalized_asymmetry": asymmetry})
    # Only antisymmetric floating-point noise is removed. Final feasibility is
    # independently checked against the caller's original, unsymmetrized Gram.
    symmetric = 0.5 * normalized + 0.5 * normalized.T
    if n:
        try:
            eigenvalues = linalg.eigvalsh(symmetric, check_finite=False)
        except linalg.LinAlgError as exc:
            raise QPSolverFailure("QP_SPECTRUM_FAILED") from exc
        smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    else:
        smallest = largest = 0.0
    if smallest < -policy.psd_relative * max(1.0, largest):
        raise QPInputError("QP_NOT_PSD", {"minimum_eigenvalue": smallest,
                                         "maximum_eigenvalue": largest})
    impossible = np.flatnonzero(zero & (rhs > 0))
    if impossible.size:
        row = min(impossible, key=lambda i: keys[i])
        raise QPZeroRowInfeasible("QP_ZERO_ROW_INFEASIBLE", {
            "row": int(row), "id": ids[row], "b": float(rhs[row]),
            "scope": "local first-order problem only"})
    metadata = {"minimum_eigenvalue": smallest, "maximum_eigenvalue": largest,
                "normalized_asymmetry": asymmetry,
                "zero_rows": np.flatnonzero(zero).tolist(),
                "ridge": 0.0, "spectral_clipping_applied_to_problem": False}
    return raw_g, rhs, keys, norms, scales, symmetric, normalized, beta, zero, metadata


def _row_limits(beta, response, policy):
    return policy.feasibility_absolute + policy.feasibility_relative * np.maximum(
        np.abs(beta), np.abs(response))


def _certificate(raw_g, b, normalized_g, beta, scales, alpha, policy):
    lam = alpha * scales
    with np.errstate(over="ignore", invalid="ignore"):
        raw_response = raw_g @ alpha
        response = normalized_g @ lam
        slack = response - beta
        raw_slack = raw_response - b
        quadratic = float(lam @ response)
        primal = 0.5 * quadratic
        dual = float(beta @ lam) - primal
        gap = primal - dual
        complementarity = lam * slack
        projected_gradient = np.minimum(lam, slack)
    finite = all(np.isfinite(x).all() for x in (
        alpha, raw_response, raw_slack, response, complementarity, projected_gradient,
        np.array([primal, dual, gap])))
    if not finite:
        raise QPSolverFailure("QP_CERTIFICATE_NONFINITE")
    limits = _row_limits(beta, response, policy)
    objective_limit = policy.objective_relative * max(1.0, abs(primal), abs(dual))
    # Scaling changes units, not the inequalities. Raw and normalized products
    # are both evaluated; neither a large row nor global matrix norm hides one.
    raw_limits = scales * limits
    if not np.isfinite(raw_limits).all():
        raise QPSolverFailure("QP_TOLERANCE_SCALE_NONFINITE")
    feasibility = bool(np.all(slack >= -limits) and np.all(raw_slack >= -raw_limits))
    dual_feasible = bool(np.all(alpha >= 0))
    stationarity = bool(np.all(np.abs(projected_gradient) <= limits))
    complementary = _maximum(np.abs(complementarity)) <= objective_limit
    gap_pass = abs(gap) <= objective_limit
    result = {
        "all_rows_checked": int(len(b)), "feasibility_pass": feasibility,
        "dual_feasibility_pass": dual_feasible, "stationarity_pass": stationarity,
        "complementarity_pass": bool(complementary), "gap_pass": bool(gap_pass),
        "KKT_pass": bool(feasibility and dual_feasible and stationarity and complementary and gap_pass),
        "primal_objective": primal, "dual_objective": dual, "gap": gap,
        "objective_tolerance": objective_limit,
        "primal_max_violation": _maximum(-raw_slack),
        "normalized_primal_max_violation": _maximum(-slack),
        "dual_max_negative_violation": _maximum(-alpha),
        "stationarity_projected_gradient_inf": _maximum(np.abs(projected_gradient)),
        "primal_stationarity": "D=sum(alpha_i*h_i) by construction; independent direction not supplied",
        "complementarity_inf": _maximum(np.abs(complementarity)),
        "slack_original_units": raw_slack.tolist(), "slack_normalized": slack.tolist(),
        "row_feasibility_tolerances_normalized": limits.tolist(),
        "row_feasibility_tolerances_original_units": raw_limits.tolist(),
        "complementarity": complementarity.tolist(),
    }
    return result


def _block(candidates, selected, violations, gram, keys, order, size):
    """Mandatory worst violation, then max-min distance to all selected rows."""
    pool = set(map(int, candidates))
    if not pool:
        return []
    worst = min(pool, key=lambda i: (-float(violations[i]), keys[i]))
    added = [worst]
    pool.remove(worst)
    if order == "most_violation":
        return added + sorted(pool, key=lambda i: (-float(violations[i]), keys[i]))[:size - 1]
    context = list(selected) + added
    max_similarity = np.max(gram[:, context], axis=1)
    while pool and len(added) < size:
        # min_j (1-cos(i,j)) = 1-max_j cos(i,j). Do not use abs(cos).
        next_row = min(pool, key=lambda i: (float(max_similarity[i]),
                                           -float(violations[i]), keys[i]))
        added.append(next_row)
        pool.remove(next_row)
        max_similarity = np.maximum(max_similarity, gram[:, next_row])
    return added


def _stationary_or_ray(matrix, rhs, policy):
    """Unconstrained stationary point or null-space descent, without a ridge."""
    try:
        factor = linalg.cho_factor(matrix, lower=True, check_finite=False)
        point = linalg.cho_solve(factor, rhs, check_finite=False)
        residual = rhs - matrix @ point
        if np.isfinite(point).all() and _maximum(np.abs(residual)) <= (
                policy.feasibility_absolute * 0.1 + policy.feasibility_relative * _maximum(np.abs(rhs))):
            return point, None
    except linalg.LinAlgError:
        pass
    try:
        values, vectors = linalg.eigh(matrix, check_finite=False)
    except linalg.LinAlgError as exc:
        raise QPSolverFailure("QP_FACE_EIGENSOLVE_FAILED") from exc
    cutoff = policy.numerical_rank_relative * max(1.0, float(values[-1]))
    retained = values > cutoff
    point = vectors[:, retained] @ ((vectors[:, retained].T @ rhs) / values[retained])
    residual = rhs - matrix @ point
    threshold = policy.feasibility_absolute * 0.1 + policy.feasibility_relative * _maximum(np.abs(rhs))
    if _maximum(np.abs(residual)) <= threshold:
        return point, None
    # A resolvably positive small eigenvalue must not be silently discarded.
    # A candidate ray is checked in the ORIGINAL Gram before any infeasibility
    # outcome. Otherwise rank trouble is a technical failure, not a fallback.
    ray = vectors[:, ~retained] @ (vectors[:, ~retained].T @ rhs)
    roundoff_bound = (policy.eigenvalue_roundoff_multiplier * np.finfo(np.float64).eps
                      * len(matrix) * max(1.0, float(values[-1])))
    if (values[0] > roundoff_bound and
            np.all(ray >= -policy.recession_residual_relative * _maximum(np.abs(ray)))):
        # A resolvably positive small eigenvalue does not constitute a null
        # recession ray. Near-rank-loss must fail technically, not impose an
        # undeclared norm cap by mislabeling its huge finite optimum infeasible.
        raise QPSolverFailure("QP_POSITIVE_SMALL_EIGENVALUE_UNRESOLVED", {
            "minimum_eigenvalue": float(values[0]), "rank_cutoff": cutoff,
            "eigenvalue_roundoff_bound": float(roundoff_bound)})
    return point, ray


def _working_solve(matrix, beta, keys, warm, policy):
    """Finite nonnegative convex-QP active set, handling singular faces.

    The passive-face minimizer is followed until a multiplier hits zero.
    On a singular face, follow its null descent to a boundary or produce a
    numerical Farkas ray. This also keeps redundant/negative-offset rows.
    """
    n = len(beta)
    alpha = np.asarray(warm, dtype=np.float64).copy()
    passive = set(np.flatnonzero(alpha > 0).tolist())
    if not passive:
        passive = set(np.flatnonzero(beta > 0).tolist())
    trace = []
    for iteration in range(policy.max_pivots):
        if passive:
            p = sorted(passive, key=lambda i: keys[i])
            sub = matrix[np.ix_(p, p)]
            point, ray = _stationary_or_ray(sub, beta[p], policy)
            if not np.isfinite(point).all():
                raise QPSolverFailure("QP_FACE_SOLUTION_NONFINITE", {"trace": trace})
            if ray is not None:
                magnitude = _maximum(np.abs(ray))
                if not magnitude or not np.isfinite(ray).all():
                    raise QPSolverFailure("QP_SINGULAR_FACE_UNRESOLVED", {"trace": trace})
                ray /= magnitude
                negative = np.flatnonzero(ray < -policy.recession_residual_relative)
                if not negative.size:
                    witness = np.zeros(n)
                    witness[p] = np.maximum(ray, 0)
                    witness /= np.sum(witness)
                    residual = matrix @ witness
                    gain = float(beta @ witness)
                    ray_norm_sq = float(witness @ residual)
                    tolerance = policy.recession_residual_relative * max(1.0, _maximum(np.abs(matrix)))
                    evidence = {"witness_normalized_rows": witness.tolist(),
                                "gram_witness_inf": _maximum(np.abs(residual)),
                                "witness_norm_squared": ray_norm_sq, "b_dot_witness": gain,
                                "certificate_tolerance": tolerance, "trace": trace,
                                "scope": "numerically certified local first-order infeasibility only"}
                    if (_maximum(np.abs(residual)) <= tolerance and abs(ray_norm_sq) <= tolerance
                            and gain > policy.feasibility_absolute):
                        raise QPInfeasible("QP_INFEASIBLE", evidence)
                    raise QPSolverFailure("QP_RECESSION_UNRESOLVED", evidence)
                step = min(float(alpha[p[j]] / -ray[j]) for j in negative)
                alpha[p] += step * ray
                blockers = [p[j] for j in negative
                            if alpha[p[j]] <= policy.feasibility_absolute * 0.01]
                for j in blockers:
                    alpha[j] = 0.0
                    passive.remove(j)
                if not blockers:
                    raise QPSolverFailure("QP_NULL_STEP_NO_BLOCKER", {"trace": trace})
                trace.append({"iteration": iteration, "action": "NULL_FACE_DROP", "rows": blockers})
                continue
            negative = np.flatnonzero(point < 0)
            if negative.size:
                direction = point - alpha[p]
                step = min(float(alpha[p[j]] / -direction[j]) for j in negative)
                alpha[p] += min(1.0, step) * direction
                blockers = [p[j] for j in negative
                            if alpha[p[j]] <= policy.feasibility_absolute * 0.01]
                for j in blockers:
                    alpha[j] = 0.0
                    passive.remove(j)
                if not blockers:
                    raise QPSolverFailure("QP_STEP_NO_BLOCKER", {"trace": trace})
                trace.append({"iteration": iteration, "action": "FACE_DROP", "rows": blockers})
                continue
            alpha[:] = 0.0
            alpha[p] = point
        response = matrix @ alpha
        violation = beta - response
        limits = _row_limits(beta, response, policy)
        missing = [j for j in range(n) if j not in passive and violation[j] > limits[j]]
        if not missing:
            if np.any(alpha < 0) or not np.isfinite(alpha).all():
                raise QPSolverFailure("QP_DUAL_NONFINITE_OR_NEGATIVE", {"trace": trace})
            return alpha, {"iterations": iteration + 1, "active_set_trace": trace}
        row = min(missing, key=lambda i: (-float(violation[i]), keys[i]))
        passive.add(row)
        trace.append({"iteration": iteration, "action": "FACE_ADD", "row": row})
    raise QPSolverFailure("QP_ITERATION_LIMIT", {"iterations": policy.max_pivots, "trace": trace})


def solve(gram, b, ids: Sequence, order: str = "gss", *,
          policy: QPPolicy = DEFAULT_QP_POLICY) -> dict:
    """Return a JSON-safe certified solution or raise a typed QPError.

    ``alpha`` is in the ORIGINAL input row units, so callers reconstruct with
    their unnormalized h. ``gss`` and ``most_violation`` use growing blocks;
    ``full`` solves the same complete exposed bank directly. Every mode scans
    every input inequality, including initially safe and exactly zero rows.
    """
    started = time.perf_counter()
    if order not in ("gss", "most_violation", "full"):
        raise QPInputError("QP_ORDER")
    if policy.max_pivots < 1 or policy.block_size < 1 or policy.max_rows < 1:
        raise QPInputError("QP_POLICY")
    (raw_g, rhs, keys, norms, scales, matrix, normalized_g, beta,
     zero, spectrum) = _prepare(gram, b, ids, policy)
    lam = np.zeros(len(rhs), dtype=np.float64)
    eligible = np.flatnonzero(~zero).tolist()
    history = []
    working: list[int] = []
    if np.any(beta > 0):
        if order == "full":
            working = sorted(eligible, key=lambda i: keys[i])
        else:
            working = _block(eligible, [], beta, matrix, keys, order, policy.block_size)
    added = list(working)
    while working:
        indices = np.asarray(working, dtype=int)
        try:
            values, internal = _working_solve(matrix[np.ix_(indices, indices)], beta[indices],
                                              [keys[i] for i in working], lam[indices], policy)
        except QPError as exc:
            exc.receipt.update({"order": order, "working_rows": working,
                                "working_ids": [ids[i] for i in working],
                                "working_set_history": history, "policy": asdict(policy)})
            if isinstance(exc, QPInfeasible) and "witness_normalized_rows" in exc.receipt:
                local_witness = np.asarray(exc.receipt["witness_normalized_rows"])
                witness = np.zeros(len(rhs))
                witness[indices] = local_witness
                # A subset Farkas witness must be null against the ENTIRE Gram.
                residual = normalized_g @ witness
                if _maximum(np.abs(residual)) > policy.recession_residual_relative:
                    raise QPSolverFailure("QP_FULL_RECESSION_UNRESOLVED", exc.receipt) from exc
                exc.receipt["witness_original_rows"] = (witness / scales).tolist()
                exc.receipt["full_gram_witness_inf"] = _maximum(np.abs(residual))
            raise
        lam[:] = 0.0
        lam[indices] = values
        alpha = lam / scales
        certificate = _certificate(raw_g, rhs, normalized_g, beta, scales, alpha, policy)
        working_certificate = _certificate(
            raw_g[np.ix_(indices, indices)], rhs[indices],
            normalized_g[np.ix_(indices, indices)], beta[indices], scales[indices],
            alpha[indices], policy)
        if not working_certificate["KKT_pass"]:
            raise QPSolverFailure("QP_WORKING_KKT_UNRESOLVED", {
                "working_rows": working, "working_ids": [ids[i] for i in working],
                "diagnostics": working_certificate, "full_diagnostics": certificate,
                "working_set_history": history, "policy": asdict(policy)})
        response = normalized_g @ lam
        violation = beta - response
        limits = _row_limits(beta, response, policy)
        present = set(working)
        missing = [i for i in eligible if i not in present and violation[i] > limits[i]]
        history.append({"round": len(history) + 1, "added_rows": list(added),
                        "added_ids": [ids[i] for i in added], "working_size": len(working),
                        "all_rows_checked": int(len(rhs)),
                        "violated_missing_rows": len(missing),
                        "normalized_max_violation": _maximum(violation),
                        "working_KKT_pass": True,
                        "primal_objective": certificate["primal_objective"], **internal})
        if not missing:
            break
        added = _block(missing, working, violation, matrix, keys, order, policy.block_size)
        working.extend(added)
    alpha = lam / scales
    certificate = _certificate(raw_g, rhs, normalized_g, beta, scales, alpha, policy)
    result = {"status": "CERTIFIED", "alpha": alpha.tolist(), "order": order,
              "method": "FP64_NONNEGATIVE_DUAL_ACTIVE_SET_NO_RIDGE",
              "policy": asdict(policy), "row_ids": list(ids), "row_norms": norms.tolist(),
              "normalized_b": beta.tolist(), "working_rows": working,
              "active_rows": np.flatnonzero(alpha > 0).tolist(),
              "working_set_history": history, "diagnostics": certificate,
              "spectrum": spectrum, "elapsed_seconds": time.perf_counter() - started,
              "reconstruction": "+sum(alpha_i * original_h_i)",
              "model_validation": "NOT_RUN_CPU_LOCAL_QP_ONLY"}
    if not certificate["KKT_pass"]:
        raise QPSolverFailure("QP_KKT_UNRESOLVED", result)
    return result


def audit_same_problem(gram, b, ids: Sequence, *,
                       policy: QPPolicy = DEFAULT_QP_POLICY) -> dict:
    """Replay all orderings; compare directions/objectives, not nonunique duals."""
    results = {order: solve(gram, b, ids, order, policy=policy)
               for order in ("full", "gss", "most_violation")}
    matrix = np.asarray(gram, dtype=np.float64)
    full = results["full"]
    objective = full["diagnostics"]["primal_objective"]
    comparisons = {}
    passed = True
    for order in ("gss", "most_violation"):
        difference = np.asarray(results[order]["alpha"]) - np.asarray(full["alpha"])
        direction_squared = float(difference @ matrix @ difference)
        objective_difference = abs(results[order]["diagnostics"]["primal_objective"] - objective)
        tolerance = policy.objective_relative * max(1.0, abs(objective))
        # A near-zero signed quadratic can be negative from cancellation; keep
        # its actual value and require its absolute magnitude to pass too.
        equal = abs(direction_squared) <= 4 * tolerance and objective_difference <= tolerance
        comparisons[order] = {"direction_difference_squared": direction_squared,
                              "objective_difference": objective_difference,
                              "objective_tolerance": tolerance, "pass": bool(equal)}
        passed = passed and equal
    return {"status": "PASS_CPU_SAME_PROBLEM_ONLY" if passed else "FAIL_CPU_SAME_PROBLEM",
            "pass": bool(passed), "comparisons": comparisons, "results": results,
            "alpha_equality_required": False, "model_validation": "NOT_RUN"}
