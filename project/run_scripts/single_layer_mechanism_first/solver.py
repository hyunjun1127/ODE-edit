"""FP64 two-phase convex coefficient QCQP, dimension at most five.

The reference slack variables are eliminated exactly: xi=max(0,-mu-Ja).
This gives a continuously differentiable convex squared-hinge objective.
SLSQP supplies a candidate only. Independent primal and convex dual-bound
audits determine whether it is a usable two-phase solution. Infeasibility is
reported only with an explicitly checked ball/linear-row dual certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize, nnls, linprog, root
from threadpoolctl import threadpool_limits

ROW_RELATIVE_TOLERANCE = 1e-8
PHASE2_RISK_ABSOLUTE_TOLERANCE = 1e-12
PHASE2_RISK_RELATIVE_TOLERANCE = 1e-8
SOLVER_FTOL = 1e-13
SOLVER_MAXITER = 2000
KKT_ACTIVE_TOLERANCE = 2e-7
PHASE2_NORM_DUAL_GAP_TOLERANCE = 2e-7
BACKTRACKING_SCALES = (1.0, 0.5, 0.25, 0.125)


class SolverTechnicalError(ValueError):
    """Invalid state/nonfinite input, not finite scientific fallback."""


@dataclass
class CoefficientResult:
    coefficients: np.ndarray | None
    status: str
    receipt: dict


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    if hasattr(value, "detach"):
        if value.device.type != "cpu":
            raise SolverTechnicalError(f"{name}: explicit CPU ownership required")
        value = value.detach().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise SolverTechnicalError(f"{name}: wrong dimension or nonfinite")
    return array


def risk(mu: np.ndarray, jacobian: np.ndarray, a: np.ndarray) -> tuple[float, np.ndarray]:
    deficits = np.minimum(mu + jacobian @ a, 0.0)
    return float(np.mean(deficits * deficits)), 2.0 * jacobian.T @ deficits / len(mu)


def proposal_radius(psi: float, projected_gradient_norm: float, gradient_norm: float,
                    *, reference_risk: float | None = None,
                    tie_mismatches: int = 0, guards_valid: bool = True) -> dict:
    numbers = [psi, projected_gradient_norm, gradient_norm]
    if reference_risk is not None:
        numbers.append(reference_risk)
    if any(not np.isfinite(x) or x < 0 for x in numbers) or tie_mismatches < 0:
        raise SolverTechnicalError("invalid risk/gradient norm/tie count")
    phi = psi if reference_risk is None else reference_risk
    tolerance = 1e-10 + 1e-6 * max(psi, phi)
    if psi == 0:
        status = "TIE_ONLY_NO_DIRECTION" if tie_mismatches else (
            "NO_OP_ZERO_RISK" if guards_valid else "ZERO_RISK_GUARD_UNRESOLVED")
    elif psi <= tolerance:
        status = "BELOW_RISK_RESOLUTION"
    elif gradient_norm == 0 or projected_gradient_norm <= 1e-12 * gradient_norm:
        status = "NO_PROJECTED_DIRECTION"
    else:
        radius = psi / projected_gradient_norm
        if not np.isfinite(radius):
            raise SolverTechnicalError("nonfinite proposal radius")
        return {"status": "RADIUS_READY", "radius": radius, "psi": psi,
                "tau_risk": tolerance, "safety_certificate": False}
    return {"status": status, "radius": None, "psi": psi,
            "tau_risk": tolerance, "safety_certificate": False}


def _nnls(matrix: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
    if not matrix.shape[1]:
        return np.empty(0), True
    try:
        values, _ = nnls(matrix, target, maxiter=max(300, 10 * matrix.shape[1]))
        return values, True
    except (RuntimeError, ValueError):
        return np.zeros(matrix.shape[1]), False


def _primal(x: np.ndarray, matrix: np.ndarray, lower: np.ndarray) -> dict:
    slack = matrix @ x - lower
    ball_slack = 1.0 - float(x @ x)
    return {"scaled_row_min_slack": float(np.min(slack, initial=np.inf)),
            "scaled_row_violation": max(0.0, -float(np.min(slack, initial=0.0))),
            "unit_ball_slack": ball_slack,
            "passed": bool(np.all(slack >= -ROW_RELATIVE_TOLERANCE)
                           and ball_slack >= -ROW_RELATIVE_TOLERANCE)}


def _polish_phase1(x: np.ndarray, mu: np.ndarray, jacobian: np.ndarray,
                    matrix: np.ndarray, lower: np.ndarray) -> tuple[np.ndarray, dict]:
    """Solve local active-set KKT equations without relaxing the problem.

SLSQP's objective stopping test alone can leave a 1e-7 stationarity residual
even on a smooth, five-dimensional ball problem. Newton polishing uses the
exact piecewise-quadratic Hessian, then independently rechecks every row.
It never discards rows from the final primal/dual audit.
"""
    value, gradient = risk(mu, jacobian, x)
    active = np.flatnonzero(matrix@x-lower <= KKT_ACTIVE_TOLERANCE)
    ball = 1-float(x@x) <= KKT_ACTIVE_TOLERANCE
    columns = [-matrix[index] for index in active]
    if ball:
        columns.append(2*x)
    design = np.stack(columns, axis=1) if columns else np.empty((len(x), 0))
    multipliers, solved = _nnls(design, -gradient)
    if not solved:
        return x, {"attempted": False, "reason": "MULTIPLIER_INITIALIZATION_UNRESOLVED"}
    # NNLS supplies sparse positive support even with hundreds of repeated rows.
    keep = np.flatnonzero(multipliers[:len(active)] > 0)
    selected = active[keep]
    dual0 = multipliers[keep]
    eta0 = multipliers[-1] if ball else 0.0
    a, b = matrix[selected], lower[selected]
    n, k = len(x), len(selected)
    y0 = np.r_[x, dual0, eta0] if ball else np.r_[x, dual0]
    def equations(y):
        point, dual = y[:n], y[n:n+k]
        eta = y[-1] if ball else 0.0
        _, g = risk(mu, jacobian, point)
        stationarity = g-a.T@dual+2*eta*point
        return np.r_[stationarity, a@point-b, float(point@point)-1] if ball else np.r_[stationarity, a@point-b]
    def derivative(y):
        point = y[:n]
        eta = y[-1] if ball else 0.0
        negative = mu+jacobian@point < 0
        hessian = 2*(jacobian[negative].T@jacobian[negative])/len(mu)+2*eta*np.eye(n)
        out = np.zeros((len(y),len(y)))
        out[:n,:n] = hessian
        out[:n,n:n+k] = -a.T
        out[n:n+k,:n] = a
        if ball:
            out[:n,-1] = 2*point
            out[-1,:n] = 2*point
        return out
    refined = root(equations, y0, jac=derivative, method="hybr",
                   options={"xtol": 1e-11, "maxfev": 500})
    receipt = {"attempted": True, "method": "EXACT_HINGE_HESSIAN_ACTIVE_KKT_ROOT",
               "success": bool(refined.success), "message": str(refined.message),
               "evaluations": int(refined.nfev), "selected_linear_rows": selected.tolist(),
               "ball_active": ball}
    if not np.isfinite(refined.x).all():
        raise SolverTechnicalError("nonfinite active-set KKT solver proposal")
    candidate = refined.x[:n]
    receipt["kkt_equation_l2"] = float(np.linalg.norm(equations(refined.x)))
    candidate_value, candidate_gradient = risk(mu,jacobian,candidate)
    old_audit = _dual_audit(x,value,gradient,matrix,lower)
    new_audit = _dual_audit(candidate,candidate_value,candidate_gradient,matrix,lower)
    accepted = (_primal(candidate,matrix,lower)["passed"] and
                new_audit["dual_gap_upper_bound"] <= old_audit["dual_gap_upper_bound"] and
                candidate_value <= value+1e-12+1e-8*abs(value))
    return (candidate if accepted else x), {**receipt, "accepted": bool(accepted)}


def _dual_audit(x: np.ndarray, value: float, gradient: np.ndarray,
                matrix: np.ndarray, lower: np.ndarray,
                *, risk_value: float | None = None,
                risk_gradient: np.ndarray | None = None,
                risk_limit: float | None = None, risk_scale: float = 1.0) -> dict:
    slack = matrix @ x - lower
    active = np.flatnonzero(slack <= KKT_ACTIVE_TOLERANCE)
    columns = [-matrix[index] for index in active]
    labels = [f"linear:{index}" for index in active]
    constraint_values = [-slack[index] for index in active]
    ball_slack = 1.0 - float(x @ x)
    if ball_slack <= KKT_ACTIVE_TOLERANCE:
        columns.append(2.0 * x)
        labels.append("ball")
        constraint_values.append(-ball_slack)
    if risk_limit is not None and (risk_limit - risk_value) / risk_scale <= KKT_ACTIVE_TOLERANCE:
        columns.append(risk_gradient / risk_scale)
        labels.append("risk_bound")
        constraint_values.append((risk_value - risk_limit) / risk_scale)
    matrix_active = np.stack(columns, axis=1) if columns else np.empty((len(x), 0))
    multipliers, solved = _nnls(matrix_active, -gradient)
    residual = gradient + matrix_active @ multipliers
    # Convexity on ||z||<=1 gives L(z)>=L(x)-||grad L(x)||(1+||x||).
    # This bound is valid even if the chosen point is slightly infeasible.
    lagrangian = value + float(multipliers @ np.asarray(constraint_values))
    lower_bound = lagrangian - float(np.linalg.norm(residual)) * (1 + float(np.linalg.norm(x)))
    complementarity = multipliers * np.asarray(constraint_values)
    return {"nonnegative_multiplier_solve": solved, "labels": labels,
            "multipliers": multipliers.tolist(), "stationarity_vector": residual.tolist(),
            "stationarity_l2": float(np.linalg.norm(residual)),
            "complementarity_max_abs": float(np.max(np.abs(complementarity), initial=0.0)),
            "convex_lower_bound": lower_bound, "dual_gap_upper_bound": max(0.0, value - lower_bound),
            "value": value, "units": "objective_units; coefficients use a/radius"}


def _certificate(matrix: np.ndarray, lower: np.ndarray) -> dict | None:
    """Verify lambda*l > ||A.T lambda||, lambda>=0, on unit ball."""
    def verify(weights: np.ndarray, source: str) -> dict | None:
        if not np.isfinite(weights).all() or np.min(weights, initial=0.0) < 0:
            return None
        left = float(weights @ lower)
        right = float(np.linalg.norm(matrix.T @ weights))
        uncertainty = 128 * np.finfo(float).eps * max(1.0, abs(left), right,
                                                     float(np.abs(weights) @ np.abs(lower)))
        gap = left - right
        if gap > uncertainty:
            return {"type": "LINEAR_ROWS_VS_UNIT_BALL_DUAL", "source": source,
                    "weights": weights.tolist(), "weights_nonnegative": True,
                    "lower_dot_weights": left, "adjoint_norm": right,
                    "strict_contradiction_gap": gap, "roundoff_envelope": uncertainty,
                    "verified": True}
        return None
    for index in np.flatnonzero(lower > np.linalg.norm(matrix, axis=1)):
        weights = np.zeros(len(lower)); weights[index] = 1.0
        checked = verify(weights, "SINGLE_ROW")
        if checked:
            return checked
    if not len(lower):
        return None
    # A positive linear-only Farkas ray is also a valid ball certificate.
    equality = np.vstack((matrix.T, np.ones(len(lower))))
    rhs = np.r_[np.zeros(matrix.shape[1]), 1.0]
    lp = linprog(-lower, A_eq=equality, b_eq=rhs, bounds=(0, None), method="highs")
    if lp.success:
        checked = verify(np.asarray(lp.x), "LINEAR_FARKAS_LP_CHECKED_IN_ORIGINAL_FP64_ROWS")
        if checked:
            return checked
    # Closest point of the row polyhedron supplies candidate positive duals.
    phase0 = minimize(lambda x: float(x @ x), np.zeros(matrix.shape[1]),
                      jac=lambda x: 2*x, method="SLSQP",
                      constraints=[{"type": "ineq", "fun": lambda x: matrix @ x-lower,
                                    "jac": lambda x: matrix}],
                      options={"ftol": SOLVER_FTOL, "maxiter": SOLVER_MAXITER, "disp": False})
    if np.isfinite(phase0.x).all():
        active = np.flatnonzero(matrix @ phase0.x - lower <= KKT_ACTIVE_TOLERANCE)
        if len(active):
            dual, solved = _nnls(matrix[active].T, phase0.x)
            if solved:
                weights = np.zeros(len(lower)); weights[active] = dual
                checked = verify(weights, "MIN_NORM_POLYHEDRON_CANDIDATE_DUAL")
                if checked:
                    return checked
    return None


def solve_coefficients(mu_ref: Any, J_ref: Any, radius: float, *,
                       history_slack: Any | None = None, history_J: Any | None = None,
                       current_slack: Any | None = None, current_J: Any | None = None,
                       threads: int = 8) -> CoefficientResult:
    """Solve the exact declared local problem, not the nonlinear model.

History/current inputs encode h+J*a>=0. Current rows are normally absent
because exact Q_E makes their Jacobian zero; if supplied they are not altered.
No scientific row slack or ridge is introduced. Returning coefficients=None
means no certified two-phase solution was established.
"""
    mu, j = _array(mu_ref, "mu_ref", 1), _array(J_ref, "J_ref", 2)
    if len(mu) < 1 or j.shape[0] != len(mu) or j.shape[1] > 5:
        raise SolverTechnicalError("reference cardinality or q>5")
    if not np.isfinite(radius) or radius < 0:
        raise SolverTechnicalError("invalid radius")
    q = j.shape[1]
    rows = [j]
    lowers = [np.minimum(mu, 0.0) - mu]
    anchors = [mu]
    labels = [f"reference:{index}" for index in range(len(mu))]
    for name, slack, jac in (("history", history_slack, history_J),
                             ("current", current_slack, current_J)):
        if (slack is None) != (jac is None):
            raise SolverTechnicalError(f"{name}: slack/J must be paired")
        if slack is not None:
            h, hj = _array(slack, name, 1), _array(jac, name+"_J", 2)
            if hj.shape != (len(h), q):
                raise SolverTechnicalError(f"{name}: row shape mismatch")
            rows.append(hj); lowers.append(-h); anchors.append(h)
            labels.extend(f"{name}:{index}" for index in range(len(h)))
    raw_matrix, raw_lower = np.vstack(rows), np.concatenate(lowers)
    anchor = np.concatenate(anchors)
    row_scale = np.maximum.reduce((np.ones(len(anchor)), np.abs(anchor),
                                    radius * np.linalg.norm(raw_matrix, axis=1)))
    matrix, lower = radius * raw_matrix / row_scale[:, None], raw_lower / row_scale
    initial_risk, _ = risk(mu, j, np.zeros(q))
    evidence = {"schema_version": 1, "solver": "scipy.optimize.SLSQP_two_phase",
                "scipy_version": scipy.__version__, "dtype": "float64", "dimension": q,
                "radius": radius, "coefficient_parameterization": "a=radius*x",
                "objective": "mean_reference_squared_hinge_exact_slack_elimination",
                "reference_rows": len(mu), "row_labels": labels,
                "row_scales": row_scale.tolist(),
                "physical_row_tolerances": (ROW_RELATIVE_TOLERANCE*row_scale).tolist(),
                "row_relative_tolerance": ROW_RELATIVE_TOLERANCE,
                "phase2_risk_gap_absolute": PHASE2_RISK_ABSOLUTE_TOLERANCE,
                "phase2_risk_gap_relative": PHASE2_RISK_RELATIVE_TOLERANCE,
                "solver_ftol": SOLVER_FTOL, "solver_maxiter": SOLVER_MAXITER,
                "initial_reference_risk": initial_risk, "nonlinear_optimum_claim": False}
    if q == 0 or radius == 0:
        if np.all(raw_lower <= 0):
            return CoefficientResult(np.zeros(q), "ZERO_RADIUS_OR_BASIS",
                                     {**evidence, "reference_risk": initial_risk})
        return CoefficientResult(None, "LOCAL_HISTORY_CONSTRAINT_INFEASIBLE",
                                 {**evidence, "infeasibility_certificate": {
                                     "type": "ONLY_ZERO_COEFFICIENT_ALLOWED", "verified": True,
                                     "violated_rows": np.flatnonzero(raw_lower > 0).tolist()}})
    risk_scale = max(initial_risk, 1e-12)
    scaled_j = radius*j
    def objective(x):
        value, gradient = risk(mu, scaled_j, x)
        return value/risk_scale, gradient/risk_scale
    constraints = [{"type": "ineq", "fun": lambda x: matrix@x-lower,
                    "jac": lambda x: matrix},
                   {"type": "ineq", "fun": lambda x: 1-float(x@x),
                    "jac": lambda x: -2*x}]
    with threadpool_limits(limits=threads):
        phase1 = minimize(lambda x: objective(x)[0], np.zeros(q),
                          jac=lambda x: objective(x)[1], constraints=constraints,
                          method="SLSQP", options={"ftol": SOLVER_FTOL,
                                                   "maxiter": SOLVER_MAXITER, "disp": False})
        if not np.isfinite(phase1.x).all():
            raise SolverTechnicalError("nonfinite phase1 solver iterate")
        polished, polish_receipt = _polish_phase1(phase1.x, mu, scaled_j, matrix, lower)
        phase1.x = polished
        value1, grad1 = risk(mu, scaled_j, phase1.x)
        audit1 = _dual_audit(phase1.x, value1, grad1, matrix, lower)
        primal1 = _primal(phase1.x, matrix, lower)
        # Squared-hinge nonnegativity is an additional exact dual lower bound.
        audit1["convex_lower_bound"] = max(0.0, audit1["convex_lower_bound"])
        audit1["dual_gap_upper_bound"] = max(0.0, value1-audit1["convex_lower_bound"])
        gap_tolerance = PHASE2_RISK_ABSOLUTE_TOLERANCE + PHASE2_RISK_RELATIVE_TOLERANCE*abs(value1)
        evidence["phase1"] = {"success": bool(phase1.success), "message": str(phase1.message),
                              "iterations": int(phase1.nit), "risk": value1,
                              "active_kkt_polish": polish_receipt,
                              "primal": primal1, "dual_audit": audit1,
                              "certification_gap_tolerance": gap_tolerance,
                              "coefficients": (radius*phase1.x).tolist()}
        if not primal1["passed"] or audit1["dual_gap_upper_bound"] > gap_tolerance:
            certificate = _certificate(matrix, lower) if not primal1["passed"] else None
            if certificate:
                return CoefficientResult(None, "LOCAL_HISTORY_CONSTRAINT_INFEASIBLE",
                                         {**evidence, "infeasibility_certificate": certificate})
            return CoefficientResult(None, "FINITE_SOLVER_UNRESOLVED", evidence)
        limit = value1 + gap_tolerance
        risk_constraint_scale = max(risk_scale, gap_tolerance)
        phase2constraints = constraints + [{"type": "ineq",
                    "fun": lambda x: (limit-risk(mu, scaled_j, x)[0])/risk_constraint_scale,
                    "jac": lambda x: -risk(mu, scaled_j, x)[1]/risk_constraint_scale}]
        phase2 = minimize(lambda x: float(x@x), phase1.x,
                          jac=lambda x: 2*x, constraints=phase2constraints,
                          method="SLSQP", options={"ftol": SOLVER_FTOL,
                                                   "maxiter": SOLVER_MAXITER, "disp": False})
        if not np.isfinite(phase2.x).all():
            raise SolverTechnicalError("nonfinite phase2 solver iterate")
        value2, grad2 = risk(mu, scaled_j, phase2.x)
        primal2 = _primal(phase2.x, matrix, lower)
        audit2 = _dual_audit(phase2.x, float(phase2.x@phase2.x), 2*phase2.x,
                             matrix, lower, risk_value=value2, risk_gradient=grad2,
                             risk_limit=limit, risk_scale=risk_constraint_scale)
        # This tolerance checks numeric constraint solution, not additional risk slack.
        risk_numeric_error = 64*np.finfo(float).eps*max(1.0, initial_risk, abs(limit))
        risk_pass = value2 <= limit + risk_numeric_error
        phase2_pass = (primal2["passed"] and risk_pass and
                       audit2["dual_gap_upper_bound"] <= PHASE2_NORM_DUAL_GAP_TOLERANCE)
        evidence["phase2"] = {"success": bool(phase2.success), "message": str(phase2.message),
                              "iterations": int(phase2.nit), "risk": value2,
                              "risk_limit": limit, "risk_numeric_error_bound": risk_numeric_error,
                              "risk_constraint_pass": bool(risk_pass), "primal": primal2,
                              "dual_audit": audit2,
                              "norm_dual_gap_tolerance": PHASE2_NORM_DUAL_GAP_TOLERANCE,
                              "physical_squared_norm": float((radius*phase2.x)@(radius*phase2.x)),
                              "coefficients": (radius*phase2.x).tolist()}
        if not phase2_pass:
            return CoefficientResult(None, "FINITE_SOLVER_UNRESOLVED", evidence)
        coefficients = radius*phase2.x
        evidence["predicted_reference_margins"] = (mu+j@coefficients).tolist()
        evidence["reference_squared_slacks"] = np.square(np.minimum(mu+j@coefficients, 0.0)).tolist()
        evidence["raw_row_slacks"] = (raw_matrix@coefficients-raw_lower).tolist()
        return CoefficientResult(coefficients, "LOCAL_TWO_PHASE_SOLVED", evidence)


def history_backtracking_plan(coefficients: Any, history_slack: Any,
                               history_J: Any) -> list[dict]:
    a, h, j = _array(coefficients, "coefficients", 1), _array(history_slack, "history_slack", 1), _array(history_J, "history_J", 2)
    if j.shape != (len(h), len(a)):
        raise SolverTechnicalError("history backtracking dimensions")
    response = j@a
    damaged = h < 0
    impossible = damaged & (response <= 0)
    lower = float(np.max(-h[damaged & (response > 0)]/response[damaged & (response > 0)], initial=0.0))
    return [{"scale": scale, "linear_history_alpha_lower_bound": lower,
             "skip_before_model": bool(np.any(impossible) or scale < lower),
             "reason": "NONPOSITIVE_REPAIR_ON_DAMAGED_ROW" if np.any(impossible) else
                       ("BELOW_LINEAR_HISTORY_ALPHA_LOWER_BOUND" if scale < lower else "ELIGIBLE_ACTUAL_CHECK"),
             "nonlinear_feasibility_certificate": False} for scale in BACKTRACKING_SCALES]
