"""L4 고정 preservation repair의 작은 CPU 수학 검산.

Python 표준 라이브러리만 사용한다. 아래 active-set 열거기는 최대 2차원
양의 정부호 toy QP 전용이며 production solver, 실제 모델 검증 또는 성능
실험이 아니다. 원 모델·GPU·실험 산출물을 읽거나 변경하지 않는다.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path


TOL = 1e-9


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def objective(hessian, gradient, coefficients):
    return dot(gradient, coefficients) + 0.5 * dot(
        coefficients, matvec(hessian, coefficients)
    )


def solve_dense(matrix, rhs):
    """작은 정방 선형계의 pivot 소거. 특이 active set은 None을 반환한다."""
    n = len(rhs)
    assert len(matrix) == n and all(len(row) == n for row in matrix)
    augmented = [list(map(float, row)) + [float(value)]
                 for row, value in zip(matrix, rhs, strict=True)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            scale = augmented[row][column]
            augmented[row] = [left - scale * right
                              for left, right in zip(augmented[row], augmented[column], strict=True)]
    return [row[-1] for row in augmented]


def kkt_residuals(hessian, gradient, constraints, limits, coefficients, multipliers):
    slacks = [dot(row, coefficients) - limit
              for row, limit in zip(constraints, limits, strict=True)]
    stationarity = [value + gradient[j] + math.fsum(
        multipliers[i] * constraints[i][j] for i in range(len(constraints)))
        for j, value in enumerate(matvec(hessian, coefficients))]
    return {
        "primal_max_positive_violation": max([0.0] + slacks),
        "dual_max_negative_violation": max([0.0] + [-v for v in multipliers]),
        "stationarity_infinity_norm": max(map(abs, stationarity), default=0.0),
        "complementarity_infinity_norm": max(
            (abs(mu * slack) for mu, slack in zip(multipliers, slacks, strict=True)),
            default=0.0,
        ),
    }


def toy_qp(hessian, gradient, constraints, limits):
    """min g'c + c'Hc/2, A c <= limits. 1~2차원 SPD toy만 허용한다."""
    n = len(gradient)
    assert n in (1, 2)
    assert len(hessian) == n and all(len(row) == n for row in hessian)
    assert len(constraints) == len(limits) and all(len(row) == n for row in constraints)
    assert hessian[0][0] > 0.0
    if n == 2:
        assert hessian[0][1] == hessian[1][0]
        assert hessian[0][0] * hessian[1][1] - hessian[0][1] ** 2 > 0.0
    candidates = []
    for size in range(min(n, len(constraints)) + 1):
        for active in itertools.combinations(range(len(constraints)), size):
            system = [list(hessian[j]) + [constraints[i][j] for i in active]
                      for j in range(n)]
            system += [list(constraints[i]) + [0.0] * size for i in active]
            solution = solve_dense(system, [-v for v in gradient] + [limits[i] for i in active])
            if solution is None:
                continue
            coefficients = solution[:n]
            multipliers = [0.0] * len(constraints)
            for i, value in zip(active, solution[n:], strict=True):
                multipliers[i] = value
            residuals = kkt_residuals(
                hessian, gradient, constraints, limits, coefficients, multipliers
            )
            if max(residuals.values()) > TOL:
                continue
            candidates.append({
                "coefficients": coefficients,
                "objective_change": objective(hessian, gradient, coefficients),
                "active_constraint_indices": list(active),
                "multipliers": multipliers,
                "KKT": residuals,
            })
    assert candidates, "TOY_QP_NO_CERTIFIED_CANDIDATE"
    return min(candidates, key=lambda row: (
        row["objective_change"], dot(row["coefficients"], row["coefficients"]),
        row["active_constraint_indices"],
    ))


def box(radius=1.0):
    return [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], [radius] * 4


def close(actual, expected, tolerance=TOL):
    assert abs(actual - expected) <= tolerance, (actual, expected, tolerance)


def softmax(logits):
    maximum = max(logits)
    weights = [math.exp(v - maximum) for v in logits]
    total = math.fsum(weights)
    return [v / total for v in weights]


def kl(teacher, logits):
    observed = softmax(logits)
    return math.fsum(p * (math.log(p) - math.log(q))
                     for p, q in zip(teacher, observed, strict=True))


def run_checks():
    checks = []
    hessian = [[1.0, 0.0], [0.0, 1.0]]
    constraints, limits = box()
    constraints += [[0.0, 1.0], [0.0, -1.0]]
    limits += [0.0, 0.0]  # toy edit response c2를 정확히 고정한다.

    zero = [0.0, 0.0]
    assert all(dot(row, zero) <= limit for row, limit in zip(constraints, limits, strict=True))
    checks.append({"name": "zero_is_feasible", "status": "PASS",
                   "maximum_guard_violation": max(dot(row, zero) - limit
                       for row, limit in zip(constraints, limits, strict=True))})

    repair = toy_qp(hessian, [-2.0, 0.0], constraints, limits)
    close(repair["coefficients"][0], 1.0)
    close(repair["coefficients"][1], 0.0)
    close(repair["objective_change"], -1.5)
    checks.append({"name": "known_feasible_preservation_repair", "status": "PASS",
                   "base_objective_before": 2.0, "base_objective_after": 0.5,
                   "edit_response_change": 0.0, "solution": repair})

    unconstrained = toy_qp(hessian, [-2.0, 0.0], [], [])
    assert unconstrained["objective_change"] <= repair["objective_change"]
    close(unconstrained["objective_change"], -2.0)
    checks.append({"name": "unconstrained_objective_is_lower_bound", "status": "PASS",
                   "unconstrained": unconstrained["objective_change"],
                   "constrained": repair["objective_change"]})

    conflict_rows, conflict_limits = box()
    conflict_rows.append([1.0, 0.0]); conflict_limits.append(0.0)
    conflict = toy_qp(hessian, [-1.0, 0.0], conflict_rows, conflict_limits)
    close(dot(conflict["coefficients"], conflict["coefficients"]), 0.0)
    checks.append({"name": "edit_conflict_selects_zero", "status": "PASS", "solution": conflict,
                   "interpretation": "이 toy 방향 공간에서 edit guard와 Base descent가 충돌한다."})

    nonlinear_rows, nonlinear_limits = box()
    nonlinear_rows.append([0.0, 1.0]); nonlinear_limits.append(0.0)
    linear = toy_qp(hessian, [-1.0, 0.0], nonlinear_rows, nonlinear_limits)
    c1, c2 = linear["coefficients"]
    first_order_guard = c2
    finite_guard = c2 + c1 * c1  # f(0)=0, grad f(0)=(0,1).
    base_improves = linear["objective_change"] < 0.0
    accepted = base_improves and finite_guard <= TOL
    assert first_order_guard <= TOL and finite_guard > TOL and not accepted
    checks.append({"name": "finite_curvature_invalidates_linear_guard", "status": "PASS",
                   "first_order_guard_change": first_order_guard,
                   "finite_guard_change": finite_guard, "base_improves": base_improves,
                   "actual_screen_accepts": accepted, "fallback_coefficients": zero,
                   "solution": linear})

    w4, w8, x, delta8 = 2.0, 3.0, 1.0, -0.5
    output_before = w8 * (w4 * x)
    output_after = (w8 + delta8) * (w4 * x)
    assert output_before != output_after and w4 == 2.0
    checks.append({"name": "fixed_L4_does_not_fix_final_output", "status": "PASS",
                   "W4_before_and_after": w4, "output_before": output_before,
                   "output_after": output_after})

    pn = [0.4, 0.6]
    z = [math.log(p) for p in pn]
    step = 1e-4
    hessian_estimates = []
    gradient_estimates = []
    for teacher in ([0.8, 0.2], [0.3, 0.7]):
        plus, minus = [z[0] + step, z[1]], [z[0] - step, z[1]]
        first = (kl(teacher, plus) - kl(teacher, minus)) / (2.0 * step)
        second = (kl(teacher, plus) - 2.0 * kl(teacher, z) + kl(teacher, minus)) / step**2
        close(first, pn[0] - teacher[0], 1e-7)
        close(second, pn[0] * (1.0 - pn[0]), 1e-6)
        gradient_estimates.append(first); hessian_estimates.append(second)
    checks.append({"name": "KL_teacher_and_GN_metric_are_distinct", "status": "PASS",
                   "p_at_WN": pn, "teacher_dependent_logit_gradients": gradient_estimates,
                   "teacher_independent_logit_H00": hessian_estimates,
                   "analytic_Fisher_H00": 0.24,
                   "scope": "logit Hessian=F(pN); weight GN는 logit 비선형성의 추가 Hessian항을 생략한다."})

    first_at_reference = (kl(pn, [z[0] + step, z[1]])
                          - kl(pn, [z[0] - step, z[1]])) / (2.0 * step)
    nonzero_step_kl = kl(pn, [z[0] + 0.5, z[1]])
    close(first_at_reference, 0.0, 1e-7)
    assert nonzero_step_kl > 0.0
    checks.append({"name": "KL_guard_zero_first_gradient_is_insufficient", "status": "PASS",
                   "reference_KL_gradient_finite_difference": first_at_reference,
                   "finite_nonzero_step_KL": nonzero_step_kl})
    return checks


def main():
    checks = run_checks()
    source = Path(__file__).resolve()
    result = {
        "schema": "l4-preserving-repair-math-checks/v1",
        "status": "PASS_STDLIB_CPU_TOY_ONLY",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_calls": 0,
        "GPU_calls": 0,
        "production_solver": False,
        "QP_scope": "최대2차원 SPD, finite linear inequality active-set enumeration",
        "KKT_absolute_tolerance": TOL,
        "checks_passed": len(checks),
        "checks": checks,
        "limits": [
            "실제 Llama, native writer, JVP 또는 full-matrix direction을 검증하지 않았다.",
            "저차원 toy의 성공/실패는 모델 성능·RS/PS 유지·intrinsic capacity의 증거가 아니다.",
            "실제 nonlinear endpoint 검증을 linear feasibility 또는 KKT residual로 대체할 수 없다.",
            "Production의 scaling, numerical damping, trust radius, tolerance는 별도 계약이 필요하다.",
        ],
    }
    output = source.with_name("math-checks.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks_passed": len(checks),
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
