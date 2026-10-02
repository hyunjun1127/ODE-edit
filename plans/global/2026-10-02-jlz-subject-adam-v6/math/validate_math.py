#!/usr/bin/env python3
"""Small CPU-only algebra checks; standard library, no model, no autograd.

This validates a proposed method's identities and bookkeeping. It does not
establish an experimental speedup, convergence, or preservation guarantee.
"""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import sys


def zeros(n, m):
    return [[0.0 for _ in range(m)] for _ in range(n)]


def eye(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def transpose(x):
    return [list(row) for row in zip(*x)]


def matmul(a, b):
    bt = transpose(b)
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def add(a, b, multiplier=1.0):
    return [[x + multiplier * y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(a, value):
    return [[value * x for x in row] for row in a]


def diagonal(values):
    return [[value if i == j else 0.0 for j in range(len(values))]
            for i, value in enumerate(values)]


def trace(a):
    return sum(a[i][i] for i in range(len(a)))


def max_abs(a):
    return max(abs(x) for row in a for x in row)


def quadratic(d, metric):
    return trace(matmul(matmul(d, metric), transpose(d)))


def solve_spd(a, b):
    """Cholesky solve AX=B without materializing an inverse."""
    n = len(a)
    columns = len(b[0])
    lower = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            v = a[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                assert v > 0.0
                lower[i][j] = math.sqrt(v)
            else:
                lower[i][j] = v / lower[j][j]
    y = zeros(n, columns)
    x = zeros(n, columns)
    for col in range(columns):
        for i in range(n):
            y[i][col] = (b[i][col] - sum(lower[i][j] * y[j][col]
                                        for j in range(i))) / lower[i][i]
        for i in reversed(range(n)):
            x[i][col] = (y[i][col] - sum(lower[j][i] * x[j][col]
                                        for j in range(i + 1, n))) / lower[i][i]
    return x


def close(name, a, b, checks, atol=2e-10, rtol=2e-10):
    error = abs(a - b)
    limit = atol + rtol * max(abs(a), abs(b))
    assert error <= limit, (name, a, b, error, limit)
    checks.append({"name": name, "absolute_error": error, "limit": limit})


def random_matrix(rng, n, m):
    return [[rng.uniform(-1.0, 1.0) for _ in range(m)] for _ in range(n)]


def geometry(rng, batch, din, dout, checks):
    # The first context is canonical. Its mass is 1/2; the rest share 1/2.
    counts = [2 + r % 3 for r in range(batch)]
    contexts = sum(counts)
    z = zeros(batch, contexts)
    alpha = []
    column = 0
    for r, count in enumerate(counts):
        for c in range(count):
            z[r][column] = 1.0
            alpha.append(0.5 if c == 0 else 0.5 / (count - 1))
            column += 1
    weights = diagonal(alpha)
    weighted_z = matmul(matmul(z, weights), transpose(z))
    close("per_request_context_mass_identity", max_abs(add(weighted_z, eye(batch), -1)),
          0.0, checks)
    # Shared key component makes cross-request terms observable.
    k = random_matrix(rng, din, contexts)
    common = [rng.uniform(0.4, 1.2) for _ in range(din)]
    k = [[value + common[i] for value in row] for i, row in enumerate(k)]
    a_factor = random_matrix(rng, din, din)
    a = add(matmul(a_factor, transpose(a_factor)), scale(eye(din), 0.7))
    kbar = matmul(matmul(k, weights), transpose(z))
    m = add(a, matmul(matmul(k, weights), transpose(k)))
    p = solve_spd(m, kbar)
    g = matmul(matmul(transpose(p), a), p)
    error_map = add(matmul(transpose(p), k), z, -1)
    e = matmul(matmul(error_map, weights), transpose(error_map))
    compact = add(eye(batch), matmul(transpose(kbar), p), -1)
    close("E_plus_G_equals_I_minus_KbarT_P", max_abs(add(add(e, g), compact, -1)),
          0.0, checks)
    close("normal_equation", max_abs(add(matmul(m, p), kbar, -1)), 0.0, checks)
    close("G_symmetric", max_abs(add(g, transpose(g), -1)), 0.0, checks)
    close("E_symmetric", max_abs(add(e, transpose(e), -1)), 0.0, checks)
    d = random_matrix(rng, dout, batch)
    u = matmul(d, transpose(p))
    physical_q = quadratic(u, a)
    compact_g_q = quadratic(d, g)
    residual = add(matmul(u, k), matmul(d, z), -1)
    explicit_e_q = quadratic(residual, weights)
    compact_e_q = quadratic(d, e)
    close("physical_write_cost_explicit_equals_compact", physical_q, compact_g_q, checks)
    close("realization_cost_explicit_equals_compact", explicit_e_q, compact_e_q, checks)
    assert physical_q >= 0 and explicit_e_q >= 0
    coupling = {}
    for name, metric in [("G", g), ("E", e)]:
        full = quadratic(d, metric)
        diagonal_only = quadratic(d, diagonal([metric[i][i] for i in range(batch)]))
        off_diagonal = full - diagonal_only
        coupling[name] = {"full_quadratic": full,
                          "diagonal_only_quadratic": diagonal_only,
                          "off_diagonal_contribution": off_diagonal}
        if batch > 1:
            assert abs(off_diagonal) > 1e-6
    return {"D": d, "G": g, "E": e, "sigma": 0.8 + din / 10.0,
            "batch": batch, "din": din, "dout": dout, "contexts": contexts,
            "coupling": coupling}


def cost(layers, metric_name, arm):
    batch = layers[0]["batch"]
    terms = [max(0.0, quadratic(layer["D"], layer[metric_name])) /
             (batch * layer["sigma"] ** 2) for layer in layers]
    return sum(math.sqrt(q) for q in terms) if arm == "A" else math.sqrt(sum(terms))


def analytic_gradients(layers, metric_name, arm):
    batch = layers[0]["batch"]
    total = cost(layers, metric_name, arm)
    result = []
    for layer in layers:
        q = max(0.0, quadratic(layer["D"], layer[metric_name]))
        root = math.sqrt(q)
        if arm == "A":
            denominator = math.sqrt(batch) * layer["sigma"] * root
        else:
            denominator = batch * layer["sigma"] ** 2 * total
        # At a root-norm origin this is a chosen zero subgradient, not a
        # claim that the norm is differentiable there.
        result.append(scale(matmul(layer["D"], layer[metric_name]),
                            1.0 / denominator if denominator else 0.0))
    return result


def gradient_check(layers, metric_name, arm):
    analytic = analytic_gradients(layers, metric_name, arm)
    errors = []
    h = 1e-6
    for layer_idx, layer in enumerate(layers):
        for i, row in enumerate(layer["D"]):
            for j, original in enumerate(row):
                row[j] = original + h
                plus = cost(layers, metric_name, arm)
                row[j] = original - h
                minus = cost(layers, metric_name, arm)
                row[j] = original
                observed = (plus - minus) / (2 * h)
                expected = analytic[layer_idx][i][j]
                errors.append(abs(observed - expected))
                assert abs(observed - expected) < 5e-8
    return {"arm": arm, "metric": metric_name, "coordinate_count": len(errors),
            "finite_difference_h": h, "max_absolute_error": max(errors),
            "absolute_tolerance": 5e-8}


def clamp_columns(d, anchors, factor=0.75):
    norms = [math.sqrt(sum(row[j] ** 2 for row in d)) for j in range(len(anchors))]
    factors = [min(1.0, factor * anchor / norm) if norm else 1.0
               for anchor, norm in zip(anchors, norms)]
    return [[value * factors[j] for j, value in enumerate(row)] for row in d]


def projection_check(batch, checks):
    # Alternate an interior column, an exterior column and exact zero.
    anchors = [1.0 + j / 5.0 for j in range(batch)]
    d = [[0.0 for _ in range(batch)] for _ in range(3)]
    for j in range(batch):
        d[0][j] = [0.2, 2.0, 0.0][j % 3] * anchors[j]
    projected = clamp_columns(d, anchors)
    for j in range(batch):
        norm = math.sqrt(sum(row[j] ** 2 for row in projected))
        close("native_group_projection", norm,
              min(abs(d[0][j]), 0.75 * anchors[j]), checks)
        assert norm <= 0.75 * anchors[j] + 1e-12
    return {"batch": batch, "radius_factor": 0.75,
            "interior_unchanged": True, "exterior_projected": batch >= 2,
            "zero_unchanged": batch >= 3}


def coverage_check(batch, past_count):
    current = list(range(batch))
    past = list(range(past_count))
    probe_candidate_indices = [5, 10, 15, 20]
    partitions = []
    seen_current, seen_past = [], []
    for partition, candidate_index in enumerate(probe_candidate_indices):
        curr = [r for r in current if r % 4 == partition]
        previous = [r for r in past if r % 4 == partition]
        seen_current.extend(curr)
        seen_past.extend(previous)
        updates_before_probe = candidate_index - 1
        assert updates_before_probe in [4, 9, 14, 19]
        partitions.append({"native_candidate_index": candidate_index,
                           "adam_updates_before_probe": updates_before_probe,
                           "timing": "before_update",
                           "next_adam_update_index": candidate_index,
                           "current_count": len(curr), "past_count": len(previous),
                           "skip_current_forward": not curr,
                           "skip_past_forward": not previous})
    assert sorted(seen_current) == current and len(set(seen_current)) == batch
    assert sorted(seen_past) == past and len(set(seen_past)) == past_count
    native_candidate_indices = list(range(1, 26))  # Candidate 1 is the initial state.
    native_evaluation_states = list(range(25))  # Number of completed Adam updates.
    updates = list(range(1, 25))
    assert len(native_evaluation_states) == len(updates) + 1
    assert len(native_candidate_indices) == len(native_evaluation_states)
    assert all(state == candidate - 1 for candidate, state in
               zip(native_candidate_indices, native_evaluation_states))
    assert set(probe_candidate_indices).issubset(updates)
    feedback_rows = len(seen_current) + len(seen_past)
    native_forward_rows = len(native_candidate_indices) * batch
    native_backward_rows = len(updates) * batch
    terminal_physical_forward_rows = batch
    total_forward_rows = native_forward_rows + feedback_rows + terminal_physical_forward_rows
    total_backward_rows = native_backward_rows + feedback_rows
    assert total_forward_rows == 27 * batch + past_count
    assert total_backward_rows == 25 * batch + past_count
    return {"batch": batch, "past_count": past_count,
            "native_candidate_evaluations": len(native_evaluation_states),
            "adam_updates": len(updates), "probe_partitions": partitions,
            "total_feedback_request_rows": feedback_rows,
            "request_row_costs": {
                "native_forward": native_forward_rows,
                "native_backward": native_backward_rows,
                "feedback_forward": feedback_rows,
                "feedback_backward": feedback_rows,
                "terminal_physical_forward": terminal_physical_forward_rows,
                "terminal_physical_backward": 0,
                "terminal_extra_adam_updates": 0,
                "total_forward": total_forward_rows,
                "total_backward": total_backward_rows,
                "forward_formula": "27*B+M",
                "backward_formula": "25*B+M"},
            "all_requests_seen_exactly_once": True,
            "note": "Request rows, not token rows or GPU forward-call counts."}


def run():
    rng = random.Random(20261002)
    checks, geometries, gradient_checks, projections = [], [], [], []
    for batch in [1, 3, 7]:
        layers = [geometry(rng, batch, din, dout, checks)
                  for din, dout in [(4, 3), (6, 5), (5, 4)]]
        geometries.extend([{key: layer[key] for key in
                            ["batch", "din", "dout", "contexts", "sigma", "coupling"]}
                           for layer in layers])
        for metric in ["G", "E"]:
            for arm in ["A", "B"]:
                gradient_checks.append(dict(batch=batch, **gradient_check(layers, metric, arm)))
        projections.append(projection_check(batch, checks))
    identical = geometry(rng, 3, 5, 4, checks)
    split_results = []
    for count in [1, 2, 5]:
        split = [dict(identical, D=scale(identical["D"], 1.0 / count))
                 for _ in range(count)]
        for metric in ["G", "E"]:
            original = cost([identical], metric, "A")
            ratio_a = cost(split, metric, "A") / original
            ratio_b = cost(split, metric, "B") / original
            close("identical_path_arm_A_split_neutral", ratio_a, 1.0, checks)
            close("identical_path_arm_B_spread_incentive", ratio_b, 1 / math.sqrt(count), checks)
            split_results.append({"layers": count, "metric": metric,
                                  "arm_A_ratio": ratio_a, "arm_B_ratio": ratio_b,
                                  "expected_arm_B_ratio": 1 / math.sqrt(count)})
    zero = dict(identical, D=zeros(4, 3))
    zero_subgradient = []
    for metric in ["G", "E"]:
        for arm in ["A", "B"]:
            assert cost([zero], metric, arm) == 0.0
            assert max_abs(analytic_gradients([zero], metric, arm)[0]) == 0.0
            zero_subgradient.append({"metric": metric, "arm": arm, "selected_subgradient": 0.0})
    coverage = [coverage_check(b, m) for b in [1, 3, 7] for m in [0, 1, 5, 10]]
    return {"schema": "jlz-subject-adam-v6-math-validation-v1", "passed": True,
            "seed": 20261002, "cpu_only": True, "standard_library_only": True,
            "checks": checks, "geometries": geometries,
            "gradient_checks": gradient_checks, "native_projection_checks": projections,
            "identical_path_split_checks": split_results,
            "zero_subgradient_convention": zero_subgradient,
            "feedback_coverage_checks": coverage,
            "limitations": [
                "Synthetic matrices validate algebra, not model efficacy or causal mechanisms.",
                "Split neutrality and 1/sqrt(L) scaling assume identical layer paths and sigma.",
                "Root norms need a selected subgradient at zero; no ordinary derivative is claimed.",
                "E and G use a fixed entry geometry; changing keys after writes changes realized effects.",
                "25 native candidate evaluations exclude additional writer-feedback and terminal physical forwards.",
                "The 27B+M/25B+M costs count optimization request rows, excluding geometry/teacher preparation and benchmark evaluation.",
                "Logical request coverage does not imply a fixed token count or runtime across batch sizes."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("validation-results.json"))
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    script = Path(__file__).resolve()
    receipt = {"command": shlex.join([sys.executable, str(script), "--output", str(args.output.resolve())]),
               "python_version": sys.version, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
               "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
               "passed": result["passed"], "result_path": str(args.output.resolve())}
    args.output.with_name("run-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": True, "identity_checks": len(result["checks"]),
                      "gradient_checks": len(result["gradient_checks"]),
                      "coverage_cases": len(result["feedback_coverage_checks"]),
                      "max_gradient_absolute_error": max(c["max_absolute_error"] for c in result["gradient_checks"]),
                      "result": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
