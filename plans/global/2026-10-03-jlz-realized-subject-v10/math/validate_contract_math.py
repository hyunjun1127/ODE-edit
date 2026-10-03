"""Small CPU algebra checks; no language-model experiment or performance claim."""
import json
from pathlib import Path

import numpy as np


def main():
    rng = np.random.default_rng(1032026)
    f = rng.normal(size=(8, 8))
    a = f @ f.T + 2 * np.eye(8)
    k = rng.normal(size=(8, 3))
    r = rng.normal(size=(5, 3))
    ainvk = np.linalg.solve(a, k)
    s = k.T @ ainvk
    p = np.linalg.solve(a + k @ k.T, k)
    u = r @ p.T
    m = p.T @ k
    y = u @ k
    exact_same_y = y @ np.linalg.solve(s, ainvk.T)
    energy = np.trace(u @ a @ u.T)
    energy_y = np.trace(y @ np.linalg.solve(s, y.T))
    conditional = []
    for i in range(k.shape[1]):
        others = np.delete(k, i, axis=1)
        conditional.append(float(k[:, i] @ np.linalg.solve(a + others @ others.T, k[:, i])))
    odds = np.diag(m) / (1 - np.diag(m))
    errors = {
        "same_y_same_U_max_abs": float(np.max(np.abs(u - exact_same_y))),
        "same_y_energy_abs": float(abs(energy - energy_y)),
        "M_identity_max_abs": float(np.max(np.abs(m - s @ np.linalg.solve(np.eye(3) + s, np.eye(3))))),
        "diag_odds_conditional_capacity_max_abs": float(np.max(np.abs(odds - conditional))),
    }
    assert all(v < 1e-10 for v in errors.values()), errors

    low, high = 1.63, .80  # quoted illustrative capacities; not raw telemetry
    ratios = {
        "old_G_plus_E_root_same_R_L8_over_L4": ((1 + low) / (1 + high)) ** .5,
        "old_G_plus_E_root_same_Y_L8_over_L4": ((1 + high) / (1 + low)) ** .5 * low / high,
        "new_G_only_root_same_Y_L8_over_L4": (low / high) ** .5,
        "new_G_only_energy_same_Y_L8_over_L4": low / high,
    }
    regularization_ratios = {
        str(gamma): {
            "old_over_new_decay_at_same_realized_effect": 1 / gamma,
            "old_over_new_root_allocation_at_same_realized_effect": 1 / gamma ** .5,
        } for gamma in [.60, .45]
    }

    # Two-token, two-layer scalar causal toy. Layer 2 subject key mixes prefix.
    # Fit masks layer 1's identical U to the subject; deployment applies it to both.
    x = np.array([1., 2.])
    u1 = .9 * x[1] / (2 + x[1] ** 2)
    masked = x.copy()
    masked[1] += u1 * x[1]
    actual = x + u1 * x
    k2_fit = masked[1] + .5 * masked[0]
    k2_actual = actual[1] + .5 * actual[0]
    u2_fit = 1.2 * k2_fit / (2 + k2_fit ** 2)
    u2_actual = 1.2 * k2_actual / (2 + k2_actual ** 2)
    toy = {
        "description": "Synthetic causal mixing example, not a measured JLZ result",
        "layer1_U": float(u1),
        "layer2_K_fit": float(k2_fit),
        "layer2_K_actual": float(k2_actual),
        "layer2_U_from_fit_K": float(u2_fit),
        "layer2_U_from_actual_K": float(u2_actual),
        "fit_local_increment": float(u2_fit * k2_fit),
        "deployed_fixed_U_local_increment": float(u2_fit * k2_actual),
        "rebuilt_U_local_increment": float(u2_actual * k2_actual),
        "fixed_U_application_same_input_identity_error": float(abs((1 + u2_fit) * k2_actual - (k2_actual + u2_fit * k2_actual))),
        "S_direct_change_gap": float(u2_fit * (k2_actual - k2_fit)),
        "T_direct_change_gap": float(u2_actual * (k2_actual - k2_fit)),
        "Tprime_direct_change_gap": 0.,
        "Tprime_remaining_base_gap": float(k2_actual - k2_fit),
    }
    assert k2_fit != k2_actual and u2_fit != u2_actual
    assert toy["fixed_U_application_same_input_identity_error"] < 1e-12
    result = {
        "kind": "cpu_algebra_only_not_model_validation",
        "seed": 1032026,
        "matrix_shape": {"key": [8, 3], "coefficient": [5, 3]},
        "matrix_identity_errors": errors,
        "scalar_cost_ratios": ratios,
        "scalar_regularization_ratios_not_terminal_magnitude_predictions": regularization_ratios,
        "fixed_writer_vs_changed_state_toy": toy,
    }
    Path(__file__).with_name("results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
