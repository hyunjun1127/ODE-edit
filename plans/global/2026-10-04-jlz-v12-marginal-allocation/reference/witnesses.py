"""Deterministic toy witnesses; these are not model editing experiments."""

import numpy as np

try:
    from .budget import project
    from .kkt import kkt_diagnostics
    from .optimizer import EfficiencyAdam, objective_gradients
except ImportError:
    from budget import project
    from kkt import kkt_diagnostics
    from optimizer import EfficiencyAdam, objective_gradients


def non_kkt_witness(steps=24):
    """A convex objective with a non-KKT projected EfficiencyAdam fixed point.

    F=(3,4).u1+(1,2).u2, decay=.1, shared radius=.75. The default
    epsilon slightly perturbs the diagonal direction; four fixed-point
    substitutions determine it to float64 precision before optimizer steps.
    The optimizer starts with zero moments at this nonzero feasible point.
    """
    linear = [np.array([3.0, 4.0]), np.array([1.0, 2.0])]
    radius, decay = 0.75, 0.1
    opt = EfficiencyAdam([(2,), (2,)], lr=0.1)
    q = -np.ones(2) / np.sqrt(2.0)
    for _ in range(4):
        grad = linear[0] + decay * q
        direction = grad / (np.abs(grad) + opt.eps)
        q = -direction / np.linalg.norm(direction)
    u = [radius * q, np.zeros(2)]
    initial = [b.copy() for b in u]
    max_move = 0.0
    for _ in range(steps):
        candidate, gamma = opt.step(u, objective_gradients(u, linear, decay))
        u, tau_proj, _ = project(candidate, radius)
        max_move = max(max_move, max(float(np.max(np.abs(a - b))) for a, b in zip(u, initial)))
    objective = sum(float(g @ b) + decay * np.linalg.norm(b) for g, b in zip(linear, u))
    optimum = radius * (-np.linalg.norm(linear[0]) + decay)
    return {
        "status": "EXPECTED_LIMITATION",
        "claim": "Projected EfficiencyAdam can have non-KKT fixed points, even on this convex objective.",
        "initialization": "Nonzero feasible fixed point, zero Adam moments; no model calls.",
        "steps": steps,
        "radius": radius,
        "decay": decay,
        "linear_F_gradients": [g.tolist() for g in linear],
        "final_blocks": [b.tolist() for b in u],
        "max_displacement_from_initial": max_move,
        "objective": float(objective),
        "exact_optimum": float(optimum),
        "objective_gap": float(objective - optimum),
        "tau_proj": float(tau_proj),
        "gamma": gamma.tolist(),
        "kkt": kkt_diagnostics(u, linear, radius, decay),
    }


def sparse_first_step_witness():
    grads = [np.array([1.0, 0.0]), np.ones(2) / np.sqrt(2.0)]
    step, gamma = EfficiencyAdam([(2,), (2,)], lr=0.1).step([np.zeros(2), np.zeros(2)], grads)
    return {"status": "EXPECTED_LIMITATION", "gradient_rms_ratio": 1.0,
            "step_norm_ratio": float(np.linalg.norm(step[0]) / np.linalg.norm(step[1])),
            "gamma": gamma.tolist(), "claim": "Equal RMS does not imply equal first-step norm with different gradient support."}
