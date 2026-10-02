"""CPU FP64 math checks for native joint local-delta v4; not a model pilot.

All five subject-shared deltas are independent optimization variables in ONE
causal nonlinear branch. This does not use R-derived Z or an auxiliary/full
branch mixture. Geometry is fixed within this toy batch. The native norm is
nonsmooth at zero, so zero/single-layer points are checked for feasibility;
directional derivatives use nonzero interior blocks.
"""
from pathlib import Path
import hashlib
import json
import torch
import torch.nn.functional as F


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    rng = torch.Generator(device="cpu").manual_seed(20261003)
    rand = lambda *shape: torch.randn(*shape, generator=rng, device="cpu")
    L, B, C, T, H, I, V = 5, 2, 6, 6, 4, 7, 11
    weights = [rand(H, I) * .3 for _ in range(L)]
    up = [rand(I, H) * .3 for _ in range(L)]
    causal_up = [rand(I, H) * .2 for _ in range(L)]
    head = rand(V, H) * .4
    x0 = rand(B * (C + 1), T, H) * .7
    requests = torch.arange(B).repeat_interleave(C + 1)
    contexts = torch.arange(C + 1).repeat(B)
    subject = requests + 1
    is_kl = contexts == C
    canonical = torch.nonzero(contexts == 0).flatten()
    target = (requests * 3 + 2) % V
    rows = torch.arange(len(requests))
    subject_mask = F.one_hot(subject, T).to(dtype=x0.dtype)[:, :, None]

    def run(deltas=None, single=None):
        x = x0
        keys, local_before, local_after = [], [], []
        for layer in range(L):
            causal = x.cumsum(1) / torch.arange(1, T + 1)[None, :, None]
            k = torch.tanh(F.linear(x, up[layer]) + F.linear(causal, causal_up[layer]))
            # Native delta is added at block output. A down_proj write also
            # contributes with coefficient one to this residual output.
            v = x + F.linear(k, weights[layer]) + .1 * torch.tanh(x)
            keys.append(k[rows, subject])
            local_before.append(v[rows, subject])
            if deltas is not None:
                v = v + subject_mask * deltas[layer].T[requests, None, :]
            elif single is not None and layer == single[0]:
                v = v + subject_mask * single[1].T[requests, None, :]
            local_after.append(v[rows, subject])
            x = v
        prediction = torch.where(is_kl, subject, T - 1)
        logits = F.linear(torch.tanh(x[rows, prediction]), head)
        return logits, keys, local_before, local_after

    with torch.no_grad():
        entry_logits, entry_keys, entry_local, _ = run()
        teacher = entry_logits.log_softmax(-1)
        anchor_sq = [v[canonical].square().sum(-1) for v in entry_local]
        assert all(bool((v > 0).all()) for v in anchor_sq)
        radii = [.75 * v.sqrt() for v in anchor_sq]
        coeff = [.5 / v for v in anchor_sq]
        geometry, point = [], []
        for layer in range(L):
            k = torch.stack([.5 * entry_keys[layer][(requests == r) & (contexts == 0)][0] +
                             .1 * entry_keys[layer][(requests == r) & (contexts > 0) & ~is_kl].sum(0)
                             for r in range(B)], dim=1)
            a0 = rand(I, I)
            a = torch.eye(I) + a0.T @ a0
            p = torch.linalg.solve(a + k @ k.T, k)
            q = k.T @ p
            q = .5 * (q + q.T)
            geometry.append((a, k, p, q))
            d = rand(H, B)
            point.append(d * (.35 * radii[layer] / d.norm(dim=0))[None, :])

    def task(logits):
        logp = logits.log_softmax(-1)
        nll = -logp[~is_kl, target[~is_kl]].sum() / C
        kl = (logp[is_kl].exp() * (logp[is_kl] - teacher[is_kl])).sum()
        return nll + .0625 * kl

    def penalty(ds):
        return sum((c * d.norm(dim=0)).sum() for c, d in zip(coeff, ds))

    def reduced_value(d, q):
        return .5 * ((d @ (torch.eye(B) - q)) * d).sum()

    def objective(ds, eta):
        logits = run(ds)[0]
        ridge = sum(reduced_value(d, g[3]) / scale.mean()
                    for d, g, scale in zip(ds, geometry, anchor_sq))
        return task(logits) + penalty(ds) + eta * ridge

    pack = lambda values: torch.cat([v.reshape(-1) for v in values])
    ridge_checks = []
    for layer, (a, k, p, q) in enumerate(geometry):
        d = point[layer].clone().requires_grad_(True)
        u = d @ p.T
        error = u @ k - d
        primal = .5 * error.square().sum() + .5 * ((u @ a) * u).sum()
        reduced = reduced_value(d, q)
        gradient = torch.autograd.grad(primal, d)[0]
        expected = d.detach() @ (torch.eye(B) - q)
        stationarity = u.detach() @ (k @ k.T + a) - d.detach() @ k.T
        row = dict(layer=4 + layer, primal=float(primal.detach()), reduced=float(reduced.detach()),
                   value_error=abs(float((primal - reduced).detach())),
                   gradient_error=float((gradient - expected).norm()),
                   writer_stationarity_error=float(stationarity.norm()),
                   minimum_A_eigenvalue=float(torch.linalg.eigvalsh(a).min()))
        assert max(row["value_error"], row["gradient_error"], row["writer_stationarity_error"]) < 1e-11
        ridge_checks.append(row)

    direction = [rand(*d.shape) for d in point]
    direction = [v / pack(direction).norm() for v in direction]
    joint_checks = {}
    for arm, eta in (("A", 0.), ("B", 1.)):
        ds = [d.clone().requires_grad_(True) for d in point]
        value = objective(ds, eta)
        gradients = torch.autograd.grad(value, ds)
        expected = float(sum((g * v).sum() for g, v in zip(gradients, direction)))
        differences = []
        for eps in (1e-4, 1e-5, 1e-6):
            with torch.no_grad():
                plus = objective([d + eps * v for d, v in zip(point, direction)], eta)
                minus = objective([d - eps * v for d, v in zip(point, direction)], eta)
            observed = float((plus - minus) / (2 * eps))
            differences.append(dict(epsilon=eps, observed=observed, analytic=expected,
                                    absolute_error=abs(observed - expected)))
        assert min(x["absolute_error"] for x in differences) < 1e-7
        joint_checks[arm] = dict(eta=eta, total=float(value.detach()), finite_difference=differences)

    # A mixed derivative directly checks coupling of the earliest/latest delta.
    ds = [d.clone().requires_grad_(True) for d in point]
    task_value = task(run(ds)[0])
    last_grad = torch.autograd.grad(task_value, ds[-1], create_graph=True)[0]
    mixed = torch.autograd.grad((last_grad * direction[-1]).sum(), ds[0])[0]
    assert float(mixed.norm()) > 1e-9
    eps = 1e-5
    projected_last = []
    for sign in (1., -1.):
        test = [d.clone().requires_grad_(True) for d in point]
        test[0] = test[0] + sign * eps * direction[0]
        last = torch.autograd.grad(task(run(test)[0]), test[-1])[0]
        projected_last.append(float((last * direction[-1]).sum()))
    cross_fd = (projected_last[0] - projected_last[1]) / (2 * eps)
    cross_ad = float((mixed * direction[0]).sum())
    assert abs(cross_fd - cross_ad) < 1e-8

    # Each singleton reduces to a conventional single-site additive-delta
    # objective at that same layer, interception site, and final readout.
    singleton_checks = []
    feasible = lambda ds: all(bool((d.norm(dim=0) <= r).all()) for d, r in zip(ds, radii))
    zero = [torch.zeros_like(d) for d in point]
    for layer in range(L):
        joint_d = point[layer].clone().requires_grad_(True)
        ds = [joint_d if i == layer else torch.zeros_like(d) for i, d in enumerate(point)]
        native_d = point[layer].clone().requires_grad_(True)
        jlogits = run(ds)[0]
        nlogits = run(single=(layer, native_d))[0]
        jloss = task(jlogits) + penalty(ds)
        nloss = task(nlogits) + (coeff[layer] * native_d.norm(dim=0)).sum()
        jgrad = torch.autograd.grad(jloss, joint_d)[0]
        ngrad = torch.autograd.grad(nloss, native_d)[0]
        row = dict(layer=4 + layer, loss_error=abs(float((jloss - nloss).detach())),
                   gradient_error=float((jgrad - ngrad).norm()),
                   logits_error=float((jlogits - nlogits).detach().abs().max()), feasible=feasible(ds))
        assert max(row["loss_error"], row["gradient_error"], row["logits_error"]) < 1e-12
        assert row["feasible"]
        singleton_checks.append(row)

    # Fixed entry absolute targets remove the lower-delta-induced change of
    # this subject's local block output, not every position in the stream.
    lower_only = [point[0]] + [torch.zeros_like(d) for d in point[1:]]
    before = run(lower_only)[2][-1][canonical]
    baseline = entry_local[-1][canonical]
    lower_response = before - baseline
    absolute_target = baseline  # Upper delta is zero in this fixture.
    wrong_added_delta = absolute_target - before
    correct_added_delta = torch.zeros_like(before)
    assert float(lower_response.norm()) > 1e-8
    assert torch.allclose(wrong_added_delta, -lower_response, rtol=0, atol=0)

    # A scalar SPD fixture is sufficient to refute monotonicity of write
    # energy alone as protection grows, while minimized writer value grows.
    history_growth = []
    for a in (4., 16., 64.):
        k, d = 2., 1.
        u = d * k / (a + k * k)
        fit = .5 * (u * k - d) ** 2
        energy = .5 * u * u * a
        history_growth.append(dict(A=a, H_increment=a - 4., U=u, realized=u * k,
                                   fit_residual_cost=fit, energy_only=energy,
                                   minimized_writer_value=fit + energy,
                                   reduced_value=.5 * d * d * (1 - k * k / (a + k * k))))
    assert all(b["energy_only"] < a["energy_only"] and
               b["minimized_writer_value"] > a["minimized_writer_value"]
               for a, b in zip(history_growth, history_growth[1:]))
    assert feasible(zero) and feasible(point)
    script = Path(__file__).resolve()
    result = dict(status="PASS_CPU_FP64_TOY_NOT_REAL_LLAMA_PILOT", device="cpu", dtype="float64",
                  torch_version=str(torch.__version__), script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                  dimensions=dict(layers=[4, 5, 6, 7, 8], requests=B, rewrite_contexts=C, KL_rows_per_request=1,
                                  tokens=T, hidden=H, intermediate=I, vocabulary=V),
                  objective=dict(NLL="sum requests, mean six rewrite contexts", KL=".0625 * current||entry",
                                 intervention_site="subject_last at native block output; independent delta per layer/request",
                                 norm="sum .5*||delta_lr||/||entry_anchor_lr||^2", clamp=".75*||entry_anchor_lr||",
                                 cost="eta * sum_l V_l(delta_l)/mean_r ||entry_anchor_lr||^2", eta={"A": 0, "B": 1},
                                 geometry_key_weights=[.5, .1, .1, .1, .1, .1], no_R_derived_Z=True, no_branch_mixture=True),
                  ridge_writer=ridge_checks, joint_gradient=joint_checks,
                  cross_layer=dict(first_layer=4, last_layer=8, mixed_gradient_norm=float(mixed.norm()),
                                   directional_autograd=cross_ad, directional_finite_difference=cross_fd,
                                   absolute_error=abs(cross_fd - cross_ad)),
                  one_active_layer_native_reduction=singleton_checks,
                  absolute_target_error=dict(local_lower_response_norm=float(lower_response.norm()),
                                             wrong_added_delta_norm=float(wrong_added_delta.norm()),
                                             correct_added_delta_norm=float(correct_added_delta.norm()),
                                             cancels_local_lower_response=True,
                                             scope="subject local block output; not all token positions in the residual stream"),
                  protection_growth_counterexample=history_growth,
                  allocation=dict(all_layers_eligible=True, zero_feasible=feasible(zero), all_five_feasible=feasible(point),
                                  all_single_layers_feasible=all(r["feasible"] for r in singleton_checks), minimum_active_layers=None),
                  limitations=["No real Llama inference, GPU, downloads, timing, or quality prediction.",
                               "Singleton reduction uses the same toy interception site and readout; it does not certify every native code path.",
                               "The norm is nonsmooth at zero; no unique gradient at zero is asserted.",
                               "The scalar protection example proves a possible energy-only failure, not that actual runs follow that scalar path."])
    out = script.with_name("math-check.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], output=str(out), cross_layer=result["cross_layer"]), indent=2))


if __name__ == "__main__":
    main()
