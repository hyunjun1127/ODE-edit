"""Reproducible, CPU-only JLZ reference audit; no model downloads or GPU jobs.

Run with single-thread BLAS, for example::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -m project.run_scripts.jlz_smoke.run --output-dir local/jlz-smoke-20261001

The default native adapter fixes the reference KL direction. --objective
reference audits the untouched attachment and exits 3 for its known KL mismatch.
Neither mode passes real-model native JLZ gates.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

# Set before NumPy is imported; record the values actually used in the receipt.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from project.run_scripts.jlz_ref.problem import JLZProblem, prox_blocks
from project.run_scripts.jlz_ref.prox import solve
from project.run_scripts.jlz_ref.toy import ToyLM
from project.run_scripts.jlz_ref.writer import write_batch as reference_write_batch
from .native_problem import NativeKLProblem
from .native_writer import write_batch as native_write_batch


class MeteredToy(ToyLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []

    def forward(self, H0, deltas=None):
        start = time.perf_counter()
        result = super().forward(H0, deltas)
        self.events.append(dict(kind="forward", prompts=len(H0), seconds=time.perf_counter() - start))
        return result

    def backward(self, cache, g_logits, layers):
        start = time.perf_counter()
        result = super().backward(cache, g_logits, layers)
        self.events.append(dict(kind="backward", prompts=len(g_logits), seconds=time.perf_counter() - start))
        return result


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def all_finite(values):
    if isinstance(values, dict):
        return all(all_finite(x) for x in values.values())
    if isinstance(values, (list, tuple)):
        return all(all_finite(x) for x in values)
    if isinstance(values, (np.ndarray, float, int, np.number)):
        return bool(np.all(np.isfinite(values)))
    return True


def digest(values):
    h = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        h.update(str(array.shape).encode())
        h.update(array.dtype.str.encode())
        h.update(array.tobytes())
    return h.hexdigest()


def source_manifest():
    root = Path(__file__).resolve().parent
    paths = [root / name for name in ("run.py", "native_problem.py", "native_writer.py")]
    paths.extend(root.parent / "jlz_ref" / name for name in ("problem.py", "prox.py", "writer.py", "toy.py"))
    return {str(path.relative_to(root.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def setup(seed, *, d=32, m=16, tol=1e-3, cap=400, objective="native"):
    model = MeteredToy(d=d, m=m, V=24, L=7, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    X = rng.standard_normal((256, d))
    X *= 2.0 / np.linalg.norm(X, axis=1, keepdims=True)
    cache = model.forward(X)
    layers = [1, 2, 3, 4, 5]
    C0 = {l: cache["xs"][l].T @ cache["xs"][l] / len(X) for l in layers}
    state = dict(C0=C0, H={l: np.zeros((m, m)) for l in layers})
    contexts = np.vstack([np.zeros(d)] + [0.6 * rng.standard_normal(d) / np.sqrt(d) for _ in range(5)])
    batch = []
    for r in range(4):
        h0 = rng.standard_normal(d)
        h0 *= 2.0 / np.linalg.norm(h0)
        logits = model.forward(h0[None])["logits"][0]
        batch.append(dict(id=f"seed{seed}-request{r}", h0=h0,
                          h_kl=h0 + 0.2 * rng.standard_normal(d) / np.sqrt(d),
                          t_new=int(np.argsort(logits)[model.V // 2])))
    cfg = dict(layers=layers, contexts=contexts, lam=float(m), tol=tol, cap=cap,
               wd=0.5, clamp=0.75, kl_factor=0.0625, stop=0.05, objective=objective)
    return model, state, batch, cfg


def make_problem(model, state, batch, cfg):
    problem_class = NativeKLProblem if cfg["objective"] == "native" else JLZProblem
    return problem_class(model, batch, cfg["layers"], cfg["contexts"], state["C0"], state["H"],
                      cfg["lam"], wd=cfg["wd"], clamp=cfg["clamp"], kl_factor=cfg["kl_factor"],
                      stop=cfg["stop"])


def kkt_block(v, g, c, rho, *, atol=1e-12):
    """Signed radial condition: boundary optimality requires radial <= 0."""
    n = float(np.linalg.norm(v))
    if n == 0.0:
        violation = max(0.0, float(np.linalg.norm(g)) - c)
        return dict(state="ZERO", norm_over_radius=0.0,
                    subgradient_ratio=float(np.linalg.norm(g)) / c,
                    stationarity_violation=violation)
    u = v / n
    total = g + c * u
    radial = float(total @ u)
    tangent = float(np.linalg.norm(total - radial * u))
    boundary = abs(n - rho) <= atol * max(1.0, rho)
    radial_violation = max(radial, 0.0) if boundary else abs(radial)
    return dict(state="BOUNDARY" if boundary else "INTERIOR", norm_over_radius=n / rho,
                signed_radial=radial, radial_violation=radial_violation,
                tangential_residual=tangent, clamp_multiplier=max(-radial, 0.0) if boundary else 0.0,
                stationarity_violation=float(np.hypot(tangent, radial_violation)))


def event_summary(events):
    return dict(forward_calls=sum(e["kind"] == "forward" for e in events),
                backward_calls=sum(e["kind"] == "backward" for e in events),
                forward_prompt_evaluations=sum(e["prompts"] for e in events if e["kind"] == "forward"),
                forward_seconds=sum(e["seconds"] for e in events if e["kind"] == "forward"),
                backward_seconds=sum(e["seconds"] for e in events if e["kind"] == "backward"))


def audit_write(model, state, batch, cfg, *, name, seed):
    """Run the selected writer and independently check its committed state."""
    require(all_finite([model.W, model.U, model.b, model.E, state, batch, cfg]), "NONFINITE_INPUT")
    before = copy.deepcopy(model)
    before.events.clear()
    W0 = [x.copy() for x in model.W]
    H0 = {l: x.copy() for l, x in state["H"].items()}
    event_start = len(model.events)
    start = time.perf_counter()
    writer = native_write_batch if cfg["objective"] == "native" else reference_write_batch
    rec = writer(model, state, batch, cfg)
    write_seconds = time.perf_counter() - start
    events = model.events[event_start:]
    n = rec["solver"]["calls"]
    forwards = [e for e in events if e["kind"] == "forward"]
    backwards = [e for e in events if e["kind"] == "backward"]
    require(len(forwards) == 2 + 2 * n + 4, "forward accounting differs from reference pipeline")
    require(len(backwards) == 2 * n, "backward accounting differs from oracle calls")
    prob, R, deltas = rec["_problem"], rec["_R"], rec["_deltas"]
    require(n <= cfg["cap"], "oracle call cap exceeded")
    require(all_finite([rec["solver"], R, deltas, state["H"], model.W, rec["allocation"]]), "NONFINITE_OUTPUT")
    audit_start = time.perf_counter()
    post_start = len(model.events)
    fit_rw = before.forward(prob.rw, deltas)
    fit_kl = before.forward(prob.klp, deltas)
    post_rw = model.forward(prob.rw)
    post_kl = model.forward(prob.klp)
    require(np.array_equal(fit_rw["logits"], post_rw["logits"]), "rewrite logits differ after commit")
    require(np.array_equal(fit_kl["logits"], post_kl["logits"]), "KL logits differ after commit")
    require(rec["consistency"] == 0.0, "NLL/KL fit and commit differ")
    require(rec["lowest_layer_key_change"] == 0.0, "lowest edited layer key moved")
    for l in range(model.L):
        expected = W0[l] + deltas[l] if l in deltas else W0[l]
        require(np.array_equal(model.W[l], expected), f"weight commit mismatch layer {l}")
    geometry = []
    for l in cfg["layers"]:
        K_post = post_rw["xs"][l].reshape(prob.B, prob.C, -1).mean(axis=1).T
        require(np.array_equal(state["H"][l], H0[l] + K_post @ K_post.T), f"H append mismatch layer {l}")
        np.testing.assert_allclose(state["H"][l], state["H"][l].T, rtol=0.0, atol=1e-12)
        K, A, adj = prob.K[l], prob.A[l], prob.adj[l]
        M = A + K @ K.T
        residual = np.linalg.norm(M @ adj - K) / max(1.0, np.linalg.norm(K))
        AinvK = np.linalg.solve(A, K)
        woodbury = AinvK @ np.linalg.solve(np.eye(prob.B) + K.T @ AinvK, np.eye(prob.B))
        relative = np.linalg.norm(adj - woodbury) / max(1.0, np.linalg.norm(adj))
        require(residual < 1e-11 and relative < 1e-11, f"adj algebra check failed layer {l}")
        G = K.T @ AinvK
        eig = np.linalg.eigvalsh(G)
        geometry.append(dict(layer=l, key_rank=int(np.linalg.matrix_rank(K)), adj_solve_residual=float(residual),
                             adj_woodbury_relative_error=float(relative), G_eigen_min=float(eig[0]), G_eigen_max=float(eig[-1])))
        for r in range(prob.B):
            require(np.linalg.norm(R[l][:, r]) <= prob.rho[l][r] * (1 + 1e-12), "infeasible block")
            if not prob.active[r]:
                require(np.count_nonzero(R[l][:, r]) == 0, "zero-step target not exactly zero")
    # prob retains entry keys/teacher; use the saved entry model, avoiding a second delta.
    prob.model = before
    try:
        f, G, terms = prob.smooth(R)
    finally:
        prob.model = model
    require(all_finite([f, G, terms]), "NONFINITE_AUDIT_ORACLE")
    rows = [dict(layer=l, request=r, **kkt_block(R[l][:, r], G[l][:, r], prob.c[l][r], prob.rho[l][r]))
            for l, r in prob.blocks]
    c, rho = prob.block_params()
    x, g = prob.pack(R), prob.pack(G)
    scale = rec["solver"].get("gradient_scale", 1.0)
    residual = float(np.linalg.norm(x - prox_blocks(x - g, 1.0, c, rho, prob.d)) / scale)
    if n:
        require(abs(residual - rec["solver"]["normalized_residual"]) < 1e-12, "return residual mismatch")
    status = rec["solver"]["status"]
    require(status in {"CONVERGED", "STALLED_AT_PRECISION", "NOT_CONVERGED", "LINESEARCH_FAILED", "POLICY_ZERO_STEP"},
            "unknown solver status")
    if status == "CONVERGED":
        require(residual <= cfg["tol"], "CONVERGED status without residual tolerance")
    active = prob.active
    initial_nll = float(np.sum(prob.nll_entry[active]))
    final_nll = float(np.sum(np.asarray(rec["nll_fit"])[active]))
    raw_kl = float(np.sum(np.asarray(rec["kl_fit"])[active]))
    decay = rec["decay"]
    total = final_nll + cfg["kl_factor"] * raw_kl + decay
    require(all_finite([initial_nll, total, rows, geometry]), "NONFINITE_RECEIPT")
    if n:
        require(abs(total - rec["solver"]["value"]) < 1e-10, "objective decomposition mismatch")
    require(np.array_equal(prob.K[min(cfg["layers"])],
                           post_rw["xs"][min(cfg["layers"])].reshape(prob.B, prob.C, -1).mean(axis=1).T),
            "lowest layer full key mismatch")
    audit_events = before.events + model.events[post_start:]
    audit_seconds = time.perf_counter() - audit_start
    config_identity = {key: value for key, value in cfg.items() if key != "contexts"}
    config_identity["contexts_sha256"] = digest([cfg["contexts"]])
    config_sha256 = hashlib.sha256(json.dumps(config_identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    fixture_arrays = [cfg["contexts"], model.E, *model.U, *model.b, *W0]
    fixture_arrays.extend(state["C0"][l] for l in cfg["layers"])
    fixture_arrays.extend(H0[l] for l in cfg["layers"])
    for request in batch:
        fixture_arrays.extend([request["h0"], request["h_kl"], np.asarray(request["t_new"], dtype=np.int64)])
    return dict(name=name, seed=seed, request_ids=[q["id"] for q in batch], batch_size=len(batch),
                technical_pass=True, objective_fidelity_to_native=("TOY_NATIVE_KL_DIRECTION" if cfg["objective"] == "native" else "FAIL_KL_DIRECTION"),
                optimization_converged=status in {"CONVERGED", "POLICY_ZERO_STEP"},
                solver=rec["solver"], active_requests=int(active.sum()), zero_step=rec["zero_step"],
                active_blocks=sum(row["state"] != "ZERO" for row in rows), free_blocks=len(prob.blocks),
                initial=dict(total=initial_nll, nll=initial_nll, kl=0.0, weighted_kl=0.0, decay=0.0),
                final=dict(total=total, nll=final_nll, kl=raw_kl, weighted_kl=cfg["kl_factor"] * raw_kl, decay=decay),
                per_request=dict(nll_entry=rec["nll_entry"], nll_fit=rec["nll_fit"], kl_fit=rec["kl_fit"]),
                checks=dict(adj_algebra=True, fit_commit_weight_bitwise=True, fit_commit_logits_bitwise=True,
                            fit_commit_loss_max_abs=rec["consistency"], lowest_key_bitwise=True,
                            H_append_bitwise=True, feasible=True, finite=True),
                identity=dict(W_entry=digest(W0), W_post=digest(model.W),
                              H_entry=digest(H0.values()), H_post=digest(state["H"].values()),
                              config_sha256=config_sha256, input_fixture_sha256=digest(fixture_arrays)),
                history_entry_frobenius=float(np.sqrt(sum(np.linalg.norm(x) ** 2 for x in H0.values()))),
                work=dict(write_wall_seconds=write_seconds, oracle_calls=n,
                          entry_forward=event_summary(forwards[:2]),
                          oracle=event_summary(forwards[2:2 + 2 * n] + backwards),
                          fit_forward=event_summary(forwards[2 + 2 * n:4 + 2 * n]),
                          post_forward=event_summary(forwards[4 + 2 * n:]),
                          pipeline_total=event_summary(events),
                          audit_extra=dict(wall_seconds=audit_seconds, **event_summary(audit_events)),
                          audit_extra_oracle_calls=1),
                kkt=dict(normalized_prox_residual=residual,
                         max_raw_stationarity_violation=max([row["stationarity_violation"] for row in rows], default=0.0),
                         blocks=rows), allocation=rec["allocation"], geometry=geometry)


def diagnostic_checks(cfg):
    model, state, batch, _ = setup(0, d=cfg["d"], m=cfg["m"], tol=cfg["tol"], cap=cfg["cap"], objective=cfg["objective"])
    local_cfg = dict(cfg["config"])
    local_cfg["contexts"] = cfg["config"]["contexts"].copy()
    prob = make_problem(model, state, batch[:2], local_cfg)
    rng = np.random.default_rng(4242)
    c, rho = prob.block_params()
    x = rng.standard_normal(len(prob.blocks) * prob.d)
    x = prox_blocks(x, 0.0, c, 0.1 * rho, prob.d)
    _, G, _ = prob.smooth(prob.unpack(x))
    errors = []
    for _ in range(3):
        direction = rng.standard_normal(x.size)
        direction /= np.linalg.norm(direction)
        epsilon = 1e-5
        fp = prob.smooth(prob.unpack(x + epsilon * direction))[0]
        fm = prob.smooth(prob.unpack(x - epsilon * direction))[0]
        analytic = float(prob.pack(G) @ direction)
        error = abs((fp - fm) / (2 * epsilon) - analytic) / max(1.0, abs(analytic))
        require(error < 1e-7, "packed finite difference gradient check failed")
        errors.append(error)
    # Independent native API evidence at nonzero R; zero-R KL cannot distinguish directions.
    import torch
    import torch.nn.functional as F
    _, actual_kl, _, kl_cache = prob.terms(prob.deltas(prob.unpack(x)))
    logits_t = torch.tensor(kl_cache["logits"], dtype=torch.float64, requires_grad=True)
    lp_t = torch.log_softmax(logits_t, dim=-1)
    native_kl_t = F.kl_div(torch.tensor(prob.logp0), lp_t, log_target=True, reduction="none").sum(-1)
    native_kl_t.sum().backward()
    from project.run_scripts.jlz_ref.toy import log_softmax
    lp = log_softmax(kl_cache["logits"])
    p = np.exp(lp)
    native_kl = np.sum(p * (lp - prob.logp0), axis=1)
    analytic_kl_gradient = p * (lp - prob.logp0 - native_kl[:, None])
    native_value_error = float(np.max(np.abs(native_kl - native_kl_t.detach().numpy())))
    native_gradient_error = float(np.max(np.abs(analytic_kl_gradient - logits_t.grad.numpy())))
    problem_vs_native_error = float(np.max(np.abs(actual_kl - native_kl)))
    require(native_value_error < 1e-12 and native_gradient_error < 1e-12, "native torch KL parity failed")
    if cfg["objective"] == "native":
        require(problem_vs_native_error < 1e-12, "adapter differs from native KL")
    else:
        require(problem_vs_native_error > 1e-7, "reference KL direction fixture did not expose mismatch")
    R = prob.zeros()
    nll0 = prob.terms(prob.deltas(R))[0][0]
    layer = local_cfg["layers"][2]
    R[layer][:, 1] = 0.2 * rng.standard_normal(prob.d) / np.sqrt(prob.d)
    nll1 = prob.terms(prob.deltas(R))[0][0]
    coupling = float(nll1 - nll0)
    require(abs(coupling) > 1e-7, "request coupling probe is ineffective")
    # This rejects the false boundary certificate omitted from reference kkt().
    invalid = kkt_block(np.array([1.0, 0.0]), np.zeros(2), 0.5, 1.0)
    valid = kkt_block(np.array([1.0, 0.0]), np.array([-0.8, 0.0]), 0.5, 1.0)
    require(invalid["radial_violation"] == 0.5 and valid["stationarity_violation"] == 0.0,
            "signed boundary KKT diagnostic failed")
    fun = lambda x: (float(0.5 * np.sum(np.array([1., 10.]) * x * x) - np.sum(x)),
                     np.array([1., 10.]) * x - 1.0)
    _, _, capped = solve(fun, lambda x: 0.0, lambda x, t: x, np.zeros(2), tol=1e-12, cap=2)
    require(capped["status"] == "NOT_CONVERGED" and capped["calls"] == 2 and capped["final_recomputed"],
            "restrictive cap did not reserve return evaluation")
    _, _, stalled = solve(lambda x: (float(1.0 + 1e-20 * np.sum(x)), np.full_like(x, 1e-20)),
                          lambda x: 0.0, lambda x, t: x, np.zeros(1), tol=1e-30, cap=20)
    require(stalled["status"] == "STALLED_AT_PRECISION" and stalled["normalized_residual"] > 1e-30,
            "precision plateau incorrectly called convergence")
    _, _, linesearch_failed = solve(lambda x: (float(np.sum(x)), -np.ones_like(x)),
                                    lambda x: 0.0, lambda x, t: x, np.zeros(1),
                                    tol=1e-12, cap=20, max_trials=3)
    require(linesearch_failed["status"] == "LINESEARCH_FAILED", "failed line search mislabeled")
    try:
        solve(fun, lambda x: 0., lambda x, t: x, np.zeros(2), tol=1e-3, cap=1)
    except ValueError as error:
        require(str(error) == "RETURN_EVALUATION_RESERVE", "wrong cap exception")
    else:
        raise AssertionError("cap 1 was accepted")
    nonfinite = []
    for bad_f, bad_g in ((np.nan, np.zeros(1)), (0.0, np.array([np.inf]))):
        try:
            solve(lambda x: (bad_f, bad_g), lambda x: 0.0, lambda x, t: x,
                  np.zeros(1), tol=1e-3, cap=3)
        except FloatingPointError as error:
            require(str(error) == "NONFINITE_ORACLE", "wrong nonfinite exception")
            nonfinite.append(str(error))
        else:
            raise AssertionError("nonfinite oracle was accepted")
    # A nonfinite input must fail before touching W/H in this audit entrypoint.
    model.E[0, 0] = np.nan
    w_before = digest(model.W)
    h_before = digest(state["H"].values())
    try:
        audit_write(model, state, batch[:1], local_cfg, name="nonfinite-input", seed=0)
    except AssertionError as error:
        require(str(error) == "NONFINITE_INPUT", "wrong nonfinite input exception")
    else:
        raise AssertionError("nonfinite input was accepted")
    require(w_before == digest(model.W) and h_before == digest(state["H"].values()), "nonfinite input mutated state")
    return dict(technical_pass=True, packed_fd_relative_errors=errors, cross_request_nll_change=coupling,
                native_kl_torch_probe=dict(nonzero_R=True, torch_version=torch.__version__,
                                          native_value_max_abs_error=native_value_error,
                                          native_gradient_max_abs_error=native_gradient_error,
                                          problem_vs_native_max_abs_error=problem_vs_native_error),
                boundary_invalid_example=invalid, boundary_valid_example=valid, restrictive_cap=capped,
                plateau=stalled, failed_line_search=linesearch_failed,
                cap_one_rejected=True, nonfinite_oracle_rejections=nonfinite,
                nonfinite_input_rejected_without_mutation=True,
                work=event_summary(model.events))


def zero_fixture(cfg, *, all_zero):
    model, state, batch, local_cfg = setup(71, d=cfg["d"], m=cfg["m"], tol=cfg["tol"], cap=cfg["cap"], objective=cfg["objective"])
    local_cfg["contexts"] *= 0.01
    h = model.forward(batch[0]["h0"][None])["hs"][-1][0]
    model.E[:] = 0.0
    model.E[0] = 24.0 * h / float(h @ h)
    pair = [copy.deepcopy(batch[0]), copy.deepcopy(batch[0])]
    pair[0].update(id="zero-request0", t_new=0)
    pair[1].update(id="zero-request1" if all_zero else "active-request1", t_new=0 if all_zero else 1)
    prob = make_problem(model, state, pair, local_cfg)
    require(prob.active.tolist() == ([False, False] if all_zero else [False, True]), "zero fixture screening changed")
    return model, state, pair, local_cfg


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=["native", "reference"], default="native")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--cap", type=int, default=400)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results, errors = [], []
    diagnostics = {}
    setup_events = []

    def run_case(name, seed, model, state, batch, cfg):
        try:
            rec = audit_write(model, state, batch, cfg, name=name, seed=seed)
            results.append(rec)
            print(f"{name} seed={seed}: {rec['solver']['status']}, calls={rec['solver']['calls']}, "
                  f"objective={rec['initial']['total']:.6g}->{rec['final']['total']:.6g}", flush=True)
            return rec
        except Exception as error:
            errors.append(dict(name=name, seed=seed, error=repr(error), traceback=traceback.format_exc()))
            print(f"{name} seed={seed}: TECHNICAL_FAIL {error}", flush=True)
            return None

    for seed in args.seeds:
        model, state, batch, cfg = setup(seed, d=args.d, m=args.m, tol=args.tol, cap=args.cap, objective=args.objective)
        setup_events.extend(model.events)
        model.events.clear()
        for r in range(2):
            run_case(f"bs1_independent_r{r}", seed, copy.deepcopy(model), copy.deepcopy(state), [batch[r]], cfg)
        run_case("bs2_coupled", seed, copy.deepcopy(model), copy.deepcopy(state), batch[:2], cfg)
        seq_model, seq_state = copy.deepcopy(model), copy.deepcopy(state)
        first = run_case("bs2_sequential_batch1", seed, seq_model, seq_state, batch[:2], cfg)
        if first is not None:
            second = run_case("bs2_sequential_batch2", seed, seq_model, seq_state, batch[2:], cfg)
            if second is not None:
                if first["identity"]["W_post"] != second["identity"]["W_entry"] or first["identity"]["H_post"] != second["identity"]["H_entry"]:
                    errors.append(dict(name="sequential-chain", seed=seed, error="W/H identity chain mismatch"))
        else:
            errors.append(dict(name="bs2_sequential_batch2", seed=seed, error="blocked by first batch technical failure"))
    base_model, base_state, base_batch, base_cfg = setup(0, d=args.d, m=args.m, tol=args.tol, cap=args.cap, objective=args.objective)
    cfg_data = dict(d=args.d, m=args.m, tol=args.tol, cap=args.cap, config=base_cfg, objective=args.objective)
    try:
        diagnostics = diagnostic_checks(cfg_data)
    except Exception as error:
        errors.append(dict(name="diagnostics", error=repr(error), traceback=traceback.format_exc()))
    for all_zero in (False, True):
        name = "all_zero_step" if all_zero else "mixed_zero_step"
        try:
            model, state, pair, cfg = zero_fixture(cfg_data, all_zero=all_zero)
            if not all_zero:
                prob = make_problem(model, state, pair, cfg)
                rng = np.random.default_rng(337)
                direction = rng.standard_normal(len(prob.blocks) * prob.d)
                direction /= np.linalg.norm(direction)
                c, rho = prob.block_params()
                x = prox_blocks(0.1 * direction, 0.0, c, rho, prob.d)
                _, G, _ = prob.smooth(prob.unpack(x))
                epsilon = 1e-5
                fd = (prob.smooth(prob.unpack(x + epsilon * direction))[0]
                      - prob.smooth(prob.unpack(x - epsilon * direction))[0]) / (2 * epsilon)
                analytic = float(prob.pack(G) @ direction)
                error = abs(fd - analytic) / max(1.0, abs(analytic))
                require(error < 1e-7, "mixed zero-step packed FD failed")
                diagnostics["mixed_zero_step_packed_fd_relative_error"] = error
            rec = run_case(name, 71, model, state, pair, cfg)
            if rec is not None:
                require(rec["zero_step"] == ([0, 1] if all_zero else [0]), "zero-step receipt mismatch")
                require(all(g["key_rank"] == 1 for g in rec["geometry"]), "duplicate-key fixture not rank deficient")
                require(rec["identity"]["H_entry"] != rec["identity"]["H_post"], "zero-step history was not appended")
                if all_zero:
                    require(rec["solver"]["status"] == "POLICY_ZERO_STEP" and rec["solver"]["calls"] == 0, "all-zero solver shortcut missing")
                    require(rec["identity"]["W_entry"] == rec["identity"]["W_post"], "all-zero altered weights")
        except Exception as error:
            errors.append(dict(name=name, error=repr(error), traceback=traceback.format_exc()))
    # Nonzero H is also exercised by every sequential second batch; this case
    # combines independent pre-existing history with duplicate active keys.
    rng = np.random.default_rng(919)
    for l in base_cfg["layers"]:
        history = rng.standard_normal((args.m, 3)) * 0.1
        base_state["H"][l] = history @ history.T
    duplicate = [copy.deepcopy(base_batch[0]), copy.deepcopy(base_batch[0])]
    duplicate[1]["id"] += "-duplicate"
    rec = run_case("duplicate_keys_nonzero_H", 0, base_model, base_state, duplicate, base_cfg)
    if rec is not None and not all(g["key_rank"] == 1 for g in rec["geometry"]):
        errors.append(dict(name="duplicate_keys_nonzero_H", error="expected rank-one key matrix"))
    primary = [r for r in results if r["name"].startswith("bs")]
    technical_pass = not errors
    optimization_pass = bool(primary) and all(r["optimization_converged"] for r in primary)
    fidelity = "TOY_NATIVE_KL_DIRECTION" if args.objective == "native" else "FAIL_KL_DIRECTION"
    receipt = dict(schema="jlz-cpu-smoke-v1", objective=args.objective, technical_pass=technical_pass,
                   primary_optimization_gate=optimization_pass,
                   objective_fidelity_to_native=fidelity, native_model_gates_passed=False,
                   limitation="Native adapter uses KL(current||entry); untouched reference uses KL(entry||current). No native-model J1/J3/J5 or performance claim.",
                   parameters=dict(seeds=args.seeds, d=args.d, m=args.m, V=24, total_layers=7,
                                   edit_layers=base_cfg["layers"], contexts=6, tol=args.tol, cap=args.cap,
                                   lambda_toy=float(args.m), lambda_native_design=15000,
                                   wd=0.5, clamp=0.75, kl_factor=0.0625, stop=0.05,
                                   zero_fixture="same threshold; constructed high-confidence output head and contexts scaled by 0.01"),
                   environment=dict(python=platform.python_version(), numpy=np.__version__,
                                    blas_threads={k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}),
                   source_sha256=source_manifest(),
                   source_delta="native_problem overrides terms/smooth KL only; native_writer function body is copied from reference with imports changed",
                   wall_seconds=time.perf_counter() - started,
                   primary_setup_work=event_summary(setup_events), diagnostics=diagnostics,
                   batches=results, errors=errors,
                   exit_semantics={"0": "native-adapter toy technical and primary optimization checks pass",
                                   "1": "technical failure", "2": "primary optimization gate not passed",
                                   "3": "reference technical and optimization pass but native KL direction differs"})
    path = args.output_dir / f"{args.objective}-smoke.json"
    path.write_text(json.dumps(safe_json(receipt), indent=2, allow_nan=False) + "\n")
    columns = ["name", "seed", "batch_size", "technical_pass", "status", "oracle_calls", "pipeline_forward_calls",
               "extra_audit_forward_calls", "normalized_residual", "initial_objective", "final_objective",
               "initial_nll", "final_nll", "final_kl", "final_decay", "active_blocks", "free_blocks", "write_wall_seconds"]
    with (args.output_dir / f"{args.objective}-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow(dict(name=r["name"], seed=r["seed"], batch_size=r["batch_size"], technical_pass=r["technical_pass"],
                                 status=r["solver"]["status"], oracle_calls=r["solver"]["calls"],
                                 pipeline_forward_calls=r["work"]["pipeline_total"]["forward_calls"],
                                 extra_audit_forward_calls=r["work"]["audit_extra"]["forward_calls"],
                                 normalized_residual=r["kkt"]["normalized_prox_residual"],
                                 initial_objective=r["initial"]["total"], final_objective=r["final"]["total"],
                                 initial_nll=r["initial"]["nll"], final_nll=r["final"]["nll"], final_kl=r["final"]["kl"],
                                 final_decay=r["final"]["decay"], active_blocks=r["active_blocks"], free_blocks=r["free_blocks"],
                                 write_wall_seconds=r["work"]["write_wall_seconds"]))
    print(f"technical_pass={technical_pass}, primary_optimization_gate={optimization_pass}, native_fidelity={fidelity}", flush=True)
    print(path.resolve(), flush=True)
    return 1 if not technical_pass else (2 if not optimization_pass else (3 if args.objective == "reference" else 0))


if __name__ == "__main__":
    raise SystemExit(main())
