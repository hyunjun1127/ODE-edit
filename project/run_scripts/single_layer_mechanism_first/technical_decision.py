"""Bounded actual decision T0; never a scientific full-bank/model-runner gate.

Four hash-selected references and the caller's four fixed Current requests are
the complete technical panel. No fitting, generation, submission, observer, or
stage continuation is performed here. A supplied retained native result lets
the native FP32 solve be replayed without an extra z calculation.

FD constants/grid are inherited verbatim from EN's technical numerical policy;
the objective is now an exposed fixed pair margin, not KL. Every perturbation
receipt is created before its physical forward. A small/unresolved signal is
NOT_ESTABLISHED, never a widened tolerance or selected favorable probe.
"""
from __future__ import annotations

from copy import deepcopy
import inspect
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from .basis import build_functional_basis
from .config import NUMERIC
from .current import fixed_panel
from .decision import DecisionOracle, EndpointBinding, risk_tolerance
from project.run_scripts.baseline_mechanism_first.fixtures import capture_rng, restore_rng
from project.run_scripts.single_layer_edit_preserving_correction import geometry
from project.run_scripts.single_layer_edit_preserving_correction.binding import score_rows
from project.run_scripts.single_layer_edit_preserving_correction.common import (
    digest, member, save_tensor, sha, tensor_sha, write,
)
from project.run_scripts.single_layer_edit_preserving_correction.runtime import invariant
from project.run_scripts.single_layer_edit_preserving_correction.technical import NUMERIC as EN_NUMERIC


FD_POLICY = {
    key: deepcopy(EN_NUMERIC[key]) for key in (
        "FD_scales", "FD_initial_native_relative", "FD_relative", "FD_signal_noise", "FD_adjacent",
        "FD_noise", "FD_selection", "FD_small_AD", "FD_zero_direction")
}
FD_POLICY.update(objective="fixed exposed reference pair margin at immutable WN",
    noise_objective_adaptation="range of three identical physical pair-margin evaluations, then eps64*max(1,abs(margin))",
    direction="one prospectively constructed Q-basis coefficient vector, unit Frobenius norm",
    thresholds_not_retuned=True, full_bank_validation=False)


class TechnicalDecisionError(RuntimeError):
    pass


def _require(value, message):
    if not value:
        raise TechnicalDecisionError(message)


def inherited_fd_grid(native_norm):
    """The existing twelve descending h values; no adaptive grid extension."""
    _require(math.isfinite(native_norm) and native_norm >= 0, "FINITE_NATIVE_NORM")
    return [FD_POLICY["FD_initial_native_relative"] * native_norm / (2 ** k)
            for k in range(FD_POLICY["FD_scales"])]


def classify_fd(points, *, AD, noise, direction_norm):
    """Mechanical original adjacent-window/signal rule, with full points kept."""
    _require(math.isfinite(AD) and math.isfinite(noise) and noise > 0 and
             math.isfinite(direction_norm) and direction_norm >= 0, "FINITE_FD_INPUTS")
    if direction_norm == 0:
        return dict(status="NO_DIRECTION", derivative_pass=False, applicable=False,
                    selected_window=None, full_grid=points)
    _require(len(points) == FD_POLICY["FD_scales"], "EXACT_TWELVE_FD_POINTS")
    classified = []
    for k, point in enumerate(points):
        p = deepcopy(point)
        _require(p["k"] == k and p["h"] > 0 and
                 math.isfinite(p["plus_margin"]) and math.isfinite(p["minus_margin"]), "FINITE_ORDERED_FD_POINT")
        if k:
            _require(p["h"] == points[0]["h"] / (2 ** k), "FD_GRID_DRIFT")
        fd = (p["plus_margin"] - p["minus_margin"]) / (2 * p["h"])
        signal = abs(p["plus_margin"] - p["minus_margin"]) / 2
        relative = abs(fd - AD) / abs(AD) if AD else None
        resolved = (signal > FD_POLICY["FD_signal_noise"] * noise and
                    p["plus_sha256"] != p["minus_sha256"] and
                    p["plus_nonzero"] > 0 and p["minus_nonzero"] > 0)
        p.update(AD=AD, FD=fd, signal=signal, noise=noise, absolute_error=abs(fd - AD),
                 relative_error=relative, resolved=resolved,
                 match=bool(resolved and relative is not None and relative <= FD_POLICY["FD_relative"]))
        classified.append(p)
    width = FD_POLICY["FD_adjacent"]
    windows = [list(range(i, i + width)) for i in range(len(points) - width + 1)
               if all(p["match"] for p in classified[i:i + width])]
    small = max(p["h"] * abs(AD) for p in points) <= FD_POLICY["FD_signal_noise"] * noise
    status = "PASS" if windows else "SMALL_AD_UNRESOLVED" if small else "NO_RESOLVED_ADJACENT_WINDOW"
    return dict(status=status, derivative_pass=bool(windows), applicable=True,
                selected_window=windows[0] if windows else None,
                AD=AD, noise=noise, full_grid=classified,
                numerical_validation="ESTABLISHED_ON_THIS_FIXED_PAIR" if windows else "NOT_ESTABLISHED")


def repeated_panel_metrics(panels):
    _require(len(panels) == 3 and all(p["indices"] == panels[0]["indices"] for p in panels), "THREE_IDENTICAL_PANELS")
    spread = 0.
    ids_exact = True
    for doc in range(4):
        rows = [p["rows"][doc] for p in panels]
        _require(all(r["positions"] == rows[0]["positions"] and r["labels"] == rows[0]["labels"] for r in rows),
                 "REPEAT_TOKEN_IDENTITY")
        for position in range(len(rows[0]["margins"])):
            values = [r["margins"][position] for r in rows]
            spread = max(spread, max(values) - min(values))
        ids_exact &= all(r["predictions"] == rows[0]["predictions"] for r in rows)
    risks = [p["panel_phi"] for p in panels]
    tau = risk_tolerance(risks[0], 0.)
    risk_spread = max(risks) - min(risks)
    return dict(margin_max_spread=spread, panel_risk_spread=risk_spread,
                panel_risk_values=risks, token_IDs_exact=ids_exact,
                margin_ceiling=NUMERIC["repeat_spread_fraction"] * NUMERIC["reference_deficit_allowance"],
                risk_ceiling=NUMERIC["repeat_spread_fraction"] * tau,
                pass_=bool(ids_exact and spread <= NUMERIC["repeat_spread_fraction"] * NUMERIC["reference_deficit_allowance"]
                    and risk_spread <= NUMERIC["repeat_spread_fraction"] * tau),
                scope="FOUR_REFERENCE_PANEL_ONLY; whole512 acceptance remains mandatory")


def cross_term_measurement(entry_displacement, update, keys):
    """Actual tensors, FP64 algebra, no new model or reference selection."""
    e, d, k = (torch.as_tensor(x).double() for x in (entry_displacement, update, keys))
    _require(e.shape == d.shape and e.shape[1] == k.shape[0], "CROSS_TERM_SHAPE")
    before, change = e @ k, d @ k
    after = (e + d) @ k
    direct = float(after.square().sum() - before.square().sum())
    cross = float(2 * (before * change).sum())
    update_energy = float(change.square().sum())
    expanded = cross + update_energy
    denominator = max(1., abs(direct), abs(cross), abs(update_energy), abs(expanded))
    relative = abs(direct - expanded) / denominator
    # Use the existing FP64 algebra/projector ceiling, not an outcome-tuned one.
    return dict(direct_energy_change=direct, cross_term=cross, update_energy=update_energy,
                expanded_energy_change=expanded, relative_arithmetic_gap=relative,
                ceiling=NUMERIC["projector_fp64"], pass_=relative <= NUMERIC["projector_fp64"])


def _native_replay(rt, native, WN, directory):
    if not isinstance(native, dict) or not native.get("captures"):
        return dict(status="NOT_ESTABLISHED", pass_=False,
                    reason="RETAINED_NATIVE_CAPTURE_PAYLOAD_REQUIRED_NO_REFIT_PERFORMED", new_z_calls=0)
    receipt, cap = native["receipt"], native["captures"]
    _require(receipt["entry_weight_sha256"] == tensor_sha(rt.W0) and
             receipt["history_sha256"] == tensor_sha(rt.M) and
             receipt["projector_sha256"] == tensor_sha(rt.P), "NATIVE_REPLAY_ENTRY_P_M_BINDING")
    _require(receipt["history_append"] == 0 and receipt["compute_z"] == 4 and
             receipt["solve"] == 1, "RETAINED_NATIVE_FIXED_FOUR_COUNTS")
    _require(len(cap["compute_ks"]) == len(cap["get_module_input_output_at_words"]) == 1 and
             len(cap["compute_z"]) == 4, "RETAINED_NATIVE_CAPTURE_COUNTS")
    device = rt.W.device
    with torch.no_grad():
        k = cap["compute_ks"][0].T.to(device)
        z = torch.stack(cap["compute_z"], dim=1).to(device)
        current = cap["get_module_input_output_at_words"][0].T.to(device)
        _require(k.dtype == z.dtype == current.dtype == torch.float32 and z.shape == current.shape,
                 "NATIVE_CAPTURE_DTYPE_SHAPE")
        residual = z - current
        _require(k.shape[1] % residual.shape[1] == 0, "NATIVE_CONTEXT_REPEAT_FACTOR")
        residual = residual.repeat_interleave(k.shape[1] // residual.shape[1], dim=1)
        p, m = rt.P[0].to(device), rt.M[0].to(device)
        lhs = p @ (k @ k.T + m) + rt.hp.L2 * torch.eye(k.shape[0], dtype=torch.float32, device=device)
        rhs = (p @ k) @ residual.T
        update = torch.linalg.solve(lhs, rhs).T
        replayed = (rt.W0.to(device) + update.float()).cpu()
    gap = replayed.double() - WN.double()
    exact = torch.equal(replayed, WN)
    evidence = dict(status="PASS" if exact else "NOT_ESTABLISHED_EXACT_NATIVE_REPLAY", pass_=exact,
        expected_weight_sha256=tensor_sha(WN), replay_weight_sha256=tensor_sha(replayed),
        weight_max_abs=float(gap.abs().max()), weight_l2=float(gap.norm()),
        actual_delta_norm=float((WN.double() - rt.W0.double()).norm()),
        native_source=receipt["source"], new_z_calls=0, diagnostic_dense_solves=1,
        history_appends=0, numerical_comparison="EXACT_FP32_REPLAY; NO_INVENTED_DELTA_TOLERANCE",
        weight_snapshot_saved=False, storage_policy="USER_NO_W_M_CHECKPOINT_SCALAR_HASH_EVIDENCE_ONLY")
    return evidence


def probe_evidence(native, probe, ideal, scale):
    """Describe the actual in-memory probe without persisting any W snapshot."""
    actual = probe.double() - native.double()
    return dict(native_sha256=tensor_sha(native), probe_sha256=tensor_sha(probe),
        ideal_delta_sha256=tensor_sha(ideal), actual_delta_sha256=tensor_sha(actual),
        scale=scale, ideal_delta_norm=float(ideal.norm()), actual_delta_norm=float(actual.norm()),
        rounding_delta_norm=float((actual - ideal).norm()),
        actual_changed_elements=int(torch.count_nonzero(actual)),
        science_candidate=False, weight_snapshot_saved=False,
        storage_policy="USER_NO_W_M_CHECKPOINT_SCALAR_HASH_EVIDENCE_ONLY")


def _allowed_and_space(rt, K, directory):
    descriptor = rt.lock["P_star_basis"]
    path = Path(descriptor["path"])
    _require(path.is_file() and path.stat().st_size == descriptor["bytes"] and sha(path) == descriptor["sha256"],
             "SEALED_ALLOWED_BASIS_MEMBER")
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    _require(payload.get("P_raw_sha") == geometry.matrix_sha256(rt.P[0]), "ALLOWED_BASIS_NATIVE_P_BINDING")
    basis = payload["basis"].numpy()
    allowed = geometry.RightSpace(basis, np.empty((basis.shape[1], 0)),
        "RESOLVED" if basis.shape[1] else "REPAIR_SPACE_EMPTY", payload["diagnostic"])
    space = geometry.edit_null_space(allowed, K)
    receipt = dict(allowed=allowed.receipt(), space=space.receipt(),
        allowed_member=deepcopy(descriptor), prior_allowed_numerical_evidence="REUSED_EXACT_P_BOUND_BASIS",
        current_new_space="EXISTING_NUMPY_SCIPY_TSQR_SVD_AND_FIXED_CUTOFF",
        full_dense_P_eigen_recheck=False)
    write(directory / "geometry.json", receipt)
    save_tensor(directory / "current-space.pt", dict(blocked=torch.from_numpy(space.blocked),
        status=space.status, shared_allowed_basis=descriptor, protected_K=K))
    return allowed, space, receipt


def _projector_probe(space):
    if space.status == "RANK_UNRESOLVED":
        return dict(status="RANK_UNRESOLVED", pass_=False, numerical_validation="NOT_ESTABLISHED")
    rng = np.random.default_rng(20260919)
    x, y = rng.standard_normal((4, space.basis.shape[0])), rng.standard_normal((4, space.basis.shape[0]))
    qx, qy = space.project(x), space.project(y)
    idem = float(np.linalg.norm(space.project(qx) - qx) / max(1., np.linalg.norm(qx)))
    symmetry = abs(float(np.sum(qx * y) - np.sum(x * qy))) / max(1., np.linalg.norm(x) * np.linalg.norm(y))
    return dict(status="PASS" if max(idem, symmetry) <= NUMERIC["projector_fp64"] else "FAIL",
        pass_=max(idem, symmetry) <= NUMERIC["projector_fp64"], idempotence=idem, symmetry=symmetry,
        ceiling=NUMERIC["projector_fp64"], probes=4, seed=20260919,
        scope="BOUNDED_FP64_OPERATOR_PROBES_PLUS_REUSED_ALLOWED_BASIS; NOT_FULL_OPERATOR_CERTIFICATE")


def _physical_margin(reference, index, weight, position, label, competitor):
    with torch.no_grad(), reference.physical_weight(weight):
        hidden = reference._physical_hidden(index)
        logits = reference._head(hidden, torch.tensor([position]))
        _require(bool(torch.isfinite(logits).all()), "NONFINITE_PHYSICAL_PAIR_LOGITS")
        return float(logits[0, label] - logits[0, competitor])


def _pair_check(rt, decision, reference, binding, index, direction, native_norm, output):
    output.mkdir(parents=True, exist_ok=False)
    row, activation, keys = decision._document(binding, index, derivative=True)
    cached = (activation.T @ keys).double().cpu()
    factor = activation.detach().cpu()
    position, label, competitor = row["worst_position"], row["worst_label"], row["worst_competitor"]
    factor_member = save_tensor(output / "activation-factor.pt", dict(A=factor,
        endpoint_identity=binding.identity, cache_index=index, exposed_row=row))
    del activation, keys
    with reference.physical_weight(binding.cpu, gradient=True) as leaf, torch.enable_grad():
        hidden = reference._physical_hidden(index)
        logits = reference._head(hidden, torch.tensor([position]))
        scalar = logits[0, label] - logits[0, competitor]
        direct, = torch.autograd.grad(scalar, leaf)
        direct = direct.detach().double().cpu()
        physical_margin = float(scalar.detach())
        alternatives = logits[0].detach().clone()
        alternatives[label] = -torch.inf
        competitor_tie = int((alternatives == alternatives.max()).sum()) > 1
    gradient_relative = float((cached - direct).norm() / direct.norm().clamp_min(1e-300))
    repeat = [_physical_margin(reference, index, binding.cpu, position, label, competitor) for _ in range(3)]
    noise = max(max(repeat) - min(repeat), np.finfo(np.float64).eps * max(1., abs(repeat[0])))
    direction = direction.double()
    norm = float(direction.norm())
    d = direction / norm if norm else direction
    AD = float((direct * d).sum())
    cached_J = float((factor.double() * (d @ reference.caches[index].valid_keys().double()).T).sum())
    scalar_error = abs(physical_margin - row["mu"])
    relative_J = abs(cached_J - AD) / abs(AD) if AD else None
    base = dict(index=index, position=position, label=label, competitor=competitor,
        competitor_tie_at_center=competitor_tie, label_pair_tie_at_center=row["mu"] == 0,
        fixed_pair_through_FD=True, changing_worst_branch_FD_claim=False,
        exposed_reference=row, factor=factor_member, physical_full_parameter_leaf=True,
        physical_bypasses_cached_prefix_and_optimized_endpoint_session=True,
        gradient_relative=gradient_relative, gradient_ceiling=NUMERIC["gradient_relative"],
        cached_scalar=row["mu"], physical_scalar=physical_margin, scalar_absolute_gap=scalar_error,
        cached_factor_coefficient_J=cached_J, physical_AD_coefficient=AD,
        coefficient_relative_gap=relative_J, physical_repeat=repeat, physical_noise=noise,
        all_valid_input_gradient_shape=list(factor.shape), direction_norm=norm)
    write(output / "AD-and-factor.json", base)
    if norm == 0 or native_norm == 0:
        fd = dict(status="NO_DIRECTION" if norm == 0 else "ZERO_NATIVE_FD_SCALE", derivative_pass=False,
                  applicable=False, selected_window=None, full_grid=[])
    else:
        points = []
        # ULP spacing is fixed at immutable WN and need not be recalculated for
        # every sign. It is a bounded T0 diagnostic, not a method criterion.
        spacing = (torch.nextafter(binding.cpu, torch.full_like(binding.cpu, float("inf"))) - binding.cpu).double().abs()
        for k, h in enumerate(inherited_fd_grid(native_norm)):
            values, perturbations = {}, {}
            for sign, multiplier in (("plus", 1.), ("minus", -1.)):
                weight = (binding.cpu.double() + multiplier * h * d).float()
                actual = weight.double() - binding.cpu.double()
                moved = actual != 0
                ulp = actual.abs() / spacing.clamp_min(torch.finfo(torch.float32).tiny)
                perturb = dict(k=k, h=h, sign=sign, weight_sha256=tensor_sha(weight),
                    actual_norm=float(actual.norm()), nonzero=int(torch.count_nonzero(actual)),
                    nominal_actual_error_norm=float((actual - multiplier * h * d).norm()),
                    max_ULP=float(ulp.max()), median_nonzero_ULP=float(ulp[moved].median()) if bool(moved.any()) else 0.,
                    nominal_AD=AD, actual_linear_delta=float((direct * actual).sum()))
                write(output / f"k{k:02d}-{sign}-perturbation.json", perturb)
                value = _physical_margin(reference, index, weight, position, label, competitor)
                write(output / f"k{k:02d}-{sign}-margin.json", dict(margin=value, fixed_exposed_pair=True))
                values[sign], perturbations[sign] = value, perturb
                del weight, actual, ulp
            points.append(dict(k=k, h=h, plus_margin=values["plus"], minus_margin=values["minus"],
                plus_sha256=perturbations["plus"]["weight_sha256"], minus_sha256=perturbations["minus"]["weight_sha256"],
                plus_nonzero=perturbations["plus"]["nonzero"], minus_nonzero=perturbations["minus"]["nonzero"],
                rounded_linear_FD=(perturbations["plus"]["actual_linear_delta"] -
                                   perturbations["minus"]["actual_linear_delta"]) / (2 * h)))
        fd = classify_fd(points, AD=AD, noise=noise, direction_norm=norm)
    write(output / "FD-result.json", fd)
    # Zero mathematical direction is classified, not claimed derivative PASS.
    # A nonzero unresolved/small AD signal remains a blocking unknown.
    passed = (gradient_relative <= NUMERIC["gradient_relative"] and scalar_error <= NUMERIC["noop_logits"]
              and (fd["derivative_pass"] or fd["status"] == "NO_DIRECTION")
              and (relative_J is not None and relative_J <= NUMERIC["gradient_relative"] or norm == 0))
    result = dict(**base, FD=fd, pass_=passed,
                  status="PASS" if passed else "NOT_ESTABLISHED", full512_pass=False)
    write(output / "result.json", result)
    return result


def _current_checks(rt, oracle, rows, WN, probe, ideal, keys, allowed, output):
    class Physical:
        def logits_at(self, index, weight, positions):
            return oracle.logits_at(index, weight, positions, route="physical")
    cached = score_rows(oracle, WN, rows)
    physical = score_rows(Physical(), WN, rows)
    rt.sync_oracles()
    nll = max(abs(cached[k]["nll"] - physical[k]["nll"]) for k in cached)
    ids = all(cached[k]["strict"] == physical[k]["strict"] and
              cached[k]["predictions"] == physical[k]["predictions"] for k in cached)
    physical_parity, stationarity, affine = [], [], []
    for index, cache in enumerate(oracle.caches):
        stationarity.append(oracle.key_stationarity(index, WN))
        physical_parity.append(oracle.compare_logits(index, WN, WN, left_route="cached", right_route="physical"))
        # Direct physical module capture, independent of suffix/prefix cache.
        captured = []
        with torch.no_grad(), oracle.physical_weight(WN):
            hook = oracle.decoder.layers[4].mlp.down_proj.register_forward_hook(
                lambda _module, _args, value: captured.append(value.detach().cpu().clone()))
            try:
                oracle._physical_hidden(index)
            finally:
                hook.remove()
        _require(len(captured) == 1, "PHYSICAL_L4_OUTPUT_CAPTURE_COUNT")
        k = cache.valid_keys().double()
        valid = cache.valid_positions()
        physical_H = captured[0][0, valid].T.double()
        # H0 is the actual FP32 module calculation, then exact actual FP32
        # weight displacement is applied in FP64 for the affine identity test.
        h0 = F.linear(cache.keys.to(oracle.device), rt.W0.to(oracle.device))[0, valid.to(oracle.device)].T.double().cpu()
        predicted = h0 + (WN.double() - rt.W0.double()) @ k
        error = predicted - physical_H
        normalized = torch.linalg.vector_norm(error, dim=0) / torch.linalg.vector_norm(physical_H, dim=0).clamp_min(1.)
        affine.append(dict(index=index, valid_tokens=k.shape[1], max_abs=float(error.abs().max()),
            max_token_normalized=float(normalized.max()), ceiling=NUMERIC["actual_DK"],
            pass_=float(normalized.max()) <= NUMERIC["actual_DK"],
            equation="actual_FP32_H0 + (FP64(WN)-FP64(W0)) K versus physical_FP32_HN"))
    rt.sync_oracles()
    inv = invariant(oracle, rows, cached, probe, WN, ideal, keys, allowed)
    probe_parity = [oracle.compare_logits(i, probe, probe, left_route="cached", right_route="physical")
                    for i in range(len(oracle.caches))]
    rt.sync_oracles()
    passed = (nll <= NUMERIC["noop_nll"] and ids and all(r["byte_equal"] for r in stationarity) and
              all(r["max_abs"] <= NUMERIC["noop_logits"] for r in physical_parity + probe_parity) and
              all(r["pass_"] for r in affine) and inv["pass"])
    result = dict(pass_=passed, status="PASS" if passed else "FAIL", fixed_current_requests=4,
        actual_TF_paths=len(oracle.caches), old_new_native_canonical_rows=len(rows),
        cached_rows=cached, physical_rows=physical, max_NLL_gap=nll, exact_target_IDs=ids,
        key_stationarity=stationarity, affine=affine, physical_parity=physical_parity,
        protected_invariant=inv, corrected_physical_parity=probe_parity,
        physical_route="Full decoder with actual selected Parameter copy/restore; optimized Current cache bypassed",
        all_valid_tokens_checked=True, full_vocabulary=True,
        probe_policy="first inherited FD scale in one frozen Q-basis direction; technical only, not method proposal")
    write(output / "current-checks.json", result)
    return result


def run_checks(rt, reference, records4, native_weight, output):
    """Return durable receipt with pass_/typed limits; never fit or submit.

    ``native_weight`` should be the full actual-write-native.pt dictionary.
    A bare weight has no independent writer evidence and returns a typed
    NOT_ESTABLISHED receipt rather than silently refitting or claiming PASS.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stage = "policy"
    before_rng = capture_rng()
    before_W, before_M = tensor_sha(rt.W), tensor_sha(rt.M)
    WN = native_weight["weight"] if isinstance(native_weight, dict) else native_weight
    expected = [rt.records[i]["case_id"] for i in fixed_panel(rt.records[:100], lambda r: r["case_id"])]
    _require(len(records4) == 4 and [r["case_id"] for r in records4] == expected, "FIXED_FOUR_CURRENT_HASH_PANEL")
    _require(WN.dtype == torch.float32 and tuple(WN.shape) == tuple(rt.W.shape) and bool(torch.isfinite(WN).all()),
             "FINITE_NATIVE_ENDPOINT")
    from project.run_scripts.single_layer_edit_preserving_correction import technical as inherited
    policy = dict(numerical=deepcopy(NUMERIC), inherited_FD=deepcopy(FD_POLICY),
        inherited_source=member(inherited.__file__), current_case_ids=expected,
        max_extra_native_fit=0, full512_acceptance="NOT_TESTED_HERE; MANDATORY_IN_SCIENTIFIC_RUNTIME",
        B2_nonempty_history="NOT_TESTED_BY_THIS_COLD_PANEL", posthoc_tolerance_adjustment=False,
        W_M_checkpoint_saved=False, new_weight_snapshots=False,
        retained_50974_input="READ_ONLY; NEVER_DELETED_OR_REWRITTEN")
    write(output / "technical-contract.json", policy)
    results = {}
    binding = None
    try:
        stage = "native_dense_replay"
        results[stage] = _native_replay(rt, native_weight, WN, output)
        write(output / "native-dense-replay.json", results[stage])
        if not results[stage]["pass_"]:
            receipt = dict(status="NOT_ESTABLISHED", pass_=False, results=results,
                blocking=[stage], seconds=time.monotonic() - started, new_z_calls=0)
            write(output / "checks.json", receipt)
            return receipt
        stage = "Current_Q_geometry"
        current, rows, K, meta = rt.protected_oracle(records4)
        write(output / "protected-inputs.json", meta)
        allowed, space, geometry_receipt = _allowed_and_space(rt, K, output)
        results[stage] = _projector_probe(space)
        write(output / "projector-probe.json", results[stage])
        if not results[stage]["pass_"]:
            receipt = dict(status="NOT_ESTABLISHED", pass_=False, results=results,
                blocking=[stage], seconds=time.monotonic() - started, new_z_calls=0)
            write(output / "checks.json", receipt)
            return receipt
        decision = DecisionOracle(reference)
        indices = fixed_panel(decision.capsules[:512], lambda c: c["source_row_id"])
        write(output / "reference-panel.json", dict(indices=indices,
            source_row_ids=[decision.capsules[i]["source_row_id"] for i in indices],
            selection="fixed SHA256 identity only; no damage/gradient/performance selection"))
        binding = EndpointBinding(WN, reference.device, endpoint_id="T0_FIXED4_NATIVE",
                                  source_identity=dict(execution=rt.lock["execution"], panel=expected))
        stage = "reference_repeat"
        repeats = [decision.check_panel(binding, indices) for _ in range(3)]
        results[stage] = repeated_panel_metrics(repeats)
        write(output / "reference-repeats.json", dict(panels=repeats, **results[stage]))
        stage = "panel_gradient_basis"
        panel_G = torch.zeros(tuple(WN.shape), dtype=torch.float64, device=reference.device)
        factor_count = 0
        for index in indices:
            row, a, keys = decision._document(binding, index, derivative=True)
            panel_G.add_((a.T @ keys).double(), alpha=-2 * max(0., -row["mu"]) / 4)
            factor_count += 1
            del a, keys
        G = panel_G.cpu()
        del panel_G
        H = -space.project(G)
        projected_norm = float(H.norm())
        direction_origin = "FIXED_PANEL_RISK_GRADIENT; NOT_FULL512_METHOD_GRADIENT"
        if projected_norm == 0 and space.dimension:
            # Predeclared numerical-path probe only; not a replacement method
            # direction, science policy or successful correction claim.
            generator = torch.Generator().manual_seed(20260919)
            left = torch.randn((WN.shape[0], 1), dtype=torch.float64, generator=generator)
            right = torch.randn((1, WN.shape[1]), dtype=torch.float64, generator=generator)
            H = space.project(left @ right)
            direction_origin = "ZERO_PANEL_RISK_FIXED_SEED_RANK_ONE_TECHNICAL_Q_PROBE"
        basis = build_functional_basis(H, project=space.project)
        if basis.rank:
            direction = torch.from_numpy(sum(basis.directions) / math.sqrt(basis.rank)).double()
            direction /= direction.norm()
        else:
            direction = torch.zeros_like(WN, dtype=torch.float64)
        native_norm = float((WN.double() - rt.W0.double()).norm())
        scales = inherited_fd_grid(native_norm)
        write(output / "technical-direction.json", dict(direction_sha256=tensor_sha(direction),
            direction_norm=float(direction.norm()), shape=list(direction.shape), source=direction_origin,
            basis=basis.receipt, native_norm=native_norm, FD_scales=scales,
            weight_snapshot_saved=False))
        results[stage] = dict(pass_=True, basis=basis.receipt, source=direction_origin,
            reference_factor_backwards=factor_count, actual_full512_gradient=False,
            directional_rank=basis.rank, extra_z_calls=0)
        write(output / "basis.json", results[stage])
        stage = "pair_factor_AD_FD"
        pair_results = [_pair_check(rt, decision, reference, binding, index, direction, native_norm,
                                   output / f"reference-{index:03d}") for index in indices]
        results[stage] = dict(pass_=all(r["pass_"] for r in pair_results), rows=pair_results,
                             documents=4, full512_pass=False)
        rt.sync_oracles()
        stage = "Current_affine_physical_invariant"
        ideal = direction * scales[0]
        probe = (WN.double() + ideal).float()
        write(output / "technical-probe.json", probe_evidence(WN, probe, ideal, scales[0]))
        results[stage] = _current_checks(rt, current, rows, WN, probe, ideal, K, allowed, output)
        stage = "cross_term_STEP_CUM"
        E = WN.double() - rt.W0.double()
        cross = [dict(index=i, **cross_term_measurement(E, probe.double() - WN.double(), reference.caches[i].valid_keys()))
                 for i in indices]
        # Cold entry E_entry=0: STEP and CUM feed exactly the same displacement
        # into the same covariance action; no duplicated expensive contraction.
        step = WN.double() - rt.W0.double()
        cumulative = WN.double() - rt.W0.double()
        results[stage] = dict(pass_=all(r["pass_"] for r in cross) and torch.equal(step, cumulative),
            rows=cross, STEP_CUM_input_exact=torch.equal(step, cumulative),
            STEP_CUM_covariance_action="IDENTICAL_INPUT_AND_SAME_DETERMINISTIC_OPERATOR; NOT_EXECUTED_TWICE",
            actual_zero_entry=True, full512_covariance="NOT_ESTABLISHED_BY_FOUR_REFERENCE_PANEL")
        write(output / "cross-term.json", results[stage])
        stage = "restore"
        rt.guard()
        results[stage] = dict(pass_=before_W == tensor_sha(rt.W) and before_M == tensor_sha(rt.M) and
            digest(before_rng) == digest(capture_rng()), weight_exact=before_W == tensor_sha(rt.W),
            memory_exact=before_M == tensor_sha(rt.M), RNG_exact=digest(before_rng) == digest(capture_rng()),
            history_appends=0, independent_GPU_continuation="NOT_TESTED")
        blocking = [name for name, result in results.items() if not result["pass_"]]
        receipt = dict(status="PASS" if not blocking else "NOT_ESTABLISHED", pass_=not blocking,
            scope="FIXED4_REFERENCE_AND_FIXED4_CURRENT_T0_NOT_FULL512_SCIENCE", results=results,
            blocking=blocking, new_z_calls=0, scientific_batches_executed=0,
            seconds=time.monotonic() - started, full512_coverage="REQUIRED_BY_B1_RUNTIME_NOT_TESTED_HERE",
            past="COLD_EMPTY_ACTUAL; NONEMPTY_CPU_FIXTURES_SEPARATE", numeric_policy=policy)
        write(output / "checks.json", receipt)
        return receipt
    except BaseException as exc:
        write(output / "failure.json", dict(status="TECHNICAL_FAILURE", stage=stage, error=repr(exc),
            traceback=traceback.format_exc(), completed_results=results, seconds=time.monotonic() - started,
            new_z_calls=0, thresholds_unchanged=True, full512_pass=False))
        raise
    finally:
        restore_rng(before_rng)
        if binding is not None:
            binding.close()
