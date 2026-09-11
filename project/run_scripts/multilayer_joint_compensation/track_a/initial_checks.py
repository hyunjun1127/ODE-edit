"""Bounded actual-model correctness checks; no model loading or submission.

The caller supplies first-request Current contexts, the pinned We teacher panel,
and a frozen JointView. One nonzero joint state is compared against temporary
physical materialization. JVP finite differences are validation-only. The GGN
is the shared functional action, never a finite-difference/full Hessian.
"""
from __future__ import annotations

import math
import time
from typing import Any, Sequence

import torch

from ..contracts import tensor_sha, digest
from ..evaluation import materialized
from ..functional import FunctionalPanel, context_nll
from ..linear_solve import dot, norm, add
from ..observations import JointView, PredictionBatch


SEED = 20260911
FD_EPSILONS = (2. ** -7, 2. ** -8, 2. ** -9)
FP32_ROUNDOFF = 64 * torch.finfo(torch.float32).eps
OPERATOR_TOLERANCE = max(FP32_ROUNDOFF, 1e-4)


class InitialCheckBoundary(RuntimeError):
    """Full diagnostics survive a technical failure; caller persists receipt."""

    def __init__(self, receipt: dict[str, Any]):
        self.receipt = receipt
        super().__init__("INITIAL_TECHNICAL_UNRESOLVED: " + "; ".join(receipt.get("failures", [])))


def _snapshot(view: JointView) -> list[dict[str, Any]]:
    return [dict(name=name, pointer=p.data_ptr(), version=p._version,
                 shape=list(p.shape), dtype=str(p.dtype), device=str(p.device), sha256=tensor_sha(p))
            for name, p in view.model.named_parameters()]


def _norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach(), dtype=torch.float64))


def _comparison(reference: torch.Tensor, observed: torch.Tensor) -> dict[str, Any]:
    if reference.shape != observed.shape:
        return {"finite": False, "shape_match": False}
    if not bool(torch.isfinite(reference).all() and torch.isfinite(observed).all()):
        return {"finite": False, "shape_match": True}
    a, b = reference.detach().double().reshape(-1), observed.detach().double().reshape(-1)
    an, bn = _norm(a), _norm(b)
    error = _norm(b - a)
    cosine = float(torch.dot(a, b) / (an * bn)) if an and bn else None
    return dict(finite=True, shape_match=True, reference_l2=an, observed_l2=bn,
                max_abs_error=float((b - a).abs().max()) if a.numel() else 0.,
                absolute_l2_error=error, relative_l2_error=error / an if an else None,
                cosine=cosine, byte_equal=bool(torch.equal(reference, observed)))


def _directions(view: JointView, seed: int, projectors: Sequence | None) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = []
    if projectors is not None and len(projectors) != len(view.entry):
        raise ValueError("PROJECTOR_SUPPORT_MISMATCH")
    for index, weight in enumerate(view.entry):
        # CPU RNG yields the same deterministic candidate on both hosts. It is
        # normalized after the complete P* action, not after rank truncation.
        direction = torch.randn(weight.shape, generator=generator, dtype=torch.float32)
        if projectors is not None:
            direction = projectors[index].right(direction.double()).float()
        size, target = _norm(direction), _norm(weight)
        if size:
            direction.mul_(target / size)
        result.append(direction.to(weight.device))
    return tuple(result)


def _direct_logits(view: JointView, batch: PredictionBatch) -> torch.Tensor:
    """Bypass JointView only inside the explicit temporary observation state."""
    view.ledger.add("initial_direct_model_forward_invocations")
    view.ledger.add("initial_direct_forward_nonpadding_tokens", int(batch.attention_mask.sum()))
    output = view.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    logits = output.logits if hasattr(output, "logits") else output
    return logits[batch.positions]


def run_initial_checks(view: JointView, prediction_batches: Sequence[PredictionBatch],
                       ggn_panel: FunctionalPanel, *, projectors: Sequence | None = None) -> dict[str, Any]:
    """Check existing model without editing targets/history or planning A0.

    All supplied prediction batches must be the first Current request's exact
    contexts. The caller supplies a bounded, globally normalized GGN panel
    (e.g. its first context), with We teachers already bound by identity.
    Signal rule and tolerances are outcome-independent constants above. At
    least one adjacent epsilon pair must have sufficient signal and satisfy
    JVP/FD cosine>=.99 and relativeL2<=.05 at both epsilons; all pairs and all
    errors are retained. No epsilon search or tolerance relaxation occurs.
    """
    receipt: dict[str, Any] = dict(
        schema="multilayer-joint-initial-checks-v1", status="RUNNING", seed=SEED,
        fd_epsilons=list(FD_EPSILONS), fd_cosine_min=.99, fd_relative_l2_max=.05,
        sufficient_signal_rule="||f_plus-f_minus||2 > 64 eps32 (||f_plus||2+||f_minus||2)",
        primal_relative_roundoff=FP32_ROUNDOFF, ggn_relative_tolerance=OPERATOR_TOLERANCE,
        ggn_tolerance_rule="max(64 eps_FP32, pinned_PCG_relative_target_1e-4)",
        failures=[], virtual_materialized=[], directional_checks=[], teacher_checks=[], ggn={},
        scope="first Current request contexts plus bounded GGN panel; observation-only",
        full_model_clone_count=0, model_load_count=0, native_z_calls=0, history_appends=0,
        fd_production_calls=0, full_hessian_count=0,
    )
    started = time.perf_counter()
    before = _snapshot(view)
    before_panel = dict(ggn_panel.counts)
    receipt["full_parameter_count"] = len(before)
    receipt["full_parameter_entry_root"] = digest(before)
    receipt["selected_entry_sha256"] = list(view.entry_sha)
    phase = "input"
    try:
        batches = tuple(prediction_batches)
        if not batches or any(b.view is not view for b in batches):
            raise ValueError("INITIAL_PREDICTION_BATCH_BINDING")
        if not ggn_panel.batches:
            raise ValueError("INITIAL_GGN_PANEL_EMPTY")
        if view.model.training:
            raise ValueError("INITIAL_MODEL_MUST_BE_EVAL")
        receipt["context_identities"] = [str(row["identity"]) for batch in batches for row in batch.rows]
        ordinals = {row["ordinal"] for batch in batches for row in batch.rows if "ordinal" in row}
        if len(ordinals) > 1:
            raise ValueError("INITIAL_SCOPE_MORE_THAN_ONE_REQUEST")
        receipt["request_ordinals"] = sorted(ordinals)
        receipt["ggn_panel_identities"] = [batch.identity for batch in ggn_panel.batches]
        receipt["ggn_role"] = ggn_panel.role
        view.assert_live(bytes_check=True)
        phase = "We_teacher"
        with torch.no_grad():
            for index, batch in enumerate(ggn_panel.batches):
                entry_logp = ggn_panel._logits(view.entry, batch).log_softmax(-1)
                metrics = _comparison(entry_logp, batch.teacher_logp)
                nll_metrics = _comparison(context_nll(entry_logp, batch), batch.reference_nll)
                logp_limit = FP32_ROUNDOFF * max(float(entry_logp.abs().max()), 1.)
                nll_limit = FP32_ROUNDOFF * max(float(batch.reference_nll.abs().max()), 1.)
                passed = (metrics["finite"] and nll_metrics["finite"]
                          and metrics.get("max_abs_error", math.inf) <= logp_limit
                          and nll_metrics.get("max_abs_error", math.inf) <= nll_limit)
                receipt["teacher_checks"].append(dict(batch=index, passed=passed,
                    logp=metrics, reference_nll=nll_metrics, logp_limit=logp_limit, nll_limit=nll_limit))
                if not passed:
                    receipt["failures"].append(f"WE_TEACHER_MISMATCH:{index}")
        direction = _directions(view, SEED, projectors)
        other = _directions(view, SEED + 1, projectors)
        receipt["direction_layers"] = [dict(name=name, weight_frobenius=_norm(w),
                                            direction_frobenius=_norm(d), sha256=tensor_sha(d))
                                       for name, w, d in zip(view.names, view.entry, direction)]
        state = tuple(w + FD_EPSILONS[1] * d for w, d in zip(view.entry, direction))
        receipt["nonzero_probe_state_scale"] = FD_EPSILONS[1]
        receipt["probe_weight_sha256"] = [tensor_sha(w) for w in state]
        phase = "virtual_materialized"
        with torch.no_grad():
            virtual = [batch.logits(state).detach().clone() for batch in batches]
            with materialized(view.model, view.names, state, view.ledger, purpose="initial_parity"):
                direct = [_direct_logits(view, batch).detach() for batch in batches]
            for batch_index, (expected, observed) in enumerate(zip(virtual, direct)):
                metrics = _comparison(expected, observed)
                limit = FP32_ROUNDOFF * max(float(expected.abs().max()), 1.)
                passed = metrics["finite"] and metrics.get("max_abs_error", math.inf) <= limit
                receipt["virtual_materialized"].append(dict(batch=batch_index, limit=limit, passed=passed, **metrics))
                if not passed:
                    receipt["failures"].append(f"VIRTUAL_MATERIALIZED_LOGITS:batch{batch_index}")
        view.assert_live(bytes_check=True)

        phase = "jvp_fd"
        isolated = [tuple(d if index == layer else torch.zeros_like(d) for index, d in enumerate(direction))
                    for layer in range(len(direction))]
        named_directions = [("joint", direction)] + [(f"layer:{name}", d) for name, d in zip(view.names, isolated)]
        for name, tangent in named_directions:
            for batch_index, batch in enumerate(batches):
                _, jvp = torch.func.jvp(lambda *ws: batch.logits(ws), state, tangent)
                view.ledger.add("initial_validation_jvp_calls")
                rows = []
                for epsilon in FD_EPSILONS:
                    with torch.no_grad():
                        plus = batch.logits(tuple(w + epsilon * d for w, d in zip(state, tangent)))
                        minus = batch.logits(tuple(w - epsilon * d for w, d in zip(state, tangent)))
                        fd = (plus.double() - minus.double()) / (2 * epsilon)
                    metrics = _comparison(jvp, fd)
                    signal = _norm(plus.double() - minus.double())
                    noise = FP32_ROUNDOFF * (_norm(plus) + _norm(minus))
                    sufficient = math.isfinite(signal) and math.isfinite(noise) and signal > noise
                    matches = (metrics["finite"] and metrics.get("cosine") is not None
                               and metrics["cosine"] >= .99
                               and metrics["relative_l2_error"] <= .05)
                    rows.append(dict(epsilon=epsilon, signal_l2=signal, fp32_noise_envelope=noise,
                                     sufficient_signal=sufficient, matches_jvp=bool(matches), **metrics))
                    view.ledger.add("initial_validation_fd_forward_calls", 2)
                adjacent = []
                for index in range(2):
                    eligible = rows[index]["sufficient_signal"] and rows[index + 1]["sufficient_signal"]
                    passed = eligible and rows[index]["matches_jvp"] and rows[index + 1]["matches_jvp"]
                    adjacent.append(dict(indices=[index, index + 1], sufficient=eligible, passed=passed))
                passed = any(pair["passed"] for pair in adjacent)
                receipt["directional_checks"].append(dict(direction=name, batch=batch_index,
                                                          epsilons=rows, adjacent_pairs=adjacent, passed=passed))
                if not passed:
                    kind = "JVP_NUMERICALLY_UNRESOLVED" if not any(pair["sufficient"] for pair in adjacent) else "JVP_FD_MISMATCH"
                    receipt["failures"].append(f"{kind}:{name}:batch{batch_index}")

        phase = "ggn"
        gv = ggn_panel.ggn(state, direction)
        gu = ggn_panel.ggn(state, other)
        vu, uv = dot(direction, gu), dot(other, gv)
        sym_scale = max(norm(direction) * norm(gu), norm(other) * norm(gv))
        symmetry_pass = abs(vu - uv) <= OPERATOR_TOLERANCE * sym_scale
        psd_rows = []
        for name, vector, action in (("v", direction, gv), ("u", other, gu)):
            energy = dot(vector, action)
            scale = norm(vector) * norm(action)
            passed = energy >= -OPERATOR_TOLERANCE * scale
            psd_rows.append(dict(direction=name, quadratic_form=energy, scale=scale, passed=passed))
            if not passed:
                receipt["failures"].append(f"GGN_NEGATIVE_CURVATURE:{name}")
        if not symmetry_pass:
            receipt["failures"].append("GGN_NONSYMMETRIC")
        per_layer = [ggn_panel.ggn(state, vector) for vector in isolated]
        summed = tuple(sum(action[index] for action in per_layer) for index in range(len(state)))
        linear_error = norm(add(summed, gv, -1.))
        linear_scale = max(norm(summed), norm(gv))
        linear_pass = linear_error <= OPERATOR_TOLERANCE * linear_scale
        if not linear_pass:
            receipt["failures"].append("GGN_JOINT_LINEARITY")
        cross = []
        for source, action in enumerate(per_layer):
            for target, block in enumerate(action):
                if source != target:
                    cross.append(dict(source_layer=view.names[source], target_layer=view.names[target],
                                      cross_action_frobenius=_norm(block),
                                      signed_bilinear=dot((direction[target],), (block,)),
                                      zero_is_failure=False))
        receipt["ggn"] = dict(symmetry=dict(v_Gu=vu, u_Gv=uv, absolute_error=abs(vu - uv),
                                           scale=sym_scale, passed=symmetry_pass),
                              psd=psd_rows, joint_linearity=dict(error=linear_error, scale=linear_scale,
                                                               passed=linear_pass), cross_actions=cross,
                              operator="shared FunctionalPanel.ggn; full-sequence JVP/VJP")
    except Exception as error:
        receipt["failures"].append(f"EXECUTION_EXCEPTION:{phase}:{type(error).__name__}")
        receipt["original_exception"] = dict(type=type(error).__name__, message=str(error), phase=phase)
    finally:
        after = _snapshot(view)
        receipt["full_parameter_exit_root"] = digest(after)
        receipt["full_live_pointer_version_bytes_restored"] = before == after
        changed = [item["name"] for item, final in zip(before, after) if item != final]
        receipt["changed_parameters"] = changed
        if before != after:
            receipt["failures"].append("FULL_LIVE_STATE_CHANGED")
        receipt["ggn_compute_counts"] = {key: ggn_panel.counts[key] - before_panel[key] for key in before_panel}
        receipt["wall_seconds"] = time.perf_counter() - started
        receipt["timer_scope"] = "caller synchronizes CUDA around this API for allocated wall-time accounting"
    receipt["status"] = "INITIAL_CORRECTNESS_PASS" if not receipt["failures"] else "INITIAL_TECHNICAL_UNRESOLVED"
    # Even nonfinite failures must be persistable by canonical JSON. Preserve
    # their exact nonfinite class as an explicit string, never a fabricated 0.
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return f"NONFINITE:{value!r}"
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item) for item in value]
        return value
    receipt = safe(receipt)
    if receipt["failures"]:
        raise InitialCheckBoundary(receipt)
    return receipt
