"""BG-1 one-correction math and actual-FP32 finite candidate screen.

No model hooks, data sampler, teacher generation, history commit or submission
are performed here.  The caller freezes parent/K/P/M and owns all transactions.
Numerical choices are explicit and must be locked before model measurements.
"""
from dataclasses import asdict, dataclass
import math

import torch


class CorrectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class NumericalPolicy:
    epsilon: float
    ball_atol: float
    trust_atol: float
    screen_tolerance: float
    alpha_cap: float | None
    zeta: float = .25
    tau: float = .1
    mu: float = .01

    def __post_init__(self):
        if (self.zeta, self.tau, self.mu) != (.25, .1, .01):
            raise CorrectionError("BG1_MANUSCRIPT_CONSTANTS")
        values = (self.epsilon, self.ball_atol, self.trust_atol,
                  self.screen_tolerance)
        if not all(math.isfinite(x) for x in values) or self.epsilon <= 0:
            raise CorrectionError("NUMERICAL_POLICY_NONFINITE_OR_EPSILON")
        if min(self.ball_atol, self.trust_atol, self.screen_tolerance) < 0:
            raise CorrectionError("NEGATIVE_NUMERICAL_TOLERANCE")
        if self.alpha_cap is not None and (
                not math.isfinite(self.alpha_cap) or self.alpha_cap <= 0):
            raise CorrectionError("ALPHA_CAP")

    def receipt(self):
        return asdict(self)


def relaxed_barrier(h, tau=.1):
    """C2 relaxed log barrier, well-defined including negative slack.

    The inactive logarithmic branch is clamped at tau to avoid NaN autograd
    from evaluating log(negative) inside torch.where.  Active values unchanged.
    """
    if tau != .1:
        raise CorrectionError("BG1_TAU")
    if not torch.is_floating_point(h) or not bool(torch.isfinite(h).all()):
        raise CorrectionError("BARRIER_INPUT")
    upper = -torch.log(h.clamp_min(tau))
    lower = -math.log(tau) - (h-tau)/tau + (h-tau).square()/(2*tau*tau)
    return torch.where(h >= tau, upper, lower)


def full_control_slope(d64, ceiling, policy):
    """The ONE scalar d[mu*b_tau((b-D)/b)]/dD for the full D64.

    Use this frozen slope to weight every document's gradient.  Applying the
    barrier to separate microbatch means defines a different objective.
    """
    if not math.isfinite(d64) or not math.isfinite(ceiling) or ceiling <= 0:
        raise CorrectionError("CONTROL_MEAN_OR_CEILING")
    h = (ceiling-d64)/ceiling
    derivative_h = -1/h if h >= policy.tau else -1/policy.tau+(h-policy.tau)/(policy.tau**2)
    slope = -policy.mu*derivative_h/ceiling
    if not math.isfinite(slope):
        raise CorrectionError("CONTROL_SLOPE_NONFINITE")
    return slope


def weighted_chunk_mean(chunks, *, expected_count):
    """Reduce [(scalar mean, number of documents/requests)] with true mass."""
    if not chunks or type(expected_count) is not int or expected_count <= 0:
        raise CorrectionError("EMPTY_OR_INVALID_MASS")
    count = 0
    result = None
    for mean, n in chunks:
        if type(n) is not int or n <= 0 or mean.ndim != 0:
            raise CorrectionError("MICROBATCH_MASS_OR_MEAN")
        if not bool(torch.isfinite(mean)):
            raise CorrectionError("MICROBATCH_NONFINITE")
        term = mean * (n/expected_count)
        result = term if result is None else result + term
        count += n
    if count != expected_count:
        raise CorrectionError("MICROBATCH_TOTAL_MASS")
    return result


def accumulate_route_gradient(current_chunks, control_chunks, *, d64, ceiling,
                              policy, request_count=100, document_count=64):
    """Combine gradients of chunk MEANS; native target loss is not E_current.

    Each item is (detached gradient wrt the SAME residual, integer item count).
    The caller first measures all D64 with one frozen endpoint, then does F/B
    with this full mean's slope.  No hidden per-chunk barrier or token average.
    """
    slope = full_control_slope(d64, ceiling, policy)
    shape = None
    dtype = None
    device = None
    def reduce(chunks, total):
        nonlocal shape, dtype, device
        if total <= 0 or not chunks:
            raise CorrectionError("GRADIENT_EMPTY_MASS")
        out = None
        seen = 0
        for gradient, n in chunks:
            if type(n) is not int or n <= 0 or gradient.ndim != 2:
                raise CorrectionError("GRADIENT_MASS_OR_SCHEMA")
            if gradient.dtype != torch.float32 or not bool(torch.isfinite(gradient).all()):
                raise CorrectionError("GRADIENT_FP32_FINITE")
            if shape is None:
                shape, dtype, device = gradient.shape, gradient.dtype, gradient.device
            if gradient.shape != shape or gradient.dtype != dtype or gradient.device != device:
                raise CorrectionError("GRADIENT_ENDPOINT_SCHEMA")
            term = gradient.detach() * (n/total)
            out = term.clone() if out is None else out + term
            seen += n
        if seen != total:
            raise CorrectionError("GRADIENT_TOTAL_MASS")
        return out
    current = reduce(current_chunks, request_count)
    control = reduce(control_chunks, document_count)
    result = current + slope*control
    if not bool(torch.isfinite(result).all()):
        raise CorrectionError("ROUTE_GRADIENT_NONFINITE")
    return result, {"full_D64": d64, "control_slope": slope,
                    "request_count": request_count, "document_count": document_count,
                    "current_gradient_norm": float(current.double().norm()),
                    "control_gradient_norm": float(control.double().norm()),
                    "scaled_control_gradient_norm": float((slope*control).double().norm()),
                    "barrier_applied_to_microbatch_means": False}


def _matrix(name, value, shape=None):
    if value.dtype != torch.float32 or value.ndim != 2:
        raise CorrectionError(f"{name}_FP32_MATRIX")
    if shape is not None and value.shape != shape:
        raise CorrectionError(f"{name}_SHAPE")
    if not bool(torch.isfinite(value).all()):
        raise CorrectionError(f"{name}_NONFINITE")


def correct_once(proposal_z, canonical_h, anchors, radii, gradient, native_map,
                 *, policy):
    """One gradient step -> requestwise native ball -> executable trust.

    Tensors are [output, request]; anchors come from native compute_z and are
    NOT silently substituted with canonical_h.  Requested and realized FP32
    residual displacements are both checked and separately recorded.
    """
    _matrix("PROPOSAL", proposal_z)
    for name, value in (("H", canonical_h), ("ANCHOR", anchors), ("GRADIENT", gradient)):
        _matrix(name, value, proposal_z.shape)
        if value.device != proposal_z.device:
            raise CorrectionError("CORRECTION_DEVICE")
    if (radii.dtype != torch.float32 or radii.shape != (proposal_z.shape[1],)
            or radii.device != proposal_z.device or not bool(torch.isfinite(radii).all())
            or bool((radii < 0).any())):
        raise CorrectionError("NATIVE_RADIUS_SCHEMA")
    with torch.no_grad():
        native_distance = torch.linalg.vector_norm(proposal_z-anchors, dim=0)
        if bool((native_distance > radii+policy.ball_atol).any()):
            raise CorrectionError("NATIVE_PROPOSAL_OUTSIDE_OWN_BALL")
        rprop = proposal_z-canonical_h
        native_norm = float(native_map(rprop).double().norm())
        gradient_norm = float(gradient.double().norm())
        if not math.isfinite(native_norm):
            raise CorrectionError("NATIVE_ACTION_NONFINITE")
        info = {"numerical_policy": policy.receipt(), "native_action_norm": native_norm,
                "gradient_norm": gradient_norm, "proposal_ball_max_excess":
                float((native_distance-radii).max()), "extra_gradient_count": 1}
        if native_norm == 0. or gradient_norm == 0.:
            info.update(status="ZERO_NATIVE_OR_GRADIENT_RAW_ONLY", corrected_available=False,
                        alpha=0., trust_scale=0., actual_correction_action_norm=0.)
            return rprop.clone(), info
        map_gradient_norm = float(native_map(gradient).double().norm())
        alpha_uncapped = policy.zeta*native_norm/(map_gradient_norm+policy.epsilon)
        alpha = min(alpha_uncapped, policy.alpha_cap) if policy.alpha_cap is not None else alpha_uncapped
        if not math.isfinite(alpha):
            raise CorrectionError("ALPHA_NONFINITE")
        rtmp = rprop-alpha*gradient
        ztmp = canonical_h+rtmp
        relative = ztmp-anchors
        distances = torch.linalg.vector_norm(relative, dim=0)
        scale = torch.ones_like(radii)
        outside = distances > radii
        scale[outside] = radii[outside]/distances[outside]
        zprojected = anchors+relative*scale.unsqueeze(0)
        correction = zprojected-proposal_z
        proposed_action_norm = float(native_map(correction).double().norm())
        bound = policy.zeta*native_norm
        trust_scale = min(1., bound/proposed_action_norm) if proposed_action_norm else 1.
        correction = correction*trust_scale
        rcorr = rprop+correction
        realized = rcorr-rprop
        realized_norm = float(native_map(realized).double().norm())
        corrected_z = proposal_z+realized
        final_distance = torch.linalg.vector_norm(corrected_z-anchors, dim=0)
        reconstructed_z = canonical_h+rcorr
        reconstructed_distance = torch.linalg.vector_norm(reconstructed_z-anchors, dim=0)
        if (bool((final_distance > radii+policy.ball_atol).any()) or
                bool((reconstructed_distance > radii+policy.ball_atol).any())):
            raise CorrectionError("CORRECTED_TARGET_BALL_ROUNDING_EXCESS")
        if realized_norm > bound+policy.trust_atol:
            raise CorrectionError("EXECUTABLE_TRUST_ROUNDING_EXCESS")
        info.update(status="CORRECTED", corrected_available=True, alpha=alpha,
                    alpha_uncapped=alpha_uncapped, alpha_cap_hit=(alpha != alpha_uncapped),
                    map_gradient_norm=map_gradient_norm, projected_requests=int(outside.sum()),
                    projected_correction_action_norm=proposed_action_norm,
                    trust_scale=trust_scale, trust_bound=bound,
                    actual_correction_action_norm=realized_norm,
                    actual_target_ball_max_excess=float((final_distance-radii).max()),
                    reconstructed_H_plus_R_ball_max_excess=float((reconstructed_distance-radii).max()),
                    H_plus_R_vs_Zprop_plus_c_max_abs=float((reconstructed_z-corrected_z).abs().max()),
                    residual_add_rounding_max_abs=float((realized-correction).abs().max()))
        return rcorr, info


CANDIDATE_ORDER = ("RAW1", "CORR1", "CORR.5", "CORR.25")


def materialize_candidate(parent, *, candidate, raw_native_endpoint,
                          corrected_map_update=None):
    """RAW1 copies actual native; corrected fractional steps use CPU FP32.

    CORR eta1 first forms FP32(parent+RA).  eta .5/.25 then uses
    FP32(parent+eta*FP32(full_corrected_candidate-parent)), matching the
    existing actual-stored-candidate materialization convention.  This is
    not parent+eta*an ideal unrounded solve.  No model is mutated here.
    """
    _matrix("PARENT", parent)
    _matrix("RAW_NATIVE_ENDPOINT", raw_native_endpoint, parent.shape)
    if candidate not in CANDIDATE_ORDER:
        raise CorrectionError("UNAPPROVED_CANDIDATE")
    with torch.no_grad():
        entry = parent.detach().to(device="cpu", dtype=torch.float32)
        if candidate == "RAW1":
            endpoint = raw_native_endpoint.detach().cpu().clone()
            eta = 1.
        else:
            if corrected_map_update is None:
                raise CorrectionError("CORRECTED_MAP_UPDATE_REQUIRED")
            _matrix("CORRECTED_MAP_UPDATE", corrected_map_update, parent.shape)
            full = entry+corrected_map_update.detach().cpu()
            eta = {"CORR1": 1., "CORR.5": .5, "CORR.25": .25}[candidate]
            endpoint = full.clone() if eta == 1. else entry+eta*(full-entry)
        _matrix("ACTUAL_ENDPOINT", endpoint, parent.shape)
        actual_norm = float((endpoint.double()-entry.double()).norm())
    return endpoint, {"candidate": candidate, "eta": eta,
                      "actual_delta_norm": actual_norm,
                      "materialization_dtype": "float32",
                      "materialization_device": "cpu",
                      "endpoint_copy": candidate == "RAW1",
                      "fractional_path": "FP32(parent+eta*FP32(full_candidate-parent))"}


def choose_candidate(measurements, *, ceiling, policy, corrected_available):
    """Select only actual-forward screened candidates, else unchanged parent.

    Required rows: candidate, D64, Ecur, actual_delta_norm, actual_endpoint_sha.
    Exact scalar ties are resolved by actual delta norm then fixed menu order.
    Nonfinite values are typed technical errors, not performance rejections.
    """
    expected = CANDIDATE_ORDER if corrected_available else CANDIDATE_ORDER[:1]
    if [row.get("candidate") for row in measurements] != list(expected):
        raise CorrectionError("CANDIDATE_MENU_ORDER_OR_COUNT")
    if not math.isfinite(ceiling) or ceiling <= 0:
        raise CorrectionError("SCREEN_CEILING")
    annotated = []
    for index, row in enumerate(measurements):
        values = [row.get(key) for key in ("D64", "Ecur", "actual_delta_norm")]
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
            raise CorrectionError("CANDIDATE_NONFINITE_OR_MISSING")
        if row["actual_delta_norm"] < 0 or not row.get("actual_endpoint_sha"):
            raise CorrectionError("CANDIDATE_ACTUAL_ENDPOINT_IDENTITY")
        annotated.append(dict(row, order=index,
                              feasible=row["D64"] <= ceiling+policy.screen_tolerance))
    feasible = [row for row in annotated if row["feasible"]]
    selected = min(feasible, key=lambda row: (row["Ecur"], row["actual_delta_norm"], row["order"])) if feasible else None
    return {"selected": selected["candidate"] if selected else "PARENT",
            "write_accepted": selected is not None,
            "screen_ceiling": ceiling,
            "screen_tolerance": policy.screen_tolerance,
            "actual_screen_ceiling": ceiling+policy.screen_tolerance,
            "candidates": annotated, "native_fallback": False}
