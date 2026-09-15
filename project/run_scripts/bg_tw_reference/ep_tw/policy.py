"""EP-TW-1 v3 residual policy; no model, history, data, or scheduler access.

All executable tensors are FP32. Scalar diagnostics use FP64 reductions, not
a different model/kernel policy. ``Zp``, anchors, and residuals are [out, B],
``A`` is [B, in], and stored weights are [out, in]. The caller supplies the
actual native proposal; this module never reconstructs RAW from its map.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import torch


POLICY = "EP-TW-1"
CANDIDATE_MENU = (("RAW", 0.0), ("C1", 1.0), ("C05", 0.5), ("C025", 0.25))
_EPS32 = torch.finfo(torch.float32).eps


class PolicyError(RuntimeError):
    """Input/state/native failure, never a successful RAW fallback."""


@dataclass(frozen=True)
class NumericalPolicy:
    """Pre-execution numerical choices; not tuned using candidate quality.

    Defaults are explicit starting choices requiring a model-level numerical
    receipt. Relative representational checks allow eight FP32 ulps; there is
    *no* positive E allowance. D tie resolution is not a quality allowance.
    alpha_cap=1 limits the unscaled residual displacement; epsilon only guards
    division. Exact zeros, rather than a fitted gradient floor, are skipped.
    """

    alpha_cap: float | None = 1.0
    epsilon_num: float = 1e-12
    zeta: float = 0.25
    zero_norm_atol: float = 0.0
    ball_atol: float = 0.0
    ball_rtol: float = 8 * _EPS32
    trust_atol: float = 0.0
    trust_rtol: float = 8 * _EPS32
    d_tie_atol: float = 0.0
    d_tie_rtol: float = 8 * _EPS32
    max_identical_rechecks: int = 0
    alpha_cap_mode: str = "bounded"

    def __post_init__(self):
        values = asdict(self)
        mode = values.pop("alpha_cap_mode")
        cap = values.pop("alpha_cap")
        if mode not in ("bounded", "disabled"):
            raise PolicyError("ALPHA_CAP_MODE")
        if mode == "disabled":
            if cap is not None:
                raise PolicyError("DISABLED_ALPHA_CAP_MUST_BE_NULL")
        elif (isinstance(cap, bool) or not isinstance(cap, (int, float))
              or not math.isfinite(cap) or cap <= 0):
            raise PolicyError("BOUNDED_ALPHA_CAP_POSITIVE_FINITE")
        if not all(math.isfinite(v) for v in values.values()):
            raise PolicyError("NUMERICAL_POLICY_NONFINITE")
        if self.zeta != 0.25 or self.epsilon_num <= 0:
            raise PolicyError("POLICY_ZETA_OR_POSITIVE_NUMERICS")
        if any(v < 0 for v in values.values()):
            raise PolicyError("NEGATIVE_NUMERICAL_POLICY")
        if type(self.max_identical_rechecks) is not int or self.max_identical_rechecks != 0:
            raise PolicyError("THIS_IMPLEMENTATION_HAS_NO_RECHECK_SEARCH")

    def receipt(self):
        result = asdict(self)
        result.update(
            schema="EP_TW1_NUMERICAL_POLICY_V1", quality_positive_allowance=None,
            numerical_constants_are_not_performance_tuned=True,
            model_numeric_resolution_status="REQUIRES_SEPARATE_TECHNICAL_RECEIPT",
            tensor_arithmetic="FP32", diagnostic_scalar_reductions="FP64",
            rationale={
                "alpha_cap": "explicit bounded positive finite cap or disabled/null; fixed before outcomes",
                "epsilon_num": "division guard; exact-zero direction/native action skips",
                "ball_trust_rtol": "eight FP32 eps relative representational envelope",
                "d_tie": "conservative RAW-priority tie only; never relaxes E<=Ep",
                "zero_norm_atol": "exact zero, no fitted small-gradient threshold",
            },
        )
        return result


def validate_dispatch(dispatch: Mapping[str, Any]):
    """Fail closed against accidentally iterating historical policy cells."""
    expected = {
        "policy": POLICY, "new_scientific_policies": [POLICY],
        "new_scientific_chains": 1, "initial_model": "pre-edit W0",
        "layer": 4, "native_l2": 1, "batch_size": 100,
        "batches_per_chain": 10, "unique_requests": 1000,
        "ordinals": [0, 1000], "new_editing_logical_batches": 10,
        "baseline_editing_reruns_allowed": False,
        "N4_calibration_required": False, "preservation_budget": None,
        "tv_allowance": None, "base_penalty_weight": None, "log_barrier": False,
        "candidate_ids": [x[0] for x in CANDIDATE_MENU],
        "candidate_betas": [x[1] for x in CANDIDATE_MENU],
        "candidate_scale": "correction_only", "fallback": "actual_RAW_native_not_parent",
    }
    for name, value in expected.items():
        if name not in dispatch or dispatch[name] != value:
            raise PolicyError("DISPATCH_SCOPE_" + name)
    if dispatch.get("gradient_sweeps") != {"current": 1, "S64": 1}:
        raise PolicyError("DISPATCH_GRADIENT_SWEEPS")
    return {"policy": POLICY, "new_scientific_chains": 1,
            "N4_calibration_required": False, "scope_checked": list(expected)}


def _matrix(name, tensor, *, shape=None, device=None, finite=True):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tensor.ndim != 2:
        raise PolicyError(name + "_FP32_MATRIX")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise PolicyError(name + "_SHAPE")
    if device is not None and tensor.device != device:
        raise PolicyError(name + "_DEVICE")
    if finite and not bool(torch.isfinite(tensor).all()):
        raise PolicyError(name + "_NONFINITE")


def _norm(tensor):
    return float(torch.linalg.vector_norm(tensor.detach().double()))


def _dot(first, second):
    return float((first.detach().double() * second.detach().double()).sum())


def _json_safe(value):
    """Invalid observations retain typed reasons without emitting JSON NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"not_finite": repr(value)}
    if isinstance(value, dict):
        return {name: _json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def tensor_sha256(tensor):
    """Dtype+shape header bridged to exact contiguous raw tensor bytes."""
    data = tensor.detach().contiguous().cpu()
    header = json.dumps({"dtype": str(data.dtype), "shape": list(data.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(header + b"\n")
    digest.update(memoryview(data.view(torch.uint8).numpy()))
    return digest.hexdigest()


@torch.no_grad()
def project_direction(gE, gD):
    """One Euclidean projection of -gD onto <gE,d><=0 at C=0."""
    _matrix("GE", gE)
    _matrix("GD", gD, shape=gE.shape, device=gE.device)
    q, e2 = _dot(gE, gD), _dot(gE, gE)
    coefficient = min(q, 0.0) / e2 if e2 != 0.0 else 0.0
    # Scalar dot/norm stability does not turn the executable residual into FP64.
    direction = -gD if coefficient == 0.0 else -gD + coefficient * gE
    if not bool(torch.isfinite(direction).all()):
        raise PolicyError("PROJECTED_DIRECTION_NONFINITE")
    stationarity = direction.double() + gD.double() - coefficient * gE.double()
    ed = _dot(gE, direction)
    return direction.detach().clone(), {
        "q_ge_gd": q, "ge_norm_squared": e2, "ge_dot_d": ed,
        "gd_dot_d": _dot(gD, direction), "coefficient": coefficient,
        "kkt_lambda": -coefficient, "kkt_stationarity_norm": float(stationarity.norm()),
        "kkt_complementarity": -coefficient * ed,
        "first_order_violation_recorded_not_silently_reprojected": max(ed, 0.0),
        "ge_exact_zero": e2 == 0.0, "direction_norm": _norm(direction),
        "metric": "RESIDUAL_EUCLIDEAN", "gradient_sweeps_current": 1,
        "gradient_sweeps_S64": 1,
    }


@dataclass
class CorrectionResult:
    correction: torch.Tensor
    diagnostics: dict
    status: str

    @property
    def C(self):
        return self.correction

    @property
    def available(self):
        return self.status == "CORRECTION_AVAILABLE"

    def to_dict(self):
        return _json_safe({"status": self.status, "available": self.available,
                           "correction_sha256": tensor_sha256(self.correction),
                           "correction_shape": list(self.correction.shape),
                           "diagnostics": self.diagnostics})

    receipt = to_dict


@torch.no_grad()
def build_correction(Vp, Wentry, Zp, anchors, radii, A, gE, gD,
                     config=NumericalPolicy()):
    """Direction -> native target balls -> C-only executable-map trust.

    Nonfinite *native* tensors and incorrect bindings are technical errors.
    Nonfinite gradients/unexecutable correction are explicit RAW-only reasons.
    ``A`` must be the frozen, source-validated native map supplied by the caller.
    """
    _matrix("VP", Vp)
    _matrix("WENTRY", Wentry, shape=Vp.shape, device=Vp.device)
    _matrix("ZP", Zp, device=Vp.device)
    _matrix("ANCHORS", anchors, shape=Zp.shape, device=Vp.device)
    _matrix("A", A, shape=(Zp.shape[1], Vp.shape[1]), device=Vp.device)
    if Zp.shape[0] != Vp.shape[0]:
        raise PolicyError("TARGET_WEIGHT_OUTPUT_ORIENTATION")
    for name, gradient in (("GE", gE), ("GD", gD)):
        _matrix(name, gradient, shape=Zp.shape, device=Vp.device, finite=False)
    if (radii.dtype != torch.float32 or radii.shape != (Zp.shape[1],)
            or radii.device != Vp.device or not bool(torch.isfinite(radii).all())
            or bool((radii < 0).any())):
        raise PolicyError("NATIVE_RADII_SCHEMA")
    initial_distance = (Zp.double() - anchors.double()).norm(dim=0)
    allowed_radius = radii.double() * (1 + config.ball_rtol) + config.ball_atol
    if bool((initial_distance > allowed_radius).any()):
        raise PolicyError("NATIVE_ZP_OUTSIDE_ANCHOR_BALL")
    native_delta = Vp - Wentry
    native_norm = _norm(native_delta)
    diagnostics = {
        "numerical_policy": config.receipt(), "actual_native_delta_norm": native_norm,
        "native_proposal_sha256": tensor_sha256(Vp),
        "native_proposal_ball_max_excess": float((initial_distance - radii.double()).max()),
        "native_trust_limit": config.zeta * native_norm,
        "raw_anchor": "ACTUAL_STORED_VP", "native_delta_reconstructed": False,
    }

    def raw_only(reason):
        diagnostics.update(status=reason, corrected_available=False)
        return CorrectionResult(torch.zeros_like(Zp), _json_safe(diagnostics), reason)

    if not math.isfinite(native_norm):
        raise PolicyError("ACTUAL_NATIVE_DELTA_NONFINITE")
    if not bool(torch.isfinite(gE).all() and torch.isfinite(gD).all()):
        return raw_only("GRADIENT_NONFINITE_RAW_ONLY")
    if native_norm <= config.zero_norm_atol:
        return raw_only("ZERO_NATIVE_ACTION_RAW_ONLY")
    try:
        direction, projection = project_direction(gE, gD)
    except PolicyError as error:
        if str(error) != "PROJECTED_DIRECTION_NONFINITE":
            raise
        return raw_only("PROJECTED_DIRECTION_NONFINITE_RAW_ONLY")
    diagnostics["projection"] = projection
    mapped_direction = direction @ A
    mapped_norm = _norm(mapped_direction)
    diagnostics["mapped_direction_norm"] = mapped_norm
    if not math.isfinite(mapped_norm):
        return raw_only("MAPPED_DIRECTION_NONFINITE_RAW_ONLY")
    if mapped_norm <= config.zero_norm_atol:
        return raw_only("ZERO_MAPPED_DIRECTION_RAW_ONLY")
    trust_limit = config.zeta * native_norm
    alpha_norm = trust_limit / (mapped_norm + config.epsilon_num)
    alpha = min(config.alpha_cap, alpha_norm) if config.alpha_cap_mode == "bounded" else alpha_norm
    if not math.isfinite(alpha):
        return raw_only("ALPHA_NONFINITE_RAW_ONLY")
    temporary = Zp + alpha * direction
    if not bool(torch.isfinite(temporary).all()):
        return raw_only("TARGET_DISPLACEMENT_NONFINITE_RAW_ONLY")
    from_anchor = temporary - anchors
    distances = from_anchor.double().norm(dim=0)
    factors = torch.ones_like(distances)
    outside = distances > radii.double()
    factors[outside] = radii.double()[outside] / distances[outside]
    projected = anchors + from_anchor * factors.to(torch.float32).unsqueeze(0)
    # Avoid reconstructing already-inside targets, including signed-zero bytes.
    projected[:, ~outside] = temporary[:, ~outside]
    correction = projected - Zp
    projected_norm = _norm(correction @ A)
    if not math.isfinite(projected_norm):
        return raw_only("PROJECTED_MAP_NONFINITE_RAW_ONLY")
    retract = min(1.0, trust_limit / projected_norm) if projected_norm > 0 else 1.0
    if retract < 1.0:
        correction = correction * retract
    mapped = correction @ A
    mapped_final_norm = _norm(mapped)
    final_distance = ((Zp + correction).double() - anchors.double()).norm(dim=0)
    diagnostics.update(
        alpha=alpha, alpha_norm=alpha_norm, alpha_cap=config.alpha_cap,
        alpha_cap_mode=config.alpha_cap_mode,
        alpha_cap_active=(config.alpha_cap_mode == "bounded" and config.alpha_cap < alpha_norm),
        pre_ball_residual_norm=_norm(alpha * direction),
        post_ball_mapped_norm=projected_norm,
        ball_projected_requests=int(outside.sum()), trust_retraction=retract,
        residual_correction_norm=_norm(correction), mapped_correction_norm=mapped_final_norm,
        ge_dot_correction=_dot(gE, correction), gd_dot_correction=_dot(gD, correction),
        postprojection_ball_max_excess=float((final_distance - radii.double()).max()),
        postprojection_first_order_sign_is_observation_not_guarantee=True,
    )
    if bool((final_distance > allowed_radius).any()):
        return raw_only("PROJECTED_BALL_NUMERIC_FAILURE_RAW_ONLY")
    if mapped_final_norm > trust_limit * (1 + config.trust_rtol) + config.trust_atol:
        return raw_only("MAPPED_TRUST_NUMERIC_FAILURE_RAW_ONLY")
    if mapped_final_norm <= config.zero_norm_atol:
        return raw_only("ZERO_PROJECTED_CORRECTION_RAW_ONLY")
    diagnostics.update(status="CORRECTION_AVAILABLE", corrected_available=True)
    return CorrectionResult(correction.detach(), diagnostics, "CORRECTION_AVAILABLE")


@dataclass
class Candidate:
    id: str
    beta: float
    weight: torch.Tensor
    sha256: str
    actual_correction_norm: float
    trust_valid: bool
    duplicate_of: str | None = None
    exclusion_reason: str | None = None

    def receipt(self):
        return _json_safe({"id": self.id, "beta": self.beta, "sha256": self.sha256,
                "actual_correction_norm": self.actual_correction_norm,
                "trust_valid": self.trust_valid, "duplicate_of": self.duplicate_of,
                "exclusion_reason": self.exclusion_reason})

    to_dict = receipt


@torch.no_grad()
def materialize_candidates(Vp, correction, A, *, native_delta_norm,
                           config=NumericalPolicy()):
    """Exact RAW clone; other candidates independently FP32(Vp+beta*CA).

    Actual FP32 correction trust is checked after materialization. No candidate
    is restored by replaying a previous correction. Byte duplicates are aliases
    to the first candidate and must not receive a second evaluator call.
    """
    _matrix("VP", Vp)
    _matrix("C", correction, device=Vp.device)
    _matrix("A", A, shape=(correction.shape[1], Vp.shape[1]), device=Vp.device)
    if correction.shape[0] != Vp.shape[0] or not math.isfinite(native_delta_norm) or native_delta_norm < 0:
        raise PolicyError("MATERIALIZATION_SCHEMA_OR_NATIVE_NORM")
    mapped = correction @ A
    if not bool(torch.isfinite(mapped).all()):
        raise PolicyError("MATERIALIZATION_MAP_NONFINITE")
    candidates, by_hash = [], {}
    limit = config.zeta * native_delta_norm
    for name, beta in CANDIDATE_MENU:
        value = Vp.detach().clone() if name == "RAW" else Vp + mapped * beta
        finite = bool(torch.isfinite(value).all())
        norm = _norm(value - Vp) if finite else math.inf
        valid = finite and norm <= limit * (1 + config.trust_rtol) + config.trust_atol
        digest = tensor_sha256(value)
        duplicate = by_hash.get(digest)
        by_hash.setdefault(digest, name)
        reason = None if valid else ("NONFINITE_MATERIALIZATION" if not finite else "ACTUAL_FP32_TRUST_EXCEEDED")
        candidates.append(Candidate(name, beta, value.detach(), digest, norm, valid, duplicate, reason))
    return candidates


@dataclass(frozen=True)
class CandidateObservation:
    mean_nll: float
    strict_ids: frozenset
    D64: float
    request_ids: tuple


@dataclass
class Selection:
    selected_id: str
    reason: str
    candidate_receipts: list

    def receipt(self):
        return _json_safe(asdict(self))

    to_dict = receipt


def choose_candidate(candidates, observations: Mapping[str, CandidateObservation],
                     config=NumericalPolicy(), *, expected_count=100):
    """Finite exact-quality feasible min-D; RAW-priority ambiguous comparisons.

    Observations contain only Current and S64, deliberately excluding all
    observer populations. Identity includes ordered IDs, not an equal count.
    """
    if [(x.id, x.beta) for x in candidates] != list(CANDIDATE_MENU):
        raise PolicyError("CANDIDATE_MENU_NOT_FIXED_FOUR")
    if "RAW" not in observations:
        raise PolicyError("RAW_METRICS_MISSING")
    raw = observations["RAW"]
    ids = tuple(raw.request_ids)
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise PolicyError("RAW_REQUEST_DENOMINATOR_OR_DUPLICATE")
    if not set(raw.strict_ids).issubset(ids):
        raise PolicyError("RAW_STRICT_IDS_OUTSIDE_CURRENT")
    if not math.isfinite(raw.mean_nll) or not math.isfinite(raw.D64):
        raise PolicyError("NATIVE_RAW_METRICS_NONFINITE")
    by_id = {x.id: x for x in candidates}
    if not by_id["RAW"].trust_valid or by_id["RAW"].duplicate_of is not None:
        raise PolicyError("NATIVE_RAW_CANDIDATE_INVALID")
    unknown = set(observations) - set(by_id)
    if unknown:
        raise PolicyError("EXTRA_CANDIDATE_OBSERVATION")
    feasible, rows = [], []
    effective_observations = {}
    for candidate in candidates:
        row = candidate.receipt()
        if not candidate.trust_valid:
            row.update(feasible=False, reason=candidate.exclusion_reason)
            rows.append(row)
            continue
        source_id = candidate.duplicate_of or candidate.id
        if source_id not in observations:
            raise PolicyError("UNIQUE_CANDIDATE_METRICS_MISSING_" + source_id)
        if candidate.duplicate_of and candidate.id in observations:
            raise PolicyError("DUPLICATE_BYTES_EVALUATED_TWICE")
        obs = observations[source_id]
        if tuple(obs.request_ids) != ids or not set(obs.strict_ids).issubset(ids):
            raise PolicyError("CANDIDATE_CURRENT_IDENTITY_MISMATCH")
        finite = math.isfinite(obs.mean_nll) and math.isfinite(obs.D64)
        lost = set(raw.strict_ids) - set(obs.strict_ids)
        e_ok = finite and obs.mean_nll <= raw.mean_nll
        is_feasible = finite and e_ok and not lost
        reasons = ([] if finite else ["NONFINITE_CANDIDATE_METRICS"])
        if finite and not e_ok:
            reasons.append("E_EXCEEDS_OWN_RAW_NO_ALLOWANCE")
        if lost:
            reasons.append("RAW_STRICT_IDS_LOST")
        row.update(mean_nll=obs.mean_nll, D64=obs.D64, strict_success_count=len(obs.strict_ids),
                   raw_strict_lost_count=len(lost), raw_strict_lost_ids=sorted(lost, key=str),
                   E_le_own_raw=e_ok, raw_strict_subset=not lost,
                   feasible=is_feasible, reason=reasons or ["FINITE_EXACT_QUALITY_FEASIBLE"],
                   evaluation_source_id=source_id)
        rows.append(row)
        effective_observations[candidate.id] = obs
        if is_feasible:
            feasible.append(candidate)
    if not any(x.id == "RAW" for x in feasible):
        raise PolicyError("RAW_ANCHOR_NOT_FEASIBLE")
    best_d = min(effective_observations[x.id].D64 for x in feasible)

    def tied(value):
        return value - best_d <= config.d_tie_atol + config.d_tie_rtol * max(abs(value), abs(best_d))

    shortlist = [x for x in feasible if tied(effective_observations[x.id].D64)]
    order = {name: index for index, (name, _) in enumerate(CANDIDATE_MENU)}
    selected = min(shortlist, key=lambda x: (x.id != "RAW", x.actual_correction_norm, order[x.id]))
    reason = "MIN_D_EXACT_QUALITY_FEASIBLE_CORRECTION"
    if selected.id == "RAW":
        reason = "RAW_NATIVE_NO_UNAMBIGUOUS_FEASIBLE_IMPROVEMENT"
    return Selection(selected.id, reason, rows)
