#!/usr/bin/env python3
"""Deterministic analysis for the paired AlphaEdit direct-z diagnostic.

The paired panel is deliberately a narrow, single-model diagnostic.  It
compares the genuine AlphaEdit ``P``-inside-solve construction, an explicitly
post-hoc ``B @ P`` construction, and the ODE/BF trajectory under one frozen
direct-z lineage.  It reports sign-normalized paired effects, but never turns
those effects into a pooled model verdict, a method-superiority decision, or a
lifelong-editing claim.

Only scalar sanitized outcome records are accepted.  A failed case remains in
the fixed eight-case intention-to-diagnose denominator and contributes zero to
every paired effect.  A completed run is scientifically valid only when all
eight cases passed and the frozen-target/projector/rollback firewall contract
is exact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "ode-edit-direct-z-alpha-paired-manifest/v1"
SUMMARY_SCHEMA = "ode-edit-direct-z-alpha-paired-summary/v1"
STREAM_SCHEMA = "ode-edit-direct-z-alpha-paired-stream/v1"
STREAM_EVENT = "direct_z_alpha_outcome"
ANALYSIS_SCHEMA = "ode-edit-direct-z-alpha-paired-analysis/v1"

EXPECTED_CASE_COUNT = 8
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_RESAMPLES = 4000
STRICT_MAJORITY = EXPECTED_CASE_COUNT // 2 + 1
CLAIM_BOUNDARY = "atomic_geometry_signal_only_not_lifelong_or_method_superiority"

MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
EXPECTED_ALPHA_RUN_BY_MODEL = {
    "llama3-8b-inst": "dzf_alpha_llama_p0_v1",
    "qwen2.5-7b-inst": "dzf_alpha_qwen_p0_v1",
}
EXPECTED_MEMIT_RUN_BY_MODEL = {
    "llama3-8b-inst": "dzf_llama_p0_v2",
    "qwen2.5-7b-inst": "dzf_qwen_p0_v2",
}
EXPECTED_MEMIT_MANIFEST_BY_MODEL = {
    "llama3-8b-inst": "2eda8963ae8b854ee58c1835ba6184bf1cdf88e8748ac46a208bb80542235e15",
    "qwen2.5-7b-inst": "1df5dbb1cf69679cf302938f270d2bb94f2cdd324903b8f930b986b477e88a7b",
}
EXPECTED_REPLAY_LOCK_ID = (
    "8ee67660c167233e95f77cc170011eccf1526d08bd1337f5fd6da612db641f6e"
)
ARM_ORDER = (
    "no_op_replay",
    "oracle_do_z",
    "alpha_genuine_ordered_full",
    "alpha_genuine_ordered_c_matched",
    "alpha_posthoc_bp_full",
    "alpha_posthoc_bp_c_matched",
    "alpha_bf_genuine_refreshed_k4",
    "alpha_sync_z_cone_genuine",
)
EXPECTED_OUTCOME_COUNT = EXPECTED_CASE_COUNT * len(ARM_ORDER)

NO_OP = ARM_ORDER[0]
ORACLE_DO_Z = ARM_ORDER[1]
GENUINE_FULL = ARM_ORDER[2]
GENUINE_C = ARM_ORDER[3]
POSTHOC_FULL = ARM_ORDER[4]
POSTHOC_C = ARM_ORDER[5]
BF_GENUINE = ARM_ORDER[6]
SYNC_CONE_GENUINE = ARM_ORDER[7]

# This is deliberately explicit metadata rather than an inferred solver name:
# a post-hoc right projection must never be silently relabeled as genuine
# AlphaEdit.  The runner writes these fixed values into scalar-only outcomes.
ARM_CONSTRUCTION = {
    NO_OP: "no_weight_write",
    ORACLE_DO_Z: "weight_free_direct_z_ceiling",
    GENUINE_FULL: "genuine_alphaedit_isolated_ordered_solve",
    GENUINE_C: "genuine_alphaedit_isolated_ordered_solve_c_matched",
    POSTHOC_FULL: "posthoc_unprojected_alpha_base_right_projection",
    POSTHOC_C: "posthoc_unprojected_alpha_base_right_projection_c_matched",
    BF_GENUINE: "genuine_alphaedit_refreshed_k4_endpoint_c_matched",
    SYNC_CONE_GENUINE: "genuine_alphaedit_sync_cone_diagnostic",
}
GENUINE_ALPHA_ARMS = frozenset(
    (GENUINE_FULL, GENUINE_C, BF_GENUINE, SYNC_CONE_GENUINE)
)
POSTHOC_BP_ARMS = frozenset((POSTHOC_FULL, POSTHOC_C))

# The values are numerical diagnostics, not an accuracy composite.  The
# normalized sign is used only so every positive paired effect means "lhs is
# better on this axis".  The raw lhs-minus-rhs mean is retained in JSON.
METRIC_DIRECTIONS: dict[str, str] = {
    "z_residual_ratio": "lower_is_better",
    "generated_delta_error_mean": "lower_is_better",
    "generated_delta_error_worst": "lower_is_better",
    "delta_gain": "higher_is_better",
    "delta_cosine": "higher_is_better",
    "off_token_spill_ratio": "lower_is_better",
    "output_progress": "higher_is_better",
    "output_nll_reduction": "higher_is_better",
    "exact_margin_min": "higher_is_better",
    "exact_satisfied": "higher_is_better",
    "paraphrase_nll_reduction": "higher_is_better",
    "heldout_kl": "lower_is_better",
    "preservation_score": "higher_is_better",
    "endpoint_c_energy": "lower_is_better",
    "endpoint_frobenius_norm": "lower_is_better",
    "nfe": "lower_is_better",
    "endpoint_right_projector_violation_ratio": "lower_is_better",
}

CONTRASTS = (
    ("genuine_c_vs_posthoc_c", GENUINE_C, POSTHOC_C),
    ("bf_vs_genuine_c", BF_GENUINE, GENUINE_C),
    ("cone_vs_noop", SYNC_CONE_GENUINE, NO_OP),
    ("oracle_vs_noop", ORACLE_DO_Z, NO_OP),
    ("genuine_full_vs_c_matched", GENUINE_FULL, GENUINE_C),
    ("posthoc_full_vs_c_matched", POSTHOC_FULL, POSTHOC_C),
)

REFERENCE_C_REL_TOL = 5e-5
REFERENCE_C_ABS_TOL = 1e-10
ZERO_ABS_TOL = 1e-12
PRESERVATION_ABS_TOL = 1e-12
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{1,127}$")
_SAFE_CONSTRUCTION_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,127}$")
_UNSAFE_KEY_FRAGMENTS = (
    "prompt",
    "tensor",
    "logit",
    "token",
    "generation",
    "weight",
)

_MANIFEST_REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "model_alias",
    "case_count",
    "arm_order",
    "selection_sha256",
    "config_sha256",
    "bootstrap_seed",
    "bootstrap_resamples",
    "paired_memit_run_id",
    "paired_memit_manifest_sha256",
    "replay_lock_id",
    "claim_boundary",
}
_SUMMARY_REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "model_alias",
    "planned_case_count",
    "attempted_case_count",
    "pass_case_count",
    "failed_case_count",
    "outcome_count",
    "arm_order",
    "selection_sha256",
    "config_sha256",
    "frozen_target_replay_exact",
    "frozen_target_load_once_per_case",
    "direct_z_recompute_count_total",
    "projector_precomputed_only",
    "projector_integrity_exact",
    "all_rollbacks_exact",
    "firewall_pass",
    "receipt_before_outcome",
    "paired_memit_run_id",
    "paired_memit_manifest_sha256",
    "replay_lock_id",
    "genuine_alpha_ordered_solve_once_per_case",
    "genuine_alpha_bf_all_hops_genuine",
    "posthoc_arms_projection_only",
}
_ENVELOPE_FIELDS = {
    "schema_version",
    "run_id",
    "sequence",
    "recorded_at",
    "event",
    "payload",
}
_OUTCOME_FIELDS = {
    "case_id",
    "request_id",
    "arm_id",
    "case_pass",
    "arm_success",
    "technical_pass",
    "rollback_exact",
    "firewall_pass",
    "receipt_before_outcome",
    "z_residual_ratio",
    "generated_delta_error_mean",
    "generated_delta_error_worst",
    "delta_gain",
    "delta_cosine",
    "off_token_spill_ratio",
    "output_progress",
    "output_nll_reduction",
    "exact_margin_min",
    "exact_satisfied",
    "paraphrase_nll_reduction",
    "heldout_kl",
    "preservation_score",
    "endpoint_c_energy",
    "endpoint_frobenius_norm",
    "nfe",
    "arm_construction",
    "target_anchor_id",
    "frozen_target_replay_exact",
    "projector_precomputed_only",
    "projector_integrity_exact",
    "genuine_alpha_solver_used",
    "posthoc_b_at_p_only",
    "reference_c_energy",
    "endpoint_right_projector_violation_ratio",
}


class DirectZAlphaAnalysisError(ValueError):
    """Raised when a scalar artifact violates the paired-run contract."""


def _reject_nonstandard_constant(value: str) -> None:
    raise DirectZAlphaAnalysisError(f"non-finite JSON constant rejected: {value}")


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    extra = actual - expected
    for key in extra:
        lowered = key.lower()
        if any(fragment in lowered for fragment in _UNSAFE_KEY_FRAGMENTS):
            raise DirectZAlphaAnalysisError(f"{path}: unsafe key rejected")
    if actual != expected:
        raise DirectZAlphaAnalysisError(
            f"{path}: exact key set mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(extra)}"
        )


def _required_keys(value: Mapping[str, Any], required: set[str], path: str) -> None:
    """Accept safe provenance extensions while rejecting raw-bearing fields.

    The paired runner records immutable target/projector lineage metadata in
    its manifest and summary.  Those fields can grow without changing the
    scalar analysis contract, so the analyzer consumes its required subset and
    never serializes unknown values.  A raw-bearing unknown key is rejected
    rather than silently admitted.
    """

    actual = set(value)
    missing = required - actual
    if missing:
        raise DirectZAlphaAnalysisError(
            f"{path}: required key set mismatch; missing={sorted(missing)}"
        )
    for key in actual - required:
        lowered = key.lower()
        if any(fragment in lowered for fragment in _UNSAFE_KEY_FRAGMENTS):
            raise DirectZAlphaAnalysisError(f"{path}: unsafe extension key rejected")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectZAlphaAnalysisError(f"{path}: object required")
    return value


def _bool(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise DirectZAlphaAnalysisError(f"{path}: bool required")
    return value


def _int(value: Any, path: str, *, nonnegative: bool = True) -> int:
    if type(value) is not int or (nonnegative and value < 0):
        raise DirectZAlphaAnalysisError(f"{path}: invalid integer")
    return value


def _finite(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float):
        raise DirectZAlphaAnalysisError(f"{path}: finite scalar required")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise DirectZAlphaAnalysisError(f"{path}: scalar outside contract")
    return result


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DirectZAlphaAnalysisError(f"{path}: lowercase SHA-256 required")
    return value


def _run_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or _RUN_ID_RE.fullmatch(value) is None:
        raise DirectZAlphaAnalysisError(f"{path}: sanitized run id required")
    return value


def _safe_provenance_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise DirectZAlphaAnalysisError(f"{path}: sanitized provenance id required")
    return value


def _case_id(value: Any, path: str) -> str:
    if type(value) is int and value >= 0:
        return str(value)
    if isinstance(value, str) and _SAFE_ID_RE.fullmatch(value) is not None:
        return value
    raise DirectZAlphaAnalysisError(f"{path}: sanitized case id required")


def _safe_timestamp(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or any(ord(character) < 32 for character in value)
    ):
        raise DirectZAlphaAnalysisError(f"{path}: invalid recorded_at")
    return value


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DirectZAlphaAnalysisError(f"{label}: regular non-symlink file required")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, parse_constant=_reject_nonstandard_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectZAlphaAnalysisError(f"{label}: invalid JSON") from exc
    return _mapping(value, label)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    _required_keys(raw, _MANIFEST_REQUIRED_FIELDS, "manifest")
    if raw["schema_version"] != MANIFEST_SCHEMA:
        raise DirectZAlphaAnalysisError("manifest: schema mismatch")
    model_alias = raw["model_alias"]
    if model_alias not in MODEL_ALIASES:
        raise DirectZAlphaAnalysisError("manifest: unsupported model")
    arm_order = raw["arm_order"]
    if not isinstance(arm_order, list) or tuple(arm_order) != ARM_ORDER:
        raise DirectZAlphaAnalysisError("manifest: exact arm order mismatch")
    if _int(raw["case_count"], "manifest.case_count") != EXPECTED_CASE_COUNT:
        raise DirectZAlphaAnalysisError("manifest: exact case count mismatch")
    if _int(raw["bootstrap_seed"], "manifest.bootstrap_seed") != BOOTSTRAP_SEED:
        raise DirectZAlphaAnalysisError("manifest: bootstrap seed mismatch")
    if (
        _int(raw["bootstrap_resamples"], "manifest.bootstrap_resamples")
        != BOOTSTRAP_RESAMPLES
    ):
        raise DirectZAlphaAnalysisError("manifest: bootstrap resamples mismatch")
    if raw["claim_boundary"] != CLAIM_BOUNDARY:
        raise DirectZAlphaAnalysisError("manifest: claim boundary mismatch")
    run_id = _run_id(raw["run_id"], "manifest.run_id")
    paired_memit_run_id = _run_id(
        raw["paired_memit_run_id"], "manifest.paired_memit_run_id"
    )
    paired_memit_manifest_sha256 = _sha256(
        raw["paired_memit_manifest_sha256"],
        "manifest.paired_memit_manifest_sha256",
    )
    replay_lock_id = _sha256(raw["replay_lock_id"], "manifest.replay_lock_id")
    if (
        run_id != EXPECTED_ALPHA_RUN_BY_MODEL[model_alias]
        or paired_memit_run_id != EXPECTED_MEMIT_RUN_BY_MODEL[model_alias]
        or paired_memit_manifest_sha256
        != EXPECTED_MEMIT_MANIFEST_BY_MODEL[model_alias]
        or replay_lock_id != EXPECTED_REPLAY_LOCK_ID
    ):
        raise DirectZAlphaAnalysisError("manifest: fixed replay identity mismatch")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": run_id,
        "model_alias": model_alias,
        "case_count": EXPECTED_CASE_COUNT,
        "arm_order": list(ARM_ORDER),
        "selection_sha256": _sha256(
            raw["selection_sha256"], "manifest.selection_sha256"
        ),
        "config_sha256": _sha256(raw["config_sha256"], "manifest.config_sha256"),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "paired_memit_run_id": paired_memit_run_id,
        "paired_memit_manifest_sha256": paired_memit_manifest_sha256,
        "replay_lock_id": replay_lock_id,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _validate_summary(raw: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    _required_keys(raw, _SUMMARY_REQUIRED_FIELDS, "summary")
    if raw["schema_version"] != SUMMARY_SCHEMA:
        raise DirectZAlphaAnalysisError("summary: schema mismatch")
    run_id = _run_id(raw["run_id"], "summary.run_id")
    model_alias = raw["model_alias"]
    arm_order = raw["arm_order"]
    paired_memit_run_id = _run_id(
        raw["paired_memit_run_id"], "summary.paired_memit_run_id"
    )
    paired_memit_manifest_sha256 = _sha256(
        raw["paired_memit_manifest_sha256"],
        "summary.paired_memit_manifest_sha256",
    )
    replay_lock_id = _sha256(raw["replay_lock_id"], "summary.replay_lock_id")
    if (
        run_id != manifest["run_id"]
        or model_alias != manifest["model_alias"]
        or not isinstance(arm_order, list)
        or tuple(arm_order) != ARM_ORDER
        or _sha256(raw["selection_sha256"], "summary.selection_sha256")
        != manifest["selection_sha256"]
        or _sha256(raw["config_sha256"], "summary.config_sha256")
        != manifest["config_sha256"]
        or paired_memit_run_id != manifest["paired_memit_run_id"]
        or paired_memit_manifest_sha256
        != manifest["paired_memit_manifest_sha256"]
        or replay_lock_id != manifest["replay_lock_id"]
    ):
        raise DirectZAlphaAnalysisError("summary: manifest identity mismatch")
    planned = _int(raw["planned_case_count"], "summary.planned_case_count")
    attempted = _int(raw["attempted_case_count"], "summary.attempted_case_count")
    passed = _int(raw["pass_case_count"], "summary.pass_case_count")
    failed = _int(raw["failed_case_count"], "summary.failed_case_count")
    outcomes = _int(raw["outcome_count"], "summary.outcome_count")
    if (
        planned != EXPECTED_CASE_COUNT
        or attempted != EXPECTED_CASE_COUNT
        or passed + failed != EXPECTED_CASE_COUNT
        or outcomes != EXPECTED_OUTCOME_COUNT
    ):
        raise DirectZAlphaAnalysisError("summary: denominator/count mismatch")
    return {
        "schema_version": SUMMARY_SCHEMA,
        "run_id": run_id,
        "model_alias": model_alias,
        "planned_case_count": planned,
        "attempted_case_count": attempted,
        "pass_case_count": passed,
        "failed_case_count": failed,
        "outcome_count": outcomes,
        "arm_order": list(ARM_ORDER),
        "selection_sha256": manifest["selection_sha256"],
        "config_sha256": manifest["config_sha256"],
        "frozen_target_replay_exact": _bool(
            raw["frozen_target_replay_exact"], "summary.frozen_target_replay_exact"
        ),
        "frozen_target_load_once_per_case": _bool(
            raw[
                "frozen_target_load_once_per_case"
            ],
            "summary.frozen_target_load_once_per_case",
        ),
        "direct_z_recompute_count_total": _int(
            raw["direct_z_recompute_count_total"],
            "summary.direct_z_recompute_count_total",
        ),
        "projector_precomputed_only": _bool(
            raw["projector_precomputed_only"], "summary.projector_precomputed_only"
        ),
        "projector_integrity_exact": _bool(
            raw["projector_integrity_exact"], "summary.projector_integrity_exact"
        ),
        "all_rollbacks_exact": _bool(
            raw["all_rollbacks_exact"], "summary.all_rollbacks_exact"
        ),
        "firewall_pass": _bool(raw["firewall_pass"], "summary.firewall_pass"),
        "receipt_before_outcome": _bool(
            raw["receipt_before_outcome"], "summary.receipt_before_outcome"
        ),
        "paired_memit_run_id": paired_memit_run_id,
        "paired_memit_manifest_sha256": paired_memit_manifest_sha256,
        "replay_lock_id": replay_lock_id,
        "genuine_alpha_ordered_solve_once_per_case": _bool(
            raw[
                "genuine_alpha_ordered_solve_once_per_case"
            ],
            "summary.genuine_alpha_ordered_solve_once_per_case",
        ),
        "genuine_alpha_bf_all_hops_genuine": _bool(
            raw[
                "genuine_alpha_bf_all_hops_genuine"
            ],
            "summary.genuine_alpha_bf_all_hops_genuine",
        ),
        "posthoc_arms_projection_only": _bool(
            raw["posthoc_arms_projection_only"],
            "summary.posthoc_arms_projection_only",
        ),
    }


def _arm_construction(value: Any, path: str, arm_id: str) -> str:
    if not isinstance(value, str) or _SAFE_CONSTRUCTION_RE.fullmatch(value) is None:
        raise DirectZAlphaAnalysisError(f"{path}: sanitized construction required")
    expected = ARM_CONSTRUCTION[arm_id]
    if value != expected:
        raise DirectZAlphaAnalysisError(
            f"{path}: construction mismatch for {arm_id}; expected {expected}"
        )
    return value


def _normalize_payload(raw: Mapping[str, Any], path: str) -> dict[str, Any]:
    _exact_keys(raw, _OUTCOME_FIELDS, path)
    arm_id = raw["arm_id"]
    if arm_id not in ARM_ORDER:
        raise DirectZAlphaAnalysisError(f"{path}: unknown arm")
    delta_cosine = _finite(raw["delta_cosine"], f"{path}.delta_cosine")
    if delta_cosine < -1.0 or delta_cosine > 1.0:
        raise DirectZAlphaAnalysisError(f"{path}.delta_cosine: outside [-1,1]")
    heldout_kl = _finite(raw["heldout_kl"], f"{path}.heldout_kl", nonnegative=True)
    preservation_score = _finite(
        raw["preservation_score"], f"{path}.preservation_score"
    )
    if not math.isclose(
        preservation_score,
        -heldout_kl,
        rel_tol=0.0,
        abs_tol=PRESERVATION_ABS_TOL,
    ):
        raise DirectZAlphaAnalysisError(
            f"{path}: preservation_score must equal -heldout_kl"
        )
    endpoint_c_energy = _finite(
        raw["endpoint_c_energy"], f"{path}.endpoint_c_energy", nonnegative=True
    )
    reference_c_energy = _finite(
        raw["reference_c_energy"], f"{path}.reference_c_energy", nonnegative=True
    )
    if reference_c_energy <= 0.0:
        raise DirectZAlphaAnalysisError(f"{path}: reference C-energy must be positive")
    endpoint_frobenius_norm = _finite(
        raw["endpoint_frobenius_norm"],
        f"{path}.endpoint_frobenius_norm",
        nonnegative=True,
    )
    projector_violation = _finite(
        raw["endpoint_right_projector_violation_ratio"],
        f"{path}.endpoint_right_projector_violation_ratio",
        nonnegative=True,
    )
    if arm_id in (NO_OP, ORACLE_DO_Z) and (
        abs(endpoint_c_energy) > ZERO_ABS_TOL
        or abs(endpoint_frobenius_norm) > ZERO_ABS_TOL
        or abs(projector_violation) > ZERO_ABS_TOL
    ):
        raise DirectZAlphaAnalysisError(
            f"{path}: non-parameter arm endpoint diagnostics must be zero"
        )
    target_anchor_id = _sha256(raw["target_anchor_id"], f"{path}.target_anchor_id")
    frozen_target_replay_exact = _bool(
        raw["frozen_target_replay_exact"], f"{path}.frozen_target_replay_exact"
    )
    projector_precomputed_only = _bool(
        raw["projector_precomputed_only"], f"{path}.projector_precomputed_only"
    )
    projector_integrity_exact = _bool(
        raw["projector_integrity_exact"], f"{path}.projector_integrity_exact"
    )
    genuine_alpha_solver_used = _bool(
        raw["genuine_alpha_solver_used"], f"{path}.genuine_alpha_solver_used"
    )
    posthoc_b_at_p_only = _bool(
        raw["posthoc_b_at_p_only"], f"{path}.posthoc_b_at_p_only"
    )
    if not frozen_target_replay_exact:
        raise DirectZAlphaAnalysisError(f"{path}: frozen target replay must be exact")
    if not projector_precomputed_only or not projector_integrity_exact:
        raise DirectZAlphaAnalysisError(f"{path}: projector provenance must be exact")
    if genuine_alpha_solver_used != (arm_id in GENUINE_ALPHA_ARMS):
        raise DirectZAlphaAnalysisError(
            f"{path}: genuine Alpha solver flag disagrees with arm semantics"
        )
    if posthoc_b_at_p_only != (arm_id in POSTHOC_BP_ARMS):
        raise DirectZAlphaAnalysisError(
            f"{path}: post-hoc B@P flag disagrees with arm semantics"
        )
    if genuine_alpha_solver_used and posthoc_b_at_p_only:
        raise DirectZAlphaAnalysisError(
            f"{path}: genuine and post-hoc construction flags are exclusive"
        )
    return {
        "case_id": _case_id(raw["case_id"], f"{path}.case_id"),
        "request_id": _sha256(raw["request_id"], f"{path}.request_id"),
        "arm_id": arm_id,
        "case_pass": _bool(raw["case_pass"], f"{path}.case_pass"),
        "arm_success": _bool(raw["arm_success"], f"{path}.arm_success"),
        "technical_pass": _bool(raw["technical_pass"], f"{path}.technical_pass"),
        "rollback_exact": _bool(raw["rollback_exact"], f"{path}.rollback_exact"),
        "firewall_pass": _bool(raw["firewall_pass"], f"{path}.firewall_pass"),
        "receipt_before_outcome": _bool(
            raw["receipt_before_outcome"], f"{path}.receipt_before_outcome"
        ),
        "z_residual_ratio": _finite(
            raw["z_residual_ratio"], f"{path}.z_residual_ratio", nonnegative=True
        ),
        "generated_delta_error_mean": _finite(
            raw["generated_delta_error_mean"],
            f"{path}.generated_delta_error_mean",
            nonnegative=True,
        ),
        "generated_delta_error_worst": _finite(
            raw["generated_delta_error_worst"],
            f"{path}.generated_delta_error_worst",
            nonnegative=True,
        ),
        "delta_gain": _finite(raw["delta_gain"], f"{path}.delta_gain"),
        "delta_cosine": delta_cosine,
        "off_token_spill_ratio": _finite(
            raw["off_token_spill_ratio"],
            f"{path}.off_token_spill_ratio",
            nonnegative=True,
        ),
        "output_progress": _finite(raw["output_progress"], f"{path}.output_progress"),
        "output_nll_reduction": _finite(
            raw["output_nll_reduction"], f"{path}.output_nll_reduction"
        ),
        "exact_margin_min": _finite(
            raw["exact_margin_min"], f"{path}.exact_margin_min"
        ),
        "exact_satisfied": _bool(raw["exact_satisfied"], f"{path}.exact_satisfied"),
        "paraphrase_nll_reduction": _finite(
            raw["paraphrase_nll_reduction"], f"{path}.paraphrase_nll_reduction"
        ),
        "heldout_kl": heldout_kl,
        "preservation_score": preservation_score,
        "endpoint_c_energy": endpoint_c_energy,
        "endpoint_frobenius_norm": endpoint_frobenius_norm,
        "nfe": _int(raw["nfe"], f"{path}.nfe"),
        "arm_construction": _arm_construction(
            raw["arm_construction"], f"{path}.arm_construction", arm_id
        ),
        "target_anchor_id": target_anchor_id,
        "frozen_target_replay_exact": frozen_target_replay_exact,
        "projector_precomputed_only": projector_precomputed_only,
        "projector_integrity_exact": projector_integrity_exact,
        "genuine_alpha_solver_used": genuine_alpha_solver_used,
        "posthoc_b_at_p_only": posthoc_b_at_p_only,
        "reference_c_energy": reference_c_energy,
        "endpoint_right_projector_violation_ratio": projector_violation,
    }


def _load_outcomes(path: Path, run_id: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise DirectZAlphaAnalysisError("outcomes: regular non-symlink file required")
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    wrapper = json.loads(line, parse_constant=_reject_nonstandard_constant)
                except json.JSONDecodeError as exc:
                    raise DirectZAlphaAnalysisError(
                        f"outcomes line {line_number}: invalid JSON"
                    ) from exc
                wrapper = _mapping(wrapper, f"outcomes[{line_number}]")
                _exact_keys(wrapper, _ENVELOPE_FIELDS, f"outcomes[{line_number}]")
                if (
                    wrapper["schema_version"] != STREAM_SCHEMA
                    or _run_id(wrapper["run_id"], f"outcomes[{line_number}].run_id")
                    != run_id
                    or _int(wrapper["sequence"], f"outcomes[{line_number}].sequence")
                    != len(records)
                    or wrapper["event"] != STREAM_EVENT
                ):
                    raise DirectZAlphaAnalysisError(
                        f"outcomes line {line_number}: envelope identity/order mismatch"
                    )
                _safe_timestamp(
                    wrapper["recorded_at"], f"outcomes[{line_number}].recorded_at"
                )
                records.append(
                    _normalize_payload(
                        _mapping(wrapper["payload"], f"outcomes[{line_number}].payload"),
                        f"outcomes[{line_number}].payload",
                    )
                )
    except OSError as exc:
        raise DirectZAlphaAnalysisError("outcomes: read failure") from exc
    if len(records) != EXPECTED_OUTCOME_COUNT:
        raise DirectZAlphaAnalysisError(
            f"outcomes: expected {EXPECTED_OUTCOME_COUNT}, got {len(records)}"
        )
    return records


def _c_matches_reference(value: float, reference: float) -> bool:
    return math.isclose(
        value,
        reference,
        rel_tol=REFERENCE_C_REL_TOL,
        abs_tol=REFERENCE_C_ABS_TOL,
    )


def _c_within_reference_cap(value: float, reference: float) -> bool:
    return value <= reference * (1.0 + REFERENCE_C_REL_TOL) + REFERENCE_C_ABS_TOL


def _group_cases(
    records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    cases: list[dict[str, Any]] = []
    technical_failures: list[str] = []
    for case_index in range(EXPECTED_CASE_COUNT):
        block = records[case_index * len(ARM_ORDER) : (case_index + 1) * len(ARM_ORDER)]
        first = block[0]
        case_id = first["case_id"]
        request_id = first["request_id"]
        target_anchor_id = first["target_anchor_id"]
        reference_c_energy = first["reference_c_energy"]
        if (
            tuple(record["arm_id"] for record in block) != ARM_ORDER
            or {record["case_id"] for record in block} != {case_id}
            or {record["request_id"] for record in block} != {request_id}
            or {record["target_anchor_id"] for record in block} != {target_anchor_id}
            or {record["reference_c_energy"] for record in block}
            != {reference_c_energy}
            or len({record["case_pass"] for record in block}) != 1
        ):
            raise DirectZAlphaAnalysisError(
                f"case block {case_index}: identity/arm order mismatch"
            )
        arms = {record["arm_id"]: dict(record) for record in block}
        case_pass = bool(first["case_pass"])
        all_arm_success = all(record["arm_success"] for record in block)
        if case_pass != all_arm_success:
            technical_failures.append(f"case={case_id}: case/arm success mismatch")
        all_technical_flags = True
        for flag in (
            "technical_pass",
            "rollback_exact",
            "firewall_pass",
            "receipt_before_outcome",
        ):
            if not all(record[flag] for record in block):
                technical_failures.append(f"case={case_id}: {flag}=false")
                all_technical_flags = False
        if case_pass:
            for arm_id in (GENUINE_C, POSTHOC_C, BF_GENUINE):
                if not _c_matches_reference(
                    arms[arm_id]["endpoint_c_energy"], reference_c_energy
                ):
                    technical_failures.append(
                        f"case={case_id}: {arm_id} reference C-match violation"
                    )
                    all_technical_flags = False
            cone = arms[SYNC_CONE_GENUINE]
            if not _c_within_reference_cap(
                cone["endpoint_c_energy"], reference_c_energy
            ):
                technical_failures.append(f"case={case_id}: cone C-cap violation")
                all_technical_flags = False
            if cone["endpoint_c_energy"] == 0.0 and (
                cone["endpoint_frobenius_norm"] > ZERO_ABS_TOL
                or cone["endpoint_right_projector_violation_ratio"] > ZERO_ABS_TOL
            ):
                technical_failures.append(
                    f"case={case_id}: zero-cone endpoint diagnostics are nonzero"
                )
                all_technical_flags = False
        cases.append(
            {
                "case_id": case_id,
                "request_id": request_id,
                "target_anchor_id": target_anchor_id,
                "reference_c_energy": reference_c_energy,
                "case_pass": case_pass,
                "analysis_success": case_pass and all_arm_success and all_technical_flags,
                "arms": arms,
            }
        )
    if len({case["case_id"] for case in cases}) != EXPECTED_CASE_COUNT:
        raise DirectZAlphaAnalysisError("case ids must be unique")
    if len({case["request_id"] for case in cases}) != EXPECTED_CASE_COUNT:
        raise DirectZAlphaAnalysisError("request ids must be unique")
    if len({case["target_anchor_id"] for case in cases}) != EXPECTED_CASE_COUNT:
        raise DirectZAlphaAnalysisError("target anchor ids must be unique")
    observed_pass = sum(
        case["case_pass"] and all(record["arm_success"] for record in case["arms"].values())
        for case in cases
    )
    if (
        observed_pass != summary["pass_case_count"]
        or EXPECTED_CASE_COUNT - observed_pass != summary["failed_case_count"]
    ):
        raise DirectZAlphaAnalysisError("summary/outcomes pass-count mismatch")
    observed_flags = {
        "all_rollbacks_exact": all(record["rollback_exact"] for record in records),
        "firewall_pass": all(record["firewall_pass"] for record in records),
        "receipt_before_outcome": all(
            record["receipt_before_outcome"] for record in records
        ),
        "frozen_target_replay_exact": all(
            record["frozen_target_replay_exact"] for record in records
        ),
        "projector_precomputed_only": all(
            record["projector_precomputed_only"] for record in records
        ),
        "projector_integrity_exact": all(
            record["projector_integrity_exact"] for record in records
        ),
    }
    for key, observed in observed_flags.items():
        if observed != summary[key]:
            raise DirectZAlphaAnalysisError(
                f"summary/outcomes flag mismatch: {key}"
            )
    if summary["failed_case_count"] != 0:
        technical_failures.append("summary: failed_case_count must be zero")
    required_summary_flags = (
        "frozen_target_replay_exact",
        "frozen_target_load_once_per_case",
        "projector_precomputed_only",
        "projector_integrity_exact",
        "all_rollbacks_exact",
        "firewall_pass",
        "receipt_before_outcome",
        "genuine_alpha_ordered_solve_once_per_case",
        "genuine_alpha_bf_all_hops_genuine",
        "posthoc_arms_projection_only",
    )
    for flag in required_summary_flags:
        if not summary[flag]:
            technical_failures.append(f"summary: {flag}=false")
    if summary["direct_z_recompute_count_total"] != 0:
        technical_failures.append("summary: direct_z_recompute_count_total must equal zero")
    return cases, technical_failures


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise DirectZAlphaAnalysisError("quantile requires values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap_mean_ci(values: Sequence[float], stream: int) -> list[float]:
    if len(values) != EXPECTED_CASE_COUNT:
        raise DirectZAlphaAnalysisError("bootstrap ITD denominator mismatch")
    rng = random.Random(BOOTSTRAP_SEED + stream)
    estimates = sorted(
        math.fsum(values[rng.randrange(EXPECTED_CASE_COUNT)] for _ in values)
        / EXPECTED_CASE_COUNT
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return [_quantile(estimates, 0.025), _quantile(estimates, 0.975)]


def _effect_summary(values: Sequence[float], stream: int) -> dict[str, Any]:
    if len(values) != EXPECTED_CASE_COUNT or any(not math.isfinite(v) for v in values):
        raise DirectZAlphaAnalysisError("effect summary denominator/value mismatch")
    mean = math.fsum(values) / EXPECTED_CASE_COUNT
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    return {
        "n_itd": EXPECTED_CASE_COUNT,
        "mean": mean,
        "median": statistics.median(values),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": EXPECTED_CASE_COUNT - positive - negative,
        "positive_fraction": positive / EXPECTED_CASE_COUNT,
        "lenient_positive": mean > 0.0 and positive >= STRICT_MAJORITY,
        "lenient_negative": mean < 0.0 and negative >= STRICT_MAJORITY,
        "paired_bootstrap_mean_ci95": _bootstrap_mean_ci(values, stream),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "ci_exclusion_required_for_gate": False,
    }


def _metric_value(record: Mapping[str, Any], metric: str) -> float:
    value = record[metric]
    if type(value) is bool:
        return float(value)
    return float(value)


def _build_contrasts(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    stream = 0
    for contrast_id, lhs_arm, rhs_arm in CONTRASTS:
        metrics: dict[str, Any] = {}
        for metric, direction in METRIC_DIRECTIONS.items():
            raw_values: list[float] = []
            normalized_values: list[float] = []
            for case in cases:
                if case["analysis_success"]:
                    raw = _metric_value(case["arms"][lhs_arm], metric) - _metric_value(
                        case["arms"][rhs_arm], metric
                    )
                    normalized = raw if direction == "higher_is_better" else -raw
                else:
                    raw = normalized = 0.0
                raw_values.append(raw)
                normalized_values.append(normalized)
            summary = _effect_summary(normalized_values, stream)
            stream += 1
            summary.update(
                {
                    "direction": direction,
                    "sign_normalized": True,
                    "raw_mean_lhs_minus_rhs": math.fsum(raw_values)
                    / EXPECTED_CASE_COUNT,
                    "raw_median_lhs_minus_rhs": statistics.median(raw_values),
                }
            )
            metrics[metric] = summary
        result[contrast_id] = {
            "lhs_arm": lhs_arm,
            "rhs_arm": rhs_arm,
            "positive_meaning": "lhs_better_on_this_axis",
            "metrics": metrics,
        }
    return result


def _signal(contrasts: Mapping[str, Any], contrast_id: str, metric: str) -> bool:
    return bool(contrasts[contrast_id]["metrics"][metric]["lenient_positive"])


def _classify(contrasts: Mapping[str, Any], technical_pass: bool) -> dict[str, Any]:
    # Each item below is deliberately an axis-level signal.  No aggregate
    # score, model pooling, or "winner" label is constructed.
    signals = {
        "oracle_activation_ceiling": _signal(
            contrasts, "oracle_vs_noop", "z_residual_ratio"
        )
        and _signal(contrasts, "oracle_vs_noop", "output_nll_reduction"),
        "genuine_cone_actual_write_signal": _signal(
            contrasts, "cone_vs_noop", "z_residual_ratio"
        )
        and _signal(contrasts, "cone_vs_noop", "output_nll_reduction"),
        "genuine_c_vs_posthoc_c_z_fidelity_signal": _signal(
            contrasts, "genuine_c_vs_posthoc_c", "z_residual_ratio"
        ),
        "bf_vs_genuine_c_z_fidelity_signal": _signal(
            contrasts, "bf_vs_genuine_c", "z_residual_ratio"
        ),
        "genuine_full_vs_c_matched_z_signal": _signal(
            contrasts, "genuine_full_vs_c_matched", "z_residual_ratio"
        ),
        "posthoc_full_vs_c_matched_z_signal": _signal(
            contrasts, "posthoc_full_vs_c_matched", "z_residual_ratio"
        ),
    }
    return {
        "verdict": (
            "PAIRED_ALPHA_SIGNALS_REPORTED_NO_SUPERIORITY_DECISION"
            if technical_pass
            else "BLOCK_TECHNICAL_INVALID"
        ),
        "technical_pass": technical_pass,
        "axis_level_lenient_signals": signals,
        "gate": {
            "strict_majority": STRICT_MAJORITY,
            "requires_positive_mean": True,
            "requires_ci_exclusion": False,
            "method_superiority_decision": False,
            "model_pooling": False,
            "lifelong_or_sequential_claim": False,
        },
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def analyze_run(run_directory: str | Path) -> dict[str, Any]:
    root = Path(run_directory).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise DirectZAlphaAnalysisError("run directory must be a regular directory")
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    outcomes_path = root / "outcomes.jsonl"
    manifest = _validate_manifest(_read_json_object(manifest_path, "manifest"))
    summary = _validate_summary(_read_json_object(summary_path, "summary"), manifest)
    records = _load_outcomes(outcomes_path, manifest["run_id"])
    cases, technical_failures = _group_cases(records, summary)
    contrasts = _build_contrasts(cases)
    technical_pass = not technical_failures
    classification = _classify(contrasts, technical_pass)
    analysis: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA,
        "run_id": manifest["run_id"],
        "model_alias": manifest["model_alias"],
        "scope": "single_model_paired_alpha_direct_z_signal",
        "claim_boundary": CLAIM_BOUNDARY,
        "fixed_contract": {
            "case_count": EXPECTED_CASE_COUNT,
            "arm_order": list(ARM_ORDER),
            "arm_construction": dict(ARM_CONSTRUCTION),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "failed_cases_itd_zero_effect": True,
            "ci_exclusion_required_for_gate": False,
            "lower_is_better_axes_sign_normalized": True,
            "endpoint_right_projector_violation_ratio": "reported_axis_not_validity_gate",
        },
        "artifact_validation": {
            "valid": technical_pass,
            "failures": technical_failures,
            "planned_case_count": EXPECTED_CASE_COUNT,
            "successful_case_count": sum(
                bool(case["analysis_success"]) for case in cases
            ),
            "failed_case_count": sum(
                not bool(case["analysis_success"]) for case in cases
            ),
            "itd_denominator": EXPECTED_CASE_COUNT,
            "outcome_count": len(records),
            "summary_technical_contract": {
                "frozen_target_replay_exact": summary[
                    "frozen_target_replay_exact"
                ],
                "frozen_target_load_once_per_case": summary[
                    "frozen_target_load_once_per_case"
                ],
                "direct_z_recompute_count_total": summary[
                    "direct_z_recompute_count_total"
                ],
                "projector_precomputed_only": summary[
                    "projector_precomputed_only"
                ],
                "projector_integrity_exact": summary[
                    "projector_integrity_exact"
                ],
                "all_rollbacks_exact": summary["all_rollbacks_exact"],
                "firewall_pass": summary["firewall_pass"],
                "receipt_before_outcome": summary["receipt_before_outcome"],
                "genuine_alpha_ordered_solve_once_per_case": summary[
                    "genuine_alpha_ordered_solve_once_per_case"
                ],
                "genuine_alpha_bf_all_hops_genuine": summary[
                    "genuine_alpha_bf_all_hops_genuine"
                ],
                "posthoc_arms_projection_only": summary[
                    "posthoc_arms_projection_only"
                ],
                "paired_memit_run_id": summary["paired_memit_run_id"],
                "paired_memit_manifest_sha256": summary[
                    "paired_memit_manifest_sha256"
                ],
                "replay_lock_id": summary["replay_lock_id"],
            },
        },
        "contrasts": contrasts,
        "classification": classification,
        "compute": {
            "nfe_total_itd": sum(
                int(record["nfe"])
                for case in cases
                for record in case["arms"].values()
            ),
            "endpoint_c_energy_genuine_posthoc_c_matched": all(
                not case["case_pass"]
                or (
                    _c_matches_reference(
                        case["arms"][GENUINE_C]["endpoint_c_energy"],
                        case["reference_c_energy"],
                    )
                    and _c_matches_reference(
                        case["arms"][POSTHOC_C]["endpoint_c_energy"],
                        case["reference_c_energy"],
                    )
                )
                for case in cases
            ),
            "endpoint_c_energy_bf_genuine_c_matched": all(
                not case["case_pass"]
                or (
                    _c_matches_reference(
                        case["arms"][GENUINE_C]["endpoint_c_energy"],
                        case["reference_c_energy"],
                    )
                    and _c_matches_reference(
                        case["arms"][BF_GENUINE]["endpoint_c_energy"],
                        case["reference_c_energy"],
                    )
                )
                for case in cases
            ),
            "endpoint_c_energy_cone_within_reference_cap": all(
                not case["case_pass"]
                or _c_within_reference_cap(
                    case["arms"][SYNC_CONE_GENUINE]["endpoint_c_energy"],
                    case["reference_c_energy"],
                )
                for case in cases
            ),
            "max_endpoint_right_projector_violation_ratio_by_arm": {
                arm_id: max(
                    float(case["arms"][arm_id]["endpoint_right_projector_violation_ratio"])
                    for case in cases
                )
                for arm_id in ARM_ORDER
            },
        },
        "case_diagnostics": [
            {
                "case_id": case["case_id"],
                "analysis_success": case["analysis_success"],
                "genuine_c_z_fidelity_over_posthoc_c": (
                    case["arms"][POSTHOC_C]["z_residual_ratio"]
                    - case["arms"][GENUINE_C]["z_residual_ratio"]
                    if case["analysis_success"]
                    else 0.0
                ),
                "bf_z_fidelity_over_genuine_c": (
                    case["arms"][GENUINE_C]["z_residual_ratio"]
                    - case["arms"][BF_GENUINE]["z_residual_ratio"]
                    if case["analysis_success"]
                    else 0.0
                ),
                "cone_z_fidelity_over_noop": (
                    case["arms"][NO_OP]["z_residual_ratio"]
                    - case["arms"][SYNC_CONE_GENUINE]["z_residual_ratio"]
                    if case["analysis_success"]
                    else 0.0
                ),
                "oracle_z_fidelity_over_noop": (
                    case["arms"][NO_OP]["z_residual_ratio"]
                    - case["arms"][ORACLE_DO_Z]["z_residual_ratio"]
                    if case["analysis_success"]
                    else 0.0
                ),
            }
            for case in cases
        ],
        "source_hashes": {
            "manifest_json": _file_sha256(manifest_path),
            "summary_json": _file_sha256(summary_path),
            "outcomes_jsonl": _file_sha256(outcomes_path),
            "analysis_code": _file_sha256(Path(__file__).resolve(strict=True)),
        },
        "interpretation_limits": [
            "single-model paired diagnostic; no model pooling",
            "axis-level signals only; no method-superiority decision",
            "bootstrap interval is descriptive and not a gate",
            "no lifelong, sequential-retention, or collapse-prevention claim",
            "direct-z fidelity, edit proxies, preservation, norm, spill, and NFE are reported separately",
        ],
    }
    analysis["analysis_sha256"] = hashlib.sha256(_canonical_json(analysis)).hexdigest()
    return analysis


def render_markdown(analysis: Mapping[str, Any]) -> str:
    contrasts = analysis["contrasts"]
    focus_rows = (
        ("genuine C vs posthoc C", "genuine_c_vs_posthoc_c"),
        ("BF vs genuine C", "bf_vs_genuine_c"),
        ("genuine cone vs no-op", "cone_vs_noop"),
        ("oracle vs no-op", "oracle_vs_noop"),
        ("genuine full vs C-match", "genuine_full_vs_c_matched"),
        ("posthoc full vs C-match", "posthoc_full_vs_c_matched"),
    )
    primary_metrics = (
        "z_residual_ratio",
        "output_nll_reduction",
        "preservation_score",
        "endpoint_frobenius_norm",
        "endpoint_right_projector_violation_ratio",
    )
    lines = [
        f"# Paired Alpha direct-z analysis — {analysis['model_alias']}",
        "",
        f"- run: `{analysis['run_id']}`",
        f"- verdict: `{analysis['classification']['verdict']}`",
        f"- technical validity: `{analysis['artifact_validation']['valid']}`",
        f"- ITD: `{analysis['artifact_validation']['successful_case_count']}/"
        f"{analysis['artifact_validation']['itd_denominator']}` successful",
        f"- analysis SHA-256: `{analysis['analysis_sha256']}`",
        "- every displayed effect is sign-normalized: positive means the listed lhs is better on that individual axis.",
        "- scope: a single-model paired signal report, not a method-superiority or lifelong-editing analysis.",
        "",
        "| Paired contrast / metric | Normalized mean | Positive | Bootstrap mean CI95 |",
        "|---|---:|---:|---:|",
    ]
    for label, contrast_id in focus_rows:
        for metric in primary_metrics:
            item = contrasts[contrast_id]["metrics"][metric]
            lo, hi = item["paired_bootstrap_mean_ci95"]
            lines.append(
                f"| {label}: `{metric}` | {item['mean']:.8f} | "
                f"{item['positive_count']}/{item['n_itd']} | "
                f"[{lo:.8f}, {hi:.8f}] |"
            )
    contract = analysis["artifact_validation"]["summary_technical_contract"]
    lines.extend(
        [
            "",
            "## Technical contract",
            "",
            f"- frozen target replay exact: `{contract['frozen_target_replay_exact']}`",
            f"- frozen target loaded once per case: `{contract['frozen_target_load_once_per_case']}`",
            f"- direct-z recomputes: `{contract['direct_z_recompute_count_total']}` (must be `0`)",
            f"- precomputed projector only / integrity exact: "
            f"`{contract['projector_precomputed_only']}` / `{contract['projector_integrity_exact']}`",
            f"- genuine ordered once/case, BF all hops genuine, post-hoc projection-only: "
            f"`{contract['genuine_alpha_ordered_solve_once_per_case']}` / "
            f"`{contract['genuine_alpha_bf_all_hops_genuine']}` / "
            f"`{contract['posthoc_arms_projection_only']}`",
            f"- paired MEMIT run / replay lock: `{contract['paired_memit_run_id']}` / "
            f"`{contract['replay_lock_id']}`",
            f"- exact rollback / firewall / receipt: "
            f"`{contract['all_rollbacks_exact']}` / `{contract['firewall_pass']}` / "
            f"`{contract['receipt_before_outcome']}`",
            "",
            "## Interpretation boundary",
            "",
            "- The JSON reports every scalar axis, including direct-z fidelity, output/edit proxies, preservation, spill, endpoint C/Frobenius cost, NFE, and projector leak.  Projector leak is a reported axis, not a validity gate.",
            "- The 4,000-resample paired-bootstrap intervals are descriptive; the lenient signal gate is positive mean plus at least 5/8 positive cases.",
            "- Do not pool Llama and Qwen, relabel a post-hoc BP update as genuine AlphaEdit, infer method superiority, or infer lifelong/sequential-collapse behavior.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    analysis: Mapping[str, Any], output_json: str | Path, output_markdown: str | Path
) -> None:
    json_path = Path(output_json).expanduser().resolve()
    markdown_path = Path(output_markdown).expanduser().resolve()
    if json_path == markdown_path:
        raise DirectZAlphaAnalysisError("JSON and Markdown outputs must differ")
    for path in (json_path, markdown_path):
        if path.exists() or path.is_symlink():
            raise DirectZAlphaAnalysisError("analysis outputs are exclusive")
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise DirectZAlphaAnalysisError("output parent must already exist")
    json_text = (
        json.dumps(
            analysis,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    markdown_text = render_markdown(analysis)
    try:
        with json_path.open("x", encoding="utf-8") as handle:
            handle.write(json_text)
        with markdown_path.open("x", encoding="utf-8") as handle:
            handle.write(markdown_text)
    except OSError as exc:
        raise DirectZAlphaAnalysisError("exclusive analysis output failed") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ode-edit-direct-z-alpha-analysis", allow_abbrev=False
    )
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis = analyze_run(args.run_directory)
        write_outputs(analysis, args.output_json, args.output_markdown)
    except DirectZAlphaAnalysisError as exc:
        print(f"direct-z Alpha analysis rejected: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "model_alias": analysis["model_alias"],
                "verdict": analysis["classification"]["verdict"],
                "analysis_sha256": analysis["analysis_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if analysis["artifact_validation"]["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
