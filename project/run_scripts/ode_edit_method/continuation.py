"""Fail-closed helpers for the prelocked P1 ``S_max=8`` continuation.

The continuation is deliberately narrower than a second P1 run.  It selects
only the Full trajectories that terminated at the prelocked resolution cap,
pins their complete R2 provenance, reuses their direct-z artifacts read-only,
and requires rounds 1--6 to reproduce exactly before rounds 7--8 may be read.
No prompt, target, context string, evaluation field, or timer enters the
selection or prefix identity.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from project.run_scripts.ode_edit_motivation.contracts import (
    ExpectedFileIdentity,
)
from project.run_scripts.ode_edit_motivation.hooks import tensor_sha256

from .contracts import (
    ArmRunResult,
    ControllerConfig,
    MethodContractError,
    canonical_hash,
)
from .easyedit_backend import EasyEditMemitBackend
from .events import ControllerRequest


CANONICAL_CASE_ORDER = ("2022", "12498", "20964", "768")
EXPECTED_AFFECTED_CASES = {
    "llama3-8b-inst": ("2022", "20964", "768"),
    "qwen2.5-7b-inst": (),
}
R2_EXECUTION_HEAD = "e285c96d704a04287f65d1c51539ebbb81451d28"
R2_PROPOSAL_ID = "83eec5068acec673f7d9d0788e57ce0e9873a55785f7a6932026a4a2420c330b"
R2_LOCK_SHA256 = "95376c554f01e00c9c5d71e99dfa3cbb5339a7534db8a62a9a05bc69c0be1bbb"
SERVER1_GPU_CAP = 3
BASE_S_MAX = 6
CONTINUATION_S_MAX = 8
PROMOTION_MINIMUM_HITS = 2


@dataclass(frozen=True, slots=True)
class R2SourceSpec:
    model_alias: str
    root: Path
    terminal_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class AffectedTrajectory:
    model_alias: str
    case_id: str
    order_position: int
    pre_edit_state_id: str
    pre_edit_target_weight_state_id: str
    direct_z_relative_path: str
    direct_z_identity: ExpectedFileIdentity
    direct_z_tensor_sha256: str
    direct_z_source_state_id: str
    prefix_payload: Mapping[str, Any]
    prefix_fingerprint: str

    def to_raw_free_dict(self) -> dict[str, Any]:
        return {
            "model_alias": self.model_alias,
            "case_id": self.case_id,
            "order_position": self.order_position,
            "pre_edit_state_id": self.pre_edit_state_id,
            "pre_edit_target_weight_state_id": self.pre_edit_target_weight_state_id,
            "direct_z_relative_path": self.direct_z_relative_path,
            "direct_z_sha256": self.direct_z_identity.sha256,
            "direct_z_size": self.direct_z_identity.size,
            "direct_z_tensor_sha256": self.direct_z_tensor_sha256,
            "direct_z_source_state_id": self.direct_z_source_state_id,
            "prefix_fingerprint": self.prefix_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ValidatedR2Source:
    spec: R2SourceSpec
    manifest_sha256: str
    summary_sha256: str
    affected: tuple[AffectedTrajectory, ...]
    file_identities: tuple[tuple[str, ExpectedFileIdentity], ...]

    def identity_for(self, relative_path: str) -> ExpectedFileIdentity:
        try:
            return dict(self.file_identities)[relative_path]
        except KeyError as exc:
            raise MethodContractError(
                f"R2 terminal manifest is missing {relative_path}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ContinuationSelection:
    sources: tuple[ValidatedR2Source, ...]

    @property
    def pooled_affected_count(self) -> int:
        return sum(len(source.affected) for source in self.sources)

    def for_alias(self, model_alias: str) -> tuple[AffectedTrajectory, ...]:
        for source in self.sources:
            if source.spec.model_alias == model_alias:
                return source.affected
        raise MethodContractError(f"R2 source alias is absent: {model_alias}")

    def source_for_alias(self, model_alias: str) -> ValidatedR2Source:
        for source in self.sources:
            if source.spec.model_alias == model_alias:
                return source
        raise MethodContractError(f"R2 source alias is absent: {model_alias}")


DEFAULT_R2_SOURCE_SPECS = (
    R2SourceSpec(
        model_alias="llama3-8b-inst",
        root=Path(
            "local/results/session02-p1-four-case-mechanism-r2-"
            "llama3-8b-inst-83eec506"
        ),
        terminal_manifest_sha256=(
            "19d7619400dd4387f69b05192046297b5331ee5d4e581486ab9544eeafc0da6f"
        ),
    ),
    R2SourceSpec(
        model_alias="qwen2.5-7b-inst",
        root=Path(
            "local/results/session02-p1-four-case-mechanism-r2-"
            "qwen2.5-7b-inst-83eec506"
        ),
        terminal_manifest_sha256=(
            "c830799669a6d7a537cb6b4893f8e2d580f307f5bf445bae68c51885cbb9c381"
        ),
    ),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise MethodContractError(f"non-finite JSON constant in {path.name}: {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MethodContractError(f"invalid R2 JSON artifact: {path.name}") from exc


def _strict_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MethodContractError(f"invalid R2 JSONL artifact: {path.name}") from exc
    for index, line in enumerate(lines):
        if not line:
            raise MethodContractError(f"empty line in R2 {path.name}:{index + 1}")
        try:
            row = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    MethodContractError(
                        f"non-finite JSON constant in {path.name}:{index + 1}: {value}"
                    )
                ),
            )
        except (json.JSONDecodeError, MethodContractError) as exc:
            raise MethodContractError(
                f"invalid R2 JSONL artifact: {path.name}:{index + 1}"
            ) from exc
        if not isinstance(row, Mapping):
            raise MethodContractError(f"R2 {path.name}:{index + 1} is not an object")
        rows.append(row)
    return tuple(rows)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MethodContractError(f"{label} is not a canonical SHA-256")
    return value


_FORBIDDEN_RAW_KEYS = frozenset(
    {
        "prompt",
        "subject",
        "target_new",
        "target_old",
        "raw_context",
        "context_templates",
        "templates",
    }
)


def _assert_raw_free(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_RAW_KEYS.intersection(value)
        if forbidden:
            raise MethodContractError(
                f"{label} contains forbidden raw fields: {sorted(forbidden)}"
            )
        for child in value.values():
            _assert_raw_free(child, label)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_raw_free(child, label)


def _zero_omega(value: Any, label: str) -> None:
    if not isinstance(value, Mapping) or not value:
        raise MethodContractError(f"{label} is not a non-empty Omega mapping")
    for layer, amount in value.items():
        try:
            numeric = float(amount)
        except (TypeError, ValueError) as exc:
            raise MethodContractError(f"{label} is not exactly zero") from exc
        if (
            not isinstance(layer, str)
            or isinstance(amount, bool)
            or not math.isfinite(numeric)
            or numeric != 0.0
        ):
            raise MethodContractError(f"{label} is not exactly zero")


_PREFIX_FIELDS = (
    "position",
    "layer",
    "direction_ids",
    "solver_coefficients",
    "applied_coefficients",
    "hard_phi_before",
    "hard_phi_after",
    "smooth_phi_before",
    "smooth_phi_after",
    "accepted",
    "reason",
)


def canonical_six_round_prefix(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Project all attempts through accepted round six into a raw-free payload."""

    projected: list[dict[str, Any]] = []
    accepted_count = 0
    for raw in steps:
        if not isinstance(raw, Mapping):
            raise MethodContractError("controller step is not an object")
        missing = set(_PREFIX_FIELDS) - set(raw)
        if missing:
            raise MethodContractError(
                f"controller step lacks prefix fields: {sorted(missing)}"
            )
        row = {field: raw[field] for field in _PREFIX_FIELDS}
        if row["accepted"] is not True and row["accepted"] is not False:
            raise MethodContractError("controller step accepted flag is not boolean")
        if not isinstance(row["direction_ids"], list) or not row["direction_ids"]:
            raise MethodContractError("controller step direction IDs are absent")
        for key in ("solver_coefficients", "applied_coefficients"):
            if not isinstance(row[key], list) or len(row[key]) != len(
                row["direction_ids"]
            ):
                raise MethodContractError(f"controller step {key} shape differs")
        projected.append(row)
        if row["accepted"]:
            accepted_count += 1
            if accepted_count == BASE_S_MAX:
                break
    if accepted_count != BASE_S_MAX:
        raise MethodContractError(
            "trajectory does not contain six accepted prefix rounds"
        )
    return {
        "schema_version": "ode-edit-p1-smax8-prefix/v1",
        "accepted_rounds": BASE_S_MAX,
        "attempts": projected,
    }


def assert_prefix_match(
    expected: Mapping[str, Any], observed_steps: Sequence[Mapping[str, Any]]
) -> str:
    observed = canonical_six_round_prefix(observed_steps)
    if observed != expected:
        raise MethodContractError(
            "S_max=8 continuation rounds 1-6 differ from the pinned R2 prefix"
        )
    return canonical_hash(observed)


def _verify_terminal_files(
    root: Path, terminal: Mapping[str, Any]
) -> tuple[tuple[str, ExpectedFileIdentity], ...]:
    if (
        terminal.get("schema_version") != "ode-edit-session02-terminal-manifest/v1"
        or terminal.get("status") != "COMPLETE"
        or not isinstance(terminal.get("files"), Mapping)
    ):
        raise MethodContractError("R2 terminal manifest schema/status differs")
    records: list[tuple[str, ExpectedFileIdentity]] = []
    for relative, raw_identity in sorted(terminal["files"].items()):
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
        ):
            raise MethodContractError("R2 terminal manifest path is invalid")
        candidate = (root / relative).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise MethodContractError(
                "R2 terminal file escaped its source root"
            ) from exc
        if not candidate.is_file() or not isinstance(raw_identity, Mapping):
            raise MethodContractError("R2 terminal file identity is invalid")
        try:
            identity = ExpectedFileIdentity.from_value(raw_identity)
        except ValueError as exc:
            raise MethodContractError("R2 terminal file identity is invalid") from exc
        if (
            candidate.stat().st_size != identity.size
            or file_sha256(candidate) != identity.sha256
        ):
            raise MethodContractError(f"R2 terminal file identity differs: {relative}")
        records.append((relative, identity))
    required = {
        "manifest.json",
        "controller_steps.jsonl",
        "compute.jsonl",
        "mechanism.jsonl",
        "evaluation.jsonl",
        "summary.json",
    }
    if not required.issubset(dict(records)):
        raise MethodContractError("R2 terminal manifest lacks required artifacts")
    return tuple(records)


def _validate_source(spec: R2SourceSpec, *, repo: Path) -> ValidatedR2Source:
    if spec.model_alias not in EXPECTED_AFFECTED_CASES:
        raise MethodContractError("R2 source has a non-canonical model alias")
    root_candidate = spec.root if spec.root.is_absolute() else repo / spec.root
    root = root_candidate.resolve(strict=True)
    if not root.is_dir():
        raise MethodContractError("R2 source root is not a directory")
    terminal_path = root / "terminal_manifest.json"
    if file_sha256(terminal_path) != _sha(
        spec.terminal_manifest_sha256, "R2 terminal manifest SHA-256"
    ):
        raise MethodContractError("R2 terminal manifest identity differs")
    terminal = _strict_json(terminal_path)
    if not isinstance(terminal, Mapping):
        raise MethodContractError("R2 terminal manifest is not an object")
    file_identities = _verify_terminal_files(root, terminal)
    identities = dict(file_identities)

    manifest = _strict_json(root / "manifest.json")
    summary = _strict_json(root / "summary.json")
    if not isinstance(manifest, Mapping) or not isinstance(summary, Mapping):
        raise MethodContractError("R2 manifest/summary is not an object")
    if (
        manifest.get("status") != "COMPLETE_TECHNICAL_MECHANISM_ONLY"
        or manifest.get("git", {}).get("commit") != R2_EXECUTION_HEAD
        or manifest.get("git", {}).get("proposal_id") != R2_PROPOSAL_ID
        or manifest.get("model", {}).get("model_alias") != spec.model_alias
        or manifest.get("model", {}).get("checkpoint_original_dtype")
        != "torch.bfloat16"
        or manifest.get("model", {}).get("observed_parameter_dtype")
        != "torch.bfloat16"
        or manifest.get("model", {}).get("config_torch_dtype") != "torch.bfloat16"
        or tuple(manifest.get("selection", {}).get("case_ids", ()))
        != CANONICAL_CASE_ORDER
        or manifest.get("expected_run_count") != 16
        or manifest.get("artifact_firewall", {}).get("evaluation_executed") is not False
        or manifest.get("artifact_firewall", {}).get("generation_executed") is not False
    ):
        raise MethodContractError("R2 manifest provenance/policy differs")
    policy = manifest.get("policy", {})
    if (
        policy.get("controller", {}).get("s_max") != BASE_S_MAX
        or policy.get("event_backend") != "two-separate-teacher-forced-forwards"
        or policy.get("event_nfe") != 2
        or policy.get("trial_backend")
        != "quantized-full-linear-commit-emulator"
        or policy.get("model_specific_rescue") is not False
    ):
        raise MethodContractError("R2 method policy differs")
    if (
        summary.get("status")
        != "COMPLETE_TECHNICAL_MECHANISM_ONLY_NO_EVALUATION"
        or summary.get("record_count") != 16
        or summary.get("evaluation_count") != 0
        or summary.get("scientific_outcome_count") != 0
        or (root / "evaluation.jsonl").stat().st_size != 0
    ):
        raise MethodContractError("R2 summary/evaluation firewall differs")

    controller_rows = _strict_jsonl(root / "controller_steps.jsonl")
    compute_rows = _strict_jsonl(root / "compute.jsonl")
    mechanism_rows = _strict_jsonl(root / "mechanism.jsonl")
    if (
        len(controller_rows) != 16
        or len(compute_rows) != 16
        or len(mechanism_rows) != 16
    ):
        raise MethodContractError("R2 artifact record count differs")
    _assert_raw_free(controller_rows, "R2 controller artifacts")
    _assert_raw_free(compute_rows, "R2 compute artifacts")
    _assert_raw_free(mechanism_rows, "R2 mechanism artifacts")
    controller_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    compute_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for rows, destination, label in (
        (controller_rows, controller_by_key, "controller"),
        (compute_rows, compute_by_key, "compute"),
    ):
        for row in rows:
            key = (str(row.get("arm")), str(row.get("case_id")))
            if key in destination:
                raise MethodContractError(f"duplicate R2 {label} record")
            destination[key] = row
    expected_keys = {
        (arm, case_id)
        for arm in (
            "native-memit",
            "static-synchronous",
            "one-refresh",
            "full-ode-edit",
        )
        for case_id in CANONICAL_CASE_ORDER
    }
    if set(controller_by_key) != expected_keys or set(compute_by_key) != expected_keys:
        raise MethodContractError("R2 arm/case record matrix differs")

    affected: list[AffectedTrajectory] = []
    for order_position, case_id in enumerate(CANONICAL_CASE_ORDER):
        key = ("full-ode-edit", case_id)
        row = controller_by_key[key]
        compute = compute_by_key[key]
        if (
            row.get("order_position") != order_position
            or compute.get("order_position") != order_position
        ):
            raise MethodContractError("R2 canonical order position differs")
        if row.get("status") != "resolution_cap_unresolved":
            continue
        result = row.get("result")
        sequential = row.get("sequential_state")
        omega = row.get("Omega")
        hashes = row.get("hashes")
        if not all(
            isinstance(value, Mapping)
            for value in (result, sequential, omega, hashes)
        ):
            raise MethodContractError("R2 affected record schema differs")
        if (
            result.get("status") != "resolution_cap_unresolved"
            or result.get("omega_appended") is not False
            or result.get("direct_z_compute_count") != 1
            or result.get("failure_type") != "resolution_cap_unresolved"
            or result.get("terminal_state_id") != row.get("post_edit_state_id")
            or row.get("pre_edit_state_id") != row.get("post_edit_state_id")
            or row.get("pre_edit_target_weight_state_id")
            != row.get("post_edit_target_weight_state_id")
            or sequential.get("history_length_before") != 0
            or sequential.get("history_length_after") != 0
            or sequential.get("accepted_terminal_count") != 0
            or sequential.get("current_edit_terminal_appended") is not False
            or compute.get("K_acc") != BASE_S_MAX
            or compute.get("N_reject") != 0
            or compute.get("N_z") != 1
            or compute.get("N_write") != BASE_S_MAX
            or compute.get("N_trial") != BASE_S_MAX
            or compute.get("N_field") != BASE_S_MAX
            or compute.get("N_bw") != BASE_S_MAX
            or compute.get("N_eval") != 0
            or compute.get("status") != "resolution_cap_unresolved"
        ):
            raise MethodContractError(
                "R2 affected trajectory is not an exact zero-history cap failure"
            )
        _zero_omega(omega.get("before"), "R2 Omega before")
        _zero_omega(omega.get("after"), "R2 Omega after")
        _zero_omega(sequential.get("Omega_before"), "R2 sequential Omega before")
        _zero_omega(sequential.get("Omega_after"), "R2 sequential Omega after")
        if omega.get("receipts_through_current_edit") != []:
            raise MethodContractError("R2 affected trajectory retained Omega receipts")
        steps = result.get("steps")
        if not isinstance(steps, list):
            raise MethodContractError("R2 affected trajectory steps are absent")
        prefix = canonical_six_round_prefix(steps)
        if (
            len(steps) != BASE_S_MAX
            or len(prefix["attempts"]) != BASE_S_MAX
            or any(step["accepted"] is not True for step in prefix["attempts"])
        ):
            raise MethodContractError(
                "R2 affected prefix is not six rejection-free accepts"
            )

        direct_z = hashes.get("direct_z")
        if not isinstance(direct_z, Mapping):
            raise MethodContractError("R2 affected direct-z identity is absent")
        relative = (
            f"direct_z/{spec.model_alias}-full-ode-edit-position-{order_position}-"
            f"case-{case_id}.pt"
        )
        identity = identities.get(relative)
        if (
            identity is None
            or direct_z.get("artifact_sha256") != identity.sha256
            or direct_z.get("artifact_size") != identity.size
            or direct_z.get("source_state_id") != row.get("pre_edit_state_id")
            or hashes.get("proposal_id") != R2_PROPOSAL_ID
        ):
            raise MethodContractError("R2 affected direct-z/source identity differs")
        affected.append(
            AffectedTrajectory(
                model_alias=spec.model_alias,
                case_id=case_id,
                order_position=order_position,
                pre_edit_state_id=str(row["pre_edit_state_id"]),
                pre_edit_target_weight_state_id=str(
                    row["pre_edit_target_weight_state_id"]
                ),
                direct_z_relative_path=relative,
                direct_z_identity=identity,
                direct_z_tensor_sha256=_sha(
                    direct_z.get("tensor_sha256"), "R2 direct-z tensor SHA-256"
                ),
                direct_z_source_state_id=str(direct_z["source_state_id"]),
                prefix_payload=prefix,
                prefix_fingerprint=canonical_hash(prefix),
            )
        )

    expected = EXPECTED_AFFECTED_CASES[spec.model_alias]
    if tuple(item.case_id for item in affected) != expected:
        raise MethodContractError(
            "R2 affected-case eligibility differs from the prelocked result"
        )
    if "12498" in expected or any(item.case_id == "12498" for item in affected):
        raise MethodContractError(
            "trust-rejection case 12498 is not continuation eligible"
        )
    if affected:
        baseline_ids = {item.pre_edit_target_weight_state_id for item in affected}
        if len(baseline_ids) != 1:
            raise MethodContractError(
                "R2 affected trajectories do not share the same W0 identity"
            )

    return ValidatedR2Source(
        spec=replace(spec, root=root),
        manifest_sha256=identities["manifest.json"].sha256,
        summary_sha256=identities["summary.json"].sha256,
        affected=tuple(affected),
        file_identities=file_identities,
    )


def validate_r2_sources(
    repo: str | Path,
    specs: Sequence[R2SourceSpec] = DEFAULT_R2_SOURCE_SPECS,
) -> ContinuationSelection:
    root = Path(repo).resolve(strict=True)
    aliases = tuple(spec.model_alias for spec in specs)
    if aliases != ("llama3-8b-inst", "qwen2.5-7b-inst"):
        raise MethodContractError("R2 source pair/order differs")
    selection = ContinuationSelection(
        tuple(_validate_source(spec, repo=root) for spec in specs)
    )
    if selection.pooled_affected_count != 3 or selection.pooled_affected_count < 2:
        raise MethodContractError(
            "prelocked pooled continuation trigger is not exactly 3 >= 2"
        )
    return selection


def continuation_config(base: ControllerConfig) -> ControllerConfig:
    if base.s_max != BASE_S_MAX:
        raise MethodContractError("base controller S_max is not 6")
    extended = replace(base, s_max=CONTINUATION_S_MAX)
    before = base.to_dict()
    after = extended.to_dict()
    changed = {key for key in before if before[key] != after[key]}
    if changed != {"s_max"} or after["s_max"] != CONTINUATION_S_MAX:
        raise MethodContractError(
            "continuation changed a controller field other than S_max"
        )
    return extended


def dispatch_eligibility(
    selection: ContinuationSelection,
    model_alias: str,
    *,
    output_root: Path,
) -> tuple[AffectedTrajectory, ...]:
    """Return eligibility before output creation; Qwen is a mutation-free no-op."""

    affected = selection.for_alias(model_alias)
    if not affected and output_root.exists():
        raise MethodContractError(
            "zero-eligibility dispatch must not reuse/create an output root"
        )
    return affected


def continuation_hit_round(result: ArmRunResult | Mapping[str, Any]) -> int | None:
    raw = result.to_dict() if isinstance(result, ArmRunResult) else result
    steps = raw.get("steps")
    if not isinstance(steps, Sequence):
        raise MethodContractError("continuation result steps are absent")
    accepted = sum(
        1
        for step in steps
        if isinstance(step, Mapping) and step.get("accepted") is True
    )
    if raw.get("status") == "event_hit":
        if accepted not in (7, 8):
            raise MethodContractError(
                "continuation event hit did not occur at round 7 or 8"
            )
        return accepted
    if accepted > CONTINUATION_S_MAX:
        raise MethodContractError("continuation exceeded S_max=8")
    return None


def promotion_decision(hit_rounds: Sequence[int | None]) -> Mapping[str, Any]:
    if len(hit_rounds) != 3:
        raise MethodContractError(
            "continuation promotion requires exactly three affected cases"
        )
    if any(value not in (None, 7, 8) for value in hit_rounds):
        raise MethodContractError(
            "continuation promotion received an invalid hit round"
        )
    hits = sum(value in (7, 8) for value in hit_rounds)
    return {
        "affected_count": 3,
        "first_hit_round_7_or_8_count": hits,
        "minimum_required": PROMOTION_MINIMUM_HITS,
        "common_future_s_max_candidate": (
            CONTINUATION_S_MAX if hits >= PROMOTION_MINIMUM_HITS else BASE_S_MAX
        ),
        "candidate_pass": hits >= PROMOTION_MINIMUM_HITS,
        "model_specific_rescue": False,
    }


class PinnedDirectZEasyEditBackend(EasyEditMemitBackend):
    """Concrete backend that can only load one fully pinned R2 direct-z file."""

    def __init__(
        self,
        *,
        expected_direct_z_identity: ExpectedFileIdentity,
        expected_direct_z_tensor_sha256: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._expected_direct_z_identity = expected_direct_z_identity
        self._expected_direct_z_tensor_sha256 = _sha(
            expected_direct_z_tensor_sha256, "expected direct-z tensor SHA-256"
        )
        if not self.direct_z_cache_path.is_file():
            raise MethodContractError("pinned R2 direct-z artifact is missing")
        if (
            self.direct_z_cache_path.stat().st_size != expected_direct_z_identity.size
            or file_sha256(self.direct_z_cache_path)
            != expected_direct_z_identity.sha256
        ):
            raise MethodContractError("pinned R2 direct-z artifact identity differs")

    def compute_direct_z(self, request: ControllerRequest) -> Any:
        if request != self.request:
            raise MethodContractError("direct-z request differs from backend edit")
        if self._direct_z is not None:
            raise MethodContractError("direct-z backend was invoked more than once")
        result = self.bridge.load_or_compute_direct_z(
            self.model,
            self.tokenizer,
            (self.motivation_request,),
            self.hparams,
            self.contexts,
            model_id=self.runtime.spec.snapshot_name,
            local_cache_root=self.direct_z_cache_root,
            cache_path=self.direct_z_cache_path,
            expected_identity=self._expected_direct_z_identity,
        )
        if result.source_state_id != self.current_state_id():
            raise MethodContractError(
                "reused direct-z origin differs from continuation entry"
            )
        if result.tensor_sha256 != self._expected_direct_z_tensor_sha256:
            raise MethodContractError("reused direct-z tensor identity differs")
        self._direct_z = result
        self._direct_z_tensor_sha256 = tensor_sha256(result.values)
        if self._direct_z_tensor_sha256 != self._expected_direct_z_tensor_sha256:
            raise MethodContractError("reused direct-z tensor bytes differ")
        return result
