"""Fail-closed locks and one-arm registry for P1R14."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import load_rooted_json, sha256_file
from .contracts import MODEL_ALIASES, ODEBFContractError, canonical_hash
from .integrated_physical_writer import (
    INTEGRATED_PHYSICAL_WRITER_AMENDMENT_IDS,
    INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
    INTEGRATED_PHYSICAL_WRITER_METHOD_ID,
    integrated_physical_writer_source_contract,
    physical_writer_compute_contract,
)
from .integrated_physical_writer_selection import (
    R14_EXCLUSION_SCHEMA,
    R14_FRESH_SEAL_SCHEMA,
    R14_P_ANCHOR_COUNT,
    R14_SELECTION_BASE,
    verify_r14_fresh_seal,
    verify_r14_historical_exclusion,
)


P1R14_SCHEMA = "ode-edit-s05-integrated-physical-writer-p1r14/v1"
P1R14_ARM_ID = INTEGRATED_PHYSICAL_WRITER_METHOD_ID
P1R14_ARM_REGISTRY = (P1R14_ARM_ID,)
P1R14_RESULT_TOKEN = "integrated-physical-writer-p1r14-v1"
P1R14_RUN_ATTEMPT_ID = (
    "ODEEDIT-S05-ODE-BF-INTEGRATED-PHYSICAL-WRITER-P1R14-V1-TECH-R4-NAMESPACE"
)
P1R14_RUN_ATTEMPT_SUFFIX = "tech-r4-v1"
P1R14_EXCLUSION_FILE = "p1r14_historical_exclusion.json"
P1R14_FRESH_SEAL_FILE = "p1r14_fresh_cf_b10_seal.json"
P1R14_NUMERICAL_LOCK_FILE = "numerical_lock_s05_integrated_physical_writer.json"
P1R14_SOURCE_MANIFEST_FILE = "source_manifest_s05_integrated_physical_writer.json"
P1R14_FORECAST_SECONDS = 82_800
P1R14_ALLOCATION_SECONDS = 86_340

P1R14_SCIENTIFIC_SOURCE_PATHS = (
    "project/run_scripts/ode_bf/integrated_physical_writer.py",
    "project/run_scripts/ode_bf/integrated_physical_writer_solver.py",
    "project/run_scripts/ode_bf/integrated_physical_writer_runtime.py",
    "project/run_scripts/ode_bf/integrated_physical_writer_selection.py",
    "project/run_scripts/ode_bf/integrated_physical_writer_experiment.py",
    "project/run_scripts/ode_bf/integrated_physical_writer_terminal.py",
    "project/run_scripts/ode_bf/p1_integrated_physical_writer_panel.py",
)
P1R14_SESSION_SOURCE_PATHS = (
    "project/run_scripts/session05_ode_bf_integrated_physical_writer.py",
    "project/run_scripts/session05_ode_bf_integrated_physical_writer_dry_plan.py",
    "project/run_scripts/session05_ode_bf_integrated_physical_writer_package.py",
    "project/run_scripts/session05_ode_bf_integrated_physical_writer.sbatch",
    "project/run_scripts/session05_ode_bf_submit_integrated_physical_writer.py",
)
P1R14_TEST_SOURCE_PATHS = (
    "project/run_scripts/ode_bf/tests/test_integrated_physical_writer.py",
    "project/run_scripts/ode_bf/tests/test_integrated_physical_writer_system.py",
    "project/run_scripts/ode_bf/tests/test_integrated_physical_writer_launch.py",
    "project/run_scripts/ode_bf/tests/test_integrated_physical_writer_package.py",
)

_FORBIDDEN_MODULES = (
    "p1_runtime",
    "fixed_e8_runtime",
    "common_coldcoord_fixed_e8_runtime",
    "fixed_e8_soft_routing",
    "p1_universal_observability_panel",
    "ode_bf_observability",
)
_FORBIDDEN_DISPATCH_TOKENS = (
    "RS-NEUTRAL",
    "RS-SOFT",
    "BG-NEUTRAL",
    "BG-SOFT",
    "TARGET-HOLD",
    "TARGET_HOLD",
    "factorial_arm",
    "positive_direction_mask",
    "ZERO_POSITIVE_DIRECTION",
)

_MODEL_ALIAS_LITERALS = frozenset(MODEL_ALIASES)


@dataclass(frozen=True, slots=True)
class P1R14Forecast:
    arm_count: int
    alias_count_per_job: int
    gpu_count: int
    cpu_count: int
    host_memory_mib: int
    allocation_seconds: int
    forecast_seconds: int
    online_forward_group_count: int
    online_forward_group_ceiling: int
    fits: bool

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def expected_p1r14_result_name(alias: str) -> str:
    """Return the immutable method-level result name used by legacy provenance."""

    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1R14 alias differs")
    return f"s05-integrated-physical-writer-p1r14-{alias}-v1"


def expected_p1r14_attempt_result_name(alias: str) -> str:
    """Return the exact create-once basename for the authorized TECH-R4 attempt."""

    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1R14 attempt alias differs")
    return (
        f"s05-integrated-physical-writer-p1r14-{alias}-"
        f"{P1R14_RUN_ATTEMPT_SUFFIX}"
    )


def validate_p1r14_attempt_output_namespace(
    *,
    repo_root: Path,
    alias: str,
    output_root: Path,
    run_attempt_id: str,
) -> dict[str, Any]:
    """Validate one exact alias-specific attempt path without creating it."""

    if run_attempt_id != P1R14_RUN_ATTEMPT_ID:
        raise ODEBFContractError("P1R14 run-attempt identity differs")
    expected_name = expected_p1r14_attempt_result_name(alias)
    root_input = Path(repo_root)
    if not root_input.is_absolute() or root_input.is_symlink():
        raise ODEBFContractError("P1R14 repository namespace differs")
    root = root_input.resolve(strict=True)
    if root_input != root or not root.is_dir():
        raise ODEBFContractError("P1R14 repository namespace differs")
    parent = root / "local" / "odebf" / "results"
    if (
        parent.is_symlink()
        or not parent.is_dir()
        or parent.resolve(strict=True) != parent
    ):
        raise ODEBFContractError("P1R14 output parent namespace differs")
    candidate = Path(output_root)
    expected = parent / expected_name
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or candidate.parent != parent
        or candidate.name != expected_name
        or candidate != expected
    ):
        raise ODEBFContractError("P1R14 run-attempt output namespace differs")
    payload = {
        "schema": f"{P1R14_SCHEMA}-run-attempt-output-namespace",
        "method_result_token": P1R14_RESULT_TOKEN,
        "run_attempt_id": P1R14_RUN_ATTEMPT_ID,
        "run_attempt_suffix": P1R14_RUN_ATTEMPT_SUFFIX,
        "model_alias": alias,
        "result_parent": "local/odebf/results",
        "result_name": expected_name,
        "decision_influence_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    payload["result_root"] = str(expected)
    return payload


def p1r14_common_science_config() -> dict[str, Any]:
    payload = {
        "method_id": P1R14_ARM_ID,
        "arm_registry": list(P1R14_ARM_REGISTRY),
        "target_allocation": "BATCH_GLOBAL",
        "target_dynamics": "DYNAMIC",
        "routing": "PHYSICAL_NOHOOK_E_THEN_P_THEN_CAPACITY",
        "grid_count": 8,
        "h": 0.125,
        "retry_count": 0,
        "first_hit_decision_influence_count": 0,
        "hard_h_p_budget_veto_count": 0,
        "native_or_direct_z_controller_access_count": 0,
        "scientific_promotion_authorized": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def p1r14_forecast() -> P1R14Forecast:
    compute = physical_writer_compute_contract(fixed_online_forward_groups=0)
    fits = bool(
        compute.total_online_forward_groups < compute.online_forward_group_ceiling
        and P1R14_FORECAST_SECONDS < P1R14_ALLOCATION_SECONDS
    )
    return P1R14Forecast(
        1,
        1,
        1,
        8,
        65_000,
        P1R14_ALLOCATION_SECONDS,
        P1R14_FORECAST_SECONDS,
        compute.total_online_forward_groups,
        compute.online_forward_group_ceiling,
        fits,
    )


def load_and_validate_p1r14_seals(
    locks: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    exclusion, _ = load_rooted_json(
        locks / P1R14_EXCLUSION_FILE,
        expected_schema=R14_EXCLUSION_SCHEMA,
    )
    exclusion = verify_r14_historical_exclusion(exclusion)
    seal, _ = load_rooted_json(
        locks / P1R14_FRESH_SEAL_FILE,
        expected_schema=R14_FRESH_SEAL_SCHEMA,
    )
    seal = verify_r14_fresh_seal(
        seal, exclusion_root=exclusion["root_digest"]
    )
    if (
        seal.get("functional_p_anchor_count") != R14_P_ANCHOR_COUNT
        or seal.get("selection_base_commit") != R14_SELECTION_BASE
    ):
        raise ODEBFContractError("P1R14 fresh seal inventory differs")
    return exclusion, seal


def load_and_validate_p1r14_numerical_lock(
    path: Path,
    *,
    exclusion_root: str,
    fresh_seal_root: str,
    artifact_lock_root: str,
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = load_rooted_json(
        path, expected_schema=f"{P1R14_SCHEMA}-numerical-lock"
    )
    source = integrated_physical_writer_source_contract()
    compute = physical_writer_compute_contract(fixed_online_forward_groups=0)
    expected = {
        "instruction_id": INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
        "amendment_ids": list(INTEGRATED_PHYSICAL_WRITER_AMENDMENT_IDS),
        "base": R14_SELECTION_BASE,
        "method_id": P1R14_ARM_ID,
        "arm_registry": list(P1R14_ARM_REGISTRY),
        "common_science_config_sha256": p1r14_common_science_config()[
            "identity_sha256"
        ],
        "source_contract_sha256": source["identity_sha256"],
        "historical_exclusion_root": exclusion_root,
        "fresh_seal_root": fresh_seal_root,
        "artifact_lock_root": artifact_lock_root,
        "online_forward_group_count": compute.total_online_forward_groups,
        "online_forward_group_ceiling": compute.online_forward_group_ceiling,
        "fixed_online_forward_group_count": 0,
        "endpoint_transaction_count": 1,
        "persistent_commit_count": 0,
        "history_append_count": 0,
        "scientific_promotion_authorized": False,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ODEBFContractError("P1R14 numerical lock differs")
    return value, file_sha256


def load_and_validate_p1r14_source_manifest(
    repo_root: Path,
    path: Path,
    *,
    source_head: str,
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = load_rooted_json(
        path, expected_schema=f"{P1R14_SCHEMA}-source-manifest"
    )
    entries = value.get("entries")
    required = tuple(
        sorted(
            (
                *P1R14_SCIENTIFIC_SOURCE_PATHS,
                *P1R14_SESSION_SOURCE_PATHS,
                *P1R14_TEST_SOURCE_PATHS,
            )
        )
    )
    if (
        value.get("source_base") != R14_SELECTION_BASE
        or value.get("execution_head_binding") != "RUNTIME_EXACT_GIT_HEAD"
        or len(source_head) != 40
        or not isinstance(entries, list)
        or tuple(item.get("path") for item in entries) != required
    ):
        raise ODEBFContractError("P1R14 source manifest inventory differs")
    for item in entries:
        relative = str(item["path"])
        source_path = repo_root / relative
        if source_path.is_symlink():
            raise ODEBFContractError("P1R14 source manifest member differs")
        observed = source_path.resolve(strict=True)
        data = observed.read_bytes()
        expected_mode = 0o100755 if observed.stat().st_mode & 0o111 else 0o100644
        expected_blob = hashlib.sha1(
            f"blob {len(data)}\0".encode("ascii") + data
        ).hexdigest()
        if (
            not observed.is_file()
            or item.get("size_bytes") != observed.stat().st_size
            or item.get("sha256") != sha256_file(observed)
            or item.get("mode") != expected_mode
            or item.get("git_blob_oid") != expected_blob
            or item.get("object_algorithm") != "GIT_BLOB_SHA1"
        ):
            raise ODEBFContractError("P1R14 source manifest member differs")
    if value.get("entry_root") != canonical_hash(entries):
        raise ODEBFContractError("P1R14 source manifest root differs")
    return value, file_sha256


def validate_p1r14_source_closure(
    repo_root: Path,
    paths: Sequence[str] = (
        *P1R14_SCIENTIFIC_SOURCE_PATHS,
        *P1R14_SESSION_SOURCE_PATHS,
    ),
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in paths:
        path = repo_root / relative
        source = path.read_text(encoding="utf-8")
        if path.suffix != ".py":
            if any(token in source for token in _FORBIDDEN_DISPATCH_TOKENS):
                raise ODEBFContractError(
                    "P1R14 session helper contains a forbidden arm dispatch"
                )
            records.append(
                {
                    "path": relative,
                    "import_sha256": canonical_hash([]),
                    "ast_sha256": canonical_hash(
                        {"non_python_sha256": sha256_file(path)}
                    ),
                }
            )
            continue
        tree = ast.parse(source, filename=relative)
        imports: list[str] = []
        constants: list[str] = []
        identifiers: list[str] = []
        policy_constant_ids: set[int] = set()
        for statement in tree.body:
            if (
                isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id in ("_FORBIDDEN_MODULES", "_FORBIDDEN_DISPATCH_TOKENS")
                    for target in statement.targets
                )
            ):
                policy_constant_ids.update(
                    id(item)
                    for item in ast.walk(statement.value)
                    if isinstance(item, ast.Constant)
                )
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.append(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.append(node.attr)
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
                imports.extend(alias.name for alias in node.names)
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in policy_constant_ids
            ):
                constants.append(node.value)
            if isinstance(node, ast.Dict):
                literal_keys = {
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                }
                if literal_keys & _MODEL_ALIAS_LITERALS:
                    raise ODEBFContractError(
                        "P1R14 contains alias-specific mapping dispatch"
                    )
            if isinstance(node, ast.Match):
                for case in node.cases:
                    pattern_literals = {
                        item.value
                        for item in ast.walk(case.pattern)
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                    }
                    if pattern_literals & _MODEL_ALIAS_LITERALS:
                        raise ODEBFContractError(
                            "P1R14 contains alias-specific match dispatch"
                        )
            if isinstance(node, (ast.If, ast.IfExp)):
                test_literals = {
                    item.value
                    for item in ast.walk(node.test)
                    if isinstance(item, ast.Constant)
                    and isinstance(item.value, str)
                }
                if test_literals & _MODEL_ALIAS_LITERALS:
                    raise ODEBFContractError(
                        "P1R14 contains alias-specific conditional dispatch"
                    )
        if any(
            token in imported
            for imported in imports
            for token in _FORBIDDEN_MODULES
        ):
            raise ODEBFContractError("P1R14 imported a forbidden legacy runtime")
        joined = "\n".join(constants)
        joined_identifiers = "\n".join(identifiers)
        if any(
            token in joined or token in joined_identifiers
            for token in _FORBIDDEN_DISPATCH_TOKENS
        ):
            raise ODEBFContractError("P1R14 contains a forbidden arm dispatch")
        records.append(
            {
                "path": relative,
                "import_sha256": canonical_hash(sorted(imports)),
                "ast_sha256": canonical_hash(ast.dump(tree, include_attributes=False)),
            }
        )
    payload = {
        "schema": f"{P1R14_SCHEMA}-source-closure",
        "paths": records,
        "arm_registry": list(P1R14_ARM_REGISTRY),
        "legacy_runtime_import_count": 0,
        "legacy_arm_dispatch_count": 0,
        "alias_specific_science_config_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


__all__ = [
    "P1R14_ALLOCATION_SECONDS",
    "P1R14_ARM_ID",
    "P1R14_ARM_REGISTRY",
    "P1R14_EXCLUSION_FILE",
    "P1R14_FRESH_SEAL_FILE",
    "P1R14_NUMERICAL_LOCK_FILE",
    "P1R14_RESULT_TOKEN",
    "P1R14_RUN_ATTEMPT_ID",
    "P1R14_RUN_ATTEMPT_SUFFIX",
    "P1R14_SCHEMA",
    "P1R14_SCIENTIFIC_SOURCE_PATHS",
    "P1R14_SESSION_SOURCE_PATHS",
    "P1R14_TEST_SOURCE_PATHS",
    "P1R14_SOURCE_MANIFEST_FILE",
    "expected_p1r14_result_name",
    "expected_p1r14_attempt_result_name",
    "load_and_validate_p1r14_numerical_lock",
    "load_and_validate_p1r14_source_manifest",
    "load_and_validate_p1r14_seals",
    "p1r14_common_science_config",
    "p1r14_forecast",
    "validate_p1r14_source_closure",
    "validate_p1r14_attempt_output_namespace",
]
