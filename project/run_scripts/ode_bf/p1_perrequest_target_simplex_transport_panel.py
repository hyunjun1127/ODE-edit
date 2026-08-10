"""P1R15 Stage-A lock, source-closure, namespace, and forecast helpers."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import load_rooted_json
from .contracts import MODEL_ALIASES, ODEBFContractError, canonical_hash
from .integrated_physical_writer import (
    PHYSICAL_WRITER_ONLINE_FORWARD_GROUP_CEILING,
    physical_writer_compute_contract,
)
from .p1_common_coldcoord_fixed_e8_panel import (
    COMMON_COLD_CASE_SEAL_FILE,
    load_common_cold_requests,
    verify_common_cold_case_seal,
)
from .perrequest_target_simplex_transport import (
    P1R15_AMENDMENT_ID,
    P1R15_INSTRUCTION_ID,
    P1R15_METHOD_ID,
    p1r15_source_contract,
)
from .perrequest_target_simplex_transport_experiment import (
    P1R15_STAGE_A_R13_REQUEST_ORDER,
    P1R15_STAGE_A_R13_SEAL_ROOT,
)


P1R15_NUMERICAL_LOCK_FILE = (
    "numerical_lock_s05_perrequest_target_simplex_transport.json"
)
P1R15_SOURCE_MANIFEST_FILE = (
    "source_manifest_s05_perrequest_target_simplex_transport.json"
)
P1R15_STAGE_A_RESULT_TOKEN = "perrequest-simplex-transport-p1r15-a1-stage-a-v1"
P1R15_STAGE_B_MATERIAL_COUNT = 0
P1R15_P_ANCHOR_SEAL_FILE = "p1r14_fresh_cf_b10_seal.json"
P1R15_P_ANCHOR_SEAL_ROOT = (
    "11fee1f32d608064f72a1befdab32e2184ceefee4c72f8a00439fc3d748d2adc"
)
P1R15_NUMERICAL_LOCK_ROOT = (
    "c3b543d6b7b9c1a02997d3a360d49bb157900d44cd518148daa75f3f960d3824"
)

P1R15_CORE_SOURCE_PATHS = (
    "project/run_scripts/ode_bf/perrequest_target_simplex_transport.py",
    "project/run_scripts/ode_bf/perrequest_target_simplex_transport_runtime.py",
    "project/run_scripts/ode_bf/perrequest_target_simplex_transport_experiment.py",
    "project/run_scripts/ode_bf/p1_perrequest_target_simplex_transport_panel.py",
)
P1R15_SESSION_SOURCE_PATHS = (
    "project/run_scripts/session05_ode_bf_perrequest_simplex_transport.py",
    "project/run_scripts/session05_ode_bf_perrequest_simplex_transport.sbatch",
    "project/run_scripts/session05_ode_bf_perrequest_simplex_transport_dry_plan.py",
    "project/run_scripts/session05_ode_bf_perrequest_simplex_transport_package.py",
    "project/run_scripts/session05_ode_bf_submit_perrequest_simplex_transport.py",
)
P1R15_TEST_SOURCE_PATHS = (
    "project/run_scripts/ode_bf/tests/test_perrequest_target_simplex_transport.py",
    "project/run_scripts/ode_bf/tests/test_perrequest_target_simplex_transport_launch.py",
    "project/run_scripts/ode_bf/tests/test_perrequest_target_simplex_transport_package.py",
)


@dataclass(frozen=True, slots=True)
class P1R15ResourceForecast:
    alias: str
    gpu_peak_mib: int
    host_peak_mib: int
    wall_seconds: int
    requested_memory_mib: int
    allocation_seconds: int
    fits: bool
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return asdict(self)


def expected_p1r15_stage_a_result_name(alias: str) -> str:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1R15 alias differs")
    return f"s05-{P1R15_STAGE_A_RESULT_TOKEN}-{alias}"


def validate_p1r15_stage_a_output_root(
    *, repo_root: Path, alias: str, output_root: Path
) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    expected = (
        root / "local" / "odebf" / "results" / expected_p1r15_stage_a_result_name(alias)
    )
    observed = output_root.resolve(strict=False)
    if observed != expected or observed.name != expected.name:
        raise ODEBFContractError("P1R15 Stage-A output namespace differs")
    payload = {
        "schema": "ode-edit-s05-p1r15-stage-a-output-namespace/v1",
        "alias": alias,
        "result_name": expected.name,
        "relative_parent": "local/odebf/results",
        "stage_a_only": True,
        "stage_b_material_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def load_and_validate_p1r15_stage_a_seal(
    locks: Path,
) -> tuple[dict[str, Any], str]:
    path = locks / COMMON_COLD_CASE_SEAL_FILE
    loaded, file_sha256 = load_rooted_json(path)
    value = verify_common_cold_case_seal(loaded)
    if (
        value["root_digest"] != P1R15_STAGE_A_R13_SEAL_ROOT
        or value["batch_ordered_request_digest_v1"]
        != [P1R15_STAGE_A_R13_REQUEST_ORDER]
    ):
        raise ODEBFContractError("P1R15 Stage-A R13 seal differs")
    return value, file_sha256


def expected_p1r15_stage_a_context_sha256(
    seal: Mapping[str, Any], alias: str
) -> str:
    """Return the exact frozen R13 controller-context identity for an alias."""

    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1R15 Stage-A context alias differs")
    try:
        value = seal["warm_source"]["source_roots"][alias]["context_sha256"]
    except (KeyError, TypeError) as exc:
        raise ODEBFContractError("P1R15 Stage-A context seal differs") from exc
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ODEBFContractError("P1R15 Stage-A context identity differs")
    return value


def load_and_validate_p1r15_numerical_lock(
    path: Path,
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = load_rooted_json(path)
    required = {
        "instruction_id": P1R15_INSTRUCTION_ID,
        "amendment_id": P1R15_AMENDMENT_ID,
        "method_id": P1R15_METHOD_ID,
        "stage_a_case_root": P1R15_STAGE_A_R13_SEAL_ROOT,
        "stage_a_request_order_sha256": P1R15_STAGE_A_R13_REQUEST_ORDER,
        "stage_b_material_count": 0,
        "stage_b_access_count": 0,
        "online_forward_group_count": 104,
        "online_forward_group_ceiling": 110,
        "model_forwards_per_full_field": 132,
        "backwards_per_full_field": 22,
        "root_digest": P1R15_NUMERICAL_LOCK_ROOT,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ODEBFContractError("P1R15 numerical lock differs")
    if value.get("source_contract_sha256") != p1r15_source_contract()[
        "identity_sha256"
    ]:
        raise ODEBFContractError("P1R15 source contract lock differs")
    return value, file_sha256


def _forbidden_alias_science(tree: ast.AST) -> bool:
    aliases = set(MODEL_ALIASES)
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp, ast.Match, ast.Dict)) and any(
            isinstance(child, ast.Constant) and child.value in aliases
            for child in ast.walk(node)
        ):
            return True
    return False


def validate_p1r15_source_closure(
    repo_root: Path,
    *,
    paths: Sequence[str] = P1R15_CORE_SOURCE_PATHS,
) -> dict[str, Any]:
    root = repo_root.resolve(strict=True)
    members: list[dict[str, Any]] = []
    forbidden_names = {
        "TARGET_HOLD",
        "ZERO_POSITIVE_DIRECTION",
        "active=np.flatnonzero",
        "stage_b_salt",
        "stage_b_case",
        "stage_b_seal",
    }
    for relative in paths:
        path = root / relative
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        is_policy_helper = relative == P1R15_CORE_SOURCE_PATHS[-1]
        if not is_policy_helper and any(token in source for token in forbidden_names):
            raise ODEBFContractError("P1R15 forbidden legacy/science path differs")
        if not is_policy_helper and _forbidden_alias_science(tree):
            raise ODEBFContractError("P1R15 alias-specific science differs")
        members.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = {
        "schema": "ode-edit-s05-p1r15-source-closure/v1",
        "members": members,
        "member_root": canonical_hash(members),
        "one_arm_registry_count": 1,
        "stage_b_material_count": P1R15_STAGE_B_MATERIAL_COUNT,
        "alias_science_branch_count": 0,
        "r13_runtime_import_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def p1r15_stage_a_forecast(alias: str) -> P1R15ResourceForecast:
    if alias not in MODEL_ALIASES:
        raise ODEBFContractError("P1R15 forecast alias differs")
    gpu = 31_000 if alias == "llama3-8b-inst" else 30_500
    host = 48_000
    wall = 23 * 60 * 60
    allocation = 23 * 60 * 60 + 59 * 60
    payload = {
        "schema": "ode-edit-s05-p1r15-stage-a-resource-forecast/v1",
        "alias": alias,
        "gpu_peak_mib": gpu,
        "host_peak_mib": host,
        "wall_seconds": wall,
        "requested_memory_mib": 65_000,
        "allocation_seconds": allocation,
        "fits": gpu < 65_000 and host < 65_000 and wall < allocation,
        "online_forward_groups": physical_writer_compute_contract(
            fixed_online_forward_groups=0
        ).total_online_forward_groups,
        "online_forward_group_ceiling": PHYSICAL_WRITER_ONLINE_FORWARD_GROUP_CEILING,
    }
    return P1R15ResourceForecast(
        alias,
        gpu,
        host,
        wall,
        65_000,
        allocation,
        bool(payload["fits"]),
        canonical_hash(payload),
    )


def load_and_validate_p1r15_source_manifest(
    repo_root: Path, path: Path, *, source_head: str
) -> tuple[dict[str, Any], str]:
    value, file_sha256 = load_rooted_json(
        path,
        expected_schema="ode-edit-s05-p1r15-perrequest-simplex-transport-a1/v1-source-manifest",
    )
    entries = value.get("entries")
    required = tuple(
        sorted(
            (
                *P1R15_CORE_SOURCE_PATHS,
                *P1R15_SESSION_SOURCE_PATHS,
                *P1R15_TEST_SOURCE_PATHS,
            )
        )
    )
    if (
        value.get("source_base")
        != "6184857ef7171920e5193e7609f5fc1596bc67b0"
        or value.get("execution_head_binding") != "RUNTIME_EXACT_GIT_HEAD"
        or len(source_head) != 40
        or not isinstance(entries, list)
        or tuple(item.get("path") for item in entries) != required
    ):
        raise ODEBFContractError("P1R15 source manifest inventory differs")
    for item in entries:
        member = repo_root / str(item["path"])
        data = member.read_bytes()
        expected_mode = 0o100755 if member.stat().st_mode & 0o111 else 0o100644
        expected_blob = hashlib.sha1(
            f"blob {len(data)}\0".encode("ascii") + data
        ).hexdigest()
        if (
            not member.is_file()
            or member.is_symlink()
            or member.stat().st_size != item["size_bytes"]
            or hashlib.sha256(data).hexdigest() != item["sha256"]
            or item.get("mode") != expected_mode
            or item.get("git_blob_oid") != expected_blob
            or item.get("object_algorithm") != "GIT_BLOB_SHA1"
        ):
            raise ODEBFContractError("P1R15 source manifest member differs")
    if value.get("entry_root") != canonical_hash(entries):
        raise ODEBFContractError("P1R15 source manifest root differs")
    return value, file_sha256


__all__ = [
    "P1R15_CORE_SOURCE_PATHS",
    "P1R15_NUMERICAL_LOCK_FILE",
    "P1R15_NUMERICAL_LOCK_ROOT",
    "P1R15_P_ANCHOR_SEAL_FILE",
    "P1R15_P_ANCHOR_SEAL_ROOT",
    "P1R15_SOURCE_MANIFEST_FILE",
    "P1R15_SESSION_SOURCE_PATHS",
    "P1R15_TEST_SOURCE_PATHS",
    "expected_p1r15_stage_a_context_sha256",
    "expected_p1r15_stage_a_result_name",
    "load_and_validate_p1r15_numerical_lock",
    "load_and_validate_p1r15_source_manifest",
    "load_and_validate_p1r15_stage_a_seal",
    "load_common_cold_requests",
    "p1r15_stage_a_forecast",
    "validate_p1r15_source_closure",
    "validate_p1r15_stage_a_output_root",
]
