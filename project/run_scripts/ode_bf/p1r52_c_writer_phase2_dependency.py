"""Read-only Phase-2 terminal gate for the dependent Phase-3 launcher."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .artifacts import sha256_file
from .contracts import ODEBFStateError, canonical_hash
from .p1r52_b100x10_stream import BATCH_SIZE, ROUND_COUNT
from .p1r52_c_writer_kstep_independent import ARMS, ROLES, expected_result_name
from .scalable_batched_runtime import P1R23_GRID_COUNT


@dataclass(frozen=True, slots=True)
class Phase2DependencyGateReceipt:
    status: str
    phase2_source_head: str
    phase2_parent: str
    arm_count: int
    valid_case_count: int
    valid_request_count: int
    k_writer_call_count: int
    terminal_sha256: tuple[str, ...]
    manifest_sha256: tuple[str, ...]
    w0_restore_count: int
    technical_failure_count: int
    scientific_failure_count: int
    imputation_count: int
    identity_sha256: str

    def raw_free_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "phase2_source_head": self.phase2_source_head,
            "phase2_parent": self.phase2_parent,
            "arm_count": self.arm_count,
            "valid_case_count": self.valid_case_count,
            "valid_request_count": self.valid_request_count,
            "k_writer_call_count": self.k_writer_call_count,
            "terminal_sha256": list(self.terminal_sha256),
            "manifest_sha256": list(self.manifest_sha256),
            "w0_restore_count": self.w0_restore_count,
            "technical_failure_count": self.technical_failure_count,
            "scientific_failure_count": self.scientific_failure_count,
            "imputation_count": self.imputation_count,
            "identity_sha256": self.identity_sha256,
        }


def _read_verified(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ODEBFStateError(f"Phase3 dependency {label} path differs")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEBFStateError(f"Phase3 dependency {label} JSON differs") from exc
    if not isinstance(value, dict) or not isinstance(value.get("identity_sha256"), str):
        raise ODEBFStateError(f"Phase3 dependency {label} identity is absent")
    sealed = dict(value)
    claimed = sealed.pop("identity_sha256")
    if canonical_hash(sealed) != claimed:
        raise ODEBFStateError(f"Phase3 dependency {label} identity differs")
    return value


def verify_phase2_terminal_completeness(
    phase2_parent: Path,
    *,
    expected_source_head: str,
) -> Phase2DependencyGateReceipt:
    """Validate all three Phase-2 arms before any Phase-3 model load."""

    parent = phase2_parent.resolve(strict=True)
    terminal_shas: list[str] = []
    manifest_shas: list[str] = []
    valid_cases = valid_requests = k_calls = restores = 0
    technical = scientific = imputations = 0
    for role, arm in zip(ROLES, ARMS, strict=True):
        root = parent / expected_result_name(role)
        terminal_path = root / "terminal.json"
        manifest_path = root / "manifest.json"
        terminal = _read_verified(terminal_path, label=f"{arm} terminal")
        manifest = _read_verified(manifest_path, label=f"{arm} manifest")
        terminal_sha = sha256_file(terminal_path)
        case_rows = terminal.get("cases")
        case_shas = terminal.get("case_terminal_sha256")
        if (
            terminal.get("schema") != "ode-edit-s05-p1r52-c-writer-phase2-terminal/v1"
            or terminal.get("status") != "TERMINAL_VALID"
            or terminal.get("source_head") != expected_source_head
            or terminal.get("role") != role
            or terminal.get("arm") != arm
            or terminal.get("case_count") != ROUND_COUNT
            or terminal.get("valid_request_count") != ROUND_COUNT * BATCH_SIZE
            or terminal.get("K_writer_call_count") != ROUND_COUNT * P1R23_GRID_COUNT
            or terminal.get("alpha_cache_status") != "ALPHA_CACHE_OFF_CONTROL"
            or terminal.get("cross_case_state_count") != 0
            or terminal.get("W0_restored") is not True
            or terminal.get("technical_failure_count") != 0
            or terminal.get("scientific_failure_count") != 0
            or terminal.get("imputation_count") != 0
            or not isinstance(case_rows, list)
            or len(case_rows) != ROUND_COUNT
            or not isinstance(case_shas, list)
            or len(case_shas) != ROUND_COUNT
        ):
            raise ODEBFStateError(f"Phase3 dependency {arm} terminal completeness differs")
        dtype = terminal.get("dtype_contract")
        if (
            not isinstance(dtype, dict)
            or dtype.get("status") != "FULL_FP32_PASS"
            or dtype.get("numeric_storage_cast_count") != 0
            or dtype.get("bf16_fp16_path_count") != 0
        ):
            raise ODEBFStateError(f"Phase3 dependency {arm} dtype differs")
        for index, (row, expected_sha) in enumerate(zip(case_rows, case_shas, strict=True), start=1):
            case_path = root / "raw" / "cases" / f"case-{index:02d}" / "terminal.json"
            case = _read_verified(case_path, label=f"{arm} case {index}")
            if (
                not isinstance(row, dict)
                or row.get("case_index") != index
                or row.get("terminal_sha256") != expected_sha
                or sha256_file(case_path) != expected_sha
                or case.get("case_index") != index
                or case.get("arm") != arm
                or case.get("request_count") != BATCH_SIZE
                or len(case.get("kstep_executions", ())) != P1R23_GRID_COUNT
                or case.get("alpha_cache_status") != "ALPHA_CACHE_OFF_CONTROL"
                or case.get("cross_case_W_cache_history_carry_count") != 0
                or case.get("retry_count") != 0
                or case.get("imputation_count") != 0
            ):
                raise ODEBFStateError(f"Phase3 dependency {arm} case {index} differs")
        if (
            manifest.get("schema") != "ode-edit-s05-p1r52-c-writer-phase2-manifest/v1"
            or manifest.get("source_head") != expected_source_head
            or manifest.get("role") != role
            or manifest.get("terminal_sha256") != terminal_sha
            or manifest.get("case_terminal_sha256") != case_shas
            or manifest.get("W0_restored") is not True
        ):
            raise ODEBFStateError(f"Phase3 dependency {arm} manifest differs")
        terminal_shas.append(terminal_sha)
        manifest_shas.append(sha256_file(manifest_path))
        valid_cases += int(terminal["case_count"])
        valid_requests += int(terminal["valid_request_count"])
        k_calls += int(terminal["K_writer_call_count"])
        restores += 1
        technical += int(terminal["technical_failure_count"])
        scientific += int(terminal["scientific_failure_count"])
        imputations += int(terminal["imputation_count"])

    payload: dict[str, Any] = {
        "status": "PHASE2_TERMINAL_VALID_DEPENDENCY_PASS",
        "phase2_source_head": expected_source_head,
        "phase2_parent": str(parent),
        "arm_count": len(ARMS),
        "valid_case_count": valid_cases,
        "valid_request_count": valid_requests,
        "k_writer_call_count": k_calls,
        "terminal_sha256": terminal_shas,
        "manifest_sha256": manifest_shas,
        "w0_restore_count": restores,
        "technical_failure_count": technical,
        "scientific_failure_count": scientific,
        "imputation_count": imputations,
    }
    identity = canonical_hash(payload)
    return Phase2DependencyGateReceipt(
        payload["status"],
        expected_source_head,
        str(parent),
        len(ARMS),
        valid_cases,
        valid_requests,
        k_calls,
        tuple(terminal_shas),
        tuple(manifest_shas),
        restores,
        technical,
        scientific,
        imputations,
        identity,
    )


__all__ = ["Phase2DependencyGateReceipt", "verify_phase2_terminal_completeness"]
