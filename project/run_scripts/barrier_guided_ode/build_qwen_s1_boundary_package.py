#!/usr/bin/env python3
"""Seal the Qwen BGODE-R1 S1 scientific-boundary factual package.

This analyzer is intentionally raw-read-only.  It publishes no performance
endpoint because the six-arm panel stopped at the first singular-aware
Rayleighian boundary before a terminal panel could be constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_PREFIX = "ode-edit-bgode-r1-s1-qwen-boundary"
RUN_ID = "s05-bgode-r1-s1-qwen-b1-request000-six-arm-v1"
JOB_ID = "26605"
EXPECTED_SOURCE_HEAD = "44c569a50c178dda655c0d16395dc0087069c2ba"
EXPECTED_SOURCE_TREE = "4104b58bd5860022c7717a8044c32cb70522d55e"
EXPECTED_TERMINAL_SHA = "31d84a931d9492321256c8d7a170057c75520eceb8b7571207367e8a1008febe"
EXPECTED_FIDELITY_SHA = "4d61b6ada645949e258b556eb6bdfc7583539ac65a0fedcf4852af09baa827e9"
EXPECTED_FIDELITY_ID = "b33925fefb2af705254844de8a5c820e98bd887161267943fc907a6d2342a820"
EXPECTED_FAILURE_MESSAGE_SHA = "fe8e88199cb47a9e0484c7eff59d9bd96cd8901e188507f86deade4fda32f835"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_regular(path: Path, *, expected_mode: int | None = None) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"symlink input forbidden: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"regular input required: {path}")
    if expected_mode is not None and stat.S_IMODE(resolved.stat().st_mode) != expected_mode:
        raise RuntimeError(f"mode mismatch for {path}")
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    with require_regular(path, expected_mode=0o600).open("rb") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"object JSON required: {path}")
    return value


def file_receipt(path: Path, *, classification: str) -> dict[str, Any]:
    resolved = require_regular(path)
    info = resolved.stat()
    return {
        "path": str(resolved),
        "classification": classification,
        "bytes": info.st_size,
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
        "sha256": sha256_file(resolved),
        "regular_non_symlink": True,
    }


def write_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"create-once output already exists: {path}")
    path.write_bytes(canonical_bytes(value))
    path.chmod(0o600)


def write_text(path: Path, value: str) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"create-once output already exists: {path}")
    path.write_text(value.rstrip() + "\n", encoding="utf-8")
    path.chmod(0o600)


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def scheduler_row() -> dict[str, Any]:
    raw = subprocess.check_output(
        [
            "sacct", "-j", JOB_ID, "--starttime", "2026-08-27", "-X", "-P", "-n",
            "--format=JobIDRaw,JobName,State,ExitCode,Elapsed,Timelimit,AllocTRES,NodeList",
        ],
        text=True,
    ).strip().splitlines()
    rows = [row for row in raw if row.split("|", 1)[0] == JOB_ID]
    if len(rows) != 1:
        raise RuntimeError("exact scheduler row not found")
    fields = rows[0].split("|")
    if len(fields) < 8:
        raise RuntimeError("scheduler row has wrong width")
    names = ("job_id", "job_name", "state", "exit_code", "elapsed", "time_limit", "alloc_tres", "node_list")
    value = dict(zip(names, fields[:8], strict=True))
    if value["state"] != "FAILED" or value["exit_code"] != "1:0":
        raise RuntimeError("unexpected TECH-R6 scheduler terminal")
    return value


def assert_source_control_flow(repo: Path) -> dict[str, Any]:
    experiment_path = repo / "project/run_scripts/barrier_guided_ode/s1_experiment.py"
    contract_path = repo / "project/run_scripts/barrier_guided_ode/s1_contract.py"
    trajectory_path = repo / "project/run_scripts/barrier_guided_ode/s1_alphaedit_runtime.py"
    controller_path = repo / "project/run_scripts/barrier_guided_ode/rayleighian_controller.py"
    experiment = require_regular(experiment_path).read_text(encoding="utf-8")
    contract = require_regular(contract_path).read_text(encoding="utf-8")
    trajectory = require_regular(trajectory_path).read_text(encoding="utf-8")
    controller = require_regular(controller_path).read_text(encoding="utf-8")
    required = {
        "fixed_arm_order": "S1_ARM_ORDER = tuple(S1Arm)" in contract,
        "plain_branch_precedes_solver": "if arm is S1Arm.PLAIN_DYNAMIC:\n                velocity, numerical = _plain_velocity" in experiment,
        "solver_is_nonplain_branch": "solution = solve_equality_rayleighian(" in experiment,
        "apply_follows_solver": "action = trajectory.apply(proposal, coefficients)" in experiment,
        "arm_loop_is_serial": "for arm in S1_ARM_ORDER[1:]:" in experiment,
        "trajectory_always_restores": "def __exit__(self, exc_type" in trajectory and "self.restore()" in trajectory,
        "range_failure_is_typed": 'raise NumericalRankBoundary("equality direction is outside range(G)")' in controller,
    }
    if not all(required.values()):
        raise RuntimeError(f"source control-flow proof failed: {required}")
    return {
        "checks": required,
        "source_order_inference": {
            "first_rayleighian_arm": "fisher-only-dynamic",
            "node": 0,
            "reason": "Native and Plain precede Fisher; Plain uses _plain_velocity and cannot emit the observed Rayleighian exception; later solver arms are unreachable.",
            "fisher_writer_action_count": 0,
            "reason_writer_zero": "solve_equality_rayleighian raises before coefficient construction and trajectory.apply.",
            "plain_endpoint_publication": "NOT_PUBLISHED_PANEL_ABORTED",
            "plain_internal_action_count": "NOT_RECORDED_RAW_RECEIPT",
        },
        "sources": [
            file_receipt(experiment_path, classification="EXPERIMENT_CONTROL_FLOW"),
            file_receipt(contract_path, classification="ARM_ORDER"),
            file_receipt(trajectory_path, classification="ATOMIC_RESTORE"),
            file_receipt(controller_path, classification="LOCKED_RANK_GATE"),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--llama-terminal", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve(strict=True)
    output = args.output
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"create-once package already exists: {output}")
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)

    head = git_value(repo, "rev-parse", "HEAD")
    tree = git_value(repo, "rev-parse", "HEAD^{tree}")
    if (head, tree) != (EXPECTED_SOURCE_HEAD, EXPECTED_SOURCE_TREE):
        raise RuntimeError(f"source mismatch: {(head, tree)}")

    root = repo / "local/odebf/results/s05-bgode-r1-s1-qwen-b1-request000-six-arm-v1-tech-r6" / RUN_ID
    terminal_path = root / "terminal.json"
    fidelity_path = root / "native-adapter-fidelity.json"
    direct_z_path = root / "private/direct-z.pt"
    terminal = load_json(terminal_path)
    fidelity = load_json(fidelity_path)
    preflight = load_json(args.preflight)
    llama_terminal = load_json(args.llama_terminal)

    if sha256_file(terminal_path) != EXPECTED_TERMINAL_SHA:
        raise RuntimeError("TECH-R6 terminal SHA mismatch")
    if terminal.get("status") != "BGODE_R1_S1_TERMINAL_BOUNDARY" or terminal.get("failure_type") != "NumericalRankBoundary":
        raise RuntimeError("TECH-R6 is not the expected numerical-rank terminal")
    if terminal.get("failure_message_sha256") != EXPECTED_FAILURE_MESSAGE_SHA:
        raise RuntimeError("TECH-R6 failure identity mismatch")
    if sha256_file(fidelity_path) != EXPECTED_FIDELITY_SHA or fidelity.get("identity_sha256") != EXPECTED_FIDELITY_ID:
        raise RuntimeError("native fidelity receipt identity mismatch")
    if fidelity.get("pass") is not True or fidelity.get("relative_frobenius") != 0.0:
        raise RuntimeError("native fidelity did not pass exactly")
    if any(layer.get("relative_frobenius") != 0.0 or layer.get("cosine") != 1.0 for layer in fidelity.get("layers", [])):
        raise RuntimeError("layer fidelity is not exact")
    if preflight.get("source_head") != head or preflight.get("source_tree") != tree:
        raise RuntimeError("preflight source mismatch")
    if preflight.get("status") != "BGODE_R1_S1_PRE_GPU_PASS" or preflight.get("model_alias") != "qwen2.5-7b-inst":
        raise RuntimeError("Qwen preflight mismatch")

    control_flow = assert_source_control_flow(repo)
    scheduler = scheduler_row()
    log_root = repo / "local/odebf/logs/s05-bgode-r1-s1-qwen-b1-request000-six-arm-v1-tech-r6"
    stderr_path = log_root / f"slurm-{JOB_ID}.err"
    stdout_path = log_root / f"slurm-{JOB_ID}.out"
    stderr = require_regular(stderr_path, expected_mode=0o600).read_text(encoding="utf-8")
    if "NumericalRankBoundary: equality direction is outside range(G)" not in stderr:
        raise RuntimeError("scheduler stderr lacks locked boundary")

    history_specs = (
        ("TECH-R1", "26587", "TECHNICAL_EXCLUSION_VOCABULARY_BINDING"),
        ("TECH-R2", "26588", "TECHNICAL_EXCLUSION_NATIVE_FIDELITY"),
        ("TECH-R3", "26595", "TECHNICAL_EXCLUSION_NATIVE_FIDELITY"),
        ("TECH-R4", "26599", "TECHNICAL_EXCLUSION_PRE_FIDELITY_VALIDATOR"),
        ("TECH-R5", "26602", "TECHNICAL_EXCLUSION_RANK_ONE_REPRESENTATION"),
        ("TECH-R6", "26605", "SCIENTIFIC_BOUNDARY_LOCKED_RANGE_G"),
    )
    technical_history = []
    for label, job_id, classification in history_specs:
        suffix = label.lower()
        terminal_file = repo / f"local/odebf/results/s05-bgode-r1-s1-qwen-b1-request000-six-arm-v1-{suffix}" / RUN_ID / "terminal.json"
        item = load_json(terminal_file)
        technical_history.append(
            {
                "lineage": label,
                "job_id": job_id,
                "classification": classification,
                "scientific_denominator_contribution": 0,
                "terminal": file_receipt(terminal_file, classification=classification),
                "terminal_status": item.get("status"),
                "failure_type": item.get("failure_type"),
                "failure_message_sha256": item.get("failure_message_sha256"),
                "source_head": item.get("source_head"),
                "source_tree": item.get("source_tree"),
                "wall_seconds": item.get("total_wall_seconds"),
            }
        )

    sample = preflight["sample"]
    llama_sample = llama_terminal.get("sample")
    shared_fields = (
        "all_request_order_sha256", "batch_label", "batch_ordinal", "case_id",
        "edit_request_id", "raw_bytes", "raw_sha256", "request_ordinal",
        "request_sha256", "sample_payload_sha256", "selection_rule", "stream_root_sha256",
    )
    matched = all(sample.get(key) == llama_sample.get(key) for key in shared_fields)
    if not matched or preflight.get("sample_identity") != llama_terminal.get("sample_identity"):
        raise RuntimeError("Llama/Qwen matched sample identity failed")

    artifact_inventory = {
        "schema": f"{SCHEMA_PREFIX}-artifact-inventory/v1",
        "raw_root_immutable": True,
        "members": [
            file_receipt(terminal_path, classification="QWEN_TECH_R6_SCIENTIFIC_BOUNDARY"),
            file_receipt(fidelity_path, classification="QWEN_OFFICIAL_FIDELITY_EXACT_PASS"),
            file_receipt(direct_z_path, classification="PRIVATE_FIXED_Z_REFERENCE_ONLY_NOT_COMMITTED"),
            file_receipt(stdout_path, classification="RAW_LOG_REFERENCE_ONLY_NOT_COMMITTED"),
            file_receipt(stderr_path, classification="RAW_LOG_REFERENCE_ONLY_NOT_COMMITTED"),
            file_receipt(args.preflight, classification="PRE_GPU_INPUT_BINDING"),
            file_receipt(repo / "project/run_scripts/barrier_guided_ode/locks/bgode-r1-s1-qwen-numerical-lock.json", classification="NUMERICAL_LOCK"),
            file_receipt(repo / "project/run_scripts/barrier_guided_ode/locks/bgode-r1-s1-qwen-source-manifest.json", classification="SOURCE_MANIFEST"),
        ],
        "raw_classes_committed": 0,
    }
    matched_inputs = {
        "schema": f"{SCHEMA_PREFIX}-matched-inputs/v1",
        "status": "MATCHED_SAMPLE_INPUT_ONLY",
        "same_sample_fields": {key: sample.get(key) for key in shared_fields},
        "sample_identity": preflight.get("sample_identity"),
        "shared_sample_identity_pass": True,
        "qwen": {
            "model_alias": "qwen2.5-7b-inst",
            "tokenization": preflight.get("tokenization"),
            "termination_boundary": preflight.get("termination_boundary"),
            "source_head": head,
            "source_tree": tree,
            "terminal_status": terminal.get("status"),
            "terminal_sha256": EXPECTED_TERMINAL_SHA,
        },
        "llama_cross_reference_only": {
            "canonical_analysis_owner": "GH",
            "model_alias": "llama3-8b-inst",
            "tokenization": llama_terminal.get("tokenization"),
            "termination": llama_terminal.get("termination"),
            "source_head": llama_terminal.get("source_head"),
            "source_tree": llama_terminal.get("source_tree"),
            "terminal_status": llama_terminal.get("status"),
            "terminal_path": str(args.llama_terminal.resolve(strict=True)),
            "terminal_sha256": sha256_file(args.llama_terminal),
            "performance_metrics_copied": False,
        },
        "cross_model_scientific_endpoint_comparison": "NOT_COMPARABLE_QWEN_PANEL_DENOMINATOR_ZERO",
        "scientific_promotion": False,
    }
    summary = {
        "schema": f"{SCHEMA_PREFIX}-summary/v1",
        "status": "SCIENTIFIC_BOUNDARY_LOCKED_RANGE_G",
        "instruction_id": terminal.get("instruction_id"),
        "run_id": RUN_ID,
        "job": scheduler,
        "source_head": head,
        "source_tree": tree,
        "terminal": {
            "status": terminal.get("status"),
            "failure_type": terminal.get("failure_type"),
            "failure_message_sha256": terminal.get("failure_message_sha256"),
            "wall_seconds": terminal.get("total_wall_seconds"),
            "sha256": EXPECTED_TERMINAL_SHA,
        },
        "native_fidelity": {
            "status": "EXACT_PASS",
            "receipt_sha256": EXPECTED_FIDELITY_SHA,
            "identity_sha256": EXPECTED_FIDELITY_ID,
            "global_relative_frobenius": fidelity.get("relative_frobenius"),
            "all_layer_relative_frobenius": [layer.get("relative_frobenius") for layer in fidelity.get("layers", [])],
            "all_layer_cosine": [layer.get("cosine") for layer in fidelity.get("layers", [])],
            "controller_influence_count": fidelity.get("controller_influence_count"),
        },
        "panel": {
            "planned_arm_count": 6,
            "terminal_valid_arm_count": 0,
            "terminal_valid_denominator": "0/6",
            "plain_uses_rayleighian_solver": False,
            "plain_endpoint_status": "NOT_PUBLISHED_PANEL_ABORTED",
            "first_rayleighian_boundary_arm": "fisher-only-dynamic",
            "first_rayleighian_boundary_node": 0,
            "fisher_writer_action_count": 0,
            "later_arms_reached": False,
            "performance_metrics_published": False,
        },
        "atomicity": {
            "status": "SOURCE_ENFORCED_EXCEPTION_UNWIND_RESTORE",
            "pointer_check": True,
            "byte_sha_check": True,
            "terminal_panel_w0_receipt": "NOT_PUBLISHED_PANEL_ABORTED",
        },
        "control_flow_proof": control_flow,
        "technical_history": technical_history,
        "resubmit_count_after_scientific_ruling": 0,
        "forbidden_repairs": {
            "damping": 0,
            "ridge": 0,
            "tolerance_relaxation": 0,
            "fallback": 0,
            "pinv_cutoff_adjustment": 0,
        },
        "llama_analysis_duplicated": False,
        "p1r55_mutation_count": 0,
        "scientific_promotion": False,
    }

    write_json(output / "qwen-boundary-summary.json", summary)
    write_json(output / "qwen-llama-matched-inputs.json", matched_inputs)
    write_json(output / "qwen-technical-history.json", technical_history)
    write_json(output / "qwen-artifact-inventory.json", artifact_inventory)

    report = f"""# BGODE-R1 S1 Qwen 과학 경계 사실 보고서

## 판정

Qwen TECH-R6 job `{JOB_ID}`은 Official AlphaEdit fidelity를 **정확히 통과**한 뒤, 최초 Rayleighian arm인 `fisher-only-dynamic`의 node 0에서 `NumericalRankBoundary: equality direction is outside range(G)`로 fail-close되었다. 이는 잠긴 singular-aware 과학 경계이며 기술 수리 대상이 아니다. damping, ridge, 허용오차 완화, fallback, pseudoinverse cutoff 변경 및 재제출은 모두 0이다.

`scientific_promotion=false`이며, 6-arm panel terminal-valid denominator는 **0/6**이다. 따라서 Qwen 성능 수치나 Llama 대비 endpoint 비교를 만들지 않는다.

## 무엇이 유효한가

| 항목 | 사실 |
|---|---|
| job | {JOB_ID}, scheduler `FAILED`, exit `1:0`, elapsed `00:15:27` |
| source | `{head}` / `{tree}` |
| FULL-FP32 preflight | PASS; model=`qwen2.5-7b-inst`; boundary=`<|im_end|>`/151645 |
| Official dense RHS/native adapter | exact PASS |
| global relative Frobenius error | 0.0 |
| layer 4–8 relative error | 모두 0.0 |
| layer 4–8 cosine | 모두 1.0 |
| terminal boundary | equality direction outside `range(G)` |
| scientific result denominator | 0/6 |
| scientific promotion | false |

Official fidelity receipt는 `{EXPECTED_FIDELITY_SHA}`이고 identity는 `{EXPECTED_FIDELITY_ID}`이다. 이는 exact dense RHS identity-basis factor가 Official FP32 update를 byte/numerical 관점에서 정확히 재현했음을 증명하지만, six-arm scientific panel 완성을 뜻하지 않는다.

## arm 경계

고정 arm 순서는 Native → Plain → Fisher → Full-moving → One-step-Full → Frozen-field-N4이다. Plain은 Rayleighian solver를 호출하지 않는다. 관측된 예외는 Rayleighian solver 내부에서 발생했으므로, source order상 최초 가능한 위치는 **Fisher node 0**이다. solver는 coefficient 구성과 `trajectory.apply`보다 먼저 호출되므로 Fisher arm writer action은 0이다.

Plain은 제어 흐름상 Fisher 전에 실행됐지만 전체 panel publication 이전에 중단되었으므로 Plain endpoint도 유효 denominator에 넣지 않는다. Plain 내부 physical action count는 별도 raw receipt에 기록되지 않았고, `AtomicWeightTrajectory.__exit__`가 예외 유무와 관계없이 pointer를 유지한 채 W0 bytes를 복원·SHA 검증한다. 따라서 보고서는 Plain 성능을 추정하거나 공개하지 않는다.

## lineage 분리

| lineage | job | 분류 | scientific denominator |
|---|---:|---|---:|
| TECH-R1 | 26587 | technical exclusion: vocabulary binding | 0 |
| TECH-R2 | 26588 | technical exclusion: native fidelity | 0 |
| TECH-R3 | 26595 | technical exclusion: native fidelity | 0 |
| TECH-R4 | 26599 | technical exclusion: pre-fidelity validator | 0 |
| TECH-R5 | 26602 | technical exclusion: rank-one representation | 0 |
| TECH-R6 | 26605 | locked scientific boundary: `range(G)` | 0 |

TECH-R1–R5 root와 TECH-R6 raw root는 모두 immutable이며 imputation과 partial endpoint 재사용은 0이다.

## Llama/Qwen 결속 범위

두 모델은 sealed Phase123 B1 request ordinal 0, case `19795`, request SHA `{sample['request_sha256']}`, stream root `{sample['stream_root_sha256']}`, order root `{sample['all_request_order_sha256']}`를 공유한다. tokenizer/model/termination은 모델별로 별도 봉인된다. Llama canonical 분석 소유자는 GH이며 이 package는 Llama 성능 표나 판정을 복사·결합하지 않는다. Qwen denominator가 0/6이므로 matched cross-model endpoint comparison은 `NOT_COMPARABLE`이다.

## 비주장

- H1/H2/H3의 Qwen terminal 판정은 만들지 않는다.
- barrier attribution, ODE attribution, efficacy/locality/heldout 성능을 추정하지 않는다.
- rank gate를 피하기 위한 수치 보정이나 fallback을 제안하지 않는다.
- Llama GH package 또는 P1R55 source/result를 변경하지 않는다.

모든 raw 경로와 SHA는 `qwen-artifact-inventory.json`, lineage는 `qwen-technical-history.json`, 공통 입력은 `qwen-llama-matched-inputs.json`에 결속되어 있다.
"""
    report_name = "bgode-r1-s1-qwen-scientific-boundary-factual-ko.md"
    write_text(output / report_name, report)

    package_members = [
        report_name,
        "qwen-boundary-summary.json",
        "qwen-llama-matched-inputs.json",
        "qwen-technical-history.json",
        "qwen-artifact-inventory.json",
    ]
    entries = []
    for name in package_members:
        path = output / name
        entries.append({
            "path": name,
            "bytes": path.stat().st_size,
            "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
            "sha256": sha256_file(path),
        })
    members_root = sha256_bytes(canonical_bytes(entries))
    manifest = {
        "schema": f"{SCHEMA_PREFIX}-manifest/v1",
        "package_status": "SCIENTIFIC_BOUNDARY_LOCKED_RANGE_G",
        "entries": entries,
        "entries_count": len(entries),
        "members_root_sha256": members_root,
        "source_head": head,
        "source_tree": tree,
        "raw_files_committed": 0,
        "scientific_promotion": False,
    }
    manifest_path = output / "analysis-manifest.json"
    write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)
    receipt_payload = {
        "schema": f"{SCHEMA_PREFIX}-rooted-receipt/v1",
        "status": "QWEN_SCIENTIFIC_BOUNDARY_PACKAGE_COMPLETE",
        "manifest_sha256": manifest_sha,
        "manifest_bytes": manifest_path.stat().st_size,
        "members_root_sha256": members_root,
        "terminal_sha256": EXPECTED_TERMINAL_SHA,
        "fidelity_receipt_sha256": EXPECTED_FIDELITY_SHA,
        "fidelity_identity_sha256": EXPECTED_FIDELITY_ID,
        "terminal_valid_denominator": "0/6",
        "source_head": head,
        "source_tree": tree,
        "resubmit_after_scientific_boundary": 0,
        "llama_analysis_duplicated": False,
        "p1r55_mutation_count": 0,
        "scientific_promotion": False,
    }
    receipt_payload["identity_sha256"] = sha256_bytes(canonical_bytes(receipt_payload))
    receipt_path = output / "rooted-receipt.json"
    write_json(receipt_path, receipt_payload)

    print(canonical_bytes({
        "status": receipt_payload["status"],
        "output": str(output.resolve()),
        "report_sha256": sha256_file(output / report_name),
        "manifest_sha256": manifest_sha,
        "members_root_sha256": members_root,
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_identity_sha256": receipt_payload["identity_sha256"],
    }).decode().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
