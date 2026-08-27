#!/usr/bin/env python3
"""Seal the immutable BGODE-R3 G2 all-prefix JVP/FD numerical boundary."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
OUTPUT = REPO / "audits/servers/server1/2026-08-28-bgode-r3-g2-numerical-boundary-v1"
RESULT = REPO / "local/odebf/results/s05-bgode-r3-g2-rho-free-convergence-v1"
LOG = REPO / "local/odebf/logs/s05-bgode-r3-g2-rho-free-convergence-v1"
STATE = REPO / "local/state/bgode-r3-g2-rho-free-convergence-v1/submission-receipt.json"
RUN_HEAD = "423782776354b6779b8f5f312248e20111f75ba3"
RUN_TREE = "d672d23f049fc2f9bfeb569f2c4a2e84c179661b"
CONTRACT_SHA = "fa0a6923efe944d485dc49802772b4d08f7d02f7c7113dbee302e7d9744642dc"
ERROR_TEXT = "NumericalBoundary: all-prefix normalized JVP/central-FD identity failed"

TASKS = (
    {
        "array_task": 0,
        "model_alias": "llama3-8b-inst",
        "elapsed": "00:27:18",
        "started": "2026-08-27T22:46:10",
        "ended": "2026-08-27T23:13:28",
        "max_rss_kib": 11127232,
        "private": (
            "task-0/s05-bgode-r3-g2-llama3-8b-inst-natural-unequal-nonprefix-convergence-v1/private/direct-z.pt",
            "task-0/s05-bgode-r3-g2-llama3-8b-inst-natural-unequal-nonprefix-convergence-v1/private/w0-q0.pt",
        ),
    },
    {
        "array_task": 1,
        "model_alias": "qwen2.5-7b-inst",
        "elapsed": "05:18:14",
        "started": "2026-08-27T22:46:10",
        "ended": "2026-08-28T04:04:24",
        "max_rss_kib": 15843488,
        "private": (
            "task-1/s05-bgode-r3-g2-qwen2.5-7b-inst-natural-unequal-nonprefix-convergence-v1/private/direct-z.pt",
            "task-1/s05-bgode-r3-g2-qwen2.5-7b-inst-natural-unequal-nonprefix-convergence-v1/private/w0-q0.pt",
        ),
    },
)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    status = path.lstat()
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(status.st_mode) != 0o600:
        raise SystemExit(f"boundary artifact is not regular mode0600: {path}")
    return {
        "bytes": status.st_size,
        "mode": "0600",
        "path": str(path.relative_to(REPO)),
        "sha256": file_sha(path),
    }


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise SystemExit(f"existing G2 boundary member differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def main() -> None:
    submission = json.loads(STATE.read_text())
    if submission["job_id"] != 26916 or submission["source_head"] != RUN_HEAD:
        raise SystemExit("G2 submission binding differs")

    artifact_rows = [record(STATE)]
    model_rows = []
    for task in TASKS:
        index = task["array_task"]
        stdout = LOG / f"26916_{index}.out"
        stderr = LOG / f"26916_{index}.err"
        stderr_text = stderr.read_text()
        if ERROR_TEXT not in stderr_text:
            raise SystemExit(f"task {index} does not contain the locked NumericalBoundary")
        if "PASS repository=" not in stdout.read_text():
            raise SystemExit(f"task {index} did not pass the session boundary")
        artifact_rows.extend((record(stdout), record(stderr)))
        private_rows = []
        for relative in task["private"]:
            item = record(RESULT / relative)
            private_rows.append(item)
            artifact_rows.append(item)
        model_rows.append(
            {
                "array_task": index,
                "elapsed": task["elapsed"],
                "ended": task["ended"],
                "exit_code": "1:0",
                "failed_arm": "NOT_RECORDED_SCHEMA_GAP",
                "failed_n": "NOT_RECORDED_SCHEMA_GAP",
                "failed_node": "NOT_RECORDED_SCHEMA_GAP",
                "failed_prefix": "NOT_RECORDED_SCHEMA_GAP",
                "fixed_artifacts": private_rows,
                "g2_dynamic_writer_action_count": "NOT_RECORDED_SCHEMA_GAP",
                "locked_fd_absolute_tolerance": 0.02,
                "locked_fd_epsilon": 0.00390625,
                "locked_fd_maximum_absolute_error": "NOT_RECORDED_UNCAUGHT_EXCEPTION_RECEIPT",
                "locked_fd_relative_tolerance": 0.08,
                "max_rss_kib": task["max_rss_kib"],
                "model_alias": task["model_alias"],
                "scheduler_state": "FAILED",
                "science_definition_change_count": 0,
                "started": task["started"],
                "trajectory_panel_endpoint_count": 0,
                "w0_restore": "SOURCE_GUARANTEED_ATOMIC_CONTEXT_ON_EXCEPTION_NOT_RAW_RECEIPTED",
            }
        )

    artifact_rows = sorted(artifact_rows, key=lambda item: str(item["path"]))
    inventory = {
        "schema": "ode-edit-bgode-r3-g2-boundary-artifact-inventory/v1",
        "artifact_count": len(artifact_rows),
        "artifacts": artifact_rows,
        "artifacts_root": digest_bytes(canonical(artifact_rows).rstrip(b"\n")),
    }
    summary = {
        "schema": "ode-edit-bgode-r3-g2-numerical-boundary-summary/v1",
        "status": "R3_G2_ALL_PREFIX_JVP_FD_NUMERICAL_BOUNDARY_UNLOCALIZED_SCHEMA_GAP",
        "contract_sha256": CONTRACT_SHA,
        "job_id": 26916,
        "source_head": RUN_HEAD,
        "source_tree": RUN_TREE,
        "sample": {"ordinal": 26, "case_id": "17454", "topology": "unequal-non-prefix"},
        "locked_matrix": {"arms": ["Plain", "Fisher", "Full"], "n": [4, 8, 16, 32]},
        "models": model_rows,
        "classification": {
            "boundary_type": "LOCKED_NUMERICAL_BOUNDARY",
            "exact_exception": ERROR_TEXT,
            "failure_receipt_publication": "NOT_RECORDED_SCHEMA_GAP",
            "technical_repair_authorized": False,
            "resubmit_count": 0,
            "tolerance_change_count": 0,
            "threshold_change_count": 0,
            "damping_ridge_fallback_count": 0,
            "imputation_count": 0,
        },
        "denominators": {
            "scheduler_failed_tasks": "2/2",
            "terminal_model_endpoints": "0/2",
            "trajectory_panel_endpoints": "0/24",
            "cauchy_arm_assessments": "0/6",
        },
        "preservation": {
            "failed_result_roots_immutable": True,
            "failed_logs_immutable": True,
            "r1_r2_p1r55_unrelated_mutation_count": 0,
        },
        "scientific_promotion": False,
    }

    inventory_path = OUTPUT / "artifact-inventory.json"
    summary_path = OUTPUT / "g2-boundary-summary.json"
    write_once(inventory_path, canonical(inventory))
    write_once(summary_path, canonical(summary))
    rows = "\n".join(
        "| {model_alias} | FAILED 1:0 | {elapsed} | all-prefix JVP/FD | "
        "NOT_RECORDED | NOT_RECORDED | NOT_RECORDED | 0 |".format(**row)
        for row in model_rows
    )
    report = f"""# BGODE-R3 G2 수치 경계 사실 보고서

## 판정

`R3_G2_ALL_PREFIX_JVP_FD_NUMERICAL_BOUNDARY_UNLOCALIZED_SCHEMA_GAP`이다. job 26916의 Llama/Qwen 두 task 모두 locked all-prefix normalized-actuator serial-JVP 대 central-FD identity에서 `NumericalBoundary`로 종료했다. 이는 Cauchy 결과를 보고 threshold를 조정하는 기술 실패가 아니라 predeclared numerical gate의 실패이다. 따라서 tolerance·threshold·ridge·damping·fallback·pinv를 변경하거나 재제출하지 않았다.

## 모델별 사실

| 모델 | scheduler | elapsed | boundary | arm/N/node | prefix | max abs error | panel endpoint |
|---|---|---:|---|---|---|---:|---:|
{rows}

G2 구현은 node receipt와 trajectory panel을 terminal JSON에서만 원자적으로 publish한다. 예외에는 `NumericalBoundary.receipt`가 있었지만 top-level failure artifact로 serialize하지 않았으므로 실패 arm/N/node/prefix, 최대 절대오차, 예외 전 dynamic write 수는 raw에 남지 않았다. 실행 시간으로 이를 역추정하거나 G1 값으로 대체하지 않았다. 이 누락은 `NOT_RECORDED_SCHEMA_GAP`이며 imputation은 0이다.

## 분모와 보존 경계

- scheduler failure: 2/2 tasks
- terminal endpoint: 0/2 models
- trajectory panel endpoint: 0/24 (3 arms × 4 N × 2 models)
- Cauchy arm assessment: 0/6
- sample: sealed ordinal 26 / case 17454 / natural unequal-nonprefix
- run source HEAD/tree: `{RUN_HEAD}` / `{RUN_TREE}`
- 기존 result/log/private prefix는 immutable이다.
- R1/R2/P1R55/unrelated mutation 0, resubmit 0, scientific promotion false이다.

## 해석 한계

G1의 같은 natural sample은 locked all-prefix FD gate를 통과했지만, 그 값으로 G2의 실패 값을 대신할 수 없다. G2에서 어느 trajectory node가 처음 실패했는지는 저장되지 않았으므로 Full/Fisher/Plain 또는 N별 과학 비교와 Cauchy 판정은 모두 수행 불가이다. 이 보고서는 실패를 숨기지 않고 numerical boundary와 failure-schema limitation만 결속한다.
""".encode()
    report_path = OUTPUT / "bgode-r3-g2-numerical-boundary-factual-ko.md"
    write_once(report_path, report)

    members = []
    for path in sorted((inventory_path, summary_path, report_path)):
        members.append(record(path))
    manifest = {
        "schema": "ode-edit-bgode-r3-g2-boundary-manifest/v1",
        "member_count": len(members),
        "members": members,
        "members_root": digest_bytes(canonical(members).rstrip(b"\n")),
    }
    manifest_path = OUTPUT / "source-manifest.json"
    write_once(manifest_path, canonical(manifest))
    receipt_body = {
        "schema": "ode-edit-bgode-r3-g2-boundary-rooted-receipt/v1",
        "manifest_sha256": file_sha(manifest_path),
        "members_root": manifest["members_root"],
        "run_source_head": RUN_HEAD,
        "run_source_tree": RUN_TREE,
        "status": summary["status"],
        "scientific_promotion": False,
    }
    receipt = dict(receipt_body)
    receipt["receipt_identity"] = digest_bytes(canonical(receipt_body).rstrip(b"\n"))
    write_once(OUTPUT / "rooted-receipt.json", canonical(receipt))


if __name__ == "__main__":
    main()
