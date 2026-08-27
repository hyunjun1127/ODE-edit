#!/usr/bin/env python3
"""Seal the immutable BGODE-R3 G1 natural-topology terminal facts."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
OUTPUT = REPO / "audits/servers/server1/2026-08-27-bgode-r3-g1-terminal-v1"
RESULT_ROOT = REPO / "local/odebf/results/s05-bgode-r3-g1-natural-multitoken-tech-r1-v1"
SOURCE_HEAD = "072a8addee4a331929db014752c0ee246a5f01b2"
SOURCE_TREE = "596d77414965f03bb5e1c18e8e094aab19b8b07a"
MODEL_RUNS = {
    "llama3-8b-inst": (
        "task-0/s05-bgode-r3-g1-llama3-8b-inst-natural-unequal-nonprefix-multitoken-tech-r1-v1/terminal.json"
    ),
    "qwen2.5-7b-inst": (
        "task-1/s05-bgode-r3-g1-qwen2.5-7b-inst-natural-unequal-nonprefix-multitoken-tech-r1-v1/terminal.json"
    ),
}


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


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise SystemExit(f"existing G1 terminal member differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _max_fd(terminal: dict[str, Any]) -> tuple[float, float, float]:
    receipts = terminal["prefix_fd"]
    return (
        max(float(item["maximum_absolute_error"]) for item in receipts),
        max(float(item["absolute_tolerance"]) for item in receipts),
        max(float(item["relative_tolerance"]) for item in receipts),
    )


def _model_fact(alias: str, path: Path, terminal: dict[str, Any]) -> dict[str, Any]:
    if terminal["source_head"] != SOURCE_HEAD or terminal["source_tree"] != SOURCE_TREE:
        raise SystemExit(f"{alias} source identity differs")
    if terminal["model"]["model_alias"] != alias:
        raise SystemExit(f"{alias} model binding differs")
    if terminal["status"] != "R3_G1_ACTUATOR_FEASIBLE_G2_HOLD":
        raise SystemExit(f"{alias} did not reach the natural G1 terminal gate")
    if terminal["case"]["ordinal"] != 26 or str(terminal["case"]["case_id"]) != "17454":
        raise SystemExit(f"{alias} natural request identity differs")
    if terminal["execution_boundary"]["fixed_target_compute_count"] != 1:
        raise SystemExit(f"{alias} fixed-z compute count differs")
    forbidden = (
        "fixed_target_recompute_count",
        "node_history_append_count",
        "terminal_history_append_count",
        "localizer_call_count",
        "endpoint_correction_count",
        "accepted_step_search_count",
        "alternate_solver_count",
        "controller_evaluator_influence_count",
        "retained_factor_count",
    )
    if any(terminal["execution_boundary"][name] != 0 for name in forbidden):
        raise SystemExit(f"{alias} forbidden counter differs from zero")
    if terminal["w0_restore"] != {"bytes_exact": True, "pointer_exact": True}:
        raise SystemExit(f"{alias} W0 restore differs")
    absolute, absolute_tolerance, relative_tolerance = _max_fd(terminal)
    return {
        "adapter_relative_frobenius": float(
            terminal["native_adapter_fidelity"]["relative_frobenius"]
        ),
        "event_count": int(terminal["node0_event"]["event_count"]),
        "fisher_equality_residual": float(terminal["fisher"]["receipt"]["equality_residual"]),
        "fixed_z_compute_count": 1,
        "fixed_z_recompute_count": 0,
        "full_equality_residual": float(terminal["full"]["receipt"]["equality_residual"]),
        "g2_release_eligibility": "ELIGIBLE_AFTER_PREDECLARED_CAUCHY_LOCK",
        "history_append_count": 0,
        "jvp_calls": int(terminal["jvp_compute"]["jvp_call_count"]),
        "max_prefix_fd_absolute_error": absolute,
        "max_prefix_fd_absolute_tolerance": absolute_tolerance,
        "max_prefix_fd_relative_error": "NOT_RECORDED_SEPARATELY",
        "max_prefix_fd_relative_tolerance": relative_tolerance,
        "model_alias": alias,
        "model_forward_invocations": int(terminal["jvp_compute"]["model_forward_invocations"]),
        "native_target_logit_horizon": float(terminal["native_target_logit_horizon"]),
        "normalization_log_residual": float(terminal["node0_event"]["normalization_log_residual"]),
        "normalization_tolerance": float(terminal["node0_event"]["normalization_tolerance"]),
        "ordinal": 26,
        "case_id": "17454",
        "peak_gpu_allocated_bytes": int(terminal["peak_gpu_allocated_bytes"]),
        "peak_gpu_reserved_bytes": int(terminal["peak_gpu_reserved_bytes"]),
        "score_centering_norm": float(terminal["node0_event"]["score_centering_norm"]),
        "status": terminal["status"],
        "step_size": float(terminal["step_size"]),
        "t0_full_fisher_difference_norm": float(
            terminal["t0_full_fisher_identity"]["difference_norm"]
        ),
        "t0_full_fisher_tolerance": float(terminal["t0_full_fisher_identity"]["tolerance"]),
        "terminal_bytes": path.stat().st_size,
        "terminal_identity": terminal["terminal_identity"],
        "terminal_path": str(path.relative_to(REPO)),
        "terminal_sha256": file_sha(path),
        "total_wall_seconds": float(terminal["total_wall_seconds"]),
        "w0_bytes_restore": True,
        "w0_pointer_restore": True,
    }


def main() -> None:
    model_facts: list[dict[str, Any]] = []
    for alias, relative in MODEL_RUNS.items():
        path = RESULT_ROOT / relative
        status = path.lstat()
        if path.is_symlink() or not path.is_file() or stat.S_IMODE(status.st_mode) != 0o600:
            raise SystemExit(f"terminal is not regular mode0600: {path}")
        model_facts.append(_model_fact(alias, path, json.loads(path.read_text())))

    summary = {
        "schema": "ode-edit-bgode-r3-g1-terminal-summary/v1",
        "instruction_id": "ODEEDIT-S05-BGODE-R3-FINE-EVENT-TARGET-EXCLUDED-FACTOR-SPACE",
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "scheduler": {
            "initial_job": 26885,
            "initial_job_status": "TECHNICAL_EXCLUSION_JSON_SERIALIZATION_AFTER_W0_RESTORE",
            "initial_endpoint_denominator": "0/2",
            "tech_r1_job": 26899,
            "tech_r1_terminal_denominator": "2/2",
        },
        "natural_topology": {
            "available_cells": 2,
            "available_topology": "unequal-non-prefix",
            "ordinal": 26,
            "case_id": "17454",
            "unavailable_cells": 6,
            "unavailable_status": "NATURAL_TOPOLOGY_UNAVAILABLE",
            "synthetic_replacement_count": 0,
        },
        "models": model_facts,
        "g1_terminal_valid": True,
        "g2_release": "MODELWISE_ELIGIBLE_AFTER_PREDECLARED_CAUCHY_LOCK",
        "scientific_promotion": False,
        "action_counts_after_terminal": {"model": 0, "gpu": 0, "slurm": 0},
        "status": "R3_G1_NATURAL_UNEQUAL_NONPREFIX_TERMINAL_PASS",
    }
    summary_path = OUTPUT / "g1-terminal-summary.json"
    write_once(summary_path, canonical(summary))
    rows_path = OUTPUT / "g1-model-gates.json"
    write_once(rows_path, canonical(model_facts))

    rows = []
    for fact in model_facts:
        rows.append(
            "| {model_alias} | {event_count} | {normalization_log_residual:.3e} | "
            "{max_prefix_fd_absolute_error:.3e} / NOT_RECORDED | "
            "{adapter_relative_frobenius:.3e} | {t0_full_fisher_difference_norm:.3e} | "
            "{native_target_logit_horizon:.6f} | {total_wall_seconds:.3f} | PASS |".format(**fact)
        )
    report = f"""# BGODE-R3 G1 자연 multi-token 사실 보고서

## 판정

`R3_G1_NATURAL_UNEQUAL_NONPREFIX_TERMINAL_PASS`이다. sealed ordinal 26/case 17454의 자연 unequal-non-prefix 요청에서 Llama/Qwen 2/2가 terminal-valid이다. 다른 topology 6 cell은 `NATURAL_TOPOLOGY_UNAVAILABLE`이며 합성 대체는 0이다. scientific promotion은 false이다.

초기 job 26885의 두 task는 모든 과학 동작과 W0 restore 뒤 관찰 receipt JSON 변환에서만 실패했으며 endpoint denominator는 0/2이다. 과학식·요청·허용오차·writer를 바꾸지 않은 TECH-R1 job 26899가 2/2 terminal을 새 namespace에서 완성했다.

## 모델별 gate

| 모델 | event 수 | normalization residual | max FD abs / rel | Official adapter rel-F | t0 Full−Fisher | T_AE | wall(s) | W0 restore |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

두 모델 모두 fixed-z compute1/recompute0, node/terminal history append0, localizer/root0, retained factor0, FP32 model·FP64 event/controller·FP32 physical write 경계를 지켰다. Genuine P-inside ordered proposal의 Official dense endpoint fidelity, 전체 내부 prefix serial JVP/central FD, fine-event normalization/score-centering, equality residual, t=0 Full=Fisher, h_probe=T_AE/32 one-step write와 pointer+bytes exact W0 restore를 통과했다.

## G2 경계

두 모델의 자연 unequal-non-prefix G1 gate는 PASS했다. G2는 결과 전에 봉인하는 Cauchy numerical lock이 아직 필요하므로 이 package 자체는 실행을 release하지 않는다. 모델별로 동일 ordinal/request/W0/fixed-z/T_AE와 N={{4,8,16,32}}, arms={{Plain,Fisher,Full}}를 사용하며 unavailable topology를 대체하지 않는다.

## identity

- source HEAD/tree: `{SOURCE_HEAD}` / `{SOURCE_TREE}`
- TECH-R1 job: 26899, terminal 2/2
- initial technical exclusion: job26885, endpoint 0/2
- promotion: false
""".encode()
    report_path = OUTPUT / "bgode-r3-g1-natural-multitoken-factual-ko.md"
    write_once(report_path, report)

    members = []
    for path in sorted((summary_path, rows_path, report_path)):
        mode = stat.S_IMODE(path.lstat().st_mode)
        members.append(
            {
                "bytes": path.stat().st_size,
                "mode": format(mode, "04o"),
                "path": str(path.relative_to(REPO)),
                "sha256": file_sha(path),
            }
        )
    manifest = {
        "schema": "ode-edit-bgode-r3-g1-terminal-manifest/v1",
        "member_count": len(members),
        "members": members,
        "members_root": digest_bytes(canonical(members).rstrip(b"\n")),
    }
    manifest_path = OUTPUT / "source-manifest.json"
    write_once(manifest_path, canonical(manifest))
    receipt_body = {
        "schema": "ode-edit-bgode-r3-g1-terminal-rooted-receipt/v1",
        "manifest_sha256": file_sha(manifest_path),
        "members_root": manifest["members_root"],
        "source_head": SOURCE_HEAD,
        "source_tree": SOURCE_TREE,
        "terminal_identities": {
            fact["model_alias"]: fact["terminal_identity"] for fact in model_facts
        },
        "status": summary["status"],
        "scientific_promotion": False,
    }
    receipt = dict(receipt_body)
    receipt["receipt_identity"] = digest_bytes(canonical(receipt_body).rstrip(b"\n"))
    write_once(OUTPUT / "rooted-receipt.json", canonical(receipt))


if __name__ == "__main__":
    main()
