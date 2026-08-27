#!/usr/bin/env python3
"""Build the immutable BGODE-R2 Stage-C topology package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from project.run_scripts.barrier_guided_ode.r2.stage_c_contract import TOPOLOGY_NAMES


SCHEMA = "ode-edit-bgode-r2-stage-c-terminal-package/v1"
RUNTIME_HEAD = "8588772459c528321de999915da7dca6fa2f7bd2"
RUNTIME_TREE = "12d8980c5377991fca2b6d3198a13044904c4334"
EXPECTED = (
    "ba6d5da2332ab53ae4410b88ef98d931c5ace6998360fcfc0e19247e7db65112",
    "4bf7e50a7eeae0e00997beddd23569bd9cfecb28b0dadd5783c71a1279261ac0",
    "8c86b04a02fb553cc7e8c520581fc1abb27f45bf834fa111c69fdee2b608809a",
    "a20f399a700dab853467347c0314330990100daaa205170e508814d75acba915",
    "5ae751afd43222a5ea42d209a28f4d37b684a6e05e7af73036a3f8148fc802a0",
    "556980bed2bb8e554a692613a491c6b2d899a697a4773a59a21344b70e07c6e5",
)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_once(path: Path, value: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
            raise SystemExit(f"existing Stage-C output differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(value)


def record(path: Path, label: str) -> dict[str, Any]:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or path.is_symlink() or stat.S_IMODE(status.st_mode) != 0o600:
        raise SystemExit(f"regular non-symlink mode0600 required: {path}")
    data = path.read_bytes()
    return {"label": label, "path": str(path.resolve()), "bytes": len(data), "mode": "0600", "sha256": sha(data)}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    inputs = []
    statuses = []
    for cell in range(6):
        model = "llama3-8b-inst" if cell < 3 else "qwen2.5-7b-inst"
        topology = TOPOLOGY_NAMES[cell % 3]
        task = args.input_root / f"task-{cell}"
        paths = tuple(task.glob("*/terminal.json"))
        if len(paths) != 1:
            raise SystemExit(f"cell {cell} terminal cardinality differs")
        path = paths[0]
        rec = record(path, f"cell-{cell}")
        if rec["sha256"] != EXPECTED[cell]:
            raise SystemExit(f"cell {cell} terminal SHA differs")
        value = json.loads(path.read_text())
        if value["source_head"] != RUNTIME_HEAD or value["source_tree"] != RUNTIME_TREE:
            raise SystemExit(f"cell {cell} runtime source differs")
        if value["stage_context"]["topology"] != topology:
            raise SystemExit(f"cell {cell} topology differs")
        if value["native_adapter_fidelity"]["relative_frobenius"] != 0.0 or value["w0_restore"] != {"bytes_exact": True, "pointer_exact": True}:
            raise SystemExit(f"cell {cell} fidelity/W0 differs")
        forbidden = (
            "dynamic_writer_action_count", "node_history_append_count", "fixed_target_recompute_count",
            "controller_evaluator_influence_count", "scalar_localizer_count", "probability_floor_count",
            "ridge_count", "damping_count", "fallback_count",
        )
        if any(value[key] for key in forbidden):
            raise SystemExit(f"cell {cell} forbidden counter differs")
        boundary = value.get("boundary")
        solution = value.get("fisher_node0_solution")
        row = {
            "cell": cell,
            "model_alias": model,
            "topology": topology,
            "status": value["status"],
            "terminal_sha256": rec["sha256"],
            "terminal_identity": value["terminal_identity"],
            "prefix_relation": value["fisher_node0_event"]["prefix_relation"],
            "event_count": value["fisher_node0_event"]["event_count"],
            "normalization_log_residual": value["fisher_node0_event"]["normalization_log_residual"],
            "score_centering_norm": value["fisher_node0_event"]["score_centering_norm"],
            "fd_allclose": value["finite_difference"]["allclose"],
            "official_relative_frobenius": value["native_adapter_fidelity"]["relative_frobenius"],
            "matrix_rank": None if solution is None else solution["receipt"]["matrix_rank"],
            "equality_rank": None if solution is None else solution["receipt"]["equality_rank"],
            "equality_residual": None if solution is None else solution["receipt"]["equality_residual"],
            "boundary_type": None if boundary is None else boundary["type"],
            "boundary_receipt": None if boundary is None else boundary["receipt"],
            "w0_pointer_exact": True,
            "w0_bytes_exact": True,
            "wall_seconds": value["total_wall_seconds"],
        }
        rows.append(row)
        inputs.append(rec)
        statuses.append(value["status"])
    passed = sum(status == "BGODE_R2_STAGE_C_FP64_TOPOLOGY_PASS" for status in statuses)
    bounded = sum(status == "BGODE_R2_STAGE_C_NUMERICAL_IMPLEMENTATION_BOUNDARY" for status in statuses)
    if passed != 4 or bounded != 2:
        raise SystemExit("Stage-C terminal denominator differs")
    summary = {
        "schema": SCHEMA + "/summary",
        "array_job_id": 26686,
        "terminal_complete": "6/6",
        "topology_pass": "4/6",
        "numerical_implementation_boundary": "2/6",
        "technical_failure": 0,
        "stage_d_release": False,
        "stage_d_hold_reason": "STAGE_C_ALL_TOPOLOGY_GATE_NOT_PASS",
        "runtime_source_head": RUNTIME_HEAD,
        "runtime_source_tree": RUNTIME_TREE,
        "rows": rows,
        "scientific_promotion": False,
    }
    report = """# BGODE-R2 Stage C multi-token topology 사실 보고

Stage C는 outcome-independent tokenizer-token manifest로 Llama/Qwen 각각 unequal non-prefix, source-prefix, target-prefix를 검사했다. 여섯 scheduler cell은 모두 exit0 terminal을 게시했으며 raw result는 변경하지 않았다.

| 모델 | topology | event partition | FP64 two-equality | W0 |
|---|---|---:|---:|---:|
| Llama3-8B | unequal non-prefix | 정상화 PASS | NUMERICAL_IMPLEMENTATION_BOUNDARY | exact |
| Llama3-8B | source-prefix | 정상화 PASS | PASS | exact |
| Llama3-8B | target-prefix | 정상화 PASS | PASS | exact |
| Qwen2.5-7B | unequal non-prefix | 정상화 PASS | NUMERICAL_IMPLEMENTATION_BOUNDARY | exact |
| Qwen2.5-7B | source-prefix | 정상화 PASS | PASS | exact |
| Qwen2.5-7B | target-prefix | 정상화 PASS | PASS | exact |

두 unequal non-prefix boundary는 event/JVP 오류가 아니다. event normalization, score centering, serial JVP/FD, Official adapter fidelity, FULL-FP32 model과 W0 restore는 통과했고, 고정 FP64 `range(G)` backward-error gate만 거부했다. Llama direction residual은 1.282e-2/4.709e-5, bound는 1.560e-6/1.362e-6이다. Qwen은 2.401e-2/9.108e-2, bound는 7.456e-4/1.624e-4이다.

계약상 Stage D는 Stage B와 Stage C가 모두 통과한 뒤에만 release된다. 따라서 tolerance/pinv/ridge/damping/fallback을 바꾸거나 재제출하지 않고 Stage D를 HOLD한다. scientific promotion은 false이다. 상세 spectrum과 모든 terminal identity는 `stage-c-summary.json`에 있다.
"""
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "bgode-r2-stage-c-topology-factual-ko.md"
    summary_path = args.output / "stage-c-summary.json"
    write_once(report_path, report.encode())
    write_once(summary_path, canonical(summary))
    outputs = [record(path, path.name) for path in (report_path, summary_path)]
    root = sha(canonical(outputs))
    manifest = {
        "schema": SCHEMA + "/manifest",
        "runtime_inputs": inputs,
        "outputs": outputs,
        "members_root": root,
        "raw_files_committed": 0,
        "scientific_promotion": False,
    }
    manifest_path = args.output / "analysis-manifest.json"
    write_once(manifest_path, canonical(manifest))
    receipt = {
        "schema": SCHEMA + "/rooted-receipt",
        "status": "BGODE_R2_STAGE_C_PARTIAL_PASS_NUMERICAL_BOUNDARY",
        "array_job_id": 26686,
        "terminal_complete": "6/6",
        "topology_pass": "4/6",
        "numerical_implementation_boundary": "2/6",
        "stage_d_submission_count": 0,
        "manifest_sha256": sha(manifest_path.read_bytes()),
        "members_root": root,
        "runtime_source_head": RUNTIME_HEAD,
        "runtime_source_tree": RUNTIME_TREE,
        "p1r55_mutation_count": 0,
        "r1_package_mutation_count": 0,
        "scientific_promotion": False,
    }
    receipt["identity"] = sha(canonical(receipt))
    receipt_path = args.output / "rooted-receipt.json"
    write_once(receipt_path, canonical(receipt))
    print(canonical({
        "status": receipt["status"], "output": str(args.output.resolve()),
        "report_sha256": sha(report_path.read_bytes()), "manifest_sha256": receipt["manifest_sha256"],
        "members_root": root, "receipt_sha256": sha(receipt_path.read_bytes()), "receipt_identity": receipt["identity"],
    }).decode().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
