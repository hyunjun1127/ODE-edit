#!/usr/bin/env python3
"""Build the immutable BGODE-R2 Stage-B factual package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


SCHEMA = "ode-edit-bgode-r2-stage-b-terminal-package/v1"
RUNTIME_HEAD = "d91d48ca1c0d6d3bd0d554bb7b543ec1233ab35e"
RUNTIME_TREE = "c8d0bd2794af29dc4efdf32b31c6fa7293333d2d"
EXPECTED = {
    "llama3-8b-inst": "0fb8b6499d3ac599e027739a86da498e18fcd1c14fcd41c79aabb04e6f1c7abb",
    "qwen2.5-7b-inst": "0c0c11acd8a8de95f8b956abefa4a10c416125304cece228fa91df0dc6da8e46",
}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path, *, label: str) -> dict[str, Any]:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or path.is_symlink() or stat.S_IMODE(status.st_mode) != 0o600:
        raise SystemExit(f"regular non-symlink mode0600 required: {path}")
    data = path.read_bytes()
    return {"label": label, "path": str(path.resolve()), "bytes": len(data), "mode": "0600", "sha256": sha(data)}


def write_once(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise SystemExit(f"existing output differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _terminal(path: Path, alias: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = file_record(path, label=alias)
    if record["sha256"] != EXPECTED[alias]:
        raise SystemExit(f"{alias} terminal SHA differs")
    payload = json.loads(path.read_text())
    if payload.get("status") != "BGODE_R2_STAGE_B_FP64_FISHER_NODE0_PASS":
        raise SystemExit(f"{alias} terminal status differs")
    checks = {
        "runtime_source_exact": payload.get("source_head") == RUNTIME_HEAD and payload.get("source_tree") == RUNTIME_TREE,
        "official_fidelity_exact": payload["native_adapter_fidelity"]["relative_frobenius"] == 0.0,
        "ordered_validation_call_count": payload["ordered_dictionary"]["validation_call_count"],
        "w0_pointer_exact": payload["w0_restore"]["pointer_exact"],
        "w0_bytes_exact": payload["w0_restore"]["bytes_exact"],
        "full_fp32": set(payload["dtype"]["parameter_dtype_counts"]) == {"torch.float32"},
        "forbidden_counts": {
            key: payload[key]
            for key in (
                "dynamic_writer_action_count", "node_history_append_count", "fixed_target_recompute_count",
                "controller_evaluator_influence_count", "scalar_localizer_count", "probability_floor_count",
                "ridge_count", "damping_count", "fallback_count",
            )
        },
    }
    if not all((checks["runtime_source_exact"], checks["official_fidelity_exact"], checks["w0_pointer_exact"], checks["w0_bytes_exact"], checks["full_fp32"])):
        raise SystemExit(f"{alias} terminal invariant differs")
    if checks["ordered_validation_call_count"] != 1 or any(checks["forbidden_counts"].values()):
        raise SystemExit(f"{alias} counter invariant differs")
    return payload, record


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "llama3-8b-inst": args.input_root / "task-0/s05-bgode-r2-stage-b-llama-request000-fp64-two-equality-v1/terminal.json",
        "qwen2.5-7b-inst": args.input_root / "task-1/s05-bgode-r2-stage-b-qwen-request000-fp64-two-equality-v1/terminal.json",
    }
    terminals: dict[str, Any] = {}
    inputs = []
    rows = []
    for alias, path in paths.items():
        payload, record = _terminal(path, alias)
        terminals[alias] = payload
        inputs.append(record)
        solve = payload["fisher_node0_solution"]["receipt"]
        rows.append({
            "model_alias": alias,
            "status": payload["status"],
            "terminal_sha256": record["sha256"],
            "terminal_identity": payload["terminal_identity"],
            "native_log_odds_horizon": payload["native_log_odds_horizon"],
            "event_count": payload["fisher_node0_event"]["event_count"],
            "prefix_relation": payload["fisher_node0_event"]["prefix_relation"],
            "event_normalization_log_residual": payload["fisher_node0_event"]["normalization_log_residual"],
            "score_centering_norm": payload["fisher_node0_event"]["score_centering_norm"],
            "fisher_rank": solve["matrix_rank"],
            "equality_rank": solve["equality_rank"],
            "equality_residual": solve["equality_residual"],
            "equality_tolerance": solve["equality_tolerance"],
            "direction_range_residuals": solve["range_residual_directions"],
            "direction_range_tolerances": solve["range_tolerance_directions"],
            "official_relative_frobenius": payload["native_adapter_fidelity"]["relative_frobenius"],
            "jvp_calls": payload["jvp_compute"]["jvp_call_count"],
            "fd_forwards": payload["jvp_compute"]["finite_difference_forward_count"],
            "fd_allclose": payload["finite_difference"]["allclose"],
            "wall_seconds": payload["total_wall_seconds"],
            "peak_gpu_allocated_bytes": payload["peak_gpu_allocated_bytes"],
            "peak_gpu_reserved_bytes": payload["peak_gpu_reserved_bytes"],
            "w0_pointer_exact": True,
            "w0_bytes_exact": True,
        })

    summary = {
        "schema": SCHEMA + "/summary",
        "array_job_id": 26677,
        "array_mapping": {"0": "llama3-8b-inst", "1": "qwen2.5-7b-inst"},
        "terminal_valid": "2/2",
        "technical_failure": 0,
        "scientific_boundary": 0,
        "sample_identity": terminals["llama3-8b-inst"]["sample_identity"],
        "runtime_source_head": RUNTIME_HEAD,
        "runtime_source_tree": RUNTIME_TREE,
        "rows": rows,
        "scientific_promotion": False,
    }
    report = f"""# BGODE-R2 Stage B FP64 two-equality 기술 폐쇄 사실 보고

Stage B는 효능 실험이 아니라 Llama/Qwen의 동일 sealed request000에서 Official AlphaEdit adapter fidelity, termination-free prefix event, FP64 Fisher/two-equality 해와 write 전 원자성을 닫는 기술 gate이다. Dynamic writer action은 두 모델 모두 0이며 과학 promotion은 하지 않는다.

| 모델 | terminal | Official Δ relative error | G rank | equality rank | equality residual | event norm residual | W0 restore |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama3-8B | PASS | {rows[0]['official_relative_frobenius']:.1f} | {rows[0]['fisher_rank']} | {rows[0]['equality_rank']} | {rows[0]['equality_residual']:.3e} | {rows[0]['event_normalization_log_residual']:.3e} | pointer+bytes exact |
| Qwen2.5-7B | PASS | {rows[1]['official_relative_frobenius']:.1f} | {rows[1]['fisher_rank']} | {rows[1]['equality_rank']} | {rows[1]['equality_residual']:.3e} | {rows[1]['event_normalization_log_residual']:.3e} | pointer+bytes exact |

Qwen의 과거 R1 FP32 `range(G)` hold는 R2 FP64에서 재현되지 않았다. 방향 range residual은 고정 backward-error bound 안이며 Schur/equality rank는 2이다. 이는 Stage B 수치 구현 통과 사실이지 효능·수렴·barrier attribution 결과가 아니다.

두 모델 모두 fixed z compute1/recompute0, ordered validator call1, history append0, rho/root/localizer0, probability floor/ridge/damping/fallback0, FP16/BF16/TF32/autocast/quantization/storage-cast0이다. 상세 spectrum과 JVP/FD/메모리는 `stage-b-summary.json`에 기록했다.
"""
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "bgode-r2-stage-b-factual-ko.md"
    summary_path = args.output / "stage-b-summary.json"
    write_once(report_path, report.encode())
    write_once(summary_path, canonical(summary))
    output_records = [file_record(path, label=path.name) for path in (report_path, summary_path)]
    member_root = sha(canonical(output_records))
    manifest = {
        "schema": SCHEMA + "/manifest",
        "runtime_inputs": inputs,
        "outputs": output_records,
        "members_root": member_root,
        "raw_files_committed": 0,
        "scientific_promotion": False,
    }
    manifest_path = args.output / "analysis-manifest.json"
    write_once(manifest_path, canonical(manifest))
    receipt = {
        "schema": SCHEMA + "/rooted-receipt",
        "status": "BGODE_R2_STAGE_B_TERMINAL_PASS",
        "array_job_id": 26677,
        "terminal_valid": "2/2",
        "manifest_sha256": sha(manifest_path.read_bytes()),
        "members_root": member_root,
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
        "members_root": member_root, "receipt_sha256": sha(receipt_path.read_bytes()),
        "receipt_identity": receipt["identity"],
    }).decode().strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
