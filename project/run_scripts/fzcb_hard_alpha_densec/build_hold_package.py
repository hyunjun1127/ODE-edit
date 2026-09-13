"""Build the create-once factual package for an HA1 numerical HOLD."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any


CONTRACT_SHA = "65d1f80291d64a6cdf97f7e6072d7fe33cb1595f5fcd7462f42e8ad7ecd1b35a"
OFFICIAL_HEAD = "3488a66ee988d83ee7891a8abbbe6bcb24a77daf"
OFFICIAL_TREE = "1f6d5e9a4a95daa15a4b5a8963dcd133876e47b3"
QWEN_EVIDENCE = {
    "layer": 4,
    "history": "zero",
    "relative_update_difference": 0.00096076593035832047,
    "tolerance": 0.00048828125,
    "official_equation_residual": 1.1518749261085759e-06,
    "woodbury_solve_relative": 3.019503935774992e-08,
    "woodbury_direct_d_relative": 0.00017931935144588351,
    "small_condition": 1.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_once(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def file_identity(path: Path) -> dict[str, Any]:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or path.is_symlink():
        raise RuntimeError(f"not a regular non-symlink: {path}")
    return {"path": str(path), "bytes": value.st_size, "mode": f"{stat.S_IMODE(value.st_mode):04o}", "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--llama-result", type=Path, required=True)
    parser.add_argument("--qwen-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_root.resolve()
    if git(source, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked source is dirty")
    if sha256(args.contract) != CONTRACT_SHA:
        raise RuntimeError("contract identity mismatch")
    if git(args.official_root, "rev-parse", "HEAD") != OFFICIAL_HEAD or git(args.official_root, "rev-parse", "HEAD^{tree}") != OFFICIAL_TREE:
        raise RuntimeError("Official source identity mismatch")
    if git(args.official_root, "status", "--porcelain"):
        raise RuntimeError("Official source is dirty")
    llama = json.loads(args.llama_result.read_text())
    if llama["status"] != "TERMINAL_VALID" or llama["model"] != "llama3-8b-inst":
        raise RuntimeError("Llama HA1 result is not terminal-valid")
    qwen_text = args.qwen_log.read_text()
    for name, value in QWEN_EVIDENCE.items():
        if name in {"layer", "history"}:
            token = f"{name}={value}"
        else:
            token = f"{name}={value:.17g}"
        if token not in qwen_text:
            raise RuntimeError(f"Qwen evidence missing: {token}")

    output = args.output_dir
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    layer_rows = []
    for layer in llama["layers"]:
        for history, parity in layer["parity"].items():
            layer_rows.append({"layer": layer["layer"], "history": history, **parity})
    summary = {
        "schema": "odeedit.s06.fzcb-hard-alpha-densec.ha1-hold-summary.v1",
        "status": "SENSITIVITY_NUMERICALLY_INCONCLUSIVE",
        "scientific_failure": False,
        "scientific_promotion": False,
        "instruction_id": "ODEEDIT-S06-FZCB-HARD-ALPHA-DENSEC-HA0-B10-V1",
        "source": {"branch": git(source, "branch", "--show-current"), "head": git(source, "rev-parse", "HEAD"), "tree": git(source, "rev-parse", "HEAD^{tree}"), "parent": git(source, "rev-parse", "HEAD^")},
        "contract": file_identity(args.contract),
        "official_source": {"path": str(args.official_root), "head": OFFICIAL_HEAD, "tree": OFFICIAL_TREE, "tracked_clean": True},
        "ha0": {"focused_tests": {"passed": 13, "total": 13}, "preflight": file_identity(args.preflight), "gpu_action_count": 0},
        "ha1": {
            "llama": {"job": "31020_0", "result": file_identity(args.llama_result), "status": "TERMINAL_VALID", "case_id": llama["case_id"], "layers": layer_rows, "w0_pointer_restore": llama["w0_pointer_restore"], "w0_bytes_restore": llama["w0_bytes_restore"], "cache_append_count": llama["cache_append_count"]},
            "qwen": {"jobs": ["31020_1", "31024_1"], "status": "SENSITIVITY_NUMERICALLY_INCONCLUSIVE", "evidence": QWEN_EVIDENCE, "terminal_endpoint_count": 0, "imputation_count": 0, "log": file_identity(args.qwen_log)},
        },
        "technical_history": [
            {"job": "31006_[0-1]", "label": "PRE_MODEL_PREFLIGHT_SHA_ARGUMENT_DIAGNOSTIC_GAP", "model_load_count": 0},
            {"job": "31015_[0-1]", "label": "PINNED_OFFICIAL_NETHOOK_WITH_KWARGS_SIGNATURE", "target_compute_count": 0},
            {"job": "31020_1", "label": "QWEN_DENSE_PARITY_BOUNDARY"},
            {"job": "31024_1", "label": "QWEN_DENSE_PARITY_EVIDENCE_REPRODUCED"},
        ],
        "gates": {"ha0": "PASS", "ha1_llama": "PASS", "ha1_qwen": "HOLD", "ha2_atomic_b10": "NOT_RELEASED", "ha3_b10x10_submission_count": 0, "b100x10_submission_count": 0},
        "plan": {"ha2_release_condition": "Qwen direct/Woodbury numerical parity authority resolved without changing equation or outcome", "ha3": "plan only after HA2 terminal-valid and GH review", "b100x10": "not authorized; compute/memory estimate remains NOT_RECORDED before HA2 timing"},
    }
    summary_path = output / "ha1-hold-summary.json"
    write_once(summary_path, json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")

    llama_diffs = [row["relative_update_difference"] for row in layer_rows]
    llama_direct = [row["woodbury_direct_d_relative"] for row in layer_rows]
    report = f"""# FzCB Hard-Alpha Dense-C HA0/HA1 수치 HOLD 사실 보고서

## 한눈에 보는 결론

|단계|Llama|Qwen|다음 단계|
|---|---:|---:|---|
|HA0 model-free dense backend|13/13 PASS|공통|HA1 허용|
|HA1 Official replay B1|TERMINAL_VALID|SENSITIVITY_NUMERICALLY_INCONCLUSIVE|HA2/B10 미제출|

Llama는 zero/nonzero dense-history 전 layer에서 direct Official 식과 LU/Woodbury backend parity, FP32, padding/position, cache mutation0, W0 pointer+bytes restore를 통과했다. Qwen layer4 zero-history에서는 두 solve residual이 모두 작고 D 차이는 lock 이내였지만, residual과 곱한 최종 update 차이는 lock 밖으로 이동했다. 계약의 numerical-boundary 규칙에 따라 tolerance를 완화하거나 이를 구조적 hard-Alpha 실패로 부르지 않고 중단했다.

## 1. 권위 입력과 source

- contract: `{args.contract}` / `{CONTRACT_SHA}` / 27,954 bytes / 1,437 lines / mode0600.
- implementation: `{summary['source']['branch']}` / HEAD `{summary['source']['head']}` / tree `{summary['source']['tree']}` / parent `{summary['source']['parent']}`.
- Official source: `{args.official_root}` / HEAD `{OFFICIAL_HEAD}` / tree `{OFFICIAL_TREE}` / clean read-only.
- pinned user EasyEdit dirty checkout edit/reset 영향0. Official `AlphaEdit_main.py` normal equation, RHS, shape adapter, post-terminal K append semantics는 preflight에서 독립 봉인했다.

## 2. HA0 backend 및 transaction

- nonsymmetric `A_H=lambda I+P C_H`: LU factorization/solve. Cholesky/CG/inverse0.
- `D=Q S^-1`: `solve(S.T,Q.T).T`; C_H=0/nonzero direct parity, P=I, hard-range/right leakage, incidence, orientation PASS.
- matrix-free LSQR/null projection/envelope FP64 FD/analytic alpha/terminal limit PASS; explicit Gram/pseudoinverse/null basis/Kronecker0.
- cache checkpoint+append-only WAL replay, rejected append0, successful append-once test PASS.
- pinned Official PyTorch hook signature는 clean source를 수정하지 않고 ODE thin adapter에서 `(module,args,kwargs,output)`으로만 교정했다.

## 3. HA1 Llama 결과

- job `31020_0`, case `{llama['case_id']}`, result `{args.llama_result}`.
- update relative difference 범위 `{min(llama_diffs):.6e}..{max(llama_diffs):.6e}`; D direct/Woodbury relative 범위 `{min(llama_direct):.6e}..{max(llama_direct):.6e}`.
- target compute1/recompute0, cache append0/mutation0, FULL FP32, BF16/FP16/autocast0.
- W0 pointer restore={llama['w0_pointer_restore']}, bytes restore={llama['w0_bytes_restore']}.

## 4. HA1 Qwen 첫 false gate

|항목|값|
|---|---:|
|layer/history|4 / zero|
|Official direct equation residual|{QWEN_EVIDENCE['official_equation_residual']:.9e}|
|Woodbury full-equation residual|{QWEN_EVIDENCE['woodbury_solve_relative']:.9e}|
|D direct↔Woodbury relative|{QWEN_EVIDENCE['woodbury_direct_d_relative']:.9e}|
|최종 update relative|{QWEN_EVIDENCE['relative_update_difference']:.9e}|
|고정 backend tolerance|{QWEN_EVIDENCE['tolerance']:.9e}|
|small-system condition|{QWEN_EVIDENCE['small_condition']:.1f}|

동일 evidence는 job `31020_1`과 진단-only job `31024_1`에서 재현됐다. D parity는 통과하지만 최종 update parity는 실패하므로 어떤 receipt를 authoritative parity로 채택하는지에 따라 HA1 판정이 바뀐다. 이는 `SENSITIVITY_NUMERICALLY_INCONCLUSIVE`이며 임의 tolerance 변경·direct-only fallback·post-hoc projection을 적용하지 않았다. Qwen terminal endpoint denominator는 0, imputation0이다.

## 5. 기술 이력과 실행 경계

- `31006_[0-1]`: model load/science0, preflight SHA mismatch 메시지의 actual/expected 증거 누락; fail root immutable.
- `31015_[0-1]`: model load 후 target0, pinned Official hook signature incompatibility; thin adapter로만 수리, source bytes 불변.
- `31020_0`: Llama valid. `31020_1`: Qwen numerical boundary. `31024_1`: Qwen evidence-only 재현.
- HA2 four-arm B1/Atomic B10, HA3 B10×10, B100×10 submission은 모두 0. cache history claim0, scientific promotion=false.

## 6. 후속 plan

HA2 release에는 Qwen direct/Woodbury parity authority를 식·target·stream·tolerance 변경 없이 명시적으로 닫는 numerical decision이 필요하다. 그 전에는 Official/Frozen/Equality/FzCB 네 arm의 B1 또는 B10을 제출하지 않는다. HA3와 B100×10은 계획만 유지하며, HA2 timing이 없으므로 B100×10 compute/memory 수치 추정은 `NOT_RECORDED`이다.
"""
    report_path = output / "fzcb-hard-alpha-densec-ha0-ha1-hold-factual-ko.md"
    write_once(report_path, report.encode())

    members = [file_identity(report_path), file_identity(summary_path)]
    root_digest = hashlib.sha256(canonical([{"name": Path(item["path"]).name, "sha256": item["sha256"], "bytes": item["bytes"], "mode": item["mode"]} for item in members])).hexdigest()
    manifest = {"schema": "odeedit.s06.fzcb-hard-alpha-densec.ha1-hold-analysis-manifest.v1", "status": summary["status"], "members": members, "root_digest": root_digest}
    manifest_path = output / "analysis-manifest.json"
    write_once(manifest_path, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
    receipt = {"schema": "odeedit.s06.fzcb-hard-alpha-densec.ha1-hold-rooted-receipt.v1", "status": summary["status"], "manifest": file_identity(manifest_path), "root_digest": root_digest, "model_gpu_slurm_rerun_after_hold": 0, "ha2_submission_count": 0, "ha3_submission_count": 0, "b100x10_submission_count": 0}
    receipt_path = output / "rooted-analysis-receipt.json"
    write_once(receipt_path, json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
    print(json.dumps({"report": file_identity(report_path), "manifest": file_identity(manifest_path), "receipt": file_identity(receipt_path), "root_digest": root_digest}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
