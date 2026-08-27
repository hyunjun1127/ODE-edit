#!/usr/bin/env python3
"""Build compact BGODE-R3 G2 pre-GPU factual package."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
OUTPUT = REPO / "audits/servers/server1/2026-08-27-bgode-r3-g2-pre-gpu-v1"
SOURCE_HEAD = "16616d872def3b2dcceafa2fd181187f957e3ad9"
SOURCE_TREE = "3533c36befbf48dbef876167a3fbbc771d06586c"
LOCK_PATH = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g2-cauchy-numerical-lock.json"
MANIFEST_PATH = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g2-source-manifest.json"


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise SystemExit(f"existing G2 pre-GPU member differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text())
    source = json.loads(MANIFEST_PATH.read_text())
    preflights = {
        "llama3-8b-inst": {
            "fixed_input_manifest_id": "273d06af367904aac61931331e6aa2ea09204b0db1f6c7557bca16a9e2c84d2b",
            "easyedit_bridge_manifest_id": "41d480254ad717b9c616a7226d810f9aa44b54410ff877541fde487bb25b76fd",
            "tokenization_identity": "91630af59900b02ed8f70a42ad5588f1497138b8c4d99cd3888b1929cbd6ca58",
            "status": "PASS",
        },
        "qwen2.5-7b-inst": {
            "fixed_input_manifest_id": "d247b27edf638b7a9625954e8bf2aeac745b61247230f8bcf733e991a3ed9d16",
            "easyedit_bridge_manifest_id": "41d480254ad717b9c616a7226d810f9aa44b54410ff877541fde487bb25b76fd",
            "tokenization_identity": "1d4410cda84be7f98ccf978fbd25f00f57b5801a74715af2c35bbcb59ba809da",
            "status": "PASS",
        },
    }
    summary = {
        "schema": "ode-edit-bgode-r3-g2-pre-gpu-summary/v1",
        "instruction_id": lock["instruction_id"],
        "implementation_head": SOURCE_HEAD,
        "implementation_tree": SOURCE_TREE,
        "source_manifest_sha256": sha_file(MANIFEST_PATH),
        "source_members_root": source["members_root"],
        "numerical_lock_sha256": sha_file(LOCK_PATH),
        "numerical_lock_identity": lock["identity"],
        "g1_terminal_receipt_identity": lock["g1_terminal_receipt_identity"],
        "natural_case": {"ordinal": 26, "case_id": "17454", "topology": "unequal-non-prefix"},
        "matrix": {
            "array_mapping": lock["array_mapping"],
            "array_throttle": 2,
            "gpu_per_task": 1,
            "project_gpu_cap": 3,
            "arms": lock["arms"],
            "n_grid": lock["n_grid"],
            "panel_count_per_model": 12,
        },
        "cauchy_gate": {
            "pairs": lock["cauchy_pairs"],
            "first_order_physical_ratio_floor": lock["first_order_physical_ratio_floor"],
            "final_relative_physical_distance_ceiling": lock[
                "final_relative_physical_distance_ceiling"
            ],
            "final_normalized_target_logit_gap_ceiling": lock[
                "final_normalized_target_logit_gap_ceiling"
            ],
            "final_normalized_q_kl_gap_ceiling": lock[
                "final_normalized_q_kl_gap_ceiling"
            ],
        },
        "focused_tests": {"passed": 41, "failed": 0},
        "compile": "PASS",
        "bash": "PASS",
        "session": "PASS",
        "external_preflight": preflights,
        "future_fixed_T_curve": {"status": "PREDECLARED_NOT_RUN", **lock["future_fixed_T_curve_predeclared_not_run"]},
        "scientific_promotion": False,
        "action_counts": {"model": 0, "gpu": 0, "slurm": 0},
        "status": "R3_G2_PRE_GPU_PASS",
    }
    summary_path = OUTPUT / "g2-pre-gpu-summary.json"
    write_once(summary_path, canonical(summary))
    report = f"""# BGODE-R3 G2 pre-GPU 사실 보고서

## 판정

`R3_G2_PRE_GPU_PASS`이다. G1 natural unequal-non-prefix는 Llama/Qwen 2/2 terminal PASS이며, G2는 동일 ordinal 26/case 17454만 사용한다. 실행 전 model/GPU/Slurm action은 0/0/0이다.

## 고정 matrix

- array: 0=Llama, 1=Qwen, `0-1%2`, 1 GPU/task, project cap3
- 각 모델: Plain/Fisher/Full × N={{4,8,16,32}} = 12 panel
- W0, fixed-z, T_AE, normalized actuator, equality `[1,0]`이 arm/N 사이 동일하다.
- 모든 node에서 ordered dictionary 재구축, 전체 internal-prefix serial JVP+central FD, factor-space residual과 actual dense W delta를 기록한다.
- history append0, rho/root/localizer/ridge/damping/floor/fallback0, evaluator/controller influence0이다.

## 결과 전 봉인한 Cauchy gate

- pair: D4,8 / D8,16 / D16,32
- actual dense block W endpoint 거리, target-logit gap, q0-conditional KL gap 모두 successive contraction 필요
- physical first-order ratio floor: 1.25
- final relative physical distance ceiling: 0.05
- final normalized target-logit/q-KL gap ceiling: 0.05/0.05
- rounding bound: `256*eps_FP32*(1+observed scalar scale)`
- 기준 실패 시 tolerance를 완화하지 않고 `R3_G2_NUMERICAL_CONVERGENCE_HOLD`로 보존한다.

## gate와 identity

- focused tests 41/41; py_compile/bash/session PASS
- Llama/Qwen EasyEdit/HF/tokenization preflight PASS
- source HEAD/tree: `{SOURCE_HEAD}` / `{SOURCE_TREE}`
- source members root: `{source['members_root']}`
- numerical lock identity: `{lock['identity']}`
- G1 terminal receipt identity: `{lock['g1_terminal_receipt_identity']}`

Future fixed-T curve T={{0.5,1,2,3,5}}, N=32는 predeclare만 했고 이번 G2에서 실행하지 않는다. scientific promotion=false이다.
""".encode()
    report_path = OUTPUT / "bgode-r3-g2-pre-gpu-factual-ko.md"
    write_once(report_path, report)
    members = []
    for path in sorted((summary_path, report_path)):
        members.append(
            {
                "bytes": path.stat().st_size,
                "mode": format(stat.S_IMODE(path.lstat().st_mode), "04o"),
                "path": str(path.relative_to(REPO)),
                "sha256": sha_file(path),
            }
        )
    manifest = {
        "schema": "ode-edit-bgode-r3-g2-pre-gpu-manifest/v1",
        "member_count": len(members),
        "members": members,
        "members_root": sha_bytes(canonical(members).rstrip(b"\n")),
    }
    manifest_path = OUTPUT / "source-manifest.json"
    write_once(manifest_path, canonical(manifest))
    receipt_body = {
        "schema": "ode-edit-bgode-r3-g2-pre-gpu-rooted-receipt/v1",
        "manifest_sha256": sha_file(manifest_path),
        "members_root": manifest["members_root"],
        "numerical_lock_identity": lock["identity"],
        "source_members_root": source["members_root"],
        "status": "R3_G2_PRE_GPU_PASS",
        "scientific_promotion": False,
    }
    receipt = dict(receipt_body)
    receipt["receipt_identity"] = sha_bytes(canonical(receipt_body).rstrip(b"\n"))
    write_once(OUTPUT / "rooted-receipt.json", canonical(receipt))


if __name__ == "__main__":
    main()
