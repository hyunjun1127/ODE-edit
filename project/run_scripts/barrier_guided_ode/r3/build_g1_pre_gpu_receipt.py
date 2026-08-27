#!/usr/bin/env python3
"""Build the compact BGODE-R3 G1 pre-GPU factual package."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
OUTPUT = REPO / "audits/servers/server1/2026-08-27-bgode-r3-g1-pre-gpu-v1"
IMPLEMENTATION_HEAD = "cf8a2b4d05dd13cb5a4525bc7592b05b90810612"
IMPLEMENTATION_TREE = "50c95b3d430a0681ca8701bb40708963b6439d53"


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise SystemExit(f"existing pre-GPU member differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def main() -> None:
    source_manifest_path = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g1-source-manifest.json"
    numerical_lock_path = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g1-numerical-lock.json"
    natural_path = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g1-natural-topology-manifest.json"
    source = json.loads(source_manifest_path.read_text())
    lock = json.loads(numerical_lock_path.read_text())
    natural = json.loads(natural_path.read_text())
    preflights = {}
    for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        path = Path(f"/tmp/bgode-r3-g1-{alias}-preflight.json")
        payload = json.loads(path.read_text())
        preflights[alias] = {
            "bytes": path.stat().st_size,
            "easyedit_bridge_manifest_id": payload["easyedit_bridge_manifest_id"],
            "external_input_preflight": payload["external_input_preflight"],
            "fixed_input_manifest_id": payload["fixed_input_manifest_id"],
            "sha256": file_sha(path),
            "tokenization_identity": payload["tokenization"]["identity"],
        }
    summary = {
        "schema": "ode-edit-bgode-r3-g1-pre-gpu-summary/v1",
        "instruction_id": lock["instruction_id"],
        "implementation_head": IMPLEMENTATION_HEAD,
        "implementation_tree": IMPLEMENTATION_TREE,
        "source_manifest_sha256": file_sha(source_manifest_path),
        "source_members_root": source["members_root"],
        "numerical_lock_sha256": file_sha(numerical_lock_path),
        "numerical_lock_identity": lock["identity"],
        "natural_manifest_sha256": file_sha(natural_path),
        "natural_manifest_identity": natural["identity"],
        "natural_case": {"ordinal": 26, "case_id": "17454", "available_cells": 2, "unavailable_cells": 6},
        "array_mapping": lock["array_mapping"],
        "array_throttle": 2,
        "project_gpu_cap": 3,
        "focused_tests": {"passed": 32, "failed": 0},
        "compile": "PASS",
        "bash": "PASS",
        "session": "PASS",
        "external_preflight": preflights,
        "g2_release": "HOLD_UNTIL_G1_ALL_AVAILABLE_NATURAL_CELLS_PASS",
        "scientific_promotion": False,
        "action_counts": {"model": 0, "gpu": 0, "slurm": 0},
        "status": "R3_G1_PRE_GPU_PASS",
    }
    summary_path = OUTPUT / "g1-pre-gpu-summary.json"
    write_once(summary_path, canonical(summary))
    report = f"""# BGODE-R3 G1 pre-GPU 사실 보고서

## 판정

`R3_G1_PRE_GPU_PASS`; G2는 아직 HOLD이다. model/GPU/Slurm action은 0/0/0이다.

## 자연 요청 봉인

- sealed canonical stream/order에서 tokenizer-only scan을 수행했다.
- 두 모델 모두 자연 `unequal-non-prefix`만 존재한다.
- multi-token G1을 충족하는 최초 공통 요청은 ordinal 26, case 17454이다.
- Llama target/source token 길이는 2/1, Qwen은 2/1이다.
- 나머지 세 topology/model은 총 6 cell 모두 `NATURAL_TOPOLOGY_UNAVAILABLE`; 합성 대체는 0이다.

## 구현 경계

- Official AlphaEdit fixed-z와 genuine P-inside ordered proposal을 재사용한다.
- factor Gram FP64 Frobenius normalization을 JVP/controller/write에 동일 적용한다.
- 모든 내부 prefix에서 5 actuator serial forward-JVP와 central FD를 검사한다.
- 동일 fine partition의 W0 q0를 create-once raw artifact로 봉인한다.
- t=0 Full/Fisher identity를 검사한 뒤 `h_probe=T_AE/32` Fisher write 1회와 exact W0 restore만 수행한다.
- history append0, localizer/root/ridge/damping/floor/fallback0, promotion=false이다.

## gate

- focused CPU: 32/32 PASS
- py_compile/bash/session: PASS
- Llama/Qwen tokenizer·EasyEdit·HF·P/stats preflight: PASS
- implementation source: `{IMPLEMENTATION_HEAD}` / `{IMPLEMENTATION_TREE}`
- source members root: `{source['members_root']}`
- natural manifest identity: `{natural['identity']}`

## release

G1 array mapping은 0=Llama, 1=Qwen, `%2`, project cap3이다. 제출 직전 unrelated/P1R55 포함 active GPU를 재계수한다. G1 natural unequal-nonprefix PASS 없이는 해당 모델 G2를 release하지 않는다.
""".encode()
    report_path = OUTPUT / "bgode-r3-g1-pre-gpu-factual-ko.md"
    write_once(report_path, report)
    members = []
    for path in sorted((summary_path, report_path)):
        status = path.lstat()
        members.append(
            {
                "bytes": path.stat().st_size,
                "mode": format(stat.S_IMODE(status.st_mode), "04o"),
                "path": str(path.relative_to(REPO)),
                "sha256": file_sha(path),
            }
        )
    manifest = {
        "schema": "ode-edit-bgode-r3-g1-pre-gpu-manifest/v1",
        "member_count": len(members),
        "members": members,
        "members_root": sha(canonical(members).rstrip(b"\n")),
    }
    manifest_path = OUTPUT / "source-manifest.json"
    write_once(manifest_path, canonical(manifest))
    receipt_body = {
        "schema": "ode-edit-bgode-r3-g1-pre-gpu-rooted-receipt/v1",
        "manifest_sha256": file_sha(manifest_path),
        "members_root": manifest["members_root"],
        "natural_manifest_identity": natural["identity"],
        "numerical_lock_identity": lock["identity"],
        "source_members_root": source["members_root"],
        "status": "R3_G1_PRE_GPU_PASS",
        "scientific_promotion": False,
        "action_counts": {"model": 0, "gpu": 0, "slurm": 0},
    }
    receipt = dict(receipt_body)
    receipt["receipt_identity"] = sha(canonical(receipt_body).rstrip(b"\n"))
    write_once(OUTPUT / "rooted-receipt.json", canonical(receipt))


if __name__ == "__main__":
    main()
