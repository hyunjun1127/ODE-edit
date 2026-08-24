#!/usr/bin/env python3
"""No-model/no-CUDA final gate for P1R54 FZ sequential B1->B10."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.artifacts import ODEBFArtifactGuard, sha256_file
from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1r52_target_timescale_deployment import (
    build_target_timescale_deployment,
)
from project.run_scripts.ode_bf.p1r54_fz_c3_sequential import (
    INSTRUCTION_ID,
    RESULT_NAME,
    ROLE,
    STREAM_ORDER,
    STREAM_ROOT,
    atomic_b1_source_equivalence,
)
from project.run_scripts.session05_ode_bf_p1r52_target_timescale_b100_preflight import (
    EXTRACT_ROOT,
    PRIOR_HF_GATE,
    _read_json,
    _rooted,
    _verify_stream,
    _write_or_verify_once,
)
from project.run_scripts.session05_ode_bf_p1r54_fz_c3_sequential_dry_plan import (
    build_plan,
)


PROJECT_GPU_CAP = 4
JOB_GPU_COUNT = 1
NUMERICAL_LOCK = Path(
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p1r54_fz_c3_sequential_10xb100_v1.json"
)
ATOMIC_FZ_ROOT = Path(
    "/data/janghj/ODE-edit/local/worktrees/"
    "p1r54-energyfree-localz-llama-b100-v1/local/odebf/results/"
    "s05-p1r54-energyfree-localz-llama-b100-fz-tech-r1-v1"
)
ATOMIC_FZ_TERMINAL_SHA = "89ad4231916278f6ba087004106fb1870a3227f903e7e499eef63e7dab1b08d2"
ATOMIC_FZ_MANIFEST_SHA = "697450bc79649b92abc50582d61f02106ffcbda12a966cbcba6c2ca294a6f478"
PHASE123_STATUS = Path(
    "experiment-reports/servers/server1/2026-08-23-p1r52-c-writer-phase123/"
    "phase123-canonical-status.json"
)
PHASE123_STATUS_SHA = "a2e4d5997fb4b616298c3ebd827fe4d2c38eebb713cc70dd2e2ab750999297a6"
REFERENCE_REPORTS = {
    "phase1_one_shot_and_official_sequential": (
        Path(
            "experiment-reports/servers/server1/2026-08-23-p1r52-c-writer-phase1/"
            "final-presentation-v5/p1r52-c-writer-phase1-full-fp32-sequential-"
            "tech-r1-exhaustive-factual-ko.md"
        ),
        "b93391e5c344bd5635fdd1c2f6532d41e683a88d85dabb3be21e949f05f3891b",
    ),
    "phase2_c3_independent": (
        Path(
            "experiment-reports/servers/server1/2026-08-23-p1r52-c-writer-phase2/"
            "exhaustive-v1/p1r52-c-writer-phase2-independent-kstep-full-fp32-"
            "exhaustive-factual-ko.md"
        ),
        "7d34b40f4e1983061113dbce1afb771436c187ae54286e4cc349d47616e2a6b6",
    ),
    "phase3_c3_cache_sequential": (
        Path(
            "experiment-reports/servers/server1/2026-08-23-p1r52-c-writer-phase3/"
            "exhaustive-v1/p1r52-c-writer-phase3-kstep-cache-sequential-full-"
            "fp32-exhaustive-factual-ko.md"
        ),
        "fbb790618655f7c975e0f5c42dff3860c8159880be44e38a258fe046a26047d8",
    ),
}
SOURCE_FILES = (
    "agents/server4/alphaedit-runtime-path-seal.json",
    "agents/server4/p4-hf-consumed-closure-seal.json",
    "project/run_scripts/alphaedit_runtime_path_seal.py",
    "project/run_scripts/ode_alloc/p0_artifacts.py",
    "project/run_scripts/ode_bf/artifacts.py",
    "project/run_scripts/ode_bf/contracts.py",
    "project/run_scripts/ode_bf/p1_runtime.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep_cache.py",
    "project/run_scripts/ode_bf/p1r52_c_writer_kstep_cache_sequential.py",
    "project/run_scripts/ode_bf/p1r52_joint_pc_fp32_runtime.py",
    "project/run_scripts/ode_bf/p1r52_sequential_runtime.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale.py",
    "project/run_scripts/ode_bf/p1r52_target_timescale_deployment.py",
    "project/run_scripts/ode_bf/p1r54_energyfree_localz.py",
    "project/run_scripts/ode_bf/p1r54_energyfree_localz_b100.py",
    "project/run_scripts/ode_bf/p1r54_fz_c3_sequential.py",
    "project/run_scripts/ode_bf/p4_hf_consumed_closure.py",
    "project/run_scripts/ode_bf/tests/test_p1r52_c_writer_kstep_cache.py",
    "project/run_scripts/ode_bf/tests/test_p1r54_energyfree_localz.py",
    "project/run_scripts/ode_bf/tests/test_p1r54_fz_c3_sequential.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_c3_sequential.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_c3_sequential_dry_plan.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_c3_sequential_preflight.py",
    "project/run_scripts/session05_ode_bf_p1r54_fz_c3_sequential_server4.sbatch",
    str(NUMERICAL_LOCK),
    "project/run_scripts/ode_bf/locks/p1r52_sequential_b100x10_stream_seal.json",
    str(PHASE123_STATUS),
    *(str(path) for path, _ in REFERENCE_REPORTS.values()),
)


def source_manifest(head: str, tree: str) -> Mapping[str, Any]:
    entries = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ODEBFContractError(f"P1R54 FZ sequential source member differs: {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    payload: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-c3-sequential-source-manifest/v1",
        "source_head": head,
        "source_tree": tree,
        "entries": entries,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def verify_references(*, b1_order: str) -> Mapping[str, Any]:
    atomic_terminal_path = ATOMIC_FZ_ROOT / "terminal.json"
    atomic_manifest_path = ATOMIC_FZ_ROOT / "manifest.json"
    atomic = _read_json(atomic_terminal_path)
    if (
        sha256_file(atomic_terminal_path) != ATOMIC_FZ_TERMINAL_SHA
        or sha256_file(atomic_manifest_path) != ATOMIC_FZ_MANIFEST_SHA
        or atomic.get("status") != "TERMINAL_VALID"
        or atomic.get("role") != "r54-energyfree-localz-b100-fz"
        or atomic.get("request_count") != 100
        or atomic.get("request_order_sha256") != b1_order
        or atomic.get("stream_root") != STREAM_ROOT
        or atomic.get("stream_order") != STREAM_ORDER
        or atomic.get("W0_restored") is not True
        or atomic.get("amplitude_policy_terminal", {}).get("arm") != "FZ"
        or atomic.get("amplitude_policy_terminal", {}).get("rho_refresh_count") != 0
        or atomic.get("p1r54_writer_layer_apply_count") != 40
    ):
        raise ODEBFContractError("P1R54 atomic FZ reference differs")

    status_path = REPO_ROOT / PHASE123_STATUS
    status = _read_json(status_path)
    if (
        sha256_file(status_path) != PHASE123_STATUS_SHA
        or status.get("status") != "PHASE123_TERMINAL_VALID_CANONICAL_REPORTS_COMPLETE"
        or status.get("full_fp32") is not True
        or status.get("scientific_promotion") is not False
        or status.get("stream", {}).get("stream_root_sha256") != STREAM_ROOT
        or status.get("stream", {}).get("all_request_order_sha256") != STREAM_ORDER
        or status.get("stream", {}).get("request_count") != 1000
        or status.get("stream", {}).get("sample_payload_count") != 1
        or status.get("stream", {}).get("sample_duplication_count") != 0
        or any(
            status.get("phases", {}).get(phase, {}).get("terminal_state")
            != "COMPLETED_0_0"
            for phase in ("phase1", "phase2", "phase3")
        )
    ):
        raise ODEBFContractError("P1R54 Phase123 reference status differs")
    reports = {}
    for label, (relative, expected_sha) in REFERENCE_REPORTS.items():
        path = REPO_ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
            raise ODEBFContractError(f"P1R54 reference report differs: {label}")
        reports[label] = {
            "path": str(path),
            "sha256": expected_sha,
            "execution_influence_count": 0,
        }
    return {
        "atomic_fz": {
            "root": str(ATOMIC_FZ_ROOT),
            "terminal_sha256": ATOMIC_FZ_TERMINAL_SHA,
            "manifest_sha256": ATOMIC_FZ_MANIFEST_SHA,
            "terminal_identity": atomic.get("identity_sha256"),
            "execution_influence_count": 0,
        },
        "phase123_status": {
            "path": str(status_path),
            "sha256": PHASE123_STATUS_SHA,
        },
        "reports": reports,
        "reference_execution_count": 0,
    }


def build_receipt(*, source_head: str, final_receipt: Path, session_id: str) -> Mapping[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, text=True
    ).strip()
    role = subprocess.check_output(
        ["git", "config", "--get", "agent.role"], cwd=REPO_ROOT, text=True
    ).strip()
    agent_host = subprocess.check_output(
        ["git", "config", "--get", "agent.hostname"], cwd=REPO_ROOT, text=True
    ).strip()
    if (
        socket.gethostname() != "server4"
        or role != "server-head"
        or agent_host != "server4"
        or head != source_head
        or os.environ.get("PROJECT_GPU_CAP", str(PROJECT_GPU_CAP))
        != str(PROJECT_GPU_CAP)
        or subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
    ):
        raise ODEBFContractError("P1R54 FZ sequential host/source/cap differs")

    stream = _verify_stream()
    lock, lock_sha = _rooted(REPO_ROOT / NUMERICAL_LOCK)
    if (
        lock.get("instruction_id") != INSTRUCTION_ID
        or lock.get("model_alias") != "llama3-8b-inst"
        or lock.get("stream_root") != STREAM_ROOT
        or lock.get("order_root") != STREAM_ORDER
        or lock.get("sequential_batches")
        != [f"B{index}" for index in range(1, 11)]
        or lock.get("total_request_count") != 1000
        or lock.get("arm") != "FZ"
        or lock.get("target_clock", {}).get("field_evaluations_total") != 80
        or lock.get("writer", {}).get("calls_total") != 80
        or lock.get("writer", {}).get("layer_applies_total") != 400
        or any(lock.get("forbidden_decision_access", {}).values())
    ):
        raise ODEBFContractError("P1R54 FZ sequential numerical lock differs")
    references = verify_references(b1_order=str(stream["B1_request_order_sha256"]))
    deployment = build_target_timescale_deployment(
        repo_root=REPO_ROOT,
        stream_extract_root=EXTRACT_ROOT,
        prior_final_pre_gpu_path=PRIOR_HF_GATE,
        dataset_identity=str(stream["dataset_identity"]),
    )
    guard = ODEBFArtifactGuard(
        REPO_ROOT,
        REPO_ROOT / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
        "llama3-8b-inst",
        require_held_ode_alloc=False,
        runtime_path_seal=deployment.runtime_path_seal,
        evaluator_source_paths=deployment.evaluator_source_paths,
        base_guard_override=deployment.base_guard,
    )
    artifact = guard.preflight()
    guard.assert_unchanged()
    dry_plan = build_plan()
    final: dict[str, Any] = {
        "schema": "ode-edit-s05-p1r54-fz-c3-sequential-final-pre-gpu/v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "FINAL_PRE_GPU_PASS",
        "model_load_authorized": True,
        "hostname": "server4",
        "agent_role": role,
        "agent_hostname": agent_host,
        "session_id": session_id,
        "project_gpu_cap": PROJECT_GPU_CAP,
        "job_gpu_count": JOB_GPU_COUNT,
        "source_head": head,
        "source_tree": tree,
        "source_manifest": source_manifest(head, tree),
        "authoritative_envelope": {
            "provenance": "GH_DELEGATED_AUTHORITATIVE_TASK_ENVELOPE",
            "instruction_id": INSTRUCTION_ID,
            "full_read": True,
        },
        "stream_binding": stream,
        "numerical_lock": {
            "path": str(REPO_ROOT / NUMERICAL_LOCK),
            "sha256": lock_sha,
            "root_digest": lock["root_digest"],
        },
        "atomic_b1_source_equivalence": atomic_b1_source_equivalence(),
        "external_references": references,
        "reference_execution_count": 0,
        "deployment_identity": deployment.identity_sha256,
        "easyedit_runtime_seal_root": deployment.runtime_path_seal.root_digest,
        "artifact_identity": artifact.identity(),
        "dry_plan": dry_plan,
        "result_name": RESULT_NAME,
        "role": ROLE,
        "scientific_promotion": False,
    }
    final["identity_sha256"] = canonical_hash(final)
    _write_or_verify_once(final_receipt, final)
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--final-receipt", required=True, type=Path)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            build_receipt(
                source_head=args.source_head,
                final_receipt=args.final_receipt,
                session_id=args.session_id,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
