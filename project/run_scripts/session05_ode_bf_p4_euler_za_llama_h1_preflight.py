#!/usr/bin/env python3
"""No-model/no-CUDA preflight for Llama P4-Euler ZA h=1 B2-B10."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p4_euler_integrator import P4_EULER_INSTRUCTION_ID
from project.run_scripts.ode_bf.p4_hf_consumed_closure import (
    load_p4_hf_consumed_closure_seal,
)
from project.run_scripts.ode_bf.p4_sealed_stream import (
    TRANSFER_V2_EVALUATOR_IDENTITY,
    TRANSFER_V2_ORDER_SHA256,
    TRANSFER_V2_STREAM_ROOT,
    preflight_transferred_stream_v2,
)
from project.run_scripts.session05_ode_bf_p4_za_final_pre_gpu import (
    _load_rooted,
    _sha256,
    _write_or_verify_once,
)


SOURCE_MANIFEST = Path(
    "project/run_scripts/ode_bf/locks/"
    "source_manifest_s05_p4_euler_za_llama_h1_b2b10.json"
)
NUMERICAL_LOCK = Path(
    "project/run_scripts/ode_bf/locks/"
    "numerical_lock_s05_p4_euler_za_llama_h1_b2b10.json"
)
HF_SEAL = Path("agents/server4/p4-hf-consumed-closure-seal.json")
EASYEDIT_SEAL = Path("agents/server4/alphaedit-runtime-path-seal.json")
EASYEDIT_ROOT = Path("/data/janghj/EasyEdit")
DATASET = EASYEDIT_ROOT / "data/counterfact/counterfact.json"
HPARAMS_RELATIVE = Path("hparams/AlphaEdit/llama3-8b.yaml")
HPARAMS_SIZE = 1_134
HPARAMS_SHA256 = "d403e1875e62096b089be5343d33896510e454b0cdd2b256618ab53de6609ef8"
MODEL_ALIAS = "llama3-8b-inst"
CASE_INDICES = tuple(range(2, 11))
SELECTED_H = 1.0
SELECTED_M = 5
SELECTED_TARGET_HORIZON = 5.0
LLAMA_BUNDLE = "2cd8517f65f476e3a8be4edddc81fc345ecd96a49634e41769bbacd4206af77e"


def _source_gate(path: Path, source_head: str) -> dict[str, object]:
    value, raw_sha = _load_rooted(
        path,
        schema="ode-edit-s05-p4-euler-za-llama-h1-b2b10-source-manifest/v1",
    )
    if (
        value.get("instruction_id") != P4_EULER_INSTRUCTION_ID
        or value.get("source_parent") is None
        or subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                str(value["source_parent"]),
                source_head,
            ],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("P4 Euler ZA source ancestry differs")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("P4 Euler ZA source manifest is empty")
    paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("P4 Euler ZA source manifest member differs")
        relative = str(entry["path"])
        member = REPO_ROOT / relative
        if (
            member.is_symlink()
            or not member.is_file()
            or member.stat().st_size != entry.get("size")
            or _sha256(member) != entry.get("sha256")
        ):
            raise ValueError(f"P4 Euler ZA source bytes differ: {relative}")
        paths.append(relative)
    if paths != sorted(set(paths)):
        raise ValueError("P4 Euler ZA source manifest ordering differs")
    return {
        "path": str(path),
        "sha256": raw_sha,
        "root_digest": value["root_digest"],
        "member_count": len(entries),
    }


def _evaluation_rows(identity_path: Path) -> dict[int, Mapping[str, Any]]:
    if identity_path.is_symlink() or not identity_path.is_file():
        raise ValueError("P4 Euler ZA evaluator identity file differs")
    value = json.loads(identity_path.read_bytes())
    rows = value.get("evaluation_context_identities")
    rows = rows.get("rows") if isinstance(rows, Mapping) else None
    selected = {
        int(row["case_index"]): row
        for row in rows or []
        if isinstance(row, Mapping)
        and row.get("model_alias") == MODEL_ALIAS
        and row.get("case_index") in CASE_INDICES
    }
    if tuple(sorted(selected)) != CASE_INDICES:
        raise ValueError("P4 Euler ZA evaluator B2-B10 rows differ")
    return selected


def build_receipt(
    *,
    archive: Path,
    extract_root: Path,
    transfer_receipt_path: Path,
    final_receipt_path: Path,
    source_head: str,
    session_id: str,
) -> Mapping[str, Any]:
    role = subprocess.run(
        ["git", "config", "--get", "agent.role"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    agent_host = subprocess.run(
        ["git", "config", "--get", "agent.hostname"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if (
        socket.gethostname() != "server4"
        or role != "server-head"
        or agent_host != "server4"
        or os.environ.get("PROJECT_GPU_CAP", "2") != "2"
    ):
        raise ValueError("P4 Euler ZA server4 role/cap differs")
    observed_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if (
        observed_head != source_head
        or subprocess.run(
            ["git", "diff", "--quiet"], cwd=REPO_ROOT, check=False
        ).returncode
        != 0
        or subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("P4 Euler ZA source/worktree differs")

    transfer = preflight_transferred_stream_v2(
        extract_root, archive=archive, dataset_path=DATASET
    )
    _write_or_verify_once(transfer_receipt_path, transfer)
    if (
        transfer.get("status") != "TRANSFER_FULL_READ_PASS"
        or len(transfer.get("batch_ordered_request_digest_v1", [])) != 10
        or transfer.get("binding", {}).get("slice_membership_root")
        != TRANSFER_V2_STREAM_ROOT
        or transfer.get("binding", {}).get("order_root")
        != TRANSFER_V2_ORDER_SHA256
        or transfer.get("binding", {}).get("evaluator_identity")
        != TRANSFER_V2_EVALUATOR_IDENTITY
    ):
        raise ValueError("P4 Euler ZA transfer identity differs")

    source = _source_gate(REPO_ROOT / SOURCE_MANIFEST, source_head)
    numerical, numerical_sha = _load_rooted(
        REPO_ROOT / NUMERICAL_LOCK,
        schema="ode-edit-s05-p4-euler-za-llama-h1-b2b10-lock/v1",
    )
    if (
        numerical.get("instruction_id") != P4_EULER_INSTRUCTION_ID
        or numerical.get("policy_provenance")
        != "USER_DIRECTED_POST_CALIBRATION_HYPERPARAMETER_REVISION"
        or numerical.get("selected_setting")
        != {
            "h": SELECTED_H,
            "M": SELECTED_M,
            "T_z": SELECTED_TARGET_HORIZON,
            "formula": "T_z=M*h",
            "under_edit_mitigation_intent": True,
            "model_or_arm_specific_branch_count": 0,
        }
        or numerical.get("confirmatory_stream", {}).get("slice_indices")
        != list(CASE_INDICES)
    ):
        raise ValueError("P4 Euler ZA numerical lock differs")

    hf = load_p4_hf_consumed_closure_seal(
        REPO_ROOT / HF_SEAL, repo_root=REPO_ROOT
    )
    model = hf.model(MODEL_ALIAS)
    easyedit, easyedit_sha = _load_rooted(
        REPO_ROOT / EASYEDIT_SEAL,
        schema="ode-edit-alphaedit-runtime-path-seal/v1",
    )
    llama_assets = easyedit.get("models", {}).get(MODEL_ALIAS)
    mapping = easyedit.get("mappings", {}).get("easyedit_root")
    hparams_path = EASYEDIT_ROOT / HPARAMS_RELATIVE
    observed_hparams = hparams_path.lstat()
    if (
        easyedit.get("root_digest")
        != numerical["asset_identity"]["easyedit_runtime_seal_root"]
        or not isinstance(llama_assets, Mapping)
        or llama_assets.get("bundle_sha256") != LLAMA_BUNDLE
        or mapping
        != {
            "logical_root": "/mnt/raid5/janghj/EasyEdit",
            "runtime_root": str(EASYEDIT_ROOT),
        }
        or hparams_path.is_symlink()
        or not stat.S_ISREG(observed_hparams.st_mode)
        or observed_hparams.st_size != HPARAMS_SIZE
        or _sha256(hparams_path) != HPARAMS_SHA256
        or hf.root_digest
        != numerical["asset_identity"]["hf_consumed_closure_seal_root"]
        or model.closure_identity
        != numerical["asset_identity"]["llama_hf_closure"]
    ):
        raise ValueError("P4 Euler ZA deployment asset identity differs")

    evaluator_identity_path = extract_root / (
        "canonical/stream-order-context-identity.json"
    )
    evaluator_rows = _evaluation_rows(evaluator_identity_path)
    digests = transfer["batch_ordered_request_digest_v1"]
    case_bindings: dict[str, Any] = {}
    for case_index in CASE_INDICES:
        request_order = digests[case_index - 1]
        row = evaluator_rows[case_index]
        locked = numerical["confirmatory_stream"]["ordered_request_sha256"][
            f"B{case_index}"
        ]
        if (
            request_order != locked
            or row.get("request_order_sha256") != request_order
            or row.get("evaluator_source_sha256")
            != "25c3f49039e7bacb58de6d9ab8139f6f24f21532434a08690e9ec71ea939e145"
            or row.get("aggregator_source_sha256")
            != "64f009b2fb648627a956b95abd801838d86edf04c8e1888d90ade0ec07b745c0"
        ):
            raise ValueError("P4 Euler ZA case/evaluator binding differs")
        case_bindings[f"B{case_index}"] = {
            "case_index": case_index,
            "request_order_sha256": request_order,
            "evaluation_case_identity_sha256": row[
                "evaluation_case_identity_sha256"
            ],
            "objective_plan_sha256_observation_only": row[
                "objective_plan_sha256"
            ],
            "capture_plan_sha256_observation_only": row["capture_plan_sha256"],
            "confirmatory_eligibility": True,
        }

    final: dict[str, Any] = {
        "schema": "ode-edit-s05-p4-euler-za-llama-h1-b2b10-final-pre-gpu/v1",
        "instruction_id": P4_EULER_INSTRUCTION_ID,
        "status": "FINAL_PRE_GPU_PASS",
        "model_load_authorized": True,
        "source_head": source_head,
        "source_tree": tree,
        "source_manifest": source,
        "session_id": session_id,
        "hostname": "server4",
        "agent_role": "server-head",
        "agent_hostname": "server4",
        "project_gpu_cap": 2,
        "transfer_receipt": transfer,
        "model_input_binding": {
            "alias": MODEL_ALIAS,
            "native_name": model.native_name,
            "revision": model.revision,
            "closure_identity": model.closure_identity,
            "snapshot_path": model.snapshot_path,
            "stream_root": TRANSFER_V2_STREAM_ROOT,
            "order_sha256": TRANSFER_V2_ORDER_SHA256,
            "evaluator_identity": TRANSFER_V2_EVALUATOR_IDENTITY,
            "case_bindings": case_bindings,
            "B1_excluded": True,
        },
        "hf_closure_binding": {
            MODEL_ALIAS: {
                "alias": MODEL_ALIAS,
                "closure_identity": model.closure_identity,
                "native_name": model.native_name,
                "required_bytes": model.required_bytes,
                "required_count": len(model.required_members),
                "required_root": model.required_root,
                "revision": model.revision,
                "seal_root_digest": hf.root_digest,
                "seal_sha256": hf.seal_sha256,
                "snapshot_path": model.snapshot_path,
            }
        },
        "hf_content_verification": (
            "PRIOR_ACCEPTED_EXACT_CONSUMED_CLOSURE_REUSED"
        ),
        "hf_duplicate_rehash_count": 0,
        "easyedit_seal": {
            "path": str(REPO_ROOT / EASYEDIT_SEAL),
            "sha256": easyedit_sha,
            "root_digest": easyedit["root_digest"],
            "runtime_root": str(EASYEDIT_ROOT),
            "llama_bundle": LLAMA_BUNDLE,
            "hparams": {
                "path": str(hparams_path),
                "size": HPARAMS_SIZE,
                "sha256": HPARAMS_SHA256,
            },
        },
        "hf_seal": {
            "path": str(REPO_ROOT / HF_SEAL),
            "root_digest": hf.root_digest,
            "seal_sha256": hf.seal_sha256,
            "required_root": model.required_root,
            "closure_identity": model.closure_identity,
            "duplicate_rehash_count": 0,
        },
        "numerical_lock": {
            "path": str(REPO_ROOT / NUMERICAL_LOCK),
            "sha256": numerical_sha,
            "root_digest": numerical["root_digest"],
        },
        "execution_binding": {
            "stage": "ZA_TARGET_ONLY",
            "interpretation": "EXPLORATORY_CONFIRMATION_AFTER_CALIBRATION",
            "policy_provenance": (
                "USER_DIRECTED_POST_CALIBRATION_HYPERPARAMETER_REVISION"
            ),
            "model": MODEL_ALIAS,
            "case_indices": list(CASE_INDICES),
            "arms": ["Z+", "Z±", "Native-Z"],
            "causal_arms": ["Z+", "Z±"],
            "h": SELECTED_H,
            "M": SELECTED_M,
            "T_z": SELECTED_TARGET_HORIZON,
            "same_W0": True,
            "writer_materialization_count": 0,
            "cache_append_count": 0,
            "outer_k8_repeat_count": 0,
            "optimizer": "NONE",
            "adam_state_count": 0,
            "loss_backward_count": 0,
            "parameter_gradient_count": 0,
            "final_iterate_only": True,
            "heldout_terminal_only": True,
            "high_clamp_fraction_stop_rule_count": 0,
            "nonfinite_or_W_mutation_fail_close": True,
            "qwen_submission_count": 0,
            "ZB_submission_count": 0,
        },
        "full_fp32_offline": {
            "loader": "load_p4_full_fp32_from_sealed_snapshot",
            "snapshot_argument": "sealed_absolute_snapshot",
            "dtype": "torch.float32",
            "autocast": False,
            "quantization": False,
            "local_files_only": True,
        },
        "resources_per_cell": {
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
            "walltime_hours": 48,
            "array_concurrency_max": 2,
        },
        "cuda_api_access_count": 0,
        "model_load_count": 0,
        "slurm_submit_count": 0,
        "scientific_promotion": False,
    }
    final["identity_sha256"] = canonical_hash(final)
    _write_or_verify_once(final_receipt_path, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--transfer-receipt", type=Path, required=True)
    parser.add_argument("--final-receipt", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    result = build_receipt(
        archive=args.archive.resolve(strict=True),
        extract_root=args.extract_root.resolve(strict=True),
        transfer_receipt_path=args.transfer_receipt,
        final_receipt_path=args.final_receipt,
        source_head=args.source_head,
        session_id=args.session_id,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "identity_sha256": result["identity_sha256"],
                "case_count": len(
                    result["model_input_binding"]["case_bindings"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
