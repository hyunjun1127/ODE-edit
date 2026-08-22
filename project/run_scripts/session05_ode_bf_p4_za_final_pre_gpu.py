#!/usr/bin/env python3
"""Essential-gates-only, no-model/no-CUDA final preflight for the P4 ZA pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p4_hf_consumed_closure import (
    EXPECTED_ALIASES,
    load_p4_hf_consumed_closure_seal,
)
from project.run_scripts.ode_bf.p4_sealed_stream import (
    TRANSFER_V2_EVALUATOR_IDENTITY,
    TRANSFER_V2_ORDER_SHA256,
    TRANSFER_V2_STREAM_ROOT,
    preflight_transferred_stream_v2,
)
from project.run_scripts.ode_bf.p4_semantic_barrier import P4_INSTRUCTION_ID
from project.run_scripts.ode_bf.p1_runtime import _atomic_write_once


SOURCE_MANIFEST = Path(
    "project/run_scripts/ode_bf/locks/source_manifest_s05_p4_za_pilot.json"
)
HF_SEAL = Path("agents/server4/p4-hf-consumed-closure-seal.json")
EASYEDIT_SEAL = Path("agents/server4/alphaedit-runtime-path-seal.json")
NUMERICAL_LOCK = Path(
    "project/run_scripts/ode_bf/locks/numerical_lock_s05_p4_target_side_semantic_barrier.json"
)
DATASET = Path("/data/janghj/EasyEdit/data/counterfact/counterfact.json")
EASYEDIT_ROOT = Path("/data/janghj/EasyEdit")
HPARAMS = {
    "llama3-8b-inst": (
        "hparams/AlphaEdit/llama3-8b.yaml",
        1_134,
        "d403e1875e62096b089be5343d33896510e454b0cdd2b256618ab53de6609ef8",
    ),
    "qwen2.5-7b-inst": (
        "hparams/AlphaEdit/qwen2.5-7b.yaml",
        695,
        "82d04976c4ab65e67c537ac3bd1b04d42c8f7527e2a749bdefcce63e43b995c3",
    ),
}


def _write_or_verify_once(path: Path, value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    expected = hashlib.sha256(payload).hexdigest()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError("P4 ZA create-once receipt differs")
        return expected
    return _atomic_write_once(path, value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_rooted(path: Path, *, schema: str) -> tuple[dict[str, object], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("rooted preflight input is absent or symlinked")
    raw = path.read_bytes()
    value = json.loads(raw)
    root = value.get("root_digest")
    payload = dict(value)
    payload.pop("root_digest", None)
    if value.get("schema_version", value.get("schema")) != schema or root != canonical_hash(payload):
        raise ValueError("rooted preflight input identity differs")
    return value, hashlib.sha256(raw).hexdigest()


def _source_gate(path: Path, source_head: str) -> dict[str, object]:
    value, raw_sha = _load_rooted(
        path,
        schema="ode-edit-s05-p4-za-pilot-source-manifest/v1",
    )
    if (
        value.get("instruction_id") != P4_INSTRUCTION_ID
        or value.get("source_parent") is None
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", str(value["source_parent"]), source_head],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
        != 0
    ):
        raise ValueError("P4 ZA source ancestry differs")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("P4 ZA source manifest is empty")
    paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("P4 ZA source manifest member differs")
        relative = str(entry["path"])
        member = REPO_ROOT / relative
        if (
            member.is_symlink()
            or not member.is_file()
            or member.stat().st_size != entry.get("size")
            or _sha256(member) != entry.get("sha256")
        ):
            raise ValueError("P4 ZA source bytes differ")
        paths.append(relative)
    if paths != sorted(set(paths)):
        raise ValueError("P4 ZA source manifest ordering differs")
    return {
        "path": str(path),
        "sha256": raw_sha,
        "root_digest": value["root_digest"],
        "member_count": len(entries),
    }


def build_receipts(
    *,
    archive: Path,
    extract_root: Path,
    transfer_receipt_path: Path,
    final_receipt_path: Path,
    source_head: str,
    session_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if (
        socket.gethostname() != "server4"
        or subprocess.run(
            ["git", "config", "--get", "agent.role"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        != "server-head"
        or subprocess.run(
            ["git", "config", "--get", "agent.hostname"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        != "server4"
        or os.environ.get("PROJECT_GPU_CAP", "2") != "2"
    ):
        raise ValueError("P4 ZA server4 agent/cap identity differs")
    observed_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, check=True, text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if observed_head != source_head or subprocess.run(
        ["git", "diff", "--quiet"], cwd=REPO_ROOT, check=False
    ).returncode != 0 or subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT, check=False
    ).returncode != 0:
        raise ValueError("P4 ZA source HEAD/worktree differs")

    transfer = preflight_transferred_stream_v2(
        extract_root, archive=archive, dataset_path=DATASET
    )
    _write_or_verify_once(transfer_receipt_path, transfer)

    source = _source_gate(REPO_ROOT / SOURCE_MANIFEST, source_head)
    numerical, numerical_sha = _load_rooted(
        REPO_ROOT / NUMERICAL_LOCK,
        schema="ode-edit-s05-p4-target-side-semantic-barrier-lock/v1",
    )
    if numerical.get("instruction_id") != P4_INSTRUCTION_ID:
        raise ValueError("P4 ZA numerical lock differs")

    hf = load_p4_hf_consumed_closure_seal(REPO_ROOT / HF_SEAL, repo_root=REPO_ROOT)
    hf_binding: dict[str, object] = {}
    for alias in EXPECTED_ALIASES:
        model = hf.model(alias)
        hf_binding[alias] = {
            "alias": alias,
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

    easyedit_value, easyedit_sha = _load_rooted(
        REPO_ROOT / EASYEDIT_SEAL,
        schema="ode-edit-alphaedit-runtime-path-seal/v1",
    )
    mapping = easyedit_value.get("mappings")
    if (
        not isinstance(mapping, dict)
        or mapping.get("easyedit_root")
        != {
            "logical_root": "/mnt/raid5/janghj/EasyEdit",
            "runtime_root": str(EASYEDIT_ROOT),
        }
    ):
        raise ValueError("P4 ZA EasyEdit root seal differs")
    hparams: dict[str, object] = {}
    for alias, (relative, size, sha) in HPARAMS.items():
        path = EASYEDIT_ROOT / relative
        observed = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_size != size
            or _sha256(path) != sha
        ):
            raise ValueError("P4 ZA hparams identity differs")
        hparams[alias] = {"path": str(path), "size": size, "sha256": sha}

    case01 = transfer["batch_ordered_request_digest_v1"][0]
    model_input_binding = {
        alias: {
            "alias": alias,
            "native_name": hf.model(alias).native_name,
            "case_index": 1,
            "request_order_sha256": case01,
            "stream_root": TRANSFER_V2_STREAM_ROOT,
            "all_request_order_sha256": TRANSFER_V2_ORDER_SHA256,
            "evaluator_identity": TRANSFER_V2_EVALUATOR_IDENTITY,
            "dataset_identity": transfer["binding"]["dataset_identity"],
        }
        for alias in EXPECTED_ALIASES
    }
    final: dict[str, object] = {
        "schema": "ode-edit-s05-p4-za-server4-final-pre-gpu/v1",
        "instruction_id": P4_INSTRUCTION_ID,
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
        "model_input_binding": model_input_binding,
        "hf_seal_path": str(REPO_ROOT / HF_SEAL),
        "hf_closure_binding": hf_binding,
        "hf_content_verification": "PRIOR_ACCEPTED_EXACT_CONSUMED_CLOSURE_REUSED",
        "hf_duplicate_rehash_count": 0,
        "easyedit_seal": {
            "path": str(REPO_ROOT / EASYEDIT_SEAL),
            "sha256": easyedit_sha,
            "root_digest": easyedit_value["root_digest"],
            "runtime_root": str(EASYEDIT_ROOT),
            "hparams": hparams,
        },
        "numerical_lock": {
            "path": str(REPO_ROOT / NUMERICAL_LOCK),
            "sha256": numerical_sha,
            "root_digest": numerical["root_digest"],
        },
        "execution_binding": {
            "stage": "ZA",
            "case_index": 1,
            "arms": ["Z+", "Z±", "Native-Z"],
            "same_W0": True,
            "writer_count": 0,
            "positive_negative_only_difference": "softplus(s_minus-s_plus)",
            "inner_iterations": 5,
            "outer_steps": 8,
            "final_iterate_only": True,
            "moment_reset_each_outer": True,
            "heldout_decision_influence_count": 0,
        },
        "full_fp32_offline": {
            "loader": "load_p4_full_fp32_from_sealed_snapshot",
            "snapshot_argument": "sealed_absolute_snapshot",
            "dtype": "torch.float32",
            "autocast": False,
            "quantization": False,
            "local_files_only": True,
            "environment": {
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_DISABLED": "true",
            },
        },
        "resources_per_cell": {
            "gpu": 1,
            "cpu": 8,
            "memory_mib": 65000,
            "walltime_hours": 48,
        },
        "cuda_api_access_count": 0,
        "model_load_count": 0,
        "slurm_submit_count": 0,
    }
    final["identity_sha256"] = canonical_hash(final)
    _write_or_verify_once(final_receipt_path, final)
    return transfer, final


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--extract-root", required=True, type=Path)
    parser.add_argument("--transfer-receipt", required=True, type=Path)
    parser.add_argument("--final-receipt", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    transfer, final = build_receipts(
        archive=args.archive.resolve(strict=True),
        extract_root=args.extract_root.resolve(strict=True),
        transfer_receipt_path=args.transfer_receipt,
        final_receipt_path=args.final_receipt,
        source_head=args.source_head,
        session_id=args.session_id,
    )
    print(json.dumps({
        "transfer": transfer["status"],
        "transfer_identity": transfer["identity_sha256"],
        "final": final["status"],
        "final_identity": final["identity_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
