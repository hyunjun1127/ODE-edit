"""Read-only source/artifact closure and pre-GPU receipt builder."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

import numpy as np
import torch

from .contracts import EngineeringBoundary, NumericalLock


PACKAGE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, expected_sha: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise EngineeringBoundary(f"regular non-symlink required: {path}")
    digest = sha256(path)
    if expected_sha is not None and digest != expected_sha:
        raise EngineeringBoundary(f"SHA mismatch: {path}")
    if expected_bytes is not None and metadata.st_size != expected_bytes:
        raise EngineeringBoundary(f"byte mismatch: {path}")
    return {
        "path": str(path),
        "sha256": digest,
        "bytes": metadata.st_size,
        "mode": format(stat.S_IMODE(metadata.st_mode), "04o"),
    }


def git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def canonical(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def source_call_inventory() -> dict[str, int]:
    counts = {
        "torch_inverse": 0,
        "torch_linalg_inv": 0,
        "torch_pinverse": 0,
        "torch_linalg_pinv": 0,
        "explicit_kron": 0,
        "dense_backend_cholesky": 0,
        "dense_backend_cg": 0,
    }
    production = [
        PACKAGE / "dense_backend.py",
        PACKAGE / "matrix_free.py",
        PACKAGE / "controller.py",
        PACKAGE / "cache_transaction.py",
    ]
    for path in production:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            try:
                name = ast.unparse(node.func)
            except Exception:
                continue
            mapping = {
                "torch.inverse": "torch_inverse",
                "torch.linalg.inv": "torch_linalg_inv",
                "torch.pinverse": "torch_pinverse",
                "torch.linalg.pinv": "torch_linalg_pinv",
                "torch.kron": "explicit_kron",
            }
            if name in mapping:
                counts[mapping[name]] += 1
            if path.name == "dense_backend.py" and "cholesky" in name.lower():
                counts["dense_backend_cholesky"] += 1
            if path.name == "dense_backend.py" and name.lower().endswith("cg"):
                counts["dense_backend_cg"] += 1
    if any(counts.values()):
        raise EngineeringBoundary(f"forbidden production call inventory: {counts}")
    return counts


def verify_official(lock: dict[str, Any]) -> dict[str, Any]:
    root = Path(lock["runtime_root"])
    if git(root, "rev-parse", "HEAD") != lock["head"] or git(root, "rev-parse", "HEAD^{tree}") != lock["tree"]:
        raise EngineeringBoundary("Official source HEAD/tree mismatch")
    if git(root, "status", "--porcelain"):
        raise EngineeringBoundary("Official source is not clean")
    members = {relative: identity(root / relative, digest) for relative, digest in lock["members"].items()}
    source = (root / "easyeditor/models/alphaedit/AlphaEdit_main.py").read_text()
    required = {
        "dense_normal_equation": "P[i,:,:].to(f\"cuda:{hparams.device}\") @ (layer_ks.to(f\"cuda:{hparams.device}\") @ layer_ks.T.to(f\"cuda:{hparams.device}\") + cache_c[i,:,:].to(f\"cuda:{hparams.device}\"))",
        "rhs": "P[i,:,:].to(f\"cuda:{hparams.device}\") @ layer_ks.to(f\"cuda:{hparams.device}\") @ resid.T.to(f\"cuda:{hparams.device}\")",
        "shape_adapter": "upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)",
        "post_key": "layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T",
        "append_once": "cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T",
    }
    missing = [key for key, text in required.items() if text not in source]
    if missing:
        raise EngineeringBoundary(f"Official equation/orientation source closure missing: {missing}")
    return {
        "head": lock["head"],
        "tree": lock["tree"],
        "tracked_clean": True,
        "members": members,
        "equation_orientation": {key: "PASS" for key in required},
        "dirty_user_checkout_influence_count": 0,
    }


def _mapped(path: str) -> Path:
    if not path.startswith("/data/janghj/"):
        raise EngineeringBoundary(f"unexpected server4 source prefix: {path}")
    return Path("/mnt/raid5/janghj") / path.removeprefix("/data/janghj/")


def verify_model_members(alias: str, spec: dict[str, Any], seal: dict[str, Any]) -> dict[str, Any]:
    row = seal["models"][alias]
    snapshot = Path(spec["snapshot"])
    if row["revision"] != spec["revision"] or row["required_root"] != spec["required_root"]:
        raise EngineeringBoundary(f"model revision/root mismatch: {alias}")
    members = []
    for member in row["required_members"]:
        logical = snapshot / member["relative_path"]
        resolved = logical.resolve(strict=True)
        metadata = resolved.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != member["size"]:
            raise EngineeringBoundary(f"model member metadata mismatch: {logical}")
        digest = resolved.name if len(resolved.name) == 64 else sha256(resolved)
        if digest != member["sha256"]:
            raise EngineeringBoundary(f"model member SHA mismatch: {logical}")
        members.append({"relative_path": member["relative_path"], "resolved": str(resolved), "bytes": metadata.st_size, "sha256": digest})
    return {
        "snapshot": str(snapshot),
        "revision": row["revision"],
        "required_root": row["required_root"],
        "closure_identity": row["closure_identity"],
        "members": members,
    }


def verify_alpha_artifacts(alias: str, spec: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    row = manifest["models"][alias]
    projector_member = next(member for member in row["members"] if member["kind"] == "projector")
    projector = identity(Path(spec["projector"]), projector_member["sha256"], projector_member["size"])
    if projector["sha256"] != spec["projector_sha256"]:
        raise EngineeringBoundary(f"projector lock mismatch: {alias}")
    value = torch.load(spec["projector"], map_location="cpu", weights_only=True, mmap=True)
    projector.update({"shape": list(value.shape), "dtype": str(value.dtype), "finite_probe": bool(torch.isfinite(value[:, :16, :16]).all())})
    if projector["shape"] != projector_member["shape"] or projector["dtype"] != "torch.float32" or not projector["finite_probe"]:
        raise EngineeringBoundary(f"projector tensor mismatch: {alias}")
    del value
    covariances = []
    for member in row["members"]:
        if member["kind"] != "covariance":
            continue
        path = _mapped("/data/janghj/EasyEdit/" + member["relative_path"])
        file_id = identity(path, member["sha256"], member["size"])
        with np.load(path, mmap_mode="r") as archive:
            matrix = archive["mom2.mom2"]
            metadata = {
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "sample_size": int(archive["sample_size"]),
                "count": int(archive["mom2.count"]),
            }
        if metadata["shape"] != member["shape"] or metadata["dtype"] != member["dtype"]:
            raise EngineeringBoundary(f"covariance tensor mismatch: {path}")
        covariances.append({**file_id, **metadata, "layer": member["layer"]})
    if [row["layer"] for row in covariances] != [4, 5, 6, 7, 8]:
        raise EngineeringBoundary(f"five-layer covariance closure mismatch: {alias}")
    return {"projector": projector, "covariances": covariances, "artifact_bundle_sha256": row["bundle_sha256"]}


def verify_cases(dataset: Path, case_manifest: Path) -> dict[str, Any]:
    rows = json.loads(dataset.read_text())
    by_id = {int(row["case_id"]): row for row in rows}
    manifest = json.loads(case_manifest.read_text())
    selected = []
    for row in manifest["cases"]:
        case_id = int(row["case_id"])
        if case_id not in by_id or canonical(by_id[case_id]) != row["row_sha256"]:
            raise EngineeringBoundary(f"case identity mismatch: {case_id}")
        selected.append({"ordinal": row["ordinal"], "case_id": case_id, "row_sha256": row["row_sha256"]})
    if [row["ordinal"] for row in selected] != list(range(10)) or len({row["case_id"] for row in selected}) != 10:
        raise EngineeringBoundary("Atomic B10 case/order closure mismatch")
    return {"case_ids": [row["case_id"] for row in selected], "order_sha256": canonical(selected), "request_denominator": 10}


def build(source_root: Path, contract: Path) -> dict[str, Any]:
    artifact_path = PACKAGE / "config/artifact-lock-v1.json"
    official_path = PACKAGE / "config/official-source-lock-v1.json"
    case_path = PACKAGE / "config/case-manifest-v1.json"
    numerical_path = PACKAGE / "config/numerical-lock-v1.json"
    artifact = json.loads(artifact_path.read_text())
    official = json.loads(official_path.read_text())
    model_manifest_id = identity(source_root / artifact["model_closure_manifest"]["path"], artifact["model_closure_manifest"]["sha256"])
    alpha_manifest_id = identity(source_root / artifact["alphaedit_artifact_manifest"]["path"], artifact["alphaedit_artifact_manifest"]["sha256"])
    model_seal = json.loads(Path(model_manifest_id["path"]).read_text())
    alpha_seal = json.loads(Path(alpha_manifest_id["path"]).read_text())
    dataset = identity(Path(artifact["dataset"]["path"]), artifact["dataset"]["sha256"], artifact["dataset"]["bytes"])
    models = {}
    for alias, spec in artifact["models"].items():
        models[alias] = {
            "model": verify_model_members(alias, spec, model_seal),
            "alphaedit": verify_alpha_artifacts(alias, spec, alpha_seal),
            "padding": {
                "official": "right",
                "candidate": "left_with_attention_mask_and_explicit_position_ids",
                "pad_equals_eos": True,
                "target_alignment": "semantic_nonpad_columns",
            },
        }
    source_status = git(source_root, "status", "--porcelain", "--untracked-files=no")
    if source_status:
        raise EngineeringBoundary("tracked source worktree is dirty")
    numerical = json.loads(numerical_path.read_text())
    if numerical != NumericalLock().payload():
        raise EngineeringBoundary("numerical lock/class mismatch")
    return {
        "schema": "odeedit.s06.fzcb-hard-alpha-densec.pre-gpu.v1",
        "status": "PRE_GPU_PASS",
        "contract": identity(contract, "65d1f80291d64a6cdf97f7e6072d7fe33cb1595f5fcd7462f42e8ad7ecd1b35a", 27954),
        "source": {
            "head": git(source_root, "rev-parse", "HEAD"),
            "tree": git(source_root, "rev-parse", "HEAD^{tree}"),
            "branch": git(source_root, "branch", "--show-current"),
            "tracked_clean": True,
        },
        "official": verify_official(official),
        "artifact_lock": identity(artifact_path),
        "official_source_lock": identity(official_path),
        "case_manifest": identity(case_path),
        "numerical_lock": identity(numerical_path),
        "dataset": dataset,
        "cases": verify_cases(Path(dataset["path"]), case_path),
        "upstream_manifests": {"model": model_manifest_id, "alphaedit": alpha_manifest_id},
        "models": models,
        "forbidden_production_calls": source_call_inventory(),
        "ha3_submission_count": 0,
        "b100x10_submission_count": 0,
        "model_action_count": 0,
        "gpu_action_count": 0,
        "slurm_action_count": 0,
        "result_action_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite preflight receipt")
    payload = build(args.source_root.resolve(), args.contract.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    os.chmod(args.output, 0o600)


if __name__ == "__main__":
    main()
