"""Bounded artifact/source verification and G0 receipt builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import MODEL_SPECS, NumericalLock, TechnicalBoundary
from .forbidden_imports import scan


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path: Path, *, expected_sha: str | None = None, expected_bytes: int | None = None) -> dict[str, Any]:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or path.is_symlink():
        raise TechnicalBoundary(f"not a regular non-symlink artifact: {path}")
    digest = sha256(path)
    if expected_sha is not None and digest != expected_sha:
        raise TechnicalBoundary(f"SHA mismatch: {path}")
    if expected_bytes is not None and st.st_size != expected_bytes:
        raise TechnicalBoundary(f"byte mismatch: {path}")
    return {"path": str(path), "sha256": digest, "bytes": st.st_size, "mode": format(stat.S_IMODE(st.st_mode), "04o")}


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _verify_model_closure(source_root: Path, alias: str) -> dict[str, Any]:
    seal_path = source_root / "agents/server4/p4-hf-consumed-closure-seal.json"
    seal = json.loads(seal_path.read_text())
    row = seal["models"][alias]
    spec = MODEL_SPECS[alias]
    if row["revision"] != spec.model_revision or Path(row["snapshot_path"]) != spec.model_path:
        raise TechnicalBoundary(f"model revision/path mismatch: {alias}")
    members = []
    for member in row["required_members"]:
        path = spec.model_path / member["relative_path"]
        # HF snapshots are pinned symlinks; validate the resolved regular blob.
        resolved = path.resolve(strict=True)
        st = resolved.stat()
        if not stat.S_ISREG(st.st_mode) or st.st_size != member["size"]:
            raise TechnicalBoundary(f"model closure member mismatch: {path}")
        # Shard blob names are their sha256; small members are streamed.
        digest = resolved.name if len(resolved.name) == 64 else sha256(resolved)
        if digest != member["sha256"]:
            raise TechnicalBoundary(f"model closure SHA mismatch: {path}")
        members.append({"relative_path": member["relative_path"], "resolved": str(resolved), "bytes": st.st_size, "sha256": digest})
    return {"required_root": row["required_root"], "closure_identity": row["closure_identity"], "members": members, "extra_load_influence_count": 0}


def _verify_stats(spec: Any, expected: dict[str, Any]) -> dict[str, Any]:
    path = spec.statistics_root / spec.statistics_model_dir / "wikipedia_stats" / "model.layers.8.mlp.down_proj_float32_mom2_100000.npz"
    file_id = identity(path, expected_sha=expected["statistics_layer8_sha256"], expected_bytes=expected["statistics_layer8_bytes"])
    with np.load(path, mmap_mode="r") as archive:
        matrix = archive["mom2.mom2"]
        metadata = {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "sample_size": int(archive["sample_size"]),
            "count": int(archive["mom2.count"]),
            "constructor": str(archive["mom2.constructor"]),
        }
    if metadata["shape"] != expected["statistics_shape"] or metadata["dtype"] != expected["statistics_dtype"] or metadata["sample_size"] != expected["statistics_sample_size"]:
        raise TechnicalBoundary(f"statistics metadata mismatch: {spec.alias}")
    return {**file_id, **metadata, "statistic_type": "wikipedia_second_moment", "covariance_regularization": "official_hparams_only"}


def _verify_projector(spec: Any, expected: dict[str, Any]) -> dict[str, Any]:
    file_id = identity(spec.projector, expected_sha=expected["projector_sha256"], expected_bytes=expected["projector_bytes"])
    value = torch.load(spec.projector, map_location="cpu", weights_only=True, mmap=True)
    metadata = {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."), "finite": bool(torch.isfinite(value[0, :64, :64]).all())}
    if metadata["shape"] != expected["projector_shape"] or metadata["dtype"] != "float32" or not metadata["finite"]:
        raise TechnicalBoundary(f"projector metadata mismatch: {spec.alias}")
    del value
    return {**file_id, **metadata, "convention": "official_alphaedit_left_input_projection"}


def build(source_root: Path) -> dict[str, Any]:
    lock_path = source_root / "project/run_scripts/fixed_z_nonuniqueness/config/artifact-lock-v1.json"
    lock = json.loads(lock_path.read_text())
    easyedit = Path(lock["easyedit"]["root"])
    if _git(easyedit, "rev-parse", "HEAD") != lock["easyedit"]["head"] or _git(easyedit, "rev-parse", "HEAD^{tree}") != lock["easyedit"]["tree"] or _git(easyedit, "status", "--porcelain", "--untracked-files=no"):
        raise TechnicalBoundary("stock EasyEdit identity/clean gate failed")
    official = {}
    for relative, digest in lock["easyedit"]["members"].items():
        official[relative] = identity(easyedit / relative, expected_sha=digest)
    dataset = identity(Path(lock["dataset"]["path"]), expected_sha=lock["dataset"]["sha256"], expected_bytes=lock["dataset"]["bytes"])
    package = source_root / "project/run_scripts/fixed_z_nonuniqueness"
    violations = scan(package)
    if violations:
        raise TechnicalBoundary(f"forbidden imports: {violations}")
    models = {}
    for alias, spec in MODEL_SPECS.items():
        expected = lock["models"][alias]
        models[alias] = {
            "model": _verify_model_closure(source_root, alias),
            "projector": _verify_projector(spec, expected),
            "statistics": _verify_stats(spec, expected),
            "config_sha256": sha256(spec.model_path / "config.json"),
            "tokenizer_sha256": sha256(spec.model_path / "tokenizer.json"),
            "model_revision": spec.model_revision,
            "padding_contract": {"official": spec.official_padding_side, "hook": spec.hook_padding_side, "pad_policy": spec.pad_policy},
            "existing_full_fp32_endpoint_lock": {
                **identity(
                    Path(lock["existing_full_fp32_endpoint_locks"][alias]["path"]),
                    expected_sha=lock["existing_full_fp32_endpoint_locks"][alias]["sha256"],
                ),
                "endpoint_relative_tolerance": lock["existing_full_fp32_endpoint_locks"][alias]["endpoint_relative_tolerance"],
            },
        }
    return {
        "schema": "odeedit.s06.fixed-z-nonuniqueness.pre-gpu.v1",
        "status": "PRE_GPU_PASS",
        "source": {"head": _git(source_root, "rev-parse", "HEAD"), "tree": _git(source_root, "rev-parse", "HEAD^{tree}"), "tracked_clean": not bool(_git(source_root, "status", "--porcelain", "--untracked-files=no"))},
        "easyedit": {"head": lock["easyedit"]["head"], "tree": lock["easyedit"]["tree"], "members": official},
        "dataset": dataset,
        "models": models,
        "numerical_lock": NumericalLock().payload(),
        "forbidden_import_count": 0,
        "dense_inverse_count": 0,
        "final_audit_open_count": 0,
        "full_fp32": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite preflight receipt")
    payload = build(args.source_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.chmod(args.output, 0o600)


if __name__ == "__main__":
    main()
