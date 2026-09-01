"""Bounded source/artifact/stream preflight for the observational study."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any

from project.run_scripts.fixed_z_nonuniqueness.contracts import MODEL_SPECS
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p4_sealed_stream import preflight_transferred_stream_v2
from project.run_scripts.ode_bf.p1r52_official_sequential_baselines import run_official_memit_apply
from project.run_scripts.ode_bf.scalable_batched_native import run_official_native_apply

from .contracts import (
    DATASET,
    EASYEDIT_HEAD,
    EASYEDIT_ROOT,
    EASYEDIT_TREE,
    EVALUATOR_IDENTITY,
    INSTRUCTION_ID,
    LAYERS,
    ORDER_IDENTITY,
    ObservationBoundary,
    ObservationLock,
    STREAM_ARCHIVE,
    STREAM_IDENTITY,
    STREAM_ROOT,
)


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path) -> dict[str, Any]:
    observed = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise ObservationBoundary(f"not a regular non-symlink: {path}")
    return {"path": str(path), "bytes": observed.st_size, "sha256": _sha256(path)}


def build(source_root: Path, expected_head: str, focused_gate: Path) -> dict[str, Any]:
    head = _git(source_root, "rev-parse", "HEAD")
    tree = _git(source_root, "rev-parse", "HEAD^{tree}")
    if head != expected_head or _git(source_root, "status", "--porcelain", "--untracked-files=no"):
        raise ObservationBoundary("source HEAD/clean gate failed")
    if _git(EASYEDIT_ROOT, "rev-parse", "HEAD") != EASYEDIT_HEAD or _git(EASYEDIT_ROOT, "rev-parse", "HEAD^{tree}") != EASYEDIT_TREE or _git(EASYEDIT_ROOT, "status", "--porcelain", "--untracked-files=no"):
        raise ObservationBoundary("stock EasyEdit identity/clean gate failed")
    lock = json.loads((source_root / "project/run_scripts/fixed_z_nonuniqueness/config/artifact-lock-v1.json").read_text())
    official_members = {
        relative: _regular(EASYEDIT_ROOT / relative)
        for relative in lock["easyedit"]["members"]
    }
    if any(official_members[name]["sha256"] != digest for name, digest in lock["easyedit"]["members"].items()):
        raise ObservationBoundary("stock EasyEdit member identity differs")
    stream = preflight_transferred_stream_v2(STREAM_ROOT, archive=STREAM_ARCHIVE, dataset_path=DATASET)
    if stream["stream_root"] != STREAM_IDENTITY or stream["order_sha256"] != ORDER_IDENTITY or stream["evaluator_identity"] != EVALUATOR_IDENTITY:
        raise ObservationBoundary("stream/evaluator binding differs")
    gate = json.loads(focused_gate.read_text())
    if gate.get("status") != "PASS" or gate.get("failed") != 0:
        raise ObservationBoundary("focused gate receipt differs")
    signatures = {
        "memit": str(inspect.signature(run_official_memit_apply)),
        "alphaedit": str(inspect.signature(run_official_native_apply)),
    }
    if any("observer" not in value or "call_audit" not in value for value in signatures.values()):
        raise ObservationBoundary("optional wrapper interface is absent")
    models = {}
    for alias, spec in MODEL_SPECS.items():
        hparams = [source_root / spec.memit_hparams, source_root / spec.alpha_hparams]
        if not spec.model_path.is_dir() or any(not path.is_file() for path in hparams):
            raise ObservationBoundary(f"model/hparams deployment missing: {alias}")
        manifest = json.loads((source_root / "agents/server4/p4-hf-consumed-closure-seal.json").read_text())
        row = manifest["models"][alias]
        if row["revision"] != spec.model_revision or Path(row["snapshot_path"]) != spec.model_path:
            raise ObservationBoundary(f"HF consumed closure differs: {alias}")
        models[alias] = {
            "revision": spec.model_revision,
            "snapshot": str(spec.model_path),
            "required_root": row["required_root"],
            "closure_identity": row["closure_identity"],
            "projector": {"path": str(spec.projector), "bytes": spec.projector.stat().st_size},
            "statistics_root": str(spec.statistics_root),
            "hparams": [_regular(path) for path in hparams],
        }
    payload = {
        "schema": "odeedit.s06.official-layer-realization-debt.pre-gpu.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "PRE_GPU_PASS",
        "source": {"head": head, "tree": tree, "tracked_clean": True},
        "easyedit": {"head": EASYEDIT_HEAD, "tree": EASYEDIT_TREE, "tracked_clean": True, "members": official_members},
        "stream": stream,
        "models": models,
        "wrapper_signatures": signatures,
        "layers": list(LAYERS),
        "observation_lock": ObservationLock().payload(),
        "focused_gate": {"path": str(focused_gate), "sha256": _sha256(focused_gate), "passed": gate["passed"]},
        "easyedit_source_edit_count": 0,
        "baseline_equation_change_count": 0,
        "gpu_slurm_action_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--focused-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("refusing to overwrite preflight")
    payload = build(args.source_root.resolve(), args.expected_head, args.focused_gate.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x") as handle:
        handle.write(raw)
    os.chmod(args.output, 0o600)


if __name__ == "__main__":
    main()
