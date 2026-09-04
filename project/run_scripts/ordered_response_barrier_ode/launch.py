"""Thin fail-closed launcher for one ORBODE array cell and wave."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from .preflight import (
    INSTRUCTION_ID,
    ORDER_ROOT,
    STREAM_ROOT,
    PreflightBoundary,
    _assert_no_symlink_components,
    canonical_hash,
    canonical_json,
    cell_spec,
    run_preflight,
    wave_rounds,
)


def _assert_private_regular(path: Path) -> None:
    _assert_no_symlink_components(path)
    try:
        observed = path.lstat()
    except OSError as exc:
        raise PreflightBoundary(f"missing private receipt: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode):
        raise PreflightBoundary(f"private receipt must be a regular non-symlink: {path}")
    if stat.S_IMODE(observed.st_mode) != 0o600:
        raise PreflightBoundary(f"private receipt mode differs: {path}")


def write_create_once_json(path: Path, value: Mapping[str, object]) -> None:
    target = path.absolute()
    parent = target.parent
    _assert_no_symlink_components(parent)
    if target.exists() or target.is_symlink():
        raise PreflightBoundary(f"create-once receipt exists: {target}")
    if parent.is_symlink() or not parent.is_dir():
        raise PreflightBoundary(f"receipt parent differs: {parent}")
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A failed exclusive write is evidence.  Do not unlink or overwrite it.
        raise
    _assert_private_regular(target)


def load_shared_preflight(
    path: Path,
    *,
    source_head: str,
    source_tree: str,
    wave: str,
) -> dict[str, Any]:
    _assert_private_regular(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightBoundary("cannot load shared pre-GPU receipt") from exc
    if not isinstance(value, dict):
        raise PreflightBoundary("shared pre-GPU receipt is not an object")
    body = dict(value)
    identity = body.pop("receipt_identity_sha256", None)
    if identity != canonical_hash(body):
        raise PreflightBoundary("shared pre-GPU receipt identity differs")
    wave_value = value.get("wave")
    stream = value.get("stream")
    source = value.get("source")
    model_artifacts = value.get("model_artifacts")
    b1_gate = value.get("b1_common_gate")
    round0_gate = value.get("round0_common_gate")
    if (
        value.get("instruction_id") != INSTRUCTION_ID
        or value.get("status") != "PRE_GPU_BINDING_PASS"
        or not isinstance(wave_value, dict)
        or wave_value.get("name") != wave
        or wave_value.get("round_indices") != list(wave_rounds(wave))
        or not isinstance(stream, dict)
        or stream.get("stream_root") != STREAM_ROOT
        or stream.get("order_root") != ORDER_ROOT
        or not isinstance(source, dict)
        or source.get("head") != source_head
        or source.get("tree") != source_tree
        or value.get("full_fp32_required") is not True
        or not isinstance(model_artifacts, dict)
        or model_artifacts.get("deep_hash") is not True
    ):
        raise PreflightBoundary("shared pre-GPU receipt binding differs")
    if wave == "b1" and (b1_gate is not None or round0_gate is not None):
        raise PreflightBoundary("B1 shared pre-GPU receipt gate ordering differs")
    if wave == "round0" and (
        not isinstance(b1_gate, dict)
        or b1_gate.get("status") != "B1_COMMON_INTEGRITY_PASS"
        or b1_gate.get("endpoint_count") != 20
        or round0_gate is not None
    ):
        raise PreflightBoundary("round0 shared pre-GPU receipt lacks B1 common gate")
    if wave == "remaining" and (
        b1_gate is not None
        or not isinstance(round0_gate, dict)
        or round0_gate.get("status") != "ROUND0_COMMON_INTEGRITY_PASS"
        or round0_gate.get("endpoint_count") != 2000
    ):
        raise PreflightBoundary("remaining shared pre-GPU receipt lacks round0 common gate")
    return value


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--authoritative-root", type=Path, required=True)
    parser.add_argument("--easyedit-source-root", type=Path, required=True)
    parser.add_argument("--easyedit-artifact-root", type=Path, required=True)
    parser.add_argument("--hf-hub-cache", type=Path, required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--wave", choices=("b1", "round0", "remaining"), required=True)
    parser.add_argument("--b1-root", type=Path)
    parser.add_argument("--round0-root", type=Path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(allow_abbrev=False)
    commands = value.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal-preflight", allow_abbrev=False)
    _common_arguments(seal)
    seal.add_argument("--receipt", type=Path, required=True)
    run = commands.add_parser("run-cell", allow_abbrev=False)
    _common_arguments(run)
    run.add_argument("--preflight-receipt", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--cell-id", type=int, required=True)
    run.add_argument("--run-token", required=True)
    return value


def _seal_preflight(args: argparse.Namespace) -> int:
    receipt = run_preflight(
        repo_root=args.repo_root,
        authoritative_root=args.authoritative_root,
        easyedit_source_root=args.easyedit_source_root,
        easyedit_artifact_root=args.easyedit_artifact_root,
        hf_hub_cache=args.hf_hub_cache,
        source_head=args.source_head,
        source_tree=args.source_tree,
        cell_id=0,
        wave=args.wave,
        deep_artifact_hash=True,
        b1_root=args.b1_root,
        round0_root=args.round0_root,
    )
    write_create_once_json(args.receipt, receipt)
    print(canonical_json({"receipt": str(args.receipt.absolute()), "status": receipt["status"]}))
    return 0


def _run_cell(args: argparse.Namespace) -> int:
    spec = cell_spec(args.cell_id)
    expected_token = f"s06-orbode-server1-{args.wave}-cell{spec.cell_id}-{spec.model_alias}-{spec.writer_family.lower()}-r1"
    if args.run_token != expected_token:
        raise PreflightBoundary("run token or array mapping differs")
    load_shared_preflight(
        args.preflight_receipt,
        source_head=args.source_head,
        source_tree=args.source_tree,
        wave=args.wave,
    )
    # Repeat cheap mutable-path/source checks immediately before model import.
    binding = run_preflight(
        repo_root=args.repo_root,
        authoritative_root=args.authoritative_root,
        easyedit_source_root=args.easyedit_source_root,
        easyedit_artifact_root=args.easyedit_artifact_root,
        hf_hub_cache=args.hf_hub_cache,
        source_head=args.source_head,
        source_tree=args.source_tree,
        cell_id=args.cell_id,
        wave=args.wave,
        deep_artifact_hash=False,
        b1_root=args.b1_root,
        round0_root=args.round0_root,
    )
    if args.output_root.exists() or args.output_root.is_symlink():
        raise PreflightBoundary("task create-once output root already exists")
    _assert_no_symlink_components(args.output_root.absolute().parent)

    # This is intentionally the first import of the model-facing runtime.
    from .runtime import run_cell

    result = run_cell(
        repo_root=args.repo_root,
        easyedit_source_root=args.easyedit_source_root,
        easyedit_artifact_root=args.easyedit_artifact_root,
        hf_hub_cache=args.hf_hub_cache,
        output_root=args.output_root,
        source_head=args.source_head,
        source_tree=args.source_tree,
        cell_id=args.cell_id,
        wave=args.wave,
        round_indices=wave_rounds(args.wave),
        run_token=args.run_token,
    )
    if not isinstance(result, dict):
        raise PreflightBoundary("runtime result is not a receipt object")
    print(
        canonical_json(
            {
                "binding_receipt_identity": binding["receipt_identity_sha256"],
                "cell_id": args.cell_id,
                "runtime_status": result.get("status", "NOT_RECORDED"),
                "wave": args.wave,
            }
        )
    )
    return 0


def main() -> int:
    args = parser().parse_args()
    return _seal_preflight(args) if args.command == "seal-preflight" else _run_cell(args)


if __name__ == "__main__":
    raise SystemExit(main())
