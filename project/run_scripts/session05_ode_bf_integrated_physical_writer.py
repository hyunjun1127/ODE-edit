#!/usr/bin/env python3
"""Create-once entry point for the single-arm P1R14 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import MODEL_ALIASES
from project.run_scripts.ode_bf.integrated_physical_writer import (
    INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
)
from project.run_scripts.ode_bf.integrated_physical_writer_experiment import (
    IntegratedOutputRootCollision,
    run_integrated_physical_writer_experiment,
)
from project.run_scripts.ode_bf.p1_integrated_physical_writer_panel import (
    P1R14_RESULT_TOKEN,
    P1R14_RUN_ATTEMPT_ID,
    validate_p1r14_attempt_output_namespace,
)


def _valid_head(value: str) -> bool:
    return len(value) == 40 and all(item in "0123456789abcdef" for item in value)


def _write_failure_once(root: Path, value: Mapping[str, Any]) -> str | None:
    if root.is_symlink() or not root.is_dir():
        return None
    destination = root / "failure.json"
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        return None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def _last_stage(root: Path) -> str | None:
    stage_root = root / "raw" / "stages"
    if stage_root.is_symlink() or not stage_root.is_dir():
        return None
    names = sorted(
        item.name
        for item in stage_root.iterdir()
        if item.is_file() and not item.is_symlink()
    )
    return names[-1] if names else None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(P1R14_RESULT_TOKEN,))
    parser.add_argument("--run-attempt", required=True, choices=(P1R14_RUN_ATTEMPT_ID,))
    parser.add_argument("--namespace-probe", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not _valid_head(args.source_head):
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_INVALID_SOURCE_HEAD",
                    "instruction_id": INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
                    "model_alias": args.model,
                    "failure_receipt_written": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    try:
        namespace = validate_p1r14_attempt_output_namespace(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            run_attempt_id=args.run_attempt,
        )
        if args.namespace_probe:
            print(
                json.dumps(
                    {
                        "status": "P1R14_RUN_ATTEMPT_NAMESPACE_PASS_NO_MODEL",
                        "instruction_id": INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
                        "model_alias": args.model,
                        "namespace_identity_sha256": namespace["identity_sha256"],
                        "result_name": namespace["result_name"],
                        "model_load": False,
                        "gpu_use": False,
                        "result_root_creation": False,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                flush=True,
            )
            return 0
        result = run_integrated_physical_writer_experiment(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            run_attempt_id=args.run_attempt,
        )
    except IntegratedOutputRootCollision as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_OUTPUT_ROOT_COLLISION",
                    "instruction_id": INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
                    "model_alias": args.model,
                    "exception_class": type(exc).__name__,
                    "exception_message_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                    "failure_receipt_written": False,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    except Exception as exc:
        failure = {
            "schema": "ode-edit-s05-integrated-physical-writer-p1r14-failure/v1",
            "status": "FAIL_CLOSED",
            "instruction_id": INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
            "model_alias": args.model,
            "source_head": args.source_head,
            "exception_class": type(exc).__name__,
            "exception_message_sha256": hashlib.sha256(
                str(exc).encode("utf-8")
            ).hexdigest(),
            "last_completed_stage": _last_stage(args.output_root),
            "scientific_promotion_authorized": False,
        }
        failure_sha = _write_failure_once(args.output_root, failure)
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED",
                    "instruction_id": INTEGRATED_PHYSICAL_WRITER_INSTRUCTION_ID,
                    "model_alias": args.model,
                    "exception_class": failure["exception_class"],
                    "exception_message_sha256": failure[
                        "exception_message_sha256"
                    ],
                    "last_completed_stage": failure["last_completed_stage"],
                    "failure_sha256": failure_sha,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
