#!/usr/bin/env python3
"""Fail-closed entry point for one approved Session 04 P0 model alias."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_alloc.contracts import MODEL_ALIASES
from project.run_scripts.ode_alloc.p0_runtime import (
    R3_RUN_TOKEN,
    run_p0,
    write_diagnostic_failure_once,
    write_failure_once,
)


PACKAGE_ROOT = REPO_ROOT / "project" / "run_scripts" / "ode_alloc"
EASYEDIT_ROOT = Path("/mnt/raid5/janghj/EasyEdit")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="session04-ode-alloc-p0", allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(R3_RUN_TOKEN,))
    parser.add_argument(
        "--numerical-lock",
        type=Path,
        default=PACKAGE_ROOT / "numerical_lock_proposal.json",
    )
    parser.add_argument(
        "--artifact-lock",
        type=Path,
        default=PACKAGE_ROOT / "p0_artifact_lock_r1.json",
    )
    parser.add_argument(
        "--seal",
        type=Path,
        default=PACKAGE_ROOT / "split_anchor_seal_candidate.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible.casefold() in {"none", "nodevfiles"}:
        raise RuntimeError("P0 requires exactly one Slurm-visible GPU")
    try:
        result = run_p0(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            numerical_lock_path=args.numerical_lock,
            artifact_lock_path=args.artifact_lock,
            seal_path=args.seal,
            run_token=args.run_token,
        )
    except BaseException as exc:
        diagnostic_sha = None
        diagnostic_error = None
        try:
            diagnostic_sha = write_diagnostic_failure_once(
                args.output_root,
                exc,
                repo_root=REPO_ROOT,
                easyedit_root=EASYEDIT_ROOT,
            )
        except Exception as diagnostic_exc:
            diagnostic_error = {
                "error_type": type(diagnostic_exc).__name__,
                "error_sha256": hashlib.sha256(
                    str(diagnostic_exc).encode("utf-8")
                ).hexdigest(),
            }
        failure_sha = write_failure_once(args.output_root, exc)
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_NO_RETRY",
                    "model_alias": args.model,
                    "error_type": type(exc).__name__,
                    "error_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                    "failure_receipt_sha256": failure_sha,
                    "diagnostic_failure_receipt_sha256": diagnostic_sha,
                    "diagnostic_capture_error": diagnostic_error,
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
