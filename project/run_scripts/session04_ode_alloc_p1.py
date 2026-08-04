#!/usr/bin/env python3
"""Fail-closed entry point for one approved Session 04 P1 model alias."""

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
from project.run_scripts.ode_alloc.p1_contracts import P1_RUN_TOKEN
from project.run_scripts.ode_alloc.p1_runtime import (
    P1_EXECUTION_TOKEN,
    run_p1,
    write_p1_failure_once,
)


PACKAGE_ROOT = REPO_ROOT / "project" / "run_scripts" / "ode_alloc"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="session04-ode-alloc-p1", allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(P1_RUN_TOKEN,))
    parser.add_argument(
        "--execution-token", required=True, choices=(P1_EXECUTION_TOKEN,)
    )
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
        raise RuntimeError("P1 requires exactly one Slurm-visible GPU")
    try:
        result = run_p1(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
            numerical_lock_path=args.numerical_lock,
            artifact_lock_path=args.artifact_lock,
            seal_path=args.seal,
            run_token=args.run_token,
            execution_token=args.execution_token,
        )
    except BaseException as exc:
        failure_sha = write_p1_failure_once(args.output_root, exc)
        print(
            json.dumps(
                {
                    "status": "FAIL_CLOSED_NO_RETRY",
                    "model_alias": args.model,
                    "error_type": type(exc).__name__,
                    "error_sha256": hashlib.sha256(
                        str(exc).encode("utf-8")
                    ).hexdigest(),
                    "failure_receipt_sha256": failure_sha,
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
