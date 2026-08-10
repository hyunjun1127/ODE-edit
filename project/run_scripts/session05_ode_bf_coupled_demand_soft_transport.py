#!/usr/bin/env python3
"""Create-once P1R16 Stage-A single-arm entry point."""

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
from project.run_scripts.ode_bf.coupled_demand_soft_transport import (
    P1R16_INSTRUCTION_ID,
)
from project.run_scripts.ode_bf.coupled_demand_soft_transport_experiment import (
    P1R16OutputRootCollision,
    run_p1r16_stage_a_experiment,
)
from project.run_scripts.ode_bf.p1_coupled_demand_soft_transport_panel import (
    P1R16_STAGE_A_RESULT_TOKEN,
    validate_p1r16_stage_a_output_root,
)


def _valid_head(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _write_failure_once(root: Path, value: Mapping[str, Any]) -> str | None:
    if root.is_symlink() or not root.is_dir():
        return None
    path = root / "failure.json"
    encoded = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return None
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _last_stage(root: Path) -> str | None:
    stage_root = root / "raw" / "stages"
    if stage_root.is_symlink() or not stage_root.is_dir():
        return None
    names = sorted(
        path.name
        for path in stage_root.iterdir()
        if path.is_file() and not path.is_symlink()
    )
    return names[-1] if names else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", required=True, choices=MODEL_ALIASES)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--run-token", required=True, choices=(P1R16_STAGE_A_RESULT_TOKEN,))
    parser.add_argument("--namespace-probe", action="store_true")
    args = parser.parse_args(argv)
    if not _valid_head(args.source_head):
        return 2
    try:
        namespace = validate_p1r16_stage_a_output_root(
            repo_root=REPO_ROOT, alias=args.model, output_root=args.output_root
        )
        if args.namespace_probe:
            print(
                json.dumps(
                    {
                        "status": "P1R16_STAGE_A_NAMESPACE_PASS_NO_MODEL",
                        "instruction_id": P1R16_INSTRUCTION_ID,
                        "model_alias": args.model,
                        "namespace_identity_sha256": namespace["identity_sha256"],
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
        result = run_p1r16_stage_a_experiment(
            repo_root=REPO_ROOT,
            alias=args.model,
            output_root=args.output_root,
            source_head=args.source_head,
        )
    except P1R16OutputRootCollision:
        return 1
    except Exception as exc:
        failure = {
            "schema": "ode-edit-s05-p1r16-stage-a-failure/v1",
            "status": "FAIL_CLOSED",
            "instruction_id": P1R16_INSTRUCTION_ID,
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
                    "exception_class": failure["exception_class"],
                    "exception_message_sha256": failure["exception_message_sha256"],
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
