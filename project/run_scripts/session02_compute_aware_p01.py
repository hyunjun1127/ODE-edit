"""Fail-closed P0/P1 plan renderer; this module never executes a job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from project.run_scripts.ode_edit_method.lock import LOCK_PATH, dry_plan, load_lock


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="session02-compute-aware-p01",
        description="Validate and print the paired-model P0/P1 dry plan",
    )
    parser.add_argument("--stage", choices=("p0", "p1", "both"), required=True)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--dry-run", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lock = load_lock(args.lock)
    lock.pop("proposal_id")
    stages = ("p0", "p1") if args.stage == "both" else (args.stage,)
    print(
        json.dumps(
            {"plans": [dry_plan(lock, stage) for stage in stages]},
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
