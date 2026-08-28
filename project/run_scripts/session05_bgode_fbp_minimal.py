"""CLI for the minimal fixed-basis barrier ODE experiment."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from project.run_scripts.barrier_guided_ode.fixed_basis_ode.experiment import run_experiment
from project.run_scripts.barrier_guided_ode.s1_experiment import _write_json_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--easyedit-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        result = run_experiment(
            easyedit_root=args.easyedit_root,
            output_root=args.output_root,
            source_head=args.source_head,
            source_tree=args.source_tree,
            model_alias=args.model_alias,
            run_id=args.run_id,
        )
    except BaseException as exc:
        output = Path(args.output_root).expanduser().resolve(strict=False) / args.run_id
        if output.is_dir() and not (output / "failure-boundary.json").exists():
            _write_json_once(
                output / "failure-boundary.json",
                {
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                    "science_change_count": 0,
                    "threshold_change_count": 0,
                    "tolerance_change_count": 0,
                },
            )
        raise
    print(json.dumps({"run_id": result["run_id"], "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
