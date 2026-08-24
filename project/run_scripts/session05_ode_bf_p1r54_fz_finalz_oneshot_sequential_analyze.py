#!/usr/bin/env python3
"""Analysis-only writer-cadence count comparison entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1_runtime import _atomic_write_once
from project.run_scripts.ode_bf.p1r54_fz_finalz_oneshot_analysis import (
    build_writer_cadence_comparison,
)


def _read(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ODEBFContractError("writer-cadence analysis input differs")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ODEBFContractError("writer-cadence analysis JSON differs")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--treatment-terminal", required=True, type=Path)
    parser.add_argument("--control-terminal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_writer_cadence_comparison(
        treatment=_read(args.treatment_terminal),
        control=_read(args.control_terminal),
    )
    digest = _atomic_write_once(args.output, payload)
    print(json.dumps({"status": "ANALYSIS_VALID", "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
