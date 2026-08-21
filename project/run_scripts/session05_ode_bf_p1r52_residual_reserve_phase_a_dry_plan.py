#!/usr/bin/env python3
"""Outcome-free four-cell Phase-A plan."""

from __future__ import annotations

import argparse
import json

from project.run_scripts.ode_bf.p1r52_residual_reserve_phase_a_execution import (
    PHASE_A_ARMS,
    expected_phase_a_result_name,
)


def build_plan(source_head: str, alias: str) -> dict[str, object]:
    return {
        "source_head": source_head,
        "alias": alias,
        "array": "0-3%4",
        "cells": [
            {"index": index, "arm": arm, "result_name": expected_phase_a_result_name(alias, arm)}
            for index, arm in enumerate(PHASE_A_ARMS)
        ],
        "gpu_per_cell": 1,
        "max_concurrent_gpu": 4,
        "batch_size": 10,
        "target_step_count": 8,
        "history_mode": "OFF",
        "heldout_inner_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--alias", required=True)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.source_head, args.alias), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
