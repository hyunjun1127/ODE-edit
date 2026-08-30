"""Terminal Phase-A gate with process-restart nuisance in matched CVaR units."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .hashing import file_sha256, write_json_once


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentinel-result", type=Path, action="append", required=True)
    parser.add_argument("--screen-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sentinel = {}
    for path in args.sentinel_result:
        row = json.loads(path.read_text())
        sentinel[(row["model"], row["method"])] = (path, row)
    cells = []
    for path in args.screen_result:
        row = json.loads(path.read_text())
        key = (row["model"], row["method"])
        sentinel_path, prior = sentinel[key]
        first_screen = row["cases"][0]
        first_sentinel = prior["cases"][0]
        screen_cvar = {candidate["candidate_id"]: candidate["observation"]["functional"]["cvar_0_875"] for candidate in first_screen["candidates"]}
        sentinel_cvar = {candidate["candidate_id"]: candidate["observation"]["functional"]["cvar_0_875"] for candidate in first_sentinel["candidates"]}
        process_restart = max(abs(screen_cvar[candidate] - sentinel_cvar[candidate]) for candidate in screen_cvar)
        base_nuisance = max(case["epsilon_nuisance"] for case in row["cases"])
        epsilon = max(base_nuisance, process_restart)
        spreads = [case["functional_spread"] for case in row["cases"]]
        valid = sum(case["valid_candidate_count"] == case["candidate_denominator"] for case in row["cases"])
        spread_cases = sum(value > 3.0 * epsilon for value in spreads)
        gate = {
            "valid_cases": valid, "valid_denominator": 8, "valid_gate": valid >= 7,
            "spread_gt_3x_nuisance_cases": spread_cases, "spread_denominator": 8, "spread_gate": spread_cases >= 6,
            "median_spread": statistics.median(spreads), "epsilon_nuisance": epsilon,
            "median_gate": statistics.median(spreads) > 5.0 * epsilon,
        }
        gate["phase_a_pass"] = all(gate[name] for name in ("valid_gate", "spread_gate", "median_gate"))
        cells.append({
            "model": row["model"], "method": row["method"], "gate": gate,
            "process_restart_cvar_max_abs": process_restart,
            "screen_result": {"path": str(path), "sha256": file_sha256(path)},
            "sentinel_result": {"path": str(sentinel_path), "sha256": file_sha256(sentinel_path)},
        })
    if len(cells) != 4:
        raise SystemExit("Phase A terminal requires four cells")
    payload = {
        "schema": "odeedit.s06.barrier-usefulness.phase-a-terminal.v1",
        "status": "PHASE_A_PASS" if all(cell["gate"]["phase_a_pass"] for cell in cells) else "PHASE_A_HOLD",
        "cells": sorted(cells, key=lambda row: (row["model"], row["method"])),
        "q_gate_edited_access_count": 0,
        "technical_exclusions": [
            {"job": "29294_[1-3]", "reason": "PENDING_ONLY_RESOURCE_MEMORY_REPAIR", "science_endpoint_count": 0, "replacement_job": "29301_1,29307_[2-3]"},
            {"job": "29301_[2-3]", "reason": "PRE_MODEL_TRACKED_SOURCE_DIRTY_TECHNICAL", "science_endpoint_count": 0, "replacement_job": "29307_[2-3]"},
        ],
        "scientific_promotion": False,
    }
    write_json_once(args.output, payload)


if __name__ == "__main__":
    main()
