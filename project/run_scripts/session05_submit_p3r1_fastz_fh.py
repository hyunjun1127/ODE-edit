#!/usr/bin/env python3
"""Held-inspect-release submitter for P3R1 under the server2 cap of two."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


PROJECT_GPU_CAP = 2
ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")


def active_allocations() -> list[dict[str, str]]:
    output = subprocess.check_output(
        ["squeue", "-h", "-w", "server2", "-u", "janghj", "-o", "%A|%T|%b|%j"],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        job, state, gres, name = line.split("|", 3)
        if state in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"} and "gpu" in gres.lower():
            rows.append({"job": job, "state": state, "gres": gres, "name": name})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--result-parent", required=True, type=Path)
    parser.add_argument("--case-index", required=True, type=int, choices=range(1, 11))
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    rows = active_allocations()
    requested = len(ALIASES)
    payload = {
        "schema": "ode-edit-s05-p3r1-held-inspect-release/v1",
        "project_gpu_cap": PROJECT_GPU_CAP,
        "observed_active_pending_gpu_allocations": rows,
        "observed_allocation_count": len(rows),
        "requested_gpu_count": requested,
        "release_allowed": len(rows) + requested <= PROJECT_GPU_CAP,
        "case_index": args.case_index,
        "aliases": list(ALIASES),
        "released": False,
        "job_ids": [],
    }
    if args.release:
        if not payload["release_allowed"]:
            raise SystemExit(json.dumps(payload, sort_keys=True))
        for alias in ALIASES:
            output = subprocess.check_output(
                [
                    "sbatch",
                    "--parsable",
                    f"--job-name=p3r1-{alias}-c{args.case_index:02d}",
                    "project/run_scripts/session05_ode_bf_p3r1_fastz_fh.sbatch",
                    args.source_head,
                    str(args.result_parent),
                    alias,
                    str(args.case_index),
                ],
                text=True,
            ).strip()
            payload["job_ids"].append(output)
        payload["released"] = True
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
