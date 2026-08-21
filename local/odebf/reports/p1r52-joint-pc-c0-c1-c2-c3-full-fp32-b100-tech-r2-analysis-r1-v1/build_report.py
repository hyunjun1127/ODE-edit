#!/usr/bin/env python3
"""Build the canonical R1 package while preserving the pre-seal V1 package."""

from __future__ import annotations

import csv
import importlib.util
import io
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "p1r52-joint-pc-c0-c1-c2-c3-full-fp32-b100-tech-r2-v1/build_report.py"
SPEC = importlib.util.spec_from_file_location("fp32_base_report", BASE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("base report builder unavailable")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def dump_csv_lf(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV rows: {path}")
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    MOD.create_once(path, buf.getvalue().encode())


MOD.OUT = HERE
MOD.dump_csv = dump_csv_lf
MOD.main()
