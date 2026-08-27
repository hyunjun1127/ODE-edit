#!/usr/bin/env python3
"""Build the create-once BGODE-R3 G2 TECH-R1 observation-only source manifest."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from project.run_scripts.barrier_guided_ode.r3.build_g2_release import (
    LOCK_PATH,
    MEMBERS,
    NATURAL_PATH,
)


REPO = Path(__file__).resolve().parents[4]
OUTPUT = REPO / "project/run_scripts/barrier_guided_ode/r3/locks/bgode-r3-g2-tech-r1-source-manifest.json"
ADDITIONAL = (
    "project/run_scripts/barrier_guided_ode/r3/tests/test_g2_failure_receipt.py",
    "project/run_scripts/session05_bgode_r3_g2_tech_r1.sbatch",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    members = tuple(relative for relative in MEMBERS if not relative.endswith("session05_bgode_r3_g2.sbatch"))
    relatives = sorted((*members, *ADDITIONAL, str(NATURAL_PATH.relative_to(REPO)), str(LOCK_PATH.relative_to(REPO))))
    records = []
    for relative in relatives:
        path = REPO / relative
        status = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(status.st_mode):
            raise SystemExit(f"G2 TECH-R1 source member is not regular: {relative}")
        records.append(
            {
                "bytes": status.st_size,
                "mode": format(stat.S_IMODE(status.st_mode), "04o"),
                "path": relative,
                "sha256": digest(path),
            }
        )
    payload = {
        "schema": "ode-edit-bgode-r3-g2-tech-r1-source-manifest/v1",
        "repair": "OBSERVATION_ONLY_NUMERICAL_BOUNDARY_RECEIPT_PUBLICATION",
        "science_change_count": 0,
        "tolerance_change_count": 0,
        "threshold_change_count": 0,
        "member_count": len(records),
        "members": records,
        "members_root": hashlib.sha256(canonical(records)).hexdigest(),
    }
    data = canonical(payload) + b"\n"
    OUTPUT.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        if OUTPUT.is_symlink() or not OUTPUT.is_file() or OUTPUT.read_bytes() != data:
            raise SystemExit(f"existing G2 TECH-R1 manifest differs: {OUTPUT}")
        return
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


if __name__ == "__main__":
    main()
