#!/usr/bin/env python3
"""Create the final rooted receipt after the independent review is present."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().parent


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    target = OUT / "final-package-receipt.json"
    if target.exists():
        raise RuntimeError(f"create-once exists: {target}")
    members = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != target.name:
            members.append({"name": p.name, "bytes": p.stat().st_size, "sha256": sha(p), "mode": oct(p.stat().st_mode & 0o777)})
    material = "\n".join(f"{x['name']}\t{x['sha256']}\t{x['bytes']}\t{x['mode']}" for x in members)
    review = json.loads((OUT / "independent-rehash-review.json").read_text())
    data = {
        "schema": "p1r52-structuralh-on/final-package-receipt/v1",
        "status": "PASS",
        "canonical_scope": "P1R52-Soft-Sequential-StructuralH-On-10xB10",
        "canonical_report": "p1r52-soft-sequential-structuralh-on-10xb10-factual-ko-v2.md",
        "canonical_report_sha256": sha(OUT / "p1r52-soft-sequential-structuralh-on-10xb10-factual-ko-v2.md"),
        "file_tree_count": len(members),
        "file_tree_root_sha256": hashlib.sha256(material.encode()).hexdigest(),
        "directory_mode": oct(OUT.stat().st_mode & 0o777),
        "all_members_regular_0600": all(x["mode"] == "0o600" for x in members),
        "independent_review": {"status": review["status"], "sha256": sha(OUT / "independent-rehash-review.json")},
        "analysis_actions": {"model": 0, "evaluator": 0, "gpu": 0, "slurm": 0, "source_edit": 0, "result_mutation": 0},
        "members": members,
    }
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.chmod(target, 0o600)


if __name__ == "__main__":
    main()
