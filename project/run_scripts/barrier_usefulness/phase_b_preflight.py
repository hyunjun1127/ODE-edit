"""Identity-only release gate from corrected Phase A to fresh ctrl-only Phase B."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .hashing import canonical_hash, file_sha256, write_json_once


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--phase-a-terminal", type=Path, required=True)
    parser.add_argument("--anchor-seal", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _git(args.source_root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Phase B source is tracked-dirty")
    phase_a = json.loads(args.phase_a_terminal.read_text())
    if phase_a["status"] != "PHASE_A_PASS" or phase_a["q_gate_edited_access_count"] != 0 or len(phase_a["cells"]) != 4:
        raise SystemExit("Phase A terminal does not release Phase B")
    anchors = []
    for path in sorted(args.anchor_seal, key=str):
        row = json.loads(path.read_text())
        if row["status"] != "W0_ANCHOR_SEAL_PASS" or row["edited_candidate_access_count"] != 0 or len(row["cases"]) != 8:
            raise SystemExit("W0 anchor seal invalid")
        if any(case["ctrl"]["selected_count"] != 32 or case["gate"]["selected_count"] != 32 or case["selected_overlap_count"] != 0 for case in row["cases"]):
            raise SystemExit("W0 exact ctrl/gate anchor completeness failed")
        anchors.append({"model": row["model"], "path": str(path), "sha256": file_sha256(path), "root": row["root_digest"]})
    if {row["model"] for row in anchors} != {"llama3-8b-inst", "qwen2.5-7b-inst"}:
        raise SystemExit("two-model W0 anchor seals absent")
    payload = {
        "schema": "odeedit.s06.barrier-usefulness.phase-b-preflight.v1", "status": "PHASE_B_CTRL_RELEASE_PASS",
        "source": {"head": _git(args.source_root, "rev-parse", "HEAD"), "tree": _git(args.source_root, "rev-parse", "HEAD^{tree}"), "clean": True},
        "phase_a": {"path": str(args.phase_a_terminal), "sha256": file_sha256(args.phase_a_terminal)},
        "anchors": anchors, "edited_gate_access_count": 0, "selector_lock_count": 0,
        "replacement_count": 0, "final_audit_open_count": 0, "full_fp32": True,
    }
    payload["identity"] = canonical_hash(payload)
    write_json_once(args.output, payload)


if __name__ == "__main__":
    main()
