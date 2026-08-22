#!/usr/bin/env python3
"""Source-only Phase-1 dry plan."""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p1r52_b100x10_stream import SEAL_FILE, verify_p1r52_b100x10_stream
from project.run_scripts.ode_bf.p1r52_c_writer_phase1_sequential import ALPHA_APPLICABLE, CELL_LABELS, role_for_cell


def build_plan() -> dict[str, object]:
    stream = verify_p1r52_b100x10_stream(json.loads((REPO_ROOT / "project/run_scripts/ode_bf/locks" / SEAL_FILE).read_text(encoding="utf-8")))
    rows = []
    for cell, label in enumerate(CELL_LABELS):
        applicable = label in ALPHA_APPLICABLE
        rows.append({
            "cell": cell,
            "role": role_for_cell(cell),
            "label": label,
            "batch_count": 10,
            "batch_size": 100,
            "sequential_W_carry": True,
            "full_fp32": True,
            "alpha_cache_status": "ALPHA_CACHE_CONTINUITY_ON" if applicable else "ALPHA_CACHE_NOT_APPLICABLE_NATIVE_MEMIT",
            "alpha_cache_influence_count": 1 if applicable else 0,
            "request_history_width": 1000 if applicable else 0,
            "scientific_call_path": (
                "OFFICIAL_EASYEDIT_ALPHAEDIT" if label == "OFFICIAL-ALPHAEDIT"
                else "OFFICIAL_EASYEDIT_MEMIT" if label == "OFFICIAL-MEMIT"
                else "P1R52_ACCEPTED_Z_PLUS_RELEASED_C_WRITER"
            ),
        })
    payload: dict[str, object] = {
        "schema": "ode-edit-s05-p1r52-c-writer-phase1-dry-plan/v1",
        "array": "0-4%3",
        "project_gpu_cap": 3,
        "rows": rows,
        "cache_valid_denominator": "4/4_APPLICABLE_ARMS",
        "memit_alpha_cache_denominator_inclusion_count": 0,
        "stream_root": stream["root_digest"],
        "stream_order": stream["all_request_order_sha256"],
        "bf16_fp16_autocast_quantization_count": 0,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(build_plan(), sort_keys=True))
