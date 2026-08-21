#!/usr/bin/env python3
"""Independent raw-free rehash verifier for the canonical R1 package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


manifest = json.loads((HERE / "analysis-manifest.json").read_text())
for row in manifest["members"]:
    path = HERE / row["path"]
    assert path.is_file() and not path.is_symlink()
    assert path.stat().st_size == row["bytes"]
    assert path.read_bytes().count(b"\n") == row["lines"]
    assert sha(path) == row["sha256"]
receipt = json.loads((HERE / "rooted-analysis-receipt.json").read_text())
assert receipt["analysis_manifest_sha256"] == sha(HERE / "analysis-manifest.json")
assert receipt["same_accepted_z"] and receipt["same_writer_entry_W0"] and receipt["W0_restored"]
assert receipt["attempted"] == receipt["valid"] == 100
dtype = json.loads((HERE / "dtype-receipts.json").read_text())
assert dtype["global"]["model_storage"] == "torch.float32"
assert dtype["global"]["bf16_conversion_count"] == dtype["global"]["fp16_conversion_count"] == 0
assert dtype["C3"]["native_compute_z_call_count"] == 0
assert dtype["C3"]["p1r52_pc_router_decision_influence_count"] == 0
assert len(json.loads((HERE / "per-request.json").read_text())) == 3300
assert len(json.loads((HERE / "per-layer.json").read_text())) == 35
print(f"PASS members={len(manifest['members'])} per_request=3300 per_layer=35")
