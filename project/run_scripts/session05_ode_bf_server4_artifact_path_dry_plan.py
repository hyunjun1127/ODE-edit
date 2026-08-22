#!/usr/bin/env python3
"""Server4 no-model AlphaEdit runtime-path preflight and dry plan."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.run_scripts.alphaedit_runtime_path_seal import (
    MODEL_ALIASES,
    load_alphaedit_runtime_path_seal,
    verify_focused_source_manifest,
)


SCHEMA = "ode-edit-s4-m1-r1-server4-artifact-path-dry-plan/v1"
SEAL_RELATIVE = Path("agents/server4/alphaedit-runtime-path-seal.json")
SOURCE_MANIFEST_RELATIVE = Path(
    "project/run_scripts/ode_bf/locks/source_manifest_s4_m1_r1_server4_path_seal.json"
)
BF_LOCK_RELATIVE = Path("project/run_scripts/ode_bf/locks/p0_artifact_lock.json")
BASE_LOCK_RELATIVE = Path("project/run_scripts/ode_alloc/p0_artifact_lock_r1.json")


def build_plan(*, repository_root: Path = REPO_ROOT) -> dict[str, object]:
    root = repository_root.resolve(strict=True)
    source_manifest_sha256 = verify_focused_source_manifest(
        root / SOURCE_MANIFEST_RELATIVE,
        repo_root=root,
    )
    seal = load_alphaedit_runtime_path_seal(
        root / SEAL_RELATIVE,
        repo_root=root,
    )
    bf_lock = json.loads((root / BF_LOCK_RELATIVE).read_text(encoding="utf-8"))
    base_lock = json.loads((root / BASE_LOCK_RELATIVE).read_text(encoding="utf-8"))

    receipts: list[dict[str, object]] = []
    for alias in MODEL_ALIASES:
        seal.assert_odebf_model_contract(alias, bf_lock["models"][alias])
        seal.assert_p0_model_contract(alias, base_lock["models"][alias])
        receipts.append(asdict(seal.preflight_alias(alias)))

    required_mappings = (
        "easyedit_root",
        "hf_hub_cache",
        "alphaedit_evaluator_root",
    )
    missing_mappings = [
        name for name in required_mappings if not seal.has_mapping(name)
    ]
    withheld = dict(seal.withheld_mappings)
    blockers = [
        {"mapping": name, "reason": withheld.get(name, "mapping is absent")}
        for name in missing_mappings
    ]
    return {
        "schema_version": SCHEMA,
        "source_manifest_sha256": source_manifest_sha256,
        "seal": {
            "path": str(root / SEAL_RELATIVE),
            "seal_id": seal.seal_id,
            "sha256": seal.seal_sha256,
            "root_digest": seal.root_digest,
        },
        "runtime_identity": asdict(seal.identity),
        "guard_contract_delivery": {
            "P0ArtifactGuard": True,
            "ODEBFArtifactGuard": True,
        },
        "artifact_receipts": receipts,
        "active_mappings": {
            mapping.name: {
                "logical_locked_path": mapping.logical_root,
                "resolved_runtime_path": mapping.runtime_root,
            }
            for mapping in seal.mappings
        },
        "mapping_blockers": blockers,
        "easyedit_seal_ready": seal.has_mapping("easyedit_root"),
        "launcher_ready": not missing_mappings,
        "model_load": False,
        "gpu_use": False,
        "slurm_submit": False,
        "scientific_submit": "HOLD",
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            build_plan(repository_root=args.repository_root),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
