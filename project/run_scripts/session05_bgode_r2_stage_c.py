#!/usr/bin/env python3
"""Fail-closed entrypoint for BGODE-R2 Stage-C topology validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from project.run_scripts.barrier_guided_ode.r2.stage_b_contract import MODEL_BINDINGS
from project.run_scripts.barrier_guided_ode.r2.stage_b_probe import INSTRUCTION_ID, run_stage_b
from project.run_scripts.barrier_guided_ode.r2.stage_c_contract import (
    TOPOLOGY_MANIFEST,
    TOPOLOGY_NAMES,
    load_stage_c_manifest,
    topology_case,
)
from project.run_scripts.barrier_guided_ode.s1_experiment import s1_easyedit_pins
from project.run_scripts.ode_edit_motivation.contracts import canonical_json
from project.run_scripts.ode_edit_motivation.easyedit_bridge import EasyEditBridge
from project.run_scripts.ode_edit_motivation.manifests import preflight_fixed_artifacts
from project.run_scripts.session05_bgode_r2_stage_b import validate_source_release


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "project/run_scripts/barrier_guided_ode/r2/locks/bgode-r2-stage-c-numerical-lock.json"
MANIFEST_PATH = REPO_ROOT / "project/run_scripts/barrier_guided_ode/r2/locks/bgode-r2-stage-c-source-manifest.json"
SCHEMA = "ode-edit-bgode-r2-stage-c-topology-validation/v1"


def run_id(model_alias: str, topology: str) -> str:
    if model_alias not in MODEL_BINDINGS or topology not in TOPOLOGY_NAMES:
        raise ValueError("Stage-C run identity input differs")
    return f"s05-bgode-r2-stage-c-{model_alias}-{topology}-fp64-two-equality-v1"


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--easyedit-root", type=Path, default=Path("/mnt/raid5/janghj/EasyEdit"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--model-alias", choices=tuple(MODEL_BINDINGS), required=True)
    parser.add_argument("--topology", choices=TOPOLOGY_NAMES, required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    expected = run_id(args.model_alias, args.topology)
    if args.run_token != expected:
        raise ValueError("Stage-C run token differs")
    release = validate_source_release(
        args.source_head,
        args.source_tree,
        manifest_path=MANIFEST_PATH,
        lock_path=LOCK_PATH,
    )
    topology_manifest, topology_manifest_sha = load_stage_c_manifest(TOPOLOGY_MANIFEST)
    case = topology_case(args.model_alias, args.topology)
    plan = {
        "schema": SCHEMA + "/dry-plan",
        "instruction_id": INSTRUCTION_ID,
        "source_head": args.source_head,
        "source_tree": args.source_tree,
        "model_alias": args.model_alias,
        "topology": args.topology,
        "run_id": expected,
        "case": case.receipt(),
        "topology_manifest_identity": topology_manifest["identity"],
        "topology_manifest_sha256": topology_manifest_sha,
        "selection_influence": topology_manifest["selection_influence"],
        "termination_token_count": 0,
        "gpu_count": 1,
        "dynamic_writer_action_count": 0,
        "release": release,
        "scientific_promotion": False,
    }
    if args.preflight_only:
        fixed = preflight_fixed_artifacts(args.easyedit_root, model_alias=args.model_alias)
        bridge = EasyEditBridge(
            args.easyedit_root,
            expected_files=s1_easyedit_pins(),
            include_alphaedit_reference=True,
        ).preflight()
        plan.update({
            "fixed_input_manifest_id": fixed.manifest_id,
            "easyedit_bridge_manifest_id": bridge.manifest_id,
            "external_input_preflight": "PASS",
        })
        print(canonical_json(plan))
        return 0
    if args.output_root is None:
        parser.error("--output-root is required")
    result = run_stage_b(
        easyedit_root=args.easyedit_root,
        output_root=args.output_root,
        source_head=args.source_head,
        source_tree=args.source_tree,
        release_receipt=release,
        model_alias=args.model_alias,
        run_id=expected,
        expected_run_id=expected,
        event_token_override=(case.target_token_ids, case.source_token_ids),
        terminal_schema=SCHEMA,
        pass_status="BGODE_R2_STAGE_C_FP64_TOPOLOGY_PASS",
        numerical_boundary_status="BGODE_R2_STAGE_C_NUMERICAL_IMPLEMENTATION_BOUNDARY",
        infeasible_status="BGODE_R2_STAGE_C_ACTUATOR_EQUALITY_INFEASIBLE",
        stage_context={
            "topology": case.topology,
            "case_identity": case.case_identity,
            "topology_manifest_identity": topology_manifest["identity"],
            "topology_manifest_sha256": topology_manifest_sha,
            "selection_influence": topology_manifest["selection_influence"],
        },
    )
    print(canonical_json({"status": result["status"], "run_id": result["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
