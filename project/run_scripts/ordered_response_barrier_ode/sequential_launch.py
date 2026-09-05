"""Thin CLI for the Server4 ORBODE B100x10 sequential campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import shutil
import hashlib
from typing import Any, Mapping

from .preflight import (
    CELL_SPECS,
    PreflightBoundary,
    _assert_no_symlink_components,
    canonical_hash,
    canonical_json,
    cell_spec,
    MODEL_BINDINGS,
    validate_easyedit_source,
    validate_model_artifacts,
    validate_stream,
)
from .sequential_contracts import ARM_ORDER, INSTRUCTION_ID
from .sequential_preflight import (
    AUTHORITATIVE_DOC_ROOT,
    EASYEDIT_ARTIFACT_ROOT,
    EASYEDIT_SOURCE_ROOT,
    HF_HUB_CACHE_ROOT,
    MEMORY_MIB_PER_TASK,
    parse_local_cap,
    run_preflight,
    validate_authoritative_docs,
    validate_execution_lock,
    validate_source,
)


def _assert_private_regular(path: Path) -> None:
    _assert_no_symlink_components(path.absolute())
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or stat.S_IMODE(observed.st_mode) != 0o600:
        raise PreflightBoundary(f"private receipt identity differs: {path}")


def write_create_once_json(path: Path, value: Mapping[str, Any]) -> None:
    target = path.absolute()
    _assert_no_symlink_components(target.parent)
    if target.exists() or target.is_symlink() or not target.parent.is_dir():
        raise PreflightBoundary(f"create-once receipt target differs: {target}")
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _assert_private_regular(target)


def load_preflight(
    path: Path,
    *,
    source_head: str,
    source_tree: str,
    output_parent: Path,
    log_root: Path,
) -> dict[str, Any]:
    _assert_private_regular(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightBoundary("pre-GPU receipt is not an object")
    body = dict(value)
    identity = body.pop("identity_sha256", None)
    source = value.get("source")
    create_once = value.get("create_once")
    resource = value.get("resource")
    if (
        identity != canonical_hash(body)
        or value.get("instruction_id") != INSTRUCTION_ID
        or value.get("status") != "PRE_GPU_BINDING_PASS"
        or not isinstance(source, Mapping)
        or source.get("head") != source_head
        or source.get("tree") != source_tree
        or not isinstance(create_once, Mapping)
        or create_once.get("output_parent") != str(output_parent.absolute())
        or create_once.get("log_root") != str(log_root.absolute())
        or not isinstance(resource, Mapping)
        or resource.get("array") != "0-3%2"
        or resource.get("memory_mib_per_task") != MEMORY_MIB_PER_TASK
    ):
        raise PreflightBoundary("pre-GPU receipt binding differs")
    return value


def repeat_runtime_binding(
    *,
    repo_root: Path,
    source_head: str,
    source_tree: str,
    easyedit_source_root: Path,
    easyedit_artifact_root: Path,
    hf_hub_cache: Path,
    local_caps: Path,
) -> dict[str, Any]:
    """Repeat mutable identity checks immediately before model import."""

    value: dict[str, Any] = {
        "source": validate_source(repo_root, head=source_head, tree=source_tree),
        "authoritative_inputs": validate_authoritative_docs(),
        "execution_lock": validate_execution_lock(repo_root),
        "easyedit_source": validate_easyedit_source(easyedit_source_root),
        "stream": validate_stream(repo_root, easyedit_artifact_root),
        "model_artifacts": validate_model_artifacts(
            repo_root, easyedit_artifact_root, hf_hub_cache, deep_hash=False
        ),
        "resource": parse_local_cap(local_caps),
    }
    value["identity_sha256"] = canonical_hash(value)
    return value


def build_dry_plan(
    *,
    repo_root: Path,
    source_head: str,
    source_tree: str,
    easyedit_source_root: Path,
    easyedit_artifact_root: Path,
    hf_hub_cache: Path,
    local_caps: Path,
    output_parent: Path,
    log_root: Path,
) -> dict[str, Any]:
    binding = run_preflight(
        repo_root=repo_root,
        source_head=source_head,
        source_tree=source_tree,
        easyedit_source_root=easyedit_source_root,
        easyedit_artifact_root=easyedit_artifact_root,
        hf_hub_cache=hf_hub_cache,
        local_caps=local_caps,
        output_parent=output_parent,
        log_root=log_root,
        deep_artifact_hash=False,
    )
    value: dict[str, Any] = {
        "schema": "orbode.server4.sequential-dry-plan.v1",
        "instruction_id": INSTRUCTION_ID,
        "status": "DRY_PLAN_PASS",
        "source_head": source_head,
        "source_tree": source_tree,
        "binding_identity_sha256": binding["identity_sha256"],
        "array": "0-3%2",
        "mapping": {
            str(item.cell_id): [item.model_alias, item.writer_family]
            for item in CELL_SPECS
        },
        "arms": list(ARM_ORDER),
        "execution": "ARM_LOCAL_B1_TO_B10_CUMULATIVE_W_CACHE_SEQUENTIAL",
        "batch_count_per_arm": 10,
        "request_count_per_arm": 1_000,
        "memory_mib_per_task": MEMORY_MIB_PER_TASK,
        "action_counts": {"gpu": 0, "model": 0, "slurm": 0},
    }
    value["identity_sha256"] = canonical_hash(value)
    return value


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(allow_abbrev=False)
    commands = value.add_subparsers(dest="command", required=True)
    for name in ("dry-plan", "seal-preflight", "run-cell"):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--source-head", required=True)
        command.add_argument("--source-tree", required=True)
        command.add_argument("--easyedit-source-root", type=Path, default=EASYEDIT_SOURCE_ROOT)
        command.add_argument("--easyedit-artifact-root", type=Path, default=EASYEDIT_ARTIFACT_ROOT)
        command.add_argument("--hf-hub-cache", type=Path, default=HF_HUB_CACHE_ROOT)
        command.add_argument("--local-caps", type=Path, required=True)
        command.add_argument("--output-parent", type=Path, required=True)
        command.add_argument("--log-root", type=Path, required=True)
        command.add_argument("--cumulative-observation", action="store_true")
    commands.choices["dry-plan"].add_argument("--receipt", type=Path, required=True)
    commands.choices["seal-preflight"].add_argument("--receipt", type=Path, required=True)
    commands.choices["run-cell"].add_argument("--preflight-receipt", type=Path, required=True)
    commands.choices["run-cell"].add_argument("--cell-id", type=int, required=True)
    commands.choices["run-cell"].add_argument("--run-token", required=True)
    return value


def _dry_plan(args: argparse.Namespace) -> int:
    value = build_dry_plan(
        repo_root=args.repo_root,
        source_head=args.source_head,
        source_tree=args.source_tree,
        easyedit_source_root=args.easyedit_source_root,
        easyedit_artifact_root=args.easyedit_artifact_root,
        hf_hub_cache=args.hf_hub_cache,
        local_caps=args.local_caps,
        output_parent=args.output_parent,
        log_root=args.log_root,
    )
    if args.cumulative_observation:
        value['cumulative_observation']=True
        value['cumulative_measurement_plan']=cumulative_measurement_plan(args.repo_root,args.output_parent,args.hf_hub_cache)
        value.pop('identity_sha256');value['identity_sha256']=canonical_hash(value)
    write_create_once_json(args.receipt, value)
    print(canonical_json({"receipt": str(args.receipt.absolute()), "status": value["status"]}))
    return 0


def cumulative_measurement_plan(repo:Path,output:Path,hub:Path)->dict[str,Any]:
    estimates={};total=0
    for alias,spec in MODEL_BINDINGS.items():
        path=hub/str(spec['hf_repo'])/'snapshots'/str(spec['revision'])/'config.json'
        data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=spec['config_sha256']:
            raise PreflightBoundary('cumulative storage model-config identity')
        config=json.loads(data);one=5*4*int(config['hidden_size'])*int(config['intermediate_size'])
        estimates[alias]={'five_edited_FP32_tensor_bytes':one,'checkpoint_count_two_methods':100}
        total+=100*one
    free=shutil.disk_usage(output if output.exists() else output.parent).free
    # Reserve 128 GiB plus 25% for journals, target tensors and serialization.
    if free<total*1.25+128*2**30:
        raise PreflightBoundary('cumulative storage reserve insufficient')
    mapping=repo/'project/run_scripts/ordered_response_barrier_ode/cumulative_claim_measurement_map.md'
    return {'status':'CUMULATIVE_MEASUREMENT_RESOURCE_PASS','claim_map_path':str(mapping),
        'claim_map_sha256':hashlib.sha256(mapping.read_bytes()).hexdigest(),
        'checkpoint_storage_bytes':total,'storage_estimates':estimates,'available_bytes':free,
        'reserve_bytes':128*2**30,'checkpoints':200,'request_state_rows':110000,
        'evaluation_type':'CHECKPOINT_W_ON_ALL_SEEN_REQUESTS',
        'final_requests_per_arm':1000,'final_rewrite_per_arm':1000,'final_rephrase_per_arm':2000,'final_neighborhood_per_arm':10000,
        'historical_layer_probe_cohorts':'ALL_SEEN_EVERY_VISITED_LAYER',
        'controller_feedback':0,'z_reoptimization':0,'old_results_overwrite':0,
        'gpu_hours_planning_range':[64,144],'estimate_basis':'prior four-cell total45.8 GPUh; added 110000 request-state evaluations and all-seen activation probes, not measured runtime',
        'native_Creg_action':'UNBOUND_NOT_FROBENIUS_EQUIVALENT','scientific_promotion':False}


def _seal_preflight(args: argparse.Namespace) -> int:
    value = run_preflight(
        repo_root=args.repo_root,
        source_head=args.source_head,
        source_tree=args.source_tree,
        easyedit_source_root=args.easyedit_source_root,
        easyedit_artifact_root=args.easyedit_artifact_root,
        hf_hub_cache=args.hf_hub_cache,
        local_caps=args.local_caps,
        output_parent=args.output_parent,
        log_root=args.log_root,
        deep_artifact_hash=True,
    )
    if args.cumulative_observation:
        value['cumulative_observation']=True
        value['cumulative_measurement_lock']='cumulative_claim_measurement_map.md'
        value['cumulative_measurement_plan']=cumulative_measurement_plan(args.repo_root,args.output_parent,args.hf_hub_cache)
        value.pop('identity_sha256')
        value['identity_sha256']=canonical_hash(value)
    write_create_once_json(args.receipt, value)
    print(canonical_json({"receipt": str(args.receipt.absolute()), "status": value["status"]}))
    return 0


def _run_cell(args: argparse.Namespace) -> int:
    spec = cell_spec(args.cell_id)
    expected_token = (
        f"s06-orbode-sequential-server4-cell{spec.cell_id}-"
        f"{spec.model_alias}-{spec.writer_family.lower()}-r1"
    )
    if args.run_token != expected_token:
        raise PreflightBoundary("sequential run token differs")
    sealed=load_preflight(
        args.preflight_receipt,
        source_head=args.source_head,
        source_tree=args.source_tree,
        output_parent=args.output_parent,
        log_root=args.log_root,
    )
    if bool(sealed.get('cumulative_observation',False))!=args.cumulative_observation:
        raise PreflightBoundary('cumulative observer/preflight binding differs')
    repeat_runtime_binding(
        repo_root=args.repo_root,
        source_head=args.source_head,
        source_tree=args.source_tree,
        easyedit_source_root=args.easyedit_source_root,
        easyedit_artifact_root=args.easyedit_artifact_root,
        hf_hub_cache=args.hf_hub_cache,
        local_caps=args.local_caps,
    )
    task_root = args.output_parent.absolute() / f"task-{args.cell_id}"
    if task_root.exists() or task_root.is_symlink():
        raise PreflightBoundary("cell create-once result root exists")

    from .sequential_runtime import run_cell

    terminal = run_cell(
        repo_root=args.repo_root,
        easyedit_source_root=args.easyedit_source_root,
        easyedit_artifact_root=args.easyedit_artifact_root,
        hf_hub_cache=args.hf_hub_cache,
        output_root=task_root,
        source_head=args.source_head,
        source_tree=args.source_tree,
        cell_id=args.cell_id,
        run_token=args.run_token,
        cumulative_observation=args.cumulative_observation,
    )
    print(canonical_json({"cell_id": args.cell_id, "status": terminal["status"]}))
    return 0


def main() -> int:
    args = parser().parse_args()
    if args.command == "dry-plan":
        return _dry_plan(args)
    if args.command == "seal-preflight":
        return _seal_preflight(args)
    return _run_cell(args)


__all__ = [
    "build_dry_plan",
    "load_preflight",
    "main",
    "repeat_runtime_binding",
    "write_create_once_json",
]
