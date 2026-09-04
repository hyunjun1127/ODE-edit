"""Arm-local B1-to-B10 cumulative ORBODE runtime for Server4.

The numerical field, writer construction, response controller, integrator,
endpoint evaluator, and raw-free reducers are imported from the accepted
ORBODE implementation.  This module owns only the cross-B100 transaction
lifecycle: each arm begins at the same cold W0, commits ten B100 endpoints in
sealed order, and is then restored before the next arm.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import stat
import time
import traceback
from typing import Any, Mapping, Sequence

import torch

from .artifacts import (
    reduce_derived_endpoint_payload,
    reduce_endpoint_payload,
    reduce_evaluation_payload,
)
from .contracts import (
    ArmId,
    ORBODERuntimeLock,
    TechnicalBoundary,
    assert_full_fp32,
    canonical_arm_configs,
)
from .fp32_overlay import tensor_set_sha256
from .preflight import MODEL_BINDINGS, ORDER_ROOT, STREAM_ROOT, canonical_hash
from .sequential_contracts import (
    ARM_ORDER,
    DERIVED_ARM,
    INSTRUCTION_ID,
    NONCE,
    ROUND_INDICES,
    SequentialRuntimeLock,
    validate_batch_chain,
)
from .terminal_jvp import seal_eager_attention
from .runtime import (
    FamilyRuntime,
    _arm_run,
    _bootstrap_easyedit,
    _cell,
    _clone_selected,
    _create_once_json,
    _git,
    _install_model_forward_counter,
    _load_hparams,
    _load_stream,
    _method_module,
    _model_forward_count,
    _model_name,
    _model_snapshot,
    _official_requests,
    _parameter_inventory,
    _request_order_identity,
    _restore_selected,
    _selected,
    _single_request_preamble,
    _sync,
)


RESULT_SCHEMA = "orbode.server4.sequential-cell.v1"
BATCH_SCHEMA = "orbode.server4.sequential-batch.v1"
ARM_SCHEMA = "orbode.server4.sequential-arm.v1"
FAILURE_SCHEMA = "orbode.server4.sequential-failure.v1"
FIRST_GATE_SCHEMA = "orbode.server4.sequential-first-valid-gate.v1"


def _method_state_snapshot(family: str, module: Any) -> torch.Tensor | None:
    if family != "AlphaEdit":
        return None
    value = getattr(module, "cache_c", None)
    if not isinstance(value, torch.Tensor) or value.dtype is not torch.float32:
        raise TechnicalBoundary("AlphaEdit cache snapshot is unavailable")
    return value.detach().clone()


def _restore_method_state(
    family: str,
    module: Any,
    alpha_cache: torch.Tensor | None,
    *,
    alpha_cache_new: bool,
    memit_versions: Mapping[tuple[Any, ...], int],
) -> None:
    if family == "AlphaEdit":
        if alpha_cache is None:
            raise TechnicalBoundary("AlphaEdit cold cache snapshot is absent")
        module.cache_c.copy_(alpha_cache)
        module.cache_c_new = bool(alpha_cache_new)
    elif {key: int(value._version) for key, value in module.COV_CACHE.items()} != dict(
        memit_versions
    ):
        raise TechnicalBoundary("MEMIT static covariance cache mutated")


def _raw_free_batch(
    *,
    cell_id: int,
    model_alias: str,
    writer_family: str,
    arm: str,
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
    request_order_sha256: str,
    sealed_batch_order_digest: str,
    family: FamilyRuntime,
    fixed_z: Any,
    raw_arm: Mapping[str, Any],
    commit: Mapping[str, Any],
    entry_evaluation: Mapping[str, Any],
    runtime_preamble: Mapping[str, Any] | None,
    entry_history_width: int,
    exit_history_width: int,
) -> dict[str, Any]:
    case_ids = [int(item["case_id"]) for item in batch]
    request_sha256 = [str(item["request_sha256"]) for item in batch]
    pre = reduce_evaluation_payload(
        entry_evaluation,
        case_ids=case_ids,
        request_sha256=request_sha256,
        request_order_sha256=request_order_sha256,
    )
    endpoint = reduce_endpoint_payload(
        raw_arm,
        expected_arm=arm,
        case_ids=case_ids,
        request_sha256=request_sha256,
        request_order_sha256=request_order_sha256,
        entry_evaluation=entry_evaluation,
    )
    derived = None
    if arm == ArmId.ORDERED_RESPONSE_FIXED_HORIZON.value:
        raw_derived = raw_arm.get("derived_endpoint")
        if not isinstance(raw_derived, Mapping):
            raise TechnicalBoundary("ORBFH derived ORBHit endpoint is absent")
        derived = reduce_derived_endpoint_payload(
            raw_derived,
            case_ids=case_ids,
            request_sha256=request_sha256,
            request_order_sha256=request_order_sha256,
            entry_evaluation=entry_evaluation,
        )
    payload: dict[str, Any] = {
        "schema": BATCH_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "nonce": NONCE,
        "status": "SEQUENTIAL_BATCH_TERMINAL_VALID",
        "cell_id": cell_id,
        "model_alias": model_alias,
        "writer_family": writer_family,
        "arm": arm,
        "batch_index": batch_index,
        "request_count": len(batch),
        "case_ids": case_ids,
        "request_sha256": request_sha256,
        "request_order_sha256": request_order_sha256,
        "sealed_batch_order_digest": sealed_batch_order_digest,
        "fixed_z": {
            "identity_sha256": fixed_z.identity_sha256,
            "target_context_identity_sha256": fixed_z.target_context_identity_sha256,
            "compute_count": fixed_z.request_count,
            "recompute_count": 0,
        },
        "entry_evaluation": pre,
        "endpoint": endpoint,
        "derived_endpoint": derived,
        "runtime_preamble": runtime_preamble,
        "entry_weight_sha256": str(commit["entry_weight_sha256"]),
        "committed_weight_sha256": str(commit["committed_weight_sha256"]),
        "entry_method_state_sha256": str(commit["entry_method_state_sha256"]),
        "committed_method_state_sha256": str(
            commit["committed_method_state_sha256"]
        ),
        "sequential_commit": dict(commit),
        "alpha_history_width": {
            "applicable": writer_family == "AlphaEdit",
            "entry": entry_history_width if writer_family == "AlphaEdit" else 0,
            "exit": exit_history_width if writer_family == "AlphaEdit" else 0,
            "append": exit_history_width - entry_history_width
            if writer_family == "AlphaEdit"
            else 0,
        },
        "memit_covariance_static": writer_family == "MEMIT",
        "controller_evaluator_influence_count": 0,
        "dynamic_z_recompute_count": 0,
        "retry_count": 0,
        "backtracking_count": 0,
        "fallback_count": 0,
        "full_fp32": True,
        "nonfinite_count": 0,
        "imputation_count": 0,
        "literal_prompt_target_token_prediction_publication_count": 0,
        "scientific_promotion": False,
    }
    payload["identity_sha256"] = canonical_hash(payload)
    return payload


def _first_gate(
    *,
    root: Path,
    cell_id: int,
    model_alias: str,
    writer_family: str,
    source_head: str,
    source_tree: str,
    official_records: Sequence[Mapping[str, Any]],
    qcl_first_batch: Mapping[str, Any],
) -> dict[str, Any]:
    if len(official_records) < 2:
        raise TechnicalBoundary("Official B1 to B2 continuity evidence is absent")
    first, second = official_records[0], official_records[1]
    if first["committed_weight_sha256"] != second["entry_weight_sha256"]:
        raise TechnicalBoundary("Official B1 exit does not bind B2 entry")
    if first["committed_method_state_sha256"] != second["entry_method_state_sha256"]:
        raise TechnicalBoundary("Official B1 method state does not bind B2 entry")
    telemetry = qcl_first_batch["endpoint"]["mechanism_telemetry"]
    facts = telemetry.get("telemetry") if isinstance(telemetry, Mapping) else None
    steps = facts.get("steps") if isinstance(facts, Mapping) else None
    if not isinstance(steps, list):
        raise TechnicalBoundary("QCL dynamic step ledger is absent")
    first_step = steps[0] if steps else None
    canonical_ns = qcl_first_batch["endpoint"]["evaluation"]["locality"].get(
        "canonical_ns"
    )
    if (
        not isinstance(canonical_ns, Mapping)
        or canonical_ns.get("prompt_denominator") != 1_000
        or qcl_first_batch["fixed_z"]["compute_count"] != 100
        or qcl_first_batch["fixed_z"]["recompute_count"] != 0
    ):
        raise TechnicalBoundary("QCL first endpoint/NS/fixed-z denominator differs")
    value: dict[str, Any] = {
        "schema": FIRST_GATE_SCHEMA,
        "instruction_id": INSTRUCTION_ID,
        "status": "FIRST_VALID_SEQUENTIAL_CONTINUITY_GATE_PASS",
        "cell_id": cell_id,
        "model_alias": model_alias,
        "writer_family": writer_family,
        "source_head": source_head,
        "source_tree": source_tree,
        "official_B1_exit_sha256": first["committed_weight_sha256"],
        "official_B2_entry_sha256": second["entry_weight_sha256"],
        "official_weight_link_pass": True,
        "official_method_state_link_pass": True,
        "qcl_first_batch_identity_sha256": qcl_first_batch["identity_sha256"],
        "qcl_first_dynamic_node": (
            {
                "sweep": first_step["sweep"],
                "layer": first_step["layer"],
                "visit_ordinal": first_step["visit_ordinal"],
                "built_state_version": first_step["built_state_version"],
                "resulting_state_version": first_step["resulting_state_version"],
            }
            if first_step is not None
            else None
        ),
        "qcl_dynamic_node_observation": (
            "FIRST_VALID_NODE_RECORDED"
            if first_step is not None
            else "NONBLOCKING_ZERO_OR_ENTRY_HIT_OBSERVATION"
        ),
        "fixed_z_compute_count": 100,
        "fixed_z_recompute_count": 0,
        "canonical_ns_prompt_denominator": 1_000,
        "controller_evaluator_influence_count": 0,
        "full_fp32": True,
        "nonfinite_count": 0,
        "scientific_promotion": False,
    }
    value["identity_sha256"] = canonical_hash(value)
    _create_once_json(root / "first-valid-gate.json", value)
    return value


def run_cell(
    *,
    repo_root: Path,
    easyedit_source_root: Path,
    easyedit_artifact_root: Path,
    hf_hub_cache: Path,
    output_root: Path,
    source_head: str,
    source_tree: str,
    cell_id: int,
    run_token: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = output_root.absolute()
    if root.exists() or root.is_symlink():
        raise TechnicalBoundary(f"create-once task root exists: {root}")
    root.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    failure_path = root / "failure-boundary.json"
    model: torch.nn.Module | None = None
    forward_handle: Any | None = None
    active_family: FamilyRuntime | None = None
    completed: list[dict[str, Any]] = []
    progress: dict[str, Any] = {
        "stage": "TASK_ROOT_CREATED",
        "arm": None,
        "batch_index": None,
        "completed_batch_count": 0,
    }
    try:
        cell = _cell(cell_id)
        expected_token = (
            f"s06-orbode-sequential-server4-cell{cell.cell_id}-"
            f"{cell.model_alias}-{cell.writer_family.lower()}-r1"
        )
        if run_token != expected_token:
            raise TechnicalBoundary("sequential run token or array mapping differs")
        if (
            _git(repo_root, "rev-parse", "HEAD") != source_head
            or _git(repo_root, "rev-parse", "HEAD^{tree}") != source_tree
            or _git(repo_root, "status", "--porcelain", "--untracked-files=no")
        ):
            raise TechnicalBoundary("queued source HEAD/tree/clean identity differs")
        easyedit = _bootstrap_easyedit(easyedit_source_root)
        if (
            easyedit["head"] != "14cea8245f06715684592ab55184939b99d70784"
            or not easyedit["tracked_clean"]
        ):
            raise TechnicalBoundary("pinned stock EasyEdit identity differs")

        os.environ.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "WANDB_DISABLED": "true",
            }
        )
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise TechnicalBoundary("runtime requires exactly one visible CUDA device")
        torch.cuda.set_device(0)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        snapshot = _model_snapshot(hf_hub_cache, cell.model_alias)
        _sync()
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map={"": "cuda:0"},
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False, use_fast=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.eval()
        attention_receipt = dict(seal_eager_attention(model))
        config_path = (snapshot / "config.json").resolve(strict=True)
        if not stat.S_ISREG(config_path.stat().st_mode):
            raise TechnicalBoundary("HF model config is not a regular file")
        attention_receipt.update(
            {
                "hf_config_path": str(config_path),
                "hf_config_sha256": __import__("hashlib").sha256(
                    config_path.read_bytes()
                ).hexdigest(),
                "hf_config_bytes": config_path.stat().st_size,
            }
        )
        forward_handle = _install_model_forward_counter(model)
        dtype_receipt = assert_full_fp32(model)
        load_seconds = time.perf_counter() - load_started
        progress["stage"] = "MODEL_LOAD_FP32_ATTENTION_PASS"

        hparams, hparams_path = _load_hparams(
            repo_root, cell.writer_family, cell.model_alias
        )
        hparams.device = 0
        hparams.stats_dir = str(easyedit_artifact_root / "examples/data/stats")
        if cell.writer_family == "AlphaEdit":
            hparams.P_loc = str(
                easyedit_artifact_root
                / str(MODEL_BINDINGS[cell.model_alias]["projector"][0])
            )
        if tuple(int(item) for item in hparams.layers) != (4, 5, 6, 7, 8):
            raise TechnicalBoundary("writer layer inventory differs")
        module = _method_module(cell.writer_family)
        module.CONTEXT_TEMPLATES_CACHE = None
        with _model_name(model, str(hparams.model_name)):
            contexts = module.get_context_templates(model, tokenizer)
        if not isinstance(contexts, list) or not contexts:
            raise TechnicalBoundary("stock compute-z context builder failed")

        dataset = (easyedit_artifact_root / "data/counterfact/counterfact.json").resolve(
            strict=True
        )
        seal, batches, raw_map = _load_stream(
            repo_root, dataset, ROUND_INDICES, wave="remaining"
        )
        parameters = _selected(model, hparams)
        global_w0 = _clone_selected(parameters)
        global_w0_sha = tensor_set_sha256(global_w0)
        global_pointers = {name: int(value.data_ptr()) for name, value in parameters.items()}

        first_requests = _official_requests(batches[0])
        setup = FamilyRuntime(
            model=model,
            tokenizer=tokenizer,
            family=cell.writer_family,
            hparams=hparams,
            module=module,
            requests=first_requests,
            endpoint_records=tuple(raw_map[int(item["case_id"])] for item in batches[0]),
            request_order_sha256=_request_order_identity(batches[0]),
            contexts=contexts,
        )
        setup.prepare_method_state()
        setup.reset_entry()
        initial_method_identity = setup.method_state_identity()
        initial_alpha_cache = _method_state_snapshot(cell.writer_family, module)
        initial_alpha_cache_new = bool(getattr(module, "cache_c_new", False))
        memit_versions = dict(setup._cov_versions)
        parameter_inventory = _parameter_inventory(model)
        runtime_preamble: Mapping[str, Any] | None = None
        arm_receipts: list[dict[str, Any]] = []
        first_gate: Mapping[str, Any] | None = None

        for config in canonical_arm_configs(sweeps=4)[:5]:
            arm = config.arm.value
            progress.update(stage="ARM_RESTORE_COLD_ENTRY", arm=arm, batch_index=None)
            _restore_selected(parameters, global_w0)
            _restore_method_state(
                cell.writer_family,
                module,
                initial_alpha_cache,
                alpha_cache_new=initial_alpha_cache_new,
                memit_versions=memit_versions,
            )
            if tensor_set_sha256(parameters) != global_w0_sha:
                raise TechnicalBoundary("arm did not begin at common original W0")
            arm_records: list[dict[str, Any]] = []
            arm_journals: list[dict[str, Any]] = []
            logical_history_width = 0
            previous_commit = global_w0_sha
            previous_method_state = initial_method_identity

            for batch_index, batch in enumerate(batches):
                progress.update(
                    stage="BATCH_ENTRY",
                    arm=arm,
                    batch_index=batch_index,
                    completed_batch_count=len(completed),
                )
                requests = _official_requests(batch)
                order_identity = _request_order_identity(batch)
                endpoint_records = tuple(
                    raw_map[int(item["case_id"])] for item in batch
                )
                active_family = FamilyRuntime(
                    model=model,
                    tokenizer=tokenizer,
                    family=cell.writer_family,
                    hparams=hparams,
                    module=module,
                    requests=requests,
                    endpoint_records=endpoint_records,
                    request_order_sha256=order_identity,
                    contexts=contexts,
                    capture_persistent_endpoint=True,
                )
                active_family.bind_existing_method_state()
                active_family.reset_entry()
                if (
                    active_family.w0_sha256 != previous_commit
                    or active_family.method_state_identity() != previous_method_state
                ):
                    raise TechnicalBoundary("previous B100 commit does not bind next entry")
                try:
                    fixed_z = active_family.adapter(compute=True).compute_fixed_z_once()
                    active_family.pre_evaluation = active_family.evaluate_endpoint()
                    if arm == "O" and batch_index == 0:
                        progress["stage"] = "FAST_RUNTIME_PREAMBLE"
                        runtime_preamble = _single_request_preamble(
                            family=active_family,
                            raw_request=batch[0],
                            endpoint_record=endpoint_records[0],
                        )
                        if runtime_preamble.get("status") != "FAST_RUNTIME_PREAMBLE_PASS":
                            raise TechnicalBoundary("FAST runtime preamble did not pass")
                    progress["stage"] = "ARM_BATCH_EXECUTION"
                    raw_arm = _arm_run(active_family, config)
                    endpoint = raw_arm.get("endpoint")
                    if not isinstance(endpoint, Mapping):
                        raise TechnicalBoundary("arm endpoint payload is absent")
                    expected_sha = str(endpoint.get("selected_weight_endpoint_sha256"))
                    commit = active_family.commit_captured_endpoint(
                        expected_sha256=expected_sha
                    )
                    history_append = int(endpoint.get("history_append_count", 0))
                    entry_width = logical_history_width
                    if cell.writer_family == "AlphaEdit":
                        logical_history_width += 100 * history_append
                    batch_payload = _raw_free_batch(
                        cell_id=cell_id,
                        model_alias=cell.model_alias,
                        writer_family=cell.writer_family,
                        arm=arm,
                        batch_index=batch_index,
                        batch=batch,
                        request_order_sha256=order_identity,
                        sealed_batch_order_digest=seal[
                            "batch_ordered_request_digest_v1"
                        ][batch_index],
                        family=active_family,
                        fixed_z=fixed_z,
                        raw_arm=raw_arm,
                        commit=commit,
                        entry_evaluation=active_family.pre_evaluation,
                        runtime_preamble=(
                            runtime_preamble
                            if arm == "O" and batch_index == 0
                            else None
                        ),
                        entry_history_width=entry_width,
                        exit_history_width=logical_history_width,
                    )
                    arm_dir = root / f"arm-{arm}"
                    arm_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                    batch_path = arm_dir / f"batch-{batch_index + 1:02d}.json"
                    batch_sha = _create_once_json(batch_path, batch_payload)
                except BaseException:
                    active_family.reset_entry()
                    raise

                record = {
                    "status": "SEQUENTIAL_BATCH_TERMINAL_VALID",
                    "batch_index": batch_index,
                    "request_count": len(batch),
                    "fixed_z_compute_count": fixed_z.request_count,
                    "fixed_z_recompute_count": 0,
                    "entry_weight_sha256": str(commit["entry_weight_sha256"]),
                    "committed_weight_sha256": str(
                        commit["committed_weight_sha256"]
                    ),
                    "entry_method_state_sha256": str(
                        commit["entry_method_state_sha256"]
                    ),
                    "committed_method_state_sha256": str(
                        commit["committed_method_state_sha256"]
                    ),
                }
                arm_records.append(record)
                arm_journals.append(
                    {
                        "batch_index": batch_index,
                        "relative_path": str(batch_path.relative_to(root)),
                        "file_sha256": batch_sha,
                        "identity_sha256": batch_payload["identity_sha256"],
                        "request_count": 100,
                    }
                )
                completed.append({"arm": arm, "batch_index": batch_index})
                previous_commit = str(commit["committed_weight_sha256"])
                previous_method_state = str(
                    commit["committed_method_state_sha256"]
                )
                progress.update(
                    stage="BATCH_TERMINAL_VALID",
                    completed_batch_count=len(completed),
                )

                if arm == "QCL" and batch_index == 0 and first_gate is None:
                    official_path = root / "arm-O"
                    official_records = []
                    for ordinal in (1, 2):
                        value = json.loads(
                            (official_path / f"batch-{ordinal:02d}.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        official_records.append(value)
                    first_gate = _first_gate(
                        root=root,
                        cell_id=cell_id,
                        model_alias=cell.model_alias,
                        writer_family=cell.writer_family,
                        source_head=source_head,
                        source_tree=source_tree,
                        official_records=official_records,
                        qcl_first_batch=batch_payload,
                    )

            chain = validate_batch_chain(arm_records, family=cell.writer_family)
            arm_terminal = {
                "schema": ARM_SCHEMA,
                "instruction_id": INSTRUCTION_ID,
                "status": "SEQUENTIAL_ARM_TERMINAL_VALID",
                "cell_id": cell_id,
                "model_alias": cell.model_alias,
                "writer_family": cell.writer_family,
                "arm": arm,
                "chain": chain,
                "journal_count": len(arm_journals),
                "journals": arm_journals,
                "entry_weight_sha256": global_w0_sha,
                "terminal_weight_sha256": previous_commit,
                "entry_method_state_sha256": initial_method_identity,
                "terminal_method_state_sha256": previous_method_state,
                "alpha_history_terminal_width": logical_history_width
                if cell.writer_family == "AlphaEdit"
                else 0,
                "scientific_promotion": False,
            }
            arm_terminal["identity_sha256"] = canonical_hash(arm_terminal)
            arm_path = root / f"arm-{arm}" / "terminal-receipt.json"
            arm_sha = _create_once_json(arm_path, arm_terminal)
            arm_receipts.append(
                {
                    "arm": arm,
                    "relative_path": str(arm_path.relative_to(root)),
                    "file_sha256": arm_sha,
                    "identity_sha256": arm_terminal["identity_sha256"],
                }
            )

        if first_gate is None:
            raise TechnicalBoundary("cell first-valid gate was not published")
        _restore_selected(parameters, global_w0)
        _restore_method_state(
            cell.writer_family,
            module,
            initial_alpha_cache,
            alpha_cache_new=initial_alpha_cache_new,
            memit_versions=memit_versions,
        )
        if (
            tensor_set_sha256(parameters) != global_w0_sha
            or {name: int(value.data_ptr()) for name, value in parameters.items()}
            != global_pointers
        ):
            raise TechnicalBoundary("terminal global W0 pointer/bytes restore differs")
        terminal_method_identity = setup.method_state_identity()
        if terminal_method_identity != initial_method_identity:
            raise TechnicalBoundary("terminal cold method-state restore differs")

        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "instruction_id": INSTRUCTION_ID,
            "nonce": NONCE,
            "status": "SEQUENTIAL_CELL_TERMINAL_VALID",
            "cell_id": cell_id,
            "model_alias": cell.model_alias,
            "writer_family": cell.writer_family,
            "run_token": run_token,
            "source": {
                "head": source_head,
                "tree": source_tree,
                "tracked_clean": True,
            },
            "easyedit": easyedit,
            "hparams_path": str(hparams_path),
            "stream_root": STREAM_ROOT,
            "order_root": ORDER_ROOT,
            "runtime_lock": SequentialRuntimeLock().payload(),
            "orbode_numerical_lock": asdict(ORBODERuntimeLock()),
            "arms": list(ARM_ORDER),
            "derived_arm": DERIVED_ARM,
            "batch_count_per_arm": 10,
            "request_count_per_arm": 1_000,
            "primary_endpoint_count": 5_000,
            "fixed_z_compute_count": 5_000,
            "fixed_z_recompute_count": 0,
            "arm_receipts": arm_receipts,
            "first_valid_gate": dict(first_gate),
            "full_fp32": True,
            "dtype_receipt": dtype_receipt,
            "attention_backend_receipt": attention_receipt,
            "parameter_inventory": parameter_inventory,
            "model_load_count": 1,
            "model_load_seconds": load_seconds,
            "model_forward_invocation_count": _model_forward_count(model),
            "terminal_w0_pointer_bytes_restore_pass": True,
            "terminal_method_state_restore_pass": True,
            "inter_arm_state_carry_count": 0,
            "inter_batch_weight_link_count": 45,
            "inter_batch_method_state_link_count": 45,
            "technical_failure_count": 0,
            "scientific_failure_count": 0,
            "nonfinite_count": 0,
            "imputation_count": 0,
            "retry_count": 0,
            "scientific_promotion": False,
            "memory": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            },
            "wall_seconds": time.perf_counter() - started,
        }
        result["identity_sha256"] = canonical_hash(result)
        result_sha = _create_once_json(root / "result.json", result)
        terminal = {
            "schema": "orbode.server4.sequential-terminal-receipt.v1",
            "instruction_id": INSTRUCTION_ID,
            "status": "SEQUENTIAL_CELL_TERMINAL_VALID",
            "cell_id": cell_id,
            "model_alias": cell.model_alias,
            "writer_family": cell.writer_family,
            "source_head": source_head,
            "source_tree": source_tree,
            "result_sha256": result_sha,
            "result_identity_sha256": result["identity_sha256"],
            "arm_count": 5,
            "batch_count": 50,
            "request_count": 5_000,
            "terminal_w0_pointer_bytes_restore_pass": True,
            "terminal_method_state_restore_pass": True,
            "scientific_promotion": False,
        }
        terminal["identity_sha256"] = canonical_hash(terminal)
        _create_once_json(root / "terminal-receipt.json", terminal)
        if forward_handle is not None:
            forward_handle.remove()
            forward_handle = None
        return terminal
    except BaseException as exc:
        restored = False
        try:
            if active_family is not None:
                active_family.reset_entry()
                restored = True
        except BaseException:
            restored = False
        failure = {
            "schema": FAILURE_SCHEMA,
            "instruction_id": INSTRUCTION_ID,
            "status": "TECHNICAL_OR_NUMERICAL_BOUNDARY_HOLD",
            "cell_id": cell_id,
            "source_head": source_head,
            "source_tree": source_tree,
            "run_token": run_token,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-30:],
            "progress": progress,
            "completed_batch_journal_count": len(completed),
            "failed_batch_entry_restore_pass": restored,
            "partial_scientific_denominator_in_final_aggregate": 0,
            "imputation_count": 0,
            "science_change_count": 0,
            "tolerance_change_count": 0,
            "threshold_change_count": 0,
            "scientific_promotion": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        failure["identity_sha256"] = canonical_hash(failure)
        try:
            if not failure_path.exists() and not failure_path.is_symlink():
                _create_once_json(failure_path, failure)
        except BaseException:
            pass
        if forward_handle is not None:
            forward_handle.remove()
        raise


__all__ = ["run_cell"]
