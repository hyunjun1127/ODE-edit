"""Minimal matched-model fixed-basis barrier ODE experiment."""

from __future__ import annotations

import json
import math
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from project.run_scripts.barrier_guided_ode.r2.actuator_guard import propose_validated_ordered
from project.run_scripts.barrier_guided_ode.r2.stage_b_contract import seal_model_vocabulary
from project.run_scripts.barrier_guided_ode.r2.stage_b_probe import _assert_w0, _native_fidelity
from project.run_scripts.barrier_guided_ode.r3.events import FineEventLayout, evaluate_fine_events
from project.run_scripts.barrier_guided_ode.r3.g1_jvp import R3SerialForwardJVPBackend
from project.run_scripts.barrier_guided_ode.r3.g1_probe import _event_payload, _jsonable
from project.run_scripts.barrier_guided_ode.r3.natural import canonical_records, tokenize_record
from project.run_scripts.barrier_guided_ode.s1_alphaedit_runtime import AtomicWeightTrajectory, BGODETargetAuthority
from project.run_scripts.barrier_guided_ode.s1_experiment import (
    LAYERS,
    RUN_SEED,
    S1_CONTEXT_SEED,
    _assert_full_fp32,
    _load_alpha_hparams,
    _metrics_payload,
    _silence_upstream,
    _snapshot,
    _write_json_once,
    s1_easyedit_pins,
)
from project.run_scripts.ode_bf.p1r52_target_official_alphaedit_writer import (
    accepted_z_cache_template,
    isolated_alphaedit_module_state,
)
from project.run_scripts.ode_bf.scalable_batched_native import run_official_native_apply
from project.run_scripts.ode_edit_motivation.alphaedit_proposal_adapter import AlphaEditProposalAdapter
from project.run_scripts.ode_edit_motivation.alphaedit_reference import load_alphaedit_solver_config
from project.run_scripts.ode_edit_motivation.contracts import EditRequest, canonical_json, sha256_bytes
from project.run_scripts.ode_edit_motivation.direct_z import DirectZCache
from project.run_scripts.ode_edit_motivation.direct_z_alpha_possibility import _capture_teacher
from project.run_scripts.ode_edit_motivation.direct_z_possibility import (
    _observe_branch,
    _teacher_forced_prompts,
    load_heldout_evaluation_case,
)
from project.run_scripts.ode_edit_motivation.easyedit_bridge import EasyEditBridge
from project.run_scripts.ode_edit_motivation.gpu_runtime import load_fixed_model, offline_environment, seed_runtime
from project.run_scripts.ode_edit_motivation.hooks import resolve_parameter, tensor_sha256
from project.run_scripts.ode_edit_motivation.manifests import fixed_model_spec, preflight_fixed_artifacts
from project.run_scripts.ode_edit_motivation.mv0_fidelity import _freeze_contexts
from project.run_scripts.ode_edit_motivation.mv1_calibration import build_exact_teacher_batch
from project.run_scripts.ode_edit_motivation.projector_adapter import AlphaEditProjectorBank

from .alphaedit_basis import FixedAlphaEditBasis
from .euler_writer import run_barrier_euler
from .event_distribution import TargetExcludedReference
from .functional_observer import FixedBasisObserver


INSTRUCTION_ID = "ODEEDIT-S05-BGODE-FBP-MINIMAL-FIXED-BASIS-BARRIER-ODE-V1"
SCHEMA = "ode-edit-bgode-fbp-minimal-fixed-basis-barrier-ode/v1"
ARM_ORDER = ("NATIVE_ALPHAEDIT_N1_T1", "STATIC_SPLIT_OFF_N4", "BARRIER_ODE_N2", "BARRIER_ODE_N4")
COHORT_PATH = Path(__file__).with_name("cohort-lock.json")


class ExperimentBoundary(RuntimeError):
    """The locked minimal experiment cannot continue."""


def expected_run_id(model_alias: str) -> str:
    if model_alias not in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        raise ExperimentBoundary("unsupported model alias")
    return f"s05-bgode-fbp-minimal-{model_alias}-8case-four-arm-v1"


def _cohort(model_alias: str) -> tuple[Mapping[str, Any], ...]:
    value = json.loads(COHORT_PATH.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "identity"}
    if (
        value.get("schema") != "ode-edit-fixed-basis-barrier-ode-cohort/v1"
        or value.get("case_count") != 8
        or sha256_bytes(canonical_json(body).encode("utf-8")) != value.get("identity")
    ):
        raise ExperimentBoundary("cohort lock identity differs")
    for row in value["cases"]:
        if model_alias not in row["tokenization"]:
            raise ExperimentBoundary("cohort lacks model tokenization")
    return tuple(value["cases"])


def _request(row: Mapping[str, Any]) -> tuple[Any, EditRequest, str]:
    records = canonical_records()
    ordinal = row.get("ordinal")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0 or ordinal >= len(records):
        raise ExperimentBoundary("cohort ordinal is invalid")
    record = records[ordinal]
    if record.case_id != str(row["case_id"]) or record.raw_sha256 != row["raw_sha256"] or record.request_sha256 != row["request_sha256"]:
        raise ExperimentBoundary("cohort request bytes differ")
    rewrite = record.row["requested_rewrite"]
    request = EditRequest.from_mapping(
        {
            "case_id": record.case_id,
            "prompt": rewrite["prompt"],
            "subject": rewrite["subject"],
            "target_new": {"str": rewrite["target_new"]["str"]},
        }
    )
    return record, request, str(rewrite["target_true"]["str"]).strip()


def _true_metrics(*, runtime: Any, bindings: Any, hparams: Any, request: Any, heldout: Any, target_true: str) -> dict[str, Any]:
    prompt = request.prompt.format(request.subject)
    rewrite = _teacher_forced_prompts(
        runtime=runtime,
        bindings=bindings,
        hparams=hparams,
        prompts=(prompt,),
        target_text=target_true,
        subject=request.subject,
    )
    rephrase = _teacher_forced_prompts(
        runtime=runtime,
        bindings=bindings,
        hparams=hparams,
        prompts=heldout.paraphrase_prompts,
        target_text=target_true,
        subject=request.subject,
    )
    return {"rewrite_target_true": rewrite.compact(), "rephrase_target_true": rephrase.compact()}


def _run_case(
    *,
    output: Path,
    root: Path,
    runtime: Any,
    bindings: Any,
    bridge: Any,
    contexts: Any,
    projector: Any,
    hparams: Any,
    alpha_config: Any,
    provenance_id: str,
    row: Mapping[str, Any],
    model_alias: str,
    run_native_fidelity: bool,
) -> dict[str, Any]:
    case_started = time.perf_counter()
    raw_record, request, target_true = _request(row)
    tokenization = tokenize_record(runtime.tokenizer, raw_record, model_alias=model_alias)
    locked = row["tokenization"][model_alias]
    if tokenization.identity != locked["identity"] or tokenization.topology != "unequal-non-prefix":
        raise ExperimentBoundary("runtime tokenization differs from outcome-free cohort lock")
    vocabulary = seal_model_vocabulary(runtime.model, tokenization)
    layout = FineEventLayout.build(
        source_tokens=tokenization.source_token_ids,
        target_tokens=tokenization.target_token_ids,
        output_vocabulary_size=vocabulary.output_head_vocabulary_size,
        tokenizer_vocabulary_size=vocabulary.tokenizer_vocabulary_size,
    )
    weight_names = tuple(f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in LAYERS)
    origin = _snapshot(
        runtime=runtime,
        request=request,
        contexts=contexts,
        hparams=hparams,
        weight_names=weight_names,
        provenance_id=provenance_id,
    )
    pointers = {name: int(resolve_parameter(runtime.model, name).data_ptr()) for name in weight_names}
    hashes = {name: origin.parameter(name).sha256 for name in weight_names}
    teacher = build_exact_teacher_batch(runtime.tokenizer, request, contexts)
    if tuple(int(value) for value in teacher.target_ids.tolist()) != tokenization.target_token_ids:
        raise ExperimentBoundary("event/compute-z/writer target tokens differ")
    from easyeditor.models.alphaedit import compute_z as alpha_compute_z

    z_calls = 0

    def compute_z_once() -> torch.Tensor:
        nonlocal z_calls
        z_calls += 1
        if z_calls != 1:
            raise ExperimentBoundary("native target was recomputed")
        value = alpha_compute_z.compute_z(
            runtime.model,
            runtime.tokenizer,
            request.to_easyedit(),
            hparams,
            LAYERS[-1],
            contexts.to_easyedit(),
        )
        return value.detach().cpu().float().reshape(-1, 1)

    private = output / "private" / f"case-{raw_record.ordinal:04d}"
    direct_z = DirectZCache(private, "direct-z.pt").load_or_compute(
        source_snapshot=origin,
        z_layer=LAYERS[-1],
        compute=compute_z_once,
    )
    authority = BGODETargetAuthority.start(
        direct_z=direct_z,
        origin_snapshot=origin,
        target_token_ids=teacher.target_ids,
    )
    adapter = AlphaEditProposalAdapter(
        bridge=bridge,
        config=alpha_config,
        projector_bank=projector,
        model=runtime.model,
        tokenizer=runtime.tokenizer,
        request=request,
        memit_hparams=hparams,
        contexts=contexts,
        model_id=runtime.spec.snapshot_name,
        direct_z=direct_z,
        target_token_ids=teacher.target_ids,
        provenance_id=provenance_id,
        isolated_solve_backend="upstream-dense",
    )
    heldout = load_heldout_evaluation_case(root, request)
    baseline = _observe_branch(
        runtime=runtime,
        bindings=bindings,
        hparams=hparams,
        contexts=contexts,
        request=request,
        heldout=heldout,
    )
    base_teacher = _capture_teacher(
        runtime=runtime,
        bindings=bindings,
        hparams=hparams,
        contexts=contexts,
        request=request,
    )
    z_delta = direct_z.values[:, 0].float() - base_teacher.subject_states[0]
    z_branch = _observe_branch(
        runtime=runtime,
        bindings=bindings,
        hparams=hparams,
        contexts=contexts,
        request=request,
        heldout=heldout,
        patch_delta=z_delta,
    )

    def endpoint() -> dict[str, Any]:
        branch = _observe_branch(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            contexts=contexts,
            request=request,
            heldout=heldout,
        )
        return {
            **_metrics_payload(branch, baseline),
            **_true_metrics(
                runtime=runtime,
                bindings=bindings,
                hparams=hparams,
                request=request,
                heldout=heldout,
                target_true=target_true,
            ),
        }

    primal = R3SerialForwardJVPBackend(runtime.model, tokenization)
    initial_event = evaluate_fine_events(layout, primal.observe_primal(layout))
    request_payload = {
        **request.to_easyedit(),
        "case_id": int(request.case_id) if request.case_id.isdigit() else request.case_id,
        "request_sha256": raw_record.request_sha256,
    }
    with isolated_alphaedit_module_state() as alpha_state:
        with accepted_z_cache_template(
            (request_payload,),
            direct_z.values,
            hparams,
            parent=private,
            accepted_z_source="FIXED_BASIS_NATIVE_TARGET_W0",
        ) as (cache_template, target_receipt):
            native_apply, originals = run_official_native_apply(
                runtime.model,
                runtime.tokenizer,
                (request_payload,),
                hparams,
                touched={name: resolve_parameter(runtime.model, name) for name in weight_names},
                reset_cache=True,
                cache_history_width=0,
                cache_template=cache_template,
                expected_native_compute_z_call_count=0,
                accepted_z_source="FIXED_BASIS_NATIVE_TARGET_W0",
            )
            native_metrics = endpoint()
            native_event = evaluate_fine_events(layout, primal.observe_primal(layout))
            official_weights = {name: resolve_parameter(runtime.model, name).detach().clone() for name in weight_names}
            with torch.no_grad():
                for name in weight_names:
                    resolve_parameter(runtime.model, name).copy_(originals[name])
    if not alpha_state.get("restored"):
        raise ExperimentBoundary("Official AlphaEdit module state did not restore")
    _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
    with _silence_upstream():
        validated = propose_validated_ordered(
            adapter,
            origin_lineage=authority,
            construction="genuine-p-inside-solve",
            solver_suffix=f"fixed-basis/{model_alias}/case-{raw_record.ordinal}",
        )
    basis = FixedAlphaEditBasis.capture(
        validated.build.proposal,
        factor_sha256=validated.factor_sha256,
    )
    fidelity = None
    if run_native_fidelity:
        fidelity = _native_fidelity(
            runtime=runtime,
            proposal=basis.raw_proposal,
            official_weights=official_weights,
            entry_weights=originals,
            weight_names=weight_names,
        )
        _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
    observer = FixedBasisObserver(runtime.model, tokenization, basis.normalized_proposal)
    q0_event = evaluate_fine_events(layout, observer.observe(layout, torch.zeros(5, dtype=torch.float64), with_jvp=False))
    reference = TargetExcludedReference.capture(q0_event)

    arms: dict[str, Any] = {
        ARM_ORDER[0]: {
            "endpoint": native_metrics,
            "event": reference.observe(native_event),
            "native_apply": native_apply,
            "authoritative_terminal_write_count": 1,
            "theta": [float(value) for value in basis.theta_ae.tolist()],
        }
    }

    def materialize(name: str, theta: torch.Tensor, nodes: tuple[dict[str, object], ...]) -> None:
        arm_started = time.perf_counter()
        coefficients = basis.raw_coefficients(theta)
        with AtomicWeightTrajectory(runtime.model, weight_names) as trajectory:
            action = trajectory.apply(basis.raw_proposal, coefficients)
            arm_metrics = endpoint()
            arm_event = evaluate_fine_events(layout, primal.observe_primal(layout))
        _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
        arms[name] = {
            "endpoint": arm_metrics,
            "event": reference.observe(arm_event),
            "nodes": list(nodes),
            "theta": [float(value) for value in theta.tolist()],
            "theta_norm": float(torch.linalg.vector_norm(theta)),
            "raw_coefficients": [float(value) for value in coefficients.tolist()],
            "action": asdict(action),
            "authoritative_terminal_write_count": 1,
            "node_physical_write_count": 0,
            "wall_seconds": time.perf_counter() - arm_started,
        }

    static = torch.zeros(5, dtype=torch.float64)
    for _ in range(4):
        static = static + 0.25 * basis.theta_ae
    materialize(ARM_ORDER[1], static, ())
    for name, steps in ((ARM_ORDER[2], 2), (ARM_ORDER[3], 4)):
        trajectory = run_barrier_euler(
            theta_ae=basis.theta_ae,
            reference=reference,
            observe=lambda theta: evaluate_fine_events(layout, observer.observe(layout, theta, with_jvp=True)),
            steps=steps,
        )
        materialize(name, trajectory.theta, trajectory.nodes)
    if z_calls != 1:
        raise ExperimentBoundary("fixed native target compute count differs")
    return _jsonable(
        {
            "ordinal": raw_record.ordinal,
            "case_id": raw_record.case_id,
            "request_sha256": raw_record.request_sha256,
            "raw_sha256": raw_record.raw_sha256,
            "tokenization": tokenization.receipt(),
            "vocabulary": vocabulary.receipt(),
            "initial_event": _event_payload(initial_event),
            "accepted_z": {
                **_metrics_payload(z_branch, baseline),
                "tensor_sha256": direct_z.tensor_sha256,
                "compute_count": z_calls,
                "recompute_count": 0,
            },
            "basis": basis.receipt(),
            "ordered_dictionary": {
                "validation_call_count": validated.validation_call_count,
                "dictionary": asdict(validated.dictionary),
            },
            "native_adapter_fidelity": fidelity,
            "accepted_target_receipt": target_receipt,
            "arms": arms,
            "compute": asdict(observer.ledger),
            "w0_restore": {"pointer_exact": True, "bytes_exact": True},
            "case_wall_seconds": time.perf_counter() - case_started,
        }
    )


def run_experiment(
    *,
    easyedit_root: str | Path,
    output_root: str | Path,
    source_head: str,
    source_tree: str,
    model_alias: str,
    run_id: str,
) -> dict[str, Any]:
    if run_id != expected_run_id(model_alias):
        raise ExperimentBoundary("run ID differs from locked model namespace")
    output = Path(output_root).expanduser().resolve(strict=False) / run_id
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    fixed = preflight_fixed_artifacts(root, model_alias=model_alias)
    bridge = EasyEditBridge(root, expected_files=s1_easyedit_pins(), include_alphaedit_reference=True)
    bridge_manifest = bridge.preflight()
    started = time.perf_counter()
    with offline_environment():
        seed = seed_runtime(RUN_SEED)
        bindings = bridge.load()
        model_started = time.perf_counter()
        runtime = load_fixed_model(model_alias)
        torch.cuda.synchronize(0)
        model_load_seconds = time.perf_counter() - model_started
        dtype = _assert_full_fp32(runtime)
        spec = fixed_model_spec(model_alias)
        hparams = _load_alpha_hparams(root, bindings, spec, model_alias)
        contexts = _freeze_contexts(bridge, runtime, seed=S1_CONTEXT_SEED)
        projector = AlphaEditProjectorBank.open(root, spec)
        alpha_config, alpha_reference = load_alphaedit_solver_config(root, model_alias, memit_hparams=hparams)
        rows: list[dict[str, Any]] = []
        for index, cohort_row in enumerate(_cohort(model_alias)):
            row = _run_case(
                output=output,
                root=root,
                runtime=runtime,
                bindings=bindings,
                bridge=bridge,
                contexts=contexts,
                projector=projector,
                hparams=hparams,
                alpha_config=alpha_config,
                provenance_id=bindings.provenance.manifest_id,
                row=cohort_row,
                model_alias=model_alias,
                run_native_fidelity=index == 0,
            )
            _write_json_once(output / "cases" / f"case-{row['ordinal']:04d}.json", row)
            rows.append(row)
        terminal = _jsonable(
            {
                "instruction_id": INSTRUCTION_ID,
                "schema": SCHEMA,
                "run_id": run_id,
                "source_head": source_head,
                "source_tree": source_tree,
                "model_alias": model_alias,
                "model": runtime.metadata(),
                "dtype": dtype,
                "seed": seed,
                "cohort_identity": json.loads(COHORT_PATH.read_text(encoding="utf-8"))["identity"],
                "case_count": len(rows),
                "arm_order": list(ARM_ORDER),
                "cases": rows,
                "fixed_inputs": {
                    "fixed_manifest_id": fixed.manifest_id,
                    "bridge_manifest_id": bridge_manifest.manifest_id,
                    "alpha_config_id": alpha_config.config_id,
                    "alpha_reference_manifest_id": alpha_reference.manifest_id,
                    "projector": projector.metadata(),
                },
                "model_load_seconds": model_load_seconds,
                "total_wall_seconds": time.perf_counter() - started,
                "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "node_factor_rebuild_count": 0,
                "node_physical_write_count": 0,
                "terminal_write_count_per_non_native_arm": 1,
                "scientific_promotion": False,
                "status": "TERMINAL_VALID",
            }
        )
        terminal["terminal_identity"] = sha256_bytes(canonical_json(terminal).encode("utf-8"))
        _write_json_once(output / "terminal.json", terminal)
        return terminal


__all__ = ["ARM_ORDER", "INSTRUCTION_ID", "SCHEMA", "expected_run_id", "run_experiment"]
