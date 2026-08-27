"""BGODE-R3 G1 natural multi-token node-0 and one-step GPU gate."""

from __future__ import annotations

import hashlib
import io
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from project.run_scripts.barrier_guided_ode.r2.actuator_guard import propose_validated_ordered
from project.run_scripts.barrier_guided_ode.r2.stage_b_contract import seal_model_vocabulary
from project.run_scripts.barrier_guided_ode.r2.stage_b_probe import _assert_w0, _native_fidelity
from project.run_scripts.barrier_guided_ode.s1_alphaedit_runtime import (
    AtomicWeightTrajectory,
    BGODETargetAuthority,
)
from project.run_scripts.barrier_guided_ode.s1_experiment import (
    LAYERS,
    RUN_SEED,
    S1_CONTEXT_SEED,
    _assert_full_fp32,
    _load_alpha_hparams,
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
from project.run_scripts.ode_edit_motivation.contracts import canonical_json, sha256_bytes
from project.run_scripts.ode_edit_motivation.direct_z import DirectZCache
from project.run_scripts.ode_edit_motivation.easyedit_bridge import EasyEditBridge
from project.run_scripts.ode_edit_motivation.gpu_runtime import (
    load_fixed_model,
    offline_environment,
    seed_runtime,
)
from project.run_scripts.ode_edit_motivation.hooks import resolve_parameter, tensor_sha256
from project.run_scripts.ode_edit_motivation.manifests import fixed_model_spec, preflight_fixed_artifacts
from project.run_scripts.ode_edit_motivation.mv0_fidelity import _freeze_contexts
from project.run_scripts.ode_edit_motivation.mv1_calibration import build_exact_teacher_batch
from project.run_scripts.ode_edit_motivation.projector_adapter import AlphaEditProjectorBank
from project.run_scripts.ode_edit_motivation.frozen_target_lineage import proposal_direction_hash

from .actuators import NodeBuildLedger, NormalizedActuatorBasis
from .errors import EqualityInfeasible, NumericalBoundary, R3ScientificBoundary
from .events import FineEventEvaluation, FineEventLayout, evaluate_fine_events
from .g1_jvp import R3SerialForwardJVPBackend
from .moments import aggregate_fine_event_moments, differential_identity
from .natural import load_natural_record, natural_case, tokenize_record
from .reference import W0ConditionalSeal, barrier_decomposition
from .solver import ControllerArm, barrier_attribution, solve_factor_space
from .telemetry import R3ExecutionBoundary


INSTRUCTION_ID = "ODEEDIT-S05-BGODE-R3-FINE-EVENT-TARGET-EXCLUDED-FACTOR-SPACE"
SCHEMA = "ode-edit-bgode-r3-g1-natural-multitoken-actuator/v1"


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _write_once(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise R3ScientificBoundary(f"existing private artifact differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)


def _seal_q0(output: Path, seal: W0ConditionalSeal) -> dict[str, object]:
    payload = {
        "initial_log_probabilities": seal.initial_log_probabilities,
        "layout_identity": seal.layout_identity,
        "log_q0": seal.log_q0,
        "non_target_indices": torch.tensor(seal.non_target_indices, dtype=torch.int64),
        "target_index": seal.target_index,
        "tensor_identity": seal.tensor_identity,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    path = output / "private/w0-q0.pt"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _write_once(path, data)
    return {
        "bytes": len(data),
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "shape": list(seal.log_q0.shape),
        "tensor_identity": seal.tensor_identity,
    }


def _event_payload(value: FineEventEvaluation) -> dict[str, object]:
    target = value.target_log_probability
    source = value.source_log_probability
    non_target = torch.logsumexp(
        value.log_probabilities[
            [index for index in range(value.layout.event_count) if index not in value.layout.target_event_indices]
        ],
        dim=0,
    )
    return {
        "event_count": value.layout.event_count,
        "layout_identity": value.layout.identity,
        "normalization_log_residual": value.normalization_log_residual,
        "normalization_tolerance": value.normalization_tolerance,
        "reducer": value.reducer,
        "score_centering_norm": value.score_centering_norm,
        "source_log_probability": float(source),
        "source_probability": float(source.exp()),
        "target_log_probability": float(target),
        "target_logit": float(target - non_target),
        "target_probability": float(target.exp()),
        "topology": value.layout.topology,
    }


def _solution_payload(value: Any) -> dict[str, object]:
    return {
        "receipt": value.receipt.payload(),
        "retained_direction_count": int(value.retained_directions.shape[1]),
        "velocity": [float(item) for item in value.velocity.cpu().tolist()],
        "velocity_sha256": _tensor_sha(value.velocity),
    }


def _t0_identity(fisher: Any, full: Any) -> dict[str, object]:
    difference = float(torch.linalg.vector_norm(fisher.velocity - full.velocity).cpu().item())
    scale = 1.0 + float(torch.linalg.vector_norm(fisher.velocity).cpu().item())
    tolerance = 512.0 * torch.finfo(torch.float64).eps * scale
    passed = math.isfinite(difference) and difference <= tolerance
    if not passed:
        raise NumericalBoundary(
            "t=0 Full/Fisher identity failed",
            receipt={"difference_norm": difference, "tolerance": tolerance},
        )
    return {"difference_norm": difference, "tolerance": tolerance, "pass": True}


def run_g1(
    *,
    easyedit_root: str | Path,
    output_root: str | Path,
    source_head: str,
    source_tree: str,
    release_receipt: Mapping[str, Any],
    model_alias: str,
    run_id: str,
) -> dict[str, Any]:
    expected = f"s05-bgode-r3-g1-{model_alias}-natural-unequal-nonprefix-multitoken-v1"
    if run_id != expected:
        raise R3ScientificBoundary("G1 run identity differs")
    output = Path(output_root).expanduser().resolve(strict=False) / run_id
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.perf_counter()
    case = natural_case(model_alias, "unequal-non-prefix")
    raw_record, request, target_true = load_natural_record(case)
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    spec = fixed_model_spec(model_alias)
    fixed = preflight_fixed_artifacts(root, model_alias=model_alias)
    bridge = EasyEditBridge(root, expected_files=s1_easyedit_pins(), include_alphaedit_reference=True)
    bridge_manifest = bridge.preflight()
    with offline_environment():
        seed_receipt = seed_runtime(RUN_SEED)
        bindings = bridge.load()
        runtime = load_fixed_model(model_alias)
        torch.cuda.synchronize(0)
        dtype_receipt = _assert_full_fp32(runtime)
        tokenization = tokenize_record(runtime.tokenizer, raw_record, model_alias=model_alias)
        if tokenization.identity != case["tokenization_identity"]:
            raise R3ScientificBoundary("runtime natural tokenization differs from presealed manifest")
        if request.to_easyedit()["target_new"] != tokenization.target_normalized:
            raise R3ScientificBoundary("event/compute_z/writer target bytes differ")
        vocabulary = seal_model_vocabulary(runtime.model, tokenization)
        layout = FineEventLayout.build(
            source_tokens=tokenization.source_token_ids,
            target_tokens=tokenization.target_token_ids,
            output_vocabulary_size=vocabulary.output_head_vocabulary_size,
            tokenizer_vocabulary_size=vocabulary.tokenizer_vocabulary_size,
        )
        if layout.topology != "unequal-non-prefix":
            raise R3ScientificBoundary("runtime natural topology differs")
        hparams = _load_alpha_hparams(root, bindings, spec, model_alias)
        contexts = _freeze_contexts(bridge, runtime, seed=S1_CONTEXT_SEED)
        projector = AlphaEditProjectorBank.open(root, spec)
        alpha_config, alpha_reference = load_alphaedit_solver_config(
            root, model_alias, memit_hparams=hparams
        )
        weight_names = tuple(f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in LAYERS)
        origin = _snapshot(
            runtime=runtime,
            request=request,
            contexts=contexts,
            hparams=hparams,
            weight_names=weight_names,
            provenance_id=bindings.provenance.manifest_id,
        )
        pointers = {name: int(resolve_parameter(runtime.model, name).data_ptr()) for name in weight_names}
        hashes = {name: origin.parameter(name).sha256 for name in weight_names}
        teacher = build_exact_teacher_batch(runtime.tokenizer, request, contexts)
        if tuple(int(value) for value in teacher.target_ids.tolist()) != tokenization.target_token_ids:
            raise R3ScientificBoundary("compute_z teacher target IDs differ from event/writer target")
        from easyeditor.models.alphaedit import compute_z as alpha_compute_z

        fixed_target_calls = 0

        def compute_z_once() -> torch.Tensor:
            nonlocal fixed_target_calls
            fixed_target_calls += 1
            if fixed_target_calls != 1:
                raise R3ScientificBoundary("fixed native target was recomputed")
            value = alpha_compute_z.compute_z(
                runtime.model,
                runtime.tokenizer,
                request.to_easyedit(),
                hparams,
                LAYERS[-1],
                contexts.to_easyedit(),
            )
            return value.detach().cpu().float().reshape(-1, 1)

        direct_z = DirectZCache(output / "private", "direct-z.pt").load_or_compute(
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
            provenance_id=bindings.provenance.manifest_id,
            isolated_solve_backend="upstream-dense",
        )
        primal_backend = R3SerialForwardJVPBackend(runtime.model, tokenization)
        initial = evaluate_fine_events(layout, primal_backend.observe_primal(layout))
        q0 = W0ConditionalSeal.capture(initial)
        q0_artifact = _seal_q0(output, q0)
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
                parent=output / "private",
                accepted_z_source="BGODE_R3_FIXED_NATIVE_TARGET_W0",
            ) as (cache_template, accepted_target_receipt):
                official_apply, originals = run_official_native_apply(
                    runtime.model,
                    runtime.tokenizer,
                    (request_payload,),
                    hparams,
                    touched={name: resolve_parameter(runtime.model, name) for name in weight_names},
                    reset_cache=True,
                    cache_history_width=0,
                    cache_template=cache_template,
                    expected_native_compute_z_call_count=0,
                    accepted_z_source="BGODE_R3_FIXED_NATIVE_TARGET_W0",
                )
                native_event = evaluate_fine_events(layout, primal_backend.observe_primal(layout))
                official_weights = {
                    name: resolve_parameter(runtime.model, name).detach().clone()
                    for name in weight_names
                }
                with torch.no_grad():
                    for name in weight_names:
                        resolve_parameter(runtime.model, name).copy_(originals[name])
        if not alpha_state.get("restored"):
            raise R3ScientificBoundary("Official AlphaEdit module state did not restore")
        _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
        initial_payload = _event_payload(initial)
        native_payload = _event_payload(native_event)
        horizon = float(native_payload["target_logit"]) - float(initial_payload["target_logit"])
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise R3ScientificBoundary("NONPOSITIVE_NATIVE_TARGET_LOGIT_HORIZON")

        build_ledger = NodeBuildLedger()
        with _silence_upstream():
            validated = propose_validated_ordered(
                adapter,
                origin_lineage=authority,
                construction="genuine-p-inside-solve",
                solver_suffix=f"bgode-r3/g1/{model_alias}/node-0",
            )
        raw_proposal = validated.build.proposal
        fidelity = _native_fidelity(
            runtime=runtime,
            proposal=raw_proposal,
            official_weights=official_weights,
            entry_weights=originals,
            weight_names=weight_names,
        )
        basis = NormalizedActuatorBasis.from_factors(raw_proposal.factors)
        normalized_proposal = basis.normalized_proposal(raw_proposal)
        build_ledger.publish(basis, validation_call_count=validated.validation_call_count)
        _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
        try:
            observations = primal_backend.observe_all_prefix_fd(
                layout=layout,
                proposal=normalized_proposal,
            )
            node = evaluate_fine_events(layout, observations)
            moments = aggregate_fine_event_moments(node, seal=q0, reference_time=0.0)
            fisher = solve_factor_space(
                moments.factor,
                moments.objective_offset,
                moments.equality_directions,
                moments.equality_rates,
                arm=ControllerArm.FISHER,
            )
            full = solve_factor_space(
                moments.factor,
                moments.objective_offset,
                moments.equality_directions,
                moments.equality_rates,
                arm=ControllerArm.FULL,
            )
            t0_identity = _t0_identity(fisher, full)
            fisher_retained = primal_backend.validate_retained(fisher)
            full_retained = primal_backend.validate_retained(full)
            attribution = barrier_attribution(
                moments.factor,
                moments.objective_offset,
                fisher,
                full,
            )
            step_size = horizon / 32.0
            beta = fisher.coefficient(step_size)
            if basis.raw_coefficients_fp32(beta).shape != beta.shape:
                raise R3ScientificBoundary("normalized-to-raw coefficient binding differs")
            with AtomicWeightTrajectory(runtime.model, weight_names) as trajectory:
                action = trajectory.apply(normalized_proposal, beta.to(dtype=torch.float32))
                post = evaluate_fine_events(layout, primal_backend.observe_primal(layout))
                post_hashes = {
                    name: tensor_sha256(resolve_parameter(runtime.model, name)) for name in weight_names
                }
            _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
            mismatch = basis.ordered_prefix_mismatch(beta)
            node_barrier = barrier_decomposition(q0, node, time=0.0)
            post_barrier = barrier_decomposition(q0, post, time=step_size)
        finally:
            build_ledger.release_node()

        execution = R3ExecutionBoundary(
            batch_size=1,
            fixed_target_compute_count=fixed_target_calls,
            fixed_target_recompute_count=0,
            model_numeric_dtype="torch.float32",
            event_numeric_dtype="torch.float64",
            physical_write_dtype="torch.float32",
            node_history_append_count=0,
            terminal_history_append_count=0,
            localizer_call_count=0,
            endpoint_correction_count=0,
            accepted_step_search_count=0,
            alternate_solver_count=0,
            controller_evaluator_influence_count=0,
            retained_factor_count=build_ledger.retained_factor_count,
            scientific_promotion=False,
        )
        result: dict[str, Any] = {
            "accepted_target_receipt": accepted_target_receipt,
            "action": asdict(action),
            "attribution": asdict(attribution),
            "barrier_node0": asdict(node_barrier),
            "barrier_post_probe": asdict(post_barrier),
            "case": dict(case),
            "contexts": {"manifest_id": contexts.manifest_id, "seed": S1_CONTEXT_SEED},
            "dtype": dtype_receipt,
            "execution_boundary": execution.payload(),
            "fixed_inputs": {
                "alpha_config_id": alpha_config.config_id,
                "alpha_reference_manifest_id": alpha_reference.manifest_id,
                "bridge_manifest_id": bridge_manifest.manifest_id,
                "fixed_manifest_id": fixed.manifest_id,
                "projector": projector.metadata(),
            },
            "fixed_target": {
                "artifact_sha256": direct_z.artifact.sha256,
                "compute_count": fixed_target_calls,
                "recompute_count": 0,
                "tensor_sha256": direct_z.tensor_sha256,
            },
            "fisher": _solution_payload(fisher),
            "fisher_retained_fd": list(fisher_retained),
            "full": _solution_payload(full),
            "full_retained_fd": list(full_retained),
            "initial_event": initial_payload,
            "instruction_id": INSTRUCTION_ID,
            "jvp_compute": asdict(primal_backend.ledger),
            "model": runtime.metadata(),
            "native_adapter_fidelity": fidelity,
            "native_event": native_payload,
            "native_official_apply": official_apply,
            "native_target_logit_horizon": horizon,
            "node0_differential_identity": differential_identity(moments, fisher.velocity),
            "node0_event": _event_payload(node),
            "normalized_actuator": {
                "basis_count": len(basis.normalized_factors),
                "beta": [float(value) for value in beta.cpu().tolist()],
                "beta_norm": basis.block_frobenius(beta),
                "normalized_proposal_identity": proposal_direction_hash(normalized_proposal),
                "ordered_prefix_mismatch_beta_based": list(mismatch),
                "raw_coefficients": [
                    float(value) for value in basis.raw_coefficients_fp32(beta).cpu().tolist()
                ],
                "raw_factor_scales": [float(value) for value in basis.raw_scales.tolist()],
                "raw_factor_sha256": list(basis.raw_factor_sha256),
            },
            "ordered_dictionary": {
                **asdict(validated.dictionary),
                "build_ledger": asdict(build_ledger),
                "factor_sha256": list(validated.factor_sha256),
                "validation_call_count": validated.validation_call_count,
            },
            "post_probe_event": _event_payload(post),
            "post_probe_parameter_hashes": post_hashes,
            "prefix_fd": [asdict(value) for value in primal_backend.prefix_fd_receipts],
            "q0_artifact": q0_artifact,
            "release_receipt": dict(release_receipt),
            "run_id": run_id,
            "schema": SCHEMA,
            "scientific_promotion": False,
            "seed": seed_receipt,
            "source_head": source_head,
            "source_tree": source_tree,
            "status": "R3_G1_ACTUATOR_FEASIBLE_G2_HOLD",
            "step_size": step_size,
            "t0_full_fisher_identity": t0_identity,
            "tokenization": tokenization.receipt(),
            "vocabulary": vocabulary.receipt(),
            "w0_restore": {"bytes_exact": True, "pointer_exact": True},
        }
        result["total_wall_seconds"] = time.perf_counter() - started
        result["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
        result["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
        result["terminal_identity"] = sha256_bytes(canonical_json(result).encode("utf-8"))
        _write_json_once(output / "terminal.json", result)
        return result


__all__ = ["INSTRUCTION_ID", "SCHEMA", "run_g1"]
