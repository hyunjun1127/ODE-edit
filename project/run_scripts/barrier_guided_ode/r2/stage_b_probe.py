"""BGODE-R2 Stage-B one-request technical GPU closure."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from project.run_scripts.barrier_guided_ode.s1_alphaedit_runtime import (
    AtomicWeightTrajectory,
    BGODETargetAuthority,
    SerialForwardJVPBackend,
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
from project.run_scripts.ode_edit_motivation.manifests import (
    fixed_model_spec,
    preflight_fixed_artifacts,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import _freeze_contexts
from project.run_scripts.ode_edit_motivation.mv1_calibration import build_exact_teacher_batch
from project.run_scripts.ode_edit_motivation.projector_adapter import AlphaEditProjectorBank

from .actuator_guard import propose_validated_ordered
from .controller import solve_two_equality_rayleighian
from .errors import ActuatorEqualityInfeasible, NumericalImplementationBoundary, R2ScientificBoundary
from .events import PrefixEventLayout, evaluate_prefix_events
from .moments import aggregate_r2_event_moments
from .stage_b_contract import (
    load_sealed_s1_sample,
    model_binding,
    seal_model_vocabulary,
    seal_tokenization,
)


INSTRUCTION_ID = "ODEEDIT-S05-BGODE-R2-RHO-FREE-PREFIX-EVENT-FP64-TWO-EQUALITY"
SCHEMA = "ode-edit-bgode-r2-stage-b-technical-closure/v1"


def _event_payload(value: Any) -> dict[str, Any]:
    return {
        "event_count": len(value.layout.events),
        "prefix_relation": value.layout.prefix_relation,
        "target_log_probability": float(value.target_log_probability),
        "source_log_probability": float(value.source_log_probability),
        "target_probability": float(value.target_log_probability.exp()),
        "source_probability": float(value.source_log_probability.exp()),
        "pair_mass": float(value.pair_mass),
        "log_odds": float(value.log_odds),
        "normalization_log_residual": value.normalization_log_residual,
        "normalization_tolerance": value.normalization_tolerance,
        "score_centering_norm": value.score_centering_norm,
    }


def _assert_w0(
    runtime: Any,
    *,
    weight_names: tuple[str, ...],
    pointers: Mapping[str, int],
    hashes: Mapping[str, str],
) -> None:
    for name in weight_names:
        parameter = resolve_parameter(runtime.model, name)
        if int(parameter.data_ptr()) != pointers[name] or tensor_sha256(parameter) != hashes[name]:
            raise R2ScientificBoundary("Stage-B W0 pointer/bytes boundary failed")


def _native_fidelity(
    *,
    runtime: Any,
    proposal: Any,
    official_weights: Mapping[str, torch.Tensor],
    entry_weights: Mapping[str, torch.Tensor],
    weight_names: tuple[str, ...],
) -> dict[str, Any]:
    coefficients = torch.ones(len(proposal.factors), dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    numerator = 0.0
    denominator = 0.0
    with AtomicWeightTrajectory(runtime.model, weight_names) as trajectory:
        action = trajectory.apply(proposal, coefficients)
        for name in weight_names:
            candidate = resolve_parameter(runtime.model, name).detach().float()
            official = official_weights[name].detach().float()
            entry = entry_weights[name].detach().float()
            difference_energy = float(torch.sum((candidate - official).square()).item())
            official_energy = float(torch.sum((official - entry).square()).item())
            numerator += difference_energy
            denominator += official_energy
            rows.append(
                {
                    "weight_name": name,
                    "candidate_sha256": tensor_sha256(candidate),
                    "official_sha256": tensor_sha256(official),
                    "difference_frobenius": math.sqrt(difference_energy),
                    "official_delta_frobenius": math.sqrt(official_energy),
                }
            )
        relative = math.sqrt(numerator / max(denominator, torch.finfo(torch.float32).tiny))
        result = {
            "identity_scale": 1.0,
            "relative_frobenius": relative,
            "tolerance": 5.0e-5,
            "pass": math.isfinite(relative) and relative <= 5.0e-5,
            "technical_apply_count": action.assignment_count,
            "dynamic_writer_action_count": 0,
            "layers": rows,
        }
    if not result["pass"]:
        raise R2ScientificBoundary("ordered adapter differs from Official AlphaEdit endpoint")
    return result


def _ordered_prefix_metrics(proposal: Any, coefficients: torch.Tensor | None) -> dict[str, Any]:
    factor_norms: list[float] = []
    for factor in proposal.factors:
        left_gram = factor.left.float().T @ factor.left.float()
        right_gram = factor.right.float().T @ factor.right.float()
        value = float(torch.sqrt(torch.clamp_min(torch.sum(left_gram * right_gram), 0.0)).item())
        if not math.isfinite(value):
            raise R2ScientificBoundary("ordered factor norm is non-finite")
        factor_norms.append(value)
    rows: list[dict[str, Any]] = []
    for index, factor in enumerate(proposal.factors):
        unit = math.sqrt(sum(value * value for value in factor_norms[:index]))
        row: dict[str, Any] = {
            "weight_name": factor.weight_name,
            "factor_frobenius": factor_norms[index],
            "unit_prefix_frobenius": unit,
        }
        if coefficients is None:
            row.update(
                {
                    "effective_prefix_frobenius": "NOT_AVAILABLE_BEFORE_NUMERICAL_SOLUTION",
                    "unit_effective_mismatch_frobenius": "NOT_AVAILABLE_BEFORE_NUMERICAL_SOLUTION",
                }
            )
        else:
            values = [float(value) for value in coefficients.detach().cpu().tolist()]
            effective = math.sqrt(
                sum((values[position] * factor_norms[position]) ** 2 for position in range(index))
            )
            mismatch = math.sqrt(
                sum(((values[position] - 1.0) * factor_norms[position]) ** 2 for position in range(index))
            )
            row.update(
                {
                    "effective_prefix_frobenius": effective,
                    "unit_effective_mismatch_frobenius": mismatch,
                }
            )
        rows.append(row)
    return {
        "block_separated_parameter_geometry": True,
        "rows": rows,
    }


def run_stage_b(
    *,
    easyedit_root: str | Path,
    output_root: str | Path,
    source_head: str,
    source_tree: str,
    release_receipt: Mapping[str, Any],
    model_alias: str,
    run_id: str,
) -> dict[str, Any]:
    binding = model_binding(model_alias)
    if run_id != binding.run_id:
        raise R2ScientificBoundary("Stage-B run identity differs")
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve(strict=False) / run_id
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    started = time.perf_counter()
    sample = load_sealed_s1_sample()
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
        tokenization = seal_tokenization(runtime.tokenizer, sample, model_alias=model_alias)
        vocabulary = seal_model_vocabulary(runtime.model, tokenization)
        layout = PrefixEventLayout.build(
            source_tokens=tokenization.source_token_ids,
            target_tokens=tokenization.target_token_ids,
            output_vocabulary_size=vocabulary.output_head_vocabulary_size,
            tokenizer_vocabulary_size=vocabulary.tokenizer_vocabulary_size,
        )
        hparams = _load_alpha_hparams(root, bindings, spec, model_alias)
        contexts = _freeze_contexts(bridge, runtime, seed=S1_CONTEXT_SEED)
        projector = AlphaEditProjectorBank.open(root, spec)
        alpha_config, alpha_reference = load_alphaedit_solver_config(root, model_alias, memit_hparams=hparams)
        request = sample.edit_request
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
        target_ids = build_exact_teacher_batch(runtime.tokenizer, request, contexts).target_ids
        from easyeditor.models.alphaedit import compute_z as alpha_compute_z

        fixed_target_calls = 0

        def compute_z_once() -> torch.Tensor:
            nonlocal fixed_target_calls
            fixed_target_calls += 1
            if fixed_target_calls != 1:
                raise R2ScientificBoundary("fixed native target was recomputed")
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
            target_token_ids=target_ids,
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
            target_token_ids=target_ids,
            provenance_id=bindings.provenance.manifest_id,
            isolated_solve_backend="upstream-dense",
        )
        primal_backend = SerialForwardJVPBackend(runtime.model, tokenization)
        initial = evaluate_prefix_events(layout, primal_backend.observe_primal(layout))
        request_payload = {
            **request.to_easyedit(),
            "case_id": int(request.case_id),
            "request_sha256": sample.request_sha256,
        }
        official_weights: dict[str, torch.Tensor] = {}
        with isolated_alphaedit_module_state() as alpha_state:
            with accepted_z_cache_template(
                (request_payload,),
                direct_z.values,
                hparams,
                parent=output / "private",
                accepted_z_source="BGODE_R2_FIXED_NATIVE_TARGET_W0",
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
                    accepted_z_source="BGODE_R2_FIXED_NATIVE_TARGET_W0",
                )
                native_event = evaluate_prefix_events(layout, primal_backend.observe_primal(layout))
                official_weights = {
                    name: resolve_parameter(runtime.model, name).detach().clone() for name in weight_names
                }
                with torch.no_grad():
                    for name in weight_names:
                        resolve_parameter(runtime.model, name).copy_(originals[name])
        if not alpha_state.get("restored"):
            raise R2ScientificBoundary("Official AlphaEdit module state did not restore")
        _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)

        with _silence_upstream():
            validated = propose_validated_ordered(
                adapter,
                origin_lineage=authority,
                construction="genuine-p-inside-solve",
                solver_suffix=f"bgode-r2/stage-b/{model_alias}/node-0",
            )
        proposal = validated.build.proposal
        fidelity = _native_fidelity(
            runtime=runtime,
            proposal=proposal,
            official_weights=official_weights,
            entry_weights=originals,
            weight_names=weight_names,
        )
        _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
        observations = primal_backend.observe(
            layout=layout,
            proposal=proposal,
            validate_first_node_fd=True,
        )
        node = evaluate_prefix_events(layout, observations)
        moments = aggregate_r2_event_moments(node, initial=initial, reference_time=0.0)
        common = {
            "schema": SCHEMA,
            "instruction_id": INSTRUCTION_ID,
            "run_id": run_id,
            "source_head": source_head,
            "source_tree": source_tree,
            "release_receipt": dict(release_receipt),
            "sample": sample.receipt(),
            "sample_identity": sample.identity,
            "tokenization": tokenization.receipt(),
            "vocabulary": vocabulary.receipt(),
            "model": runtime.metadata(),
            "dtype": dtype_receipt,
            "seed": seed_receipt,
            "contexts": {"seed": S1_CONTEXT_SEED, "manifest_id": contexts.manifest_id},
            "fixed_inputs": {
                "fixed_manifest_id": fixed.manifest_id,
                "bridge_manifest_id": bridge_manifest.manifest_id,
                "alpha_reference_manifest_id": alpha_reference.manifest_id,
                "alpha_config_id": alpha_config.config_id,
                "projector": projector.metadata(),
            },
            "fixed_target": {
                "compute_count": fixed_target_calls,
                "recompute_count": 0,
                "tensor_sha256": direct_z.tensor_sha256,
                "artifact_sha256": direct_z.artifact.sha256,
            },
            "initial_event": _event_payload(initial),
            "native_event": _event_payload(native_event),
            "native_log_odds_horizon": float(native_event.log_odds - initial.log_odds),
            "official_apply": official_apply,
            "accepted_target_receipt": accepted_target_receipt,
            "ordered_dictionary": {
                **asdict(validated.dictionary),
                "validation_call_count": validated.validation_call_count,
                "factor_sha256": list(validated.factor_sha256),
                "fidelity_identity_scale_prefix_mismatch": _ordered_prefix_metrics(
                    proposal, torch.ones(len(proposal.factors), dtype=torch.float64)
                ),
            },
            "native_adapter_fidelity": fidelity,
            "fisher_node0_event": _event_payload(node),
            "fisher_node0_moments": {
                "fisher": moments.fisher.tolist(),
                "moving_gradient": moments.moving_gradient.tolist(),
                "progress_direction": moments.progress_direction.tolist(),
                "normalized_pair_mass_direction": moments.normalized_pair_mass_direction.tolist(),
                "equality_directions": moments.equality_directions.tolist(),
                "equality_rates": moments.equality_rates.tolist(),
                "score_expectation": moments.score_expectation.tolist(),
            },
            "jvp_compute": asdict(primal_backend.ledger),
            "finite_difference": (
                None
                if primal_backend.finite_difference_receipt is None
                else asdict(primal_backend.finite_difference_receipt)
            ),
            "dynamic_writer_action_count": 0,
            "node_history_append_count": 0,
            "fixed_target_recompute_count": 0,
            "controller_evaluator_influence_count": 0,
            "scalar_localizer_count": 0,
            "probability_floor_count": 0,
            "ridge_count": 0,
            "damping_count": 0,
            "fallback_count": 0,
            "scientific_promotion": False,
        }
        try:
            solution = solve_two_equality_rayleighian(
                moments.fisher,
                torch.zeros_like(moments.moving_gradient),
                moments.equality_directions,
                moments.equality_rates,
            )
        except NumericalImplementationBoundary as boundary:
            common.update(
                {
                    "status": "BGODE_R2_STAGE_B_NUMERICAL_IMPLEMENTATION_BOUNDARY",
                    "boundary": {"type": type(boundary).__name__, "message": str(boundary), "receipt": boundary.receipt},
                    "ordered_prefix_mismatch": _ordered_prefix_metrics(proposal, None),
                }
            )
        except ActuatorEqualityInfeasible as boundary:
            common.update(
                {
                    "status": "BGODE_R2_STAGE_B_ACTUATOR_EQUALITY_INFEASIBLE",
                    "boundary": {"type": type(boundary).__name__, "message": str(boundary), "receipt": boundary.receipt},
                    "ordered_prefix_mismatch": _ordered_prefix_metrics(proposal, None),
                }
            )
        else:
            common.update(
                {
                    "status": "BGODE_R2_STAGE_B_FP64_FISHER_NODE0_PASS",
                    "fisher_node0_solution": {
                        "velocity": solution.velocity.tolist(),
                        "receipt": solution.receipt.as_payload(),
                        "physical_coefficient_dtype": str(solution.physical_coefficient_fp32(0.25).dtype),
                    },
                    "ordered_prefix_mismatch": _ordered_prefix_metrics(proposal, solution.velocity),
                }
            )
        _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
        common["w0_restore"] = {"pointer_exact": True, "bytes_exact": True}
        common["total_wall_seconds"] = time.perf_counter() - started
        common["peak_gpu_allocated_bytes"] = int(torch.cuda.max_memory_allocated(0))
        common["peak_gpu_reserved_bytes"] = int(torch.cuda.max_memory_reserved(0))
        common["terminal_identity"] = sha256_bytes(canonical_json(common).encode("utf-8"))
        _write_json_once(output / "terminal.json", common)
        return common


__all__ = ["INSTRUCTION_ID", "SCHEMA", "run_stage_b"]
