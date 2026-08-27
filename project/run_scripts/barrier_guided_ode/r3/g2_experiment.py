"""BGODE-R3 G2 matched-model rho-free convergence experiment."""

from __future__ import annotations

import hashlib
import math
import resource
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from project.run_scripts.barrier_guided_ode.r2.stage_b_contract import seal_model_vocabulary
from project.run_scripts.barrier_guided_ode.r2.stage_b_probe import _assert_w0
from project.run_scripts.barrier_guided_ode.s1_alphaedit_runtime import BGODETargetAuthority
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
from project.run_scripts.ode_edit_motivation.contracts import canonical_json, sha256_bytes
from project.run_scripts.ode_edit_motivation.direct_z import DirectZCache
from project.run_scripts.ode_edit_motivation.direct_z_alpha_possibility import _capture_teacher
from project.run_scripts.ode_edit_motivation.direct_z_possibility import (
    _observe_branch,
    _teacher_forced_prompts,
    load_heldout_evaluation_case,
)
from project.run_scripts.ode_edit_motivation.easyedit_bridge import EasyEditBridge
from project.run_scripts.ode_edit_motivation.gpu_runtime import (
    load_fixed_model,
    offline_environment,
    seed_runtime,
)
from project.run_scripts.ode_edit_motivation.hooks import resolve_parameter
from project.run_scripts.ode_edit_motivation.manifests import fixed_model_spec, preflight_fixed_artifacts
from project.run_scripts.ode_edit_motivation.mv0_fidelity import _freeze_contexts
from project.run_scripts.ode_edit_motivation.mv1_calibration import build_exact_teacher_batch
from project.run_scripts.ode_edit_motivation.projector_adapter import AlphaEditProjectorBank

from .errors import NumericalBoundary, R3ScientificBoundary
from .events import FineEventLayout, evaluate_fine_events
from .g1_jvp import R3SerialForwardJVPBackend
from .g1_probe import _event_payload, _jsonable, _seal_q0
from .g2_convergence import G2Endpoint, N_GRID, assess_cauchy, block_distance
from .g2_trajectory import run_g2_trajectory
from .natural import load_natural_record, natural_case, tokenize_record
from .reference import W0ConditionalSeal
from .solver import ControllerArm


INSTRUCTION_ID = "ODEEDIT-S05-BGODE-R3-FINE-EVENT-TARGET-EXCLUDED-FACTOR-SPACE"
SCHEMA = "ode-edit-bgode-r3-g2-rho-free-convergence/v1"
ARM_ORDER = (ControllerArm.PLAIN, ControllerArm.FISHER, ControllerArm.FULL)


def expected_run_id(model_alias: str) -> str:
    if model_alias not in ("llama3-8b-inst", "qwen2.5-7b-inst"):
        raise R3ScientificBoundary("G2 model alias is outside the locked matrix")
    return f"s05-bgode-r3-g2-{model_alias}-natural-unequal-nonprefix-convergence-v1"


def _true_metrics(
    *,
    runtime: Any,
    bindings: Any,
    hparams: Any,
    request: Any,
    heldout: Any,
    target_true: str,
) -> dict[str, Any]:
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
    return {
        "rewrite_target_true": rewrite.compact(),
        "rephrase_target_true": rephrase.compact(),
    }


def run_g2(
    *,
    easyedit_root: str | Path,
    output_root: str | Path,
    source_head: str,
    source_tree: str,
    release_receipt: Mapping[str, Any],
    model_alias: str,
    run_id: str,
) -> dict[str, Any]:
    if run_id != expected_run_id(model_alias):
        raise R3ScientificBoundary("G2 run identity differs")
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
        model_started = time.perf_counter()
        runtime = load_fixed_model(model_alias)
        torch.cuda.synchronize(0)
        model_load_wall = time.perf_counter() - model_started
        dtype_receipt = _assert_full_fp32(runtime)
        tokenization = tokenize_record(runtime.tokenizer, raw_record, model_alias=model_alias)
        if tokenization.identity != case["tokenization_identity"]:
            raise R3ScientificBoundary("G2 runtime tokenization differs from the G1 seal")
        vocabulary = seal_model_vocabulary(runtime.model, tokenization)
        layout = FineEventLayout.build(
            source_tokens=tokenization.source_token_ids,
            target_tokens=tokenization.target_token_ids,
            output_vocabulary_size=vocabulary.output_head_vocabulary_size,
            tokenizer_vocabulary_size=vocabulary.tokenizer_vocabulary_size,
        )
        if layout.topology != "unequal-non-prefix":
            raise R3ScientificBoundary("G2 natural topology differs")
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
            raise R3ScientificBoundary("G2 compute_z/event/writer target tokens differ")
        from easyeditor.models.alphaedit import compute_z as alpha_compute_z

        fixed_z_calls = 0

        def compute_z_once() -> torch.Tensor:
            nonlocal fixed_z_calls
            fixed_z_calls += 1
            if fixed_z_calls != 1:
                raise R3ScientificBoundary("G2 fixed native target was recomputed")
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
        authority0 = BGODETargetAuthority.start(
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
        initial_backend = R3SerialForwardJVPBackend(runtime.model, tokenization)
        initial_event = evaluate_fine_events(layout, initial_backend.observe_primal(layout))
        q0 = W0ConditionalSeal.capture(initial_event)
        q0_artifact = _seal_q0(output, q0)

        heldout = load_heldout_evaluation_case(root, request)
        baseline_observation = _observe_branch(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            contexts=contexts,
            request=request,
            heldout=heldout,
        )
        baseline_true = _true_metrics(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            request=request,
            heldout=heldout,
            target_true=target_true,
        )
        base_teacher = _capture_teacher(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            contexts=contexts,
            request=request,
        )
        z_delta = direct_z.values[:, 0].float() - base_teacher.subject_states[0]
        z_observation = _observe_branch(
            runtime=runtime,
            bindings=bindings,
            hparams=hparams,
            contexts=contexts,
            request=request,
            heldout=heldout,
            patch_delta=z_delta,
        )

        def observe_live_endpoint() -> dict[str, Any]:
            torch.cuda.synchronize(0)
            observation_started = time.perf_counter()
            branch = _observe_branch(
                runtime=runtime,
                bindings=bindings,
                hparams=hparams,
                contexts=contexts,
                request=request,
                heldout=heldout,
            )
            true = _true_metrics(
                runtime=runtime,
                bindings=bindings,
                hparams=hparams,
                request=request,
                heldout=heldout,
                target_true=target_true,
            )
            torch.cuda.synchronize(0)
            return {
                **_metrics_payload(branch, baseline_observation),
                **true,
                "observation_wall_seconds": time.perf_counter() - observation_started,
                "controller_influence_count": 0,
            }

        request_payload = {
            **request.to_easyedit(),
            "case_id": int(request.case_id) if request.case_id.isdigit() else request.case_id,
            "request_sha256": raw_record.request_sha256,
        }
        native_observation: dict[str, Any] | None = None
        with isolated_alphaedit_module_state() as alpha_state:
            with accepted_z_cache_template(
                (request_payload,),
                direct_z.values,
                hparams,
                parent=output / "private",
                accepted_z_source="BGODE_R3_FIXED_NATIVE_TARGET_W0",
            ) as (cache_template, accepted_target_receipt):
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
                    accepted_z_source="BGODE_R3_FIXED_NATIVE_TARGET_W0",
                )
                native_event = evaluate_fine_events(
                    layout,
                    initial_backend.observe_primal(layout),
                )
                native_observation = observe_live_endpoint()
                with torch.no_grad():
                    for name in weight_names:
                        resolve_parameter(runtime.model, name).copy_(originals[name])
        if not alpha_state.get("restored") or native_observation is None:
            raise R3ScientificBoundary("G2 Official AlphaEdit state did not restore")
        _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
        horizon = float(_event_payload(native_event)["target_logit"]) - float(
            _event_payload(initial_event)["target_logit"]
        )
        if not math.isfinite(horizon) or horizon <= 0.0:
            raise R3ScientificBoundary("NONPOSITIVE_NATIVE_TARGET_LOGIT_HORIZON")

        def snapshot_factory() -> Any:
            return _snapshot(
                runtime=runtime,
                request=request,
                contexts=contexts,
                hparams=hparams,
                weight_names=weight_names,
                provenance_id=bindings.provenance.manifest_id,
            )

        panels: dict[str, Any] = {}
        convergence: dict[str, Any] = {}
        for arm in ARM_ORDER:
            previous_blocks: tuple[torch.Tensor, ...] | None = None
            previous_n: int | None = None
            endpoints: dict[int, G2Endpoint] = {}
            distances: dict[tuple[int, int], float] = {}
            for step_count in N_GRID:
                try:
                    result = run_g2_trajectory(
                        arm=arm,
                        step_count=step_count,
                        horizon=horizon,
                        runtime=runtime,
                        adapter=adapter,
                        layout=layout,
                        q0=q0,
                        authority0=authority0,
                        snapshot_factory=snapshot_factory,
                        weight_names=weight_names,
                        tokenization=tokenization,
                        endpoint_observer=observe_live_endpoint,
                        solver_prefix=f"bgode-r3/g2/{model_alias}",
                    )
                except NumericalBoundary as error:
                    _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
                    failure = {
                        "schema": "ode-edit-bgode-r3-g2-numerical-boundary/v1",
                        "status": "R3_G2_LOCKED_NUMERICAL_BOUNDARY",
                        "instruction_id": INSTRUCTION_ID,
                        "source_head": source_head,
                        "source_tree": source_tree,
                        "model_alias": model_alias,
                        "run_id": run_id,
                        "case": dict(case),
                        "boundary_type": type(error).__name__,
                        "boundary_message": str(error),
                        "boundary_receipt": _jsonable(error.receipt),
                        "completed_panel_count": len(panels),
                        "completed_panel_keys": sorted(panels),
                        "fixed_z_compute_count": fixed_z_calls,
                        "fixed_z_recompute_count": 0,
                        "history_append_count": 0,
                        "w0_pointer_restore": True,
                        "w0_bytes_restore": True,
                        "retained_factor_count_after_boundary": len(adapter.builds),
                        "science_definition_change_count": 0,
                        "tolerance_change_count": 0,
                        "threshold_change_count": 0,
                        "scientific_promotion": False,
                    }
                    failure = _jsonable(failure)
                    failure["failure_identity"] = sha256_bytes(
                        canonical_json(failure).encode("utf-8")
                    )
                    _write_json_once(output / "failure-boundary.json", failure)
                    raise
                _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
                panels[f"{arm.value}/N{step_count}"] = result.payload
                endpoints[step_count] = result.endpoint
                if previous_blocks is not None and previous_n is not None:
                    distances[(previous_n, step_count)] = block_distance(
                        previous_blocks,
                        result.endpoint_blocks,
                    )
                    del previous_blocks
                previous_blocks = result.endpoint_blocks
                previous_n = step_count
                torch.cuda.empty_cache()
            if previous_blocks is not None:
                del previous_blocks
            convergence[arm.value] = assess_cauchy(endpoints, distances).payload()
        passed = all(bool(value["passed"]) for value in convergence.values())
        status = (
            "R3_G2_RHO_FREE_CONVERGENCE_PASS_SCIENTIFIC_ATTRIBUTION_HOLD"
            if passed
            else "R3_G2_NUMERICAL_CONVERGENCE_HOLD"
        )
        _assert_w0(runtime, weight_names=weight_names, pointers=pointers, hashes=hashes)
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "instruction_id": INSTRUCTION_ID,
            "run_id": run_id,
            "status": status,
            "source_head": source_head,
            "source_tree": source_tree,
            "release_receipt": dict(release_receipt),
            "case": dict(case),
            "tokenization": tokenization.receipt(),
            "vocabulary": vocabulary.receipt(),
            "model": runtime.metadata(),
            "dtype": dtype_receipt,
            "seed": seed_receipt,
            "contexts": {"manifest_id": contexts.manifest_id, "seed": S1_CONTEXT_SEED},
            "fixed_inputs": {
                "alpha_config_id": alpha_config.config_id,
                "alpha_reference_manifest_id": alpha_reference.manifest_id,
                "bridge_manifest_id": bridge_manifest.manifest_id,
                "fixed_manifest_id": fixed.manifest_id,
                "projector": projector.metadata(),
            },
            "fixed_target": {
                "artifact_sha256": direct_z.artifact.sha256,
                "compute_count": fixed_z_calls,
                "recompute_count": 0,
                "tensor_sha256": direct_z.tensor_sha256,
            },
            "q0_artifact": q0_artifact,
            "initial_event": _event_payload(initial_event),
            "native_event": _event_payload(native_event),
            "native_target_logit_horizon": horizon,
            "native_official_apply": native_apply,
            "accepted_target_receipt": accepted_target_receipt,
            "observations": {
                "w0": {**_metrics_payload(baseline_observation, baseline_observation), **baseline_true},
                "fixed_z_star": _metrics_payload(z_observation, baseline_observation),
                "native_alphaedit": native_observation,
            },
            "arm_order": [arm.value for arm in ARM_ORDER],
            "n_grid": list(N_GRID),
            "panels": panels,
            "cauchy": convergence,
            "cauchy_all_arms_pass": passed,
            "future_fixed_T_curve_predeclared_not_run": {
                "T": [0.5, 1.0, 2.0, 3.0, 5.0],
                "N": 32,
                "execution_count": 0,
                "selection_influence_count": 0,
            },
            "execution_boundary": {
                "batch_size": 1,
                "panel_count": 12,
                "fixed_z_compute_count": fixed_z_calls,
                "fixed_z_recompute_count": 0,
                "history_append_count": 0,
                "localizer_call_count": 0,
                "endpoint_correction_count": 0,
                "ridge_damping_floor_fallback_count": 0,
                "controller_evaluator_influence_count": 0,
                "retained_factor_count": len(adapter.builds),
                "w0_pointer_restore": True,
                "w0_bytes_restore": True,
            },
            "resources": {
                "model_load_wall_seconds": model_load_wall,
                "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "total_wall_seconds": time.perf_counter() - started,
            },
            "scientific_promotion": False,
        }
        if result["execution_boundary"]["retained_factor_count"] != 0:
            raise R3ScientificBoundary("G2 terminal retained full AlphaEdit builds")
        result = _jsonable(result)
        result["terminal_identity"] = sha256_bytes(canonical_json(result).encode("utf-8"))
        _write_json_once(output / "terminal.json", result)
        return result


__all__ = ["ARM_ORDER", "INSTRUCTION_ID", "SCHEMA", "expected_run_id", "run_g2"]
