"""BGODE-R1 S1 single-request six-arm GPU experiment."""

from __future__ import annotations

import contextlib
import json
import math
import os
import resource
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from project.run_scripts.ode_edit_motivation.alphaedit_proposal_adapter import (
    AlphaEditProposalAdapter,
)
from project.run_scripts.ode_edit_motivation.alphaedit_reference import (
    load_alphaedit_solver_config,
)
from project.run_scripts.ode_edit_motivation.contracts import (
    EditRequest,
    LowRankFactor,
    MemitFactorProposal,
    ProposalSemantics,
    canonical_json,
    sha256_bytes,
)
from project.run_scripts.ode_edit_motivation.direct_z import DirectZCache
from project.run_scripts.ode_edit_motivation.direct_z_alpha_possibility import (
    _capture_teacher,
)
from project.run_scripts.ode_edit_motivation.direct_z_possibility import (
    _kl_from_w0,
    _observe_branch,
    _teacher_forced_prompts,
    load_heldout_evaluation_case,
)
from project.run_scripts.ode_edit_motivation.easyedit_bridge import (
    APPROVED_ALPHAEDIT_REFERENCE_FILES,
    EasyEditBridge,
)
from project.run_scripts.ode_edit_motivation.gpu_runtime import (
    FixedModelRuntime,
    load_fixed_model,
    offline_environment,
    seed_runtime,
)
from project.run_scripts.ode_edit_motivation.hooks import (
    capture_snapshot,
    resolve_parameter,
    tensor_sha256,
)
from project.run_scripts.ode_edit_motivation.manifests import (
    ALPHAEDIT_HPARAM_BY_MODEL,
    FIXED_FILE_IDENTITIES,
    fixed_model_spec,
    preflight_fixed_artifacts,
)
from project.run_scripts.ode_edit_motivation.mv0_fidelity import (
    _bridge_pins,
    _freeze_contexts,
    _safe_payload,
    _silence_upstream,
)
from project.run_scripts.ode_edit_motivation.mv1_calibration import (
    build_exact_teacher_batch,
)
from project.run_scripts.ode_edit_motivation.projector_adapter import AlphaEditProjectorBank
from project.run_scripts.ode_bf.p1r52_target_official_alphaedit_writer import (
    accepted_z_cache_template,
    isolated_alphaedit_module_state,
)
from project.run_scripts.ode_bf.scalable_batched_native import run_official_native_apply

from .errors import BGODEScientificBoundary
from .policies import explicit_native_bypass
from .rayleighian_controller import solve_equality_rayleighian
from .s1_alphaedit_runtime import (
    AtomicWeightTrajectory,
    BGODETargetAuthority,
    SerialForwardJVPBackend,
)
from .s1_contract import (
    S1Arm,
    S1_ARM_ORDER,
    S1_CONTEXT_SEED,
    S1_MODEL_ALIAS,
    load_sealed_s1_sample,
    seal_s1_tokenization,
    s1_model_binding,
    validate_s1_horizon,
)
from .s1_streaming_events import (
    StreamingEventState,
    StreamingTrieLayout,
    aggregate_streaming_moments,
    evaluate_streaming_state,
)


INSTRUCTION_ID = "ODEEDIT-S05-BGODE-R1-SEQUENCE-EVENT-BARRIER-GUIDED-ODE-SCIENCE-R1"
RUN_ID = "s05-bgode-r1-s1-llama-b1-request000-six-arm-v1"
SCHEMA = "ode-edit-bgode-r1-s1-six-arm-terminal/v1"
RUN_SEED = 41
EULER_STEPS = 4
LAYERS = (4, 5, 6, 7, 8)


def s1_easyedit_pins() -> dict[str, Mapping[str, Any]]:
    """Return the exact verified MEMIT plus Official AlphaEdit import closure."""

    pins = dict(_bridge_pins())
    for relative in APPROVED_ALPHAEDIT_REFERENCE_FILES:
        identity = FIXED_FILE_IDENTITIES[relative]
        pins[relative] = {"sha256": identity.sha256, "size": identity.size}
    return pins


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise BGODEScientificBoundary(f"create-once output already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _snapshot(
    *,
    runtime: FixedModelRuntime,
    request: EditRequest,
    contexts: Any,
    hparams: Any,
    weight_names: tuple[str, ...],
    provenance_id: str,
):
    return capture_snapshot(
        runtime.model,
        model_id=runtime.spec.snapshot_name,
        requests=(request,),
        context_id=contexts.manifest_id,
        hparams=hparams,
        weight_names=weight_names,
        provenance_ids=(provenance_id,),
    )


def _rebase_frozen_proposal(
    proposal: MemitFactorProposal,
    snapshot: Any,
) -> MemitFactorProposal:
    factors = tuple(
        LowRankFactor(
            weight_name=factor.weight_name,
            left=factor.left,
            right=factor.right,
            expected_weight_sha256=snapshot.parameter(factor.weight_name).sha256,
            native_update_transposed=factor.native_update_transposed,
        )
        for factor in proposal.factors
    )
    return MemitFactorProposal(
        snapshot=snapshot,
        factors=factors,
        semantics=ProposalSemantics.ORDERED_GAUSS_SEIDEL,
        solver_name=f"{proposal.solver_name}/frozen-field-rebased",
        residual_denominator=None,
    )


def _plain_velocity(progress: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    if progress.dtype != torch.float32 or progress.shape != (5,) or not bool(torch.isfinite(progress).all()):
        raise BGODEScientificBoundary("Plain progress sensitivity is invalid")
    denominator = torch.sum(progress)
    scale = max(1.0, float(torch.linalg.vector_norm(progress).item()))
    tolerance = 32.0 * torch.finfo(torch.float32).eps * 5.0 * scale
    value = float(denominator.item())
    if not math.isfinite(value) or abs(value) <= tolerance:
        raise BGODEScientificBoundary("Plain equal-direction denominator is zero/rank invalid")
    velocity = torch.ones(5, dtype=torch.float32) / denominator
    residual = abs(float(torch.dot(progress, velocity).item()) - 1.0)
    if residual > 512.0 * torch.finfo(torch.float32).eps * 5.0:
        raise BGODEScientificBoundary("Plain exact progress equality residual is too large")
    return velocity, {
        "denominator": value,
        "denominator_tolerance": tolerance,
        "equality_residual": residual,
    }


def _proposal_update_energy(proposal: MemitFactorProposal, coefficients: torch.Tensor) -> dict[str, Any]:
    per_layer: dict[str, float] = {}
    for factor, coefficient in zip(proposal.factors, coefficients, strict=True):
        left = factor.left.float() * float(coefficient.item())
        right = factor.right.float()
        energy = float(torch.sum((left.T @ left) * (right.T @ right)).item())
        per_layer[factor.weight_name] = energy
    total = sum(per_layer.values())
    return {
        "per_layer_energy": per_layer,
        "total_energy": total,
        "per_layer_share": {
            name: (value / total if total > 0.0 else 0.0) for name, value in per_layer.items()
        },
    }


def _event_payload(state: StreamingEventState) -> dict[str, Any]:
    return {
        "target_log_probability": state.target_log_probability,
        "source_log_probability": state.source_log_probability,
        "log_odds": state.log_odds,
        "pair_mass": state.pair_mass,
        "normalization_log_residual": state.normalization_log_residual,
    }


def _metrics_payload(branch: Any, baseline: Any) -> dict[str, Any]:
    return {
        "rewrite_target_new": branch.teacher.rewrite.compact(),
        "rephrase_target_new": branch.paraphrase.compact(),
        "locality_forward_kl": _kl_from_w0(
            baseline.preservation_logits, branch.preservation_logits
        ),
    }


def _true_metrics(
    *,
    runtime: FixedModelRuntime,
    bindings: Any,
    hparams: Any,
    sample: Any,
    heldout: Any,
) -> dict[str, Any]:
    prompt = sample.edit_request.prompt.format(sample.edit_request.subject)
    rewrite = _teacher_forced_prompts(
        runtime=runtime,
        bindings=bindings,
        hparams=hparams,
        prompts=(prompt,),
        target_text=sample.target_true,
        subject=sample.edit_request.subject,
    )
    rephrase = _teacher_forced_prompts(
        runtime=runtime,
        bindings=bindings,
        hparams=hparams,
        prompts=heldout.paraphrase_prompts,
        target_text=sample.target_true,
        subject=sample.edit_request.subject,
    )
    return {
        "rewrite_target_true": rewrite.compact(),
        "rephrase_target_true": rephrase.compact(),
    }


def _load_alpha_hparams(root: Path, bindings: Any, spec: Any, model_alias: str) -> Any:
    # The bridge has already installed and pinned the minimal EasyEdit namespace.
    from easyeditor.models.alphaedit.AlphaEdit_hparams import AlphaEditHyperParams

    path = root / ALPHAEDIT_HPARAM_BY_MODEL[model_alias]
    hparams = AlphaEditHyperParams.from_hparams(str(path))
    if tuple(int(layer) for layer in hparams.layers) != LAYERS:
        raise BGODEScientificBoundary("Official AlphaEdit layer order differs")
    hparams.device = 0
    hparams.stats_dir = str((root / "examples/data/stats").resolve(strict=True))
    hparams.P_loc = str((root / spec.projector_path).resolve(strict=True))
    return hparams


def _assert_full_fp32(runtime: FixedModelRuntime) -> dict[str, Any]:
    counts: dict[str, int] = {}
    quantized = 0
    for parameter in runtime.model.parameters():
        if parameter.is_floating_point():
            counts[str(parameter.dtype)] = counts.get(str(parameter.dtype), 0) + 1
        if parameter.__class__.__name__.lower().find("quant") >= 0:
            quantized += 1
    if set(counts) != {"torch.float32"} or quantized != 0:
        raise BGODEScientificBoundary("S1 model parameter inventory is not unquantized FP32")
    if torch.is_autocast_enabled() or torch.is_autocast_enabled("cuda"):
        raise BGODEScientificBoundary("S1 autocast must be disabled")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise BGODEScientificBoundary("S1 TF32 must be disabled")
    return {
        "parameter_dtype_counts": counts,
        "quantized_parameter_count": quantized,
        "autocast_enabled": False,
        "tf32_enabled": False,
        "bf16_conversion_count": 0,
        "fp16_conversion_count": 0,
        "numeric_storage_cast_count": 0,
    }


def _run_dynamic_arm(
    *,
    arm: S1Arm,
    runtime: FixedModelRuntime,
    adapter: AlphaEditProposalAdapter,
    initial: StreamingEventState,
    layout: StreamingTrieLayout,
    tokenization: Any,
    authority0: BGODETargetAuthority,
    horizon: float,
    snapshot_factory: Any,
    weight_names: tuple[str, ...],
    endpoint_observer: Callable[[], Mapping[str, Any]],
) -> tuple[dict[str, Any], StreamingEventState]:
    if arm not in {
        S1Arm.PLAIN_DYNAMIC,
        S1Arm.FISHER_DYNAMIC,
        S1Arm.FULL_DYNAMIC,
        S1Arm.ONE_STEP_FULL,
        S1Arm.FROZEN_FIELD_N4,
    }:
        raise BGODEScientificBoundary("unknown dynamic S1 arm")
    steps = 1 if arm is S1Arm.ONE_STEP_FULL else EULER_STEPS
    step_size = horizon / steps
    backend = SerialForwardJVPBackend(runtime.model, tokenization)
    authority = authority0
    node_rows: list[dict[str, Any]] = []
    frozen_proposal: MemitFactorProposal | None = None
    frozen_observations: Mapping[Any, Any] | None = None
    with AtomicWeightTrajectory(runtime.model, weight_names) as trajectory:
        for node in range(steps):
            node_started = time.perf_counter()
            current_snapshot = snapshot_factory()
            if frozen_proposal is None or arm is not S1Arm.FROZEN_FIELD_N4:
                torch.cuda.synchronize(next(runtime.model.parameters()).device)
                dictionary_started = time.perf_counter()
                with _silence_upstream():
                    proposal = adapter.propose_ordered(
                        origin_lineage=authority,
                        construction="genuine-p-inside-solve",
                        solver_suffix=f"bgode-r1-s1/{arm.value}/node-{node}",
                    ).proposal
                torch.cuda.synchronize(next(runtime.model.parameters()).device)
                dictionary_wall = time.perf_counter() - dictionary_started
                dictionary_refresh = True
                jvp_wall_before = backend.ledger.wall_seconds
                observations = backend.observe(
                    layout=layout,
                    proposal=proposal,
                    validate_first_node_fd=(arm is S1Arm.PLAIN_DYNAMIC and node == 0),
                )
                jvp_wall = backend.ledger.wall_seconds - jvp_wall_before
                if arm is S1Arm.FROZEN_FIELD_N4:
                    frozen_proposal = proposal
                    frozen_observations = observations
            else:
                proposal = _rebase_frozen_proposal(frozen_proposal, current_snapshot)
                assert frozen_observations is not None
                observations = frozen_observations
                dictionary_wall = 0.0
                dictionary_refresh = False
                jvp_wall = 0.0
            reference_time = float(node * step_size)
            moments = aggregate_streaming_moments(
                initial=initial,
                observations=observations,
                reference_time=reference_time,
            )
            if arm is S1Arm.PLAIN_DYNAMIC:
                velocity, numerical = _plain_velocity(moments.progress_sensitivity)
            else:
                gradient = (
                    moments.moving_gradient
                    if arm in {S1Arm.FULL_DYNAMIC, S1Arm.ONE_STEP_FULL, S1Arm.FROZEN_FIELD_N4}
                    else torch.zeros(5, dtype=torch.float32)
                )
                solution = solve_equality_rayleighian(
                    moments.fisher.matrix,
                    gradient,
                    moments.progress_sensitivity,
                    1.0,
                )
                velocity = solution.velocity
                numerical = asdict(solution.receipt)
            coefficients = (velocity * step_size).detach().cpu().to(dtype=torch.float32).contiguous()
            energy = _proposal_update_energy(proposal, coefficients)
            action = trajectory.apply(proposal, coefficients)
            child = snapshot_factory()
            authority = authority.derive(
                proposal=proposal,
                coefficients=coefficients,
                child_snapshot=child,
                applied_hashes=dict(action.parameter_hashes),
            )
            endpoint = evaluate_streaming_state(layout, backend.observe_primal(layout))
            expected_progress = initial.log_odds + (node + 1) * step_size
            node_rows.append(
                {
                    "node": node,
                    "time_entry": reference_time,
                    "step_size": step_size,
                    "proposal_sha256": action.action_sha256,
                    "dictionary_refreshed": dictionary_refresh,
                    "dictionary_wall_seconds": dictionary_wall,
                    "dictionary_logical_key_capture_count": 5 if dictionary_refresh else 0,
                    "dictionary_logical_terminal_capture_count": 5 if dictionary_refresh else 0,
                    "dictionary_actual_model_forward_count": "NOT_RESOLVED_SOURCE_LEVEL_HELPER_INTERNALS",
                    "serial_jvp_wall_seconds": jvp_wall,
                    "dictionary_state_id": proposal.snapshot.state_id,
                    "coefficients": [float(value) for value in coefficients],
                    "velocity": [float(value) for value in velocity],
                    "numerical": numerical,
                    "event_entry": _event_payload(moments.current),
                    "event_exit": _event_payload(endpoint),
                    "expected_exit_log_odds": expected_progress,
                    "nonlinear_progress_residual": endpoint.log_odds - expected_progress,
                    "moving_reference_kl_entry": moments.moving_reference_kl,
                    "anchored_kl_entry": moments.anchored_kl,
                    "score_centering_norm": float(torch.linalg.vector_norm(moments.score_expectation).item()),
                    "event_score_buffer_peak_bytes": moments.event_score_buffer_peak_bytes,
                    "pair_mass_sensitivity": [float(value) for value in moments.pair_mass_sensitivity],
                    "update_energy": energy,
                    "physical_action": asdict(action),
                    "history_append_count": 0,
                    "fixed_z_recompute_count": 0,
                    "node_total_wall_seconds": time.perf_counter() - node_started,
                }
            )
        final = evaluate_streaming_state(layout, backend.observe_primal(layout))
        # This callback is deliberately outside the controller loop and its
        # return value is never visible to any subsequent controller decision.
        # It observes the exact live endpoint before the transaction restores W0.
        terminal_observation = dict(endpoint_observer())
        result = {
            "arm": arm.value,
            "steps": steps,
            "horizon": horizon,
            "step_size": step_size,
            "nodes": node_rows,
            "terminal_event": _event_payload(final),
            "terminal_observation": terminal_observation,
            "compute": asdict(backend.ledger),
            "finite_difference": (
                None if backend.finite_difference_receipt is None else asdict(backend.finite_difference_receipt)
            ),
            "dictionary_build_count": 1 if arm is S1Arm.FROZEN_FIELD_N4 else steps,
            "dictionary_actual_model_forward_count": "NOT_RESOLVED_SOURCE_LEVEL_HELPER_INTERNALS",
            "ordered_unit_prefix_write_count": 4 * (1 if arm is S1Arm.FROZEN_FIELD_N4 else steps),
            "ordered_unit_prefix_restore_count": 4 * (1 if arm is S1Arm.FROZEN_FIELD_N4 else steps),
            "physical_final_write_count": steps,
            "physical_layer_apply_count": steps * 5,
            "history_append_inside_node_count": 0,
            "controller_evaluator_input_count": 0,
            "heldout_decision_influence_count": 0,
        }
    return result, final


def run_s1(
    *,
    easyedit_root: str | Path,
    output_root: str | Path,
    source_head: str,
    source_tree: str,
    release_receipt: Mapping[str, Any],
    model_alias: str = S1_MODEL_ALIAS,
    run_id: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    model_binding = s1_model_binding(model_alias)
    selected_run_id = model_binding.run_id if run_id is None else run_id
    if selected_run_id != model_binding.run_id:
        raise BGODEScientificBoundary("S1 run identity does not match model binding")
    root = Path(easyedit_root).expanduser().resolve(strict=True)
    output = Path(output_root).expanduser().resolve(strict=False) / selected_run_id
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    status_path = output / "terminal.json"
    try:
        sample = load_sealed_s1_sample()
        spec = fixed_model_spec(model_alias)
        fixed = preflight_fixed_artifacts(root, model_alias=model_alias)
        bridge = EasyEditBridge(
            root,
            expected_files=s1_easyedit_pins(),
            include_alphaedit_reference=True,
        )
        bridge_manifest = bridge.preflight()
        with offline_environment():
            seed_receipt = seed_runtime(RUN_SEED)
            bindings = bridge.load()
            model_load_started = time.perf_counter()
            runtime = load_fixed_model(model_alias)
            torch.cuda.synchronize(0)
            model_load_wall = time.perf_counter() - model_load_started
            dtype_receipt = _assert_full_fp32(runtime)
            tokenization = seal_s1_tokenization(
                runtime.tokenizer, sample, model_alias=model_alias
            )
            hparams = _load_alpha_hparams(root, bindings, spec, model_alias)
            contexts = _freeze_contexts(bridge, runtime, seed=S1_CONTEXT_SEED)
            projector = AlphaEditProjectorBank.open(root, spec)
            alpha_config, alpha_reference = load_alphaedit_solver_config(
                root, model_alias, memit_hparams=hparams
            )
            request = sample.edit_request
            weight_names = tuple(
                f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in LAYERS
            )
            origin = _snapshot(
                runtime=runtime,
                request=request,
                contexts=contexts,
                hparams=hparams,
                weight_names=weight_names,
                provenance_id=bindings.provenance.manifest_id,
            )
            target_ids = build_exact_teacher_batch(runtime.tokenizer, request, contexts).target_ids
            from easyeditor.models.alphaedit import compute_z as alpha_compute_z

            z_calls = 0
            z_compute_wall = 0.0

            def compute_z_once() -> torch.Tensor:
                nonlocal z_calls, z_compute_wall
                z_calls += 1
                if z_calls != 1:
                    raise BGODEScientificBoundary("fixed native z* was recomputed")
                torch.cuda.synchronize(0)
                z_started = time.perf_counter()
                value = alpha_compute_z.compute_z(
                    runtime.model,
                    runtime.tokenizer,
                    request.to_easyedit(),
                    hparams,
                    LAYERS[-1],
                    contexts.to_easyedit(),
                )
                torch.cuda.synchronize(0)
                z_compute_wall = time.perf_counter() - z_started
                return value.detach().cpu().float().reshape(-1, 1)

            direct_z = DirectZCache(output / "private", "direct-z.pt").load_or_compute(
                source_snapshot=origin,
                z_layer=LAYERS[-1],
                compute=compute_z_once,
            )
            authority0 = BGODETargetAuthority.start(
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
            )
            layout = StreamingTrieLayout.build(
                source_tokens=tokenization.source_token_ids,
                target_tokens=tokenization.target_token_ids,
                boundary_token=tokenization.boundary_token_id,
                vocabulary_size=tokenization.tokenizer_vocab_size,
            )
            primal_backend = SerialForwardJVPBackend(runtime.model, tokenization)
            initial = evaluate_streaming_state(layout, primal_backend.observe_primal(layout))

            # The observation payload is sealed once and is not accepted by any
            # target, dictionary, moment, velocity, or writer API below.  It is
            # used only by endpoint callbacks after an arm has completed all
            # controller actions.
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
                sample=sample,
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
                true_metrics = _true_metrics(
                    runtime=runtime,
                    bindings=bindings,
                    hparams=hparams,
                    sample=sample,
                    heldout=heldout,
                )
                torch.cuda.synchronize(0)
                return {
                    **_metrics_payload(branch, baseline_observation),
                    **true_metrics,
                    "observation_wall_seconds": time.perf_counter() - observation_started,
                    "controller_influence_count": 0,
                }

            # Native Official AlphaEdit consumes the already-computed z* through
            # its exact cache interface; native compute_z therefore remains zero.
            request_payload = {
                **request.to_easyedit(),
                "case_id": int(request.case_id),
                "request_sha256": sample.request_sha256,
            }
            base_pointers = {name: int(resolve_parameter(runtime.model, name).data_ptr()) for name in weight_names}
            native_observation = None
            native_event = None
            native_apply = None
            with isolated_alphaedit_module_state() as alpha_state:
                with accepted_z_cache_template(
                    (request_payload,),
                    direct_z.values,
                    hparams,
                    parent=output / "private",
                    accepted_z_source="BGODE_R1_FIXED_NATIVE_Z_STAR_W0",
                ) as (cache_template, z_bridge):
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
                        accepted_z_source="BGODE_R1_FIXED_NATIVE_Z_STAR_W0",
                    )
                    native_event = evaluate_streaming_state(layout, primal_backend.observe_primal(layout))
                    native_observation = observe_live_endpoint()
                    native_weights = {
                        name: resolve_parameter(runtime.model, name).detach().clone() for name in weight_names
                    }
                    with torch.no_grad():
                        for name in weight_names:
                            resolve_parameter(runtime.model, name).copy_(originals[name])
                    if any(
                        int(resolve_parameter(runtime.model, name).data_ptr()) != base_pointers[name]
                        or tensor_sha256(resolve_parameter(runtime.model, name)) != origin.parameter(name).sha256
                        for name in weight_names
                    ):
                        raise BGODEScientificBoundary("Official native arm did not restore W0")
            if not alpha_state.get("restored"):
                raise BGODEScientificBoundary("Official AlphaEdit module state did not restore")
            assert native_event is not None and native_apply is not None and native_observation is not None
            horizon = validate_s1_horizon(native_event.log_odds - initial.log_odds)

            # Adapter self-fidelity to the exact Official endpoint, without a
            # second model forward or target computation.
            with _silence_upstream():
                native_proposal = adapter.propose_ordered(
                    origin_lineage=authority0,
                    construction="genuine-p-inside-solve",
                    solver_suffix="bgode-r1-s1/native-self-fidelity",
                ).proposal
            bypass = explicit_native_bypass(
                batch_size=1, euler_steps=1, barrier_enabled=False, factor_count=5
            )
            with AtomicWeightTrajectory(runtime.model, weight_names) as fidelity_trajectory:
                fidelity_trajectory.apply(native_proposal, bypass.coefficients)
                numerator = 0.0
                denominator = 0.0
                for name in weight_names:
                    candidate = resolve_parameter(runtime.model, name).detach().float()
                    official = native_weights[name].detach().float()
                    numerator += float(torch.sum((candidate - official).square()).item())
                    denominator += float(torch.sum((official - originals[name].float()).square()).item())
                relative = math.sqrt(numerator / max(denominator, torch.finfo(torch.float32).tiny))
                if not math.isfinite(relative) or relative > 5.0e-5:
                    raise BGODEScientificBoundary("P-inside adapter differs from Official AlphaEdit endpoint")
            del native_weights
            del originals

            def snapshot_factory():
                return _snapshot(
                    runtime=runtime,
                    request=request,
                    contexts=contexts,
                    hparams=hparams,
                    weight_names=weight_names,
                    provenance_id=bindings.provenance.manifest_id,
                )

            arms: dict[str, Any] = {
                S1Arm.NATIVE_ALPHAEDIT.value: {
                    "arm": S1Arm.NATIVE_ALPHAEDIT.value,
                    "native_bypass": {
                        "coefficients": [float(value) for value in bypass.coefficients],
                        "rho": bypass.rho,
                        "controller_call_count": bypass.controller_call_count,
                        "root_localization_count": bypass.root_localization_count,
                        "barrier_influence_count": bypass.barrier_influence_count,
                    },
                    "official_apply": native_apply,
                    "accepted_z_bridge": z_bridge,
                    "terminal_event": _event_payload(native_event),
                    "physical_final_write_count": 1,
                    "physical_layer_apply_count": 5,
                }
            }
            endpoint_states: dict[str, StreamingEventState] = {
                S1Arm.NATIVE_ALPHAEDIT.value: native_event
            }
            for arm in S1_ARM_ORDER[1:]:
                result, state = _run_dynamic_arm(
                    arm=arm,
                    runtime=runtime,
                    adapter=adapter,
                    initial=initial,
                    layout=layout,
                    tokenization=tokenization,
                    authority0=authority0,
                    horizon=horizon,
                    snapshot_factory=snapshot_factory,
                    weight_names=weight_names,
                    endpoint_observer=observe_live_endpoint,
                )
                arms[arm.value] = result
                endpoint_states[arm.value] = state

            if any(
                int(resolve_parameter(runtime.model, name).data_ptr()) != base_pointers[name]
                or tensor_sha256(resolve_parameter(runtime.model, name))
                != origin.parameter(name).sha256
                for name in weight_names
            ):
                raise BGODEScientificBoundary("six-arm terminal W0 pointer/bytes restore failed")

            observations: dict[str, Any] = {
                "w0": {
                    **_metrics_payload(baseline_observation, baseline_observation),
                    **baseline_true,
                },
                "fixed_z_star": _metrics_payload(z_observation, baseline_observation),
                S1Arm.NATIVE_ALPHAEDIT.value: native_observation,
            }

            for arm in S1_ARM_ORDER[1:]:
                observations[arm.value] = arms[arm.value]["terminal_observation"]
            for arm in S1_ARM_ORDER:
                observations[arm.value]["event"] = _event_payload(endpoint_states[arm.value])

            torch.cuda.synchronize(0)
            peak_allocated = int(torch.cuda.max_memory_allocated(0))
            peak_reserved = int(torch.cuda.max_memory_reserved(0))
            result = {
                "schema": SCHEMA,
                "instruction_id": INSTRUCTION_ID,
                "run_id": selected_run_id,
                "status": "BGODE_R1_S1_TERMINAL_PASS",
                "source_head": source_head,
                "source_tree": source_tree,
                "release_receipt": dict(release_receipt),
                "sample": sample.receipt(),
                "sample_identity": sample.identity,
                "tokenization": tokenization.receipt(),
                "termination": {
                    "boundary_string": tokenization.boundary_string,
                    "boundary_token_id": tokenization.boundary_token_id,
                    "alternate_boundary_count": 0,
                },
                "model": runtime.metadata(),
                "dtype": dtype_receipt,
                "seed": seed_receipt,
                "contexts": {
                    "seed": S1_CONTEXT_SEED,
                    "manifest_id": contexts.manifest_id,
                },
                "fixed_inputs": {
                    "fixed_manifest_id": fixed.manifest_id,
                    "bridge_manifest_id": bridge_manifest.manifest_id,
                    "alpha_reference_manifest_id": alpha_reference.manifest_id,
                    "alpha_config_id": alpha_config.config_id,
                    "projector": projector.metadata(),
                },
                "initial_event": _event_payload(initial),
                "native_horizon": {
                    "T_AE": horizon,
                    "r0": initial.log_odds,
                    "r_AE": native_event.log_odds,
                    "dynamic_N": 4,
                    "dynamic_h": horizon / 4.0,
                    "one_step_N": 1,
                },
                "fixed_z": {
                    "compute_count": z_calls,
                    "recompute_count": 0,
                    "compute_wall_seconds": z_compute_wall,
                    "tensor_sha256": direct_z.tensor_sha256,
                    "artifact_sha256": direct_z.artifact.sha256,
                    "artifact_size": direct_z.artifact.size,
                },
                "native_adapter_fidelity": {
                    "relative_frobenius": relative,
                    "tolerance": 5.0e-5,
                    "pass": True,
                },
                "arm_order": [arm.value for arm in S1_ARM_ORDER],
                "arms": arms,
                "observations": observations,
                "controller_firewall": {
                    "endpoint_observation_after_each_arm_action_freeze": True,
                    "endpoint_observation_return_consumed_by_controller": False,
                    "heldout_decision_influence_count": 0,
                    "locality_decision_influence_count": 0,
                    "pair_mass_decision_influence_count": 0,
                    "pair_mass_constraint_count": 0,
                    "sign_cap_damping_ridge_count": 0,
                    "history_append_inside_euler_count": 0,
                    "terminal_history_append_count": 0,
                },
                "resources": {
                    "model_load_wall_seconds": model_load_wall,
                    "peak_gpu_allocated_bytes": peak_allocated,
                    "peak_gpu_reserved_bytes": peak_reserved,
                    "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                    "total_wall_seconds": time.perf_counter() - started,
                },
                "hypotheses": {
                    "H1_event_validity": "MEASURED",
                    "H2_barrier_attribution": "FACTUAL_ENDPOINT_ONLY_NO_PROMOTION",
                    "H3_ode_attribution": "FACTUAL_ENDPOINT_ONLY_NO_CONVERGENCE_CLAIM",
                },
                "scientific_promotion": False,
                "claims_forbidden": [
                    "classical_CBF_invariance",
                    "monotone_barrier_decrease",
                    "global_locality",
                    "Euler_convergence_from_single_pilot",
                ],
            }
            _safe_payload(result)
            projector.assert_hash_current()
            fixed.assert_current()
            bridge_manifest.assert_current()
            _write_json_once(status_path, result)
            return result
    except BaseException as exc:
        failure = {
            "schema": SCHEMA,
            "instruction_id": INSTRUCTION_ID,
            "run_id": selected_run_id,
            "status": "BGODE_R1_S1_TERMINAL_BOUNDARY",
            "failure_type": type(exc).__name__,
            "failure_message_sha256": sha256_bytes(str(exc).encode("utf-8")),
            "source_head": source_head,
            "source_tree": source_tree,
            "scientific_promotion": False,
            "total_wall_seconds": time.perf_counter() - started,
        }
        _write_json_once(status_path, failure)
        raise


__all__ = ["INSTRUCTION_ID", "RUN_ID", "run_s1", "s1_easyedit_pins"]
