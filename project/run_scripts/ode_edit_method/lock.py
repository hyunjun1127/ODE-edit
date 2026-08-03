"""Strict loader and dry-plan projection for the outcome-free v6 proposal."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .contracts import Arm, ControllerConfig, MethodContractError, canonical_hash
from .events import EVENT_BACKEND_MODE, EVENT_MODEL_FORWARD_CALLS


LOCK_PATH = Path(__file__).with_name("numerical_lock_proposal.json")
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
LOCK_SCHEMA = "ode-edit-compute-aware-numerical-lock-proposal/v6"
LOCK_INSTRUCTION = "ODEEDIT-S02-TWO-FORWARD-EVENT-V6-IMPL-V1"
LOCK_PARENT = "ODEEDIT-S02-P0-ORIGINAL-DTYPE-SIMPLE-T-V5-PAIR-V1"
LOCK_BASE_COMMIT = "96b314b18585470275d20c586b46e63ac9574c57"
CONTEXT_LOCKS: Mapping[str, Mapping[str, Any]] = {
    "llama3-8b-inst": {
        "source": "llama3-8b-inst:fresh-seed-17",
        "manifest_id": "22c26dc11fb13acd51d5bdc483e4b9dd46fa40029fe10b167bff1a1642f7e686",
        "group_sizes": [1, 5],
        "generation_dtype_policy": "checkpoint-original",
        "templates_sha256": "0a2069beafc60e170251103028fde716a160a60a8048bb000a649cf26c233bb0",
        "repeat_count": 2,
        "exact_match": True,
        "probe_execution_head": "37a713233b95614d2743a12abc4d1036daed95f0",
        "terminal_manifest_sha256": "d37bc471629449100e369e3aec2e9508ea32d29ffeb530e2a7eb07ea6c19c94e",
        "raw_context_manifest_sha256": "f1f428a0cd8ed5c512179d971a01f40c8b3ba0dfea7269d7ec16e24559953991",
        "legacy_provenance": {
            "generation_dtype_policy": "legacy-motivation-float32",
            "manifest_id": "3020b3f5cea62e6cfbd173f0c99a4348cecf7e84425bb087720ced7f395482e5",
            "method_evidence": False,
        },
    },
    "qwen2.5-7b-inst": {
        "source": "qwen2.5-7b-inst:fresh-seed-17",
        "manifest_id": "5b7144416638fb3deec1f12f204a401e06edd1293aa2fe4a22f9080f3e8bd41b",
        "group_sizes": [1, 5],
        "generation_dtype_policy": "checkpoint-original",
        "templates_sha256": "ff84e360b7a4a413275da818862c7c4431ac21c7fbeaf26e4d2296278f9a4bf9",
        "repeat_count": 2,
        "exact_match": True,
        "probe_execution_head": "37a713233b95614d2743a12abc4d1036daed95f0",
        "terminal_manifest_sha256": "bd7e169c2c3abe255e11563220deb558f7c30c1e22713d8be030a760c7c9d79d",
        "raw_context_manifest_sha256": "6bd9ba263f4774b65fce31c965c27344625402ee1f19f4129542fd5e08132084",
        "legacy_provenance": {
            "generation_dtype_policy": "legacy-motivation-float32",
            "manifest_id": "e0c5f61d874334a2cab26f82fb3594d9ab88c9e5816be390274a0bb5e98c93bd",
            "method_evidence": False,
        },
    },
}
P01_ARMS = (
    Arm.NATIVE_MEMIT,
    Arm.STATIC_SYNCHRONOUS,
    Arm.ONE_REFRESH,
    Arm.FULL_ODE_EDIT,
)


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MethodContractError(f"lock section {name} is not a mapping")
    return value


def _positive_float(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodContractError(f"lock value {name} is not numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise MethodContractError(f"lock value {name} must be positive")
    return number


def load_lock(path: str | Path = LOCK_PATH) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MethodContractError("numerical lock proposal is unreadable") from exc
    validate_lock(payload)
    specification = _mapping("canonical_spec", payload.get("canonical_spec"))
    relative = specification.get("path")
    expected_sha = specification.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_sha, str):
        raise MethodContractError("canonical spec identity is incomplete")
    repo = Path(__file__).resolve().parents[3]
    spec_path = (repo / relative).resolve(strict=True)
    try:
        spec_path.relative_to(repo)
    except ValueError as exc:
        raise MethodContractError("canonical spec escaped repository") from exc
    if hashlib.sha256(spec_path.read_bytes()).hexdigest() != expected_sha:
        raise MethodContractError("canonical spec bytes differ from numerical lock")
    result = dict(payload)
    result["proposal_id"] = canonical_hash(payload)
    return result


def controller_config(payload: Mapping[str, Any]) -> ControllerConfig:
    controller = _mapping("controller", payload["controller"])
    scalar = _mapping("scalar_first_hit", payload["scalar_first_hit"])
    event = _mapping("event", payload["event"])
    return ControllerConfig.from_mapping(
        {
            "tau": event["tau"],
            "h0_fraction": controller.get("h0_fraction", controller.get("h0")),
            "h_max_fraction": controller.get(
                "h_max_fraction", controller.get("radius_upper_cap")
            ),
            "kappa": controller["kappa"],
            "beta": controller["beta"],
            "eta_reject": controller["eta_reject"],
            "eta_expand": controller["eta_expand"],
            "gamma_down": controller["gamma_down"],
            "gamma_up": controller["gamma_up"],
            "s_max": controller["s_max"],
            "max_rejections_per_state": controller["max_rejections_per_state"],
            "slope_epsilon": controller["slope_support_epsilon"],
            "progress_epsilon": controller["progress_epsilon"],
            "qp_equality_tolerance": controller["qp_equality_tolerance"],
            "qp_trust_tolerance": controller["qp_trust_tolerance"],
            "qp_bisection_iterations": controller[
                "qp_trust_dual_bisection_iterations"
            ],
            "event_tolerance": event["event_tolerance"],
            "hard_worsening_tolerance": event["hard_worsening_tolerance"],
            "trust_denominator_epsilon": controller["trust_denominator_epsilon"],
            "load_denominator_epsilon": controller["load_denominator_epsilon"],
            "scalar_grid": scalar["alpha_grid"],
            "scalar_bisection_tolerance": scalar["bisection_tolerance"],
            "scalar_bisection_iterations": scalar["bisection_iterations"],
            "scalar_nonmonotonic_tolerance": scalar["nonmonotonic_tolerance"],
            "functional_commit_atol": payload["trial_backend"][
                "functional_vs_committed_atol"
            ],
            "functional_commit_rtol": payload["trial_backend"][
                "functional_vs_committed_rtol"
            ],
        }
    )


def validate_lock(payload: Any) -> None:
    root = _mapping("root", payload)
    if root.get("schema_version") != LOCK_SCHEMA:
        raise MethodContractError("unknown numerical lock proposal schema")
    if root.get("status") != "OUTCOME_FREE_PROPOSAL_PENDING_GH_APPROVAL":
        raise MethodContractError("numerical lock is not an outcome-free proposal")
    if root.get("outcome_count_at_proposal") != 0:
        raise MethodContractError("numerical lock proposal observed an outcome")
    if root.get("execution_seed") != 17:
        raise MethodContractError("common execution seed differs")
    if (
        root.get("instruction_id") != LOCK_INSTRUCTION
        or root.get("revision_id") != LOCK_INSTRUCTION
        or root.get("parent_instruction_id") != LOCK_PARENT
        or root.get("canonical_main_commit") != LOCK_BASE_COMMIT
    ):
        raise MethodContractError("v6 instruction/provenance identity differs")

    dtype_contract = _mapping("dtype_contract", root.get("dtype_contract"))
    if (
        dtype_contract.get("method_dtype_policy") != "checkpoint-original"
        or dtype_contract.get("checkpoint_original_dtype") != "torch.bfloat16"
        or dtype_contract.get("observed_parameter_dtype_required")
        != "torch.bfloat16"
        or dtype_contract.get("config_torch_dtype_required") != "torch.bfloat16"
        or dtype_contract.get("mixed_floating_parameter_dtype") != "FAIL_CLOSED"
        or dtype_contract.get("legacy_dtype_policy")
        != "legacy-motivation-float32"
        or dtype_contract.get("legacy_p0_is_original_dtype_evidence") is not False
        or dtype_contract.get("model_specific_dtype_rescue") is not False
    ):
        raise MethodContractError("checkpoint-original dtype contract differs")

    selection = _mapping("selection", root.get("selection"))
    case_ids = selection.get("canonical_case_ids")
    if case_ids != ["2022", "12498", "20964", "768"]:
        raise MethodContractError("canonical four-case order differs")
    if canonical_hash(case_ids) != selection.get("canonical_order_sha256"):
        raise MethodContractError("canonical case order hash differs")
    request_ids = selection.get("request_ids")
    if (
        not isinstance(request_ids, list)
        or len(request_ids) != len(case_ids)
        or any(not isinstance(value, str) or len(value) != 64 for value in request_ids)
        or canonical_hash(request_ids) != selection.get("request_order_sha256")
    ):
        raise MethodContractError("canonical request identity/order differs")
    if selection.get("p0_profiler_case_ids") != case_ids[:1]:
        raise MethodContractError("P0 profiler case is not the canonical prefix")
    if selection.get("p1_case_ids") != case_ids:
        raise MethodContractError("P1 cases differ from the canonical four")

    models = _mapping("models", root.get("models"))
    if tuple(models) != MODEL_ALIASES:
        raise MethodContractError("lock model aliases/order differ")
    context_ids: set[str] = set()
    for alias in MODEL_ALIASES:
        model = _mapping(alias, models[alias])
        context = _mapping("context_manifest", model.get("context_manifest"))
        expected_context = CONTEXT_LOCKS[alias]
        if context != expected_context:
            raise MethodContractError(
                f"{alias} original-BF16 context provenance differs from v6"
            )
        context_ids.add(str(context["manifest_id"]))
        if (
            not isinstance(model.get("revision"), str)
            or len(model["revision"]) != 40
            or not isinstance(model.get("hparams_sha256"), str)
            or len(model["hparams_sha256"]) != 64
            or not isinstance(model.get("hparams_size_bytes"), int)
            or model["hparams_size_bytes"] <= 0
            or model.get("checkpoint_original_dtype") != "torch.bfloat16"
            or model.get("config_torch_dtype") != "torch.bfloat16"
            or model.get("safetensors_bfloat16_tensors")
            != model.get("safetensors_total_tensors")
            or not isinstance(model.get("safetensors_total_tensors"), int)
            or model["safetensors_total_tensors"] <= 0
        ):
            raise MethodContractError("model revision/hparams pin is incomplete")
    if len(context_ids) != len(MODEL_ALIASES):
        raise MethodContractError("model context manifest IDs must be distinct")

    config = controller_config(root)
    if config.s_max != 6:
        raise MethodContractError("initial common S_max is not six")
    controller = _mapping("controller", root["controller"])
    if (
        controller.get("h0_fraction") != 0.25
        or controller.get("h_max_fraction") != 0.5
        or controller.get("trust_scale_reference")
        != "raw-synchronous-joint-C-distance-before-unit-normalization"
        or "h0" in controller
        or "radius_upper_cap" in controller
    ):
        raise MethodContractError("entry-relative trust scale differs from v3")
    if controller.get("one_refresh_accepted_cap") != 2:
        raise MethodContractError("One-refresh accepted cap is not two")
    if controller.get("model_specific_policy") is not False:
        raise MethodContractError("model-specific controller policy is enabled")
    scalar = _mapping("scalar_first_hit", root["scalar_first_hit"])
    if scalar.get("alpha_above_one_allowed") is not False:
        raise MethodContractError("scalar alpha rescue is enabled")
    if scalar.get("alpha_grid", [])[-1:] != [1.0]:
        raise MethodContractError("scalar alpha grid does not stop at one")

    derivative = _mapping("derivative_backend", root.get("derivative_backend"))
    if (
        derivative.get("primary") != "activation-actuator-hook"
        or derivative.get("dense_target_gradient") != "reference-only"
        or derivative.get("target_weight_grad_materialization_allowed") is not False
        or derivative.get("p0_finite_difference_epsilon") != 0.001
    ):
        raise MethodContractError("primary derivative backend violates the compute lock")
    artifacts = _mapping("read_only_artifacts", root.get("read_only_artifacts"))
    if any(
        artifacts.get(name) is not False
        for name in (
            "covariance_recompute_or_download",
            "projector_recompute_or_download",
            "wikipedia_statistics_recompute_or_download",
            "easyedit_write",
        )
    ):
        raise MethodContractError("read-only artifact policy enables a forbidden write")
    trial = _mapping("trial_backend", root.get("trial_backend"))
    if (
        trial.get("cached_trial_graph") != "UNSUPPORTED_FAIL_CLOSED"
        or trial.get("cached_mode_scientific_arm") is not False
        or trial.get("selected_common_backend")
        != "quantized-rowblock-commit-emulator"
        or trial.get("selection_policy") != "simple-T-every-adaptive-candidate"
        or trial.get("two_tier_pretrial") is not False
        or trial.get("row_block") != 64
        or trial.get("continuous_low_rank_overlay")
        != "reference-and-diagnostic-only"
        or trial.get("dense_weight_copy_for_trial") is not False
        or trial.get("target_weight_mutation_for_trial") is not False
        or trial.get("per_trial_checkpoint") is not False
        or trial.get("unchanged_trial_state_guard")
        != "storage-pointer-version-shape-dtype-device-without-dense-rehash"
    ):
        raise MethodContractError("current MEMIT trial backend is not locked simple-T")
    concrete = _mapping("concrete_backend", root.get("concrete_backend"))
    execution = _mapping("execution_path", root.get("execution_path"))
    if (
        concrete.get("common_model_code_path") is not True
        or concrete.get("easyedit_access") != "verified-read-only-bridge"
        or concrete.get("accepted_state_mode")
        != "quantized-rowblock-trial-then-commit-then-dedicated-rebuild"
        or execution.get("method_loader") != "load_fixed_model_checkpoint_original"
        or execution.get("retry_output_root_template")
        != "local/results/session02-p0-original-dtype-simple-t-v1-{model_alias}-{proposal_prefix}"
        or execution.get("runner_can_submit_slurm") is not False
        or execution.get("runner_requires_execute_flag") is not True
    ):
        raise MethodContractError("concrete backend/executable boundary differs")
    transaction = _mapping("transaction", root.get("transaction"))
    if (
        transaction.get("edit_entry_checkpoint_count") != 1
        or transaction.get("per_accepted_full_cpu_weight_backup") != 0
    ):
        raise MethodContractError("outer transaction still permits per-step backup")
    terminal = _mapping("terminal_geometry", root.get("terminal_geometry"))
    if (
        terminal.get("micro_step_energy_sum_allowed") is not False
        or terminal.get("p1_low_rank_cross_term_trigger_fraction") != 0.1
    ):
        raise MethodContractError("terminal net geometry trigger differs")
    event_backend = _mapping("event_backend", root.get("event_backend"))
    if (
        event_backend.get("old_new_forward") != EVENT_BACKEND_MODE
        or event_backend.get("actual_model_calls_per_event")
        != EVENT_MODEL_FORWARD_CALLS
        or event_backend.get("differentiable_autograd_calls_per_field") != 1
        or event_backend.get("individual_object_length_normalization") is not True
        or event_backend.get("combined_scorer_runtime_role")
        != "diagnostic-reference-only"
        or event_backend.get("conditional_fallback") is not False
    ):
        raise MethodContractError("two-forward event backend lock differs")
    _positive_float("event_backend.p0_two_forward_atol", event_backend.get("p0_two_forward_atol"))
    _positive_float("event_backend.p0_two_forward_rtol", event_backend.get("p0_two_forward_rtol"))

    compute = _mapping("compute_accounting", root.get("compute_accounting"))
    required_counters = [
        "N_z",
        "N_model_fwd",
        "N_event_fwd",
        "N_field_state_fwd",
        "N_field",
        "N_proposal_build",
        "N_native_sweep",
        "N_bw",
        "K_acc",
        "N_trial",
        "N_reject",
        "N_eval",
        "N_write",
    ]
    if compute.get("required_counters") != required_counters:
        raise MethodContractError("fair compute counters differ from canonical order")
    if "N_state_fwd" in compute.get("required_counters", []):
        raise MethodContractError("ambiguous N_state_fwd remains enabled")

    artifact_schema = _mapping("artifact_schema", root.get("artifact_schema"))
    controller_fields = artifact_schema.get("controller_step_required_fields")
    compute_fields = artifact_schema.get("compute_required_fields")
    manifest_model_fields = artifact_schema.get("manifest_model_required_fields")
    manifest_policy_fields = artifact_schema.get("manifest_policy_required_fields")
    required_record_fields = {
        "model",
        "case_id",
        "arm",
        "order_sha256",
        "seed",
        "commit",
        "hashes",
        "status",
    }
    if (
        not isinstance(controller_fields, list)
        or not (
            required_record_fields | {"result", "event", "Omega"}
        )
        <= set(controller_fields)
        or not isinstance(compute_fields, list)
        or not (required_record_fields | {"counters"}) <= set(compute_fields)
        or not set(required_counters) <= set(compute_fields)
        or not isinstance(manifest_model_fields, list)
        or not {
            "dtype",
            "observed_parameter_dtype",
            "checkpoint_original_dtype",
            "dtype_policy",
        }
        <= set(manifest_model_fields)
        or not isinstance(manifest_policy_fields, list)
        or not {
            "trial_backend",
            "event_backend",
            "event_nfe",
            "dtype_policy",
            "checkpoint_original_dtype",
            "observed_parameter_dtype",
        }
        <= set(manifest_policy_fields)
    ):
        raise MethodContractError("artifact records do not expose the locked P0 schema")

    timing = _mapping("timing", root.get("timing"))
    if (
        timing.get("each_non_entry_hit_repetition_n_z") != 1
        or "each_repetition_n_z" in timing
    ):
        raise MethodContractError("P0 direct-z timing rule ignores entry-hit freeze")

    stages = _mapping("stages", root.get("stages"))
    expected = [arm.value for arm in P01_ARMS]
    for stage in ("p0", "p1"):
        if _mapping(stage, stages.get(stage)).get("arms") != expected:
            raise MethodContractError(f"{stage} arms differ from the compute-aware lock")
    if _mapping("p3", stages.get("p3")).get("arms", []).count(
        Arm.ORDERED_ADAPTIVE.value
    ):
        raise MethodContractError("Ordered adaptive leaked into P3")

    resources = _mapping("resource_forecast", root.get("resource_forecast"))
    if (
        resources.get("submission_authorized") is not False
        or resources.get("concurrent_project_gpus", 99) > resources.get("project_gpu_cap", 0)
        or resources.get("host_memory_mib_per_job", 1)
        > resources.get("host_memory_cap_mib_per_gpu", 0)
        or resources.get("p0_forecast_gpu_hours_per_model")
        != "UNMEASURED_CHECKPOINT_ORIGINAL_BF16_P0"
        or resources.get("p1_forecast_gpu_hours_per_model")
        != "HOLD_UNTIL_CHECKPOINT_ORIGINAL_BF16_P0"
        or resources.get("gpu_peak_reserved_gib_forecast")
        != "UNMEASURED_CHECKPOINT_ORIGINAL_BF16_P0"
        or resources.get("simple_t_cpu_microfixture_median_ratio")
        != 3.855567094593102
        or resources.get("simple_t_extra_mac_ratio") != 96.0
        or resources.get("simple_t_microfixture_is_model_scale_forecast") is not False
    ):
        raise MethodContractError("dry proposal grants submission or exceeds GPU cap")
    v6_counter_bounds = {
        "p0_n_event_fwd_upper_bound_per_model_per_repetition": 150,
        "p0_profile_n_event_fwd_upper_bound_per_model": 600,
        "p1_n_event_fwd_upper_bound_per_model": 600,
        "p0_n_field_state_fwd_upper_bound_per_model_per_repetition": 72,
        "p0_profile_n_field_state_fwd_upper_bound_per_model": 288,
        "p1_n_field_state_fwd_upper_bound_per_model": 288,
        "p0_profile_n_reference_gate_fwd_upper_bound_per_model": 2,
        "p0_profile_n_reference_gate_bw_upper_bound_per_model": 1,
        "p1_n_reference_gate_fwd_upper_bound_per_model": 0,
        "p1_n_reference_gate_bw_upper_bound_per_model": 0,
    }
    if any(resources.get(name) != value for name, value in v6_counter_bounds.items()):
        raise MethodContractError("v6 two-forward counter bounds differ")
    boundary = _mapping("execution_boundary", root.get("execution_boundary"))
    if (
        boundary.get("gpu_now") != 0
        or boundary.get("slurm_now") is not False
        or boundary.get("checkpoint_original_dtype_required") != "torch.bfloat16"
        or boundary.get("legacy_float32_results_are_original_dtype_evidence")
        is not False
    ):
        raise MethodContractError("implementation prep unexpectedly enables GPU/Slurm")


def dry_plan(payload: Mapping[str, Any], stage: str) -> dict[str, Any]:
    if stage not in {"p0", "p1"}:
        raise MethodContractError("dry plan stage must be p0 or p1")
    validate_lock(payload)
    selection = _mapping("selection", payload["selection"])
    stages = _mapping("stages", payload["stages"])
    stage_lock = _mapping(stage, stages[stage])
    cases = selection["p0_profiler_case_ids" if stage == "p0" else "p1_case_ids"]
    resources = _mapping("resource_forecast", payload["resource_forecast"])
    execution = _mapping("execution_path", payload["execution_path"])
    dtype_contract = _mapping("dtype_contract", payload["dtype_contract"])
    proposal_id = canonical_hash(payload)
    jobs = []
    for alias in MODEL_ALIASES:
        model = _mapping(alias, payload["models"][alias])
        prefix = "p0_profile" if stage == "p0" else "p1"
        output_root = str(execution["retry_output_root_template"]).format(
            model_alias=alias,
            proposal_prefix=proposal_id[:8],
        )
        jobs.append(
            {
                "model": alias,
                "repository_id": model["repository_id"],
                "revision": model["revision"],
                "hparams_relative_path": model["hparams_relative_path"],
                "hparams_sha256": model["hparams_sha256"],
                "case_ids": list(cases),
                "order_sha256": canonical_hash(list(cases)),
                "arms": list(stage_lock["arms"]),
                "backend": payload["trial_backend"]["selected_common_backend"],
                "trial_backend": "quantized-rowblock-commit-emulator",
                "event_backend": EVENT_BACKEND_MODE,
                "event_nfe": EVENT_MODEL_FORWARD_CALLS,
                "dtype_policy": dtype_contract["method_dtype_policy"],
                "checkpoint_original_dtype": model["checkpoint_original_dtype"],
                "observed_parameter_dtype_required": dtype_contract[
                    "observed_parameter_dtype_required"
                ],
                "gpus": resources["gpus_per_job"],
                "cpus": resources["cpus_per_job"],
                "host_memory_mib": resources["host_memory_mib_per_job"],
                "wall_limit": resources[f"{stage}_wall_limit"],
                "submission_authorized": False,
                "executable_command": [
                    "/mnt/raid5/janghj/EasyEdit/.venv/bin/python",
                    "-B",
                    "project/run_scripts/session02_compute_aware_p0.py",
                    "--model-alias",
                    alias,
                    "--output-root",
                    output_root,
                    "--execute",
                ] if stage == "p0" else None,
                "sbatch_template": (
                    "project/run_scripts/session02_compute_aware_p0.sbatch"
                    if stage == "p0"
                    else None
                ),
                "forecast_upper": {
                    "N_z": resources[
                        "p0_profile_n_z_upper_bound_per_model"
                        if stage == "p0"
                        else "p1_n_z_per_model"
                    ],
                    "N_model_fwd": resources["n_model_fwd_upper_bound"],
                    "N_event_fwd": resources[
                        f"{prefix}_n_event_fwd_upper_bound_per_model"
                    ],
                    "N_field_state_fwd": resources[
                        f"{prefix}_n_field_state_fwd_upper_bound_per_model"
                    ],
                    "N_field": resources[
                        f"{prefix}_n_field_upper_bound_per_model"
                    ],
                    "N_proposal_build": resources[
                        f"{prefix}_n_proposal_build_upper_bound_per_model"
                    ],
                    "N_native_sweep": resources[
                        f"{prefix}_n_native_sweep_upper_bound_per_model"
                    ],
                    "N_bw": resources[f"{prefix}_n_bw_upper_bound_per_model"],
                    "N_reference_gate_fwd": resources[
                        f"{prefix}_n_reference_gate_fwd_upper_bound_per_model"
                    ],
                    "N_reference_gate_bw": resources[
                        f"{prefix}_n_reference_gate_bw_upper_bound_per_model"
                    ],
                    "K_acc": resources[
                        f"{prefix}_k_acc_upper_bound_per_model"
                    ],
                    "N_trial": resources[
                        f"{prefix}_n_trial_upper_bound_per_model"
                    ],
                    "N_reject": resources[
                        f"{prefix}_n_reject_upper_bound_per_model"
                    ],
                    "N_eval": resources[
                        f"{prefix}_n_eval_upper_bound_per_model"
                    ],
                    "N_write": resources[
                        f"{prefix}_n_write_upper_bound_per_model"
                    ],
                },
            }
        )
    return {
        "schema_version": "ode-edit-session02-dry-plan/v2",
        "status": "DRY_RUN_ONLY; NO_GPU; NO_SLURM",
        "stage": stage,
        "proposal_id": proposal_id,
        "same_submission_batch_required": True,
        "required_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "required_preflight": [
            "fixed-artifact-hash-and-size",
            "model-revision-present-in-offline-cache",
            "checkpoint-original-bfloat16-config-and-parameter-set",
            "simple-quantized-rowblock-T-runtime-primary",
            "fresh-context-manifest-id-match-before-action",
            "staged-agent-access-check",
        ],
        "jobs": jobs,
        "aggregate_gpus": sum(job["gpus"] for job in jobs),
    }
