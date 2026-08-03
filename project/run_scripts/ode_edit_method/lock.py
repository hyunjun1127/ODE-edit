"""Strict loader and dry-plan projection for the outcome-free v2 proposal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import Arm, ControllerConfig, MethodContractError, canonical_hash


LOCK_PATH = Path(__file__).with_name("numerical_lock_proposal.json")
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
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


def load_lock(path: str | Path = LOCK_PATH) -> dict[str, Any]:
    source = Path(path).resolve(strict=True)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MethodContractError("numerical lock proposal is unreadable") from exc
    validate_lock(payload)
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
            "h0": controller["h0"],
            "h_max": controller["radius_upper_cap"],
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
        }
    )


def validate_lock(payload: Any) -> None:
    root = _mapping("root", payload)
    if root.get("schema_version") != "ode-edit-compute-aware-numerical-lock-proposal/v2":
        raise MethodContractError("unknown numerical lock proposal schema")
    if root.get("status") != "OUTCOME_FREE_PROPOSAL_PENDING_GH_APPROVAL":
        raise MethodContractError("numerical lock is not an outcome-free proposal")
    if root.get("outcome_count_at_proposal") != 0:
        raise MethodContractError("numerical lock proposal observed an outcome")
    if root.get("execution_seed") != 17:
        raise MethodContractError("common execution seed differs")

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
    context_ids = {
        _mapping("context_manifest", _mapping(alias, models[alias])["context_manifest"])[
            "manifest_id"
        ]
        for alias in MODEL_ALIASES
    }
    if len(context_ids) != 2 or any(
        not isinstance(value, str) or len(value) != 64 for value in context_ids
    ):
        raise MethodContractError("model context manifests are incomplete")
    for alias in MODEL_ALIASES:
        model = _mapping(alias, models[alias])
        if (
            not isinstance(model.get("revision"), str)
            or len(model["revision"]) != 40
            or not isinstance(model.get("hparams_sha256"), str)
            or len(model["hparams_sha256"]) != 64
            or not isinstance(model.get("hparams_size_bytes"), int)
            or model["hparams_size_bytes"] <= 0
        ):
            raise MethodContractError("model revision/hparams pin is incomplete")

    config = controller_config(root)
    if config.s_max != 6:
        raise MethodContractError("initial common S_max is not six")
    controller = _mapping("controller", root["controller"])
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
        or not str(trial.get("selected_common_backend", "")).startswith("no-grad-trial")
    ):
        raise MethodContractError("current MEMIT trial backend is not fail-closed no-grad")

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
    ):
        raise MethodContractError("dry proposal grants submission or exceeds GPU cap")
    boundary = _mapping("execution_boundary", root.get("execution_boundary"))
    if boundary.get("gpu_now") != 0 or boundary.get("slurm_now") is not False:
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
    jobs = []
    for alias in MODEL_ALIASES:
        model = _mapping(alias, payload["models"][alias])
        prefix = "p0_profile" if stage == "p0" else "p1"
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
                "gpus": resources["gpus_per_job"],
                "cpus": resources["cpus_per_job"],
                "host_memory_mib": resources["host_memory_mib_per_job"],
                "wall_limit": resources[f"{stage}_wall_limit"],
                "submission_authorized": False,
                "forecast_upper": {
                    "N_z": resources[
                        "p0_profile_n_z_upper_bound_per_model"
                        if stage == "p0"
                        else "p1_n_z_per_model"
                    ],
                    "N_state_fwd": resources[
                        f"{prefix}_n_state_fwd_upper_bound_per_model"
                    ],
                    "N_field": resources[
                        f"{prefix}_n_field_upper_bound_per_model"
                    ],
                    "N_bw": resources[f"{prefix}_n_bw_upper_bound_per_model"],
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
        "schema_version": "ode-edit-session02-dry-plan/v1",
        "status": "DRY_RUN_ONLY; NO_GPU; NO_SLURM",
        "stage": stage,
        "proposal_id": canonical_hash(payload),
        "same_submission_batch_required": True,
        "required_environment": {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        "required_preflight": [
            "fixed-artifact-hash-and-size",
            "model-revision-present-in-offline-cache",
            "fresh-context-manifest-id-match-before-action",
            "staged-agent-access-check",
        ],
        "jobs": jobs,
        "aggregate_gpus": sum(job["gpus"] for job in jobs),
    }
