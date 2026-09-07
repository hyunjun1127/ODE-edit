"""Immutable actual S grid; prefix reuse never changes the scientific endpoint.

This module has no model, scheduler, diagnosis/replay, or runtime global mutation.
The shared trajectory owner consumes each path's frozen TrajectoryConfig.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .contracts import TrajectoryConfig


MODELS = ("llama3-8b-inst", "qwen2.5-7b-inst")
ACTUAL_LAMBDAS = (.01, .03162277660168379, .1, .31622776601683794, 1.)


class SweepBoundary(RuntimeError):
    """A requested grid or prefix binding differs from the sealed S contract."""


@dataclass(frozen=True, slots=True)
class EndpointSpec:
    candidate_id: str
    path_id: str
    config: TrajectoryConfig
    completed_nodes: int
    derived_observation_only: bool


@dataclass(frozen=True, slots=True)
class PathSpec:
    path_id: str
    config: TrajectoryConfig
    endpoint_ids: tuple[str, ...]
    priority_group: str


def _config(lam: float, T: float, N: int) -> TrajectoryConfig:
    return TrajectoryConfig(lambda_response=lam, T=T, N=N, normalization_id="N0_SOURCE")


def paths() -> tuple[PathSpec, ...]:
    """Seven physical paths, in outcome-independent budget priority order."""
    return (
        PathSpec("S-PATH-T4", _config(.1, 4., 8),
                 ("JV-HOR-T1", "JV-BASE", "JV-HOR-T4"), "BASE_TIME_COARSE_LAMBDA"),
        PathSpec("S-PATH-N2", _config(.1, 2., 2), ("JV-RES-N2",), "BASE_TIME_COARSE_LAMBDA"),
        PathSpec("S-PATH-N8", _config(.1, 2., 8), ("JV-RES-N8",), "BASE_TIME_COARSE_LAMBDA"),
        PathSpec("S-PATH-LAM-001", _config(.01, 2., 4), ("JV-LAM-001",), "BASE_TIME_COARSE_LAMBDA"),
        PathSpec("S-PATH-LAM-1", _config(1., 2., 4), ("JV-LAM-1",), "BASE_TIME_COARSE_LAMBDA"),
        PathSpec("S-PATH-LAM-00316", _config(.03162277660168379, 2., 4),
                 ("JV-LAM-00316",), "INTERMEDIATE_LAMBDA_ACTUAL"),
        PathSpec("S-PATH-LAM-0316", _config(.31622776601683794, 2., 4),
                 ("JV-LAM-0316",), "INTERMEDIATE_LAMBDA_ACTUAL"),
    )


def endpoints() -> tuple[EndpointSpec, ...]:
    """Nine declared outcomes per model, including all five actual lambdas."""
    return (
        EndpointSpec("JV-BASE", "S-PATH-T4", _config(.1, 2., 4), 4, True),
        EndpointSpec("JV-LAM-001", "S-PATH-LAM-001", _config(.01, 2., 4), 4, False),
        EndpointSpec("JV-LAM-00316", "S-PATH-LAM-00316", _config(.03162277660168379, 2., 4), 4, False),
        EndpointSpec("JV-LAM-0316", "S-PATH-LAM-0316", _config(.31622776601683794, 2., 4), 4, False),
        EndpointSpec("JV-LAM-1", "S-PATH-LAM-1", _config(1., 2., 4), 4, False),
        EndpointSpec("JV-RES-N2", "S-PATH-N2", _config(.1, 2., 2), 2, False),
        EndpointSpec("JV-RES-N8", "S-PATH-N8", _config(.1, 2., 8), 8, False),
        EndpointSpec("JV-HOR-T1", "S-PATH-T4", _config(.1, 1., 2), 2, True),
        EndpointSpec("JV-HOR-T4", "S-PATH-T4", _config(.1, 4., 8), 8, False),
    )


def endpoint_binding(endpoint: EndpointSpec, parent: PathSpec) -> dict[str, Any]:
    cfg, pcfg = endpoint.config, parent.config
    if (endpoint.path_id != parent.path_id or endpoint.candidate_id not in parent.endpoint_ids
            or endpoint.completed_nodes != cfg.N or cfg.N > pcfg.N
            or cfg.lambda_response != pcfg.lambda_response or cfg.h != pcfg.h
            or cfg.normalization_id != pcfg.normalization_id
            or endpoint.derived_observation_only != (cfg.N < pcfg.N)):
        raise SweepBoundary("PREFIX_CONFIG_BINDING")
    return {
        "candidate_id": endpoint.candidate_id, "path_id": parent.path_id,
        "effective_T": cfg.T, "effective_N": cfg.N, "h": cfg.h,
        "parent_T": pcfg.T, "parent_N": pcfg.N,
        "lambda_response": cfg.lambda_response,
        "normalization_id": cfg.normalization_id,
        "completed_nodes": endpoint.completed_nodes,
        "derived_observation_only": endpoint.derived_observation_only,
        "history_append_count": 0 if endpoint.derived_observation_only else 1,
        "persistent_endpoint_capture_count": 0 if endpoint.derived_observation_only else 1,
        "prefix_is_sequential_resume_checkpoint": False,
        "actual_gpu_endpoint_required": True,
    }


def dry_plan() -> dict[str, Any]:
    path_rows = paths()
    path_by_id = {row.path_id: row for row in path_rows}
    endpoint_rows = [endpoint_binding(row, path_by_id[row.path_id]) for row in endpoints()]
    if len(path_rows) != 7 or sum(row.config.N for row in path_rows) != 34 or len(endpoint_rows) != 9:
        raise SweepBoundary("S_GRID_COMPLETENESS")
    return {
        "schema": "alpha-jv-diagnosis.s-sweep-plan.v1", "track": "S",
        "models": list(MODELS), "fixture": "S_DEV", "requests_per_model": 100,
        "same_case_ids_and_order_across_models": True,
        "D_exact_replay_dependency": False,
        "fixed_z_capture_per_model_cohort": 1,
        "entry_reference_count_per_model": 1, "O_NATIVE_count_per_model": 1,
        "actual_lambda_values": list(ACTUAL_LAMBDAS),
        "paths_per_model": 7, "nodes_per_model_nominal": 34,
        "main_target_JVP_per_model_nominal": 170,
        "model_count": 2, "JV_endpoint_count": 18,
        "all_endpoint_count_including_entry_and_Official": 22,
        "paths": [dict(asdict(row), h=row.config.h) for row in path_rows],
        "endpoints": endpoint_rows,
        "budget_priority": "both models BASE_TIME_COARSE_LAMBDA, then both INTERMEDIATE_LAMBDA_ACTUAL",
        "optional_N16": "NOT_ENABLED_SEPARATE_MANIFEST_REQUIRED",
        "audit_reserved_execution": "NOT_AUTHORIZED_IN_BASIC_GRID",
        "compute_exclusions_from_nominal": ["model_load", "compute_z", "Official", "entry_reference_solves",
                                             "evaluator", "FD", "history_finalization"],
        "main_target_JVP_actual_reuse": "MEASURE_AT_RUNTIME_NOT_ASSUMED",
        "GPU_hour_cap": None, "GPU_submission_status": "GPU_HOUR_BUDGET_UNASSIGNED",
        "scientific_promotion": False,
    }
