"""Shared executable core for Session 03 CT-K4 P0/P1."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch

_REPO_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_IMPORT_ROOT))

from project.run_scripts.ode_edit_motivation.contracts import EditRequest
from project.run_scripts.ode_edit_motivation.gpu_runtime import (
    load_fixed_model_checkpoint_original,
    offline_environment,
    seed_runtime,
)
from project.run_scripts.ode_edit_motivation.hooks import resolve_parameter, tensor_sha256
from project.run_scripts.ode_edit_method.contracts import MethodContractError, canonical_hash
from project.run_scripts.ode_edit_method.ct_k4 import (
    CT_ARM_ORDER,
    CTArm,
    assert_event_identity,
    assert_shared_direct_z_identities,
    run_ct_arm,
)
from project.run_scripts.ode_edit_method.ct_k4_evaluation import (
    evaluate_frozen_endpoint,
    load_evaluation_payloads,
    make_firewall,
)
from project.run_scripts.ode_edit_method.ct_k4_lock import (
    CT_K4_LOCK_PATH,
    load_ct_k4_lock,
)
from project.run_scripts.ode_edit_method.easyedit_backend import (
    OracleAbsoluteMeanMarginEasyEditBackend,
)
from project.run_scripts.ode_edit_method.event_strength import assert_raw_free
from project.run_scripts.ode_edit_method.hooks import TorchCheckpoint, base_weight_c_energy
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.lock import LOCK_PATH, controller_config, load_lock
from project.run_scripts.ode_edit_method.oracle_absolute_lock import (
    ORACLE_ABSOLUTE_LOCK_PATH,
    load_oracle_absolute_lock,
)
from project.run_scripts.ode_edit_method.preflight import (
    assert_static_lock_identities,
    preflight_static_inputs,
    prepare_concrete_environment,
)
from project.run_scripts.session02_compute_aware_p0 import (
    EASYEDIT_ROOT,
    _file_sha256,
    _git_head,
    _json_write,
    _jsonl_append,
)
from project.run_scripts.session02_compute_aware_p1 import (
    _checkpoint_original_runtime_metadata,
)


EXECUTION_TOKENS = {
    "p0": "session03-ct-k4-p0-v1",
    "p1": "session03-ct-k4-p1-r2-v1",
}
OUTPUT_PREFIXES = {
    "p0": "session03-ct-k4-p0-r3",
    "p1": "session03-ct-k4-p1-r2",
}


@dataclass(frozen=True, slots=True)
class Session03CTSpec:
    """Experiment-local arm and lock wiring for the shared CT runner."""

    slug: str
    lock_path: Path
    load_lock: Callable[[str | Path], dict[str, Any]]
    arm_order: tuple[Any, ...]
    run_arm: Callable[..., Any]
    assert_shared_direct_z: Callable[..., None]
    native_arm: Any
    one_shot_arm: Any
    frozen_arm: Any
    finite_reference_arm: Any
    field_build_counts: Mapping[Any, int]
    field_build_max_counts: Mapping[Any, int]
    first_step_nonzero_arms: frozenset[Any]
    authorization_env_prefix: str
    execution_tokens: Mapping[str, str]
    output_prefixes: Mapping[str, str]


CT_K4_SPEC = Session03CTSpec(
    slug="ct-k4",
    lock_path=CT_K4_LOCK_PATH,
    load_lock=load_ct_k4_lock,
    arm_order=tuple(CT_ARM_ORDER),
    run_arm=run_ct_arm,
    assert_shared_direct_z=assert_shared_direct_z_identities,
    native_arm=CTArm.NATIVE_MEMIT,
    one_shot_arm=CTArm.BF_ONESHOT_FULL,
    frozen_arm=CTArm.BF_FROZEN_CT_K4,
    finite_reference_arm=CTArm.ODE_REFRESH_CT_K4,
    field_build_counts={
        CTArm.BF_FROZEN_CT_K4: 1,
        CTArm.ODE_REFRESH_CT_K4: 4,
    },
    field_build_max_counts={CTArm.ODE_REFRESH_CT_K4_ES: 4},
    first_step_nonzero_arms=frozenset(),
    authorization_env_prefix="ODEEDIT_SESSION03_CT_K4",
    execution_tokens=EXECUTION_TOKENS,
    output_prefixes=OUTPUT_PREFIXES,
)


def expected_output_root(
    stage: str,
    model_alias: str,
    proposal_id: str,
    spec: Session03CTSpec = CT_K4_SPEC,
) -> Path:
    return Path(
        "local/results/"
        f"{spec.output_prefixes[stage]}-{model_alias}-{proposal_id[:8]}"
    )


def _require_session03_output_root(
    repo: Path,
    candidate: Path,
    expected_relative: Path,
) -> Path:
    """Create exactly one sealed Session 03 result root without following links."""

    repo_root = repo.resolve(strict=True)
    if (
        expected_relative.is_absolute()
        or expected_relative.parent != Path("local/results")
        or not expected_relative.name.startswith("session03-")
    ):
        raise ValueError("invalid Session 03 expected output root")
    expected = repo_root / expected_relative
    candidate_absolute = Path(os.path.abspath(os.fspath(candidate)))
    if candidate_absolute != expected:
        raise ValueError("Session 03 output root differs from exact expected path")

    parent = repo_root
    for component in ("local", "results"):
        parent = parent / component
        if os.path.lexists(parent):
            if parent.is_symlink() or not parent.is_dir():
                raise ValueError("Session 03 output parent is not a real directory")
        else:
            parent.mkdir(mode=0o755, exist_ok=False)
    if parent.resolve(strict=True) != repo_root / "local" / "results":
        raise ValueError("Session 03 output parent escaped repository")
    if expected.parent != parent or expected.name != expected_relative.name:
        raise ValueError("Session 03 output basename differs")
    if os.path.lexists(expected):
        raise FileExistsError(f"Session 03 output root already exists: {expected}")
    expected.mkdir(mode=0o755, parents=False, exist_ok=False)
    if expected.is_symlink() or expected.resolve(strict=True) != expected:
        raise ValueError("Session 03 output root is not a sealed directory")
    return expected


def dry_plan(
    lock: Mapping[str, Any],
    stage: str,
    spec: Session03CTSpec = CT_K4_SPEC,
) -> dict[str, Any]:
    resources = lock["resources"]
    return {
        "schema_version": f"ode-edit-session03-{spec.slug}-{stage}-dry-plan/v1",
        "submission_authorized": False,
        "proposal_id": lock["proposal_id"],
        "lock_sha256": lock["lock_sha256"],
        "stage": stage,
        "case_ids": list(lock["selection"][f"{stage}_case_ids"]),
        "arms": [arm.value for arm in spec.arm_order],
        "jobs": [
            {
                "model_alias": alias,
                "output_root": str(
                    expected_output_root(stage, alias, lock["proposal_id"], spec)
                ),
                "resources": {
                    "gpu": resources["gpu_per_job"],
                    "cpu": resources["cpu_per_job"],
                    "host_memory_mib": resources["host_memory_mib_per_job"],
                    "time": resources[f"{stage}_time"],
                },
            }
            for alias in lock["models"]
        ],
        "pair_gpu": resources["pair_gpu"],
        "server1_project_gpu_cap": resources["server1_project_gpu_cap"],
        "p1_evaluation_enabled": stage == "p1",
        "automatic_p1_condition": "both-p0-terminal-technical-pass",
    }


def _evaluation_firewall_metadata(stage: str) -> dict[str, bool]:
    return {
        "enabled": stage == "p1",
        "controller_access": False,
        "action_freeze_required": True,
    }


class _PostActionForwardCounter:
    """Count post-controller top-level forwards without touching controller totals."""

    _SCOPES = ("terminal_residual", "endpoint_metrics")

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        gpu_device: torch.device | str | None = None,
    ) -> None:
        self._model = model
        self._counts = {name: 0 for name in self._SCOPES}
        self._wall_seconds = {name: 0.0 for name in self._SCOPES}
        self._gpu_seconds = {name: 0.0 for name in self._SCOPES}
        self._gpu_events: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._active_scope: str | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._finalized = False
        self._gpu_device: torch.device | None = None
        if gpu_device is not None:
            resolved = torch.device(gpu_device)
            if resolved.type != "cuda" or not torch.cuda.is_available():
                raise MethodContractError("post-action GPU timing device is unavailable")
            self._gpu_device = resolved

    @property
    def attached(self) -> bool:
        return self._handle is not None

    def __enter__(self) -> "_PostActionForwardCounter":
        if self._handle is not None or self._finalized:
            raise MethodContractError("post-action counter lifecycle differs")

        def count(_module: torch.nn.Module, _inputs: Any) -> None:
            if self._active_scope is None:
                raise MethodContractError("unscoped post-action model forward")
            self._counts[self._active_scope] += 1

        self._handle = self._model.register_forward_pre_hook(count)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    @contextmanager
    def scope(self, name: str) -> Iterator[None]:
        if name not in self._counts or self._handle is None or self._finalized:
            raise MethodContractError("post-action forward scope is invalid")
        if self._active_scope is not None:
            raise MethodContractError("post-action forward scopes cannot nest")
        self._active_scope = name
        start = time.perf_counter()
        gpu_start: torch.cuda.Event | None = None
        gpu_end: torch.cuda.Event | None = None
        if self._gpu_device is not None:
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.device(self._gpu_device):
                gpu_start.record()
        try:
            yield
        finally:
            if self._gpu_device is not None:
                assert gpu_start is not None and gpu_end is not None
                with torch.cuda.device(self._gpu_device):
                    gpu_end.record()
                self._gpu_events.append((name, gpu_start, gpu_end))
            self._wall_seconds[name] += time.perf_counter() - start
            self._active_scope = None

    def finalize(self) -> dict[str, Any]:
        if self._handle is not None or self._active_scope is not None or self._finalized:
            raise MethodContractError("post-action counter finalized while active")
        if self._gpu_device is not None:
            torch.cuda.synchronize(self._gpu_device)
            for name, start, end in self._gpu_events:
                self._gpu_seconds[name] += float(start.elapsed_time(end)) / 1000.0
        self._finalized = True
        return {
            "N_post_action_model_fwd": sum(self._counts.values()),
            "post_action_model_fwd_scope_counts": dict(self._counts),
            "post_action_model_fwd_provenance": (
                "top-level-forward-pre-hook-after-controller-detach"
            ),
            "post_action_wall_seconds_by_scope": dict(self._wall_seconds),
            "post_action_gpu_seconds_by_scope": dict(self._gpu_seconds),
        }


class _PostActionEvaluationInstrumentation:
    """Keep logical N_eval while routing endpoint work outside controller timing."""

    def __init__(
        self,
        controller: EditInstrumentation,
        post_action: _PostActionForwardCounter,
    ) -> None:
        self._controller = controller
        self._post_action = post_action

    def increment(self, name: str, amount: int = 1) -> None:
        if name != "N_eval":
            raise MethodContractError("post-action evaluator counter differs")
        self._controller.increment(name, amount)

    @contextmanager
    def component(self, name: str) -> Iterator[None]:
        if name != "evaluation":
            raise MethodContractError("post-action evaluator component differs")
        with self._post_action.scope("endpoint_metrics"):
            yield


def build_parser(
    stage: str, spec: Session03CTSpec = CT_K4_SPEC
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--model-alias",
        required=True,
        choices=("llama3-8b-inst", "qwen2.5-7b-inst"),
    )
    parser.add_argument("--lock", type=Path, default=spec.lock_path)
    parser.add_argument("--base-lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--v3-lock", type=Path, default=ORACLE_ABSOLUTE_LOCK_PATH)
    parser.add_argument("--easyedit-root", type=Path, default=EASYEDIT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.set_defaults(stage=stage)
    return parser


def _request_ids(requests: Sequence[Any]) -> list[str]:
    return [
        EditRequest.from_mapping(
            {
                "case_id": request.case_id,
                "prompt": request.prompt,
                "subject": request.subject,
                "target_new": request.target_new,
            }
        ).request_id
        for request in requests
    ]


def _backend(
    *,
    runtime: Any,
    prepared: Any,
    request: Any,
    output_root: Path,
    config: Any,
    base_lock: Mapping[str, Any],
    metrics: EditInstrumentation,
    record_mechanism: bool,
    finite_reference_gate: bool,
) -> OracleAbsoluteMeanMarginEasyEditBackend:
    cache_root = output_root / "direct_z"
    cache_path = cache_root / f"{runtime.spec.alias}-{request.case_id}.pt"
    return OracleAbsoluteMeanMarginEasyEditBackend(
        runtime=runtime,
        bridge=prepared.bridge,
        hparams=prepared.hparams,
        contexts=prepared.contexts,
        request=request,
        covariance_specs=prepared.covariance_specs,
        covariance_contract=prepared.covariance_contract,
        covariance_by_layer=prepared.covariance_by_layer,
        direct_z_cache_root=cache_root,
        direct_z_cache_path=cache_path,
        tau=config.tau,
        oracle_epsilon=config.event_tolerance,
        unit_c_norm_epsilon=base_lock["controller"]["unit_c_norm_epsilon"],
        unit_c_identity_atol=base_lock["controller"]["unit_c_identity_atol"],
        unit_c_identity_rtol=base_lock["controller"]["unit_c_identity_rtol"],
        instrumentation=metrics,
        finite_difference_gate_epsilon=(
            base_lock["derivative_backend"]["p0_finite_difference_epsilon"]
            if finite_reference_gate
            else None
        ),
        finite_difference_gate_atol=base_lock["derivative_backend"]["p0_scalar_gate_atol"],
        finite_difference_gate_rtol=base_lock["derivative_backend"]["p0_scalar_gate_rtol"],
        record_mechanism=record_mechanism,
    )


def _parameter_hashes(model: torch.nn.Module, weight_names: Sequence[str]) -> dict[str, str]:
    return {
        name: tensor_sha256(resolve_parameter(model, name))
        for name in sorted(weight_names)
    }


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    numerator = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def _mechanism(
    result: Any, backend: Any, *, schema_slug: str = "ct-k4"
) -> dict[str, Any]:
    steps = result.steps
    transitions = []
    allocation_cosines = []
    ranking_changes = []
    support_turnover = []
    for before, after in zip(steps, steps[1:]):
        transitions.append(before.direction_ids != after.direction_ids)
        allocation_cosines.append(
            _cosine(before.corrector_coefficients, after.corrector_coefficients)
        )
        before_rank = tuple(
            sorted(
                range(len(before.corrector_coefficients)),
                key=lambda index: (-before.corrector_coefficients[index], index),
            )
        )
        after_rank = tuple(
            sorted(
                range(len(after.corrector_coefficients)),
                key=lambda index: (-after.corrector_coefficients[index], index),
            )
        )
        ranking_changes.append(before_rank != after_rank)
        before_support = {
            index for index, value in enumerate(before.corrector_coefficients) if value > 0.0
        }
        after_support = {
            index for index, value in enumerate(after.corrector_coefficients) if value > 0.0
        }
        union = before_support | after_support
        support_turnover.append(
            0.0 if not union else len(before_support ^ after_support) / len(union)
        )
    return {
        "schema_version": f"ode-edit-session03-{schema_slug}-mechanism/v1",
        "direction_id_transition_rate": (
            None if not transitions else sum(transitions) / len(transitions)
        ),
        "allocation_cosines": allocation_cosines,
        "ranking_drift_rate": (
            None if not ranking_changes else sum(ranking_changes) / len(ranking_changes)
        ),
        "support_turnover": support_turnover,
        "field_history": list(backend.mechanism_field_history),
        "exact_c_cosine_deferred": True,
        "retained_factor_tensors": False,
    }


def _total_energy(energy: Mapping[int, float]) -> float:
    value = math.fsum(float(item) for item in energy.values())
    if not math.isfinite(value) or value < 0.0:
        raise MethodContractError("terminal C-energy is invalid")
    return value


def _assert_frozen_endpoint_identity(
    *,
    one_shot_hashes: Mapping[str, str] | None,
    frozen_hashes: Mapping[str, str],
    one_shot_event: Any,
    frozen_event: Any,
    one_shot_energy: Mapping[int, float] | None,
    frozen_energy: Mapping[int, float],
    one_shot_evaluation: Mapping[str, Any] | None,
    frozen_evaluation: Mapping[str, Any] | None,
    atol: float,
    rtol: float,
) -> None:
    """Lock the frozen cumulative endpoint to its one-shot BF reference."""

    if one_shot_hashes is None or dict(frozen_hashes) != dict(one_shot_hashes):
        raise RuntimeError("BF frozen endpoint bytes differ from one-shot")
    assert_event_identity(one_shot_event, frozen_event, atol=atol, rtol=rtol)
    if one_shot_energy is None or dict(frozen_energy) != dict(one_shot_energy):
        raise RuntimeError("BF frozen terminal C-energy differs from one-shot")
    if frozen_evaluation != one_shot_evaluation:
        raise RuntimeError("BF frozen endpoint evaluation differs from one-shot")


def _assert_case_direct_z_accounting(
    spec: Session03CTSpec,
    source_identity: Mapping[str, Any],
    arm_identities: Mapping[Any, Mapping[str, Any]],
    n_z_by_arm: Mapping[Any, int],
) -> None:
    """Derive, rather than assume, the one-compute shared direct-z contract."""

    if not spec.arm_order or spec.arm_order[0] is not spec.native_arm:
        raise RuntimeError("native arm must own the case/model direct-z compute")
    expected_counts = {
        arm: 1 if arm is spec.native_arm else 0 for arm in spec.arm_order
    }
    if dict(n_z_by_arm) != expected_counts:
        raise RuntimeError("case/model direct-z counter distribution differs")
    spec.assert_shared_direct_z(
        source_identity,
        arm_identities,
        global_n_z=sum(n_z_by_arm.values()),
    )


def _validate_source_locks(
    lock_path: Path,
    base_lock_path: Path,
    v3_lock_path: Path,
    spec: Session03CTSpec = CT_K4_SPEC,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock = spec.load_lock(lock_path)
    base = load_lock(base_lock_path)
    v3 = load_oracle_absolute_lock(v3_lock_path)
    pinned = lock["base_method"]
    if (
        base["proposal_id"] != pinned["proposal_id"]
        or _file_sha256(base_lock_path) != pinned["lock_sha256"]
        or v3["proposal_id"] != pinned["v3_event_proposal_id"]
        or _file_sha256(v3_lock_path) != pinned["v3_event_lock_sha256"]
    ):
        raise MethodContractError("CT-K4 base/V3 source identities differ")
    return lock, base, v3


def run(
    args: argparse.Namespace, spec: Session03CTSpec = CT_K4_SPEC
) -> int:
    stage = str(args.stage)
    if stage not in {"p0", "p1"}:
        raise RuntimeError("unknown CT-K4 stage")
    repo = Path(__file__).resolve().parents[2]
    easyedit_root = args.easyedit_root.resolve(strict=True)
    if easyedit_root != EASYEDIT_ROOT.resolve(strict=True):
        raise RuntimeError("CT-K4 EasyEdit root differs")
    lock_path = args.lock.resolve(strict=True)
    base_lock_path = args.base_lock.resolve(strict=True)
    v3_lock_path = args.v3_lock.resolve(strict=True)
    lock, base_lock, _v3_lock = _validate_source_locks(
        lock_path, base_lock_path, v3_lock_path, spec
    )
    plan = dry_plan(lock, stage, spec)
    expected_relative = expected_output_root(
        stage, args.model_alias, lock["proposal_id"], spec
    )
    expected = (repo / expected_relative).resolve()
    candidate = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    if candidate.resolve() != expected:
        raise RuntimeError("CT-K4 output root differs from dry plan")
    if args.dry_run:
        print(json.dumps(plan, allow_nan=False, sort_keys=True))
        return 0
    token_name = f"{spec.authorization_env_prefix}_{stage.upper()}_AUTHORIZED"
    if os.environ.get(token_name) != spec.execution_tokens[stage]:
        raise RuntimeError("separate CT-K4 stage execution authority is required")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key}=1 is required")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("CT-K4 requires exactly one visible GPU")

    output_root = _require_session03_output_root(
        repo, candidate, expected_relative
    )
    for name in (
        "controller_steps.jsonl",
        "compute.jsonl",
        "mechanism.jsonl",
        "evaluation.jsonl",
    ):
        (output_root / name).touch(exist_ok=False)
    case_ids = tuple(lock["selection"][f"{stage}_case_ids"])
    expected_request_ids = tuple(lock["selection"][f"{stage}_request_ids"])
    seed = int(lock["policy"]["seed"])
    seed_runtime(seed)
    setup_start = time.perf_counter()
    fixed_artifacts, requests = preflight_static_inputs(
        easyedit_root, model_alias=args.model_alias, case_ids=case_ids
    )
    if tuple(_request_ids(requests)) != expected_request_ids:
        raise RuntimeError("CT-K4 request identities differ")
    model_lock = base_lock["models"][args.model_alias]
    assert_static_lock_identities(
        easyedit_root,
        fixed_artifacts=fixed_artifacts,
        model_lock=model_lock,
        selection_lock=base_lock["selection"],
    )
    model_load_start = time.perf_counter()
    with offline_environment():
        runtime = load_fixed_model_checkpoint_original(args.model_alias)
    model_load_seconds = time.perf_counter() - model_load_start
    runtime_metadata = _checkpoint_original_runtime_metadata(runtime)
    prepared = prepare_concrete_environment(
        easyedit_root,
        runtime=runtime,
        fixed_artifacts=fixed_artifacts,
        controller_requests=requests,
        expected_model_lock=model_lock,
        seed=seed,
    )
    config = controller_config(base_lock)
    weight_by_layer = {
        layer: f"{prepared.hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in prepared.runtime.spec.layers
    }
    denominators = base_weight_c_energy(
        runtime.model,
        prepared.covariance_by_layer,
        weight_by_layer,
        denominator_epsilon=config.load_denominator_epsilon,
    )
    baseline = TorchCheckpoint.capture(
        runtime.model, tuple(weight_by_layer.values()), backup_device="cpu"
    )
    private_evaluation = (
        load_evaluation_payloads(easyedit_root, requests) if stage == "p1" else {}
    )
    setup_seconds = time.perf_counter() - setup_start - model_load_seconds
    manifest = {
        "schema_version": f"ode-edit-session03-{spec.slug}-{stage}-manifest/v1",
        "status": "RUNNING",
        "instruction_id": lock["instruction_id"],
        "git": {
            "commit": _git_head(repo),
            "proposal_id": lock["proposal_id"],
            "lock_sha256": lock["lock_sha256"],
        },
        "model": runtime_metadata,
        "selection": {
            "case_ids": list(case_ids),
            "request_ids": list(expected_request_ids),
            "arms": [arm.value for arm in spec.arm_order],
            "seed": seed,
            "cases_independent_atomic_reset": True,
        },
        "policy": dict(lock["policy"]),
        "context_manifest_id": prepared.contexts.manifest_id,
        "fixed_artifact_manifest_id": fixed_artifacts.manifest_id,
        "easyedit_source_manifest_id": prepared.bridge.load().provenance.manifest_id,
        "trial_backend": base_lock["trial_backend"]["selected_common_backend"],
        "event_backend": "direct-z-oracle-absolute-new-uniform-mean-margin-v3",
        "evaluation_firewall": _evaluation_firewall_metadata(stage),
        "setup_wall_seconds": setup_seconds,
        "model_load_wall_seconds_excluded": model_load_seconds,
        "server1_project_gpu_cap": 3,
    }
    assert_raw_free(manifest)
    _json_write(output_root / "manifest.json", manifest)

    summaries: list[dict[str, Any]] = []
    try:
        for request in requests:
            baseline.restore(runtime.model)
            baseline.assert_exact(runtime.model, include_rng=True)
            shared_target = None
            shared_source_identity: Mapping[str, Any] | None = None
            shared_entry_residual: float | None = None
            direct_identities: dict[CTArm, Mapping[str, Any]] = {}
            direct_z_counts: dict[CTArm, int] = {}
            one_shot_hashes: Mapping[str, str] | None = None
            one_shot_event = None
            one_shot_energy: Mapping[int, float] | None = None
            one_shot_evaluation: Mapping[str, Any] | None = None
            case_rows: list[dict[str, Any]] = []
            for arm in spec.arm_order:
                baseline.restore(runtime.model)
                baseline.assert_exact(runtime.model, include_rng=True)
                metrics = EditInstrumentation(
                    f"session03:{stage}:{args.model_alias}:{request.case_id}:{arm.value}",
                    gpu_timing=True,
                    gpu_device="cuda:0",
                )
                metrics.add_wall_seconds(
                    "context_setup", setup_seconds / (len(requests) * len(CT_ARM_ORDER))
                )
                backend = _backend(
                    runtime=runtime,
                    prepared=prepared,
                    request=request,
                    output_root=output_root,
                    config=config,
                    base_lock=base_lock,
                    metrics=metrics,
                    record_mechanism=arm is not spec.native_arm,
                    finite_reference_gate=(
                        stage == "p0" and arm is spec.finite_reference_arm
                    ),
                )
                metrics.attach_model(runtime.model)
                post_action_payload: Mapping[str, Any] | None = None
                try:
                    if shared_target is None:
                        with metrics.component("direct_z"):
                            shared_target = backend.compute_direct_z(request)
                        metrics.increment("N_z")
                        with metrics.component("direct_z"):
                            shared_entry_residual = backend.direct_z_residual_norm(
                                shared_target
                            )
                        shared_source_identity = dict(backend.direct_z_identity or {})
                    else:
                        backend.attach_shared_direct_z(shared_target)
                    identity = dict(backend.direct_z_identity or {})
                    direct_identities[arm] = identity
                    result = spec.run_arm(
                        arm,
                        request=request,
                        backend=backend,
                        frozen_target=shared_target,
                        denominators=denominators,
                        config=config,
                        instrumentation=metrics,
                    )
                    metrics.detach_model()
                    if metrics.tracks_model_forwards:
                        raise RuntimeError("controller model-forward hook remained attached")
                    post_action = _PostActionForwardCounter(
                        runtime.model, gpu_device="cuda:0"
                    )
                    with post_action:
                        firewall = make_firewall(
                            request,
                            private_evaluation.get(request.case_id),
                        )
                        action_hash = firewall.freeze_action(
                            {
                                "arm": arm.value,
                                "case_id": request.case_id,
                                "terminal_state_id": result.terminal_state_id,
                                "step_count": len(result.steps),
                            }
                        )
                        with post_action.scope("terminal_residual"):
                            terminal_residual = backend.direct_z_residual_norm(
                                shared_target
                            )
                        endpoint_hashes = _parameter_hashes(
                            runtime.model, tuple(weight_by_layer.values())
                        )
                        evaluation_row = None
                        if stage == "p1":
                            endpoint = TorchCheckpoint.capture(
                                runtime.model,
                                tuple(weight_by_layer.values()),
                                backup_device="cpu",
                            )
                            evaluation_row = evaluate_frozen_endpoint(
                                runtime=runtime,
                                hparams=prepared.hparams,
                                request=request,
                                firewall=firewall,
                                baseline_checkpoint=baseline,
                                endpoint_checkpoint=endpoint,
                                instrumentation=(
                                    _PostActionEvaluationInstrumentation(
                                        metrics, post_action
                                    )
                                ),
                            )
                            endpoint.assert_exact(runtime.model, include_rng=True)
                    post_action_payload = post_action.finalize()
                    first_step_nonzero = bool(
                        result.steps
                        and result.steps[0].source_state_id
                        != result.steps[0].terminal_state_id
                    )
                    if (
                        arm in spec.first_step_nonzero_arms
                        and not first_step_nonzero
                    ):
                        raise RuntimeError(
                            "fixed-horizon first step preserved target-weight state"
                        )
                    if arm is spec.one_shot_arm:
                        one_shot_hashes = dict(endpoint_hashes)
                        one_shot_event = result.terminal_event
                        one_shot_energy = dict(result.terminal_net_energy)
                        one_shot_evaluation = evaluation_row
                    elif arm is spec.frozen_arm:
                        _assert_frozen_endpoint_identity(
                            one_shot_hashes=one_shot_hashes,
                            frozen_hashes=endpoint_hashes,
                            one_shot_event=one_shot_event,
                            frozen_event=result.terminal_event,
                            one_shot_energy=one_shot_energy,
                            frozen_energy=result.terminal_net_energy,
                            one_shot_evaluation=one_shot_evaluation,
                            frozen_evaluation=evaluation_row,
                            atol=config.functional_commit_atol,
                            rtol=config.functional_commit_rtol,
                        )
                    mechanism = _mechanism(
                        result, backend, schema_slug=spec.slug
                    )
                    target = backend.oracle_target
                    terminal_mean_new = math.fsum(
                        result.terminal_event.target_new_log_likelihoods
                    ) / len(result.terminal_event.target_new_log_likelihoods)
                    terminal_mean_margin = math.fsum(
                        result.terminal_event.context_margins
                    ) / len(result.terminal_event.context_margins)
                    q_new = (
                        terminal_mean_new - target.entry_mean_new
                    ) / target.new_denominator
                    q_margin = terminal_mean_margin / target.oracle_mean_margin
                finally:
                    if metrics.tracks_model_forwards:
                        metrics.detach_model()
                if post_action_payload is None:
                    raise RuntimeError("post-action accounting is absent")
                snapshot = metrics.finalize().to_dict()
                counters = snapshot["counters"]
                direct_z_counts[arm] = counters["N_z"]
                if counters["N_eval"] != (0 if stage == "p0" else 5):
                    raise RuntimeError("CT-K4 evaluation counter differs")
                expected_fields = spec.field_build_counts.get(arm)
                if expected_fields is not None and counters["N_field"] != expected_fields:
                    raise RuntimeError("fixed-horizon field count differs")
                maximum_fields = spec.field_build_max_counts.get(arm)
                if maximum_fields is not None and counters["N_field"] > maximum_fields:
                    raise RuntimeError("fixed-horizon field count exceeds cap")
                if arm is not spec.native_arm and counters["N_bw"] != counters["N_field"]:
                    raise RuntimeError("fixed-horizon one-backward field count differs")
                record = {
                    "model": args.model_alias,
                    "stage": stage,
                    "case_id": request.case_id,
                    "arm": arm.value,
                    "action_hash": action_hash,
                    "result": result.to_dict(),
                    "direct_z_identity": identity,
                    "direct_z_entry_residual_norm": shared_entry_residual,
                    "direct_z_terminal_residual_norm": terminal_residual,
                    "direct_z_terminal_residual_fraction": (
                        None
                        if not shared_entry_residual
                        else terminal_residual / shared_entry_residual
                    ),
                    "terminal_mean_new": terminal_mean_new,
                    "terminal_mean_margin": terminal_mean_margin,
                    "q_new": q_new,
                    "q_margin_diagnostic_only": q_margin,
                    "endpoint_parameter_hashes": endpoint_hashes,
                    "oracle_calibration": backend.oracle_calibration.to_dict(),
                    "oracle_target": backend.oracle_target.to_dict(),
                    "hook_reference_gate": (
                        None
                        if backend.hook_reference_gate is None
                        else list(backend.hook_reference_gate)
                    ),
                }
                if spec.first_step_nonzero_arms:
                    record["first_step_target_weight_state_change_nonzero"] = (
                        first_step_nonzero
                    )
                compute_row = {
                    "schema_version": f"ode-edit-session03-{spec.slug}-compute/v1",
                    "model": args.model_alias,
                    "stage": stage,
                    "case_id": request.case_id,
                    "arm": arm.value,
                    **snapshot,
                    **counters,
                    **post_action_payload,
                }
                mechanism_row = {
                    "model": args.model_alias,
                    "stage": stage,
                    "case_id": request.case_id,
                    "arm": arm.value,
                    **mechanism,
                }
                for row in (record, compute_row, mechanism_row):
                    assert_raw_free(row)
                _jsonl_append(output_root / "controller_steps.jsonl", record)
                _jsonl_append(output_root / "compute.jsonl", compute_row)
                _jsonl_append(output_root / "mechanism.jsonl", mechanism_row)
                if evaluation_row is not None:
                    evaluation_serialized = {
                        "model": args.model_alias,
                        "stage": stage,
                        "arm": arm.value,
                        **evaluation_row,
                    }
                    assert_raw_free(evaluation_serialized)
                    _jsonl_append(
                        output_root / "evaluation.jsonl", evaluation_serialized
                    )
                case_row = {
                    "case_id": request.case_id,
                    "arm": arm.value,
                    "status": result.status,
                    "step_count": len(result.steps),
                    "first_hit_step": result.first_hit_step,
                    "terminal_hard_phi": result.terminal_event.hard_phi,
                    "terminal_mean_margin": terminal_mean_margin,
                    "q_new": q_new,
                    "q_margin_diagnostic_only": q_margin,
                    "terminal_c_energy": _total_energy(result.terminal_net_energy),
                    "direct_z_residual_fraction": record["direct_z_terminal_residual_fraction"],
                    "controller_wall_seconds": snapshot["controller_wall_seconds"],
                    "controller_gpu_seconds": snapshot["controller_gpu_seconds"],
                    "peak_memory_allocated_bytes": snapshot["peak_memory_allocated_bytes"],
                    "peak_memory_reserved_bytes": snapshot["peak_memory_reserved_bytes"],
                    "evaluation_metrics": evaluation_row,
                }
                if spec.first_step_nonzero_arms:
                    case_row["first_step_target_weight_state_change_nonzero"] = (
                        first_step_nonzero
                    )
                case_rows.append(case_row)
                summaries.append(case_row)
                baseline.restore(runtime.model)
                baseline.assert_exact(runtime.model, include_rng=True)
            assert shared_source_identity is not None
            _assert_case_direct_z_accounting(
                spec,
                shared_source_identity,
                direct_identities,
                direct_z_counts,
            )
            by_arm = {row["arm"]: row for row in case_rows}
            native = by_arm[spec.native_arm.value]
            for row in case_rows:
                row["c_energy_ratio_to_native"] = (
                    None
                    if native["terminal_c_energy"] == 0.0
                    else row["terminal_c_energy"] / native["terminal_c_energy"]
                )
                row["gpu_ratio_to_native"] = (
                    None
                    if native["controller_gpu_seconds"] == 0.0
                    else row["controller_gpu_seconds"]
                    / native["controller_gpu_seconds"]
                )
    except BaseException:
        baseline.restore(runtime.model)
        baseline.assert_exact(runtime.model, include_rng=True)
        raise

    baseline.restore(runtime.model)
    baseline.assert_exact(runtime.model, include_rng=True)
    fixed_artifacts.assert_current()
    prepared.bridge.load().provenance.assert_current()
    prepared.covariance_contract.manifest.assert_current()
    endpoint_aggregates: dict[str, Any] = {}
    if stage == "p1":
        metric_names = (
            "efficacy_token_accuracy",
            "generalization_token_accuracy",
            "locality_pre_post_token_agreement",
            "neighborhood_target_true_token_accuracy",
        )
        for arm in spec.arm_order:
            rows = [row for row in summaries if row["arm"] == arm.value]
            successful = [
                row
                for row in rows
                if row["evaluation_metrics"]["efficacy_token_accuracy"]
                >= 1.0 - 1e-12
            ]
            endpoint_aggregates[arm.value] = {
                "all_request_count": len(rows),
                "success_condition": "efficacy_token_accuracy==1",
                "success_conditioned_count": len(successful),
                "all_request": {
                    name: math.fsum(
                        row["evaluation_metrics"][name] for row in rows
                    )
                    / len(rows)
                    for name in metric_names
                },
                "success_conditioned": (
                    None
                    if not successful
                    else {
                        name: math.fsum(
                            row["evaluation_metrics"][name]
                            for row in successful
                        )
                        / len(successful)
                        for name in metric_names
                    }
                ),
            }
    summary = {
        "schema_version": f"ode-edit-session03-{spec.slug}-{stage}-summary/v1",
        "status": "COMPLETE_TECHNICAL_PASS",
        "model": args.model_alias,
        "stage": stage,
        "records": summaries,
        "endpoint_aggregates": endpoint_aggregates,
        "technical_pass": True,
        "p1_automatic_submission_eligible": stage == "p0",
        "evaluation_count": 0 if stage == "p0" else len(summaries),
        "generation_count": 0,
    }
    assert_raw_free(summary)
    _json_write(output_root / "summary.json", summary)
    manifest["status"] = "COMPLETE_TECHNICAL_PASS"
    _json_write(output_root / "manifest.json", manifest)
    files = [
        output_root / "manifest.json",
        output_root / "controller_steps.jsonl",
        output_root / "compute.jsonl",
        output_root / "mechanism.jsonl",
        output_root / "evaluation.jsonl",
        output_root / "summary.json",
        *sorted((output_root / "direct_z").glob("*.pt")),
    ]
    _json_write(
        output_root / "terminal_manifest.json",
        {
            "schema_version": f"ode-edit-session03-{spec.slug}-terminal/v1",
            "status": "COMPLETE",
            "files": {
                str(path.relative_to(output_root)): {
                    "sha256": _file_sha256(path),
                    "size": path.stat().st_size,
                }
                for path in files
            },
        },
    )
    return 0
