"""Concrete common-model backend for the Session 04 P1 matched pilot."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import resource
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .accounting import ComputeLedger, assert_matched_call_identity
from .contracts import (
    ARM_ORDER,
    MODEL_ALIASES,
    Arm,
    ODEAllocContractError,
    canonical_hash,
    canonical_json,
)
from .events import RewriteGateConfig, RewriteGateReading
from .frozen_inputs import FrozenInputReceipt, OneShotEditInputs
from .functional import (
    FP32FactorOverlay,
    QuantizedBF16FunctionalTrial,
    quantized_effective_weight,
)
from .gauge import FactorPair, FixedEnergyGauge, QDomainError, factor_gram_energy
from .p0_artifacts import P0ArtifactGuard, sha256_file
from .p0_runtime import (
    ComponentTimer,
    ForwardCounter,
    ScorePanel,
    _capture_native_factors,
    _event,
    _event_record,
    _flatten_contexts,
    _fresh_contexts_twice,
    _load_original_bf16,
    _seed_all,
    factors_sha256,
    score_panel,
    tensor_sha256,
)
from .p1_contracts import (
    EXPECTED_BASE,
    INNER_MICROBATCH_SIZE,
    P1_CASE_ORDER,
    P1_INSTRUCTION_ID,
    P1_REQUEST_HASHES,
    P1_RUN_TOKEN,
    SEED,
    SEAL_ROOT,
    SESSION_ID,
    DamageAggregate,
    EndpointFreezeToken,
    HistoryRequest,
    P1Policy,
    assert_p1_seal,
    differentiable_damage_aggregate,
    expected_p1_result_name,
    historical_damage_aggregate,
    pretrained_damage_aggregate,
    remove_obsolete_requests,
    request_key,
)
from .p1_evaluator import (
    evaluate_frozen_endpoint,
    load_evaluation_cases_after_freeze,
)
from .p1_scoring import TeacherForcedScorer, TeacherForcedSpec
from .selection import (
    assert_seal_source_current,
    load_and_verify_seal_candidate,
    load_projected_request,
)
from .solver import (
    CBFProjector,
    CandidateVerdict,
    ConstraintLinearization,
    EvaluationCost,
    FieldEvaluation,
    FixedGridSolver,
    GenericProjector,
    LinearizationEvaluation,
    ProjectionResult,
    SolverEndpoint,
    VelocityProjector,
)
from .transaction import AtomicLayerTransaction


NUMERICAL_LOCK_SHA256 = "905a3bbccfb71a5b780586cde49503bde88df92bf8af378d8b88a85b4245f37d"
ARTIFACT_LOCK_SHA256 = "6c327c563e0e805eb73a09cf3a577dea3771bc44127748fe64aba70a803292d2"
SEAL_FILE_SHA256 = "48e8c96acee8e95ffc2b82689c314b4b9f84c8a6979431c8bf03f8987e2b549e"


@contextlib.contextmanager
def _discard_easyedit_console_output() -> Any:
    """Prevent EasyEdit's request-bearing prints from entering Slurm logs."""

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield


def _write_canonical_once(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError("P1 metadata is create-once")
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise
    return hashlib.sha256(encoded).hexdigest()


class ProgressRecorder:
    """Raw-free create-once P1 progress receipts for initial monitoring."""

    def __init__(self, raw_root: Path) -> None:
        self.root = raw_root / "stages"
        self.root.mkdir(mode=0o700)
        self.completed: list[tuple[str, str]] = []
        self.wall_seconds = 0.0

    def complete(self, stage: str, **safe: Any) -> str:
        if (
            not stage
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in stage)
            or any(key in {"prompt", "subject", "target", "text", "tokens"} for key in safe)
        ):
            raise ODEAllocContractError("P1 progress receipt is not raw-free")
        ordinal = len(self.completed)
        payload = {
            "schema_version": "ode-alloc-s04-p1-progress/v1",
            "ordinal": ordinal,
            "stage": stage,
            **safe,
        }
        started = time.perf_counter()
        try:
            digest = _write_canonical_once(
                self.root / f"{ordinal:03d}-{stage}.json", payload
            )
        finally:
            self.wall_seconds += time.perf_counter() - started
        self.completed.append((stage, digest))
        return digest

    @property
    def last_stage(self) -> str | None:
        return self.completed[-1][0] if self.completed else None

    def root_sha256(self) -> str:
        return canonical_hash(tuple(self.completed))


@dataclass(slots=True)
class RecordingProjector:
    inner: VelocityProjector
    records: list[dict[str, Any]]

    @property
    def name(self) -> str:
        return self.inner.name

    def project(
        self,
        nominal: torch.Tensor,
        linearization: ConstraintLinearization,
    ) -> ProjectionResult:
        result = self.inner.project(nominal, linearization)
        self.records.append(
            {
                "ordinal": len(self.records),
                "labels": linearization.labels,
                "barrier_values": tuple(float(value) for value in linearization.barrier_values),
                "nominal_linf": float(torch.max(torch.abs(nominal))),
                "projected_linf": float(torch.max(torch.abs(result.velocity))),
                "slack": tuple(float(value) for value in result.slack),
                "slack_sum": float(result.slack.sum()) if result.slack.numel() else 0.0,
                "slack_max": float(result.slack.max()) if result.slack.numel() else 0.0,
                "qp_cpu_seconds": result.qp_cpu_seconds,
                "max_constraint_residual": result.max_constraint_residual,
                "active_subset_count": len(result.subset_diagnostics),
                "all_subset_kkt_valid": all(
                    item.kkt_valid for item in result.subset_diagnostics
                ),
            }
        )
        return result


def _ledger_state(ledger: ComputeLedger) -> dict[str, int | float]:
    return {
        name: getattr(ledger, name)
        for name in ledger.__dataclass_fields__
    }


def _accumulate_ledger_delta(
    target: ComputeLedger,
    before: Mapping[str, int | float],
    after: Mapping[str, int | float],
) -> dict[str, int | float]:
    if set(before) != set(after) or set(before) != set(target.__dataclass_fields__):
        raise ODEAllocContractError("P1 controller accounting schema differs")
    delta: dict[str, int | float] = {}
    for name in target.__dataclass_fields__:
        observed = after[name] - before[name]
        if observed < 0:
            raise ODEAllocContractError("P1 controller accounting regressed")
        delta[name] = observed
        if name == "peak_memory_bytes":
            target.observe_peak_memory(int(after[name]))
        elif isinstance(getattr(target, name), float):
            target.add_seconds(name, float(observed))
        else:
            target.increment(name, int(observed))
    return delta


def _matched_call_differences(
    left: ComputeLedger, right: ComputeLedger
) -> dict[str, tuple[int, int]]:
    fields = (
        "model_forward_calls",
        "processed_tokens",
        "backward_calls",
        "constraint_vjp_calls",
        "constraint_jvp_calls",
        "quantized_trial_calls",
        "commit_count",
    )
    return {
        name: (int(getattr(left, name)), int(getattr(right, name)))
        for name in fields
        if getattr(left, name) != getattr(right, name)
    }


@dataclass(frozen=True, slots=True)
class ExactReading:
    q: tuple[float, ...]
    ratios: tuple[float, ...]
    rewrite: RewriteGateReading
    historical: DamageAggregate
    pretrained: DamageAggregate
    feasible: bool
    objective: float
    q_cap_hit: bool
    event_id: str
    panel_id: str | None

    def record(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "ratios": self.ratios,
            "q_nonzero": any(value != 0.0 for value in self.q),
            "q_cap_hit": self.q_cap_hit,
            "rewrite": _event_record(self.rewrite),
            "historical": self.historical.record(),
            "pretrained": self.pretrained.record(),
            "feasible": self.feasible,
            "objective": self.objective,
            "event_id": self.event_id,
            "preservation_panel_id": self.panel_id,
        }


@dataclass(slots=True)
class EditBasis:
    factors_by_weight: dict[str, FactorPair]
    active_factors: dict[str, FactorPair]
    gauge: FixedEnergyGauge
    entry_panel: ScorePanel
    direct_z_panel: ScorePanel
    native_panel: ScorePanel
    native_event: RewriteGateReading
    native_weight_hashes: dict[str, str]
    entry_weight_hashes: dict[str, str]
    entry_pointers: dict[str, int]
    frozen_receipt: FrozenInputReceipt
    underlying_counts: dict[str, int]
    replay_writer_calls: int
    oneshot: OneShotEditInputs


def _weight_root(parameters: Mapping[str, torch.Tensor]) -> str:
    return canonical_hash(
        {name: tensor_sha256(value) for name, value in sorted(parameters.items())}
    )


def _source_freeze(repo: Path, source_head: str) -> None:
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    if observed != source_head or source_head == EXPECTED_BASE:
        raise ODEAllocContractError("P1 source HEAD differs from the frozen checkpoint")
    changed = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "project/run_scripts/ode_alloc",
            "project/run_scripts/session04_ode_alloc_p1.py",
            "project/run_scripts/session04_ode_alloc_p1.sbatch",
            "project/run_scripts/session04_ode_alloc_p1_dry_plan.py",
            "project/run_scripts/session04_ode_alloc_submit_p1.py",
        ],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    if changed:
        raise ODEAllocContractError("P1 frozen source contains changed paths")


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(result, dict):
        raise ODEAllocContractError("P1 JSON root is not an object")
    return result


def _rewrite_specs(
    request: Mapping[str, Any],
    prompt_templates: Sequence[tuple[str, str]],
) -> tuple[tuple[TeacherForcedSpec, ...], tuple[str, ...], tuple[str, ...]]:
    specs: list[TeacherForcedSpec] = []
    new_ids: list[str] = []
    old_ids: list[str] = []
    for context_id, template in prompt_templates:
        prompt = template.format(str(request["subject"]))
        new_id = f"E-new-{context_id}"
        old_id = f"E-old-{context_id}"
        specs.extend(
            (
                TeacherForcedSpec(new_id, prompt, str(request["target_new"])),
                TeacherForcedSpec(old_id, prompt, str(request["target_old"])),
            )
        )
        new_ids.append(new_id)
        old_ids.append(old_id)
    return tuple(specs), tuple(new_ids), tuple(old_ids)


def _preservation_specs(
    history: Sequence[HistoryRequest],
    anchors: Sequence[Mapping[str, Any]],
    history_prompt_templates: Sequence[Sequence[tuple[str, str]]],
) -> tuple[
    tuple[TeacherForcedSpec, ...],
    tuple[tuple[str, ...], ...],
    tuple[str, ...],
]:
    if len(history) != len(history_prompt_templates):
        raise ODEAllocContractError("P1 historical rewrite-context panels differ")
    specs: list[TeacherForcedSpec] = []
    history_groups: list[tuple[str, ...]] = []
    for item, prompt_templates in zip(
        history, history_prompt_templates, strict=True
    ):
        if len(prompt_templates) != 6:
            raise ODEAllocContractError("P1 historical rewrite-context count differs")
        ids: list[str] = []
        for context_id, template in prompt_templates:
            item_id = f"H-{item.case_id}-{context_id}"
            specs.append(
                TeacherForcedSpec(
                    item_id,
                    template.format(str(item.request["subject"])),
                    str(item.request["target_new"]),
                )
            )
            ids.append(item_id)
        history_groups.append(tuple(ids))
    anchor_ids: list[str] = []
    for index, request in enumerate(anchors):
        item_id = f"P-{index}"
        specs.append(
            TeacherForcedSpec(
                item_id,
                str(request["prompt"]).format(str(request["subject"])),
                str(request["target_old"]),
            )
        )
        anchor_ids.append(item_id)
    return tuple(specs), tuple(history_groups), tuple(anchor_ids)


def _means(
    values: Mapping[str, torch.Tensor], groups: Sequence[Sequence[str]]
) -> tuple[torch.Tensor, ...]:
    result: list[torch.Tensor] = []
    for group in groups:
        if not group:
            raise ODEAllocContractError("P1 preservation group is empty")
        result.append(torch.stack(tuple(values[item_id] for item_id in group)).mean())
    return tuple(result)


def _prepare_edit_basis(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    request: Mapping[str, Any],
    hparams: Any,
    guard: P0ArtifactGuard,
    ledger: ComputeLedger,
    prompt_templates: Sequence[tuple[str, str]],
    touched: Mapping[str, torch.nn.Parameter],
    mutation_lock: threading.RLock,
    policy: P1Policy,
) -> EditBasis:
    entry_weight_hashes = {name: tensor_sha256(value) for name, value in touched.items()}
    entry_pointers = {name: value.data_ptr() for name, value in touched.items()}
    entry_panel = score_panel(model, tokenizer, prompt_templates, request)
    oneshot = OneShotEditInputs(f"counterfact:{request['case_id']}")
    capture: dict[str, Any] = {}

    def build() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        with _discard_easyedit_console_output():
            result = _capture_native_factors(
                model, tokenizer, request, hparams, guard, ledger
            )
        capture["direct_z"] = result[1]
        capture["keys"] = result[2]
        capture["covariance"] = result[3]
        capture["counts"] = result[4]
        capture["memit_main"] = result[5]
        return result[0]

    deltas = oneshot.capture_from("native_factors", build, factors_sha256)
    direct_z = oneshot.capture_from(
        "direct_z", lambda: capture["direct_z"], tensor_sha256
    )
    oneshot.capture_from(
        "key",
        lambda: capture["keys"],
        lambda values: canonical_hash(
            {str(layer): tensor_sha256(value) for layer, value in sorted(values.items())}
        ),
    )
    oneshot.capture_from(
        "covariance", lambda: capture["covariance"], canonical_hash
    )
    frozen_receipt = oneshot.seal()
    if {name: tensor_sha256(value) for name, value in touched.items()} != entry_weight_hashes:
        raise ODEAllocContractError("P1 Native factor capture did not restore arm state")
    factors_by_weight: dict[str, FactorPair] = {}
    weight_names = tuple(touched)
    for layer, name in zip(hparams.layers, weight_names, strict=True):
        key_matrix, residual = deltas[name]
        factors_by_weight[name] = FactorPair(int(layer), residual, key_matrix)
    z_module = hparams.layer_module_tmp.format(hparams.layers[-1])
    direct_z_panel = score_panel(
        model,
        tokenizer,
        prompt_templates,
        request,
        direct_z=direct_z,
        z_module_name=z_module,
        fact_token=hparams.fact_token,
    )
    gate = RewriteGateConfig(policy.rho, policy.denominator_epsilon)
    memit_main = capture["memit_main"]
    original_execute = memit_main.execute_memit
    replay_calls = 0

    def replay_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal replay_calls
        del args, kwargs
        replay_calls += 1
        if replay_calls != 1:
            raise ODEAllocContractError("P1 Native factor replay repeated")
        return deltas

    memit_main.execute_memit = replay_once
    try:
        with _discard_easyedit_console_output():
            _, snapshots = memit_main.apply_memit_to_model(
                model,
                tokenizer,
                [dict(request)],
                hparams,
                copy=False,
                return_orig_weights=True,
                cache_template=None,
            )
    finally:
        memit_main.execute_memit = original_execute
    if replay_calls != 1 or set(snapshots) != set(touched):
        raise ODEAllocContractError("P1 independent Native writer receipt differs")
    native_weight_hashes = {name: tensor_sha256(value) for name, value in touched.items()}
    native_panel = score_panel(model, tokenizer, prompt_templates, request)
    native_event = _event(native_panel, entry_panel, direct_z_panel, gate)
    exact_zero: dict[int, bool] = {}
    for name, pair in factors_by_weight.items():
        expected = quantized_effective_weight(snapshots[name], pair, 1.0)
        if not torch.equal(expected, touched[name]):
            raise ODEAllocContractError("P1 q=0 bytes differ from EasyEdit Native")
        if factor_gram_energy(pair) == 0.0:
            exact_zero[pair.layer] = bool(torch.equal(expected, snapshots[name]))
        del expected
    with torch.no_grad():
        for name in touched:
            touched[name].copy_(snapshots[name])
    snapshots.clear()
    if (
        {name: tensor_sha256(value) for name, value in touched.items()}
        != entry_weight_hashes
        or any(touched[name].data_ptr() != entry_pointers[name] for name in touched)
    ):
        raise ODEAllocContractError("P1 Native replay restore differs")
    gauge = FixedEnergyGauge(
        {pair.layer: pair for pair in factors_by_weight.values()},
        basis_energy_epsilon=policy.basis_energy_epsilon,
        max_abs_centered_q=policy.max_abs_centered_q,
        quantized_zero_by_layer=exact_zero,
    )
    zero_reading = gauge.evaluate(gauge.zeros())
    if not torch.equal(zero_reading.ratios, torch.ones_like(zero_reading.ratios)):
        raise ODEAllocContractError("P1 q=0 gauge ratios differ")
    active_factors = {
        name: pair
        for name, pair in factors_by_weight.items()
        if pair.layer in gauge.layers
    }
    del direct_z
    return EditBasis(
        factors_by_weight=factors_by_weight,
        active_factors=active_factors,
        gauge=gauge,
        entry_panel=entry_panel,
        direct_z_panel=direct_z_panel,
        native_panel=native_panel,
        native_event=native_event,
        native_weight_hashes=native_weight_hashes,
        entry_weight_hashes=entry_weight_hashes,
        entry_pointers=entry_pointers,
        frozen_receipt=frozen_receipt,
        underlying_counts=dict(capture["counts"]),
        replay_writer_calls=replay_calls,
        oneshot=oneshot,
    )


class EditProblem:
    """One common callback implementation shared by Generic and ODE."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        request: Mapping[str, Any],
        prompt_templates: Sequence[tuple[str, str]],
        history: Sequence[HistoryRequest],
        history_prompt_templates: Sequence[Sequence[tuple[str, str]]],
        anchors: Sequence[Mapping[str, Any]],
        theta0_anchor_teacher: Sequence[float],
        basis: EditBasis,
        policy: P1Policy,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.request = request
        self.basis = basis
        self.policy = policy
        self.prompt_templates = tuple(prompt_templates)
        self.rewrite_specs, self.new_ids, self.old_ids = _rewrite_specs(
            request, prompt_templates
        )
        (
            self.preservation_specs,
            self.history_groups,
            self.anchor_ids,
        ) = _preservation_specs(history, anchors, history_prompt_templates)
        if len(theta0_anchor_teacher) != 16:
            raise ODEAllocContractError("theta0 anchor teacher count differs")
        self.theta0_anchor_teacher = tuple(float(value) for value in theta0_anchor_teacher)
        scorer = TeacherForcedScorer(
            model, tokenizer, microbatch_size=INNER_MICROBATCH_SIZE
        )
        entry = scorer.score(self.preservation_specs, gradient=False)
        history_values = _means(entry.logp, self.history_groups)
        self.history_teacher = tuple(
            float(value.detach().to(device="cpu")) for value in history_values
        )
        self.readings: dict[str, ExactReading] = {}
        self.trial_count = 0
        self.feasible_trial_count = 0
        self.violation_trial_count = 0
        self.q_cap_hit_count = 0
        self.max_live_effective_bf16_weights = 0
        self.max_fp32_delta_block_elements = 0
        self.replacement_linear_calls = 0
        self._linearization_cache: dict[tuple[int, str], ConstraintLinearization] = {}

    def _q_reading(
        self, q: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[float, ...], dict[int, float], bool]:
        centered = q.to(dtype=torch.float64, device="cpu") - q.mean()
        try:
            reading = self.basis.gauge.evaluate(
                {
                    layer: centered[index]
                    for index, layer in enumerate(self.basis.gauge.layers)
                }
            )
            q_cap_hit = False
        except QDomainError:
            reading = self.basis.gauge.evaluate(self.basis.gauge.zeros())
            q_cap_hit = True
            self.q_cap_hit_count += 1
        ratios = {
            layer: float(value.detach().to(device="cpu"))
            for layer, value in reading.ratio_by_layer().items()
        }
        q_values = tuple(float(value) for value in centered)
        return centered, q_values, ratios, q_cap_hit

    def _build_exact(
        self,
        *,
        q_values: tuple[float, ...],
        ratios: Mapping[int, float],
        q_cap_hit: bool,
        rewrite_panel: ScorePanel,
        preservation: Mapping[str, torch.Tensor],
        panel_id: str | None,
    ) -> ExactReading:
        rewrite = _event(
            rewrite_panel,
            self.basis.entry_panel,
            self.basis.direct_z_panel,
            RewriteGateConfig(self.policy.rho, self.policy.denominator_epsilon),
        )
        history_values = _means(preservation, self.history_groups)
        historical_damage = tuple(
            max(0.0, teacher - float(value.detach().to(device="cpu")))
            for value, teacher in zip(
                history_values, self.history_teacher, strict=True
            )
        )
        anchor_values = tuple(preservation[item_id] for item_id in self.anchor_ids)
        pretrained_damage = tuple(
            max(0.0, teacher - float(value.detach().to(device="cpu")))
            for value, teacher in zip(
                anchor_values, self.theta0_anchor_teacher, strict=True
            )
        )
        historical = historical_damage_aggregate(historical_damage, policy=self.policy)
        pretrained = pretrained_damage_aggregate(pretrained_damage, policy=self.policy)
        objective = 0.5 * (historical.normalized + pretrained.normalized)
        feasible = (
            not q_cap_hit
            and rewrite.feasible
            and historical.feasible
            and pretrained.feasible
        )
        payload = {
            "schema": "ode-alloc-s04-p1-exact-event/v1",
            "q": q_values,
            "ratios": tuple(ratios[layer] for layer in self.basis.gauge.layers),
            "q_cap_hit": q_cap_hit,
            "rewrite_event_id": rewrite.event_id,
            "historical": historical.record(),
            "pretrained": pretrained.record(),
            "feasible": feasible,
            "objective": objective if not q_cap_hit else 1.0e300,
            "panel_id": panel_id,
        }
        event_id = canonical_hash(payload)
        return ExactReading(
            q=q_values,
            ratios=tuple(ratios[layer] for layer in self.basis.gauge.layers),
            rewrite=rewrite,
            historical=historical,
            pretrained=pretrained,
            feasible=feasible,
            objective=objective if not q_cap_hit else 1.0e300,
            q_cap_hit=q_cap_hit,
            event_id=event_id,
            panel_id=panel_id,
        )

    def exact_candidate(self, q: torch.Tensor) -> CandidateVerdict:
        centered, q_values, ratios, q_cap_hit = self._q_reading(q)
        trial = QuantizedBF16FunctionalTrial(
            self.model, self.basis.active_factors, ratios, row_block=64
        )
        with trial:
            rewrite_panel = score_panel(
                self.model,
                self.tokenizer,
                tuple(
                    (context_id, template)
                    for context_id, template in self._rewrite_prompt_templates()
                ),
                self.request,
            )
            preservation_result = TeacherForcedScorer(
                self.model,
                self.tokenizer,
                microbatch_size=INNER_MICROBATCH_SIZE,
            ).score(self.preservation_specs, gradient=False)
        self.max_live_effective_bf16_weights = max(
            self.max_live_effective_bf16_weights,
            trial.max_live_effective_weights,
        )
        self.max_fp32_delta_block_elements = max(
            self.max_fp32_delta_block_elements,
            trial.max_fp32_delta_block_elements,
        )
        self.replacement_linear_calls += trial.replacement_linear_calls
        reading = self._build_exact(
            q_values=q_values,
            ratios=ratios,
            q_cap_hit=q_cap_hit,
            rewrite_panel=rewrite_panel,
            preservation=preservation_result.logp,
            panel_id=preservation_result.panel_id,
        )
        if all(value == 0.0 for value in centered):
            if rewrite_panel != self.basis.native_panel or reading.rewrite != self.basis.native_event:
                raise ODEAllocContractError("P1 q=0 functional output/event differs from Native")
        self.readings[reading.event_id] = reading
        self.trial_count += 1
        self.feasible_trial_count += int(reading.feasible)
        self.violation_trial_count += int(not reading.feasible)
        return CandidateVerdict(
            feasible=reading.feasible,
            objective=reading.objective,
            event_id=reading.event_id,
        )

    def _rewrite_prompt_templates(self) -> tuple[tuple[str, str], ...]:
        return self.prompt_templates

    def _gradient_state(
        self, q: torch.Tensor
    ) -> tuple[torch.Tensor, ConstraintLinearization, EvaluationCost]:
        q_variable = (
            q.detach().to(dtype=torch.float64, device="cpu").clone().requires_grad_(True)
        )
        gauge = self.basis.gauge.evaluate(
            {
                layer: q_variable[index]
                for index, layer in enumerate(self.basis.gauge.layers)
            }
        )
        ratio_tensors = gauge.ratio_by_layer()
        specs = self.rewrite_specs + self.preservation_specs
        with FP32FactorOverlay(
            self.model, self.basis.active_factors, ratio_tensors
        ):
            values = TeacherForcedScorer(
                self.model,
                self.tokenizer,
                microbatch_size=INNER_MICROBATCH_SIZE,
            ).score(specs, gradient=True).logp
            candidate_new = torch.stack(tuple(values[item_id] for item_id in self.new_ids)).mean()
            candidate_old = torch.stack(tuple(values[item_id] for item_id in self.old_ids)).mean()
            mean_margin = candidate_new - candidate_old
            entry_new = sum(self.basis.entry_panel.new.values()) / len(self.basis.entry_panel.new)
            direct_z_new = sum(self.basis.direct_z_panel.new.values()) / len(
                self.basis.direct_z_panel.new
            )
            ratio_barrier = candidate_new - entry_new - self.policy.rho * (
                direct_z_new - entry_new
            )
            h_e = torch.minimum(mean_margin, ratio_barrier)
            history_values = _means(values, self.history_groups)
            historical = differentiable_damage_aggregate(
                history_values,
                self.history_teacher,
                fixed_scale=self.policy.historical_scale,
                smooth_temperature=self.policy.smooth_temperature,
                zero_reference=q_variable,
            )
            anchor_values = tuple(values[item_id] for item_id in self.anchor_ids)
            pretrained = differentiable_damage_aggregate(
                anchor_values,
                self.theta0_anchor_teacher,
                fixed_scale=self.policy.pretrained_scale,
                smooth_temperature=self.policy.smooth_temperature,
                zero_reference=q_variable,
            )
            preservation_objective = 0.5 * (
                historical.aggregate / self.policy.historical_scale
                + pretrained.aggregate / self.policy.pretrained_scale
            )
            objective = -mean_margin if float(h_e.detach().to(device="cpu")) < 0.0 else preservation_objective
            nominal = -torch.autograd.grad(
                objective, q_variable, retain_graph=True
            )[0]
            barrier_tensors = (h_e, historical.barrier_value, pretrained.barrier_value)
            rows = tuple(
                torch.autograd.grad(value, q_variable, retain_graph=True)[0]
                for value in barrier_tensors
            )
        nominal = nominal.detach().to(device="cpu", dtype=torch.float64)
        nominal = nominal - nominal.mean()
        maximum = float(torch.max(torch.abs(nominal)))
        if maximum > self.policy.max_abs_centered_q:
            nominal = nominal * (self.policy.max_abs_centered_q / maximum)
        matrix = torch.stack(
            tuple(
                (row.detach().to(device="cpu", dtype=torch.float64) - row.detach().to(device="cpu", dtype=torch.float64).mean())
                for row in rows
            )
        )
        linearization = ConstraintLinearization(
            matrix=matrix,
            barrier_values=torch.tensor(
                tuple(float(value.detach().to(device="cpu")) for value in barrier_tensors),
                dtype=torch.float64,
            ),
            kappa=self.policy.cbf_kappa,
            labels=("E", "H", "P"),
        )
        return (
            nominal,
            linearization,
            EvaluationCost(backward_calls=4, constraint_vjp_calls=3),
        )

    def direction(self, step: int, q: torch.Tensor) -> FieldEvaluation:
        nominal, linearization, cost = self._gradient_state(q)
        key = (step, canonical_hash(tuple(float(value) for value in q)))
        if key in self._linearization_cache:
            raise ODEAllocContractError("P1 linearization cache repeats")
        self._linearization_cache[key] = linearization
        return FieldEvaluation(nominal, cost)

    def linearize(self, step: int, q: torch.Tensor) -> LinearizationEvaluation:
        key = (step, canonical_hash(tuple(float(value) for value in q)))
        try:
            value = self._linearization_cache.pop(key)
        except KeyError as exc:
            raise ODEAllocContractError("P1 matched linearization cache is absent") from exc
        return LinearizationEvaluation(value, EvaluationCost())

    def score_committed(self, q: torch.Tensor) -> ExactReading:
        _, q_values, ratios, q_cap_hit = self._q_reading(q)
        if q_cap_hit:
            raise ODEAllocContractError("committed P1 endpoint cannot hit q cap")
        rewrite_panel = score_panel(
            self.model,
            self.tokenizer,
            self._rewrite_prompt_templates(),
            self.request,
        )
        preservation = TeacherForcedScorer(
            self.model,
            self.tokenizer,
            microbatch_size=INNER_MICROBATCH_SIZE,
        ).score(self.preservation_specs, gradient=False)
        return self._build_exact(
            q_values=q_values,
            ratios=ratios,
            q_cap_hit=False,
            rewrite_panel=rewrite_panel,
            preservation=preservation.logp,
            panel_id=preservation.panel_id,
        )


def _projector_for_arm(arm: Arm, policy: P1Policy) -> RecordingProjector:
    if arm is Arm.GENERIC_ADAPTIVE:
        inner: VelocityProjector = GenericProjector()
    elif arm is Arm.ODE_ALLOC:
        inner = CBFProjector(
            slack_penalty_weight=policy.slack_penalty_weight,
            residual_tolerance=policy.qp_residual_tolerance,
        )
    else:
        raise ODEAllocContractError("adaptive projector requested for Native")
    return RecordingProjector(inner=inner, records=[])


def _commit_endpoint(
    *,
    touched: Mapping[str, torch.nn.Parameter],
    basis: EditBasis,
    ratios: Mapping[int, float],
    mutation_lock: threading.RLock,
    ledger: ComputeLedger,
) -> dict[str, str]:
    fault = AtomicLayerTransaction(touched, mutation_lock=mutation_lock)
    for name, pair in basis.factors_by_weight.items():
        candidate = quantized_effective_weight(
            touched[name], pair, ratios.get(pair.layer, 1.0)
        )
        fault.stage(name, candidate)
        del candidate
    raised = False
    try:
        fault.commit(fault_after_writes=max(1, len(touched) // 2))
    except RuntimeError as exc:
        if "injected" not in str(exc):
            raise
        raised = True
    if (
        not raised
        or {name: tensor_sha256(value) for name, value in touched.items()}
        != basis.entry_weight_hashes
    ):
        raise ODEAllocContractError("P1 edit-local fault rollback differs")
    transaction = AtomicLayerTransaction(touched, mutation_lock=mutation_lock)
    for name, pair in basis.factors_by_weight.items():
        candidate = quantized_effective_weight(
            touched[name], pair, ratios.get(pair.layer, 1.0)
        )
        transaction.stage(name, candidate)
        del candidate
    receipt = transaction.commit(ledger=ledger)
    if receipt.commit_count != 1 or receipt.rolled_back:
        raise ODEAllocContractError("P1 endpoint transaction receipt differs")
    if any(
        touched[name].data_ptr() != basis.entry_pointers[name]
        or touched[name].grad is not None
        or touched[name].requires_grad
        for name in touched
    ):
        raise ODEAllocContractError("P1 endpoint pointer/grad contract differs")
    return {name: tensor_sha256(value) for name, value in touched.items()}


def _restore_exact(
    touched: Mapping[str, torch.nn.Parameter],
    snapshots: Mapping[str, torch.Tensor],
    *,
    mutation_lock: threading.RLock,
) -> None:
    transaction = AtomicLayerTransaction(touched, mutation_lock=mutation_lock)
    for name in touched:
        transaction.stage(name, snapshots[name])
    transaction.commit()


def _run_arm(
    *,
    arm: Arm,
    model: torch.nn.Module,
    tokenizer: Any,
    hparams: Any,
    requests: Sequence[Mapping[str, Any]],
    request_hashes: Sequence[str],
    anchors: Sequence[Mapping[str, Any]],
    theta0_anchor_teacher: Sequence[float],
    context_templates: Sequence[Sequence[str]],
    guard: P0ArtifactGuard,
    policy: P1Policy,
    source_head: str,
    progress: ProgressRecorder,
    mutation_lock: threading.RLock,
) -> tuple[dict[str, Any], ComputeLedger, ComputeLedger]:
    weight_names = tuple(
        f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in hparams.layers
    )
    parameters = dict(model.named_parameters())
    touched = {name: parameters[name] for name in weight_names}
    arm_pointers = {name: value.data_ptr() for name, value in touched.items()}
    arm_snapshots = {
        name: value.detach().to(device="cpu").clone() for name, value in touched.items()
    }
    arm_entry_hashes = {name: tensor_sha256(value) for name, value in touched.items()}
    ledger = ComputeLedger()
    matched_controller_ledger = ComputeLedger()
    counter = ForwardCounter(model, ledger)
    history: list[HistoryRequest] = []
    edit_rows: list[dict[str, Any]] = []
    restored = False
    diagnostic_before = progress.wall_seconds
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    arm_started = time.perf_counter()
    try:
        for edit_index, (request, request_hash) in enumerate(
            zip(requests, request_hashes, strict=True)
        ):
            kept, obsolete = remove_obsolete_requests(history, request)
            prompt_templates = _flatten_contexts(
                context_templates, str(request["prompt"])
            )
            basis = _prepare_edit_basis(
                model=model,
                tokenizer=tokenizer,
                request=request,
                hparams=hparams,
                guard=guard,
                ledger=ledger,
                prompt_templates=prompt_templates,
                touched=touched,
                mutation_lock=mutation_lock,
                policy=policy,
            )
            progress.complete(
                f"{arm.value}-edit-{edit_index}-basis",
                arm=arm.value,
                edit_index=edit_index,
                case_id=int(request["case_id"]),
                layer_count=len(basis.factors_by_weight),
                active_layer_count=len(basis.active_factors),
                factor_receipt_id=basis.frozen_receipt.receipt_id,
            )
            history_prompt_templates = tuple(
                _flatten_contexts(
                    context_templates, str(item.request["prompt"])
                )
                for item in kept
            )
            controller_before = _ledger_state(ledger)
            problem = EditProblem(
                model=model,
                tokenizer=tokenizer,
                request=request,
                prompt_templates=prompt_templates,
                history=kept,
                history_prompt_templates=history_prompt_templates,
                anchors=anchors,
                theta0_anchor_teacher=theta0_anchor_teacher,
                basis=basis,
                policy=policy,
            )
            initial_q = torch.zeros(
                len(basis.gauge.layers), dtype=torch.float64, device="cpu"
            )
            projector_records: list[dict[str, Any]] = []
            if arm is Arm.NATIVE:
                verdict = problem.exact_candidate(initial_q)
                endpoint = SolverEndpoint(
                    q=initial_q,
                    verdict=verdict,
                    fallback_to_native=False,
                    completed_steps=0,
                    projector_name="native-memit-q0",
                )
                ledger.increment("quantized_trial_calls")
            else:
                projector = _projector_for_arm(arm, policy)
                endpoint = FixedGridSolver(
                    budget=policy.solver_budget,
                    projector=projector,
                ).run(
                    initial_q,
                    direction_fn=problem.direction,
                    linearize_fn=problem.linearize,
                    verdict_fn=problem.exact_candidate,
                    ledger=ledger,
                )
                projector_records = projector.records
            try:
                selected = problem.readings[endpoint.verdict.event_id]
            except KeyError as exc:
                raise ODEAllocContractError("P1 selected exact reading is absent") from exc
            ratio_map = {
                layer: ratio
                for layer, ratio in zip(
                    basis.gauge.layers, selected.ratios, strict=True
                )
            }
            committed_hashes = _commit_endpoint(
                touched=touched,
                basis=basis,
                ratios=ratio_map,
                mutation_lock=mutation_lock,
                ledger=ledger,
            )
            committed = problem.score_committed(endpoint.q)
            controller_accounting = _accumulate_ledger_delta(
                matched_controller_ledger,
                controller_before,
                _ledger_state(ledger),
            )
            if committed.event_id != selected.event_id:
                raise ODEAllocContractError("P1 committed output/event differs from exact verdict")
            if all(value == 0.0 for value in selected.q) and committed_hashes != basis.native_weight_hashes:
                raise ODEAllocContractError("P1 committed q=0 bytes differ from Native")
            basis.oneshot.assert_current()
            if problem.max_live_effective_bf16_weights != 1:
                raise ODEAllocContractError("P1 verdict retained multiple BF16 effective weights")
            maximum_weight = max(touched[name].numel() for name in basis.active_factors)
            if problem.max_fp32_delta_block_elements >= maximum_weight:
                raise ODEAllocContractError("P1 verdict retained a dense FP32 delta")
            history = list(kept) + [
                HistoryRequest(
                    case_id=int(request["case_id"]),
                    request_hash=request_hash,
                    request_key=request_key(request),
                    request=dict(request),
                )
            ]
            edit_row = {
                "edit_index": edit_index,
                "case_id": int(request["case_id"]),
                "request_hash": request_hash,
                "history_hard_count": len(kept),
                "history_obsolete_audit_only_count": len(obsolete),
                "history_obsolete_case_ids": tuple(item.case_id for item in obsolete),
                "basis": {
                    "layers": basis.gauge.layers,
                    "dimension": basis.gauge.dimension,
                    "energies": tuple(float(value) for value in basis.gauge.energies),
                    "excluded_exact_zero_layers": basis.gauge.excluded_layers,
                    "frozen_receipt_id": basis.frozen_receipt.receipt_id,
                    "capture_counts": dict(basis.frozen_receipt.capture_counts),
                    "underlying_counts": basis.underlying_counts,
                    "native_replay_writer_calls": basis.replay_writer_calls,
                },
                "endpoint": committed.record(),
                "fallback_to_native": endpoint.fallback_to_native,
                "completed_steps": endpoint.completed_steps,
                "projector_name": endpoint.projector_name,
                "projection": projector_records,
                "matched_controller_accounting": controller_accounting,
                "trials": {
                    "count": problem.trial_count,
                    "feasible_count": problem.feasible_trial_count,
                    "violation_count": problem.violation_trial_count,
                    "q_cap_hit_count": problem.q_cap_hit_count,
                    "fixed_budget": (
                        1
                        if arm is Arm.NATIVE
                        else policy.solver_budget.quantized_trial_budget
                    ),
                },
                "functional": {
                    "max_live_effective_bf16_weights": problem.max_live_effective_bf16_weights,
                    "max_fp32_delta_block_elements": problem.max_fp32_delta_block_elements,
                    "full_fp32_dense_delta_retained": 0,
                    "gradient_overlay_used_as_verdict": False,
                    "replacement_linear_calls": problem.replacement_linear_calls,
                },
                "committed_weight_root_sha256": canonical_hash(committed_hashes),
                "fault_injected_rollback_exact": True,
                "logical_commit_count": 1,
            }
            edit_rows.append(edit_row)
            progress.complete(
                f"{arm.value}-edit-{edit_index}-commit",
                arm=arm.value,
                edit_index=edit_index,
                case_id=int(request["case_id"]),
                endpoint_event_id=committed.event_id,
                q_nonzero=any(value != 0.0 for value in committed.q),
                feasible=committed.feasible,
                fallback_to_native=endpoint.fallback_to_native,
            )
            del problem, basis
            torch.cuda.empty_cache()
        action_sha = canonical_hash(
            tuple(
                {
                    "case_id": row["case_id"],
                    "event_id": row["endpoint"]["event_id"],
                    "q": row["endpoint"]["q"],
                    "fallback": row["fallback_to_native"],
                }
                for row in edit_rows
            )
        )
        parameter_sha = _weight_root(touched)
        freeze = EndpointFreezeToken.create(
            arm=arm,
            source_head=source_head,
            parameter_sha256=parameter_sha,
            action_sha256=action_sha,
            edit_count=len(edit_rows),
        )
        end_event.record()
        torch.cuda.synchronize()
        controller_gpu_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
        controller_wall_seconds = (
            time.perf_counter() - arm_started - (progress.wall_seconds - diagnostic_before)
        )
        if controller_wall_seconds < ledger.controller_wall_seconds:
            raise ODEAllocContractError("P1 controller timing exclusion differs")
        ledger.add_seconds(
            "controller_wall_seconds",
            controller_wall_seconds - ledger.controller_wall_seconds,
        )
        ledger.add_seconds("gpu_seconds", controller_gpu_seconds)
        controller_peak_allocated = int(torch.cuda.max_memory_allocated(0))
        controller_peak_reserved = int(torch.cuda.max_memory_reserved(0))
        ledger.observe_peak_memory(controller_peak_allocated)
        counter.close()
        progress.complete(
            f"{arm.value}-endpoint-frozen",
            arm=arm.value,
            freeze_token_id=freeze.token_id,
            parameter_sha256=parameter_sha,
            edit_count=len(edit_rows),
        )
        evaluation_cases = load_evaluation_cases_after_freeze(
            guard.dataset,
            freeze=freeze,
            approved=tuple(
                {"case_id": case_id, "request_hash": request_hash}
                for case_id, request_hash in zip(
                    P1_CASE_ORDER, P1_REQUEST_HASHES, strict=True
                )
            ),
            inner_requests=requests,
        )
        evaluation = evaluate_frozen_endpoint(
            model,
            tokenizer,
            freeze=freeze,
            cases=evaluation_cases,
            anchor_requests=anchors,
            theta0_anchor_teacher=theta0_anchor_teacher,
        )
        endpoint_peak_allocated = int(torch.cuda.max_memory_allocated(0))
        endpoint_peak_reserved = int(torch.cuda.max_memory_reserved(0))
        progress.complete(
            f"{arm.value}-endpoint-evaluated",
            arm=arm.value,
            freeze_token_id=freeze.token_id,
            evaluation_id=evaluation["evaluation_id"],
            n_eval_specs=evaluation["accounting"]["n_eval_specs"],
        )
        _restore_exact(touched, arm_snapshots, mutation_lock=mutation_lock)
        restored = True
        if (
            {name: tensor_sha256(value) for name, value in touched.items()}
            != arm_entry_hashes
            or any(
                touched[name].data_ptr() != arm_pointers[name]
                or touched[name].grad is not None
                or touched[name].requires_grad
                for name in touched
            )
        ):
            raise ODEAllocContractError("P1 arm terminal W0 restore differs")
        progress.complete(
            f"{arm.value}-arm-restored",
            arm=arm.value,
            restored_weight_root_sha256=_weight_root(touched),
        )
        if ledger.commit_count != 4:
            raise ODEAllocContractError("P1 arm logical commit count differs")
        expected_trials = 4 if arm is Arm.NATIVE else 4 * policy.solver_budget.quantized_trial_budget
        if ledger.quantized_trial_calls != expected_trials:
            raise ODEAllocContractError("P1 arm quantized trial budget differs")
        record = {
            "arm": arm.value,
            "arm_order": tuple(item.value for item in ARM_ORDER),
            "edit_count": len(edit_rows),
            "edits": edit_rows,
            "endpoint_freeze": {
                "token_id": freeze.token_id,
                "parameter_sha256": freeze.parameter_sha256,
                "action_sha256": freeze.action_sha256,
            },
            "evaluation": evaluation,
            "accounting": ledger.snapshot(),
            "matched_controller_accounting": matched_controller_ledger.snapshot(),
            "memory": {
                "controller_peak_allocated_bytes": controller_peak_allocated,
                "controller_peak_reserved_bytes": controller_peak_reserved,
                "endpoint_peak_allocated_bytes": endpoint_peak_allocated,
                "endpoint_peak_reserved_bytes": endpoint_peak_reserved,
                "host_max_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                )
                * 1024,
            },
            "transaction": {
                "logical_commit_count": ledger.commit_count,
                "fault_injected_rollback_count": len(edit_rows),
                "arm_terminal_w0_restore_exact": True,
                "parameter_pointer_grad_contract": True,
            },
            "instrumentation": {
                "progress_wall_seconds": progress.wall_seconds - diagnostic_before,
                "excluded_from_controller_wall": True,
            },
        }
        record["arm_id"] = canonical_hash(record)
        return record, ledger, matched_controller_ledger
    finally:
        with contextlib.suppress(Exception):
            counter.close()
        if not restored:
            with torch.no_grad():
                for name in touched:
                    touched[name].copy_(arm_snapshots[name])
            observed = {name: tensor_sha256(value) for name, value in touched.items()}
            if observed != arm_entry_hashes:
                raise ODEAllocContractError("P1 exception cleanup W0 restore differs")
        arm_snapshots.clear()


def run_p1(
    *,
    repo_root: Path,
    alias: str,
    output_root: Path,
    source_head: str,
    numerical_lock_path: Path,
    artifact_lock_path: Path,
    seal_path: Path,
    run_token: str,
) -> dict[str, Any]:
    started = time.time()
    if alias not in MODEL_ALIASES:
        raise ODEAllocContractError("P1 model alias is not approved")
    if run_token != P1_RUN_TOKEN:
        raise ODEAllocContractError("P1 run token differs")
    _source_freeze(repo_root, source_head)
    numerical_lock_path = numerical_lock_path.resolve(strict=True)
    artifact_lock_path = artifact_lock_path.resolve(strict=True)
    seal_path = seal_path.resolve(strict=True)
    if (
        sha256_file(numerical_lock_path) != NUMERICAL_LOCK_SHA256
        or sha256_file(artifact_lock_path) != ARTIFACT_LOCK_SHA256
        or sha256_file(seal_path) != SEAL_FILE_SHA256
    ):
        raise ODEAllocContractError("P1 lock/seal file digest differs")
    lock = _strict_json(numerical_lock_path)
    policy = P1Policy.from_lock(lock)
    expected_name = expected_p1_result_name(alias, NUMERICAL_LOCK_SHA256, run_token)
    destination = output_root.resolve(strict=False)
    allowed_parent = (repo_root / "local" / "odealloc" / "results").resolve(strict=True)
    if (
        destination.parent != allowed_parent
        or destination.name != expected_name
        or destination.exists()
        or destination.is_symlink()
    ):
        raise ODEAllocContractError("P1 output root identity differs")
    destination.mkdir(mode=0o700)
    raw_root = destination / "raw"
    raw_root.mkdir(mode=0o700)
    progress = ProgressRecorder(raw_root)
    guard = P0ArtifactGuard(artifact_lock_path, alias)
    artifact_receipt = guard.preflight()
    seal = load_and_verify_seal_candidate(seal_path)
    assert_p1_seal(seal)
    assert_seal_source_current(seal, guard.dataset)
    requests = tuple(
        load_projected_request(
            guard.dataset,
            case_id=case_id,
            expected_request_hash=request_hash,
        )
        for case_id, request_hash in zip(
            P1_CASE_ORDER, P1_REQUEST_HASHES, strict=True
        )
    )
    anchor_approved = tuple(seal["pretrained_anchor"])
    anchors = tuple(
        load_projected_request(
            guard.dataset,
            case_id=int(item["case_id"]),
            expected_request_hash=str(item["request_hash"]),
        )
        for item in anchor_approved
    )
    _seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(0)
    setup_timers = ComponentTimer()
    with setup_timers.measure("model_load"):
        model, tokenizer, hparams = _load_original_bf16(guard)
    progress.complete(
        "model-loaded",
        alias=alias,
        observed_dtype=str(next(model.parameters()).dtype),
        visible_gpu_count=torch.cuda.device_count(),
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )
    setup_ledger = ComputeLedger()
    setup_counter = ForwardCounter(model, setup_ledger)
    try:
        with setup_timers.measure("context_generation_twice"):
            with _discard_easyedit_console_output():
                context_templates, context_first, context_second = (
                    _fresh_contexts_twice(model, tokenizer)
                )
    finally:
        setup_counter.close()
    _write_canonical_once(
        raw_root / "context_templates.json",
        {
            "schema_version": "ode-alloc-s04-p1-raw-contexts/v1",
            "contexts": context_templates,
        },
    )
    prompt_templates = _flatten_contexts(
        context_templates, str(requests[0]["prompt"])
    )
    progress.complete(
        "contexts-locked",
        context_count=len(prompt_templates),
        first_sha256=context_first,
        second_sha256=context_second,
        exact_match=context_first == context_second,
    )
    weight_names = tuple(
        f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in hparams.layers
    )
    parameters = dict(model.named_parameters())
    touched = {name: parameters[name] for name in weight_names}
    global_w0_hashes = {name: tensor_sha256(value) for name, value in touched.items()}
    mutation_lock = threading.RLock()
    setup_counter = ForwardCounter(model, setup_ledger)
    anchor_specs = tuple(
        TeacherForcedSpec(
            f"theta0-anchor-{index}",
            str(request["prompt"]).format(str(request["subject"])),
            str(request["target_old"]),
        )
        for index, request in enumerate(anchors)
    )
    try:
        with setup_timers.measure("theta0_anchor_teacher"):
            theta0_result = TeacherForcedScorer(
                model, tokenizer, microbatch_size=INNER_MICROBATCH_SIZE
            ).score(anchor_specs, gradient=False)
    finally:
        setup_counter.close()
    theta0_anchor_teacher = tuple(
        float(theta0_result.logp[spec.item_id].detach().to(device="cpu"))
        for spec in anchor_specs
    )
    setup_ledger.add_seconds(
        "controller_wall_seconds", sum(setup_timers.wall.values())
    )
    setup_ledger.add_seconds("gpu_seconds", sum(setup_timers.gpu.values()))
    setup_ledger.observe_peak_memory(int(torch.cuda.max_memory_allocated(0)))
    progress.complete(
        "theta0-anchors-scored",
        anchor_count=len(theta0_anchor_teacher),
        panel_id=theta0_result.panel_id,
        model_forward_calls=setup_ledger.model_forward_calls,
        processed_token_count=setup_ledger.processed_tokens,
    )
    arm_records: list[dict[str, Any]] = []
    ledgers: dict[Arm, ComputeLedger] = {}
    matched_ledgers: dict[Arm, ComputeLedger] = {}
    try:
        for arm in ARM_ORDER:
            _seed_all(SEED)
            record, ledger, matched_ledger = _run_arm(
                arm=arm,
                model=model,
                tokenizer=tokenizer,
                hparams=hparams,
                requests=requests,
                request_hashes=P1_REQUEST_HASHES,
                anchors=anchors,
                theta0_anchor_teacher=theta0_anchor_teacher,
                context_templates=context_templates,
                guard=guard,
                policy=policy,
                source_head=source_head,
                progress=progress,
                mutation_lock=mutation_lock,
            )
            arm_records.append(record)
            ledgers[arm] = ledger
            matched_ledgers[arm] = matched_ledger
        assert_matched_call_identity(
            matched_ledgers[Arm.GENERIC_ADAPTIVE],
            matched_ledgers[Arm.ODE_ALLOC],
        )
        full_actual_call_differences = _matched_call_differences(
            ledgers[Arm.GENERIC_ADAPTIVE], ledgers[Arm.ODE_ALLOC]
        )
        if {name: tensor_sha256(value) for name, value in touched.items()} != global_w0_hashes:
            raise ODEAllocContractError("P1 terminal global W0 differs")
        guard.assert_unchanged()
        manifest: dict[str, Any] = {
            "schema_version": "ode-alloc-s04-p1-matched-pair/v1",
            "status": "PASS_MOTIVATION_SCALE_TERMINAL_ONLY",
            "instruction_id": P1_INSTRUCTION_ID,
            "session_id": SESSION_ID,
            "run_token": run_token,
            "execution_seed": SEED,
            "source_head": source_head,
            "expected_parent": EXPECTED_BASE,
            "model_alias": alias,
            "model": {
                "config_dtype": str(model.config.dtype),
                "observed_parameter_dtype": str(next(model.parameters()).dtype),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "visible_cuda_device_count": torch.cuda.device_count(),
            },
            "locks": {
                "numerical_lock_sha256": NUMERICAL_LOCK_SHA256,
                "artifact_lock_sha256": ARTIFACT_LOCK_SHA256,
                "artifact_lock_canonical_sha256": artifact_receipt.lock_canonical_sha256,
                "seal_file_sha256": SEAL_FILE_SHA256,
                "seal_root_digest": SEAL_ROOT,
                "dataset_sha256": artifact_receipt.counterfact,
                "hparams_sha256": artifact_receipt.hparams,
                "model_revision": artifact_receipt.revision,
                "common_policy_id": policy.policy_id,
                "budget_id": policy.solver_budget.identity(),
            },
            "selection": {
                "case_order": P1_CASE_ORDER,
                "request_hashes": P1_REQUEST_HASHES,
                "anchor_count": len(anchor_approved),
                "anchor_root_sha256": canonical_hash(anchor_approved),
            },
            "contexts": {
                "count": len(prompt_templates),
                "first_sha256": context_first,
                "second_sha256": context_second,
                "exact_match": context_first == context_second,
                "raw_location": "raw/context_templates.json",
            },
            "theta0_anchor_teacher": {
                "count": len(theta0_anchor_teacher),
                "panel_id": theta0_result.panel_id,
                "setup_accounting": setup_ledger.snapshot(),
                "setup_component_wall_seconds": setup_timers.wall,
                "setup_component_gpu_seconds": setup_timers.gpu,
            },
            "arms": arm_records,
            "matched_generic_ode_calls": True,
            "matched_scope": "post-basis-controller-proposal-and-verdict",
            "full_actual_generic_ode_call_differences": full_actual_call_differences,
            "full_actual_generic_ode_calls_equal": not full_actual_call_differences,
            "generic_ode_call_identity_fields": (
                "model_forward_calls",
                "processed_tokens",
                "backward_calls",
                "constraint_vjp_calls",
                "constraint_jvp_calls",
                "quantized_trial_calls",
                "commit_count",
            ),
            "terminal_global_w0_restore_exact": True,
            "artifact_postflight_unchanged": True,
            "evaluation_feedback_to_controller": False,
            "model_generate_calls": 0,
            "scientific_scope": "motivation-scale-four-edit-no-superiority-claim",
            "progress": {
                "last_stage": progress.last_stage,
                "receipt_root_sha256": progress.root_sha256(),
                "instrumentation_wall_seconds": progress.wall_seconds,
            },
            "peak": {
                "allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                "host_max_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                )
                * 1024,
            },
            "started_unix": started,
            "finished_unix": time.time(),
        }
        manifest["manifest_id"] = canonical_hash(manifest)
        manifest_sha = _write_canonical_once(destination / "manifest.json", manifest)
        summary = {
            "schema_version": "ode-alloc-s04-p1-terminal-summary/v1",
            "status": manifest["status"],
            "model_alias": alias,
            "source_head": source_head,
            "numerical_lock_sha256": NUMERICAL_LOCK_SHA256,
            "seal_root_digest": SEAL_ROOT,
            "case_order": P1_CASE_ORDER,
            "arms": tuple(
                {
                    "arm": row["arm"],
                    "nonzero_feasible_edit_count": sum(
                        bool(edit["endpoint"]["q_nonzero"])
                        and bool(edit["endpoint"]["feasible"])
                        for edit in row["edits"]
                    ),
                    "fallback_count": sum(
                        bool(edit["fallback_to_native"]) for edit in row["edits"]
                    ),
                    "violation_trial_count": sum(
                        int(edit["trials"]["violation_count"])
                        for edit in row["edits"]
                    ),
                    "slack_sum": sum(
                        float(step["slack_sum"])
                        for edit in row["edits"]
                        for step in edit["projection"]
                    ),
                    "evaluation_id": row["evaluation"]["evaluation_id"],
                    "accounting_identity": row["accounting"]["identity"],
                    "peak_allocated_bytes": row["memory"][
                        "endpoint_peak_allocated_bytes"
                    ],
                    "peak_reserved_bytes": row["memory"][
                        "endpoint_peak_reserved_bytes"
                    ],
                }
                for row in arm_records
            ),
            "matched_generic_ode_calls": True,
            "matched_scope": "post-basis-controller-proposal-and-verdict",
            "full_actual_generic_ode_call_differences": full_actual_call_differences,
            "full_actual_generic_ode_calls_equal": not full_actual_call_differences,
            "terminal_global_w0_restore_exact": True,
            "model_generate_calls": 0,
            "scientific_outcome_count": sum(
                int(row["evaluation"]["accounting"]["n_eval_specs"])
                for row in arm_records
            ),
            "elapsed_seconds": time.time() - started,
        }
        summary["summary_id"] = canonical_hash(summary)
        summary_sha = _write_canonical_once(destination / "summary.json", summary)
        terminal = {
            "schema_version": "ode-alloc-s04-p1-terminal-receipt/v1",
            "status": "PASS",
            "model_alias": alias,
            "manifest_sha256": manifest_sha,
            "summary_sha256": summary_sha,
        }
        terminal["terminal_id"] = canonical_hash(terminal)
        terminal_sha = _write_canonical_once(destination / "terminal.json", terminal)
        progress.complete(
            "terminal-metadata-written",
            alias=alias,
            manifest_sha256=manifest_sha,
            summary_sha256=summary_sha,
            terminal_sha256=terminal_sha,
        )
        return {
            "status": "PASS",
            "alias": alias,
            "root": str(destination),
            "manifest_sha256": manifest_sha,
            "summary_sha256": summary_sha,
            "terminal_sha256": terminal_sha,
        }
    finally:
        with contextlib.suppress(Exception):
            from easyeditor.models.memit import memit_main

            memit_main.COV_CACHE.clear()
            memit_main.CONTEXT_TEMPLATES_CACHE = None


def write_p1_failure_once(output_root: Path, exc: BaseException) -> str | None:
    if not output_root.exists() or not output_root.is_dir():
        return None
    destination = output_root / "failure.json"
    if destination.exists() or destination.is_symlink():
        return None
    stages = output_root / "raw" / "stages"
    completed = tuple(sorted(path.name for path in stages.glob("*.json"))) if stages.is_dir() else ()
    payload = {
        "schema_version": "ode-alloc-s04-p1-failure/v1",
        "status": "FAIL_CLOSED_NO_RETRY",
        "instruction_id": P1_INSTRUCTION_ID,
        "exception_type": type(exc).__name__,
        "exception_message_sha256": hashlib.sha256(
            str(exc).encode("utf-8")
        ).hexdigest(),
        "completed_stage_count": len(completed),
        "last_completed_stage_file": completed[-1] if completed else None,
        "retry_allowed": False,
    }
    return _write_canonical_once(destination, payload)
