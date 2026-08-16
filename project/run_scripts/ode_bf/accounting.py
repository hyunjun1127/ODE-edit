"""Joint-batch, K/T/cycle, compute, and longitudinal accounting contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Sequence

from .contracts import (
    BATCH_SIZE,
    DirectZDiagnostics,
    ODEBFContractError,
    RolloutBudget,
    canonical_hash,
    finite,
)


INTEGER_COUNTERS = (
    "model_forward",
    "processed_tokens",
    "backward",
    "target_backward",
    "constraint_jvp",
    "constraint_vjp",
    "qp_solve",
    "qp_certificate",
    "trial",
    "reject",
    "fallback",
    "commit",
    "rollback",
    "functional_h_replay",
    "functional_p_replay",
    "evaluator_forward",
    "evaluator_tokens",
    "effective_bf16_weight_peak_live",
    "dense_fp32_full_delta_live",
    "native_baseline_dense_delta_peak_live",
    "native_baseline_dense_delta_bytes_peak",
)


@dataclass(frozen=True, slots=True)
class LayerFactorReceipt:
    layer: int
    key_shape: tuple[int, int]
    residual_shape: tuple[int, int]
    numerical_rank: int
    dtype: str
    device_class: str

    def __post_init__(self) -> None:
        if self.key_shape[1] != BATCH_SIZE or self.residual_shape[1] != BATCH_SIZE:
            raise ODEBFContractError("joint native factors must expose ten columns")
        if self.numerical_rank < 2 or self.numerical_rank > BATCH_SIZE:
            raise ODEBFContractError("joint native factor collapsed or exceeds batch rank")
        if self.dtype not in ("torch.float32", "torch.float64"):
            raise ODEBFContractError("native factor receipt dtype is unsupported")
        if self.device_class not in ("cpu", "cuda"):
            raise ODEBFContractError("native factor receipt device class is invalid")


@dataclass(frozen=True, slots=True)
class JointInitializationReceipt:
    editor_invocations: int
    request_count: int
    direct_z_initializations: int
    target_initializations: int
    hidden_singleton_editor_calls: int
    shared_key_computes: int
    covariance_reads: int
    projector_reads: int
    native_factor_computes: int
    covariance_decision_enabled: bool
    factors: tuple[LayerFactorReceipt, ...]
    request_order_sha256: str

    def __post_init__(self) -> None:
        if self.editor_invocations != 1 or self.hidden_singleton_editor_calls != 0:
            raise ODEBFContractError("backend decomposed the joint editor invocation")
        if self.request_count != BATCH_SIZE:
            raise ODEBFContractError("initialization did not receive ten requests")
        if self.direct_z_initializations != BATCH_SIZE or self.target_initializations != BATCH_SIZE:
            raise ODEBFContractError("direct-z/target initialization must be per request")
        if len(self.factors) < 2 or len({factor.layer for factor in self.factors}) != len(self.factors):
            raise ODEBFContractError("joint factor receipt lacks distinct layers")
        expected_shared = len(self.factors)
        for name, value in (
            ("shared_key_computes", self.shared_key_computes),
            ("projector_reads", self.projector_reads),
            ("native_factor_computes", self.native_factor_computes),
        ):
            if value != expected_shared:
                raise ODEBFContractError(f"{name} does not reflect one shared joint pass/layer")
        expected_covariance = expected_shared if self.covariance_decision_enabled else 0
        if self.covariance_reads != expected_covariance:
            raise ODEBFContractError(
                "covariance receipt does not match whether the decision screen was enabled"
            )
        if len(self.request_order_sha256) != 64:
            raise ODEBFContractError("request order digest is invalid")


@dataclass(slots=True)
class ComputeLedger:
    counters: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in INTEGER_COUNTERS}
    )
    component_wall_seconds: dict[str, float] = field(default_factory=dict)
    component_gpu_seconds: dict[str, float] = field(default_factory=dict)
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    host_maxrss_kib: int = 0
    accepted_t: float = 0.0
    completed_k_total: int = 0
    completed_correction_cycles: int = 0
    direct_z_diagnostics: list[DirectZDiagnostics] = field(default_factory=list)

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in self.counters:
            raise ODEBFContractError("unknown compute counter")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ODEBFContractError("compute counter increment must be nonnegative integer")
        self.counters[name] += amount

    def add_time(self, component: str, *, wall_seconds: float, gpu_seconds: float = 0.0) -> None:
        wall = finite("wall_seconds", wall_seconds)
        gpu = finite("gpu_seconds", gpu_seconds)
        if wall < 0.0 or gpu < 0.0:
            raise ODEBFContractError("component time cannot be negative")
        if not component:
            raise ODEBFContractError("accounting component is empty")
        self.component_wall_seconds[component] = self.component_wall_seconds.get(component, 0.0) + wall
        self.component_gpu_seconds[component] = self.component_gpu_seconds.get(component, 0.0) + gpu

    def observe_memory(self, *, allocated_bytes: int, reserved_bytes: int, maxrss_kib: int) -> None:
        for name, value in (
            ("allocated_bytes", allocated_bytes),
            ("reserved_bytes", reserved_bytes),
            ("maxrss_kib", maxrss_kib),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ODEBFContractError(f"{name} must be a nonnegative integer")
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, allocated_bytes)
        self.peak_reserved_bytes = max(self.peak_reserved_bytes, reserved_bytes)
        self.host_maxrss_kib = max(self.host_maxrss_kib, maxrss_kib)

    def record_accepted_step(self, *, accepted_dt: float, completed_k_total: int) -> None:
        value = finite("accepted_dt", accepted_dt)
        if value < 0.0:
            raise ODEBFContractError("accepted time cannot decrease")
        if completed_k_total != self.completed_k_total + 1:
            raise ODEBFContractError("K_total accounting is not contiguous")
        self.accepted_t += value
        self.completed_k_total = completed_k_total

    def finish_cycle(self, correction_cycle: int) -> None:
        if correction_cycle != self.completed_correction_cycles:
            raise ODEBFContractError("correction-cycle accounting is not contiguous")
        self.completed_correction_cycles += 1

    def add_direct_z_diagnostics(self, diagnostics: DirectZDiagnostics) -> None:
        self.direct_z_diagnostics.append(diagnostics)

    def validate_strict(self, budget: RolloutBudget) -> None:
        if self.completed_k_total != budget.k_total:
            raise ODEBFContractError("compute ledger did not execute locked K_total")
        if self.completed_correction_cycles != budget.correction_cycles:
            raise ODEBFContractError("compute ledger did not execute locked cycles")
        if self.counters["effective_bf16_weight_peak_live"] > 1:
            raise ODEBFContractError("more than one effective BF16 target weight was live")
        if self.counters["dense_fp32_full_delta_live"] != 0:
            raise ODEBFContractError("a dense FP32 full delta was retained")

    def raw_free_payload(self) -> dict[str, object]:
        return {
            "counters": dict(sorted(self.counters.items())),
            "component_wall_seconds": dict(sorted(self.component_wall_seconds.items())),
            "component_gpu_seconds": dict(sorted(self.component_gpu_seconds.items())),
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "host_maxrss_kib": self.host_maxrss_kib,
            "accepted_t": self.accepted_t,
            "completed_k_total": self.completed_k_total,
            "completed_correction_cycles": self.completed_correction_cycles,
            "direct_z_diagnostics": [asdict(item) for item in self.direct_z_diagnostics],
        }

    def identity(self) -> str:
        return canonical_hash(self.raw_free_payload())


def assert_matched_rollout_accounting(
    generic: ComputeLedger,
    bf: ComputeLedger,
    *,
    budget: RolloutBudget,
) -> None:
    generic.validate_strict(budget)
    bf.validate_strict(budget)
    # Projector work (constraint derivatives/QP/certificate) may differ.  The raw
    # proposal and exact-trial budget may not.
    required_equal = (
        "model_forward",
        "processed_tokens",
        "backward",
        "target_backward",
        "trial",
        "evaluator_forward",
        "evaluator_tokens",
    )
    mismatch = {
        name: (generic.counters[name], bf.counters[name])
        for name in required_equal
        if generic.counters[name] != bf.counters[name]
    }
    if mismatch:
        raise ODEBFContractError("Generic/BF matched proposal accounting differs")
    if generic.completed_k_total != bf.completed_k_total or generic.accepted_t != bf.accepted_t:
        raise ODEBFContractError("Generic/BF K/T accounting differs")


@dataclass(frozen=True, slots=True)
class LongitudinalBatchPoint:
    arm: str
    sequential_batch_index: int
    cumulative_edits: int
    official_efficacy_aggregate: float
    official_generalization_aggregate: float
    official_locality_aggregate: float
    active_history_retention: float
    all_ever_history_retention: float
    historical_functional_damage: float
    pretrained_functional_damage: float
    exact_hit_count: int
    reject_count: int
    rollback_count: int
    fallback_count: int
    net_bf16_capacity: float
    layer_concentration: float
    compute_identity: str

    def __post_init__(self) -> None:
        if self.sequential_batch_index < 0:
            raise ODEBFContractError("sequential batch index is invalid")
        expected = (self.sequential_batch_index + 1) * BATCH_SIZE
        if self.cumulative_edits != expected:
            raise ODEBFContractError("cumulative edit count is not a B10 trajectory")
        for name in (
            "official_efficacy_aggregate",
            "official_generalization_aggregate",
            "official_locality_aggregate",
            "active_history_retention",
            "all_ever_history_retention",
        ):
            value = finite(name, getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ODEBFContractError(f"{name} is outside [0,1]")
        for name in (
            "historical_functional_damage",
            "pretrained_functional_damage",
            "net_bf16_capacity",
            "layer_concentration",
        ):
            if finite(name, getattr(self, name)) < 0.0:
                raise ODEBFContractError(f"{name} is negative")
        for name in ("exact_hit_count", "reject_count", "rollback_count", "fallback_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ODEBFContractError(f"{name} is invalid")
        if len(self.compute_identity) != 64:
            raise ODEBFContractError("compute identity is not SHA-256")


def assert_sequential_b10_trajectory(points: Sequence[LongitudinalBatchPoint]) -> None:
    if not points:
        raise ODEBFContractError("longitudinal trajectory is empty")
    arm = points[0].arm
    if any(point.arm != arm for point in points):
        raise ODEBFContractError("one trajectory mixed arm states")
    if tuple(point.sequential_batch_index for point in points) != tuple(range(len(points))):
        raise ODEBFContractError("sequential B10 trajectory has a gap or reset")
