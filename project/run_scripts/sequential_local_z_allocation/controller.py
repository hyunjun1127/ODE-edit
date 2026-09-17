"""Production CPU controller derived from the immutable v2 CPU reference.

Native fitting and model scoring remain exclusively in the backend. Immutable
RAM snapshots are opaque handles, not a disk-checkpoint requirement. This module
adds durable event hooks, explicit incomplete/failure provenance and selection
sealing before observers/history, without changing the reference search math.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Integral, Real
from typing import Any, Callable, Mapping, Optional, Protocol


HISTORY_LAYERS = (4, 5, 6, 7, 8)
ARM_LAYERS = {"N4": (4,), "F48": (4, 8), "G48": (4, 8), "C4": (4,),
              "C48": (4, 8), "C45678": HISTORY_LAYERS}
REFERENCE_SOURCE = "audits/global/2026-09-17-sequential-local-z-allocation/controller_reference.py"


def json_safe(value):
    """No model/snapshot payload is accepted by this compact event serializer."""
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if not math.isfinite(value):
            raise ContractError("NONFINITE_LEDGER_VALUE")
        return float(value)
    raise ContractError("OPAQUE_OBJECT_IN_COMPACT_LEDGER:" + type(value).__name__)


class ContractError(RuntimeError):
    """기술 오류는 정상적인 품질 fallback과 구분한다."""


class BudgetExceeded(RuntimeError):
    def __init__(self, reason, *, owner=None):
        super().__init__(reason)
        self.reason, self.owner = reason, owner


@dataclass(frozen=True)
class Scores:
    base_kl: float
    training_e: float
    current_strict: frozenset = frozenset()
    current_pair: Optional[frozenset] = None
    past_h: Optional[float] = None
    past_strict: frozenset = frozenset()
    past_pair: Optional[frozenset] = None
    canonical_e: Optional[float] = None
    # 각 ID의 최악 token margin 및 old-NLL minus new-NLL이다.
    current_token_margins: Mapping = field(default_factory=dict)
    current_pair_margins: Mapping = field(default_factory=dict)
    past_token_margins: Mapping = field(default_factory=dict)
    past_pair_margins: Mapping = field(default_factory=dict)

    def validate(self):
        if self.base_kl is None or self.training_e is None:
            raise ContractError("REQUIRED_SCORE_MISSING")
        values = [self.base_kl, self.training_e, self.past_h, self.canonical_e]
        for name in ("current_token_margins", "current_pair_margins",
                     "past_token_margins", "past_pair_margins"):
            values.extend(getattr(self, name).values())
        if any(v is not None and not math.isfinite(v) for v in values):
            raise ContractError("NONFINITE_SCORE")


@dataclass(frozen=True)
class Limits:
    max_search_proposals: int = 64
    max_endpoints: int = 28  # N4 baseline score 1회 제외
    max_suffix_fits: int = 40  # native L4 fit 1회 제외
    reserve_endpoints_for_pruning: int = 4
    reserve_suffix_fits_for_pruning: int = 8
    max_extra_adam: Optional[int] = 9600
    max_adam_per_suffix_fit: int = 2400  # B100 × native 최대24 Adam
    epsilon_l: float = 1e-4
    epsilon_b: float = 1e-6
    mean_scale: float = .05  # 단위 조절이며 품질 plateau가 아니다.
    guard_canonical_mean: bool = False
    proposal_margin_guards: bool = True

    def __post_init__(self):
        if self.max_search_proposals < 1 or self.max_endpoints < 1:
            raise ValueError("POSITIVE_SEARCH_BUDGET_REQUIRED")
        if not 0 <= self.reserve_endpoints_for_pruning <= self.max_endpoints:
            raise ValueError("INVALID_ENDPOINT_RESERVE")
        if not 0 <= self.reserve_suffix_fits_for_pruning <= self.max_suffix_fits:
            raise ValueError("INVALID_FIT_RESERVE")
        if self.mean_scale <= 0 or self.epsilon_l < 0 or self.epsilon_b <= 0:
            raise ValueError("INVALID_NUMERICAL_SCALE")
        if self.max_extra_adam is not None and self.max_extra_adam < 0:
            raise ValueError("INVALID_ADAM_BUDGET")
        if self.max_adam_per_suffix_fit <= 0:
            raise ValueError("INVALID_PER_FIT_ADAM_RESERVE")


class Backend(Protocol):
    """Snapshot은 handle이어도 된다. 실제 큰 tensor 저장 정책을 정하지 않는다.

    state_token은 history 이외의 model/context/RNG 상태를 정확히 식별한다.
    history_token은 native M들을 식별한다. fit_native는 선택층만 수정하고
    history는 건드리지 않는다. finalize_history만 최종 history를 갱신한다.
    """
    def snapshot(self) -> Any: ...
    def restore(self, snapshot: Any) -> None: ...
    def state_token(self) -> str: ...
    def history_token(self) -> str: ...
    def fit_native(self, layer: int) -> int: ...  # 실제 Adam step 합계 반환
    def apply_gate(self, layer: int, before: Any, native: Any, gate: float) -> None: ...
    def score(self) -> Scores: ...
    def changed_layers(self, before: Any, after: Any) -> tuple: ...
    def action_norm(self, before: Any, after: Any) -> float: ...
    def finalize_history(self, layers: tuple) -> None: ...


@dataclass(frozen=True)
class Candidate:
    gates: tuple
    state: Any
    state_token: Optional[str]
    scores: Optional[Scores]
    feasible: bool
    reasons: tuple
    active_layers: tuple = ()
    action_norm: float = 0.
    is_n4: bool = False
    phase: str = "baseline"


class Controller:
    def __init__(self, backend: Backend, layers=(4, 8), *, namespace: str,
                 limits: Limits = Limits(), history_layers=None,
                 event_sink: Optional[Callable[[dict], None]] = None):
        self.backend, self.layers = backend, tuple(layers)
        if not self.layers or self.layers[0] != 4 or tuple(sorted(set(layers))) != self.layers:
            raise ValueError("ASCENDING_UNIQUE_LAYERS_STARTING_AT_4_REQUIRED")
        if not namespace:
            raise ValueError("CAPSULE_NAMESPACE_REQUIRED")
        # 실제 v2 runner는 모든 arm에 history_layers=(4,5,6,7,8)을 지정한다.
        # 작은 toy backend는 명시하지 않으면 자신의 search layers만 유지한다.
        self.history_layers = tuple(history_layers) if history_layers is not None else self.layers
        if tuple(sorted(set(self.history_layers))) != self.history_layers or not set(self.layers).issubset(self.history_layers):
            raise ValueError("INVALID_HISTORY_LAYERS")
        self.namespace, self.limits = namespace, limits
        self.event_sink = event_sink
        self.events, self.incomplete_records = [], []
        self.sealed, self.selected, self.failed = False, None, None
        self.policy_arm = None
        self._fixed_candidate = None
        self.entry = backend.snapshot()
        self.entry_token, self.entry_history = backend.state_token(), backend.history_token()
        self.fit_cache, self.score_cache, self.gate_cache = {}, {}, {}
        self.fit_cache_identity = {}
        self.reduction_cache = {}
        self.records, self.finalized = [], False
        self.search_completed_gates = []
        self.search_coverage = None
        self.pruning_status = None
        self.counts = dict(l4_fits=0, suffix_fits=0, baseline_scores=0, endpoints=0,
                           search_proposals=0, pruning_proposals=0, fit_cache_hits=0,
                           endpoint_cache_hits=0, gate_cache_hits=0, bound_rejections=0,
                           commits=0, l4_adam=0, extra_adam=0, reduction_cache_hits=0,
                           raw_callback_calls=0, extra_adam_reserved=0,
                           completed_suffix_fits=0, incomplete_candidates=0,
                           technical_failures=0)
        self._emit("CONTROLLER_START", layers=self.layers, history_layers=self.history_layers,
                   namespace=namespace, entry_token=self.entry_token,
                   entry_history=self.entry_history, limits=asdict(limits))
        n4 = (1.,) + (0.,) * (len(self.layers) - 1)
        try:
            state, token = self._path(n4, "baseline")
            score = self._score(token, baseline=True)
            if limits.guard_canonical_mean and score.canonical_e is None:
                raise ContractError("CANONICAL_MEAN_REQUIRED")
            self.reference = score
            self.objective_scale = max(score.base_kl, limits.epsilon_b)
            self.n4 = Candidate(n4, state, token, score, True, (),
                                backend.changed_layers(self.entry, state),
                                backend.action_norm(self.entry, state), True)
            self.records.append(self.n4)
            self.gate_cache[n4] = self.n4
            self._emit("BASELINE_COMPLETE", candidate=self.candidate_receipt(self.n4))
        except Exception as exc:
            self._technical_failure(exc, stage="BASELINE")
            raise
        finally:
            self._restore_entry()

    def _emit(self, event, **values):
        row = json_safe(dict(sequence=len(self.events), event=event, counts=dict(self.counts), **values))
        # Validate exact JSON serializability before a caller's create-once writer.
        json.dumps(row, allow_nan=False)
        if self.event_sink is not None:
            try:
                self.event_sink(row)
            except Exception as exc:
                self.failed = "LEDGER_WRITE_FAILURE"
                raise ContractError("LEDGER_WRITE_FAILURE") from exc
        self.events.append(row)
        return row

    def _ensure_open(self):
        if self.failed is not None:
            raise ContractError("CONTROLLER_TECHNICAL_FAILURE:" + self.failed)
        if self.finalized:
            raise ContractError("ALREADY_FINALIZED")
        if self.sealed:
            raise ContractError("SELECTION_ALREADY_SEALED")

    def _restore_entry(self):
        self._validate_snapshot(self.entry, self.entry_token, self.entry_history)
        self.backend.restore(self.entry)
        if self.backend.state_token() != self.entry_token or self.backend.history_token() != self.entry_history:
            self.failed = "ENTRY_RESTORE_MISMATCH"
            raise ContractError("ENTRY_RESTORE_MISMATCH")

    def _validate_snapshot(self, snapshot, state_token, history_token):
        # Production backend verifies immutable RAM handle/version/hash binding;
        # reference toy backends may rely on their immutable-value contract.
        verifier = getattr(self.backend, "validate_snapshot", None)
        if verifier is not None:
            verifier(snapshot, state_token, history_token)

    def _technical_failure(self, exc, *, stage, **values):
        self.failed = type(exc).__name__ + ":" + str(exc)
        self.counts["technical_failures"] += 1
        self._emit("TECHNICAL_FAILURE", stage=stage, error=self.failed, **values)

    def _budget(self, reason):
        self._emit("BUDGET_STOP", reason=reason)
        raise BudgetExceeded(reason, owner=id(self))

    def candidate_receipt(self, row):
        return json_safe(dict(gates=row.gates, state_token=row.state_token,
                              scores=row.scores, feasible=row.feasible, reasons=row.reasons,
                              active_layers=row.active_layers, action_norm=row.action_norm,
                              is_n4=row.is_n4, phase=row.phase))

    def _limit(self, kind, phase):
        cap = getattr(self.limits, "max_" + kind)
        if phase == "search":
            cap -= getattr(self.limits, "reserve_" + kind + "_for_pruning")
        return cap

    def _path(self, gates, phase):
        b = self.backend
        self._restore_entry()
        for layer, gate in zip(self.layers, gates):
            if gate == 0:
                self._emit("EXACT_ZERO_SKIP", layer=layer, phase=phase, gates=gates,
                           prefix_state=b.state_token(), history=b.history_token())
                continue
            before = b.snapshot()
            key = (self.namespace, layer, b.state_token(), b.history_token())
            if key in self.fit_cache:
                native = self.fit_cache[key]
                self._validate_snapshot(native, *self.fit_cache_identity[key])
                self.counts["fit_cache_hits"] += 1
                self._emit("FIT_CACHE_HIT", cache_key=key, layer=layer, phase=phase)
            else:
                if layer == 4:
                    if self.counts["l4_fits"]:
                        raise ContractError("L4_ENTRY_CACHE_IDENTITY_CHANGED")
                    self.counts["l4_fits"] += 1
                else:
                    if self.counts["endpoints"] >= self._limit("endpoints", phase):
                        self._budget("ENDPOINT_RESERVE")
                    if self.counts["suffix_fits"] >= self._limit("suffix_fits", phase):
                        self._budget("SUFFIX_FIT_CAP")
                    if (self.limits.max_extra_adam is not None and
                            self.limits.max_extra_adam - self.counts["extra_adam"] < self.limits.max_adam_per_suffix_fit):
                        self._budget("EXTRA_ADAM_RESERVE")
                    self.counts["suffix_fits"] += 1
                    self.counts["extra_adam_reserved"] = self.limits.max_adam_per_suffix_fit
                history = b.history_token()
                self._emit("FIT_RESERVED", layer=layer, phase=phase, cache_key=key,
                           baseline=layer == 4, extra_reservation=self.counts["extra_adam_reserved"])
                try:
                    steps = b.fit_native(layer)
                except BaseException as exc:
                    # Consumed work on an interrupted native fit is not invented.
                    self._emit("FIT_FAILED", layer=layer, phase=phase, cache_key=key,
                               error=repr(exc), actual_adam="NOT_RECORDED",
                               reservation_unreconciled=self.counts["extra_adam_reserved"])
                    raise
                if not isinstance(steps, int) or isinstance(steps, bool) or not 0 <= steps <= self.limits.max_adam_per_suffix_fit:
                    raise ContractError("INVALID_ACTUAL_ADAM_COUNT")
                self.counts["l4_adam" if layer == 4 else "extra_adam"] += steps
                released = self.counts["extra_adam_reserved"] - steps if layer != 4 else 0
                self.counts["extra_adam_reserved"] = 0
                if layer != 4:
                    self.counts["completed_suffix_fits"] += 1
                # 직렬 호출이므로 worst-case reserve는 검사만 한다. 실제 step만
                # 차감하여 사용하지 않은 reserve는 다음 fit에 바로 반환한다.
                native = b.snapshot()
                if b.history_token() != history:
                    raise ContractError("INNER_HISTORY_MUTATION")
                if set(b.changed_layers(before, native)) - {layer}:
                    raise ContractError("NATIVE_NONSELECTED_MUTATION")
                self.fit_cache[key] = native
                self.fit_cache_identity[key] = (b.state_token(), b.history_token())
                self._emit("FIT_COMPLETE", layer=layer, phase=phase, cache_key=key,
                           actual_adam=steps, unused_adam_reservation_released=released,
                           native_state=b.state_token(), native_history=b.history_token())
            b.restore(before)
            b.apply_gate(layer, before, native, gate)
            if b.history_token() != self.entry_history:
                raise ContractError("GATE_HISTORY_MUTATION")
            if set(b.changed_layers(before, b.snapshot())) - {layer}:
                raise ContractError("GATE_NONSELECTED_MUTATION")
            self._emit("GATE_MATERIALIZED", layer=layer, gate=gate, phase=phase,
                       state_token=b.state_token(), history=b.history_token())
        return b.snapshot(), b.state_token()

    def _score(self, token, *, baseline=False, phase="search"):
        key = (self.namespace, self.entry_history, token)
        if key in self.score_cache:
            self.counts["endpoint_cache_hits"] += 1
            self._emit("ENDPOINT_SCORE_CACHE_HIT", cache_key=key, phase=phase)
            return self.score_cache[key]
        if baseline:
            self.counts["baseline_scores"] += 1
        else:
            if self.counts["endpoints"] >= self._limit("endpoints", phase):
                self._budget("ENDPOINT_CAP")
            self.counts["endpoints"] += 1
        self._emit("SCORE_RESERVED", token=token, baseline=baseline, phase=phase)
        score = self.backend.score()
        score.validate()
        if self.backend.state_token() != token or self.backend.history_token() != self.entry_history:
            raise ContractError("OBSERVER_STATE_MUTATION")
        self.score_cache[key] = score
        self._emit("SCORE_COMPLETE", token=token, baseline=baseline, phase=phase, scores=score)
        return score

    def _quality_reasons(self, score):
        r, eps, reasons = self.reference, self.limits.epsilon_l, []
        if score.training_e > r.training_e + eps:
            reasons.append("CURRENT_TRAINING_MEAN")
        if not r.current_strict.issubset(score.current_strict):
            reasons.append("CURRENT_STRICT_IDS")
        if r.current_pair is not None and (score.current_pair is None or not r.current_pair.issubset(score.current_pair)):
            reasons.append("CURRENT_PAIR_IDS")
        if self.limits.guard_canonical_mean:
            if score.canonical_e is None or score.canonical_e > r.canonical_e + eps:
                reasons.append("CURRENT_CANONICAL_MEAN")
        if r.past_h is not None:
            if score.past_h is None or score.past_h > r.past_h + eps:
                reasons.append("PAST_MEAN")
            if not r.past_strict.issubset(score.past_strict):
                reasons.append("PAST_STRICT_IDS")
            if r.past_pair is not None and (score.past_pair is None or not r.past_pair.issubset(score.past_pair)):
                reasons.append("PAST_PAIR_IDS")
        return tuple(reasons)

    def evaluate(self, gates, *, phase="search"):
        self._ensure_open()
        if phase not in ("search", "pruning"):
            raise ValueError("UNKNOWN_PHASE")
        gates = tuple(float(x) for x in gates)
        if len(gates) != len(self.layers) or not all(math.isfinite(x) for x in gates):
            raise ContractError("INVALID_GATE_VECTOR")
        if gates in self.gate_cache:
            self.counts["gate_cache_hits"] += 1
            self._emit("GATE_CACHE_HIT", gates=gates, phase=phase,
                       candidate=self.candidate_receipt(self.gate_cache[gates]))
            return self.gate_cache[gates]
        counter = "search_proposals" if phase == "search" else "pruning_proposals"
        if phase == "search" and self.counts[counter] >= self.limits.max_search_proposals:
            self._budget("SEARCH_PROPOSAL_CAP")
        self.counts[counter] += 1
        self._emit("GATE_PROPOSAL", gates=gates, phase=phase)
        if any(x < 0 or x > 1 for x in gates):
            self.counts["bound_rejections"] += 1
            row = Candidate(gates, None, None, None, False, ("RAW_BOUNDS",))
            self.gate_cache[gates] = row
            self._emit("GATE_BOUNDS_REJECTED", gates=gates, phase=phase, model_called=False)
            return row
        counts_before = dict(self.counts)
        try:
            # 완성된 endpoint만 품질 검사한다. 중간 prefix는 탈락시키지 않는다.
            state, token = self._path(gates, phase)
            score = self._score(token, phase=phase)
            reasons = self._quality_reasons(score)
            norm = self.backend.action_norm(self.entry, state)
            if not math.isfinite(norm) or norm < 0:
                raise ContractError("INVALID_ACTION_NORM")
            row = Candidate(gates, state, token, score, not reasons, reasons,
                            self.backend.changed_layers(self.entry, state), norm, phase=phase)
            self.records.append(row)
            self.gate_cache[gates] = row
            if phase == "search":
                self.search_completed_gates.append(gates)
            self._emit("CANDIDATE_COMPLETE", candidate=self.candidate_receipt(row),
                       incumbent=self.candidate_receipt(self.best()))
            return row
        except BudgetExceeded as exc:
            if exc.owner != id(self):
                failure = ContractError("BACKEND_BUDGET_EXCEPTION_NOT_CONTROLLER_STOP")
                self._technical_failure(failure, stage="CANDIDATE", gates=gates, phase=phase)
                raise failure from exc
            row = json_safe(dict(gates=gates, phase=phase, reason=str(exc),
                                 completed=False, scored=False, feasible=False,
                                 partial_state=self.backend.state_token(),
                                 counts_before=counts_before, counts_after=dict(self.counts)))
            self.incomplete_records.append(row)
            self.counts["incomplete_candidates"] += 1
            self._emit("INCOMPLETE_BUDGET", candidate=row)
            raise
        except Exception as exc:
            self._technical_failure(exc, stage="CANDIDATE", gates=gates, phase=phase)
            raise
        finally:
            self._restore_entry()

    def _margin_panels(self):
        r = self.reference
        panels = [(r.current_strict, "current_token_margins", 1.),
                  (r.current_pair or (), "current_pair_margins", self.limits.mean_scale)]
        panels.extend([(r.past_strict if r.past_h is not None else (), "past_token_margins", 1.),
                       ((r.past_pair or ()) if r.past_h is not None else (), "past_pair_margins", self.limits.mean_scale)])
        return panels

    def objective_and_constraints(self, gates):
        row = self.evaluate(gates)
        box = tuple(v for a in row.gates for v in (a, 1. - a))
        mean_count = 1 + int(self.reference.past_h is not None) + int(self.limits.guard_canonical_mean)
        margin_panels = self._margin_panels() if self.limits.proposal_margin_guards else []
        if row.scores is None:
            # 외부 영역의 해석적 연장값이며 모델 관측값이 아니다. clipping 없음.
            return 1., box + (-1.,) * (mean_count + len(margin_panels))
        s, r, eps, scale = row.scores, self.reference, self.limits.epsilon_l, self.limits.mean_scale
        slack = [(r.training_e + eps - s.training_e) / scale]
        if r.past_h is not None:
            if s.past_h is None:
                raise ContractError("MISSING_PAST_MEAN")
            slack.append((r.past_h + eps - s.past_h) / scale)
        if self.limits.guard_canonical_mean:
            if s.canonical_e is None:
                raise ContractError("MISSING_CANONICAL_MEAN")
            slack.append((r.canonical_e + eps - s.canonical_e) / scale)
        for ids, name, margin_scale in margin_panels:
            if not ids:
                slack.append(1.)
                continue
            values = getattr(s, name)
            if not set(ids).issubset(values):
                raise ContractError("MISSING_PROTECTED_MARGIN:" + name)
            slack.append(min(values[i] for i in ids) / margin_scale)
        return s.base_kl / self.objective_scale, box + tuple(slack)

    def measure_reductions(self, reductions):
        """Raw u를 먼저 검사하여 1-u 반올림이 bounds 위반을 숨기지 않게 한다."""
        self._ensure_open()
        u = tuple(float(x) for x in reductions)
        if len(u) != len(self.layers) or not all(math.isfinite(x) for x in u):
            raise ContractError("INVALID_REDUCTION_VECTOR")
        self.counts["raw_callback_calls"] += 1
        self._emit("RAW_CALLBACK", raw_u=u, transformed_gates_not_yet_evaluated=True)
        if u in self.reduction_cache:
            self.counts["reduction_cache_hits"] += 1
            self._emit("RAW_CALLBACK_CACHE_HIT", raw_u=u,
                       result=self.reduction_cache[u], model_called=False)
            return self.reduction_cache[u]
        # Gate 하한=1-u, gate 상한 slack=u. 후자는 다시 1-(1-u)로 계산하지 않는다.
        box = tuple(v for x in u for v in (1. - x, x))
        if any(x < 0 or x > 1 for x in u):
            if self.counts["search_proposals"] >= self.limits.max_search_proposals:
                self._budget("SEARCH_PROPOSAL_CAP")
            self.counts["search_proposals"] += 1
            self.counts["bound_rejections"] += 1
            mean_count = 1 + int(self.reference.past_h is not None) + int(self.limits.guard_canonical_mean)
            margin_count = len(self._margin_panels()) if self.limits.proposal_margin_guards else 0
            result = (1., box + (-1.,) * (mean_count + margin_count))
            self._emit("RAW_BOUNDS_REJECTED", raw_u=u, dummy_objective=1.,
                       box_slack=box, model_called=False, clipped=False,
                       scored_candidate=False)
        else:
            objective, slack = self.objective_and_constraints(tuple(1. - x for x in u))
            result = (objective, box + slack[2 * len(u):])
        self.reduction_cache[u] = result
        self._emit("RAW_CALLBACK_RETURN", raw_u=u, objective=result[0], constraints=result[1])
        return result

    def coverage(self):
        """탐색 충분성의 count proxy이며 실제 COBYLA simplex 복원은 아니다."""
        count = len(self.search_completed_gates)
        distinct_a4 = len({g[0] for g in self.search_completed_gates})
        return dict(completed_search_gate_vectors=count, distinct_a4=distinct_a4,
                    completed_unique_state_tokens=len({r.state_token for r in self.records if r.phase == "search"}),
                    at_least_two_a4_values=distinct_a4 >= 2,
                    beyond_initial_simplex_count_proxy=count >= len(self.layers) + 2,
                    adequate_by_count_proxy=distinct_a4 >= 2 and count >= len(self.layers) + 2,
                    definition="N4 제외; 완료된 새 gate vector 포함(동일 endpoint score cache 허용); 실제 simplex 판정 아님")

    def best(self):
        if self.failed is not None:
            raise ContractError("NO_FALLBACK_AFTER_TECHNICAL_FAILURE:" + self.failed)
        feasible = [r for r in self.records if r.feasible]
        minimum = min(r.scores.base_kl for r in feasible)
        tied = [r for r in feasible if r.scores.base_kl <= minimum + self.limits.epsilon_b]
        # 매번 모든 실측 feasible 후보의 global minimum을 사용: epsilon 누적 금지.
        return min(tied, key=lambda r: (not r.is_n4, len(r.active_layers), r.action_norm, r.gates, r.state_token))

    def prune(self):
        """후속층을 역순으로 한 번씩 제거한다. L4는 삭제 대상이 아니다."""
        self._ensure_open()
        if self.pruning_status is not None:
            raise ContractError("PRUNING_ALREADY_ATTEMPTED")
        self.pruning_status = dict(attempted=0, completed=0, deleted=0, complete=False,
                                   budget_reason=None, attempted_layers=[], deleted_layers=[])
        for index in range(len(self.layers) - 1, 0, -1):
            incumbent = self.best()
            if incumbent.gates[index] == 0:
                continue
            gates = list(incumbent.gates)
            gates[index] = 0.
            self.pruning_status["attempted"] += 1
            self.pruning_status["attempted_layers"].append(self.layers[index])
            self._emit("PRUNING_ATTEMPT", layer=self.layers[index],
                       from_gates=incumbent.gates, proposed_gates=gates)
            try:
                self.evaluate(gates, phase="pruning")
            except BudgetExceeded as exc:
                if exc.owner != id(self):
                    raise
                self.pruning_status["budget_reason"] = str(exc)
                self._emit("PRUNING_INCOMPLETE", status=self.pruning_status)
                return self.best()
            self.pruning_status["completed"] += 1
            if self.best().gates[index] == 0:
                self.pruning_status["deleted"] += 1
                self.pruning_status["deleted_layers"].append(self.layers[index])
            self._emit("PRUNING_RESULT", layer=self.layers[index],
                       incumbent=self.candidate_receipt(self.best()), status=self.pruning_status)
        self.pruning_status["complete"] = True
        self._emit("PRUNING_COMPLETE", status=self.pruning_status)
        return self.best()

    def seal_selection(self, selected=None, *, allow_fixed=None):
        """Durably freeze the choice before any official observer/history append.

        The only infeasible selectable endpoint is the declared raw F48 endpoint
        produced by ``run_arm``. This is never a generic guard-bypass switch.
        """
        self._ensure_open()
        selected = self.best() if selected is None else selected
        if not any(selected is row for row in self.records):
            raise ContractError("SELECTION_NOT_COMPLETED_SCORED_CANDIDATE")
        fixed = self.policy_arm == "F48" and selected is self._fixed_candidate
        if allow_fixed is None:
            allow_fixed = fixed
        if not selected.feasible and not (allow_fixed and fixed):
            raise ContractError("INFEASIBLE_SELECTION_FORBIDDEN")
        if self.policy_arm == "F48" and not fixed:
            raise ContractError("F48_MUST_SELECT_RAW_FIXED_NO_FALLBACK")
        if not fixed and selected is not self.best():
            raise ContractError("SELECTION_NOT_GLOBAL_BEST_FEASIBLE")
        selected.scores.validate()
        self._validate_snapshot(selected.state, selected.state_token, self.entry_history)
        self._restore_entry()
        self._emit("SELECTION_SEALED", candidate=self.candidate_receipt(selected),
                   policy_arm=self.policy_arm, raw_fixed_exception=fixed,
                   history_appended=False, official_observer_access=False)
        self.selected, self.sealed = selected, True
        return selected

    def finalize(self, selected=None):
        if self.finalized:
            raise ContractError("ALREADY_FINALIZED")
        if self.failed is not None:
            raise ContractError("NO_FINALIZE_AFTER_TECHNICAL_FAILURE:" + self.failed)
        if not self.sealed:
            self.seal_selection(selected)
        elif selected is not None and selected is not self.selected:
            raise ContractError("SELECTED_ENDPOINT_CHANGED_AFTER_SEAL")
        selected = self.selected
        self._validate_snapshot(selected.state, selected.state_token, self.entry_history)
        self.backend.restore(selected.state)
        if self.backend.history_token() != self.entry_history:
            raise ContractError("SELECTED_SNAPSHOT_HISTORY_CHANGED")
        token = self.backend.state_token()
        if token != selected.state_token:
            raise ContractError("SELECTED_SNAPSHOT_STATE_CHANGED")
        try:
            self._emit("FINALIZATION_BEGIN", selected_token=token,
                       history_before=self.entry_history, layers=self.history_layers)
            self.backend.finalize_history(self.history_layers)
            if self.backend.state_token() != token:
                raise ContractError("FINALIZER_WEIGHT_MUTATION")
            self._emit("FINALIZATION_COMPLETE", selected_token=token,
                       history_after=self.backend.history_token(), layers=self.history_layers,
                       history_counts_verification="BACKEND_RESPONSIBILITY")
        except Exception as exc:
            self._technical_failure(exc, stage="FINALIZATION")
            self._restore_entry()
            raise
        self.finalized = True
        self.counts["commits"] += 1
        return selected


def run_cobyla(controller: Controller):
    """SciPy 1.15.3 legacy COBYLA. 실제 모델 또는 SciPy 검증을 대신하지 않는다."""
    import numpy as np
    import scipy
    from scipy.optimize import minimize
    if scipy.__version__ != "1.15.3":
        raise ContractError("SCIPY_VERSION_MUST_BE_1.15.3")

    def measured(u):
        return controller.measure_reductions(u)

    # u=0은 모든 physical gate=1이다. 최초 양의 좌표 probe는 영역 안쪽이다.
    try:
        result = minimize(lambda u: measured(u)[0], np.zeros(len(controller.layers)),
                          method="COBYLA", constraints=[dict(type="ineq", fun=lambda u: np.asarray(measured(u)[1]))],
                          options=dict(rhobeg=.25, tol=.01, catol=1e-8,
                                       maxiter=controller.limits.max_search_proposals, disp=False))
        stop = dict(reason="SOLVER_RETURN", status=int(result.status),
                    success=bool(result.success), nfev=int(result.nfev))
    except BudgetExceeded as exc:
        if exc.owner != id(controller):
            raise ContractError("NONCONTROLLER_BUDGET_EXCEPTION") from exc
        stop = dict(reason=str(exc), success=False, status=None, nfev=None)
    except Exception as exc:
        if controller.failed is None:
            controller._technical_failure(exc, stage="COBYLA_CALLBACK")
        raise
    controller.search_coverage = controller.coverage()
    controller._emit("SEARCH_STOP", stop=stop, coverage=controller.search_coverage,
                     scipy_version=scipy.__version__, implementation="legacy_COBYLA_not_PRIMA")
    # Solver 반환점이 아닌 모든 실측 품질-feasible 후보 중 incumbent를 반환한다.
    return controller.best(), stop


def run_arm(controller: Controller, arm: str, *, finalize=False):
    """Run exactly one declared six-arm policy; no observer/model task is added.

    Returns ``(Candidate, JSON-safe stop metadata)``. Normally the runner calls
    ``seal_selection(candidate)``, does read-only post-seal observers and finally
    ``finalize(candidate)``. With ``finalize=True`` the seal still precedes history.
    """
    controller._ensure_open()
    if arm not in ARM_LAYERS or controller.layers != ARM_LAYERS[arm]:
        raise ContractError("ARM_LAYER_ALLOWLIST_MISMATCH")
    if controller.history_layers != HISTORY_LAYERS:
        raise ContractError("ALL_SIX_ARMS_REQUIRE_FIVE_HISTORY_LAYERS")
    if controller.limits != Limits():
        raise ContractError("PRODUCTION_V2_LIMITS_CHANGED")
    if controller.policy_arm is not None:
        raise ContractError("ARM_POLICY_ALREADY_RUN")
    controller.policy_arm = arm
    controller._emit("ARM_POLICY_BEGIN", arm=arm)
    if arm == "N4":
        selected, stop = controller.n4, {"reason": "NATIVE_FULL", "optimizer_run": False}
    elif arm == "F48":
        selected = controller.evaluate((.75, .5))
        controller._fixed_candidate = selected
        stop = {"reason": "RAW_FIXED_NO_QUALITY_FALLBACK", "optimizer_run": False,
                "quality_observation": {"feasible": selected.feasible, "reasons": selected.reasons}}
    elif arm == "G48":
        for a4 in (.75, 1.):
            for a8 in (0., .5, 1.):
                controller.evaluate((a4, a8))
        selected, stop = controller.best(), {"reason": "COMPLETE_SIX_POINT_GRID", "optimizer_run": False}
    else:
        selected, stop = run_cobyla(controller)
        stop["optimizer_run"] = True
        if arm in ("C48", "C45678"):
            selected = controller.prune()
            stop["pruning"] = dict(controller.pruning_status)
        stop["coverage"] = dict(controller.search_coverage)
        stop["operational_search_status"] = ("COUNT_PROXY_ADEQUATE" if controller.search_coverage["adequate_by_count_proxy"]
                                              else "INSUFFICIENT_SEARCH")
    controller._emit("ARM_POLICY_COMPLETE", arm=arm, stop=stop,
                     selected=controller.candidate_receipt(selected))
    if finalize:
        controller.seal_selection(selected)
        controller.finalize(selected)
    return selected, json_safe(stop)
