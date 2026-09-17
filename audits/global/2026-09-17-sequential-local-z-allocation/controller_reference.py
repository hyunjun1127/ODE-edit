"""순차 local-z 제어기의 CPU 계약 참조 구현.

실제 모델 runner가 아니다. Backend의 opaque snapshot/token은 실제 adapter에서
전체 parameter/context/RNG/source 연결을 보장해야 한다. 이 파일은 z나 native
solve를 근사하지 않으며, callback 순서·캐시·예산·품질·commit만 정의한다.
SciPy는 run_cobyla에서만 import한다. 기본 테스트는 Python 표준 라이브러리다.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional, Protocol


class ContractError(RuntimeError):
    """기술 오류는 정상적인 품질 fallback과 구분한다."""


class BudgetExceeded(RuntimeError):
    pass


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
                 limits: Limits = Limits(), history_layers=None):
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
        self.entry = backend.snapshot()
        self.entry_token, self.entry_history = backend.state_token(), backend.history_token()
        self.fit_cache, self.score_cache, self.gate_cache = {}, {}, {}
        self.reduction_cache = {}
        self.records, self.finalized = [], False
        self.search_completed_gates = []
        self.search_coverage = None
        self.pruning_status = None
        self.counts = dict(l4_fits=0, suffix_fits=0, baseline_scores=0, endpoints=0,
                           search_proposals=0, pruning_proposals=0, fit_cache_hits=0,
                           endpoint_cache_hits=0, gate_cache_hits=0, bound_rejections=0,
                           commits=0, l4_adam=0, extra_adam=0, reduction_cache_hits=0)
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
        finally:
            backend.restore(self.entry)

    def _limit(self, kind, phase):
        cap = getattr(self.limits, "max_" + kind)
        if phase == "search":
            cap -= getattr(self.limits, "reserve_" + kind + "_for_pruning")
        return cap

    def _path(self, gates, phase):
        b = self.backend
        b.restore(self.entry)
        for layer, gate in zip(self.layers, gates):
            if gate == 0:
                continue
            before = b.snapshot()
            key = (self.namespace, layer, b.state_token(), b.history_token())
            if key in self.fit_cache:
                native = self.fit_cache[key]
                self.counts["fit_cache_hits"] += 1
            else:
                if layer == 4:
                    if self.counts["l4_fits"]:
                        raise ContractError("L4_ENTRY_CACHE_IDENTITY_CHANGED")
                    self.counts["l4_fits"] += 1
                else:
                    if self.counts["endpoints"] >= self._limit("endpoints", phase):
                        raise BudgetExceeded("ENDPOINT_RESERVE")
                    if self.counts["suffix_fits"] >= self._limit("suffix_fits", phase):
                        raise BudgetExceeded("SUFFIX_FIT_CAP")
                    if (self.limits.max_extra_adam is not None and
                            self.limits.max_extra_adam - self.counts["extra_adam"] < self.limits.max_adam_per_suffix_fit):
                        raise BudgetExceeded("EXTRA_ADAM_RESERVE")
                    self.counts["suffix_fits"] += 1
                history = b.history_token()
                steps = b.fit_native(layer)
                if not isinstance(steps, int) or isinstance(steps, bool) or not 0 <= steps <= self.limits.max_adam_per_suffix_fit:
                    raise ContractError("INVALID_ACTUAL_ADAM_COUNT")
                self.counts["l4_adam" if layer == 4 else "extra_adam"] += steps
                # 직렬 호출이므로 worst-case reserve는 검사만 한다. 실제 step만
                # 차감하여 사용하지 않은 reserve는 다음 fit에 바로 반환한다.
                native = b.snapshot()
                if b.history_token() != history:
                    raise ContractError("INNER_HISTORY_MUTATION")
                if set(b.changed_layers(before, native)) - {layer}:
                    raise ContractError("NATIVE_NONSELECTED_MUTATION")
                self.fit_cache[key] = native
            b.restore(before)
            b.apply_gate(layer, before, native, gate)
            if b.history_token() != self.entry_history:
                raise ContractError("GATE_HISTORY_MUTATION")
            if set(b.changed_layers(before, b.snapshot())) - {layer}:
                raise ContractError("GATE_NONSELECTED_MUTATION")
        return b.snapshot(), b.state_token()

    def _score(self, token, *, baseline=False, phase="search"):
        key = (self.namespace, self.entry_history, token)
        if key in self.score_cache:
            self.counts["endpoint_cache_hits"] += 1
            return self.score_cache[key]
        if baseline:
            self.counts["baseline_scores"] += 1
        else:
            if self.counts["endpoints"] >= self._limit("endpoints", phase):
                raise BudgetExceeded("ENDPOINT_CAP")
            self.counts["endpoints"] += 1
        score = self.backend.score()
        score.validate()
        if self.backend.state_token() != token or self.backend.history_token() != self.entry_history:
            raise ContractError("OBSERVER_STATE_MUTATION")
        self.score_cache[key] = score
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
        if self.finalized:
            raise ContractError("ALREADY_FINALIZED")
        if phase not in ("search", "pruning"):
            raise ValueError("UNKNOWN_PHASE")
        gates = tuple(float(x) for x in gates)
        if len(gates) != len(self.layers) or not all(math.isfinite(x) for x in gates):
            raise ContractError("INVALID_GATE_VECTOR")
        if gates in self.gate_cache:
            self.counts["gate_cache_hits"] += 1
            return self.gate_cache[gates]
        counter = "search_proposals" if phase == "search" else "pruning_proposals"
        if phase == "search" and self.counts[counter] >= self.limits.max_search_proposals:
            raise BudgetExceeded("SEARCH_PROPOSAL_CAP")
        self.counts[counter] += 1
        if any(x < 0 or x > 1 for x in gates):
            self.counts["bound_rejections"] += 1
            row = Candidate(gates, None, None, None, False, ("RAW_BOUNDS",))
            self.gate_cache[gates] = row
            return row
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
            return row
        finally:
            self.backend.restore(self.entry)

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
        if self.finalized:
            raise ContractError("ALREADY_FINALIZED")
        u = tuple(float(x) for x in reductions)
        if len(u) != len(self.layers) or not all(math.isfinite(x) for x in u):
            raise ContractError("INVALID_REDUCTION_VECTOR")
        if u in self.reduction_cache:
            self.counts["reduction_cache_hits"] += 1
            return self.reduction_cache[u]
        # Gate 하한=1-u, gate 상한 slack=u. 후자는 다시 1-(1-u)로 계산하지 않는다.
        box = tuple(v for x in u for v in (1. - x, x))
        if any(x < 0 or x > 1 for x in u):
            if self.counts["search_proposals"] >= self.limits.max_search_proposals:
                raise BudgetExceeded("SEARCH_PROPOSAL_CAP")
            self.counts["search_proposals"] += 1
            self.counts["bound_rejections"] += 1
            mean_count = 1 + int(self.reference.past_h is not None) + int(self.limits.guard_canonical_mean)
            margin_count = len(self._margin_panels()) if self.limits.proposal_margin_guards else 0
            result = (1., box + (-1.,) * (mean_count + margin_count))
        else:
            objective, slack = self.objective_and_constraints(tuple(1. - x for x in u))
            result = (objective, box + slack[2 * len(u):])
        self.reduction_cache[u] = result
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
        feasible = [r for r in self.records if r.feasible]
        minimum = min(r.scores.base_kl for r in feasible)
        tied = [r for r in feasible if r.scores.base_kl <= minimum + self.limits.epsilon_b]
        # 매번 모든 실측 feasible 후보의 global minimum을 사용: epsilon 누적 금지.
        return min(tied, key=lambda r: (not r.is_n4, len(r.active_layers), r.action_norm, r.gates, r.state_token))

    def prune(self):
        """후속층을 역순으로 한 번씩 제거한다. L4는 삭제 대상이 아니다."""
        if self.finalized:
            raise ContractError("ALREADY_FINALIZED")
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
            try:
                self.evaluate(gates, phase="pruning")
            except BudgetExceeded as exc:
                self.pruning_status["budget_reason"] = str(exc)
                return self.best()
            self.pruning_status["completed"] += 1
            if self.best().gates[index] == 0:
                self.pruning_status["deleted"] += 1
                self.pruning_status["deleted_layers"].append(self.layers[index])
        self.pruning_status["complete"] = True
        return self.best()

    def finalize(self):
        if self.finalized:
            raise ContractError("ALREADY_FINALIZED")
        selected = self.best()
        self.backend.restore(selected.state)
        token = self.backend.state_token()
        try:
            self.backend.finalize_history(self.history_layers)
            if self.backend.state_token() != token:
                raise ContractError("FINALIZER_WEIGHT_MUTATION")
        except Exception:
            self.backend.restore(self.entry)
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
        stop = dict(reason=str(exc), success=False, status=None, nfev=None)
    controller.search_coverage = controller.coverage()
    # Solver 반환점이 아닌 모든 실측 품질-feasible 후보 중 incumbent를 반환한다.
    return controller.best(), stop
