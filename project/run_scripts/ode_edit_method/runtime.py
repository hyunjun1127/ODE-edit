"""Backend-neutral common-controller execution state machine.

The state machine is usable with a toy CPU backend today and with a guarded
MEMIT backend after a separate GPU execution envelope.  It never accepts an
evaluation prompt or held-out outcome as a controller input.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Mapping, Protocol, Sequence

from .contracts import (
    Arm,
    ArmRunResult,
    ControllerConfig,
    EventReading,
    MethodContractError,
    ProposalBatch,
    ProposalSemantics,
    StepRecord,
)
from .controller import (
    OmegaLedger,
    ScalarFirstHitSearch,
    assess_trial,
    normalized_share,
    solve_progress_qp,
    static_coefficients,
)
from .instrumentation import EditInstrumentation
from .events import ControllerRequest


class BackendCheckpoint(Protocol):
    state_id: str


class BackendTrial(Protocol):
    applied_coefficients: tuple[float, ...]


class MethodBackend(Protocol):
    """Minimal actuator/runtime surface required by every locked arm."""

    @property
    def layers(self) -> tuple[int, ...]: ...

    def current_state_id(self) -> str: ...

    def checkpoint(self) -> BackendCheckpoint: ...

    def restore(self, checkpoint: BackendCheckpoint) -> None: ...

    def assert_checkpoint(self, checkpoint: BackendCheckpoint) -> None: ...

    def compute_direct_z(self, request: ControllerRequest) -> Any: ...

    def event(self, request: ControllerRequest) -> EventReading: ...

    def build_native_terminal(self, frozen_target: Any) -> ProposalBatch: ...

    def build_synchronous(self, frozen_target: Any) -> ProposalBatch: ...

    def build_coordinate(self, frozen_target: Any, layer: int) -> ProposalBatch: ...

    def rebind_frozen(self, entry_batch: ProposalBatch) -> ProposalBatch: ...

    def entry_synchronous_distance(self) -> float: ...

    def trial(
        self, batch: ProposalBatch, coefficients: Sequence[float]
    ) -> ContextManager[BackendTrial]: ...

    def commit(
        self, batch: ProposalBatch, coefficients: Sequence[float]
    ) -> tuple[float, ...]: ...

    def terminal_net_energy(
        self, entry_checkpoint: BackendCheckpoint
    ) -> Mapping[int, float]: ...


@dataclass(slots=True)
class _DirectZOnce:
    backend: MethodBackend
    request: ControllerRequest
    instrumentation: EditInstrumentation | None = None
    value: Any = None
    compute_count: int = 0

    def get(self) -> Any:
        if self.compute_count == 0:
            timer = (
                self.instrumentation.component("direct_z")
                if self.instrumentation is not None
                else nullcontext()
            )
            with timer:
                self.value = self.backend.compute_direct_z(self.request)
            self.compute_count = 1
            if self.instrumentation is not None:
                self.instrumentation.increment("N_z")
        return self.value


def _prepare_event_target_if_required(
    backend: MethodBackend,
    target: _DirectZOnce,
) -> Any:
    """Prepare an edit-local event target before the first event, if required.

    Legacy backends deliberately have no ``prepare_event_target`` method and
    retain the original lazy direct-z/entry-hit behavior.  The oracle-mean V2
    backend implements the method, so direct-z is computed exactly once before
    its entry target and no N_z=0 path is representable for that backend.
    """

    prepare = getattr(backend, "prepare_event_target", None)
    if prepare is None:
        return None
    if not callable(prepare):
        raise MethodContractError("backend event-target preparer is not callable")
    frozen_target = target.get()
    prepare(frozen_target)
    return frozen_target


def _assert_batch_current(backend: MethodBackend, batch: ProposalBatch) -> None:
    current = backend.current_state_id()
    if batch.snapshot_id != current:
        raise MethodContractError(
            f"proposal snapshot {batch.snapshot_id} differs from current state {current}"
        )


def _relative_discrepancy(left: float, right: float) -> float:
    absolute = abs(left - right)
    denominator = max(abs(left), abs(right))
    return 0.0 if denominator == 0.0 else absolute / denominator


def _functional_commit_mismatch_message(
    trial: EventReading,
    committed: EventReading,
    *,
    atol: float,
    rtol: float,
) -> str:
    """Render only numeric T/C diagnostics; never include request payloads."""

    hard_abs = abs(trial.hard_phi - committed.hard_phi)
    smooth_abs = abs(trial.smooth_phi - committed.smooth_phi)
    return (
        "functional trial event differs from committed write: "
        f"trial_hard_phi={trial.hard_phi:.17g}, "
        f"committed_hard_phi={committed.hard_phi:.17g}, "
        f"hard_abs={hard_abs:.17g}, "
        f"hard_rel={_relative_discrepancy(trial.hard_phi, committed.hard_phi):.17g}, "
        f"trial_smooth_phi={trial.smooth_phi:.17g}, "
        f"committed_smooth_phi={committed.smooth_phi:.17g}, "
        f"smooth_abs={smooth_abs:.17g}, "
        f"smooth_rel={_relative_discrepancy(trial.smooth_phi, committed.smooth_phi):.17g}, "
        f"atol={atol:.17g}, rtol={rtol:.17g}"
    )


class FiveArmRunner:
    def __init__(
        self,
        config: ControllerConfig,
        omega: OmegaLedger,
        *,
        scale_rejected_progress: bool = False,
    ) -> None:
        self.config = config
        self.omega = omega
        self.scale_rejected_progress = bool(scale_rejected_progress)

    def run(
        self,
        arm: Arm,
        *,
        edit_id: str,
        request: ControllerRequest,
        backend: MethodBackend,
        instrumentation: EditInstrumentation | None = None,
    ) -> ArmRunResult:
        if tuple(backend.layers) != self.omega.layers:
            raise MethodContractError("backend layers differ from Omega geometry")
        if not isinstance(request, ControllerRequest):
            raise MethodContractError("runner accepts only a rewrite-only ControllerRequest")
        if instrumentation is not None:
            instrumentation.start_controller()
        try:
            if arm is Arm.NATIVE_MEMIT:
                return self._run_native(edit_id, request, backend, instrumentation)
            if arm is Arm.SCALAR_FIRST_HIT:
                return self._run_scalar(edit_id, request, backend, instrumentation)
            if arm in {
                Arm.STATIC_SYNCHRONOUS,
                Arm.ONE_REFRESH,
                Arm.ORDERED_ADAPTIVE,
                Arm.FULL_ODE_EDIT,
            }:
                return self._run_adaptive(
                    arm, edit_id, request, backend, instrumentation
                )
            raise MethodContractError(f"unsupported arm: {arm}")
        finally:
            if instrumentation is not None:
                instrumentation.stop_controller()

    def _run_native(
        self,
        edit_id: str,
        request: ControllerRequest,
        backend: MethodBackend,
        instrumentation: EditInstrumentation | None,
    ) -> ArmRunResult:
        entry = self._checkpoint(backend, instrumentation)
        frozen_omega = self.omega.frozen_for_edit(edit_id)
        del frozen_omega  # Native does not route, but freezes the same edit ledger.
        target = _DirectZOnce(backend, request, instrumentation)
        try:
            target_value = _prepare_event_target_if_required(backend, target)
            before = self._event(backend, request, instrumentation)
            if before.is_hit(self.config.event_tolerance):
                if instrumentation is not None:
                    instrumentation.mark_first_hit()
                terminal_energy = self._terminal_energy(
                    backend, entry, instrumentation
                )
                self.omega.append_terminal(edit_id, terminal_energy)
                return ArmRunResult(
                    arm=Arm.NATIVE_MEMIT,
                    edit_id=edit_id,
                    status="event_hit",
                    direct_z_compute_count=target.compute_count,
                    terminal_state_id=backend.current_state_id(),
                    omega_appended=True,
                    steps=(),
                )
            if target_value is None:
                target_value = target.get()
            batch = self._build_native(
                lambda: backend.build_native_terminal(target_value),
                instrumentation,
            )
            if batch.semantics is not ProposalSemantics.NATIVE_ORDERED_TERMINAL:
                raise MethodContractError("native backend returned non-native semantics")
            _assert_batch_current(backend, batch)
            coefficients = tuple(1.0 for _ in batch.proposals)
            applied = self._commit(
                backend, batch, coefficients, instrumentation
            )
            if instrumentation is not None:
                instrumentation.increment("K_acc")
            if applied != coefficients:
                raise MethodContractError("native applied coefficients changed")
            after = self._event(backend, request, instrumentation)
            if after.is_hit(self.config.event_tolerance) and instrumentation is not None:
                instrumentation.mark_first_hit()
            terminal_energy = self._terminal_energy(backend, entry, instrumentation)
            step = StepRecord(
                arm=Arm.NATIVE_MEMIT,
                position=0,
                layer=None,
                snapshot_id=batch.snapshot_id,
                direction_ids=batch.direction_ids,
                solver_coefficients=coefficients,
                applied_coefficients=coefficients,
                accepted=True,
                reason="canonical-native-terminal",
                hard_phi_before=before.hard_phi,
                hard_phi_after=after.hard_phi,
                smooth_phi_before=before.smooth_phi,
                smooth_phi_after=after.smooth_phi,
                trust_ratio=0.0,
            )
            result = ArmRunResult(
                arm=Arm.NATIVE_MEMIT,
                edit_id=edit_id,
                status=(
                    "event_hit"
                    if after.is_hit(self.config.event_tolerance)
                    else "native_event_fail"
                ),
                direct_z_compute_count=target.compute_count,
                terminal_state_id=backend.current_state_id(),
                omega_appended=True,
                steps=(step,),
                failure_type=(
                    None
                    if after.is_hit(self.config.event_tolerance)
                    else "native_event_fail"
                ),
            )
            self.omega.append_terminal(edit_id, terminal_energy)
            return result
        except BaseException:
            self._restore(backend, entry, instrumentation)
            raise

    def _run_scalar(
        self,
        edit_id: str,
        request: ControllerRequest,
        backend: MethodBackend,
        instrumentation: EditInstrumentation | None,
    ) -> ArmRunResult:
        entry = self._checkpoint(backend, instrumentation)
        self.omega.frozen_for_edit(edit_id)
        target = _DirectZOnce(backend, request, instrumentation)
        try:
            target_value = _prepare_event_target_if_required(backend, target)
            entry_event = self._event(backend, request, instrumentation)
            if entry_event.is_hit(self.config.event_tolerance):
                if instrumentation is not None:
                    instrumentation.mark_first_hit()
                terminal_energy = self._terminal_energy(
                    backend, entry, instrumentation
                )
                result = ArmRunResult(
                    arm=Arm.SCALAR_FIRST_HIT,
                    edit_id=edit_id,
                    status="event_hit",
                    direct_z_compute_count=target.compute_count,
                    terminal_state_id=backend.current_state_id(),
                    omega_appended=True,
                    steps=(),
                    scalar_alpha=0.0,
                    scalar_nonmonotonic=False,
                )
                self.omega.append_terminal(edit_id, terminal_energy)
                return result
            if target_value is None:
                target_value = target.get()
            batch = self._build_native(
                lambda: backend.build_native_terminal(target_value),
                instrumentation,
            )
            if batch.semantics is not ProposalSemantics.NATIVE_ORDERED_TERMINAL:
                raise MethodContractError("scalar arm did not freeze native terminal writes")
            _assert_batch_current(backend, batch)

            def probe(alpha: float) -> float:
                if alpha == 0.0:
                    return entry_event.hard_phi
                coefficients = tuple(alpha for _ in batch.proposals)
                state_before_trial_id = backend.current_state_id()
                if instrumentation is not None:
                    instrumentation.increment("N_trial")
                trial_timer = (
                    instrumentation.component("trial")
                    if instrumentation is not None
                    else nullcontext()
                )
                with trial_timer, backend.trial(batch, coefficients) as trial:
                    if tuple(trial.applied_coefficients) != coefficients:
                        raise MethodContractError("scalar trial alpha changed")
                    reading = self._event(
                        backend,
                        request,
                        instrumentation,
                        component=None,
                    )
                # Functional trials already guard storage/version/RNG.  The
                # state-id check is deliberately copy-free; exact dense entry
                # hashing on every scalar probe would hide a full-weight read
                # outside the trial timer.
                if backend.current_state_id() != state_before_trial_id:
                    raise MethodContractError("scalar trial changed model state")
                return reading.hard_phi

            search = ScalarFirstHitSearch(self.config).run(probe)
            if not search.hit or search.alpha is None:
                self._restore(backend, entry, instrumentation)
                return ArmRunResult(
                    arm=Arm.SCALAR_FIRST_HIT,
                    edit_id=edit_id,
                    status="scalar_event_fail",
                    direct_z_compute_count=target.compute_count,
                    terminal_state_id=backend.current_state_id(),
                    omega_appended=False,
                    steps=(),
                    failure_type="scalar_event_fail",
                    scalar_alpha=None,
                    scalar_nonmonotonic=search.nonmonotonic,
                )

            coefficients = tuple(search.alpha for _ in batch.proposals)
            before = entry_event
            applied = self._commit(
                backend, batch, coefficients, instrumentation
            )
            if instrumentation is not None:
                instrumentation.increment("K_acc")
            if applied != coefficients:
                raise MethodContractError("scalar applied alpha changed")
            after = self._event(backend, request, instrumentation)
            if not after.is_hit(self.config.event_tolerance):
                raise MethodContractError("scalar selected alpha did not reproduce first hit")
            if instrumentation is not None:
                instrumentation.mark_first_hit()
            terminal_energy = self._terminal_energy(backend, entry, instrumentation)
            step = StepRecord(
                arm=Arm.SCALAR_FIRST_HIT,
                position=0,
                layer=None,
                snapshot_id=batch.snapshot_id,
                direction_ids=batch.direction_ids,
                solver_coefficients=coefficients,
                applied_coefficients=coefficients,
                accepted=True,
                reason="scalar-first-hit",
                hard_phi_before=before.hard_phi,
                hard_phi_after=after.hard_phi,
                smooth_phi_before=before.smooth_phi,
                smooth_phi_after=after.smooth_phi,
                trust_ratio=0.0,
            )
            result = ArmRunResult(
                arm=Arm.SCALAR_FIRST_HIT,
                edit_id=edit_id,
                status="event_hit",
                direct_z_compute_count=target.compute_count,
                terminal_state_id=backend.current_state_id(),
                omega_appended=True,
                steps=(step,),
                scalar_alpha=search.alpha,
                scalar_nonmonotonic=search.nonmonotonic,
            )
            self.omega.append_terminal(edit_id, terminal_energy)
            return result
        except BaseException:
            self._restore(backend, entry, instrumentation)
            raise

    def _run_adaptive(
        self,
        arm: Arm,
        edit_id: str,
        request: ControllerRequest,
        backend: MethodBackend,
        instrumentation: EditInstrumentation | None,
    ) -> ArmRunResult:
        entry = self._checkpoint(backend, instrumentation)
        frozen_omega = self.omega.frozen_for_edit(edit_id)
        target = _DirectZOnce(backend, request, instrumentation)
        radius: float | None = None
        radius_cap: float | None = None
        steps: list[StepRecord] = []
        layers = tuple(backend.layers)
        accepted_count = 0
        accepted_cap = 2 if arm is Arm.ONE_REFRESH else self.config.s_max
        coordinate_cursor = 0
        zero_slope_streak = 0
        rejections = 0
        static_entry: ProposalBatch | None = None
        static_share: tuple[float, ...] | None = None
        retry_field: tuple[EventReading, ProposalBatch, int | None, str] | None = None
        rejected_candidate: tuple[str, float, tuple[float, ...]] | None = None
        cached_current_event: EventReading | None = None

        target_value: Any = None
        try:
            target_value = _prepare_event_target_if_required(backend, target)
            while accepted_count < accepted_cap:
                if retry_field is not None:
                    before, batch, layer, retry_state_id = retry_field
                    if backend.current_state_id() != retry_state_id:
                        raise MethodContractError(
                            "rejected retry state differs from its cached field"
                        )
                    _assert_batch_current(backend, batch)
                else:
                    if cached_current_event is None:
                        before = self._event(backend, request, instrumentation)
                    else:
                        before = cached_current_event
                        cached_current_event = None
                    if before.is_hit(self.config.event_tolerance):
                        if instrumentation is not None:
                            instrumentation.mark_first_hit()
                        terminal_energy = self._terminal_energy(
                            backend, entry, instrumentation
                        )
                        result = ArmRunResult(
                            arm=arm,
                            edit_id=edit_id,
                            status="event_hit",
                            direct_z_compute_count=target.compute_count,
                            terminal_state_id=backend.current_state_id(),
                            omega_appended=True,
                            steps=tuple(steps),
                        )
                        self.omega.append_terminal(edit_id, terminal_energy)
                        return result

                    if target_value is None:
                        target_value = target.get()

                    layer = None
                    if arm in {Arm.FULL_ODE_EDIT, Arm.ONE_REFRESH}:
                        batch = self._build_field(
                            lambda: backend.build_synchronous(target_value),
                            instrumentation,
                        )
                        if batch.semantics is not ProposalSemantics.CURRENT_SAME_SNAPSHOT:
                            raise MethodContractError(
                                "joint refresh requires current same-snapshot proposals"
                            )
                    elif arm is Arm.STATIC_SYNCHRONOUS:
                        if static_entry is None:
                            static_entry = self._build_field(
                                lambda: backend.build_synchronous(target_value),
                                instrumentation,
                            )
                            if (
                                static_entry.semantics
                                is not ProposalSemantics.CURRENT_SAME_SNAPSHOT
                            ):
                                raise MethodContractError(
                                    "Static entry is not same-snapshot"
                                )
                            batch = static_entry
                        else:
                            batch = backend.rebind_frozen(static_entry)
                            if batch.direction_ids != static_entry.direction_ids:
                                raise MethodContractError(
                                    "Static direction changed after entry"
                                )
                    else:
                        layer = layers[coordinate_cursor % len(layers)]
                        batch = self._build_field(
                            lambda: backend.build_coordinate(target_value, layer),
                            instrumentation,
                        )
                        if (
                            batch.semantics is not ProposalSemantics.CURRENT_COORDINATE
                            or batch.layers != (layer,)
                        ):
                            raise MethodContractError(
                                "Ordered arm violated coordinate identity"
                            )
                    _assert_batch_current(backend, batch)

                if radius is None:
                    scale_timer = (
                        instrumentation.component("trust_scale")
                        if instrumentation is not None
                        else nullcontext()
                    )
                    with scale_timer:
                        d_sync_entry = float(backend.entry_synchronous_distance())
                    if (
                        not math.isfinite(d_sync_entry)
                        or d_sync_entry <= self.config.trust_denominator_epsilon
                    ):
                        raise MethodContractError(
                            "entry synchronous C-distance is degenerate"
                        )
                    radius = self.config.h0_fraction * d_sync_entry
                    radius_cap = self.config.h_max_fraction * d_sync_entry

                assert radius_cap is not None
                retry_index = rejections if retry_field is not None else 0
                retry_scale = (
                    self.config.gamma_down**retry_index
                    if self.scale_rejected_progress
                    else 1.0
                )

                qp_timer = (
                    instrumentation.component("qp")
                    if instrumentation is not None
                    else nullcontext()
                )
                with qp_timer:
                    if arm is Arm.STATIC_SYNCHRONOUS and static_share is not None:
                        solution = static_coefficients(
                            layers=batch.layers,
                            frozen_share=static_share,
                            frozen_slopes=static_entry.slopes,
                            hard_phi=before.hard_phi,
                            trust_radius=radius,
                            config=self.config,
                            progress_scale=retry_scale,
                        )
                    else:
                        solution = solve_progress_qp(
                            layers=batch.layers,
                            slopes=batch.slopes,
                            omega={
                                layer_id: frozen_omega[layer_id]
                                for layer_id in batch.layers
                            },
                            denominators={
                                layer_id: self.omega.denominators[layer_id]
                                for layer_id in batch.layers
                            },
                            hard_phi=before.hard_phi,
                            trust_radius=radius,
                            config=self.config,
                            progress_scale=retry_scale,
                        )
                        if (
                            arm is Arm.STATIC_SYNCHRONOUS
                            and solution.coefficient_norm > 0.0
                        ):
                            static_share = normalized_share(
                                solution.coefficients,
                                self.config.progress_epsilon,
                            )

                if self.scale_rejected_progress and retry_index > 0:
                    if rejected_candidate is None:
                        raise MethodContractError(
                            "unchanged-state retry lacks its rejected candidate"
                        )
                    rejected_state, rejected_progress, rejected_coefficients = (
                        rejected_candidate
                    )
                    if rejected_state != backend.current_state_id():
                        raise MethodContractError(
                            "unchanged-state retry candidate state differs"
                        )
                    if (
                        solution.coefficient_norm > self.config.progress_epsilon
                        and (
                            solution.requested_progress == rejected_progress
                            or solution.coefficients == rejected_coefficients
                        )
                    ):
                        raise MethodContractError(
                            "unchanged-state rejection retry replayed its QP candidate"
                        )

                if solution.predicted_progress <= self.config.progress_epsilon:
                    if arm is Arm.ORDERED_ADAPTIVE:
                        coordinate_cursor += 1
                        zero_slope_streak += 1
                        rejections = 0
                        retry_field = None
                        if zero_slope_streak >= len(layers):
                            self._restore(backend, entry, instrumentation)
                            return ArmRunResult(
                                arm=arm,
                                edit_id=edit_id,
                                status="no_positive_slope",
                                direct_z_compute_count=target.compute_count,
                                terminal_state_id=backend.current_state_id(),
                                omega_appended=False,
                                steps=tuple(steps),
                                failure_type="no_positive_slope",
                            )
                        cached_current_event = before
                        continue
                    self._restore(backend, entry, instrumentation)
                    return ArmRunResult(
                        arm=arm,
                        edit_id=edit_id,
                        status="no_positive_slope",
                        direct_z_compute_count=target.compute_count,
                        terminal_state_id=backend.current_state_id(),
                        omega_appended=False,
                        steps=tuple(steps),
                        failure_type="no_positive_slope",
                    )

                # A functional trial is read-only.  Use the state identity as
                # the rollback guard; taking another dense checkpoint here
                # would reintroduce a dense trial copy.
                state_before_trial_id = backend.current_state_id()
                if instrumentation is not None:
                    instrumentation.increment("N_trial")
                trial_timer = (
                    instrumentation.component("trial")
                    if instrumentation is not None
                    else nullcontext()
                )
                with trial_timer, backend.trial(batch, solution.coefficients) as trial:
                    after = self._event(
                        backend,
                        request,
                        instrumentation,
                        component=None,
                    )
                    verdict = assess_trial(
                        before=before,
                        after=after,
                        predicted_progress=solution.predicted_progress,
                        trust_radius=radius,
                        trust_radius_cap=radius_cap,
                        config=self.config,
                    )
                    applied = tuple(trial.applied_coefficients)
                    if applied != solution.coefficients:
                        raise MethodContractError("post-QP applied coefficients changed")
                if backend.current_state_id() != state_before_trial_id:
                    raise MethodContractError("read-only trial changed model state")
                if verdict.accepted:
                    committed = self._commit(
                        backend,
                        batch,
                        solution.coefficients,
                        instrumentation,
                    )
                    if committed != solution.coefficients:
                        raise MethodContractError("accepted write coefficients changed")
                    if instrumentation is not None:
                        instrumentation.increment("K_acc")
                    committed_event = self._event(
                        backend,
                        request,
                        instrumentation,
                    )
                    if not (
                        math.isclose(
                            committed_event.hard_phi,
                            after.hard_phi,
                            abs_tol=self.config.functional_commit_atol,
                            rel_tol=self.config.functional_commit_rtol,
                        )
                        and math.isclose(
                            committed_event.smooth_phi,
                            after.smooth_phi,
                            abs_tol=self.config.functional_commit_atol,
                            rel_tol=self.config.functional_commit_rtol,
                        )
                    ):
                        raise MethodContractError(
                            _functional_commit_mismatch_message(
                                after,
                                committed_event,
                                atol=self.config.functional_commit_atol,
                                rtol=self.config.functional_commit_rtol,
                            )
                        )
                    after = committed_event
                steps.append(
                    StepRecord(
                        arm=arm,
                        position=accepted_count,
                        layer=layer,
                        snapshot_id=batch.snapshot_id,
                        direction_ids=batch.direction_ids,
                        solver_coefficients=solution.coefficients,
                        applied_coefficients=applied,
                        accepted=verdict.accepted,
                        reason=verdict.reason,
                        hard_phi_before=before.hard_phi,
                        hard_phi_after=after.hard_phi,
                        smooth_phi_before=before.smooth_phi,
                        smooth_phi_after=after.smooth_phi,
                        trust_ratio=verdict.ratio,
                        radius=radius,
                        radius_cap=radius_cap,
                        requested_progress=solution.requested_progress,
                        predicted_progress=solution.predicted_progress,
                        coefficient_l2=solution.coefficient_norm,
                        retry_index=retry_index,
                        retry_scale=retry_scale,
                    )
                )
                radius = verdict.next_radius
                if verdict.accepted:
                    rejections = 0
                    retry_field = None
                    rejected_candidate = None
                    accepted_count += 1
                    zero_slope_streak = 0
                    if arm is Arm.ORDERED_ADAPTIVE:
                        coordinate_cursor += 1
                    committed_hit = after.is_hit(self.config.event_tolerance)
                    if not committed_hit:
                        cached_current_event = after
                    if committed_hit:
                        if instrumentation is not None:
                            instrumentation.mark_first_hit()
                        terminal_energy = self._terminal_energy(
                            backend, entry, instrumentation
                        )
                        result = ArmRunResult(
                            arm=arm,
                            edit_id=edit_id,
                            status="event_hit",
                            direct_z_compute_count=target.compute_count,
                            terminal_state_id=backend.current_state_id(),
                            omega_appended=True,
                            steps=tuple(steps),
                        )
                        self.omega.append_terminal(edit_id, terminal_energy)
                        return result
                else:
                    rejections += 1
                    if instrumentation is not None:
                        instrumentation.increment("N_reject")
                    retry_field = (
                        before,
                        batch,
                        layer,
                        backend.current_state_id(),
                    )
                    rejected_candidate = (
                        backend.current_state_id(),
                        solution.requested_progress,
                        solution.coefficients,
                    )
                    if rejections >= self.config.max_rejections_per_state:
                        self._restore(backend, entry, instrumentation)
                        return ArmRunResult(
                            arm=arm,
                            edit_id=edit_id,
                            status="trust_rejection_limit",
                            direct_z_compute_count=target.compute_count,
                            terminal_state_id=backend.current_state_id(),
                            omega_appended=False,
                            steps=tuple(steps),
                            failure_type="trust_rejection_limit",
                        )

            self._restore(backend, entry, instrumentation)
            return ArmRunResult(
                arm=arm,
                edit_id=edit_id,
                status=(
                    "one_refresh_event_fail"
                    if arm is Arm.ONE_REFRESH
                    else "resolution_cap_unresolved"
                ),
                direct_z_compute_count=target.compute_count,
                terminal_state_id=backend.current_state_id(),
                omega_appended=False,
                steps=tuple(steps),
                failure_type=(
                    "one_refresh_event_fail"
                    if arm is Arm.ONE_REFRESH
                    else "resolution_cap_unresolved"
                ),
            )
        except BaseException:
            self._restore(backend, entry, instrumentation)
            raise

    @staticmethod
    def _event(
        backend: MethodBackend,
        request: ControllerRequest,
        instrumentation: EditInstrumentation | None,
        *,
        component: str | None = "event",
    ) -> EventReading:
        timer = (
            instrumentation.component(component)
            if instrumentation is not None and component is not None
            else nullcontext()
        )
        forward_scope = (
            instrumentation.model_forward_scope("event")
            if instrumentation is not None
            else nullcontext()
        )
        with timer, forward_scope:
            reading = backend.event(request)
        if (
            instrumentation is not None
            and not instrumentation.tracks_model_forwards
            and reading.nfe
        ):
            for _ in range(reading.nfe):
                instrumentation.record_model_forward("event")
        return reading

    @staticmethod
    def _build_field(
        build: Any,
        instrumentation: EditInstrumentation | None,
    ) -> ProposalBatch:
        # The concrete backend records non-overlapping proposal, state-forward,
        # and hook-backward components.  This wrapper owns only logical build
        # counters so those component timers are never nested/double-counted.
        batch = build()
        if instrumentation is not None:
            instrumentation.increment("N_field")
            instrumentation.increment("N_proposal_build")
        return batch

    @staticmethod
    def _build_native(
        build: Any,
        instrumentation: EditInstrumentation | None,
    ) -> ProposalBatch:
        timer = (
            instrumentation.component("native_proposal")
            if instrumentation is not None
            else nullcontext()
        )
        with timer:
            batch = build()
        if instrumentation is not None:
            instrumentation.increment("N_proposal_build")
            instrumentation.increment("N_native_sweep")
        return batch

    @staticmethod
    def _commit(
        backend: MethodBackend,
        batch: ProposalBatch,
        coefficients: Sequence[float],
        instrumentation: EditInstrumentation | None,
    ) -> tuple[float, ...]:
        timer = (
            instrumentation.component("commit_write")
            if instrumentation is not None
            else nullcontext()
        )
        with timer:
            applied = tuple(backend.commit(batch, coefficients))
        if instrumentation is not None:
            instrumentation.increment("N_write")
        return applied

    @staticmethod
    def _checkpoint(
        backend: MethodBackend,
        instrumentation: EditInstrumentation | None,
    ) -> BackendCheckpoint:
        timer = (
            instrumentation.component("entry_checkpoint")
            if instrumentation is not None
            else nullcontext()
        )
        with timer:
            return backend.checkpoint()

    @staticmethod
    def _restore(
        backend: MethodBackend,
        checkpoint: BackendCheckpoint,
        instrumentation: EditInstrumentation | None,
    ) -> None:
        timer = (
            instrumentation.component("restore")
            if instrumentation is not None
            else nullcontext()
        )
        with timer:
            backend.restore(checkpoint)

    @staticmethod
    def _terminal_energy(
        backend: MethodBackend,
        checkpoint: BackendCheckpoint,
        instrumentation: EditInstrumentation | None,
    ) -> Mapping[int, float]:
        timer = (
            instrumentation.component("terminal_geometry")
            if instrumentation is not None
            else nullcontext()
        )
        with timer:
            return backend.terminal_net_energy(checkpoint)
