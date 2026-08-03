"""Backend-neutral common-controller execution state machine.

The state machine is usable with a toy CPU backend today and with a guarded
MEMIT backend after a separate GPU execution envelope.  It never accepts an
evaluation prompt or held-out outcome as a controller input.
"""

from __future__ import annotations

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


def _assert_batch_current(backend: MethodBackend, batch: ProposalBatch) -> None:
    current = backend.current_state_id()
    if batch.snapshot_id != current:
        raise MethodContractError(
            f"proposal snapshot {batch.snapshot_id} differs from current state {current}"
        )


class FiveArmRunner:
    def __init__(self, config: ControllerConfig, omega: OmegaLedger) -> None:
        self.config = config
        self.omega = omega

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

    def _run_native(
        self,
        edit_id: str,
        request: ControllerRequest,
        backend: MethodBackend,
        instrumentation: EditInstrumentation | None,
    ) -> ArmRunResult:
        entry = backend.checkpoint()
        frozen_omega = self.omega.frozen_for_edit(edit_id)
        del frozen_omega  # Native does not route, but freezes the same edit ledger.
        target = _DirectZOnce(backend, request, instrumentation)
        try:
            before = self._event(backend, request, instrumentation)
            batch = backend.build_native_terminal(target.get())
            if batch.semantics is not ProposalSemantics.NATIVE_ORDERED_TERMINAL:
                raise MethodContractError("native backend returned non-native semantics")
            _assert_batch_current(backend, batch)
            coefficients = tuple(1.0 for _ in batch.proposals)
            applied = self._commit(
                backend, batch, coefficients, instrumentation
            )
            if applied != coefficients:
                raise MethodContractError("native applied coefficients changed")
            after = self._event(backend, request, instrumentation)
            if after.is_hit(self.config.event_tolerance) and instrumentation is not None:
                instrumentation.mark_first_hit()
            terminal_energy = backend.terminal_net_energy(entry)
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
            backend.restore(entry)
            raise

    def _run_scalar(
        self,
        edit_id: str,
        request: ControllerRequest,
        backend: MethodBackend,
        instrumentation: EditInstrumentation | None,
    ) -> ArmRunResult:
        entry = backend.checkpoint()
        self.omega.frozen_for_edit(edit_id)
        target = _DirectZOnce(backend, request, instrumentation)
        try:
            target.get()
            entry_event = self._event(backend, request, instrumentation)
            if entry_event.is_hit(self.config.event_tolerance):
                if instrumentation is not None:
                    instrumentation.mark_first_hit()
                terminal_energy = backend.terminal_net_energy(entry)
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
            batch = backend.build_native_terminal(target.get())
            if batch.semantics is not ProposalSemantics.NATIVE_ORDERED_TERMINAL:
                raise MethodContractError("scalar arm did not freeze native terminal writes")
            _assert_batch_current(backend, batch)

            def probe(alpha: float) -> float:
                if alpha == 0.0:
                    return entry_event.hard_phi
                coefficients = tuple(alpha for _ in batch.proposals)
                if instrumentation is not None:
                    instrumentation.increment("N_trial")
                with backend.trial(batch, coefficients) as trial:
                    if tuple(trial.applied_coefficients) != coefficients:
                        raise MethodContractError("scalar trial alpha changed")
                    reading = self._event(
                        backend,
                        request,
                        instrumentation,
                        component="trial",
                    )
                backend.assert_checkpoint(entry)
                return reading.hard_phi

            search = ScalarFirstHitSearch(self.config).run(probe)
            if not search.hit or search.alpha is None:
                backend.restore(entry)
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
            if applied != coefficients:
                raise MethodContractError("scalar applied alpha changed")
            after = self._event(backend, request, instrumentation)
            if not after.is_hit(self.config.event_tolerance):
                raise MethodContractError("scalar selected alpha did not reproduce first hit")
            if instrumentation is not None:
                instrumentation.mark_first_hit()
            terminal_energy = backend.terminal_net_energy(entry)
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
            backend.restore(entry)
            raise

    def _run_adaptive(
        self,
        arm: Arm,
        edit_id: str,
        request: ControllerRequest,
        backend: MethodBackend,
        instrumentation: EditInstrumentation | None,
    ) -> ArmRunResult:
        entry = backend.checkpoint()
        frozen_omega = self.omega.frozen_for_edit(edit_id)
        target = _DirectZOnce(backend, request, instrumentation)
        radius = self.config.h0
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

        try:
            target_value = target.get()
            while accepted_count < accepted_cap:
                if retry_field is not None:
                    before, batch, layer, retry_state_id = retry_field
                    if backend.current_state_id() != retry_state_id:
                        raise MethodContractError(
                            "rejected retry state differs from its cached field"
                        )
                    _assert_batch_current(backend, batch)
                else:
                    before = self._event(backend, request, instrumentation)
                    if before.is_hit(self.config.event_tolerance):
                        if instrumentation is not None:
                            instrumentation.mark_first_hit()
                        terminal_energy = backend.terminal_net_energy(entry)
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
                        )
                        if (
                            arm is Arm.STATIC_SYNCHRONOUS
                            and solution.coefficient_norm > 0.0
                        ):
                            static_share = normalized_share(
                                solution.coefficients,
                                self.config.progress_epsilon,
                            )

                if solution.predicted_progress <= self.config.progress_epsilon:
                    if arm is Arm.ORDERED_ADAPTIVE:
                        coordinate_cursor += 1
                        zero_slope_streak += 1
                        rejections = 0
                        retry_field = None
                        if zero_slope_streak >= len(layers):
                            backend.restore(entry)
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
                        continue
                    backend.restore(entry)
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
                with backend.trial(batch, solution.coefficients) as trial:
                    after = self._event(
                        backend,
                        request,
                        instrumentation,
                        component="trial",
                    )
                    verdict = assess_trial(
                        before=before,
                        after=after,
                        predicted_progress=solution.predicted_progress,
                        trust_radius=radius,
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
                    )
                )
                radius = verdict.next_radius
                if verdict.accepted:
                    rejections = 0
                    retry_field = None
                    accepted_count += 1
                    zero_slope_streak = 0
                    if arm is Arm.ORDERED_ADAPTIVE:
                        coordinate_cursor += 1
                    if verdict.hit:
                        if instrumentation is not None:
                            instrumentation.mark_first_hit()
                        terminal_energy = backend.terminal_net_energy(entry)
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
                    if rejections >= self.config.max_rejections_per_state:
                        backend.restore(entry)
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

            backend.restore(entry)
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
            backend.restore(entry)
            raise

    @staticmethod
    def _event(
        backend: MethodBackend,
        request: ControllerRequest,
        instrumentation: EditInstrumentation | None,
        *,
        component: str = "event",
    ) -> EventReading:
        timer = (
            instrumentation.component(component)
            if instrumentation is not None
            else nullcontext()
        )
        with timer:
            reading = backend.event(request)
        if instrumentation is not None and reading.nfe:
            instrumentation.increment("N_state_fwd", reading.nfe)
        return reading

    @staticmethod
    def _build_field(
        build: Any,
        instrumentation: EditInstrumentation | None,
    ) -> ProposalBatch:
        timer = (
            instrumentation.component("field")
            if instrumentation is not None
            else nullcontext()
        )
        with timer:
            batch = build()
        if instrumentation is not None:
            instrumentation.increment("N_field")
            instrumentation.increment("N_bw")
            instrumentation.increment("N_state_fwd")
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
