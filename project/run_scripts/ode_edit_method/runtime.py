"""Backend-neutral five-arm execution state machine.

The state machine is usable with a toy CPU backend today and with a guarded
MEMIT backend after a separate GPU execution envelope.  It never accepts an
evaluation prompt or held-out outcome as a controller input.
"""

from __future__ import annotations

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


class BackendCheckpoint(Protocol):
    state_id: str


class BackendTrial(Protocol):
    applied_coefficients: tuple[float, ...]

    def commit(self) -> None: ...


class MethodBackend(Protocol):
    """Minimal actuator/runtime surface required by the five-arm engine."""

    @property
    def layers(self) -> tuple[int, ...]: ...

    def current_state_id(self) -> str: ...

    def checkpoint(self) -> BackendCheckpoint: ...

    def restore(self, checkpoint: BackendCheckpoint) -> None: ...

    def assert_checkpoint(self, checkpoint: BackendCheckpoint) -> None: ...

    def compute_direct_z(self, request: Any) -> Any: ...

    def event(self, request: Any) -> EventReading: ...

    def build_native_terminal(self, frozen_target: Any) -> ProposalBatch: ...

    def build_synchronous(self, frozen_target: Any) -> ProposalBatch: ...

    def build_coordinate(self, frozen_target: Any, layer: int) -> ProposalBatch: ...

    def rebind_frozen(self, entry_batch: ProposalBatch) -> ProposalBatch: ...

    def trial(
        self, batch: ProposalBatch, coefficients: Sequence[float]
    ) -> ContextManager[BackendTrial]: ...

    def terminal_net_energy(
        self, entry_checkpoint: BackendCheckpoint
    ) -> Mapping[int, float]: ...


@dataclass(slots=True)
class _DirectZOnce:
    backend: MethodBackend
    request: Any
    value: Any = None
    compute_count: int = 0

    def get(self) -> Any:
        if self.compute_count == 0:
            self.value = self.backend.compute_direct_z(self.request)
            self.compute_count = 1
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
        request: Any,
        backend: MethodBackend,
    ) -> ArmRunResult:
        if tuple(backend.layers) != self.omega.layers:
            raise MethodContractError("backend layers differ from Omega geometry")
        if arm is Arm.NATIVE_MEMIT:
            return self._run_native(edit_id, request, backend)
        if arm is Arm.SCALAR_FIRST_HIT:
            return self._run_scalar(edit_id, request, backend)
        if arm in {
            Arm.STATIC_SYNCHRONOUS,
            Arm.ORDERED_ADAPTIVE,
            Arm.FULL_ODE_EDIT,
        }:
            return self._run_adaptive(arm, edit_id, request, backend)
        raise MethodContractError(f"unsupported arm: {arm}")

    def _run_native(
        self, edit_id: str, request: Any, backend: MethodBackend
    ) -> ArmRunResult:
        entry = backend.checkpoint()
        frozen_omega = self.omega.frozen_for_edit(edit_id)
        del frozen_omega  # Native does not route, but freezes the same edit ledger.
        target = _DirectZOnce(backend, request)
        before = backend.event(request)
        try:
            batch = backend.build_native_terminal(target.get())
            if batch.semantics is not ProposalSemantics.NATIVE_ORDERED_TERMINAL:
                raise MethodContractError("native backend returned non-native semantics")
            _assert_batch_current(backend, batch)
            coefficients = tuple(1.0 for _ in batch.proposals)
            with backend.trial(batch, coefficients) as trial:
                after = backend.event(request)
                if tuple(trial.applied_coefficients) != coefficients:
                    raise MethodContractError("native applied coefficients changed")
                trial.commit()
            receipt = self.omega.append_terminal(
                edit_id, backend.terminal_net_energy(entry)
            )
            del receipt
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
            return ArmRunResult(
                arm=Arm.NATIVE_MEMIT,
                edit_id=edit_id,
                status=(
                    "event_hit" if after.is_hit(self.config.event_tolerance) else "native_event_fail"
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
        except BaseException:
            backend.restore(entry)
            raise

    def _run_scalar(
        self, edit_id: str, request: Any, backend: MethodBackend
    ) -> ArmRunResult:
        entry = backend.checkpoint()
        self.omega.frozen_for_edit(edit_id)
        target = _DirectZOnce(backend, request)
        try:
            batch = backend.build_native_terminal(target.get())
            if batch.semantics is not ProposalSemantics.NATIVE_ORDERED_TERMINAL:
                raise MethodContractError("scalar arm did not freeze native terminal writes")
            _assert_batch_current(backend, batch)

            def probe(alpha: float) -> float:
                coefficients = tuple(alpha for _ in batch.proposals)
                with backend.trial(batch, coefficients):
                    reading = backend.event(request)
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

            before = backend.event(request)
            coefficients = tuple(search.alpha for _ in batch.proposals)
            with backend.trial(batch, coefficients) as trial:
                after = backend.event(request)
                if tuple(trial.applied_coefficients) != coefficients:
                    raise MethodContractError("scalar applied alpha changed")
                if not after.is_hit(self.config.event_tolerance):
                    raise MethodContractError("scalar selected alpha did not reproduce first hit")
                trial.commit()
            self.omega.append_terminal(edit_id, backend.terminal_net_energy(entry))
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
            return ArmRunResult(
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
        except BaseException:
            backend.restore(entry)
            raise

    def _run_adaptive(
        self,
        arm: Arm,
        edit_id: str,
        request: Any,
        backend: MethodBackend,
    ) -> ArmRunResult:
        entry = backend.checkpoint()
        frozen_omega = self.omega.frozen_for_edit(edit_id)
        target = _DirectZOnce(backend, request)
        target_value = target.get()
        radius = self.config.h0
        steps: list[StepRecord] = []
        layers = tuple(backend.layers)
        max_positions = self.config.smax_cycles * (
            len(layers) if arm is Arm.ORDERED_ADAPTIVE else 1
        )
        position = 0
        rejections = 0
        static_entry: ProposalBatch | None = None
        static_share: tuple[float, ...] | None = None

        try:
            while position < max_positions:
                before = backend.event(request)
                if before.is_hit(self.config.event_tolerance):
                    self.omega.append_terminal(edit_id, backend.terminal_net_energy(entry))
                    return ArmRunResult(
                        arm=arm,
                        edit_id=edit_id,
                        status="event_hit",
                        direct_z_compute_count=target.compute_count,
                        terminal_state_id=backend.current_state_id(),
                        omega_appended=True,
                        steps=tuple(steps),
                    )

                layer: int | None = None
                if arm is Arm.FULL_ODE_EDIT:
                    batch = backend.build_synchronous(target_value)
                    if batch.semantics is not ProposalSemantics.CURRENT_SAME_SNAPSHOT:
                        raise MethodContractError("Full requires current same-snapshot proposals")
                elif arm is Arm.STATIC_SYNCHRONOUS:
                    if static_entry is None:
                        static_entry = backend.build_synchronous(target_value)
                        if static_entry.semantics is not ProposalSemantics.CURRENT_SAME_SNAPSHOT:
                            raise MethodContractError("Static entry is not same-snapshot")
                        batch = static_entry
                    else:
                        batch = backend.rebind_frozen(static_entry)
                        if batch.direction_ids != static_entry.direction_ids:
                            raise MethodContractError("Static direction changed after entry")
                else:
                    layer = layers[position % len(layers)]
                    batch = backend.build_coordinate(target_value, layer)
                    if (
                        batch.semantics is not ProposalSemantics.CURRENT_COORDINATE
                        or batch.layers != (layer,)
                    ):
                        raise MethodContractError("Ordered arm violated coordinate identity")
                _assert_batch_current(backend, batch)

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
                        omega={layer_id: frozen_omega[layer_id] for layer_id in batch.layers},
                        denominators={
                            layer_id: self.omega.denominators[layer_id]
                            for layer_id in batch.layers
                        },
                        hard_phi=before.hard_phi,
                        trust_radius=radius,
                        config=self.config,
                    )
                    if arm is Arm.STATIC_SYNCHRONOUS and solution.coefficient_norm > 0.0:
                        static_share = normalized_share(
                            solution.coefficients, self.config.progress_epsilon
                        )

                if solution.predicted_progress <= self.config.progress_epsilon:
                    if arm is Arm.ORDERED_ADAPTIVE:
                        position += 1
                        rejections = 0
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

                state_before_trial = backend.checkpoint()
                with backend.trial(batch, solution.coefficients) as trial:
                    after = backend.event(request)
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
                    if verdict.accepted:
                        trial.commit()
                if not verdict.accepted:
                    backend.assert_checkpoint(state_before_trial)
                steps.append(
                    StepRecord(
                        arm=arm,
                        position=position,
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
                    position += 1
                    if verdict.hit:
                        self.omega.append_terminal(
                            edit_id, backend.terminal_net_energy(entry)
                        )
                        return ArmRunResult(
                            arm=arm,
                            edit_id=edit_id,
                            status="event_hit",
                            direct_z_compute_count=target.compute_count,
                            terminal_state_id=backend.current_state_id(),
                            omega_appended=True,
                            steps=tuple(steps),
                        )
                else:
                    rejections += 1
                    if rejections > self.config.max_rejections_per_state:
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
                status="smax_event_fail",
                direct_z_compute_count=target.compute_count,
                terminal_state_id=backend.current_state_id(),
                omega_appended=False,
                steps=tuple(steps),
                failure_type="smax_event_fail",
            )
        except BaseException:
            backend.restore(entry)
            raise
