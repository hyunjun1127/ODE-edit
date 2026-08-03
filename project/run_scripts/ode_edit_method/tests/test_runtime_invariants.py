from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import unittest

from project.run_scripts.ode_edit_method.contracts import (
    Arm,
    ControllerConfig,
    EventReading,
    LayerProposal,
    ProposalBatch,
    ProposalSemantics,
    canonical_hash,
)
from project.run_scripts.ode_edit_method.controller import OmegaLedger
from project.run_scripts.ode_edit_method.runtime import FiveArmRunner


def _test_config() -> ControllerConfig:
    """CPU fixture constants only; not a Session 02 numerical lock."""

    return ControllerConfig(
        tau=0.1,
        h0=2.0,
        kappa=1.0,
        beta=0.8,
        eta_reject=0.1,
        eta_expand=0.75,
        gamma_down=0.5,
        gamma_up=1.5,
        smax_cycles=6,
        max_rejections_per_state=2,
        slope_epsilon=1e-12,
        progress_epsilon=1e-12,
        qp_equality_tolerance=1e-9,
        event_tolerance=1e-9,
        trust_denominator_epsilon=1e-12,
        scalar_grid=(0.0, 0.5, 1.0),
        scalar_bisection_tolerance=1e-6,
        scalar_bisection_iterations=8,
        scalar_nonmonotonic_tolerance=1e-9,
    )


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    weights: tuple[float, float]
    state_id: str


class _Trial(AbstractContextManager["_Trial"]):
    def __init__(
        self,
        backend: "_ToyBackend",
        batch: ProposalBatch,
        coefficients: tuple[float, ...],
    ) -> None:
        self.backend = backend
        self.batch = batch
        self.applied_coefficients = coefficients
        self.backup = tuple(backend.weights)
        self.committed = False

    def __enter__(self) -> "_Trial":
        self.backend.trial_calls += 1
        for layer, coefficient in zip(
            self.batch.layers, self.applied_coefficients, strict=True
        ):
            self.backend.weights[layer] += coefficient
        return self

    def commit(self) -> None:
        self.committed = True
        self.backend.write_calls += 1

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None or not self.committed:
            self.backend.weights[:] = self.backup
        return False


class _ToyBackend:
    layers = (0, 1)

    def __init__(self) -> None:
        self.weights = [0.0, 0.0]
        self.direct_z_calls = 0
        self.event_calls = 0
        self.field_calls = 0
        self.trial_calls = 0
        self.write_calls = 0

    def current_state_id(self) -> str:
        return canonical_hash(self.weights)

    def checkpoint(self) -> _Checkpoint:
        return _Checkpoint(tuple(self.weights), self.current_state_id())

    def restore(self, checkpoint: _Checkpoint) -> None:
        self.weights[:] = checkpoint.weights

    def assert_checkpoint(self, checkpoint: _Checkpoint) -> None:
        if tuple(self.weights) != checkpoint.weights:
            raise AssertionError("toy checkpoint differs")

    def compute_direct_z(self, request: object) -> object:
        self.direct_z_calls += 1
        return ("frozen-z", request)

    def event(self, request: object) -> EventReading:
        del request
        self.event_calls += 1
        hard = 1.0 - sum(self.weights)
        return EventReading(
            hard_phi=hard,
            smooth_phi=hard,
            context_margins=(-hard,),
            nfe=1,
        )

    def _batch(self, semantics: ProposalSemantics) -> ProposalBatch:
        self.field_calls += 1
        state = self.current_state_id()
        return ProposalBatch(
            snapshot_id=state,
            proposals=tuple(
                LayerProposal(
                    layer=layer,
                    state_id=state,
                    direction_id=f"layer-{layer}-field-{self.field_calls}",
                )
                for layer in self.layers
            ),
            slopes=(1.0, 1.0),
            semantics=semantics,
        )

    def build_synchronous(self, frozen_target: object) -> ProposalBatch:
        del frozen_target
        return self._batch(ProposalSemantics.CURRENT_SAME_SNAPSHOT)

    def build_coordinate(self, frozen_target: object, layer: int) -> ProposalBatch:
        del frozen_target
        full = self._batch(ProposalSemantics.CURRENT_SAME_SNAPSHOT)
        proposal = full.proposals[layer]
        return ProposalBatch(
            snapshot_id=full.snapshot_id,
            proposals=(proposal,),
            slopes=(1.0,),
            semantics=ProposalSemantics.CURRENT_COORDINATE,
        )

    def build_native_terminal(self, frozen_target: object) -> ProposalBatch:
        del frozen_target
        return self._batch(ProposalSemantics.NATIVE_ORDERED_TERMINAL)

    def rebind_frozen(self, entry_batch: ProposalBatch) -> ProposalBatch:
        return entry_batch.rebind_frozen(self.current_state_id())

    def trial(self, batch: ProposalBatch, coefficients) -> _Trial:
        return _Trial(self, batch, tuple(coefficients))

    def terminal_net_energy(self, entry_checkpoint: _Checkpoint) -> dict[int, float]:
        return {
            layer: (self.weights[layer] - entry_checkpoint.weights[layer]) ** 2
            for layer in self.layers
        }


class RuntimeInvariantTests(unittest.TestCase):
    def test_full_first_hit_stops_all_later_controller_work(self) -> None:
        backend = _ToyBackend()
        omega = OmegaLedger({0: 1.0, 1: 1.0})
        result = FiveArmRunner(_test_config(), omega).run(
            Arm.FULL_ODE_EDIT,
            edit_id="edit-1",
            request={"rewrite": "only"},
            backend=backend,
        )
        self.assertEqual(result.status, "event_hit")
        self.assertEqual(result.direct_z_compute_count, 1)
        self.assertEqual(backend.direct_z_calls, 1)
        self.assertEqual(backend.field_calls, 1)
        self.assertEqual(backend.trial_calls, 1)
        self.assertEqual(backend.write_calls, 1)
        self.assertEqual(backend.event_calls, 2)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(len(omega.receipts), 1)


if __name__ == "__main__":
    unittest.main()
