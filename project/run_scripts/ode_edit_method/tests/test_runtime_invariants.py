from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import unittest

from project.run_scripts.ode_edit_method.contracts import (
    Arm,
    ControllerConfig,
    EventReading,
    LayerProposal,
    MethodContractError,
    ProposalBatch,
    ProposalSemantics,
    canonical_hash,
)
from project.run_scripts.ode_edit_method.controller import OmegaLedger
from project.run_scripts.ode_edit_method.events import ControllerRequest
from project.run_scripts.ode_edit_method.instrumentation import EditInstrumentation
from project.run_scripts.ode_edit_method.runtime import FiveArmRunner


def _test_config() -> ControllerConfig:
    """CPU fixture constants only; not a Session 02 numerical lock."""

    return ControllerConfig(
        tau=0.1,
        h0=2.0,
        h_max=2.0,
        kappa=1.0,
        beta=0.8,
        eta_reject=0.1,
        eta_expand=0.75,
        gamma_down=0.5,
        gamma_up=1.5,
        s_max=6,
        max_rejections_per_state=2,
        slope_epsilon=1e-12,
        progress_epsilon=1e-12,
        qp_equality_tolerance=1e-9,
        qp_trust_tolerance=1e-9,
        qp_bisection_iterations=128,
        event_tolerance=1e-9,
        hard_worsening_tolerance=1e-9,
        trust_denominator_epsilon=1e-12,
        load_denominator_epsilon=1e-12,
        scalar_grid=(0.0, 0.5, 1.0),
        scalar_bisection_tolerance=1e-6,
        scalar_bisection_iterations=8,
        scalar_nonmonotonic_tolerance=1e-9,
    )


def _request() -> ControllerRequest:
    return ControllerRequest(
        case_id="toy-case",
        prompt="{} is",
        subject="subject",
        target_new="new object",
        target_old="old object",
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

    def __enter__(self) -> "_Trial":
        self.backend.trial_calls += 1
        self.backend.virtual_delta = {
            layer: coefficient
            for layer, coefficient in zip(
                self.batch.layers, self.applied_coefficients, strict=True
            )
        }
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.backend.virtual_delta = {}
        self.backend.weights[:] = self.backup
        return False


class _ToyBackend:
    layers = (0, 1)

    def __init__(self, *, goal: float = 1.0, reject_first_trial: bool = False) -> None:
        self.weights = [0.0, 0.0]
        self.goal = goal
        self.reject_first_trial = reject_first_trial
        self.direct_z_calls = 0
        self.event_calls = 0
        self.field_calls = 0
        self.trial_calls = 0
        self.write_calls = 0
        self.virtual_delta: dict[int, float] = {}

    def current_state_id(self) -> str:
        return canonical_hash(self.weights)

    def checkpoint(self) -> _Checkpoint:
        return _Checkpoint(tuple(self.weights), self.current_state_id())

    def restore(self, checkpoint: _Checkpoint) -> None:
        self.weights[:] = checkpoint.weights

    def assert_checkpoint(self, checkpoint: _Checkpoint) -> None:
        if tuple(self.weights) != checkpoint.weights:
            raise AssertionError("toy checkpoint differs")

    def compute_direct_z(self, request: ControllerRequest) -> object:
        self.direct_z_calls += 1
        return ("frozen-z", request)

    def event(self, request: ControllerRequest) -> EventReading:
        del request
        self.event_calls += 1
        trial_total = sum(self.virtual_delta.values())
        if self.reject_first_trial and trial_total > 0.0:
            self.reject_first_trial = False
            hard = self.goal + 1.0
        else:
            hard = self.goal - sum(self.weights) - trial_total
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

    def commit(self, batch: ProposalBatch, coefficients) -> tuple[float, ...]:
        applied = tuple(float(value) for value in coefficients)
        for layer, coefficient in zip(batch.layers, applied, strict=True):
            self.weights[layer] += coefficient
        self.write_calls += 1
        return applied

    def terminal_net_energy(self, entry_checkpoint: _Checkpoint) -> dict[int, float]:
        return {
            layer: (self.weights[layer] - entry_checkpoint.weights[layer]) ** 2
            for layer in self.layers
        }


class RuntimeInvariantTests(unittest.TestCase):
    def test_runner_rejects_untyped_evaluation_payload(self) -> None:
        with self.assertRaisesRegex(MethodContractError, "rewrite-only"):
            FiveArmRunner(
                _test_config(), OmegaLedger({0: 1.0, 1: 1.0})
            ).run(
                Arm.FULL_ODE_EDIT,
                edit_id="edit-firewall",
                request={"rewrite": "only", "paraphrase_prompts": ["forbidden"]},  # type: ignore[arg-type]
                backend=_ToyBackend(),
            )

    def test_full_first_hit_stops_all_later_controller_work(self) -> None:
        backend = _ToyBackend()
        omega = OmegaLedger({0: 1.0, 1: 1.0})
        result = FiveArmRunner(_test_config(), omega).run(
            Arm.FULL_ODE_EDIT,
            edit_id="edit-1",
            request=_request(),
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

    def test_reject_reuses_exact_field_and_only_retries_trial(self) -> None:
        backend = _ToyBackend(reject_first_trial=True)
        result = FiveArmRunner(
            _test_config(), OmegaLedger({0: 1.0, 1: 1.0})
        ).run(
            Arm.FULL_ODE_EDIT,
            edit_id="edit-retry",
            request=_request(),
            backend=backend,
        )
        self.assertEqual(result.status, "event_hit")
        self.assertEqual(backend.field_calls, 1)
        self.assertEqual(backend.trial_calls, 2)
        self.assertEqual(sum(not step.accepted for step in result.steps), 1)
        self.assertEqual(result.steps[0].snapshot_id, result.steps[1].snapshot_id)
        self.assertEqual(result.steps[0].direction_ids, result.steps[1].direction_ids)

    def test_static_freezes_entry_field_while_full_refreshes_current_state(self) -> None:
        static_backend = _ToyBackend(goal=3.0)
        static = FiveArmRunner(
            _test_config(), OmegaLedger({0: 1.0, 1: 1.0})
        ).run(
            Arm.STATIC_SYNCHRONOUS,
            edit_id="edit-static-identity",
            request=_request(),
            backend=static_backend,
        )
        self.assertEqual(static.status, "event_hit")
        self.assertEqual(static_backend.field_calls, 1)
        self.assertEqual(len(static.steps), 2)
        self.assertEqual(static.steps[0].direction_ids, static.steps[1].direction_ids)
        self.assertNotEqual(static.steps[0].snapshot_id, static.steps[1].snapshot_id)

        full_backend = _ToyBackend(goal=3.0)
        full = FiveArmRunner(
            _test_config(), OmegaLedger({0: 1.0, 1: 1.0})
        ).run(
            Arm.FULL_ODE_EDIT,
            edit_id="edit-full-refresh",
            request=_request(),
            backend=full_backend,
        )
        self.assertEqual(full.status, "event_hit")
        self.assertEqual(full_backend.field_calls, 2)
        self.assertEqual(len(full.steps), 2)
        self.assertNotEqual(full.steps[0].direction_ids, full.steps[1].direction_ids)
        self.assertNotEqual(full.steps[0].snapshot_id, full.steps[1].snapshot_id)

    def test_ordered_is_ascending_current_coordinate_and_revisitable(self) -> None:
        backend = _ToyBackend(goal=4.0)
        result = FiveArmRunner(
            _test_config(), OmegaLedger({0: 1.0, 1: 1.0})
        ).run(
            Arm.ORDERED_ADAPTIVE,
            edit_id="edit-ordered-identity",
            request=_request(),
            backend=backend,
        )
        self.assertEqual(result.status, "event_hit")
        self.assertEqual([step.layer for step in result.steps], [0, 1, 0])
        self.assertTrue(all(len(step.solver_coefficients) == 1 for step in result.steps))
        self.assertEqual(len({step.snapshot_id for step in result.steps}), 3)

    def test_native_and_scalar_share_frozen_terminal_direction(self) -> None:
        native_backend = _ToyBackend(goal=1.0)
        native = FiveArmRunner(
            _test_config(), OmegaLedger({0: 1.0, 1: 1.0})
        ).run(
            Arm.NATIVE_MEMIT,
            edit_id="edit-native",
            request=_request(),
            backend=native_backend,
        )
        self.assertEqual(native.status, "event_hit")
        self.assertEqual(native.steps[0].applied_coefficients, (1.0, 1.0))

        scalar_backend = _ToyBackend(goal=1.0)
        scalar = FiveArmRunner(
            _test_config(), OmegaLedger({0: 1.0, 1: 1.0})
        ).run(
            Arm.SCALAR_FIRST_HIT,
            edit_id="edit-scalar",
            request=_request(),
            backend=scalar_backend,
        )
        self.assertEqual(scalar.status, "event_hit")
        self.assertIsNotNone(scalar.scalar_alpha)
        assert scalar.scalar_alpha is not None
        self.assertGreaterEqual(scalar.scalar_alpha, 0.0)
        self.assertLessEqual(scalar.scalar_alpha, 1.0)
        self.assertEqual(scalar.steps[0].direction_ids, native.steps[0].direction_ids)
        self.assertEqual(scalar_backend.field_calls, 1)
        self.assertEqual(scalar_backend.write_calls, 1)

    def test_entry_first_hit_performs_no_field_trial_or_write(self) -> None:
        backend = _ToyBackend(goal=0.0)
        result = FiveArmRunner(
            _test_config(), OmegaLedger({0: 1.0, 1: 1.0})
        ).run(
            Arm.FULL_ODE_EDIT,
            edit_id="edit-entry-hit",
            request=_request(),
            backend=backend,
        )
        self.assertEqual(result.status, "event_hit")
        self.assertEqual(result.direct_z_compute_count, 0)
        self.assertEqual(backend.direct_z_calls, 0)
        self.assertEqual(backend.field_calls, 0)
        self.assertEqual(backend.trial_calls, 0)
        self.assertEqual(backend.write_calls, 0)

    def test_runtime_compute_accounting_and_reject_reuse(self) -> None:
        backend = _ToyBackend(reject_first_trial=True)
        metrics = EditInstrumentation("edit-accounting")
        result = FiveArmRunner(
            _test_config(), OmegaLedger({0: 1.0, 1: 1.0})
        ).run(
            Arm.FULL_ODE_EDIT,
            edit_id="edit-accounting",
            request=_request(),
            backend=backend,
            instrumentation=metrics,
        )
        self.assertEqual(result.status, "event_hit")
        observed = metrics.finalize().to_dict()
        counters = observed["counters"]
        self.assertEqual(counters["N_z"], 1)
        self.assertEqual(counters["N_model_fwd"], 3)
        self.assertEqual(counters["N_event_fwd"], 3)
        self.assertEqual(counters["N_field_state_fwd"], 0)
        self.assertEqual(counters["N_proposal_build"], 1)
        self.assertEqual(counters["N_bw"], 1)
        self.assertEqual(counters["K_acc"], 1)
        self.assertEqual(counters["N_trial"], 2)
        self.assertEqual(counters["N_reject"], 1)
        self.assertEqual(counters["N_write"], 1)
        self.assertTrue(observed["first_hit"])

    def test_failure_after_accepted_write_restores_entry_before_omega_append(self) -> None:
        class _TerminalFailureBackend(_ToyBackend):
            def terminal_net_energy(self, entry_checkpoint: _Checkpoint) -> dict[int, float]:
                del entry_checkpoint
                raise RuntimeError("injected terminal failure")

        backend = _TerminalFailureBackend()
        omega = OmegaLedger({0: 1.0, 1: 1.0})
        with self.assertRaisesRegex(RuntimeError, "injected terminal failure"):
            FiveArmRunner(_test_config(), omega).run(
                Arm.FULL_ODE_EDIT,
                edit_id="edit-terminal-failure",
                request=_request(),
                backend=backend,
            )
        self.assertEqual(backend.weights, [0.0, 0.0])
        self.assertEqual(omega.receipts, ())

    def test_one_refresh_and_ordered_caps_count_accepted_transitions(self) -> None:
        one_backend = _ToyBackend(goal=10.0)
        one = FiveArmRunner(
            _test_config(), OmegaLedger({0: 1.0, 1: 1.0})
        ).run(
            Arm.ONE_REFRESH,
            edit_id="edit-one-refresh",
            request=_request(),
            backend=one_backend,
        )
        self.assertEqual(one.status, "one_refresh_event_fail")
        self.assertEqual(sum(step.accepted for step in one.steps), 2)
        self.assertEqual(one_backend.field_calls, 2)
        self.assertEqual(one_backend.weights, [0.0, 0.0])

        ordered_backend = _ToyBackend(goal=10.0)
        ordered = FiveArmRunner(
            _test_config(), OmegaLedger({0: 1.0, 1: 1.0})
        ).run(
            Arm.ORDERED_ADAPTIVE,
            edit_id="edit-ordered-cap",
            request=_request(),
            backend=ordered_backend,
        )
        self.assertEqual(ordered.status, "resolution_cap_unresolved")
        self.assertEqual(sum(step.accepted for step in ordered.steps), 6)
        self.assertEqual(ordered_backend.trial_calls, 6)
        self.assertEqual(ordered_backend.weights, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
