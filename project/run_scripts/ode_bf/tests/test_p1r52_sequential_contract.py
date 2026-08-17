from __future__ import annotations

import ast
import hashlib
import inspect
import threading
import types
import unittest

import numpy as np
import torch

from project.run_scripts.ode_bf.p1_backend import _validate_history_keys
from project.run_scripts.ode_bf.p1_state import P1HistoryRecord
from project.run_scripts.ode_bf.p1r29_sequential_preparation import SequentialArmState
from project.run_scripts.ode_bf.p1r43_full_strength_routing import solve_p1r43_full_strength_routing
from project.run_scripts.ode_bf.fixed_e8_soft_routing import FixedE8Arm
from project.run_scripts.ode_bf.functional import WaypointFactor
from project.run_scripts.ode_bf.p1r52_sequential_contract import (
    LifetimeAnchor,
    LifetimeAnchorLedger,
    SequentialHRouter,
    assemble_terminal_candidates,
    commit_sequential_batch,
    dry_plan,
    scoped_atomic_sequential_adapter,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    _b1_scientific_payload,
    run_p1r52_sequential,
    structural_h_off_control_receipts,
)
from project.run_scripts.ode_bf.p1r24_atomic_strength import p1r24_disable_historical
from project.run_scripts.ode_bf.routing import QuadraticBarrier, RoutingProblem
from project.run_scripts.ode_bf.scalable_batched_runtime import P1R23_LAYER_ORDER
from project.run_scripts.ode_bf.woodbury import ProjectorCertificate, solve_alpha_woodbury


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _records(version: int) -> tuple[P1HistoryRecord, ...]:
    return tuple(
        P1HistoryRecord(
            _sha(f"request-{version}-{index}"),
            version * 100 + index,
            _sha(f"collision-{version}-{index}"),
            _sha(f"target-{version}-{index}"),
            _sha(f"event-{version}"),
            version,
        )
        for index in range(10)
    )


def _anchors(version: int) -> tuple[LifetimeAnchor, ...]:
    return tuple(
        LifetimeAnchor.create(
            request_sha256=_sha(f"request-{version}-{index}"),
            case_id=version * 100 + index,
            collision_sha256=_sha(f"collision-{version}-{index}"),
            round_index=version,
            target_new_nll_by_context=[float(version + index)] * 6,
            margin_by_context=[1.0] * 6,
            context_provenance_sha256=_sha(f"context-{version}-{index}"),
            endpoint_weight_sha256=_sha(f"weight-{version}"),
        )
        for index in range(10)
    )


def _keys(version: int) -> dict[int, torch.Tensor]:
    return {
        layer: torch.full((3, 10), float(version + layer), dtype=torch.float32)
        for layer in P1R23_LAYER_ORDER
    }


class P1R52SequentialContractTests(unittest.TestCase):
    def test_b1_cross_process_identity_filter_is_exact_and_rejects_science_delta(self) -> None:
        reference = {
            "field_sha256": "a" * 64,
            "target_objective": {
                "model_state_sha256": "b" * 64,
                "identity_sha256": "c" * 64,
                "loss": 0.5,
                "target_gradient_sha256": "d" * 64,
            },
            "routing": {"velocity": [0.1, 0.2, 0.3, 0.2, 0.2]},
        }
        cross_process = {
            **reference,
            "field_sha256": "e" * 64,
            "target_objective": {
                **reference["target_objective"],
                "model_state_sha256": "f" * 64,
                "identity_sha256": "0" * 64,
            },
        }
        self.assertEqual(
            _b1_scientific_payload(reference),
            _b1_scientific_payload(cross_process),
        )
        changed = {
            **cross_process,
            "routing": {"velocity": [0.1, 0.2, 0.3, 0.1, 0.3]},
        }
        self.assertNotEqual(
            _b1_scientific_payload(reference),
            _b1_scientific_payload(changed),
        )

    def test_terminal_candidate_assembly_binds_authoritative_row_block(self) -> None:
        entry = torch.zeros((3, 4), dtype=torch.bfloat16)
        factor = WaypointFactor(
            "module.weight",
            4,
            0,
            0,
            0,
            1.0,
            torch.ones((3, 10), dtype=torch.float32),
            torch.ones((4, 10), dtype=torch.float32),
        )
        candidates, receipt = assemble_terminal_candidates(
            {"module.weight": entry},
            {"module.weight": (factor,)},
        )
        self.assertEqual(candidates["module.weight"].dtype, torch.bfloat16)
        self.assertEqual(receipt["weights"]["module.weight"]["factor_count"], 1)

    def test_atomic_call_uses_only_the_p1r52_method_flag(self) -> None:
        tree = ast.parse(inspect.getsource(run_p1r52_sequential))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_ode_arm"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {item.arg: item.value for item in calls[0].keywords}
        self.assertIsInstance(keywords["p1r52"], ast.Constant)
        self.assertIs(keywords["p1r52"].value, True)
        self.assertNotIn("p1r51", keywords)

    def test_history_width_zero_through_ninety(self) -> None:
        for width in range(0, 91, 10):
            values = {
                layer: torch.zeros((3, width), dtype=torch.float32)
                for layer in P1R23_LAYER_ORDER
            }
            observed = _validate_history_keys(
                values,
                P1R23_LAYER_ORDER,
                maximum_history_columns=90,
            )
            self.assertEqual({value.shape[1] for value in observed.values()}, {width})
        with self.assertRaises(Exception):
            _validate_history_keys(
                {
                    layer: torch.zeros((3, 100), dtype=torch.float32)
                    for layer in P1R23_LAYER_ORDER
                },
                P1R23_LAYER_ORDER,
                maximum_history_columns=90,
            )
        with self.assertRaises(Exception):
            _validate_history_keys(
                {
                    layer: torch.zeros((3, 40), dtype=torch.float32)
                    for layer in P1R23_LAYER_ORDER
                },
                P1R23_LAYER_ORDER,
            )

    def test_h_empty_is_exact_atomic_and_active_preserves_strength(self) -> None:
        dimension = len(P1R23_LAYER_ORDER)
        identity = np.eye(dimension, dtype=np.float64)
        historical = QuadraticBarrier(
            "historical",
            0.0,
            np.zeros(dimension),
            np.diag([20.0, 1.0, 1.0, 1.0, 1.0]),
            100.0,
            "layer-local-diagonal",
        )
        pretrained = QuadraticBarrier(
            "pretrained", 0.0, np.zeros(dimension), identity, 100.0, "layer-local-diagonal"
        )
        problem = RoutingProblem(
            np.ones(dimension), identity, identity, 100.0, np.full(dimension, 100.0),
            1.0, 1.0, historical, pretrained,
        )
        atomic = solve_p1r43_full_strength_routing(problem, arm=FixedE8Arm.SOFT, alpha_req=1.0)
        empty_router = SequentialHRouter(0, solve_p1r43_full_strength_routing)
        empty = empty_router.solve(problem, arm=FixedE8Arm.SOFT, alpha_req=1.0)
        self.assertEqual(empty.identity_sha256, atomic.identity_sha256)
        self.assertEqual(empty_router.receipts[0].status, "H_EMPTY_EXACT_ATOMIC_EQUIVALENCE")
        active_router = SequentialHRouter(40, solve_p1r43_full_strength_routing)
        active = active_router.solve(problem, arm=FixedE8Arm.SOFT, alpha_req=1.0)
        self.assertLess(abs(sum(active.velocity) - 1.0), 1.0e-8)
        self.assertLessEqual(
            problem.historical.value(np.asarray(active.velocity)),
            problem.historical.value(np.asarray(atomic.velocity)) + 1.0e-8,
        )
        self.assertGreater(
            np.linalg.norm(np.asarray(active.velocity) - np.asarray(atomic.velocity)),
            1.0e-10,
        )
        self.assertEqual(active_router.receipts[0].status, "H_ACTIVE_CERTIFIED")

    def test_b2_alpha_cache_on_and_structural_h_decision_off(self) -> None:
        state, anchors = self._state_with_rounds(1)
        self.assertEqual(state.history_entry_count(), 10)
        self.assertEqual(len(anchors.anchors), 10)
        dimension = len(P1R23_LAYER_ORDER)
        identity = np.eye(dimension, dtype=np.float64)
        problem = RoutingProblem(
            np.ones(dimension),
            identity,
            identity,
            100.0,
            np.full(dimension, 100.0),
            1.0,
            1.0,
            QuadraticBarrier(
                "historical",
                0.0,
                np.zeros(dimension),
                np.diag([20.0, 1.0, 1.0, 1.0, 1.0]),
                100.0,
                "layer-local-diagonal",
            ),
            QuadraticBarrier(
                "pretrained",
                0.0,
                np.zeros(dimension),
                identity,
                100.0,
                "layer-local-diagonal",
            ),
        )
        experiment = types.SimpleNamespace(
            build_scalable_dynamic_field=lambda *args, **kwargs: None,
            p1r24_disable_historical=p1r24_disable_historical,
            solve_p1r43_full_strength_routing=solve_p1r43_full_strength_routing,
        )
        router = SequentialHRouter(0, solve_p1r43_full_strength_routing)
        with scoped_atomic_sequential_adapter(
            experiment,
            state,
            router,
            structural_h_decision_enabled=False,
        ):
            routing_problem = experiment.p1r24_disable_historical(problem)
            selected = experiment.solve_p1r43_full_strength_routing(
                routing_problem,
                arm=FixedE8Arm.SOFT,
                alpha_req=1.0,
            )
        expected = solve_p1r43_full_strength_routing(
            p1r24_disable_historical(problem),
            arm=FixedE8Arm.SOFT,
            alpha_req=1.0,
        )
        self.assertEqual(selected.identity_sha256, expected.identity_sha256)
        self.assertEqual(float(np.trace(routing_problem.historical.gram)), 0.0)
        receipt = structural_h_off_control_receipts(
            router.receipts,
            alpha_solve_history_width=state.history_entry_count(),
            anchor_observation_width=len(anchors.anchors),
        )[0]
        self.assertEqual(receipt["alpha_solve_history_width"], 10)
        self.assertEqual(receipt["alpha_solve_cache_consume_count"], 10)
        self.assertEqual(receipt["alpha_solve_cache_append_count"], 10)
        self.assertEqual(receipt["structural_h_decision_history_width"], 0)
        self.assertEqual(receipt["structural_h_decision_influence_count"], 0)
        self.assertEqual(receipt["added_model_forward_count"], 0)
        self.assertEqual(receipt["added_backward_count"], 0)
        self.assertEqual(receipt["added_materialization_count"], 0)

    def test_b90_history_woodbury_is_finite_and_changes_solve(self) -> None:
        generator = torch.Generator().manual_seed(52)
        dimension = 128
        projector = torch.eye(dimension, dtype=torch.float32)
        current = torch.randn((dimension, 10), generator=generator) * 0.01
        history = torch.randn((dimension, 90), generator=generator) * 0.01
        certificate = ProjectorCertificate(
            _sha("projector"), 0.0, 0.0, "full-matrix", 1.0e-10
        )
        with_history = solve_alpha_woodbury(
            projector,
            current,
            history_keys=history,
            regularization=1.0,
            projector_certificate=certificate,
            residual_tolerance=1.0e-8,
        )
        empty = solve_alpha_woodbury(
            projector,
            current,
            history_keys=torch.empty((dimension, 0), dtype=torch.float32),
            regularization=1.0,
            projector_certificate=certificate,
            residual_tolerance=1.0e-8,
        )
        self.assertTrue(with_history.certificate.passed)
        self.assertTrue(torch.isfinite(with_history.q).all())
        self.assertEqual(with_history.certificate.small_dimension, 100)
        self.assertGreater(torch.linalg.norm(with_history.q - empty.q).item(), 0.0)

    def _state_with_rounds(self, count: int) -> tuple[SequentialArmState, LifetimeAnchorLedger]:
        state = SequentialArmState("test", P1R23_LAYER_ORDER)
        anchors = LifetimeAnchorLedger.empty()
        for version in range(1, count + 1):
            prospective = state.ledger.prospective(
                transaction_id=f"prefill-{version}",
                expected_version=version - 1,
                records=_records(version),
                solve_keys_by_layer=_keys(version),
                risk_keys_by_layer=_keys(version),
            )
            state.ledger.finalize(
                prospective,
                post_commit_verified=True,
                load_increment_by_layer={layer: 0.0 for layer in P1R23_LAYER_ORDER},
            )
            anchors.append_once(f"prefill-{version}", _anchors(version))
        return state, anchors

    def _fault_rollback(self, prior_rounds: int, fault: str) -> None:
        state, anchors = self._state_with_rounds(prior_rounds)
        version = prior_rounds + 1
        prospective = state.ledger.prospective(
            transaction_id=f"round-{version}",
            expected_version=prior_rounds,
            records=_records(version),
            solve_keys_by_layer=_keys(version),
            risk_keys_by_layer=_keys(version),
        )
        parameter = torch.nn.Parameter(torch.zeros((3, 3), dtype=torch.bfloat16), requires_grad=False)
        parameters = {"weight": parameter}
        before_weight = parameter.detach().clone()
        before_history = state.ledger.snapshot().digest
        before_anchors = anchors.identity()
        with self.assertRaises(RuntimeError):
            commit_sequential_batch(
                parameters,
                {"weight": torch.ones((3, 3), dtype=torch.bfloat16)},
                mutation_lock=threading.RLock(),
                transaction_id=f"round-{version}",
                history_state=state,
                prospective=prospective,
                load_increment_by_layer={layer: 1.0 for layer in P1R23_LAYER_ORDER},
                anchor_ledger=anchors,
                anchor_factory=lambda: (_anchors(version), {"round": version}),
                fault_phase=fault,
            )
        self.assertTrue(torch.equal(parameter, before_weight))
        self.assertEqual(state.ledger.snapshot().digest, before_history)
        self.assertEqual(anchors.identity(), before_anchors)

    def test_b5_and_b10_joint_transaction_fault_rollback(self) -> None:
        self._fault_rollback(4, "after_anchor")
        self._fault_rollback(9, "after_history")

    def test_history_and_anchor_factories_observe_postcommit_weight(self) -> None:
        state, anchors = self._state_with_rounds(0)
        parameter = torch.nn.Parameter(
            torch.zeros((3, 3), dtype=torch.bfloat16), requires_grad=False
        )

        def history_factory():
            self.assertTrue(torch.equal(parameter, torch.ones_like(parameter)))
            prospective = state.ledger.prospective(
                transaction_id="round-1",
                expected_version=0,
                records=_records(1),
                solve_keys_by_layer=_keys(1),
                risk_keys_by_layer=_keys(1),
            )
            return prospective, {"post_commit_capture": True}

        def anchor_factory():
            self.assertTrue(torch.equal(parameter, torch.ones_like(parameter)))
            return _anchors(1), {"post_commit_capture": True}

        receipt = commit_sequential_batch(
            {"weight": parameter},
            {"weight": torch.ones_like(parameter)},
            mutation_lock=threading.RLock(),
            transaction_id="round-1",
            history_state=state,
            prospective=None,
            load_increment_by_layer={layer: 1.0 for layer in P1R23_LAYER_ORDER},
            anchor_ledger=anchors,
            anchor_factory=anchor_factory,
            history_factory=history_factory,
        )
        self.assertEqual(state.history_entry_count(), 10)
        self.assertEqual(len(anchors.anchors), 10)
        self.assertTrue(receipt["history_capture"]["post_commit_capture"])
        self.assertEqual(receipt["idempotent_history_replay_append_count"], 0)
        self.assertEqual(receipt["idempotent_anchor_replay_append_count"], 0)

    def test_dry_plan_has_exact_widths_and_zero_extra_accuracy_compute(self) -> None:
        plan = dry_plan()
        self.assertEqual(plan["history_width_at_entry"], list(range(0, 100, 10)))
        self.assertEqual(plan["sequential_accuracy_added_forward_backward_generation"], [0, 0, 0])
        self.assertEqual(plan["batch_entry_pre_evaluations"], 10)
        self.assertEqual(plan["batch_entry_pre_evaluator_added_forward_count"], 100)
        self.assertEqual(plan["batch_entry_pre_evaluator_added_backward_generation"], [0, 0])
        self.assertEqual(len(plan["pre_post_final_machine_tables"]), 4)


if __name__ == "__main__":
    unittest.main()
