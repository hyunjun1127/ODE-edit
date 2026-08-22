from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.contracts import canonical_hash
from project.run_scripts.ode_bf.p4_cache_transaction import P4CacheTransaction
from project.run_scripts.ode_bf.p4_fixed_target_solver import (
    P4SolverEvaluation,
    run_fixed_m_target_adam,
)
from project.run_scripts.ode_bf.p4_panel_analysis import (
    factual_barrier_delta,
    validate_paired_panel,
)
from project.run_scripts.ode_bf.p4_semantic_barrier import (
    P4TargetArm,
    semantic_gradient_coefficients,
    semantic_gradient_comparison,
    smooth_semantic_logodds_potential,
)
from project.run_scripts.ode_bf.p4_waypoint_writer import (
    build_p4_write_waypoint,
    run_p4_official_writer_step,
    verify_current_state_refresh,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
ROOT = Path(__file__).resolve().parents[4]


class SemanticPotentialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.new = torch.tensor(
            [[-2.0, -1.0, -0.5, -3.0, -4.0, -2.5], [-1.5, -2.5, -3.5, -0.7, -1.2, -2.2]],
            dtype=torch.float32,
            requires_grad=True,
        )
        self.old = torch.tensor(
            [[-1.0, -2.0, -0.2, -4.0, -3.0, -2.0], [-2.0, -2.0, -4.0, -0.5, -2.2, -1.2]],
            dtype=torch.float32,
            requires_grad=True,
        )

    def test_analytic_reference_equality_and_signs(self) -> None:
        result = smooth_semantic_logodds_potential(
            self.new, self.old, arm=P4TargetArm.POSITIVE_NEGATIVE
        )
        reference = torch.mean(
            torch.mean(
                -self.new
                + torch.logaddexp(torch.zeros_like(self.new), self.old - self.new),
                dim=1,
            )
        )
        self.assertTrue(torch.allclose(result.objective, reference, atol=1e-6, rtol=0.0))
        new_coefficient, old_coefficient = semantic_gradient_coefficients(
            self.new, self.old, arm=P4TargetArm.POSITIVE_NEGATIVE
        )
        new_gradient, old_gradient = torch.autograd.grad(
            result.objective, (self.new, self.old)
        )
        scale = float(self.new.numel())
        self.assertTrue(torch.allclose(new_gradient * scale, new_coefficient, atol=1e-6, rtol=0.0))
        self.assertTrue(torch.allclose(old_gradient * scale, old_coefficient, atol=1e-6, rtol=0.0))
        self.assertTrue(bool(torch.all(new_coefficient < 0.0)))
        self.assertTrue(bool(torch.all(old_coefficient > 0.0)))

    def test_positive_arm_has_no_old_decision_gradient(self) -> None:
        result = smooth_semantic_logodds_potential(
            self.new, self.old, arm=P4TargetArm.POSITIVE
        )
        new_gradient = torch.autograd.grad(result.objective, self.new)[0]
        new_coefficient, old_coefficient = semantic_gradient_coefficients(
            self.new, self.old, arm=P4TargetArm.POSITIVE
        )
        self.assertTrue(torch.allclose(new_gradient * self.new.numel(), new_coefficient))
        self.assertTrue(torch.count_nonzero(old_coefficient) == 0)
        self.assertEqual(result.receipt["margin_threshold_access_count"], 0)

    def test_full_fp32_fail_close(self) -> None:
        with self.assertRaises(ODEBFContractError):
            smooth_semantic_logodds_potential(
                self.new.double(), self.old.double(), arm=P4TargetArm.POSITIVE
            )

    def test_gradient_observation(self) -> None:
        receipt = semantic_gradient_comparison(
            torch.eye(2, dtype=torch.float32),
            torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
        )
        self.assertEqual(receipt["decision_influence_count"], 0)
        self.assertAlmostEqual(receipt["gradient_cosine_by_request"][0], 1.0)


class FixedSolverTests(unittest.TestCase):
    def test_fixed_five_final_only_matches_adam(self) -> None:
        initial = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
        origin = initial.clone()
        goal = torch.tensor([[0.0, 1.0], [2.0, -1.0]], dtype=torch.float32)

        def evaluate(target: torch.Tensor, iteration: int) -> P4SolverEvaluation:
            delta = target - goal
            return P4SolverEvaluation(
                torch.sum(torch.square(delta), dim=0),
                2.0 * delta,
                {"iteration": iteration, "heldout_access_count": 0},
            )

        observed = run_fixed_m_target_adam(
            initial,
            origin,
            evaluate,
            learning_rate=0.1,
            clamp_factor=100.0,
            outer_step_index=0,
            arm="Z+",
            paired_input_identity=SHA_A,
        )
        expected = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([expected], lr=0.1)
        for _ in range(5):
            optimizer.zero_grad()
            torch.sum(torch.square(expected - goal)).backward()
            optimizer.step()
        self.assertTrue(torch.allclose(observed.final_target, expected.detach(), atol=1e-7, rtol=0.0))
        self.assertEqual(observed.receipt["executed_inner_iterations"], 5)
        self.assertEqual(observed.receipt["selected_iterate_ordinal"], 5)
        self.assertEqual(observed.receipt["best_iterate_decision_influence_count"], 0)
        self.assertEqual(observed.receipt["moment_reset_count"], 1)

    def test_nonfinite_is_technical_failure(self) -> None:
        initial = torch.ones((2, 1), dtype=torch.float32)

        def evaluate(_target: torch.Tensor, _iteration: int) -> P4SolverEvaluation:
            return P4SolverEvaluation(
                torch.tensor([float("nan")], dtype=torch.float32),
                torch.ones((2, 1), dtype=torch.float32),
                {},
            )

        with self.assertRaises(ODEBFContractError):
            run_fixed_m_target_adam(
                initial,
                initial,
                evaluate,
                learning_rate=0.1,
                clamp_factor=1.0,
                outer_step_index=0,
                arm="Z+",
                paired_input_identity=SHA_A,
            )


class WaypointAndCacheTests(unittest.TestCase):
    def test_all_k8_waypoints(self) -> None:
        terminal = torch.zeros((2, 1), dtype=torch.float32)
        target = torch.full((2, 1), 8.0, dtype=torch.float32)
        for index in range(8):
            waypoint, receipt = build_p4_write_waypoint(
                terminal, target, outer_step_index=index
            )
            self.assertTrue(
                torch.equal(waypoint, target / float(8 - index))
            )
            self.assertEqual(receipt["lambda_k"], 1.0 / (8 - index))

    def test_current_state_refresh_call_counts(self) -> None:
        rows = [
            {
                "outer_step_index": index,
                "target_gradient_refresh_count": 1,
                "current_terminal_residual_refresh_count": 1,
                "current_key_refresh_count": 1,
                "alpha_solve_refresh_count": 1,
                "official_writer_apply_count": 1,
                "native_compute_z_call_count": 0,
            }
            for index in range(8)
        ]
        receipt = verify_current_state_refresh(rows, arm="A±")
        self.assertEqual(receipt["official_writer_apply_count"], 8)
        self.assertEqual(receipt["native_compute_z_call_count"], 0)
        rows[4] = {**rows[4], "current_key_refresh_count": 0}
        with self.assertRaises(ODEBFContractError):
            verify_current_state_refresh(rows, arm="A±")

    def test_official_writer_adapter_compute_z_counts(self) -> None:
        calls: list[dict[str, object]] = []

        def apply(*_args: object, **kwargs: object):
            template = kwargs["cache_template"]
            if template is not None:
                self.assertTrue(Path(str(template).format(8, 0.75, 1)).is_file())
            calls.append(dict(kwargs))
            return {"call": len(calls)}, {}

        requests = [{"case_id": 1, "request_sha256": SHA_A}]
        hparams = SimpleNamespace(layers=[4, 5, 6, 7, 8], clamp_norm_factor=0.75)
        with tempfile.TemporaryDirectory() as temporary:
            ours, _ = run_p4_official_writer_step(
                object(),
                object(),
                requests,
                hparams,
                arm="A±",
                outer_step_index=0,
                touched={},
                private_root=Path(temporary),
                write_waypoint=torch.ones((2, 1), dtype=torch.float32),
                reset_cache=True,
                batch_entry_cache_width=0,
                apply_callable=apply,
            )
            native, _ = run_p4_official_writer_step(
                object(),
                object(),
                requests,
                hparams,
                arm="Native",
                outer_step_index=0,
                touched={},
                private_root=Path(temporary),
                write_waypoint=None,
                reset_cache=True,
                batch_entry_cache_width=0,
                apply_callable=apply,
            )
        self.assertEqual(ours["native_compute_z_expected_count"], 0)
        self.assertEqual(native["native_compute_z_expected_count"], 1)
        self.assertEqual(calls[0]["expected_native_compute_z_call_count"], 0)
        self.assertEqual(calls[1]["expected_native_compute_z_call_count"], 1)

    def test_cache_snapshot_append_once_and_rollback(self) -> None:
        transaction = P4CacheTransaction(SHA_A, 30, 10)
        for index in range(8):
            transaction.observe_outer(
                outer_step_index=index,
                consumed_snapshot_sha256=SHA_A,
                consumed_snapshot_width=30,
                current_batch_append_count=0,
            )
        receipt = transaction.commit(exit_sha256=SHA_B, exit_width=40, append_count=1)
        self.assertEqual(receipt["outer_snapshot_reuse_count"], 8)
        self.assertEqual(receipt["post_k8_append_count"], 1)

        aborted = P4CacheTransaction(SHA_A, 30, 10)
        aborted.observe_outer(
            outer_step_index=0,
            consumed_snapshot_sha256=SHA_A,
            consumed_snapshot_width=30,
            current_batch_append_count=0,
        )
        failure = aborted.abort(rollback_count=1)
        self.assertEqual(failure["post_k8_append_count"], 0)
        self.assertEqual(failure["exit_sha256"], SHA_A)

    def test_cache_rejects_current_batch_append(self) -> None:
        transaction = P4CacheTransaction(SHA_A, 0, 10)
        with self.assertRaises(ODEBFContractError):
            transaction.observe_outer(
                outer_step_index=0,
                consumed_snapshot_sha256=SHA_A,
                consumed_snapshot_width=0,
                current_batch_append_count=1,
            )


class PanelTests(unittest.TestCase):
    def test_numerical_lock_is_rooted_and_exact(self) -> None:
        path = (
            ROOT
            / "project/run_scripts/ode_bf/locks/numerical_lock_s05_p4_target_side_semantic_barrier.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        observed = value.pop("root_digest")
        self.assertEqual(observed, canonical_hash(value))
        self.assertEqual(value["target"]["inner_iterations"], 5)
        self.assertEqual(value["outer"]["K"], 8)
        self.assertEqual(value["cache_transaction"]["append_after_successful_k8_commit"], 1)

    def test_paired_panel_and_factual_delta(self) -> None:
        rows = []
        for arm in ("Z+", "Z±", "Native-Z"):
            rows.append(
                {
                    "arm": arm,
                    "model_alias": "llama3-8b-inst",
                    "model_revision": "revision",
                    "sample_order_sha256": SHA_A,
                    "evaluator_identity_sha256": SHA_B,
                    "w0_sha256": "c" * 64,
                    "context_sha256": "d" * 64,
                }
            )
        receipt = validate_paired_panel(rows, stage="ZA")
        self.assertEqual(receipt["arms"], ["Z+", "Z±", "Native-Z"])
        delta = factual_barrier_delta(
            {"new_nll": 2.0, "margin": 0.1},
            {"new_nll": 1.8, "margin": 0.4},
            stage="ZA",
        )
        self.assertAlmostEqual(delta["delta_positive_negative_minus_positive"]["new_nll"], -0.2)
        self.assertFalse(delta["automatic_promotion"])


if __name__ == "__main__":
    unittest.main()
