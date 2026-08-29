from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

import torch

from easyeditor.editors.utils import _prepare_requests
from project.run_scripts.alphaedit_strength_neutral_barrier.contracts import (
    TokenizerContractBoundary,
    require_right_padding,
    target_token_ids,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.endpoint_adoption import (
    adopt_exact_endpoint,
    capture_exact_endpoint,
    fp32_comparison,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.geometry import (
    ActionableNormalBoundary,
    apply_velocity_,
    factorize_alphaedit_velocity,
    low_rank_pullback,
    project_one_sided_velocity,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.target_path import (
    build_target_event_batch,
    capture_reference,
    evaluate_values,
    sequence_nll_by_request,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.writer import (
    BarrierArm,
    BarrierWriterConfig,
    apply_strength_neutral_barrier_to_model,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    bos_token_id = 1
    unk_token_id = 99
    padding_side = "right"

    def __call__(self, text, add_special_tokens=True):
        ids = [2 + (ord(char) % 17) for char in text]
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(char) % 17) for char in text]


def _requests() -> list[dict]:
    return [
        {
            "case_id": 7,
            "prompt": "{} lives in",
            "subject": "Ada",
            "target_new": " XY",
        },
        {
            "case_id": 8,
            "prompt": "{} works at",
            "subject": "Lin",
            "target_new": " Z",
        },
    ]


def _factorization(*, history: bool = True):
    torch.manual_seed(3)
    output, width, requests = 5, 7, 3
    projector = torch.diag(torch.tensor([1, 1, 1, 1, 1, 0, 0], dtype=torch.float32))
    keys = torch.randn(width, requests)
    residual = torch.randn(output, requests)
    basis = torch.randn(width, 2)
    cache = basis @ basis.T if history else torch.zeros(width, width)
    return factorize_alphaedit_velocity(
        residual=residual,
        keys=keys,
        projector=projector,
        history_cache=cache,
        l2=0.3,
    ), cache


class TargetPathTests(unittest.TestCase):
    def setUp(self):
        self.batch = build_target_event_batch(
            _Tokenizer(), _requests(), device=torch.device("cpu")
        )
        torch.manual_seed(1)
        self.logits = torch.randn(self.batch.event_count, 32, dtype=torch.float32)
        self.reference = capture_reference(self.logits, self.batch)

    def test_target_excluded_normalization_and_zero_reference(self):
        values = evaluate_values(self.logits, self.batch, self.reference)
        self.assertTrue(torch.allclose(values.barrier, torch.tensor(0.0), atol=1e-7))
        self.assertTrue(torch.allclose(self.reference.q0.sum(-1), torch.ones(self.batch.event_count)))

    def test_target_logit_has_zero_direct_qkl_effect(self):
        changed = self.logits.clone()
        changed.scatter_add_(
            1,
            self.batch.target_ids[:, None],
            torch.full((self.batch.event_count, 1), 3.0),
        )
        values = evaluate_values(changed, self.batch, self.reference)
        self.assertTrue(torch.allclose(values.barrier, torch.tensor(0.0), atol=1e-6))
        baseline = evaluate_values(self.logits, self.batch, self.reference)
        self.assertTrue(torch.all(values.target_log_odds > baseline.target_log_odds))

    def test_macro_event_weights_sum_per_request(self):
        sums = torch.zeros(self.batch.request_count)
        for event, weight in zip(self.batch.events, self.batch.event_weights):
            sums[event.request_index] += weight.cpu()
        expected = torch.full_like(sums, 1.0 / self.batch.request_count)
        self.assertTrue(torch.allclose(sums, expected))

    def test_sequence_nll_is_token_mean_not_sum(self):
        values = evaluate_values(self.logits, self.batch, self.reference)
        grouped = sequence_nll_by_request(values.target_nll, self.batch)
        for request_index in range(self.batch.request_count):
            indices = [
                event.event_index
                for event in self.batch.events
                if event.request_index == request_index
            ]
            self.assertTrue(
                torch.allclose(grouped[request_index], values.target_nll[indices].mean())
            )

    def test_teacher_forced_target_ids_cover_unequal_lengths(self):
        grouped = {index: [] for index in range(len(_requests()))}
        for event in self.batch.events:
            grouped[event.request_index].append(event.target_token_id)
        for index, request in enumerate(_requests()):
            self.assertEqual(grouped[index], target_token_ids(_Tokenizer(), request["target_new"]))


class GeometryTests(unittest.TestCase):
    def test_factorization_matches_stock_solve_and_AJ_equals_PK(self):
        factors, _ = _factorization()
        self.assertLessEqual(factors.solve_backward_error, factors.solve_backward_tolerance)
        self.assertLessEqual(factors.stock_velocity_relative, factors.solve_backward_tolerance)
        self.assertTrue(torch.allclose(factors.velocity, factors.residual @ factors.writer_map.T))

    def test_metric_is_psd_and_fast_matches_energy(self):
        factors, _ = _factorization()
        receipt = factors.metric_receipt
        self.assertGreaterEqual(receipt.minimum_eigenvalue, -receipt.psd_tolerance)
        self.assertLessEqual(receipt.fast_energy_relative, receipt.fast_energy_tolerance)
        self.assertEqual(factors.metric.shape, (3, 3))

    def test_history_cache_changes_projected_velocity(self):
        with_history, cache = _factorization(history=True)
        without_history, empty = _factorization(history=False)
        pullback = with_history.residual.clone()
        projected_history = project_one_sided_velocity(
            residual=with_history.residual,
            writer_map=with_history.writer_map,
            metric=with_history.metric,
            metric_pinv=with_history.metric_pinv,
            barrier_pullback=pullback,
            keys=with_history.keys,
            history_cache=cache,
            l2=0.3,
        )
        projected_empty = project_one_sided_velocity(
            residual=without_history.residual,
            writer_map=without_history.writer_map,
            metric=without_history.metric,
            metric_pinv=without_history.metric_pinv,
            barrier_pullback=pullback,
            keys=without_history.keys,
            history_cache=empty,
            l2=0.3,
        )
        self.assertFalse(torch.allclose(projected_history.residual_velocity, projected_empty.residual_velocity))

    def test_one_sided_negative_rate_keeps_native_velocity(self):
        factors, cache = _factorization()
        projected = project_one_sided_velocity(
            residual=factors.residual,
            writer_map=factors.writer_map,
            metric=factors.metric,
            metric_pinv=factors.metric_pinv,
            barrier_pullback=-factors.residual,
            keys=factors.keys,
            history_cache=cache,
            l2=0.3,
        )
        self.assertFalse(projected.active)
        self.assertTrue(torch.equal(projected.residual_velocity, factors.residual))
        self.assertLessEqual(projected.projected_rate, 0.0)

    def test_positive_rate_projects_to_halfspace_and_energy_identity(self):
        factors, cache = _factorization()
        projected = project_one_sided_velocity(
            residual=factors.residual,
            writer_map=factors.writer_map,
            metric=factors.metric,
            metric_pinv=factors.metric_pinv,
            barrier_pullback=factors.residual,
            keys=factors.keys,
            history_cache=cache,
            l2=0.3,
        )
        self.assertTrue(projected.active)
        tolerance = 64 * torch.finfo(torch.float32).eps * abs(projected.native_rate)
        self.assertLessEqual(abs(projected.projected_rate), tolerance)
        self.assertAlmostEqual(
            projected.correction_energy,
            projected.current_key_energy + projected.history_cache_energy + projected.l2_energy,
            delta=max(projected.correction_energy, 1.0) * 1e-4,
        )

    def test_projected_point_is_qp_minimum_against_feasible_perturbations(self):
        factors, cache = _factorization()
        pullback = factors.residual.clone()
        projected = project_one_sided_velocity(
            residual=factors.residual,
            writer_map=factors.writer_map,
            metric=factors.metric,
            metric_pinv=factors.metric_pinv,
            barrier_pullback=pullback,
            keys=factors.keys,
            history_cache=cache,
            l2=0.3,
        )
        optimum = projected.residual_velocity
        optimum_distance = projected.correction_energy
        torch.manual_seed(11)
        for _ in range(32):
            tangent = torch.randn_like(optimum)
            rate = torch.sum(pullback * tangent)
            if rate > 0:
                tangent = tangent - (rate / torch.sum(pullback**2)) * pullback
            candidate = optimum + 0.2 * tangent
            self.assertLessEqual(float(torch.sum(pullback * candidate)), 2e-5)
            delta = candidate - factors.residual
            distance = float(torch.sum((delta @ factors.metric) * delta))
            self.assertGreaterEqual(distance + 2e-4, optimum_distance)

    def test_low_rank_pullback_matches_dense_GJ(self):
        torch.manual_seed(17)
        left = torch.randn(5, 9)
        right = torch.randn(7, 9)
        writer_map = torch.randn(7, 3)
        observed = low_rank_pullback(left, right, writer_map)
        expected = (left @ right.T) @ writer_map
        self.assertTrue(torch.allclose(observed, expected, atol=1e-5, rtol=1e-5))

    def test_eta_zero_positive_rate_fails_closed(self):
        residual = torch.ones(2, 1)
        with self.assertRaises(ActionableNormalBoundary):
            project_one_sided_velocity(
                residual=residual,
                writer_map=torch.zeros(2, 1),
                metric=torch.zeros(1, 1),
                metric_pinv=torch.zeros(1, 1),
                barrier_pullback=torch.ones_like(residual),
                keys=torch.zeros(2, 1),
                history_cache=torch.zeros(2, 2),
                l2=0.0,
            )

    def test_euler_step_applies_h_once(self):
        entry = torch.randn(4, 5)
        velocity = torch.randn_like(entry)
        observed = entry.clone()
        for _ in range(4):
            apply_velocity_(observed, velocity, step_size=0.25)
        self.assertTrue(torch.allclose(observed, entry + velocity, atol=2e-6, rtol=2e-6))

    def test_exact_endpoint_adoption_avoids_subtract_add_drift(self):
        entry = torch.tensor([1.0e-20, 1.0, -3.0], dtype=torch.float32)
        temporary = torch.tensor([-1.0e-19, 1.0000001192092896, -2.999999761581421])
        weights = {"w": entry.clone()}
        endpoint = {"w": temporary.clone()}
        receipt = adopt_exact_endpoint(weights, endpoint)
        self.assertTrue(receipt["pass"])
        self.assertTrue(torch.equal(weights["w"], temporary))
        snapshot = capture_exact_endpoint(weights)
        self.assertTrue(fp32_comparison(snapshot["w"], temporary)["bitwise_equal"])


class BoundaryTests(unittest.TestCase):
    def test_base_editor_official_parity_request_preserves_subject(self):
        requests = _prepare_requests(
            ["Ada lives in"], [" Paris"], [" London"], subject=["Ada"]
        )
        self.assertEqual(requests[0]["subject"], "Ada")

    def test_arm_step_boundary_and_stock_n1_bypass(self):
        BarrierWriterConfig(BarrierArm.OFFICIAL, 1)
        BarrierWriterConfig(BarrierArm.SPLIT, 2)
        BarrierWriterConfig(BarrierArm.PROJECTED, 4)
        with self.assertRaises(ValueError):
            BarrierWriterConfig(BarrierArm.OFFICIAL, 2)
        with self.assertRaises(ValueError):
            BarrierWriterConfig(BarrierArm.PROJECTED, 8)

    def test_controller_has_no_forbidden_input_or_fallback_path(self):
        package = Path(__file__).parents[1]
        writer_text = (package / "writer.py").read_text()
        writer = ast.parse(writer_text)
        called_names = {
            node.func.id
            for node in ast.walk(writer)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("compute_locality_quality", called_names)
        self.assertNotIn("retry", called_names)
        self.assertNotIn("target_gradients", writer_text)
        self.assertNotIn('["target_true"]', writer_text)
        self.assertNotIn("build_sequence_event_batch", writer_text)
        self.assertNotIn("fallback", writer_text.lower())

    def test_writer_entry_fail_closes_non_right_padding(self):
        tok = _Tokenizer()
        tok.padding_side = "left"
        with self.assertRaises(TokenizerContractBoundary):
            require_right_padding(tok, caller="test")
        with self.assertRaises(TokenizerContractBoundary):
            apply_strength_neutral_barrier_to_model(
                None,
                tok,
                [],
                None,
                BarrierWriterConfig(BarrierArm.OFFICIAL, 1),
            )

    def test_projected_path_has_no_first_node_bypass(self):
        from project.run_scripts.alphaedit_strength_neutral_barrier import writer

        source = inspect.getsource(writer._execute_guided)
        self.assertNotIn("layer_index == 0 and node == 0", source)
        self.assertIn("if config.arm is BarrierArm.PROJECTED", source)


if __name__ == "__main__":
    unittest.main()
