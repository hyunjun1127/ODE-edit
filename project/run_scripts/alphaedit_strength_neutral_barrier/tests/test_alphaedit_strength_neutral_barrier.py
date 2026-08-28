from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

import torch

from easyeditor.editors.utils import _prepare_requests
from project.run_scripts.alphaedit_strength_neutral_barrier.geometry import (
    FactorTerm,
    LowRankFactor,
    apply_factor_terms_,
    factor_dense_inner,
    factor_terms_times_keys,
    project_right,
    solve_strength_neutral_correction,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.contracts import (
    TokenizerContractBoundary,
    require_right_padding,
    select_multitoken_alignment_fixtures,
    target_token_ids,
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
from project.run_scripts.alphaedit_strength_neutral_barrier.run_preflight import (
    _max_ulp_distance,
)
from project.run_scripts.alphaedit_strength_neutral_barrier.endpoint_adoption import (
    adopt_exact_endpoint,
    capture_exact_endpoint,
    fp32_comparison,
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


class TargetPathTests(unittest.TestCase):
    def setUp(self):
        self.batch = build_target_event_batch(
            _Tokenizer(),
            [
                {
                    "case_id": 7,
                    "prompt": "{} lives in",
                    "subject": "Ada",
                    "target_new": " XY",
                }
            ],
            device=torch.device("cpu"),
        )
        torch.manual_seed(1)
        self.logits = torch.randn(self.batch.event_count, 32, dtype=torch.float32)
        self.reference = capture_reference(self.logits, self.batch)

    def test_target_excluded_normalization_and_zero_reference(self):
        values = evaluate_values(self.logits, self.batch, self.reference)
        self.assertTrue(torch.allclose(values.barrier, torch.tensor(0.0), atol=1e-7))
        self.assertTrue(torch.allclose(self.reference.q0.sum(-1), torch.ones(3)))

    def test_target_strength_is_separate_from_non_target_q(self):
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

    def test_non_target_redistribution_activates_barrier(self):
        changed = self.logits.clone()
        changed[:, 4] += 2.0
        values = evaluate_values(changed, self.batch, self.reference)
        self.assertGreater(float(values.barrier), 0.0)

    def test_sequence_nll_groups_all_target_events_by_request(self):
        values = evaluate_values(self.logits, self.batch, self.reference)
        grouped = sequence_nll_by_request(values.target_nll, self.batch)
        self.assertEqual(tuple(grouped.shape), (1,))
        self.assertTrue(torch.allclose(grouped[0], values.target_nll.sum()))

    def test_multitoken_fixture_selection_and_teacher_forced_ids(self):
        def record(case_id, new, true):
            return {
                "case_id": case_id,
                "requested_rewrite": {
                    "prompt": "{} lives in",
                    "subject": "Ada",
                    "target_new": {"str": new},
                    "target_true": {"str": true},
                },
            }

        records = [
            record(1, "AA", "B"),
            record(2, "C", "DDD"),
            record(3, "EEEE", "F"),
        ]
        fixtures = select_multitoken_alignment_fixtures(records, _Tokenizer())
        self.assertEqual(
            [fixture.criterion for fixture in fixtures],
            ["target_new_multitoken", "source_longer", "target_longer"],
        )
        requests = [
            {
                "case_id": record["case_id"],
                "prompt": record["requested_rewrite"]["prompt"],
                "subject": record["requested_rewrite"]["subject"],
                "target_new": record["requested_rewrite"]["target_new"]["str"],
            }
            for record in records
        ]
        batch = build_target_event_batch(
            _Tokenizer(), requests, device=torch.device("cpu")
        )
        self.assertGreater(batch.event_count, batch.request_count)
        grouped = {index: [] for index in range(len(requests))}
        for event in batch.events:
            grouped[event.request_index].append(event.target_token_id)
        for index, request in enumerate(requests):
            self.assertEqual(
                grouped[index], target_token_ids(_Tokenizer(), request["target_new"])
            )


class GeometryTests(unittest.TestCase):
    def test_exact_endpoint_adoption_avoids_subtract_add_drift(self):
        entry = torch.tensor([1.0e-20, 1.0, -3.0], dtype=torch.float32)
        temporary = torch.tensor([-1.0e-19, 1.0000001192092896, -2.999999761581421], dtype=torch.float32)
        delta_replay = entry + (temporary - entry)
        self.assertFalse(torch.equal(delta_replay, temporary))
        weights = {"w": entry.clone()}
        endpoint = {"w": temporary.clone()}
        receipt = adopt_exact_endpoint(weights, endpoint)
        self.assertTrue(receipt["pass"])
        self.assertTrue(torch.equal(weights["w"], temporary))

    def test_endpoint_snapshot_is_independent_and_exact(self):
        weights = {"w": torch.arange(8, dtype=torch.float32)}
        endpoint = capture_exact_endpoint(weights)
        weights["w"].zero_()
        self.assertTrue(torch.equal(endpoint["w"], torch.arange(8, dtype=torch.float32)))
        comparison = fp32_comparison(endpoint["w"], endpoint["w"].clone())
        self.assertTrue(comparison["bitwise_equal"])
        self.assertEqual(comparison["max_ulp"], 0)

    def test_closed_form_preserves_keys_strength_and_direction(self):
        torch.manual_seed(3)
        out_dim, in_dim, rank = 5, 7, 4
        left = torch.randn(out_dim, rank)
        right = torch.randn(in_dim, rank)
        barrier = LowRankFactor(left.float(), right.float())
        target = LowRankFactor(torch.randn(out_dim, rank), right.float())
        projector = torch.eye(in_dim)
        keys = torch.zeros(in_dim, 1)
        keys[0, 0] = 1.0
        projected_right, receipt = project_right(right, projector, keys)
        dense_g = left @ right.T
        native = dense_g.clone()
        solution = solve_strength_neutral_correction(
            barrier_gradient=barrier,
            target_gradients=[target],
            native_delta=native,
            projected_right=projected_right,
            keys=keys,
        )
        self.assertTrue(solution.active)
        correction = torch.zeros_like(native)
        for term in solution.terms:
            correction += float(term.coefficient) * term.factor.left @ term.factor.right.T
        self.assertLess(float(torch.linalg.vector_norm(correction @ keys)), 5e-5)
        self.assertLess(abs(float(torch.sum((target.left @ target.right.T) * correction))), 5e-4)
        self.assertLessEqual(solution.directional_rate, 5e-4)
        self.assertLess(receipt.key_residual_max_abs, 5e-6)
        self.assertEqual(len(solution.target_strength_inner_abs), 1)
        self.assertLess(solution.target_strength_inner_abs[0], 5e-4)

    def test_factor_dense_inner_matches_materialization(self):
        torch.manual_seed(4)
        factor = LowRankFactor(torch.randn(4, 3), torch.randn(6, 3))
        dense = torch.randn(4, 6)
        expected = torch.sum((factor.left @ factor.right.T) * dense)
        self.assertTrue(torch.allclose(factor_dense_inner(factor, dense), expected))

    def test_split_fractional_write_matches_native_fp32_tolerance(self):
        torch.manual_seed(5)
        entry = torch.randn(11, 13)
        delta = torch.randn_like(entry) * 1e-2
        official = entry + delta
        split = entry.clone()
        for _ in range(8):
            apply_factor_terms_(split, delta, (), step_scale=1 / 8)
        self.assertTrue(torch.allclose(split, official, atol=2e-6, rtol=2e-6))

    def test_actual_dk_uses_sum_of_low_rank_terms(self):
        torch.manual_seed(9)
        keys = torch.randn(6, 2)
        factors = [
            FactorTerm(torch.tensor(0.7), LowRankFactor(torch.randn(4, 3), torch.randn(6, 3))),
            FactorTerm(torch.tensor(-0.2), LowRankFactor(torch.randn(4, 2), torch.randn(6, 2))),
        ]
        observed = factor_terms_times_keys(factors, keys)
        dense = sum(
            float(term.coefficient) * term.factor.left @ term.factor.right.T
            for term in factors
        )
        self.assertTrue(torch.allclose(observed, dense @ keys, atol=1e-6, rtol=1e-6))

    def test_fp32_ulp_distance_is_exact_and_ordered(self):
        value = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
        self.assertEqual(_max_ulp_distance(value, value.clone()), 0)
        adjacent = value.clone()
        adjacent[0] = torch.nextafter(
            adjacent[0], torch.tensor(float("inf"), dtype=torch.float32)
        )
        self.assertEqual(_max_ulp_distance(value, adjacent), 1)


class BoundaryTests(unittest.TestCase):
    def test_base_editor_official_parity_request_preserves_subject(self):
        requests = _prepare_requests(
            ["Ada lives in"],
            [" Paris"],
            [" London"],
            subject=["Ada"],
        )
        self.assertEqual(requests[0]["subject"], "Ada")
        self.assertIn(requests[0]["subject"], requests[0]["prompt"])

    def test_arm_step_boundary(self):
        BarrierWriterConfig(BarrierArm.OFFICIAL, 1)
        BarrierWriterConfig(BarrierArm.SPLIT, 2)
        BarrierWriterConfig(BarrierArm.BARRIER, 8)
        with self.assertRaises(ValueError):
            BarrierWriterConfig(BarrierArm.OFFICIAL, 2)
        with self.assertRaises(ValueError):
            BarrierWriterConfig(BarrierArm.BARRIER, 1)

    def test_controller_has_no_locality_or_retry_input_path(self):
        package = Path(__file__).parents[1]
        writer = ast.parse((package / "writer.py").read_text())
        called_names = {
            node.func.id
            for node in ast.walk(writer)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("compute_locality_quality", called_names)
        self.assertNotIn("retry", called_names)

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

    def test_barrier_first_node_has_no_native_bypass_branch(self):
        from project.run_scripts.alphaedit_strength_neutral_barrier import writer

        source = inspect.getsource(writer._execute_guided)
        self.assertNotIn("and not native_predictor", source)
        self.assertNotIn("layer_index == 0 and node == 0", source)
        self.assertIn("if config.arm is BarrierArm.BARRIER", source)


if __name__ == "__main__":
    unittest.main()
