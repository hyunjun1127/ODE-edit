"""Compact synthetic CPU fixtures; no model, GPU, Slurm or external artifacts."""
import ast
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from project.run_scripts.native_response_ode_v31.algebra import FrozenNormalization, nnls_response
from .normalization_views import NormalizationView, NormalizationBoundary, source_residual
from .requestwise import (contributions, RequestwiseBoundary, all_row_influence,
                          publication_rows, global_summary)
from .diagnosis import (capture_repeatability, same_state_attribution, cross_objective_scores,
                        inspect_sealed_local_member, historical_availability)


class DiagnosisTests(unittest.TestCase):
    def fixture(self):
        generator = torch.Generator().manual_seed(20260907)
        target = torch.randn((7, 4), generator=generator, dtype=torch.float32)
        terminal = torch.randn((7, 4), generator=generator, dtype=torch.float32)
        target[:, 1] = terminal[:, 1]  # exact-zero remains in native request inventory
        raw = torch.randn((3, 7, 4), generator=generator, dtype=torch.float32)
        source = FrozenNormalization.capture(target, terminal, "sealed-entry")
        q = torch.tensor([.3, 1.7, 2.4], dtype=torch.float64)
        return target, terminal, raw, source, q

    def test_n0_exact_delegate_and_tiny_positive(self):
        target = torch.tensor([[0., 1e-8, .234567], [0., 2e-8, .789012]], dtype=torch.float32)
        source = FrozenNormalization.capture(target, torch.zeros_like(target), "entry")
        view = NormalizationView.from_source(source)
        self.assertEqual(view.active.tolist(), [False, True, True])
        self.assertTrue(torch.equal(view.weight(target), source.weight(target)))
        self.assertIsNone(view.receipt()["applied_denominators"][0])
        self.assertEqual(view.receipt()["request_count"], 3)
        self.assertEqual(view.receipt()["request_removal_count"], 0)

    def test_nrms_fp32_scale_fp64_rms_and_shared_active(self):
        target, terminal, raw, source, q = self.fixture()
        view = NormalizationView.from_source(source, "NRMS_ENTRY")
        scales = source.scales[source.active].double()
        denominator = scales.square().mean().sqrt()*int(source.active.sum())**.5
        expected = ((target-terminal).double()[:, source.active]/denominator).reshape(-1)
        self.assertTrue(torch.equal(view.weight(target-terminal), expected))
        self.assertTrue(torch.equal(view.active, source.active))
        self.assertTrue(torch.equal(view.scales, source.scales))
        self.assertEqual(view.receipt()["normalization_id"], "NRMS_ENTRY")

    def test_frozen_copies_and_mutation_rejection(self):
        _, _, _, source, _ = self.fixture()
        view = NormalizationView.from_source(source)
        before = view.scales
        source.scales.mul_(2)
        self.assertTrue(torch.equal(before, view.scales))
        returned = view.scales
        returned.zero_()
        self.assertTrue(torch.equal(before, view.scales))
        view._source.scales.mul_(2)
        with self.assertRaisesRegex(NormalizationBoundary, "MUTATED"):
            view.receipt()

    def test_primary_fp32_subtraction_and_repeat_capture(self):
        target = torch.tensor([[1e10, 1.]], dtype=torch.float32)
        terminal = torch.tensor([[-3.14159, 1.]], dtype=torch.float32)
        primary = source_residual(target, terminal)
        self.assertEqual(primary.dtype, torch.float32)
        self.assertFalse(torch.equal(primary.double(), target.double()-terminal.double()))
        fact = capture_repeatability(target, [terminal.clone() for _ in range(3)],
                                     semantic_identities=["same"]*3)
        self.assertEqual(fact["normalization_initialization_count"], 1)
        self.assertGreater(fact["captures"][0]["fp32_subtract_vs_fp64_subtract_max_abs"], 0)
        self.assertEqual(fact["captures"][2]["repeat_l2_deviation"], 0)

    def test_requestwise_sum_production_flatten_and_inactive(self):
        target, terminal, raw, source, q = self.fixture()
        view = NormalizationView.from_source(source)
        bundle = contributions(target-terminal, raw, q, view, (4, 6, 8))
        expected = torch.stack([source.weight(raw[i])/q[i].sqrt() for i in range(3)], 1)
        self.assertTrue(torch.equal(bundle.psi, expected))
        self.assertTrue(torch.allclose(bundle.H_by_request.sum(0), bundle.H, atol=1e-10, rtol=1e-10))
        self.assertTrue(torch.allclose(bundle.g_by_request.sum(0), bundle.g, atol=1e-10, rtol=1e-10))
        self.assertEqual(float(bundle.H_by_request[1].abs().sum()), 0)
        self.assertEqual(float(bundle.g_by_request[1].abs().sum()), 0)
        solution = nnls_response(bundle.e, bundle.psi, torch.eye(3), .1)
        rows = publication_rows(bundle, ["a", "b", "c", "d"], solution.coefficients)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["g_i"][1], 0.)
        self.assertEqual(rows[1]["applied_denominator"], None)

    def test_whitening_coordinate_scaling_and_layer_permutation(self):
        target, terminal, raw, source, q = self.fixture()
        view = NormalizationView.from_source(source)
        baseline = contributions(target-terminal, raw, q, view, (4, 6, 8))
        # Powers of two preserve raw FP32 scaling exactly.
        scale = torch.tensor([2., .5, 4.], dtype=torch.float64)
        scaled = contributions(target-terminal, raw*scale.float()[:, None, None],
                               q*scale.square(), view, (4, 6, 8))
        self.assertTrue(torch.equal(baseline.psi, scaled.psi))
        c = nnls_response(baseline.e, baseline.psi, torch.eye(3), .1).coefficients
        scaled_c = nnls_response(scaled.e, scaled.psi, torch.eye(3), .1).coefficients
        self.assertTrue(torch.equal(c, scaled_c))
        self.assertTrue(torch.equal(c/q.sqrt(), scaled_c/(q*scale.square()).sqrt()*scale))
        permutation = [2, 0, 1]
        permuted = contributions(target-terminal, raw[permutation], q[permutation], view, (8, 4, 6))
        cp = nnls_response(permuted.e, permuted.psi, torch.eye(3), .1).coefficients
        self.assertTrue(torch.allclose(cp, c[permutation], atol=1e-10, rtol=1e-10))

    def test_request_permutation_preserves_global_solve(self):
        target, terminal, raw, source, q = self.fixture()
        baseline = contributions(target-terminal, raw, q, NormalizationView.from_source(source), (4, 6, 8))
        order = [3, 1, 0, 2]
        permuted_source = FrozenNormalization.capture(target[:, order], terminal[:, order], 'permuted-entry')
        permuted = contributions((target-terminal)[:, order], raw[:, :, order], q,
                                 NormalizationView.from_source(permuted_source), (4, 6, 8))
        self.assertTrue(torch.allclose(permuted.g_by_request, baseline.g_by_request[order], atol=1e-10, rtol=1e-10))
        c = nnls_response(baseline.e, baseline.psi, torch.eye(3), .1).coefficients
        cp = nnls_response(permuted.e, permuted.psi, torch.eye(3), .1).coefficients
        self.assertTrue(torch.allclose(c, cp, atol=1e-10, rtol=1e-10))
        self.assertEqual(global_summary(baseline, c)['error_span_observation']['controller_influence_count'], 0)

    def test_all_row_influence_uses_existing_solver_not_request_deletion(self):
        target, terminal, raw, source, q = self.fixture()
        view = NormalizationView.from_source(source)
        bundle = contributions(target-terminal, raw, q, view, (4, 6, 8))
        e_before, psi_before = bundle.e.clone(), bundle.psi.clone()
        records = all_row_influence(bundle, torch.eye(3), .31622776601683794)
        self.assertEqual(len(records), 4)
        self.assertTrue(torch.equal(bundle.e, e_before))
        self.assertTrue(torch.equal(bundle.psi, psi_before))
        for record in records:
            self.assertEqual(record["original_active_request_count"], 3)
            self.assertEqual(record["lambda_response"], .31622776601683794)
            self.assertEqual(record["controller_influence_count"], 0)
            self.assertEqual(record["request_removal_count"], 0)
            self.assertEqual(record["reconstruction"]["H"]["status"], "PASS")
        self.assertEqual(records[1]["coefficient_delta_observed"], [0., 0., 0.])

    def test_empty_active_and_empty_native_directions(self):
        target = torch.zeros((2, 3), dtype=torch.float32)
        source = FrozenNormalization.capture(target, target.clone(), "zero")
        for normalization_id in ("N0_SOURCE", "NRMS_ENTRY"):
            view = NormalizationView.from_source(source, normalization_id)
            bundle = contributions(target, torch.empty((0, 2, 3), dtype=torch.float32),
                                   torch.empty(0), view, ())
            self.assertEqual(bundle.psi.shape, (0, 0))
            records = all_row_influence(bundle, torch.empty((0, 0)), .1)
            self.assertEqual(len(records), 3)
            self.assertIsNone(global_summary(bundle, [])['top1_trace_share'])

    def test_nonfinite_wrong_dtype_and_zero_q_boundaries(self):
        target, terminal, raw, source, q = self.fixture()
        view = NormalizationView.from_source(source)
        with self.assertRaises(NormalizationBoundary):
            view.weight(target.double())
        invalid = target.clone(); invalid[0, 0] = float('nan')
        with self.assertRaises(NormalizationBoundary):
            view.weight(invalid)
        with self.assertRaises(RequestwiseBoundary):
            contributions(target-terminal, raw, [0., 1., 1.], view, (4, 6, 8))
        with self.assertRaises(NormalizationBoundary):
            NormalizationView.from_source(source, 'SILENT_FLOOR')

    def test_same_state_shadow_and_cross_objective_are_observer_only(self):
        target, terminal, raw, source, q = self.fixture()
        record = same_state_attribution(source_normalization=source, residual32=target-terminal,
            raw_responses32=raw, q_layers=q, layer_ids=(4, 6, 8), metric=torch.eye(3),
            lambda_response=.03162277660168379, request_ids=['a', 'b', 'c', 'd'])
        self.assertEqual(record['actual_write_count'], 0)
        self.assertEqual(set(record['views']), {'N0_SOURCE', 'NRMS_ENTRY'})
        for value in record['views'].values():
            self.assertEqual(value['lambda_response'], .03162277660168379)
        scores = cross_objective_scores(target, terminal, source)
        self.assertEqual(scores['history_append_count'], 0)
        self.assertAlmostEqual(scores['objectives']['N0_SOURCE']['V'], .5, places=6)
        self.assertAlmostEqual(scores['objectives']['NRMS_ENTRY']['V'], .5, places=6)

    def test_bounded_inventory_missing_and_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # Fixture creation is isolated to the temporary test directory.
            path = root/'fixture'; path.write_bytes(b'fixed')
            sha = hashlib.sha256(b'fixed').hexdigest()
            self.assertEqual(inspect_sealed_local_member(path, sha, 5)['status'], 'SEALED_BYTES_REHASH_PASS')
            link = root/'linked'; link.symlink_to(path)
            self.assertEqual(inspect_sealed_local_member(link, sha, 5)['status'], 'SYMLINK_BOUNDARY')
            missing = root/'missing'
            rows = [dict(alias='llama3-8b-inst', arm='JV_NATIVE', batch='5', path=str(missing),
                         bytes='5', sha256=sha, journal_replay_parity='NOT_TESTED')]
            availability = historical_availability(checkpoint_rows=rows, context_rows=[])
            self.assertEqual(availability['status'], 'HISTORICAL_EXACT_STATE_UNAVAILABLE_ON_SERVER1')
            self.assertFalse(availability['S_blocked_by_D'])
            self.assertFalse(availability['W10_as_W9_allowed'])

    def test_observer_dependency_firewall(self):
        package = Path(__file__).parent
        for filename in ('requestwise.py', 'diagnosis.py', 'normalization_views.py'):
            tree = ast.parse((package/filename).read_text())
            imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            for module in imported:
                self.assertFalse(any(name in (module or '') for name in
                    ('runtime', 'overlay', 'native_binding', 'terminal_jvp', 'trajectory')))
            calls = [node.func.attr for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
            self.assertFalse(any(name in calls for name in
                ('run_joint', 'run_official', 'finalize', 'compute_fixed_z', 'load_model', 'cuda')))


if __name__ == '__main__':
    unittest.main()
