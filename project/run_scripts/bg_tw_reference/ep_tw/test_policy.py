"""Deterministic CPU arithmetic/selector fixtures, not a Llama/GPU gate."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

import torch

from .policy import (CandidateObservation, NumericalPolicy, PolicyError,
                     build_correction, choose_candidate, materialize_candidates,
                     project_direction, tensor_sha256, validate_dispatch)


class PolicyFixtures(unittest.TestCase):
    def fixture(self):
        return dict(Vp=torch.eye(2) * 2, Wentry=torch.zeros(2, 2),
                    Zp=torch.zeros(2, 2), anchors=torch.zeros(2, 2),
                    radii=torch.ones(2) * 10, A=torch.eye(2),
                    gE=torch.zeros(2, 2), gD=-torch.ones(2, 2))

    def candidates(self, correction=None):
        vp = torch.eye(2) * 2
        correction = torch.ones(2, 2) * .1 if correction is None else correction
        return materialize_candidates(vp, correction, torch.eye(2), native_delta_norm=3.)

    @staticmethod
    def observation(e=1., d=.5, successes=(1, 2), ids=(0, 1, 2)):
        return CandidateObservation(e, frozenset(successes), d, tuple(ids))

    def test_projection_interior(self):
        ge, gd = torch.tensor([[1., 0.]]), torch.tensor([[1., 2.]])
        d, info = project_direction(ge, gd)
        self.assertTrue(torch.equal(d, -gd))
        self.assertLess(info["ge_dot_d"], 0.)
        self.assertEqual(info["kkt_lambda"], 0.)

    def test_projection_conflict_and_kkt(self):
        ge, gd = torch.tensor([[1., 0.]]), torch.tensor([[-1., 2.]])
        d, info = project_direction(ge, gd)
        self.assertTrue(torch.equal(d, torch.tensor([[0., -2.]])))
        self.assertEqual(info["ge_dot_d"], 0.)
        self.assertEqual(info["kkt_lambda"], 1.)
        self.assertEqual(info["kkt_stationarity_norm"], 0.)

    def test_projection_zero_and_antiparallel(self):
        zeros, ones = torch.zeros(2, 3), torch.ones(2, 3)
        self.assertTrue(torch.equal(project_direction(zeros, ones)[0], -ones))
        self.assertTrue(torch.equal(project_direction(ones, zeros)[0], zeros))
        self.assertTrue(torch.equal(project_direction(ones, -ones)[0], zeros))

    def test_projection_matches_feasible_nearest_grid(self):
        ge, gd = torch.tensor([[1., 0.]]), torch.tensor([[-1., 2.]])
        d, _ = project_direction(ge, gd)
        optimum = ((d + gd) ** 2).sum().item()
        for x in (-2., -1., 0.):
            for y in (-3., -2., -1., 0.):
                self.assertLessEqual(optimum, ((torch.tensor([[x, y]]) + gd) ** 2).sum().item())

    def test_correction_uses_actual_native_norm_and_trust(self):
        f = self.fixture()
        out = build_correction(**f)
        self.assertTrue(out.available)
        self.assertAlmostEqual(out.diagnostics["actual_native_delta_norm"], 8. ** .5)
        self.assertLessEqual(float((out.C @ f["A"]).double().norm()), .25 * 8. ** .5 * (1 + 1e-6))
        self.assertEqual(out.diagnostics["raw_anchor"], "ACTUAL_STORED_VP")
        json.dumps(out.to_dict(), allow_nan=False)

    def test_native_ball_invalid_is_technical_not_raw_fallback(self):
        f = self.fixture()
        f["Zp"] = torch.ones(2, 2) * 20
        with self.assertRaisesRegex(PolicyError, "NATIVE_ZP_OUTSIDE"):
            build_correction(**f)

    def test_native_nonfinite_is_technical(self):
        f = self.fixture()
        f["Vp"][0, 0] = float("nan")
        with self.assertRaisesRegex(PolicyError, "VP_NONFINITE"):
            build_correction(**f)

    def test_nonfinite_gradient_is_typed_raw_only(self):
        f = self.fixture()
        f["gD"][0, 0] = float("nan")
        out = build_correction(**f)
        self.assertFalse(out.available)
        self.assertIn("GRADIENT_NONFINITE", out.status)
        self.assertEqual(torch.count_nonzero(out.C), 0)
        json.dumps(out.to_dict(), allow_nan=False)

    def test_mapped_overflow_is_json_safe_raw_only(self):
        f = self.fixture()
        f["gD"] = -torch.ones(2, 2) * 1e30
        f["A"] = torch.ones(2, 2) * 1e20
        out = build_correction(**f)
        self.assertIn("MAPPED_DIRECTION_NONFINITE", out.status)
        json.dumps(out.to_dict(), allow_nan=False)

    def test_zero_native_and_zero_map(self):
        f = self.fixture()
        f["Wentry"] = f["Vp"].clone()
        self.assertIn("ZERO_NATIVE", build_correction(**f).status)
        f = self.fixture()
        f["A"].zero_()
        self.assertIn("ZERO_MAPPED", build_correction(**f).status)

    def test_ball_projection_and_boundary_zero(self):
        f = self.fixture()
        f["radii"] = torch.ones(2) * .01
        out = build_correction(**f)
        self.assertTrue(out.available)
        self.assertTrue(bool(((f["Zp"] + out.C).norm(dim=0) <= .010001).all()))
        self.assertEqual(out.diagnostics["ball_projected_requests"], 2)
        f["radii"].zero_()
        self.assertIn("ZERO_PROJECTED", build_correction(**f).status)

    def test_alpha_cap_is_fixed_not_quality_feedback(self):
        out = build_correction(**self.fixture(), config=NumericalPolicy(alpha_cap=.01))
        self.assertEqual(out.diagnostics["alpha"], .01)

    def test_raw_exact_signed_zero_and_correction_only(self):
        vp = torch.tensor([[-0., 2.], [1., 3.]])
        correction = torch.ones(2, 2) * .1
        candidates = materialize_candidates(vp, correction, torch.eye(2), native_delta_norm=3.)
        self.assertEqual(tensor_sha256(vp), candidates[0].sha256)
        self.assertTrue(torch.signbit(candidates[0].weight[0, 0]))
        self.assertTrue(torch.equal(candidates[2].weight, vp + .5 * correction))
        self.assertFalse(torch.equal(candidates[2].weight, .5 * (vp + correction)))
        self.assertTrue(torch.equal(vp, torch.tensor([[-0., 2.], [1., 3.]])))

    def test_dedup_exact_bytes_not_nominal_map(self):
        candidates = self.candidates(torch.zeros(2, 2))
        self.assertEqual([x.duplicate_of for x in candidates], [None, "RAW", "RAW", "RAW"])
        selected = choose_candidate(candidates, {"RAW": self.observation()}, expected_count=3)
        self.assertEqual(selected.selected_id, "RAW")
        with self.assertRaisesRegex(PolicyError, "DUPLICATE_BYTES_EVALUATED"):
            choose_candidate(candidates, {"RAW": self.observation(), "C1": self.observation()}, expected_count=3)

    def test_actual_fp32_trust_excludes_large_correction(self):
        candidates = materialize_candidates(torch.zeros(2, 2), torch.ones(2, 2),
                                             torch.eye(2), native_delta_norm=1.)
        self.assertTrue(candidates[0].trust_valid)
        self.assertTrue(all(not x.trust_valid for x in candidates[1:]))
        selected = choose_candidate(candidates, {"RAW": self.observation()}, expected_count=3)
        self.assertEqual(selected.selected_id, "RAW")

    def test_strict_id_subset_not_success_count(self):
        candidates = self.candidates()
        obs = {x.id: self.observation() for x in candidates}
        obs["C1"] = self.observation(e=.9, d=.1, successes=(0, 1))
        result = choose_candidate(candidates, obs, expected_count=3)
        self.assertEqual(result.selected_id, "RAW")
        self.assertIn("RAW_STRICT_IDS_LOST", result.candidate_receipts[1]["reason"])

    def test_no_positive_nll_allowance_even_when_kl_better(self):
        candidates = self.candidates()
        obs = {x.id: self.observation() for x in candidates}
        obs["C1"] = self.observation(e=1. + 1e-15, d=.01)
        result = choose_candidate(candidates, obs, expected_count=3)
        self.assertEqual(result.selected_id, "RAW")
        self.assertIn("E_EXCEEDS", result.candidate_receipts[1]["reason"][0])

    def test_minimum_d_and_raw_tie_priority(self):
        candidates = self.candidates()
        obs = {x.id: self.observation() for x in candidates}
        obs["C1"] = self.observation(e=.9, d=.2)
        obs["C05"] = self.observation(e=.95, d=.1)
        self.assertEqual(choose_candidate(candidates, obs, expected_count=3).selected_id, "C05")
        obs["C1"] = self.observation(d=.5 - 1e-8)
        obs["C05"] = self.observation()
        self.assertEqual(choose_candidate(candidates, obs, expected_count=3).selected_id, "RAW")

    def test_equal_corrected_d_prefers_actual_smaller_norm(self):
        candidates = self.candidates()
        obs = {x.id: self.observation(d=.1 if x.id != "RAW" else .5) for x in candidates}
        self.assertEqual(choose_candidate(candidates, obs, expected_count=3).selected_id, "C025")

    def test_nonfinite_candidate_excluded_but_nonfinite_raw_errors(self):
        candidates = self.candidates()
        obs = {x.id: self.observation() for x in candidates}
        obs["C1"] = self.observation(d=float("nan"))
        result = choose_candidate(candidates, obs, expected_count=3)
        self.assertEqual(result.selected_id, "RAW")
        json.dumps(result.to_dict(), allow_nan=False)
        obs["RAW"] = self.observation(d=float("nan"))
        with self.assertRaisesRegex(PolicyError, "NATIVE_RAW_METRICS_NONFINITE"):
            choose_candidate(candidates, obs, expected_count=3)

    def test_ordered_identity_and_denominator_enforced(self):
        candidates = self.candidates()
        obs = {x.id: self.observation() for x in candidates}
        with self.assertRaisesRegex(PolicyError, "RAW_REQUEST_DENOMINATOR"):
            choose_candidate(candidates, obs)
        obs["C1"] = self.observation(ids=(2, 1, 0))
        with self.assertRaisesRegex(PolicyError, "CURRENT_IDENTITY"):
            choose_candidate(candidates, obs, expected_count=3)

    def test_dispatch_one_policy_no_calibration(self):
        root = Path(__file__).resolve().parents[4]
        dispatch = json.loads((root / "plans/global/2026-09-15-ep-tw1-c4-v3-dispatch-contract.json").read_text())
        self.assertFalse(validate_dispatch(dispatch)["N4_calibration_required"])
        dispatch["new_scientific_policies"].append("N4")
        with self.assertRaisesRegex(PolicyError, "DISPATCH_SCOPE"):
            validate_dispatch(dispatch)

    def test_numerical_policy_prevents_quality_allowance_or_method_change(self):
        self.assertIsNone(NumericalPolicy().receipt()["quality_positive_allowance"])
        with self.assertRaises(PolicyError):
            NumericalPolicy(zeta=.5)
        with self.assertRaises(PolicyError):
            NumericalPolicy(epsilon_num=0.)
        with self.assertRaises(PolicyError):
            NumericalPolicy(d_tie_atol=-1.)


if __name__ == "__main__":
    unittest.main()
