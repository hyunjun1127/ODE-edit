"""CPU algebra/routing fixtures only; actual T0 is not executed by tests."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from project.run_scripts.single_layer_edit_preserving_correction import geometry
from project.run_scripts.single_layer_edit_preserving_correction.common import tensor_sha
from .technical_decision import (FD_POLICY, TechnicalDecisionError, _native_replay,
    _projector_probe, classify_fd, cross_term_measurement, inherited_fd_grid,
    probe_evidence, repeated_panel_metrics)


def points(AD=2., norm=10.):
    return [dict(k=k, h=h, plus_margin=.3 + h * AD, minus_margin=.3 - h * AD,
        plus_sha256=f"p{k}", minus_sha256=f"m{k}", plus_nonzero=10, minus_nonzero=10)
        for k, h in enumerate(inherited_fd_grid(norm))]


def panels():
    row = dict(positions=[128, 129], labels=[1, 2], predictions=[1, 2], margins=[.2, -.3])
    return [dict(indices=[1, 23, 90, 444], rows=[deepcopy(row) for _ in range(4)], panel_phi=.09)
            for _ in range(3)]


class Tests(unittest.TestCase):
    def test_inherited_grid_exact(self):
        self.assertEqual(FD_POLICY["FD_scales"], 12)
        self.assertEqual(FD_POLICY["FD_adjacent"], 2)
        self.assertEqual(FD_POLICY["FD_signal_noise"], 10)
        self.assertEqual(FD_POLICY["FD_relative"], .01)
        self.assertEqual(inherited_fd_grid(10.), [.1 / (2 ** k) for k in range(12)])

    def test_fd_first_adjacent_window(self):
        result = classify_fd(points(), AD=2., noise=1e-10, direction_norm=1.)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["selected_window"], [0, 1])
        self.assertEqual(len(result["full_grid"]), 12)

    def test_fd_cannot_cherry_pick_isolated_point(self):
        source = points()
        for p in source:
            if p["k"] != 5:
                p["plus_margin"] += 2 * p["h"]
        result = classify_fd(source, AD=2., noise=1e-12, direction_norm=1.)
        self.assertFalse(result["derivative_pass"])
        self.assertEqual(result["status"], "NO_RESOLVED_ADJACENT_WINDOW")

    def test_small_ad_not_numerical_pass(self):
        result = classify_fd(points(AD=1e-14), AD=1e-14, noise=1e-10, direction_norm=1.)
        self.assertEqual(result["status"], "SMALL_AD_UNRESOLVED")
        self.assertEqual(result["numerical_validation"], "NOT_ESTABLISHED")

    def test_no_direction_not_derivative_pass(self):
        result = classify_fd([], AD=0., noise=1e-10, direction_norm=0.)
        self.assertFalse(result["derivative_pass"])
        self.assertFalse(result["applicable"])
        self.assertEqual(result["status"], "NO_DIRECTION")

    def test_rounded_no_move_does_not_resolve(self):
        source = points()
        for p in source:
            p["plus_sha256"] = p["minus_sha256"]
        result = classify_fd(source, AD=2., noise=1e-12, direction_norm=1.)
        self.assertFalse(result["derivative_pass"])

    def test_changed_grid_or_nonfinite_fails(self):
        source = points()
        source[2]["h"] *= 2
        with self.assertRaises(TechnicalDecisionError):
            classify_fd(source, AD=2., noise=1e-12, direction_norm=1.)
        source = points()
        source[0]["plus_margin"] = float("nan")
        with self.assertRaises(TechnicalDecisionError):
            classify_fd(source, AD=2., noise=1e-12, direction_norm=1.)

    def test_repeated_panel_threshold_and_exact_ids(self):
        source = panels()
        self.assertTrue(repeated_panel_metrics(source)["pass_"])
        source[1]["rows"][1]["predictions"][0] = 3
        self.assertFalse(repeated_panel_metrics(source)["pass_"])
        source = panels()
        source[1]["rows"][1]["margins"][0] += 1e-3
        self.assertFalse(repeated_panel_metrics(source)["pass_"])

    def test_repeat_does_not_accept_different_inputs(self):
        source = panels()
        source[1]["rows"][0]["labels"][0] = 3
        with self.assertRaises(TechnicalDecisionError):
            repeated_panel_metrics(source)

    def test_cross_term_nonzero_cumulative(self):
        generator = torch.Generator().manual_seed(1)
        e = torch.randn((4, 8), generator=generator, dtype=torch.float64)
        d = torch.randn((4, 8), generator=generator, dtype=torch.float64)
        k = torch.randn((8, 7), generator=generator, dtype=torch.float64)
        value = cross_term_measurement(e, d, k)
        self.assertTrue(value["pass_"])
        self.assertNotEqual(value["cross_term"], 0.)

    def test_projector_probes_bounded_label(self):
        space = geometry.RightSpace(np.eye(8), np.eye(8)[:, :2], "RESOLVED", {})
        result = _projector_probe(space)
        self.assertTrue(result["pass_"])
        self.assertIn("NOT_FULL_OPERATOR_CERTIFICATE", result["scope"])
        unresolved = geometry.RightSpace(np.eye(8), np.empty((8, 0)), "RANK_UNRESOLVED", {})
        self.assertFalse(_projector_probe(unresolved)["pass_"])

    def test_native_replay_cpu_fixture_no_fit(self):
        w0 = torch.zeros(2, 3)
        p, memory = torch.eye(3)[None], torch.zeros(1, 3, 3)
        k = torch.tensor([[1., 2., 0.], [2., 1., 0.], [0., 1., 2.], [1., 0., 1.]])
        z = [torch.tensor([.1 * i, .2]) for i in range(4)]
        current = torch.zeros(4, 2)
        keys = k.T
        residual = torch.stack(z, dim=1) - current.T
        lhs = p[0] @ (keys @ keys.T + memory[0]) + torch.eye(3)
        endpoint = w0 + torch.linalg.solve(lhs, (p[0] @ keys) @ residual.T).T
        native = dict(weight=endpoint, captures=dict(compute_ks=[k], compute_z=z,
            get_module_input_output_at_words=[current]), receipt=dict(
                entry_weight_sha256=tensor_sha(w0), history_sha256=tensor_sha(memory),
                projector_sha256=tensor_sha(p), history_append=0, compute_z=4, solve=1,
                source={"CPU_FIXTURE": True}))
        rt = SimpleNamespace(W=w0, W0=w0, P=p, M=memory, hp=SimpleNamespace(L2=1))
        with tempfile.TemporaryDirectory() as folder:
            result = _native_replay(rt, native, endpoint, Path(folder))
            self.assertTrue(result["pass_"])
            self.assertEqual(result["new_z_calls"], 0)
            self.assertFalse(result["weight_snapshot_saved"])
            self.assertEqual(list(Path(folder).iterdir()), [])
        self.assertFalse(_native_replay(rt, endpoint, endpoint, Path("unused"))["pass_"])

    def test_probe_hash_scalar_only_no_checkpoint(self):
        native = torch.zeros((2, 3), dtype=torch.float32)
        ideal = torch.full_like(native, .01, dtype=torch.float64)
        probe = (native.double() + ideal).float()
        value = probe_evidence(native, probe, ideal, .01)
        self.assertFalse(value["weight_snapshot_saved"])
        self.assertEqual(value["probe_sha256"], tensor_sha(probe))
        self.assertEqual(value["actual_changed_elements"], 6)
        self.assertEqual(json.loads(json.dumps(value)), value)
        self.assertNotIn("weight", value)

    def test_native_replay_binds_original_state(self):
        rt = SimpleNamespace(W0=torch.zeros(2, 3), M=torch.zeros(1, 3, 3), P=torch.eye(3)[None])
        native = dict(captures={"something": 1}, receipt=dict(entry_weight_sha256="wrong"))
        with self.assertRaises(TechnicalDecisionError):
            _native_replay(rt, native, torch.zeros(2, 3), Path("unused"))


if __name__ == "__main__":
    unittest.main()
