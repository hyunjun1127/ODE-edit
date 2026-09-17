"""CPU mock integration of the real selector; no model/GPU/checkpoint payload.

The runtime is a pair of tiny CPU tensors. Evidence writers and RNG capture are
patched to memory. These fixtures certify routing/acceptance/rollback boundaries,
not actual Llama derivatives, efficacy, or GPU continuation.
"""
from __future__ import annotations

import copy
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import engine
from .qp import QPError, solve_repair_qp


class MemoryPath:
    def __init__(self, text="memory-only"):
        self.text = text

    def __truediv__(self, name):
        return MemoryPath(self.text + "/" + name)

    def mkdir(self):
        return None

    def __str__(self):
        return self.text


def panel(E=0.02, strict=(1, 2), preference=(1,)):
    return {"E": E, "strict_ids": list(strict), "preference_ids": list(preference),
            "denominator": 2, "rows": []}


def scored(E=0.02, B=0.3, strict=(1, 2), preference=(1,), past=None):
    return {"current": panel(E, strict, preference), "past": past, "base": {"B": B}}


class FakeResponse:
    def __init__(self, runtime, observations):
        self.rt = runtime
        self.observations = observations
        self.index = 0
        self.panel_calls = []
        self.base_calls = []
        self.response_calls = []

    def _row(self):
        if not self.observations:
            return scored(B=0.5)
        return self.observations[min(self.index, len(self.observations) - 1)]

    def panel(self, records, gradient=False):
        self.panel_calls.append((records, gradient))
        key = "past" if records == ["past"] else "current"
        row = copy.deepcopy(self._row()[key])
        grad = torch.tensor([0.0, 1.0]) if gradient else None
        return row, grad

    def base(self, gradient=False):
        self.base_calls.append(gradient)
        row = copy.deepcopy(self._row()["base"])
        if not gradient:
            self.index += 1
        return row, torch.tensor([-1.0, 0.0]) if gradient else None

    def response(self, Q, current, past, cur, old, include_guards):
        self.response_calls.append({"guards": include_guards, "m": len(Q)})
        return {"H": torch.eye(len(Q), dtype=torch.float64),
                "A": torch.empty((0, len(Q)), dtype=torch.float64),
                "s": torch.empty(0, dtype=torch.float64)}


class FakeRuntime:
    def __init__(self, observations=()):
        self.W = {4: torch.tensor([4.0, 4.0]), 8: torch.tensor([0.25, -0.5])}
        self.M4 = torch.zeros(2, 2)
        self.rng = {"fake": "same-sealed-anchor-rng"}
        self.apply_history = []
        self.response = FakeResponse(self, list(observations))

    def state(self):
        return {"W": {str(k): engine.tensor_sha(v) for k, v in self.W.items()},
                "M4": engine.tensor_sha(self.M4), "rng": copy.deepcopy(self.rng),
                "contexts": "constant-test-context", "P4": "constant-test-projector"}

    def observe(self, fn):
        before = self.state()
        result = fn()
        assert before == self.state(), "MOCK_OBSERVER_STATE_MUTATION"
        return result

    def apply8(self, weight, rng):
        assert torch.isfinite(weight).all()
        self.W[8].copy_(weight)
        self.rng = copy.deepcopy(rng)
        self.apply_history.append(weight.clone())


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.writes = {}
        self.tensor_writes = {}
        self.context = ExitStack()
        self.addCleanup(self.context.close)

        def save(path, value):
            key = str(path)
            self.assertNotIn(key, self.writes, "create-once evidence overwritten")
            self.writes[key] = copy.deepcopy(value)
            return {"path": key, "sha256": "memory-receipt", "bytes": 0}

        def tensor_save(path, value):
            self.tensor_writes[str(path)] = copy.deepcopy(value)
            return {"path": str(path), "sha256": "memory-tensor-receipt", "bytes": 0}

        self.context.enter_context(patch.object(engine, "save", side_effect=save))
        self.context.enter_context(patch.object(engine, "tensor_save", side_effect=tensor_save))
        self.context.enter_context(patch.object(engine, "identity", side_effect=lambda p: {
            "path": str(p), "sha256": "memory-receipt", "bytes": 0}))
        self.context.enter_context(patch.object(engine, "capture_rng", return_value={"fake": "same-sealed-anchor-rng"}))

    def model(self, rt, arm="R-QP", B=0.5, past=None):
        return {"anchor": {"current": panel(), "past": past, "base": {"B": B}},
                "anchor8": rt.W[8].clone(), "Q": [torch.tensor([1.0, 0.0])],
                "response": {"H": torch.ones((1, 1), dtype=torch.float64),
                             "A": torch.empty((0, 1), dtype=torch.float64),
                             "s": torch.empty(0, dtype=torch.float64)},
                "dots": np.array([-1.0]), "rng": copy.deepcopy(rt.rng),
                "state": rt.state(), "arm": arm}

    def select(self, rt, model, past=False):
        return engine.select(rt, ["current"], ["past"] if past else [], model, MemoryPath())

    def test_quality_exact_id_subset_not_equal_success_count(self):
        anchor = scored()
        bad = scored(strict=(2, 3), preference=(2,))
        reasons = engine.quality(anchor, bad)
        self.assertIn("current_strict_ids", reasons)
        self.assertIn("current_preference_ids", reasons)

    def test_quality_no_point05_plateau(self):
        anchor = scored(E=0.001)
        self.assertIn("current_MEAN", engine.quality(anchor, scored(E=0.01)))
        self.assertEqual(engine.quality(anchor, scored(E=0.00105)), [])

    def test_past_separate_mean_and_success_sets(self):
        anchor = scored(past=panel(E=0.4, strict=(8,), preference=(9,)))
        candidate = scored(E=0.0, past=panel(E=0.4002, strict=(7,), preference=(7,)))
        self.assertEqual(set(engine.quality(anchor, candidate)),
                         {"past_MEAN", "past_strict_ids", "past_preference_ids"})

    def test_quality_nan_is_technical_not_infeasible(self):
        with self.assertRaisesRegex(AssertionError, "NONFINITE"):
            engine.quality(scored(), scored(E=float("nan")))

    def test_first_acceptable_not_best_of_all(self):
        rt = FakeRuntime([scored(E=0.0202), scored(B=0.3), scored(B=0.01)])
        model = self.model(rt)
        with patch("project.run_scripts.l4_preserving_repair.qp.solve_repair_qp", wraps=solve_repair_qp) as qp:
            selection, weight = self.select(rt, model)
        self.assertEqual(selection["selected"], "probe-01")
        self.assertEqual(selection["reason"], "FIRST_ACCEPTABLE")
        self.assertEqual(len(selection["probes"]), 2)
        self.assertEqual(qp.call_count, 2)
        self.assertEqual(rt.response.index, 2)
        self.assertTrue(torch.equal(rt.W[8], weight))
        self.assertTrue(torch.equal(rt.W[4], torch.tensor([4.0, 4.0])))
        self.assertEqual(torch.count_nonzero(rt.M4), 0)

    def test_six_rejections_restore_exact_anchor_and_halve_radius(self):
        rt = FakeRuntime([scored(E=0.5)] * 8)
        model = self.model(rt)
        selection, weight = self.select(rt, model)
        self.assertEqual(selection["selected"], "WN")
        self.assertEqual(len(selection["probes"]), 6)
        self.assertEqual([p["radius"] for p in selection["probes"]], [1, 0.5, 0.25, 0.125, 0.0625, 0.03125])
        self.assertTrue(torch.equal(weight, model["anchor8"]))
        self.assertEqual(rt.state(), model["state"])
        self.assertEqual(rt.response.index, 6)
        # Every scored attempt is followed by WN restoration, then final WN.
        self.assertEqual(len(rt.apply_history), 13)
        for index in range(1, 12, 2):
            self.assertTrue(torch.equal(rt.apply_history[index], model["anchor8"]))

    def test_gd_still_rejects_actual_rewrite_quality(self):
        rt = FakeRuntime([scored(strict=(1, 3))] * 6)
        model = self.model(rt, arm="R-GD")
        selection, _ = self.select(rt, model)
        self.assertEqual(selection["selected"], "WN")
        self.assertTrue(all("current_strict_ids" in p["reasons"] for p in selection["probes"]))
        self.assertEqual(rt.state(), model["state"])

    def test_small_actual_gain_and_bad_agreement_are_rejections(self):
        rt = FakeRuntime([scored(B=0.5), scored(B=0.49999), scored(B=0.3)])
        selection, _ = self.select(rt, self.model(rt))
        self.assertIn("ACTUAL_SMALL_OR_NEGATIVE", selection["probes"][0]["reasons"])
        self.assertIn("AGREEMENT", selection["probes"][1]["reasons"])
        self.assertEqual(selection["selected"], "probe-02")

    def test_base_small_no_solver_no_forward_preserves_prior_l8(self):
        rt = FakeRuntime()
        model = self.model(rt, B=1e-6)
        with patch("project.run_scripts.l4_preserving_repair.qp.solve_repair_qp") as qp:
            selection, weight = self.select(rt, model)
        qp.assert_not_called()
        self.assertEqual(selection["reason"], "BASE_SMALL")
        self.assertEqual(rt.response.base_calls, [])
        self.assertTrue(torch.equal(weight, torch.tensor([0.25, -0.5])))

    def test_qp_technical_failure_propagates_without_normal_off(self):
        rt = FakeRuntime()
        model = self.model(rt)
        with patch("project.run_scripts.l4_preserving_repair.qp.solve_repair_qp", side_effect=QPError("QP_KKT_UNRESOLVED")):
            with self.assertRaisesRegex(QPError, "KKT_UNRESOLVED"):
                self.select(rt, model)
        self.assertNotIn("memory-only/selection.json", self.writes)
        self.assertEqual(rt.state(), model["state"])

    def test_zero_guard_proposal_does_not_score_six_useless_endpoints(self):
        rt = FakeRuntime()
        model = self.model(rt)
        model["response"]["A"] = torch.ones((1, 1), dtype=torch.float64)
        model["response"]["s"] = torch.zeros(1, dtype=torch.float64)
        selection, _ = self.select(rt, model)
        self.assertEqual(selection["selected"], "WN")
        self.assertEqual(len(selection["probes"]), 6)
        self.assertEqual(rt.response.base_calls, [])
        self.assertTrue(all("PREDICTED_SMALL" in p["reasons"] for p in selection["probes"]))

    def test_build_qp_tensor_response_and_no_disk_checkpoint(self):
        rt = FakeRuntime()
        model = engine.build(rt, ["current"], [], "R-QP", MemoryPath())
        self.assertEqual(len(model["Q"]), 2)
        self.assertTrue(all(q.dtype == torch.float32 and q.device.type == "cpu" for q in model["Q"]))
        self.assertIsInstance(model["response"]["H"], torch.Tensor)
        self.assertEqual(rt.response.response_calls, [{"guards": True, "m": 2}])
        self.assertEqual(set(self.tensor_writes), {"memory-only/directions.pt"})
        self.assertNotIn("weight", self.tensor_writes["memory-only/directions.pt"])

    def test_build_gd_has_no_dummy_current_gradient(self):
        rt = FakeRuntime()
        model = engine.build(rt, ["current"], [], "R-GD", MemoryPath())
        self.assertEqual(len(model["Q"]), 1)
        self.assertEqual(rt.response.panel_calls, [(["current"], False)])
        self.assertEqual(rt.response.response_calls, [{"guards": False, "m": 1}])

    def test_materialize_zero_copies_actual_prior_l8(self):
        anchor = torch.tensor([0.25, -0.5])
        output = engine.materialize(anchor, [torch.tensor([1.0, 2.0])], [0.0])
        self.assertTrue(torch.equal(output, anchor))
        self.assertNotEqual(output.data_ptr(), anchor.data_ptr())

    def test_nonfinite_candidate_base_must_not_be_accepted(self):
        rt = FakeRuntime([scored(B=float("nan"))])
        with self.assertRaises((AssertionError, ValueError, RuntimeError)):
            self.select(rt, self.model(rt))
        self.assertNotIn("memory-only/selection.json", self.writes)

    def test_nonfinite_anchor_base_must_not_be_normal_off(self):
        rt = FakeRuntime()
        with self.assertRaises((AssertionError, ValueError, RuntimeError)):
            self.select(rt, self.model(rt, B=float("nan")))
        self.assertNotIn("memory-only/selection.json", self.writes)


if __name__ == "__main__":
    unittest.main()
