"""CPU-only tiny fixtures; no actual Llama or scientific PASS is assigned."""
from contextlib import contextmanager
from copy import deepcopy
import tempfile
from types import SimpleNamespace
import unittest

import torch

from project.run_scripts.single_layer_edit_preserving_correction.alltoken import FullTokenCache
from .decision import (DecisionError, DecisionOracle, EndpointBinding, FactorArchive,
    acceptance, margins_from_logits, no_direction_reason, risk_from_margins, risk_tolerance)
from .history import (StreamingHistoryOracle, active_history, receive_all, registry_status,
                       safety_slack)


class MemoryFactors:
    def __init__(self):
        self.members, self.values, self.sealed = [], [], False

    def put(self, index, a, metadata):
        assert index == len(self.values) and not self.sealed
        self.values.append((a.detach().cpu().clone(), deepcopy(metadata)))
        item = dict(index=index, metadata=deepcopy(metadata))
        self.members.append(item)
        return item

    def seal(self, *, expected_count):
        assert len(self.values) == expected_count
        self.sealed = True

    def get(self, i):
        assert self.sealed
        return self.values[i]


class Block:
    def __call__(self, h, **kwargs):
        # Causal cross-position dependence ensures an exposed last-position
        # pair has nonzero activation factors at earlier valid input tokens.
        count = torch.arange(1, h.shape[1] + 1, dtype=h.dtype, device=h.device)[None, :, None]
        return (h + torch.tanh(h.cumsum(1) / count),)


class FakeOracle:
    """Explicit tiny CPU nonlinear suffix fixture, not model validation."""
    def __init__(self, count=640, history=False):
        self.device = torch.device("cpu")
        self.shape = (3, 4)
        head = torch.nn.Linear(3, 5, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[.3, .2, -.1], [-.2, .4, .1], [.1, -.1, .5],
                                            [.3, -.2, .2], [-.3, -.1, -.2]]))
        head.weight.requires_grad_(False)
        self.model = SimpleNamespace(lm_head=head, config=SimpleNamespace(vocab_size=5))
        self.decoder = SimpleNamespace(layers=[None] * 5 + [Block()], norm=lambda x: x)
        self.store = SimpleNamespace(receipt={"manifest_sha256": "a" * 64})
        self.caches, self._capsules = [], []
        for index in range(count):
            length = 4 if history else 129
            ids = torch.arange(length).remainder(5)[None]
            packed = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                          position_ids=torch.arange(length)[None])
            k = torch.tensor([.1, .2, -.1, .3]).repeat(1, length, 1)
            # Small doc difference makes the full-bank weighting test nontrivial.
            k = k + (index % 7) * .01
            r = torch.tensor([.1, -.2, .2]).repeat(1, length, 1)
            self.caches.append(FullTokenCache(packed, k, r, dict(index=index, fixture=True)))
            self._capsules.append(dict(role="R512" if index < 512 else "Dev128",
                ordinal=index if index < 512 else index - 512, source_row_id=f"doc-{index}",
                y0=[4], score_positions=[128], actual_length=1, tf_input_ids=ids[0].tolist()))
        self.parameter = torch.zeros(self.shape)
        self._guard_value = self._cache_guard()

    def _cache_guard(self):
        return tuple((c.keys.data_ptr(), c.keys._version, c.residual.data_ptr(), c.residual._version)
                     for c in self.caches)

    def _guard(self):
        if self._cache_guard() != self._guard_value:
            raise DecisionError("FAKE_CACHE_MUTATION")

    def _on_device(self, packed):
        return packed

    def _teacher(self, index):
        raise AssertionError("DECISION_MUST_NOT_READ_FULL_TEACHER")

    def kl(self, *args, **kwargs):
        raise AssertionError("DECISION_MUST_NOT_CALL_KL")

    def _sync(self):
        pass

    def _args(self, hidden, packed):
        return {}

    def _head(self, hidden, positions):
        return self.model.lm_head(hidden[0, positions])

    @contextmanager
    def physical_weight(self, weight, gradient=False):
        old = self.parameter
        self.parameter = weight.detach().clone().requires_grad_(gradient)
        try:
            yield self.parameter
        finally:
            self.parameter = old

    def _physical_hidden(self, index):
        cache = self.caches[index]
        h = cache.residual + torch.nn.functional.linear(cache.keys, self.parameter)
        return self.decoder.layers[5](h)[0]


def endpoint(weight=None, identity="native"):
    if weight is None:
        weight = torch.zeros((3, 4), dtype=torch.float32)
    return EndpointBinding(weight, "cpu", endpoint_id=identity, source_identity={"fixture": True})


def event(i, subject="s", relation="r", target="new"):
    return dict(case_id=i, requested_rewrite=dict(subject=subject, relation_id=relation,
                                                  target_new={"str": target}, prompt="{}"))


class DecisionTests(unittest.TestCase):
    def test_margin_exact_ties(self):
        out = margins_from_logits(torch.tensor([[2., 2., 0.], [1., 0., 1.]]), [1, 0])
        self.assertEqual(out["margins"], [0., 0.])
        self.assertEqual(out["competitors"], [0, 2])
        self.assertEqual(out["predictions"], [0, 0])
        self.assertEqual(out["correct"], [False, True])

    def test_nonfinite_is_technical(self):
        with self.assertRaises(DecisionError):
            margins_from_logits(torch.tensor([[float("nan"), 1.]]), [0])
        with self.assertRaises(DecisionError):
            risk_from_margins([-1e308])

    def test_endpoint_owned_alias_and_mutation(self):
        original = torch.zeros(3, 4)
        e = endpoint(original)
        original.add_(1)
        e.guard()
        self.assertEqual(float(e.cpu.sum()), 0.)
        e.device_weight.add_(1)
        with self.assertRaises(DecisionError):
            e.guard()

    def test_empty_or_stale_closed(self):
        e = endpoint()
        e.close()
        with self.assertRaises(DecisionError):
            e.guard()

    def test_risk_branches(self):
        self.assertEqual(risk_from_margins([-2, 1]), 2)
        self.assertEqual(no_direction_reason(0, 0, 0), "ZERO_RISK_NO_OP")
        self.assertEqual(no_direction_reason(0, 0, 1), "TIE_ONLY_NO_DIRECTION")
        self.assertEqual(no_direction_reason(1e-12, 0, 1), "BELOW_RISK_RESOLUTION")
        self.assertEqual(no_direction_reason(1, 0, 1, 1., 1e-13), "NO_PROJECTED_DIRECTION")
        self.assertIsNone(no_direction_reason(1, 0, 1, 1., .1))

    def test_fullbank_gradient_factors_and_fd(self):
        oracle = DecisionOracle(FakeOracle())
        n = endpoint()
        factors = MemoryFactors()
        obs = oracle.scan(n, gradient=True, factor_sink=factors)
        self.assertEqual(obs.coverage["documents"], 512)
        self.assertEqual(obs.coverage["positions"], 512)
        self.assertEqual(obs.work["backward_documents"], 512)
        self.assertEqual(obs.work["full_teacher_reads"], 0)
        self.assertEqual(len(factors.values), 512)
        self.assertEqual(factors.values[0][0].shape, (129, 3))
        self.assertGreater(float(factors.values[0][0][0].norm()), 0.)
        direction = torch.ones(3, 4) / 5
        jac = oracle.jacobian(obs, [direction])
        expected = sum(-2 * max(0., -r["mu"]) * float(jac[i, 0]) / 512
                       for i, r in enumerate(obs.rows))
        ad = float((obs.gradient * direction.double()).sum())
        self.assertAlmostEqual(expected, ad, places=6)
        h = .001
        p = oracle.scan(endpoint(n.cpu + h * direction, "plus"))
        m = oracle.scan(endpoint(n.cpu - h * direction, "minus"))
        self.assertAlmostEqual((p.phi_reference - m.phi_reference) / (2 * h), ad, places=4)
        with self.assertRaises(DecisionError):
            oracle.scan(n, gradient=True, factor_sink=MemoryFactors())

    def test_no_gradient_dev_and_subset(self):
        oracle = DecisionOracle(FakeOracle())
        with self.assertRaises(DecisionError):
            oracle.scan(endpoint(), role="Dev128", gradient=True, factor_sink=MemoryFactors())
        with self.assertRaises(TypeError):
            oracle.scan(endpoint(), indices=[0])
        with self.assertRaises(DecisionError):
            oracle.scan(endpoint(), role="Dev128")
        self.assertEqual(oracle.scan(endpoint(), role="Dev128", selection_seal="sealed").coverage["documents"], 128)

    def test_prefix_mutation_fails(self):
        fake = FakeOracle()
        oracle = DecisionOracle(fake)
        fake.caches[0].keys.add_(1)
        with self.assertRaises(DecisionError):
            oracle.scan(endpoint())

    def test_acceptance_no_new_flip_within_unsafe_document(self):
        oracle = DecisionOracle(FakeOracle())
        n = oracle.scan(endpoint())
        c = deepcopy(n)
        c.endpoint_identity = "candidate"
        c.phi_reference = n.phi_reference / 2
        # A document may have another unsafe token. A newly flipped native-safe
        # position is still forbidden even with globally improved risk/count.
        n.rows[0]["correct"][0] = True
        c.rows[0]["correct"][0] = False
        result = acceptance(n, c, current_guard_pass=True, invariant_pass=True, history_guard_pass=True)
        self.assertIn("REFERENCE_NEW_FLIP", result["reasons"])

    def test_acceptance_all512_and_ids_required(self):
        oracle = DecisionOracle(FakeOracle())
        n = oracle.scan(endpoint())
        c = deepcopy(n)
        c.coverage["complete"] = False
        with self.assertRaises(DecisionError):
            acceptance(n, c, current_guard_pass=True, invariant_pass=True, history_guard_pass=True)
        c = deepcopy(n)
        c.rows[0]["labels"] = [3]
        with self.assertRaises(DecisionError):
            acceptance(n, c, current_guard_pass=True, invariant_pass=True, history_guard_pass=True)

    def test_factor_archive_complete_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = FactorArchive(folder + "/factors", {"endpoint": "x"})
            archive.put(0, torch.ones(2, 3), {"index": 0})
            with self.assertRaises(DecisionError):
                archive.get(0)
            archive.seal(expected_count=1)
            self.assertEqual(tuple(archive.get(0)[0].shape), (2, 3))
            archive.members[0]["metadata"]["index"] = 1
            with self.assertRaises(DecisionError):
                archive.get(0)

    def test_bounded_physical_pair_diagnostic(self):
        oracle = DecisionOracle(FakeOracle())
        receipt = oracle.check_pair_document(endpoint(), 0, torch.ones(3, 4), [.001, .0005])
        self.assertFalse(receipt["full_bank_pass"])
        self.assertLess(receipt["gradient_relative_l2"], 1e-6)
        self.assertEqual(len(receipt["FD"]), 2)

    def test_technical_four_panel_not_fullbank(self):
        oracle = DecisionOracle(FakeOracle())
        report = oracle.check_panel(endpoint(), [0, 13, 128, 511])
        self.assertFalse(report["full_bank_pass"])
        self.assertEqual(len(report["rows"]), 4)
        with self.assertRaises(DecisionError):
            oracle.check_panel(endpoint(), [0, 1])


class HistoryTests(unittest.TestCase):
    def test_registry_all_events_overwrite_and_no_sampling(self):
        ledger = receive_all([], [event(i, subject=f"s{i}") for i in range(90)])
        self.assertEqual(len(active_history(ledger, [])), 90)
        ledger = receive_all(ledger, [event(100, subject="s0"), event(101, subject="s1", target="")])
        active = active_history(ledger, [event(200, subject="s2")])
        self.assertEqual(len(active), 89)
        self.assertIn(100, [r["case_id"] for r in active])
        self.assertIn(1, [r["case_id"] for r in active])
        self.assertNotIn(0, [r["case_id"] for r in active])
        self.assertNotIn(2, [r["case_id"] for r in active])
        self.assertEqual(registry_status(ledger)[-1]["status"], "INVALID_TARGET")
        with self.assertRaises(DecisionError):
            receive_all(ledger, [event(100)])

    @staticmethod
    def score(nll=.2, strict=True, margins=(.4,), label=1):
        return dict(nll=nll, strict=strict, margins=list(margins), positions=[0], labels=[label],
                    input_identity="same")

    def test_entry_anchor_not_native_and_tie_ids(self):
        anchor = dict(new=self.score(), old=self.score(.5, label=2))
        native = dict(new=self.score(.3), old=self.score(.6, label=2))
        out = safety_slack(anchor, native)
        self.assertFalse(out["guard_pass"])
        self.assertIn("ENTRY_DESIRED_NLL", out["reasons"])
        candidate = dict(new=self.score(.2, False, (0.,)), old=self.score(.5, label=2))
        out = safety_slack(anchor, candidate)
        self.assertEqual(out["slack"], 0.)
        self.assertFalse(out["guard_pass"])
        self.assertIn("ENTRY_STRICT_ID_LOST", out["reasons"])

    def test_missing_old_not_fabricated(self):
        out = safety_slack(dict(new=self.score()), dict(new=self.score()))
        self.assertFalse(out["old_branch_available"])
        self.assertNotIn("preference", out["component_slacks"])

    @staticmethod
    def factory(record):
        oracle = FakeOracle(2, history=True)
        rows = [dict(case_id=record["case_id"], kind="canonical", branch="new", cache=0,
                     positions=[2, 3], labels=[1, 2]),
                dict(case_id=record["case_id"], kind="canonical", branch="old", cache=1,
                     positions=[3], labels=[4])]
        return oracle, rows

    def test_empty_history_na(self):
        history = StreamingHistoryOracle([], lambda r: self.fail("EMPTY_HISTORY_FACTORY_CALLED"))
        obs = history.evaluate(endpoint())
        self.assertTrue(obs.coverage["past_empty"])
        self.assertEqual(obs.phi_history, 0.)
        self.assertTrue(obs.guard_pass)

    def test_streaming_history_gradient_and_jacobian(self):
        records = receive_all([], [event(0), event(1, subject="other")])
        history = StreamingHistoryOracle(records, self.factory)
        entry = history.evaluate(endpoint())
        e = endpoint(torch.ones(3, 4) * .2, "native")
        factors = MemoryFactors()
        native = history.evaluate(e, entry_anchor=entry.anchor(), gradient=True, factor_sink=factors)
        self.assertEqual(native.coverage["active_requests"], 2)
        self.assertEqual(native.work["peak_resident_requests"], 1)
        self.assertEqual(native.work["peak_resident_paths"], 2)
        self.assertEqual(native.work["backward_requests"], 2)
        jac = history.jacobian(native, [torch.ones(3, 4)])
        expected = sum(-2 * max(0., -r["slack"]) * float(jac[i, 0]) / 2
                       for i, r in enumerate(native.rows))
        self.assertAlmostEqual(expected, float(native.gradient.sum()), places=6)
        with self.assertRaises(DecisionError):
            history.evaluate(e, entry_anchor=entry.anchor(), gradient=True, factor_sink=MemoryFactors())

    def test_history_wrong_anchor_or_observer_path_fails(self):
        records = receive_all([], [event(0)])
        history = StreamingHistoryOracle(records, self.factory)
        obs = history.evaluate(endpoint())
        anchor = obs.anchor()
        anchor["history_identity"] = "wrong"
        with self.assertRaises(DecisionError):
            history.evaluate(endpoint(), entry_anchor=anchor)
        def bad(record):
            oracle, rows = self.factory(record)
            rows[0]["kind"] = "paraphrase"
            return oracle, rows
        with self.assertRaises(DecisionError):
            StreamingHistoryOracle(records, bad).evaluate(endpoint())


if __name__ == "__main__":
    unittest.main()
