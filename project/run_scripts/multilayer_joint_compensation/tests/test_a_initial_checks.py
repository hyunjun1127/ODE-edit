"""CPU nonlinear full-sequence tests for the bounded actual-model gate."""
import unittest
import json
from unittest.mock import patch

import torch

from project.run_scripts.multilayer_joint_compensation.observations import JointView, pack, teacher
from project.run_scripts.multilayer_joint_compensation.evaluation import panel
from project.run_scripts.multilayer_joint_compensation.track_a import initial_checks
from project.run_scripts.l4_two_memory_conflict_routing.geometry import Projector


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(31)
        self.embed = torch.nn.Embedding(7, 4)
        self.l4 = torch.nn.Linear(4, 4, bias=False)
        self.l8 = torch.nn.Linear(4, 7, bias=False)
        self.eval().requires_grad_(False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        x = self.embed(input_ids) * attention_mask[..., None]
        # Earlier tokens affect the selected prediction logits, so testing the
        # selected output cannot accidentally be a subject-only weight hook.
        x = x.cumsum(1) / torch.arange(1, x.shape[1] + 1, device=x.device)[None, :, None]
        return self.l8(torch.tanh(self.l4(x)))


def make_fixture():
    view = JointView(Toy(), ("l4.weight", "l8.weight"))
    rows = [dict(identity=f"first-request-context-{i}", input_ids=[1, 2, 3, 4][:i % 4 + 1],
                 positions=[i % 4], target_ids=[(i + 2) % 7], token_mean_weight=1., context_weight=1 / 6)
            for i in range(6)]
    batches = tuple(pack(view, rows, 0, 2))
    saved = teacher(view, view.entry, rows, 0, 2)
    bounded = panel(view, [dict(rows[0], context_weight=1.)], saved, 0, "current", 1)
    return view, batches, bounded


class InitialChecksTests(unittest.TestCase):
    def test_joint_full_forward_fd_ggn_and_exact_restore(self):
        view, batches, bounded = make_fixture()
        receipt = initial_checks.run_initial_checks(view, batches, bounded)
        self.assertEqual(receipt["status"], "INITIAL_CORRECTNESS_PASS")
        self.assertTrue(receipt["full_live_pointer_version_bytes_restored"])
        self.assertEqual(receipt["full_parameter_entry_root"], receipt["full_parameter_exit_root"])
        self.assertEqual(len(receipt["context_identities"]), 6)
        self.assertTrue(all(row["passed"] for row in receipt["teacher_checks"]))
        self.assertEqual(len(receipt["directional_checks"]), 9)
        self.assertTrue(all(row["passed"] for row in receipt["directional_checks"]))
        self.assertTrue(all(row["byte_equal"] for row in receipt["virtual_materialized"]))
        self.assertEqual(receipt["ggn_compute_counts"]["ggn_matvecs"], 4)
        # Nonzero cross action here proves the shared operator does not drop
        # the functional cross layer blocks. Actual zero is not a failure.
        self.assertTrue(all(row["cross_action_frobenius"] > 0 for row in receipt["ggn"]["cross_actions"]))
        for row in receipt["direction_layers"]:
            self.assertAlmostEqual(row["weight_frobenius"], row["direction_frobenius"], places=6)

    def test_forced_materialized_mismatch_keeps_full_failure_receipt(self):
        view, batches, bounded = make_fixture()
        original = initial_checks._direct_logits
        with patch.object(initial_checks, "_direct_logits", lambda *args: original(*args) + .01):
            with self.assertRaises(initial_checks.InitialCheckBoundary) as caught:
                initial_checks.run_initial_checks(view, batches, bounded)
        receipt = caught.exception.receipt
        self.assertTrue(any("VIRTUAL_MATERIALIZED" in value for value in receipt["failures"]))
        self.assertTrue(receipt["full_live_pointer_version_bytes_restored"])
        self.assertEqual(len(receipt["directional_checks"]), 9)
        self.assertIn("psd", receipt["ggn"])

    def test_forced_fd_mismatch_is_not_tuned_or_retried(self):
        view, batches, bounded = make_fixture()
        original = torch.func.jvp

        def incorrect(*args, **kwargs):
            output, tangent = original(*args, **kwargs)
            return output, -tangent

        with patch("torch.func.jvp", incorrect):
            with self.assertRaises(initial_checks.InitialCheckBoundary) as caught:
                initial_checks.run_initial_checks(view, batches, bounded)
        receipt = caught.exception.receipt
        self.assertTrue(any("JVP_FD_MISMATCH" in value for value in receipt["failures"]))
        self.assertEqual(receipt["fd_epsilons"], [2.**-7, 2.**-8, 2.**-9])
        self.assertTrue(receipt["full_live_pointer_version_bytes_restored"])

    def test_zero_projected_direction_is_typed_unresolved(self):
        view, batches, bounded = make_fixture()
        zero = Projector(torch.empty(4, 0, dtype=torch.float64), False, 4)
        with self.assertRaises(initial_checks.InitialCheckBoundary) as caught:
            initial_checks.run_initial_checks(view, batches, bounded, projectors=(zero, zero))
        receipt = caught.exception.receipt
        self.assertTrue(any("JVP_NUMERICALLY_UNRESOLVED" in value for value in receipt["failures"]))
        self.assertEqual(receipt["direction_layers"][0]["direction_frobenius"], 0.)
        self.assertTrue(receipt["full_live_pointer_version_bytes_restored"])
        self.assertTrue(all(row["cross_action_frobenius"] == 0 for row in receipt["ggn"]["cross_actions"]))

    def test_projected_directions_are_deterministic_complete_space(self):
        view, _, _ = make_fixture()
        vector = torch.tensor([[1.], [0.], [0.], [0.]], dtype=torch.float64)
        p = Projector(vector, True, 4)
        first = initial_checks._directions(view, initial_checks.SEED, (p, p))
        second = initial_checks._directions(view, initial_checks.SEED, (p, p))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertTrue(all(torch.count_nonzero(a[:, 0]) == 0 for a in first))

    def test_wrong_we_teacher_and_forward_exception_are_recorded(self):
        view, batches, bounded = make_fixture()
        bounded.batches[0].teacher_logp = bounded.batches[0].teacher_logp + .1
        with self.assertRaises(initial_checks.InitialCheckBoundary) as caught:
            initial_checks.run_initial_checks(view, batches, bounded)
        self.assertIn("WE_TEACHER_MISMATCH:0", caught.exception.receipt["failures"])
        json.dumps(caught.exception.receipt, allow_nan=False)

        view, batches, bounded = make_fixture()
        with patch.object(initial_checks, "_direct_logits", side_effect=RuntimeError("forced forward error")):
            with self.assertRaises(initial_checks.InitialCheckBoundary) as caught:
                initial_checks.run_initial_checks(view, batches, bounded)
        receipt = caught.exception.receipt
        self.assertEqual(receipt["original_exception"]["phase"], "virtual_materialized")
        self.assertTrue(receipt["full_live_pointer_version_bytes_restored"])
        json.dumps(receipt, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
