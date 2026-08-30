from __future__ import annotations

import math
import unittest

import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from project.run_scripts.barrier_usefulness.contracts import BarrierLock
from project.run_scripts.barrier_usefulness.geometry import build_axes
from project.run_scripts.barrier_usefulness.metrics import CandidateScore, reference_barrier, select_ctrl, teacher_kl
from project.run_scripts.barrier_usefulness.snapshots import EndpointSnapshot
from project.run_scripts.barrier_usefulness.firewall import authorize_gate_open, seal_selector_lock


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))


class BarrierCoreTests(unittest.TestCase):
    def test_actual_cross_equality_and_shell(self) -> None:
        torch.manual_seed(7)
        delta = torch.randn(5, 7)
        constraints = torch.randn(7, 3)
        covariance = torch.randn(7, 7)
        covariance = covariance @ covariance.T + torch.eye(7)
        axes = build_axes(
            base_delta=delta,
            constraints=constraints,
            matvec=lambda value: covariance @ value,
            lock=BarrierLock(), model="m", method="x", case_id="1", count=2, projector=None,
        )
        for axis in axes:
            for sign in (-1, 1):
                correction = axis.correction(sign)
                self.assertLess(float((correction @ constraints).norm() / (correction.norm() * constraints.norm())), BarrierLock().fp32_relative_tolerance)
                total = delta + correction
                q_base = (delta * (delta @ covariance)).sum()
                q_total = (total * (total @ covariance)).sum()
                self.assertLess(abs(float(q_total / q_base) - 1.05), 2e-5)

    def test_exact_snapshot_copy_not_subtractive_rollback(self) -> None:
        model = TinyModel()
        snapshot = EndpointSnapshot.capture(model, ["weight"])
        correction = torch.ones_like(model.weight)
        changed = snapshot.apply_correction(model, "weight", correction)
        self.assertNotEqual(changed, snapshot.member_hashes["weight"])
        receipt = snapshot.restore_after_candidate(model)
        self.assertTrue(receipt["exact"])
        self.assertTrue(torch.equal(model.weight.cpu(), snapshot.tensors["weight"]))

    def test_barrier_violation_is_infinite(self) -> None:
        reference = torch.tensor([[3.0, 1.0, 0.0], [2.0, 0.0, 1.0]])
        candidate = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        labels = torch.tensor([0, 0])
        row = reference_barrier(reference, candidate, labels, numerical_floor=1e-5)
        self.assertFalse(row["finite"])
        self.assertTrue(math.isinf(row["score"]))

    def test_barrier_and_kl_selectors_are_separate_and_lexical(self) -> None:
        rows = [
            CandidateScore("axis-00-minus", 0.2, True, 0.9),
            CandidateScore("axis-00-plus", 0.1, True, 1.0),
            CandidateScore("axis-01-minus", 0.3, True, 0.1),
        ]
        selected = select_ctrl(rows, "axis-00-minus")
        self.assertEqual(selected["barrier-safe"], "axis-00-plus")
        self.assertEqual(selected["kl-only"], "axis-01-minus")
        self.assertEqual(selected["barrier-adverse-finite"], "axis-01-minus")

    def test_teacher_kl_identity(self) -> None:
        logits = torch.randn(32, 17)
        row = teacher_kl(logits, logits.clone(), 0.875)
        self.assertEqual(row["count"], 32)
        self.assertAlmostEqual(row["cvar_0_875"], 0.0, places=7)

    def test_gate_root_fail_closes_without_selector_lock(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctrl = root / "ctrl"
            ctrl.mkdir()
            with self.assertRaises(FileNotFoundError):
                authorize_gate_open(ctrl_root=ctrl, gate_root=root / "gate", selector_lock=root / "missing.json", expected_identity="x")

    def test_selector_lock_precedes_disjoint_gate_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctrl = root / "ctrl"
            ctrl.mkdir()
            lock = seal_selector_lock(root / "selector" / "lock.json", {
                "edited_gate_access_count": 0,
                "candidate_bank_hash": "c",
                "selected_candidate_ids_hash": "s",
            })
            opened = authorize_gate_open(
                ctrl_root=ctrl, gate_root=root / "gate", selector_lock=root / "selector" / "lock.json",
                expected_identity=lock["selector_lock_identity"],
            )
            self.assertEqual(opened["status"], "GATE_OPEN_AUTHORIZED")


if __name__ == "__main__":
    unittest.main()
