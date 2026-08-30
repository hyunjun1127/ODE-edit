from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from project.run_scripts.fixed_z_nonuniqueness.algebra import (
    candidate_action,
    generate_axes,
    rank_one_factor,
    shape_rank_rtol,
)
from project.run_scripts.fixed_z_nonuniqueness.contracts import NumericalLock
from project.run_scripts.fixed_z_nonuniqueness.forbidden_imports import scan
from project.run_scripts.fixed_z_nonuniqueness.padding import (
    semantic_last_columns,
    semantic_batch_identity,
    semantic_position_ids,
    target_continuation_columns,
)
from project.run_scripts.fixed_z_nonuniqueness.official import official_model_name_binding


class CoreTests(unittest.TestCase):
    def test_official_model_name_binding_restores_snapshot_identity(self) -> None:
        class Config:
            _name_or_path = "/exact/local/snapshot/revision"

        class Model:
            config = Config()

        model = Model()
        with official_model_name_binding(model, "Meta-Llama-3-8B-Instruct"):
            self.assertEqual(model.config._name_or_path, "Meta-Llama-3-8B-Instruct")
        self.assertEqual(model.config._name_or_path, "/exact/local/snapshot/revision")

    def test_left_padding_semantic_positions_and_target_slice(self) -> None:
        mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
        self.assertTrue(torch.equal(semantic_position_ids(mask), torch.tensor([[0, 0, 0, 1], [0, 0, 1, 2]])))
        self.assertTrue(torch.equal(semantic_last_columns(mask), torch.tensor([3, 3])))
        spans = target_continuation_columns(mask, [1, 2])
        self.assertTrue(torch.equal(spans[0], torch.tensor([3])))
        self.assertTrue(torch.equal(spans[1], torch.tensor([2, 3])))
        left = {
            "input_ids": torch.tensor([[0, 11, 12], [21, 22, 23]]),
            "attention_mask": torch.tensor([[0, 1, 1], [1, 1, 1]]),
        }
        right = {
            "input_ids": torch.tensor([[11, 12, 0], [21, 22, 23]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        }
        self.assertEqual(semantic_batch_identity(left), semantic_batch_identity(right))

    def test_tangent_equality_orientation_and_action_pair(self) -> None:
        torch.manual_seed(7)
        out_dim, in_dim = 7, 11
        base_u = torch.randn(out_dim)
        base_v = torch.randn(in_dim)
        delta = torch.outer(base_u, base_v)
        constraints = torch.randn(in_dim, 4)
        cov = torch.randn(in_dim, in_dim)
        cov = cov @ cov.T + 0.5 * torch.eye(in_dim)
        lock = NumericalLock()
        axes, info = generate_axes(
            base_delta=delta,
            constraints=constraints,
            covariance_matvec=lambda v: cov @ v,
            lock=lock,
            model_alias="fixture",
            method="alphaedit",
            case_id="1",
        )
        self.assertEqual(len(axes), 4)
        for axis in axes:
            correction = axis.correction(1)
            self.assertLess(float((correction @ constraints).norm() / correction.norm() / constraints.norm()), 2e-5)
            plus = candidate_action(axis, 1)
            minus = candidate_action(axis, -1)
            self.assertAlmostEqual(plus, minus, places=10)
            self.assertAlmostEqual(plus, info["base_action"] * 1.05, places=4)

    def test_projector_orientation(self) -> None:
        torch.manual_seed(3)
        d = 12
        q, _ = torch.linalg.qr(torch.randn(d, 6))
        projector = q @ q.T
        constraints = torch.randn(d, 3)
        delta = torch.outer(torch.randn(5), q[:, 0])
        cov = torch.eye(d)
        axes, _ = generate_axes(
            base_delta=delta,
            constraints=constraints,
            covariance_matvec=lambda v: cov @ v,
            lock=NumericalLock(),
            model_alias="fixture",
            method="alphaedit",
            case_id="2",
            projector=projector,
        )
        for axis in axes:
            self.assertLess(float((projector @ axis.input - axis.input).norm()), 2e-5)

    def test_forbidden_import_ast(self) -> None:
        package = Path(__file__).resolve().parents[1]
        self.assertEqual(scan(package), [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.py"
            path.write_text("from project.run_scripts.ode_bf import contracts\n")
            self.assertEqual(len(scan(Path(tmp))), 1)

    def test_constraint_orientation_is_input_by_events(self) -> None:
        d_in = 13
        edit = torch.randn(d_in, 1)
        history = torch.randn(d_in, 32)
        target = torch.randn(d_in, 3)
        constraints = torch.cat([edit, history, target], dim=1)
        self.assertEqual(constraints.shape, (d_in, 36))

    def test_shape_derived_rank_one_tolerance(self) -> None:
        torch.manual_seed(17)
        base = torch.outer(torch.randn(7), torch.randn(19))
        noisy = base + 1e-7 * torch.randn_like(base)
        _, _, residual = rank_one_factor(noisy, shape_rank_rtol(tuple(noisy.shape)))
        self.assertLess(residual, shape_rank_rtol(tuple(noisy.shape)))


if __name__ == "__main__":
    unittest.main()
