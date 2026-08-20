from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import torch
from torch import nn

from project.run_scripts.ode_bf.p1r52_pir_writer import plan_pir_writer
from project.run_scripts.ode_bf.p1r52_frozen_pi_quota_writer import (
    build_current_layer_field,
)
from project.run_scripts.ode_bf.p1r52_piru_cache_continuity import (
    CACHE_COMPLETE_ROLE,
    LEGACY_ROLE,
    PIRUCacheContinuityPolicy,
    committed_history_snapshot,
    policy_for_role,
)
from project.run_scripts.ode_bf.p1r52_piru_cache_continuity_panel import (
    LOCK_FILE,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r52_piru_sequential_adapter import (
    is_piru_structural_h_role,
    pir_policy_for_role,
)
from project.run_scripts.ode_bf.p1r52_sequential_contract import (
    scoped_atomic_sequential_adapter,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    R52_ROLES,
    R52_STRUCTURAL_H_ROLES,
)
from project.run_scripts.ode_bf.p1r52_pir_writer import P1R52PIRPolicy


class P1R52PIRUCacheContinuityTests(unittest.TestCase):
    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.ModuleDict({"mlp": nn.Linear(3, 2, bias=False)}) for _ in range(9)]
            )

    def test_roles_change_only_prefix_history_policy(self) -> None:
        self.assertIs(policy_for_role(LEGACY_ROLE), PIRUCacheContinuityPolicy.LEGACY)
        self.assertIs(
            policy_for_role(CACHE_COMPLETE_ROLE),
            PIRUCacheContinuityPolicy.CACHE_COMPLETE,
        )
        for role in (LEGACY_ROLE, CACHE_COMPLETE_ROLE):
            self.assertIn(role, R52_ROLES)
            self.assertIn(role, R52_STRUCTURAL_H_ROLES)
            self.assertTrue(is_piru_structural_h_role(role))
            self.assertIs(pir_policy_for_role(role), P1R52PIRPolicy.PIR_U)

    def test_committed_snapshot_width_hash_and_exclusion(self) -> None:
        keys = {
            layer: torch.arange(300, dtype=torch.float32).reshape(3, 100) + layer
            for layer in range(4, 9)
        }
        frozen, receipt = committed_history_snapshot(
            keys,
            expected_width=100,
            ledger_version=1,
            ledger_digest="a" * 64,
        )
        self.assertTrue(all(value.shape == (3, 100) for value in frozen.values()))
        self.assertEqual(receipt["committed_history_record_count"], 100)
        self.assertEqual(receipt["current_batch_key_inclusion_count"], 0)
        self.assertEqual(receipt["current_uncommitted_prefix_key_inclusion_count"], 0)
        for layer in keys:
            self.assertNotEqual(frozen[layer].data_ptr(), keys[layer].data_ptr())
            self.assertTrue(torch.equal(frozen[layer], keys[layer]))

    def test_b1_empty_history_is_common_and_b2_b10_widths_are_exact(self) -> None:
        for width in (0, 100, 900):
            keys = {
                layer: torch.zeros((4, width), dtype=torch.float32)
                for layer in range(4, 9)
            }
            _, receipt = committed_history_snapshot(
                keys,
                expected_width=width,
                ledger_version=width // 100,
                ledger_digest="b" * 64,
            )
            self.assertEqual(receipt["committed_history_record_count"], width)

    def test_writer_and_adapter_have_no_coefficient_or_current_key_history_delta(self) -> None:
        writer_source = inspect.getsource(plan_pir_writer)
        adapter_source = inspect.getsource(scoped_atomic_sequential_adapter)
        self.assertIn("prefix_history_keys_by_layer", writer_source)
        self.assertIn("current_batch_history_inclusion_count\": 0", writer_source)
        self.assertIn("current_prefix_history_inclusion_count\": 0", writer_source)
        self.assertIn("prefix_empty_history_solve_count", writer_source)
        self.assertIn("PIRU-CACHE-COMPLETE", writer_source)
        self.assertIn("PIRUCacheContinuityPolicy.CACHE_COMPLETE", adapter_source)
        self.assertIn("history_state.ledger.solve_keys", adapter_source)
        self.assertNotIn("current_keys", adapter_source)

    def test_full_woodbury_cache_history_changes_q_without_model_work(self) -> None:
        model = self._TinyModel()
        hparams = type(
            "HParams",
            (),
            {"layers": (4, 5, 6, 7, 8), "L2": 0.2, "rewrite_module_tmp": "layers.{}.mlp"},
        )()
        projector = torch.eye(3, dtype=torch.float32).repeat(5, 1, 1)
        key = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.25]], dtype=torch.float32
        )
        history = torch.tensor(
            [[0.25, 0.0], [0.0, 0.5], [0.75, 0.25]], dtype=torch.float32
        )
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        current = torch.zeros_like(target)
        before = tuple(parameter.detach().clone() for parameter in model.parameters())
        legacy, _ = build_current_layer_field(
            model,
            hparams,
            projector,
            None,  # type: ignore[arg-type]
            layer=5,
            key=key,
            target_state=target,
            current_terminal=current,
            step_index=0,
            factor_ordinal=1,
            projector_sha256="c" * 64,
            residual_tolerance=1e-8,
            q_only=True,
        )
        complete, _ = build_current_layer_field(
            model,
            hparams,
            projector,
            None,  # type: ignore[arg-type]
            layer=5,
            key=key,
            target_state=target,
            current_terminal=current,
            step_index=0,
            factor_ordinal=1,
            projector_sha256="c" * 64,
            residual_tolerance=1e-8,
            q_only=True,
            history_keys=history,
        )
        self.assertFalse(torch.equal(legacy.q, complete.q))
        self.assertTrue(complete.woodbury_certificate.passed)
        self.assertLessEqual(complete.woodbury_certificate.alpha_linear_residual, 1e-8)
        self.assertTrue(
            all(torch.equal(value, old) for value, old in zip(model.parameters(), before, strict=True))
        )

    def test_numerical_lock_is_rooted(self) -> None:
        repo = Path(__file__).resolve().parents[4]
        lock, sha = load_and_validate_lock(
            repo / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(len(sha), 64)
        self.assertEqual(lock["stage_a_cache_complete_rounds"], [10])


if __name__ == "__main__":
    unittest.main()
