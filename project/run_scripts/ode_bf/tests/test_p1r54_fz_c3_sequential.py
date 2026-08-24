from __future__ import annotations

import inspect
import unittest

import torch

from project.run_scripts.ode_bf.contracts import ODEBFStateError
from project.run_scripts.ode_bf.p1r52_c_writer_kstep import CACHE_ARMS
from project.run_scripts.ode_bf.p1r52_c_writer_kstep_cache_sequential import (
    ROLES as LEGACY_PHASE3_ROLES,
    arm_for_role,
    validate_phase3_batch_chain,
)
from project.run_scripts.ode_bf.p1r52_r42_safe_kdc import P1R52AmplitudeContext
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE
from project.run_scripts.ode_bf.p1r54_fz_c3_sequential import (
    HELDOUT_K_INDICES,
    RESULT_NAME,
    ROLE,
    atomic_b1_source_equivalence,
    build_binding,
)


def _context(step: int, rho: torch.Tensor) -> P1R52AmplitudeContext:
    count = rho.numel()
    semantic = torch.stack(
        (
            torch.linspace(1.0, 2.0, count, dtype=torch.float64),
            torch.linspace(2.0, 1.0, count, dtype=torch.float64),
        )
    )
    norm = torch.linalg.vector_norm(semantic, dim=0)
    direction = -semantic / norm.unsqueeze(0)
    return P1R52AmplitudeContext(
        step,
        1.0,
        semantic,
        norm,
        norm,
        torch.ones(count, dtype=torch.float64),
        torch.ones(count, dtype=torch.bool),
        direction,
        torch.ones(count, dtype=torch.float64),
        0.0,
        rho,
    )


class P1R54FZC3SequentialTests(unittest.TestCase):
    def test_thin_binding_and_atomic_b1_science_equivalence(self) -> None:
        binding = build_binding()
        receipt = atomic_b1_source_equivalence()
        self.assertEqual(binding.role, ROLE)
        self.assertEqual(binding.writer_arm, "C3-KSTEP-CACHE")
        self.assertEqual(binding.heldout_step_indices, (7,))
        self.assertEqual(binding.target_subcycle_schedule.total_field_evaluations, 8)
        self.assertEqual(receipt["different_scientific_input_count"], 0)
        self.assertEqual(receipt["target_dt"], 0.125)
        self.assertEqual(receipt["heldout_k_indices"], [7])
        self.assertEqual(HELDOUT_K_INDICES, (7,))

    def test_each_batch_gets_fresh_rho_frozen_policy(self) -> None:
        binding = build_binding()
        left = binding.amplitude_policy_factory(1, 100)
        right = binding.amplitude_policy_factory(2, 100)
        rho_left = torch.linspace(1.0, 2.0, 100, dtype=torch.float64)
        rho_right = torch.linspace(2.0, 3.0, 100, dtype=torch.float64)
        for step in range(8):
            left(_context(step, rho_left))
            right(_context(step, rho_right))
        self.assertNotEqual(left.rho_sha256, right.rho_sha256)
        self.assertEqual(left.call_count, 8)
        self.assertEqual(right.call_count, 8)

    def test_batch_commit_cache_chain_positive_and_negative(self) -> None:
        rows = []
        entry = {"layer": "w0"}
        for index in range(1, 11):
            commit = {"layer": f"w{index}"}
            rows.append(
                {
                    "batch_index": index,
                    "entry_weight_sha256": entry,
                    "commit_weight_sha256": commit,
                    "cache_entry_width": (index - 1) * 100,
                    "cache_exit_width": index * 100,
                }
            )
            entry = commit
        receipt = validate_phase3_batch_chain(rows)
        self.assertEqual(receipt["commit_to_next_entry_match_count"], 9)
        self.assertEqual(receipt["entry_widths"], list(range(0, 1000, 100)))
        broken = [dict(row) for row in rows]
        broken[4]["entry_weight_sha256"] = {"layer": "stale"}
        with self.assertRaises(ODEBFStateError):
            validate_phase3_batch_chain(broken)

    def test_legacy_phase3_and_atomic_result_names_unchanged(self) -> None:
        self.assertEqual(
            tuple(arm_for_role(role) for role in LEGACY_PHASE3_ROLES),
            CACHE_ARMS,
        )
        self.assertEqual(
            expected_p1r52_sequential_result_name(
                "llama3-8b-inst", ROLE, scale=P1R52_B100X10_SCALE
            ),
            RESULT_NAME,
        )

    def test_adapter_does_not_reimplement_writer_cache_or_field(self) -> None:
        from project.run_scripts.ode_bf import p1r54_fz_c3_sequential as module

        source = inspect.getsource(module)
        for prohibited in (
            "run_official_native_apply",
            "KStepBatchEntryCachePolicy(",
            "_fp32_target_and_j0(",
            "prepare_p1r52_target_proposal(",
            "loss.backward(",
        ):
            self.assertNotIn(prohibited, source)
        self.assertIn("run_phase3(", source)


if __name__ == "__main__":
    unittest.main()
