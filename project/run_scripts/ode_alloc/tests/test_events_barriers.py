from __future__ import annotations

import unittest

from project.run_scripts.ode_alloc.barriers import (
    HistoryItem,
    RequestKey,
    assert_anchor_disjoint,
    evaluate_dual_barriers,
    normalize_damage_with_fixed_floor,
    remove_obsolete_history,
    summarize_small_buffer,
)
from project.run_scripts.ode_alloc.contracts import ODEAllocContractError
from project.run_scripts.ode_alloc.events import (
    RewriteGateConfig,
    evaluate_rewrite_gate,
)


class EventBarrierTests(unittest.TestCase):
    def test_uniform_mean_and_oracle_ratio_are_the_only_decision_gate(self) -> None:
        reading = evaluate_rewrite_gate(
            candidate_new={"c0": -0.5, "c1": -2.0},
            candidate_old={"c0": -1.0, "c1": -1.8},
            entry_new={"c0": -2.0, "c1": -3.0},
            direct_z_new={"c0": -0.1, "c1": -1.0},
            direct_z_old={"c0": -1.0, "c1": -1.5},
            config=RewriteGateConfig(rho=0.5, denominator_epsilon=1.0e-8),
        )
        self.assertTrue(reading.feasible)
        self.assertLess(reading.worst_context_margin, 0.0)
        self.assertGreaterEqual(reading.mean_margin, 0.0)
        self.assertGreaterEqual(reading.oracle_ratio, 0.5)
        self.assertEqual(len(reading.event_id), 64)

    def test_rewrite_ineligibility_fails_closed(self) -> None:
        common = {
            "candidate_new": {"c0": -1.0, "c1": -1.0},
            "candidate_old": {"c0": -2.0, "c1": -2.0},
            "entry_new": {"c0": -1.0, "c1": -1.0},
            "direct_z_new": {"c0": -1.0, "c1": -1.0},
            "direct_z_old": {"c0": -2.0, "c1": -2.0},
            "config": RewriteGateConfig(),
        }
        with self.assertRaises(ODEAllocContractError):
            evaluate_rewrite_gate(**common)
        with self.assertRaises(ODEAllocContractError):
            evaluate_rewrite_gate(
                candidate_new={"c0": -1.0},
                candidate_old={"c0": -2.0},
                entry_new={"c0": -3.0},
                direct_z_new={"c0": -1.0},
                direct_z_old={"c0": -0.5},
                config=RewriteGateConfig(),
            )
        with self.assertRaises(ODEAllocContractError):
            evaluate_rewrite_gate(
                candidate_new={"c0": -1.0},
                candidate_old={"c0": -2.0, "c1": -2.0},
                entry_new={"c0": -3.0},
                direct_z_new={"c0": -1.0},
                direct_z_old={"c0": -2.0},
                config=RewriteGateConfig(),
            )

    def test_exact_normalized_collision_is_audit_only(self) -> None:
        incoming = RequestKey(" Ada  Lovelace ", "P19", "{} was born in")
        obsolete = HistoryItem(
            RequestKey("ada lovelace", "p19", "{}  was born in"), 0.2, "old"
        )
        retained = HistoryItem(
            RequestKey("Ada Lovelace", "P20", "{} was born in"), 0.1, "other"
        )
        hard, audit = remove_obsolete_history((obsolete, retained), incoming)
        self.assertEqual(hard, (retained,))
        self.assertEqual(audit, (obsolete,))

    def test_small_buffer_uses_mean_and_smooth_max_without_tail_estimator(self) -> None:
        empty = summarize_small_buffer(
            (), fixed_scale=0.001, smooth_temperature=0.01
        )
        self.assertEqual((empty.mean, empty.smooth_max), (0.0, 0.0))
        summary = summarize_small_buffer(
            (0.0, 0.01, 0.02), fixed_scale=0.001, smooth_temperature=0.01
        )
        self.assertEqual(summary.raw, (0.0, 0.01, 0.02))
        self.assertGreater(summary.smooth_max, summary.mean)
        self.assertFalse(hasattr(summary, "cvar"))
        with self.assertRaises(ODEAllocContractError):
            summarize_small_buffer(
                (0.0, 0.01, 0.02, 0.03),
                fixed_scale=0.001,
                smooth_temperature=0.01,
            )

    def test_fixed_floor_and_anchor_disjointness(self) -> None:
        self.assertEqual(normalize_damage_with_fixed_floor(0.002, 0.001), 2.0)
        assert_anchor_disjoint(("a", "b"), ("c", "d"))
        with self.assertRaises(ODEAllocContractError):
            assert_anchor_disjoint(("a", "b"), ("b", "c"))

    def test_historical_and_pretrained_barriers_remain_separate(self) -> None:
        reading = evaluate_dual_barriers(
            historical_damage=(0.001, 0.002),
            pretrained_damage=tuple(0.0001 * index for index in range(16)),
            historical_fixed_scale=0.001,
            pretrained_fixed_scale=0.002,
            smooth_temperature=0.01,
        )
        self.assertEqual(
            reading.historical_current_teacher.fixed_scale, 0.001
        )
        self.assertEqual(
            reading.pretrained_theta0_teacher.fixed_scale, 0.002
        )
        self.assertEqual(len(reading.historical_current_teacher.raw), 2)
        self.assertEqual(len(reading.pretrained_theta0_teacher.raw), 16)


if __name__ == "__main__":
    unittest.main()
