from __future__ import annotations

import copy
import unittest

from project.run_scripts.ode_edit_motivation.capacity_history_analysis import (
    BRANCH_ALPHA_NATIVE,
    BRANCH_ALPHA_QP,
    BRANCH_MEMIT_NATIVE,
    BRANCH_MEMIT_QP,
    CapacityHistoryAnalysisError,
    analyze_pair,
    analyze_records,
)


def _contract() -> dict[str, object]:
    return {
        "selection_manifest_id": "a" * 64,
        "case_order_hash": "b" * 64,
        "chain_length": 4,
        "layers": [4, 5, 6, 7, 8],
        "seed": 41,
        "policy_id": "capacity-history-common-policy-v1",
        "policy_parameters_sha256": "c" * 64,
        "evaluator_sha256": "d" * 64,
        "action_before_evaluation": True,
        "direct_z_per_branch_edit": 1,
        "history_append_policy": "post-accepted-edit-once-history-fixed-within-edit",
    }


def _technical() -> dict[str, bool]:
    return {
        "action_committed_before_evaluation": True,
        "evaluation_firewall_pass": True,
        "direct_z_once": True,
        "state_lineage_exact": True,
        "w0_anchor_exact": True,
        "precomputed_covariance_only": True,
        "precomputed_projector_only": True,
        "alpha_history_exact": True,
        "capacity_barrier_exact": True,
        "overloaded_positive_write_suppressed": True,
        "finite_metrics": True,
    }


def _row(model: str, branch: str, index: int) -> dict[str, object]:
    is_qp = branch in {BRANCH_MEMIT_QP, BRANCH_ALPHA_QP}
    is_alpha = branch in {BRANCH_ALPHA_NATIVE, BRANCH_ALPHA_QP}
    return {
        "model_alias": model,
        "edit_id": f"edit-{index}",
        "edit_index": index,
        "branch": branch,
        "pass": True,
        "technical": _technical(),
        "fixed_contract": _contract(),
        "current_utility": 0.95 if is_qp else 1.0,
        "all_edits_mean_utility": 1.1 if is_qp else 1.0,
        "prior_mean_utility": None if index == 1 else (1.2 if is_qp else 1.0),
        "prior_success_rate": None if index == 1 else (0.9 if is_qp else 0.8),
        "neighborhood_kl": 0.1 if is_qp else 0.2,
        "generation_kl": 0.2 if is_qp else 0.3,
        "target_true_nll_delta": 0.1 if is_qp else 0.2,
        "capacity_sum": (1.0 + index * 0.1) if is_qp else (1.5 + index * 0.2),
        "max_layer_share": 0.3 if is_qp else 0.4,
        "layer_gini": 0.2 if is_qp else 0.3,
        "cumulative_frobenius": 0.8 * index if is_qp else float(index),
        "native_distance_equivalent_path_length": float(index),
        "accepted_path_distance": 0.75 if is_qp else 1.0,
        "wall_seconds": 2.0 if is_qp else 1.0,
        "proposal_build_count": 3 if is_qp else 1,
        "accepted_round_count": 2 if is_qp else 1,
        "rejected_round_count": 0,
        "first_hit_reached": is_qp,
        "history_edit_count": index if is_alpha else 0,
        "overloaded_layer_observations": 1 if is_qp else 0,
        "suppressed_overloaded_observations": 1 if is_qp else 0,
        "capacity_reroute_round_count": 1 if is_qp else 0,
        "capacity_barrier_max_violation": 0.0,
    }


def _records(model: str) -> list[dict[str, object]]:
    return [
        _row(model, branch, index)
        for branch in (
            BRANCH_MEMIT_NATIVE,
            BRANCH_MEMIT_QP,
            BRANCH_ALPHA_NATIVE,
            BRANCH_ALPHA_QP,
        )
        for index in range(1, 5)
    ]


class CapacityHistoryAnalysisTests(unittest.TestCase):
    def test_model_signal_is_family_separated_and_nfe_free(self) -> None:
        result = analyze_records(_records("llama3-8b-inst"), model_alias="llama3-8b-inst")
        self.assertEqual(result["verdict"], "CAPACITY_HISTORY_MODEL_SIGNAL")
        for family in result["families"].values():
            self.assertTrue(family["lenient_gate"]["model_gate_pass"])
            self.assertAlmostEqual(family["effects"]["current_edit_utility_mean_delta"], -0.05)
            self.assertGreater(family["effects"]["capacity_reduction"], 0.0)
            self.assertTrue(
                family["routing_diagnostic"]["all_observed_positive_overloads_suppressed"]
            )
        self.assertNotIn("nfe", repr(result).lower())

    def test_harm_is_exposed_without_model_specific_rescue(self) -> None:
        records = _records("llama3-8b-inst")
        for row in records:
            if row["branch"] == BRANCH_MEMIT_QP:
                row["current_utility"] = 0.7
        result = analyze_records(records, model_alias="llama3-8b-inst")
        self.assertEqual(
            result["families"]["memit"]["verdict"],
            "CAPACITY_QP_HARM_SIGNAL",
        )

    def test_history_count_and_contract_drift_fail_closed(self) -> None:
        records = _records("llama3-8b-inst")
        records[-1]["history_edit_count"] = 3
        with self.assertRaisesRegex(CapacityHistoryAnalysisError, "history count"):
            analyze_records(records, model_alias="llama3-8b-inst")
        records = _records("llama3-8b-inst")
        records[-1]["fixed_contract"] = copy.deepcopy(records[-1]["fixed_contract"])
        records[-1]["fixed_contract"]["policy_id"] = "model-specific-rescue"
        with self.assertRaisesRegex(CapacityHistoryAnalysisError, "fixed contract differs"):
            analyze_records(records, model_alias="llama3-8b-inst")

    def test_pair_requires_model_common_family_signal(self) -> None:
        llama = analyze_records(_records("llama3-8b-inst"), model_alias="llama3-8b-inst")
        qwen = analyze_records(_records("qwen2.5-7b-inst"), model_alias="qwen2.5-7b-inst")
        pair = analyze_pair(llama, qwen)
        self.assertEqual(pair["verdict"], "CAPACITY_HISTORY_ACTUATOR_COMMON_SIGNAL")
        self.assertEqual(set(pair["passing_families"]), {"memit", "alpha_history"})

        qwen = copy.deepcopy(qwen)
        qwen["families"]["memit"]["lenient_gate"]["model_gate_pass"] = False
        qwen["families"]["memit"]["lenient_gate"]["current_noncollapse"] = False
        pair = analyze_pair(llama, qwen)
        self.assertEqual(pair["verdict"], "CAPACITY_HISTORY_FAMILY_CONDITIONED_SIGNAL")
        self.assertEqual(pair["passing_families"], ["alpha_history"])


if __name__ == "__main__":
    unittest.main()
