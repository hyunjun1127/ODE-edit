from __future__ import annotations

import copy
import unittest

from project.run_scripts.ode_edit_motivation.microseq_analysis import (
    BRANCH_NATIVE,
    BRANCH_ODE,
    MicroseqAnalysisError,
    analyze_pair,
    analyze_records,
)


def _contract() -> dict[str, object]:
    return {
        "selection_manifest_id": "a" * 64,
        "case_order_hash": "b" * 64,
        "chain_length": 4,
        "layers": [4, 5, 6, 7, 8],
        "seed": 37,
        "policy_id": "same-policy-v1",
        "policy_parameters_sha256": "c" * 64,
        "evaluator_sha256": "d" * 64,
        "action_before_evaluation": True,
        "direct_z_per_branch_edit": 1,
    }


def _technical() -> dict[str, bool]:
    return {
        "action_committed_before_evaluation": True,
        "evaluation_firewall_pass": True,
        "direct_z_once": True,
        "state_lineage_exact": True,
        "w0_anchor_exact": True,
        "precomputed_cache_only": True,
        "finite_metrics": True,
    }


def _row(model: str, branch: str, index: int) -> dict[str, object]:
    is_ode = branch == BRANCH_ODE
    return {
        "model_alias": model,
        "edit_id": f"edit-{index}",
        "edit_index": index,
        "branch": branch,
        "pass": True,
        "technical": _technical(),
        "fixed_contract": _contract(),
        "current_utility": 0.95 if is_ode else 1.0,
        "all_edits_mean_utility": 1.1 if is_ode else 1.0,
        "prior_mean_utility": None if index == 1 else (1.2 if is_ode else 1.0),
        "prior_success_rate": None if index == 1 else (0.9 if is_ode else 0.8),
        "neighborhood_kl": 0.1 if is_ode else 0.2,
        "generation_kl": 0.2 if is_ode else 0.3,
        "target_true_nll_delta": -0.1 if is_ode else 0.0,
        "capacity_sum": (1.0 + index * 0.1) if is_ode else (1.5 + index * 0.2),
        "max_layer_share": 0.3 if is_ode else 0.4,
        "layer_gini": 0.2 if is_ode else 0.3,
        "cumulative_frobenius": float(index),
        "native_distance_equivalent_path_length": float(index),
        "wall_seconds": 2.0 if is_ode else 1.0,
        "proposal_build_count": 4 if is_ode else 1,
        "controlled_nfe": 10 if is_ode else 2,
    }


def _records(model: str) -> list[dict[str, object]]:
    return [
        _row(model, branch, index)
        for branch in (BRANCH_NATIVE, BRANCH_ODE)
        for index in range(1, 5)
    ]


class MicroseqAnalysisTests(unittest.TestCase):
    def test_model_lenient_signal(self) -> None:
        result = analyze_records(_records("llama3-8b-inst"), model_alias="llama3-8b-inst")
        self.assertEqual(result["verdict"], "MICROSEQ_MODEL_SIGNAL")
        self.assertTrue(result["lenient_gate"]["current_noncollapse"])
        self.assertAlmostEqual(result["effects"]["final_prior_retention_delta"], 0.2)
        self.assertAlmostEqual(result["effects"]["current_edit_noncollapse_mean"], -0.05)

    def test_harm_signal(self) -> None:
        records = _records("llama3-8b-inst")
        for row in records:
            if row["branch"] == BRANCH_ODE:
                row["current_utility"] = 0.7
        result = analyze_records(records, model_alias="llama3-8b-inst")
        self.assertEqual(result["verdict"], "MICROSEQ_HARM_SIGNAL")

    def test_technical_invalid(self) -> None:
        records = _records("llama3-8b-inst")
        records[-1]["technical"]["evaluation_firewall_pass"] = False
        result = analyze_records(records, model_alias="llama3-8b-inst")
        self.assertEqual(result["verdict"], "TECHNICAL_INVALID")
        self.assertFalse(result["technical_validity"]["pass"])

    def test_rejects_branch_contract_drift(self) -> None:
        records = _records("llama3-8b-inst")
        records[-1]["fixed_contract"] = copy.deepcopy(records[-1]["fixed_contract"])
        records[-1]["fixed_contract"]["policy_id"] = "model-specific-rescue"
        with self.assertRaisesRegex(MicroseqAnalysisError, "fixed contract differs"):
            analyze_records(records, model_alias="llama3-8b-inst")

    def test_pair_requires_common_axis(self) -> None:
        llama = analyze_records(
            _records("llama3-8b-inst"), model_alias="llama3-8b-inst"
        )
        qwen = analyze_records(
            _records("qwen2.5-7b-inst"), model_alias="qwen2.5-7b-inst"
        )
        pair = analyze_pair(llama, qwen)
        self.assertEqual(pair["verdict"], "MICROSEQ_LENIENT_SIGNAL")
        self.assertIn("final_prior_retention_delta", pair["common_positive_axes"])

        qwen = copy.deepcopy(qwen)
        for axis in qwen["positive_axes"]:
            qwen["positive_axes"][axis] = False
        pair = analyze_pair(llama, qwen)
        self.assertEqual(pair["verdict"], "MICROSEQ_NO_COMMON_SIGNAL")

    def test_pair_rejects_policy_mismatch(self) -> None:
        llama = analyze_records(
            _records("llama3-8b-inst"), model_alias="llama3-8b-inst"
        )
        qwen = analyze_records(
            _records("qwen2.5-7b-inst"), model_alias="qwen2.5-7b-inst"
        )
        qwen = copy.deepcopy(qwen)
        qwen["fixed_contract"]["policy_id"] = "other-policy"
        with self.assertRaisesRegex(MicroseqAnalysisError, "fixed contracts differ"):
            analyze_pair(llama, qwen)


if __name__ == "__main__":
    unittest.main()
