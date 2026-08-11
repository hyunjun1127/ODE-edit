from __future__ import annotations

import inspect
import json
import subprocess
import unittest
from pathlib import Path

from project.run_scripts.ode_bf.contracts import ODEBFStateError
from project.run_scripts.ode_bf.historical_h0_sequential_runtime import (
    EVALUATION_ROUNDS,
    METHODS,
    _run_ours_round,
    _validate_history_off_edit_receipt,
    expected_historical_h0_result_name,
)
from project.run_scripts.ode_bf.historical_h0_sequential_selection import (
    verify_historical_h0_fresh_seal,
)
from project.run_scripts.ode_bf.p1_historical_h0_sequential_panel import (
    LOCK_SCHEMA,
    load_and_validate_historical_h0_lock,
)
from project.run_scripts.ode_bf.p1_scalable_batched_experiment import _run_ode_arm
from project.run_scripts import session05_ode_bf_historical_h0_sequential_dry_plan as dry


ROOT = Path(__file__).resolve().parents[4]
LOCKS = ROOT / "project/run_scripts/ode_bf/locks"


def _history_off_receipt() -> dict[str, object]:
    return {
        "history_mode": "OFF",
        "router_visible_history_item_count": 0,
        "raw_historical_request_replay_count": 0,
        "projected_key_historical_sketch_construction_count": 0,
        "functional_h_status": "INACTIVE_BY_HISTORY_MODE_OFF",
        "functional_h_controller_input_count": 0,
        "functional_h_decision_influence_count": 0,
        "functional_h_model_forward_count": 0,
        "structural_h_status": "INACTIVE_BY_HISTORY_MODE_OFF",
        "structural_h_controller_input_count": 0,
        "structural_h_decision_influence_count": 0,
        "structural_h_model_forward_count": 0,
    }


class ProgressSimplexSequentialNoHTest(unittest.TestCase):
    def test_frozen_stream_and_noh_lock(self) -> None:
        seal = verify_historical_h0_fresh_seal(
            json.loads(
                (LOCKS / "p1r20_historical_h0_fresh_cf_b100_seal.json").read_text()
            )
        )
        lock, _ = load_and_validate_historical_h0_lock(
            LOCKS / "numerical_lock_s05_historical_h0_sequential.json"
        )
        self.assertEqual(lock["schema_version"], LOCK_SCHEMA)
        self.assertEqual(lock["fresh_seal_root"], seal["root_digest"])
        self.assertEqual(lock["history"]["mode"], "OFF")
        self.assertEqual(lock["history"]["round_entry_counts"], [0] * 10)
        self.assertEqual(EVALUATION_ROUNDS, (1, 5, 10))

    def test_exact_atomic_router_is_byte_identical(self) -> None:
        relative = "project/run_scripts/ode_bf/progress_simplex_routing.py"
        expected = subprocess.run(
            ["git", "show", f"a343d1f6967ef37763009b92d227ade85cd93de0:{relative}"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        self.assertEqual((ROOT / relative).read_bytes(), expected)

    def test_full_matrix_dry_plan_is_repeatable(self) -> None:
        first = dry.build_plan("0" * 40)
        self.assertEqual(first, dry.build_plan("0" * 40))
        self.assertEqual(first["trajectory_count"], 8)
        self.assertEqual(first["ode_trajectory_count"], 8)
        self.assertEqual(first["model_level_alphaedit_rerun_count"], 0)
        self.assertEqual(first["history_mode"], "OFF")
        self.assertEqual(first["array_max_concurrent_gpu"], 4)
        self.assertEqual(len(first["jobs"]), 8)
        self.assertEqual(len({job["result_name"] for job in first["jobs"]}), 8)

    def test_history_off_positive_and_each_negative(self) -> None:
        receipt = _history_off_receipt()
        _validate_history_off_edit_receipt(receipt)
        for key in tuple(receipt):
            changed = dict(receipt)
            if key.endswith("status") or key == "history_mode":
                changed[key] = "ACTIVE"
            else:
                changed[key] = 1
            with self.subTest(key=key), self.assertRaises(ODEBFStateError):
                _validate_history_off_edit_receipt(changed)

    def test_exact_full_six_functional_p_and_weight_accumulation_source(self) -> None:
        arm_source = inspect.getsource(_run_ode_arm)
        round_source = inspect.getsource(_run_ours_round)
        runtime_source = (ROOT / "project/run_scripts/ode_bf/historical_h0_sequential_runtime.py").read_text()
        self.assertIn("_fixed_e8_functional_basis_probe", arm_source)
        self.assertIn("slope_context_ordinals", arm_source)
        self.assertIn("full_six_slope=True", round_source)
        self.assertIn("progress_simplex=True", round_source)
        self.assertNotIn("compute_aware=True", round_source)
        self.assertNotIn("historical_sketch=", round_source)
        self.assertIn("before_hashes == initial_hashes", runtime_source)
        self.assertIn("batches=stream_batches[:round_index]", runtime_source)
        self.assertIn("postcommit_cumulative_b10_evaluation_count", runtime_source)

    def test_namespaces_and_array(self) -> None:
        names = {
            expected_historical_h0_result_name(alias, method)
            for alias in ("llama3-8b-inst", "qwen2.5-7b-inst")
            for method in METHODS
        }
        self.assertEqual(len(names), 8)
        source = (
            ROOT / "project/run_scripts/session05_ode_bf_historical_h0_sequential.sbatch"
        ).read_text()
        self.assertIn("#SBATCH --array=0-7%4", source)
        self.assertIn("#SBATCH --nodelist=devbox", source)
        self.assertNotIn("ALPHAEDIT", source)
        self.assertNotIn("COMPUTE-FULL6", source)


if __name__ == "__main__":
    unittest.main()
