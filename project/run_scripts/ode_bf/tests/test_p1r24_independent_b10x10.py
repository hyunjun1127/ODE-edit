from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

import torch

from project.run_scripts import session05_ode_bf_p1r24_independent_b10x10_dry_plan as dry
from project.run_scripts.ode_bf.p1r24_independent_b10x10_panel import (
    expected_result_name,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r24_independent_b10x10_runtime import (
    _case_failure,
    _history_off_receipt,
    _restore_exact_w0,
)
from project.run_scripts.ode_bf.p1r24_independent_b10x10_selection import (
    verify_historical_h0_fresh_seal,
)


ROOT = Path(__file__).resolve().parents[4]
LOCKS = ROOT / "project/run_scripts/ode_bf/locks"


class P1R24IndependentB10x10Test(unittest.TestCase):
    def test_lock_stream_and_dry_matrix(self) -> None:
        lock, _ = load_and_validate_lock(
            LOCKS / "numerical_lock_s05_p1r24_independent_b10x10.json"
        )
        seal = verify_historical_h0_fresh_seal(
            json.loads(
                (LOCKS / "p1r24_independent_b10x10_stream_seal.json").read_text()
            )
        )
        self.assertEqual(lock["fresh_stream_root"], seal["root_digest"])
        self.assertEqual(len(seal["batch_ordered_request_digest_v1"]), 10)
        plan = dry.build_plan("child")
        self.assertEqual(plan["job_count"], 2)
        self.assertEqual(plan["independent_atomic_b10_case_count"], 20)
        self.assertEqual(plan["array_max_concurrent_gpu"], 2)
        self.assertEqual(plan["project_gpu_cap"], 3)
        self.assertEqual(
            [item["alias"] for item in plan["jobs"]],
            ["llama3-8b-inst", "qwen2.5-7b-inst"],
        )
        self.assertTrue(all(item["method"] == "RS-P1R24-SOFT" for item in plan["jobs"]))

    def test_result_names_are_distinct(self) -> None:
        self.assertEqual(
            expected_result_name("llama3-8b-inst"),
            "s05-p1r24-rs-soft-independent-b10x10-llama3-8b-inst-tech-r1-v1",
        )
        self.assertNotEqual(
            expected_result_name("llama3-8b-inst"),
            expected_result_name("qwen2.5-7b-inst"),
        )

    def test_two_case_restore_has_pointer_and_byte_identity(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
        touched = {"w": parameter}
        base = {"w": parameter.detach().clone()}
        from project.run_scripts.ode_bf.p1_scalable_batched_experiment import _model_w0_contract

        expected = _model_w0_contract(touched)
        pointer = parameter.data_ptr()
        for delta in (3.0, -7.0):
            with torch.no_grad():
                parameter.add_(delta)
            receipt = _restore_exact_w0(
                touched,
                base,
                mutation_lock=threading.RLock(),
                expected_contract=expected,
            )
            self.assertTrue(receipt["pointer_restored_exact"])
            self.assertTrue(receipt["byte_restored_exact"])
            self.assertEqual(parameter.data_ptr(), pointer)
            self.assertTrue(torch.equal(parameter.detach(), base["w"]))

    def test_typed_failure_isolated_without_retry(self) -> None:
        receipt = {
            "pointer_restored_exact": True,
            "byte_restored_exact": True,
            "identity_sha256": "x",
        }
        with tempfile.TemporaryDirectory() as directory:
            payload = _case_failure(
                Path(directory),
                RuntimeError("NO_POSITIVE_DIRECTION"),
                case_index=4,
                method="RS-P1R24-SOFT",
                w0_restore=receipt,
            )
            self.assertEqual(payload["retry_count"], 0)
            self.assertTrue(payload["next_case_continues"])
            self.assertEqual(payload["status"], "TYPED_CASE_FAILURE_NO_RETRY_NO_IMPUTATION")
            self.assertTrue((Path(directory) / "failure.json").is_file())

    def test_history_and_atomic_entry_are_zero(self) -> None:
        value = _history_off_receipt()
        self.assertEqual(value["mode"], "OFF")
        for key in (
            "router_visible_history_item_count",
            "history_append_count",
            "raw_historical_request_replay_count",
            "projected_key_historical_sketch_construction_count",
            "functional_h_input_count",
            "functional_h_decision_influence_count",
            "structural_h_input_count",
            "structural_h_decision_influence_count",
            "cross_case_weight_state_count",
            "cross_case_controller_state_count",
            "atomic_trajectory_ledger_entry_count_at_case_entry",
        ):
            self.assertEqual(value[key], 0)

    def test_source_uses_exact_p1r24_rs_soft_and_terminal_firewall(self) -> None:
        source = (
            ROOT / "project/run_scripts/ode_bf/p1r24_independent_b10x10_runtime.py"
        ).read_text()
        self.assertIn("p1r24=True", source)
        self.assertIn('method != "RS-P1R24-SOFT"', source)
        self.assertIn("FixedE8Arm.SOFT", source)
        self.assertNotIn("run_official_native_apply", source)
        self.assertIn("actions_frozen_before_evaluator", source)
        self.assertIn("inner_k_evaluator_access_count", source)
        self.assertIn("for case_index, requests in enumerate(stream_batches, start=1)", source)
        self.assertIn("cross-case W0 state leak detected", source)
        self.assertIn("seed_all(COMMON_SEED)", source)
        self.assertIn('"cross_case_mutable_cache_count": 0', source)
        self.assertNotIn("objective_plan.context_ordinals", source)
        self.assertIn('objective_payload.get("context_count") != 6', source)
        self.assertIn("capture_ordinals != list(range(6))", source)

    def test_parent_objective_plan_has_no_context_ordinals_attribute(self) -> None:
        from dataclasses import fields
        from project.run_scripts.ode_bf.scalable_batched_model import ScalableObjectivePlan

        names = {item.name for item in fields(ScalableObjectivePlan)}
        self.assertNotIn("context_ordinals", names)

    def test_scientific_core_hashes_match_parent(self) -> None:
        from project.run_scripts.ode_bf.artifacts import sha256_file

        expected = {
            "p1r24_atomic_strength.py": "95ef3aff1e0b2d82df08453492a01be4d69d7e1c8795edd72c7d098979f54b0f",
            "p1_scalable_batched_experiment.py": "282030bba83163e069ff8ac0a2439d03b82fa626d2e0eea4547a91cc78ebe440",
            "p1r24_atomic_strength_panel.py": "0c46b34a0b6f998ce4c0f8b52c18b9e829d44ce102a41ec37d14a22cc8604c64",
        }
        for relative, digest in expected.items():
            self.assertEqual(sha256_file(ROOT / "project/run_scripts/ode_bf" / relative), digest)


if __name__ == "__main__":
    unittest.main()
