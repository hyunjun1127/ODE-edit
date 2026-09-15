"""End-to-end wiring and no-update behavior; no real-model quality claims."""
import copy
import json
import unittest

import torch

from project.run_scripts.single_layer_zflow.demo import DEFAULT_CONTRACT, run_demo


class TestPipeline(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.contract = json.loads(DEFAULT_CONTRACT.read_text())

    def test_actual_write_pipeline_and_receipt(self):
        result = run_demo(self.contract)
        self.assertEqual(result["commit_status"], "COMMITTED")
        self.assertGreater(result["accepted_steps"], 0)
        self.assertEqual(result["oracle_calls"],
                         1+result["accepted_steps"]+result["rejected_steps"])
        self.assertEqual(result["work"]["suffix_backward_tokens"],
                         result["oracle_calls"]*30)
        self.assertLess(result["terminal"]["F"], result["initial_L"])
        self.assertEqual(result["preparation"]["native_z_fits"], 0)
        self.assertTrue(result["parity"]["checked"])
        self.assertEqual(result["commit_receipt"]["history_append"], 1)
        self.assertTrue(result["duplicate_commit_was_noop"])
        self.assertFalse(result["llama_adapter_tested"])
        self.assertFalse(result["durable_transaction_tested"])
        json.dumps(result, allow_nan=False)

    def test_resource_limit_without_candidate_does_not_append(self):
        self.contract["integrator"]["max_oracle_calls"] = 1
        result = run_demo(self.contract)
        self.assertEqual(result["solver_status"], "RESOURCE_STOP")
        self.assertEqual(result["commit_status"], "NO_ACCEPTED_UPDATE")
        self.assertIsNone(result["commit_receipt"])
        self.assertEqual(result["accepted_steps"], 0)
        self.assertFalse(result["parity"]["checked"])
        self.assertEqual(result["work"]["logical_oracle_calls"], 1)

    def test_contract_does_not_silently_disable_requested_barrier(self):
        for mode in ("fixed", "exponential"):
            contract = copy.deepcopy(self.contract)
            contract["integrator"]["barrier_mode"] = mode
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                run_demo(contract)
        self.contract["integrator"]["budget"] = .1
        with self.assertRaises(ValueError):
            run_demo(self.contract)


if __name__ == "__main__":
    unittest.main()
