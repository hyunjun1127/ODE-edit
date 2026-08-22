from __future__ import annotations

import inspect
import types
import unittest

import torch

from project.run_scripts.ode_bf import p1r52_c_writer_kstep as kstep
from project.run_scripts.ode_bf import p1r52_c_writer_kstep_cache as cache
from project.run_scripts.ode_bf import p1r52_c_writer_kstep_cache_sequential as phase3


LAYERS = (4, 5, 6, 7, 8)


class KStepCacheContractTest(unittest.TestCase):
    def test_c0_c1_batch_entry_history_appends_once_after_k8(self) -> None:
        first_keys = {layer: torch.full((3, 100), float(layer), dtype=torch.float32) for layer in LAYERS}
        policy = cache.KStepBatchEntryCachePolicy(
            arm="C0-KSTEP-CACHE",
            batch_index=1,
            history_keys_by_layer=None,
            history_version=0,
        )
        for step in range(8):
            current = {
                layer: first_keys[layer] if step == 0 else first_keys[layer] + float(step)
                for layer in LAYERS
            }
            policy.close_c0_c1_step(step, current)
        commit = policy.commit_after_k8()
        self.assertEqual(commit.receipt["entry_width"], 0)
        self.assertEqual(commit.receipt["append_count"], 1)
        self.assertEqual(commit.receipt["current_batch_k_key_history_inclusion_count"], 0)
        self.assertTrue(all(value.shape == (3, 100) for value in commit.history_keys_by_layer.values()))
        self.assertTrue(all(torch.equal(commit.history_keys_by_layer[layer], first_keys[layer]) for layer in LAYERS))

        second = cache.KStepBatchEntryCachePolicy(
            arm="C1-KSTEP-CACHE",
            batch_index=2,
            history_keys_by_layer=commit.history_keys_by_layer,
            history_version=1,
        )
        for step in range(8):
            second.close_c0_c1_step(step, {layer: first_keys[layer] + 20.0 + step for layer in LAYERS})
        second_commit = second.commit_after_k8()
        self.assertEqual(second_commit.receipt["consume_width_per_k"], 100)
        self.assertEqual(second_commit.receipt["exit_width"], 200)
        self.assertTrue(all(value.shape == (3, 200) for value in second_commit.history_keys_by_layer.values()))

    def test_c3_each_k_restores_entry_and_only_k8_candidate_commits(self) -> None:
        module = types.SimpleNamespace(cache_c=torch.zeros((2, 100), dtype=torch.float32), cache_c_new={})
        entry = cache.snapshot_alpha_module_cache(module)
        policy = cache.KStepBatchEntryCachePolicy(
            arm="C3-KSTEP-CACHE",
            batch_index=2,
            history_keys_by_layer=None,
            history_version=1,
            alpha_main=module,
            alpha_entry_snapshot=entry,
            expected_official_entry_sha256="ENTRY",
        )
        for step in range(8):
            policy.prepare_c3_step(step)
            self.assertTrue(torch.equal(module.cache_c, entry["cache_c"]))
            module.cache_c = torch.full((2, 200), float(step + 1), dtype=torch.float32)
            policy.close_c3_step(
                step,
                {
                    "logical_history_width_at_entry": 100,
                    "logical_history_width_after_append": 200,
                    "solver_consumed_entry_cache": True,
                    "entry": {"sha256": "ENTRY"},
                    "exit": {"sha256": f"EXIT-{step}"},
                },
            )
            self.assertTrue(torch.equal(module.cache_c, entry["cache_c"]))
        commit = policy.commit_after_k8()
        self.assertEqual(commit.receipt["official_exit_sha256"], "EXIT-7")
        self.assertEqual(commit.receipt["append_count"], 1)
        self.assertTrue(torch.equal(module.cache_c, torch.full((2, 200), 8.0, dtype=torch.float32)))

    def test_partial_or_duplicate_cache_transaction_fails_closed(self) -> None:
        keys = {layer: torch.zeros((2, 100), dtype=torch.float32) for layer in LAYERS}
        policy = cache.KStepBatchEntryCachePolicy(
            arm="C0-KSTEP-CACHE", batch_index=1, history_keys_by_layer=None, history_version=0
        )
        policy.close_c0_c1_step(0, keys)
        with self.assertRaises(Exception):
            policy.commit_after_k8()
        with self.assertRaises(Exception):
            policy.close_c0_c1_step(0, keys)

    def test_phase3_map_and_reuse_boundary(self) -> None:
        self.assertEqual(tuple(phase3.arm_for_role(phase3.role_for_cell(i)) for i in range(3)), kstep.CACHE_ARMS)
        source = inspect.getsource(phase3)
        self.assertIn("CKStepWriterRuntime", source)
        self.assertIn("KStepBatchEntryCachePolicy", source)
        self.assertIn("policy.commit_after_k8()", source)
        self.assertIn("_restore(touched, entry_values", source)
        for prohibited in (
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "assemble_effective_bf16",
            "CachedBF16FunctionalTrial",
        ):
            self.assertNotIn(prohibited, source)


if __name__ == "__main__":
    unittest.main()
