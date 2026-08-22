from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import types
import unittest

import torch

from project.run_scripts.ode_bf import p1r52_c_writer_kstep as kstep
from project.run_scripts.ode_bf import p1r52_c_writer_kstep_cache as cache
from project.run_scripts.ode_bf import p1r52_c_writer_kstep_cache_sequential as phase3
from project.run_scripts.ode_bf import p1r52_c_writer_phase2_dependency as dependency


LAYERS = (4, 5, 6, 7, 8)


class KStepCacheContractTest(unittest.TestCase):
    def _write_receipt(self, path: Path, payload: dict) -> str:
        payload = dict(payload)
        payload["identity_sha256"] = dependency.canonical_hash(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return dependency.sha256_file(path)

    def _phase2_dependency_fixture(self, parent: Path, source_head: str) -> None:
        for role, arm in zip(dependency.ROLES, dependency.ARMS, strict=True):
            root = parent / dependency.expected_result_name(role)
            case_shas = []
            rows = []
            for index in range(1, 11):
                case_path = root / "raw" / "cases" / f"case-{index:02d}" / "terminal.json"
                case_sha = self._write_receipt(
                    case_path,
                    {
                        "case_index": index,
                        "arm": arm,
                        "request_count": 100,
                        "kstep_executions": [{} for _ in range(8)],
                        "alpha_cache_status": "ALPHA_CACHE_OFF_CONTROL",
                        "cross_case_W_cache_history_carry_count": 0,
                        "retry_count": 0,
                        "imputation_count": 0,
                    },
                )
                case_shas.append(case_sha)
                rows.append({"case_index": index, "terminal_sha256": case_sha})
            terminal_path = root / "terminal.json"
            terminal_sha = self._write_receipt(
                terminal_path,
                {
                    "schema": "ode-edit-s05-p1r52-c-writer-phase2-terminal/v1",
                    "status": "TERMINAL_VALID",
                    "source_head": source_head,
                    "role": role,
                    "arm": arm,
                    "case_count": 10,
                    "valid_request_count": 1000,
                    "K_writer_call_count": 80,
                    "alpha_cache_status": "ALPHA_CACHE_OFF_CONTROL",
                    "cross_case_state_count": 0,
                    "W0_restored": True,
                    "technical_failure_count": 0,
                    "scientific_failure_count": 0,
                    "imputation_count": 0,
                    "case_terminal_sha256": case_shas,
                    "cases": rows,
                    "dtype_contract": {
                        "status": "FULL_FP32_PASS",
                        "numeric_storage_cast_count": 0,
                        "bf16_fp16_path_count": 0,
                    },
                },
            )
            self._write_receipt(
                root / "manifest.json",
                {
                    "schema": "ode-edit-s05-p1r52-c-writer-phase2-manifest/v1",
                    "source_head": source_head,
                    "role": role,
                    "terminal_sha256": terminal_sha,
                    "case_terminal_sha256": case_shas,
                    "W0_restored": True,
                },
            )

    def test_phase3_dependency_gate_requires_three_terminal_valid_phase2_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source_head = "a" * 40
            self._phase2_dependency_fixture(parent, source_head)
            receipt = dependency.verify_phase2_terminal_completeness(
                parent, expected_source_head=source_head
            )
            self.assertEqual(receipt.status, "PHASE2_TERMINAL_VALID_DEPENDENCY_PASS")
            self.assertEqual((receipt.arm_count, receipt.valid_case_count, receipt.valid_request_count), (3, 30, 3000))
            self.assertEqual(receipt.k_writer_call_count, 240)
            self.assertEqual(receipt.w0_restore_count, 3)

    def test_phase3_dependency_gate_rejects_exit0_but_incomplete_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            source_head = "a" * 40
            self._phase2_dependency_fixture(parent, source_head)
            terminal_path = parent / dependency.expected_result_name(dependency.ROLES[1]) / "terminal.json"
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            terminal.pop("identity_sha256")
            terminal["case_count"] = 9
            self._write_receipt(terminal_path, terminal)
            with self.assertRaisesRegex(Exception, "terminal completeness"):
                dependency.verify_phase2_terminal_completeness(parent, expected_source_head=source_head)

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
