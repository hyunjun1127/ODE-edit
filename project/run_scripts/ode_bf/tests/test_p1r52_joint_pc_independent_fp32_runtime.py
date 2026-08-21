from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest
from unittest import mock

import torch

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1r52_joint_pc_independent_fp32_runtime import (
    CELL_LABELS,
    RESULT_NAMES,
    ROLES,
    _assert_w0,
    _assert_no_low_precision_activity,
    _timed,
    expected_result_name,
    full_fp32_parameter_inventory,
    role_for_cell,
)
from project.run_scripts.ode_bf.p1r52_sequential_runtime import (
    expected_p1r52_sequential_result_name,
)
from project.run_scripts.ode_bf.p1r52_sequential_scale import P1R52_B100X10_SCALE
from project.run_scripts.ode_bf.p1r36_independent_b10x10_runtime import _hashes
from project.run_scripts.ode_bf.p1_scalable_batched_experiment import _model_w0_contract
from project.run_scripts.session05_ode_bf_p1r52_joint_pc_independent_fp32_dry_plan import build_plan


REPO_ROOT = Path(__file__).resolve().parents[4]
RUNTIME = REPO_ROOT / "project/run_scripts/ode_bf/p1r52_joint_pc_independent_fp32_runtime.py"
SBATCH = REPO_ROOT / "project/run_scripts/session05_ode_bf_p1r52_joint_pc_independent_fp32.sbatch"


class JointPCIndependentFP32RuntimeTest(unittest.TestCase):
    def test_exact_five_cell_role_result_matrix(self) -> None:
        self.assertEqual(len(ROLES), 5)
        self.assertEqual(CELL_LABELS, ("BASELINE-GROUP", "C0", "C1", "C2", "C3"))
        for index, role in enumerate(ROLES):
            self.assertEqual(role_for_cell(index), role)
            self.assertEqual(expected_result_name(role), RESULT_NAMES[role])
            self.assertEqual(
                expected_p1r52_sequential_result_name(
                    "llama3-8b-inst", role, scale=P1R52_B100X10_SCALE
                ),
                RESULT_NAMES[role],
            )

    def test_full_fp32_parameter_inventory_fail_closes(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
        model.requires_grad_(False)
        receipt = full_fp32_parameter_inventory(model)
        self.assertEqual(receipt["status"], "FULL_FP32_PASS")
        self.assertEqual(receipt["parameter_dtype_counts"], {"torch.float32": 4})
        self.assertEqual(receipt["bf16_parameter_tensor_count"], 0)
        self.assertEqual(receipt["fp16_parameter_tensor_count"], 0)
        self.assertEqual(receipt["quantized_parameter_count"], 0)
        model.half()
        with self.assertRaisesRegex(ODEBFContractError, "TECHNICAL_INVALID_DTYPE"):
            full_fp32_parameter_inventory(model)
        model.float()
        model.is_loaded_in_4bit = True
        with self.assertRaisesRegex(ODEBFContractError, "TECHNICAL_INVALID_DTYPE"):
            full_fp32_parameter_inventory(model)
        model.is_loaded_in_4bit = False
        next(model.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ODEBFContractError, "TECHNICAL_INVALID_DTYPE"):
            full_fp32_parameter_inventory(model)

    def test_w0_pointer_byte_and_mapping_contract(self) -> None:
        parameter = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        touched = {"layer.weight": parameter}
        contract = _model_w0_contract(touched)
        hashes = _hashes(touched)
        pointers = {"layer.weight": int(parameter.data_ptr())}
        _assert_w0(
            touched, expected_contract=contract, expected_hashes=hashes,
            expected_pointers=pointers,
        )
        with torch.no_grad():
            parameter.add_(1.0)
        with self.assertRaisesRegex(Exception, "W0"):
            _assert_w0(
                touched, expected_contract=contract, expected_hashes=hashes,
                expected_pointers=pointers,
            )

    def test_timing_observation_executes_action_once(self) -> None:
        calls = []
        with mock.patch(
            "project.run_scripts.ode_bf.p1r52_joint_pc_independent_fp32_runtime._sync"
        ) as sync:
            value, elapsed = _timed(lambda: calls.append("action") or 7)
        self.assertEqual(value, 7)
        self.assertEqual(calls, ["action"])
        self.assertEqual(sync.call_count, 2)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_low_precision_receipt_activity_fails_closed(self) -> None:
        _assert_no_low_precision_activity({
            "dtype": "torch.float32", "bf16_path_call_count": 0,
            "autocast_enabled": False,
        })
        for forged in (
            {"dtype": "torch.bfloat16"},
            {"fp16_conversion_count": 1},
            {"nested": {"autocast_enabled": True}},
        ):
            with self.assertRaisesRegex(ODEBFContractError, "TECHNICAL_INVALID_DTYPE"):
                _assert_no_low_precision_activity(forged)

    def test_dry_plan_is_independent_full_fp32_array(self) -> None:
        plan = build_plan("a" * 40, "b" * 40)
        self.assertEqual(plan["array"], "0-4%4")
        self.assertEqual(plan["cell_count"], 5)
        self.assertEqual(plan["method_count"], 6)
        self.assertEqual(plan["scientific_request_count"], 6000)
        self.assertEqual(
            [item["independent_case_count"] for item in plan["cells"]], [10] * 5
        )
        self.assertTrue(all(item["requested_dtype"] == "torch.float32" for item in plan["cells"]))
        self.assertEqual(len(plan["batch_order_sha256"]), 10)

    def test_launcher_and_runtime_prohibited_imports_absent(self) -> None:
        source = RUNTIME.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        prohibited = {
            "AcceptedPhysicalStateMaterializer",
            "AtomicBatchTransaction",
            "CachedBF16FunctionalTrial",
            "CandidateBF16FunctionalTrial",
            "assemble_effective_bf16",
            "effective_bf16_sha256",
        }
        self.assertFalse(imported & prohibited)
        sbatch = SBATCH.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-4%4", sbatch)
        self.assertIn("#SBATCH --nodelist=server1", sbatch)
        self.assertIn("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0", sbatch)


if __name__ == "__main__":
    unittest.main()
