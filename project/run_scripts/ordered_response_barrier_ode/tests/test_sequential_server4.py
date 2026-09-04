from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from project.run_scripts.ordered_response_barrier_ode.contracts import TechnicalBoundary
from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256
from project.run_scripts.ordered_response_barrier_ode.runtime import FamilyRuntime
from project.run_scripts.ordered_response_barrier_ode.sequential_contracts import (
    ARM_ORDER,
    SequentialRuntimeLock,
    validate_batch_chain,
)
from project.run_scripts.ordered_response_barrier_ode.sequential_preflight import (
    MEMORY_MIB_PER_TASK,
    validate_execution_lock,
)
from project.run_scripts.ordered_response_barrier_ode.sequential_runtime import _first_gate


REPO_ROOT = Path(__file__).resolve().parents[4]


def _chain() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(10):
        records.append(
            {
                "status": "SEQUENTIAL_BATCH_TERMINAL_VALID",
                "batch_index": index,
                "request_count": 100,
                "fixed_z_compute_count": 100,
                "fixed_z_recompute_count": 0,
                "entry_weight_sha256": f"w{index}",
                "committed_weight_sha256": f"w{index + 1}",
                "entry_method_state_sha256": f"c{index}",
                "committed_method_state_sha256": f"c{index + 1}",
            }
        )
    return records


class SequentialContractTests(unittest.TestCase):
    def test_exact_science_and_chain(self) -> None:
        lock = SequentialRuntimeLock().payload()
        self.assertEqual(lock["arms"], list(ARM_ORDER))
        self.assertEqual((lock["T"], lock["N"], lock["h"]), (1.0, 4, 0.25))
        observed = validate_batch_chain(_chain(), family="AlphaEdit")
        self.assertEqual(observed["request_count"], 1_000)
        self.assertEqual(observed["weight_commit_to_next_entry"]["numerator"], 9)
        broken = _chain()
        broken[4]["entry_weight_sha256"] = "wrong"
        with self.assertRaises(TechnicalBoundary):
            validate_batch_chain(broken, family="MEMIT")

    def test_execution_lock_and_sbatch_are_exact(self) -> None:
        lock = validate_execution_lock(REPO_ROOT)
        self.assertEqual(len(lock["sha256"]), 64)
        script = (
            REPO_ROOT / "project/run_scripts/session06_orbode_sequential_server4.sbatch"
        ).read_text(encoding="utf-8")
        self.assertEqual(script.count("#SBATCH --mem="), 1)
        self.assertIn(f"#SBATCH --mem={MEMORY_MIB_PER_TASK}M", script)
        self.assertIn("#SBATCH --array=0-3%2", script)
        self.assertIn("#SBATCH --nodelist=server4", script)
        self.assertIn("#SBATCH --export=NONE", script)
        self.assertIn("#SBATCH --gres=gpu:rtx_pro_6000:1", script)


class PersistentEndpointTests(unittest.TestCase):
    def _alpha_family(self) -> FamilyRuntime:
        family = object.__new__(FamilyRuntime)
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
        family.family = "AlphaEdit"
        family.parameters = {"w": parameter}
        family.w0 = {"w": parameter.detach().clone()}
        family.w0_sha256 = tensor_set_sha256(family.w0)
        family.pointer_identity = {"w": int(parameter.data_ptr())}
        family.module = SimpleNamespace(
            P_loaded=True,
            cache_c=torch.zeros((5, 2, 2), dtype=torch.float32),
            cache_c_new=True,
        )
        family._prepared_method_state_identity = None
        family._alpha_cache_entry = None
        family._alpha_cache_entry_is_zero = False
        family._cov_versions = {}
        family._capture_persistent_endpoint = True
        family._captured_endpoint_weights = None
        family._captured_endpoint_method_state = None
        family._captured_endpoint_sha256 = None
        return family

    def test_alpha_endpoint_capture_restore_and_commit(self) -> None:
        family = self._alpha_family()
        family.bind_existing_method_state()
        entry_state = family.method_state_identity()
        with torch.no_grad():
            family.parameters["w"].add_(3.0)
            family.module.cache_c.add_(2.0)
        expected = tensor_set_sha256(family.parameters)
        family._capture_endpoint_for_persistence()
        family.reset_entry()
        self.assertEqual(tensor_set_sha256(family.parameters), family.w0_sha256)
        self.assertEqual(family.method_state_identity(), entry_state)
        receipt = family.commit_captured_endpoint(expected_sha256=expected)
        self.assertEqual(receipt["committed_weight_sha256"], expected)
        self.assertEqual(receipt["fixed_z_recompute_count"], 0)
        self.assertEqual(float(family.module.cache_c.mean()), 2.0)

    def test_memit_covariance_mutation_fails_closed(self) -> None:
        family = object.__new__(FamilyRuntime)
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        covariance = torch.eye(2, dtype=torch.float32)
        family.family = "MEMIT"
        family.parameters = {"w": parameter}
        family.w0 = {"w": parameter.detach().clone()}
        family.w0_sha256 = tensor_set_sha256(family.w0)
        family.pointer_identity = {"w": int(parameter.data_ptr())}
        family.module = SimpleNamespace(COV_CACHE={("m", "x"): covariance})
        family._prepared_method_state_identity = None
        family._alpha_cache_entry = None
        family._alpha_cache_entry_is_zero = False
        family._cov_versions = {}
        family.bind_existing_method_state()
        covariance.add_(1.0)
        with self.assertRaises(TechnicalBoundary):
            family.reset_entry()


class FirstGateTests(unittest.TestCase):
    def test_raw_free_telemetry_shape_is_consumed(self) -> None:
        first = {
            "committed_weight_sha256": "a" * 64,
            "committed_method_state_sha256": "b" * 64,
        }
        second = {
            "entry_weight_sha256": "a" * 64,
            "entry_method_state_sha256": "b" * 64,
        }
        qcl = {
            "identity_sha256": "c" * 64,
            "fixed_z": {"compute_count": 100, "recompute_count": 0},
            "endpoint": {
                "mechanism_telemetry": {
                    "telemetry": {
                        "steps": [
                            {
                                "sweep": 0,
                                "layer": 4,
                                "visit_ordinal": 0,
                                "built_state_version": 0,
                                "resulting_state_version": 1,
                            }
                        ]
                    }
                },
                "evaluation": {"locality": {"canonical_ns": {"prompt_denominator": 1000}}},
            },
        }
        with tempfile.TemporaryDirectory() as raw:
            value = _first_gate(
                root=Path(raw),
                cell_id=0,
                model_alias="llama3-8b-inst",
                writer_family="MEMIT",
                source_head="h",
                source_tree="t",
                official_records=[first, second],
                qcl_first_batch=qcl,
            )
            self.assertEqual(value["status"], "FIRST_VALID_SEQUENTIAL_CONTINUITY_GATE_PASS")
            self.assertEqual(
                json.loads((Path(raw) / "first-valid-gate.json").read_text())["identity_sha256"],
                value["identity_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
