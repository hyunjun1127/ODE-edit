from __future__ import annotations

import inspect
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from threading import RLock

import torch

from project.run_scripts import (
    session05_ode_bf_historical_h0_sequential_dry_plan as dry,
)
from project.run_scripts.ode_bf.historical_h0_sequential_runtime import (
    CASE_COUNT,
    HISTORY_MODE,
    INSTRUCTION_ID,
    METHODS,
    _case_failure,
    _history_off_receipt,
    _restore_exact_w0,
    expected_historical_h0_result_name,
    run_historical_h0_sequential_trajectory,
)
from project.run_scripts.ode_bf.historical_h0_sequential_selection import (
    verify_historical_h0_fresh_seal,
)
from project.run_scripts.ode_bf.p1_historical_h0_sequential_panel import (
    ALL_REQUEST_ORDER,
    LOCK_SCHEMA,
    SCIENTIFIC_CHECKPOINT,
    load_and_validate_historical_h0_lock,
)
from project.run_scripts.ode_bf.p1_scalable_batched_experiment import _model_w0_contract


ROOT = Path(__file__).resolve().parents[4]
LOCKS = ROOT / "project/run_scripts/ode_bf/locks"


class IndependentAtomicB10x10Test(unittest.TestCase):
    def test_lock_seal_matrix_and_repeatable_dry_plan(self) -> None:
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
        self.assertEqual(lock["all_request_order_sha256"], ALL_REQUEST_ORDER)
        self.assertEqual(lock["atomic"]["cardinality"], "TEN_INDEPENDENT_WHOLE_B10")
        self.assertEqual(lock["atomic"]["context_ordinals"], list(range(6)))
        first = dry.build_plan("0" * 40)
        self.assertEqual(first, dry.build_plan("0" * 40))
        self.assertEqual(first["job_count"], 10)
        self.assertEqual(first["ode_job_count"], 8)
        self.assertEqual(first["official_alphaedit_job_count"], 2)
        self.assertEqual(first["independent_atomic_b10_case_count"], 100)
        self.assertEqual(first["sequential_round_count"], 0)
        self.assertEqual(first["history_mode"], HISTORY_MODE)
        self.assertEqual([item["method"] for item in first["jobs"][:5]], list(METHODS))
        self.assertEqual(len({item["result_name"] for item in first["jobs"]}), 10)

    def test_scientific_router_and_atomic_runtime_are_exact_a343(self) -> None:
        for relative in (
            "project/run_scripts/ode_bf/progress_simplex_routing.py",
            "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py",
        ):
            expected = subprocess.run(
                ["git", "show", f"{SCIENTIFIC_CHECKPOINT}:{relative}"],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
            self.assertEqual((ROOT / relative).read_bytes(), expected)

    def test_history_off_is_total_and_has_zero_influence(self) -> None:
        receipt = _history_off_receipt()
        self.assertEqual(receipt["mode"], "OFF")
        self.assertEqual(receipt["functional_h_status"], "INACTIVE_BY_HISTORY_MODE_OFF")
        self.assertEqual(receipt["structural_h_status"], "INACTIVE_BY_HISTORY_MODE_OFF")
        for key, value in receipt.items():
            if key.endswith("_count"):
                self.assertEqual(value, 0, key)
        source = inspect.getsource(run_historical_h0_sequential_trajectory)
        self.assertNotIn("history.append", source)
        self.assertNotIn("history.stage", source)
        self.assertIn("cross-case W0 state leak detected", source)

    def test_two_consecutive_cases_restore_identical_w0_and_pointers(self) -> None:
        parameter = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        touched = {"weight": parameter}
        base = {"weight": parameter.detach().clone()}
        expected = _model_w0_contract(touched)
        pointer = parameter.data_ptr()
        for delta in (5.0, -9.0):
            with torch.no_grad():
                parameter.add_(delta)
            receipt = _restore_exact_w0(
                touched,
                base,
                mutation_lock=RLock(),
                expected_contract=expected,
            )
            self.assertTrue(receipt["byte_restored_exact"])
            self.assertTrue(receipt["pointer_restored_exact"])
            self.assertEqual(parameter.data_ptr(), pointer)
            self.assertEqual(_model_w0_contract(touched), expected)

    def test_case_failure_isolated_and_never_retried_or_imputed(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            restore = {
                "pointer_restored_exact": True,
                "byte_restored_exact": True,
                "identity_sha256": "a" * 64,
            }
            first = _case_failure(
                root / "case-01",
                RuntimeError("typed failure"),
                case_index=1,
                method=METHODS[0],
                w0_restore=restore,
            )
            second_root = root / "case-02"
            second_root.mkdir()
            self.assertEqual(first["retry_count"], 0)
            self.assertTrue(first["next_case_continues"])
            self.assertEqual(first["status"], "TYPED_CASE_FAILURE_NO_RETRY_NO_IMPUTATION")
            self.assertTrue((root / "case-01/failure.json").is_file())
            self.assertTrue(second_root.is_dir())

    def test_no_b1_or_sequential_contract_is_reachable(self) -> None:
        source = inspect.getsource(run_historical_h0_sequential_trajectory)
        self.assertEqual(CASE_COUNT, 10)
        self.assertIn("len(batch) != BATCH_SIZE", source)
        self.assertIn("for case_index, requests in enumerate(stream_batches", source)
        self.assertNotIn("request_cardinality", source)
        self.assertNotIn("B1X100", source.upper())
        self.assertNotIn("outer_round", source)
        self.assertEqual(
            INSTRUCTION_ID,
            "ODEEDIT-S05-P1R23-PROGRESS-SIMPLEX-INDEPENDENT-B10X10-V1",
        )
        self.assertIn("independent-b10x10", expected_historical_h0_result_name("a", METHODS[0]))

    def test_launcher_maps_exact_ten_jobs_and_never_mentions_b1(self) -> None:
        sbatch = (
            ROOT / "project/run_scripts/session05_ode_bf_historical_h0_sequential.sbatch"
        ).read_text()
        submitter = (
            ROOT
            / "project/run_scripts/session05_ode_bf_submit_historical_h0_sequential.py"
        ).read_text()
        self.assertIn("#SBATCH --array=0-9%4", sbatch)
        self.assertIn("#SBATCH --gres=gpu:1", sbatch)
        self.assertIn("OFFICIAL-ALPHAEDIT", sbatch)
        self.assertIn("codex/p1r23-progress-simplex-independent-b10x10-v1", sbatch)
        self.assertNotIn("b1x100", sbatch.lower())
        self.assertNotIn("b1x100", submitter.lower())
        self.assertIn('"array": "0-9%4"', submitter)


if __name__ == "__main__":
    unittest.main()
