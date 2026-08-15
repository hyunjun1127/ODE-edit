from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import subprocess
import unittest

from project.run_scripts import session05_ode_bf_p2r4_phaseb_atomic_dry_plan as dry
from project.run_scripts.ode_bf.p1_runtime import run_p1
from project.run_scripts.ode_bf.p2r1_rms_tangent_target import p2r1_target_update
from project.run_scripts.ode_bf.p2r4_clamp_off_target import (
    p2r4_clamp_off_target_update,
)
from project.run_scripts.ode_bf.p2r4_phaseb_atomic_panel import (
    LOCK_FILE,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf import p2r4_phaseb_atomic_runtime as runtime
from project.run_scripts.ode_bf.p2r4_phaseb_receipts import (
    committed_outer_h_receipt,
)


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "project/run_scripts/ode_bf"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha(commit: str, relative_path: str) -> str:
    content = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return hashlib.sha256(content).hexdigest()


class P2R4PhaseBExecutionContractTest(unittest.TestCase):
    def test_lock_binds_exact_writer_and_phase_a_target_bytes(self) -> None:
        lock, _ = load_and_validate_lock(PACKAGE / "locks" / LOCK_FILE)
        sealed_writer = lock["exact_p2r2_v2_source_head"]
        self.assertEqual(
            _git_blob_sha(
                sealed_writer,
                "project/run_scripts/ode_bf/p2r2_residual_transport_writer.py",
            ),
            lock["writer_module_sha256"],
        )
        self.assertEqual(
            _git_blob_sha(
                sealed_writer,
                "project/run_scripts/ode_bf/scalable_batched_field.py",
            ),
            lock["writer_field_module_sha256"],
        )
        self.assertEqual(
            _git_blob_sha(
                sealed_writer,
                "project/run_scripts/ode_bf/p2r2_atomic_runtime.py",
            ),
            lock["sealed_writer_runtime_sha256"],
        )
        self.assertEqual(
            _sha(PACKAGE / "p2r4_clamp_off_target.py"),
            lock["clamp_off_policy_module_sha256"],
        )

    def test_target_policy_dispatch_is_exact_on_or_off(self) -> None:
        self.assertIs(runtime._target_update_for_policy("ON"), p2r1_target_update)
        self.assertIs(
            runtime._target_update_for_policy("OFF"),
            p2r4_clamp_off_target_update,
        )
        with self.assertRaises(Exception):
            runtime._target_update_for_policy("OTHER")

    def test_runtime_keeps_current_w_refresh_and_24_continuous_microsteps(self) -> None:
        source = (PACKAGE / "p2r4_phaseb_atomic_runtime.py").read_text()
        self.assertIn("for outer in range(8):", source)
        self.assertIn("for inner in range(P2R1_MICROSTEPS_PER_OUTER_STATE):", source)
        self.assertIn("current_terminal = physical.terminal_z.clone()", source)
        self.assertIn("state = update.state_next", source)
        self.assertIn('"rms_reset_count": 0', source)
        self.assertIn("update = target_update(", source)

    def test_writer_interfaces_are_imported_not_reimplemented(self) -> None:
        source = (PACKAGE / "p2r4_phaseb_atomic_runtime.py").read_text()
        for name in (
            "build_proposal_quadratics",
            "measure_request_layer_response",
            "p2r2_waypoint_factors",
            "solve_p2r2_routing",
        ):
            self.assertIn(name, source)
            self.assertNotIn(f"def {name}(", source)
        self.assertEqual(source.count("materializer.materialize("), 1)
        self.assertIn('"one_joint_materialization_count": 1', source)
        receipt_source = (PACKAGE / "p2r4_phaseb_receipts.py").read_text()
        self.assertIn('"physical_h_application_count": 0', receipt_source)

    def test_full_current_residual_and_same_strength_writer_are_unchanged(self) -> None:
        sealed = (PACKAGE / "p2r2_atomic_runtime.py").read_text()
        phase_b = (PACKAGE / "p2r4_phaseb_atomic_runtime.py").read_text()
        expression = "current_target - current_terminal"
        self.assertIn(expression, sealed)
        self.assertIn(expression, phase_b)
        writer = (PACKAGE / "p2r2_residual_transport_writer.py").read_text()
        self.assertIn("requestwise_no_weaker", writer)
        self.assertNotIn("remaining_horizon", phase_b)
        self.assertNotIn("backtracking", inspect.getsource(runtime._target_update_for_policy))

    def test_dry_plan_is_all_eight_cells_with_task_max_two(self) -> None:
        lock, _ = load_and_validate_lock(PACKAGE / "locks" / LOCK_FILE)
        plan = dry.build_plan(lock["exact_p2r2_v2_source_head"], case_count=10)
        self.assertEqual(plan["job_count"], 4)
        self.assertEqual(plan["arm_count"], 8)
        self.assertEqual(plan["attempt_count"], 80)
        self.assertEqual(plan["request_attempt_count"], 800)
        self.assertEqual(plan["array_max_concurrent_gpu"], 2)
        identities = {
            (job["alias"], job["clamp_policy"], tuple(job["arms"]))
            for job in plan["jobs"]
        }
        self.assertEqual(len(identities), 4)

    def test_launcher_mapping_and_resources_are_exact(self) -> None:
        sbatch = (ROOT / "project/run_scripts/session05_ode_bf_p2r4_phaseb_atomic.sbatch").read_text()
        submitter = (ROOT / "project/run_scripts/session05_ode_bf_submit_p2r4_phaseb_atomic.py").read_text()
        self.assertIn("#SBATCH --array=0-3%2", sbatch)
        self.assertIn("#SBATCH --cpus-per-task=8", sbatch)
        self.assertIn("#SBATCH --mem=65000M", sbatch)
        self.assertIn("#SBATCH --gres=gpu:1", sbatch)
        self.assertIn("#SBATCH --nodelist=server2", sbatch)
        self.assertNotIn("devbox", sbatch)
        self.assertIn('"-w", "server2"', submitter)
        self.assertIn('"--nodelist", "server2"', submitter)
        self.assertIn('"ReqNodeList=server2"', submitter)
        self.assertNotIn("devbox", submitter)
        self.assertIn("readonly CLAMP_POLICIES=(ON OFF ON OFF)", sbatch)
        self.assertIn("--clamp-policy \"${CLAMP_POLICY}\"", sbatch)

    def test_p1_entrypoint_requires_explicit_clamp_policy(self) -> None:
        params = inspect.signature(run_p1).parameters
        self.assertIn("p2r4_phaseb_case_count", params)
        self.assertIn("p2r4_phaseb_clamp_policy", params)
        self.assertIn("p2r4_phaseb_attempt_suffix", params)

    def test_forbidden_runtime_counts_are_explicitly_zero(self) -> None:
        source = (PACKAGE / "p2r4_phaseb_atomic_runtime.py").read_text()
        for token in (
            '"writer_interface_mutation_count": 0',
            '"writer_equation_mutation_count": 0',
            '"off_fallback_addition_count": 0',
            '"target_retry_count": 0',
            '"target_backtracking_count": 0',
            '"target_early_stop_count": 0',
        ):
            self.assertIn(token, source)

    def test_outer_h_receipt_separates_transition_from_numeric_h(self) -> None:
        receipt = committed_outer_h_receipt(committed_joint_transition=True)
        self.assertEqual(receipt["outer_h_application_count"], 1)
        self.assertEqual(receipt["applied_coordinate_update_count"], 1)
        self.assertEqual(receipt["writer_h_numeric_multiplication_count"], 0)
        self.assertEqual(receipt["physical_h_application_count"], 0)
        self.assertEqual(receipt["second_h_application_count"], 0)
        self.assertEqual(receipt["residual_presplit_count"], 0)
        self.assertEqual(receipt["remaining_division_count"], 0)
        self.assertEqual(receipt["semantic_debt_input_count"], 0)
        self.assertEqual(receipt["decision_influence_count"], 0)

    def test_outer_h_receipt_fails_before_committed_transition(self) -> None:
        with self.assertRaises(Exception):
            committed_outer_h_receipt(committed_joint_transition=False)

    def test_outer_h_receipt_is_emitted_after_joint_materialization(self) -> None:
        source = (PACKAGE / "p2r4_phaseb_atomic_runtime.py").read_text()
        materialize = source.index("materializer.materialize(")
        receipt = source.index("outer_h_receipt = committed_outer_h_receipt(")
        payload = source.index('"outer_h_receipt_identity_sha256"')
        self.assertLess(materialize, receipt)
        self.assertLess(receipt, payload)


if __name__ == "__main__":
    unittest.main()
