from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from project.run_scripts import (
    session05_ode_bf_p1r40_velocity_decay_b10x10_dry_plan as dry,
)
from project.run_scripts.session05_ode_bf_submit_p1r40_velocity_decay_b10x10 import (
    _gpu_count,
)
from project.run_scripts.ode_bf.artifacts import sha256_file
from project.run_scripts.ode_bf.p1_scalable_batched_experiment import _run_ode_arm
from project.run_scripts.ode_bf.p1r40_independent_b10x10_panel import (
    LOCK_FILE,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r40_semantic_deficit_velocity_decay import (
    apply_p1r40_semantic_deficit_velocity_decay,
    select_p1r40_target_proposal,
)


ROOT = Path(__file__).resolve().parents[4]


class P1R40ExecutionContractTests(unittest.TestCase):
    def test_lock_parent_stream_and_single_delta(self) -> None:
        lock, _ = load_and_validate_lock(
            ROOT / "project/run_scripts/ode_bf/locks" / LOCK_FILE
        )
        self.assertEqual(
            lock["accepted_p1r38_scientific_checkpoint"],
            "6f48ac2800b257ceb16368fff5137212dfa6037f",
        )
        self.assertEqual(
            lock["fresh_stream_root"],
            "74d6896535fe46211e3f11d3d9b420c1ab36f1d503ee81f9eaace23c2fcb89e6",
        )
        self.assertEqual(lock["request_gradient_formula"], "B*BATCH_MEAN_TARGET_NEW_GRADIENT")
        self.assertEqual(lock["gamma_application"], "TARGET_DISPLACEMENT_EXACTLY_ONCE")
        self.assertEqual(lock["additional_forward_count"], 0)
        self.assertEqual(lock["additional_backward_count"], 0)
        self.assertEqual(lock["additional_materialization_count"], 0)

    def test_parent_target_and_writer_sources_are_unchanged(self) -> None:
        expected = {
            "project/run_scripts/ode_bf/p1r38_perrequest_target.py": "2ee532a3b4293cafb99116b1a587a18f4eef2489deb26b6e5cb3afcd8e3795a3",
            "project/run_scripts/ode_bf/p1r35_full_current_residual.py": "539425cc1e6bbe66cc873f4c177e8587d3e5dc955fdb081b1d909b0dc8ce617b",
            "project/run_scripts/ode_bf/p1r34_w_anchored_finite_demand.py": "b2c59ac75798b9d70fbbc370c6d9f805d33cbeb8f683be81bb9effe7d1dbfe53",
        }
        for relative, digest in expected.items():
            self.assertEqual(sha256_file(ROOT / relative), digest)

    def test_controller_is_tensor_only_and_reuses_p1r38_selection(self) -> None:
        apply_source = inspect.getsource(apply_p1r40_semantic_deficit_velocity_decay)
        select_source = inspect.getsource(select_p1r40_target_proposal)
        for forbidden in (
            "model(",
            ".backward(",
            "evaluate_scalable",
            "heldout",
            "materialize",
        ):
            self.assertNotIn(forbidden, apply_source)
            self.assertNotIn(forbidden, select_source)
        self.assertIn("request_gradient64 = batched_gradient64 * request_count", apply_source)
        self.assertIn("select_p1r38_target_proposal(", select_source)
        self.assertIn("gamma_target_to_writer_application_count", apply_source)

    def test_runtime_applies_gamma_before_existing_endpoint_forward(self) -> None:
        source = inspect.getsource(_run_ode_arm)
        apply_index = source.index("apply_p1r40_semantic_deficit_velocity_decay(")
        endpoint_index = source.index("endpoint_state =", apply_index)
        selection_index = source.index("select_p1r40_target_proposal", endpoint_index)
        self.assertLess(apply_index, endpoint_index)
        self.assertLess(endpoint_index, selection_index)
        self.assertIn("p1r38_state = P1R38AdamState.zero", source)
        self.assertIn("P1R40_SEMANTIC_DEFICIT_VELOCITY_DECAY_K8_COMPLETE", source)

    def test_dry_plan_has_four_nonaliased_cells_and_forty_cases(self) -> None:
        plan = dry.build_plan(
            "0" * 40, repository_root=ROOT, attempt_suffix="tech-r1"
        )
        self.assertEqual(plan["job_count"], 4)
        self.assertEqual(plan["independent_atomic_b10_case_count"], 40)
        self.assertEqual(plan["maximum_accepted_step_count"], 320)
        self.assertEqual(len({job["result_name"] for job in plan["jobs"]}), 4)
        self.assertTrue(
            all("-tech-r1-v1" in job["result_name"] for job in plan["jobs"])
        )
        self.assertEqual(
            [job["method"] for job in plan["jobs"]],
            [
                "SDVD-P1R38-NEUTRAL",
                "SDVD-P1R38-SOFT",
                "SDVD-P1R38-NEUTRAL",
                "SDVD-P1R38-SOFT",
            ],
        )

    def test_scheduler_gpu_count_is_allocation_aware(self) -> None:
        self.assertEqual(_gpu_count("gres/gpu:1"), 1)
        self.assertEqual(_gpu_count("gpu:a100:2"), 2)
        self.assertEqual(_gpu_count("cpu:8"), 0)
        sbatch = (
            ROOT
            / "project/run_scripts/session05_ode_bf_p1r40_velocity_decay_b10x10.sbatch"
        ).read_text(encoding="utf-8")
        submitter = (
            ROOT
            / "project/run_scripts/session05_ode_bf_submit_p1r40_velocity_decay_b10x10.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--nodelist=server2", sbatch)
        self.assertNotIn("--nodelist=devbox", sbatch)
        self.assertIn('ATTEMPT_SUFFIX = "tech-r1"', submitter)
        self.assertIn('"server2",', submitter)


if __name__ == "__main__":
    unittest.main()
