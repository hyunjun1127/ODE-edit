from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from project.run_scripts import (
    session05_ode_bf_p1r37_no_persistent_freeze_b10x10_dry_plan as dry,
)
from project.run_scripts.ode_bf.p1r24_atomic_strength import (
    P1R24AliasTargetLock,
    P1R24KLResult,
)
from project.run_scripts.ode_bf.p1r36_independent_b10x10_runtime import (
    _freeze_policy_summary,
)
from project.run_scripts.ode_bf.p1r37_independent_b10x10_panel import (
    expected_result_name,
    load_and_validate_lock,
)
from project.run_scripts.ode_bf.p1r37_instantaneous_freeze import (
    FREEZE_POLICY,
    p1r37_target_step,
)
from project.run_scripts.ode_bf.scalable_batched_field import (
    ScalableBatchGlobalMetric,
)
from project.run_scripts.ode_bf.scalable_batched_model import (
    ScalableObjectiveResult,
)


ROOT = Path(__file__).resolve().parents[4]
LOCKS = ROOT / "project/run_scripts/ode_bf/locks"


def _objective(values: tuple[float, float]) -> ScalableObjectiveResult:
    return ScalableObjectiveResult(
        loss=sum(values) / len(values),
        per_request_values=values,
        request_order_sha256="a" * 64,
        target_span_sha256="b" * 64,
        target_span_identities=("c" * 64, "d" * 64),
        suffix_token_counts=(1, 1),
        model_forward_count=1,
        backward_count=1,
        processed_token_count=2,
        padded_token_count=2,
        target_gradient=torch.ones((2, 2), dtype=torch.float32),
        coefficient_gradient=None,
        plan_sha256="e" * 64,
        model_state_sha256="f" * 64,
        identity_sha256="0" * 64,
    )


def _kl() -> P1R24KLResult:
    return P1R24KLResult(
        loss=0.0,
        per_request_values=(0.0, 0.0),
        gradient=torch.zeros((2, 2), dtype=torch.float32),
        model_forward_count=1,
        backward_count=1,
        processed_token_count=2,
        padded_token_count=2,
        identity_sha256="1" * 64,
    )


class P1R37NoPersistentFreezeTest(unittest.TestCase):
    def test_lock_stream_matrix_and_names(self) -> None:
        lock, _ = load_and_validate_lock(
            LOCKS
            / "numerical_lock_s05_p1r37_no_persistent_freeze_independent_b10x10.json"
        )
        self.assertEqual(lock["freeze_policy"], FREEZE_POLICY)
        plan = dry.build_plan("child")
        self.assertEqual(plan["job_count"], 8)
        self.assertEqual(plan["independent_atomic_b10_case_count"], 80)
        self.assertEqual(plan["array_max_concurrent_gpu"], 4)
        self.assertEqual(plan["server2_project_gpu_cap"], 4)
        self.assertEqual(
            [item["method"] for item in plan["jobs"]],
            [
                "RS-P1R35-NEUTRAL",
                "RS-P1R35-SOFT",
                "BG-P1R35-NEUTRAL",
                "BG-P1R35-SOFT",
            ]
            * 2,
        )
        self.assertEqual(
            expected_result_name("llama3-8b-inst", "RS-P1R35-SOFT"),
            "s05-p1r37-no-persistent-freeze-independent-b10x10-llama3-8b-inst-rs-soft-v1",
        )

    def test_current_phi_freezes_and_later_reactivates_without_carry(self) -> None:
        z0 = torch.ones((2, 2), dtype=torch.float32)
        metric = ScalableBatchGlobalMetric.from_z0(z0, "a" * 64)
        lock = P1R24AliasTargetLock.for_alias("llama3-8b-inst")
        first = p1r37_target_step(
            z0,
            z0,
            z0,
            _objective((0.01, 0.10)),
            _kl(),
            metric,
            lock,
            step_index=0,
            frozen_mask=(False, False),
            alias="llama3-8b-inst",
            method="RS-P1R35-SOFT",
            case_index=2,
        )
        self.assertEqual(first.frozen_mask, (True, False))
        self.assertTrue(
            torch.equal(first.target_next[:, 0], z0[:, 0])
        )
        second = p1r37_target_step(
            first.target_next,
            first.target_next,
            z0,
            _objective((0.10, 0.10)),
            _kl(),
            metric,
            lock,
            step_index=1,
            frozen_mask=first.frozen_mask,
            alias="llama3-8b-inst",
            method="RS-P1R35-SOFT",
            case_index=2,
        )
        self.assertEqual(second.frozen_mask, (False, False))
        self.assertEqual(second.receipt["frozen_to_active"], [True, False])
        self.assertEqual(second.receipt["reactivation_count"], 1)
        self.assertEqual(second.receipt["carried_frozen_input_count"], 0)
        self.assertEqual(
            second.receipt["persistent_mask_decision_influence_count"], 0
        )
        self.assertFalse(
            torch.equal(second.target_next[:, 0], first.target_next[:, 0])
        )

    def test_current_below_threshold_freezes_even_after_prior_active(self) -> None:
        z0 = torch.ones((2, 2), dtype=torch.float32)
        result = p1r37_target_step(
            z0,
            z0,
            z0,
            _objective((0.01, 0.01)),
            _kl(),
            ScalableBatchGlobalMetric.from_z0(z0, "a" * 64),
            P1R24AliasTargetLock.for_alias("llama3-8b-inst"),
            step_index=0,
            frozen_mask=(False, False),
            alias="llama3-8b-inst",
            method="BG-P1R35-NEUTRAL",
            case_index=1,
        )
        self.assertEqual(result.frozen_mask, (True, True))
        self.assertEqual(result.receipt["active_to_frozen"], [True, True])

    def test_freeze_summary_requires_no_carry_and_exact_prior_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            previous = [False] * 10
            for k in range(1, 9):
                current = [k == 1] + [False] * 9
                payload = {
                    "target_update": {
                        "freeze_policy": FREEZE_POLICY,
                        "instantaneous_mask": current,
                        "prior_step_instantaneous_mask": previous,
                        "frozen_to_active": [
                            old and not new
                            for old, new in zip(previous, current, strict=True)
                        ],
                        "active_to_frozen": [
                            not old and new
                            for old, new in zip(previous, current, strict=True)
                        ],
                        "active_request_count": 10 - sum(current),
                        "persistent_mask_decision_influence_count": 0,
                        "carried_frozen_input_count": 0,
                        "predeclared_sentinel_observation": False,
                        "identity_sha256": str(k) * 64,
                    }
                }
                path = Path(directory) / f"accepted-k{k}.json"
                path.write_text(json.dumps(payload))
                paths.append(path)
                previous = current
            summary = _freeze_policy_summary(paths, expected_policy=FREEZE_POLICY)
            self.assertEqual(summary["reactivation_count"], 1)
            self.assertEqual(summary["active_to_frozen_count"], 1)
            self.assertEqual(summary["carried_frozen_input_count"], 0)

    def test_only_policy_hook_changes_parent_target_decision_surface(self) -> None:
        experiment = (
            ROOT / "project/run_scripts/ode_bf/p1_scalable_batched_experiment.py"
        ).read_text()
        runtime = (
            ROOT / "project/run_scripts/ode_bf/p1r36_independent_b10x10_runtime.py"
        ).read_text()
        policy = (
            ROOT / "project/run_scripts/ode_bf/p1r37_instantaneous_freeze.py"
        ).read_text()
        self.assertIn("p1r24_target_step_policy", experiment)
        self.assertIn("target_step_policy_factory", runtime)
        self.assertIn("tuple(False for _ in prior)", policy)
        self.assertIn("phi", policy)
        self.assertNotIn(" or item < P1R24_FREEZE_THRESHOLD", policy)
        sealed_source_head = "27fd7714532599abfe169dc9cccf83b7a66735b9"
        for path, expected in (
            (
                "project/run_scripts/ode_bf/p1r24_atomic_strength.py",
                "95ef3aff1e0b2d82df08453492a01be4d69d7e1c8795edd72c7d098979f54b0f",
            ),
            (
                "project/run_scripts/ode_bf/p1r35_full_current_residual.py",
                "539425cc1e6bbe66cc873f4c177e8587d3e5dc955fdb081b1d909b0dc8ce617b",
            ),
        ):
            payload = subprocess.run(
                ["git", "show", f"{sealed_source_head}:{path}"],
                cwd=ROOT,
                check=True,
                capture_output=True,
            ).stdout
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected)

    def test_launcher_session_resource_and_namespace_are_closed(self) -> None:
        runner = (
            ROOT
            / "project/run_scripts/session05_ode_bf_p1r37_no_persistent_freeze_b10x10.py"
        ).read_text()
        sbatch = (
            ROOT
            / "project/run_scripts/session05_ode_bf_p1r37_no_persistent_freeze_b10x10.sbatch"
        ).read_text()
        submitter = (
            ROOT
            / "project/run_scripts/session05_ode_bf_submit_p1r37_no_persistent_freeze_b10x10.py"
        ).read_text()
        self.assertIn("p1r37_independent_b10x10_method=args.method", runner)
        self.assertIn("check-session-boundary.sh", sbatch)
        self.assertIn("#SBATCH --nodelist=server2", sbatch)
        self.assertNotIn("#SBATCH --nodelist=devbox", sbatch)
        self.assertIn(
            'EXPECTED_SESSION="019fe491-954b-70a0-8ba8-0588e9f8d741"',
            sbatch,
        )
        self.assertIn("#SBATCH --array=0-7%4", sbatch)
        self.assertIn("#SBATCH --cpus-per-task=8", sbatch)
        self.assertIn("#SBATCH --mem=60416M", sbatch)
        self.assertIn("#SBATCH --time=23:59:00", sbatch)
        self.assertIn("p1r37-no-persistent-freeze-independent-b10x10", sbatch)
        self.assertNotIn("p1r36-p1r35-independent-b10x10-${MODEL}", sbatch)
        self.assertIn('P1R36_JOB_ID = "19472"', submitter)
        self.assertIn(
            'EXPECTED_SESSION = "019fe491-954b-70a0-8ba8-0588e9f8d741"',
            submitter,
        )
        self.assertIn("def _session_boundary_gate()", submitter)
        self.assertIn('mode != 0o600', submitter)
        self.assertIn('session_boundary_sha256', submitter)
        self.assertIn('"server2",', submitter)
        self.assertIn('"ReqNodeList=server2"', submitter)
        self.assertNotIn('"devbox",', submitter)
        self.assertIn("if p1r36_active > 2", submitter)
        self.assertIn("SERVER2_PROJECT_GPU_CAP = 4", submitter)
        self.assertIn("ARRAY_MAX_CONCURRENT_GPU = 4", submitter)
        self.assertIn('"ArrayTaskThrottle=4"', submitter)
        self.assertIn('"--hold"', submitter)
        self.assertIn('["scontrol", "release", job_id]', submitter)


if __name__ == "__main__":
    unittest.main()
