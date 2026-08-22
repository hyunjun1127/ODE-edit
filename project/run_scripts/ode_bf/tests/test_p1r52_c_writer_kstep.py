from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

from project.run_scripts.ode_bf.contracts import ODEBFContractError
from project.run_scripts.ode_bf.p1_evaluator import EndpointActionFreeze
from project.run_scripts.ode_bf import p1_scalable_batched_experiment as experiment
from project.run_scripts.ode_bf import p1r52_c_writer_kstep as kstep
from project.run_scripts.ode_bf import p1r52_c_writer_kstep_independent as phase2


class CKStepContractTest(unittest.TestCase):
    def test_three_arm_map(self) -> None:
        self.assertEqual(kstep.ARMS, ("C0-KSTEP", "C1-KSTEP", "C3-KSTEP"))
        self.assertEqual(tuple(phase2.arm_for_role(phase2.role_for_cell(i)) for i in range(3)), kstep.ARMS)

    def test_closed_loop_hook_precedes_legacy_writer(self) -> None:
        source = inspect.getsource(experiment._run_ode_arm)
        branch = source.index("if p1r52_c_kstep_writer is not None:")
        legacy = source.index("field = build_scalable_dynamic_field(", branch)
        self.assertLess(branch, legacy)
        self.assertIn("current_terminal = next_physical.terminal_z.clone()", source[branch:legacy])
        self.assertIn("p1r52_state = p1r52_pending_state", source[branch:legacy])

    def test_writer_reuses_released_implementations(self) -> None:
        source = inspect.getsource(kstep)
        self.assertIn("_run_pc_arm_fp32", source)
        self.assertIn("run_official_native_apply", source)
        self.assertIn("expected_native_compute_z_call_count=0", source)
        self.assertIn("route_recomputed_at_current_k", source)
        for prohibited in ("AcceptedPhysicalStateMaterializer", "AtomicBatchTransaction", "assemble_effective_bf16", "CachedBF16FunctionalTrial"):
            self.assertNotIn(prohibited, source)

    def test_phase2_is_independent_and_cache_off(self) -> None:
        source = inspect.getsource(phase2)
        self.assertIn('alpha_cache_status="ALPHA_CACHE_OFF_CONTROL"', source)
        self.assertIn('"cross_case_state_count": 0', source)
        self.assertIn("_restore(touched, base_values", source)
        self.assertIn("K_writer_call_count", source)

    def test_final_v6_receipts_and_slice_are_exactly_bound(self) -> None:
        rooted = phase2._final_v6_rooted_receipt()
        reference = phase2._final_v6_case_reference(
            "C0-KSTEP", 1, rooted_receipt=rooted
        )
        self.assertEqual(reference["slice_identity"]["stream_root"], phase2.STREAM_ROOT)
        self.assertEqual(reference["slice_identity"]["stream_order"], phase2.STREAM_ORDER)
        self.assertEqual(reference["slice_identity"]["case_index"], 1)

    def test_forged_final_v6_receipt_rejects(self) -> None:
        rooted = phase2._final_v6_rooted_receipt()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            forged = dict(rooted)
            forged["stream_root"] = "0" * 64
            path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "identity differs"):
                phase2._verified_json_receipt(path, label="forged")

    def test_execution_raw_free_payload_preserves_identity_fields(self) -> None:
        body = {
            "arm": "C0-KSTEP",
            "step_index": 0,
            "selected_target_sha256": "a" * 64,
            "selected_bridge_identity": "b" * 64,
            "entry_weight_sha256": {"w": "c" * 64},
            "commit_weight_sha256": {"w": "d" * 64},
            "route_receipt": {"identity_sha256": "e" * 64},
            "writer_receipt": {"identity_sha256": "f" * 64},
            "prefix_capture_receipts": [],
            "metrics": {"heldout_evaluator_decision_influence_count": 0},
            "compute": {"numeric_storage_cast_count": 0},
            "alpha_cache_status": "ALPHA_CACHE_OFF_CONTROL",
            "current_K_writer_affects_next_target": True,
        }
        receipt = kstep.CKStepExecution(
            body["arm"],
            body["step_index"],
            body["selected_target_sha256"],
            body["selected_bridge_identity"],
            body["entry_weight_sha256"],
            body["commit_weight_sha256"],
            body["route_receipt"],
            body["writer_receipt"],
            (),
            body["metrics"],
            body["compute"],
            body["alpha_cache_status"],
            body["current_K_writer_affects_next_target"],
            kstep.canonical_hash(body),
        )
        observed = receipt.raw_free_payload()
        claimed = observed.pop("identity_sha256")
        self.assertEqual(claimed, kstep.canonical_hash(observed))

    def test_kstep_freeze_repairs_nonterminal_completed_slot_contract(self) -> None:
        with self.assertRaisesRegex(ODEBFContractError, "Native/K8 policy"):
            EndpointActionFreeze(
                arm="C0-KSTEP-k1",
                sequential_batch=0,
                request_order_sha256="a" * 64,
                selected_snapshot_sha256="b" * 64,
                fixed_budget_slots_completed=1,
            )
        identities = []
        for step_index in range(8):
            freeze = kstep.KStepEndpointActionFreeze(
                arm=f"C0-KSTEP-k{step_index + 1}",
                sequential_batch=step_index,
                request_order_sha256="a" * 64,
                selected_snapshot_sha256="b" * 64,
                fixed_budget_slots_completed=step_index + 1,
            )
            self.assertTrue(freeze.action_frozen)
            self.assertEqual(freeze.fixed_budget_slots_completed, step_index + 1)
            identities.append(freeze.identity())
        self.assertEqual(len(set(identities)), 8)

    def test_kstep_freeze_rejects_stale_or_future_action_count(self) -> None:
        for completed in (0, 2):
            with self.assertRaisesRegex(ODEBFContractError, "completed-action count"):
                kstep.KStepEndpointActionFreeze(
                    arm="C1-KSTEP-k1",
                    sequential_batch=0,
                    request_order_sha256="a" * 64,
                    selected_snapshot_sha256="b" * 64,
                    fixed_budget_slots_completed=completed,
                )


if __name__ == "__main__":
    unittest.main()
