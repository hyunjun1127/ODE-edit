from __future__ import annotations

import ast
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from project.run_scripts.ode_bf import (
    common_coldcoord_fixed_e8_runtime as runtime,
    p1_runtime,
)
from project.run_scripts.ode_bf.accounting import ComputeLedger
from project.run_scripts.ode_bf.contracts import (
    ODEBFContractError,
    ODEBFStateError,
    canonical_hash,
)
from project.run_scripts.ode_bf.fixed_e8_soft_routing import (
    FixedE8Arm,
    FixedE8Clock,
)
from project.run_scripts.ode_bf.p1_bg_soft_missing_cell_panel import (
    BG_SOFT_AMENDMENT_ID,
    BG_SOFT_INSTRUCTION_ID,
    BG_SOFT_REFERENCE_LOCK_FILE,
    BG_SOFT_RESULT_TOKEN,
    BG_SOFT_R10_CASE_ROOT,
    bg_soft_frozen_reference,
    bg_soft_initial_contract,
    common_cold_schedule,
    expected_bg_soft_result_name,
    load_and_validate_bg_soft_reference_lock,
)
from project.run_scripts.ode_bf.p1_controller import P1ControllerLock
from project.run_scripts.ode_bf.sampling import load_p1_sampling_seal


ROOT = Path(__file__).resolve().parents[4]
LOCKS = ROOT / "project/run_scripts/ode_bf/locks"


class _GeometryTokenizer:
    def __init__(self, padding_side: str) -> None:
        self.padding_side = padding_side

    @staticmethod
    def _ids(value: str) -> list[int]:
        return list(range(1, len(value.split()) + 2))

    def __call__(
        self,
        value: str | list[str],
        *,
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, object]:
        rows = [value] if isinstance(value, str) else list(value)
        ids = [self._ids(item) for item in rows]
        if not padding:
            return {"input_ids": ids if not isinstance(value, str) else ids[0]}
        maximum = max(len(item) for item in ids)
        attention: list[list[int]] = []
        padded: list[list[int]] = []
        for item in ids:
            count = maximum - len(item)
            zeros = [0] * count
            if self.padding_side == "left":
                padded.append(zeros + item)
                attention.append(zeros + [1] * len(item))
            else:
                padded.append(item + zeros)
                attention.append([1] * len(item) + zeros)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": attention}


class BgSoftMissingCellTests(unittest.TestCase):
    def _load_lock(self) -> tuple[dict[str, object], str]:
        schedule = common_cold_schedule(
            load_p1_sampling_seal(
                LOCKS / "p1r2_p_population_seal.json",
                stream_path=LOCKS / "p1r2_seqb10_stream_seal.json",
            )
        )
        return load_and_validate_bg_soft_reference_lock(
            LOCKS / BG_SOFT_REFERENCE_LOCK_FILE,
            controller_identity_sha256=P1ControllerLock().identity(),
            case_root_digest=BG_SOFT_R10_CASE_ROOT,
            schedule=schedule,
        )

    def test_bg_soft_is_explicit_and_r10_default_panel_is_frozen(self) -> None:
        self.assertEqual(runtime.CommonColdArm.BG_SOFT.value, "BG-SOFT")
        self.assertEqual(
            runtime.CommonColdArm.BG_SOFT.scale.value,
            "MATCHED_BATCH_GLOBAL_SCALE_V1",
        )
        self.assertIs(runtime.CommonColdArm.BG_SOFT.routing_arm, FixedE8Arm.SOFT)
        self.assertTrue(runtime.CommonColdArm.BG_SOFT_TARGET_HOLD.target_hold)
        self.assertIs(
            runtime.CommonColdArm.BG_SOFT_TARGET_HOLD.routing_arm,
            FixedE8Arm.SOFT,
        )
        self.assertEqual(
            runtime.R10_COMMON_COLD_ARMS,
            (
                runtime.CommonColdArm.RS_NEUTRAL,
                runtime.CommonColdArm.RS_SOFT,
                runtime.CommonColdArm.BG_NEUTRAL,
            ),
        )
        self.assertNotIn(runtime.CommonColdArm.BG_SOFT, runtime.R10_COMMON_COLD_ARMS)
        source = inspect.getsource(
            runtime.run_common_coldcoord_fixed_e8_diagnostic
        )
        self.assertIn("if bg_soft_missing_cell_mode", source)
        self.assertIn("CommonColdArm.BG_SOFT_TARGET_HOLD", source)
        self.assertIn("else R10_COMMON_COLD_ARMS", source)

    def test_reference_lock_is_rooted_and_alias_complete(self) -> None:
        value, file_sha256 = self._load_lock()
        self.assertEqual(value["instruction_id"], BG_SOFT_INSTRUCTION_ID)
        self.assertEqual(value["amendment_id"], BG_SOFT_AMENDMENT_ID)
        self.assertEqual(
            value["live_arms"], ["BG-SOFT", "BG-SOFT-TARGET-HOLD"]
        )
        self.assertEqual(value["case_root_digest"], BG_SOFT_R10_CASE_ROOT)
        self.assertEqual(set(value["aliases"]), {
            "llama3-8b-inst",
            "qwen2.5-7b-inst",
        })
        self.assertEqual(len(file_sha256), 64)
        for alias in value["aliases"]:
            initial = bg_soft_initial_contract(value, alias)
            frozen = bg_soft_frozen_reference(value, alias)
            self.assertEqual(
                initial["first_allowed_semantic_difference"],
                "SELECTED_ROUTING_VELOCITY",
            )
            self.assertEqual(frozen["alias"], alias)
            self.assertEqual(len(initial["identity_sha256"]), 64)
            self.assertEqual(len(frozen["terminal_sha256"]), 64)

        with tempfile.TemporaryDirectory() as temporary:
            malformed = dict(value)
            malformed["live_arms"] = ["BG-NEUTRAL"]
            path = Path(temporary) / "malformed.json"
            malformed.pop("root_digest", None)
            malformed["root_digest"] = canonical_hash(malformed)
            path.write_text(json.dumps(malformed, sort_keys=True), encoding="utf-8")
            schedule = common_cold_schedule(
                load_p1_sampling_seal(
                    LOCKS / "p1r2_p_population_seal.json",
                    stream_path=LOCKS / "p1r2_seqb10_stream_seal.json",
                )
            )
            with self.assertRaisesRegex(ODEBFContractError, "differs"):
                load_and_validate_bg_soft_reference_lock(
                    path,
                    controller_identity_sha256=P1ControllerLock().identity(),
                    case_root_digest=BG_SOFT_R10_CASE_ROOT,
                    schedule=schedule,
                )

    def test_six_context_objective_is_numeric_raw_free_and_observation_only(self) -> None:
        result = SimpleNamespace(
            per_request_values=torch.arange(10, dtype=torch.float32),
            objective=SimpleNamespace(value="TARGET_NEW_NLL"),
            loss=torch.tensor(4.5),
            request_order_sha256="a" * 64,
            target_span_sha256="b" * 64,
            context_sha256="c" * 64,
            context_group_sizes=(1, 5),
            context_count=6,
            model_forward_count=6,
            processed_token_count=321,
            generation_call_count=0,
        )
        payload = runtime._routing_objective_raw_free_payload(
            result,
            role="WEIGHT_ONLY_NO_RESIDUAL_HOOK_OBJECTIVE",
            action_frozen_before_evaluation=True,
        )
        self.assertEqual(payload["per_request_target_new_nll"], list(range(10)))
        self.assertEqual(payload["target_old_access_count"], 0)
        self.assertEqual(payload["controller_decision_influence_count"], 0)
        self.assertTrue(payload["action_frozen_before_evaluation"])
        forbidden = {
            "prompt",
            "subject",
            "target_new",
            "target_true",
            "target_old",
            "token_ids",
        }
        self.assertFalse(forbidden.intersection(payload))
        self.assertEqual(
            payload["identity_sha256"],
            canonical_hash(
                {key: value for key, value in payload.items() if key != "identity_sha256"}
            ),
        )

    def test_zero_positive_partial_clock_preserves_prefix_without_advance(self) -> None:
        clock = FixedE8Clock()
        first = clock.begin_field()
        clock.advance(first, scientific_observation={"ignored": True})
        second = clock.begin_field()
        del second
        receipt = runtime._partial_fixed_e8_clock_receipt(clock)
        self.assertEqual(receipt["grid_count"], 1)
        self.assertEqual(receipt["field_count"], 2)
        self.assertEqual(receipt["tau_final"], 0.125)
        self.assertEqual(receipt["termination_label"], "ZERO_POSITIVE_DIRECTION")
        self.assertEqual(receipt["scientific_retry_count"], 0)
        arm_source = inspect.getsource(runtime._run_common_arm)
        self.assertLess(
            arm_source.index("field_receipt.identity_sha256"),
            arm_source.index("break"),
        )
        self.assertIn('"candidate_count": 0', arm_source)
        self.assertIn('"clock_advance_count": 0', arm_source)
        self.assertIn("valid_prefix_snapshot_sha256", arm_source)

    def test_d1_qp_accounting_reconciles_routing_and_observation(self) -> None:
        ledger = ComputeLedger()
        ledger.increment("qp_solve", 7)
        ledger.increment("qp_certificate", 7)
        receipt = runtime._bg_soft_d1_operation_accounting(
            routing_qp=5,
            routing_backend=9,
            nohook_qp=2,
            nohook_backend=3,
            nohook_fallback=1,
            ledger=ledger,
        )
        self.assertEqual(
            receipt["actual_all_logical_qp_certificate_count"], 7
        )
        self.assertEqual(
            receipt["actual_all_optimizer_backend_invocation_count"], 12
        )
        self.assertEqual(
            receipt["actual_d1_nohook_logical_qp_certificate_count"], 2
        )
        self.assertEqual(receipt["d1_nohook_qp_role"], "OBSERVATION_ONLY")
        mismatch = ComputeLedger()
        mismatch.increment("qp_solve", 7)
        mismatch.increment("qp_certificate", 6)
        with self.assertRaisesRegex(ODEBFStateError, "accounting"):
            runtime._bg_soft_d1_operation_accounting(
                routing_qp=5,
                routing_backend=9,
                nohook_qp=2,
                nohook_backend=3,
                nohook_fallback=1,
                ledger=mismatch,
            )

    def test_heldout_lookup_geometry_uses_evaluator_active_padding(self) -> None:
        requests = []
        cases = []
        for index in range(10):
            request_sha256 = f"{index + 1:064x}"
            subject = f"S{index}"
            requests.append(
                {
                    "request_sha256": request_sha256,
                    "subject": subject,
                }
            )
            cases.append(
                SimpleNamespace(
                    request_sha256=request_sha256,
                    rewrite_prompt=f"{subject} short",
                    paraphrase_prompts=(
                        f"prefix {subject} medium",
                        f"a much longer prefix {subject} suffix",
                    ),
                    neighborhood_prompts=("neighbor row",) * 2,
                    target_new="new target",
                    target_true="true target",
                )
            )
        observed: dict[str, tuple[tuple[int, ...], ...]] = {}
        for padding_side in ("left", "right"):
            tokenizer = _GeometryTokenizer(padding_side)
            easyeditor = ModuleType("easyeditor")
            models = ModuleType("easyeditor.models")
            alphaedit = ModuleType("easyeditor.models.alphaedit")
            alphaedit.AlphaEdit_main = SimpleNamespace(
                find_fact_lookup_idx=mock.Mock(return_value=1)
            )
            easyeditor.models = models
            models.alphaedit = alphaedit
            with mock.patch.dict(
                sys.modules,
                {
                    "easyeditor": easyeditor,
                    "easyeditor.models": models,
                    "easyeditor.models.alphaedit": alphaedit,
                },
            ):
                positions, patched_rows, receipt = (
                    runtime._heldout_additive_lookup_geometry(
                        tokenizer,
                        requests,
                        cases,
                        fact_token_strategy="subject_first",
                    )
                )
            self.assertEqual(tokenizer.padding_side, padding_side)
            self.assertEqual(
                receipt["padding_side_during_geometry"], padding_side
            )
            self.assertTrue(receipt["padding_side_matches_evaluator"])
            self.assertEqual(patched_rows, (6,) * 10)
            self.assertTrue(all(len(item) == 10 for item in positions))
            observed[padding_side] = positions
        self.assertTrue(
            all(
                value == 1
                for request_positions in observed["right"]
                for value in request_positions[:6]
            )
        )
        self.assertNotEqual(observed["left"], observed["right"])

    def test_bg_soft_diagnostics_are_guarded_from_legacy_r10_path(self) -> None:
        build_source = inspect.getsource(runtime._build_common_field)
        arm_source = inspect.getsource(runtime._run_common_arm)
        diagnostic_source = inspect.getsource(
            runtime.run_common_coldcoord_fixed_e8_diagnostic
        )
        self.assertIn("if typed_zero_positive:", build_source)
        self.assertIn("if typed_zero_positive:", arm_source)
        self.assertIn(
            "if typed_zero_positive and entry_audit_state is not None:",
            arm_source,
        )
        self.assertIn(
            "pair_initial_capture if bg_soft_missing_cell_mode else None",
            diagnostic_source,
        )
        self.assertIn(
            "if bg_soft_missing_cell_mode:\n"
            "        bootstrap_stage_payload",
            diagnostic_source,
        )
        self.assertEqual(
            runtime.R10_COMMON_COLD_ARMS,
            (
                runtime.CommonColdArm.RS_NEUTRAL,
                runtime.CommonColdArm.RS_SOFT,
                runtime.CommonColdArm.BG_NEUTRAL,
            ),
        )

    def test_postfreeze_prefix_panel_does_not_rerun_frozen_references(self) -> None:
        panel_source = inspect.getsource(runtime._postfreeze_bg_soft_prefix_panel)
        self.assertIn("load_counterfact_cases_after_freeze", panel_source)
        self.assertIn("WEIGHT_ONLY_NO_RESIDUAL_HOOK_OBJECTIVE", panel_source)
        self.assertIn("snapshot.routing_payload", panel_source)
        self.assertIn('"w0_native_bg_neutral_rerun_count": 0', panel_source)
        self.assertNotIn("capture_p1_native_entry", panel_source)
        self.assertNotIn("_evaluate_native_rewrite", panel_source)
        self.assertLess(
            inspect.getsource(runtime.run_common_coldcoord_fixed_e8_diagnostic).index(
                "action_freeze ="
            ),
            inspect.getsource(runtime.run_common_coldcoord_fixed_e8_diagnostic).index(
                "_postfreeze_bg_soft_prefix_panel"
            ),
        )

    def test_arm_local_failure_is_reported_without_discarding_sibling(self) -> None:
        source = inspect.getsource(
            runtime.run_common_coldcoord_fixed_e8_diagnostic
        )
        self.assertIn(
            '"BG_SOFT_MISSING_CELL_DIAGNOSTIC_PARTIAL_TECHNICAL_TERMINAL"',
            source,
        )
        self.assertIn('"arm_failures": failures', source)
        self.assertIn('"arm_failure_count": len(failures)', source)
        self.assertIn('"completed_arm_count": len(bg_rollouts)', source)
        self.assertIn('"all_live_arms_scientifically_evaluable": not failures', source)
        self.assertIn("if not bg_rollouts", source)
        self.assertNotIn("if len(bg_rollouts) != 2", source)

    def test_dispatch_is_opt_in_and_result_namespace_is_exact(self) -> None:
        signature = inspect.signature(p1_runtime.run_p1)
        self.assertFalse(signature.parameters["bg_soft_missing_cell_mode"].default)
        self.assertEqual(
            expected_bg_soft_result_name("llama3-8b-inst"),
            "s05-bg-soft-missing-cell-p1r12-a1-llama3-8b-inst-v1",
        )
        self.assertEqual(
            BG_SOFT_RESULT_TOKEN, "bg-soft-missing-cell-p1r12-a1-v1"
        )
        tree = ast.parse(inspect.getsource(p1_runtime.run_p1))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertIn("bg_soft_missing_cell_mode", names)
        source = inspect.getsource(p1_runtime.run_p1)
        self.assertIn("expected_bg_soft_result_name", source)
        self.assertIn("BG_SOFT_REFERENCE_LOCK_FILE", source)
        self.assertIn("bg_soft_reference_lock=", source)


if __name__ == "__main__":
    unittest.main()
