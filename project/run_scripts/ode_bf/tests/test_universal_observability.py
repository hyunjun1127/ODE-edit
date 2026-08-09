from __future__ import annotations

import unittest
import inspect
from pathlib import Path
from unittest import mock

from project.run_scripts.ode_bf import (
    ode_bf_observability as observability,
    p1_universal_observability_panel as panel,
)
from project.run_scripts.ode_bf.contracts import (
    MODEL_ALIASES,
    ODEBFContractError,
    canonical_hash,
)
from project.run_scripts.ode_bf.common_coldcoord_fixed_e8_runtime import (
    CommonColdArm,
    _postfreeze_bg_soft_prefix_panel,
    _run_common_arm,
    run_common_coldcoord_fixed_e8_diagnostic,
)
from project.run_scripts.ode_bf.fixed_e8_soft_routing import (
    FixedE8Arm,
    _maximum_progress,
)
from project.run_scripts.ode_bf.p1_runtime import run_p1


class UniversalObservabilityTests(unittest.TestCase):
    def test_runtime_arm_registry_and_explicit_opt_in_are_exact(self) -> None:
        self.assertEqual(
            tuple(item.value for item in CommonColdArm),
            tuple(observability.ODE_BF_ARM_REGISTRY),
        )
        for arm_id, arm_contract in observability.ODE_BF_ARM_REGISTRY.items():
            with self.subTest(arm_id=arm_id):
                runtime_arm = CommonColdArm(arm_id)
                self.assertEqual(runtime_arm.target_hold, arm_contract.target_hold)
                self.assertEqual(
                    runtime_arm.routing_arm,
                    (
                        FixedE8Arm.SOFT
                        if arm_contract.layer_routing.value == "SOFT"
                        else FixedE8Arm.NEUTRAL
                    ),
                )

        run_p1_parameter = inspect.signature(run_p1).parameters[
            "universal_observability_cell"
        ]
        runtime_parameter = inspect.signature(
            run_common_coldcoord_fixed_e8_diagnostic
        ).parameters["universal_observability_cell"]
        self.assertIsNone(run_p1_parameter.default)
        self.assertIsNone(runtime_parameter.default)

    def test_runtime_persists_d1_before_the_zero_positive_guard(self) -> None:
        source = inspect.getsource(_run_common_arm)
        self.assertLess(
            source.index("_build_common_field("),
            source.index("if target_velocity is None:"),
        )
        build_source = inspect.getsource(
            __import__(
                "project.run_scripts.ode_bf.common_coldcoord_fixed_e8_runtime",
                fromlist=["_build_common_field"],
            )._build_common_field
        )
        self.assertLess(
            build_source.index("persisted = recorder.field(field_payload)"),
            build_source.index("return ("),
        )
        self.assertIn(
            "field_persisted_before_positive_guard",
            build_source,
        )

    def test_postfreeze_extension_preserves_legacy_default_contract(self) -> None:
        parameters = inspect.signature(
            _postfreeze_bg_soft_prefix_panel
        ).parameters
        self.assertEqual(parameters["output_token"].default, "bg-soft")
        self.assertIsNone(parameters["observability_contract"].default)
        self.assertIsNone(parameters["bootstrap_targets"].default)
        self.assertIsNone(parameters["controller_lookup_positions"].default)

    def test_d1_nohook_certificate_is_detached_without_legacy_relaxation(self) -> None:
        maximum_parameters = inspect.signature(_maximum_progress).parameters
        self.assertIs(maximum_parameters["fail_closed"].default, True)
        source = inspect.getsource(
            __import__(
                "project.run_scripts.ode_bf.common_coldcoord_fixed_e8_runtime",
                fromlist=["_build_common_field"],
            )._build_common_field
        )
        self.assertIn("fail_closed=False", source)
        self.assertIn("NOHOOK_NUMERIC_UNAVAILABLE", source)
        self.assertIn("NOT_APPLICABLE_NO_POSITIVE_DIRECTION", source)
        self.assertIn("nohook_numerical_certificate_required", source)
        self.assertIn("nohook_controller_decision_influence_count", source)

    def test_exact_eight_arm_registry_covers_the_locked_factorial(self) -> None:
        expected_ids = (
            "RS-NEUTRAL",
            "RS-NEUTRAL-TARGET-HOLD",
            "RS-SOFT",
            "RS-SOFT-TARGET-HOLD",
            "BG-NEUTRAL",
            "BG-NEUTRAL-TARGET-HOLD",
            "BG-SOFT",
            "BG-SOFT-TARGET-HOLD",
        )
        expected_factors = (
            ("RS", "NEUTRAL", "DYNAMIC_TARGET"),
            ("RS", "NEUTRAL", "TARGET_HOLD"),
            ("RS", "SOFT", "DYNAMIC_TARGET"),
            ("RS", "SOFT", "TARGET_HOLD"),
            ("BG", "NEUTRAL", "DYNAMIC_TARGET"),
            ("BG", "NEUTRAL", "TARGET_HOLD"),
            ("BG", "SOFT", "DYNAMIC_TARGET"),
            ("BG", "SOFT", "TARGET_HOLD"),
        )

        self.assertEqual(
            tuple(arm.arm_id for arm in observability.ODE_BF_FACTORIAL_ARMS),
            expected_ids,
        )
        self.assertEqual(tuple(observability.ODE_BF_ARM_REGISTRY), expected_ids)
        self.assertEqual(
            tuple(
                (
                    arm.target_allocation.value,
                    arm.layer_routing.value,
                    arm.target_dynamics.value,
                )
                for arm in observability.ODE_BF_FACTORIAL_ARMS
            ),
            expected_factors,
        )
        self.assertEqual(len(observability.ODE_BF_ARM_REGISTRY), 8)

    def test_dynamic_and_target_hold_arms_preserve_the_other_roles(self) -> None:
        expected_pairs = {
            "RS-NEUTRAL": "RS-NEUTRAL-TARGET-HOLD",
            "RS-SOFT": "RS-SOFT-TARGET-HOLD",
            "BG-NEUTRAL": "BG-NEUTRAL-TARGET-HOLD",
            "BG-SOFT": "BG-SOFT-TARGET-HOLD",
        }

        for dynamic_id, hold_id in expected_pairs.items():
            with self.subTest(dynamic_id=dynamic_id):
                self.assertEqual(
                    observability.paired_arm_ids(dynamic_id),
                    (dynamic_id, hold_id),
                )
                dynamic = observability.require_observability_arm(dynamic_id)
                hold = observability.require_observability_arm(hold_id)
                self.assertFalse(dynamic.target_hold)
                self.assertTrue(hold.target_hold)
                self.assertEqual(dynamic.dynamic_peer_id, dynamic_id)
                self.assertEqual(dynamic.hold_peer_id, hold_id)
                self.assertEqual(hold.dynamic_peer_id, dynamic_id)
                self.assertEqual(hold.hold_peer_id, hold_id)
                self.assertIs(hold.target_allocation, dynamic.target_allocation)
                self.assertIs(hold.layer_routing, dynamic.layer_routing)

    def test_all_new_target_hold_arms_preserve_scale_and_routing(self) -> None:
        expected = {
            "RS-NEUTRAL-TARGET-HOLD": ("ROBUST_SHARED_REQUEST_SCALE_V1", FixedE8Arm.NEUTRAL),
            "RS-SOFT-TARGET-HOLD": ("ROBUST_SHARED_REQUEST_SCALE_V1", FixedE8Arm.SOFT),
            "BG-NEUTRAL-TARGET-HOLD": ("MATCHED_BATCH_GLOBAL_SCALE_V1", FixedE8Arm.NEUTRAL),
        }
        for arm_id, (scale, routing) in expected.items():
            with self.subTest(arm_id=arm_id):
                arm = CommonColdArm(arm_id)
                self.assertTrue(arm.target_hold)
                self.assertEqual(arm.scale.value, scale)
                self.assertIs(arm.routing_arm, routing)

    def test_universal_runtime_freezes_target_and_uses_independent_weight_clock(self) -> None:
        source = inspect.getsource(_run_common_arm)
        self.assertIn("if arm.target_hold", source)
        self.assertIn("target_trial = (", source)
        self.assertIn("current_target.clone()", source)
        self.assertIn('"target_clock_advance_count": 0', source)
        self.assertIn('"weight_clock_advance_count": len(snapshots)', source)
        self.assertIn('"target_hash_invariant"', source)
        self.assertIn('"scientific_retry_count": 0', source)

    def test_universal_runtime_enforces_pair_identity_and_postfreeze_order(self) -> None:
        source = inspect.getsource(run_common_coldcoord_fixed_e8_diagnostic)
        self.assertIn("universal dynamic/hold initial contract differs", source)
        self.assertIn("universal dynamic/hold selected k0 contract differs", source)
        self.assertLess(
            source.index("action_freeze ="),
            source.index("_postfreeze_bg_soft_prefix_panel("),
        )
        self.assertIn("bootstrap_targets=bootstrap_targets", source)
        self.assertIn("controller_lookup_positions=lookup_positions", source)
        self.assertIn('"postfreeze_entry_snapshot_count"', source)

    def test_observation_receipts_have_zero_decision_influence(self) -> None:
        receipt = observability.universal_observability_contract_receipt(
            instruction_id=panel.UNIVERSAL_OBS_INSTRUCTION_ID,
            amendment_id=panel.UNIVERSAL_OBS_AMENDMENT_ID,
            cell_id="RS-SOFT",
        )

        self.assertEqual(receipt["factorial_shape"], [2, 2, 2])
        self.assertEqual(receipt["live_arms"], ["RS-SOFT", "RS-SOFT-TARGET-HOLD"])
        self.assertTrue(receipt["d1_d2_d3_observation_only"])
        self.assertFalse(receipt["scientific_promotion_authorized"])
        receipt_counts = {
            key: value
            for key, value in receipt.items()
            if key.endswith("_decision_influence_count")
        }
        self.assertEqual(
            receipt_counts,
            {"observability_total_decision_influence_count": 0},
        )
        self.assertEqual(receipt["observability_state_mutation_count"], 0)
        self.assertEqual(receipt["observability_rng_advance_count"], 0)
        self.assertEqual(receipt["heldout_open_before_action_freeze_count"], 0)
        self.assertEqual(
            receipt["identity_sha256"],
            canonical_hash(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "identity_sha256"
                }
            ),
        )

        for arm in observability.ODE_BF_FACTORIAL_ARMS:
            with self.subTest(arm_id=arm.arm_id):
                arm_receipt = arm.raw_free_payload()
                arm_counts = {
                    key: value
                    for key, value in arm_receipt.items()
                    if key.endswith("_decision_influence_count")
                }
                self.assertTrue(arm_counts)
                self.assertTrue(all(value == 0 for value in arm_counts.values()))
                self.assertEqual(
                    arm_receipt["identity_sha256"],
                    canonical_hash(
                        {
                            key: value
                            for key, value in arm_receipt.items()
                            if key != "identity_sha256"
                        }
                    ),
                )

    def test_missing_or_unknown_schema_and_unknown_arms_fail_closed(self) -> None:
        receipt = observability.universal_observability_contract_receipt(
            instruction_id=panel.UNIVERSAL_OBS_INSTRUCTION_ID,
            amendment_id=panel.UNIVERSAL_OBS_AMENDMENT_ID,
            cell_id="RS-NEUTRAL",
        )
        for label, mutation in (
            ("missing", lambda value: value.pop("schema")),
            (
                "unknown",
                lambda value: value.__setitem__(
                    "schema", "ode-edit-s05-unknown-observability-contract/v1"
                ),
            ),
        ):
            with self.subTest(schema=label):
                malformed = dict(receipt)
                mutation(malformed)
                malformed["identity_sha256"] = canonical_hash(
                    {
                        key: value
                        for key, value in malformed.items()
                        if key != "identity_sha256"
                    }
                )
                with self.assertRaises(ODEBFContractError):
                    observability.validate_observability_contract_receipt(
                        malformed,
                        instruction_id=panel.UNIVERSAL_OBS_INSTRUCTION_ID,
                        amendment_id=panel.UNIVERSAL_OBS_AMENDMENT_ID,
                        cell_id="RS-NEUTRAL",
                    )

        for arm_or_cell in ("RS-N", "BG-UNKNOWN", ""):
            with self.subTest(arm_or_cell=arm_or_cell):
                with self.assertRaises(ODEBFContractError):
                    observability.require_observability_arm(arm_or_cell)
                with self.assertRaises(ODEBFContractError):
                    observability.paired_arm_ids(arm_or_cell)

        target_hold = observability.require_observability_arm(
            "BG-SOFT-TARGET-HOLD"
        )
        self.assertTrue(target_hold.target_hold)
        with self.assertRaises(ODEBFContractError):
            observability.paired_arm_ids(target_hold.arm_id)

    def test_r13_live_cells_are_exact_and_r12_bg_soft_is_reusable_only(self) -> None:
        self.assertEqual(
            observability.R13_LIVE_CELL_IDS,
            ("RS-NEUTRAL", "RS-SOFT", "BG-NEUTRAL"),
        )
        self.assertEqual(observability.R12_REUSABLE_CELL_ID, "BG-SOFT")
        self.assertNotIn(
            observability.R12_REUSABLE_CELL_ID,
            observability.R13_LIVE_CELL_IDS,
        )
        self.assertEqual(
            observability.paired_arm_ids(observability.R12_REUSABLE_CELL_ID),
            ("BG-SOFT", "BG-SOFT-TARGET-HOLD"),
        )

        alias = next(iter(MODEL_ALIASES))
        for cell_id in observability.R13_LIVE_CELL_IDS:
            with self.subTest(cell_id=cell_id):
                self.assertEqual(
                    panel.expected_universal_observability_result_name(
                        alias, cell_id
                    ),
                    "s05-universal-obs-p1r13-a1-"
                    f"{cell_id.lower().replace('-', '_')}-{alias}-v1",
                )
        with self.assertRaises(ODEBFContractError):
            panel.expected_universal_observability_result_name(
                alias, observability.R12_REUSABLE_CELL_ID
            )
        self.assertEqual(
            panel.UNIVERSAL_OBS_SESSION_SOURCE_PATHS,
            (
                "project/run_scripts/session05_ode_bf_submit_universal_observability.py",
                "project/run_scripts/session05_ode_bf_universal_observability.py",
                "project/run_scripts/session05_ode_bf_universal_observability.sbatch",
                "project/run_scripts/session05_ode_bf_universal_observability_dry_plan.py",
                "project/run_scripts/session05_ode_bf_universal_observability_package.py",
            ),
        )

    def test_r12_reference_reuses_only_the_locked_bg_soft_mapping(self) -> None:
        locks = Path("/pure-cpu-r12-locks")
        schedule = object()
        lock = {"opaque": "r12-lock"}
        frozen_reference = {"alias": "llama3-8b-inst", "frozen": True}
        alias = "llama3-8b-inst"

        with (
            mock.patch.object(
                panel,
                "load_and_validate_bg_soft_reference_lock",
                return_value=(lock, panel.UNIVERSAL_OBS_R12_REFERENCE_SHA256),
            ) as load_reference,
            mock.patch.object(
                panel,
                "bg_soft_frozen_reference",
                return_value=frozen_reference,
            ) as map_reference,
        ):
            observed, digest = panel.load_and_validate_r12_frozen_reference(
                locks,
                controller_identity_sha256="controller-identity",
                case_root_digest=panel.UNIVERSAL_OBS_R10_CASE_ROOT,
                schedule=schedule,
                alias=alias,
            )

        self.assertEqual(observed, frozen_reference)
        self.assertEqual(digest, panel.UNIVERSAL_OBS_R12_REFERENCE_SHA256)
        load_reference.assert_called_once_with(
            locks / panel.BG_SOFT_REFERENCE_LOCK_FILE,
            controller_identity_sha256="controller-identity",
            case_root_digest=panel.UNIVERSAL_OBS_R10_CASE_ROOT,
            schedule=schedule,
        )
        map_reference.assert_called_once_with(lock, alias)

        with mock.patch.object(
            panel,
            "load_and_validate_bg_soft_reference_lock",
            return_value=(lock, "0" * 64),
        ):
            with self.assertRaises(ODEBFContractError):
                panel.load_and_validate_r12_frozen_reference(
                    locks,
                    controller_identity_sha256="controller-identity",
                    case_root_digest=panel.UNIVERSAL_OBS_R10_CASE_ROOT,
                    schedule=schedule,
                    alias=alias,
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
