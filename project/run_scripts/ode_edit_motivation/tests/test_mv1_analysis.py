import json
import unittest

from project.run_scripts.ode_edit_motivation.contracts import ContractError
from project.run_scripts.ode_edit_motivation.mv1_analysis import (
    ControllerDecision,
    ITDFailurePolicy,
    LockedAction,
    LockedActionSet,
    OutcomeStatus,
    SanitizedOutcome,
    build_confirmatory_fold_manifest,
    build_controller_decision_manifest,
    controller_effect_summary,
    oracle_empirical_ceiling,
    select_calibration_best_static,
)


def outcome(model, case, action, progress, status=OutcomeStatus.COMPLETED):
    return SanitizedOutcome(
        model_id=model,
        case_id=case,
        action_id=action,
        progress=progress,
        reached=status is OutcomeStatus.COMPLETED,
        status=status,
    )


class MV1AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.actions = LockedActionSet(
            actions=(
                LockedAction("analytic", simplicity_rank=2, compute_rank=2),
                LockedAction("static-simple", simplicity_rank=0, compute_rank=0),
                LockedAction("static-costly", simplicity_rank=1, compute_rank=1),
            ),
            near_tie_tolerance=0.05,
        )
        self.itd = ITDFailurePolicy(
            fallback_progress=-2.0,
            label="calibration-locked-worst-progress",
        )

    def test_static_is_calibration_fixed_and_tie_is_deterministic(self):
        rows = []
        for case in ("c0", "c1"):
            rows.extend(
                (
                    outcome("m", case, "static-simple", 1.0),
                    outcome("m", case, "static-costly", 1.04),
                    outcome("m", case, "analytic", 0.0),
                )
            )
        first = select_calibration_best_static(
            rows,
            model_id="m",
            calibration_case_ids=("c1", "c0"),
            action_set=self.actions,
            itd_policy=self.itd,
        )
        second = select_calibration_best_static(
            reversed(rows),
            model_id="m",
            calibration_case_ids=("c0", "c1"),
            action_set=self.actions,
            itd_policy=self.itd,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.action_id, "static-simple")
        self.assertEqual(
            first.near_tied_actions, ("static-costly", "static-simple")
        )

    def test_case_hash_folds_are_balanced_deterministic_and_outcome_free(self):
        ids = tuple(f"case-{index}" for index in range(60))
        first = build_confirmatory_fold_manifest(ids, seed="confirmatory-v1")
        second = build_confirmatory_fold_manifest(
            tuple(reversed(ids)), seed="confirmatory-v1"
        )
        self.assertEqual(first, second)
        counts = [sum(item.fold == fold for item in first.assignments) for fold in range(5)]
        self.assertEqual(counts, [12] * 5)
        serialized = json.dumps(first.to_dict(), sort_keys=True)
        self.assertNotIn("progress", serialized)
        self.assertNotIn("prompt", serialized)
        self.assertNotIn("answer", serialized)

    def test_sanitized_contracts_reject_raw_or_outcome_fields(self):
        base = {
            "model_id": "m",
            "case_id": "c",
            "action_id": "a",
            "progress": 1.0,
            "reached": True,
            "status": "completed",
        }
        for forbidden in ("prompt", "answer", "target_new", "paraphrase_prompts"):
            with self.assertRaises(ContractError):
                SanitizedOutcome.from_mapping({**base, forbidden: "secret"})

        decision = {
            "model_id": "m",
            "case_id": "c",
            "fold": 0,
            "action_id": "a",
            "pre_step_feature_hash": "a" * 64,
            "controller_lock_hash": "b" * 64,
        }
        with self.assertRaises(ContractError):
            ControllerDecision.from_mapping(
                {**decision, "realized_progress": 100.0}
            )

    def test_oracle_is_labeled_ceiling_and_preserves_failures_in_itd(self):
        calibration = []
        for case in ("k0", "k1"):
            calibration.extend(
                (
                    outcome("m", case, "static-simple", 1.0),
                    outcome("m", case, "static-costly", 0.5),
                    outcome("m", case, "analytic", 0.0),
                )
            )
        static = select_calibration_best_static(
            calibration,
            model_id="m",
            calibration_case_ids=("k0", "k1"),
            action_set=self.actions,
            itd_policy=self.itd,
        )
        confirmatory = [
            outcome("m", "x", "static-simple", 1.0),
            outcome("m", "x", "static-costly", 1.03),
            outcome("m", "x", "analytic", 2.0),
            outcome("m", "y", "static-simple", 1.0),
            outcome("m", "y", "static-costly", 0.0),
            SanitizedOutcome(
                model_id="m",
                case_id="y",
                action_id="analytic",
                progress=None,
                reached=False,
                status=OutcomeStatus.NON_FINITE,
            ),
        ]
        ceiling = oracle_empirical_ceiling(
            confirmatory,
            model_id="m",
            case_ids=("x", "y"),
            action_set=self.actions,
            static_policy=static,
            itd_policy=self.itd,
            bootstrap_seed="oracle-v1",
            bootstrap_replicates=200,
        )
        self.assertIn("not achievable", ceiling.estimand_label)
        self.assertEqual(ceiling.summary.raw.denominator, 2)
        self.assertEqual(ceiling.summary.raw.mean, 0.5)
        self.assertGreaterEqual(ceiling.summary.fallback_scored_rows, 0)
        self.assertEqual(ceiling.candidate_count, 3)
        self.assertEqual(
            dict(dict(ceiling.arm_status_counts)["analytic"])["non_finite"],
            1,
        )

        incomplete = confirmatory[:-1]
        with self.assertRaisesRegex(ContractError, "ITD panel is missing"):
            oracle_empirical_ceiling(
                incomplete,
                model_id="m",
                case_ids=("x", "y"),
                action_set=self.actions,
                static_policy=static,
                itd_policy=self.itd,
                bootstrap_seed="oracle-v1",
                bootstrap_replicates=100,
            )

    def test_controller_requires_precommitted_hash_and_bootstrap_is_deterministic(self):
        calibration = []
        for case in ("k0", "k1"):
            calibration.extend(
                (
                    outcome("m", case, "static-simple", 1.0),
                    outcome("m", case, "static-costly", 0.0),
                    outcome("m", case, "analytic", 0.0),
                )
            )
        static = select_calibration_best_static(
            calibration,
            model_id="m",
            calibration_case_ids=("k0", "k1"),
            action_set=self.actions,
            itd_policy=self.itd,
        )
        folds = build_confirmatory_fold_manifest(
            ("x", "y", "z", "w"), seed="fold-v1", fold_count=2
        )
        decisions = build_controller_decision_manifest(
            tuple(
                ControllerDecision(
                    model_id="m",
                    case_id=case,
                    fold=folds.fold_for(case),
                    action_id="analytic",
                    pre_step_feature_hash=(str(index + 1) * 64)[:64],
                    controller_lock_hash="f" * 64,
                )
                for index, case in enumerate(folds.case_ids)
            ),
            model_id="m",
            action_set=self.actions,
            folds=folds,
        )
        rows = []
        gains = {"w": 0.01, "x": 0.04, "y": 0.2, "z": 0.4}
        for case, gain in gains.items():
            rows.extend(
                (
                    outcome("m", case, "static-simple", 1.0),
                    outcome("m", case, "analytic", 1.0 + gain),
                )
            )
        first = controller_effect_summary(
            rows,
            decisions=decisions,
            expected_decision_manifest_id=decisions.manifest_id,
            action_set=self.actions,
            static_policy=static,
            itd_policy=self.itd,
            bootstrap_seed="paired-v1",
            bootstrap_replicates=300,
        )
        second = controller_effect_summary(
            reversed(rows),
            decisions=decisions,
            expected_decision_manifest_id=decisions.manifest_id,
            action_set=self.actions,
            static_policy=static,
            itd_policy=self.itd,
            bootstrap_seed="paired-v1",
            bootstrap_replicates=300,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.raw.mean, 0.1625)
        self.assertAlmostEqual(first.near_tie_sensitivity.mean, 0.15)
        self.assertEqual(first.raw.denominator, 4)

        with self.assertRaisesRegex(ContractError, "precommitted"):
            controller_effect_summary(
                rows,
                decisions=decisions,
                expected_decision_manifest_id="0" * 64,
                action_set=self.actions,
                static_policy=static,
                itd_policy=self.itd,
                bootstrap_seed="paired-v1",
                bootstrap_replicates=100,
            )


if __name__ == "__main__":
    unittest.main()
