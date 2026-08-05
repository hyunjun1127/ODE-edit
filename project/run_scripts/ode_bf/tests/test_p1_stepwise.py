from __future__ import annotations

import inspect
import unittest

from project.run_scripts.ode_bf.contracts import BATCH_SIZE, canonical_hash
from project.run_scripts.ode_bf.p1_evaluator import (
    PrefixNLLPair,
    counterfact_primary_receipt_from_scores,
)
from project.run_scripts.ode_bf.p1_stepwise import (
    PREDECLARED_QUANTILES,
    StepwiseActionFreeze,
    StepwisePrimaryReceipt,
    compare_stepwise_primary,
)


def _freeze(label: str, snapshot: str) -> StepwiseActionFreeze:
    return StepwiseActionFreeze(
        variant=label,
        request_order_sha256="1" * 64,
        rollout_sha256="2" * 64,
        snapshot_sha256=snapshot * 64,
        snapshot_index=0,
        accepted_snapshot_count=1,
        rejected_retry_count=0,
        trajectory_status="COMPLETE",
    )


def _receipt(
    label: str,
    *,
    rewrite_new: float,
    rewrite_old: float,
    generalization_new: float,
    generalization_old: float,
    locality_new: float,
    locality_old: float,
) -> StepwisePrimaryReceipt:
    freeze = _freeze(label, label[0].casefold())
    efficacy = tuple(
        (PrefixNLLPair(rewrite_new + index * 0.001, rewrite_old),)
        for index in range(BATCH_SIZE)
    )
    generalization = tuple(
        (PrefixNLLPair(generalization_new, generalization_old),) * 2
        for _ in range(BATCH_SIZE)
    )
    locality = tuple(
        (PrefixNLLPair(locality_new, locality_old),)
        for _ in range(BATCH_SIZE)
    )
    primary = counterfact_primary_receipt_from_scores(
        efficacy=efficacy,
        generalization=generalization,
        locality=locality,
        request_order_sha256="1" * 64,
        evaluation_case_identity_sha256="3" * 64,
        target_span_sha256="4" * 64,
        model_forward_count=BATCH_SIZE,
        processed_token_count=100,
        endpoint_freeze_sha256=freeze.identity(),
    )
    temporary = object.__new__(StepwisePrimaryReceipt)
    object.__setattr__(temporary, "primary", primary)
    object.__setattr__(temporary, "efficacy_scores", efficacy)
    object.__setattr__(temporary, "generalization_scores", generalization)
    object.__setattr__(temporary, "locality_scores", locality)
    object.__setattr__(temporary, "action_freeze_sha256", freeze.identity())
    object.__setattr__(temporary, "numeric_vectors_sha256", "0" * 64)
    return StepwisePrimaryReceipt(
        primary,
        efficacy,
        generalization,
        locality,
        freeze.identity(),
        canonical_hash(temporary.numeric_payload()),
    )


class StepwiseReceiptTests(unittest.TestCase):
    def test_action_freeze_is_snapshot_specific(self) -> None:
        left = _freeze("FR-A8", "a")
        right = _freeze("FR-A8", "b")
        self.assertNotEqual(left.identity(), right.identity())
        self.assertTrue(left.action_frozen)

    def test_numeric_receipt_uses_locked_metric_directions(self) -> None:
        observed = _receipt(
            "OBS",
            rewrite_new=1.0,
            rewrite_old=2.0,
            generalization_new=1.0,
            generalization_old=2.0,
            locality_new=2.0,
            locality_old=1.0,
        )
        payload = observed.numeric_payload()
        self.assertEqual(payload["efficacy"][0][0]["margin"], 1.0)
        self.assertEqual(payload["generalization"][0][0]["margin"], 1.0)
        self.assertEqual(payload["locality-preservation"][0][0]["margin"], 1.0)
        self.assertEqual(observed.primary.efficacy.numerator, 10)
        self.assertEqual(observed.primary.generalization.numerator, 20)
        self.assertEqual(observed.primary.locality.numerator, 10)

    def test_paired_native_losses_and_old_degradation_are_explicit(self) -> None:
        w0 = _receipt(
            "W0",
            rewrite_new=3.0,
            rewrite_old=2.0,
            generalization_new=3.0,
            generalization_old=2.0,
            locality_new=2.0,
            locality_old=1.0,
        )
        native = _receipt(
            "N32",
            rewrite_new=1.0,
            rewrite_old=2.0,
            generalization_new=1.0,
            generalization_old=2.0,
            locality_new=2.0,
            locality_old=1.0,
        )
        # Success is achieved only because old NLL worsens; the new-target NLL
        # does not improve relative to W0.
        observed = _receipt(
            "OURS",
            rewrite_new=3.0,
            rewrite_old=4.0,
            generalization_new=3.0,
            generalization_old=4.0,
            locality_new=2.0,
            locality_old=1.0,
        )
        result = compare_stepwise_primary(w0, native, observed)
        efficacy = result["metrics"]["efficacy"]
        self.assertEqual(efficacy["paired_native"]["loss_count"], 0)
        self.assertEqual(efficacy["old_degradation_only_success_count"], 10)
        self.assertEqual(
            tuple(efficacy["delta_vs_native"]["nll_new"]["quantiles"]),
            tuple(f"q{int(100 * value):02d}" for value in PREDECLARED_QUANTILES),
        )

    def test_native_success_loss_cannot_be_hidden_by_other_metrics(self) -> None:
        w0 = _receipt(
            "W0",
            rewrite_new=3.0,
            rewrite_old=2.0,
            generalization_new=3.0,
            generalization_old=2.0,
            locality_new=2.0,
            locality_old=1.0,
        )
        native = _receipt(
            "N32",
            rewrite_new=1.0,
            rewrite_old=2.0,
            generalization_new=1.0,
            generalization_old=2.0,
            locality_new=2.0,
            locality_old=1.0,
        )
        ours = _receipt(
            "OURS",
            rewrite_new=3.0,
            rewrite_old=2.0,
            generalization_new=1.0,
            generalization_old=2.0,
            locality_new=2.0,
            locality_old=1.0,
        )
        result = compare_stepwise_primary(w0, native, ours)
        self.assertEqual(
            result["metrics"]["efficacy"]["paired_native"][
                "native_success_loss_count"
            ],
            10,
        )
        self.assertEqual(
            result["metrics"]["generalization"]["paired_native"][
                "native_success_loss_count"
            ],
            0,
        )

    def test_stepwise_module_is_teacher_forced_and_controller_disjoint(self) -> None:
        from project.run_scripts.ode_bf import p1_adaptive_runtime, p1_stepwise

        evaluator_source = inspect.getsource(p1_stepwise)
        runtime_source = inspect.getsource(p1_adaptive_runtime)
        self.assertNotIn("model.generate", evaluator_source)
        self.assertIn("action_frozen_before_open", runtime_source)
        self.assertLess(
            runtime_source.index("action-freeze.json"),
            runtime_source.index("cases = load_counterfact_cases_after_freeze"),
        )
        self.assertIn("heldout_controller_access_count", runtime_source)


if __name__ == "__main__":
    unittest.main()
