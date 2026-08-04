from __future__ import annotations

import json
import unittest
from pathlib import Path

from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1_evaluator import (
    CounterFactEvaluationCase,
    EndpointActionFreeze,
    PrefixNLLPair,
    counterfact_primary_receipt_from_scores,
    evaluate_counterfact_primary_batch,
    load_counterfact_cases_after_freeze,
    pair_primary_native_floor,
)
from project.run_scripts.ode_bf.p1_selection import load_p1_stream_batches
from project.run_scripts.ode_bf.request_digest import ordered_request_digest_v1
from project.run_scripts.ode_bf.tests.test_evaluator_firewall import (
    ALPHAEDIT_ROOT,
    FakeCausalLM,
    FakeTokenizer,
    _canonical_function,
)


ROOT = Path(__file__).resolve().parents[4]
LOCK = Path(__file__).resolve().parents[1] / "locks/p1_seqb10_stream_seal.json"
DATASET = Path("/mnt/raid5/janghj/EasyEdit/data/counterfact/counterfact.json")


def _cases() -> tuple[CounterFactEvaluationCase, ...]:
    return tuple(
        CounterFactEvaluationCase(
            index,
            canonical_hash({"request": index}),
            f"rewrite-{index}",
            (f"paraphrase-{index}-0", f"paraphrase-{index}-1"),
            tuple(f"neighborhood-{index}-{item}" for item in range(3)),
            "alpha beta",
            "gamma delta",
        )
        for index in range(10)
    )


def _freeze(cases: tuple[CounterFactEvaluationCase, ...]) -> EndpointActionFreeze:
    return EndpointActionFreeze(
        "R_BF",
        0,
        ordered_request_digest_v1([item.request_sha256 for item in cases]),
        canonical_hash({"endpoint": "fixed"}),
        8,
    )


class P1OfficialEvaluatorTests(unittest.TestCase):
    def test_full_primary_evaluator_matches_pinned_prefix_arithmetic(self) -> None:
        canonical = _canonical_function(
            ALPHAEDIT_ROOT / "experiments/py/eval_utils_counterfact.py",
            "test_batch_prediction",
        )
        cases = _cases()
        freeze = _freeze(cases)
        for llama, alias in ((True, "llama3-8b-inst"), (False, "qwen2.5-7b-inst")):
            tokenizer = FakeTokenizer(llama=llama)
            model = FakeCausalLM(llama=llama)
            receipt = evaluate_counterfact_primary_batch(
                model,
                tokenizer,
                cases,
                model_alias=alias,
                freeze=freeze,
            )
            expected_efficacy = []
            expected_generalization = []
            expected_locality = []
            for case in cases:
                prefixes = (
                    (case.rewrite_prompt,)
                    + case.paraphrase_prompts
                    + case.neighborhood_prompts
                )
                probabilities, _ = canonical(
                    model,
                    tokenizer,
                    list(prefixes),
                    [0] * (1 + len(case.paraphrase_prompts))
                    + [1] * len(case.neighborhood_prompts),
                    case.target_new,
                    case.target_true,
                )
                scores = tuple(
                    PrefixNLLPair(item["target_new"], item["target_true"])
                    for item in probabilities
                )
                expected_efficacy.append(scores[:1])
                expected_generalization.append(scores[1:3])
                expected_locality.append(scores[3:])
            expected = counterfact_primary_receipt_from_scores(
                efficacy=expected_efficacy,
                generalization=expected_generalization,
                locality=expected_locality,
                request_order_sha256=receipt.request_order_sha256,
                evaluation_case_identity_sha256=receipt.evaluation_case_identity_sha256,
                target_span_sha256=receipt.target_span_sha256,
                model_forward_count=10,
                processed_token_count=receipt.processed_token_count,
                endpoint_freeze_sha256=freeze.identity(),
            )
            for metric in receipt.metrics():
                self.assertEqual(
                    receipt.metrics()[metric].raw_free_payload(),
                    expected.metrics()[metric].raw_free_payload(),
                )
            self.assertEqual(receipt.generation_call_count, 0)

    def test_heldout_loader_requires_freeze_and_copies_no_generation_surface(self) -> None:
        if not DATASET.is_file():
            self.skipTest("pinned CounterFact dataset unavailable")
        stream = json.loads(LOCK.read_text(encoding="utf-8"))
        batch = load_p1_stream_batches(DATASET, stream)[0]
        freeze = EndpointActionFreeze(
            "N32_NATIVE",
            0,
            ordered_request_digest_v1([item["request_sha256"] for item in batch]),
            canonical_hash({"native": 0}),
            0,
        )
        cases = load_counterfact_cases_after_freeze(DATASET, batch, freeze)
        self.assertEqual(len(cases), 10)
        self.assertTrue(all(case.paraphrase_prompts for case in cases))
        self.assertTrue(all(case.neighborhood_prompts for case in cases))
        self.assertTrue(all("generation" not in name for name in cases[0].__slots__))
        wrong = EndpointActionFreeze(
            "N32_NATIVE",
            0,
            canonical_hash({"wrong": True}),
            canonical_hash({"native": 0}),
            0,
        )
        with self.assertRaisesRegex(ODEBFContractError, "preceded action freeze"):
            load_counterfact_cases_after_freeze(DATASET, batch, wrong)

    def test_three_primary_floors_are_noncompensatory_and_pairwise_losses_visible(self) -> None:
        cases = _cases()
        freeze = _freeze(cases)
        base_scores = tuple((PrefixNLLPair(0.0, 1.0),) for _ in range(10))
        gen_scores = tuple(
            (PrefixNLLPair(0.0, 1.0), PrefixNLLPair(0.0, 1.0))
            for _ in range(10)
        )
        locality_scores = tuple(
            (PrefixNLLPair(1.0, 0.0),) for _ in range(10)
        )
        common = {
            "request_order_sha256": freeze.request_order_sha256,
            "evaluation_case_identity_sha256": canonical_hash({"cases": 10}),
            "target_span_sha256": canonical_hash({"spans": 10}),
            "model_forward_count": 10,
            "processed_token_count": 100,
            "endpoint_freeze_sha256": freeze.identity(),
        }
        native = counterfact_primary_receipt_from_scores(
            efficacy=base_scores,
            generalization=gen_scores,
            locality=locality_scores,
            **common,
        )
        damaged = list(locality_scores)
        damaged[3] = (PrefixNLLPair(0.0, 1.0),)
        ours = counterfact_primary_receipt_from_scores(
            efficacy=base_scores,
            generalization=gen_scores,
            locality=damaged,
            **common,
        )
        paired = pair_primary_native_floor(native, ours)
        self.assertFalse(paired.all_primary_pass)
        self.assertTrue(paired.pairwise_miss_review)
        locality = [item for item in paired.paired if item.metric.value.startswith("locality")][0]
        self.assertEqual(locality.loss_count, 1)
        self.assertEqual(locality.win_count, 0)

    def test_denominator_and_span_mismatch_fail_closed(self) -> None:
        cases = _cases()
        freeze = _freeze(cases)
        tokenizer = FakeTokenizer(llama=False)
        model = FakeCausalLM(llama=False)
        native = evaluate_counterfact_primary_batch(
            model, tokenizer, cases, model_alias="qwen2.5-7b-inst", freeze=freeze
        )
        altered = list(cases)
        case = altered[0]
        altered[0] = CounterFactEvaluationCase(
            case.case_id,
            case.request_sha256,
            case.rewrite_prompt,
            case.paraphrase_prompts + ("extra-paraphrase",),
            case.neighborhood_prompts,
            case.target_new,
            case.target_true,
        )
        ours = evaluate_counterfact_primary_batch(
            model, tokenizer, altered, model_alias="qwen2.5-7b-inst", freeze=freeze
        )
        with self.assertRaisesRegex(ODEBFContractError, "denominators"):
            pair_primary_native_floor(native, ours)


if __name__ == "__main__":
    unittest.main()
