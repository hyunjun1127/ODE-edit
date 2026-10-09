"""CPU regressions exercise the actual model-forward evaluator, not GPU parity."""
import copy
import json
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from official.evaluation.factual import (FactualError, build_zsre_w0_reference,
                                         evaluate_counterfact, evaluate_zsre,
                                         retained_query_work, validate_retained_observation)
from official.evaluation import factual


class CharacterTokenizer:
    """Deterministic native BOS/no-BOS tokenizer with exact concatenation."""
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
    model_max_length, padding_side = 512, "left"
    add_bos_token = True

    def __call__(self, text, add_special_tokens=True, truncation=False):
        assert truncation is False
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else [])
                              + [ord(char) + 3 for char in text]}


class TransitionLM(torch.nn.Module):
    """Tiny causal CPU LM, native logits API and no target-fitting operations."""
    def __init__(self, consume_rng=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("fixed", torch.zeros(1))
        self.child = torch.nn.Dropout()
        self.config = SimpleNamespace(max_position_embeddings=512, use_cache=True)
        self.consume_rng, self.calls = consume_rng, []
        self.prefer_true_for_B = True
        self.corrupt = None

    def forward(self, input_ids, attention_mask, use_cache):
        self.calls.append(dict(ids=input_ids.detach().cpu().tolist(),
                               mask=attention_mask.detach().cpu().tolist(), use_cache=use_cache,
                               training=self.training, grad=torch.is_grad_enabled()))
        if self.consume_rng:
            random.random(); np.random.random(); torch.rand(1)
        if self.corrupt == "weight":
            self.weight.add_(1)
        if self.corrupt == "config":
            self.config.use_cache = False
        if self.corrupt == "error":
            raise RuntimeError("originating model error")
        shape = (*input_ids.shape, 512)
        logits = torch.full(shape, -4., device=input_ids.device)
        # Every ordinary previous character predicts a separator. After the
        # separator, B-prefix prompts prefer T, others N; the next token is n/t.
        expected = torch.full_like(input_ids, ord(" ") + 3)
        true_prompt = input_ids.eq(ord("B") + 3).cumsum(1).gt(0) & self.prefer_true_for_B
        space = input_ids.eq(ord(" ") + 3)
        expected[space & ~true_prompt] = ord("N") + 3
        expected[space & true_prompt] = ord("T") + 3
        expected[input_ids.eq(ord("N") + 3)] = ord("n") + 3
        expected[input_ids.eq(ord("T") + 3)] = ord("t") + 3
        logits.scatter_(2, expected[..., None], 4.)
        if self.corrupt == "nan":
            logits[:, :, 0] = float("nan")
        return SimpleNamespace(logits=logits)


def cf(index=1, *, paraphrases=None, subject="Ada"):
    return dict(case_id=100 + index, occurrence_index=index,
                requested_rewrite=dict(prompt="{} is", subject=subject,
                                       target_new={"str": "Nn"}, target_true={"str": "Tt"}),
                paraphrase_prompts=paraphrases or ["Ada was"],
                neighborhood_prompts=["Bob is", "Bob at"])


def zr(index=1, *, target="Nn", neighborhood="Tt", subject="Ada"):
    return dict(case_id=index, occurrence_index=index,
                requested_rewrite=dict(prompt="{} is", subject=subject,
                                       target_new={"str": target}, target_true={"str": "unused"}),
                paraphrase_prompts=["Ada was"],
                neighborhood_prompts=[dict(prompt="Bob is?", target=neighborhood)])


class FactualTests(unittest.TestCase):
    def setUp(self):
        self.tok, self.model = CharacterTokenizer(), TransitionLM()

    def test_CF_native_strict_preference_and_N_desired_true(self):
        result = evaluate_counterfact(self.model, self.tok, [cf()], batch_size=2)
        self.assertEqual({key: result["summary"][key] for key in
                          ("Efficacy", "Generalization", "Specificity", "Score")},
                         dict(Efficacy=100., Generalization=100., Specificity=100., Score=100.))
        case = result["cases"][0]
        self.assertLess(case["rewrite_prompts_probs"][0]["target_new"],
                        case["rewrite_prompts_probs"][0]["target_true"])
        self.assertLess(case["neighborhood_prompts_probs"][0]["target_true"],
                        case["neighborhood_prompts_probs"][0]["target_new"])
        self.assertEqual(result["accuracy"]["neighborhood"]["token_acc_pct"], 100)
        self.assertTrue(all(row["desired_target"] == "true" for row in case["neighborhood_observations"]))
        self.assertEqual(result["work"]["candidate_sequences"], 8)
        self.assertEqual(result["work"]["target_tokens"], 24)
        self.assertEqual(result["accuracy"]["neighborhood"]["token_count"], 6)
        self.assertTrue(all(not call["use_cache"] and not call["training"] and not call["grad"]
                            for call in self.model.calls))

    def test_CF_actual_request_macro_not_prompt_micro(self):
        result = evaluate_counterfact(self.model, self.tok,
            [cf(1, paraphrases=["Bob was"] * 3), cf(2, paraphrases=["Ada was"])])
        self.assertEqual(result["summary"]["Generalization"], 50)
        self.assertEqual(result["summary"]["Score"], 75)
        self.assertEqual(result["accuracy"]["paraphrase"]["prompt_count"], 4)
        self.assertEqual(result["summary"]["requests"], 2)

    def test_CF_ties_fail_and_missing_is_not_zero(self):
        row = cf()
        row["requested_rewrite"]["target_true"] = {"str": "Nn"}
        result = evaluate_counterfact(self.model, self.tok, [row])
        self.assertEqual(result["summary"]["Efficacy"], 0)
        self.assertEqual(result["summary"]["Specificity"], 0)
        self.assertEqual(result["summary"]["Score"], 0)
        row["neighborhood_prompts"] = []
        with self.assertRaisesRegex(FactualError, "MISSING_PROMPTS"):
            evaluate_counterfact(self.model, self.tok, [row])

    def test_CF_BOS_multi_target_mask_and_padding_side(self):
        result = evaluate_counterfact(self.model, self.tok, [cf()], batch_size=8)
        row = result["cases"][0]["rewrite_observations"][0]["target_new"]
        self.assertEqual(row["input_token_ids"][0], self.tok.bos_token_id)
        self.assertNotIn(self.tok.bos_token_id, row["target_token_ids"])
        self.assertEqual(row["target_start"], len(self.tok("Ada is")["input_ids"]))
        self.assertEqual(row["target_token_ids"], [ord(x) + 3 for x in " Nn"])
        self.assertEqual(len(row["nll_by_token"]), 3)
        self.assertEqual(row["predicted_token_ids"], row["target_token_ids"])
        self.assertEqual(self.tok.padding_side, "left")
        mask = self.model.calls[0]["mask"]
        self.assertTrue(all(bits == sorted(bits, reverse=True) for bits in mask))

    def test_CF_token_boundary_rejected_before_forward(self):
        class BoundaryTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                value = super().__call__(text, **kwargs)
                if text == "Ada is Nn":
                    value["input_ids"][3] = 300
                return value
        with self.assertRaisesRegex(FactualError, "BOUNDARY_MISMATCH"):
            evaluate_counterfact(self.model, BoundaryTokenizer(), [cf()])
        self.assertEqual(self.model.calls, [])

    def test_context_limit_no_silent_truncation(self):
        self.model.config.max_position_embeddings = 4
        with self.assertRaisesRegex(FactualError, "CONTEXT_LENGTH_EXCEEDED"):
            evaluate_counterfact(self.model, self.tok, [cf()])
        self.assertEqual(self.model.calls, [])

    def test_state_RNG_and_heterogeneous_training_restored(self):
        model = TransitionLM(consume_rng=True)
        model.train(); model.child.eval()
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        original = copy.deepcopy(cf())
        evaluate_counterfact(model, self.tok, [original])
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertTrue(model.config.use_cache)
        self.assertEqual(original, cf())
        self.assertEqual(before[0], random.getstate())
        self.assertTrue(np.array_equal(before[1][1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(before[2], torch.get_rng_state()))

    def test_nonfinite_and_mutation_typed_not_quality_gates(self):
        for cause, code in (("nan", "NONFINITE_LOGITS"), ("weight", "MODEL_MUTATED"),
                            ("config", "MODEL_MUTATED")):
            with self.subTest(cause=cause):
                model = TransitionLM(); model.corrupt = cause
                with self.assertRaisesRegex(FactualError, code):
                    evaluate_counterfact(model, self.tok, [cf()])
        model = TransitionLM(consume_rng=True); model.corrupt = "error"
        state = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "originating model error"):
            evaluate_counterfact(model, self.tok, [cf()])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_zsRE_W0_reference_actual_once_and_answer_vs_agreement(self):
        rows = [zr()]
        ref = build_zsre_w0_reference(self.model, self.tok, rows, identity={"revision": "pinned"})
        calls = len(self.model.calls)
        self.assertEqual(ref["evaluation"]["summary"]["Specificity"], 100)
        self.assertEqual(ref["evaluation"]["summary"]["Specificity_loc_ans"], 100)
        self.assertEqual(ref["evaluation"]["work"]["forward_calls"], calls)
        self.assertEqual(len(ref["cases"][0]["predictions"][0]), 3)
        self.model.prefer_true_for_B = False
        later = evaluate_zsre(self.model, self.tok, rows, w0_reference=ref,
                              identity={"revision": "pinned"})
        self.assertAlmostEqual(later["summary"]["Specificity"], 200 / 3)
        self.assertAlmostEqual(later["summary"]["Specificity_loc_ans"], 200 / 3)
        # A base model may be wrong yet fully preserve its own predictions.
        self.model.prefer_true_for_B = False
        wrong = build_zsre_w0_reference(self.model, self.tok, rows)
        self.assertEqual(wrong["evaluation"]["summary"]["W0_prediction_agreement"], 100)
        self.assertLess(wrong["evaluation"]["summary"]["Specificity"], 100)
        self.assertLess(wrong["evaluation"]["summary"]["Specificity_loc_ans"], 100)

    def test_zsRE_BOS_ignored_and_native_teacher_prefix_multi_token(self):
        ref = build_zsre_w0_reference(self.model, self.tok, [zr()])
        query = ref["cases"][0]["queries"][0]
        prefix = self.tok("Bob is?")["input_ids"]
        suffix = self.tok(" Tt", add_special_tokens=False)["input_ids"]
        self.assertEqual(query["input_token_ids"], prefix + suffix)
        self.assertEqual(query["target_token_ids"], suffix)
        self.assertEqual(len(ref["evaluation"]["cases"][0]["neighborhood_prompts_correct"]), 3)

    def test_zsRE_missing_W0_reference_explicit_no_fake_zero(self):
        result = evaluate_zsre(self.model, self.tok, [zr()])
        self.assertEqual(result["summary"]["Specificity"], result["summary"]["Specificity_loc_ans"])
        self.assertNotIn("W0_prediction_agreement", result["summary"])
        self.assertEqual(result["summary"]["W0_prediction_agreement_availability"],
                         "NOT_MEASURED_W0_REFERENCE_REQUIRED")
        self.assertIsNone(result["cases"][0]["neighborhood_W0_agreement"])

    def test_zsRE_request_macro_distinct_from_token_micro(self):
        result = evaluate_zsre(self.model, self.tok,
                               [zr(1, target="T"), zr(2, target="Nn")])
        self.assertEqual(result["summary"]["Efficacy"], 75)
        self.assertEqual(result["accuracy"]["rewrite"]["token_acc_pct"], 80)
        self.assertEqual(result["accuracy"]["rewrite"]["token_count"], 5)
        self.assertEqual(result["accuracy"]["rewrite"]["token_correct_count"], 4)

    def test_no_BOS_tokenizer_first_suffix_query_is_not_shifted(self):
        class NoBOSTokenizer(CharacterTokenizer):
            bos_token_id, add_bos_token = None, False
            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(char) + 3 for char in text]}
        tok = NoBOSTokenizer()
        result = evaluate_counterfact(self.model, tok, [cf()])
        self.assertEqual(result["summary"]["Score"], 100)
        target = result["cases"][0]["rewrite_observations"][0]["target_new"]
        self.assertEqual(target["target_start"], len("Ada is"))
        self.assertEqual(target["predicted_token_ids"], target["target_token_ids"])

    def test_zsRE_reference_tamper_identity_mismatch_before_forward(self):
        ref = build_zsre_w0_reference(self.model, self.tok, [zr()], identity={"revision": "pinned"})
        cases = len(self.model.calls)
        changed = copy.deepcopy(ref)
        changed["cases"][0]["predictions"][0][0] = 333
        with self.assertRaisesRegex(FactualError, "PAYLOAD_HASH"):
            evaluate_zsre(self.model, self.tok, [zr()], w0_reference=changed,
                          identity={"revision": "pinned"})
        with self.assertRaisesRegex(FactualError, "EXTERNAL_IDENTITY"):
            evaluate_zsre(self.model, self.tok, [zr()], w0_reference=ref,
                          identity={"revision": "different"})
        with self.assertRaisesRegex(FactualError, "TOKEN_PREFIX_IDENTITY"):
            evaluate_zsre(self.model, self.tok, [zr(neighborhood="Nn")], w0_reference=ref,
                          identity={"revision": "pinned"})
        self.assertEqual(len(self.model.calls), cases)

    def test_zsRE_W0_prefix_subset_does_not_deduplicate_case_id(self):
        ref = build_zsre_w0_reference(self.model, self.tok, [zr(1), zr(2)])
        result = evaluate_zsre(self.model, self.tok, [zr(2)], w0_reference=ref)
        self.assertEqual(result["summary"]["requests"], 1)
        self.assertEqual(result["cases"][0]["occurrence_index"], 2)
        with self.assertRaisesRegex(FactualError, "DUPLICATE_OCCURRENCE"):
            evaluate_zsre(self.model, self.tok, [zr(2), zr(2)])

    def test_all_raw_json_finite_progress_counts_no_extra_forward(self):
        progress = []
        value = evaluate_counterfact(self.model, self.tok, [cf()], batch_size=3,
                                     progress=progress.append)
        json.dumps(value, allow_nan=False)
        self.assertEqual(len(progress), value["work"]["forward_calls"])
        self.assertEqual(progress[-1]["completed_candidate_sequences"], 8)
        self.assertEqual(progress[-1]["target_tokens"], 24)
        self.assertTrue(value["model_no_mutation"] and value["RNG_restored"])


def retained_fixture(dataset="cf", batch_size=4):
    """UNIT TEST ONLY synthetic retained rows; no model/tokenizer observation."""
    signatures, queries, cases = [], [], []
    for index in range(2):
        case = dict(case_id=100 + index, occurrence_index=index + 1)
        case_queries = []
        for slot, kind in enumerate(("rewrite", "paraphrase", "neighborhood")):
            targets = ("new", "true") if dataset == "cf" else (
                "loc_ans" if kind == "neighborhood" else "new",)
            for target_kind in targets:
                prefix = [1] + [10 + slot] * (1 + index + slot)
                gold = [20, 21] if target_kind == "new" else [22]
                query = dict(kind=kind, prompt_index=0, prompt="CPU_FIXTURE_PROMPT_" + kind,
                    target_kind=target_kind, target="CPU_FIXTURE_TARGET_" + target_kind,
                    input_token_ids=prefix + gold, target_start=len(prefix), target_token_ids=gold)
                case_queries.append(query)
                queries.append(dict(query, case_index=index, occurrence_index=index + 1))
        signatures.append(dict(case, queries=case_queries))
        cases.append(case)
    results = []
    for query in queries:
        gold = query["target_token_ids"]
        desired = "true" if query["kind"] == "neighborhood" else "new"
        nll = 1.0 if query["target_kind"] in (desired, "loc_ans") else 2.0
        predicted = list(gold)
        if query["case_index"] == 1 and query["kind"] == "rewrite":
            predicted[-1] = 99
        correct = [a == b for a, b in zip(predicted, gold)]
        results.append(dict(query, predicted_token_ids=predicted, token_correct=correct,
            token_count=len(gold), token_correct_count=sum(correct), strict_correct=all(correct),
            nll_by_token=[nll] * len(gold), mean_nll=nll))
    summary, accuracy = (factual._assemble_cf(cases, results) if dataset == "cf" else
                         factual._assemble_zsre(cases, results, None, signatures))
    identity = dict(schema=factual.SCHEMA, dataset=dataset, tokenization=factual.TOKENIZATION,
        cohort_sha256=factual._digest(signatures), ordered_occurrences=[1, 2],
        external_identity={"unit_test_evidence": "TEST_ONLY_NO_REAL_MODEL_CALLS"},
        padding="RIGHT_EXPLICIT_ATTENTION_MASK", use_cache=False, W0_reference_sha256=None)
    chunks = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    work = dict(forward_calls=len(chunks), candidate_sequences=len(queries),
        physical_input_tokens=sum(len(query["input_token_ids"]) for query in queries),
        target_tokens=sum(len(query["target_token_ids"]) for query in queries),
        padded_input_tokens=sum(len(chunk) * max(len(q["input_token_ids"]) for q in chunk) for chunk in chunks),
        seconds=0.001)
    return dict(summary=summary, accuracy=accuracy, cases=cases, identity=identity,
        identity_sha256=factual._digest(identity), work=work, raw_local_only=True,
        model_no_mutation=True, RNG_restored=True), signatures, queries


class RetainedObservationTests(unittest.TestCase):
    """Adversarial CPU raw checks, expressly not actual GPU/parity evidence."""
    def audit(self, endpoint, signatures, dataset="cf", **kwargs):
        with patch.object(factual, "_infer", side_effect=AssertionError("MODEL_FORWARD_FORBIDDEN")), \
             patch.object(factual, "_ids", side_effect=AssertionError("TOKENIZER_CALL_FORBIDDEN")):
            return validate_retained_observation(endpoint, dataset,
                input_signature=signatures, batch_size=4, **kwargs)

    def test_synthetic_CF_and_zsRE_completed_raw_reduces_with_no_model_or_tokenizer_call(self):
        for dataset in ("cf", "zsre"):
            value, signatures, queries = retained_fixture(dataset)
            before = copy.deepcopy(value)
            with self.subTest(dataset=dataset):
                result = self.audit(value, signatures, dataset, query_plan=queries)
                self.assertEqual(result["candidate_sequences"], len(queries))
                self.assertEqual(result["requests"], 2)
                self.assertTrue(result["observed_work_validated"])
                self.assertEqual(result["actual_model_forward_calls"], 0)
                self.assertFalse(result["GPU_parity_claim"])
                self.assertEqual(value, before)

    def test_removed_all_canonical_observations_is_not_probability_only_proof(self):
        value, signatures, _ = retained_fixture()
        for case in value["cases"]:
            for kind in ("rewrite", "paraphrase", "neighborhood"):
                case.pop(kind + "_observations")
        value["payload_sha256"] = factual._digest(value)
        with self.assertRaisesRegex(FactualError, "CANONICAL_OBSERVATIONS_REQUIRED"):
            self.audit(value, signatures)

    def test_missing_zero_or_same_wrong_resigned_work_counters_are_rejected(self):
        original, signatures, _ = retained_fixture()
        for kind in ("missing", "all0", "all123456", "padded", "forward"):
            value = copy.deepcopy(original)
            if kind == "missing":
                value.pop("work")
            elif kind in ("all0", "all123456"):
                for key in value["work"]:
                    value["work"][key] = 0 if kind == "all0" else 123456
            elif kind == "padded":
                value["work"]["padded_input_tokens"] += 1
            else:
                value["work"]["forward_calls"] += 1
            value["payload_sha256"] = factual._digest(value)
            with self.subTest(kind=kind), self.assertRaisesRegex(FactualError, "QUERY_DERIVED_WORK_REQUIRED"):
                self.audit(value, signatures)

    def test_positive_finite_elapsed_and_typed_actual_work_required(self):
        original, signatures, _ = retained_fixture()
        for elapsed in (0, -1, float("nan"), float("inf"), True, "0.001"):
            value = copy.deepcopy(original)
            value["work"]["seconds"] = elapsed
            with self.subTest(elapsed=elapsed), self.assertRaisesRegex(FactualError, "WORK_ELAPSED_FINITE"):
                self.audit(value, signatures)
        value = copy.deepcopy(original)
        value["work"]["forward_calls"] = True
        with self.assertRaisesRegex(FactualError, "QUERY_DERIVED_WORK_REQUIRED"):
            self.audit(value, signatures)

    def test_nll_mean_probabilities_margins_strict_and_desired_N_are_bound_to_raw(self):
        original, signatures, _ = retained_fixture()
        for field in ("token_nll", "mean", "probability", "margin", "strict", "desired"):
            value = copy.deepcopy(original)
            observation = value["cases"][0]["neighborhood_observations"][0]
            if field == "token_nll":
                observation["target_true"]["nll_by_token"][0] = 3.0
            elif field == "mean":
                observation["target_true"]["mean_nll"] = 3.0
            elif field == "probability":
                value["cases"][0]["neighborhood_prompts_probs"][0]["target_true"] = 3.0
            elif field == "margin":
                observation["margin_true_minus_new"] += 1
            elif field == "strict":
                value["cases"][0]["neighborhood_prompts_correct"][0] = False
            else:
                observation["desired_target"] = "new"
            value["payload_sha256"] = factual._digest(value)
            with self.subTest(field=field), self.assertRaises(FactualError):
                self.audit(value, signatures)

    def test_token_bool_corruption_and_wrong_flat_plan_are_not_equal_integer_evidence(self):
        original, signatures, queries = retained_fixture()
        value = copy.deepcopy(original)
        value["cases"][0]["rewrite_observations"][0]["target_new"]["input_token_ids"][0] = True
        with self.assertRaisesRegex(FactualError, "RAW_QUERY_TOKEN_IDENTITY"):
            self.audit(value, signatures)
        changed = copy.deepcopy(queries)
        changed[0]["target_start"] += 1
        with self.assertRaisesRegex(FactualError, "FLAT_QUERY_PLAN_MISMATCH"):
            self.audit(original, signatures, query_plan=changed)
        value = copy.deepcopy(original)
        value["cases"][0]["rewrite_observations"][0]["target_new"]["token_correct"][0] = 1
        with self.assertRaisesRegex(FactualError, "CORRECTNESS_NUMERATOR_DENOMINATOR"):
            self.audit(value, signatures)

    def test_signature_order_and_causal_prefix_cannot_be_resigned_to_replace_locked_plan(self):
        value, signatures, _ = retained_fixture()
        changed = copy.deepcopy(value)
        changed["cases"].reverse()
        with self.assertRaisesRegex(FactualError, "ORDERED_CASES"):
            self.audit(changed, signatures)
        changed_signature = copy.deepcopy(signatures)
        changed_signature[0]["queries"][0]["target_start"] = 0
        with self.assertRaisesRegex(FactualError, "CAUSAL_TOKEN_PREFIX"):
            self.audit(value, changed_signature)
        changed_signature = copy.deepcopy(signatures)
        changed_signature[1]["occurrence_index"] = 1
        with self.assertRaisesRegex(FactualError, "CASE_ORDER_OR_SCHEMA"):
            self.audit(value, changed_signature)

    def test_native_per_case_padding_work_differs_from_canonical_contiguous_layout(self):
        value, signatures, queries = retained_fixture()
        native = retained_query_work(signatures, "cf", per_case=True)
        groups = [[q for q in queries if q["case_index"] == index] for index in range(2)]
        self.assertEqual(native["forward_calls"], 2)
        self.assertEqual(native["padded_input_tokens"], sum(len(group) * max(
            len(q["input_token_ids"]) for q in group) for group in groups))
        self.assertEqual(native["physical_input_tokens"], value["work"]["physical_input_tokens"])
        self.assertEqual(native["target_tokens"], value["work"]["target_tokens"])
        self.assertNotEqual(native["forward_calls"], value["work"]["forward_calls"])

    def test_zsRE_cold_reference_pred_agreement_distinct_from_loc_answer_and_tamper_rejected(self):
        value, signatures, _ = retained_fixture("zsre")
        reference_cases = []
        for case, signature in zip(value["cases"], signatures):
            observations = case["neighborhood_observations"]
            predictions = [row["predicted_token_ids"] for row in observations]
            case["neighborhood_W0_agreement"] = [True for group in predictions for _ in group]
            reference_cases.append(dict(case_id=case["case_id"], occurrence_index=case["occurrence_index"],
                queries=[q for q in signature["queries"] if q["kind"] == "neighborhood"],
                predictions=predictions))
        value["summary"] = factual.zsre(value["cases"])
        identity = dict(schema=factual.W0_SCHEMA, tokenization=factual.TOKENIZATION,
            external_identity=value["identity"]["external_identity"], ordered_occurrences=[1, 2],
            cohort_sha256=factual._digest(signatures), state="W0_COLD_BASE_MODEL")
        reference = dict(schema=factual.W0_SCHEMA, identity=identity,
            identity_sha256=factual._digest(identity), cases=reference_cases,
            evaluation=copy.deepcopy(value), raw_local_only=True)
        reference["payload_sha256"] = factual._digest(reference)
        result = self.audit(value, signatures, "zsre", w0_reference=reference)
        self.assertFalse(result["GPU_parity_claim"])
        changed = copy.deepcopy(value)
        changed["cases"][0]["neighborhood_W0_agreement"][0] = False
        changed["identity"]["W0_reference_sha256"] = reference["identity_sha256"]
        changed["identity_sha256"] = factual._digest(changed["identity"])
        with self.assertRaisesRegex(FactualError, "PROBABILITY_STRICT_OR_AGREEMENT"):
            self.audit(changed, signatures, "zsre", w0_reference=reference)


if __name__ == "__main__":
    unittest.main()
