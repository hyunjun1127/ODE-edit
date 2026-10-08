"""CPU regressions exercise the actual model-forward evaluator, not GPU parity."""
import copy
import json
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from official.evaluation.factual import (FactualError, build_zsre_w0_reference,
                                         evaluate_counterfact, evaluate_zsre)


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
        self.assertEqual(wrong["evaluation"]["summary"]["Specificity"], 100)
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
        self.assertNotIn("Specificity", result["summary"])
        self.assertEqual(result["summary"]["Specificity_availability"],
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


if __name__ == "__main__":
    unittest.main()
