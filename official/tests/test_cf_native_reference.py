"""Deterministic CPU-fixture regressions; NEVER actual pretrained/GPU parity."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from official.evaluation.cf_native_reference import (
    CFNativeReferenceError, CPU_SCOPE, FULL_SCOPE, MATCHED_SCOPE, SMOKE_SCOPE, SOURCE_BYTES,
    SOURCE_PATH, SOURCE_SHA256, compare_native_counterfact, evaluate_native_counterfact,
    load_original_counterfact,
)
from official.evaluation.factual import evaluate_counterfact


class NativeCharacterTokenizer:
    """Exact GPT-J no-BOS or genuine Llama BOS, with native batched padding."""
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
    model_max_length, padding_side = 512, "right"
    name_or_path = "TEST_ONLY_CHARACTER_TOKENIZER"

    def __init__(self, bos=False):
        self.add_bos_token = bos

    def __call__(self, text, add_special_tokens=True, truncation=False,
                 padding=False, return_tensors=None):
        assert truncation is False
        def tokens(value):
            return ([self.bos_token_id] if self.add_bos_token and add_special_tokens else []) + [ord(char)+3 for char in value]
        if isinstance(text, str):
            return {"input_ids": tokens(text)}
        rows = [tokens(value) for value in text]
        if return_tensors is None:
            return {"input_ids": rows}
        assert padding is True and return_tensors == "pt"
        width = max(map(len, rows))
        return {"input_ids": torch.tensor([row+[0]*(width-len(row)) for row in rows]),
                "attention_mask": torch.tensor([[1]*len(row)+[0]*(width-len(row)) for row in rows])}


class FixtureLM(torch.nn.Module):
    """Fixed causal CPU model; no fitting, generation, or canonical-logit input."""
    def __init__(self, llama=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.child = torch.nn.Dropout()
        self.config = SimpleNamespace(_name_or_path="TEST_ONLY_llama" if llama else "TEST_ONLY_gpt-j",
            use_cache=False, max_position_embeddings=512, _attn_implementation="eager")
        self.calls, self.consume_rng, self.corrupt = [], False, None

    def forward(self, input_ids, attention_mask, use_cache=None):
        self.calls.append(dict(ids=input_ids.tolist(), mask=attention_mask.tolist(),
            use_cache=use_cache, training=self.training, grad=torch.is_grad_enabled()))
        if self.consume_rng:
            random.random(); np.random.random(); torch.rand(1)
        if self.corrupt == "parameter":
            self.weight.add_(1)
        if self.corrupt == "config":
            self.config.use_cache = True
        if self.corrupt == "input":
            input_ids.add_(1)
        if self.corrupt == "error":
            raise RuntimeError("TEST_ONLY originating forward failure")
        logits = torch.full((*input_ids.shape, 512), -4.)
        expected = torch.full_like(input_ids, ord(" ")+3)
        true_prompt = input_ids.eq(ord("B")+3).cumsum(1).gt(0)
        space = input_ids.eq(ord(" ")+3)
        expected[space & ~true_prompt] = ord("N")+3
        expected[space & true_prompt] = ord("T")+3
        expected[input_ids.eq(ord("N")+3)] = ord("n")+3
        expected[input_ids.eq(ord("T")+3)] = ord("t")+3
        logits.scatter_(2, expected[..., None], 4.)
        if self.corrupt == "nan":
            logits[:, :, 0] = float("nan")
        return SimpleNamespace(logits=logits)


def record(index=1, *, paraphrases=None):
    return dict(case_id=100+index, occurrence_index=index,
        requested_rewrite=dict(prompt="{} is", subject="Ada", target_new={"str": "Nn"}, target_true={"str": "Tt"}),
        paraphrase_prompts=paraphrases or ["Ada was"], neighborhood_prompts=["Bob is", "Bob at"])


IDENTITY = dict(model_identity="TEST_ONLY_FIXED_CPU_MODEL", tokenizer_identity="TEST_ONLY_CHARACTER_TOKENIZER",
                state_identity="TEST_ONLY_W0_NO_FIT")
CPU_OPTIONS = dict(identity=IDENTITY, evidence_scope=CPU_SCOPE, test_only_cpu=True)


class CFNativeReferenceTests(unittest.TestCase):
    def setUp(self):
        self.model, self.tok = FixtureLM(), NativeCharacterTokenizer()

    def canonical(self, rows):
        return evaluate_counterfact(self.model, self.tok, rows, batch_size=3, identity=IDENTITY)

    def compare(self, rows, canonical=None, **options):
        if canonical is None:
            canonical = self.canonical(rows)
        return compare_native_counterfact(self.model, self.tok, rows, canonical, **dict(CPU_OPTIONS, **options))

    def test_hash_verified_unchanged_AST_has_only_minimal_globals(self):
        source = SOURCE_PATH.read_bytes()
        self.assertEqual((len(source), hashlib.sha256(source).hexdigest()), (SOURCE_BYTES, SOURCE_SHA256))
        fn = load_original_counterfact()
        self.assertEqual(set(fn.__globals__), {"__builtins__", "typing", "np", "torch", "test_batch_prediction"})
        node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == fn.__name__)
        expected = compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE_PATH), "exec")
        self.assertEqual(fn.__code__.co_code, next(item.co_code for item in expected.co_consts if hasattr(item, "co_name") and item.co_name == fn.__name__))

    def test_GPTJ_noBOS_independent_original_forwards_multi_target_pad_and_scope(self):
        rows = [record(1), record(2, paraphrases=["Ada was", "Ada much longer is"])]
        canonical = self.canonical(rows)
        before = len(self.model.calls)
        result = self.compare(rows, canonical)
        self.assertEqual(result["status"], "CPU_FIXTURE_PASS")
        self.assertEqual(len(self.model.calls)-before, 2)
        self.assertEqual(result["work"]["forward_calls"], 2)
        self.assertEqual(result["work"]["candidate_sequences"], 18)
        self.assertEqual(result["work"]["target_tokens"], 54)
        self.assertGreater(result["work"]["padded_input_tokens"], result["work"]["physical_input_tokens"])
        self.assertEqual(result["evidence"][SMOKE_SCOPE], "NOT_OBSERVED")
        self.assertEqual(result["evidence"][FULL_SCOPE], "NOT_OBSERVED")
        calls = self.model.calls[before:]
        self.assertTrue(all(call["use_cache"] is None and not call["training"] and not call["grad"] for call in calls))
        self.assertNotIn(self.tok.bos_token_id, calls[0]["ids"][0])
        self.assertTrue(all(mask == sorted(mask, reverse=True) for call in calls for mask in call["mask"]))
        json.dumps(result, allow_nan=False)

    def test_Llama_genuineBOS_paired_prefix_and_logit_slice(self):
        self.model, self.tok = FixtureLM(llama=True), NativeCharacterTokenizer(bos=True)
        result = self.compare([record()])
        self.assertEqual(result["status"], "CPU_FIXTURE_PASS")
        query = result["native"]["case_signatures"][0]["queries"][0]
        self.assertEqual(query["input_token_ids"][0], self.tok.bos_token_id)
        self.assertNotIn(self.tok.bos_token_id, query["target_token_ids"])
        self.assertEqual(result["native"]["summary"]["Score"], 100.)

    def test_request_macro_not_prompt_micro_original_display_separate(self):
        result = self.compare([record(1, paraphrases=["Bob was"]*3), record(2)])
        self.assertEqual(result["status"], "CPU_FIXTURE_PASS")
        self.assertEqual(result["native"]["summary"]["Generalization"], 50.)
        self.assertEqual(result["native"]["summary"]["Score"], 75.)
        self.assertEqual(result["native"]["original_display"]["success"]["Generalization"], [50., 50.])
        self.assertEqual(result["display_mismatches"], [])

    def test_strict_ties_fail_without_zero_missing_or_tie_relaxation(self):
        row = record()
        row["requested_rewrite"]["target_true"] = {"str": "Nn"}
        result = self.compare([row])
        self.assertEqual(result["status"], "CPU_FIXTURE_PASS")
        self.assertEqual(result["native"]["summary"]["Efficacy"], 0.)
        self.assertEqual(result["native"]["summary"]["Specificity"], 0.)

    def test_nll_large_mismatch_reports_fixed_threshold_without_extra_canonical_forward(self):
        rows = [record()]
        canonical = self.canonical(rows)
        case = canonical["cases"][0]
        case["rewrite_prompts_probs"][0]["target_new"] += .01
        case["rewrite_observations"][0]["target_new"]["mean_nll"] += .01
        before = len(self.model.calls)
        result = self.compare(rows, canonical)
        self.assertEqual(result["status"], "MISMATCH")
        self.assertEqual(len(self.model.calls)-before, 1)
        self.assertIn("NLL", {row["type"] for row in result["mismatches"]})
        self.assertEqual(result["tolerances"]["nll_abs_nats"], 1e-4)

    def test_near_tie_boolean_flip_fails_even_when_nll_within_tolerance(self):
        row = record()
        row["requested_rewrite"]["target_true"] = {"str": "Nn"}
        canonical = self.canonical([row])
        case = canonical["cases"][0]
        case["rewrite_prompts_probs"][0]["target_new"] -= 1e-6
        case["rewrite_observations"][0]["target_new"]["mean_nll"] -= 1e-6
        result = self.compare([row], canonical)
        kinds = {item["type"] for item in result["mismatches"]}
        self.assertEqual(result["status"], "MISMATCH")
        self.assertIn("STRICT_NLL_PREFERENCE", kinds)
        self.assertNotIn("NLL", kinds)

    def test_desired_argmax_boolean_mismatch_is_exact(self):
        rows = [record()]
        canonical = self.canonical(rows)
        case = canonical["cases"][0]
        case["rewrite_prompts_correct"][0] = False
        case["rewrite_observations"][0]["target_new"]["strict_correct"] = False
        result = self.compare(rows, canonical)
        self.assertIn("STRICT_DESIRED_ARGMAX", {row["type"] for row in result["mismatches"]})
        self.assertEqual(result["status"], "MISMATCH")

    def test_identity_hash_cohort_token_rows_and_state_fail_before_native_forward(self):
        rows = [record()]
        canonical = self.canonical(rows)
        for mutate in (
            lambda value: value.update(identity_sha256="0"*64),
            lambda value: value["cases"][0].update(case_id=-1),
            lambda value: value["cases"][0]["rewrite_observations"][0]["target_new"]["input_token_ids"].append(9),
        ):
            altered = copy.deepcopy(canonical)
            mutate(altered)
            before = len(self.model.calls)
            result = self.compare(rows, altered)
            self.assertEqual(result["status"], "NOT_QUALIFIED")
            self.assertEqual(len(self.model.calls), before)
        result = self.compare(rows, canonical, state_callback=lambda: "WRONG_STATE")
        self.assertEqual(result["status"], "NOT_QUALIFIED")
        self.assertIn("STATE_IDENTITY", result["reasons"][0]["code"])

    def test_existing_external_identity_reuse_preserves_raw_and_separate_explicit_binding(self):
        rows, external = [record()], {"revision": "TEST_ONLY_ORIGINAL_RUNNER", "endpoint": "TEST_ONLY_B3"}
        canonical = evaluate_counterfact(self.model, self.tok, rows, identity=external)
        original = copy.deepcopy(canonical)
        result = compare_native_counterfact(self.model, self.tok, rows, canonical,
            identity=external, evidence_scope=CPU_SCOPE, test_only_cpu=True,
            model_identity=IDENTITY["model_identity"], tokenizer_identity=IDENTITY["tokenizer_identity"],
            state_identity=IDENTITY["state_identity"], state_callback=lambda: IDENTITY["state_identity"])
        self.assertEqual(result["status"], "CPU_FIXTURE_PASS")
        self.assertEqual(canonical, original)
        self.assertEqual(result["native"]["identity"]["external_identity"], external)
        self.assertEqual(result["native"]["identity"]["reference_binding"], IDENTITY)
        self.assertEqual(result["evidence"][MATCHED_SCOPE], "NOT_OBSERVED")

    def test_wrong_BOS_family_and_concat_boundary_are_not_adapted(self):
        for llama, bos, expected in ((True, False, "GENUINE_TARGET_BOS"), (False, True, "TARGET_BOS_FORBIDDEN")):
            result = evaluate_native_counterfact(FixtureLM(llama), NativeCharacterTokenizer(bos), [record()], **CPU_OPTIONS)
            self.assertEqual(result["status"], "NOT_QUALIFIED")
            self.assertIn(expected, result["reasons"][0]["code"])
        class BadBoundary(NativeCharacterTokenizer):
            def __call__(self, text, **kwargs):
                value = super().__call__(text, **kwargs)
                if isinstance(text, str) and text.endswith(" Nn") and not text.startswith(" "):
                    value["input_ids"][-1] += 1
                return value
        result = evaluate_native_counterfact(self.model, BadBoundary(), [record()], **CPU_OPTIONS)
        self.assertIn("CONCAT_BOUNDARY", result["reasons"][0]["code"])
        self.assertEqual(len(self.model.calls), 0)

    def test_hash_tamper_no_generation_imports_or_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"source.txt"
            path.write_bytes(SOURCE_PATH.read_bytes()+b"\n")
            with self.assertRaisesRegex(CFNativeReferenceError, "SOURCE_SHA256"):
                load_original_counterfact(path)
            result = evaluate_native_counterfact(self.model, self.tok, [record()], source_path=path, **CPU_OPTIONS)
            self.assertEqual(result["status"], "NOT_QUALIFIED")
            self.assertEqual(result["work"]["forward_calls"], 0)

    def test_malformed_rows_and_nonfinite_input_are_typed_before_forward(self):
        for rows in ([None], [dict(case_id=1, occurrence_index=1)], [dict(record(), extra=float("nan"))]):
            result = evaluate_native_counterfact(self.model, self.tok, rows, **CPU_OPTIONS)
            self.assertEqual(result["status"], "NOT_QUALIFIED")
            self.assertEqual(result["work"]["forward_calls"], 0)

    def test_display_and_request_count_are_separate_failure_types(self):
        rows = [record()]
        canonical = self.canonical(rows)
        canonical["summary"]["Score_AlphaEdit_display"] += .01
        result = self.compare(rows, canonical)
        self.assertEqual(result["status"], "MISMATCH")
        self.assertEqual(result["mismatches"], [])
        self.assertEqual(result["display_mismatches"][0]["type"], "ORIGINAL_DISPLAY_SCORE")
        canonical["summary"]["Score_AlphaEdit_display"] -= .01
        canonical["summary"]["requests"] += 1e-11
        result = self.compare(rows, canonical)
        self.assertIn("requests", {item.get("metric") for item in result["mismatches"]})

    def test_modes_rng_inputs_and_callback_state_restored(self):
        self.model.train(); self.model.child.eval(); self.model.consume_rng = True
        rows = [record()]
        original = copy.deepcopy(rows)
        canonical = self.canonical(rows)
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        result = self.compare(rows, canonical, state_callback=lambda: IDENTITY["state_identity"])
        self.assertEqual(result["status"], "CPU_FIXTURE_PASS")
        self.assertEqual(rows, original)
        self.assertTrue(self.model.training); self.assertFalse(self.model.child.training)
        self.assertEqual(random.getstate(), py_state)
        self.assertEqual(np.random.get_state()[0], np_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_guard_mutations_rejected_and_originating_error_restores(self):
        for corrupt, expected in (("parameter", "MODEL_MUTATED"), ("config", "MODEL_MUTATED"),
                                  ("input", "INPUT_MUTATED"), ("nan", "NUMERIC")):
            model = FixtureLM(); model.corrupt = corrupt
            result = evaluate_native_counterfact(model, self.tok, [record()], **CPU_OPTIONS)
            self.assertEqual(result["status"], "NOT_QUALIFIED")
            self.assertIn(expected, result["reasons"][0]["code"])
            self.assertEqual(result["work"]["forward_calls"], 1)
        self.model.corrupt, self.model.consume_rng = "error", True
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "originating forward"):
            evaluate_native_counterfact(self.model, self.tok, [record()], **CPU_OPTIONS)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(self.model.training)

    def test_cpu_never_qualifies_actual_scope_and_other_preconditions_do_not_mutate(self):
        self.model.eval()
        result = evaluate_native_counterfact(self.model, self.tok, [record(i) for i in range(1, 5)], identity=IDENTITY)
        self.assertEqual(result["status"], "NOT_QUALIFIED")
        self.assertIn("REAL_CUDA", result["reasons"][0]["code"])
        self.assertEqual(result["evidence"][SMOKE_SCOPE], "NOT_OBSERVED")
        result = evaluate_native_counterfact(self.model, self.tok, [record()], identity=IDENTITY,
            evidence_scope=MATCHED_SCOPE, state_callback=lambda: IDENTITY["state_identity"])
        self.assertEqual(result["status"], "NOT_QUALIFIED")
        self.assertEqual(result["evidence"][MATCHED_SCOPE], "NOT_OBSERVED")
        result = evaluate_native_counterfact(self.model, self.tok, [record()], **dict(CPU_OPTIONS, evidence_scope=SMOKE_SCOPE))
        self.assertIn("TEST_ONLY_SCOPE", result["reasons"][0]["code"])
        self.tok.padding_side = "left"
        result = evaluate_native_counterfact(self.model, self.tok, [record()], **CPU_OPTIONS)
        self.assertIn("RIGHT_PADDING", result["reasons"][0]["code"])
        self.assertEqual(self.tok.padding_side, "left")
        self.assertEqual(self.model.calls, [])


if __name__ == "__main__":
    unittest.main()
