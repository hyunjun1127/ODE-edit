"""CPU fixtures only: no pretrained model, GPU, native target or evaluator job."""
import math
import types
import unittest
from pathlib import Path

import torch

from project.run_scripts.alpha_key_concentration_causal import observer as obs


ROOT = Path(__file__).resolve().parents[4]


class Tokenizer:
    padding_side = "right"
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    unk_token_id = 99
    add_bos_token = True

    def __call__(self, text, add_special_tokens=True):
        ids = [3] * len(text.split())
        return {"input_ids": ([1] + ids) if add_special_tokens else ids}

    def encode(self, text, add_special_tokens=False):
        table = {"truth": 4, "new": 5, "second": 6}
        return [table[word] for word in text.split()]


class Model:
    training = False

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append({k: v.clone() if torch.is_tensor(v) else v for k, v in kwargs.items()})
        shape = (*kwargs["input_ids"].shape, 40)
        logits = torch.zeros(shape, dtype=torch.float32)
        logits[..., 4] = 3
        logits[..., 5] = 2
        logits[..., 6] = 1
        return types.SimpleNamespace(logits=logits)


def record(case=1, subject="person", target="new"):
    return {"case_id": case,
            "requested_rewrite": {"prompt": "{} lives in", "subject": subject,
                                  "relation_id": "P1", "target_new": {"str": target},
                                  "target_true": {"str": "truth"}},
            "paraphrase_prompts": ["paraphrase one", "another long paraphrase"],
            "neighborhood_prompts": [f"neighbor {i}" for i in range(10)]}


def history_fixture():
    records = [record(i, f"Person{i}") for i in range(512)]
    panel = {"n": 512, "records": [
        {"case_id": r["case_id"], "subject": r["requested_rewrite"]["subject"],
         "relation": "P1", "target": "new", "write_batch": 1,
         "timestamp_checkpoint": "B001"} for r in records]}
    return records, panel


class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = obs.bind_native_evaluator(ROOT)

    def test_exact_native_sources(self):
        self.assertEqual(obs.NATIVE_HELPER_COMMIT, "1075540b45c29269e690ac63aae44758d8d63174")
        self.assertTrue(callable(self.kernel.evaluate_pairs))

    def test_bad_source_fails_closed(self):
        with self.assertRaisesRegex(obs.ObserverBoundary, "SOURCE_MISMATCH"):
            obs.bind_native_evaluator(ROOT / "not-a-source")

    def test_native_scalar_and_input_parity(self):
        tok = Tokenizer()
        pairs = [self.kernel.PromptTarget(1, "r", 0, "short", "new second"),
                 self.kernel.PromptTarget(2, "r", 0, "longer prompt words", "truth")]
        expected_model, observed_model = Model(), Model()
        expected = self.kernel.evaluate_pairs(expected_model, tok, pairs,
                                              device=torch.device("cpu"), microbatch_size=2)
        result = obs.evaluate_pairs(observed_model, tok, pairs, device=torch.device("cpu"),
                                    endpoint_id="sealed-W", postseal=True,
                                    source_root=ROOT, microbatch_size=2, topk=32)
        for a, b in zip(expected, result["rows"]):
            for key, value in a.items():
                self.assertEqual(value, b[key])
        for a, b in zip(expected_model.calls, observed_model.calls):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                if torch.is_tensor(a[key]):
                    self.assertTrue(torch.equal(a[key], b[key]))
                else:
                    self.assertEqual(a[key], b[key])
        self.assertEqual(result["counter"]["extra_forward_calls"], 0)
        self.assertEqual(result["counter"]["forward_calls"], 1)
        self.assertEqual(result["counter"]["scored_target_tokens"], 3)
        self.assertEqual(result["rows"][0]["full_vocab"]["scoring_positions_unpadded"], [1, 2])
        self.assertEqual(result["rows"][0]["full_vocab"]["padding_tokens"], 1)
        self.assertTrue(result["rows"][0]["full_vocab"]["prompt_starts_with_bos"])

    def test_full_vocab_tie_and_top32(self):
        logits = torch.zeros((1, 40), dtype=torch.float32)
        logits[0, 2] = logits[0, 4] = 3
        result = obs._logit_diagnostics(logits, [4], topk=32)
        self.assertEqual(result["argmax_token_ids"], [2])
        self.assertEqual(result["argmax_tie_counts"], [2])
        self.assertEqual(result["target_minus_best_other_margin"], [0])
        self.assertFalse(result["all_unique_argmax_correct"])
        self.assertEqual(result["top_token_ids"][0][:4], [2, 4, 0, 1])
        self.assertEqual(len(result["top_token_ids"][0]), 32)
        self.assertEqual(result["logsumexp"], torch.logsumexp(logits, -1).tolist())

    def test_lowest_target_id_tie_is_argmax_not_unique(self):
        logits = torch.tensor([[1., 1., 0.]])
        result = obs._logit_diagnostics(logits, [0])
        self.assertEqual(result["argmax_token_ids"], [0])
        self.assertFalse(result["all_unique_argmax_correct"])

    def test_nonfinite_is_technical_error(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaisesRegex(obs.ObserverBoundary, "NONFINITE"):
                obs._logit_diagnostics(torch.tensor([[0., value]]), [0])

    def test_double_logits_not_silently_cast(self):
        with self.assertRaisesRegex(obs.ObserverBoundary, "NOT_FP32"):
            obs._logit_diagnostics(torch.zeros(1, 2, dtype=torch.float64), [0])

    def test_postseal_required(self):
        with self.assertRaisesRegex(obs.ObserverBoundary, "POSTSEAL"):
            obs.evaluate_pairs(Model(), Tokenizer(), [], device="cpu", endpoint_id="x",
                                postseal=False, source_root=ROOT)

    def test_model_eval_and_right_tokenizer_required(self):
        model = Model()
        model.training = True
        with self.assertRaisesRegex(obs.ObserverBoundary, "NOT_EVAL"):
            obs.evaluate_pairs(model, Tokenizer(), [], device="cpu", endpoint_id="x",
                                postseal=True, source_root=ROOT)
        tok = Tokenizer()
        tok.padding_side = "left"
        with self.assertRaisesRegex(obs.ObserverBoundary, "TOKENIZER"):
            obs.evaluate_pairs(Model(), tok, [], device="cpu", endpoint_id="x",
                                postseal=True, source_root=ROOT)

    def evaluated_record(self):
        return obs.evaluate_rpn(Model(), Tokenizer(), [record()], device="cpu",
                                 endpoint_id="sealed-W", postseal=True, source_root=ROOT)

    def test_rpn_denominators_and_observer_only(self):
        result = self.evaluated_record()
        compact = result["compact"]
        self.assertEqual({k: v["denominator"] for k, v in compact["metrics"].items()},
                         {"RS": 1, "PS": 2, "NS": 10})
        self.assertEqual({k: v["numerator"] for k, v in compact["metrics"].items()},
                         {"RS": 0, "PS": 0, "NS": 10})
        self.assertEqual(compact["counter"]["extra_forward_calls"], 0)
        self.assertEqual(compact["controller_influence"], 0)
        self.assertEqual(compact["state_nonmutation"], "CALLER_VERIFICATION_REQUIRED")
        self.assertIn("prompt", result["raw_local_only"]["rewrite_target_new"][0])
        self.assertNotIn("prompt", compact["metrics"]["RS"]["rows"][0])

    def test_canonical_reducer_matches_original(self):
        from project.run_scripts.blue_alphaedit_sequential_comparison.evaluation import reduce
        result = self.evaluated_record()
        original = reduce(result["raw_local_only"])
        for tag, native in original.items():
            actual = result["compact"]["metrics"][tag]
            self.assertEqual(native["numerator"], actual["numerator"])
            self.assertEqual(native["denominator"], actual["denominator"])
            for old, new in zip(native["rows"], actual["rows"]):
                for key in ("identity", "success", "new_nll", "true_nll", "new_strict", "true_strict"):
                    self.assertEqual(old[key], new[key])

    def test_unicode_identity_matches_original(self):
        from project.run_scripts.blue_alphaedit_sequential_comparison.integrity import digest
        self.assertEqual(obs.digest(["José", "서울"]), digest(["José", "서울"]))

    def test_nll_tie_is_failure_for_all_categories(self):
        raw = self.evaluated_record()["raw_local_only"]
        for rows in raw.values():
            for row in rows:
                row["nll"] = 1.0
        reduced = obs.reduce_rpn(raw)
        for result in reduced.values():
            self.assertEqual(result["numerator"], 0)
            self.assertEqual(result["nll_ties"], result["denominator"])

    def test_pair_endpoint_mismatch_is_not_reused(self):
        raw = self.evaluated_record()["raw_local_only"]
        raw["rewrite_target_true"][0]["endpoint_id"] = "other-W"
        with self.assertRaisesRegex(obs.ObserverBoundary, "PAIR_ENDPOINT"):
            obs.reduce_rpn(raw)

    def test_pair_identity_and_missing_are_errors(self):
        raw = self.evaluated_record()["raw_local_only"]
        raw["rewrite_target_true"][0]["prompt"] = "different"
        with self.assertRaisesRegex(obs.ObserverBoundary, "PAIR_IDENTITY"):
            obs.reduce_rpn(raw)
        raw["rewrite_target_true"] = []
        with self.assertRaisesRegex(obs.ObserverBoundary, "PAIR_DENOMINATOR"):
            obs.reduce_rpn(raw)

    def test_history_overwrite_functional_only(self):
        records, panel = history_fixture()
        records.append(record(600, "Person0", "truth"))
        result = obs.history_masks(panel, records)
        self.assertEqual(result["statistics_weight_sum"], 512)
        self.assertEqual(result["statistics_denominator"], 512)
        self.assertEqual(result["functional_current_valid"], 511)
        self.assertEqual(result["functional_superseded"], 1)
        self.assertFalse(result["rows"][0]["functional_current_valid"])
        self.assertEqual(result["rows"][0]["latest_event_case_id"], 600)
        self.assertEqual(result["rows"][0]["statistics_weight"], 1)

    def test_history_same_target_repeat_stays_valid(self):
        records, panel = history_fixture()
        records.append(record(600, "Person0", "new"))
        result = obs.history_masks(panel, records)
        self.assertEqual(result["functional_current_valid"], 512)
        self.assertEqual(result["rows"][0]["latest_event_case_id"], 600)

    def test_history_raw_identity_not_normalized(self):
        records, panel = history_fixture()
        records.append(record(600, "person0", "truth"))
        self.assertEqual(obs.history_masks(panel, records)["functional_current_valid"], 512)

    def test_history_missing_bank_event_fails(self):
        records, panel = history_fixture()
        with self.assertRaisesRegex(obs.ObserverBoundary, "NOT_IDENTICAL_RECEIVED"):
            obs.history_masks(panel, records[1:])

    def test_history_duplicate_case_fails(self):
        records, panel = history_fixture()
        with self.assertRaisesRegex(obs.ObserverBoundary, "DUPLICATE_RECEIVED"):
            obs.history_masks(panel, records + [records[0]])
        panel["records"][-1] = panel["records"][0]
        with self.assertRaisesRegex(obs.ObserverBoundary, "DUPLICATE_BANK"):
            obs.history_masks(panel, records)

    def test_n512_denominator_not_reselected(self):
        with self.assertRaisesRegex(obs.ObserverBoundary, "N512_CARDINALITY"):
            obs.evaluate_neighborhood512(Model(), Tokenizer(), {"n": 511, "records": []},
                                          device="cpu", endpoint_id="x", postseal=True, source_root=ROOT)

    def test_n512_all_rows_top32_and_denominator(self):
        panel = {"n": 512, "records": [
            {"case_id": i, "prompt_index": 0, "prompt": f"prompt {i}",
             "target_true": "truth", "paired_edit_target": "new",
             "base_true_nll": 1., "base_identity": obs.digest([i, 0, f"prompt {i}", "new", "truth"])}
             for i in range(512)]}
        result = obs.evaluate_neighborhood512(Model(), Tokenizer(), panel, device="cpu",
                                               endpoint_id="x", postseal=True, source_root=ROOT,
                                               microbatch_size=16, edit_target_token_ids=[4, 5])
        compact = result["compact"]
        self.assertEqual(compact["denominator"], 512)
        self.assertEqual(compact["true_argmax_correct"], 512)
        self.assertEqual(compact["counter"]["forward_calls"], 64)
        self.assertEqual(compact["counter"]["scored_target_tokens"], 1024)
        self.assertEqual(compact["rows"][0]["top32"]["topk_requested"], 32)
        self.assertTrue(compact["rows"][0]["argmax_in_seen_edit_target_tokens"])

    def test_protection_loss_recovery_exact_ids(self):
        a = [{"identity": "a", "true_argmax_correct": True, "true_nll": 1.},
             {"identity": "b", "true_argmax_correct": False, "true_nll": 3.}]
        b = [{"identity": "a", "true_argmax_correct": False, "true_nll": 2.},
             {"identity": "b", "true_argmax_correct": True, "true_nll": 2.}]
        result = obs.protection_transitions(a, b)
        self.assertEqual((result["lost"], result["recovered"], result["entry_successes"]), (1, 1, 1))
        with self.assertRaisesRegex(obs.ObserverBoundary, "IDENTITY"):
            obs.protection_transitions(a, b[::-1])


if __name__ == "__main__":
    unittest.main()
