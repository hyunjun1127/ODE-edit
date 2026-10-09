import copy
import unittest

from official.evaluation.zsre_paper import build_queries, evaluate
from official.tests.test_factual import CharacterTokenizer, TransitionLM, zr


class Tokenizer(CharacterTokenizer):
    padding_side = "right"
    def decode(self, ids):
        if isinstance(ids, int): ids = [ids]
        return "".join("B" if i == 1 else chr(i - 3) for i in ids)


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.tok, self.model = Tokenizer(), TransitionLM(consume_rng=True)
        self.model.config.model_type = "llama"
        self.rec = zr()
        self.rec["neighborhood_prompts"][0]["prompt"] = "nq question: Bob is?"

    def test_native_llama_spaces_and_loader_BOS(self):
        p = build_queries(self.tok, [self.rec], model_family="llama3")
        rw = [q for q in p["queries"] if q["group"] == "rewrite"]
        self.assertEqual(rw[0]["prompt"], "Ada is")
        self.assertEqual(rw[1]["prompt"], "Ada is  ")
        loc = [q for q in p["queries"] if q["group"] == "neighborhood"]
        self.assertEqual(len(loc), len(self.tok(" Tt")["input_ids"]))
        self.assertEqual(loc[0]["target_text"], "B")
        self.assertEqual(loc[1]["prompt"], "nq question: Bob is?B")

    def test_rightpad_bos_family_and_unexpanded_guards(self):
        self.tok.padding_side = "left"
        with self.assertRaisesRegex(ValueError, "RIGHT_PADDING"):
            build_queries(self.tok, [self.rec], model_family="llama3")
        self.tok.padding_side = "right"
        class NoBos(Tokenizer):
            def __call__(self, *args, **kwargs):
                return {"input_ids": super().__call__(*args, **kwargs)["input_ids"][1:]}
        with self.assertRaisesRegex(ValueError, "BOS_CONTRACT"):
            build_queries(NoBos(), [self.rec], model_family="llama3")
        with self.assertRaisesRegex(ValueError, "MODEL_TYPE"):
            evaluate(self.model, self.tok, [self.rec], model_family="qwen25")

    def test_eval_denominators_batching_and_state(self):
        before = copy.deepcopy(self.model.state_dict())
        a = evaluate(self.model, self.tok, [self.rec], model_family="llama3", batch_size=1)
        b = evaluate(self.model, self.tok, [self.rec], model_family="llama3", batch_size=5)
        self.assertEqual(a["cases"], b["cases"])
        self.assertEqual(a["summary"], b["summary"])
        self.assertEqual(a["summary"]["requests"], 1)
        self.assertNotIn("W0_prediction_agreement", a["summary"])
        self.assertTrue(a["model_no_mutation"] and a["RNG_restored"])
        for k, v in before.items(): self.assertTrue(v.equal(self.model.state_dict()[k]))
        self.assertTrue(all(not c["use_cache"] and not c["training"] and not c["grad"]
                            for c in self.model.calls))

    def test_fail_closed_nonfinite_and_mutation(self):
        self.model.corrupt = "nan"
        with self.assertRaisesRegex(ValueError, "NONFINITE"):
            evaluate(self.model, self.tok, [self.rec], model_family="llama3")
        self.model.corrupt = "weight"
        with self.assertRaisesRegex(ValueError, "MODEL_MUTATED"):
            evaluate(self.model, self.tok, [self.rec], model_family="llama3")

    def test_nonllama_no_extra_space_and_no_bos_slice(self):
        p = build_queries(self.tok, [self.rec], model_family="gptj")
        q = p["queries"][1]
        self.assertEqual(q["prompt"], "Ada isB")
        self.assertEqual(q["target_token_id"], 1)


if __name__ == "__main__": unittest.main()
