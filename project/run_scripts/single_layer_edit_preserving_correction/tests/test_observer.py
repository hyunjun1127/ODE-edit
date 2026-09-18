"""CPU-only observer fixtures; no pretrained model, GPU, native fit or editor."""
import copy
import hashlib
import json
from pathlib import Path
import random
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from project.run_scripts.baseline_mechanism_first.evaluation import bind_evaluation_sources
from project.run_scripts.single_layer_edit_preserving_correction.common import digest, tensor_sha
from project.run_scripts.single_layer_edit_preserving_correction.observer import (
    CanonicalObserver, ObserverBoundary, require_selection_seal, w0_correct_retention)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    bos_token_id = 1
    unk_token_id = 3
    padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        tokens = [4+ord(c)%12 for c in text.strip()]
        return [1]+tokens if add_special_tokens else tokens

    def __call__(self, text, add_special_tokens=True):
        return dict(input_ids=self.encode(text, add_special_tokens))


class GenerationConfig:
    eos_token_id = [2, 15]

    def to_dict(self):
        return dict(eos_token_id=list(self.eos_token_id))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(5)])
        self.model.layers[4].mlp = torch.nn.Module()
        self.model.layers[4].mlp.down_proj = torch.nn.Linear(4, 4, bias=False)
        self.embedding = torch.nn.Embedding(16, 4)
        self.head = torch.nn.Linear(4, 16, bias=False)
        self.config = types.SimpleNamespace(eos_token_id=[2, 15])
        self.generation_config = GenerationConfig()
        self.generation_calls = []
        self.eval().requires_grad_(False)

    def forward(self, input_ids, attention_mask, use_cache=False):
        hidden = self.embedding(input_ids)*attention_mask[..., None]
        hidden = self.model.layers[4].mlp.down_proj(hidden).cumsum(1)
        return types.SimpleNamespace(logits=self.head(torch.tanh(hidden)))

    def generate(self, **kwargs):
        self.generation_calls.append({k:v for k,v in kwargs.items() if k not in ("input_ids", "attention_mask")})
        self(input_ids=kwargs["input_ids"], attention_mask=kwargs["attention_mask"], use_cache=False)
        suffix = torch.tensor([[4+ord("x")%12, 4+ord("y")%12, 15]], device=kwargs["input_ids"].device)
        return torch.cat((kwargs["input_ids"], suffix), 1)


def record(case=1, target="xy"):
    return dict(case_id=case, requested_rewrite=dict(prompt="{} lives", subject="AB",
                target_new=dict(str=target), target_true=dict(str="z")),
                paraphrase_prompts=["one", "long other"],
                neighborhood_prompts=["n"+str(i) for i in range(10)])


class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_root = Path(__file__).resolve().parents[2]
        bind_evaluation_sources(source_root/"blue_alphaedit_sequential_comparison", helper_root=source_root)

    def setUp(self):
        torch.manual_seed(867)
        torch.set_num_threads(1)
        self.model, self.tok = Toy(), Tokenizer()
        self.observer = CanonicalObserver(self.model, self.tok, runtime_identity="b"*64)
        self.weight = self.model.model.layers[4].mlp.down_proj.weight.detach().clone()
        self.records = [record()]

    def seal(self, records=None, weight=None):
        records = self.records if records is None else records
        weight = self.weight if weight is None else weight
        return dict(status="SELECTION_SEALED", episode_id="M001", endpoint_id="EN-F",
                    endpoint_weight_sha256=tensor_sha(weight),
                    request_order_sha256=digest([r["case_id"] for r in records]),
                    selection_ledger_sha256="a"*64)

    def test_seal_required_before_official_prompt_access(self):
        incomplete_record = dict(case_id=1)
        with self.assertRaisesRegex(ObserverBoundary, "SELECTION_SEAL"):
            self.observer.observe([incomplete_record], self.weight, selection_seal={})
        self.assertEqual(self.observer.work["model_forward_calls"], 0)
        with self.assertRaisesRegex(ObserverBoundary, "ENDPOINT_BYTES"):
            self.observer.observe(self.records, self.weight+.1, selection_seal=self.seal())
        self.assertEqual(self.observer.work["model_forward_calls"], 0)

    def test_exact_historical_pair_calls_cardinality_tokens_and_ties(self):
        evaluator = self.observer.bindings["evaluator"]
        with patch.object(evaluator, "evaluate_pairs", wraps=evaluator.evaluate_pairs) as called:
            out = self.observer.observe(self.records, self.weight, selection_seal=self.seal())
        self.assertEqual(called.call_count, 6)
        self.assertTrue(all(c.kwargs["microbatch_size"] == 16 for c in called.call_args_list))
        self.assertEqual([out["metrics"][k]["denominator"] for k in ("RS", "PS", "NS")], [1, 2, 10])
        self.assertEqual(out["work"]["canonical_pair_rows_computed"], 26)
        self.assertEqual(out["strict"]["denominator"], 1)
        self.assertEqual(out["generation_denominator"], 1)
        self.assertTrue(out["generation"][0]["target_prefix_match"])
        self.assertTrue(out["generation"][0]["stopped_on_original_eos"])
        self.assertEqual(self.model.generation_calls[0]["eos_token_id"], [2, 15])
        self.assertEqual(self.model.generation_calls[0]["max_new_tokens"], 32)
        self.assertFalse(self.model.generation_calls[0]["do_sample"])
        for tag in ("RS", "PS", "NS"):
            for row in out["metrics"][tag]["rows"]:
                expected = row["true_nll"] < row["new_nll"] if tag == "NS" else row["new_nll"] < row["true_nll"]
                self.assertEqual(row["success"], expected)
        ties = copy.deepcopy(out["raw"])
        for rows in ties.values():
            for row in rows:
                row["nll"] = 1.
        reduced = self.observer.bindings["historical"].reduce(ties)
        self.assertEqual([reduced[k]["numerator"] for k in ("RS", "PS", "NS")], [0, 0, 0])

    def test_partial_raw_reuse_and_new_greedy_only(self):
        out = self.observer.observe(self.records, self.weight, selection_seal=self.seal(), greedy=False)
        reused = self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reuse=out, greedy=True)
        self.assertEqual(reused["work"]["canonical_pair_rows_computed"], 0)
        self.assertEqual(reused["work"]["canonical_pair_rows_reused"], 26)
        self.assertEqual(reused["work"]["greedy_requests_computed"], 1)
        again = self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reuse=reused)
        self.assertEqual(again["work"]["model_forward_calls"], 0)
        self.assertEqual(again["work"]["greedy_requests_reused"], 1)
        self.assertEqual(again["raw"], out["raw"])

    def test_reuse_rejects_endpoint_input_or_runtime_mismatch(self):
        out = self.observer.observe(self.records, self.weight, selection_seal=self.seal(), greedy=False)
        corrupted = copy.deepcopy(out)
        corrupted["compatibility"]["runtime_identity"] = "c"*64
        with self.assertRaisesRegex(ObserverBoundary, "REUSE_RUNTIME"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reuse=corrupted)
        corrupted = copy.deepcopy(out)
        corrupted["raw"]["rewrite_target_new"][0]["target"] = "fake"
        with self.assertRaisesRegex(ObserverBoundary, "REUSE_UNEXPECTED"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reuse=corrupted)

    def test_missing_group_recomputes_original_mb_group_not_repartition(self):
        records = [record(i+1) for i in range(3)]
        out = self.observer.observe(records, self.weight, selection_seal=self.seal(records), greedy=False)
        partial = copy.deepcopy(out)
        partial["raw"]["locality_target_new"].pop(0)
        reused = self.observer.observe(records, self.weight, selection_seal=self.seal(records), reuse=partial, greedy=False)
        self.assertEqual(reused["work"]["canonical_microbatches_computed"], 1)
        self.assertEqual(reused["work"]["canonical_pair_rows_computed"], 16)
        self.assertEqual(reused["raw"], out["raw"])

    def test_exact_weight_restore_rng_guard_and_failure_rollback(self):
        candidate = self.weight+.01
        before = torch.get_rng_state().clone()
        out = self.observer.observe(self.records, candidate, selection_seal=self.seal(weight=candidate))
        self.assertTrue(torch.equal(self.model.model.layers[4].mlp.down_proj.weight, self.weight))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(out["RNG_unchanged_and_restored"])
        original = self.model.generate
        def mutate_rng(**kwargs):
            torch.rand(1); random.random(); np.random.rand()
            return original(**kwargs)
        with patch.object(self.model, "generate", mutate_rng):
            with self.assertRaisesRegex(ObserverBoundary, "RNG_MUTATED_AND_RESTORED"):
                self.observer.observe(self.records, candidate, selection_seal=self.seal(weight=candidate))
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(torch.equal(self.model.model.layers[4].mlp.down_proj.weight, self.weight))
        self.assertEqual(len(self.model._forward_pre_hooks), 0)

    def test_w0_conditioned_retention_and_identity_rejection(self):
        out = self.observer.observe(self.records, self.weight, selection_seal=self.seal(), greedy=False)
        observed = self.observer.observe(self.records, self.weight, selection_seal=self.seal(),
                                         reuse=out, w0_result=out, greedy=False)
        retention = observed["W0_correct_NS"]
        self.assertEqual(retention["numerator"], retention["denominator"])
        self.assertEqual(retention["lost"], 0)
        corrupted = copy.deepcopy(out["metrics"])
        corrupted["NS"]["rows"][0]["identity"] = "fake"
        with self.assertRaisesRegex(ObserverBoundary, "EXACT_IDENTITY"):
            w0_correct_retention(out["metrics"], corrupted)

    def test_censored_targets_remain_denominator_and_bad_cardinality_fails(self):
        records = [record(target="x"*33)]
        out = self.observer.observe(records, self.weight, selection_seal=self.seal(records))
        self.assertEqual(out["generation_denominator"], 1)
        self.assertEqual(out["generation_censored"], 1)
        self.assertFalse(out["generation"][0]["target_prefix_match"])
        bad = [record()]
        bad[0]["paraphrase_prompts"].pop()
        with self.assertRaisesRegex(ObserverBoundary, "CARDINALITY"):
            self.observer.observe(bad, self.weight, selection_seal=self.seal(bad))

    def test_nonselected_changes_do_not_silently_become_new_baseline(self):
        with torch.no_grad():
            self.model.head.weight.add_(.01)
        with self.assertRaisesRegex(ObserverBoundary, "BASE_NONSELECTED"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal())
        self.assertEqual(self.observer.work["model_forward_calls"], 0)

    def reduced_fixture(self, result, *, w0=False):
        prior = dict(requests=result["requests"], request_order=result["request_order"],
                     metrics=copy.deepcopy(result["metrics"]), source_binding=result["source_binding"],
                     evaluator_layout=result["compatibility"]["evaluator_layout"])
        raw_bytes = json.dumps(prior, sort_keys=True, separators=(",", ":")).encode()
        proof = dict(status="COMPATIBILITY_SEALED", **result["compatibility"],
                     source_raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                     endpoint_role="ENDPOINT_CURRENT",
                     lineage_evidence="CPU fixture, not actual Llama equivalence")
        if w0:
            proof.update(endpoint_role="W0_PRETRAINED",
                         model_pretrained_weight_sha256=proof["endpoint_weight_sha256"])
        proof["proof_sha256"] = digest(proof)
        return dict(source_bytes=raw_bytes, proof=proof)

    @staticmethod
    def reseal_changed_reduced(bundle, change):
        bundle = copy.deepcopy(bundle)
        prior = json.loads(bundle["source_bytes"])
        change(prior)
        bundle["source_bytes"] = json.dumps(prior, sort_keys=True, separators=(",", ":")).encode()
        bundle["proof"]["source_raw_sha256"] = hashlib.sha256(bundle["source_bytes"]).hexdigest()
        bundle["proof"]["proof_sha256"] = digest({k:v for k,v in bundle["proof"].items() if k != "proof_sha256"})
        return bundle

    def test_reduced_pair_reuse_no_raw_synthesis_no_canonical_forwards(self):
        prior = self.observer.observe(self.records, self.weight, selection_seal=self.seal(), greedy=False)
        bundle = self.reduced_fixture(prior)
        evaluator = self.observer.bindings["evaluator"]
        with patch.object(evaluator, "evaluate_pairs", side_effect=AssertionError("covered canonical evaluation repeated")):
            out = self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reduced_metrics_reuse=bundle)
        self.assertEqual(out["metrics"], prior["metrics"])
        self.assertEqual(out["raw"], {})
        self.assertEqual(out["raw_token_payload"], "NOT_RETAINED_PRIOR")
        self.assertEqual(out["work"]["canonical_pair_rows_computed"], 0)
        self.assertEqual(out["work"]["canonical_microbatches_computed"], 0)
        self.assertEqual(out["work"]["reduced_metric_pairs_reused"], 13)
        self.assertEqual(out["work"]["greedy_requests_computed"], 1)
        self.assertEqual(out["work"]["model_forward_calls"], 1)
        self.assertEqual(out["strict"], prior["strict"])
        with self.assertRaisesRegex(ObserverBoundary, "EXPLICIT_PROOF"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reuse=out)

    def test_reduced_reuse_rejects_bad_identity_denominator_endpoint_or_source(self):
        prior = self.observer.observe(self.records, self.weight, selection_seal=self.seal(), greedy=False)
        bundle = self.reduced_fixture(prior)
        wrong = self.reseal_changed_reduced(bundle, lambda p:p["metrics"]["RS"]["rows"][0].update(identity="f"*64))
        with self.assertRaisesRegex(ObserverBoundary, "PAIR_IDENTITY"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reduced_metrics_reuse=wrong)
        wrong = self.reseal_changed_reduced(bundle, lambda p:p["metrics"]["PS"].update(denominator=20))
        with self.assertRaisesRegex(ObserverBoundary, "PAIR_CARDINALITY"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reduced_metrics_reuse=wrong)
        wrong = copy.deepcopy(bundle)
        wrong["proof"]["endpoint_weight_sha256"] = "f"*64
        wrong["proof"]["proof_sha256"] = digest({k:v for k,v in wrong["proof"].items() if k != "proof_sha256"})
        with self.assertRaisesRegex(ObserverBoundary, "ENDPOINT_WEIGHT_SHA256_MISMATCH"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reduced_metrics_reuse=wrong)
        wrong = self.reseal_changed_reduced(bundle, lambda p:p.update(source_binding={}))
        with self.assertRaisesRegex(ObserverBoundary, "SOURCE_SHA"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reduced_metrics_reuse=wrong)
        wrong = copy.deepcopy(bundle)
        wrong["source_bytes"] += b" "
        with self.assertRaisesRegex(ObserverBoundary, "SOURCE_BYTES_SHA"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reduced_metrics_reuse=wrong)

    def test_reduced_reuse_validates_inequality_ties_and_strict_counts(self):
        prior = self.observer.observe(self.records, self.weight, selection_seal=self.seal(), greedy=False)
        bundle = self.reduced_fixture(prior)
        def change(p):
            row = p["metrics"]["RS"]["rows"][0]
            row.update(new_nll=1., true_nll=1., margin=0., success=True)
        wrong = self.reseal_changed_reduced(bundle, change)
        with self.assertRaisesRegex(ObserverBoundary, "INEQUALITY_OR_TIE"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reduced_metrics_reuse=wrong)
        wrong = self.reseal_changed_reduced(bundle, lambda p:p["metrics"]["PS"]["rows"][0].update(new_token_count=99))
        with self.assertRaisesRegex(ObserverBoundary, "TOKEN_COUNT"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal(), reduced_metrics_reuse=wrong)

    def test_precomputed_W0_pairs_require_separate_pretrained_endpoint_proof(self):
        prior = self.observer.observe(self.records, self.weight, selection_seal=self.seal(), greedy=False)
        bundle, w0 = self.reduced_fixture(prior), self.reduced_fixture(prior, w0=True)
        out = self.observer.observe(self.records, self.weight, selection_seal=self.seal(), greedy=False,
                                    reduced_metrics_reuse=bundle, w0_reduced_metrics_reuse=w0)
        self.assertEqual(out["work"]["model_forward_calls"], 0)
        self.assertEqual(out["W0_correct_NS"]["lost"], 0)
        with self.assertRaisesRegex(ObserverBoundary, "PRETRAINED_ENDPOINT_PROOF"):
            self.observer.observe(self.records, self.weight, selection_seal=self.seal(), greedy=False,
                                  reduced_metrics_reuse=bundle, w0_reduced_metrics_reuse=bundle)

    def test_W0_source_population_subset_preserves_original_rows_and_layout(self):
        records = [record(i+1) for i in range(3)]
        whole = self.observer.observe(records, self.weight, selection_seal=self.seal(records), greedy=False)
        subset = [records[1]]
        actual = self.observer.observe(subset, self.weight, selection_seal=self.seal(subset), greedy=False)
        w0 = self.reduced_fixture(whole, w0=True)
        original_bytes = w0["source_bytes"]
        w0["proof"].update(actual["compatibility"])
        w0["proof"].update(endpoint_role="W0_REFERENCE_SUBSET", selected_case_ids=[2],
                            source_population=3, source_request_order_sha256=whole["request_order"])
        w0["proof"]["proof_sha256"] = digest({k:v for k,v in w0["proof"].items() if k != "proof_sha256"})
        out = self.observer.observe(subset, self.weight, selection_seal=self.seal(subset), greedy=False,
                                    reduced_metrics_reuse=self.reduced_fixture(actual), w0_reduced_metrics_reuse=w0)
        self.assertEqual(out["work"]["model_forward_calls"], 0)
        self.assertEqual(out["W0_correct_NS"]["all_requested_N"], 10)
        self.assertEqual(out["W0_reference_subset_scope"]["source_population"], 3)
        self.assertEqual(out["W0_reference_subset_scope"]["selected_population"], 1)
        self.assertEqual(out["W0_reference_subset_scope"]["current_B100_padding_bit_parity"], "NOT_CLAIMED")
        self.assertEqual(w0["source_bytes"], original_bytes)
        bad = copy.deepcopy(w0)
        bad["proof"]["selected_case_ids"] = [1]
        bad["proof"]["proof_sha256"] = digest({k:v for k,v in bad["proof"].items() if k != "proof_sha256"})
        with self.assertRaisesRegex(ObserverBoundary, "SOURCE_ORDER_PROOF"):
            self.observer.observe(subset, self.weight, selection_seal=self.seal(subset), greedy=False,
                                  reduced_metrics_reuse=self.reduced_fixture(actual), w0_reduced_metrics_reuse=bad)


if __name__ == "__main__":
    unittest.main()
