"""Independent raw/ledger CPU corruptions; no actual native GPU claims."""
import copy
from pathlib import Path
import tempfile
import unittest

from official.baselines.registry import call_options, hparams, requests
from official.evaluation.factual import build_zsre_w0_reference, evaluate_counterfact, evaluate_zsre
from official.experiments.prepare import digest, file_sha, write_new
from official.runners.server1.audit import AuditError, audit_commits, audit_factual
from official.runners.server1.common import member
from official.tests.test_factual import CharacterTokenizer, TransitionLM, cf, zr


class FactualAuditTests(unittest.TestCase):
    def setUp(self):
        self.model, self.tok = TransitionLM(), CharacterTokenizer()
        self.external = dict(model="fixture", revision="pinned", tokenizer="fixture-only")
        self.rows = [cf(1), cf(2)]
        self.endpoint = evaluate_counterfact(self.model, self.tok, self.rows, identity=self.external)

    def audit(self, endpoint=None):
        return audit_factual(endpoint or self.endpoint, self.rows, "cf", self.tok, self.external)

    def test_actual_CPU_forward_raw_counts_and_work_strip(self):
        before = len(self.model.calls)
        out = self.audit()
        self.assertEqual(out["requests"], 2)
        self.assertEqual(out["candidate_sequences"], 16)
        self.assertEqual(out["prompt_pairs"], 8)
        self.assertEqual(out["groups"]["neighborhood"]["token_count"], 12)
        self.assertEqual(out["actual_model_forward_calls"], 0)
        self.assertEqual(len(self.model.calls), before)
        scientific = {key: value for key, value in self.endpoint.items() if key != "work"}
        self.assertEqual(self.audit(scientific), out)

    def test_wrong_order_or_occurrence_cannot_pass_matching_count(self):
        endpoint = copy.deepcopy(self.endpoint)
        endpoint["cases"].reverse()
        with self.assertRaisesRegex(AuditError, "ORDERED_COHORT"):
            self.audit(endpoint)
        endpoint["cases"].reverse()
        endpoint["cases"][1]["occurrence_index"] = 1
        with self.assertRaisesRegex(AuditError, "ORDERED_COHORT"):
            self.audit(endpoint)

    def test_partial_prompt_arrays_rejected_even_if_macro_is_unchanged(self):
        endpoint = copy.deepcopy(self.endpoint)
        for name in ("neighborhood_observations", "neighborhood_prompts_probs", "neighborhood_prompts_correct"):
            endpoint["cases"][0][name].pop()
        with self.assertRaisesRegex(AuditError, "PROMPT_DENOMINATOR"):
            self.audit(endpoint)

    def test_gold_token_and_query_boundary_identity_rejected(self):
        endpoint = copy.deepcopy(self.endpoint)
        endpoint["cases"][0]["rewrite_observations"][0]["target_new"]["target_token_ids"][0] += 1
        with self.assertRaisesRegex(AuditError, "QUERY_TOKEN_IDENTITY"):
            self.audit(endpoint)

    def test_prediction_denominator_numerator_N_desired_guard(self):
        for field, value, code in (("token_count", 500, "TOKEN_DENOMINATOR"),
                                  ("token_correct_count", 0, "TOKEN_DENOMINATOR"),
                                  ("token_correct", [False] * 3, "TOKEN_NUMERATOR")):
            with self.subTest(field=field):
                endpoint = copy.deepcopy(self.endpoint)
                endpoint["cases"][0]["rewrite_observations"][0]["target_new"][field] = value
                with self.assertRaisesRegex(AuditError, code):
                    self.audit(endpoint)
        endpoint = copy.deepcopy(self.endpoint)
        endpoint["cases"][0]["neighborhood_observations"][0]["desired_target"] = "new"
        with self.assertRaisesRegex(AuditError, "DESIRED_TARGET"):
            self.audit(endpoint)

    def test_F32_mean_per_token_finite_and_reducer_checks(self):
        endpoint = copy.deepcopy(self.endpoint)
        endpoint["cases"][0]["rewrite_observations"][0]["target_new"]["mean_nll"] += 1
        with self.assertRaisesRegex(AuditError, "FP32_NLL_MEAN"):
            self.audit(endpoint)
        endpoint = copy.deepcopy(self.endpoint)
        endpoint["cases"][0]["rewrite_observations"][0]["target_new"]["nll_by_token"][0] = float("nan")
        with self.assertRaisesRegex(AuditError, "NLL_TOKEN_FINITE"):
            self.audit(endpoint)
        endpoint = copy.deepcopy(self.endpoint)
        endpoint["summary"]["Score"] += 1
        with self.assertRaisesRegex(AuditError, "REQUEST_MACRO_REDUCTION"):
            self.audit(endpoint)

    def test_source_endpoint_identity_accuracy_denominator_work_guards(self):
        endpoint = copy.deepcopy(self.endpoint)
        endpoint["identity_sha256"] = "0" * 64
        with self.assertRaisesRegex(AuditError, "ENDPOINT_IDENTITY"):
            self.audit(endpoint)
        endpoint = copy.deepcopy(self.endpoint)
        endpoint["accuracy"]["rewrite"]["token_count"] += 1
        with self.assertRaisesRegex(AuditError, "ACCURACY_REDUCTION"):
            self.audit(endpoint)
        endpoint = copy.deepcopy(self.endpoint)
        endpoint["work"]["target_tokens"] += 1
        with self.assertRaisesRegex(AuditError, "LOGICAL_WORK_COUNTS"):
            self.audit(endpoint)

    def test_zsRE_W0_self_reference_and_edited_agreement_audit(self):
        rows = [zr(1), zr(2)]
        reference = build_zsre_w0_reference(self.model, self.tok, rows, identity=self.external)
        value = audit_factual(reference["evaluation"], rows, "zsre", self.tok, self.external, reference)
        self.assertEqual(value["candidate_sequences"], 6)
        endpoint = evaluate_zsre(self.model, self.tok, rows, identity=self.external, w0_reference=reference)
        self.assertEqual(audit_factual(endpoint, rows, "zsre", self.tok, self.external, reference)["requests"], 2)
        endpoint["cases"][0]["neighborhood_W0_agreement"][0] = False
        with self.assertRaisesRegex(AuditError, "W0_PREDICTION_AGREEMENT"):
            audit_factual(endpoint, rows, "zsre", self.tok, self.external, reference)

    def test_zsRE_missing_reference_typed_omission(self):
        rows = [zr()]
        endpoint = evaluate_zsre(self.model, self.tok, rows, identity=self.external)
        audit_factual(endpoint, rows, "zsre", self.tok, self.external)
        endpoint["summary"]["Specificity"] = 0
        with self.assertRaisesRegex(AuditError, "REQUEST_MACRO_REDUCTION"):
            audit_factual(endpoint, rows, "zsre", self.tok, self.external)


class CommitAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.method = "MEMIT"
        self.identity = {name: "frozen-" + name for name in
                         ("config_sha256", "stream_sha256", "code_commit", "official_tree_sha256",
                          "model_revision", "tokenizer_sha256", "assets_sha256")}
        self.records = [cf(index + 1) for index in range(300)]
        self.ledger = []
        hp = hparams(self.method, "llama3")
        names = [hp.rewrite_module_tmp.format(layer) + ".weight" for layer in hp.layers]
        previous = dict.fromkeys(names, "1" * 64)
        native_path = Path(__file__).resolve().parents[2] / "baselines/sphere/memit/memit_main.py"
        for batch in range(1, 4):
            post = dict.fromkeys(names, str(batch + 1) * 64)
            edit = dict(schema="official-server1-native-apply-v1", method=self.method, batch=batch,
                requests=100, request_sha256=digest(requests(self.records[(batch - 1) * 100:batch * 100],
                                                           self.method, "llama3")),
                entry_selected_sha256=previous, post_selected_sha256=post,
                native_module="official.baselines.sphere.memit.memit_main",
                call_options=call_options(self.method, "llama3"), native_source_sha256=file_sha(native_path),
                same_model=True, nonselected_identity_version_unchanged=True, selected_FP32_finite=True,
                native_history=False, cache_c={}, contexts_sha256="f" * 64, quality_gate=False)
            edit["identity_sha256"] = digest(edit)
            payload = self.root / "checkpoint" / f"temporary-{batch}.pt"
            payload.parent.mkdir(exist_ok=True)
            payload.write_bytes(f"fixture-own-batch-{batch}".encode())
            sha = file_sha(payload)
            pointer = dict(batch=batch, file=f"batch-{batch:02d}-{sha[:16]}.pt", sha256=sha,
                           final_W20=False, identity_sha256=digest(self.identity))
            if batch == 3:
                payload.rename(payload.parent / pointer["file"])
            else:
                payload.unlink()
            receipt = dict(batch=batch, identity=self.identity, actual_applied_requests=100 * batch,
                           cursor=dict(completed_batch=batch, edit=edit,
                                       evaluation="NOT_SCHEDULED_THIS_BATCH"), checkpoint=pointer)
            self.ledger.append(receipt)
            write_new(self.root / "commits" / f"batch-{batch:02d}.json", receipt)
            previous = post
        write_new(self.root / "checkpoint" / "latest.json", pointer)

    def tearDown(self):
        self.temp.cleanup()

    def audit(self):
        return audit_commits(self.root, self.method, self.identity, expected20=3, records=self.records)

    def replace(self, index):
        path = self.root / "commits" / f"batch-{index + 1:02d}.json"
        path.write_text(__import__("json").dumps(self.ledger[index]))

    def test_exact_native_count_payload_hash_and_weight_chain(self):
        value = self.audit()
        self.assertEqual(value["actual_native_apply_receipts"], 3)
        self.assertEqual(value["actual_applied_requests"], 300)
        self.assertTrue(value["selected_weight_hash_chain"])
        self.assertTrue(value["native_request_digest_checked_against_stream"])
        self.assertFalse(value["GPU_resume_parity_claim"])

    def test_rehashed_broken_weight_link_rejected(self):
        edit = self.ledger[1]["cursor"]["edit"]
        edit["entry_selected_sha256"] = dict.fromkeys(edit["entry_selected_sha256"], "9" * 64)
        edit["identity_sha256"] = digest({key: value for key, value in edit.items() if key != "identity_sha256"})
        self.replace(1)
        with self.assertRaisesRegex(AuditError, "ENTRY_POST_WEIGHT_LINK"):
            self.audit()

    def test_request_stream_digest_actual_source_reject(self):
        changed = copy.deepcopy(self.records)
        changed[0]["requested_rewrite"]["target_new"]["str"] = "other"
        with self.assertRaisesRegex(AuditError, "REQUEST_STREAM_DIGEST"):
            audit_commits(self.root, self.method, self.identity, expected20=3, records=changed)

    def test_no_filecount_only_or_untrusted_native_identity(self):
        self.ledger[1]["actual_applied_requests"] = 100
        self.replace(1)
        with self.assertRaisesRegex(AuditError, "BATCH_IDENTITY_REQUESTS"):
            self.audit()
        self.ledger[1]["actual_applied_requests"] = 200
        self.ledger[1]["cursor"]["edit"]["native_source_sha256"] = "0" * 64
        self.replace(1)
        with self.assertRaisesRegex(AuditError, "APPLY_SOURCE_REQUESTS_IDENTITY"):
            self.audit()

    def test_missing_batch_or_corrupt_current_checkpoint(self):
        path = self.root / "commits" / "batch-03.json"
        path.unlink()
        with self.assertRaisesRegex(AuditError, "EXACT_COMMIT_LEDGER_NAMES"):
            self.audit()
        write_new(path, self.ledger[2])
        payload = self.root / "checkpoint" / self.ledger[2]["checkpoint"]["file"]
        payload.write_bytes(b"corrupt")
        with self.assertRaisesRegex(AuditError, "PAYLOAD_SHA"):
            self.audit()


if __name__ == "__main__":
    unittest.main()
