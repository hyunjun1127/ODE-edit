"""CPU fixtures only: existing B3 raw is unchanged and never re-evaluated.

Mock metadata/reference observations never execute a model or original oracle.
No fixture result is actual pretrained/CUDA matched-subset qualification.
"""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from official.evaluation.reduce import counterfact
from official.evaluation.factual import _assemble_cf, retained_query_work
from official.experiments.prepare import digest
from official.runners.server1 import native_parity as parity


@contextmanager
def tf32_disabled_fixture():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def external():
    return dict(model="llama3", model_revision="d" * 40, tokenizer_sha256="e" * 64,
        assets_sha256="f" * 64, stream_sha256="1" * 64, code_commit="a" * 40,
        official_tree="b" * 40, runtime=dict(torch="fixture", transformers="fixture", numpy="fixture"),
        precision="FP32_EAGER_TF32_OFF_NO_AUTOCAST", raw_local_only=True)


class MetadataModel:
    """Not a torch model and not callable: inference cannot occur."""
    training = False
    config = SimpleNamespace(use_cache=False, _attn_implementation="eager")
    def parameters(self):
        return [SimpleNamespace(dtype=torch.float32, device=SimpleNamespace(type="cuda"),
                                is_floating_point=lambda: True)]
    def buffers(self):
        return []
    def modules(self):
        return [self]


class MetadataEngine:
    method, source_verified = "FT", True
    def __init__(self):
        self.asset_manifest = dict(model=dict(identity=dict(revision=external()["model_revision"]),
                                             tokenizer_sha256=external()["tokenizer_sha256"]))
        self.state = dict(method="FT", successful_calls=3, cache_c={},
            selected_weights={"model.layers.21.mlp.down_proj.weight":
                dict(sha256="9" * 64, shape=[4096, 14336], dtype="torch.float32")},
            contexts_sha256="8" * 64)
        self.state["identity_sha256"] = digest(self.state)
    def state_identity(self):
        return deepcopy(self.state)


def fixture_record(index):
    """Synthetic locked inputs, not real CounterFact or GPU evidence."""
    return dict(case_id=901 + index, occurrence_index=1 + index,
        requested_rewrite=dict(prompt="{} is", subject="Ada", target_new={"str":"new"},
                               target_true={"str":"true"}),
        paraphrase_prompts=["Ada was"], neighborhood_prompts=["Bob is"])


def signatures_fixture(records):
    signatures = []
    for index, record in enumerate(records):
        queries = []
        rw = record["requested_rewrite"]
        groups = dict(rewrite=[rw["prompt"].format(rw["subject"])],
                      paraphrase=record["paraphrase_prompts"], neighborhood=record["neighborhood_prompts"])
        for slot, (kind, prompts) in enumerate(groups.items()):
            for prompt_index, prompt in enumerate(prompts):
                prefix = [1, 10 + slot] + [12] * (index % 2)
                for target_kind, target, gold in (("new", rw["target_new"]["str"], [40, 41]),
                                                ("true", rw["target_true"]["str"], [42, 43])):
                    queries.append(dict(kind=kind, prompt_index=prompt_index, prompt=prompt,
                        target_kind=target_kind, target=target, input_token_ids=prefix + gold,
                        target_start=len(prefix), target_token_ids=gold))
        signatures.append(dict(case_id=record["case_id"], occurrence_index=record["occurrence_index"], queries=queries))
    return signatures


def canonical_fixture(records, identity):
    signatures = signatures_fixture(records)
    cases = [dict(case_id=row["case_id"], occurrence_index=row["occurrence_index"]) for row in records]
    results = []
    for index, signature in enumerate(signatures):
        for query in signature["queries"]:
            gold = query["target_token_ids"]
            predicted = list(gold) if query["target_kind"] == "true" else [99] * len(gold)
            correct = [a == b for a,b in zip(predicted,gold)]
            nll = 1.0 if query["target_kind"] == "true" else 2.0
            results.append(dict(query, case_index=index, occurrence_index=signature["occurrence_index"],
                predicted_token_ids=predicted, token_correct=correct, token_count=len(gold),
                token_correct_count=sum(correct), strict_correct=all(correct),
                nll_by_token=[nll] * len(gold), mean_nll=nll))
    summary, accuracy = _assemble_cf(cases, results)
    ci = dict(schema="official-factual-causal-v1", dataset="cf",
        tokenization="NATIVE_CF_VERIFIED_BOUNDARY_ZSRE_EXACT_TOKEN_PREFIX_NO_TARGET_BOS",
        ordered_occurrences=[row["occurrence_index"] for row in records],
        cohort_sha256=digest(signatures), external_identity=deepcopy(identity),
        padding="RIGHT_EXPLICIT_ATTENTION_MASK", use_cache=False, W0_reference_sha256=None)
    work = retained_query_work(signatures, "cf", batch_size=16)
    work["seconds"] = .001
    return dict(identity=ci, identity_sha256=digest(ci), cases=cases, summary=summary, accuracy=accuracy,
        model_no_mutation=True, raw_local_only=True, RNG_restored=True,
        work=work)


def reference_fixture(records, canonical, identity, binding):
    signatures = signatures_fixture(records)
    locked = [dict(case_id=row["case_id"], occurrence_index=row["occurrence_index"]) for row in records]
    ni = dict(schema="official-cf-original-native-reference-v1", external_identity=deepcopy(identity),
        reference_binding=deepcopy(binding), source_sha256=parity.REFERENCE_SOURCE_SHA256,
        source_commit=parity.REFERENCE_UPSTREAM_COMMIT, source_bytes=parity.REFERENCE_BYTES,
        original_function="test_batch_prediction", evidence_scope=parity.SCOPE,
        ordered_occurrences=[row["occurrence_index"] for row in records],
        cohort_sha256=canonical["identity"]["cohort_sha256"], locked_cohort_sha256=digest(locked),
        native_device="cuda", model_dtype="FP32", padding="NATIVE_RIGHT", use_cache=False)
    work = retained_query_work(signatures, "cf", per_case=True)
    work["seconds"] = .002
    native = dict(schema="official-cf-original-native-reference-v1", status="OBSERVED_UNCOMPARED",
        identity=ni, identity_sha256=digest(ni), cases=deepcopy(canonical["cases"]),
        summary=deepcopy(canonical["summary"]), case_signatures=signatures, work=work,
        evidence={"TEST_ONLY_CPU_FIXTURE": "NOT_OBSERVED", "ACTUAL_GPU_SMOKE": "NOT_OBSERVED",
                  parity.SCOPE: "NOT_OBSERVED", "ACTUAL_FULL_2K": "NOT_OBSERVED"},
        raw_local_only=True, model_no_mutation=True, RNG_restored=True, checkpoint_saved=False,
        canonical_payload_sha256=digest(canonical))
    return dict(schema="official-cf-original-native-reference-v1", status="PASS", evidence_scope=parity.SCOPE,
        evidence={"TEST_ONLY_CPU_FIXTURE": "NOT_OBSERVED", "ACTUAL_GPU_SMOKE": "NOT_OBSERVED",
                  parity.SCOPE: "PASS", "ACTUAL_FULL_2K": "NOT_OBSERVED"},
        mismatches=[], display_mismatches=[], raw_local_only=True, checkpoint_saved=False,
        scientific_performance_promotion=False, canonical_identity_sha256=canonical["identity_sha256"],
        tolerances=dict(nll_abs_nats=1e-4, nll_relative_to_native=1e-5,
                        strict_booleans="EXACT", aggregate_abs=1e-10), native=native, work=work)


def resign_fixture(value):
    """Adversarial hash repair must not repair inconsistent raw measurements."""
    canonical, comparison = value["canonical"], value["comparison"]
    native = comparison["native"]
    canonical["identity_sha256"] = digest(canonical["identity"])
    value["canonical_payload_sha256"] = digest(canonical)
    native["canonical_payload_sha256"] = value["canonical_payload_sha256"]
    native["identity_sha256"] = digest(native["identity"])
    comparison["canonical_identity_sha256"] = canonical["identity_sha256"]
    comparison["work"] = deepcopy(native["work"])


class NativeParityConnector(unittest.TestCase):
    def setUp(self):
        self.engine, self.model = MetadataEngine(), MetadataModel()
        self.tokenizer = SimpleNamespace(padding_side="right", pad_token_id=0)
        self.records = [fixture_record(index) for index in range(2000)]
        self.canonical = canonical_fixture(self.records[:300], external())
        self.reference_calls = []
        self.module = SimpleNamespace(compare_native_counterfact=self.reference)

    def reference(self, model, tok, records, canonical, **kwargs):
        self.reference_calls.append((model, tok, records, canonical, kwargs))
        binding = {key: kwargs[key] for key in ("model_identity", "tokenizer_identity", "state_identity")}
        self.assertEqual(kwargs["state_callback"](), binding["state_identity"])
        return reference_fixture(records, canonical, kwargs["identity"], binding)

    def qualify(self, *, reference=None, canonical=None, plan=None):
        module = self.module if reference is None else SimpleNamespace(compare_native_counterfact=reference)
        with patch.object(parity, "_published_reference", return_value=module), \
             patch("official.evaluation.factual.evaluate", side_effect=AssertionError("REDUNDANT_CANONICAL_FORWARD")) as forbidden, \
             tf32_disabled_fixture():
            value = parity.qualify(self.model, self.tokenizer, self.records, external(), self.engine,
                deepcopy(parity.PLAN) if plan is None else plan, self.canonical if canonical is None else canonical)
            forbidden.assert_not_called()
            return value

    def test_existing_first300_identity_preserved_separate_bindings_no_extra_canonical(self):
        before = deepcopy(self.canonical)
        before_sha = digest(self.canonical)
        value = self.qualify()
        self.assertEqual(len(self.reference_calls), 1)
        reference = self.reference_calls[0]
        self.assertEqual(reference[2], self.records[:300])
        self.assertIs(reference[3], self.canonical)
        self.assertEqual(reference[4]["identity"], external())
        self.assertNotIn("state_identity", reference[4]["identity"])
        self.assertEqual(reference[4]["state_identity"], self.engine.state_identity())
        self.assertEqual(reference[4]["model_identity"], self.engine.asset_manifest["model"]["identity"])
        self.assertEqual(reference[4]["tokenizer_identity"], dict(sha256=external()["tokenizer_sha256"]))
        self.assertEqual(reference[4]["evidence_scope"], "ACTUAL_MATCHED_SUBSET")
        self.assertEqual(reference[4]["locked_cohort"],
            [dict(case_id=row["case_id"], occurrence_index=row["occurrence_index"]) for row in self.records[:300]])
        self.assertNotIn("test_only_cpu", reference[4])
        self.assertNotIn("source_path", reference[4])
        self.assertEqual(self.canonical, before)
        self.assertEqual(value["canonical_payload_sha256"], before_sha)
        self.assertEqual(value["canonical_observation"], "EXISTING_B3_FIRST300")
        self.assertEqual(value["additional_canonical_forward_calls"], 0)
        self.assertEqual(value["additional_fit_calls"], 0)
        self.assertEqual(value["generation_calls"], 0)

    def test_low_metric_not_gate_compact_evidence_never_full2k_or_smoke(self):
        value = self.qualify()
        self.assertEqual(value["canonical"]["summary"]["Score"], 0)
        compact = parity.validate_report(value, external(), deepcopy(parity.PLAN))
        self.assertEqual(compact["status"], "PASS")
        self.assertEqual(compact["evidence_scope"], "ACTUAL_MATCHED_SUBSET")
        self.assertEqual(compact["full_2k_parity"], "NOT_OBSERVED")
        self.assertEqual(compact["requests"], 300)
        self.assertEqual(compact["original_native_forward_calls"], 300)
        self.assertEqual(compact["existing_canonical_forward_calls"], 113)
        self.assertEqual(compact["additional_canonical_forward_calls"], 0)
        self.assertFalse(compact["performance_gate"])
        self.assertFalse({"cases", "case_id", "tokens", "identity", "canonical", "comparison"} & compact.keys())
        self.assertTrue(compact["case_ids_tokens_prompts_omitted"])

    def test_plan_tolerances_budget_scope_change_rejected_before_native(self):
        for changes in (dict(nll_abs_nats=.1), dict(evidence_scope="ACTUAL_FULL_2K"),
                        dict(evidence_scope="ACTUAL_GPU_SMOKE"), dict(requests=4),
                        dict(additional_fit_calls=False), dict(additional_canonical_forward_calls=1)):
            with self.subTest(changes=changes), self.assertRaisesRegex(parity.NativeParityError, "PLAN_CHANGED"):
                self.qualify(plan=dict(parity.PLAN, **changes))
        self.assertEqual(self.reference_calls, [])

    def test_wrong_first_cohort_or_B3_state_rejected_before_native(self):
        self.records[0]["occurrence_index"] = 5
        with self.assertRaisesRegex(parity.NativeParityError, "LOCKED_FIRST300"):
            self.qualify()
        self.records[0]["occurrence_index"] = 1
        self.engine.state["successful_calls"] = 2
        with self.assertRaisesRegex(parity.NativeParityError, "COMPLETED_B3"):
            self.qualify()
        self.assertEqual(self.reference_calls, [])

    def test_existing_identity_cohort_hash_incomplete_work_rejected_before_native(self):
        modifications = (
            lambda row: row["identity"]["external_identity"].update(state_identity=self.engine.state_identity()),
            lambda row: row["identity"].update(ordered_occurrences=[1, 2, 3, 4]),
            lambda row: row.update(identity_sha256="0" * 64),
            lambda row: row["cases"][0].update(case_id=10),
            lambda row: row.update(RNG_restored=False),
            lambda row: row.update(work={}),
            lambda row: row["cases"][0]["rewrite_prompts_probs"][0].update(target_new=float("nan")),
        )
        for change in modifications:
            bad = deepcopy(self.canonical)
            change(bad)
            with self.subTest(change=change), self.assertRaises(parity.NativeParityError):
                self.qualify(canonical=bad)
        self.assertEqual(self.reference_calls, [])

    def test_actual_profile_FP32_CUDA_eval_cacheFalse_source_proof(self):
        for target, key, value, code in ((self.engine, "source_verified", False, "FROZEN_SOURCE"),
                                        (self.model, "training", True, "EVAL_REQUIRED"),
                                        (self.model.config, "use_cache", True, "NO_CACHE"),
                                        (self.tokenizer, "padding_side", "left", "RIGHT_PADDING")):
            original = getattr(target, key)
            setattr(target, key, value)
            try:
                with self.subTest(key=key), self.assertRaisesRegex(parity.NativeParityError, code):
                    self.qualify()
            finally:
                setattr(target, key, original)
        cpu = SimpleNamespace(dtype=torch.float32, device=SimpleNamespace(type="cpu"), is_floating_point=lambda: True)
        with patch.object(self.model, "parameters", return_value=[cpu]):
            with self.assertRaisesRegex(parity.NativeParityError, "FP32_CUDA_REQUIRED"):
                self.qualify()

    def test_CPU_smoke_full_or_mismatch_cannot_satisfy_actual_matched(self):
        value = self.qualify()
        for changes in (dict(status="MISMATCH"), dict(status="CPU_FIXTURE_PASS"),
                        dict(status="NOT_QUALIFIED"), dict(evidence_scope="ACTUAL_FULL_2K"),
                        dict(evidence_scope="ACTUAL_GPU_SMOKE")):
            bad = deepcopy(value)
            bad["comparison"].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(parity.NativeParityError, "ACTUAL_MATCHED_PASS_REQUIRED"):
                parity.validate_report(bad, external(), deepcopy(parity.PLAN))

    def test_receipt_tolerance_source_binding_cost_raw_drift_rejected(self):
        value = self.qualify()
        changes = (
            lambda row: row["comparison"]["tolerances"].update(nll_abs_nats=.1),
            lambda row: row.update(RNG_restored=False),
            lambda row: row["comparison"]["native"].update(model_no_mutation=False),
            lambda row: row["comparison"]["native"]["identity"].update(source_sha256="0" * 64),
            lambda row: row["comparison"]["native"]["work"].update(forward_calls=4),
            lambda row: row["comparison"]["native"]["cases"][0]["rewrite_prompts_probs"][0].update(target_new=100),
            lambda row: row["comparison"]["native"]["cases"][0]["rewrite_prompts_correct"].__setitem__(0, True),
            lambda row: row["reference_binding"]["state_identity"].update(successful_calls=4),
            lambda row: row["canonical"]["work"].update(seconds=3),
            lambda row: row.update(additional_canonical_forward_calls=1),
            lambda row: row["comparison"]["native"].update(canonical_payload_sha256="0" * 64),
        )
        for change in changes:
            bad = deepcopy(value)
            change(bad)
            with self.subTest(change=change), self.assertRaises(ValueError):
                parity.validate_report(bad, external(), deepcopy(parity.PLAN))

    def test_resigned_removed_canonical_observations_never_probability_only_PASS(self):
        value = self.qualify()
        for case in value["canonical"]["cases"]:
            for kind in parity.GROUPS:
                case.pop(kind + "_observations")
        resign_fixture(value)
        # A retained-query requirement must reject before any receipt PASS.
        with self.assertRaises((ValueError, KeyError)):
            parity.validate_report(value, external(), deepcopy(parity.PLAN))

    def test_resigned_same_wrong_work_in_both_endpoints_cannot_replace_token_plan(self):
        original = self.qualify()
        for wrong in (0, 123456):
            value = deepcopy(original)
            for endpoint in (value["canonical"], value["comparison"]["native"]):
                for field in ("physical_input_tokens", "target_tokens", "padded_input_tokens"):
                    endpoint["work"][field] = wrong
            resign_fixture(value)
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                parity.validate_report(value, external(), deepcopy(parity.PLAN))

    def test_resigned_token_prediction_NLL_mean_probability_and_strict_mismatch_rejected(self):
        original = self.qualify()
        for field in ("token", "prediction", "nll", "mean", "probability", "strict", "nonbool"):
            value = deepcopy(original)
            case = value["canonical"]["cases"][0]
            raw = case["rewrite_observations"][0]["target_new"]
            if field == "token":
                raw["input_token_ids"][0] = 7
            elif field == "prediction":
                raw["predicted_token_ids"][0] = raw["target_token_ids"][0]
            elif field == "nll":
                raw["nll_by_token"][0] = 3.0
            elif field == "mean":
                raw["mean_nll"] = 3.0
            elif field == "probability":
                case["rewrite_prompts_probs"][0]["target_new"] = 3.0
            elif field == "strict":
                case["rewrite_prompts_correct"][0] = True
            else:
                raw["token_correct"][0] = 0
            resign_fixture(value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                parity.validate_report(value, external(), deepcopy(parity.PLAN))

    def test_resigned_missing_work_zero_forward_or_elapsed_rejected(self):
        original = self.qualify()
        for field in ("missing", "forward", "elapsed", "padded"):
            value = deepcopy(original)
            if field == "missing":
                value["canonical"].pop("work")
            elif field == "forward":
                value["canonical"]["work"]["forward_calls"] = 0
            elif field == "elapsed":
                value["canonical"]["work"]["seconds"] = 0
            else:
                value["canonical"]["work"]["padded_input_tokens"] += 1
            resign_fixture(value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                parity.validate_report(value, external(), deepcopy(parity.PLAN))

    def test_production_payload_TEST_ONLY_label_is_rejected_not_fixture_promoted(self):
        original = self.qualify()
        for field in ("report", "canonical", "native_evidence"):
            value = deepcopy(original)
            if field == "report":
                value["TEST_ONLY_NO_REAL_NATIVE_CALLS"] = True
            elif field == "canonical":
                value["canonical"]["TEST_ONLY_CPU_FIXTURE"] = "PASS"
            else:
                value["comparison"]["native"]["evidence"] = {"TEST_ONLY_CPU_FIXTURE": "PASS"}
            resign_fixture(value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "TEST_ONLY_EVIDENCE"):
                parity.validate_report(value, external(), deepcopy(parity.PLAN))

    def test_original_native_uncompared_evidence_mapping_is_exact_not_promoted(self):
        value = self.qualify()
        expected = {"TEST_ONLY_CPU_FIXTURE": "NOT_OBSERVED", "ACTUAL_GPU_SMOKE": "NOT_OBSERVED",
                    parity.SCOPE: "NOT_OBSERVED", "ACTUAL_FULL_2K": "NOT_OBSERVED"}
        self.assertEqual(value["comparison"]["native"]["evidence"], expected)
        for scope in ("TEST_ONLY_CPU_FIXTURE", parity.SCOPE):
            changed = deepcopy(value)
            changed["comparison"]["native"]["evidence"][scope] = "PASS"
            resign_fixture(changed)
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                parity.validate_report(changed, external(), deepcopy(parity.PLAN))

    def test_resigned_native_selected_weight_descriptor_missing_or_malformed_rejected(self):
        original = self.qualify()
        for field in ("missing", "shape", "dtype", "label"):
            value = deepcopy(original)
            state = value["state_before"]
            if field == "missing":
                state["selected_weights"] = {}
            else:
                weight = state["selected_weights"]["model.layers.21.mlp.down_proj.weight"]
                if field == "shape":
                    weight["shape"] = [14336, 4096]
                elif field == "dtype":
                    weight["dtype"] = "torch.float16"
                else:
                    weight["sha256"] = "PASS"
            state["identity_sha256"] = digest({key:item for key,item in state.items() if key != "identity_sha256"})
            value["state_after"] = deepcopy(state)
            value["reference_binding"]["state_identity"] = deepcopy(state)
            value["comparison"]["native"]["identity"]["reference_binding"] = deepcopy(value["reference_binding"])
            resign_fixture(value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                parity.validate_report(value, external(), deepcopy(parity.PLAN))

    def test_native_strict_flags_are_bool_not_equal_integer_or_tuple(self):
        value = self.qualify()
        for kind in ("rewrite", "paraphrase", "neighborhood"):
            for conversion in (lambda xs:[int(x) for x in xs], tuple):
                bad = deepcopy(value)
                row = bad["comparison"]["native"]["cases"][0]
                row[kind + "_prompts_correct"] = conversion(row[kind + "_prompts_correct"])
                with self.subTest(kind=kind, conversion=conversion), self.assertRaisesRegex(
                        parity.NativeParityError, "STRICT_RAW_BOOLEAN"):
                    parity.validate_report(bad, external(), deepcopy(parity.PLAN))

    def test_reference_canonical_mutation_rejected_not_relabelled(self):
        def mutate(*args, **kwargs):
            result = self.reference(*args, **kwargs)
            args[3]["summary"]["Score"] = 10
            return result
        with self.assertRaisesRegex(parity.NativeParityError, "REPORT_OBSERVATION_GUARDS"):
            self.qualify(reference=mutate)

    def test_failed_actual_gate_retains_expensive_local_report_without_returning_PASS(self):
        before = digest(self.canonical)
        def mismatch(*args, **kwargs):
            result = self.reference(*args, **kwargs)
            result["status"] = "MISMATCH"
            return result
        with self.assertRaisesRegex(parity.NativeParityError, "ACTUAL_MATCHED_PASS_REQUIRED") as caught:
            self.qualify(reference=mismatch)
        self.assertTrue(caught.exception.local_report["raw_local_only"])
        self.assertEqual(caught.exception.local_report["comparison"]["status"], "MISMATCH")
        self.assertEqual(caught.exception.local_report["comparison"]["native"]["work"]["forward_calls"], 300)
        self.assertEqual(len(self.reference_calls), 1)
        self.assertEqual(digest(self.canonical), before)
        self.assertEqual(caught.exception.local_report["additional_canonical_forward_calls"], 0)

    def test_original_error_not_masked_by_later_state_guard(self):
        def fail(*args, **kwargs):
            self.engine.state["successful_calls"] = 4
            raise RuntimeError("ORIGINAL_REFERENCE_FAILURE")
        with self.assertRaisesRegex(RuntimeError, "ORIGINAL_REFERENCE_FAILURE"):
            self.qualify(reference=fail)

    def test_published_module_missing_or_draft_path_typed_block(self):
        with patch.object(parity.importlib, "import_module", side_effect=ModuleNotFoundError("missing")):
            with self.assertRaisesRegex(parity.NativeParityError, "PUBLISHED_NATIVE_CF_REFERENCE_UNAVAILABLE"):
                parity._published_reference()
        with patch.object(parity.importlib, "import_module", return_value=SimpleNamespace(__file__="/draft/not-executed.py")):
            with self.assertRaisesRegex(parity.NativeParityError, "OUTSIDE_FROZEN_OFFICIAL"):
                parity._published_reference()

    def test_published_API_matched_scope_tolerances_cannot_change(self):
        # Fictional module metadata only, not an executed oracle or draft copy.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "official"
            path = root / "evaluation" / "cf_native_reference.py"
            path.parent.mkdir(parents=True)
            path.write_text("# fictional CPU source-location fixture, not executable oracle\n")
            module = SimpleNamespace(__file__=str(path), SOURCE_SHA256=parity.REFERENCE_SOURCE_SHA256,
                SOURCE_BYTES=parity.REFERENCE_BYTES, UPSTREAM_COMMIT=parity.REFERENCE_UPSTREAM_COMMIT,
                MATCHED_SCOPE=parity.SCOPE, NLL_ABS_TOL_NATS=1e-4, NLL_REL_TOL=1e-5,
                AGGREGATE_ABS_TOL=1e-10, compare_native_counterfact=lambda: None)
            wrapper = root / "runners" / "server1" / "native_parity.py"
            with patch.object(parity, "__file__", str(wrapper)), \
                 patch.object(parity.importlib, "import_module", return_value=module):
                self.assertIs(parity._published_reference(), module)
                module.NLL_ABS_TOL_NATS = .1
                with self.assertRaisesRegex(parity.NativeParityError, "TOLERANCE_CHANGED"):
                    parity._published_reference()


if __name__ == "__main__":
    unittest.main()
