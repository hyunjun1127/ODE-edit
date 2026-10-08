"""CPU collector regressions; fixture receipts are NOT GPU/Slurm evidence.

Model/tokenizer/source admission is stubbed only in narrow collector fixtures.
Their factual raw comes from the real official evaluator on a deterministic CPU
model. The collector itself must never call that model or a tracking SDK.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from official.baselines.registry import call_options, hparams, requests
from official.evaluation.factual import build_zsre_w0_reference, evaluate_zsre
from official.evaluation.generation.native_profile import PROFILE
from official.experiments.prepare import digest, file_sha, write_new
from official.runners.server1 import common, run
from official.tests.test_factual import CharacterTokenizer, TransitionLM, zr


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = dict(main_commit="1" * 40, official_tree="2" * 40)
        self.assets = dict(assets_sha256="3" * 64,
            model=dict(identity=dict(revision="fixture-revision"), tokenizer_path=str(self.root / "tokenizer"),
                       tokenizer_sha256="4" * 64),
            streams={name: dict(stream_sha256="5" * 64) for name in ("cf", "zsre")})
        self.external = dict(model="CPU-fixture-only", revision="not-a-GPU-qualification")
        self.records = [zr(i + 1) for i in range(2000)]
        self.tok = CharacterTokenizer()
        self.tok.eos_token = "fixture-eos"
        self.model = TransitionLM()
        self.scope = patch.object(common, "LOCAL_ROOT", self.root)
        self.scope.start()
        self.addCleanup(self.scope.stop)

    def identity(self, config):
        return dict(config_sha256=config["config_sha256"], stream_sha256="5" * 64,
            code_commit=self.source["main_commit"], official_tree_sha256=self.source["official_tree"],
            model_revision="fixture-revision", tokenizer_sha256="4" * 64, assets_sha256="3" * 64)

    def profiles(self, modes, *, dataset="zsre", purpose="pipeline"):
        self.jobs = []
        configs = []
        for index, mode in enumerate([*modes, "collect"]):
            value = dict(method="MEMIT", dataset=dataset, qualification_plan={"CPU_fixture_only": True})
            value["config_sha256"] = digest(value)
            config_path = self.root / f"config-{index}.json"
            write_new(config_path, value)
            self.jobs.append(dict(key=f"job-{index}", mode=mode,
                method="MEMIT" if mode != "collect" else None, dataset=dataset,
                output=str(self.root / f"out-{index}"), config=common.member(config_path)))
            configs.append(self.jobs[-1]["config"])
        plan = dict(jobs=self.jobs, source=self.source, purpose=purpose)
        plan_path = self.root / "plan.json"
        write_new(plan_path, plan)
        self.lock = dict(source=self.source, plan_sha256=digest(plan), profiles=self.jobs,
                         job_configs=configs, plan=common.member(plan_path))
        self.lock_path = self.root / "execution-lock.json"
        write_new(self.lock_path, self.lock)
        self.manifest = dict(schema="official-server1-job-manifest-v1", source=self.source,
            plan_sha256=digest(plan), profiles=self.jobs, execution_lock=common.member(self.lock_path),
            jobs={job["key"]: str(70000 + i) for i, job in enumerate(self.jobs)})
        self.manifest_path = self.root / "job-manifest.json"
        write_new(self.manifest_path, self.manifest)
        self.output = Path(self.jobs[-1]["output"])
        self.output.mkdir()
        self.config = common.read(self.jobs[-1]["config"]["path"])
        return self.jobs

    def invoke(self, *, ready=None):
        def bind(value, lock):
            return self.assets, self.records, self.identity(value), self.external
        with patch.object(run, "validate_config", side_effect=lambda value: value), \
                patch("official.runners.server1.costs.collect_costs",
                      return_value=dict(status="CPU_FIXTURE_NOT_REAL_ACCOUNTING")), \
                patch.object(run, "bindings", side_effect=bind), \
                patch("transformers.AutoTokenizer.from_pretrained", return_value=self.tok), \
                patch.object(run, "read_w0", return_value=ready) if ready is not None else _NullContext():
            run.collect(SimpleNamespace(job_manifest=self.manifest_path), self.config, self.lock, self.output)
        return common.read(self.output / "summary.json")

    def test_expected_chains_from_qualification_resume_and_pipeline_profiles(self):
        for count, purpose in ((0, "qualification"), (1, "resume"), (6, "pipeline")):
            with self.subTest(purpose=purpose), tempfile.TemporaryDirectory() as temporary:
                old_root = self.root
                self.root = Path(temporary)
                with patch.object(common, "LOCAL_ROOT", self.root):
                    modes = ["qualification"] if not count else ["chain"] * count
                    self.profiles(modes, purpose=purpose)
                    result = self.invoke()
                    self.assertEqual(result["expected_chains"], count)
                    self.assertEqual(result["actual_complete_chains"], 0)
                    self.assertEqual(result["coverage"], "PARTIAL_OR_NOT_OBSERVED")
                    self.assertTrue(all(row["status"] == "NOT_OBSERVED_COMPLETE" for row in result["rows"]))
                    self.assertFalse(Path(self.jobs[0]["output"]).exists())
                self.root = old_root

    def test_global_source_lock_and_profiles_are_not_inferred(self):
        self.profiles(["qualification"], purpose="qualification")
        self.manifest["source"]["main_commit"] = "9" * 40
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "SOURCE_EXECUTION_LOCK"):
            self.invoke()
        self.assertFalse((self.output / "summary.json").exists())

    def test_duplicate_or_missing_actual_job_binding_rejected(self):
        self.profiles(["chain", "chain"])
        self.manifest["jobs"]["job-1"] = self.manifest["jobs"]["job-0"]
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "ACTUAL_JOB_PROFILE_BINDING"):
            self.invoke()

    def test_changed_config_member_and_scientific_profile_rejected(self):
        self.profiles(["qualification"], purpose="qualification")
        path = Path(self.jobs[0]["config"]["path"])
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "IMMUTABLE_INPUT_MEMBER_CHANGED"):
            self.invoke()

    def test_fabricated_PASS_with_actual_false_is_validation_failed_not_GPU_PASS(self):
        self.profiles(["qualification"], purpose="qualification")
        folder = Path(self.jobs[0]["output"])
        write_new(folder / "READY.json", dict(schema="official-server1-native-resume-READY-v1",
            method="MEMIT", passed=True, actual_native_B3_and_B2_resume=False,
            actual_fit_calls=6, generation_calls=0))
        result = self.invoke()
        self.assertEqual(result["rows"][0]["status"], "VALIDATION_FAILED")
        self.assertEqual(result["rows"][0]["error_code"], "ACTUAL_NATIVE_QUALIFICATION_READY_REQUIRED")
        self.assertFalse(result["rows"][0]["actual_completion"])
        self.assertFalse(result["GPU_execution_independently_certified"])
        self.assertTrue(result["remote_WB_delivery_not_inferred_from_CPU"])

    def smoke(self):
        self.profiles(["smoke"])
        folder = Path(self.jobs[0]["output"])
        config = common.read(self.jobs[0]["config"]["path"])
        identity = self.identity(config)
        records = self.records[:100]
        reference = build_zsre_w0_reference(self.model, self.tok, records, identity=self.external)
        reference_path = self.root / "zsre-reference-fixture.json"
        write_new(reference_path, reference)
        ready = dict(zsre_reference=common.member(reference_path))
        endpoint = evaluate_zsre(self.model, self.tok, records, identity=self.external,
                                 w0_reference=reference)
        endpoint.pop("work")
        endpoint_path = folder / "factual" / "batch-01.json"
        write_new(endpoint_path, endpoint)
        hp = hparams("MEMIT", "llama3")
        names = [hp.rewrite_module_tmp.format(layer) + ".weight" for layer in hp.layers]
        native = Path(__file__).resolve().parents[2] / "baselines/sphere/memit/memit_main.py"
        edit = dict(schema="official-server1-native-apply-v1", method="MEMIT", batch=1,
            requests=100, request_sha256=digest(requests(records, "MEMIT", "llama3")),
            entry_selected_sha256=dict.fromkeys(names, "a" * 64),
            post_selected_sha256=dict.fromkeys(names, "b" * 64),
            native_source_sha256=file_sha(native), call_options=call_options("MEMIT", "llama3"),
            same_model=True, nonselected_identity_version_unchanged=True, selected_FP32_finite=True,
            native_history=False, cache_c={}, contexts_sha256="c" * 64, quality_gate=False)
        edit["identity_sha256"] = digest(edit)
        # Bytes-only fixture: never deserialized as a tensor/model/checkpoint.
        payload = folder / "checkpoint" / "fixture.bin"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"CPU collector payload SHA fixture; NOT a model")
        checksum = file_sha(payload)
        pointer = dict(batch=1, file=f"batch-01-{checksum[:16]}.pt", sha256=checksum,
                       identity_sha256=digest(identity), final_W20=False)
        payload.rename(payload.parent / pointer["file"])
        write_new(folder / "checkpoint" / "latest.json", pointer)
        cursor = dict(completed_batch=1, edit=edit, evaluation="COMPLETE", factual=common.member(endpoint_path))
        write_new(folder / "commits" / "batch-01.json", dict(batch=1, identity=identity, cursor=cursor,
                  checkpoint=pointer, actual_applied_requests=100))
        write_new(folder / "READY.json", dict(schema="official-server1-zsre-smoke-v1", source=self.source,
            assets_sha256=self.assets["assets_sha256"], actual_batch100_completed=True, requests=100,
            native_method="MEMIT", factual=common.member(endpoint_path), performance_gate=False))
        return folder, ready

    def test_valid_CPU_fixture_smoke_exact100_and_no_additional_forward_or_GPU_claim(self):
        folder, ready = self.smoke()
        calls = len(self.model.calls)
        result = self.invoke(ready=ready)
        row = result["rows"][0]
        self.assertEqual(row["status"], "VERIFIED_COMPLETE_LOCAL_RECEIPTS")
        self.assertEqual(row["CPU_audit"]["native_commits"]["actual_applied_requests"], 100)
        self.assertTrue(row["CPU_audit"]["native_commits"]["native_request_digest_checked_against_stream"])
        self.assertEqual(row["CPU_audit"]["factual"][0]["requests"], 100)
        self.assertEqual(len(self.model.calls), calls)
        self.assertFalse(result["model_loaded"] or result["GPU_execution_independently_certified"])
        self.assertEqual(result["actual_model_forward_calls"], 0)
        self.assertEqual(result["expected_chains"], 0)

    def test_smoke_false_flag_wrong_count_and_native_request_digest_rejected(self):
        folder, ready = self.smoke()
        receipt_path = folder / "READY.json"
        receipt = common.read(receipt_path)
        receipt["requests"] = 99
        receipt_path.write_text(json.dumps(receipt))
        result = self.invoke(ready=ready)
        self.assertEqual(result["rows"][0]["error_code"], "COLLECTOR_ACTUAL_ZSRE_SMOKE100_REQUIRED")

    def test_smoke_raw_corruption_not_hidden_by_unchanged_summary(self):
        folder, ready = self.smoke()
        endpoint_path = folder / "factual" / "batch-01.json"
        value = common.read(endpoint_path)
        value["cases"][0]["neighborhood_observations"][0]["token_correct_count"] += 1
        endpoint_path.write_text(json.dumps(value))
        # Re-sign all transport members: the semantic CPU audit still rejects.
        endpoint_member = common.member(endpoint_path)
        for path, location in ((folder / "READY.json", "factual"),
                (folder / "commits" / "batch-01.json", "cursor")):
            item = common.read(path)
            if location == "cursor":
                item["cursor"]["factual"] = endpoint_member
            else:
                item["factual"] = endpoint_member
            path.write_text(json.dumps(item))
        result = self.invoke(ready=ready)
        self.assertEqual(result["rows"][0]["error_code"], "AUDIT_FACTUAL_TOKEN_DENOMINATOR")

    def test_partial_receipt_files_are_counts_not_completion(self):
        self.profiles(["chain"], purpose="resume")
        folder = Path(self.jobs[0]["output"])
        write_new(folder / "commits" / "batch-01.json", {"not_validated": True})
        write_new(folder / "failures" / "fixture.json", {"private_error": "not copied to report"})
        result = self.invoke()
        row = result["rows"][0]
        self.assertEqual(row["unverified_present_commit_files"], 1)
        self.assertEqual(row["local_failure_receipt_count"], 1)
        self.assertFalse(row["actual_completion"])
        self.assertNotIn("private_error", json.dumps(result))


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False


class GenerationCollectionTests(unittest.TestCase):
    """Real native CPU reader/reference reduction, zero production forwards."""
    def setUp(self):
        from official.evaluation.generation.native_observer import NativeGenerationObserver
        from official.evaluation.generation.test_native_generator import Model, Tokenizer, prompt
        from official.evaluation.generation.test_observer import FakeAssets, record
        class SingleAndBatchTokenizer(Tokenizer):
            def __call__(self, value, **kwargs):
                if type(value) is str:
                    return dict(input_ids=[int(token) for token in value.split()])
                return super().__call__(value, **kwargs)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "base-W0"
        self.tok, self.model = SingleAndBatchTokenizer(), Model()
        self.refs = FakeAssets()
        self.assets = dict(assets_sha256="a" * 64,
            model=dict(identity=dict(revision="CPU-fixture-not-GPU"), tokenizer_sha256="b" * 64),
            generation_reference=dict(identity_sha256=self.refs.sha))
        reference_path = self.root / "reference-fixture.json"
        write_new(reference_path, {"CPU_fixture_only": True, "not_a_production_reference": True})
        self.assets["generation_reference"]["manifest"] = common.member(reference_path)
        self.lock = dict(source=dict(main_commit="c" * 40, official_tree="d" * 40))
        self.records = [dict(record(i + 1, i + 10, prompts=[prompt(100)] if i == 0 else []),
                             occurrence_index=i + 1) for i in range(2000)]
        self.config = dict(model_identity=dict(model="llama3", revision=self.assets["model"]["identity"]["revision"],
            tokenizer_sha256=self.assets["model"]["tokenizer_sha256"]), profile=PROFILE, eval_seed=20261007,
            generation_source_sha=dict(code_commit=self.lock["source"]["main_commit"],
                                       official_tree=self.lock["source"]["official_tree"]))
        observer = NativeGenerationObserver(self.model, self.tok, self.refs, self.config, self.folder / "generation")
        state = dict(base_model=self.assets["model"]["identity"], assets_sha256=self.assets["assets_sha256"],
                     actual_model_edits=0)
        self.value = observer.observe(self.records, "W0", "first2000", state)
        self.saved_path = self.folder / "cf-generation-local.json"
        write_new(self.saved_path, {k: v for k, v in self.value.items() if k != "work"})
        self.scope = patch.object(common, "LOCAL_ROOT", self.root)
        self.scope.start()
        self.addCleanup(self.scope.stop)

    def audit(self, *, records=None, assets=None, lock=None, endpoint="W0", **kwargs):
        return run._collect_generation(self.saved_path, records or self.records, self.tok,
            assets or self.assets, lock or self.lock, endpoint, references=self.refs, folder=self.folder, **kwargs)

    def test_native_full_first2000_readback_is_CPU_not_GPU_qualification(self):
        before = len(self.model.calls)
        result = self.audit()
        self.assertEqual(result["requests"], 2000)
        self.assertEqual(result["prompts"], 1)
        self.assertEqual(result["actual_model_forward_calls"], 0)
        self.assertEqual(result["native_profile"], PROFILE)
        self.assertEqual(len(self.model.calls), before)
        self.assertEqual(self.value["summary"]["missing_reason_counts"]["missing_generation_prompts"], 1999)

    def test_exact_ordered_occurrence_and_prompt_target_required_not_only_count(self):
        records = copy.deepcopy(self.records)
        records[0], records[1] = records[1], records[0]
        with self.assertRaisesRegex(ValueError, "EXACT_FIRST2000"):
            self.audit(records=records)
        records = copy.deepcopy(self.records)
        records[0]["requested_rewrite"]["target_new"]["id"] = "different-reference-target"
        with self.assertRaisesRegex(ValueError, "ORDERED_PROMPT_TARGET_IDENTITY"):
            self.audit(records=records)

    def test_source_runtime_and_endpoint_cannot_be_relabelled(self):
        from official.evaluation.generation.common import GenerationError
        lock = copy.deepcopy(self.lock)
        lock["source"]["main_commit"] = "e" * 40
        with self.assertRaisesRegex(GenerationError, "RUNTIME_IDENTITY"):
            self.audit(lock=lock)
        with self.assertRaisesRegex(ValueError, "ENDPOINT_COHORT"):
            self.audit(endpoint="W20")

    def test_reference_and_cold_zero_state_guards(self):
        assets = copy.deepcopy(self.assets)
        assets["generation_reference"]["identity_sha256"] = "wrong-actual-reference"
        with self.assertRaisesRegex(ValueError, "REFERENCE_IDENTITY"):
            self.audit(assets=assets)
        assets = copy.deepcopy(self.assets)
        assets["assets_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "COLD_W0_STATE"):
            self.audit(assets=assets)

    def test_local_summary_cannot_disagree_with_guarded_native_endpoint(self):
        value = common.read(self.saved_path)
        value["summary"]["planned_count"] = 1999
        self.saved_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "LOCAL_RAW_IDENTITY"):
            self.audit()

    def test_native_tokenizer_binding_not_only_raw_self_hash(self):
        original = self.tok.__class__.__call__
        def different_tokens(tokenizer, value, **kwargs):
            output = original(tokenizer, value, **kwargs)
            if type(value) is str:
                output["input_ids"] = [8, *output["input_ids"]]
            return output
        with patch.object(self.tok.__class__, "__call__", different_tokens):
            with self.assertRaisesRegex(ValueError, "NATIVE_INPUT_TOKENS"):
                self.audit()

    def test_edited_W20_bound_to_actual_last_native_weight_hashes(self):
        from official.evaluation.generation.native_observer import NativeGenerationObserver
        selected = dict(method="MEMIT", successful_calls=20, cache_c={},
            selected_weights={"fixture-selected-weight": dict(sha256="e" * 64, shape=[4, 4], dtype="torch.float32")},
            contexts_sha256="f" * 64)
        selected["identity_sha256"] = digest(selected)
        state = dict(method="MEMIT", model=self.assets["model"]["identity"], selected_state=selected,
                     source=self.lock["source"], actual_model_edits=2000)
        observer = NativeGenerationObserver(self.model, self.tok, self.refs, self.config,
                                             self.folder / "generation-W20")
        value = observer.observe(self.records, "W20", "first2000", state)
        self.saved_path.write_text(json.dumps({k: v for k, v in value.items() if k != "work"}))
        edit = dict(post_selected_sha256={"fixture-selected-weight": "e" * 64}, contexts_sha256="f" * 64)
        self.assertEqual(self.audit(endpoint="W20", method="MEMIT", final_edit=edit)["endpoint"], "W20")
        edit["post_selected_sha256"]["fixture-selected-weight"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "NATIVE_COMMIT_STATE"):
            self.audit(endpoint="W20", method="MEMIT", final_edit=edit)


if __name__ == "__main__":
    unittest.main()
