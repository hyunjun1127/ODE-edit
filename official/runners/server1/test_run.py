"""CPU-only runner connector regressions; fixtures are not GPU qualification."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from official.experiments import checkpoint
from official.experiments.prepare import digest, write_new
from official.runners.server1 import common, run


class FixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.eval()


class FixtureEngine:
    """One CPU scalar and native-shaped cursor; no pretrained implementation."""
    def __init__(self, model, tokenizer, method, assets, *, source_verified=False):
        self.model, self.method = model, method
        self.selected_weights = {"W": model.W}
        self.successful_calls = 0
        self.context = None

    def apply(self, records):
        if len(records) != 100:
            raise ValueError("FIXTURE_EXPECTS_REAL_EXTERNAL_BATCH100")
        self.successful_calls += 1
        self.context = [["native fixture. {}"]]
        with torch.no_grad():
            self.model.W.add_(torch.rand(1) + random.random() + np.random.random())
        return dict(batch=self.successful_calls, requests=100, same_model=True)

    def contexts(self):
        return dict(method=self.method, successful_calls=self.successful_calls, contexts=self.context)

    def restore_contexts(self, value):
        self.successful_calls = value["successful_calls"]
        self.context = value["contexts"]

    def state_identity(self):
        return dict(method=self.method, calls=self.successful_calls, contexts=self.context,
                    weights=common.rng_digest(self.selected_weights))


class FixtureTracker:
    def __init__(self, *args, **kwargs):
        self.payloads = []

    def log(self, payload):
        self.payloads.append(payload)

    def finish(self, **kwargs):
        return dict(status="CPU_FIXTURE_NOT_ONLINE_VERIFIED")


def seed_CPU():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


class RunnerConnectorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = {name: "fixture-" + name for name in checkpoint.IDENTITY_FIELDS}
        self.identity.update(code_commit="a" * 40, official_tree_sha256="b" * 40)
        self.lock = dict(source=dict(main_commit="a" * 40, official_tree="b" * 40))
        self.records = [dict(case_id=index, occurrence_index=index + 1,
                             requested_rewrite=dict(prompt="{} is", subject="A",
                                                    target_new={"str": "B"}, target_true={"str": "C"}))
                        for index in range(2000)]
        self.assets = dict(model=dict(identity=dict(model="llama3", revision="revision")),
                           assets_sha256="assets")
        self.external = dict(model="llama3", source="fixture")

    def endpoint(self, model, tokenizer, records, dataset, external, **kwargs):
        value = float(model.W.item())
        cases = [dict(case_id=record["case_id"], occurrence_index=record["occurrence_index"],
                      observed=value) for record in records]
        identity = dict(dataset=dataset, external_identity=external,
                        cohort=digest([row["occurrence_index"] for row in records]))
        return dict(identity=identity, identity_sha256=digest(identity), cases=cases,
                    summary=dict(requests=len(records), value=value), accuracy={},
                    work=dict(seconds=random.Random().random()))

    def arguments(self, output, *, stage=None, dataset="cf", mode="chain", resume=False):
        return SimpleNamespace(output=Path(output), method="FT", dataset=dataset, mode=mode,
                               qualification_stage=stage, resume=resume,
                               config=self.root / "config.json", source_lock=self.root / "lock.json")

    def connector_patches(self):
        return (patch("official.runners.server1.native.NativeEngine", FixtureEngine),
                patch.object(run, "bindings", return_value=(self.assets, self.records,
                                                            self.identity, self.external)),
                patch.object(run, "load_model", side_effect=lambda assets: (FixtureModel(), object())),
                patch.object(run, "seed_edit", seed_CPU),
                patch.object(run, "factual", self.endpoint),
                patch("official.runners.server1.w0_binding.execution_identity",
                      return_value=dict(test_only_CPU_serialization_fixture=True)),
                patch("official.runners.server1.native_parity.qualify",
                      side_effect=lambda model, tok, records, external, engine, plan, endpoint:
                          dict(state_after=engine.state_identity(), identity=dict(method=engine.method),
                               test_only_CPU_fixture=True)))

    def test_stage_continuous_B3_equals_fresh_process_B2_resume_B3_CPU_fixture(self):
        from contextlib import ExitStack
        continuous, split = self.root / "continuous", self.root / "split"
        with ExitStack() as stack:
            for context in self.connector_patches():
                stack.enter_context(context)
            run.stage(self.arguments(continuous, stage="continuous"),
                      {"qualification_plan": {"native_reference_matched_B3": "CPU_FIXTURE"}}, self.lock, continuous)
            run.stage(self.arguments(split, stage="stop"), {}, self.lock, split)
            self.assertEqual(checkpoint.load(split / "checkpoint", self.identity)["batch"], 2)
            run.stage(self.arguments(split, stage="resume"), {}, self.lock, split)
        left, right = (common.read(path / "stage-complete.json") for path in (continuous, split))
        checks = run.compare_qualification(left, right)
        self.assertEqual(checks["RNG"], "EXACT")
        self.assertEqual(checks["selected_weights"], "EXACT_SHA256")
        self.assertEqual(left["actual_native_fit_calls"], 3)
        self.assertEqual(right["actual_native_fit_calls"], 1)
        self.assertEqual(len(list((split / "checkpoint").glob("batch-*.pt"))), 1)
        # These are mocked CPU connectors, not actual native/GPU proof.
        self.assertEqual(left["method"], "FT")

    def test_compare_qualification_excludes_work_but_rejects_scalar_drift(self):
        left_path, right_path = self.root / "left.json", self.root / "right.json"
        model = FixtureModel()
        endpoint = self.endpoint(model, object(), self.records[:300], "cf", self.external)
        write_new(left_path, endpoint)
        other = deepcopy(endpoint)
        other["work"]["seconds"] += 1000
        write_new(right_path, other)
        state = dict(identity=self.identity, method="FT", completed_batch=3,
                     selected_state={"W": "hash"}, contexts_sha256="contexts",
                     RNG_sha256="rng", checkpoint_RNG_sha256="rng")
        left, right = dict(state, factual_member=common.member(left_path)), dict(state, factual_member=common.member(right_path))
        run.compare_qualification(left, right)
        other["summary"]["value"] = 42
        right_path.write_text(json.dumps(other))
        right["factual_member"] = common.member(right_path)
        with self.assertRaisesRegex(ValueError, "FACTUAL_RESUME_MISMATCH_SUMMARY"):
            run.compare_qualification(left, right)

    def test_qualification_dispatches_three_distinct_stage_processes(self):
        tracker = FixtureTracker()
        config = dict(qualification_plan={"schema": "CPU fixture plan"})
        output = self.root / "qualification"
        output.mkdir()
        factual = self.root / "qual-factual.json"
        write_new(factual, self.endpoint(FixtureModel(), object(), self.records[:300], "cf", self.external))
        state = dict(identity=self.identity, method="FT", completed_batch=3,
                     selected_state={"W": "hash"}, contexts_sha256="contexts",
                     RNG_sha256="rng", checkpoint_RNG_sha256="rng", factual_member=common.member(factual))
        calls = []
        def fake(argv, *, check):
            calls.append(argv)
            stage = argv[argv.index("--qualification-stage") + 1]
            folder = Path(argv[argv.index("--output") + 1])
            folder.mkdir(exist_ok=True)
            if stage == "stop":
                write_new(folder / "stage-stop.json", dict(state, completed_batch=2))
            else:
                write_new(folder / "stage-complete.json", state)
            return SimpleNamespace(returncode=0)
        with patch.object(run.subprocess, "run", fake), \
                patch.object(run, "verify_native_parity_stage", return_value={"test_only_CPU_fixture": True}):
            run.qualification(self.arguments(output, mode="qualification"), config, self.lock, output, tracker)
        self.assertEqual(len(calls), 3)
        self.assertEqual([argv[argv.index("--qualification-stage") + 1] for argv in calls],
                         ["continuous", "stop", "resume"])
        self.assertTrue(all("qualification_stage" in argv and "--source-lock" in argv for argv in calls))
        self.assertEqual([Path(argv[argv.index("--output") + 1]).name for argv in calls], ["continuous", "split", "split"])

    def test_native_oracle_missing_report_never_produces_qualification_READY(self):
        endpoint = self.root / "endpoint.json"
        write_new(endpoint, dict(identity=dict(external_identity=self.external)))
        value = dict(factual_member=common.member(endpoint), method="FT", selected_state={})
        with self.assertRaises(KeyError):
            run.verify_native_parity_stage(value, {"native_reference_matched_B3": {}})

    def test_native_oracle_report_must_match_continuous_B3_state(self):
        endpoint, report = self.root / "endpoint.json", self.root / "oracle.json"
        write_new(endpoint, dict(identity=dict(external_identity=self.external)))
        write_new(report, dict(state_after={"wrong": "state"}, identity=dict(method="FT")))
        value = dict(factual_member=common.member(endpoint), native_parity_member=common.member(report),
                     method="FT", selected_state={"correct": "state"})
        with self.assertRaisesRegex(ValueError, "COMPLETED_B3_STATE_IDENTITY"):
            run.verify_native_parity_stage(value, {"native_reference_matched_B3": {}})

    def test_native_oracle_stage_uses_separate_binding_not_augmented_raw_identity(self):
        endpoint, report = self.root / "endpoint.json", self.root / "oracle.json"
        state = dict(method="FT", successful_calls=3)
        original = dict(identity=dict(external_identity=self.external))
        write_new(endpoint, original)
        write_new(report, dict(state_after=state, reference_binding=dict(state_identity=state),
                               canonical=original, canonical_payload_sha256=digest(original)))
        value = dict(factual_member=common.member(endpoint), native_parity_member=common.member(report),
                     method="FT", selected_state=state)
        with patch("official.runners.server1.native_parity.validate_report", return_value="CPU_FIXTURE_VALIDATED"):
            self.assertEqual(run.verify_native_parity_stage(value, {"native_reference_matched_B3": {}}),
                             "CPU_FIXTURE_VALIDATED")

    def test_native_oracle_matching_external_different_canonical_member_is_rejected_early(self):
        endpoint, report = self.root / "endpoint.json", self.root / "oracle.json"
        state = dict(method="FT", successful_calls=3)
        original = dict(identity=dict(external_identity=self.external), observations="measured")
        other = dict(identity=dict(external_identity=self.external), observations="different")
        write_new(endpoint, original)
        write_new(report, dict(state_after=state, reference_binding=dict(state_identity=state),
                               canonical=other, canonical_payload_sha256=digest(other)))
        value = dict(factual_member=common.member(endpoint), native_parity_member=common.member(report),
                     method="FT", selected_state=state)
        with patch("official.runners.server1.native_parity.validate_report") as validator:
            with self.assertRaisesRegex(ValueError, "CANONICAL_COMPLETED_B3_MEMBER"):
                run.verify_native_parity_stage(value, {"native_reference_matched_B3": {}})
            validator.assert_not_called()

    def test_qualification_other_stream_is_rejected_before_loading_stage_members(self):
        identity = dict(self.identity, stream_sha256="a" * 64)
        value = dict(schema="official-server1-native-resume-READY-v1", method="FT", passed=True,
                     actual_native_B3_and_B2_resume=True, actual_fit_calls=6, generation_calls=0,
                     identity=dict(identity, stream_sha256="b" * 64))
        with patch.object(run, "verify") as member_verifier:
            with self.assertRaisesRegex(ValueError, "SOURCE_ASSET_IDENTITY"):
                run.verify_qualification(value, "FT", identity)
            member_verifier.assert_not_called()

    def test_zsre_caller_binds_CF_qualification_to_actual_CF_bundle_not_zsre_stream(self):
        cf, bundle, assets = (self.root / name for name in ("cf.json", "bundle.json", "assets.json"))
        write_new(cf, self.records)
        cf_member = common.member(cf)
        write_new(bundle, dict(datasets=dict(cf=dict(stream=cf_member))))
        write_new(assets, dict(streams=dict(cf=dict(stream_sha256=cf_member["sha256"]))))
        config = dict(dataset="zsre", stream_bundle_member=common.member(bundle),
                      assets_member=common.member(assets), qualification_plan={},
                      qualification_outputs={method:str(self.root / method) for method in common.METHODS})
        for method in common.METHODS:
            (self.root / method).mkdir()
            write_new(self.root / method / "READY.json", {"method":method})
        zsre_identity = dict(self.identity, stream_sha256="e" * 64)
        with patch.object(run, "verify_qualification") as qualifier:
            run.verify_qualifications(config, zsre_identity)
            self.assertEqual(qualifier.call_count, 3)
            for call in qualifier.call_args_list:
                self.assertEqual(call.args[2], dict(zsre_identity, stream_sha256=cf_member["sha256"]))
        self.assertEqual(zsre_identity["stream_sha256"], "e" * 64)

    def test_zsre_caller_rejects_CF_bundle_asset_mismatch_before_qualification(self):
        cf, bundle, assets = (self.root / name for name in ("cf.json", "bundle.json", "assets.json"))
        write_new(cf, self.records)
        write_new(bundle, dict(datasets=dict(cf=dict(stream=common.member(cf)))))
        write_new(assets, dict(streams=dict(cf=dict(stream_sha256="f" * 64))))
        config = dict(dataset="zsre", stream_bundle_member=common.member(bundle), assets_member=common.member(assets))
        with patch.object(run, "verify_qualification") as qualifier:
            with self.assertRaisesRegex(ValueError, "ACTUAL_CF_STREAM_ASSET_IDENTITY"):
                run.verify_qualifications(config, dict(self.identity, stream_sha256="e" * 64))
            qualifier.assert_not_called()

    def test_qualification_failed_stage_never_writes_READY(self):
        output = self.root / "failed-qualification"
        output.mkdir()
        with patch.object(run.subprocess, "run", return_value=SimpleNamespace(returncode=2)):
            with self.assertRaisesRegex(ValueError, "STAGE_FAILED_CONTINUOUS"):
                run.qualification(self.arguments(output, mode="qualification"),
                                  {"qualification_plan": {}}, self.lock, output, FixtureTracker())
        self.assertFalse((output / "READY.json").exists())

    def test_missing_or_false_qualification_fails_closed(self):
        outputs = {}
        for method in common.METHODS:
            folder = self.root / method
            folder.mkdir()
            outputs[method] = str(folder)
            write_new(folder / "READY.json", dict(schema="official-server1-native-resume-READY-v1",
                method=method, identity=self.identity, passed=False, actual_native_B3_and_B2_resume=False))
        with self.assertRaisesRegex(ValueError, "ACTUAL_NATIVE_QUALIFICATION_READY_REQUIRED"):
            run.verify_qualifications(dict(qualification_outputs=outputs), self.identity)

    def test_stream_binding_uses_actual_pretty_serialized_SHA_not_compact_digest(self):
        # Exactly the bytes emitted by official.experiments.prepare.prepare_stream.
        streams, assets = {}, deepcopy(self.assets)
        assets["streams"] = {}
        for dataset in ("cf", "zsre"):
            stream = self.root / (dataset + "-stream.json")
            write_new(stream, self.records)
            streams[dataset] = dict(stream=common.member(stream))
            assets["streams"][dataset] = dict(stream_sha256=common.member(stream)["sha256"])
        assets["runtime"] = dict(dependency_versions={name: "fixture" for name in ("torch", "transformers", "numpy")})
        assets["model"]["tokenizer_sha256"] = "tokenizer"
        assets_path, bundle = self.root / "assets.json", self.root / "bundle.json"
        write_new(assets_path, assets)
        write_new(bundle, dict(datasets=streams))
        config = dict(dataset="cf", config_sha256="config", assets_member=common.member(assets_path),
                      stream_bundle_member=common.member(bundle))
        self.assertNotEqual(digest(self.records), streams["cf"]["stream"]["sha256"])
        with patch("official.runners.server1.assets.verify_manifest", return_value={}), \
             patch.object(common.importlib.metadata, "version", return_value="fixture"):
            _, records, identity, _ = common.bindings(config, self.lock)
        self.assertEqual(len(records), 2000)
        self.assertEqual(identity["stream_sha256"], streams["cf"]["stream"]["sha256"])

    def test_restore_checkpoint_weights_context_RNG_exact_and_no_history(self):
        model = FixtureModel()
        engine = FixtureEngine(model, object(), "FT", {})
        seed_CPU()
        engine.apply(self.records[:100])
        checkpoint.save(self.root / "checkpoint", batch=0, weights=engine.selected_weights, cache_c={},
                        contexts=engine.contexts(), evaluation_cursor={}, identity=self.identity,
                        method="FT", evaluation_complete=True)
        payload = checkpoint.load(self.root / "checkpoint", self.identity)
        payload["batch"] = 1
        expected = common.rng_digest(payload["rng"])
        with torch.no_grad():
            model.W.zero_()
        engine.successful_calls = 0
        common.restore_checkpoint(model, engine, payload, self.identity)
        self.assertEqual(engine.successful_calls, 1)
        self.assertEqual(common.rng_digest(checkpoint.rng_snapshot()), expected)
        self.assertTrue(torch.equal(model.W.detach(), payload["weights"]["W"]))
        bad = deepcopy(payload)
        bad["cache_c"] = {"invented_H": torch.zeros(1)}
        with self.assertRaisesRegex(ValueError, "NON_HISTORY_SCHEMA"):
            common.restore_checkpoint(model, engine, bad, self.identity)

    def test_restore_checkpoint_batch_context_drift_rejected(self):
        model = FixtureModel()
        engine = FixtureEngine(model, object(), "FT", {})
        payload = dict(identity=self.identity, method="FT", batch=2, cache_c={},
                       weights={"W": torch.ones(1)}, contexts=dict(method="FT", successful_calls=3, contexts=None),
                       rng=checkpoint.rng_snapshot())
        with self.assertRaisesRegex(ValueError, "CURSOR|BATCH|CALL"):
            common.restore_checkpoint(model, engine, payload, self.identity)

    def test_latest_checkpoint_ledger_crash_window_recovered_without_refit(self):
        output = self.root / "ledger-recovery"
        cursor = dict(completed_batch=1, edit=dict(batch=1), evaluation="NOT_SCHEDULED_THIS_BATCH")
        pointer = dict(batch=1, identity_sha256=digest(self.identity), file="CPU_FIXTURE", sha256="a" * 64)
        write_new(output / "checkpoint" / "latest.json", pointer)
        payload = dict(batch=1, identity=self.identity, evaluation_cursor=cursor)
        run.recover_committed_ledger(output, payload, self.identity)
        path = output / "commits" / "batch-01.json"
        expected = common.read(path)
        self.assertEqual(expected["cursor"], cursor)
        run.recover_committed_ledger(output, payload, self.identity)
        self.assertEqual(common.read(path), expected)
        wrong = dict(expected, actual_applied_requests=0)
        path.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "EXISTING_LEDGER_CONFLICT"):
            run.recover_committed_ledger(output, payload, self.identity)

    def test_missing_older_ledger_not_reconstructed_from_latest_checkpoint(self):
        output = self.root / "older-ledger-missing"
        write_new(output / "checkpoint" / "latest.json", dict(batch=2, identity_sha256=digest(self.identity)))
        payload = dict(batch=2, identity=self.identity,
                       evaluation_cursor=dict(completed_batch=2, edit=dict(batch=2)))
        with self.assertRaisesRegex(ValueError, "OLDER_LEDGER_MISSING"):
            run.recover_committed_ledger(output, payload, self.identity)

    def test_scalar_payload_missing_is_omitted_not_zero(self):
        endpoint = dict(identity=dict(dataset="zsre"),
                        summary=dict(Efficacy=80.0, Specificity_availability="MISSING"),
                        accuracy={}, cases=[])
        payload = common.factual_payload(endpoint, "all_seen/post", 500)
        self.assertEqual(payload["official/all_seen/post/Efficacy"], 80)
        self.assertEqual(payload["post_state_edits"], 500)
        self.assertNotIn("official/all_seen/post/Specificity_availability", payload)
        self.assertFalse(any("/P/" in key for key in payload))
        from official.evaluation.generation.metrics import PUBLIC_REASONS
        generated = common.generation_payload(dict(summary={"reference_score": .1,
            "planned_count": 2000, "fluency_count": 0, "consistency_count": 1000,
            "generation_prompt_count": 20000, "generated_token_count": 100000,
            "missing_reason_counts": {reason: 0 for reason in PUBLIC_REASONS}}), 2000)
        self.assertNotIn("all_seen/post/fluency/ngram_entropy", generated)

    def test_observation_scientific_bytes_immutable_but_cost_receipts_separate(self):
        path = self.root / "observation.json"
        endpoint = self.endpoint(FixtureModel(), object(), self.records[:100], "cf", self.external)
        common.immutable_observation(path, endpoint)
        changed_work = deepcopy(endpoint)
        changed_work["work"]["seconds"] += 100
        common.immutable_observation(path, changed_work)
        self.assertNotIn("work", common.read(path))
        self.assertEqual(len(list((self.root / "observation-cost").glob("*.json"))), 2)
        changed_science = deepcopy(endpoint)
        changed_science["summary"]["value"] = 99
        with self.assertRaisesRegex(ValueError, "REFUSE_OVERWRITE"):
            common.immutable_observation(path, changed_science)

    def test_W20_generation_failure_preserves_B19_and_resume_recovers_science(self):
        from contextlib import ExitStack
        output = self.root / "chain"
        output.mkdir()
        base = self.root / "base"
        base.mkdir()
        write_new(base / "READY.json", {"CPU_fixture": True})
        factual_W0 = self.root / "cold-factual.json"
        write_new(factual_W0, self.endpoint(FixtureModel(), object(), self.records, "cf", self.external))
        ready = dict(cf_factual=common.member(factual_W0))
        config = dict(base_W0_output=str(base))
        tracker = FixtureTracker()
        class Observer:
            fail = True
            calls = 0
            def observe(self, records, endpoint, *, cohort, state_identity):
                self.calls += 1
                if self.fail:
                    raise ValueError("FIXTURE_W20_GENERATION_FAILURE")
                return dict(identity=state_identity, summary=dict(entropy=1.0), work=dict(seconds=1.0))
        observer = Observer()
        def scalar(endpoint, prefix, edits):
            return dict(edits=edits, pre_state_edits=edits, post_state_edits=edits)
        with ExitStack() as stack:
            for context in self.connector_patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(run, "verify_qualifications"))
            stack.enter_context(patch.object(run, "read_w0", return_value=ready))
            stack.enter_context(patch.object(run, "factual_payload", side_effect=scalar))
            stack.enter_context(patch.object(run, "generation_observer", return_value=observer))
            stack.enter_context(patch.object(run, "generation_payload", side_effect=lambda value, edits: dict(edits=edits, measured_generation=True)))
            with self.assertRaisesRegex(ValueError, "FIXTURE_W20_GENERATION_FAILURE"):
                run.chain(self.arguments(output), config, self.lock, output, tracker)
            self.assertEqual(checkpoint.load(output / "checkpoint", self.identity)["batch"], 19)
            self.assertFalse((output / "COMPLETE.json").exists())
            self.assertEqual(len(list((output / "commits").glob("batch-*.json"))), 19)
            original_factual = common.member(output / "factual" / "batch-20.json")
            observer.fail = False
            run.chain(self.arguments(output, resume=True), config, self.lock, output, tracker)
            self.assertEqual(common.member(output / "factual" / "batch-20.json"), original_factual)
            self.assertEqual(checkpoint.load(output / "checkpoint", self.identity)["batch"], 20)
            self.assertEqual(len(list((output / "checkpoint").glob("batch-*.pt"))), 1)
            self.assertEqual(common.read(output / "COMPLETE.json")["actual_native_apply_calls"], 20)
            self.assertEqual(sum(payload.get("measured_generation") is True for payload in tracker.payloads), 1)
            with self.assertRaisesRegex(ValueError, "CHAIN_ALREADY_COMPLETED_NO_DUPLICATE_EXECUTION"):
                run.chain(self.arguments(output, resume=True), config, self.lock, output, tracker)
        self.assertEqual(observer.calls, 2)

    def test_zsre_smoke_actual_connector_one_batch_without_generation(self):
        from contextlib import ExitStack
        output = self.root / "smoke"
        output.mkdir()
        base = self.root / "base"
        base.mkdir()
        write_new(base / "READY.json", {"CPU_fixture": True})
        reference = self.root / "reference.json"
        write_new(reference, dict(evaluation=self.endpoint(FixtureModel(), object(), self.records, "zsre", self.external)))
        config = dict(base_W0_output=str(base))
        with ExitStack() as stack:
            for context in self.connector_patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(run, "verify_qualifications"))
            stack.enter_context(patch.object(run, "read_w0", return_value=dict(zsre_reference=common.member(reference))))
            stack.enter_context(patch.object(run, "factual_payload", return_value={"edits": 0}))
            generation = stack.enter_context(patch.object(run, "generation_observer"))
            run.chain(self.arguments(output, mode="smoke", dataset="zsre"), config, self.lock, output, FixtureTracker())
        self.assertEqual(checkpoint.load(output / "checkpoint", self.identity)["batch"], 1)
        self.assertEqual(common.read(output / "READY.json")["requests"], 100)
        self.assertEqual(len(common.read(output / "factual" / "batch-01.json")["cases"]), 100)
        generation.assert_not_called()

    def test_owned_output_rejects_outside_and_symlink_ancestor(self):
        root = self.root / "owned"
        root.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        with patch.object(common, "LOCAL_ROOT", root):
            with self.assertRaisesRegex(ValueError, "OWN_IGNORED_OUTPUT_REQUIRED"):
                common.local_output(outside / "run")
            (root / "link").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "LOCAL_OUTPUT_SYMLINK_ANCESTOR"):
                common.local_output(root / "link" / "run")

    def test_frozen_source_config_input_and_environment_binding(self):
        frozen = self.root / "frozen"
        runtime = frozen / "official" / "runners" / "server1" / "common.py"
        runtime.parent.mkdir(parents=True)
        runtime.write_text("# CPU fixture bytes, not executable native source\n")
        config_path, input_path, lock_path = (self.root / name for name in ("config.json", "input.json", "lock.json"))
        write_new(config_path, {"CPU_fixture": True})
        write_new(input_path, {"scalar_input": 1})
        lock = dict(schema="official-server1-execution-lock-v1", server="server1",
            source=self.lock["source"], job_configs=[common.member(config_path)],
            inputs=[common.member(input_path)], frozen_source=dict(directory=str(frozen),
                members=[common.member(runtime)], main_membership_verified=True, official_tree_verified=True))
        write_new(lock_path, lock)
        with patch.object(common, "__file__", str(runtime)), patch.dict(os.environ,
                {"OFFICIAL_CODE_COMMIT": "a" * 40, "OFFICIAL_TREE_SHA256": "b" * 40}):
            self.assertEqual(common.source_binding(lock_path, config_path), lock)
            with patch.dict(os.environ, {"OFFICIAL_CODE_COMMIT": "c" * 40}):
                with self.assertRaisesRegex(ValueError, "SOURCE_ENV_LOCK_CONFLICT"):
                    common.source_binding(lock_path, config_path)
            input_path.write_text('{"scalar_input": 2}')
            with self.assertRaisesRegex(ValueError, "IMMUTABLE_INPUT_MEMBER_CHANGED"):
                common.source_binding(lock_path, config_path)

    def test_collector_false_qualification_READY_cannot_be_complete(self):
        folder, output = self.root / "false-qualification", self.root / "collector"
        folder.mkdir()
        output.mkdir()
        write_new(folder / "READY.json", dict(schema="official-server1-native-resume-READY-v1",
            method="FT", passed=False, actual_native_B3_and_B2_resume=False, identity=self.identity))
        config_path = self.root / "collector-config.json"
        write_new(config_path, dict(method="FT", dataset="cf", config_sha256=self.identity["config_sha256"]))
        manifest = self.root / "job-manifest.json"
        write_new(manifest, dict(source=self.lock["source"], jobs={"qual-ft": "100"},
            profiles=[dict(key="qual-ft", output=str(folder), method="FT", dataset="cf", mode="qualification",
                           config=common.member(config_path))]))
        args = self.arguments(output, mode="collect")
        args.job_manifest = manifest
        profile = dict(key="qual-ft", output=str(folder), method="FT", dataset="cf", mode="qualification")
        with self.assertRaisesRegex(ValueError, "QUALIFICATION|READY|COMPLETE"):
            run._collect_profile(profile, dict(qualification_plan={}), self.lock, folder, self.assets,
                self.records, self.identity, self.external, object(), {})
        self.assertFalse((output / "summary.json").exists())

    def test_W0_READY_not_actually_cold_is_rejected(self):
        folder = self.root / "base-W0"
        folder.mkdir()
        files = {}
        for key in ("cf_factual", "cf_generation", "zsre_reference"):
            path = folder / (key + ".json")
            write_new(path, {"CPU_fixture": True})
            files[key] = common.member(path)
        write_new(folder / "READY.json", dict(schema="official-server1-base-W0-READY-v1",
            actual_complete=True, actual_model_edits=1, model_identity=self.assets["model"]["identity"],
            source=self.lock["source"], assets_sha256=self.assets["assets_sha256"],
            CF_W0_observed_once=True, shared_across_methods=True, **files))
        with self.assertRaisesRegex(ValueError, "W0|COLD|STATE"):
            run.read_w0(dict(base_W0_output=str(folder)), self.lock, self.assets)

    def test_absent_transport_and_explicit_log_rejection_are_blockers(self):
        config = dict(tracking=dict(module="official.tracking.client", entity="wkdguswns2256",
                                   project="layer allocation", attempt="fixture", env_file="private-env-path"))
        with patch.object(common.importlib, "import_module", side_effect=ModuleNotFoundError("missing")):
            with self.assertRaisesRegex(ValueError, "OFFICIAL_WANDB_TRANSPORT_NOT_PUBLISHED"):
                common.Tracking(config, self.root, self.identity, mode="chain", method="FT", dataset="cf")
        client = SimpleNamespace(log=lambda value: False)
        tracking = common.Tracking.__new__(common.Tracking)
        tracking.tracker = client
        with self.assertRaisesRegex(ValueError, "TRACKING_ACCEPTANCE_FAILURE_NOT_SILENT_DROP"):
            tracking.log({"edits": 0})
        with self.assertRaisesRegex(ValueError, "SCALAR_ONLY_TRACKING"):
            tracking.log({"raw": [1, 2]})

    def test_original_scientific_exception_survives_finish_error(self):
        config = dict(dataset="cf", method="FT")
        write_new(self.root / "config.json", config)
        tracker = FixtureTracker()
        def fail_finish(**kwargs):
            raise RuntimeError("TRANSPORT_FINISH_ERROR")
        tracker.finish = fail_finish
        argv = ["run", "--mode", "chain", "--method", "FT", "--dataset", "cf",
                "--config", str(self.root / "config.json"), "--output", str(self.root / "out"),
                "--source-lock", str(self.root / "lock.json")]
        output = self.root / "out"
        output.mkdir()
        with patch.object(run.sys, "argv", argv), patch.object(run, "validate_config", return_value=config), \
             patch.object(run, "source_binding", return_value=self.lock), patch.object(run, "local_output", return_value=output), \
             patch.object(run, "bindings", return_value=(self.assets, self.records, self.identity, self.external)), \
             patch.object(run, "Tracking", return_value=tracker), \
             patch.object(run, "chain", side_effect=ValueError("ORIGINAL_SCIENTIFIC_ERROR")):
            with self.assertRaisesRegex(ValueError, "ORIGINAL_SCIENTIFIC_ERROR"):
                run.main()


if __name__ == "__main__":
    unittest.main()
