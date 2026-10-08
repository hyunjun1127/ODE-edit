"""CPU checks for server3's physical-run mapping and scalar boundaries."""

import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from official.experiments.prepare import build_matrix, digest, load_plan
from official.runners.server3 import run


class Server3RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        contract, profiles = load_plan()
        cls.rows = {row["run_id"]: row for row in build_matrix(contract, profiles)
                    if row["model"] == "qwen25"}

    def test_all_sixteen_physical_rows_accepted_and_alias_rejected(self):
        physical = [row for row in self.rows.values()
                    if row["execution_kind"] != "selected_grid_alias"]
        self.assertEqual(len(physical), 16)
        for row in physical:
            self.assertEqual(run.validate_config(row), row)
        with self.assertRaisesRegex(ValueError, "CF_BLUE_ALIAS_NOT_PHYSICAL_CHAIN"):
            run.validate_config(self.rows["qwen25-cf-alphaedit_blue"])

    def test_only_blue_zsre_selected_l2_may_change(self):
        original = self.rows["qwen25-zsre-alphaedit_blue"]
        selected = copy.deepcopy(original)
        selected["hparams"]["L2"] = 10
        selected["config_sha256"] = digest({k: v for k, v in selected.items()
                                            if k != "config_sha256"})
        self.assertEqual(run.validate_config(selected), selected)
        changed = copy.deepcopy(selected)
        changed["hparams"]["clamp_norm_factor"] = 0.75
        changed["config_sha256"] = digest({k: v for k, v in changed.items()
                                           if k != "config_sha256"})
        with self.assertRaisesRegex(ValueError, "DERIVED_BLUE_HPARAMS_CHANGED"):
            run.validate_config(changed)

    def test_cf_macro_score_is_not_fabricated_prompt_success_count(self):
        from official.tracking.method import harmonic, validate
        cases = [{"rewrite_prompts_probs": [{}], "paraphrase_prompts_probs": [{}, {}],
                  "neighborhood_prompts_probs": [{}] * 10} for _ in range(2000)]
        summary = {"Efficacy": 80.0, "Generalization": 70.0,
                   "Specificity": 60.0, "Score": harmonic([80.0, 70.0, 60.0]),
                   "requests": 2000}
        values = run._factual_scalars("cf", cases, summary, "W0_first2000", 0)
        self.assertEqual(values["official/W0_first2000/requests"], 2000)
        self.assertFalse(any(k.endswith("/success_count") for k in values))
        self.assertFalse(any("/R/count" in k or "/P/count" in k for k in values))
        self.assertEqual(values["official/W0_first2000/Specificity"], 60.0)
        validate(values, scientific=True, official=True)

    def test_zsre_never_calls_accuracy_nll_preference(self):
        from official.tracking.method import validate
        cases = [{"rewrite_prompts_correct": [True, False],
                  "paraphrase_prompts_correct": [True],
                  "neighborhood_W0_agreement": [False, True]}] * 100
        summary = {"Efficacy": 50.0, "Generalization": 100.0,
                   "Specificity": 50.0, "Specificity_loc_ans": 25.0,
                   "requests": 100}
        values = run._factual_scalars("zsre", cases, summary, "current/post", 100)
        self.assertEqual(values["official/current/post/Efficacy"], 50.0)
        self.assertEqual(values["official/current/post/Specificity_loc_ans"], 25.0)
        self.assertFalse(any(k.endswith("/success_pct") for k in values))
        validate(values, scientific=True, official=True)

    def test_official_tracking_config_binds_cf_and_omits_zsre_generation(self):
        from official.tracking.schema import config as check_config
        assets = {"generation_reference": {"identity_sha256": "c" * 64,
                    "native_generator_sha256": "d" * 64}}
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                "ODEEDIT_WANDB_ENV_FILE": str(Path(folder) / "wandb.env"),
                "ODEEDIT_ATTEMPT_ID": "fixture"}):
            with patch("official.tracking.init") as start:
                run._tracker(folder, arm="qwen25-cf-ft", writer="FT", dataset="cf",
                             source_sha="a" * 40, config_sha="b" * 64, assets=assets)
                cf = start.call_args.kwargs["config"]
                self.assertEqual(check_config(cf), cf)
                self.assertEqual(cf["metric_schema"], "official-baselines-scalar-v1")
                run._tracker(folder, arm="qwen25-zsre-ft", writer="FT", dataset="zsre",
                             source_sha="a" * 40, config_sha="b" * 64, assets=assets)
                zsre = start.call_args.kwargs["config"]
                self.assertEqual(check_config(zsre), zsre)
                self.assertFalse(any(k.startswith("generation_") for k in zsre))

    def test_milestone_current_is_raw_last100_of_same_all_seen_endpoint(self):
        from official.evaluation.reduce import counterfact
        from official.tracking.method import validate
        cases = []
        for index in range(500):
            new = 0.0 if index >= 400 else 2.0
            cases.append({kind + "_prompts_probs": [{"target_new": new,
                "target_true": 1.0}] for kind in ("rewrite", "paraphrase") } |
                {"neighborhood_prompts_probs": [{"target_new": 2.0,
                    "target_true": 1.0}]})
        values = run._milestone_scalars("cf", cases, counterfact(cases), 500)
        self.assertEqual(values["official/all_seen/post/requests"], 500)
        self.assertEqual(values["official/current/post/requests"], 100)
        self.assertEqual(values["official/current/post/Efficacy"], 100.0)
        self.assertEqual(values["official/all_seen/post/Efficacy"], 20.0)
        validate(values, scientific=True, official=True)

    def test_shared_factual_api_preserves_w0_reference_and_identity(self):
        from official.evaluation.reduce import counterfact
        from unittest.mock import Mock
        cases = [{"rewrite_prompts_probs": [{"target_new": 0.0, "target_true": 1.0}],
                  "paraphrase_prompts_probs": [{"target_new": 0.0, "target_true": 1.0}],
                  "neighborhood_prompts_probs": [{"target_new": 2.0, "target_true": 1.0}]}]
        observed = {"cases": cases, "summary": counterfact(cases), "accuracy": {},
                    "work": {}, "identity_sha256": "i" * 64}
        module = Mock()
        module.evaluate.return_value = observed
        reference = {"schema": "zsre-w0"}
        returned = run._evaluate_factual(module, object(), object(), ["one"], "cf",
                                         w0_reference=reference, identity={"seal": "same"})
        self.assertIs(returned, observed)
        self.assertEqual(module.evaluate.call_args.kwargs["w0_reference"], reference)
        self.assertEqual(module.evaluate.call_args.kwargs["identity"], {"seal": "same"})

    def test_zsre_loc_ans_accuracy_is_not_generic_neighborhood_true_target(self):
        observed = {"accuracy": {"rewrite": {"prompt_count": 100,
                     "token_acc_pct": 80., "prompt_acc_pct": 75., "strict_acc_pct": 60.},
                     "paraphrase": {"prompt_count": 100, "token_acc_pct": 70.,
                     "prompt_acc_pct": 65., "strict_acc_pct": 50.},
                     "neighborhood": {"prompt_count": 100,
                     "token_acc_pct": 10., "prompt_acc_pct": 10.,
                     "strict_acc_pct": 10.}}}
        values = run._accuracy_scalars("zsre", observed, "all_seen/post")
        self.assertEqual(values["all_seen/post/R/token_acc_pct"], 80.)
        self.assertFalse(any("/N/" in key for key in values))

    def test_current_tf_reuses_measured_desired_raw_and_shared_reducer(self):
        desired = {"token_count": 2, "token_correct_count": 2,
                   "strict_correct": True}
        other = {"token_count": 2, "token_correct_count": 0,
                 "strict_correct": False}
        case = {kind + "_observations": [{"desired_target":
                "true" if kind == "neighborhood" else "new",
                "target_new": other if kind == "neighborhood" else desired,
                "target_true": desired if kind == "neighborhood" else other}]
                for kind in ("rewrite", "paraphrase", "neighborhood")}
        values = run._current_accuracy_scalars("cf", [case] * 100)
        self.assertEqual(values["current/post/R/strict_acc_pct"], 100.)
        self.assertEqual(values["current/post/N/token_acc_pct"], 100.)
        self.assertEqual(values["current/post/N/count"], 100)

    def test_existing_w0_receipt_requires_full_source_asset_and_raw_identity(self):
        from official.experiments.prepare import file_sha
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "w0-cases.json").write_text("[]\n")
            lock = {"stream_sha256": "s" * 64}
            source = {"code_commit": "a" * 40, "official_tree_sha256": "t" * 64}
            assets = {"assets_sha256": "b" * 64}
            receipt = dict(model="qwen25", dataset="cf", endpoint="W0",
                           stream_sha256=lock["stream_sha256"],
                           code_commit=source["code_commit"],
                           official_tree_sha256=source["official_tree_sha256"],
                           assets_sha256=assets["assets_sha256"], observed_requests=2000,
                           factual={"cases_sha256": file_sha(out / "w0-cases.json")})
            receipt["receipt_sha256"] = digest(receipt)
            (out / "w0-receipt.json").write_text(json.dumps(receipt))
            self.assertEqual(run._verified_w0_receipt(out, "cf", lock, source, assets), receipt)
            (out / "w0-cases.json").write_text("[1]\n")
            with self.assertRaisesRegex(ValueError, "SHARED_W0_CASE_HASH"):
                run._verified_w0_receipt(out, "cf", lock, source, assets)

    def test_existing_w0_replays_missing_scalar_without_model_forward(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            cases = [{}] * 2000
            run._write_once(out / "w0-cases.json", cases)
            lock = {"stream_sha256": "s" * 64}
            source = {"code_commit": "a" * 40, "official_tree_sha256": "t" * 64}
            assets = {"assets_sha256": "b" * 64, "assets": {}}
            receipt = dict(model="qwen25", dataset="cf", endpoint="W0",
                           run_id="qwen25-cf-W0", stream_sha256=lock["stream_sha256"],
                           code_commit=source["code_commit"],
                           official_tree_sha256=source["official_tree_sha256"],
                           assets_sha256=assets["assets_sha256"], observed_requests=2000,
                           factual={"summary": {"requests": 2000, "Efficacy": 80.0},
                                    "accuracy": {},
                                    "cases_sha256": run._sha(out / "w0-cases.json")},
                           generation_rows_path=str(out / "generation/endpoints" / ("g" * 64 + ".json")),
                           generation_identity_sha256="g" * 64)
            receipt["receipt_sha256"] = digest(receipt)
            run._write_once(out / "w0-receipt.json", receipt)
            tracker = Mock(run_id="fixture", spool=out / "wandb/fixture", dropped=0)
            tracker.log.return_value = True
            context = MagicMock()
            context.__enter__.return_value = tracker
            args = SimpleNamespace(output=out, dataset="cf")
            preflight = ({"ready": True}, None, lock, cases, assets, source)
            from official.evaluation.generation.metrics import PUBLIC_REASONS
            generation_summary = dict(planned_count=2000, fluency_count=2000,
                                      consistency_count=2000, generation_prompt_count=2000,
                                      generated_token_count=10000,
                                      missing_reason_counts={name: 0 for name in PUBLIC_REASONS})
            with patch.object(run, "_preflight", return_value=preflight), \
                 patch.object(run, "_require_evaluator"), \
                 patch.object(run, "_tracker", return_value=context) as start, \
                 patch.object(run, "_load_model", side_effect=AssertionError("model loaded")), \
                 patch.object(run, "_generation_summary_for_replay",
                              return_value=generation_summary) as generation_read:
                self.assertEqual(run.w0(args), 0)
                self.assertEqual(run.w0(args), 0)
            self.assertEqual(start.call_count, 1)
            generation_read.assert_called_once_with(
                out, receipt["generation_rows_path"], receipt["generation_identity_sha256"])
            tracker.log.assert_called_once()
            self.assertEqual(tracker.log.call_args.args[0][
                "W0_first2000/generation/planned_count"], 2000)
            logged = run._read(out / "logging/w0.json")
            self.assertTrue(logged["sdk_queue_accepted"])
            self.assertEqual(logged["source_receipt_sha256"], receipt["receipt_sha256"])
            self.assertEqual(logged["remote_readback"], "NOT_ESTABLISHED_BY_SDK_QUEUE")

    def test_resume_replays_last_committed_milestone_once_and_retries_rejection(self):
        from official.evaluation.reduce import counterfact
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            desired = {"token_count": 1, "token_correct_count": 1,
                       "strict_correct": True}
            other = {"token_count": 1, "token_correct_count": 0,
                     "strict_correct": False}
            case = {kind + "_prompts_probs": [{"target_new": 0.0,
                      "target_true": 1.0}] for kind in ("rewrite", "paraphrase")} | \
                   {"neighborhood_prompts_probs": [{"target_new": 2.0,
                      "target_true": 1.0}]}
            case.update({kind + "_observations": [{"desired_target":
                         "true" if kind == "neighborhood" else "new",
                         "target_new": other if kind == "neighborhood" else desired,
                         "target_true": desired if kind == "neighborhood" else other}]
                         for kind in ("rewrite", "paraphrase", "neighborhood")})
            cases = [case] * 500
            case_path = out / "evaluations/w05-cases.json"
            run._write_once(case_path, cases)
            config = self.rows["qwen25-cf-ft"]
            lock = {"stream_sha256": "s" * 64}
            source = {"code_commit": "a" * 40, "official_tree_sha256": "t" * 64}
            identity = {"config_sha256": config["config_sha256"]}
            factual = {"summary": counterfact(cases), "accuracy": {},
                       "cases_path": str(case_path.resolve()),
                       "cases_sha256": run._sha(case_path)}
            receipt = run._run_receipt(config, lock, source, identity, batch=5,
                                       factual=factual, checkpoint={"sha256": "c" * 64})
            commit_path = out / "commits/b05.json"
            run._write_once(commit_path, receipt)
            payload = {"batch": 6, "evaluation_cursor": {"evaluated_endpoints": [0, 5]}}
            rejected = Mock(run_id="first", spool=out / "wandb/first", dropped=1)
            rejected.log.return_value = False
            self.assertTrue(run._replay_committed_batch_logging(
                out, payload, config, lock, source, identity, rejected))
            self.assertFalse(run._read(out / "logging/w05.json")["sdk_queue_accepted"])
            ten_cases = cases * 2
            ten_path = out / "evaluations/w10-cases.json"
            run._write_once(ten_path, ten_cases)
            ten_factual = {"summary": counterfact(ten_cases), "accuracy": {},
                           "cases_path": str(ten_path.resolve()),
                           "cases_sha256": run._sha(ten_path)}
            ten_receipt = run._run_receipt(config, lock, source, identity, batch=10,
                                           factual=ten_factual,
                                           checkpoint={"sha256": "d" * 64})
            ten_commit = out / "commits/b10.json"
            run._write_once(ten_commit, ten_receipt)
            run._write_once(out / "logging/w10.json",
                            dict(label="w10", sdk_queue_accepted=True,
                                 source_receipt_sha256=run._sha(ten_commit),
                                 remote_readback="NOT_ESTABLISHED_BY_SDK_QUEUE",
                                 run_id="prior", spool="prior", dropped_points_at_call=0))
            later = {"batch": 11, "evaluation_cursor": {"evaluated_endpoints": [0, 5, 10]}}
            accepted = Mock(run_id="second", spool=out / "wandb/second", dropped=0)
            accepted.log.return_value = True
            self.assertTrue(run._replay_committed_batch_logging(
                out, later, config, lock, source, identity, accepted))
            accepted.log.assert_called_once()
            self.assertEqual(accepted.log.call_args.args[0]["edits"], 500)
            self.assertTrue(run._read(out / "logging/w05-replay-second.json")["sdk_queue_accepted"])
            duplicate = Mock(run_id="third", spool=out / "wandb/third", dropped=0)
            self.assertFalse(run._replay_committed_batch_logging(
                out, later, config, lock, source, identity, duplicate))
            duplicate.log.assert_not_called()
            self.assertEqual(rejected.log.call_args.args[0], accepted.log.call_args.args[0])

    def test_generation_replay_reader_requires_endpoint_identity_and_no_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            path = out / "generation/endpoints" / ("a" * 64 + ".json")
            path.parent.mkdir(parents=True)
            path.write_text("{}\n")
            with patch("official.evaluation.generation.native_observer.read_observed",
                       return_value={"identity_sha256": "a" * 64,
                                     "summary": {"planned_count": 2000}}) as read:
                self.assertEqual(run._generation_summary_for_replay(
                    out, path, "a" * 64), {"planned_count": 2000})
                read.assert_called_once_with(path)
            with self.assertRaisesRegex(ValueError, "SCALAR_REPLAY_GENERATION_PATH_OR_IDENTITY"):
                run._generation_summary_for_replay(out, path, "b" * 64)

    def test_checkpoint_pointer_receipt_recovery_including_w20(self):
        from official.experiments.prepare import file_sha
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            config = self.rows["qwen25-cf-ft"]
            lock = {"stream_sha256": "stream-lock"}
            source = {"code_commit": "source-commit", "official_tree_sha256": "tree"}
            identity = {"config_sha256": config["config_sha256"]}
            cases_path = out / "evaluations" / "w20-cases.json"
            cases_path.parent.mkdir(parents=True)
            cases_path.write_text("[]\n")
            factual = {"cases_path": str(cases_path), "cases_sha256": file_sha(cases_path),
                       "summary": {"Score": 0.0}}
            pending = dict(batch=20, config_sha256=config["config_sha256"],
                           stream_sha256=lock["stream_sha256"],
                           code_commit=source["code_commit"],
                           official_tree_sha256=source["official_tree_sha256"],
                           checkpoint_identity=identity, factual=factual,
                           generation=None, placement={}, seconds=1.0,
                           state_hashes=None)
            checksum = digest(pending)
            folder = out / "commits"
            folder.mkdir()
            (folder / f"pending-b20-{checksum}.json").write_text(
                json.dumps(pending, sort_keys=True))
            payload = {"batch": 20, "evaluation_cursor": {
                "pending_receipt_sha256": checksum}}
            ref = {"batch": 20, "sha256": "checkpoint-sha"}
            run._recover_batch_receipts(out, payload, ref, config, lock, source, identity)
            commit = json.loads((folder / "b20.json").read_text())
            self.assertEqual(commit["checkpoint_sha256"], "checkpoint-sha")
            self.assertEqual(commit, json.loads((out / "evaluations" / "w20.json").read_text()))
            run._recover_batch_receipts(out, payload, ref, config, lock, source, identity)
            cases_path.write_text("changed")
            with self.assertRaisesRegex(ValueError, "RECEIPT_FACTUAL_HASH_MISMATCH"):
                run._recover_batch_receipts(out, payload, ref, config, lock, source, identity)

    def test_actual_tokenizer_properties_must_match_pinned_cpu_audit(self):
        audit = run._read(run.ROOT / "official/hparams/tokenizers.lock.json")["audits"][
            "qwen25-cf"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in audit["tokenizer_files_sha256"]:
                (root / name).write_text("fixture")
            tok = type("Qwen2TokenizerFast", (), dict(
                add_bos_token=None, bos_token_id=None, padding_side="right"))()
            with patch.object(run, "_sha", side_effect=lambda path:
                              audit["tokenizer_files_sha256"][path.name]):
                receipt = run._tokenizer_receipt(root, tok, {"dataset": "cf",
                                                    "stream_sha256": "stream"})
                self.assertEqual(receipt["tokenizer_sha256"], audit["tokenizer_sha256"])
                tok.padding_side = "left"
                with self.assertRaisesRegex(ValueError, "TOKENIZER_RUNTIME_CONTRACT_MISMATCH"):
                    run._tokenizer_receipt(root, tok, {"dataset": "cf",
                                                        "stream_sha256": "stream"})


if __name__ == "__main__":
    unittest.main()
