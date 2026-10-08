"""CPU checks for server3's physical-run mapping and scalar boundaries."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

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
        assets = {"generation_reference": {"identity_sha256": "c" * 64}}
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
