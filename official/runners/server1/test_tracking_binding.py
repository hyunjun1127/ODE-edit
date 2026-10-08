"""CPU connector tests: real official schema, fake transport, no SDK/network.

Fixtures below contain invented local observations, not native/model/Slurm
qualification or online delivery evidence. The fake init explicitly binds the
local NOT_APPLICABLE identity, never a made-up submitted job number.
"""
from copy import deepcopy
import math
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from official.evaluation.reduce import counterfact
from official.experiments.prepare import write_new
from official.runners.server1 import common
from official.tracking import official_generation_progress, schema
from official.tracking.method import AxisState, FIELDS


class SchemaOnlyTracker:
    def __init__(self, config):
        self.config = config
        self.payloads = []
        self.axes = AxisState()

    def log(self, values):
        accepted = schema.metrics(values, scientific=True, config_values=self.config)
        self.axes.accept(accepted)
        self.payloads.append(dict(accepted))
        return True

    def finish(self, *, exit_code, timeout):
        return dict(status="CPU_FIXTURE_NOT_ONLINE_VALIDATED", exit_code=exit_code,
                    timeout=timeout, remote_verification=False)


class SchemaOnlyTransport:
    """Expose init like the real transport, but do no SDK/spool/auth work."""
    def __init__(self):
        self.calls = []

    def init(self, *, env_file, spool, config):
        sanitized = schema.config(config)
        bound = schema.bind_job_identity(sanitized, environ={})
        tracker = SchemaOnlyTracker(bound)
        self.calls.append(dict(env_file=env_file, spool=Path(spool),
                               supplied_config=dict(config), config=bound, tracker=tracker))
        return tracker


def cf_endpoint(count):
    """Variable actual prompt populations; request macro != prompt micro."""
    cases, desired = [], {kind: [] for kind in ("rewrite", "paraphrase", "neighborhood")}
    for occurrence in range(1, count + 1):
        case = {}
        populations = dict(rewrite=1, paraphrase=2 + (occurrence == 1),
                           neighborhood=10 - (occurrence == 1))
        for kind, population in populations.items():
            observations, probs = [], []
            for prompt in range(population):
                preferred = (occurrence + prompt) % 4 != 0
                true, new = ((1.0, 2.0) if preferred else (2.0, 1.0)) if kind == "neighborhood" \
                    else ((2.0, 1.0) if preferred else (1.0, 2.0))
                token_count = 1 + prompt % 3
                token_correct = token_count if preferred else 0
                observations.append(dict(target_true=dict(mean_nll=true), target_new=dict(mean_nll=new)))
                probs.append(dict(target_true=true, target_new=new))
                desired[kind].append((token_count, token_correct, preferred))
            case[kind + "_observations"] = observations
            case[kind + "_prompts_probs"] = probs
        cases.append(case)
    accuracy = {}
    for kind, rows in desired.items():
        tokens, correct = sum(row[0] for row in rows), sum(row[1] for row in rows)
        accuracy[kind] = dict(token_acc_pct=100 * correct / tokens,
            prompt_acc_pct=100 * math.fsum(row[1] / row[0] for row in rows) / len(rows),
            strict_acc_pct=100 * sum(row[2] for row in rows) / len(rows))
    return dict(identity=dict(dataset="cf"), cases=cases,
                accuracy=accuracy, summary=counterfact(cases))


def zsre_endpoint(count):
    return dict(identity=dict(dataset="zsre"), cases=[], accuracy={},
        summary=dict(requests=count, Efficacy=81.0, Generalization=72.0,
                     Specificity=99.0, Specificity_loc_ans=49.0))


class TrackingBinding(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.reference_identity = "c" * 64
        reference = self.root / "reference-manifest.json"
        write_new(reference, dict(identity_sha256=self.reference_identity))
        assets = self.root / "assets.json"
        write_new(assets, dict(generation_reference=dict(manifest=common.member(reference))))
        self.env_path = str(self.root / "private-fixture-env-not-read")
        self.config = dict(assets_member=common.member(assets),
            tracking=dict(module="official.tracking.client", entity="wkdguswns2256",
                          project="layer allocation", attempt="official-fixture", env_file=self.env_path))
        self.identity = dict(code_commit="a" * 40, config_sha256="b" * 64)
        self.transport = SchemaOnlyTransport()

    def tracking(self, *, dataset="cf", method="MEMIT", mode="chain", config=None):
        with patch.object(common.importlib, "import_module", return_value=self.transport) as imported:
            value = common.Tracking(config or self.config, self.root, self.identity,
                                    mode=mode, method=method, dataset=dataset)
            imported.assert_called_once_with("official.tracking.client")
        return value

    def test_CF_native_generation_exact_binding_and_local_identity(self):
        tracker = self.tracking()
        call = self.transport.calls[-1]
        config = call["config"]
        expected = dict(generation_schedule="W0_AND_W20_FIRST2000",
            generation_metric_schema="counterfact-cake-generation-metrics-v1",
            generation_profile="cf-cake-native-casebatch-kv-total100-globalrng-v1",
            generation_eval_seed=20261007, reference_assets_sha256=self.reference_identity,
            generation_source_sha=self.identity["code_commit"],
            generation_repair_instruction="USER-OFFICIAL-BASELINES-20261008-R1")
        self.assertEqual({key: config[key] for key in expected}, expected)
        self.assertEqual(config["source_sha"], self.identity["code_commit"])
        self.assertEqual(config["config_sha"], self.identity["config_sha256"])
        self.assertEqual(config["metric_schema"], "official-baselines-scalar-v1")
        self.assertEqual(config["model"], "llama3")
        self.assertEqual(config["writer"], "memit")
        self.assertEqual(config["execution_backend"], "local")
        self.assertEqual(config["identity_source"], "NOT_APPLICABLE")
        self.assertFalse({"job_id", "array_job_id", "array_task_id", "step_id", "job_display_id"} & config.keys())
        self.assertTrue(schema.run_name(config).endswith("-local"))
        result = tracker.finish(exit_code=0)
        self.assertEqual(result["status"], "CPU_FIXTURE_NOT_ONLINE_VALIDATED")
        self.assertFalse(result["remote_verification"])

    def test_repeated_runtime_attempt_gets_unique_spool_and_attempt(self):
        self.tracking()
        self.tracking()
        left, right = self.transport.calls
        self.assertNotEqual(left["spool"], right["spool"])
        self.assertEqual(left["spool"].parent, self.root / "tracking")
        self.assertRegex(left["spool"].name, r"^[a-f0-9]{32}$")
        self.assertNotEqual(left["config"]["attempt"], right["config"]["attempt"])
        self.assertRegex(left["config"]["attempt"], r"^official-fixture-[a-f0-9]{12}$")

    def test_zsRE_omits_every_CF_generation_field_and_needs_no_reference(self):
        config = deepcopy(self.config)
        config.pop("assets_member")
        self.tracking(dataset="zsre", method="MEMIT_FE", config=config)
        cfg = self.transport.calls[-1]["config"]
        self.assertEqual(cfg["dataset"], "zsre")
        self.assertEqual(cfg["writer"], "memit_fe")
        self.assertFalse(any(key.startswith("generation_") or key == "reference_assets_sha256" for key in cfg))
        self.assertEqual(schema.config(cfg), cfg)

    def test_base_W0_identity_has_no_editor_or_fake_job(self):
        self.tracking(mode="base_w0", method=None)
        cfg = self.transport.calls[-1]["config"]
        self.assertEqual(cfg["writer"], "none")
        self.assertEqual(cfg["baseline"], "W0")
        self.assertEqual(cfg["arm"], "cf-base_w0")
        self.assertNotIn("job_id", cfg)

    def test_CF_actual_macro_and_nine_fields_variable_denominators_W0(self):
        tracker = self.tracking()
        endpoint = cf_endpoint(2000)
        payload = common.factual_payload(endpoint, "W0_first2000", 0)
        tracker.log(payload)
        actual = self.transport.calls[-1]["tracker"].payloads[-1]
        self.assertEqual(actual["official/W0_first2000/requests"], 2000)
        for kind, letter, count in (("rewrite", "R", 2000), ("paraphrase", "P", 4001),
                                   ("neighborhood", "N", 19999)):
            prefix = "W0_first2000/" + letter + "/"
            self.assertEqual({key.removeprefix(prefix) for key in actual if key.startswith(prefix)}, set(FIELDS))
            self.assertEqual(actual[prefix + "count"], count)
            self.assertAlmostEqual(actual[prefix + "success_pct"], 100 * actual[prefix + "success_count"] / count)
            self.assertEqual(actual[prefix + "token_acc_pct"], endpoint["accuracy"][kind]["token_acc_pct"])
            self.assertEqual(actual[prefix + "margin_true_minus_new"],
                             actual[prefix + "true_nll"] - actual[prefix + "new_nll"])
        # Official CF actual prompt populations must not inherit PRICE fixed P/N.
        with self.assertRaisesRegex(ValueError, "W0_EXACT_FIRST2000_COUNTS"):
            schema.metrics({key: value for key, value in payload.items() if not key.startswith("official/")},
                           scientific=True)
        self.assertNotEqual(actual["official/W0_first2000/Generalization"], actual["W0_first2000/P/success_pct"])
        self.assertEqual((actual["edits"], actual["pre_state_edits"], actual["post_state_edits"]), (0, 0, 0))

    def test_CF_actual_macro_and_nine_fields_measured_allseen500(self):
        tracker = self.tracking()
        endpoint = cf_endpoint(500)
        payload = common.factual_payload(endpoint, "all_seen/post", 500)
        tracker.log(payload)
        actual = self.transport.calls[-1]["tracker"].payloads[-1]
        self.assertEqual(actual["official/all_seen/post/requests"], 500)
        self.assertEqual(actual["all_seen/post/P/count"], 1001)
        self.assertEqual(actual["all_seen/post/N/count"], 4999)
        self.assertEqual(actual["post_state_edits"], 500)
        bad = dict(payload, **{"official/all_seen/post/Score": 0.0})
        with self.assertRaisesRegex(ValueError, "OFFICIAL_REQUEST_MACRO_SCORE_MISMATCH"):
            tracker.log(bad)

    def test_zsRE_actual_W0_and_milestone_macros_not_CF_NLL(self):
        tracker = self.tracking(dataset="zsre", method="FT")
        for prefix, edits, count in (("W0_first2000", 0, 2000), ("all_seen/post", 500, 500)):
            tracker.log(common.factual_payload(zsre_endpoint(count), prefix, edits))
        actual = self.transport.calls[-1]["tracker"].payloads[-1]
        self.assertEqual(actual["official/all_seen/post/Specificity"], 99.0)
        self.assertEqual(actual["official/all_seen/post/Specificity_loc_ans"], 49.0)
        self.assertFalse(any(re.search(r"/(?:R|P|N)/", key) for key in actual))
        self.assertNotIn("official/all_seen/post/Score", actual)

    def test_CF_complete_generation_W0_W20_actual_payload_shared_metric_output(self):
        from official.evaluation.generation.metrics import PUBLIC_REASONS
        tracker = self.tracking()
        summary = dict(planned_count=2000, fluency_count=1999, consistency_count=1998,
            generation_prompt_count=20000, generated_token_count=1800000,
            ngram_entropy=3.25, reference_score=.125,
            missing_reason_counts={reason: 0 for reason in PUBLIC_REASONS})
        for prefix, edits in (("W0_first2000", 0), ("all_seen/post", 2000)):
            payload = common.generation_payload(dict(summary=summary), edits, prefix=prefix)
            tracker.log(payload)
            self.assertEqual(payload[prefix + "/fluency/ngram_entropy"], 3.25)
            self.assertEqual(payload[prefix + "/consistency/reference_score"], .125)
            self.assertEqual(payload[prefix + "/generation/planned_count"], 2000)
            self.assertEqual(payload[prefix + "/generation/fluency_count"], 1999)
            self.assertEqual(payload[prefix + "/generation/consistency_count"], 1998)
        self.assertEqual(len(self.transport.calls[-1]["tracker"].payloads), 2)
        with self.assertRaisesRegex(ValueError, "OFFICIAL_GENERATION_ENDPOINT_ONLY"):
            tracker.log(common.generation_payload(dict(summary=summary), 100, prefix="current/post"))
        with self.assertRaisesRegex(ValueError, "OFFICIAL_GENERATION_EXACT_ENDPOINT"):
            tracker.log(common.generation_payload(dict(summary=summary), 500))

    def test_CF_missing_generation_omits_mean_and_never_invents_zero_score(self):
        from official.evaluation.generation.metrics import PUBLIC_REASONS
        tracker = self.tracking()
        summary = dict(planned_count=2000, fluency_count=0, consistency_count=0,
            generation_prompt_count=0, generated_token_count=0,
            missing_reason_counts={reason: 2000 if reason == "missing_generation_prompts" else 0
                                   for reason in PUBLIC_REASONS})
        payload = common.generation_payload(dict(summary=summary), 2000)
        tracker.log(payload)
        self.assertNotIn("all_seen/post/fluency/ngram_entropy", payload)
        self.assertNotIn("all_seen/post/consistency/reference_score", payload)
        self.assertEqual(payload["all_seen/post/generation/fluency_count"], 0)
        with self.assertRaisesRegex(ValueError, "GENERATION_MISSING_MEAN_COUNT"):
            tracker.log(dict(payload, **{"all_seen/post/fluency/ngram_entropy": 0.0}))

    def test_W20_native_progress_phase_mapping_is_separate_and_monotonic(self):
        tracker = self.tracking()
        for completed in (0, 32, 2000):
            raw = dict(phase="generation_evaluation", **{
                "generation_progress/step": completed,
                "generation_progress/completed_cases": completed,
                "generation_progress/total_cases": 2000,
                "generation_progress/completed_prompts": completed * 10,
                "generation_progress/total_prompts": 20000,
                "generation_progress/generated_tokens": completed * 900})
            tracker.log(official_generation_progress(raw, endpoint="W20"))
        points = self.transport.calls[-1]["tracker"].payloads
        self.assertEqual([row["phase"] for row in points], ["W20_generation"] * 3)
        self.assertEqual([row["generation_progress/step"] for row in points], [0, 32, 2000])
        self.assertTrue(all(not {"edits", "pre_state_edits", "post_state_edits"} & row.keys() for row in points))
        with self.assertRaisesRegex(ValueError, "AXIS_DECREASE"):
            tracker.log(dict(points[0]))
        with self.assertRaisesRegex(ValueError, "OFFICIAL_GENERATION_CALLBACK_ENDPOINT"):
            official_generation_progress(points[-1], endpoint="W5")

    def test_zsRE_generation_progress_is_a_blocker_not_silent_drop(self):
        tracker = self.tracking(dataset="zsre")
        values = official_generation_progress(dict(phase="generation_evaluation", **{
            "generation_progress/step": 1}), endpoint="W20")
        with self.assertRaisesRegex(ValueError, "ZSRE_GENERATION_FORBIDDEN"):
            tracker.log(values)
        self.assertEqual(self.transport.calls[-1]["tracker"].payloads, [])

    def test_schema_config_privacy_and_official_generation_changes_rejected(self):
        self.tracking()
        cfg = self.transport.calls[-1]["config"]
        for field in ("prompt", "token_ids", "argv", "WANDB_API_KEY", "full_environment"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "CONFIG_NOT_ALLOWLISTED"):
                schema.config(dict(cfg, **{field: "nonsecret-fixture"}))
        for changed in (dict(generation_schedule="W20_ONLY_FIRST2000"),
                        dict(generation_profile="cf-cake-prompt-inclusive-total100-eos-corrected-v1"),
                        dict(generation_eval_seed=0)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                schema.config(dict(cfg, **changed))
        self.assertEqual(self.transport.calls[-1]["env_file"], self.env_path)
        self.assertNotIn("env_file", cfg)
        self.assertFalse(any(str(self.root) in str(value) for value in cfg.values()))


if __name__ == "__main__":
    unittest.main()
