"""CPU fixtures for W0 content/execution separation; no model or scheduler calls."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from official.runners.server1 import w0_binding as binding


def _member(path, sha="a" * 64, size=19):
    return dict(path=path, bytes=size, sha256=sha)


def _assets(root="/cpu-fixture/one"):
    return dict(
        model=dict(identity=dict(model_id="meta-llama/Meta-Llama-3-8B", revision="a" * 40),
            config=_member(root + "/model/config.json", "b" * 64),
            weights=[_member(root + "/model/model.safetensors", "c" * 64)],
            tokenizer_files={"tokenizer.json": _member(root + "/model/tokenizer.json", "d" * 64)}),
        streams={name: dict(stream_sha256=("e" if name == "cf" else "f") * 64)
                 for name in ("cf", "zsre")},
        generation_reference=dict(identity_sha256="1" * 64,
            nltk_data_root=root + "/nltk_data",
            tokenizer_resources_and_sources=[
                _member(root + "/nltk_data/tokenizers/punkt_tab/english/collocations.tab", "2" * 64),
                _member("/cpu-fixture/nltk/tokenize/punkt.py", "3" * 64)],
            manifest=_member(root + "/refs/manifest.json"),
            files={name: _member(root + "/refs/" + name, "4" * 64)
                   for name in ("attribute_snippets.json", "idf.npy", "tfidf_vocab.json")}),
        runtime={"source_path": root + "/runtime", "hostname": "CPU_FIXTURE_HOST"})


def _records():
    return {name: [dict(occurrence_index=i) for i in range(1, 2001)]
            for name in ("cf", "zsre")}


def _execution_config(root="/cpu-fixture/one"):
    return dict(config_sha256="5" * 64, assets_member=_member(root + "/assets.json"),
        stream_bundle_member=_member(root + "/stream.json"), base_W0_output=root + "/W0")


class W0BindingTests(unittest.TestCase):
    def build(self, assets=None, records=None, *, padding="right", version="CPU_FIXTURE_VERSION",
              source_change=None, query_change=None):
        """Genuine pure fingerprint validator, mocked token planning and package metadata."""
        tokenizer = SimpleNamespace(padding_side=padding)
        def plan(rows, name, actual_tokenizer):
            self.assertIs(actual_tokenizer, tokenizer)
            return (None, None, [dict(dataset=name, occurrence_index=r["occurrence_index"],
                                     query=query_change or "FIXED_CPU_QUERY") for r in rows])
        def source_sha(path):
            logical = str(path).split("/evaluation/", 1)[-1]
            return hashlib.sha256((logical + (source_change or "")).encode()).hexdigest()
        with ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"nltk": SimpleNamespace(
                __file__="/cpu-fixture/nltk/__init__.py")}))
            planner = stack.enter_context(patch.object(binding, "_plan", side_effect=plan))
            stack.enter_context(patch.object(binding, "read", return_value={
                "fixed_vectorizer": {"class_name": "sklearn.feature_extraction.text.TfidfVectorizer"}}))
            stack.enter_context(patch.object(binding, "file_sha", side_effect=source_sha))
            stack.enter_context(patch.object(binding.platform, "python_version", return_value="3.fixture"))
            stack.enter_context(patch.object(binding.importlib.metadata, "version", return_value=version))
            value = binding.build_fingerprint(assets or _assets(), tokenizer, records or _records())
        self.assertEqual(planner.call_count, 2)
        return value

    def execution(self, *, config=None, lock=None, assets=None, output="/cpu-fixture/one/out",
                  env=None, hardware="CPU_FIXTURE_GPU_NAME"):
        fake_cuda = SimpleNamespace(get_device_name=Mock(return_value=hardware),
                                    get_device_capability=Mock(return_value=(8, 0)))
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=fake_cuda)}), \
             patch.dict("os.environ", env if env is not None else {"SLURM_JOB_ID": "12345"}, clear=True):
            result = binding.execution_identity(config or _execution_config(),
                lock or dict(source=dict(main_commit="a" * 40, official_tree="b" * 40)),
                assets or _assets(), output=output)
        return result

    def test_relocated_asset_paths_source_publication_and_hardware_not_computational_content(self):
        left, right = _assets(), _assets("/cpu-fixture/two")
        right["runtime"] = dict(source_path="/different/publication", hostname="ANOTHER_HOST")
        left_fp, right_fp = self.build(left), self.build(right)
        self.assertEqual(left_fp, right_fp)
        first_execution = self.execution(assets=left)
        other_execution = self.execution(config=_execution_config("/cpu-fixture/two"), assets=right,
            lock=dict(source=dict(main_commit="c" * 40, official_tree="d" * 40)),
            output="/cpu-fixture/two/out", hardware="OTHER_CPU_FIXTURE_GPU")
        self.assertNotEqual(first_execution, other_execution)
        self.assertFalse(first_execution["hardware"]["cross_hardware_bitwise_claim"])
        self.assertNotIn("hardware", left_fp["content"])
        self.assertNotIn("source_path", json.dumps(left_fp))
        self.assertNotIn("/cpu-fixture/", json.dumps(left_fp))

    def test_consumed_payload_revision_tokenizer_stream_and_reference_changes_change_fingerprint(self):
        original = self.build()
        for field in ("weight", "revision", "tokenizer", "stream", "reference", "resource"):
            changed = deepcopy(_assets())
            if field == "weight":
                changed["model"]["weights"][0]["sha256"] = "9" * 64
            elif field == "revision":
                changed["model"]["identity"]["revision"] = "9" * 40
            elif field == "tokenizer":
                changed["model"]["tokenizer_files"]["tokenizer.json"]["bytes"] += 1
            elif field == "stream":
                changed["streams"]["cf"]["stream_sha256"] = "9" * 64
            elif field == "reference":
                changed["generation_reference"]["files"]["idf.npy"]["sha256"] = "9" * 64
            else:
                changed["generation_reference"]["tokenizer_resources_and_sources"][0]["sha256"] = "9" * 64
            with self.subTest(field=field):
                self.assertNotEqual(original["sha256"], self.build(changed)["sha256"])

    def test_consumed_source_runtime_versions_and_actual_token_plan_are_computational(self):
        original = self.build()
        for kwargs in (dict(source_change="SOURCE_CHANGE"), dict(version="DIFFERENT_VERSION"),
                       dict(query_change="ACTUAL_OTHER_TOKEN_PLAN")):
            with self.subTest(kwargs=kwargs):
                self.assertNotEqual(original["sha256"], self.build(**kwargs)["sha256"])

    def test_real_consumed_source_members_are_current_content_not_full_repository_tree(self):
        rows = binding.consumed_source_members()
        expected = {"factual.py", "reduce.py", "w0_reference.py"} | {
            "generation/" + name for name in binding.GENERATION_CLOSURE}
        self.assertEqual(set(rows), expected)
        for name, row in rows.items():
            with self.subTest(name=name):
                path = Path(row["path"])
                self.assertEqual(row["bytes"], path.stat().st_size)
                self.assertEqual(row["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertTrue(path.is_relative_to(Path(binding.__file__).resolve().parents[2] / "evaluation"))

    def test_exact2000_occurrence_order_required_without_case_dedup_or_threshold(self):
        for kind in ("short", "duplicate", "reverse"):
            records = _records()
            if kind == "short":
                records["cf"].pop()
            elif kind == "duplicate":
                records["cf"][1]["occurrence_index"] = 1
            else:
                records["cf"].reverse()
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "EXACT_ORDERED_STREAM"):
                self.build(records=records)

    def test_right_padding_and_nltk_consumed_source_scope_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "RIGHT_PADDING_REQUIRED"):
            self.build(padding="left")
        assets = _assets()
        assets["generation_reference"]["tokenizer_resources_and_sources"][1]["path"] = "/unapproved/private.py"
        with self.assertRaisesRegex(ValueError, "NLTK_CONSUMED_SOURCE_SCOPE"):
            self.build(assets)

    def test_content_member_drops_path_and_metadata_and_rejects_missing_size_or_sha(self):
        value = dict(_member("/not-read/private-path"), producer_commit="a" * 40, private="NOT_INCLUDED")
        self.assertEqual(binding.content_member(value), {"bytes": 19, "sha256": "a" * 64})
        for bad in (dict(bytes=0, sha256="a" * 64), dict(bytes=True, sha256="a" * 64),
                    dict(bytes=1), dict(bytes="19", sha256="a" * 64)):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "CONSUMED_CONTENT_SHA_REQUIRED"):
                binding.content_member(bad)

    def test_fingerprint_real_validator_rejects_malformed_member_sha(self):
        assets = _assets()
        assets["model"]["weights"][0]["sha256"] = "NOT_A_SHA"
        with self.assertRaisesRegex(ValueError, "MODEL.WEIGHTS.SHA256"):
            self.build(assets)

    def test_actual_slurm_parent_required_no_local_dummy_job(self):
        for env in ({}, {"SLURM_JOB_ID": ""}, {"SLURM_ARRAY_JOB_ID": "12345", "SLURM_ARRAY_TASK_ID": "0"}):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, "ACTUAL_SLURM_ID_REQUIRED"):
                self.execution(env=env)

    def test_execution_reads_only_slurm_whitelist_preserves_array_zero_signed_step(self):
        env = dict(SLURM_JOB_ID="12346", SLURM_ARRAY_JOB_ID="12345", SLURM_ARRAY_TASK_ID="0",
                   SLURM_STEP_ID="-1", WANDB_API_KEY="DO_NOT_PUBLISH_SECRET", PRIVATE_FULL_ENV="PRIVATE")
        result = self.execution(env=env)
        self.assertEqual(result["slurm"], {key: env[key] for key in (
            "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_STEP_ID")})
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_malformed_parent_or_incomplete_array_identity_rejected(self):
        for env in (dict(SLURM_JOB_ID="UNKNOWN"), dict(SLURM_JOB_ID="0"),
                    dict(SLURM_JOB_ID="42", SLURM_ARRAY_TASK_ID="0"),
                    dict(SLURM_JOB_ID="42", SLURM_ARRAY_JOB_ID="41"),
                    dict(SLURM_JOB_ID="42", SLURM_STEP_ID="private step")):
            with self.subTest(env=env), self.assertRaises(ValueError):
                self.execution(env=env)

    def test_execution_identity_keeps_producer_source_config_inputs_output_and_hardware_separate(self):
        result = self.execution()
        self.assertEqual(result["source"], dict(main_commit="a" * 40, official_tree="b" * 40))
        self.assertEqual(result["config_sha256"], "5" * 64)
        self.assertEqual(result["assets_manifest"], _execution_config()["assets_member"])
        self.assertEqual(result["input_stream_bundle"], _execution_config()["stream_bundle_member"])
        self.assertEqual(result["base_W0_input"], _execution_config()["base_W0_output"])
        self.assertEqual(result["hardware"]["capability"], [8, 0])


if __name__ == "__main__":
    unittest.main()
