"""CPU-only fixed reference/schema/tokenizer controls; no model or network."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import assets


class AssetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="generation-assets-cpu-")
        self.root = Path(self.temp.name)
        self.source = self.root / "originals"
        self.source.mkdir()
        self.snippets = [
            dict(relation_id="P1", target_id="Q1", samples=[dict(text="alpha beta", name="not filtered")]),
            dict(relation_id="P1", target_id="Q1", samples=[dict(text="beta gamma", name="different subject")]),
            dict(relation_id="P2", target_id="Q2", samples=[])]
        self.vocabulary = dict(alpha=0, beta=1, gamma=2)
        self.idf = np.array([1., 2., 3.])
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        (self.source / "attribute_snippets.json").write_text(json.dumps(self.snippets))
        (self.source / "tfidf_vocab.json").write_text(json.dumps(self.vocabulary))
        np.save(self.source / "idf.npy", self.idf)

    def prepare(self, records=None):
        return assets.prepare_assets(self.source, self.root / "prepared", records)

    def test_fixed_saved_tfidf_public_setter_without_any_fit(self):
        with patch.object(TfidfVectorizer, "fit", side_effect=AssertionError("NO_REFIT")), \
             patch.object(TfidfVectorizer, "fit_transform", side_effect=AssertionError("NO_REFIT")):
            vectorizer = assets.fixed_vectorizer(self.vocabulary, self.idf)
            result = vectorizer.transform(["alpha beta beta", "gamma"]).toarray()
        expected = np.array([1., 4., 0.]); expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result[0], expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(result[1], [0., 0., 1.])
        np.testing.assert_array_equal(vectorizer.idf_, self.idf)

    def test_stream_original_list_order_and_all_exact_pair_samples(self):
        result, schema = assets.read_snippets(self.source / "attribute_snippets.json")
        self.assertEqual(result[("P1", "Q1")], ["alpha beta", "beta gamma"])
        self.assertEqual(schema["samples"], 2)
        self.assertEqual(schema["entries"], 3)
        items = list(assets._array_items(self.source / "attribute_snippets.json", chunk_chars=7))
        self.assertEqual(items, self.snippets)

    def test_stream_rejects_truncation_separator_trailing_data_and_duplicate_key(self):
        path = self.root / "invalid.json"
        for text in ('[{"a":"' + "z" * 100, '[{"a":1} {"a":2}]', '[{"a":1},]',
                     '[{"a":1}] true', '[{"a":1,"a":2}]'):
            path.write_text(text)
            with self.assertRaises(assets.AssetError):
                list(assets._array_items(path, chunk_chars=7))

    def test_snippet_schema_not_subject_filtered_or_silently_dropped(self):
        self.snippets[0]["samples"][0]["text"] = 123
        self.write()
        with self.assertRaisesRegex(assets.AssetError, "SAMPLE_TEXT"):
            assets.read_snippets(self.source / "attribute_snippets.json")

    def test_dense_indices_duplicate_bool_negative_or_gap_rejected(self):
        for vocab in (dict(alpha=0, beta=0), dict(alpha=True), dict(alpha=-1), dict(alpha=1)):
            self.vocabulary = vocab
            self.write()
            with self.assertRaises(assets.AssetError):
                assets.read_tfidf(self.source / "idf.npy", self.source / "tfidf_vocab.json")

    def test_idf_nonfinite_shape_length_and_pickle_rejected(self):
        for value in (np.array([1., np.nan, 2.]), np.ones((1, 3)), np.ones(2),
                      np.array([object()], dtype=object)):
            self.idf = value
            self.write()
            with self.assertRaises(assets.AssetError):
                assets.read_tfidf(self.source / "idf.npy", self.source / "tfidf_vocab.json")

    def test_ready_manifest_load_and_local_coverage_do_not_invent_scores(self):
        records = [dict(case_id=999, requested_rewrite=dict(relation_id="P1", target_new={"id":"Q1"}),
                        generation_prompts=["synthetic"]),
                   dict(case_id=999, requested_rewrite=dict(relation_id="P9", target_new={"id":"Q9"}),
                        generation_prompts=[])]
        manifest = self.prepare(records)
        loaded = assets.load_assets(manifest)
        self.assertEqual(loaded.snippets_for("P1", "Q1"), ["alpha beta", "beta gamma"])
        self.assertEqual(loaded.reference_for("P9", "Q9"), dict(texts=[], reason="missing_reference"))
        self.assertEqual(loaded.snippets_for(1, "Q1"), [])
        coverage = loaded.coverage(records)["summary"]
        self.assertEqual(coverage["planned_count"], 2) # occurrences, not case-ID dedup
        self.assertEqual(coverage["reference_available_count"], 1)
        self.assertEqual(coverage["missing_reference_count"], 1)
        self.assertEqual(coverage["missing_generation_prompts_count"], 1)
        self.assertTrue(coverage["coverage_not_metric_validity"])
        ready = assets.json_read(manifest.parent / "READY.json")
        self.assertEqual(ready["identity_sha256"], loaded.sha)
        self.assertEqual(loaded.word_tokenize("A sentence. Two words!"),
                         ["A", "sentence", ".", "Two", "words", "!"])
        self.assertFalse(loaded.manifest["downloads_by_loader"])
        self.assertEqual(loaded.manifest["fixed_vectorizer"]["fit_calls"], 0)
        compact = assets.json_read(manifest)
        self.assertNotIn("rows", compact["coverage"])
        self.assertNotIn("samples", compact)

    def test_create_once_failure_and_corrupt_file_identity(self):
        manifest = self.prepare()
        with self.assertRaisesRegex(assets.AssetError, "CREATE_ONCE"):
            self.prepare()
        (self.source / "tfidf_vocab.json").write_text('{"other":0}')
        with self.assertRaisesRegex(assets.AssetError, "MEMBER_IDENTITY"):
            assets.load_assets(manifest)

    def test_manifest_and_runtime_tampering_blocked(self):
        manifest = assets.json_read(self.prepare())
        bad = copy.deepcopy(manifest)
        bad["files"]["idf.npy"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(assets.AssetError, "MANIFEST_IDENTITY"):
            assets.load_assets(bad)
        with patch.object(assets, "dependency_versions", return_value={}):
            with self.assertRaisesRegex(assets.AssetError, "RUNTIME_VERSIONS"):
                assets.load_assets(manifest)

    def test_parent_config_binding_and_peer_asset_paths(self):
        manifest = self.prepare()
        loaded = assets.load_assets(dict(generation_assets=assets.member(manifest)))
        self.assertEqual(loaded.sha, assets.json_read(manifest)["identity_sha256"])
        value = assets.json_read(manifest)
        # Absolute source path can differ on a peer; bytes/SHA remain authority.
        value["files"]["idf.npy"]["path"] = "/nonexistent/server1-alias"
        peer = assets.load_assets(dict(manifest=value, asset_paths={"idf.npy":str(self.source / "idf.npy")}))
        self.assertEqual(peer.sha, loaded.sha)

    def test_download_byte_plan_mismatch_typed_unavailable(self):
        with self.assertRaisesRegex(assets.AssetError, "DOWNLOAD_SIZES") as raised:
            assets.prepare_assets(self.source, self.root / "prepared", expected_sizes=dict(zip(assets.FILES, [1, 1, 1])))
        self.assertEqual(raised.exception.status, "ASSET_NOT_AVAILABLE")
        self.assertEqual(raised.exception.reason, "asset_not_available")

    def test_nltk_active_resource_bound_and_missing_not_regex_fallback(self):
        binding = assets.nltk_binding()
        self.assertEqual(binding["language"], "english")
        self.assertTrue(binding["required_resources"])
        self.assertFalse(binding["replacement_regex"])
        with patch("nltk.data.find", side_effect=LookupError("fixture")):
            with self.assertRaisesRegex(assets.AssetError, "TOKENIZER_RESOURCE") as raised:
                assets.nltk_binding()
        self.assertEqual(raised.exception.reason, "tokenizer_not_available")


if __name__ == "__main__":
    unittest.main()
