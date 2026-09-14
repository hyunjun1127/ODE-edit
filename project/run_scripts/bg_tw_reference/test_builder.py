"""Small deterministic CPU fixtures, never a real C4/teacher/G0 readiness claim."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from . import builder as b


CONTRACT_PATH = Path(__file__).resolve().parents[3] / "plans/global/2026-09-15-bg-tw-reference-data-contract.json"


class TinyTokenizer:
    """Exactly reversible character fixture, not the actual model tokenizer."""
    bos_token_id = 1
    all_special_ids = [1]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) + 100 for char in text]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token - 100) for token in ids)


def contract():
    return json.loads(CONTRACT_PATH.read_text())


def row(i, text=None, url=None, split="train"):
    sid = f"allenai/c4@{b.REVISION}|{split}|fixture|{i}"
    return b.RankedRow(sid, b.priority_hash(sid, "document-priority"), i, split,
                      {"text": text or (f"unique{i} " * 50),
                       "url": url or f"https://fixture{i}.test/path", "timestamp": "fixture"})


class BuilderTests(unittest.TestCase):
    def test_exact_contract_and_drift_rejected(self):
        c = contract()
        b.validate_contract(c)
        for key, value in [("seed", 42)]:
            changed = copy.deepcopy(c)
            changed["sampling"][key] = value
            with self.assertRaises(b.BuildError):
                b.validate_contract(changed)
        changed = copy.deepcopy(c)
        changed["tokenization"]["score_logits_indices_half_open"] = [129, 257]
        with self.assertRaises(b.BuildError):
            b.validate_contract(changed)

    def test_url_normalization_preserves_query_scheme_path(self):
        self.assertEqual(b.canonical_url("HTTPS://BÜCHER.Example:443/A?x=1#frag"),
                         "https://xn--bcher-kva.example/A?x=1")
        self.assertEqual(b.canonical_url("http://X.test:80"), "http://x.test/")
        self.assertEqual(b.canonical_url("http://X.test:443/a"), "http://x.test:443/a")
        self.assertEqual(b.canonical_url("http://[2001:db8::1]:80"), "http://[2001:db8::1]/")
        with self.assertRaises(ValueError):
            b.canonical_url("ftp://x.test/a")
        with self.assertRaises(ValueError):
            b.canonical_url("https://x.test:invalid/a")

    def test_nfkc_casefold_whitespace_only_for_comparison(self):
        raw = " ＡＢＣ\tStraße\n DEF "
        self.assertEqual(b.normalized(raw), "abc strasse def")
        d, reason = b.tokenize_candidate(row(0, raw * 30), TinyTokenizer(), contract())
        self.assertIsNone(reason)
        self.assertEqual(d.metadata["raw_text"], raw * 30)

    def test_whole_shard_priority_not_prefix_and_original_row_id(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "fixture.json.gz"
            rows = [dict(text=str(i), url=f"https://x.test/{i}", timestamp="") for i in range(80)]
            # This malformed line still occupies original JSONL row index 1.
            lines = [json.dumps(rows[0]), "{broken", *map(json.dumps, rows[1:])]
            payload = ("\n".join(lines) + "\n").encode()
            p.write_bytes(gzip.compress(payload, mtime=0))
            spec = {"split": "train", "path": "fixture.gz",
                    "expected_bytes_from_content_range": p.stat().st_size}
            corpus = {"dataset": "allenai/c4", "revision": b.REVISION, "config": "en"}
            actual, receipt = b.scan_shard(p, spec, retain=7, corpus=corpus)
            ids = [f"allenai/c4@{b.REVISION}|train|fixture.gz|{i}"
                   for i in range(81) if i != 1]
            expected = sorted(ids, key=lambda sid: (b.priority_hash(sid, "document-priority"), sid))[:7]
            self.assertEqual([x.source_row_id for x in actual], expected)
            self.assertTrue(any(x.row_index > 7 for x in actual))
            self.assertEqual(receipt["row_counts"]["jsonl_rows"], 81)
            self.assertEqual(receipt["row_counts"]["invalid_json"], 1)
            self.assertEqual(receipt["gzip_crc_validation"], "PASS_ALL_MEMBERS_TO_EOF")
            self.assertEqual(receipt["uncompressed_bytes"], len(payload))

    def test_complete_gzip_crc_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "bad.gz"
            payload = bytearray(gzip.compress(b'{"text":"fixture"}\n', mtime=0))
            payload[-8] ^= 1
            p.write_bytes(payload)
            with self.assertRaises(gzip.BadGzipFile):
                b.scan_shard(p, {"split": "train", "path": "fixture"}, retain=1,
                             corpus={"dataset": "allenai/c4", "config": "en", "revision": b.REVISION})

    def test_exact_token_window_and_shift_no_reencoding(self):
        item = row(42, "abcdefghijklmnopqrstuvwxyz " * 30)
        d, reason = b.tokenize_candidate(item, TinyTokenizer(), contract())
        self.assertIsNone(reason)
        tokens = TinyTokenizer().encode(item.row["text"], add_special_tokens=False)
        expected_hash = hashlib.sha256((f"C4-WebRef-v2|20260915|window|{item.source_row_id}|"
                                         + b.sha_text(item.row["text"])).encode()).hexdigest()
        start = int(expected_hash, 16) % (len(tokens) - 255)
        self.assertEqual(d.input_ids, [1] + tokens[start:start + 256])
        self.assertEqual(len(d.input_ids[129:257]), 128)
        self.assertEqual(d.metadata["window_start"], start)
        self.assertEqual(len(d.input_ids), 257)

    def test_near_duplicate_exact_threshold_and_empty(self):
        common = " ".join(f"word{i}" for i in range(200))
        a, c = b.Fingerprint.make(common), b.Fingerprint.make(common + " added")
        self.assertEqual(a.duplicate_reason(c), "near_full_document")
        self.assertFalse(b.near_duplicate(frozenset(), frozenset()))
        self.assertTrue(b.near_duplicate(frozenset(range(4)), frozenset(range(5))))
        self.assertFalse(b.near_duplicate(frozenset(range(3)), frozenset(range(5))))
        self.assertEqual(b.Fingerprint.make("tiny").duplicate_reason(b.Fingerprint.make("TINY")),
                         "normalized_full_document")

    def test_evaluator_input_interface_rejects_answers_scores(self):
        for extra in ({"answers": []}, {"future_subjects": []}, {"loss": 0}):
            with self.assertRaises(b.BuildError):
                b.OverlapChecker(dict(schema_version=1, **extra))
        with self.assertRaises(b.BuildError):
            b.OverlapChecker({"schema_version": 1, "evaluator_inputs": [
                {"id": "x", "scope": "fixture", "text": "hello", "label": "positive"}]})

    def test_overlap_diagnostic_does_not_filter_but_eval_does(self):
        item = row(1)
        reference = {"id": "x", "scope": "fixture", "text": item.row["text"]}
        diag = b.OverlapChecker({"schema_version": 1, "diagnostic_inputs": [reference]})
        selected, _ = b.select_documents([item], TinyTokenizer(), contract(), needed=1,
                                         initial_limit=1, overlap=diag)
        self.assertEqual(len(selected), 1)
        evaluator = b.OverlapChecker({"schema_version": 1, "evaluator_inputs": [reference]})
        selected, counts = b.select_documents([item], TinyTokenizer(), contract(), needed=1,
                                              initial_limit=1, overlap=evaluator)
        self.assertEqual(len(selected), 0)
        self.assertEqual(counts["counts"]["available_evaluator_input_overlap"], 1)

    def test_single_extension_and_bank_wins_cross_split(self):
        items = [row(0, "short"), row(1), row(2), row(3)]
        selected, counts = b.select_documents(items, TinyTokenizer(), contract(), needed=2,
                                              initial_limit=2)
        self.assertTrue(counts["extension_used"])
        self.assertEqual(len(selected), 2)
        report_rows = [row(5, text=selected[0].metadata["raw_text"], split="validation"),
                       row(7, split="validation")]
        report, counts = b.select_documents(report_rows, TinyTokenizer(), contract(), needed=1,
                                             initial_limit=2, previous=selected)
        self.assertEqual(report[0].metadata["original_jsonl_row_index"], 7)
        self.assertEqual(counts["counts"]["selected_duplicate_normalized_full_document"], 1)

    def test_same_domain_allowed_wikipedia_tag_only(self):
        rows = [row(1, url="https://en.wikipedia.org/wiki/A"),
                row(2, url="https://en.wikipedia.org/wiki/B")]
        selected, _ = b.select_documents(rows, TinyTokenizer(), contract(), needed=2,
                                         initial_limit=2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].metadata["wiki_tags"]["domains"], ["wikipedia.org"])

    def test_psl_private_wildcards_and_exceptions(self):
        psl = b.PublicSuffixList("com\nco.uk\n*.ck\n!www.ck\n// PRIVATE\nblogspot.com\n")
        self.assertEqual(psl.registrable_domain("a.b.example.co.uk"), "example.co.uk")
        self.assertEqual(psl.registrable_domain("x.example.blogspot.com"), "example.blogspot.com")
        self.assertEqual(psl.registrable_domain("x.a.ck"), "x.a.ck")
        self.assertEqual(psl.registrable_domain("x.www.ck"), "www.ck")
        self.assertEqual(psl.registrable_domain("127.0.0.1"), "127.0.0.1")

    def test_role_hash_not_priority_and_counts(self):
        docs = []
        for i in range(768):
            item = row(i, split="train" if i < 512 else "validation")
            d, _ = b.tokenize_candidate(item, TinyTokenizer(), contract())
            docs.append(d)
        actual = b.assign_roles(docs[:512], docs[512:])
        expected = sorted(docs[:512], key=lambda d: (d.metadata["bank_role_hash"],
                                                    d.metadata["source_row_id"]))
        self.assertEqual(actual[:512], expected)
        for role, count in zip(b.ROLES, b.COUNTS):
            self.assertEqual(sum(d.metadata["split_role"] == role for d in actual), count)
        self.assertEqual([d.metadata["row_ordinal"] for d in actual], list(range(768)))

    def test_post_all_pairs_includes_cross_split(self):
        d, _ = b.tokenize_candidate(row(1), TinyTokenizer(), contract())
        e, _ = b.tokenize_candidate(row(2), TinyTokenizer(), contract())
        self.assertEqual(b.all_pairs_audit([d, e])["all_pairs_checked"], 1)
        with self.assertRaises(b.BuildError):
            b.all_pairs_audit([d, d])

    def test_npz_is_pickle_free_byte_reproducible_create_once(self):
        with tempfile.TemporaryDirectory() as directory:
            a, c = Path(directory) / "a.npz", Path(directory) / "c.npz"
            arrays = {"ids": np.arange(257, dtype="<i8"), "roles": np.array(["S64", "Dev128"])}
            b.write_npz_once(a, arrays)
            b.write_npz_once(c, arrays)
            self.assertEqual(a.read_bytes(), c.read_bytes())
            with np.load(a, allow_pickle=False) as loaded:
                self.assertTrue(np.array_equal(loaded["ids"], arrays["ids"]))
            with self.assertRaises(FileExistsError):
                b.write_npz_once(a, arrays)


if __name__ == "__main__":
    unittest.main()
