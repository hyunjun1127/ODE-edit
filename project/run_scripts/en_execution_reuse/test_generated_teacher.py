"""Synthetic CPU-only schema/coverage tests; no Llama/model validation claim."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .generated_teacher import (
    CAPSULE_SCHEMA, CPUFixture, CoverageTracker, DATA_ID, DocumentMeanAccumulator,
    GeneratedTeacherError, GeneratedTeacherStore, ROLE_COUNTS, SCHEMA,
    canonical_sha256, file_sha256, generation_policy, make_capsule, token_sha256,
    validate_binding, validate_capsule, validate_inputs,
)


def binding():
    return dict(source_sha256="1" * 64, model_revision="cpu-fixture-model",
                model_config_sha256="2" * 64, model_weights_sha256="3" * 64,
                tokenizer_revision="cpu-fixture-tokenizer", tokenizer_sha256="4" * 64,
                w0_sha256="5" * 64, bos_token_id=1, generation=generation_policy([0]),
                mask_policy="all_ones_int64", position_policy="contiguous_zero_based_int64",
                teacher_policy="canonical_W0_TF_FP32_full_vocab_log_softmax",
                selected_parameter="model.layers.4.mlp.down_proj.weight",
                runtime=dict(device="CPU_SYNTHETIC_FIXTURE", dtype="float32", attention="fixture"))


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def descriptor(root, path, array=None):
    result = dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=file_sha256(path))
    if array is not None:
        result.update(shape=list(array.shape), dtype="float32")
    return result


def fixture(root):
    """All 640 members, small vocabulary, distinct prompts, no mocked loader."""
    model_binding, rows, caps, documents = binding(), [], [], []
    for index in range(640):
        role, ordinal = ("R512", index) if index < 512 else ("Dev128", index - 512)
        prompt = [1] + [2 + index // 6**j % 6 for j in range(4)] + [2] * 124
        row = dict(role=role, ordinal=ordinal, source_row_id=f"cpu-document-{index}",
                   input_ids=prompt, prompt_token_sha256=token_sha256(prompt),
                   window_token_sha256="6" * 64,
                   source_role="S64" if index < 64 else "Reserve320" if index < 384 else
                   "AdditionalTrain128" if index < 512 else "Dev128",
                   source_text_sha256="7" * 64, window_start=0)
        rows.append(row)
        length = (1, 9, 17, 256, 256)[index % 5]
        y0 = [3] * length
        if index % 5 != 3:
            y0[-1] = 0
        cap = make_capsule(row, y0, model_binding)
        caps.append(cap)
        folder = root / role / str(ordinal)
        folder.mkdir(parents=True)
        cap_path = folder / "capsule.json"
        write_json(cap_path, cap)
        member = dict(index=index, role=role, ordinal=ordinal, source_row_id=row["source_row_id"],
                      capsule=descriptor(root, cap_path))
        logits = np.full((length, 8), -2.0, dtype=np.float64)
        logits[np.arange(length), y0] = 0
        logp = (logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))).astype(np.float32)
        for kind, array in (("logp", logp), ("keys", np.zeros((128 + length, 3), dtype=np.float32)),
                            ("residual", np.zeros((128 + length, 2), dtype=np.float32))):
            path = folder / (kind + ".npy")
            np.save(path, array, allow_pickle=False)
            member[kind] = descriptor(root, path, array)
        documents.append(member)
    inputs_path = root / "inputs.json"
    write_json(inputs_path, rows)
    manifest = dict(schema=SCHEMA, data_id=DATA_ID, status="COMPLETE", production_ready=False,
                    vocabulary_size=8, key_size=3, hidden_size=2,
                    inputs_sha256=file_sha256(inputs_path), binding=model_binding,
                    binding_sha256=canonical_sha256(model_binding), documents=documents,
                    document_counts=dict(ROLE_COUNTS), upstream_cache_status="COMPLETE",
                    position_counts={role: sum(c["actual_length"] for c in caps if c["role"] == role)
                                     for role in ROLE_COUNTS})
    write_json(root / "manifest.json", manifest)
    return rows, caps, manifest


class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="en-g256-cpu-fixture-")
        cls.root = Path(cls.temporary.name)
        cls.rows, cls.caps, cls.manifest = fixture(cls.root)
        cls.cpu_fixture = CPUFixture(8, 3, 2, cls.manifest["inputs_sha256"])
        cls.sealed_store = cls.open_store()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def open_store(cls, **kwargs):
        options = dict(expected_manifest_sha256=file_sha256(cls.root / "manifest.json"),
                       inputs_path=cls.root / "inputs.json", expected_binding=binding(),
                       cpu_fixture=cls.cpu_fixture)
        options.update(kwargs)
        return GeneratedTeacherStore(cls.root, cls.root / "manifest.json", **options)

    @contextmanager
    def changed_manifest(self, modify):
        path = self.root / "manifest.json"
        original = path.read_bytes()
        changed = deepcopy(self.manifest)
        modify(changed)
        write_json(path, changed)
        try:
            yield changed
        finally:
            path.write_bytes(original)

    @contextmanager
    def changed_payload(self, index, kind, change, *, reseal=False):
        path = self.root / self.manifest["documents"][index][kind]["path"]
        original = path.read_bytes()
        try:
            array = np.load(path, allow_pickle=False)
            updated = change(array)
            np.save(path, updated, allow_pickle=False)
            if reseal:
                def update(manifest):
                    manifest["documents"][index][kind] = descriptor(self.root, path, updated)
                with self.changed_manifest(update):
                    yield
            else:
                yield
        finally:
            path.write_bytes(original)

    def test_full_640_readonly_mmap_and_ragged_shapes(self):
        store = self.sealed_store
        self.assertEqual(store.indices("R512"), tuple(range(512)))
        self.assertEqual(store.indices("Dev128"), tuple(range(512, 640)))
        self.assertFalse(store.production_ready)
        self.assertFalse(store.receipt["model_execution_performed"])
        self.assertTrue(store.receipt["canonical_tf_argmax_verified"])
        for index, expected in ((0, 1), (1, 9), (2, 17), (3, 256), (4, 256)):
            with store.document(index) as doc:
                self.assertIsInstance(doc.logp, np.memmap)
                self.assertFalse(doc.logp.flags.writeable)
                self.assertTrue(doc.canonical_tf_argmax_verified)
                self.assertEqual(doc.logp.shape, (expected, 8))
                self.assertEqual(doc.keys.shape, (128 + expected, 3))
                self.assertEqual(len(doc.capsule["input_ids"]), 129)
                self.assertEqual(len(doc.capsule["tf_input_ids"]), 129 + expected - 1)
                self.assertEqual(doc.capsule["score_positions"], list(range(128, 128 + expected)))
                self.assertEqual(sum(len(positions) for positions, _ in doc.chunks(16)), expected)
                with self.assertRaises(ValueError):
                    doc.logp[0, 0] = 0
        self.assertEqual(len(store.capsule(3)["input_ids"] + store.capsule(3)["y0"]), 385)
        self.assertEqual(store.capsule(3)["tf_input_ids"][-1], 3)
        self.assertTrue(store.capsule(3)["length_censored"])
        self.assertFalse(store.capsule(4)["length_censored"])
        self.assertEqual(store.capsule(4)["stop_reason"], "eos")

    def test_runtime_skip_never_hashes_or_scans_payloads_and_opens_only_logp(self):
        module = "project.run_scripts.en_execution_reuse.generated_teacher"
        with patch(module + ".file_sha256", side_effect=AssertionError("runtime SHA")), \
             patch(module + ".np.isfinite", side_effect=AssertionError("runtime finite scan")), \
             patch(module + ".np.argmax", side_effect=AssertionError("runtime argmax")), \
             patch(module + ".np.exp", side_effect=AssertionError("runtime normalization")):
            store = GeneratedTeacherStore(self.root, self.root/'manifest.json',
                expected_manifest_sha256='1'*64, inputs_path=self.root/'inputs.json',
                expected_binding=binding(), cpu_fixture=self.cpu_fixture, verify_payloads=False)
            with patch(module + ".np.load", wraps=np.load) as load:
                for _ in range(2):
                    with store.document(0, kinds=('logp',)) as doc:
                        self.assertIsNone(doc.keys)
                        self.assertIsNone(doc.residual)
                        self.assertFalse(doc.canonical_tf_argmax_verified)
                        self.assertFalse(doc.logp.flags.writeable)
                self.assertEqual(load.call_count, 2)
            self.assertFalse(store.receipt['all_payload_sha256_verified'])
            self.assertEqual(store.receipt['validation_policy'], 'SKIPPED_USER_DIRECTED')

    def test_caller_mutation_does_not_mutate_internal_identity(self):
        cap = self.sealed_store.capsule(0)
        cap["y0"][0] = 3
        receipt = self.sealed_store.receipt
        receipt["position_counts"]["R512"] = 0
        self.assertEqual(self.sealed_store.capsule(0)["y0"], [0])
        self.assertGreater(self.sealed_store.receipt["position_counts"]["R512"], 0)

    def test_teacher_only_complete_requires_explicit_cache_at_runner(self):
        def teacher_only(manifest):
            manifest["upstream_cache_status"] = "NOT_BUILT"
            for member in manifest["documents"]:
                del member["keys"], member["residual"]
        with self.changed_manifest(teacher_only):
            store = self.open_store()
            self.assertFalse(store.has_upstream_cache)
            with store.document(0) as doc:
                self.assertIsNone(doc.keys)
                self.assertIsNone(doc.residual)
            with self.assertRaisesRegex(GeneratedTeacherError, "REQUIRED_UPSTREAM_CACHE_MISSING"):
                self.open_store(require_upstream_cache=True)

    def test_no_report_or_generic_filtering(self):
        for role in ("Report256", "S64", "Reserve320", "short", "train", "all"):
            with self.subTest(role=role), self.assertRaisesRegex(GeneratedTeacherError, "UNAPPROVED_ROLE"):
                self.sealed_store.indices(role)
        for index in (-1, 640, True, 0.0):
            with self.assertRaisesRegex(GeneratedTeacherError, "DOCUMENT_INDEX"):
                self.sealed_store.capsule(index)

    def test_cpu_fixture_cannot_be_claimed_production(self):
        with self.assertRaises(GeneratedTeacherError):
            self.open_store(cpu_fixture=None)
        with self.changed_manifest(lambda m: m.update(production_ready=True)):
            with self.assertRaisesRegex(GeneratedTeacherError, "PRODUCTION_FIXTURE_BOUNDARY"):
                self.open_store()

    def test_missing_duplicate_order_and_dev_contamination(self):
        def swapped(m):
            m["documents"][0], m["documents"][1] = m["documents"][1], m["documents"][0]
        modifications = [lambda m: m["documents"].pop(),
                         lambda m: m["documents"].__setitem__(1, m["documents"][0]), swapped,
                         lambda m: m["documents"].__setitem__(0, m["documents"][512])]
        for modify in modifications:
            with self.changed_manifest(modify), self.assertRaisesRegex(GeneratedTeacherError, "MEMBERSHIP"):
                self.open_store()

    def test_partial_old_schema_and_wrong_position_totals(self):
        for key, value in (("status", "PARTIAL"), ("schema", "TeacherStore"),
                           ("data_id", "BPCW-R512-G16"), ("data_id", "EN-R512-G128"),
                           ("upstream_cache_status", "PARTIAL"),
                           ("position_counts", dict(R512=512, Dev128=128))):
            with self.subTest(key=key, value=value), self.changed_manifest(lambda m: m.update({key: value})):
                with self.assertRaises(GeneratedTeacherError):
                    self.open_store()

    def test_expected_manifest_input_and_binding_seals(self):
        with self.assertRaisesRegex(GeneratedTeacherError, "MANIFEST_SHA"):
            self.open_store(expected_manifest_sha256="0" * 64)
        altered = binding()
        altered["w0_sha256"] = "8" * 64
        with self.assertRaisesRegex(GeneratedTeacherError, "SOURCE_MODEL_POLICY_BINDING"):
            self.open_store(expected_binding=altered)
        path = self.root / "inputs.json"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(GeneratedTeacherError, "PINNED_INPUTS_SHA"):
                self.open_store()
        finally:
            path.write_bytes(original)

    def test_payload_corruption_after_open_is_not_cached(self):
        for kind in ("logp", "keys", "residual"):
            with self.subTest(kind=kind), self.changed_payload(0, kind, lambda a: a + np.float32(.1)):
                with self.assertRaisesRegex(GeneratedTeacherError, "PAYLOAD_SHA"):
                    with self.sealed_store.document(0):
                        pass
                with self.assertRaisesRegex(GeneratedTeacherError, "PAYLOAD_SHA"):
                    self.open_store()

    def test_well_sealed_nonfinite_bad_dtype_and_bad_argmax_rejected(self):
        for kind in ("logp", "keys", "residual"):
            def nonfinite(a):
                a[0, 0] = np.nan
                return a
            with self.subTest(kind=kind), self.changed_payload(0, kind, nonfinite, reseal=True):
                with self.assertRaisesRegex(GeneratedTeacherError, "NONFINITE"):
                    self.open_store()
        with self.changed_payload(0, "logp", lambda a: a.astype(np.float64), reseal=True):
            with self.assertRaisesRegex(GeneratedTeacherError, "NPY_PAYLOAD_SCHEMA"):
                self.open_store()
        with self.changed_payload(0, "logp", lambda a: a[:, ::-1].copy(), reseal=True):
            with self.assertRaisesRegex(GeneratedTeacherError, "CANONICAL_TF_ARGMAX_MISMATCH"):
                self.open_store()
        with self.changed_payload(0, "logp", lambda a: a - np.float32(1), reseal=True):
            with self.assertRaisesRegex(GeneratedTeacherError, "TEACHER_LOGP_NORMALIZATION"):
                self.open_store()

    def test_missing_file_and_path_escape_rejected(self):
        for name in ("../outside.npy", "/outside.npy", "missing.npy"):
            with self.changed_manifest(lambda m: m["documents"][0]["logp"].update(path=name)):
                with self.assertRaisesRegex(GeneratedTeacherError, "PAYLOAD_PATH_ESCAPE|PAYLOAD_MISSING"):
                    self.open_store()
        with self.changed_manifest(lambda m: m["documents"][0].pop("residual")):
            with self.assertRaisesRegex(GeneratedTeacherError, "DOCUMENT_SCHEMA"):
                self.open_store()

    def test_symlinks_hardlinks_and_duplicate_payload_paths_rejected(self):
        first = self.root / self.manifest["documents"][0]["keys"]["path"]
        symlink = self.root / "key-alias.npy"
        symlink.symlink_to(first)
        try:
            with self.changed_manifest(lambda m: m["documents"][0]["keys"].update(path="key-alias.npy")):
                with self.assertRaisesRegex(GeneratedTeacherError, "PAYLOAD_SYMLINK"):
                    self.open_store()
        finally:
            symlink.unlink()
        with self.changed_manifest(lambda m: m["documents"][1].update(keys=m["documents"][0]["keys"])):
            with self.assertRaisesRegex(GeneratedTeacherError, "DUPLICATE_PAYLOAD_FILE"):
                self.open_store()
        hardlink = self.root / "key-hardlink.npy"
        hardlink.hardlink_to(first)
        try:
            with self.changed_manifest(lambda m: m["documents"][1].update(keys=descriptor(self.root, hardlink, np.zeros((129, 3), dtype=np.float32)))):
                with self.assertRaisesRegex(GeneratedTeacherError, "DUPLICATE_PAYLOAD_FILE"):
                    self.open_store()
        finally:
            hardlink.unlink()

    def test_capsule_corruption_and_manifest_mutation_after_open(self):
        path = self.root / self.manifest["documents"][0]["capsule"]["path"]
        original = path.read_bytes()
        try:
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(GeneratedTeacherError, "PAYLOAD_SIZE"):
                self.sealed_store.capsule(0)
        finally:
            path.write_bytes(original)
        with self.changed_manifest(lambda m: m.update(status="PARTIAL")):
            with self.assertRaisesRegex(GeneratedTeacherError, "MANIFEST_SHA"):
                self.sealed_store.capsule(0)


class CapsuleAndMembershipTests(unittest.TestCase):
    def setUp(self):
        self.binding = binding()
        self.row = dict(role="R512", ordinal=0, source_row_id="test", input_ids=[1] + [2] * 128,
                        prompt_token_sha256=token_sha256([1] + [2] * 128), window_token_sha256="6" * 64,
                        source_role="S64", source_text_sha256="7" * 64, window_start=0)

    def test_eos_one_middle_last_and_no_eos_cap(self):
        for tokens in ([0], [3] * 15 + [0], [3] * 255 + [0], [3] * 256):
            cap = make_capsule(self.row, tokens, self.binding)
            self.assertEqual(validate_capsule(cap, self.row, self.binding, 8), cap)
            self.assertEqual(cap["tf_input_ids"], self.row["input_ids"] + tokens[:-1])
            self.assertEqual(cap["score_positions"][-1], 127 + len(tokens))
            self.assertEqual(cap["length_censored"], tokens[-1] != 0)

    def test_empty_old16_old128_fake_eos_and_wrong_shift_rejected(self):
        for tokens in ([], [3] * 16, [3] * 128, [3, 0, 3], [3] * 257):
            cap = make_capsule(self.row, tokens, self.binding)
            with self.subTest(length=len(tokens)), self.assertRaises(GeneratedTeacherError):
                validate_capsule(cap, self.row, self.binding, 8)
        valid = make_capsule(self.row, [3] * 16 + [0], self.binding)
        for name, value in (("status", "PARTIAL"), ("schema", "legacy16"),
                            ("score_positions", list(range(129, 146))),
                            ("tf_input_ids", self.row["input_ids"] + valid["y0"]),
                            ("attention_mask", [1] * 144 + [0]), ("length_censored", True),
                            ("actual_length", 16), ("y0_sha256", "f" * 64)):
            with self.subTest(field=name), self.assertRaises(GeneratedTeacherError):
                validate_capsule(dict(valid, **{name: value}), self.row, self.binding, 8)
        # Fake EOS is detectable through canonical TF argmax, tested by StoreTests.

    def test_binding_disallows_old_generation_or_processing(self):
        for key, value in (("max_new_tokens", 16), ("max_new_tokens", 128),
                           ("raw_argmax", 1), ("min_new_tokens", False),
                           ("min_new_tokens", 256), ("append_fake_eos", True),
                           ("tie_rule", "arbitrary"), ("logits_processors", ["temperature"]),
                           ("eos_token_ids", [])):
            changed = deepcopy(self.binding)
            changed["generation"][key] = value
            with self.subTest(key=key), self.assertRaises(GeneratedTeacherError):
                validate_binding(changed, 8)

    def test_exact_membership_order_duplicates_and_dev_separation(self):
        rows = []
        for index in range(640):
            row = deepcopy(self.row)
            row.update(role="R512" if index < 512 else "Dev128", ordinal=index if index < 512 else index - 512,
                       source_row_id=str(index), source_role="S64" if index < 64 else
                       "Reserve320" if index < 384 else "AdditionalTrain128" if index < 512 else "Dev128")
            row["input_ids"][1:5] = [2 + index // 6**j % 6 for j in range(4)]
            row["prompt_token_sha256"] = token_sha256(row["input_ids"])
            rows.append(row)
        validate_inputs(rows, self.binding, 8)
        for modify in (lambda r: r.pop(), lambda r: r.reverse(),
                       lambda r: r[512].update(source_row_id=r[0]["source_row_id"]),
                       lambda r: r[0].update(role="Dev128"),
                       lambda r: r[0].update(source_role="Report256"),
                       lambda r: r[1].update(input_ids=r[0]["input_ids"],
                                           prompt_token_sha256=r[0]["prompt_token_sha256"])):
            changed = deepcopy(rows)
            modify(changed)
            with self.assertRaises(GeneratedTeacherError):
                validate_inputs(changed, self.binding, 8)


class CoverageTests(unittest.TestCase):
    def capsules(self):
        return [dict(schema=CAPSULE_SCHEMA, data_id=DATA_ID, status="COMPLETE", role="R512",
                     ordinal=i, source_row_id=str(i), actual_length=17 if i == 0 else 1,
                     score_positions=list(range(128, 145)) if i == 0 else [128]) for i in range(512)]

    def test_short_final_chunk_document_mean_not_chunk_or_token_mean(self):
        caps = self.capsules()
        coverage = CoverageTracker(caps, vocabulary_size=8, require_backward=True)
        acc = DocumentMeanAccumulator(coverage)
        acc.add_chunk(0, range(128, 144), 16.0, vocabulary_size=8)
        acc.add_chunk(0, [144], 17.0, vocabulary_size=8)
        coverage.record_backward(0, range(128, 145), vocabulary_size=8)
        for ordinal in range(1, 512):
            acc.add_chunk(ordinal, [128], -1.0, vocabulary_size=8)
            coverage.record_backward(ordinal, [128], vocabulary_size=8)
        value, means, receipt = acc.finish()
        self.assertEqual(means[0], 33 / 17)
        self.assertEqual(value, ((33 / 17) - 511) / 512)
        self.assertEqual(receipt["documents"], 512)
        self.assertEqual(receipt["positions"], 528)
        self.assertEqual(receipt["backward_documents"], 512)
        self.assertNotEqual(value, (33 - 511) / 528)
        self.assertNotEqual(means[0], (1 + 17) / 2)

    def test_nonfinite_and_partial_coverage_are_technical_errors(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            acc = DocumentMeanAccumulator(CoverageTracker(self.capsules(), vocabulary_size=8))
            with self.assertRaisesRegex(GeneratedTeacherError, "NONFINITE"):
                acc.add_chunk(0, range(128, 145), value, vocabulary_size=8)
        coverage = CoverageTracker(self.capsules(), vocabulary_size=8)
        coverage.record_forward(0, range(128, 145), vocabulary_size=8)
        with self.assertRaisesRegex(GeneratedTeacherError, "PARTIAL_FORWARD"):
            coverage.finish()

    def test_duplicate_reordered_missing_positions_and_topk_forbidden(self):
        for ordinal, positions, vocab in ((1, [128], 8), (0, [129], 8),
                                          (0, [128, 128], 8), (0, [], 8),
                                          (0, range(128, 145), 4)):
            coverage = CoverageTracker(self.capsules(), vocabulary_size=8)
            with self.subTest(ordinal=ordinal, positions=positions), self.assertRaises(GeneratedTeacherError):
                coverage.record_forward(ordinal, positions, vocabulary_size=vocab)
        coverage = CoverageTracker(self.capsules(), vocabulary_size=8)
        coverage.record_forward(0, range(128, 145), vocabulary_size=8)
        with self.assertRaisesRegex(GeneratedTeacherError, "ORDER_OR_DUPLICATE"):
            coverage.record_forward(0, range(128, 145), vocabulary_size=8)

    def test_every_reference_backward_required_and_dev_forbidden(self):
        coverage = CoverageTracker(self.capsules(), vocabulary_size=8, require_backward=True)
        for ordinal, cap in enumerate(self.capsules()):
            coverage.record_forward(ordinal, cap["score_positions"], vocabulary_size=8)
        with self.assertRaisesRegex(GeneratedTeacherError, "PARTIAL_BACKWARD"):
            coverage.finish()
        with self.assertRaisesRegex(GeneratedTeacherError, "DEV_GRADIENT_FORBIDDEN"):
            CoverageTracker([], role="Dev128", vocabulary_size=8, require_backward=True)
        with self.assertRaisesRegex(GeneratedTeacherError, "BACKWARD_ALL_POSITIONS_REQUIRED"):
            coverage.record_backward(0, [128], vocabulary_size=8)
        coverage.record_backward(0, range(128, 145), vocabulary_size=8)
        with self.assertRaisesRegex(GeneratedTeacherError, "ORDER_OR_DUPLICATE"):
            coverage.record_backward(0, range(128, 145), vocabulary_size=8)

    def test_membership_dev_duplicate_missing_order_rejected(self):
        for modify in (lambda c: c.pop(), lambda c: c.reverse(),
                       lambda c: c[0].update(role="Dev128"),
                       lambda c: c[1].update(source_row_id="0")):
            caps = self.capsules()
            modify(caps)
            with self.assertRaises(GeneratedTeacherError):
                CoverageTracker(caps, vocabulary_size=8)


if __name__ == "__main__":
    unittest.main()
