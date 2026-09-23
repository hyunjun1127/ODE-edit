"""Synthetic CPU contracts; do not execute actual G1 or model forwards."""
import copy
import unittest

import numpy as np
import torch

from project.run_scripts.alpha_key_concentration_causal import technical as tech


def checkpoint_fixture():
    weights = {f"model.layers.{l}.mlp.down_proj.weight": torch.full((2, 3), float(l))
               for l in tech.LAYERS}
    cache = torch.ones(5, 3, 3)
    state = {"weights": {k: tech.tensor_sha(v) for k, v in weights.items()}, "cache": tech.tensor_sha(cache)}
    contexts = [["{}"], [f"prefix{i} {{}}" for i in range(5)]]
    meta = {"batch": 1, "seen_ids": list(range(100)), "method": "AlphaEdit", "cache_c_is_history": True,
            "base_model_revision": tech.MODEL_REVISION, "sample_root": tech.ORDER_SHA256,
            "lock_sha256": "lock", "source": "archive", "contexts": contexts,
            "rng": {"python": [], "numpy": [], "torch": [], "cuda": []}, "state": state}
    cp = {"weights": weights, "cache_c": cache, "metadata": meta}
    commit = {"batch": 1, "history_append_passes": 1, "endpoint": copy.deepcopy(state),
              "status": "BATCH_COMMITTED", "requests": 100, "seen_requests": 100,
              "history_entries_out": 100, "context_hash": tech.digest(contexts)}
    kwargs = {"batch": 1, "seen_ids": list(range(100)), "contexts": contexts,
              "original_lock_sha": "lock", "source_archive_sha": "archive", "commit": commit,
              "weight_shape": (2, 3)}
    return cp, kwargs


def timestamp_fixture():
    rows = [{"case_id": i, "requested_rewrite": {"subject": f"subject{i}", "relation_id": "P1",
                                                 "target_new": {"str": f"target{i}"}}} for i in range(4000)]
    bank = []
    for batch in tech.TIMESTAMP_BATCHES:
        for position in range(100):
            i = (batch - 1) * 100 + position
            bank.append({"case_id": i, "write_batch": batch, "timestamp_checkpoint": f"B{batch:03d}",
                         "subject": f"subject{i}", "relation": "P1", "target": f"target{i}"})
    return rows, {"n": 512, "records": bank[:512]}


class TechnicalCPU(unittest.TestCase):
    def test_checkpoint_content_schema_commit(self):
        cp, kwargs = checkpoint_fixture()
        result = tech.validate_checkpoint_object(cp, **kwargs)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["seen_count"], 100)
        self.assertEqual(result["gpu_continuation"], "NOT_TESTED")
        self.assertEqual(len(result["weights"]), 5)

    def test_checkpoint_weight_corruption(self):
        cp, kwargs = checkpoint_fixture()
        cp["weights"]["model.layers.4.mlp.down_proj.weight"][0, 0] += 1
        with self.assertRaisesRegex(tech.TechnicalFailure, "CP_WEIGHT_HASH"):
            tech.validate_checkpoint_object(cp, **kwargs)

    def test_checkpoint_history_corruption(self):
        cp, kwargs = checkpoint_fixture()
        cp["cache_c"][0, 0, 0] += 1
        with self.assertRaisesRegex(tech.TechnicalFailure, "CP_HISTORY_HASH"):
            tech.validate_checkpoint_object(cp, **kwargs)

    def test_checkpoint_nonfinite_is_technical(self):
        cp, kwargs = checkpoint_fixture()
        cp["cache_c"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(tech.TechnicalFailure, "CP_NONFINITE"):
            tech.validate_checkpoint_object(cp, **kwargs)

    def test_checkpoint_order_mismatch(self):
        cp, kwargs = checkpoint_fixture()
        kwargs["seen_ids"] = kwargs["seen_ids"][::-1]
        with self.assertRaisesRegex(tech.TechnicalFailure, "CP_ORDER_OR_BATCH"):
            tech.validate_checkpoint_object(cp, **kwargs)

    def test_checkpoint_original_commit_mismatch(self):
        cp, kwargs = checkpoint_fixture()
        kwargs["commit"]["endpoint"]["cache"] = "different"
        with self.assertRaisesRegex(tech.TechnicalFailure, "ORIGINAL_COMMIT_CONTENT"):
            tech.validate_checkpoint_object(cp, **kwargs)

    def test_checkpoint_source_and_context_fail(self):
        cp, kwargs = checkpoint_fixture()
        cp["metadata"]["source"] = "different"
        with self.assertRaisesRegex(tech.TechnicalFailure, "SOURCE_BINDING"):
            tech.validate_checkpoint_object(cp, **kwargs)
        cp["metadata"]["source"] = "archive"
        cp["metadata"]["contexts"] = [["{}"], ["different"] * 5]
        with self.assertRaisesRegex(tech.TechnicalFailure, "CONTEXT_BINDING"):
            tech.validate_checkpoint_object(cp, **kwargs)

    def test_shape_and_precision_not_silently_cast(self):
        with self.assertRaisesRegex(tech.TechnicalFailure, "CP_TENSOR_SCHEMA"):
            tech.tensor_content(torch.zeros(2, 3, dtype=torch.float64), shape=(2, 3), name="test")
        with self.assertRaisesRegex(tech.TechnicalFailure, "CP_TENSOR_SCHEMA"):
            tech.tensor_content(torch.zeros(2, 3), shape=(3, 2), name="test")

    def test_exact_parity_does_not_invent_tolerance(self):
        a = torch.zeros(2, 3)
        b = a.clone(); b[0, 0] = 1e-20
        result = tech.compare_exact(a, b, "near")
        self.assertFalse(result["exact"])
        self.assertGreater(result["max_abs"], 0)
        self.assertIn("NOT_ESTABLISHED", result["numerical_envelope"])
        self.assertTrue(tech.compare_exact(a, a.clone(), "same")["exact"])

    def test_compare_nonfinite_not_numerical_warning(self):
        a = torch.tensor([float("inf")])
        with self.assertRaisesRegex(tech.TechnicalFailure, "PARITY_NONFINITE"):
            tech.compare_exact(a, a, "invalid")

    def test_rng_fingerprint_no_sampling(self):
        state = ((3, (1, 2), None), ("MT19937", np.array([1, 2], dtype=np.uint32), 2, 0, 0.),
                 torch.tensor([1, 2], dtype=torch.uint8), [torch.tensor([3], dtype=torch.uint8)])
        original = tech.rng_fingerprint(state)
        self.assertEqual(original, tech.rng_fingerprint(copy.deepcopy(state)))
        state[2][0] = 3
        self.assertNotEqual(original, tech.rng_fingerprint(state))

    def test_timestamp_bank_maps_original_full_batches(self):
        rows, panel = timestamp_fixture()
        mapped = tech.timestamp_selection(panel, rows)
        self.assertEqual(list(mapped), list(tech.TIMESTAMP_BATCHES))
        self.assertEqual(sum(map(len, mapped.values())), 512)
        self.assertEqual(len(mapped[40]), 12)
        self.assertEqual(mapped[40][0], {"bank_index": 500, "batch_position": 0, "case_id": 3900})

    def test_timestamp_fact_mismatch_not_subtracted(self):
        rows, panel = timestamp_fixture()
        panel["records"][0]["target"] = "different"
        with self.assertRaisesRegex(tech.TechnicalFailure, "RAW_FACT_IDENTITY"):
            tech.timestamp_selection(panel, rows)

    def test_timestamp_future_or_wrong_epoch_not_used(self):
        rows, panel = timestamp_fixture()
        panel["records"][0]["timestamp_checkpoint"] = "B100"
        with self.assertRaisesRegex(tech.TechnicalFailure, "CHECKPOINT_IDENTITY"):
            tech.timestamp_selection(panel, rows)

    def test_timestamp_duplicate_bank_fails(self):
        rows, panel = timestamp_fixture()
        panel["records"][1] = panel["records"][0]
        with self.assertRaisesRegex(tech.TechnicalFailure, "DUPLICATE_OR_EPOCH"):
            tech.timestamp_selection(panel, rows)

    def test_native_observation_last_five_not_first_five(self):
        keys = [{"layer": layer, "shape": [100, 14336], "sha256": str(index) * 64}
                for index in (0, 1) for layer in tech.LAYERS]
        result = tech.history_append_key_rows({"history_append_passes": 1, "keys": keys})
        self.assertEqual(set(result), set(tech.LAYERS))
        self.assertEqual(result[4]["sha256"], "1" * 64)

    def test_native_observation_incomplete_is_not_parity(self):
        with self.assertRaisesRegex(tech.TechnicalFailure, "HISTORY_KEY_COUNT"):
            tech.history_append_key_rows({"history_append_passes": 1, "keys": []})


if __name__ == "__main__":
    unittest.main()
