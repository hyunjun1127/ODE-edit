import copy
import importlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from official.baselines.registry import call_options, hparams, requests
from official.experiments import checkpoint as cp
from official.experiments.prepare import METHODS, MODELS
from official.ours.core.jlz_interference_l1.cap_projection import lengths


class RuntimeTests(unittest.TestCase):
    def test_every_native_hparams_parser(self):
        for model in MODELS:
            for method in METHODS:
                with self.subTest(model=model, method=method):
                    override = {"L2": 10} if (model, method) == ("qwen25", "ALPHAEDIT_BLUE") else None
                    hp = hparams(method, model, overrides=override)
                    self.assertTrue(hp.layers)
        with self.assertRaisesRegex(ValueError, "SELECTION_REQUIRED"):
            hparams("ALPHAEDIT_BLUE", "qwen25")

    def test_request_schema_is_native_without_mutating_stream(self):
        record = dict(case_id=7, requested_rewrite=dict(prompt="{} is", subject="A",
                      target_new={"str": "B"}, target_true={"str": "C"}))
        original = copy.deepcopy(record)
        self.assertEqual(requests([record], "FT", "llama3")[0]["prompt"], "A is")
        self.assertEqual(requests([record], "MEMIT_FE", "qwen25")[0]["target_new"], "B")
        self.assertEqual(requests([record], "MEMIT", "llama3")[0]["target_new"], {"str": "B"})
        self.assertEqual(record, original)
        options = call_options("MEMIT", "llama3")
        self.assertFalse(options["save_weights"])
        self.assertEqual(options["beta_hse"], 0)

    def test_ours_and_shared_generator_import_without_legacy_repo(self):
        for name in ["official.ours.core.jlz_interference_l1.cap_fit",
                     "official.ours.core.jlz_v12r.entry", "official.ours.core.jlz_price_gptj.fit",
                     "official.ours.core.jlz_price_gptj.entry", "official.ours.anchor",
                     "official.evaluation.generation.native_observer"]:
            importlib.import_module(name)

    def test_price_capped_projection_KKT(self):
        values, tau, active, _, _ = lengths([3., 2.], [1., 2.], [2., 2.], 2.)
        self.assertTrue(active)
        self.assertAlmostEqual(sum(v * w for v, w in zip(values, [1., 2.])), 2.)
        self.assertTrue(all(0 <= x <= 2 for x in values))
        self.assertGreaterEqual(tau, 0)

    def test_checkpoint_B2_resume_matches_uninterrupted_B3(self):
        identity = {k: "frozen-" + k for k in cp.IDENTITY_FIELDS}
        def step(weights, history):
            weights["W"].add_(torch.rand(2) + random.random() + np.random.random())
            history["L4"].add_(weights["W"].outer(weights["W"]))
        torch.manual_seed(4); random.seed(4); np.random.seed(4)
        W, H = {"W": torch.zeros(2)}, {"L4": torch.zeros(2, 2)}
        with tempfile.TemporaryDirectory() as directory:
            for batch in range(3):
                if batch:
                    step(W, H)
                cp.save(directory, batch=batch, weights=W, cache_c=H, contexts=[["{}"]],
                        evaluation_cursor={"batch": batch}, identity=identity,
                        method="ALPHAEDIT", evaluation_complete=True)
            step(W, H)
            expected_W, expected_H = W["W"].clone(), H["L4"].clone()
            saved = cp.load(directory, identity)
            cp.rng_restore(saved["rng"])
            step(saved["weights"], saved["cache_c"])
            self.assertTrue(torch.equal(expected_W, saved["weights"]["W"]))
            self.assertTrue(torch.equal(expected_H, saved["cache_c"]["L4"]))
            self.assertEqual(len(list(Path(directory).glob("batch-*.pt"))), 1)
            with self.assertRaisesRegex(ValueError, "IDENTITY_MISMATCH"):
                cp.load(directory, dict(identity, code_commit="changed"))

    def test_failed_checkpoint_write_keeps_previous_commit(self):
        identity = {k: "frozen-" + k for k in cp.IDENTITY_FIELDS}
        args = dict(weights={"W": torch.ones(2)}, cache_c={}, contexts=[], evaluation_cursor={},
                    identity=identity, method="FT", evaluation_complete=True)
        with tempfile.TemporaryDirectory() as directory:
            cp.save(directory, batch=0, **args)
            with patch.object(cp.torch, "save", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    cp.save(directory, batch=1, **args)
            self.assertEqual(cp.load(directory, identity)["batch"], 0)
            replace = cp.os.replace
            def fail_pointer(source, destination):
                if Path(destination).name == "latest.json":
                    raise OSError("pointer write failed")
                return replace(source, destination)
            with patch.object(cp.os, "replace", side_effect=fail_pointer):
                with self.assertRaises(OSError):
                    cp.save(directory, batch=1, **args)
            self.assertEqual(cp.load(directory, identity)["batch"], 0)
            self.assertEqual(len(list(Path(directory).glob("batch-*.pt"))), 1)
            # Simulate a process dying between payload and pointer publication.
            (Path(directory) / "batch-01-0000000000000000.pt").write_bytes(b"orphan")
            cp.save(directory, batch=1, **args)
            self.assertEqual(len(list(Path(directory).glob("batch-*.pt"))), 1)
            ref = json.loads((Path(directory) / "latest.json").read_text())
            (Path(directory) / ref["file"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "HASH_MISMATCH"):
                cp.load(directory, identity)


if __name__ == "__main__":
    unittest.main()
