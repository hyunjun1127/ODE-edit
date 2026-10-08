import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from official.ours.config import (ROOT, bind_fit, canonical_sha256, metadata, plain,
                                  resolve, wandb_config)
from official.tests.price_oracle import FIXTURE, load


class ConfigTests(unittest.TestCase):
    def altered(self, changes, *, writer=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ours"
            shutil.copytree(ROOT, root)
            path = root / "qwen25" / ("writer.json" if writer else "price.json")
            value = json.loads(path.read_text())
            (value["hparams"] if writer else value).update(changes)
            path.write_text(json.dumps(value))
            return resolve("qwen25", root=root)

    def test_defaults_match_frozen_profiles_and_native_snapshots(self):
        old_factory = load("jlz_interference_l1.cap_profile").arm_profile
        for model, label in (("llama3", "LLAMA"), ("qwen25", "QWEN"), ("gptj", "GPTJ")):
            config = resolve(model)
            expected = FIXTURE["gptj_profile"] if model == "gptj" else old_factory({}, "CAP075", label)
            for key, value in expected.items():
                if key == "status":
                    continue
                with self.subTest(model=model, key=key):
                    self.assertEqual(plain(config[key]), value)
            writer = json.loads((ROOT / model / "writer.json").read_text())
            self.assertEqual(writer["hparams"], FIXTURE["writer_snapshots"][model]["hparams"])
            self.assertEqual(writer["source"], FIXTURE["writer_snapshots"][model]["source"])
            data = plain(config)
            identity = data.pop("config_sha256")
            self.assertEqual(identity, canonical_sha256(data))
            self.assertEqual(identity, FIXTURE["default_config_sha256"][model])
            self.assertEqual(config["K_eval"], config["max_updates"] + 1)
            self.assertEqual(config["beta_max_scale"], config["beta_max_native_scale"])

    def test_nested_immutability_and_detached_receipts(self):
        config = resolve("qwen25")
        with self.assertRaises(TypeError): config["lr"] = 99
        with self.assertRaises(TypeError): config["writer_hparams"]["v_lr"] = 99
        with self.assertRaises(TypeError): config["betas"][0] = 0
        with self.assertRaises(AttributeError): config._data = {}
        detached = metadata(config)
        detached["resolved_config"]["writer_hparams"]["layers"].append(99)
        self.assertNotIn(99, config["eligible_layers"])
        self.assertEqual(wandb_config(config)["ours"]["config_sha256"], config["config_sha256"])

    def test_each_arm_changes_only_its_declared_knobs_and_derived_aliases(self):
        base = plain(resolve("qwen25"))
        for path in sorted((ROOT / "arms").glob("*.json")):
            patch = json.loads(path.read_text())
            result = plain(resolve("qwen25", path.stem))
            for key, value in patch["override"].items(): self.assertEqual(result[key], value)
            allowed = set(patch["override"]) | {"arm", "config_sha256", "configuration_sources"}
            if "beta_max_scale" in allowed: allowed.add("beta_max_native_scale")
            if "max_updates" in allowed: allowed.add("K_eval")
            self.assertFalse({key for key in base if base[key] != result[key]} - allowed)
            self.assertEqual(result["writer_hparams"], base["writer_hparams"])
            self.assertEqual(result["beta_max_scale"], result["beta_base"])
            with self.assertRaises(ValueError): resolve("llama3", path.stem)

    def test_domains_missing_unknown_duplicate_and_nonfinite_are_rejected(self):
        for change in ({"beta_base": 0}, {"beta_base": -1}, {"beta_base": float("nan")},
                       {"lr": float("inf")}, {"lr": 0}, {"eps": 0}, {"tau_F": 0},
                       {"lambda_N": -1}, {"lambda_KL": -1}, {"beta_max_scale": -1},
                       {"n_exp": -1}, {"n_exp": True}, {"K_grace": 24},
                       {"max_updates": 0}, {"max_updates": 24.0}, {"betas": [0.9, 1]},
                       {"betas": [0.9]}, {"c": None}, {"c": 0}, {"cap_mode": "other"},
                       {"K_eval": 25}, {"beta_max_native_scale": 0.8}):
            with self.subTest(change=change), self.assertRaises(ValueError): self.altered(change)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "ours"; shutil.copytree(ROOT, root)
            path = root / "qwen25/price.json"
            value = json.loads(path.read_text()); value.pop("lr"); path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "MISSING_KEYS"): resolve("qwen25", root=root)
            path.write_text('{"lr": 0.1, "lr": 0.5}')
            with self.assertRaisesRegex(ValueError, "DUPLICATE_CONFIG_KEY"): resolve("qwen25", root=root)
        with self.assertRaises(ValueError): resolve("qwen25", "../qwen25/price")

    def test_small_budget_and_independent_writer_configuration(self):
        result = self.altered(dict(max_updates=1, K_grace=0, n_exp=0, beta_max_scale=0,
                                   lambda_N=0, lambda_KL=0, cap_mode="none", c=None))
        self.assertEqual(result["K_eval"], 2)
        # A copied ours tree resolves even without any baseline file alongside it.
        changed = self.altered(dict(layers=[5, 7], v_loss_layer=26, mom2_update_weight=123), writer=True)
        self.assertEqual(changed["eligible_layers"], (5, 7))
        self.assertEqual(changed["anchor_layer"], 7)
        self.assertEqual(changed["nll_layer"], 26)
        self.assertEqual(changed["lambda_C"], 123)

    def test_mismatched_fit_config_refuses_and_wandb_receives_full_identity(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        config = resolve("qwen25"); adapter = SimpleNamespace(profile=config)
        with self.assertRaisesRegex(ValueError, "MISMATCH"):
            bind_fit(adapter, resolve("qwen25", "qwen25-beta150"))
        with self.assertRaises(TypeError): bind_fit(SimpleNamespace(profile=plain(config)))
        run = Mock()
        self.assertIs(bind_fit(adapter, config, run), config)
        run.config.update.assert_called_once_with(wandb_config(config), allow_val_change=False)


if __name__ == "__main__": unittest.main()
