"""CPU only: rent runner bindings without a model, GPU, W&B or Kubernetes."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from official.experiments.prepare import ROOT, file_sha
from official.runners.rent import freeze, run
from official.runners.rent.native import model_spec, projector_slots
from official.runners.rent.submit import render
from official.tracking import schema

MODELS = ("llama3", "qwen25", "gptj")
METHODS = ("ft", "memit", "alphaedit", "memit_fe", "sphere")


class RentRunner(unittest.TestCase):
    def test_model_specs_follow_contract(self):
        expected = dict(llama3=("Meta-Llama-3-8B-Instruct", 14336, (4, 5, 6, 7, 8)),
                        qwen25=("Qwen2.5-7B-Instruct", 18944, (4, 5, 6, 7, 8)),
                        gptj=("gpt-j-6b", 16384, (3, 4, 5, 6, 7, 8)))
        for model, (basename, width, layers) in expected.items():
            spec = model_spec(model)
            self.assertEqual((spec["basename"], spec["key_width"], spec["layers"]), (basename, width, layers))

    def test_blue_uses_first_and_last_projector_plane(self):
        self.assertEqual(projector_slots((4, 5, 6, 7, 8), [4, 8]), [0, 4])
        self.assertEqual(projector_slots((3, 4, 5, 6, 7, 8), [3, 8]), [0, 5])
        self.assertEqual(projector_slots((4, 5, 6, 7, 8), [4, 5, 6, 7, 8]), [0, 1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "NATIVE_PROJECTOR_LAYER_MAPPING"):
            projector_slots((4, 5, 6, 7, 8), [3, 8])

    def test_run_ids_resolve_and_qwen_blue_needs_a_grid_row(self):
        for model in MODELS:
            for dataset in ("cf", "zsre"):
                for method in METHODS:
                    config = run.configuration(f"{model}-{dataset}-{method}")
                    run.native_hparams(config)
        with self.assertRaisesRegex(ValueError, "QWEN_BLUE_L2_SELECTION_REQUIRED"):
            run.native_hparams(run.configuration("qwen25-cf-alphaedit_blue"))
        self.assertEqual(run.native_hparams(run.configuration("qwen25-cf-alphaedit_blue-l2-10")).L2, 10)
        with self.assertRaisesRegex(ValueError, "OFFICIAL_RUN_ID"):
            run.configuration("llama3-cf-unknown")

    def test_tracking_config_is_accepted_by_official_schema(self):
        manifest = dict(code_commit="a" * 40)
        for model in MODELS:
            for mode, dataset in (("w0", "cf"), ("w0", "zsre"), ("qualification", "cf"),
                                  ("chain", "cf"), ("chain", "zsre"), ("smoke", "zsre")):
                config = run.configuration(f"{model}-{dataset}-alphaedit")
                values = run.tracking_config(manifest, config, mode, Path(f"{mode}-job092005"))
                cfg = schema.bind_job_identity(values, {schema.RENT_JOB_ENV: "092005"})
                self.assertTrue(schema.run_name(cfg).startswith("rent-"))
                self.assertTrue(schema.run_name(cfg).endswith("-job092005"))
                self.assertEqual(cfg.get("generation_schedule") is not None, dataset == "cf")

    def test_frozen_tree_matches_server3_definition(self):
        from official.runners.server3.submit import official_tree_sha256
        root = Path(tempfile.mkdtemp())
        (root / "official/sub").mkdir(parents=True)
        (root / "official/a.py").write_text("a\n")
        (root / "official/sub/b.json").write_text("{}\n")
        digest, members = freeze.tree(root)
        self.assertEqual(digest, official_tree_sha256(root))
        self.assertEqual(members["official/a.py"], hashlib.sha256(b"a\n").hexdigest())

    def test_job_templates_render_with_all_placeholders(self):
        folder = Path(run.__file__).parent
        text = render((folder / "run.job.yaml").read_text(), "092005", dict(COMMIT12="0123456789ab",
                      RUN_IDS="qwen25-cf-alphaedit_blue-l2-10,gptj-cf-ft", MODE="qualification",
                      SLUG="q-qwen25-cf-blue-l2-10", DEADLINE="14400"))
        self.assertIn("name: janghj-q-qwen25-cf-blue-l2-10-092005", text)
        self.assertIn("for run_id in $(echo qwen25-cf-alphaedit_blue-l2-10,gptj-cf-ft | tr , ' ')", text)
        self.assertIn("manifest-${run_id%%-*}.json", text)
        self.assertIn("nvidia.com/gpu: 1", text)
        text = render((folder / "prepare.job.yaml").read_text(), "092005",
                      dict(COMMIT12="0123456789ab", MODELS="llama3,qwen25,gptj"))
        self.assertIn("NVIDIA_VISIBLE_DEVICES", text)
        with self.assertRaisesRegex(ValueError, "TEMPLATE_PLACEHOLDER_UNSET"):
            render((folder / "prepare.job.yaml").read_text(), "092005", dict(COMMIT12="0123456789ab"))
        with self.assertRaisesRegex(ValueError, "TEMPLATE_VALUE_NOT_ALLOWED"):
            render((folder / "prepare.job.yaml").read_text(), "092005", dict(COMMIT12="x; rm", MODELS="gptj"))

    def test_asset_lock_covers_each_model(self):
        lock = run.read(Path(run.__file__).with_name("assets.lock.json"))["models"]
        for model in MODELS:
            spec = model_spec(model)
            self.assertEqual(len(lock[model]), len(spec["layers"]) + 1)
            self.assertIn(f"examples/null_space_project_{spec['basename']}.pt", lock[model])

    def test_runner_imports_only_official_code(self):
        self.assertEqual(ROOT, Path(run.__file__).resolve().parents[2])
        for name in ("run.py", "native.py", "prepare.py"):
            text = (Path(run.__file__).parent / name).read_text()
            self.assertNotIn("easyeditor", text)
            self.assertNotIn("from project", text)
            self.assertNotIn("official.runners.server", text)
        self.assertTrue(file_sha(Path(run.__file__)))


if __name__ == "__main__":
    unittest.main()
