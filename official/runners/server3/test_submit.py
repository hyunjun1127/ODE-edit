"""CPU-only checks of the Qwen submission matrix and fail-closed gates."""

import copy
from pathlib import Path
import json
import getpass
import os
import pwd
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from official.experiments.prepare import digest, materialize_matrix
from official.runners.server3 import submit
from official.tracking import schema as tracking_schema
from official.tracking.method import OFFICIAL_SCHEMA


class SubmitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.matrix = self.root / "matrix"
        materialize_matrix(submit.ROOT / "official/hparams", self.matrix)
        self.rows, self.files = submit.matrix_rows(self.matrix)

    def test_physical_stages_and_alias(self):
        qualify = submit.stage_specs("qualify", self.rows, self.files, self.root)
        cf = submit.stage_specs("cf", self.rows, self.files, self.root)
        self.assertEqual([x["kind"] for x in qualify], ["w0"] + ["qualify"] * 6)
        self.assertEqual(len(cf), 10)
        self.assertTrue(all(x["kind"] == "edit" for x in cf))
        self.assertNotIn("qwen25-cf-alphaedit_blue", [x["key"] for x in cf])
        self.assertEqual(sum("-l2-" in x["key"] for x in cf), 3)
        self.assertEqual(sum("clamp075" in x["key"] for x in cf), 2)
        with self.assertRaisesRegex(submit.Blocked, "ZSRE_BLUE_SELECTION_REQUIRED"):
            submit.stage_specs("zsre", self.rows, self.files, self.root)
        row = copy.deepcopy(self.rows["qwen25-zsre-alphaedit_blue"])
        row["hparams"]["L2"] = 10
        row["config_sha256"] = digest({k: v for k, v in row.items()
                                       if k != "config_sha256"})
        selected = self.root / "selected.json"
        submit.write_new(selected, row)
        zsre = submit.stage_specs("zsre", self.rows, self.files, self.root,
                                  {"selected_zsre_config": str(selected),
                                   "selected_zsre_config_sha256": submit.file_sha(selected)})
        self.assertEqual([x["kind"] for x in zsre[:2]], ["w0", "smoke"])
        self.assertEqual(sum(x["kind"] == "edit" for x in zsre), 6)
        self.assertEqual(len({x["output"] for x in zsre}), 8)

    def test_storage_uses_editable_tensors(self):
        cf = submit.stage_specs("cf", self.rows, self.files, self.root)
        row = copy.deepcopy(self.rows["qwen25-zsre-alphaedit_blue"])
        row["hparams"]["L2"] = 1
        row["config_sha256"] = digest({k: v for k, v in row.items()
                                       if k != "config_sha256"})
        selected = self.root / "selected.json"
        submit.write_new(selected, row)
        zsre = submit.stage_specs("zsre", self.rows, self.files, self.root,
                                  {"selected_zsre_config": str(selected),
                                   "selected_zsre_config_sha256": submit.file_sha(selected)})
        scientific = [x for x in cf + zsre if x["kind"] == "edit"]
        bytes_retained = sum(submit.checkpoint_bytes(submit.read(x["config"]))
                             for x in scientific)
        self.assertGreater(bytes_retained, 60 * submit.GIB)
        self.assertLess(bytes_retained, 63 * submit.GIB)
        with patch.object(submit, "model_snapshot_bytes", return_value=14 * submit.GIB), \
             patch.object(submit.shutil, "disk_usage",
                          return_value=SimpleNamespace(free=100 * submit.GIB)):
            budget = submit.storage_reserve(self.root, scientific, {})
        self.assertEqual(budget["planned_checkpoint_bytes"], bytes_retained)
        self.assertLess(budget["reserve_bytes"], 90 * submit.GIB)

    def test_first_registration_reserves_full_program_without_checkpoint_cleanup(self):
        with patch.object(submit.shutil, "disk_usage",
                          return_value=SimpleNamespace(free=140 * submit.GIB)):
            budget = submit.full_program_storage_reserve(self.root, self.rows)
        self.assertGreater(budget["reserve_bytes"], 129 * submit.GIB)
        self.assertEqual(budget["qualification_checkpoint_copies"], 12)
        self.assertFalse(budget["cleanup_assumed"])
        with patch.object(submit.shutil, "disk_usage",
                          return_value=SimpleNamespace(free=100 * submit.GIB)):
            with self.assertRaisesRegex(submit.Blocked, "FULL_PROGRAM_DISK_LOW"):
                submit.full_program_storage_reserve(self.root, self.rows)

    def test_qualification_reserves_both_b3_checkpoints_and_atomic_temp(self):
        specs = submit.stage_specs("qualify", self.rows, self.files, self.root)
        one_each = [submit.checkpoint_bytes(submit.read(s["config"])) for s in specs
                    if s["kind"] == "qualify"]
        with patch.object(submit, "model_snapshot_bytes", return_value=14 * submit.GIB), \
             patch.object(submit.shutil, "disk_usage",
                          return_value=SimpleNamespace(free=100 * submit.GIB)):
            budget = submit.storage_reserve(self.root, specs, {})
        self.assertEqual(budget["planned_checkpoint_bytes"], 2 * sum(one_each))
        self.assertEqual(budget["atomic_peak_bytes"], max(one_each))
        self.assertEqual(budget["reserve_bytes"],
                         2 * sum(one_each) + max(one_each) + 16 * submit.GIB)

    def test_launcher_python_must_match_manifest_venv_path_and_binary(self):
        invocation = self.root / "venv" / "bin" / "python"
        invocation.parent.mkdir(parents=True)
        invocation.symlink_to(sys.executable)
        path, binary = submit.bind_runtime_python(invocation,
                                                  {"runtime_python": str(invocation)})
        self.assertEqual(path, invocation)
        self.assertEqual(binary["path"], str(Path(sys.executable).resolve()))
        with self.assertRaisesRegex(submit.Blocked, "RUNTIME_PYTHON_MANIFEST_PATH_MISMATCH"):
            submit.bind_runtime_python(sys.executable,
                                       {"runtime_python": str(invocation)})

    def test_wandb_probe_does_not_inherit_env_only_credentials(self):
        sdk = self.root / "sdk-python"
        sdk.write_text("sdk stub\n")
        sdk.chmod(0o700)
        env_file = self.root / "env"
        env_file.write_text("placeholder\n")
        settings = json.dumps({"python": str(sdk), "base_url": "https://api.wandb.ai"})
        with patch.dict(os.environ, {"WANDB_API_KEY": "SECRET_SENTINEL",
                                      "NETRC": "/tmp/custom-netrc",
                                      "WANDB_CREDENTIALS_FILE": "/tmp/custom-credential"}), \
             patch.object(submit, "command", return_value=settings) as command, \
             patch.object(submit.subprocess, "run",
                          return_value=SimpleNamespace(returncode=0)) as run:
            receipt = submit.check_wandb(self.root, env_file)
        self.assertIn("from official.tracking.schema import load_env",
                      command.call_args.args[0][2])
        self.assertEqual(command.call_args.kwargs["cwd"], self.root)
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["HOME"], pwd.getpwuid(os.getuid()).pw_dir)
        self.assertNotIn("WANDB_API_KEY", env)
        self.assertNotIn("NETRC", env)
        self.assertNotIn("WANDB_CREDENTIALS_FILE", env)
        self.assertEqual(receipt["sdk_python"], str(sdk.resolve()))

    def test_terminal_success_uses_durable_sacct_and_checks_script(self):
        attempt = self.root / "attempt"
        source = attempt / "source"
        key = "qwen25-cf-ft"
        script = attempt / "launchers" / (key + ".sh")
        row = "|".join(("12345", "COMPLETED", "0:0", submit.JOB_PREFIX + key,
                        getpass.getuser(), "sbatch --parsable --hold " + str(script),
                        str(source)))
        def accounting_only(argv, **_):
            self.assertEqual(argv[0], "sacct")
            return row
        with patch.object(submit, "command", side_effect=accounting_only):
            self.assertEqual(submit.successful_job("12345", key, attempt, source)["state"],
                             "COMPLETED")
        bad = row.replace(str(script), str(attempt / "other.sh"))
        with patch.object(submit, "command", return_value=bad):
            with self.assertRaisesRegex(submit.Blocked, "ACCOUNTING_SUBMIT_SCRIPT_MISMATCH"):
                submit.successful_job("12345", key, attempt, source)

    def test_blue_tie_break_order(self):
        rows = [{"l2": n, "metrics": {"Score": 72.0, "Specificity": 80.0,
                                       "Efficacy": 70.0, "Generalization": 75.0}}
                for n in submit.GRID]
        self.assertEqual(submit.choose_blue_winner(rows)["l2"], 1)
        rows[1]["metrics"]["Specificity"] = 81.0
        self.assertEqual(submit.choose_blue_winner(rows)["l2"], 10)
        rows[2]["metrics"]["Score"] = 72.001
        self.assertEqual(submit.choose_blue_winner(rows)["l2"], 95)
        rows[2]["metrics"]["Score"] = float("nan")
        with self.assertRaisesRegex(submit.Blocked, "BLUE_GRID_METRICS_INVALID"):
            submit.choose_blue_winner(rows)

    def test_source_digest_ignores_bytecode_but_detects_source_change(self):
        package = self.root / "official"
        package.mkdir()
        source = package / "source.py"
        source.write_text("value = 1\n")
        before = submit.official_tree_sha256(self.root)
        cache = package / "__pycache__"
        cache.mkdir()
        (cache / "source.pyc").write_bytes(b"cache")
        self.assertEqual(before, submit.official_tree_sha256(self.root))
        source.write_text("value = 2\n")
        self.assertNotEqual(before, submit.official_tree_sha256(self.root))

    def test_frozen_tracking_is_inside_official_source_closure(self):
        package = self.root / "official/tracking"
        package.mkdir(parents=True)
        for name in ("client.py", "schema.py", "method.py", "worker.py"):
            (package / name).write_text("# frozen\n")
        runner = self.root / "official/runners/server3/run.py"
        runner.parent.mkdir(parents=True)
        runner.write_text('import_module("official.tracking.client")\n')
        submit.verify_tracking_closure(self.root)
        before = submit.official_tree_sha256(self.root)
        (package / "schema.py").write_text("# changed\n")
        self.assertNotEqual(before, submit.official_tree_sha256(self.root))
        runner.write_text('import_module("project.run_scripts.experiment_tracking.client")\n')
        with self.assertRaisesRegex(submit.Blocked, "RUNNER_OFFICIAL_TRACKING_IMPORT_MISSING"):
            submit.verify_tracking_closure(self.root)
        runner.write_text('import_module("official.tracking.client")\n'
                          'import_module("project.run_scripts.experiment_tracking.client")\n')
        with self.assertRaisesRegex(submit.Blocked, "RUNNER_LEGACY_TRACKING_IMPORT"):
            submit.verify_tracking_closure(self.root)

    def test_official_zsre_schema_accepts_actual_w0_token_denominators(self):
        config = dict(server="server3", task_id="official-baselines", arm="QWEN_ZSRE_W0",
                      attempt="fixture", source_sha="a" * 40, config_sha="b" * 64,
                      model="qwen25", model_family="qwen", writer="ft", baseline="FT",
                      role="scientific", metric_schema=OFFICIAL_SCHEMA,
                      instruction_id=tracking_schema.OFFICIAL_INSTRUCTION,
                      dataset="zsre")
        values = {"edits": 0, "W0_first2000/R/count": 3915,
                  "W0_first2000/P/count": 8219, "W0_first2000/N/count": 6441}
        self.assertEqual(tracking_schema.metrics(values, scientific=True,
                                                 config_values=config), values)

    def test_online_mode_is_a_registration_gate(self):
        with patch.dict("os.environ", {"WANDB_MODE": "offline"}):
            with self.assertRaisesRegex(submit.Blocked, "WANDB_ONLINE_REQUIRED"):
                submit.check_wandb(self.root, self.root / "absent.env")

    def test_cf_registration_calls_qualification_gate_without_config_map(self):
        output = self.root / "output"
        assets = self.root / "assets.json"
        submit.write_new(assets, {"output_root": str(output)})
        args = SimpleNamespace(assets=assets, stage="cf", selection=None,
                               attempt=output / "attempts" / "attempt-test",
                               matrix_root=self.matrix)
        with patch.object(submit, "verify_qualification",
                          side_effect=submit.Blocked("QUALIFICATION_SENTINEL")) as gate:
            with self.assertRaisesRegex(submit.Blocked, "QUALIFICATION_SENTINEL"):
                submit.submit_held(args)
        gate.assert_called_once_with(output)


if __name__ == "__main__":
    unittest.main()
