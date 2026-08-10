from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from project.run_scripts.ode_bf.contracts import ODEBFContractError, canonical_hash
from project.run_scripts.ode_bf.p1_perrequest_target_simplex_transport_panel import (
    P1R15_SOURCE_MANIFEST_FILE,
    expected_p1r15_stage_a_result_name,
    load_and_validate_p1r15_source_manifest,
    validate_p1r15_source_closure,
    validate_p1r15_stage_a_output_root,
)
from project.run_scripts import session05_ode_bf_perrequest_simplex_transport as entry
from project.run_scripts import session05_ode_bf_perrequest_simplex_transport_dry_plan as dry
from project.run_scripts import session05_ode_bf_submit_perrequest_simplex_transport as submit


class P1R15LaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[4]

    def test_output_namespace_exact_and_alias_swap_fail_closed(self) -> None:
        for alias in (dry.LLAMA_ALIAS, dry.QWEN_ALIAS):
            expected = (
                self.repo
                / "local"
                / "odebf"
                / "results"
                / expected_p1r15_stage_a_result_name(alias)
            )
            receipt = validate_p1r15_stage_a_output_root(
                repo_root=self.repo, alias=alias, output_root=expected
            )
            self.assertEqual(receipt["alias"], alias)
            other = dry.QWEN_ALIAS if alias == dry.LLAMA_ALIAS else dry.LLAMA_ALIAS
            with self.assertRaises(ODEBFContractError):
                validate_p1r15_stage_a_output_root(
                    repo_root=self.repo,
                    alias=alias,
                    output_root=(
                        self.repo
                        / "local"
                        / "odebf"
                        / "results"
                        / expected_p1r15_stage_a_result_name(other)
                    ),
                )

    def test_namespace_probe_does_not_create_result_or_load_model(self) -> None:
        alias = dry.LLAMA_ALIAS
        result = (
            self.repo
            / "local"
            / "odebf"
            / "results"
            / expected_p1r15_stage_a_result_name(alias)
        )
        self.assertFalse(result.exists())
        status = entry.main(
            [
                "--model",
                alias,
                "--output-root",
                str(result),
                "--source-head",
                "1" * 40,
                "--run-token",
                "perrequest-simplex-transport-p1r15-a1-stage-a-tech-r1-v1",
                "--namespace-probe",
            ]
        )
        self.assertEqual(status, 0)
        self.assertFalse(result.exists())

    def test_sbatch_syntax_and_exact_source_parent_contract(self) -> None:
        sbatch = self.repo / "project/run_scripts/session05_ode_bf_perrequest_simplex_transport.sbatch"
        subprocess.run(["bash", "-n", str(sbatch)], check=True)
        source = sbatch.read_text(encoding="utf-8")
        self.assertIn("SLURM_SUBMIT_DIR", source)
        self.assertIn("6116994a171daec92755d846cd49ba8aa279dd8c", source)
        self.assertIn("--cpus-per-task=8", source)
        self.assertIn("--gres=gpu:1", source)
        self.assertIn("--mem=65000M", source)

    def test_launcher_binds_exact_easyedit_source_path(self) -> None:
        sbatch = self.repo / "project/run_scripts/session05_ode_bf_perrequest_simplex_transport.sbatch"
        source = sbatch.read_text(encoding="utf-8")
        expected_root = "/mnt/raid5/janghj/EasyEdit"
        self.assertIn(f'EASYEDIT_ROOT="{expected_root}"', source)
        self.assertIn('export PYTHONPATH="$EASYEDIT_ROOT"', source)
        self.assertNotIn('PYTHONPATH="${PYTHONPATH', source)
        probe = (
            "import importlib.util, pathlib; "
            "spec=importlib.util.find_spec('easyeditor'); "
            "assert spec is not None and pathlib.Path(spec.origin).resolve().is_relative_to("
            "pathlib.Path('/mnt/raid5/janghj/EasyEdit').resolve())"
        )
        positive_env = dict(os.environ)
        positive_env["PYTHONPATH"] = expected_root
        subprocess.run(
            [sys.executable, "-c", probe],
            cwd=self.repo,
            env=positive_env,
            check=True,
        )
        negative_env = dict(os.environ)
        negative_env["PYTHONPATH"] = ""
        negative = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=self.repo,
            env=negative_env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertNotEqual(negative.returncode, 0)

    def test_submit_requires_package_ack_and_held_release(self) -> None:
        source = inspect.getsource(submit.submit)
        self.assertIn("package.verify_local_package", source)
        self.assertIn("_load_ack", source)
        self.assertIn('"--hold"', source)
        self.assertIn('("scontrol", "release", job_id)', source)
        self.assertNotIn("retry", source.lower())

    def test_source_closure_rejects_forbidden_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = root / "bad.py"
            bad.write_text("TARGET_HOLD = True\n", encoding="utf-8")
            with self.assertRaises(ODEBFContractError):
                validate_p1r15_source_closure(root, paths=(bad.name,))

    def test_source_manifest_exact_and_missing_mode_fails_closed(self) -> None:
        path = (
            self.repo
            / "project"
            / "run_scripts"
            / "ode_bf"
            / "locks"
            / P1R15_SOURCE_MANIFEST_FILE
        )
        value, _sha256 = load_and_validate_p1r15_source_manifest(
            self.repo, path, source_head="0" * 40
        )
        self.assertEqual(len(value["entries"]), 12)
        malformed = dict(value)
        malformed.pop("root_digest")
        malformed["entries"] = [dict(item) for item in value["entries"]]
        malformed["entries"][0].pop("mode")
        malformed["entry_root"] = canonical_hash(malformed["entries"])
        malformed["root_digest"] = canonical_hash(malformed)
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "manifest.json"
            candidate.write_text(
                json.dumps(malformed, sort_keys=True), encoding="utf-8"
            )
            with self.assertRaises(ODEBFContractError):
                load_and_validate_p1r15_source_manifest(
                    self.repo, candidate, source_head="0" * 40
                )


if __name__ == "__main__":
    unittest.main()
