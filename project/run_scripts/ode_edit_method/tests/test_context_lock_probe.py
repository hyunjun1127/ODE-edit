from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from project.run_scripts import session02_bf16_context_lock_probe as probe
from project.run_scripts.ode_edit_motivation.contracts import ContextManifest


_TEMPLATES = (
    ("{}",),
    (
        "probe-secret-alpha {}",
        "probe-secret-beta {}",
        "probe-secret-gamma {}",
        "probe-secret-delta {}",
        "probe-secret-epsilon {}",
    ),
)


class _FakeSourceManifest:
    manifest_id = "1" * 64
    files = (object(), object())

    def __init__(self) -> None:
        self.assertions = 0

    def assert_current(self) -> None:
        self.assertions += 1


class _FakeBridge:
    def __init__(self, template_repeats: list[tuple[tuple[str, ...], ...]]) -> None:
        self.template_repeats = list(template_repeats)
        self.manifest = _FakeSourceManifest()
        self.preflight_calls = 0
        self.generation_calls: list[tuple[str, bool]] = []

    def preflight(self) -> _FakeSourceManifest:
        self.preflight_calls += 1
        return self.manifest

    def freeze_generated_contexts(
        self,
        model: torch.nn.Module,
        tokenizer: object,
        *,
        source: str,
        fresh: bool,
    ) -> ContextManifest:
        del model, tokenizer
        self.generation_calls.append((source, fresh))
        index = len(self.generation_calls) - 1
        return ContextManifest.freeze(self.template_repeats[index], source=source)


class _FakeModel(torch.nn.Module):
    def __init__(
        self,
        parameter_dtype: torch.dtype = torch.bfloat16,
        config_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones((2, 2), dtype=parameter_dtype), requires_grad=False
        )
        self.config = SimpleNamespace(torch_dtype=config_dtype, use_cache=False)


class _FakeRuntime:
    def __init__(
        self,
        alias: str,
        *,
        parameter_dtype: torch.dtype = torch.bfloat16,
        config_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        revision = "a" * 40
        self.model = _FakeModel(parameter_dtype, config_dtype)
        self.tokenizer = object()
        self._metadata = {
            "model_alias": alias,
            "repository_id": "fixed/repository",
            "revision": revision,
            "dtype": str(parameter_dtype),
            "observed_parameter_dtype": str(parameter_dtype),
            "checkpoint_original_dtype": "torch.bfloat16",
            "dtype_policy": "checkpoint-original",
            "observed_model_commit": revision,
            "observed_tokenizer_commit": revision,
        }

    def metadata(self) -> dict[str, object]:
        return dict(self._metadata)


class ContextLockProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.easyedit = Path(self.temp.name) / "EasyEdit"
        (self.repo / "local" / "results").mkdir(parents=True)
        self.easyedit.mkdir()
        self.offline = mock.patch.dict(
            os.environ,
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
            clear=False,
        )
        self.offline.start()
        self.addCleanup(self.offline.stop)

    def _args(self, alias: str, name: str) -> argparse.Namespace:
        return argparse.Namespace(
            model_alias=alias,
            output_root=self.repo / "local" / "results" / name,
            easyedit_root=self.easyedit,
            execute=True,
        )

    def _run(
        self,
        *,
        alias: str = "llama3-8b-inst",
        name: str = "session02-bf16-context-lock-probe-v1-test",
        repeats: list[tuple[tuple[str, ...], ...]] | None = None,
        runtime: _FakeRuntime | None = None,
    ) -> tuple[int, Path, _FakeBridge, list[int], str]:
        bridge = _FakeBridge(repeats or [_TEMPLATES, _TEMPLATES])
        seeds: list[int] = []
        output = self._args(alias, name).output_root
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = probe.execute_probe(
                self._args(alias, name),
                runtime_loader=lambda observed_alias: runtime
                or _FakeRuntime(observed_alias),
                bridge_factory=lambda _root: bridge,
                seed_fn=lambda seed: seeds.append(seed) or {"seed": seed},
                repo_root=self.repo,
                git_head_fn=lambda _repo: "b" * 40,
            )
        return code, output, bridge, seeds, stdout.getvalue()

    def test_deterministic_repeat_pass_and_terminal_hashes(self) -> None:
        code, output, bridge, seeds, stdout = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(seeds, [17, 17])
        self.assertEqual(bridge.preflight_calls, 1)
        self.assertEqual(bridge.manifest.assertions, 2)
        self.assertEqual(
            bridge.generation_calls,
            [
                ("llama3-8b-inst:fresh-seed-17", True),
                ("llama3-8b-inst:fresh-seed-17", True),
            ],
        )
        raw = json.loads((output / "context_manifest.json").read_text())
        summary = json.loads((output / "summary.json").read_text())
        terminal = json.loads((output / "terminal_manifest.json").read_text())
        self.assertEqual(raw["repeats"][0]["templates"], [list(x) for x in _TEMPLATES])
        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(summary["sampling"]["exact_match"])
        self.assertEqual(summary["sampling"]["repeat_count"], 2)
        self.assertEqual(terminal["status"], "PASS")
        for record in terminal["files"]:
            path = output / record["name"]
            self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["size"], path.stat().st_size)
        self.assertNotIn("probe-secret-alpha", stdout)

    def test_repeat_mismatch_is_persisted_and_fails_closed(self) -> None:
        changed = (_TEMPLATES[0], (*_TEMPLATES[1][:-1], "probe-secret-zeta {}"))
        code, output, bridge, seeds, _stdout = self._run(
            repeats=[_TEMPLATES, changed]
        )
        self.assertEqual(code, 4)
        self.assertEqual(seeds, [17, 17])
        self.assertEqual(len(bridge.generation_calls), 2)
        summary = json.loads((output / "summary.json").read_text())
        terminal = json.loads((output / "terminal_manifest.json").read_text())
        self.assertFalse(summary["sampling"]["exact_match"])
        self.assertEqual(summary["status"], "NONDETERMINISTIC_CONTEXT_HOLD")
        self.assertEqual(terminal["status"], "NONDETERMINISTIC_CONTEXT_HOLD")

    def test_summary_and_terminal_are_raw_free(self) -> None:
        _code, output, _bridge, _seeds, _stdout = self._run()
        summary_bytes = (output / "summary.json").read_text()
        terminal_bytes = (output / "terminal_manifest.json").read_text()
        for marker in ("probe-secret-alpha", "probe-secret-epsilon"):
            self.assertNotIn(marker, summary_bytes)
            self.assertNotIn(marker, terminal_bytes)
        summary = json.loads(summary_bytes)
        self.assertFalse(summary["scope"]["raw_templates_persisted_in_summary"])
        self.assertTrue(summary["scope"]["no_edit"])
        self.assertTrue(summary["scope"]["no_direct_z"])
        self.assertTrue(summary["scope"]["no_evaluation"])

    def test_output_collision_and_offline_gate_fail_before_load(self) -> None:
        name = "session02-bf16-context-lock-probe-v1-collision"
        output = self._args("llama3-8b-inst", name).output_root
        output.mkdir()
        loaded: list[str] = []
        with self.assertRaises(FileExistsError):
            probe.execute_probe(
                self._args("llama3-8b-inst", name),
                runtime_loader=lambda alias: loaded.append(alias),
                bridge_factory=lambda _root: self.fail("bridge must not load"),
                repo_root=self.repo,
            )
        self.assertEqual(loaded, [])

        with mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "0"}, clear=False):
            with self.assertRaisesRegex(probe.ContextProbeError, "HF_HUB_OFFLINE"):
                probe.execute_probe(
                    self._args(
                        "llama3-8b-inst",
                        "session02-bf16-context-lock-probe-v1-offline",
                    ),
                    runtime_loader=lambda alias: loaded.append(alias),
                    bridge_factory=lambda _root: self.fail("bridge must not load"),
                    repo_root=self.repo,
                )
        self.assertEqual(loaded, [])

    def test_dtype_and_config_mismatches_fail_before_generation(self) -> None:
        for suffix, runtime in (
            ("fp32", _FakeRuntime("llama3-8b-inst", parameter_dtype=torch.float32)),
            ("config", _FakeRuntime("llama3-8b-inst", config_dtype=torch.float32)),
        ):
            bridge = _FakeBridge([_TEMPLATES, _TEMPLATES])
            with self.assertRaisesRegex(probe.ContextProbeError, "BF16|config dtype"):
                probe.execute_probe(
                    self._args(
                        "llama3-8b-inst",
                        f"session02-bf16-context-lock-probe-v1-{suffix}",
                    ),
                    runtime_loader=lambda _alias, value=runtime: value,
                    bridge_factory=lambda _root, value=bridge: value,
                    repo_root=self.repo,
                    git_head_fn=lambda _repo: "b" * 40,
                )
            self.assertEqual(bridge.generation_calls, [])

    def test_both_aliases_share_schema_and_policy(self) -> None:
        summaries = []
        for alias in probe.MODEL_ALIASES:
            _code, output, _bridge, _seeds, _stdout = self._run(
                alias=alias,
                name=f"session02-bf16-context-lock-probe-v1-{alias}",
            )
            summaries.append(json.loads((output / "summary.json").read_text()))
        self.assertEqual(set(summaries[0]), set(summaries[1]))
        self.assertEqual(set(summaries[0]["model"]), set(summaries[1]["model"]))
        self.assertEqual(summaries[0]["sampling"]["seed"], 17)
        self.assertEqual(summaries[0]["sampling"]["repeat_count"], 2)
        self.assertEqual(
            summaries[0]["model"]["dtype_policy"],
            summaries[1]["model"]["dtype_policy"],
        )

    def test_ast_uses_only_original_loader_and_has_no_forbidden_access(self) -> None:
        source_path = Path(probe.__file__).resolve()
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imported_names = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertIn("load_fixed_model_checkpoint_original", imported_names)
        self.assertNotIn("load_fixed_model", imported_names)
        self.assertFalse(
            any(
                marker in module
                for module in imported_modules
                for marker in ("direct_z", "preflight", "evaluation")
            )
        )
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("sbatch", calls)

    def test_sbatch_is_fail_closed_and_locks_resource_shape(self) -> None:
        sbatch = Path(probe.__file__).with_name(
            "session02_bf16_context_lock_probe.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("ODEEDIT_CONTEXT_PROBE_SUBMISSION_AUTHORIZED", sbatch)
        self.assertIn('= "bf16-context-lock-probe-v1"', sbatch)
        self.assertIn("#SBATCH --gres=gpu:1", sbatch)
        self.assertIn("#SBATCH --cpus-per-task=8", sbatch)
        self.assertIn("#SBATCH --mem=65000M", sbatch)
        self.assertIn("#SBATCH --time=00:30:00", sbatch)
        self.assertNotIn("sbatch ", sbatch)


if __name__ == "__main__":
    unittest.main()
