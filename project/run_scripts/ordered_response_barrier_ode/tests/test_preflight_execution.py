from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from project.run_scripts.ordered_response_barrier_ode import artifacts, launch, preflight
from project.run_scripts.ordered_response_barrier_ode.tests.test_artifacts import (
    _round as raw_round_fixture,
)


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


class MappingTests(unittest.TestCase):
    def test_cell_and_wave_mapping_is_exact(self) -> None:
        self.assertEqual(
            [(item.cell_id, item.model_alias, item.writer_family) for item in preflight.CELL_SPECS],
            [
                (0, "llama3-8b-inst", "MEMIT"),
                (1, "llama3-8b-inst", "AlphaEdit"),
                (2, "qwen2.5-7b-inst", "MEMIT"),
                (3, "qwen2.5-7b-inst", "AlphaEdit"),
            ],
        )
        self.assertEqual(preflight.wave_rounds("b1"), (0,))
        self.assertEqual(preflight.wave_rounds("round0"), (0,))
        self.assertEqual(preflight.wave_rounds("remaining"), tuple(range(1, 10)))
        with self.assertRaises(preflight.PreflightBoundary):
            preflight.cell_spec(4)
        with self.assertRaises(preflight.PreflightBoundary):
            preflight.wave_rounds("all")

    def test_authoritative_input_identity_and_symlink_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            payload = b"one\ntwo\n"
            relative = "proposal.md"
            _write(root / relative, payload)
            expected = {
                relative: {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "lines": 2,
                }
            }
            with mock.patch.dict(preflight.AUTHORITATIVE_INPUTS, expected, clear=True):
                receipt = preflight.validate_authoritative_inputs(root)
                self.assertEqual(receipt[0]["lines"], 2)
                (root / relative).unlink()
                (root / relative).symlink_to(root / "outside")
                with self.assertRaises(preflight.PreflightBoundary):
                    preflight.validate_authoritative_inputs(root)


class SourceAndArtifactTests(unittest.TestCase):
    def test_execution_source_untracked_shadow_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "fixture"], cwd=root, check=True)
            _write(root / "README", b"fixture\n")
            subprocess.run(["git", "add", "README"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()
            self.assertTrue(preflight.validate_source_checkout(root, head, tree)["tracked_clean"])
            _write(
                root / "project/run_scripts/ordered_response_barrier_ode/shadow.py",
                b"raise RuntimeError('shadow')\n",
            )
            with self.assertRaises(preflight.PreflightBoundary):
                preflight.validate_source_checkout(root, head, tree)

    def test_clean_easyedit_source_members(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "fixture"], cwd=root, check=True)
            member = "easyeditor/models/memit/memit_main.py"
            payload = b"fixture\n"
            _write(root / member, payload)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
            head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True).strip()
            with (
                mock.patch.object(preflight, "OFFICIAL_EASYEDIT_HEAD", head),
                mock.patch.object(preflight, "OFFICIAL_EASYEDIT_TREE", tree),
                mock.patch.dict(
                    preflight.OFFICIAL_SOURCE_MEMBERS,
                    {member: hashlib.sha256(payload).hexdigest()},
                    clear=True,
                ),
            ):
                self.assertTrue(preflight.validate_easyedit_source(root)["tracked_clean"])
                (root / member).write_text("mutated\n", encoding="utf-8")
                with self.assertRaises(preflight.PreflightBoundary):
                    preflight.validate_easyedit_source(root)

    def test_small_model_asset_fixture_shallow_and_deep(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            top = Path(raw)
            repo = top / "repo"
            artifact = top / "artifact"
            hub = top / "hub"
            repo.mkdir()
            artifact.mkdir()
            hub.mkdir()
            hparams = b"alg_name: MEMIT\n"
            projector = b"projector"
            statistic = b"stat"
            config = b"config"
            tokenizer = b"tokenizer"
            _write(repo / "config/memit.yaml", hparams)
            _write(artifact / "projector.pt", projector)
            _write(artifact / "examples/data/stats/Fixture/wikipedia_stats/model.layers.4.mlp.down_proj_float32_mom2_100000.npz", statistic)
            blobs = hub / "blobs"
            _write(blobs / "config", config)
            _write(blobs / "tokenizer", tokenizer)
            snapshot = hub / "models--fixture/snapshots/revision"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").symlink_to(blobs / "config")
            (snapshot / "tokenizer.json").symlink_to(blobs / "tokenizer")
            binding = {
                "fixture": {
                    "native_name": "fixture",
                    "revision": "revision",
                    "hf_repo": "models--fixture",
                    "config_sha256": hashlib.sha256(config).hexdigest(),
                    "tokenizer_sha256": hashlib.sha256(tokenizer).hexdigest(),
                    "hparams": {"MEMIT": ("config/memit.yaml", hashlib.sha256(hparams).hexdigest())},
                    "projector": ("projector.pt", len(projector), hashlib.sha256(projector).hexdigest()),
                    "statistics": {4: (len(statistic), hashlib.sha256(statistic).hexdigest())},
                    "statistics_model_dir": "Fixture",
                }
            }
            with mock.patch.dict(preflight.MODEL_BINDINGS, binding, clear=True):
                shallow = preflight.validate_model_artifacts(repo, artifact, hub, deep_hash=False)
                self.assertEqual(
                    shallow["models"]["fixture"]["projector"]["observed_sha256"],
                    "DEFERRED_TO_PRE_GPU_DEEP_HASH",
                )
                deep = preflight.validate_model_artifacts(repo, artifact, hub, deep_hash=True)
                self.assertEqual(
                    deep["models"]["fixture"]["projector"]["observed_sha256"],
                    hashlib.sha256(projector).hexdigest(),
                )


class WaveGateTests(unittest.TestCase):
    @staticmethod
    def _canonical_b1_round() -> dict[str, object]:
        value = raw_round_fixture(1)

        def replace_case_ids(item: object) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    if key == "case_id":
                        item[key] = preflight.B1_CASE_ID
                    else:
                        replace_case_ids(child)
            elif isinstance(item, list):
                for child in item:
                    replace_case_ids(child)

        replace_case_ids(value)
        value["case_ids"] = [preflight.B1_CASE_ID]
        value["request_sha256"] = [preflight.B1_REQUEST_SHA256]
        value["canonical_request_order_sha256"] = preflight.B1_REQUEST_ORDER_SHA256
        preamble = value["runtime_preamble"]
        assert isinstance(preamble, dict)
        preamble["request_case_id"] = str(preflight.B1_CASE_ID)
        preamble["request_sha256"] = preflight.B1_REQUEST_SHA256
        return value

    def _result(
        self,
        cell: int,
        round_publication: dict[str, object],
        round_file_sha256: str,
        *,
        wave: str = "round0",
        request_count: int = 100,
    ) -> dict[str, object]:
        native_memit_exception = preflight.cell_spec(cell).writer_family == "MEMIT"
        publication_receipt = {
            "round_index": 0,
            "relative_path": "round-00-result.json",
            "file_sha256": round_file_sha256,
            "identity_sha256": round_publication["identity_sha256"],
        }
        return {
            "schema": "orbode.server1.cell-result.v1",
            "status": "TERMINAL_VALID",
            "cell_id": cell,
            "model_alias": preflight.cell_spec(cell).model_alias,
            "writer_family": preflight.cell_spec(cell).writer_family,
            "wave": wave,
            "rounds": [0],
            "request_count": request_count,
            "primary_arms": list(preflight.PRIMARY_ARMS),
            "primary_endpoint_count": request_count * 5,
            "scientific_attempted_count": request_count * 5,
            "terminal_valid_count": request_count * 5,
            "technical_failure_count": 0,
            "fixed_z_compute_count": request_count,
            "fixed_z_recompute_count": 0,
            "source_head": "h",
            "source_tree": "t",
            "stream_root": preflight.STREAM_ROOT,
            "order_root": preflight.ORDER_ROOT,
            "full_fp32": True,
            "full_fp32_claim_scope": "MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH",
            "official_native_memit_ephemeral_fp64_solve_exception": native_memit_exception,
            "all_algorithm_solves_full_fp32": not native_memit_exception,
            "unqualified_full_fp32_claim": not native_memit_exception,
            "fast_runtime_preamble_count": 1,
            "fast_runtime_preamble_status": "PASS_THIS_WAVE",
            "round_results": [round_publication],
            "round_publication_receipts": [publication_receipt],
            "round_publication_count": 1,
            "round_publication_identity_root": preflight.canonical_hash(
                [round_publication["identity_sha256"]]
            ),
            "round_publication_file_sha256_root": preflight.canonical_hash(
                [round_file_sha256]
            ),
            "literal_prompt_target_token_prediction_publication_count": 0,
        }

    def _terminal(
        self,
        cell: int,
        *,
        result_sha256: str,
        round_identity: str,
        round_file_sha256: str,
        wave: str = "round0",
        request_count: int = 100,
    ) -> dict[str, object]:
        native_memit_exception = preflight.cell_spec(cell).writer_family == "MEMIT"
        value: dict[str, object] = {
            "schema": preflight.ROUND0_SCHEMA,
            "status": "TERMINAL_VALID",
            "cell_id": cell,
            "model_alias": preflight.cell_spec(cell).model_alias,
            "writer_family": preflight.cell_spec(cell).writer_family,
            "wave": wave,
            "rounds": [0],
            "request_count": request_count,
            "primary_arms": list(preflight.PRIMARY_ARMS),
            "primary_endpoint_count": request_count * 5,
            "scientific_attempted_count": request_count * 5,
            "terminal_valid_count": request_count * 5,
            "technical_failure_count": 0,
            "entry_already_hit_count": cell,
            "fixed_z_compute_count": request_count,
            "fixed_z_recompute_count": 0,
            "model_reload_wave_count": 1,
            "source_head": "h",
            "source_tree": "t",
            "stream_root": preflight.STREAM_ROOT,
            "order_root": preflight.ORDER_ROOT,
            "full_fp32": True,
            "full_fp32_claim_scope": "MODEL_STORAGE_AND_ORBODE_DYNAMIC_PATH",
            "official_native_memit_ephemeral_fp64_solve_exception": native_memit_exception,
            "all_algorithm_solves_full_fp32": not native_memit_exception,
            "unqualified_full_fp32_claim": not native_memit_exception,
            "w0_restore_pass": True,
            "cache_restore_pass": True,
            "transaction_pass": True,
            "jvp_gate_pass": True,
            "overlay_gate_pass": True,
            "result_sha256": result_sha256,
            "round_publication_count": 1,
            "round_publication_identity_root": preflight.canonical_hash(
                [round_identity]
            ),
            "round_publication_file_sha256_root": preflight.canonical_hash(
                [round_file_sha256]
            ),
            "literal_prompt_target_token_prediction_publication_count": 0,
        }
        value["receipt_identity_sha256"] = preflight.canonical_hash(value)
        return value

    def _write_cell(
        self,
        root: Path,
        cell: int,
        round_publication: dict[str, object],
        *,
        wave: str,
        request_count: int,
    ) -> None:
        task = root / f"task-{cell}"
        task.mkdir(parents=True, exist_ok=True)
        round_path = task / "round-00-result.json"
        round_path.write_text(
            preflight.canonical_json(round_publication) + "\n", encoding="utf-8"
        )
        round_path.chmod(0o600)
        round_file_sha256 = preflight.sha256_file(round_path)
        result_path = task / "result.json"
        result_path.write_text(
            preflight.canonical_json(
                self._result(
                    cell,
                    round_publication,
                    round_file_sha256,
                    wave=wave,
                    request_count=request_count,
                )
            )
            + "\n",
            encoding="utf-8",
        )
        result_path.chmod(0o600)
        terminal = self._terminal(
            cell,
            result_sha256=preflight.sha256_file(result_path),
            round_identity=str(round_publication["identity_sha256"]),
            round_file_sha256=round_file_sha256,
            wave=wave,
            request_count=request_count,
        )
        terminal_path = task / "terminal-receipt.json"
        terminal_path.write_text(
            preflight.canonical_json(terminal) + "\n", encoding="utf-8"
        )
        terminal_path.chmod(0o600)

    def test_b1_common_gate_binds_exact_first_request_and_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for cell in range(4):
                publication = artifacts.reduce_round_payload(
                    self._canonical_b1_round(), expected_request_count=1
                )
                self._write_cell(
                    root, cell, publication, wave="b1", request_count=1
                )
            gate = preflight.validate_b1_common_gate(
                root, source_head="h", source_tree="t"
            )
            self.assertEqual(gate["status"], "B1_COMMON_INTEGRITY_PASS")
            self.assertEqual(gate["request_count"], 4)
            self.assertEqual(gate["endpoint_count"], 20)

            noncanonical = artifacts.reduce_round_payload(
                raw_round_fixture(1), expected_request_count=1
            )
            self._write_cell(
                root, 0, noncanonical, wave="b1", request_count=1
            )
            with self.assertRaisesRegex(
                preflight.PreflightBoundary, "canonical first-request identity"
            ):
                preflight.validate_b1_common_gate(
                    root, source_head="h", source_tree="t"
                )

    def test_round0_preflight_requires_and_binds_b1_common_gate(self) -> None:
        arguments = {
            "repo_root": Path("/repo"),
            "authoritative_root": preflight.AUTHORITATIVE_ROOT,
            "easyedit_source_root": preflight.OFFICIAL_EASYEDIT_ROOT,
            "easyedit_artifact_root": preflight.EASYEDIT_ARTIFACT_ROOT,
            "hf_hub_cache": preflight.HF_HUB_CACHE_ROOT,
            "source_head": "h",
            "source_tree": "t",
            "cell_id": 0,
            "wave": "round0",
            "deep_artifact_hash": False,
        }
        with self.assertRaisesRegex(
            preflight.PreflightBoundary, "requires a B1 common-gate root"
        ):
            preflight.run_preflight(**arguments)

        gate = {
            "status": "B1_COMMON_INTEGRITY_PASS",
            "cell_count": 4,
            "request_count": 4,
            "endpoint_count": 20,
        }
        with (
            mock.patch.object(preflight, "validate_b1_common_gate", return_value=gate),
            mock.patch.object(preflight, "validate_authoritative_inputs", return_value=[]),
            mock.patch.object(preflight, "validate_easyedit_source", return_value={}),
            mock.patch.object(preflight, "validate_execution_lock", return_value={}),
            mock.patch.object(preflight, "validate_model_artifacts", return_value={}),
            mock.patch.object(preflight, "validate_source_checkout", return_value={}),
            mock.patch.object(preflight, "validate_stream", return_value={}),
        ):
            receipt = preflight.run_preflight(
                **arguments, b1_root=Path("/sealed/b1")
            )
        self.assertEqual(receipt["b1_common_gate"], gate)
        self.assertIsNone(receipt["round0_common_gate"])

    def test_four_cell_common_gate_and_create_once_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for cell in range(4):
                path = root / f"task-{cell}/terminal-receipt.json"
                path.parent.mkdir(parents=True)
                round_path = path.parent / "round-00-result.json"
                round_publication = artifacts.reduce_round_payload(raw_round_fixture())
                round_path.write_text(
                    preflight.canonical_json(round_publication) + "\n", encoding="utf-8"
                )
                round_path.chmod(0o600)
                round_file_sha256 = preflight.sha256_file(round_path)
                result_path = path.parent / "result.json"
                result_path.write_text(
                    preflight.canonical_json(
                        self._result(cell, round_publication, round_file_sha256)
                    )
                    + "\n",
                    encoding="utf-8",
                )
                result_path.chmod(0o600)
                terminal = self._terminal(
                    cell,
                    result_sha256=preflight.sha256_file(result_path),
                    round_identity=str(round_publication["identity_sha256"]),
                    round_file_sha256=round_file_sha256,
                )
                path.write_text(preflight.canonical_json(terminal) + "\n", encoding="utf-8")
                path.chmod(0o600)
            receipt = preflight.validate_round0_common_gate(root, source_head="h", source_tree="t")
            self.assertEqual(receipt["endpoint_count"], 2000)
            target = root / "shared-preflight.json"
            value = {"status": "PASS"}
            launch.write_create_once_json(target, value)
            self.assertEqual(target.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(preflight.PreflightBoundary):
                launch.write_create_once_json(target, value)

            nested = root / "task-1/round-00-result.json"
            nested_value = json.loads(nested.read_text(encoding="utf-8"))
            nested_value["primary_endpoints"][0]["mechanism_telemetry"]["telemetry"][
                "factor_build_count"
            ] = 999
            nested.write_text(
                preflight.canonical_json(nested_value) + "\n", encoding="utf-8"
            )
            with self.assertRaises(preflight.PreflightBoundary):
                preflight.validate_round0_common_gate(root, source_head="h", source_tree="t")
            round_publication = artifacts.reduce_round_payload(raw_round_fixture())
            nested.write_text(
                preflight.canonical_json(round_publication) + "\n", encoding="utf-8"
            )
            nested.chmod(0o600)

            bad = root / "task-2/terminal-receipt.json"
            round_path = root / "task-2/round-00-result.json"
            result_path = root / "task-2/result.json"
            round_value = json.loads(round_path.read_text(encoding="utf-8"))
            bad_value = self._terminal(
                2,
                result_sha256=preflight.sha256_file(result_path),
                round_identity=str(round_value["identity_sha256"]),
                round_file_sha256=preflight.sha256_file(round_path),
            )
            bad_value["technical_failure_count"] = 1
            bad.write_text(preflight.canonical_json(bad_value) + "\n", encoding="utf-8")
            with self.assertRaises(preflight.PreflightBoundary):
                preflight.validate_round0_common_gate(root, source_head="h", source_tree="t")

    def test_shared_preflight_receipt_is_identity_and_wave_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "preflight.json"
            value: dict[str, object] = {
                "instruction_id": preflight.INSTRUCTION_ID,
                "status": "PRE_GPU_BINDING_PASS",
                "wave": {"name": "b1", "round_indices": [0]},
                "stream": {
                    "stream_root": preflight.STREAM_ROOT,
                    "order_root": preflight.ORDER_ROOT,
                },
                "source": {"head": "h", "tree": "t"},
                "full_fp32_required": True,
                "model_artifacts": {"deep_hash": True},
            }
            value["receipt_identity_sha256"] = preflight.canonical_hash(value)
            path.write_text(preflight.canonical_json(value) + "\n", encoding="utf-8")
            path.chmod(0o600)
            loaded = launch.load_shared_preflight(
                path, source_head="h", source_tree="t", wave="b1"
            )
            self.assertEqual(loaded["status"], "PRE_GPU_BINDING_PASS")
            with self.assertRaises(preflight.PreflightBoundary):
                launch.load_shared_preflight(
                    path, source_head="h", source_tree="t", wave="remaining"
                )

            value["wave"] = {"name": "round0", "round_indices": [0]}
            value["b1_common_gate"] = {
                "status": "B1_COMMON_INTEGRITY_PASS",
                "endpoint_count": 20,
            }
            value["round0_common_gate"] = None
            value.pop("receipt_identity_sha256")
            value["receipt_identity_sha256"] = preflight.canonical_hash(value)
            path.write_text(preflight.canonical_json(value) + "\n", encoding="utf-8")
            self.assertEqual(
                launch.load_shared_preflight(
                    path, source_head="h", source_tree="t", wave="round0"
                )["wave"]["name"],
                "round0",
            )


if __name__ == "__main__":
    unittest.main()
