from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from project.run_scripts.alphaedit_runtime_path_seal import (
    EXPECTED_MODEL_BINDINGS,
    RuntimeIdentity,
    RuntimePathSealError,
    canonical_hash,
    load_alphaedit_runtime_path_seal,
    sha256_file,
)
from project.run_scripts.ode_alloc.contracts import ODEAllocContractError
from project.run_scripts.ode_alloc.p0_artifacts import (
    P0ArtifactGuard,
    load_artifact_lock,
    validate_artifact_lock_structure,
)
from project.run_scripts.ode_bf.artifacts import ODEBFArtifactGuard
from project.run_scripts.ode_bf.contracts import canonical_hash as bf_canonical_hash


ROOT = Path(__file__).resolve().parents[4]
LEGACY_EASYEDIT = "/mnt/raid5/janghj/EasyEdit"
LEGACY_HF = "/mnt/raid5/janghj/.cache/huggingface/hub"
LEGACY_EVALUATOR = "/mnt/raid5/janghj/00.KE/00.Experiment/00.EAIR_parametric/AlphaEdit"
ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")


class RuntimePathSealTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root / "repo"
        self.easyedit = self.root / "EasyEdit"
        self.hf = self.root / "hub"
        self.evaluator = self.root / "evaluator"
        for path in (self.repo, self.easyedit, self.hf, self.evaluator):
            path.mkdir(parents=True)
        self.seal_path = self.repo / "server4-seal.json"
        self.seal_value = self._build_seal()
        self._write_rooted(self.seal_path, self.seal_value)
        self.base_lock, self.base_value = self._build_base_lock()
        self.bf_lock, self.bf_value = self._build_bf_lock()

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write_rooted(path: Path, value: dict[str, object]) -> None:
        rooted = copy.deepcopy(value)
        rooted.pop("root_digest", None)
        rooted["root_digest"] = canonical_hash(rooted)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(rooted, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_members(self, alias: str, dimension: int) -> list[dict[str, object]]:
        model_name = (
            "Meta-Llama-3-8B-Instruct"
            if alias == "llama3-8b-inst"
            else "Qwen2.5-7B-Instruct"
        )
        hparams = self.easyedit / f"hparams/AlphaEdit/{alias}.yaml"
        hparams.parent.mkdir(parents=True, exist_ok=True)
        hparams.write_text(f"alias: {alias}\n", encoding="utf-8")
        projector = self.easyedit / f"examples/null_space_project_{model_name}.pt"
        projector.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.eye(dimension).repeat(5, 1, 1).float(), projector)
        members: list[dict[str, object]] = [
            {
                "kind": "hparams",
                "layer": None,
                "relative_path": hparams.relative_to(self.easyedit).as_posix(),
                "size": hparams.stat().st_size,
                "sha256": self._sha(hparams),
                "shape": [],
                "dtype": None,
            },
            {
                "kind": "projector",
                "layer": None,
                "relative_path": projector.relative_to(self.easyedit).as_posix(),
                "size": projector.stat().st_size,
                "sha256": self._sha(projector),
                "shape": [5, dimension, dimension],
                "dtype": "float32",
            },
        ]
        for layer in range(4, 9):
            covariance = (
                self.easyedit
                / "examples/data/stats"
                / model_name
                / "wikipedia_stats"
                / f"model.layers.{layer}.mlp.down_proj_float32_mom2_100000.npz"
            )
            covariance.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                covariance,
                **{
                    "mom2.constructor": np.array("CombinedStat"),
                    "mom2.count": np.array(100, dtype=np.int64),
                    "mom2.mom2": np.eye(dimension, dtype=np.float32),
                    "sample_size": np.array(100, dtype=np.int64),
                },
            )
            members.append(
                {
                    "kind": "covariance",
                    "layer": layer,
                    "relative_path": covariance.relative_to(self.easyedit).as_posix(),
                    "size": covariance.stat().st_size,
                    "sha256": self._sha(covariance),
                    "shape": [dimension, dimension],
                    "dtype": "float32",
                }
            )
        return members

    def _build_seal(self) -> dict[str, object]:
        models: dict[str, object] = {}
        for alias, dimension in zip(ALIASES, (4, 6), strict=True):
            binding = EXPECTED_MODEL_BINDINGS[alias]
            members = self._write_members(alias, dimension)
            manifest = {
                "alias": alias,
                "layers": [4, 5, 6, 7, 8],
                "members": members,
                "native_name": binding["native_name"],
                "representation": "model.layers.{L}.mlp.down_proj",
                "revision": binding["revision"],
            }
            models[alias] = {
                "native_name": binding["native_name"],
                "revision": binding["revision"],
                "layers": [4, 5, 6, 7, 8],
                "representation": "model.layers.{L}.mlp.down_proj",
                "bundle_sha256": binding["bundle_sha256"],
                "member_manifest_sha256": canonical_hash(manifest),
                "members": members,
            }
        return {
            "schema_version": "ode-edit-alphaedit-runtime-path-seal/v1",
            "seal_id": "TEST-SERVER4-SEAL",
            "hostname": "server4",
            "agent_hostname": "server4",
            "agent_role": "server-head",
            "mappings": {
                "easyedit_root": {
                    "logical_root": LEGACY_EASYEDIT,
                    "runtime_root": str(self.easyedit),
                },
                "hf_hub_cache": {
                    "logical_root": LEGACY_HF,
                    "runtime_root": str(self.hf),
                },
                "alphaedit_evaluator_root": {
                    "logical_root": LEGACY_EVALUATOR,
                    "runtime_root": str(self.evaluator),
                },
            },
            "withheld_mappings": {},
            "models": models,
        }

    def _build_base_lock(self) -> tuple[Path, dict[str, object]]:
        easy_source = self.easyedit / "easy_source.py"
        easy_source.write_text("VALUE = 1\n", encoding="utf-8")
        counterfact = self.easyedit / "data/counterfact.json"
        counterfact.parent.mkdir(parents=True)
        counterfact.write_text("[]\n", encoding="utf-8")
        models: dict[str, object] = {}
        for alias in ALIASES:
            binding = EXPECTED_MODEL_BINDINGS[alias]
            repo_cache = f"models--test--{alias}"
            snapshot = self.hf / repo_cache / "snapshots" / binding["revision"]
            snapshot.mkdir(parents=True)
            memit = self.easyedit / f"hparams/MEMIT/{alias}.yaml"
            memit.parent.mkdir(parents=True, exist_ok=True)
            memit.write_text("layers: [4, 5, 6, 7, 8]\n", encoding="utf-8")
            seal_model = self.seal_value["models"][alias]
            covariance = {
                str(member["layer"]): [
                    member["relative_path"],
                    member["size"],
                    member["sha256"],
                ]
                for member in seal_model["members"]
                if member["kind"] == "covariance"
            }
            models[alias] = {
                "repo_cache": repo_cache,
                "revision": binding["revision"],
                "native_name": binding["native_name"],
                "hparams_path": memit.relative_to(self.easyedit).as_posix(),
                "hparams_sha256": self._sha(memit),
                "config_torch_dtype": "bfloat16",
                "snapshot_files": {"config.json": ["../../blobs/x", 1, None]},
                "covariance": covariance,
            }
        value: dict[str, object] = {
            "schema_version": "ode-alloc-s04-p0-artifact-lock-r1/v1",
            "instruction_id": "ODEEDIT-S04-ODE-ALLOC-P0-LOCK-R1-PAIR-V1",
            "easyedit_root": LEGACY_EASYEDIT,
            "hf_hub_cache": LEGACY_HF,
            "easyedit_sources": {
                easy_source.relative_to(self.easyedit).as_posix(): self._sha(easy_source)
            },
            "counterfact": {
                "path": counterfact.relative_to(self.easyedit).as_posix(),
                "size": counterfact.stat().st_size,
                "sha256": self._sha(counterfact),
            },
            "models": models,
        }
        path = self.repo / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path, value

    def _build_bf_lock(self) -> tuple[Path, dict[str, object]]:
        models: dict[str, object] = {}
        for alias in ALIASES:
            binding = EXPECTED_MODEL_BINDINGS[alias]
            members = self.seal_value["models"][alias]["members"]
            hparams = next(item for item in members if item["kind"] == "hparams")
            projector = next(item for item in members if item["kind"] == "projector")
            models[alias] = {
                "native_name": binding["native_name"],
                "revision": binding["revision"],
                "hparams_path": hparams["relative_path"],
                "hparams_sha256": hparams["sha256"],
                "projector_path": projector["relative_path"],
                "projector_size": projector["size"],
                "projector_sha256": projector["sha256"],
                "layers": [4, 5, 6, 7, 8],
                "parameter_dtype": "torch.bfloat16",
            }
        value: dict[str, object] = {
            "schema_version": "ode-edit-s04-ode-bf-p0-artifact-lock/v1",
            "easyedit_root": LEGACY_EASYEDIT,
            "alphaedit_evaluator_root": LEGACY_EVALUATOR,
            "hf_hub_cache": LEGACY_HF,
            "models": models,
            "base_model_artifact_lock": {
                "path": self.base_lock.relative_to(self.repo).as_posix(),
                "sha256": sha256_file(self.base_lock),
            },
        }
        rooted = dict(value)
        rooted["root_digest"] = bf_canonical_hash(value)
        path = self.repo / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(rooted), encoding="utf-8")
        return path, rooted

    def _load(self):
        identity = RuntimeIdentity("server4", "server4", "server-head")
        with mock.patch(
            "project.run_scripts.alphaedit_runtime_path_seal.detect_runtime_identity",
            return_value=identity,
        ):
            return load_alphaedit_runtime_path_seal(
                self.seal_path,
                repo_root=self.repo,
            )

    def test_default_none_preserves_legacy_lock_identity_and_rejection(self) -> None:
        original = load_artifact_lock(self.base_lock)
        explicit_none = load_artifact_lock(self.base_lock, runtime_path_seal=None)
        self.assertEqual(original, explicit_none)
        self.assertEqual(
            validate_artifact_lock_structure(copy.deepcopy(self.base_value)),
            self.base_value,
        )
        changed = copy.deepcopy(self.base_value)
        changed["easyedit_root"] = str(self.easyedit)
        with self.assertRaisesRegex(ODEAllocContractError, "root differs"):
            validate_artifact_lock_structure(changed)

    def test_server4_seal_positive_and_guard_delivery(self) -> None:
        seal = self._load()
        for alias in ALIASES:
            seal.assert_p0_model_contract(alias, self.base_value["models"][alias])
            seal.assert_odebf_model_contract(alias, self.bf_value["models"][alias])
            receipt = seal.preflight_alias(alias)
            self.assertEqual(receipt.resolved_runtime_path, str(self.easyedit))
            self.assertEqual(receipt.verified_member_count, 7)
            p0 = P0ArtifactGuard(
                self.base_lock,
                alias,
                runtime_path_seal=seal,
            )
            bf = ODEBFArtifactGuard(
                self.repo,
                self.bf_lock,
                alias,
                require_held_ode_alloc=False,
                runtime_path_seal=seal,
            )
            self.assertIs(p0.runtime_path_seal, seal)
            self.assertIs(bf.runtime_path_seal, seal)
            self.assertIs(bf.base_guard.runtime_path_seal, seal)
            self.assertEqual(p0.easyedit_root, self.easyedit)
            self.assertEqual(bf.easyedit_root, self.easyedit)

    def test_wrong_host_and_role_fail_closed(self) -> None:
        for identity in (
            RuntimeIdentity("server2", "server4", "server-head"),
            RuntimeIdentity("server4", "server4", "worker"),
        ):
            with self.subTest(identity=identity), mock.patch(
                "project.run_scripts.alphaedit_runtime_path_seal.detect_runtime_identity",
                return_value=identity,
            ), self.assertRaisesRegex(RuntimePathSealError, "host or agent role"):
                load_alphaedit_runtime_path_seal(self.seal_path, repo_root=self.repo)

    def test_wrong_logical_root_and_symlink_runtime_root_fail_closed(self) -> None:
        seal = self._load()
        with self.assertRaisesRegex(RuntimePathSealError, "logical locked root"):
            seal.resolve_root("easyedit_root", "/different/EasyEdit")
        linked = self.root / "linked-easyedit"
        linked.symlink_to(self.easyedit, target_is_directory=True)
        changed = copy.deepcopy(self.seal_value)
        changed["mappings"]["easyedit_root"]["runtime_root"] = str(linked)
        self._write_rooted(self.seal_path, changed)
        with mock.patch(
            "project.run_scripts.alphaedit_runtime_path_seal.detect_runtime_identity",
            return_value=RuntimeIdentity("server4", "server4", "server-head"),
        ), self.assertRaisesRegex(RuntimePathSealError, "unresolved or symlinked"):
            load_alphaedit_runtime_path_seal(self.seal_path, repo_root=self.repo)

        empty = self.root / "empty-easyedit"
        empty.mkdir()
        changed["mappings"]["easyedit_root"]["runtime_root"] = str(empty)
        self._write_rooted(self.seal_path, changed)
        seal = self._load()
        with self.assertRaisesRegex(RuntimePathSealError, "unresolved or corrupt"):
            seal.preflight_alias("llama3-8b-inst")

    def test_bundle_member_hash_and_shape_mismatch_fail_closed(self) -> None:
        changed = copy.deepcopy(self.seal_value)
        changed["models"]["llama3-8b-inst"]["bundle_sha256"] = "0" * 64
        self._write_rooted(self.seal_path, changed)
        with mock.patch(
            "project.run_scripts.alphaedit_runtime_path_seal.detect_runtime_identity",
            return_value=RuntimeIdentity("server4", "server4", "server-head"),
        ), self.assertRaisesRegex(RuntimePathSealError, "model binding"):
            load_alphaedit_runtime_path_seal(self.seal_path, repo_root=self.repo)

        self._write_rooted(self.seal_path, self.seal_value)
        seal = self._load()
        member = seal.model("llama3-8b-inst").members[0]
        (self.easyedit / member.relative_path).write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimePathSealError, "SHA256 or size"):
            seal.preflight_alias("llama3-8b-inst")

        (self.easyedit / member.relative_path).write_text(
            "alias: llama3-8b-inst\n",
            encoding="utf-8",
        )
        changed = copy.deepcopy(self.seal_value)
        projector = changed["models"]["llama3-8b-inst"]["members"][1]
        projector["shape"] = [5, 5, 5]
        model = changed["models"]["llama3-8b-inst"]
        manifest = {
            "alias": "llama3-8b-inst",
            "layers": model["layers"],
            "members": model["members"],
            "native_name": model["native_name"],
            "representation": model["representation"],
            "revision": model["revision"],
        }
        model["member_manifest_sha256"] = canonical_hash(manifest)
        self._write_rooted(self.seal_path, changed)
        seal = self._load()
        with self.assertRaisesRegex(RuntimePathSealError, "shape or dtype"):
            seal.preflight_alias("llama3-8b-inst")

    def test_alias_and_locked_member_mismatch_fail_closed(self) -> None:
        changed = copy.deepcopy(self.seal_value)
        del changed["models"]["qwen2.5-7b-inst"]
        self._write_rooted(self.seal_path, changed)
        with mock.patch(
            "project.run_scripts.alphaedit_runtime_path_seal.detect_runtime_identity",
            return_value=RuntimeIdentity("server4", "server4", "server-head"),
        ), self.assertRaisesRegex(RuntimePathSealError, "aliases"):
            load_alphaedit_runtime_path_seal(self.seal_path, repo_root=self.repo)

        self._write_rooted(self.seal_path, self.seal_value)
        seal = self._load()
        spec = copy.deepcopy(self.bf_value["models"]["llama3-8b-inst"])
        spec["projector_sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimePathSealError, "members differ"):
            seal.assert_odebf_model_contract("llama3-8b-inst", spec)

    def test_tracked_server4_seal_maps_only_easyedit(self) -> None:
        seal = load_alphaedit_runtime_path_seal(
            ROOT / "agents/server4/alphaedit-runtime-path-seal.json",
            repo_root=ROOT,
        )
        self.assertEqual(
            seal.resolve_root("easyedit_root", LEGACY_EASYEDIT),
            Path("/data/janghj/EasyEdit"),
        )
        self.assertFalse(seal.has_mapping("hf_hub_cache"))
        self.assertFalse(seal.has_mapping("alphaedit_evaluator_root"))
        with self.assertRaises(FileNotFoundError):
            P0ArtifactGuard(
                ROOT / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
                "llama3-8b-inst",
                runtime_path_seal=None,
            )
        with self.assertRaises(FileNotFoundError):
            ODEBFArtifactGuard(
                ROOT,
                ROOT / "project/run_scripts/ode_bf/locks/p0_artifact_lock.json",
                "llama3-8b-inst",
                runtime_path_seal=None,
            )


if __name__ == "__main__":
    unittest.main()
