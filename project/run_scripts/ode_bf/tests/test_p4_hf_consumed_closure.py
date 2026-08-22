from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from project.run_scripts.alphaedit_runtime_path_seal import (
    RuntimeIdentity,
    canonical_hash,
)
from project.run_scripts.ode_bf.p4_hf_consumed_closure import (
    ClosureMember,
    P4HFClosureError,
    PinnedModelClosure,
    _closure_identity,
    _member_root,
    load_p4_full_fp32_from_sealed_snapshot,
    load_p4_hf_consumed_closure_seal,
)


IDENTITY = RuntimeIdentity("server4", "server4", "server-head")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class P4HFConsumedClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.hf = self.root / "hub"
        self.hf.mkdir()
        self.source = self.root / "loader.py"
        self.source.write_text("PINNED = True\n", encoding="utf-8")
        self.models: dict[str, dict[str, object]] = {}
        self._make_model(
            alias="llama3-8b-inst",
            native="meta-llama/Meta-Llama-3-8B-Instruct",
            cache="models--meta-llama--Meta-Llama-3-8B-Instruct",
            revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
            model_type="llama",
            architecture="LlamaForCausalLM",
            tokenizer_class="PreTrainedTokenizerFast",
            tokenizer_files=("special_tokens_map.json", "tokenizer.json"),
        )
        self._make_model(
            alias="qwen2.5-7b-inst",
            native="Qwen/Qwen2.5-7B-Instruct",
            cache="models--Qwen--Qwen2.5-7B-Instruct",
            revision="a09a35458c702b33eeacc393d103063234e8bc28",
            model_type="qwen2",
            architecture="Qwen2ForCausalLM",
            tokenizer_class="Qwen2Tokenizer",
            tokenizer_files=("merges.txt", "tokenizer.json", "vocab.json"),
        )
        self.seal_path = self.root / "seal.json"
        self._write_seal()
        self.identity_patch = mock.patch(
            "project.run_scripts.ode_bf.p4_hf_consumed_closure.detect_runtime_identity",
            return_value=IDENTITY,
        )
        self.identity_patch.start()

    def tearDown(self) -> None:
        self.identity_patch.stop()
        self.temporary.cleanup()

    def _make_blob_link(self, cache: str, snapshot: Path, name: str, raw: bytes) -> None:
        blob = self.hf / cache / "blobs" / hashlib.sha256(raw).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(raw)
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(os.path.relpath(blob, destination.parent))

    def _make_model(
        self,
        *,
        alias: str,
        native: str,
        cache: str,
        revision: str,
        model_type: str,
        architecture: str,
        tokenizer_class: str,
        tokenizer_files: tuple[str, ...],
    ) -> None:
        snapshot = self.hf / cache / "snapshots" / revision
        snapshot.mkdir(parents=True)
        config = {
            "architectures": [architecture],
            "model_type": model_type,
            "torch_dtype": "bfloat16",
        }
        tokenizer_config = {"tokenizer_class": tokenizer_class}
        index = {"metadata": {}, "weight_map": {"x": "model.safetensors"}}
        content = {
            "config.json": json.dumps(config, sort_keys=True).encode(),
            "generation_config.json": b'{"max_length": 20}',
            "model.safetensors.index.json": json.dumps(index, sort_keys=True).encode(),
            "model.safetensors": (alias + "-weights").encode(),
            "tokenizer_config.json": json.dumps(tokenizer_config, sort_keys=True).encode(),
            **{name: (alias + "-" + name).encode() for name in tokenizer_files},
            "README.md": (alias + "-extra").encode(),
        }
        for name, raw in content.items():
            self._make_blob_link(cache, snapshot, name, raw)
        required_names = sorted(set(content) - {"README.md"})
        members = tuple(
            ClosureMember(
                relative_path=name,
                link_target=os.readlink(snapshot / name),
                size=(snapshot / name).stat().st_size,
                sha256=_sha(snapshot / name),
                kind=(
                    "weight_shard"
                    if name == "model.safetensors"
                    else "tokenizer_file"
                    if name in tokenizer_files
                    else "tokenizer_config"
                    if name == "tokenizer_config.json"
                    else "weight_index"
                    if name.endswith("index.json")
                    else "generation_config"
                    if name == "generation_config.json"
                    else "model_config"
                ),
            )
            for name in required_names
        )
        extra = (
            ClosureMember(
                relative_path="README.md",
                link_target=os.readlink(snapshot / "README.md"),
                size=(snapshot / "README.md").stat().st_size,
                sha256=_sha(snapshot / "README.md"),
                kind="observation_only_extra",
            ),
        )
        model = PinnedModelClosure(
            alias=alias,
            native_name=native,
            repo_cache=cache,
            revision=revision,
            snapshot_path=str(snapshot),
            model_type=model_type,
            architecture=architecture,
            tokenizer_class=tokenizer_class,
            tokenizer_members=tokenizer_files,
            required_members=members,
            required_root=_member_root(members),
            required_bytes=sum(item.size for item in members),
            extra_members=extra,
            extra_directories=(),
            extra_root=_member_root(extra),
            extra_bytes=sum(item.size for item in extra),
            closure_identity="0" * 64,
        )
        model = PinnedModelClosure(
            **{
                **{key: getattr(model, key) for key in model.__dataclass_fields__},
                "closure_identity": _closure_identity(model),
            }
        )
        self.models[alias] = {
            key: (
                [item.identity_payload() for item in value]
                if key in ("required_members", "extra_members")
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in {
                field: getattr(model, field) for field in model.__dataclass_fields__
            }.items()
            if key != "alias"
        }

    def _seal_value(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema": "ode-edit-s05-p4-server4-hf-consumed-closure-seal/v1",
            "seal_id": "TEST-P4-HF-SEAL",
            "hostname": "server4",
            "agent_hostname": "server4",
            "agent_role": "server-head",
            "hf_hub_root": str(self.hf),
            "loader_contract": {
                "transformers_version": "4.57.1",
                "entrypoint": (
                    "project.run_scripts.ode_bf.p4_hf_consumed_closure."
                    "load_p4_full_fp32_from_sealed_snapshot"
                ),
                "snapshot_argument": "sealed_absolute_snapshot",
                "tokenizer_kwargs": {
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "use_fast": True,
                },
                "model_kwargs": {
                    "device_map": {"": 0},
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": False,
                    "torch_dtype": "torch.float32",
                },
                "dtype_policy": "full-fp32",
                "offline_environment": {
                    "HF_DATASETS_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                    "TRANSFORMERS_OFFLINE": "1",
                    "WANDB_DISABLED": "true",
                },
                "sources": [
                    {
                        "path": str(self.source),
                        "size": self.source.stat().st_size,
                        "sha256": _sha(self.source),
                    }
                ],
            },
            "models": self.models,
        }
        value["root_digest"] = canonical_hash(value)
        return value

    def _write_seal(self, mutate=None) -> None:
        value = self._seal_value()
        if mutate is not None:
            mutate(value)
            value.pop("root_digest", None)
            value["root_digest"] = canonical_hash(value)
        self.seal_path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    def _load(self):
        return load_p4_hf_consumed_closure_seal(
            self.seal_path, repo_root=self.repo
        )

    def test_positive_both_aliases_and_extra_influence_zero(self) -> None:
        seal = self._load()
        for alias in ("llama3-8b-inst", "qwen2.5-7b-inst"):
            receipt = seal.preflight_alias(alias)
            self.assertEqual(receipt.extra_influence_count, 0)
            self.assertFalse(receipt.model_loaded)
            self.assertEqual(
                receipt.loader_kwargs["model"]["torch_dtype"], "torch.float32"
            )
            self.assertEqual(
                receipt.loader_kwargs["model"]["pretrained_model_name_or_path"],
                receipt.snapshot_path,
            )
            self.assertEqual(receipt.offline_environment["HF_HUB_OFFLINE"], "1")

    def test_wrong_alias_and_revision_fail(self) -> None:
        with self.assertRaises(P4HFClosureError):
            self._load().preflight_alias("not-sealed")
        self._write_seal(
            lambda value: value["models"]["llama3-8b-inst"].__setitem__(
                "revision", "0" * 40
            )
        )
        with self.assertRaises(P4HFClosureError):
            self._load().preflight_alias("llama3-8b-inst")

    def test_wrong_role_fails_at_load(self) -> None:
        with mock.patch(
            "project.run_scripts.ode_bf.p4_hf_consumed_closure.detect_runtime_identity",
            return_value=RuntimeIdentity("server4", "server4", "worker"),
        ):
            with self.assertRaises(P4HFClosureError):
                self._load()

    def test_wrong_symlink_ref_fails(self) -> None:
        model = self.models["llama3-8b-inst"]
        snapshot = Path(model["snapshot_path"])
        path = snapshot / "README.md"
        path.unlink()
        path.symlink_to("../../blobs/not-the-sealed-ref")
        with self.assertRaises(P4HFClosureError):
            self._load().preflight_alias("llama3-8b-inst", verify_sources=False)

    def test_repo_ref_path_as_loader_snapshot_fails(self) -> None:
        seal = self._load()
        model = seal.model("llama3-8b-inst")
        ref_path = self.hf / model.repo_cache / "refs" / "main"
        with self.assertRaises(P4HFClosureError):
            seal.preflight_alias(
                "llama3-8b-inst",
                requested_snapshot=ref_path,
                verify_sources=False,
            )

    def test_wrong_config_index_shard_and_tokenizer_fail(self) -> None:
        names = (
            "config.json",
            "model.safetensors.index.json",
            "model.safetensors",
            "tokenizer.json",
        )
        for name in names:
            with self.subTest(name=name):
                snapshot = Path(self.models["llama3-8b-inst"]["snapshot_path"])
                resolved = (snapshot / name).resolve()
                original = resolved.read_bytes()
                try:
                    resolved.write_bytes(original + b"tampered")
                    with self.assertRaises(P4HFClosureError):
                        self._load().preflight_alias(
                            "llama3-8b-inst", verify_sources=False
                        )
                finally:
                    resolved.write_bytes(original)

    def test_extra_as_consumed_fails(self) -> None:
        seal = self._load()
        model = seal.model("qwen2.5-7b-inst")
        requested = [item.relative_path for item in model.required_members]
        requested.append("README.md")
        with self.assertRaises(P4HFClosureError):
            seal.preflight_alias(
                "qwen2.5-7b-inst",
                requested_paths=requested,
                verify_sources=False,
            )

    def test_model_load_entrypoint_is_unreachable_before_final_gates(self) -> None:
        with self.assertRaises(P4HFClosureError):
            load_p4_full_fp32_from_sealed_snapshot(
                self._load(),
                "llama3-8b-inst",
                stream_receipt={},
                final_pre_gpu_receipt={},
            )


if __name__ == "__main__":
    unittest.main()
