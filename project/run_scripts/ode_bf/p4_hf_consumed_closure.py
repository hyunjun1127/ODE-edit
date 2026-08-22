"""Server4-only pinned Hugging Face consumed-closure seal for P4.

This module does not load a tokenizer or model.  It reconstructs the files the
pinned Transformers call may consume, verifies every byte against a tracked
seal, and returns the one exact snapshot path that a later loader may use.
Files outside the reconstructed closure are recorded but never supplied to the
loader.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping

from project.run_scripts.alphaedit_runtime_path_seal import (
    canonical_hash,
    detect_runtime_identity,
)


SCHEMA = "ode-edit-s05-p4-server4-hf-consumed-closure-seal/v1"
EXPECTED_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
EXPECTED_BINDINGS = {
    "llama3-8b-inst": {
        "native_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        "model_type": "llama",
        "architecture": "LlamaForCausalLM",
        "tokenizer_class": "PreTrainedTokenizerFast",
    },
    "qwen2.5-7b-inst": {
        "native_name": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "model_type": "qwen2",
        "architecture": "Qwen2ForCausalLM",
        "tokenizer_class": "Qwen2Tokenizer",
    },
}


class P4HFClosureError(RuntimeError):
    """Raised whenever the P4 deployment closure cannot be proven exactly."""


def sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise P4HFClosureError(f"invalid closure seal field: {label}")
    return value


def _require_sha(value: object, label: str) -> str:
    text = _require_string(value, label)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise P4HFClosureError(f"invalid closure seal SHA256: {label}")
    return text


def _require_int(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise P4HFClosureError(f"invalid closure seal integer: {label}")
    return value


def _safe_relative(value: object, label: str) -> str:
    text = _require_string(value, label)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise P4HFClosureError(f"closure member escapes snapshot: {label}")
    return text


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P4HFClosureError(f"invalid consumed JSON: {label}") from exc
    if not isinstance(value, dict):
        raise P4HFClosureError(f"consumed JSON is not an object: {label}")
    return value


def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    observed = path.stat()
    if not stat.S_ISREG(observed.st_mode):
        raise P4HFClosureError("resolved closure member is not a regular file")
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )


@dataclass(frozen=True, slots=True)
class ClosureMember:
    relative_path: str
    link_target: str
    size: int
    sha256: str
    kind: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "link_target": self.link_target,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class PinnedModelClosure:
    alias: str
    native_name: str
    repo_cache: str
    revision: str
    snapshot_path: str
    model_type: str
    architecture: str
    tokenizer_class: str
    tokenizer_members: tuple[str, ...]
    required_members: tuple[ClosureMember, ...]
    required_root: str
    required_bytes: int
    extra_members: tuple[ClosureMember, ...]
    extra_directories: tuple[str, ...]
    extra_root: str
    extra_bytes: int
    closure_identity: str


@dataclass(frozen=True, slots=True)
class P4HFClosureReceipt:
    seal_id: str
    seal_path: str
    seal_sha256: str
    seal_root_digest: str
    hostname: str
    agent_hostname: str
    agent_role: str
    alias: str
    native_name: str
    revision: str
    snapshot_path: str
    required_count: int
    required_bytes: int
    required_root: str
    extra_count: int
    extra_bytes: int
    extra_root: str
    closure_identity: str
    loader_kwargs: Mapping[str, object]
    offline_environment: Mapping[str, str]
    extra_influence_count: int
    model_loaded: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _member_root(members: Iterable[ClosureMember]) -> str:
    return canonical_hash([member.identity_payload() for member in members])


def _closure_identity(model: PinnedModelClosure) -> str:
    return canonical_hash(
        {
            "alias": model.alias,
            "architecture": model.architecture,
            "model_type": model.model_type,
            "native_name": model.native_name,
            "required_bytes": model.required_bytes,
            "required_root": model.required_root,
            "revision": model.revision,
            "snapshot_path": model.snapshot_path,
            "tokenizer_class": model.tokenizer_class,
            "tokenizer_members": list(model.tokenizer_members),
        }
    )


def _parse_members(value: object, label: str) -> tuple[ClosureMember, ...]:
    if not isinstance(value, list):
        raise P4HFClosureError(f"closure member list differs: {label}")
    result: list[ClosureMember] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise P4HFClosureError(f"closure member is not an object: {label}")
        result.append(
            ClosureMember(
                relative_path=_safe_relative(
                    raw.get("relative_path"), f"{label}:{index}:path"
                ),
                link_target=_require_string(
                    raw.get("link_target"), f"{label}:{index}:target"
                ),
                size=_require_int(raw.get("size"), f"{label}:{index}:size"),
                sha256=_require_sha(raw.get("sha256"), f"{label}:{index}:sha"),
                kind=_require_string(raw.get("kind"), f"{label}:{index}:kind"),
            )
        )
    paths = [member.relative_path for member in result]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise P4HFClosureError(f"closure members are not unique and sorted: {label}")
    return tuple(result)


class P4HFConsumedClosureSeal:
    """Parsed immutable deployment seal and no-load preflight interface."""

    def __init__(
        self,
        *,
        seal_id: str,
        source_path: str,
        seal_sha256: str,
        root_digest: str,
        hf_hub_root: str,
        hostname: str,
        agent_hostname: str,
        agent_role: str,
        loader_contract: Mapping[str, object],
        models: tuple[PinnedModelClosure, ...],
    ) -> None:
        self.seal_id = seal_id
        self.source_path = source_path
        self.seal_sha256 = seal_sha256
        self.root_digest = root_digest
        self.hf_hub_root = hf_hub_root
        self.hostname = hostname
        self.agent_hostname = agent_hostname
        self.agent_role = agent_role
        self.loader_contract = loader_contract
        self.models = models

    def model(self, alias: str) -> PinnedModelClosure:
        if alias not in EXPECTED_ALIASES:
            raise P4HFClosureError("unsealed P4 model alias")
        model = next((item for item in self.models if item.alias == alias), None)
        if model is None:
            raise P4HFClosureError("P4 model alias absent from closure seal")
        return model

    def _verify_sources(self) -> None:
        if (
            self.loader_contract.get("entrypoint")
            != (
                "project.run_scripts.ode_bf.p4_hf_consumed_closure."
                "load_p4_full_fp32_from_sealed_snapshot"
            )
            or self.loader_contract.get("snapshot_argument")
            != "sealed_absolute_snapshot"
            or self.loader_contract.get("tokenizer_kwargs")
            != {
                "local_files_only": True,
                "trust_remote_code": False,
                "use_fast": True,
            }
            or self.loader_contract.get("model_kwargs")
            != {
                "device_map": {"": 0},
                "local_files_only": True,
                "low_cpu_mem_usage": True,
                "trust_remote_code": False,
                "torch_dtype": "torch.float32",
            }
            or self.loader_contract.get("dtype_policy") != "full-fp32"
            or self.loader_contract.get("offline_environment")
            != {
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_DISABLED": "true",
            }
        ):
            raise P4HFClosureError("loader offline/snapshot call contract differs")
        version = self.loader_contract.get("transformers_version")
        if version != "4.57.1":
            raise P4HFClosureError("Transformers version binding differs")
        try:
            installed_version = package_version("transformers")
        except PackageNotFoundError as exc:
            raise P4HFClosureError("Transformers package is unavailable") from exc
        if installed_version != version:
            raise P4HFClosureError("installed Transformers version differs")
        raw_sources = self.loader_contract.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise P4HFClosureError("loader source closure is absent")
        for raw in raw_sources:
            if not isinstance(raw, dict):
                raise P4HFClosureError("loader source closure differs")
            path = Path(_require_string(raw.get("path"), "loader_source:path"))
            expected_size = _require_int(raw.get("size"), "loader_source:size")
            expected_sha = _require_sha(raw.get("sha256"), "loader_source:sha")
            if path.is_symlink() or not path.is_file():
                raise P4HFClosureError("loader source is unresolved or symlinked")
            before = _fingerprint(path)
            if before[2] != expected_size or sha256_file(path) != expected_sha:
                raise P4HFClosureError("loader source SHA256 or size differs")
            if _fingerprint(path) != before:
                raise P4HFClosureError("loader source changed during verification")

    @staticmethod
    def _derive_required(
        snapshot: Path, model: PinnedModelClosure
    ) -> tuple[set[str], dict[str, object]]:
        config = _json_object(snapshot / "config.json", "config.json")
        tokenizer = _json_object(
            snapshot / "tokenizer_config.json", "tokenizer_config.json"
        )
        generation = _json_object(
            snapshot / "generation_config.json", "generation_config.json"
        )
        index = _json_object(
            snapshot / "model.safetensors.index.json",
            "model.safetensors.index.json",
        )
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise P4HFClosureError("weight index has no weight map")
        shards = set()
        for value in weight_map.values():
            shards.add(_safe_relative(value, "weight_map:shard"))
        required = {
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
            *model.tokenizer_members,
            *shards,
        }
        return required, {
            "config": config,
            "tokenizer": tokenizer,
            "generation": generation,
            "index": index,
            "shards": sorted(shards),
        }

    @staticmethod
    def _verify_binding(
        model: PinnedModelClosure, parsed: Mapping[str, object]
    ) -> None:
        config = parsed["config"]
        tokenizer = parsed["tokenizer"]
        if not isinstance(config, dict) or not isinstance(tokenizer, dict):
            raise P4HFClosureError("parsed model binding differs")
        architectures = config.get("architectures")
        if (
            config.get("model_type") != model.model_type
            or architectures != [model.architecture]
            or tokenizer.get("tokenizer_class") != model.tokenizer_class
            or config.get("auto_map") is not None
            or tokenizer.get("auto_map") is not None
        ):
            raise P4HFClosureError("model/tokenizer/source binding differs")

    def _verify_member(
        self, snapshot: Path, member: ClosureMember, hf_root: Path
    ) -> None:
        path = snapshot / member.relative_path
        if not path.is_symlink() or os.readlink(path) != member.link_target:
            raise P4HFClosureError("snapshot member symlink identity differs")
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(hf_root / Path(snapshot).parent.parent.name / "blobs")
        except (OSError, ValueError) as exc:
            raise P4HFClosureError("snapshot member target escapes pinned blob root") from exc
        before = _fingerprint(resolved)
        if before[2] != member.size or sha256_file(resolved) != member.sha256:
            raise P4HFClosureError("snapshot member SHA256 or size differs")
        if _fingerprint(resolved) != before:
            raise P4HFClosureError("snapshot member changed during verification")

    def preflight_alias(
        self,
        alias: str,
        *,
        requested_paths: Iterable[str] | None = None,
        requested_snapshot: Path | None = None,
        verify_sources: bool = True,
    ) -> P4HFClosureReceipt:
        model = self.model(alias)
        if verify_sources:
            self._verify_sources()
        hf_root = Path(self.hf_hub_root)
        snapshot = Path(model.snapshot_path)
        expected_snapshot = hf_root / model.repo_cache / "snapshots" / model.revision
        if (
            hf_root.is_symlink()
            or snapshot.is_symlink()
            or not snapshot.is_dir()
            or hf_root.resolve(strict=True) != hf_root
            or snapshot.resolve(strict=True) != snapshot
            or snapshot != expected_snapshot
        ):
            raise P4HFClosureError("pinned snapshot/root/revision path differs")
        if requested_snapshot is not None and requested_snapshot != snapshot:
            raise P4HFClosureError("loader requested a ref/repo/non-pinned snapshot")
        binding = EXPECTED_BINDINGS[alias]
        if any(
            (
                model.native_name != binding["native_name"],
                model.revision != binding["revision"],
                model.model_type != binding["model_type"],
                model.architecture != binding["architecture"],
                model.tokenizer_class != binding["tokenizer_class"],
            )
        ):
            raise P4HFClosureError("sealed alias/revision/model binding differs")

        derived, parsed = self._derive_required(snapshot, model)
        required_paths = {member.relative_path for member in model.required_members}
        if derived != required_paths:
            raise P4HFClosureError("source-derived consumed closure differs from seal")
        self._verify_binding(model, parsed)
        if requested_paths is not None and set(requested_paths) != required_paths:
            raise P4HFClosureError("requested loader closure differs from exact seal")

        extras = {member.relative_path for member in model.extra_members}
        if required_paths & extras:
            raise P4HFClosureError("extra file was admitted to consumed closure")
        actual_files: set[str] = set()
        actual_directories: set[str] = set()
        for path in snapshot.rglob("*"):
            relative = path.relative_to(snapshot).as_posix()
            if path.is_symlink():
                actual_files.add(relative)
            elif path.is_dir():
                actual_directories.add(relative)
            else:
                raise P4HFClosureError("snapshot contains an unsealed regular member")
        if actual_files != required_paths | extras:
            raise P4HFClosureError("snapshot contains missing or unsealed files")
        if actual_directories != set(model.extra_directories):
            raise P4HFClosureError("snapshot directory closure differs")

        if (
            _member_root(model.required_members) != model.required_root
            or sum(item.size for item in model.required_members) != model.required_bytes
            or _member_root(model.extra_members) != model.extra_root
            or sum(item.size for item in model.extra_members) != model.extra_bytes
            or _closure_identity(model) != model.closure_identity
        ):
            raise P4HFClosureError("closure aggregate identity differs")
        for member in (*model.required_members, *model.extra_members):
            self._verify_member(snapshot, member, hf_root)

        return P4HFClosureReceipt(
            seal_id=self.seal_id,
            seal_path=self.source_path,
            seal_sha256=self.seal_sha256,
            seal_root_digest=self.root_digest,
            hostname=self.hostname,
            agent_hostname=self.agent_hostname,
            agent_role=self.agent_role,
            alias=model.alias,
            native_name=model.native_name,
            revision=model.revision,
            snapshot_path=model.snapshot_path,
            required_count=len(model.required_members),
            required_bytes=model.required_bytes,
            required_root=model.required_root,
            extra_count=len(model.extra_members),
            extra_bytes=model.extra_bytes,
            extra_root=model.extra_root,
            closure_identity=model.closure_identity,
            loader_kwargs={
                "tokenizer": {
                    "pretrained_model_name_or_path": model.snapshot_path,
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "use_fast": True,
                },
                "model": {
                    "pretrained_model_name_or_path": model.snapshot_path,
                    "device_map": {"": 0},
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": False,
                    "torch_dtype": "torch.float32",
                },
            },
            offline_environment={
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_DISABLED": "true",
            },
            extra_influence_count=0,
            model_loaded=False,
        )


def load_p4_hf_consumed_closure_seal(
    path: Path, *, repo_root: Path
) -> P4HFConsumedClosureSeal:
    if path.is_symlink() or not path.is_file():
        raise P4HFClosureError("P4 HF seal is not a regular file")
    source = path.resolve(strict=True)
    raw_sha = sha256_file(source)
    value = _json_object(source, "P4 HF seal")
    observed_root = value.get("root_digest")
    rooted = dict(value)
    rooted.pop("root_digest", None)
    expected_root = canonical_hash(rooted)
    if observed_root != expected_root:
        raise P4HFClosureError("P4 HF seal root digest differs")
    if value.get("schema") != SCHEMA:
        raise P4HFClosureError("P4 HF seal schema differs")

    identity = detect_runtime_identity(repo_root)
    hostname = _require_string(value.get("hostname"), "hostname")
    agent_hostname = _require_string(value.get("agent_hostname"), "agent_hostname")
    agent_role = _require_string(value.get("agent_role"), "agent_role")
    if (
        identity.physical_hostname != hostname
        or identity.agent_hostname != agent_hostname
        or identity.agent_role != agent_role
        or hostname != "server4"
        or agent_hostname != "server4"
        or agent_role != "server-head"
    ):
        raise P4HFClosureError("P4 HF host/agent role differs")

    hf_hub_root = _require_string(value.get("hf_hub_root"), "hf_hub_root")
    if not Path(hf_hub_root).is_absolute():
        raise P4HFClosureError("HF hub root is not absolute")
    loader_contract = value.get("loader_contract")
    if not isinstance(loader_contract, dict):
        raise P4HFClosureError("loader contract is absent")
    raw_models = value.get("models")
    if not isinstance(raw_models, dict) or tuple(sorted(raw_models)) != tuple(
        sorted(EXPECTED_ALIASES)
    ):
        raise P4HFClosureError("P4 HF aliases differ")

    models: list[PinnedModelClosure] = []
    for alias in EXPECTED_ALIASES:
        raw = raw_models[alias]
        if not isinstance(raw, dict):
            raise P4HFClosureError("P4 HF model seal differs")
        required = _parse_members(raw.get("required_members"), f"{alias}:required")
        extras = _parse_members(raw.get("extra_members"), f"{alias}:extras")
        tokenizer_members = raw.get("tokenizer_members")
        extra_directories = raw.get("extra_directories")
        if (
            not isinstance(tokenizer_members, list)
            or not all(isinstance(item, str) for item in tokenizer_members)
            or not isinstance(extra_directories, list)
            or not all(isinstance(item, str) for item in extra_directories)
        ):
            raise P4HFClosureError("tokenizer/directory closure differs")
        tokenizer_members = [
            _safe_relative(item, f"{alias}:tokenizer_member")
            for item in tokenizer_members
        ]
        extra_directories = [
            _safe_relative(item, f"{alias}:extra_directory")
            for item in extra_directories
        ]
        model = PinnedModelClosure(
            alias=alias,
            native_name=_require_string(raw.get("native_name"), f"{alias}:name"),
            repo_cache=_safe_relative(raw.get("repo_cache"), f"{alias}:repo_cache"),
            revision=_require_string(raw.get("revision"), f"{alias}:revision"),
            snapshot_path=_require_string(raw.get("snapshot_path"), f"{alias}:snapshot"),
            model_type=_require_string(raw.get("model_type"), f"{alias}:model_type"),
            architecture=_require_string(
                raw.get("architecture"), f"{alias}:architecture"
            ),
            tokenizer_class=_require_string(
                raw.get("tokenizer_class"), f"{alias}:tokenizer_class"
            ),
            tokenizer_members=tuple(tokenizer_members),
            required_members=required,
            required_root=_require_sha(
                raw.get("required_root"), f"{alias}:required_root"
            ),
            required_bytes=_require_int(
                raw.get("required_bytes"), f"{alias}:required_bytes"
            ),
            extra_members=extras,
            extra_directories=tuple(extra_directories),
            extra_root=_require_sha(raw.get("extra_root"), f"{alias}:extra_root"),
            extra_bytes=_require_int(
                raw.get("extra_bytes"), f"{alias}:extra_bytes", allow_zero=True
            ),
            closure_identity=_require_sha(
                raw.get("closure_identity"), f"{alias}:closure_identity"
            ),
        )
        models.append(model)

    return P4HFConsumedClosureSeal(
        seal_id=_require_string(value.get("seal_id"), "seal_id"),
        source_path=str(source),
        seal_sha256=raw_sha,
        root_digest=expected_root,
        hf_hub_root=hf_hub_root,
        hostname=hostname,
        agent_hostname=agent_hostname,
        agent_role=agent_role,
        loader_contract=loader_contract,
        models=tuple(models),
    )


def load_p4_full_fp32_from_sealed_snapshot(
    seal: P4HFConsumedClosureSeal,
    alias: str,
    *,
    stream_receipt: Mapping[str, object],
    final_pre_gpu_receipt: Mapping[str, object],
    auto_model_class: Any | None = None,
    auto_tokenizer_class: Any | None = None,
    cuda_api: Any | None = None,
) -> tuple[Any, Any, P4HFClosureReceipt]:
    """The only P4 model-load entrypoint; unreachable before final gates pass.

    It deliberately accepts neither a repository id, revision, nor arbitrary
    path.  The absolute snapshot comes solely from the verified server4 seal.
    """

    binding = stream_receipt.get("binding")
    if (
        stream_receipt.get("status") != "TRANSFER_FULL_READ_PASS"
        or stream_receipt.get("full_read") is not True
        or not isinstance(binding, Mapping)
        or binding.get("model_aliases") != list(EXPECTED_ALIASES)
        or not binding.get("evaluator_identity")
        or not binding.get("dataset_identity")
    ):
        raise P4HFClosureError("sealed stream/evaluator gate is not closed")
    if (
        final_pre_gpu_receipt.get("status") != "FINAL_PRE_GPU_PASS"
        or final_pre_gpu_receipt.get("model_load_authorized") is not True
    ):
        raise P4HFClosureError("final PRE-GPU gate has not authorized model load")

    model_seal = seal.model(alias)
    receipt = seal.preflight_alias(
        alias,
        requested_paths=[
            member.relative_path for member in model_seal.required_members
        ],
        requested_snapshot=Path(model_seal.snapshot_path),
    )
    try:
        import torch
        from project.run_scripts.ode_edit_motivation.gpu_runtime import (
            assert_single_visible_gpu,
            offline_environment,
        )

        if auto_model_class is None or auto_tokenizer_class is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            auto_model_class = auto_model_class or AutoModelForCausalLM
            auto_tokenizer_class = auto_tokenizer_class or AutoTokenizer
        assert_single_visible_gpu(torch.cuda if cuda_api is None else cuda_api)
        snapshot = receipt.snapshot_path
        with offline_environment():
            tokenizer = auto_tokenizer_class.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
            model = auto_model_class.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                device_map={"": 0},
                low_cpu_mem_usage=True,
            )
        model.eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        floating = [
            parameter for parameter in model.parameters() if parameter.is_floating_point()
        ]
        if (
            not floating
            or any(parameter.dtype is not torch.float32 for parameter in floating)
            or any(parameter.requires_grad for parameter in floating)
            or torch.is_autocast_enabled()
            or torch.is_autocast_enabled("cpu")
            or str(getattr(model.config, "_name_or_path", "")) != snapshot
            or str(getattr(tokenizer, "name_or_path", "")) != snapshot
        ):
            raise P4HFClosureError("loaded P4 model/tokenizer FP32 snapshot differs")
        for observed in (
            getattr(model.config, "_commit_hash", None),
            getattr(tokenizer, "_commit_hash", None),
            getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            if isinstance(getattr(tokenizer, "init_kwargs", None), dict)
            else None,
        ):
            if observed is not None and observed != receipt.revision:
                raise P4HFClosureError("loaded P4 revision differs")
    except P4HFClosureError:
        raise
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        raise P4HFClosureError("P4 sealed full-FP32 load failed") from exc
    return model, tokenizer, receipt
