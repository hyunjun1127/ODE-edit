"""Immutable host-specific path remap for locked AlphaEdit artifacts.

The scientific locks remain authoritative.  A runtime seal may only replace an
exact locked root with an exact host-local root, and every sealed member is
verified before a caller can load a model.
"""

from __future__ import annotations

import hashlib
import json
import socket
import stat
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PATH_SEAL_SCHEMA = "ode-edit-alphaedit-runtime-path-seal/v1"
SOURCE_MANIFEST_SCHEMA = "ode-edit-s4-m1-r1-source-manifest/v1"
MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")
EXPECTED_MODEL_BINDINGS = {
    "llama3-8b-inst": {
        "native_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        "bundle_sha256": "2cd8517f65f476e3a8be4edddc81fc345ecd96a49634e41769bbacd4206af77e",
    },
    "qwen2.5-7b-inst": {
        "native_name": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "bundle_sha256": "55a6c2557b3a89f6880a7c56f65fb2487bafee53476dd248b57a3a48185a6638",
    },
}
EXPECTED_LAYERS = (4, 5, 6, 7, 8)
EXPECTED_REPRESENTATION = "model.layers.{L}.mlp.down_proj"
ROOT_NAMES = ("easyedit_root", "hf_hub_cache", "alphaedit_evaluator_root")


class RuntimePathSealError(RuntimeError):
    """Raised when a deployment path seal fails closed."""


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def _regular_fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    observed = path.stat()
    if not stat.S_ISREG(observed.st_mode):
        raise RuntimePathSealError("sealed member is not a regular file")
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimePathSealError(f"invalid runtime path seal field: {label}")
    return value


def _required_sha256(value: object, label: str) -> str:
    text = _required_string(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimePathSealError(f"invalid runtime path seal SHA256: {label}")
    return text


def _required_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RuntimePathSealError(f"invalid runtime path seal integer: {label}")
    return value


def _safe_relative_text(value: object, label: str) -> str:
    text = _required_string(value, label)
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimePathSealError(f"runtime path seal member escapes root: {label}")
    return relative.as_posix()


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    physical_hostname: str
    agent_hostname: str
    agent_role: str


@dataclass(frozen=True, slots=True)
class RootMapping:
    name: str
    logical_root: str
    runtime_root: str


@dataclass(frozen=True, slots=True)
class ArtifactMember:
    kind: str
    relative_path: str
    size: int
    sha256: str
    shape: tuple[int, ...]
    dtype: str | None
    layer: int | None

    def canonical_payload(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "kind": self.kind,
            "layer": self.layer,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "shape": list(self.shape),
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class ModelSeal:
    alias: str
    native_name: str
    revision: str
    layers: tuple[int, ...]
    representation: str
    bundle_sha256: str
    member_manifest_sha256: str
    members: tuple[ArtifactMember, ...]


@dataclass(frozen=True, slots=True)
class RuntimePathSealReceipt:
    seal_id: str
    seal_path: str
    seal_sha256: str
    seal_root_digest: str
    physical_hostname: str
    agent_hostname: str
    agent_role: str
    alias: str
    native_name: str
    revision: str
    logical_locked_path: str
    resolved_runtime_path: str
    bundle_sha256: str
    member_manifest_sha256: str
    verified_member_count: int
    member_verification: str


def detect_runtime_identity(repo_root: Path) -> RuntimeIdentity:
    root = repo_root.resolve(strict=True)

    def git_value(key: str) -> str:
        completed = subprocess.run(
            ["git", "config", "--local", "--get", key],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        value = completed.stdout.strip()
        if completed.returncode != 0 or not value:
            raise RuntimePathSealError(f"missing local Git identity: {key}")
        return value

    return RuntimeIdentity(
        physical_hostname=socket.gethostname().split(".", 1)[0],
        agent_hostname=git_value("agent.hostname"),
        agent_role=git_value("agent.role"),
    )


def _parse_member(value: object, *, alias: str, index: int) -> ArtifactMember:
    if not isinstance(value, dict):
        raise RuntimePathSealError("runtime path seal member is not an object")
    kind = _required_string(value.get("kind"), f"{alias}:member:{index}:kind")
    if kind not in ("hparams", "projector", "covariance"):
        raise RuntimePathSealError("runtime path seal member kind differs")
    layer_value = value.get("layer")
    if kind == "covariance":
        if layer_value not in EXPECTED_LAYERS:
            raise RuntimePathSealError("runtime path seal covariance layer differs")
        layer = int(layer_value)
    else:
        if layer_value is not None:
            raise RuntimePathSealError("non-covariance runtime member has a layer")
        layer = None
    shape_value = value.get("shape")
    if not isinstance(shape_value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in shape_value
    ):
        raise RuntimePathSealError("runtime path seal member shape differs")
    shape = tuple(shape_value)
    dtype_value = value.get("dtype")
    if dtype_value is not None and not isinstance(dtype_value, str):
        raise RuntimePathSealError("runtime path seal member dtype differs")
    if kind == "hparams" and (shape or dtype_value is not None):
        raise RuntimePathSealError("hparams runtime member metadata differs")
    if kind != "hparams" and (not shape or dtype_value != "float32"):
        raise RuntimePathSealError("tensor runtime member metadata differs")
    return ArtifactMember(
        kind=kind,
        relative_path=_safe_relative_text(
            value.get("relative_path"), f"{alias}:member:{index}:path"
        ),
        size=_required_int(value.get("size"), f"{alias}:member:{index}:size"),
        sha256=_required_sha256(
            value.get("sha256"), f"{alias}:member:{index}:sha256"
        ),
        shape=shape,
        dtype=dtype_value,
        layer=layer,
    )


def _model_manifest_payload(model: ModelSeal) -> dict[str, object]:
    return {
        "alias": model.alias,
        "layers": list(model.layers),
        "members": [member.canonical_payload() for member in model.members],
        "native_name": model.native_name,
        "representation": model.representation,
        "revision": model.revision,
    }


@dataclass(frozen=True, slots=True)
class AlphaEditRuntimePathSeal:
    seal_id: str
    source_path: str
    seal_sha256: str
    root_digest: str
    identity: RuntimeIdentity
    mappings: tuple[RootMapping, ...]
    models: tuple[ModelSeal, ...]
    withheld_mappings: tuple[tuple[str, str], ...]

    def _mapping(self, name: str) -> RootMapping | None:
        if name not in ROOT_NAMES:
            raise RuntimePathSealError("unknown runtime root mapping")
        return next((mapping for mapping in self.mappings if mapping.name == name), None)

    def has_mapping(self, name: str) -> bool:
        return self._mapping(name) is not None

    def assert_logical_root(self, name: str, logical_locked_root: str | Path) -> None:
        mapping = self._mapping(name)
        if mapping is not None and str(logical_locked_root) != mapping.logical_root:
            raise RuntimePathSealError("logical locked root differs from path seal")

    def resolve_root(self, name: str, logical_locked_root: str | Path) -> Path:
        logical = str(logical_locked_root)
        mapping = self._mapping(name)
        if mapping is None:
            return Path(logical).resolve(strict=True)
        if logical != mapping.logical_root:
            raise RuntimePathSealError("logical locked root differs from path seal")
        runtime = Path(mapping.runtime_root)
        if runtime.is_symlink() or runtime.resolve(strict=True) != runtime:
            raise RuntimePathSealError("sealed runtime root differs")
        return runtime

    def model(self, alias: str) -> ModelSeal:
        if alias not in MODEL_ALIASES:
            raise RuntimePathSealError("unknown sealed model alias")
        model = next((item for item in self.models if item.alias == alias), None)
        if model is None:
            raise RuntimePathSealError("sealed model alias is absent")
        return model

    def assert_p0_model_contract(self, alias: str, spec: Mapping[str, Any]) -> None:
        model = self.model(alias)
        if (
            spec.get("native_name") != model.native_name
            or spec.get("revision") != model.revision
        ):
            raise RuntimePathSealError("P0 model identity differs from path seal")
        covariance = spec.get("covariance")
        if not isinstance(covariance, Mapping):
            raise RuntimePathSealError("P0 covariance identity is absent")
        sealed = {
            str(member.layer): (
                member.relative_path,
                member.size,
                member.sha256,
            )
            for member in model.members
            if member.kind == "covariance"
        }
        observed = {
            str(layer): (str(value[0]), int(value[1]), str(value[2]))
            for layer, value in covariance.items()
        }
        if observed != sealed:
            raise RuntimePathSealError("P0 covariance members differ from path seal")

    def assert_odebf_model_contract(self, alias: str, spec: Mapping[str, Any]) -> None:
        model = self.model(alias)
        if (
            spec.get("native_name") != model.native_name
            or spec.get("revision") != model.revision
            or tuple(spec.get("layers", ())) != model.layers
        ):
            raise RuntimePathSealError("ODE-BF model identity differs from path seal")
        hparams = [member for member in model.members if member.kind == "hparams"]
        projector = [member for member in model.members if member.kind == "projector"]
        if len(hparams) != 1 or len(projector) != 1:
            raise RuntimePathSealError("sealed AlphaEdit member cardinality differs")
        if (
            spec.get("hparams_path") != hparams[0].relative_path
            or spec.get("hparams_sha256") != hparams[0].sha256
            or spec.get("projector_path") != projector[0].relative_path
            or spec.get("projector_sha256") != projector[0].sha256
            or spec.get("projector_size") != projector[0].size
        ):
            raise RuntimePathSealError("ODE-BF members differ from path seal")

    def _member_path(self, root: Path, member: ArtifactMember) -> Path:
        candidate = root
        for part in Path(member.relative_path).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise RuntimePathSealError("sealed member path uses a symlink")
        resolved_parent = candidate.parent.resolve(strict=True)
        resolved_parent.relative_to(root)
        if candidate.is_symlink():
            raise RuntimePathSealError("sealed member uses a symlink")
        return candidate

    @staticmethod
    def _npz_tensor_metadata(path: Path) -> tuple[tuple[int, ...], str]:
        import numpy as np

        expected_names = {
            "mom2.constructor.npy",
            "mom2.count.npy",
            "mom2.mom2.npy",
            "sample_size.npy",
        }
        with zipfile.ZipFile(path, "r") as archive:
            if set(archive.namelist()) != expected_names:
                raise RuntimePathSealError("sealed covariance NPZ members differ")
            with archive.open("mom2.mom2.npy", "r") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                else:
                    shape, _fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        return tuple(int(item) for item in shape), str(dtype)

    @staticmethod
    def _projector_metadata(path: Path) -> tuple[tuple[int, ...], str]:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise RuntimePathSealError("sealed projector format differs")
        return tuple(int(item) for item in value.shape), str(value.dtype).removeprefix("torch.")

    def preflight_alias(self, alias: str) -> RuntimePathSealReceipt:
        model = self.model(alias)
        mapping = self._mapping("easyedit_root")
        if mapping is None:
            raise RuntimePathSealError("EasyEdit runtime root mapping is absent")
        root = self.resolve_root("easyedit_root", mapping.logical_root)
        fingerprints: list[tuple[str, tuple[int, int, int, int, int]]] = []
        for member in model.members:
            try:
                path = self._member_path(root, member)
                fingerprint = _regular_fingerprint(path)
                if fingerprint[2] != member.size or sha256_file(path) != member.sha256:
                    raise RuntimePathSealError("sealed member SHA256 or size differs")
                if member.kind == "projector":
                    metadata = self._projector_metadata(path)
                elif member.kind == "covariance":
                    metadata = self._npz_tensor_metadata(path)
                else:
                    metadata = ((), None)
                if _regular_fingerprint(path) != fingerprint:
                    raise RuntimePathSealError("sealed member changed during preflight")
            except RuntimePathSealError:
                raise
            except (OSError, ValueError, RuntimeError, EOFError, zipfile.BadZipFile) as exc:
                raise RuntimePathSealError("sealed member is unresolved or corrupt") from exc
            if metadata != (member.shape, member.dtype):
                raise RuntimePathSealError("sealed member shape or dtype differs")
            fingerprints.append((member.relative_path, fingerprint))
        if len(fingerprints) != 7:
            raise RuntimePathSealError("sealed member verification count differs")
        return RuntimePathSealReceipt(
            seal_id=self.seal_id,
            seal_path=self.source_path,
            seal_sha256=self.seal_sha256,
            seal_root_digest=self.root_digest,
            physical_hostname=self.identity.physical_hostname,
            agent_hostname=self.identity.agent_hostname,
            agent_role=self.identity.agent_role,
            alias=model.alias,
            native_name=model.native_name,
            revision=model.revision,
            logical_locked_path=mapping.logical_root,
            resolved_runtime_path=mapping.runtime_root,
            bundle_sha256=model.bundle_sha256,
            member_manifest_sha256=model.member_manifest_sha256,
            verified_member_count=len(fingerprints),
            member_verification="sha256-size-shape-dtype-match",
        )


def load_alphaedit_runtime_path_seal(
    path: Path,
    *,
    repo_root: Path,
) -> AlphaEditRuntimePathSeal:
    source = path.resolve(strict=True)
    if path.is_symlink() or not source.is_file():
        raise RuntimePathSealError("runtime path seal is not a regular file")
    raw_sha = sha256_file(source)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimePathSealError("runtime path seal is invalid JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != PATH_SEAL_SCHEMA:
        raise RuntimePathSealError("runtime path seal schema differs")
    observed_root = value.get("root_digest")
    rooted = dict(value)
    rooted.pop("root_digest", None)
    expected_root = canonical_hash(rooted)
    if observed_root != expected_root:
        raise RuntimePathSealError("runtime path seal root digest differs")
    seal_id = _required_string(value.get("seal_id"), "seal_id")
    expected_hostname = _required_string(value.get("hostname"), "hostname")
    expected_agent_hostname = _required_string(
        value.get("agent_hostname"), "agent_hostname"
    )
    expected_agent_role = _required_string(value.get("agent_role"), "agent_role")
    identity = detect_runtime_identity(repo_root)
    if (
        identity.physical_hostname != expected_hostname
        or identity.agent_hostname != expected_agent_hostname
        or identity.agent_role != expected_agent_role
    ):
        raise RuntimePathSealError("runtime host or agent role differs")

    raw_mappings = value.get("mappings")
    if not isinstance(raw_mappings, dict) or "easyedit_root" not in raw_mappings:
        raise RuntimePathSealError("EasyEdit runtime root mapping is absent")
    mappings: list[RootMapping] = []
    for name, raw in sorted(raw_mappings.items()):
        if name not in ROOT_NAMES or not isinstance(raw, dict):
            raise RuntimePathSealError("runtime root mapping differs")
        logical = _required_string(raw.get("logical_root"), f"{name}:logical_root")
        runtime = _required_string(raw.get("runtime_root"), f"{name}:runtime_root")
        logical_path = Path(logical)
        runtime_path = Path(runtime)
        if not logical_path.is_absolute() or not runtime_path.is_absolute():
            raise RuntimePathSealError("runtime root mapping is not absolute")
        if (
            runtime_path.is_symlink()
            or not runtime_path.is_dir()
            or runtime_path.resolve(strict=True) != runtime_path
        ):
            raise RuntimePathSealError("runtime root mapping is unresolved or symlinked")
        mappings.append(RootMapping(name, logical, runtime))

    raw_models = value.get("models")
    if not isinstance(raw_models, dict) or tuple(sorted(raw_models)) != tuple(
        sorted(MODEL_ALIASES)
    ):
        raise RuntimePathSealError("runtime path seal aliases differ")
    models: list[ModelSeal] = []
    for alias in MODEL_ALIASES:
        raw_model = raw_models[alias]
        if not isinstance(raw_model, dict):
            raise RuntimePathSealError("runtime model seal differs")
        binding = EXPECTED_MODEL_BINDINGS[alias]
        native_name = _required_string(raw_model.get("native_name"), f"{alias}:name")
        revision = _required_string(raw_model.get("revision"), f"{alias}:revision")
        bundle = _required_sha256(
            raw_model.get("bundle_sha256"), f"{alias}:bundle_sha256"
        )
        layers_value = raw_model.get("layers")
        if not isinstance(layers_value, list):
            raise RuntimePathSealError("runtime model layers differ")
        layers = tuple(layers_value)
        representation = _required_string(
            raw_model.get("representation"), f"{alias}:representation"
        )
        if (
            native_name != binding["native_name"]
            or revision != binding["revision"]
            or bundle != binding["bundle_sha256"]
            or layers != EXPECTED_LAYERS
            or representation != EXPECTED_REPRESENTATION
        ):
            raise RuntimePathSealError("runtime model binding differs")
        raw_members = raw_model.get("members")
        if not isinstance(raw_members, list):
            raise RuntimePathSealError("runtime model members differ")
        members = tuple(
            _parse_member(item, alias=alias, index=index)
            for index, item in enumerate(raw_members)
        )
        kinds = [member.kind for member in members]
        covariance_layers = tuple(
            member.layer for member in members if member.kind == "covariance"
        )
        if (
            kinds.count("hparams") != 1
            or kinds.count("projector") != 1
            or kinds.count("covariance") != 5
            or covariance_layers != EXPECTED_LAYERS
            or len({member.relative_path for member in members}) != len(members)
        ):
            raise RuntimePathSealError("runtime model member binding differs")
        model = ModelSeal(
            alias=alias,
            native_name=native_name,
            revision=revision,
            layers=layers,
            representation=representation,
            bundle_sha256=bundle,
            member_manifest_sha256=_required_sha256(
                raw_model.get("member_manifest_sha256"),
                f"{alias}:member_manifest_sha256",
            ),
            members=members,
        )
        if canonical_hash(_model_manifest_payload(model)) != model.member_manifest_sha256:
            raise RuntimePathSealError("runtime member manifest digest differs")
        models.append(model)

    raw_withheld = value.get("withheld_mappings", {})
    if not isinstance(raw_withheld, dict):
        raise RuntimePathSealError("withheld runtime mappings differ")
    withheld = tuple(
        (str(name), _required_string(reason, f"withheld:{name}"))
        for name, reason in sorted(raw_withheld.items())
    )
    if any(name in raw_mappings for name, _reason in withheld):
        raise RuntimePathSealError("runtime mapping is both active and withheld")
    return AlphaEditRuntimePathSeal(
        seal_id=seal_id,
        source_path=str(source),
        seal_sha256=raw_sha,
        root_digest=expected_root,
        identity=identity,
        mappings=tuple(mappings),
        models=tuple(models),
        withheld_mappings=withheld,
    )


def verify_focused_source_manifest(path: Path, *, repo_root: Path) -> str:
    source = path.resolve(strict=True)
    if path.is_symlink() or not source.is_file():
        raise RuntimePathSealError("path-seal source manifest is not a regular file")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SOURCE_MANIFEST_SCHEMA:
        raise RuntimePathSealError("path-seal source manifest schema differs")
    observed_root = value.get("root_digest")
    rooted = dict(value)
    rooted.pop("root_digest", None)
    if observed_root != canonical_hash(rooted):
        raise RuntimePathSealError("path-seal source manifest root differs")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimePathSealError("path-seal source manifest entries differ")
    root = repo_root.resolve(strict=True)
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimePathSealError("path-seal source manifest entry differs")
        relative = _safe_relative_text(entry.get("path"), "source_manifest:path")
        candidate = root / relative
        if candidate.is_symlink():
            raise RuntimePathSealError("path-seal focused source uses a symlink")
        fingerprint = _regular_fingerprint(candidate)
        if (
            fingerprint[2] != _required_int(entry.get("size"), f"{relative}:size")
            or sha256_file(candidate)
            != _required_sha256(entry.get("sha256"), f"{relative}:sha256")
        ):
            raise RuntimePathSealError("path-seal focused source differs")
    return sha256_file(source)
