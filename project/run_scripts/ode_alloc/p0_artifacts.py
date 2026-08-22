"""Pinned, offline, read-only artifacts for the Session 04 P0 identity pair."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from project.run_scripts.alphaedit_runtime_path_seal import (
    AlphaEditRuntimePathSeal,
    RuntimePathSealReceipt,
)

from .contracts import MODEL_ALIASES, ODEAllocContractError, canonical_hash


LOCK_SCHEMA = "ode-alloc-s04-p0-artifact-lock-r1/v1"
EXPECTED_LAYERS = (4, 5, 6, 7, 8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ODEAllocContractError("artifact lock path is not relative")
    candidate = root / relative
    resolved_parent = candidate.parent.resolve(strict=True)
    resolved_parent.relative_to(root.resolve(strict=True))
    return candidate


def _triple(value: Any, label: str) -> tuple[str, int, str | None]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not isinstance(value[0], str)
        or not isinstance(value[1], int)
        or isinstance(value[1], bool)
        or value[1] <= 0
        or (value[2] is not None and (not isinstance(value[2], str) or len(value[2]) != 64))
    ):
        raise ODEAllocContractError(f"invalid artifact lock tuple: {label}")
    return value[0], value[1], value[2]


def validate_artifact_lock_structure(
    value: Any,
    *,
    runtime_path_seal: AlphaEditRuntimePathSeal | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != LOCK_SCHEMA:
        raise ODEAllocContractError("P0 artifact lock schema differs")
    if value.get("instruction_id") != "ODEEDIT-S04-ODE-ALLOC-P0-LOCK-R1-PAIR-V1":
        raise ODEAllocContractError("P0 artifact lock instruction differs")
    if value.get("easyedit_root") != "/mnt/raid5/janghj/EasyEdit":
        raise ODEAllocContractError("EasyEdit artifact root differs")
    if value.get("hf_hub_cache") != "/mnt/raid5/janghj/.cache/huggingface/hub":
        raise ODEAllocContractError("HF artifact root differs")
    if runtime_path_seal is not None:
        runtime_path_seal.assert_logical_root(
            "easyedit_root", value["easyedit_root"]
        )
        runtime_path_seal.assert_logical_root(
            "hf_hub_cache", value["hf_hub_cache"]
        )
    sources = value.get("easyedit_sources")
    if not isinstance(sources, dict) or not sources:
        raise ODEAllocContractError("EasyEdit source lock is empty")
    for relative, digest in sources.items():
        if not isinstance(relative, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ODEAllocContractError("EasyEdit source lock entry is invalid")
    counterfact = value.get("counterfact")
    if not isinstance(counterfact, dict):
        raise ODEAllocContractError("CounterFact lock is absent")
    _triple(
        [counterfact.get("path"), counterfact.get("size"), counterfact.get("sha256")],
        "counterfact",
    )
    models = value.get("models")
    if not isinstance(models, dict) or tuple(sorted(models)) != tuple(sorted(MODEL_ALIASES)):
        raise ODEAllocContractError("artifact lock model aliases differ")
    for alias in MODEL_ALIASES:
        model = models[alias]
        if not isinstance(model, dict) or model.get("config_torch_dtype") != "bfloat16":
            raise ODEAllocContractError("model artifact lock is not BF16")
        for name in ("repo_cache", "revision", "native_name", "hparams_path"):
            if not isinstance(model.get(name), str) or not model[name]:
                raise ODEAllocContractError(f"model artifact field is invalid: {name}")
        if len(model.get("revision", "")) != 40 or len(model.get("hparams_sha256", "")) != 64:
            raise ODEAllocContractError("model revision or hparams digest differs")
        files = model.get("snapshot_files")
        if not isinstance(files, dict) or not files:
            raise ODEAllocContractError("model snapshot lock is empty")
        for name, item in files.items():
            _triple(item, f"{alias}:{name}")
        covariance = model.get("covariance")
        if not isinstance(covariance, dict) or tuple(sorted(map(int, covariance))) != EXPECTED_LAYERS:
            raise ODEAllocContractError("covariance layer lock differs")
        for layer, item in covariance.items():
            _, _, digest = _triple(item, f"{alias}:covariance:{layer}")
            if digest is None:
                raise ODEAllocContractError("covariance requires a content digest")
    return value


def load_artifact_lock(
    path: Path,
    *,
    runtime_path_seal: AlphaEditRuntimePathSeal | None = None,
) -> tuple[dict[str, Any], str, str]:
    source = path.resolve(strict=True)
    raw_sha = sha256_file(source)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ODEAllocContractError("P0 artifact lock is invalid JSON") from exc
    value = validate_artifact_lock_structure(
        value,
        runtime_path_seal=runtime_path_seal,
    )
    return value, raw_sha, canonical_hash(value)


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    alias: str
    lock_sha256: str
    lock_canonical_sha256: str
    snapshot: str
    revision: str
    hparams: str
    counterfact: str
    covariance: tuple[tuple[int, str], ...]
    easyedit_sources_root: str


def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    observed = path.stat()
    if not stat.S_ISREG(observed.st_mode):
        raise ODEAllocContractError("locked artifact is not a regular file")
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )


class P0ArtifactGuard:
    """Full preflight digest validation plus point-in-time mutation detection."""

    def __init__(
        self,
        lock_path: Path,
        alias: str,
        *,
        runtime_path_seal: AlphaEditRuntimePathSeal | None = None,
    ) -> None:
        if alias not in MODEL_ALIASES:
            raise ODEAllocContractError("unknown P0 model alias")
        self.lock_path = lock_path.resolve(strict=True)
        self.runtime_path_seal = runtime_path_seal
        self.runtime_path_seal_receipt: RuntimePathSealReceipt | None = None
        self.value, self.lock_sha256, self.lock_canonical_sha256 = load_artifact_lock(
            self.lock_path,
            runtime_path_seal=runtime_path_seal,
        )
        self.alias = alias
        if runtime_path_seal is None:
            self.easyedit_root = Path(self.value["easyedit_root"]).resolve(strict=True)
            self.hf_hub_cache = Path(self.value["hf_hub_cache"]).resolve(strict=True)
        else:
            self.easyedit_root = runtime_path_seal.resolve_root(
                "easyedit_root", self.value["easyedit_root"]
            )
            self.hf_hub_cache = runtime_path_seal.resolve_root(
                "hf_hub_cache", self.value["hf_hub_cache"]
            )
        self.spec: Mapping[str, Any] = self.value["models"][alias]
        if runtime_path_seal is not None:
            runtime_path_seal.assert_p0_model_contract(alias, self.spec)
        self.snapshot = (
            self.hf_hub_cache
            / self.spec["repo_cache"]
            / "snapshots"
            / self.spec["revision"]
        ).resolve(strict=True)
        self.hparams = _safe_relative(self.easyedit_root, self.spec["hparams_path"])
        self.dataset = _safe_relative(
            self.easyedit_root, self.value["counterfact"]["path"]
        )
        self._fingerprints: dict[str, tuple[int, int, int, int, int]] = {}

    def _check_digest(self, label: str, path: Path, expected_size: int, expected_sha: str) -> None:
        observed = _fingerprint(path)
        if observed[2] != expected_size or sha256_file(path) != expected_sha:
            raise ODEAllocContractError(f"locked artifact digest differs: {label}")
        self._fingerprints[label] = observed

    def preflight(self) -> ArtifactReceipt:
        if self.runtime_path_seal is not None:
            self.runtime_path_seal_receipt = self.runtime_path_seal.preflight_alias(
                self.alias
            )
        source_digests: dict[str, str] = {}
        for relative, expected in sorted(self.value["easyedit_sources"].items()):
            path = _safe_relative(self.easyedit_root, relative)
            observed = sha256_file(path)
            if observed != expected:
                raise ODEAllocContractError("EasyEdit source digest differs")
            self._fingerprints[f"source:{relative}"] = _fingerprint(path)
            source_digests[relative] = observed
        expected_hparams = self.spec["hparams_sha256"]
        if sha256_file(self.hparams) != expected_hparams:
            raise ODEAllocContractError("MEMIT hparams digest differs")
        self._fingerprints["hparams"] = _fingerprint(self.hparams)
        cf = self.value["counterfact"]
        self._check_digest("counterfact", self.dataset, cf["size"], cf["sha256"])

        observed_names = {path.name for path in self.snapshot.iterdir()}
        expected_names = set(self.spec["snapshot_files"])
        if observed_names != expected_names:
            raise ODEAllocContractError("pinned model snapshot file set differs")
        for name, locked in sorted(self.spec["snapshot_files"].items()):
            expected_target, expected_size, expected_sha = _triple(locked, name)
            link = self.snapshot / name
            if not link.is_symlink() or os.readlink(link) != expected_target:
                raise ODEAllocContractError("pinned model snapshot symlink differs")
            resolved = link.resolve(strict=True)
            resolved.relative_to(self.hf_hub_cache)
            observed = _fingerprint(resolved)
            if observed[2] != expected_size:
                raise ODEAllocContractError("pinned model snapshot size differs")
            if expected_sha is not None and sha256_file(resolved) != expected_sha:
                raise ODEAllocContractError("pinned model snapshot digest differs")
            self._fingerprints[f"snapshot:{name}"] = observed

        covariance_receipts: list[tuple[int, str]] = []
        for layer_text, locked in sorted(
            self.spec["covariance"].items(), key=lambda item: int(item[0])
        ):
            relative, expected_size, expected_sha = _triple(
                locked, f"covariance:{layer_text}"
            )
            assert expected_sha is not None
            path = _safe_relative(self.easyedit_root, relative)
            label = f"covariance:{layer_text}"
            self._check_digest(label, path, expected_size, expected_sha)
            covariance_receipts.append((int(layer_text), expected_sha))

        config = json.loads((self.snapshot / "config.json").read_text(encoding="utf-8"))
        if config.get("torch_dtype") != self.spec["config_torch_dtype"]:
            raise ODEAllocContractError("pinned model config dtype differs")
        return ArtifactReceipt(
            alias=self.alias,
            lock_sha256=self.lock_sha256,
            lock_canonical_sha256=self.lock_canonical_sha256,
            snapshot=str(self.snapshot),
            revision=self.spec["revision"],
            hparams=expected_hparams,
            counterfact=cf["sha256"],
            covariance=tuple(covariance_receipts),
            easyedit_sources_root=canonical_hash(source_digests),
        )

    def assert_unchanged(self) -> None:
        if not self._fingerprints:
            raise ODEAllocContractError("artifact guard was not preflighted")
        for label, expected in self._fingerprints.items():
            if label == "hparams":
                path = self.hparams
            elif label == "counterfact":
                path = self.dataset
            elif label.startswith("source:"):
                path = _safe_relative(self.easyedit_root, label.removeprefix("source:"))
            elif label.startswith("snapshot:"):
                path = (self.snapshot / label.removeprefix("snapshot:")).resolve(strict=True)
            elif label.startswith("covariance:"):
                layer = label.removeprefix("covariance:")
                relative = self.spec["covariance"][layer][0]
                path = _safe_relative(self.easyedit_root, relative)
            else:
                raise ODEAllocContractError("unknown artifact mutation receipt")
            if _fingerprint(path) != expected:
                raise ODEAllocContractError(f"locked artifact changed during P0: {label}")
