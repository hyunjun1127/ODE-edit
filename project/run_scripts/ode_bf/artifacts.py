"""Pinned, offline, read-only AlphaEdit/benchmark/model artifact guard."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from project.run_scripts.alphaedit_runtime_path_seal import (
    AlphaEditRuntimePathSeal,
    RuntimePathSealReceipt,
)
from project.run_scripts.ode_alloc.p0_artifacts import P0ArtifactGuard

from .contracts import MODEL_ALIASES, ODEBFContractError, canonical_hash


LOCK_SCHEMA = "ode-edit-s04-ode-bf-p0-artifact-lock/v1"


def sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_regular_tree(root: Path) -> tuple[str, int]:
    """Hash a local artifact tree without exposing or decoding raw contents."""

    source = root.resolve(strict=True)
    if root.is_symlink() or not source.is_dir():
        raise ODEBFContractError("held artifact root is not a real directory")
    records: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise ODEBFContractError("held artifact tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ODEBFContractError("held artifact tree contains a special file")
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return canonical_hash(records), len(records)


def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    observed = path.stat()
    if not stat.S_ISREG(observed.st_mode):
        raise ODEBFContractError("locked artifact is not a regular file")
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_size),
        int(observed.st_mtime_ns),
        int(observed.st_ctime_ns),
    )


def _safe_relative(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ODEBFContractError("artifact lock path is not relative")
    candidate = root / relative
    candidate.parent.resolve(strict=True).relative_to(root.resolve(strict=True))
    return candidate


def load_rooted_json(path: Path, *, expected_schema: str | None = None) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    observed = value.pop("root_digest", None)
    expected = canonical_hash(value)
    value["root_digest"] = observed
    if observed != expected:
        raise ODEBFContractError("rooted JSON digest differs")
    if expected_schema is not None and value.get("schema_version") != expected_schema:
        raise ODEBFContractError("rooted JSON schema differs")
    return value, sha256_file(path)


@dataclass(frozen=True, slots=True)
class ODEBFArtifactReceipt:
    alias: str
    lock_sha256: str
    lock_root_digest: str
    base_model_lock_sha256: str
    base_model_revision: str
    alpha_sources_digest: str
    evaluator_sources_digest: str
    hparams_sha256: str
    projector_sha256: str
    proposal_sha256: str
    numerical_lock_sha256: str
    p0_b10_seal_sha256: str
    cpu_sampling_seal_sha256: str
    held_ode_alloc_tree_sha256: str
    held_ode_alloc_file_count: int
    held_ode_alloc_verified: bool
    held_ode_alloc_used: bool


class ODEBFArtifactGuard:
    def __init__(
        self,
        repo_root: Path,
        lock_path: Path,
        alias: str,
        *,
        require_held_ode_alloc: bool = True,
        runtime_path_seal: AlphaEditRuntimePathSeal | None = None,
    ) -> None:
        if alias not in MODEL_ALIASES:
            raise ODEBFContractError("unknown ODE-BF model alias")
        self.repo_root = repo_root.resolve(strict=True)
        self.lock_path = lock_path.resolve(strict=True)
        self.value, self.lock_sha256 = load_rooted_json(
            self.lock_path,
            expected_schema=LOCK_SCHEMA,
        )
        self.alias = alias
        self.runtime_path_seal = runtime_path_seal
        self.runtime_path_seal_receipt: RuntimePathSealReceipt | None = None
        if runtime_path_seal is None:
            self.easyedit_root = Path(self.value["easyedit_root"]).resolve(strict=True)
            self.evaluator_root = Path(
                self.value["alphaedit_evaluator_root"]
            ).resolve(strict=True)
            self.hf_hub_cache = Path(self.value["hf_hub_cache"]).resolve(strict=True)
        else:
            self.easyedit_root = runtime_path_seal.resolve_root(
                "easyedit_root", self.value["easyedit_root"]
            )
            self.evaluator_root = runtime_path_seal.resolve_root(
                "alphaedit_evaluator_root",
                self.value["alphaedit_evaluator_root"],
            )
            self.hf_hub_cache = runtime_path_seal.resolve_root(
                "hf_hub_cache", self.value["hf_hub_cache"]
            )
        self.spec: Mapping[str, Any] = self.value["models"][alias]
        if runtime_path_seal is not None:
            runtime_path_seal.assert_odebf_model_contract(alias, self.spec)
        self.require_held_ode_alloc = bool(require_held_ode_alloc)
        self.hparams = _safe_relative(self.easyedit_root, self.spec["hparams_path"])
        self.projector = _safe_relative(self.easyedit_root, self.spec["projector_path"])
        base_relative = self.value["base_model_artifact_lock"]["path"]
        self.base_lock_path = _safe_relative(self.repo_root, base_relative)
        self.base_guard = P0ArtifactGuard(
            self.base_lock_path,
            alias,
            runtime_path_seal=runtime_path_seal,
        )
        self._fingerprints: dict[Path, tuple[int, int, int, int, int]] = {}
        self._held_tree: tuple[Path, str, int] | None = None

    def _validate_file(
        self,
        path: Path,
        *,
        expected_sha256: str,
        expected_size: int | None = None,
    ) -> None:
        if path.is_symlink():
            raise ODEBFContractError("ODE-BF locked file unexpectedly uses a symlink")
        observed = _fingerprint(path)
        if expected_size is not None and observed[2] != expected_size:
            raise ODEBFContractError("ODE-BF locked file size differs")
        if sha256_file(path) != expected_sha256:
            raise ODEBFContractError("ODE-BF locked file digest differs")
        self._fingerprints[path] = observed

    def preflight(self) -> ODEBFArtifactReceipt:
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise ODEBFContractError("offline environment is not locked")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.easyedit_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        if head != self.value["easyedit_git_head"]:
            raise ODEBFContractError("EasyEdit source HEAD differs")
        expected_base_sha = self.value["base_model_artifact_lock"]["sha256"]
        if sha256_file(self.base_lock_path) != expected_base_sha:
            raise ODEBFContractError("base model artifact lock digest differs")
        self._fingerprints[self.base_lock_path] = _fingerprint(self.base_lock_path)
        base_receipt = self.base_guard.preflight()
        self.runtime_path_seal_receipt = self.base_guard.runtime_path_seal_receipt
        if base_receipt.revision != self.spec["revision"]:
            raise ODEBFContractError("AlphaEdit/base-model revision differs")

        alpha_observed: dict[str, str] = {}
        for relative, expected in sorted(self.value["easyedit_alpha_sources"].items()):
            path = _safe_relative(self.easyedit_root, relative)
            self._validate_file(path, expected_sha256=expected)
            alpha_observed[relative] = expected
        evaluator_observed: dict[str, str] = {}
        for relative, expected in sorted(self.value["benchmark_evaluator_sources"].items()):
            path = _safe_relative(self.evaluator_root, relative)
            self._validate_file(path, expected_sha256=expected)
            evaluator_observed[relative] = expected

        self._validate_file(self.hparams, expected_sha256=self.spec["hparams_sha256"])
        self._validate_file(
            self.projector,
            expected_sha256=self.spec["projector_sha256"],
            expected_size=int(self.spec["projector_size"]),
        )
        if tuple(self.spec["layers"]) != (4, 5, 6, 7, 8):
            raise ODEBFContractError("AlphaEdit candidate layers differ")

        lock_digests: dict[str, str] = {}
        for relative, expected in sorted(self.value["locks"].items()):
            path = _safe_relative(self.repo_root, relative)
            self._validate_file(path, expected_sha256=expected)
            lock_digests[relative] = expected

        proposal = self.value["proposal"]
        blob = subprocess.run(
            ["git", "show", f"{proposal['commit']}:project/proposals/ODE_BF_Dynamic_Layer_Proposal.md"],
            cwd=self.repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        if (
            len(blob) != proposal["bytes"]
            or blob.count(b"\n") != proposal["lines"]
            or hashlib.sha256(blob).hexdigest() != proposal["sha256"]
        ):
            raise ODEBFContractError("canonical ODE-BF proposal identity differs")
        observed_blob = subprocess.run(
            ["git", "rev-parse", f"{proposal['commit']}:project/proposals/ODE_BF_Dynamic_Layer_Proposal.md"],
            cwd=self.repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        if observed_blob != proposal["blob"]:
            raise ODEBFContractError("canonical ODE-BF proposal blob differs")

        held = self.value["held_ode_alloc_local"]
        held_digest = str(held["tree_sha256"])
        held_count = int(held["file_count"])
        if self.require_held_ode_alloc:
            held_root = _safe_relative(self.repo_root, held["relative_path"])
            observed_digest, observed_count = sha256_regular_tree(held_root)
            if (
                observed_digest != held_digest
                or observed_count != held_count
            ):
                raise ODEBFContractError(
                    "held ODE-Alloc local artifact tree differs"
                )
            self._held_tree = (held_root, observed_digest, observed_count)

        return ODEBFArtifactReceipt(
            self.alias,
            self.lock_sha256,
            self.value["root_digest"],
            expected_base_sha,
            base_receipt.revision,
            canonical_hash(alpha_observed),
            canonical_hash(evaluator_observed),
            self.spec["hparams_sha256"],
            self.spec["projector_sha256"],
            proposal["sha256"],
            lock_digests["project/run_scripts/ode_bf/locks/numerical_lock_p0.json"],
            lock_digests["project/run_scripts/ode_bf/locks/p0_b10_seal.json"],
            lock_digests["project/run_scripts/ode_bf/locks/p0_cpu_sampling_seal.json"],
            held_digest,
            held_count,
            self.require_held_ode_alloc,
            False,
        )

    def assert_unchanged(self) -> None:
        if not self._fingerprints:
            raise ODEBFContractError("ODE-BF artifact guard was not preflighted")
        self.base_guard.assert_unchanged()
        for path, expected in self._fingerprints.items():
            if _fingerprint(path) != expected:
                raise ODEBFContractError("locked ODE-BF artifact changed during execution")
        if self.require_held_ode_alloc and self._held_tree is None:
            raise ODEBFContractError("held ODE-Alloc artifact tree was not preflighted")
        if self._held_tree is not None:
            held_root, held_digest, held_count = self._held_tree
            if sha256_regular_tree(held_root) != (held_digest, held_count):
                raise ODEBFContractError("held ODE-Alloc local artifact tree changed")
