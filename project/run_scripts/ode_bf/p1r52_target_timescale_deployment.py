"""Server4 deployment binding for the target-timescale B100 task.

The adapter reuses the accepted AlphaEdit runtime path seal and the exact P4
HF consumed-closure seal.  It does not relax the legacy P0/ODE-BF guards: the
override exists only in the explicitly selected target-timescale launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from project.run_scripts.alphaedit_runtime_path_seal import (
    AlphaEditRuntimePathSeal,
    load_alphaedit_runtime_path_seal,
)
from project.run_scripts.ode_alloc.p0_artifacts import load_artifact_lock

from .contracts import ODEBFContractError, canonical_hash
from .p4_hf_consumed_closure import (
    P4HFConsumedClosureSeal,
    load_p4_full_fp32_from_sealed_snapshot,
    load_p4_hf_consumed_closure_seal,
)


MODEL_ALIAS = "llama3-8b-inst"
EXPECTED_HF_CLOSURE = "54e45cdfb1a594a17775b9608864df06e87a036047de5592d20830f486be3e3a"
EXPECTED_HF_REQUIRED_ROOT = "a1795dc0fe12a307e16432a7c2049da1a795e4170e3b6f6822d760a5d6ea8056"
EXPECTED_STREAM_ROOT = "467e5946ec0eb975284ca25e16f63f3b8ae0093503ca8b84948409689e0ad25a"
EXPECTED_STREAM_ORDER = "018be113361157d6f4050c37a4fec14fff78e60388e3898253d66f070d78cfc3"
EXPECTED_EVALUATOR_IDENTITY = "8d8196eedd2c8c675a6bff17912916bdd99a96715b2e7a714417b7ebc16da159"


@dataclass(frozen=True, slots=True)
class TargetTimescaleBaseReceipt:
    revision: str
    snapshot: str
    closure_identity: str
    required_root: str
    prior_full_verification_receipt_sha256: str
    bounded_member_fingerprint_count: int
    extra_influence_count: int


class TargetTimescaleHFBaseGuard:
    """P0-compatible read-only facade over an already full-verified closure."""

    def __init__(
        self,
        *,
        repo_root: Path,
        hf_seal_path: Path,
        p0_lock_path: Path,
        runtime_path_seal: AlphaEditRuntimePathSeal,
        prior_final_pre_gpu_path: Path,
        stream_receipt: Mapping[str, object],
    ) -> None:
        self.alias = MODEL_ALIAS
        self.runtime_path_seal_receipt = None
        self.seal: P4HFConsumedClosureSeal = load_p4_hf_consumed_closure_seal(
            hf_seal_path, repo_root=repo_root
        )
        p0_value, _raw_sha, _canonical_sha = load_artifact_lock(
            p0_lock_path,
            runtime_path_seal=runtime_path_seal,
        )
        self.p0_value = p0_value
        self.spec = p0_value["models"][MODEL_ALIAS]
        self.easyedit_root = runtime_path_seal.resolve_root(
            "easyedit_root", p0_value["easyedit_root"]
        )
        self.dataset = self.easyedit_root / str(p0_value["counterfact"]["path"])
        self.model_seal = self.seal.model(MODEL_ALIAS)
        self.snapshot = Path(self.model_seal.snapshot_path)
        self.prior_final_pre_gpu_path = prior_final_pre_gpu_path
        self.prior_final_pre_gpu = json.loads(
            prior_final_pre_gpu_path.read_text(encoding="utf-8")
        )
        self.stream_receipt = dict(stream_receipt)
        self._fingerprints: dict[Path, tuple[int, int, int, int, int]] = {}
        if (
            self.model_seal.closure_identity != EXPECTED_HF_CLOSURE
            or self.model_seal.required_root != EXPECTED_HF_REQUIRED_ROOT
            or self.model_seal.revision != self.spec["revision"]
            or self.model_seal.native_name != self.spec["native_name"]
        ):
            raise ODEBFContractError("target-timescale HF/P0 alias binding differs")

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
        observed = path.stat()
        if not stat.S_ISREG(observed.st_mode):
            raise ODEBFContractError("target-timescale HF member is not regular")
        return (
            int(observed.st_dev),
            int(observed.st_ino),
            int(observed.st_size),
            int(observed.st_mtime_ns),
            int(observed.st_ctime_ns),
        )

    def preflight(self) -> TargetTimescaleBaseReceipt:
        row = self.prior_final_pre_gpu.get("hf_closure_binding", {}).get(
            MODEL_ALIAS
        )
        if (
            self.prior_final_pre_gpu_path.is_symlink()
            or not self.prior_final_pre_gpu_path.is_file()
            or self.prior_final_pre_gpu.get("status") != "FINAL_PRE_GPU_PASS"
            or self.prior_final_pre_gpu.get("model_load_authorized") is not True
            or self.prior_final_pre_gpu.get("hf_content_verification")
            != "PRIOR_ACCEPTED_EXACT_CONSUMED_CLOSURE_REUSED"
            or self.prior_final_pre_gpu.get("hf_duplicate_rehash_count") != 0
            or not isinstance(row, Mapping)
            or row.get("closure_identity") != EXPECTED_HF_CLOSURE
            or row.get("required_root") != EXPECTED_HF_REQUIRED_ROOT
            or row.get("snapshot_path") != str(self.snapshot)
            or self.stream_receipt.get("status") != "TRANSFER_FULL_READ_PASS"
            or self.stream_receipt.get("full_read") is not True
        ):
            raise ODEBFContractError("target-timescale prior HF proof differs")
        binding = self.stream_receipt.get("binding")
        if (
            not isinstance(binding, Mapping)
            or binding.get("stream_root") != EXPECTED_STREAM_ROOT
            or binding.get("order_root") != EXPECTED_STREAM_ORDER
            or binding.get("evaluator_identity") != EXPECTED_EVALUATOR_IDENTITY
            or binding.get("model_aliases") != [
                "llama3-8b-inst",
                "qwen2.5-7b-inst",
            ]
            or not binding.get("dataset_identity")
        ):
            raise ODEBFContractError("target-timescale stream/HF gate differs")
        if (
            self.snapshot.is_symlink()
            or not self.snapshot.is_dir()
            or self.snapshot.resolve(strict=True) != self.snapshot
        ):
            raise ODEBFContractError("target-timescale pinned snapshot path differs")
        for member in self.model_seal.required_members:
            link = self.snapshot / member.relative_path
            if not link.is_symlink() or os.readlink(link) != member.link_target:
                raise ODEBFContractError("target-timescale HF member link differs")
            resolved = link.resolve(strict=True)
            fingerprint = self._fingerprint(resolved)
            if fingerprint[2] != member.size:
                raise ODEBFContractError("target-timescale HF member size differs")
            self._fingerprints[resolved] = fingerprint
        dataset_lock = self.p0_value["counterfact"]
        dataset = self.dataset
        if (
            dataset.is_symlink()
            or self._fingerprint(dataset)[2] != int(dataset_lock["size"])
            or _sha256(dataset) != str(dataset_lock["sha256"])
        ):
            raise ODEBFContractError("target-timescale CounterFact identity differs")
        self._fingerprints[dataset] = self._fingerprint(dataset)
        for layer_text, locked in sorted(
            self.spec["covariance"].items(), key=lambda item: int(item[0])
        ):
            relative, expected_size, expected_sha = locked
            covariance = self.easyedit_root / str(relative)
            if (
                covariance.is_symlink()
                or self._fingerprint(covariance)[2] != int(expected_size)
                or _sha256(covariance) != str(expected_sha)
            ):
                raise ODEBFContractError(
                    f"target-timescale covariance L{layer_text} differs"
                )
            self._fingerprints[covariance] = self._fingerprint(covariance)
        return TargetTimescaleBaseReceipt(
            revision=self.model_seal.revision,
            snapshot=str(self.snapshot),
            closure_identity=self.model_seal.closure_identity,
            required_root=self.model_seal.required_root,
            prior_full_verification_receipt_sha256=_sha256(
                self.prior_final_pre_gpu_path
            ),
            bounded_member_fingerprint_count=len(self._fingerprints),
            extra_influence_count=0,
        )

    def assert_unchanged(self) -> None:
        if not self._fingerprints or any(
            self._fingerprint(path) != expected
            for path, expected in self._fingerprints.items()
        ):
            raise ODEBFContractError("target-timescale HF closure changed")

    def load_full_fp32(self, alias: str) -> tuple[Any, Any, Any]:
        return load_p4_full_fp32_from_sealed_snapshot(
            self.seal,
            alias,
            stream_receipt=self.stream_receipt,
            final_pre_gpu_receipt=self.prior_final_pre_gpu,
            reuse_final_verified_closure=True,
        )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TargetTimescaleDeployment:
    runtime_path_seal: AlphaEditRuntimePathSeal
    base_guard: TargetTimescaleHFBaseGuard
    evaluator_source_paths: Mapping[str, Path]
    stream_receipt: Mapping[str, object]
    identity_sha256: str


def build_target_timescale_deployment(
    *,
    repo_root: Path,
    stream_extract_root: Path,
    prior_final_pre_gpu_path: Path,
    dataset_identity: str,
) -> TargetTimescaleDeployment:
    runtime_seal = load_alphaedit_runtime_path_seal(
        repo_root / "agents/server4/alphaedit-runtime-path-seal.json",
        repo_root=repo_root,
    )
    evaluator_paths = {
        "experiments/py/eval_utils_counterfact.py": stream_extract_root
        / "evaluator/eval_utils_counterfact.py",
        "experiments/py/eval_utils_zsre.py": stream_extract_root
        / "evaluator/eval_utils_zsre.py",
        "experiments/summarize.py": stream_extract_root / "evaluator/summarize.py",
    }
    stream_receipt: dict[str, object] = {
        "schema": "ode-edit-s05-p1r52-target-timescale-stream-hf-binding/v1",
        "status": "TRANSFER_FULL_READ_PASS",
        "full_read": True,
        "binding": {
            "stream_root": EXPECTED_STREAM_ROOT,
            "order_root": EXPECTED_STREAM_ORDER,
            "evaluator_identity": EXPECTED_EVALUATOR_IDENTITY,
            "dataset_identity": dataset_identity,
            "model_aliases": ["llama3-8b-inst", "qwen2.5-7b-inst"],
            "selected_model_alias": MODEL_ALIAS,
            "selected_batch": "B1",
            "sample_duplication_count": 0,
        },
    }
    stream_receipt["identity_sha256"] = canonical_hash(stream_receipt)
    base_guard = TargetTimescaleHFBaseGuard(
        repo_root=repo_root,
        hf_seal_path=repo_root / "agents/server4/p4-hf-consumed-closure-seal.json",
        p0_lock_path=repo_root
        / "project/run_scripts/ode_alloc/p0_artifact_lock_r1.json",
        runtime_path_seal=runtime_seal,
        prior_final_pre_gpu_path=prior_final_pre_gpu_path,
        stream_receipt=stream_receipt,
    )
    payload = {
        "runtime_path_seal_root": runtime_seal.root_digest,
        "hf_closure_identity": base_guard.model_seal.closure_identity,
        "hf_required_root": base_guard.model_seal.required_root,
        "snapshot_path": str(base_guard.snapshot),
        "stream_binding": stream_receipt["identity_sha256"],
        "evaluator_paths": {
            name: str(path) for name, path in sorted(evaluator_paths.items())
        },
    }
    return TargetTimescaleDeployment(
        runtime_seal,
        base_guard,
        evaluator_paths,
        stream_receipt,
        canonical_hash(payload),
    )


__all__ = [
    "EXPECTED_EVALUATOR_IDENTITY",
    "EXPECTED_HF_CLOSURE",
    "EXPECTED_HF_REQUIRED_ROOT",
    "EXPECTED_STREAM_ORDER",
    "EXPECTED_STREAM_ROOT",
    "TargetTimescaleBaseReceipt",
    "TargetTimescaleDeployment",
    "TargetTimescaleHFBaseGuard",
    "build_target_timescale_deployment",
]
