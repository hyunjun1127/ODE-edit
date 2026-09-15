"""CPU checkpoint publication for SL-ZFlow; no model/parity claim is made here.

The caller verifies actual Llama materialization and supplies its immutable
parity/cost evidence.  This module prepares W/M without mutating live tensors,
serializes a complete bundle, hashes it, and atomically publishes it.  A bundle
is the resume authority, not an independently written weight file or log.

Failed private staging directories are deliberately retained as evidence.
Linux renameat2(RENAME_NOREPLACE) and directory fsync are required: silently
falling back to an overwrite-capable rename would weaken the contract.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable, Mapping

import torch

from .transaction import CommitConflict, CommitError, materialized_cost


FORMAT = "SL_ZFLOW_DURABLE_V1"
_BATCH_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


class CheckpointIntegrityError(CommitError):
    """A bundle is incomplete, corrupt, or incompatible with the caller."""


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")


def _json_copy(value: Any) -> Any:
    # Also prohibits tensors, arbitrary Python objects, and NaN metadata.
    return json.loads(_json_bytes(value))


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Dtype/shape/data digest, independent of torch.save archive metadata."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256(_json_bytes({"shape": list(tensor.shape),
                                       "dtype": str(tensor.dtype)}))
    # Byte view supports bfloat16 too, without a full bytes() duplicate.
    view = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
    for offset in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[offset:offset + 8 * 1024 * 1024])
    return digest.hexdigest()


def _owned_payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone(memory_format=torch.contiguous_format)
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _owned_payload(member) for key, member in value.items()}
    if isinstance(value, tuple):
        return tuple(_owned_payload(member) for member in value)
    if isinstance(value, list):
        return [_owned_payload(member) for member in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("RNG payload must contain only finite primitives and tensors")


def _descriptor(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": {"shape": list(value.shape), "dtype": str(value.dtype),
                           "sha256": tensor_sha256(value)}}
    if isinstance(value, dict):
        return {"dict": {key: _descriptor(member) for key, member in sorted(value.items())}}
    if isinstance(value, tuple):
        return {"tuple": [_descriptor(member) for member in value]}
    if isinstance(value, list):
        return {"list": [_descriptor(member) for member in value]}
    return value


def _fp32_matrix(value: torch.Tensor, name: str) -> None:
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            or value.dtype != torch.float32 or value.ndim != 2
            or not bool(torch.isfinite(value).all())):
        raise ValueError(f"{name} must be a finite CPU FP32 matrix")


@dataclass(frozen=True)
class PreparedState:
    """Private state prepared without modifying caller-owned W/M."""
    tensors: dict[str, torch.Tensor]
    details: dict[str, Any]


def prepare_state(entry_weight: torch.Tensor, entry_history: torch.Tensor,
                  candidate_weight: torch.Tensor, x: torch.Tensor,
                  writer: torch.Tensor, metric: torch.Tensor,
                  keys: torch.Tensor, *, accepted: int, request_count: int,
                  lambda_write: float = 1.0,
                  parity_evidence: Mapping[str, Any] | None = None,
                  cost_evidence: Mapping[str, Any] | None = None) -> PreparedState:
    """Prepare a terminal candidate only, never a predictor/rejected trial.

    ``candidate_weight`` is the caller's actual FP32 materialization.  The
    caller's accepted-state parity and numerical cost gates must both pass.
    Evidence is bound but is not independently interpreted as Llama proof.
    Actual cost is independently computed from FP64 subtraction of stored
    FP32 endpoints.  M is formed in native CPU FP32 order, without subtraction
    rollback.  No-update requires X=0 and bit-identical entry W.
    """
    for value, name in ((entry_weight, "entry_weight"),
                        (entry_history, "entry_history"),
                        (candidate_weight, "candidate_weight"), (keys, "keys")):
        _fp32_matrix(value, name)
    if (not isinstance(accepted, int) or isinstance(accepted, bool) or accepted < 0
            or not isinstance(request_count, int) or isinstance(request_count, bool)
            or request_count <= 0 or not math.isfinite(lambda_write)
            or lambda_write < 0):
        raise ValueError("invalid accepted count or cost geometry")
    dout, din = entry_weight.shape
    if (candidate_weight.shape != entry_weight.shape
            or entry_history.shape != (din, din)
            or keys.shape != (din, request_count)):
        raise ValueError("weight/history/key dimensions differ")
    for value, name in ((x, "X"), (writer, "B"), (metric, "S")):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype not in (torch.float32, torch.float64)
                or value.ndim != 2 or not bool(torch.isfinite(value).all())):
            raise ValueError(f"{name} must be a finite CPU FP32/FP64 matrix")
    q = writer.shape[0]
    if x.shape != (dout, q) or writer.shape[1] != din or metric.shape != (q, q):
        raise ValueError("X/B/S dimensions differ")
    if accepted == 0:
        if bool(torch.count_nonzero(x)) or not torch.equal(candidate_weight, entry_weight):
            raise ValueError("no-update requires X=0 and unchanged entry weight")
        append = 0
        next_history = entry_history.clone()
        actual_cost = 0.0
    else:
        if not isinstance(parity_evidence, Mapping) or parity_evidence.get("passed") is not True:
            raise CommitError("COMMIT_PARITY_FAIL: passed actual-write evidence required")
        if not isinstance(cost_evidence, Mapping) or cost_evidence.get("passed") is not True:
            raise CommitError("COMMIT_COST_FAIL: passed materialized cost evidence required")
        actual_delta = candidate_weight.double() - entry_weight.double()
        actual_cost = materialized_cost(actual_delta, entry_history,
                                        lambda_write, request_count)
        if not math.isfinite(actual_cost):
            raise CommitError("COMMIT_COST_FAIL: nonfinite actual cost")
        with torch.no_grad():
            gram = keys @ keys.T  # CPU FP32 product; do not reassociate.
            next_history = entry_history + gram
        if not bool(torch.isfinite(next_history).all()):
            raise CommitError("nonfinite committed history")
        append = 1
    tensors = {name: value.detach().clone(memory_format=torch.contiguous_format)
               for name, value in (("W", candidate_weight), ("M", next_history),
                                   ("X", x), ("B", writer), ("S", metric), ("K", keys))}
    details = {
        "entry_weight_sha256": tensor_sha256(entry_weight),
        "entry_history_sha256": tensor_sha256(entry_history),
        "weight_sha256": tensor_sha256(tensors["W"]),
        "history_sha256": tensor_sha256(tensors["M"]),
        "accepted": accepted, "history_append": append,
        "status": "COMMITTED" if accepted else "NO_UPDATE",
        "request_count": request_count, "lambda_write": lambda_write,
        "actual_delta_cost_fp64": actual_cost,
        "history_order": "CPU_FP32_K_MATMUL_KT_THEN_M_ENTRY_PLUS_GRAM",
        "parity_evidence": _json_copy(dict(parity_evidence or {})),
        "cost_evidence": _json_copy(dict(cost_evidence or {})),
        "tensor_sha256": {name: tensor_sha256(value) for name, value in tensors.items()},
    }
    return PreparedState(tensors, details)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _rename_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise CommitError("durable publication requires Linux renameat2")
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                       ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise CommitConflict("checkpoint destination already exists")
        raise OSError(code, os.strerror(code), str(destination))


class CheckpointStore:
    """Create-once checkpoint store with an advisory single-writer lock.

    A parent must be an already verified bundle in this store.  The first
    scientific batch has index 1, parent=None, and next_batch_index=2.  A
    no-update batch is still a complete chain node and advances that index.
    ``source`` and ``config`` are immutable JSON identity/config objects;
    context may contain native prompt material and belongs only in local/.
    """
    def __init__(self, root: str | Path):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("checkpoint root must be a real directory")

    def _path(self, batch_id: str) -> Path:
        if not isinstance(batch_id, str) or not _BATCH_ID.fullmatch(batch_id):
            raise ValueError("batch_id must be one safe path component")
        return self.root / batch_id

    def publish(self, batch_id: str, parent_receipt: Mapping[str, Any] | None,
                next_batch_index: int, prepared: PreparedState, *,
                config: Mapping[str, Any], source: Mapping[str, Any],
                context: Any, rng: Mapping[str, Any], ledger: Any,
                cache_resume_fingerprint: str,
                crash_hook: Callable[[str], None] | None = None) -> dict[str, Any]:
        destination = self._path(batch_id)
        if (not isinstance(next_batch_index, int) or isinstance(next_batch_index, bool)
                or next_batch_index < 2 or not isinstance(cache_resume_fingerprint, str)
                or not cache_resume_fingerprint or not config or not source or not rng):
            raise ValueError("complete config/source/RNG/cache identity and next index required")
        if set(prepared.tensors) != {"W", "M", "X", "B", "S", "K"}:
            raise ValueError("complete W/M/X/B/S/K prepared state required")
        # Own a stable snapshot throughout hashing/serialization. PreparedState
        # freezes its fields, not PyTorch storage, so caller mutation must not
        # produce an acknowledged bundle that later fails its own manifest.
        payload = _owned_payload({"tensors": prepared.tensors, "rng": dict(rng)})
        for name, identity in (("W", "weight_sha256"), ("M", "history_sha256")):
            _fp32_matrix(payload["tensors"][name], name)
            if tensor_sha256(payload["tensors"][name]) != prepared.details[identity]:
                raise CommitConflict("prepared endpoint was modified before publication")
        if {name: tensor_sha256(value) for name, value in payload["tensors"].items()} != prepared.details["tensor_sha256"]:
            raise CommitConflict("prepared tensor inventory was modified before publication")
        if (prepared.details["history_append"] != int(prepared.details["accepted"] > 0)
                or prepared.details["status"] != ("COMMITTED" if prepared.details["accepted"] else "NO_UPDATE")):
            raise CommitConflict("prepared append/status disagrees with accepted count")
        parent = None
        if parent_receipt is not None:
            parent = {key: parent_receipt[key] for key in
                      ("batch_id", "payload_sha256", "receipt_sha256")}
        metadata = {
            "format": FORMAT, "batch_id": batch_id, "parent": parent,
            "batch_index": next_batch_index - 1, "next_batch_index": next_batch_index,
            "config": _json_copy(dict(config)), "source": _json_copy(dict(source)),
            "context": _json_copy(context), "ledger": _json_copy(ledger),
            "cache_resume_fingerprint": cache_resume_fingerprint,
            "prepared": _json_copy(prepared.details),
        }
        metadata["config_sha256"] = _sha_bytes(_json_bytes(metadata["config"]))
        metadata["source_sha256"] = _sha_bytes(_json_bytes(metadata["source"]))
        payload_descriptor = _descriptor(payload)
        payload_sha = _sha_bytes(_json_bytes({"metadata": metadata,
                                             "payload": payload_descriptor}))
        lock_fd = os.open(self.root / ".publish.lock",
                          os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if destination.exists() or destination.is_symlink():
                existing = self.load(batch_id)
                if existing["receipt"]["payload_sha256"] != payload_sha:
                    raise CommitConflict("batch ID already commits a different payload")
                return dict(existing["receipt"], replayed=True)
            self._check_parent(parent, metadata, prepared)
            staging = Path(tempfile.mkdtemp(prefix=f".{batch_id}.partial-", dir=self.root))
            def boundary(name: str) -> None:
                if crash_hook is not None:
                    crash_hook(name)
            with (staging / "state.pt").open("xb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            boundary("after_payload")
            _write_json(staging / "metadata.json", metadata)
            boundary("after_metadata")
            members = [{"path": name, "bytes": (staging / name).stat().st_size,
                        "sha256": _file_sha(staging / name)}
                       for name in ("state.pt", "metadata.json")]
            manifest = {"format": FORMAT, "status": "COMPLETE", "batch_id": batch_id,
                        "payload_sha256": payload_sha, "payload_descriptor": payload_descriptor,
                        "members": members}
            _write_json(staging / "manifest.json", manifest)
            _fsync_directory(staging)
            boundary("after_manifest")
            boundary("before_publish")
            _rename_no_replace(staging, destination)
            _fsync_directory(self.root)
            boundary("after_publish")
            return self._receipt(destination, metadata, manifest, replayed=False)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _check_parent(self, parent: dict[str, str] | None,
                      metadata: dict[str, Any], prepared: PreparedState) -> None:
        if parent is None:
            if metadata["batch_index"] != 1:
                raise CommitConflict("only the first batch may omit its parent")
            return
        loaded = self.load(parent["batch_id"])
        receipt = loaded["receipt"]
        prior = loaded["metadata"]
        if any(receipt[key] != value for key, value in parent.items()):
            raise CommitConflict("parent receipt identity mismatch")
        if prior["next_batch_index"] != metadata["batch_index"]:
            raise CommitConflict("next batch index does not continue parent")
        if (prior["config_sha256"] != metadata["config_sha256"]
                or prior["source_sha256"] != metadata["source_sha256"]):
            raise CommitConflict("source/config changed inside a chain")
        for entry_key, endpoint_key in (("entry_weight_sha256", "weight_sha256"),
                                       ("entry_history_sha256", "history_sha256")):
            if prepared.details[entry_key] != prior["prepared"][endpoint_key]:
                raise CommitConflict("batch entry W/M does not match committed parent")

    @staticmethod
    def _receipt(path: Path, metadata: dict[str, Any], manifest: dict[str, Any],
                 *, replayed: bool) -> dict[str, Any]:
        receipt = {
            "format": FORMAT, "status": metadata["prepared"]["status"],
            "batch_id": metadata["batch_id"], "path": str(path),
            "payload_sha256": manifest["payload_sha256"],
            "manifest_sha256": _file_sha(path / "manifest.json"),
            "config_sha256": metadata["config_sha256"],
            "source_sha256": metadata["source_sha256"],
            "next_batch_index": metadata["next_batch_index"],
            "weight_sha256": metadata["prepared"]["weight_sha256"],
            "history_sha256": metadata["prepared"]["history_sha256"],
            "history_append": metadata["prepared"]["history_append"],
        }
        receipt["receipt_sha256"] = _sha_bytes(_json_bytes(receipt))
        receipt["replayed"] = replayed
        return receipt

    def load(self, batch_id: str, *, expected_config_sha256: str | None = None,
             expected_source_sha256: str | None = None,
             expected_cache_resume_fingerprint: str | None = None) -> dict[str, Any]:
        """Verify the entire bundle before deserializing any CPU tensor state.

        This does not restore a model/RNG or invoke a writer. The caller copies
        verified W/M, restores RNG, creates fresh batch caches and binds index.
        """
        path = self._path(batch_id)
        if path.is_symlink() or not path.is_dir():
            raise CheckpointIntegrityError("complete published checkpoint is absent")
        expected = {"state.pt", "metadata.json", "manifest.json"}
        if {member.name for member in path.iterdir()} != expected:
            raise CheckpointIntegrityError("checkpoint has incomplete/unexpected members")
        for name in expected:
            mode = (path / name).lstat().st_mode
            if not stat.S_ISREG(mode):
                raise CheckpointIntegrityError("checkpoint members must be regular non-symlink files")
        try:
            manifest = json.loads((path / "manifest.json").read_bytes())
            if (manifest["format"] != FORMAT or manifest["status"] != "COMPLETE"
                    or manifest["batch_id"] != batch_id
                    or [member["path"] for member in manifest["members"]]
                    != ["state.pt", "metadata.json"]):
                raise CheckpointIntegrityError("manifest schema or identity mismatch")
            for member in manifest["members"]:
                target = path / member["path"]
                if target.stat().st_size != member["bytes"] or _file_sha(target) != member["sha256"]:
                    raise CheckpointIntegrityError("checkpoint member SHA/size mismatch")
            metadata = json.loads((path / "metadata.json").read_bytes())
            if (metadata["format"] != FORMAT or metadata["batch_id"] != batch_id
                    or _sha_bytes(_json_bytes(metadata["config"])) != metadata["config_sha256"]
                    or _sha_bytes(_json_bytes(metadata["source"])) != metadata["source_sha256"]):
                raise CheckpointIntegrityError("metadata source/config/identity mismatch")
            for key, expected_value in (("config_sha256", expected_config_sha256),
                                        ("source_sha256", expected_source_sha256),
                                        ("cache_resume_fingerprint", expected_cache_resume_fingerprint)):
                if expected_value is not None and metadata[key] != expected_value:
                    raise CheckpointIntegrityError(f"resume {key} mismatch")
            payload = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
            if set(payload) != {"tensors", "rng"} or set(payload["tensors"]) != {"W", "M", "X", "B", "S", "K"}:
                raise CheckpointIntegrityError("incomplete tensor/RNG state")
            descriptor = _descriptor(payload)
            if descriptor != manifest["payload_descriptor"]:
                raise CheckpointIntegrityError("tensor/RNG semantic digest mismatch")
            if {name: tensor_sha256(value) for name, value in payload["tensors"].items()} != metadata["prepared"]["tensor_sha256"]:
                raise CheckpointIntegrityError("prepared tensor inventory mismatch")
            digest = _sha_bytes(_json_bytes({"metadata": metadata, "payload": descriptor}))
            if digest != manifest["payload_sha256"]:
                raise CheckpointIntegrityError("checkpoint payload digest mismatch")
            for name, key in (("W", "weight_sha256"), ("M", "history_sha256")):
                _fp32_matrix(payload["tensors"][name], name)
                if tensor_sha256(payload["tensors"][name]) != metadata["prepared"][key]:
                    raise CheckpointIntegrityError("endpoint tensor identity mismatch")
        except CheckpointIntegrityError:
            raise
        except Exception as error:
            raise CheckpointIntegrityError("checkpoint failed complete validation") from error
        return {"tensors": payload["tensors"], "rng": payload["rng"],
                "metadata": metadata,
                "receipt": self._receipt(path, metadata, manifest, replayed=True)}
