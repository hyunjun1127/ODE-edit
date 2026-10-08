"""Atomic batch checkpoints for native W/cache_c, RNG and immutable run identity."""
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .prepare import digest, file_sha


IDENTITY_FIELDS = ("config_sha256", "stream_sha256", "code_commit", "official_tree_sha256",
                   "model_revision", "tokenizer_sha256", "assets_sha256")


def rng_snapshot():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def rng_restore(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA_RNG_RESTORE_REQUIRES_CUDA")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_identity(identity):
    if set(identity) != set(IDENTITY_FIELDS) or any(not identity[k] for k in IDENTITY_FIELDS):
        raise ValueError("INCOMPLETE_CHECKPOINT_IDENTITY")


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(folder, *, batch, weights, cache_c, contexts, evaluation_cursor, identity,
         method, evaluation_complete):
    """Call only after that batch's edit and scheduled evaluations both finish."""
    validate_identity(identity)
    if not evaluation_complete or type(batch) is not int or not 0 <= batch <= 20:
        raise ValueError("BATCH_NOT_COMMITTABLE")
    has_history = method in ("ALPHAEDIT", "ALPHAEDIT_BLUE", "SPHERE")
    if has_history != bool(cache_c):
        raise ValueError("NATIVE_HISTORY_SCHEMA")
    if not weights or any(t.dtype != torch.float32 for t in weights.values()):
        raise ValueError("FP32_WEIGHTS_REQUIRED")
    if any(t.dtype != torch.float32 for t in cache_c.values()):
        raise ValueError("FP32_CACHE_C_REQUIRED")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pointer = folder / "latest.json"
    previous = json.loads(pointer.read_text()) if pointer.exists() else None
    if previous:
        # Avoid deserializing multi-GiB H matrices a second time at every save.
        if previous.get("identity_sha256") != digest(identity):
            raise ValueError("CHECKPOINT_IDENTITY_MISMATCH")
        if Path(previous["file"]).name != previous["file"]:
            raise ValueError("INVALID_CHECKPOINT_MEMBER")
        if not (folder / previous["file"]).is_file():
            raise ValueError("PREVIOUS_CHECKPOINT_MISSING")
        if batch != previous["batch"] + 1:
            raise ValueError("NONCONTIGUOUS_CHECKPOINT_BATCH")
    elif batch != 0:
        raise ValueError("INITIAL_W0_CHECKPOINT_REQUIRED")
    payload = dict(schema="official-baseline-checkpoint-v1", batch=batch, identity=identity,
                   method=method, weights={k: v.detach().cpu() for k, v in weights.items()},
                   cache_c={k: v.detach().cpu() for k, v in cache_c.items()}, contexts=contexts,
                   rng=rng_snapshot(), evaluation_cursor=evaluation_cursor)
    temporary = target = None
    committed = False
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".checkpoint-", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        checksum = file_sha(temporary)
        target = folder / f"batch-{batch:02d}-{checksum[:16]}.pt"
        os.replace(temporary, target)
        temporary = None
        if file_sha(target) != checksum:
            raise ValueError("CHECKPOINT_HASH_MISMATCH")
        ref = dict(batch=batch, file=target.name, sha256=checksum, final_W20=batch == 20,
                   identity_sha256=digest(identity))
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".latest-", mode="w", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(ref, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pointer)
        temporary = None
        committed = True
        _sync_directory(folder)
        # Also reclaim unreferenced files left by an interrupted earlier save.
        # This directory belongs to one run and may have only one writer.
        for stale in folder.glob("batch-??-????????????????.pt"):
            if stale != target:
                stale.unlink()
        _sync_directory(folder)
        return ref
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if target is not None and not committed:
            target.unlink(missing_ok=True)


def load(folder, identity):
    validate_identity(identity)
    folder = Path(folder)
    ref = json.loads((folder / "latest.json").read_text())
    if ref.get("identity_sha256") != digest(identity):
        raise ValueError("CHECKPOINT_IDENTITY_MISMATCH")
    if Path(ref["file"]).name != ref["file"]:
        raise ValueError("INVALID_CHECKPOINT_MEMBER")
    path = folder / ref["file"]
    if file_sha(path) != ref["sha256"]:
        raise ValueError("CHECKPOINT_HASH_MISMATCH")
    # Only this run's own local checkpoints are inputs; arbitrary .pt files are not accepted.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["identity"] != identity or payload["batch"] != ref["batch"]:
        raise ValueError("CHECKPOINT_IDENTITY_MISMATCH")
    return payload
