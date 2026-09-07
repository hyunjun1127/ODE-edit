"""Create-once local-only raw evidence, incremental factors and endpoint bytes.

Tensor files are never publication members. Saving an increment journal does
not certify replay: only verify_reconstruction performs and records that test.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

import torch

from project.run_scripts.ordered_response_barrier_ode.fp32_overlay import tensor_set_sha256, tensor_sha256
from project.run_scripts.ordered_response_barrier_ode.preflight import canonical_hash


class RawStoreBoundary(RuntimeError):
    pass


def plain_tree(value: Any):
    """Clone complete dataclass fields into torch weights-only readable types."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone().contiguous()
    if is_dataclass(value):
        return {field.name: plain_tree(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): plain_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain_tree(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise RawStoreBoundary("UNSUPPORTED_RAW_SCHEMA_TYPE: " + type(value).__name__)


def tree_identity(value: Any):
    def view(item):
        if isinstance(item, torch.Tensor):
            return {"tensor_sha256": tensor_sha256(item), "shape": list(item.shape), "dtype": str(item.dtype)}
        if is_dataclass(item):
            return {field.name: view(getattr(item, field.name)) for field in fields(item)}
        if isinstance(item, dict):
            return {str(key): view(val) for key, val in item.items()}
        if isinstance(item, (tuple, list)):
            return [view(val) for val in item]
        return item
    return canonical_hash(view(value))


def _reject_links(path):
    for member in (path, *path.parents):
        try:
            mode = member.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise RawStoreBoundary("SYMLINK_RAW_PATH: " + str(member))


class LocalRawStore:
    def __init__(self, root: Path, *, contexts, source_identity):
        self.root = Path(root).absolute()
        _reject_links(self.root)
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.contexts = plain_tree(contexts)
        self.source_identity = plain_tree(source_identity)
        self._journals = {}
        self._entry = None
        self._endpoints = {}

    def _path(self, relative):
        child = Path(relative)
        if child.is_absolute() or ".." in child.parts or not child.parts:
            raise RawStoreBoundary("RAW_ROOT_CONTAINMENT")
        path = self.root / child
        _reject_links(path)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_links(path)
        return path

    def _member(self, path):
        _reject_links(path)
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            raise RawStoreBoundary("RAW_FILE_TYPE_OR_MODE")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return dict(path=str(path), bytes=info.st_size, sha256=digest.hexdigest(), mode="0600",
                    regular_non_symlink=True, publication_allowed=False)

    def _tensor_file(self, relative, value):
        path = self._path(relative)
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as handle:
            torch.save(plain_tree(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
        reloaded = torch.load(path, map_location="cpu", weights_only=True)
        if tree_identity(reloaded) != tree_identity(plain_tree(value)):
            raise RawStoreBoundary("RAW_SAVE_RELOAD_IDENTITY")
        return dict(self._member(path), content_identity=tree_identity(reloaded))

    def _json(self, relative, value):
        path = self._path(relative)
        payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return self._member(path)

    def fixed_target(self, bundle, entry, normalization, reference):
        if self._entry is not None or canonical_hash(self.contexts) != bundle.contexts_sha256:
            raise RawStoreBoundary("FIXED_TARGET_CONTEXT_OR_REPEAT")
        entry.assert_sealed()
        normalization.assert_frozen()
        self._entry = entry
        state = self._tensor_file("entry-state.pt", dict(weights=entry.weights, alpha_cache=entry.alpha_cache,
                                  cache_c_new=entry.cache_c_new, W_sha256=entry.W_sha256, M_sha256=entry.M_sha256))
        target = self._tensor_file("fixed-target.pt", dict(bundle=bundle, contexts=self.contexts,
            source_identity=self.source_identity, source_scales=normalization.scales,
            active_mask=normalization.active, denominators=normalization.denominators,
            normalization=normalization.receipt(), reference=reference))
        return self._json("fixed-target-receipt.json", dict(schema="alpha-jv.raw-fixed-target.v1",
            source_identity=self.source_identity, target=target, entry=state,
            contexts_sha256=bundle.contexts_sha256, fixed_z_sha256=bundle.artifact.identity_sha256,
            full_semantic_inventory_saved=True, fixed_z_recompute_count=0,
            raw_scope="LOCAL_ONLY_NO_GIT", replay_verified=False))

    def journal(self, path_id, deltas):
        self._safe_id(path_id)
        current = self._journals.setdefault(path_id, [])
        incoming = [tree_identity(delta) for delta in deltas]
        if len(incoming) < len(current) or incoming[:len(current)] != [row["factor_identity"] for row in current]:
            raise RawStoreBoundary("CHRONOLOGICAL_JOURNAL_PREFIX_CHANGED")
        for ordinal in range(len(current), len(deltas)):
            delta = deltas[ordinal]
            member = self._tensor_file(f"{path_id}/journal/increment-{ordinal:04}.pt", delta)
            current.append(dict(ordinal=ordinal, factor_identity=incoming[ordinal], member=member))
        return dict(increment_count=len(current), ordered_factor_root=canonical_hash(incoming),
                    replay_verified=False, status="STORED_NOT_RECONSTRUCTED")

    @staticmethod
    def _safe_id(value):
        if not isinstance(value, str) or not value or Path(value).name != value or value in (".", ".."):
            raise RawStoreBoundary("UNSAFE_RAW_ID")

    def node(self, path_id, node, payload):
        self._safe_id(path_id)
        if not isinstance(node, int) or isinstance(node, bool) or node < 0:
            raise RawStoreBoundary("RAW_NODE_INDEX")
        stage = payload.get("stage", "PRE_NODE")
        if stage not in ("PRE_NODE", "POST_NODE"):
            raise RawStoreBoundary("RAW_NODE_STAGE")
        if set(payload) & {"weights", "shadow_weights", "entry_weights", "alpha_cache", "state_dict"}:
            raise RawStoreBoundary("DENSE_NODE_STATE_NOT_AUTHORIZED")
        factors = payload.get("increments_after", payload.get("increments_before", ()))
        journal = self.journal(path_id, factors)
        values = {key: val for key, val in payload.items() if key not in ("increments_after", "increments_before")}
        member = self._tensor_file(f"{path_id}/node-{node:02}-{stage.lower()}.pt", values)
        return self._json(f"{path_id}/node-{node:02}-{stage.lower()}-receipt.json",
                          dict(node=node, stage=stage, raw=member, journal=journal, dense_W_per_node_saved=False))

    def endpoint(self, path_id, label, endpoint, weights, activation, entry, method_state, increments=()):
        self._safe_id(path_id)
        self._safe_id(label)
        entry.assert_sealed()
        if self._entry is None or entry.W_sha256 != self._entry.W_sha256 or entry.M_sha256 != self._entry.M_sha256:
            raise RawStoreBoundary("ENDPOINT_ENTRY_BINDING")
        actual = tensor_set_sha256(weights)
        if actual != endpoint.get("selected_weight_endpoint_sha256"):
            raise RawStoreBoundary("ENDPOINT_ACTUAL_WEIGHT_SHA")
        if tensor_sha256(activation) != endpoint.get("terminal_activation_sha256"):
            raise RawStoreBoundary("ENDPOINT_ACTIVATION_SHA")
        prefix = endpoint.get("history_append_count") == 0
        if (not isinstance(method_state, dict) or not isinstance(method_state.get("alpha_cache"), torch.Tensor)
                or not isinstance(method_state.get("cache_c_new"), bool)
                or (prefix and (tensor_sha256(method_state["alpha_cache"]) != entry.M_sha256
                                or method_state["cache_c_new"] != entry.cache_c_new))):
            raise RawStoreBoundary("ENDPOINT_METHOD_STATE_BOUNDARY")
        journal = self.journal(path_id, increments)
        state = dict(weights=weights, alpha_cache=method_state["alpha_cache"],
                     cache_c_new=method_state["cache_c_new"], activation=activation,
                     W_sha256=actual, M_sha256=tensor_sha256(method_state["alpha_cache"]))
        member = self._tensor_file(f"{path_id}/endpoint-{label}.pt", state)
        receipt = dict(schema="alpha-jv.raw-endpoint.v1", path_id=path_id, candidate_id=label,
            snapshot=member, W_sha256=actual, M_sha256=state["M_sha256"],
            cache_c_new=state["cache_c_new"], cache_flag_inferred=False,
            method_state_semantics="ENTRY_M_UNAPPENDED" if prefix else "CAPTURED_TERMINAL_M",
            effective_T=endpoint.get("effective_T"), effective_N=endpoint.get("effective_N"),
            parent_T=endpoint.get("parent_T"), parent_N=endpoint.get("parent_N"),
            journal=journal, replay_verified=False, source_identity=self.source_identity)
        self._endpoints[(path_id, label)] = receipt
        return self._json(f"{path_id}/endpoint-{label}-receipt.json", receipt)

    def verify_reconstruction(self, path_id, label):
        """Explicit CPU replay of saved low-rank bytes; not implied by storage."""
        receipt = self._endpoints[(path_id, label)]
        entry = torch.load(self.root / "entry-state.pt", map_location="cpu", weights_only=True)
        replay = {name: value.clone() for name, value in entry["weights"].items()}
        for row in self._journals[path_id][:receipt["journal"]["increment_count"]]:
            delta = torch.load(row["member"]["path"], map_location="cpu", weights_only=True)
            replay[delta["weight_name"]].addmm_(delta["left"], delta["right"].T, alpha=delta["coefficient"])
        replay_sha = tensor_set_sha256(replay)
        matched = replay_sha == receipt["W_sha256"]
        result = dict(status="EXACT_RECONSTRUCTION_PASS" if matched else "RECONSTRUCTION_MISMATCH",
                      actual_cpu_replay_performed=True, replay_verified=matched,
                      replay_W_sha256=replay_sha, endpoint_W_sha256=receipt["W_sha256"])
        self._json(f"{path_id}/endpoint-{label}-reconstruction.json", result)
        if not matched:
            raise RawStoreBoundary("RECONSTRUCTION_MISMATCH")
        return result
