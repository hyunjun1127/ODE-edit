"""G0/G1 content binding and bounded actual native-key technical checks.

No import-time model/GPU work. ``validate_checkpoint_inventory`` is CPU-only;
``run_g1`` and ``reconstruct_timestamp_bank`` require an already bound runtime
and execute only when explicitly called by the authorized GPU runner. All new
persistent tensors are the approved diagnostic K bank, never W/M/RNG resume.

The original scalar SHA convention is used. No allclose threshold is invented:
an exact-key mismatch is FAIL with numerical envelope NOT_ESTABLISHED. A future
predeclared repeat envelope must be a separately sealed implementation/version.
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import digest, file_sha, rng_get, rng_set, save, tensor_file, tensor_sha


LAYERS = (4, 5, 6, 7, 8)
CHECKPOINT_BATCHES = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
TIMESTAMP_BATCHES = (1, 5, 10, 20, 30, 40)
MODEL_REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
ORDER_SHA256 = "5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729"


class TechnicalFailure(RuntimeError):
    def __init__(self, code: str, details: Any = None):
        self.code, self.details = code, details
        super().__init__(f"{code}: {details}")


def _require(condition: bool, code: str, details: Any = None) -> None:
    if not condition:
        raise TechnicalFailure(code, details)


def tensor_content(tensor: Any, *, shape: Sequence[int], name: str) -> dict:
    """Bounded CPU finite scan and exact original tensor SHA (mmap friendly)."""
    import torch
    _require(isinstance(tensor, torch.Tensor), "CP_NOT_TENSOR", name)
    _require(tensor.device.type == "cpu", "CP_INSPECTION_NOT_CPU", name)
    _require(tensor.dtype == torch.float32 and list(tensor.shape) == list(shape),
             "CP_TENSOR_SCHEMA", {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    _require(tensor.is_contiguous(), "CP_NONCONTIGUOUS_STORAGE", name)
    flat = tensor.view(-1)
    for start in range(0, flat.numel(), 1 << 20):
        _require(bool(torch.isfinite(flat[start:start + (1 << 20)]).all()), "CP_NONFINITE", name)
    return {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "elements": tensor.numel(), "bytes": tensor.numel() * tensor.element_size(),
            "finite": True, "sha256": tensor_sha(tensor)}


def validate_checkpoint_object(cp: Mapping[str, Any], *, batch: int, seen_ids: Sequence[int],
                               contexts: Any, original_lock_sha: str, source_archive_sha: str,
                               commit: Mapping[str, Any],
                               weight_shape: tuple[int, int] = (4096, 14336)) -> dict:
    """Independent CPU schema, content and original commit binding."""
    _require(set(cp) == {"weights", "cache_c", "metadata"}, "CP_TOPLEVEL_SCHEMA")
    meta = cp["metadata"]
    _require(meta["batch"] == batch and list(meta["seen_ids"]) == list(seen_ids), "CP_ORDER_OR_BATCH")
    _require(len(seen_ids) == batch * 100 and len(set(seen_ids)) == len(seen_ids), "CP_SEEN_CARDINALITY")
    _require(meta["method"] == "AlphaEdit" and meta["cache_c_is_history"] is True, "CP_METHOD_HISTORY")
    _require(meta["base_model_revision"] == MODEL_REVISION, "CP_MODEL_REVISION")
    _require(meta["sample_root"] == ORDER_SHA256, "CP_FIXED_ORDER")
    _require(meta["lock_sha256"] == original_lock_sha and meta["source"] == source_archive_sha,
             "CP_FROZEN_SOURCE_BINDING")
    _require(meta["contexts"] == contexts and list(map(len, contexts)) == [1, 5] and contexts[0] == ["{}"],
             "CP_CONTEXT_BINDING")
    _require(set(meta["rng"]) == {"python", "numpy", "torch", "cuda"}, "CP_RNG_SCHEMA")
    _require(commit["batch"] == batch and commit["history_append_passes"] == 1 and
             commit["status"] == "BATCH_COMMITTED" and commit["requests"] == 100 and
             commit["seen_requests"] == batch * 100 and commit["history_entries_out"] == batch * 100,
             "CP_ORIGINAL_COMMIT_HISTORY")
    _require(commit["context_hash"] == digest(contexts), "CP_ORIGINAL_COMMIT_CONTEXT")
    _require(commit["endpoint"] == meta["state"], "CP_ORIGINAL_COMMIT_CONTENT_IDENTITY")
    expected_names = {f"model.layers.{layer}.mlp.down_proj.weight" for layer in LAYERS}
    _require(set(cp["weights"]) == expected_names, "CP_SELECTED_LAYER_SET")
    content = {}
    for name in sorted(expected_names):
        row = tensor_content(cp["weights"][name], shape=weight_shape, name=name)
        _require(row["sha256"] == meta["state"]["weights"][name], "CP_WEIGHT_HASH", row)
        content[name] = row
    history = tensor_content(cp["cache_c"], shape=(5, weight_shape[1], weight_shape[1]), name="cache_c")
    _require(history["sha256"] == meta["state"]["cache"], "CP_HISTORY_HASH", history)
    return {"batch": batch, "status": "PASS", "weights": content, "history": history,
            "seen_count": len(seen_ids), "seen_ids_sha256": digest(list(seen_ids)),
            "contexts_sha256": digest(contexts), "rng_metadata_sha256": digest(meta["rng"]),
            "metadata_sha256": digest(meta), "original_commit_endpoint": commit["endpoint"],
            "cpu_load": "weights_only=True,mmap=True,map_location=cpu",
            "gpu_continuation": "NOT_TESTED"}


def validate_checkpoint_inventory(rt: Any, output: str | Path, *, rehash_files: bool = False) -> dict:
    """All12 CPU content verification; existing exact transfer file SHA reusable.

    Default uses the source/destination verified transfer receipt + current stat
    for archive identity, but independently hashes every W/M tensor and binds it
    to the historical commit. ``rehash_files=True`` additionally rereads 63.4 GB.
    This distinction is recorded, not reported as a fresh file-level rehash.
    """
    import torch
    output = Path(output)
    manifest = rt.cp_manifest
    transfer_path = rt.root / "receipts/checkpoint-transfer-r1.json"
    transfer = json.loads(transfer_path.read_text())
    _require(transfer["status"] == "PASS" and len(transfer["members"]) == 12, "CP_TRANSFER_NOT_COMPLETE")
    transfers = {r["destination_relative_path"]: r for r in transfer["members"]}
    _require(len(manifest["files"]) == 12 and len(transfers) == 12, "CP_MANIFEST_CARDINALITY")
    observed_batches, rows = [], []
    started = time.monotonic()
    try:
        for member in manifest["files"]:
            relative = member["path"]
            path = rt.cp_root / relative
            batch = int(Path(relative).parts[0][1:])
            observed_batches.append(batch)
            receipt = transfers[relative]
            _require(receipt["status"] in {"COPIED_VERIFIED", "REUSED_VERIFIED"}, "CP_TRANSFER_MEMBER_NOT_VERIFIED")
            _require(receipt["bytes"] == member["bytes"] and receipt["sha256"] == member["sha256"],
                     "CP_TRANSFER_MANIFEST_IDENTITY")
            _require(path.resolve() == Path(receipt["destination"]).resolve(), "CP_TRANSFER_PATH_IDENTITY")
            stat = path.stat()
            _require(path.is_file() and not path.is_symlink() and stat.st_size == member["bytes"], "CP_ACTUAL_FILE_SCHEMA")
            if rehash_files:
                _require(file_sha(path) == member["sha256"], "CP_FULL_FILE_HASH")
            cp = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            row = validate_checkpoint_object(
                cp, batch=batch, seen_ids=[int(r["case_id"]) for r in rt.rows[:batch * 100]],
                contexts=rt.contexts, original_lock_sha=rt.contract["source_lock"]["sha256"],
                source_archive_sha=rt.old["local_source_sha256"], commit=rt.local_sidecar(batch, "commit.json"))
            row.update(path=str(path), file_bytes=stat.st_size, file_sha256=member["sha256"],
                       file_verification="CURRENT_FULL_SHA" if rehash_files else "PRIOR_TRANSFER_FULL_SHA_PLUS_CURRENT_STAT",
                       tensor_content_verification="CURRENT_FULL_TENSOR_SHA",
                       current_stat={"device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
                                     "size": stat.st_size}, transfer_receipt_sha256=file_sha(transfer_path))
            rows.append(row)
            save(output / f"CP-B{batch:03d}.json", row)
            del cp
        _require(tuple(sorted(observed_batches)) == CHECKPOINT_BATCHES, "CP_EPOCH_SET")
        result = {"status": "PASS", "checkpoints": 12, "rows": rows,
                  "wall_seconds": time.monotonic() - started, "new_gpu": 0}
        result["receipt"] = save(output / "checkpoint-content.json", result)
        return result
    except Exception as exc:
        save(output / "checkpoint-content-failure.json", {"status": "FAIL", "exception": type(exc).__name__,
             "first_cause": str(exc), "completed_batches": [r["batch"] for r in rows],
             "new_gpu": 0, "wall_seconds": time.monotonic() - started})
        raise


def rng_fingerprint(state: Any) -> str:
    """No RNG sampling or initialization; works on an already captured tuple."""
    python, numpy, cpu, cuda = state
    return digest({"python": python, "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
                   "torch": cpu.tolist(), "cuda": [x.tolist() for x in cuda]})


def compare_exact(left: Any, right: Any, label: str) -> dict:
    import torch
    _require(left.shape == right.shape and left.dtype == right.dtype, "PARITY_SCHEMA", label)
    _require(bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all()), "PARITY_NONFINITE", label)
    a, b = tensor_sha(left), tensor_sha(right)
    difference = left.detach().cpu().double() - right.detach().cpu().double()
    return {"label": label, "left_sha256": a, "right_sha256": b,
            "shape": list(left.shape), "dtype": str(left.dtype), "exact": a == b,
            "max_abs": float(difference.abs().max()) if difference.numel() else 0.,
            "rms": float(difference.square().mean().sqrt()) if difference.numel() else 0.,
            "numerical_envelope": "NOT_ESTABLISHED; exact equality required"}


def _check_capture_pair(left: Mapping[str, Any], right: Mapping[str, Any], name: str) -> list[dict]:
    _require(left["token_receipt"] == right["token_receipt"], "PARITY_TOKEN_RECEIPT", name)
    rows = []
    for layer in LAYERS:
        for field in ("keys", "means"):
            rows.append(compare_exact(left[field][layer], right[field][layer], f"{name}/{field}/L{layer}"))
    return rows


def token_fixtures(rt: Any, records: Sequence[Mapping[str, Any]]) -> dict:
    """Actual writer IDs/lookups, no prompt strings in compact receipt."""
    requests = [r["requested_rewrite"] for r in records]
    templates = [context.format(r["prompt"]) for r in requests for group in rt.contexts for context in group]
    words = [r["subject"] for r in requests for group in rt.contexts for _ in group]
    indices = rt.repr.get_words_idxs_in_templates(rt.tok, templates, words, "last")
    texts = [p.format(w) for p, w in zip(templates, words, strict=True)]
    pack = rt.tok(texts, padding=True, return_tensors="pt")
    rows = []
    for index, (ids, mask, lookups) in enumerate(zip(pack["input_ids"], pack["attention_mask"], indices, strict=True)):
        count = int(mask.sum())
        _require(len(lookups) == 1 and 0 <= lookups[0] < count, "WRITER_LOOKUP_RANGE", index)
        _require(bool(mask[:count].all()) and not bool(mask[count:].any()), "WRITER_RIGHT_PADDING", index)
        values = ids[:count].tolist()
        _require(bool(values), "WRITER_EMPTY_VALID_TOKENS", index)
        _require(bool((ids[count:] == rt.tok.pad_token_id).all()), "WRITER_PADDING_TOKEN_ID", index)
        rows.append({"case_id": int(records[index // 6]["case_id"]), "context_index": index % 6,
                     "input_token_ids": values, "subject_last": int(lookups[0]), "valid_tokens": count,
                     "padding_tokens": len(ids) - count, "subject_token_id": int(values[lookups[0]])})
    _require(getattr(rt.tok, "add_bos_token", None) is False and rt.tok.padding_side == "right", "WRITER_TOKENIZER_CONFIG")
    from .token_binding import compare_reference
    _require(hasattr(rt,'writer_tokenizer_reference'), 'WRITER_NATIVE_REFERENCE_REQUIRED')
    native_binding=compare_reference(rt.tok,rows,rt.contexts,rt.writer_tokenizer_reference)
    return {"rows": rows, "sequences": len(rows), "writer_add_bos_token": False,
            "writer_add_bos_token_is_config_attribute_not_encoded_claim": True,
            "native_token_binding": native_binding,
            "actual_bos_sequences":native_binding['actual_bos_sequences'],
            "tokenizer_padding_side": "right", "microbatch_limit": 128, "sha256": digest(rows)}


def run_g1(rt: Any, output: str | Path, *, entry: int = 50) -> dict:
    """Tiny2 actual key parity/invariance and exact RAM-only state restoration."""
    import numpy as np
    import torch
    output = Path(output)
    _require(rt.M is not None, "G1_REQUIRES_LOADED_ENTRY")
    original = rt.snapshot()
    original_signature, original_rng = rt.signature(), rng_fingerprint(original["rng"])
    comparisons, timings = [], {}
    started = time.monotonic()
    first_error = None
    try:
        rt.set_state(entry)
        records = rt.rows[entry * 100:entry * 100 + 2]
        _require(len(records) == 2, "G1_TINY_PANEL_MISSING")
        fixtures = token_fixtures(rt, records)
        save(output / "G1-token-fixtures.json", fixtures)
        bound = rt.signature()
        state = rt.snapshot()
        t = time.monotonic(); prefix = rt.capture(records, contexts=True, full=False); timings["prefix_seconds"] = time.monotonic() - t
        t = time.monotonic(); full = rt.capture(records, contexts=True, full=True); timings["full_seconds"] = time.monotonic() - t
        t = time.monotonic(); repeat = rt.capture(records, contexts=True, full=False); timings["repeat_prefix_seconds"] = time.monotonic() - t
        comparisons += _check_capture_pair(prefix, full, "prefix_vs_full")
        comparisons += _check_capture_pair(prefix, repeat, "prefix_repeat")
        requests = [dict(r["requested_rewrite"], case_id=int(r["case_id"])) for r in records]
        t = time.monotonic()
        with torch.no_grad():
            for layer in LAYERS:
                native = rt.native.compute_ks(rt.model, rt.tok, requests, rt.hp, layer, rt.contexts)
                comparisons.append(compare_exact(prefix["means"][layer], native, f"prefix_mean_vs_native_compute_ks/L{layer}"))
                del native
        timings["five_original_compute_ks_seconds"] = time.monotonic() - t
        _require(rt.signature() == bound, "G1_OBSERVATION_STATE_MUTATION")
        # Own down-projection and all upper writes cannot change their input key.
        with torch.no_grad():
            scalar = rt.weights[8].view(-1)[0]
            before = float(scalar)
            scalar.copy_(torch.nextafter(scalar, torch.full_like(scalar, math.inf)))
            _require(float(scalar) != before, "G1_PERTURBATION_ZERO")
        changed = rt.capture(records, contexts=True, full=False)
        comparisons += _check_capture_pair(prefix, changed, "W8_own_K8_and_upper_lower_invariance")
        # Perturb all state families only in RAM, then verify bytes and RNG.
        with torch.no_grad():
            scalar = rt.M.view(-1)[0]
            scalar.copy_(torch.nextafter(scalar, torch.full_like(scalar, math.inf)))
        rt.batch += 1
        rt.contexts = [["changed-only-for-restore-check"]]
        rt.native.CONTEXT_TEMPLATES_CACHE = copy.deepcopy(rt.contexts)
        random.random(); np.random.random(); torch.rand(1); torch.rand(1, device=rt.weights[8].device)
        rt.restore(state)
        _require(rt.signature() == bound, "G1_SELECTED_W_M_CONTEXT_CURSOR_RESTORE")
        _require(rng_fingerprint(rng_get()) == rng_fingerprint(state["rng"]), "G1_RNG_RESTORE")
        rt.assert_nonselected()
        restored = rt.capture(records, contexts=True, full=False)
        comparisons += _check_capture_pair(prefix, restored, "post_restore_prefix")
        save(output / "G1-parity-comparisons.json", comparisons)
        _require(all(row["exact"] for row in comparisons), "G1_EXACT_PARITY_NOT_ESTABLISHED",
                 [r for r in comparisons if not r["exact"]])
        result = {"status": "PASS", "entry": entry, "requests": 2, "sequences": 12,
                  "case_ids": [int(r["case_id"]) for r in records], "comparisons": comparisons,
                  "timings": timings, "history_appends": 0, "target_optimizations": 0,
                  "restore": "W/M/context/cursor full hashes and RNG exact",
                  "nonselected_guard": "pointer/version; NOT independent full-byte nonselected hash",
                  "weight_delta_hook_parity": "WRITER_GATE_OWNED_NOT_TESTED_HERE",
                  "timestamp_reconstruction": "SEPARATE_GATE_NOT_TESTED_HERE"}
    except Exception as exc:
        first_error = exc
        save(output / "G1-first-failure.json", {"status": "FAIL", "exception": type(exc).__name__,
             "first_cause": str(exc), "comparisons": comparisons, "timings": timings,
             "numerical_envelope": "NOT_ESTABLISHED", "wall_seconds": time.monotonic() - started})
    finally:
        try:
            rt.restore(original)
            _require(rt.signature() == original_signature and rng_fingerprint(rng_get()) == original_rng,
                     "G1_CALLER_ENTRY_RESTORE")
        except Exception as cleanup:
            save(output / "G1-restore-failure.json", {"status": "FAIL", "cleanup_cause": str(cleanup),
                 "first_cause": str(first_error) if first_error else None})
            if first_error is None:
                raise
    if first_error is not None:
        raise first_error
    result["wall_seconds"] = time.monotonic() - started
    result["receipt"] = save(output / "G1.json", result)
    return result


def history_append_key_rows(observation: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    """Actual native-observation schema: five solve reads then five append reads."""
    keys = observation["keys"]
    _require(observation["history_append_passes"] == 1 and len(keys) == 10,
             "TIMESTAMP_ORIGINAL_HISTORY_KEY_COUNT")
    _require([int(row["layer"]) for row in keys] == list(LAYERS) * 2,
             "TIMESTAMP_ORIGINAL_KEY_LAYER_ORDER")
    _require(all(row["shape"] == [100, 14336] and len(row["sha256"]) == 64 for row in keys),
             "TIMESTAMP_ORIGINAL_KEY_SCHEMA")
    return {int(row["layer"]): row for row in keys[-5:]}


def timestamp_selection(panel: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict]]:
    """Map fixed512 IDs to their original fullB100 rows; never success-filter."""
    _require(panel["n"] == len(panel["records"]) == 512, "TIMESTAMP_BANK_DENOMINATOR")
    seen = set()
    result = {batch: [] for batch in TIMESTAMP_BATCHES}
    for bank_index, entry in enumerate(panel["records"]):
        case_id, batch = int(entry["case_id"]), int(entry["write_batch"])
        _require(case_id not in seen and batch in TIMESTAMP_BATCHES, "TIMESTAMP_BANK_DUPLICATE_OR_EPOCH")
        seen.add(case_id)
        block = rows[(batch - 1) * 100:batch * 100]
        positions = [i for i, row in enumerate(block) if int(row["case_id"]) == case_id]
        _require(len(block) == 100 and len(positions) == 1, "TIMESTAMP_ORIGINAL_BATCH_IDENTITY", entry)
        record = block[positions[0]]["requested_rewrite"]
        _require((record["subject"], record["relation_id"], record["target_new"]["str"]) ==
                 (entry["subject"], entry["relation"], entry["target"]), "TIMESTAMP_RAW_FACT_IDENTITY", case_id)
        _require(entry["timestamp_checkpoint"] == f"B{batch:03d}", "TIMESTAMP_CHECKPOINT_IDENTITY")
        result[batch].append({"bank_index": bank_index, "batch_position": positions[0], "case_id": case_id})
    _require(sum(map(len, result.values())) == 512 and all(result.values()), "TIMESTAMP_ALL_EPOCHS_REQUIRED")
    return result


def reconstruct_timestamp_bank(rt: Any, output: str | Path) -> dict:
    """Re-encode original B100 at own endpoint using identical native128 route.

    Returns banks[layer] of FP32[512,14336], ordered exactly as history512.json.
    Raw bank is permitted diagnostic K, not a resume checkpoint. Every fullB100
    history SHA must match before its512-subset may be used in subtraction.
    """
    import torch
    output = Path(output)
    panel_path = rt.design / "history512.json"
    panel = json.loads(panel_path.read_text())
    selection = timestamp_selection(panel, rt.rows)
    _require(rt.M is not None, "TIMESTAMP_REQUIRES_LOADED_ENTRY")
    original = rt.snapshot()
    signature, rng = rt.signature(), rng_fingerprint(original["rng"])
    banks = {layer: torch.empty(512, 14336, dtype=torch.float32) for layer in LAYERS}
    receipts = []
    first_error = None
    started = time.monotonic()
    try:
        for batch in TIMESTAMP_BATCHES:
            rt.set_state(batch)
            block = rt.rows[(batch - 1) * 100:batch * 100]
            old = rt.local_sidecar(batch, "native-observation.json")
            expected = history_append_key_rows(old)
            before = rt.signature()
            capture = rt.capture(block, contexts=True, full=False)
            _require(rt.signature() == before, "TIMESTAMP_CAPTURE_STATE_MUTATION", batch)
            rows = [{"layer": layer, "shape": list(capture["means"][layer].shape),
                     "actual_sha256": tensor_sha(capture["means"][layer]),
                     "original_append_sha256": expected[layer]["sha256"]} for layer in LAYERS]
            exact = all(row["actual_sha256"] == row["original_append_sha256"] and row["shape"] == [100, 14336] for row in rows)
            receipt = {"batch": batch, "status": "PASS" if exact else "FAIL",
                       "full_original_batch_requests": 100, "sequences": 600,
                       "native_microbatch": 128, "token_receipt": capture["token_receipt"],
                       "keys": rows, "source_history_pass": "native-observation.keys[5:10]",
                       "bank_selected_rows": selection[batch], "capture_seconds": capture["seconds"],
                       "numerical_envelope": "NOT_ESTABLISHED; exact original append SHA required"}
            receipts.append(receipt)
            save(output / f"timestamp-B{batch:03d}.json", receipt)
            if not exact:
                tensor_file(output / f"timestamp-B{batch:03d}-mismatch-keys.pt",
                            {"schema": "diagnostic-native-history-key-mismatch-v1", "keys": capture["means"],
                             "case_ids": [int(r["case_id"]) for r in block], "layers": list(LAYERS)})
                raise TechnicalFailure("TIMESTAMP_EXACT_PARITY_NOT_ESTABLISHED", receipt)
            for item in selection[batch]:
                for layer in LAYERS:
                    banks[layer][item["bank_index"]].copy_(capture["means"][layer][item["batch_position"]])
            del capture
    except Exception as exc:
        first_error = exc
        save(output / "timestamp-first-failure.json", {"status": "FAIL", "first_cause": str(exc),
             "exception": type(exc).__name__, "completed_checks": receipts,
             "subtract_permission": False, "wall_seconds": time.monotonic() - started})
    finally:
        try:
            rt.restore(original)
            _require(rt.signature() == signature and rng_fingerprint(rng_get()) == rng,
                     "TIMESTAMP_CALLER_STATE_RESTORE")
        except Exception as cleanup:
            save(output / "timestamp-restore-failure.json", {"status": "FAIL", "cleanup_cause": str(cleanup),
                 "first_cause": str(first_error) if first_error else None})
            if first_error is None:
                raise
    if first_error is not None:
        raise first_error
    case_ids = [int(r["case_id"]) for r in panel["records"]]
    artifact = tensor_file(output / "timestamp-K-H512.pt", {
        "schema": "diagnostic-native-timestamp-key-bank-v1", "keys": banks,
        "layers": list(LAYERS), "case_ids": case_ids, "panel_sha256": file_sha(panel_path),
        "statistics_weight": 1, "superseded_included": True, "is_resume_checkpoint": False})
    summary = {"status": "PASS", "records": 512, "layers": list(LAYERS),
               "case_order_sha256": digest(case_ids), "bank_tensor_sha256": {str(k): tensor_sha(v) for k, v in banks.items()},
               "artifact": artifact, "checks": receipts, "wall_seconds": time.monotonic() - started,
               "subtract_permission": True, "history_appends": 0, "target_optimizations": 0,
               "caller_state_restored": True}
    receipt = save(output / "timestamp-bank.json", summary)
    return {"bank": banks, "case_ids": case_ids, "receipt": receipt, "summary": summary}
