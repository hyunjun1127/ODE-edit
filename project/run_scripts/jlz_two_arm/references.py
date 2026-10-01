"""Fixed JLZ general/replay membership and loss units (no model or I/O side effects).

Selection matches the sealed ``inspect_design.py`` including zero-based batch
indices, compact UTF-8 rank payloads and the order of the four exclusion filters.
Historical occurrence multiplicity belongs to H, not to this replay registry.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path


DEFAULT_CONFIG = {
    "seed": 20261002,
    "general_count": 16,
    "general_relation_matched_quota": 8,
    "replay_max_count": 16,
}


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def normalize_text(text):
    return unicodedata.normalize("NFC", " ".join(text.split()))


norm = normalize_text


def claim(record):
    req = record["requested_rewrite"]
    return norm(req["subject"]), req["relation_id"]


def canonical_prompt(record):
    req = record["requested_rewrite"]
    return norm(req["prompt"].format(req["subject"]))


prompt = canonical_prompt


def _unique_ids(records, label):
    ids = [r["case_id"] for r in records]
    if any(not isinstance(case_id, int) or isinstance(case_id, bool) for case_id in ids):
        raise ValueError(f"{label}: case IDs must be integers")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate case ID")


def verified_records(path, expected_sha256):
    """Load a small JSON dataset only after exact byte identity is established."""
    raw = Path(path).read_bytes()
    actual = digest(raw)
    if actual != expected_sha256:
        raise ValueError(f"DATASET_SHA_MISMATCH: {path}")
    records = json.loads(raw)
    if not isinstance(records, list):
        raise ValueError("DATASET_SCHEMA: expected record list")
    _unique_ids(records, str(path))
    return records, {"path": str(path), "bytes": len(raw), "sha256": actual,
                     "records": len(records)}


def filter_general_pool(full, stream):
    """Exclude all fixed10k IDs/claims/evaluation prompts/subjects in sealed order."""
    _unique_ids(full, "full")
    _unique_ids(stream, "stream")
    ids = {r["case_id"] for r in stream}
    claims = {claim(r) for r in stream}
    subjects = {c[0] for c in claims}
    eval_prompts = {
        p for r in stream for p in (
            prompt(r), *map(norm, r["paraphrase_prompts"]),
            *map(norm, r["neighborhood_prompts"]))
    }
    pool = [r for r in full if r["case_id"] not in ids]
    counts = {"outside_fixed_ids": len(pool)}
    pool = [r for r in pool if claim(r) not in claims]
    counts["after_claim_exclusion"] = len(pool)
    pool = [r for r in pool if prompt(r) not in eval_prompts]
    counts["after_eval_prompt_exclusion"] = len(pool)
    pool = [r for r in pool if claim(r)[0] not in subjects]
    counts["after_subject_exclusion"] = len(pool)
    counts["relations"] = len({r["requested_rewrite"]["relation_id"] for r in pool})
    return pool, counts


def _config(cfg):
    value = dict(DEFAULT_CONFIG)
    if cfg is not None:
        value.update(cfg)
    # These are fixed experiment settings, not configurable selection sweeps.
    for key, expected in DEFAULT_CONFIG.items():
        if value[key] != expected:
            raise ValueError(f"REFERENCE_CONFIG_CHANGED: {key}")
    return value


def rank(records, seed, batch, group):
    return sorted(records, key=lambda r: (
        digest(compact([seed, batch, group, r["case_id"]])), r["case_id"]))


def select_general(pool, current, cfg=None, batch=0):
    cfg = _config(cfg)
    relations = {r["requested_rewrite"]["relation_id"] for r in current}
    matched = [r for r in pool if r["requested_rewrite"]["relation_id"] in relations]
    chosen = rank(matched, cfg["seed"], batch, "general_relation")[:cfg["general_relation_matched_quota"]]
    used = {r["case_id"] for r in chosen}
    rest = [r for r in pool if r["case_id"] not in used]
    chosen += rank(rest, cfg["seed"], batch, "general_global")[:cfg["general_count"] - len(chosen)]
    if len(chosen) != cfg["general_count"] or len({r["case_id"] for r in chosen}) != len(chosen):
        raise ValueError("GENERAL_REFERENCE_CARDINALITY")
    return chosen


def select_replay(past, current, cfg=None, batch=0):
    """Latest *committed* occurrence per claim; current claims are excluded.

    Input ``past`` must be the actual committed ledger in arrival order. There is
    no quality/success filter. Repeated targets still update latest occurrence.
    """
    cfg = _config(cfg)
    latest = {}
    for pos, record in enumerate(past):
        latest[claim(record)] = (pos, record)
    exclude = {claim(r) for r in current}
    ordered = [r for _, r in sorted(latest.values(), key=lambda pair: pair[0])
               if claim(r) not in exclude]
    chosen = []
    for q in range(4):
        part = ordered[len(ordered) * q // 4:len(ordered) * (q + 1) // 4]
        chosen += rank(part, cfg["seed"], batch, f"replay_age_{q}")[:4]
    used = {r["case_id"] for r in chosen}
    rest = [r for r in ordered if r["case_id"] not in used]
    chosen += rank(rest, cfg["seed"], batch, "replay_fill")[:cfg["replay_max_count"] - len(chosen)]
    if len(chosen) > cfg["replay_max_count"] or len({claim(r) for r in chosen}) != len(chosen):
        raise ValueError("REPLAY_REFERENCE_IDENTITY")
    return chosen


def membership_receipt(current, general, replay, *, batch_index, batch_size):
    row = {"batch_index": batch_index, "batch_size": batch_size}
    for name, records in (("current", current), ("general", general), ("replay", replay)):
        ids = [r["case_id"] for r in records]
        row[f"{name}_ids"] = ids
        row[f"{name}_count"] = len(ids)
        row[f"{name}_ids_sha256"] = digest(compact(ids))
    return row


def build_reference_schedule(pool, stream, cfg=None):
    """Lock independent BS4x2 and fresh BS100x20 memberships before outcomes.

    Runtime rechecks each row against its own committed prefix; this planned
    schedule never represents a future request as a committed replay reference.
    """
    cfg = _config(cfg)
    if len(stream) < 2000:
        raise ValueError("STREAM_SHORTER_THAN_MAIN2000")
    _unique_ids(stream, "stream")
    output = {}
    for stage, size, batches in (("pilot", 4, 2), ("main", 100, 20)):
        rows = []
        for b in range(batches):
            past = stream[:size * b]
            current = stream[size * b:size * (b + 1)]
            general = select_general(pool, current, cfg, b)
            replay = select_replay(past, current, cfg, b)
            rows.append(membership_receipt(current, general, replay,
                                          batch_index=b, batch_size=size))
        output[stage] = rows
    output["policy"] = "fixed-membership; actual own committed prefix required"
    output["semantic_disjointness"] = "NOT_CERTIFIED"
    return output


def teacher_identity(*, model_sha256, tokenizer_sha256, input_ids, target_ids,
                     prediction_positions, attention_mask, teacher_sha256,
                     model_epoch="W0", kl_direction="current||W0"):
    """Content-key a complete full-target TF teacher; no pointer/version key."""
    values = {"model_sha256": model_sha256, "tokenizer_sha256": tokenizer_sha256,
              "input_ids": list(input_ids), "target_ids": list(target_ids),
              "prediction_positions": list(prediction_positions),
              "attention_mask": list(attention_mask), "teacher_sha256": teacher_sha256,
              "model_epoch": model_epoch, "kl_direction": kl_direction}
    if model_epoch != "W0" or kl_direction != "current||W0":
        raise ValueError("GENERAL_TEACHER_IDENTITY")
    if not values["target_ids"] or len(values["target_ids"]) != len(values["prediction_positions"]):
        raise ValueError("TEACHER_TARGET_POSITIONS")
    if len(values["input_ids"]) != len(values["attention_mask"]):
        raise ValueError("TEACHER_MASK_LENGTH")
    positions = values["prediction_positions"]
    if len(set(positions)) != len(positions) or any(p < 0 or p >= len(input_ids) for p in positions):
        raise ValueError("TEACHER_POSITION_RANGE")
    if any(values["attention_mask"][p] != 1 for p in positions):
        raise ValueError("TEACHER_PAD_POSITION")
    return {**values, "identity_sha256": digest(compact(values))}


def prompt_token_means(per_prompt_token_losses):
    """Token mean per prompt, never a token-weighted average across prompts."""
    means = []
    for losses in per_prompt_token_losses:
        if len(losses) == 0:
            raise ValueError("REFERENCE_EMPTY_TARGET")
        means.append(sum(losses) / len(losses))
    return means


def group_mean_reference_loss(general_prompt_means, replay_prompt_means, batch_size,
                              beta_general=0.0625, beta_replay=1.0):
    """B times independent prompt-group means, preserving scalar/tensor gradients."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("REFERENCE_BATCH_SIZE")
    if beta_general != 0.0625 or beta_replay != 1.0:
        raise ValueError("REFERENCE_COEFFICIENT_CHANGED")
    if len(general_prompt_means) == 0:
        raise ValueError("EMPTY_GENERAL_REFERENCE")
    general = sum(general_prompt_means) / len(general_prompt_means)
    replay = sum(replay_prompt_means) / len(replay_prompt_means) if len(replay_prompt_means) else general * 0
    return batch_size * (beta_general * general + beta_replay * replay)
