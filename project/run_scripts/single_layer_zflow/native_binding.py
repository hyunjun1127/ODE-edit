"""Pinned BLUE request/token/key binding, without target fitting or writes.

This module ports the *input construction* from BLUE 311b076a's AlphaEdit
compute_z/compute_ks/repr_tools. It never invokes compute_z, a writer, a native
endpoint, or history finalization. Sequence contents are private runtime data;
only hashes/aggregate metadata belong in published reports.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PINNED_BLUE_COMMIT = "311b076a92e4ed0f14f5c8b4909732da781bc5f7"
NATIVE_SOURCE_SHA256 = {
    "AlphaEdit/compute_ks.py": "6c5b53d4a1fae02d7b303fe67acc724b31261e2a82b915de4c3862661023bdba",
    "AlphaEdit/compute_z.py": "a941a492d9e9e45f2aa7b88ab0137ae20910b423d7e59186b20c85bb285ff79f",
    "AlphaEdit/AlphaEdit_main.py": "79da927aad5ab817fd008c5958768adcd00556989a8efbaa2c4bdc80d8fc842e",
    "AlphaEdit/AlphaEdit_hparams.py": "bffe280e509c1d699d39a6ea6457ec521954278e574006f9f46ef766d77c990e",
    "rome/repr_tools.py": "b3f7f07358437e7927d556b64666bf0e67d03eb6489c00613a8865e83372849f",
}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def verify_native_source(source_root: str | Path) -> dict[str, Any]:
    """Read/hash the exact source ports; no imports or source execution."""
    root = Path(source_root).resolve(strict=True)
    members = []
    for relative, expected in NATIVE_SOURCE_SHA256.items():
        path = root / relative
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise ValueError(f"NATIVE_SOURCE_MISMATCH: {relative}: {actual}")
        members.append({"path": str(path), "relative": relative,
                        "bytes": len(data), "sha256": actual})
    return {"blue_commit": PINNED_BLUE_COMMIT, "members": members,
            "implementation": "source-pinned construction port; native source not imported",
            "optimizer_or_endpoint_called": False}


def normalize_requests(requests: Sequence[Mapping[str, Any]]) -> tuple[dict, ...]:
    """Native leading-space normalization, preserving every caller-owned byte."""
    result = deepcopy(list(requests))
    for request in result:
        for name in ("prompt", "subject", "target_new", "case_id"):
            if name not in request:
                raise ValueError(f"request missing {name}")
        target = request["target_new"]["str"]
        if not isinstance(target, str) or not target:
            raise ValueError("target_new.str must be a nonempty string")
        if not target.startswith(" "):
            request["target_new"]["str"] = " " + target
        if request["prompt"].count("{}") != 1:
            raise ValueError("native prompt must contain exactly one subject placeholder")
    ids = [str(r["case_id"]) for r in result]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case_id in logical batch")
    return tuple(result)


def _input_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    ids = tokenizer(text)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("unexpected batched scalar tokenizer result")
        ids = ids[0]
    return tuple(int(i) for i in ids)


def native_subject_last_index(tokenizer: Any, template: str, subject: str) -> int:
    """Exact repr_tools subject_last: len(encode(prefix + subject)) - 1."""
    if template.count("{}") != 1:
        raise ValueError("native lookup requires exactly one placeholder")
    prefix, _ = template.split("{}")
    return len(tokenizer.encode(prefix + subject)) - 1


@dataclass(frozen=True)
class BoundSequence:
    ids: tuple[int, ...]
    request_index: int
    case_id: Any
    kind: str
    context_index: int
    lookup_index: int
    edit_positions: tuple[int, ...] = ()
    edit_labels: tuple[int, ...] = ()
    edit_weights: tuple[float, ...] = ()
    kl_pos: int | None = None
    kl_weight: float = 0.0
    canonical_key_is_token_prefix: bool | None = None

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.ids


@dataclass(frozen=True)
class BindingBatch:
    requests: tuple[dict, ...]
    sequences: tuple[BoundSequence, ...]
    key_sequences: tuple[BoundSequence, ...]
    group_sizes: tuple[int, ...]
    metadata: dict[str, Any]

    @property
    def m(self) -> int:
        return len(self.requests)


def build_training_sequences(tokenizer: Any, requests: Sequence[Mapping[str, Any]],
                             contexts: Sequence[Sequence[str]]) -> BindingBatch:
    """Build six native edit sequences and one entry-essence sequence/request.

    A target is tokenized with native special-token defaults; its first BOS/UNK
    is removed once. Native text decode(target_ids[:-1]) is appended *before*
    subject formatting, then retokenized. Prediction positions are exactly
    [sequence_length - target_length, sequence_length), not a second shifted
    range. Canonical keys are separately tokenized and never cache-shared on an
    assumed prefix relation.
    """
    if getattr(tokenizer, "padding_side", None) != "right":
        raise ValueError("NATIVE_PADDING_MISMATCH: pinned N4 uses right padding")
    groups = tuple(tuple(group) for group in contexts)
    if tuple(map(len, groups)) != (1, 5) or groups[0] != ("{}",):
        raise ValueError("NATIVE_CONTEXT_MISMATCH: expected clean1/generated5")
    flat = tuple(context for group in groups for context in group)
    if any(context.count("{}") != 1 for context in flat):
        raise ValueError("each native context requires one request placeholder")
    normalized = normalize_requests(requests)
    sequences, keys = [], []
    m, c = len(normalized), len(flat)
    for ri, request in enumerate(normalized):
        target = _input_ids(tokenizer, request["target_new"]["str"])
        if target and target[0] in (getattr(tokenizer, "bos_token_id", None),
                                    getattr(tokenizer, "unk_token_id", None)):
            target = target[1:]
        if not target:
            raise ValueError("NATIVE_TARGET_EMPTY after BOS/UNK removal")
        decoded_prefix = tokenizer.decode(list(target[:-1]))
        for ci, context in enumerate(flat):
            key_template = context.format(request["prompt"])
            key_ids = _input_ids(tokenizer, key_template.format(request["subject"]))
            key_lookup = native_subject_last_index(tokenizer, key_template, request["subject"])
            template = key_template + decoded_prefix
            ids = _input_ids(tokenizer, template.format(request["subject"]))
            lookup = native_subject_last_index(tokenizer, template, request["subject"])
            start = len(ids) - len(target)
            if start < 0 or not (0 <= lookup < len(ids)) or not (0 <= key_lookup < len(key_ids)):
                raise ValueError("NATIVE_TOKEN_ALIGNMENT_INVALID")
            keys.append(BoundSequence(key_ids, ri, request["case_id"], "key", ci, key_lookup))
            sequences.append(BoundSequence(
                ids, ri, request["case_id"], "edit", ci, lookup,
                tuple(range(start, len(ids))), target,
                (1.0 / (m * c * len(target)),) * len(target),
                canonical_key_is_token_prefix=ids[:len(key_ids)] == key_ids))
        template = "{} is a"
        ids = _input_ids(tokenizer, template.format(request["subject"]))
        lookup = native_subject_last_index(tokenizer, template, request["subject"])
        if not (0 <= lookup < len(ids)):
            raise ValueError("NATIVE_ESSENCE_LOOKUP_INVALID")
        sequences.append(BoundSequence(ids, ri, request["case_id"], "essence", -1,
                                       lookup, kl_pos=lookup, kl_weight=1.0 / m))
    metadata = {
        "m": m, "q": m, "incidence": "E=I_m; one mean-within/mean-across key per request",
        "request_order": [r["case_id"] for r in normalized],
        "request_digest": _digest(normalized), "contexts_digest": _digest(groups),
        "group_sizes": [1, 5], "key_context_weights": [.5, .1, .1, .1, .1, .1],
        "edit_context_weights": [1 / 6] * 6,
        "edit_token_weight": "1/(m * c_i * T_i)", "essence_request_weight": "1/m",
        "essence_template": "{} is a", "fact_token": "subject_last",
        "padding_side": "right", "position_ids": "arange(L), broadcast to every row",
        "label_positions": "native ex_len - target_len : ex_len; already prediction-shifted",
        "native_target_construction": "strip initial BOS/UNK once; decode(ids[:-1]); append; format; retokenize",
        "key_training_prefix_equal": sum(s.canonical_key_is_token_prefix is True for s in sequences),
        "key_training_prefix_not_equal": sum(s.canonical_key_is_token_prefix is False for s in sequences),
        "key_and_training_cache_shared": False,
        "sequences_digest": _digest([s.__dict__ for s in sequences]),
        "keys_digest": _digest([s.__dict__ for s in keys]),
        "native_target_optimizer_calls": 0,
    }
    return BindingBatch(normalized, tuple(sequences), tuple(keys), (1, 5), metadata)


def pack_sequences(rows: Sequence[BoundSequence], pad_token_id: int,
                   *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Right padding only; weights retain full-logical-batch normalization."""
    if not rows:
        raise ValueError("cannot pack an empty physical microbatch")
    width = max(len(row.ids) for row in rows)
    ids = torch.full((len(rows), width), int(pad_token_id), dtype=torch.long)
    mask = torch.zeros_like(ids)
    er, ec, el, ew, kr, kc, kw = [], [], [], [], [], [], []
    for i, row in enumerate(rows):
        if len(row.edit_positions) != len(row.edit_labels) or len(row.edit_labels) != len(row.edit_weights):
            raise ValueError("edit prediction metadata lengths disagree")
        ids[i, :len(row.ids)] = torch.tensor(row.ids, dtype=torch.long)
        mask[i, :len(row.ids)] = 1
        if any(p < 0 or p >= len(row.ids) for p in row.edit_positions):
            raise ValueError("edit prediction position outside valid tokens")
        er.extend([i] * len(row.edit_positions)); ec.extend(row.edit_positions)
        el.extend(row.edit_labels); ew.extend(row.edit_weights)
        if row.kl_pos is not None:
            if row.kl_pos < 0 or row.kl_pos >= len(row.ids) or row.kl_weight <= 0:
                raise ValueError("invalid essence prediction metadata")
            kr.append(i); kc.append(row.kl_pos); kw.append(row.kl_weight)
    packed = {"input_ids": ids, "attention_mask": mask,
              "position_ids": torch.arange(width, dtype=torch.long).unsqueeze(0).expand(len(rows), -1),
              "edit_rows": torch.tensor(er, dtype=torch.long),
              "edit_cols": torch.tensor(ec, dtype=torch.long),
              "edit_labels": torch.tensor(el, dtype=torch.long),
              "edit_weights": torch.tensor(ew, dtype=torch.float64),
              "kl_rows": torch.tensor(kr, dtype=torch.long),
              "kl_cols": torch.tensor(kc, dtype=torch.long),
              "kl_weights": torch.tensor(kw, dtype=torch.float64)}
    return {key: value.to(device) for key, value in packed.items()}


def aggregate_native_keys(context_keys: torch.Tensor,
                          group_sizes: Sequence[int] = (1, 5)) -> torch.Tensor:
    """Native FP32 mean-within/mean-across, returning K[din, m]."""
    if context_keys.ndim != 2 or not group_sizes or any(n <= 0 for n in group_sizes):
        raise ValueError("invalid native context key layout")
    n = sum(group_sizes)
    if context_keys.shape[0] % n:
        raise ValueError("request/context incidence is not divisible")
    if not torch.isfinite(context_keys).all():
        raise ValueError("nonfinite native keys")
    result = []
    for offset in range(0, context_keys.shape[0], n):
        start, group_means = offset, []
        for count in group_sizes:
            group_means.append(context_keys[start:start + count].mean(0))
            start += count
        result.append(torch.stack(group_means, 0).mean(0))
    if not result:
        return context_keys.new_empty((context_keys.shape[1], 0))
    return torch.stack(result, 0).T.contiguous()


def request_incidence(batch: BindingBatch, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if len(batch.key_sequences) != batch.m * sum(batch.group_sizes):
        raise ValueError("native request/context column incidence mismatch")
    for ri in range(batch.m):
        rows = batch.key_sequences[ri * 6:(ri + 1) * 6]
        if any(row.request_index != ri or row.context_index != ci for ci, row in enumerate(rows)):
            raise ValueError("native request/context column order mismatch")
    return torch.eye(batch.m, dtype=dtype)


def select_l4_projector(stack: torch.Tensor,
                        physical_inventory: Sequence[int] = (4, 5, 6, 7, 8)) -> tuple[torch.Tensor, dict]:
    """Explicit physical layer4 -> source stack0 -> singleton local0 binding."""
    if tuple(physical_inventory) != (4, 5, 6, 7, 8):
        raise ValueError("pinned projector physical inventory mismatch")
    if stack.ndim != 3 or stack.shape[0] != 5 or stack.shape[1] != stack.shape[2]:
        raise ValueError("pinned projector stack shape mismatch")
    return stack[0], {"physical_layer": 4, "source_stack_index": 0, "local_index": 0,
                      "source_inventory": list(physical_inventory), "projection_modified": False}


def validate_native_config(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"layers": [4], "fact_token": "subject_last", "L2": 1,
                "blue": True, "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
                "layer_module_tmp": "model.layers.{}"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"NATIVE_CONFIG_MISMATCH: {key}")
    return {"config_semantic_sha256": _digest(config), "checked": expected,
            "native_target_optimizer_called": False}


def capture_native_keys(model: Any, batch: BindingBatch, *, pad_token_id: int,
                        microbatch: int = 1) -> tuple[torch.Tensor, dict[str, Any]]:
    """Read-only full-sequence L4 down-projection input capture.

    The Llama base-model forward (including every transformer layer, excluding
    only the unused LM head) supplies the same hook value as native compute_ks.
    Changing native's physical microbatch128 is explicit telemetry, not a claim
    of cross-batching bitwise parity. Actual-model native parity is a separate
    technical gate owned by the runner.
    """
    if microbatch < 1 or model.training:
        raise ValueError("positive microbatch and model.eval() are required")
    params = tuple(model.parameters())
    if any(p.requires_grad for p in params):
        raise ValueError("key capture requires already-frozen model parameters")
    before = tuple((p.data_ptr(), p._version) for p in params)
    device = model.model.embed_tokens.weight.device
    module = model.model.layers[4].mlp.down_proj
    if module.weight.dtype != torch.float32:
        raise ValueError("NATIVE_PRECISION_MISMATCH: actual L4 write must be FP32")
    captured, work_tokens, padded_tokens = [], 0, 0
    hook_value = []

    def hook(_module, inputs):
        if len(inputs) != 1 or inputs[0].ndim != 3:
            raise ValueError("Llama down_proj input layout mismatch")
        hook_value.append(inputs[0].detach())

    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, len(batch.key_sequences), microbatch):
                rows = batch.key_sequences[start:start + microbatch]
                packed = pack_sequences(rows, pad_token_id, device=device)
                hook_value.clear()
                model.model(input_ids=packed["input_ids"], attention_mask=packed["attention_mask"],
                            position_ids=packed["position_ids"], use_cache=False, return_dict=True)
                if len(hook_value) != 1 or hook_value[0].shape[:2] != packed["input_ids"].shape:
                    raise ValueError("Llama down_proj hook count/shape mismatch")
                values = hook_value.pop()
                # Source selects a singleton list then mean(0), retaining its
                # exact reduction orientation instead of flattened averaging.
                for i, row in enumerate(rows):
                    captured.append(values[i, [row.lookup_index]].mean(0).detach())
                work_tokens += int(packed["attention_mask"].sum())
                padded_tokens += packed["input_ids"].numel()
    finally:
        handle.remove()
    if tuple((p.data_ptr(), p._version) for p in params) != before:
        raise RuntimeError("KEY_CAPTURE_PARAMETER_MUTATION")
    if any(p.grad is not None for p in params):
        raise RuntimeError("KEY_CAPTURE_DENSE_PARAMETER_GRAD")
    context_keys = torch.stack(captured) if captured else torch.empty((0, module.in_features), dtype=module.weight.dtype)
    # Native computes both levels of context means on the model device. Move
    # only the completed request keys to CPU, not the pre-reduction vectors.
    keys = aggregate_native_keys(context_keys, batch.group_sizes).cpu()
    incidence = request_incidence(batch)
    if keys.shape[1] != incidence.shape[0]:
        raise ValueError("KEY_CAPTURE_REQUEST_ORDER_MISMATCH")
    return keys, {"m": batch.m, "q": keys.shape[1], "key_shape": list(keys.shape),
                  "key_dtype": str(keys.dtype), "group_sizes": list(batch.group_sizes),
                  "valid_tokens": work_tokens, "padded_tokens": padded_tokens,
                  "physical_microbatch": microbatch, "native_reference_microbatch": 128,
                  "unused_lm_head_skipped": True, "full_sequence_transformer_forward": True,
                  "parameter_pointer_version_unchanged": True, "history_append": 0,
                  "native_z_calls": 0, "native_endpoint_calls": 0,
                  "cross_microbatch_bitwise_parity": "NOT_CLAIMED"}
