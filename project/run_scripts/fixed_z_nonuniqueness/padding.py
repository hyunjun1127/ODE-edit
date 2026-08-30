"""Semantic-position-safe tokenization for left-padded causal-LM batches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .contracts import TechnicalBoundary


def semantic_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise TechnicalBoundary("attention_mask must be rank two")
    pos = attention_mask.to(torch.long).cumsum(dim=-1) - 1
    return pos.masked_fill(attention_mask == 0, 0)


def semantic_token_columns(attention_mask: torch.Tensor) -> list[torch.Tensor]:
    """Return real-token raw columns; callers never infer from padded width."""
    if attention_mask.ndim != 2:
        raise TechnicalBoundary("attention_mask must be rank two")
    return [torch.nonzero(row, as_tuple=False).flatten() for row in attention_mask]


def semantic_last_columns(attention_mask: torch.Tensor) -> torch.Tensor:
    cols = semantic_token_columns(attention_mask)
    if any(item.numel() == 0 for item in cols):
        raise TechnicalBoundary("empty semantic sequence")
    return torch.stack([item[-1] for item in cols])


def target_continuation_columns(
    attention_mask: torch.Tensor, target_lengths: Iterable[int]
) -> list[torch.Tensor]:
    answer: list[torch.Tensor] = []
    for real, target_len in zip(semantic_token_columns(attention_mask), target_lengths, strict=True):
        count = int(target_len)
        if count <= 0 or count > real.numel():
            raise TechnicalBoundary("invalid target continuation length")
        answer.append(real[-count:])
    return answer


def gather_batch_positions(hidden: torch.Tensor, columns: list[torch.Tensor]) -> list[torch.Tensor]:
    if hidden.ndim != 3 or hidden.shape[0] != len(columns):
        raise TechnicalBoundary("hidden/column batch mismatch")
    return [hidden[row, col] for row, col in enumerate(columns)]


def build_position_safe_batch(tokenizer: Any, texts: list[str], *, padding_side: str) -> Any:
    if padding_side not in {"left", "right"}:
        raise TechnicalBoundary("padding_side must be explicit")
    previous = tokenizer.padding_side
    try:
        tokenizer.padding_side = padding_side
        batch = tokenizer(texts, padding=True, return_tensors="pt")
    finally:
        tokenizer.padding_side = previous
    batch["position_ids"] = semantic_position_ids(batch["attention_mask"])
    return batch


def bind_padding(tokenizer: Any, model: Any, *, padding_side: str) -> dict[str, Any]:
    """Bind pad=eos exactly and explicitly for both model configs."""
    if tokenizer.eos_token_id is None:
        raise TechnicalBoundary("pinned tokenizer has no eos token")
    tokenizer.padding_side = padding_side
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    generation = getattr(model, "generation_config", None)
    if generation is not None:
        generation.pad_token_id = tokenizer.pad_token_id
        generation.eos_token_id = tokenizer.eos_token_id
    return {
        "padding_side": tokenizer.padding_side,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "model_pad_token_id": int(model.config.pad_token_id),
        "generation_pad_token_id": None if generation is None else int(generation.pad_token_id),
        "generation_eos_token_id": None if generation is None else int(generation.eos_token_id),
    }


def batch_identity(batch: Any) -> str:
    h = hashlib.sha256()
    for key in ("input_ids", "attention_mask", "position_ids"):
        tensor = batch[key].detach().to("cpu").contiguous()
        h.update(key.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def semantic_batch_identity(batch: Any) -> str:
    """Hash semantic tokens and positions independently of pad columns."""

    h = hashlib.sha256()
    positions = semantic_position_ids(batch["attention_mask"])
    for row, cols in enumerate(semantic_token_columns(batch["attention_mask"])):
        for key, tensor in (("input_ids", batch["input_ids"]), ("position_ids", positions)):
            value = tensor[row, cols].detach().to("cpu").contiguous()
            h.update(key.encode())
            h.update(value.numpy().tobytes())
    return h.hexdigest()


@dataclass
class OfficialTokenizerHook:
    """Thin tokenizer proxy that records Official inputs without source edits.

    Stock EasyEdit is kept on its pinned right-padding convention.  Candidate
    and validation batches use :func:`build_position_safe_batch` with left
    padding and explicit position IDs.
    """

    tokenizer: Any
    call_identities: list[dict[str, Any]]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self.tokenizer(*args, **kwargs)
        if kwargs.get("return_tensors") == "pt" and "attention_mask" in output:
            official_semantic = semantic_batch_identity(output)
            previous = self.tokenizer.padding_side
            try:
                self.tokenizer.padding_side = "left"
                replay = self.tokenizer(*args, **kwargs)
            finally:
                self.tokenizer.padding_side = previous
            replay_semantic = semantic_batch_identity(replay)
            semantic_equal = official_semantic == replay_semantic
            if not semantic_equal:
                raise TechnicalBoundary("Official/hook semantic token-position identity failed")
            row = {
                "padding_side": self.tokenizer.padding_side,
                "input_ids_shape": list(output["input_ids"].shape),
                "attention_mask_shape": list(output["attention_mask"].shape),
                "semantic_lengths": [int(v) for v in output["attention_mask"].sum(-1)],
                "official_raw_batch_sha256": batch_identity({
                    "input_ids": output["input_ids"],
                    "attention_mask": output["attention_mask"],
                    "position_ids": semantic_position_ids(output["attention_mask"]),
                }),
                "left_replay_raw_batch_sha256": batch_identity({
                    "input_ids": replay["input_ids"],
                    "attention_mask": replay["attention_mask"],
                    "position_ids": semantic_position_ids(replay["attention_mask"]),
                }),
                "official_semantic_sha256": official_semantic,
                "left_replay_semantic_sha256": replay_semantic,
                "semantic_input_attention_position_equal": semantic_equal,
            }
            self.call_identities.append(row)
        return output
