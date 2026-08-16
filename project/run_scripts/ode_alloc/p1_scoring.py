"""Shared teacher-forced scoring with exact suffix and padding contracts."""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch

from .contracts import ODEAllocContractError, canonical_hash, canonical_json


@dataclass(frozen=True, slots=True)
class TeacherForcedSpec:
    item_id: str
    prompt: str
    target: str

    def __post_init__(self) -> None:
        if not self.item_id or not self.prompt or not self.target:
            raise ODEAllocContractError("teacher-forced spec is empty")


@dataclass(frozen=True, slots=True)
class TeacherForcedResult:
    logp: Mapping[str, torch.Tensor]
    model_forward_calls: int
    processed_tokens: int
    panel_id: str | None


def _as_ids(value: Any, *, label: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu").reshape(-1).tolist()
    if not isinstance(value, list) or not value or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ODEAllocContractError(f"{label} tokenization differs")
    return value


class TeacherForcedScorer:
    """Right-padded batched causal scoring; never calls model.generate."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, *, microbatch_size: int) -> None:
        if (
            isinstance(microbatch_size, bool)
            or not isinstance(microbatch_size, int)
            or microbatch_size <= 0
        ):
            raise ODEAllocContractError("teacher-forced microbatch must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.microbatch_size = microbatch_size
        self.device = next(model.parameters()).device
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
            raise ODEAllocContractError("teacher-forced pad token differs")
        self.pad_token_id = pad

    def _encode(self, spec: TeacherForcedSpec) -> tuple[list[int], list[int], list[int]]:
        prompt_ids = _as_ids(
            self.tokenizer.encode(spec.prompt, add_special_tokens=True),
            label="prompt",
        )
        locked_target = spec.target if spec.target.startswith(" ") else " " + spec.target
        target_ids = _as_ids(
            self.tokenizer.encode(locked_target, add_special_tokens=False),
            label="target",
        )
        while target_ids and target_ids[0] in {
            self.tokenizer.bos_token_id,
            self.tokenizer.unk_token_id,
        }:
            target_ids = target_ids[1:]
        if not target_ids:
            raise ODEAllocContractError("normalized target suffix is empty")
        combined = _as_ids(
            self.tokenizer.encode(
                spec.prompt + locked_target,
                add_special_tokens=True,
            ),
            label="combined prompt/target",
        )
        if (
            len(combined) != len(prompt_ids) + len(target_ids)
            or combined[: len(prompt_ids)] != prompt_ids
            or combined[-len(target_ids) :] != target_ids
        ):
            raise ODEAllocContractError("normalized target is not the exact combined suffix")
        return prompt_ids, target_ids, combined

    def score(
        self,
        specs: Iterable[TeacherForcedSpec],
        *,
        gradient: bool,
    ) -> TeacherForcedResult:
        locked = tuple(specs)
        if not locked or len({item.item_id for item in locked}) != len(locked):
            raise ODEAllocContractError("teacher-forced spec identities differ")
        encoded = tuple((spec, *self._encode(spec)) for spec in locked)
        outputs: dict[str, torch.Tensor] = {}
        digest = hashlib.sha256()
        forward_calls = 0
        processed_tokens = 0
        context = contextlib.nullcontext() if gradient else torch.no_grad()
        with context:
            for start in range(0, len(encoded), self.microbatch_size):
                batch = encoded[start : start + self.microbatch_size]
                maximum = max(len(item[3]) for item in batch)
                input_ids = torch.full(
                    (len(batch), maximum),
                    self.pad_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
                attention = torch.zeros_like(input_ids)
                for row, (_, _, _, combined) in enumerate(batch):
                    values = torch.tensor(combined, dtype=torch.long, device=self.device)
                    input_ids[row, : len(combined)].copy_(values)
                    attention[row, : len(combined)] = 1
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    use_cache=False,
                ).logits
                forward_calls += 1
                processed_tokens += int(attention.detach().sum().item())
                for row, (spec, prompt_ids, target_ids, _) in enumerate(batch):
                    begin = len(prompt_ids) - 1
                    selected = logits[row, begin : begin + len(target_ids), :]
                    if selected.shape[0] != len(target_ids):
                        raise ODEAllocContractError("teacher-forced target window differs")
                    target = torch.tensor(target_ids, dtype=torch.long, device=self.device)
                    token_logp = torch.log_softmax(selected.float(), dim=-1).gather(
                        1, target.unsqueeze(1)
                    )
                    outputs[spec.item_id] = token_logp.mean()
                    if not gradient:
                        logical = selected.detach().contiguous().to(device="cpu").reshape(-1)
                        digest.update(spec.item_id.encode("ascii"))
                        digest.update(str(selected.dtype).encode("ascii"))
                        digest.update(canonical_json(tuple(selected.shape)).encode("ascii"))
                        digest.update(logical.view(torch.uint8).numpy().tobytes(order="C"))
                    del target, token_logp, selected
                del logits, input_ids, attention
        panel_id = None
        if not gradient:
            panel_id = canonical_hash(
                {
                    "schema": "ode-alloc-teacher-forced-panel/v1",
                    "spec_ids": tuple(item.item_id for item in locked),
                    "selected_logits_sha256": digest.hexdigest(),
                    "logp": {
                        key: float(value.detach().to(device="cpu"))
                        for key, value in sorted(outputs.items())
                    },
                }
            )
        return TeacherForcedResult(
            logp=outputs,
            model_forward_calls=forward_calls,
            processed_tokens=processed_tokens,
            panel_id=panel_id,
        )
