"""Outcome-independent sample and tokenizer binding for BGODE-R2 Stage B."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from project.run_scripts.barrier_guided_ode.s1_contract import (
    SealedS1Sample,
    load_sealed_s1_sample,
)
from project.run_scripts.ode_edit_motivation.contracts import canonical_json, sha256_bytes

from .errors import PrefixEventBoundary


@dataclass(frozen=True, slots=True)
class R2ModelBinding:
    alias: str
    revision: str
    run_id: str


MODEL_BINDINGS: Mapping[str, R2ModelBinding] = MappingProxyType(
    {
        "llama3-8b-inst": R2ModelBinding(
            alias="llama3-8b-inst",
            revision="8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
            run_id="s05-bgode-r2-stage-b-llama-request000-fp64-two-equality-v1",
        ),
        "qwen2.5-7b-inst": R2ModelBinding(
            alias="qwen2.5-7b-inst",
            revision="a09a35458c702b33eeacc393d103063234e8bc28",
            run_id="s05-bgode-r2-stage-b-qwen-request000-fp64-two-equality-v1",
        ),
    }
)


def model_binding(alias: str) -> R2ModelBinding:
    try:
        return MODEL_BINDINGS[alias]
    except KeyError as exc:
        raise PrefixEventBoundary(f"unsupported BGODE-R2 model alias: {alias}") from exc


@dataclass(frozen=True, slots=True)
class R2Tokenization:
    model_alias: str
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    source_token_ids: tuple[int, ...]
    tokenizer_name_or_path: str
    tokenizer_observed_commit: str
    tokenizer_expected_revision: str
    tokenizer_vocabulary_size: int
    identity: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-bgode-r2-termination-free-tokenization/v1",
            "model_alias": self.model_alias,
            "prompt_token_ids": list(self.prompt_token_ids),
            "target_token_ids": list(self.target_token_ids),
            "source_token_ids": list(self.source_token_ids),
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "tokenizer_observed_commit": self.tokenizer_observed_commit,
            "tokenizer_expected_revision": self.tokenizer_expected_revision,
            "tokenizer_vocabulary_size": self.tokenizer_vocabulary_size,
            "termination_token_count": 0,
            "identity": self.identity,
        }


def seal_tokenization(
    tokenizer: Any,
    sample: SealedS1Sample,
    *,
    model_alias: str,
) -> R2Tokenization:
    binding = model_binding(model_alias)
    prompt_text = sample.edit_request.prompt.format(sample.edit_request.subject)
    target_text = sample.edit_request.target_new
    source_text = sample.target_true
    target_text = target_text if target_text.startswith(" ") else f" {target_text}"
    source_text = source_text if source_text.startswith(" ") else f" {source_text}"
    prompt_ids = tuple(int(value) for value in tokenizer.encode(prompt_text, add_special_tokens=True))
    target_ids = tuple(int(value) for value in tokenizer.encode(target_text, add_special_tokens=False))
    source_ids = tuple(int(value) for value in tokenizer.encode(source_text, add_special_tokens=False))
    if not prompt_ids or not target_ids or not source_ids or target_ids == source_ids:
        raise PrefixEventBoundary("R2 prompt/source/target tokenization is empty or equal")
    for text, suffix in ((target_text, target_ids), (source_text, source_ids)):
        combined = tuple(int(value) for value in tokenizer.encode(prompt_text + text, add_special_tokens=True))
        if combined != prompt_ids + suffix:
            raise PrefixEventBoundary("prompt/completion token concatenation is not exact")
    vocabulary_size = int(len(tokenizer))
    if vocabulary_size <= 1 or any(
        token < 0 or token >= vocabulary_size
        for token in (*prompt_ids, *target_ids, *source_ids)
    ):
        raise PrefixEventBoundary("R2 tokenizer vocabulary binding failed")
    commit = str(
        getattr(tokenizer, "_commit_hash", "")
        or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash", "")
    )
    payload = {
        "model_alias": binding.alias,
        "prompt_token_ids": list(prompt_ids),
        "target_token_ids": list(target_ids),
        "source_token_ids": list(source_ids),
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_observed_commit": commit,
        "tokenizer_expected_revision": binding.revision,
        "tokenizer_vocabulary_size": vocabulary_size,
        "termination_token_count": 0,
    }
    return R2Tokenization(
        model_alias=binding.alias,
        prompt_text=prompt_text,
        prompt_token_ids=prompt_ids,
        target_token_ids=target_ids,
        source_token_ids=source_ids,
        tokenizer_name_or_path=payload["tokenizer_name_or_path"],
        tokenizer_observed_commit=commit,
        tokenizer_expected_revision=binding.revision,
        tokenizer_vocabulary_size=vocabulary_size,
        identity=sha256_bytes(canonical_json(payload).encode("utf-8")),
    )


@dataclass(frozen=True, slots=True)
class R2VocabularyBinding:
    tokenizer_vocabulary_size: int
    config_vocabulary_size: int
    output_head_vocabulary_size: int
    identity: str

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "ode-edit-bgode-r2-vocabulary-binding/v1",
            "tokenizer_vocabulary_size": self.tokenizer_vocabulary_size,
            "config_vocabulary_size": self.config_vocabulary_size,
            "output_head_vocabulary_size": self.output_head_vocabulary_size,
            "vocabularies_equal": self.tokenizer_vocabulary_size == self.output_head_vocabulary_size,
            "identity": self.identity,
        }


def seal_model_vocabulary(model: Any, tokenization: R2Tokenization) -> R2VocabularyBinding:
    config_size = getattr(getattr(model, "config", None), "vocab_size", None)
    output = model.get_output_embeddings()
    shape = getattr(getattr(output, "weight", None), "shape", ())
    if (
        isinstance(config_size, bool)
        or not isinstance(config_size, int)
        or config_size <= 1
        or len(shape) != 2
        or int(shape[0]) != config_size
    ):
        raise PrefixEventBoundary("model output/config vocabulary binding failed")
    output_size = int(shape[0])
    tokenizer_size = tokenization.tokenizer_vocabulary_size
    if tokenizer_size > output_size or any(
        token >= output_size
        for token in (*tokenization.prompt_token_ids, *tokenization.target_token_ids, *tokenization.source_token_ids)
    ):
        raise PrefixEventBoundary("sealed token lies outside output head")
    body = {
        "tokenizer_vocabulary_size": tokenizer_size,
        "config_vocabulary_size": config_size,
        "output_head_vocabulary_size": output_size,
    }
    return R2VocabularyBinding(
        tokenizer_vocabulary_size=tokenizer_size,
        config_vocabulary_size=config_size,
        output_head_vocabulary_size=output_size,
        identity=sha256_bytes(canonical_json(body).encode("utf-8")),
    )


__all__ = [
    "MODEL_BINDINGS",
    "R2ModelBinding",
    "R2Tokenization",
    "R2VocabularyBinding",
    "load_sealed_s1_sample",
    "model_binding",
    "seal_model_vocabulary",
    "seal_tokenization",
]
