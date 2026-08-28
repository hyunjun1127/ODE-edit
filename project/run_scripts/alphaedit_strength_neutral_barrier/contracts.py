"""Shared tokenizer and target-alignment contracts for barrier experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, List, Mapping, Sequence


class TokenizerContractBoundary(RuntimeError):
    """The native AlphaEdit tokenizer/alignment contract is not satisfied."""


def require_right_padding(tok: Any, *, caller: str) -> None:
    observed = getattr(tok, "padding_side", None)
    if observed != "right":
        raise TokenizerContractBoundary(
            f"{caller} requires tok.padding_side='right', observed={observed!r}"
        )


def normalize_target_text(target: str) -> str:
    if not target:
        raise TokenizerContractBoundary("target text is empty")
    return target if target.startswith(" ") else " " + target


def target_token_ids(tok: Any, target: str) -> List[int]:
    ids = list(tok.encode(normalize_target_text(str(target)), add_special_tokens=False))
    while ids and ids[0] in {tok.bos_token_id, tok.unk_token_id}:
        ids = ids[1:]
    if not ids:
        raise TokenizerContractBoundary("target tokenization is empty")
    return [int(value) for value in ids]


def prompt_token_ids(tok: Any, prompt: str) -> List[int]:
    ids = list(tok(prompt, add_special_tokens=True)["input_ids"])
    if not ids:
        raise TokenizerContractBoundary("rewrite prompt tokenization is empty")
    return [int(value) for value in ids]


@dataclass(frozen=True)
class AlignmentFixture:
    criterion: str
    ordinal: int
    case_id: int
    target_new_ids: tuple[int, ...]
    target_true_ids: tuple[int, ...]
    identity_sha256: str


def _fixture(
    criterion: str,
    ordinal: int,
    record: Mapping[str, Any],
    new_ids: Sequence[int],
    true_ids: Sequence[int],
) -> AlignmentFixture:
    rewrite = record["requested_rewrite"]
    digest = sha256()
    for value in (
        criterion,
        str(ordinal),
        str(record["case_id"]),
        str(rewrite["prompt"]),
        str(rewrite["subject"]),
        str(rewrite["target_new"]["str"]),
        str(rewrite["target_true"]["str"]),
        ",".join(str(value) for value in new_ids),
        ",".join(str(value) for value in true_ids),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return AlignmentFixture(
        criterion=criterion,
        ordinal=ordinal,
        case_id=int(record["case_id"]),
        target_new_ids=tuple(int(value) for value in new_ids),
        target_true_ids=tuple(int(value) for value in true_ids),
        identity_sha256=digest.hexdigest(),
    )


def select_multitoken_alignment_fixtures(
    records: Sequence[Dict[str, Any]], tok: Any
) -> tuple[AlignmentFixture, ...]:
    """Select the first outcome-independent token-length fixtures in stream order."""

    selected: Dict[str, AlignmentFixture] = {}
    for ordinal, record in enumerate(records):
        rewrite = record["requested_rewrite"]
        new_ids = target_token_ids(tok, rewrite["target_new"]["str"])
        true_ids = target_token_ids(tok, rewrite["target_true"]["str"])
        criteria = []
        if len(new_ids) >= 2:
            criteria.append("target_new_multitoken")
        if len(true_ids) > len(new_ids):
            criteria.append("source_longer")
        if len(new_ids) > len(true_ids):
            criteria.append("target_longer")
        for criterion in criteria:
            if criterion not in selected:
                selected[criterion] = _fixture(
                    criterion, ordinal, record, new_ids, true_ids
                )
        if len(selected) == 3:
            break
    required = ("target_new_multitoken", "source_longer", "target_longer")
    missing = [criterion for criterion in required if criterion not in selected]
    if missing:
        raise TokenizerContractBoundary(
            f"deterministic multi-token alignment fixtures are unavailable: {missing}"
        )
    return tuple(selected[criterion] for criterion in required)
