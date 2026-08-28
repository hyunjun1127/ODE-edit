"""Outcome-independent shared 8-request cohort sealing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from project.run_scripts.barrier_guided_ode.r3.natural import canonical_records, tokenize_record
from project.run_scripts.ode_edit_motivation.contracts import canonical_json, sha256_bytes
from project.run_scripts.ode_edit_motivation.gpu_runtime import offline_environment
from project.run_scripts.ode_edit_motivation.manifests import fixed_model_spec
from project.run_scripts.ode_edit_motivation.gpu_runtime import fixed_pretrained_kwargs


MODEL_ALIASES = ("llama3-8b-inst", "qwen2.5-7b-inst")


def build_shared_cohort(tokenizers: dict[str, Any], *, count: int = 8) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for record in canonical_records():
        tokens = {alias: tokenize_record(tokenizer, record, model_alias=alias) for alias, tokenizer in tokenizers.items()}
        if all(
            value.topology == "unequal-non-prefix"
            and max(len(value.target_token_ids), len(value.source_token_ids)) >= 2
            for value in tokens.values()
        ):
            cases.append(
                {
                    "ordinal": record.ordinal,
                    "batch_index": record.batch_index,
                    "batch_ordinal": record.batch_ordinal,
                    "case_id": record.case_id,
                    "raw_sha256": record.raw_sha256,
                    "request_sha256": record.request_sha256,
                    "tokenization": {alias: value.receipt() for alias, value in tokens.items()},
                }
            )
            if len(cases) == count:
                break
    if len(cases) != count:
        raise RuntimeError("canonical stream has fewer than eight shared natural multi-token cases")
    body = {
        "schema": "ode-edit-fixed-basis-barrier-ode-cohort/v1",
        "selection": "EARLIEST_CANONICAL_ORDINAL_SHARED_UNEQUAL_NONPREFIX_WITH_MULTITOKEN_SIDE",
        "outcome_influence_count": 0,
        "case_count": count,
        "cases": cases,
    }
    return {**body, "identity": sha256_bytes(canonical_json(body).encode("utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizers: dict[str, Any] = {}
    with offline_environment():
        for alias in MODEL_ALIASES:
            tokenizers[alias] = AutoTokenizer.from_pretrained(
                **fixed_pretrained_kwargs(fixed_model_spec(alias)), use_fast=True
            )
    payload = build_shared_cohort(tokenizers)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
