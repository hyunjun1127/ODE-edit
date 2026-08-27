#!/usr/bin/env python3
"""Seal the outcome-independent Stage-C topology token tuples."""

from __future__ import annotations

import json
import os
from pathlib import Path

from project.run_scripts.barrier_guided_ode.r2.stage_b_contract import MODEL_BINDINGS, seal_tokenization
from project.run_scripts.barrier_guided_ode.r2.stage_c_contract import TOPOLOGY_MANIFEST
from project.run_scripts.barrier_guided_ode.s1_contract import load_sealed_s1_sample
from project.run_scripts.ode_edit_motivation.contracts import canonical_json, sha256_bytes
from project.run_scripts.ode_edit_motivation.gpu_runtime import fixed_pretrained_kwargs
from project.run_scripts.ode_edit_motivation.manifests import fixed_model_spec


def _write_once(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise SystemExit(f"existing topology manifest differs: {path}")
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def main() -> int:
    from transformers import AutoTokenizer

    sample = load_sealed_s1_sample()
    cases = []
    tokenizations = []
    for alias in MODEL_BINDINGS:
        tokenizer = AutoTokenizer.from_pretrained(**fixed_pretrained_kwargs(fixed_model_spec(alias)))
        sealed = seal_tokenization(tokenizer, sample, model_alias=alias)
        if len(sealed.target_token_ids) != 1 or len(sealed.source_token_ids) != 1:
            raise SystemExit("Stage-C deterministic construction requires sealed request000 singleton tokens")
        target = sealed.target_token_ids[0]
        source = sealed.source_token_ids[0]
        if target == source:
            raise SystemExit("Stage-C deterministic anchors are equal")
        definitions = (
            ("both-multi-unequal-nonprefix", (target, source), (source, target, source)),
            ("source-prefix-target", (source, target), (source,)),
            ("target-prefix-source", (target,), (target, source)),
        )
        tokenizations.append(sealed.receipt())
        for topology, target_ids, source_ids in definitions:
            row = {
                "model_alias": alias,
                "topology": topology,
                "construction": "SEALED_REQUEST000_TOKEN_CONCATENATION_OUTCOME_INDEPENDENT",
                "target_token_ids": list(target_ids),
                "source_token_ids": list(source_ids),
                "base_tokenization_identity": sealed.identity,
                "sample_identity": sample.identity,
            }
            row["case_identity"] = sha256_bytes(canonical_json(row).encode("utf-8"))
            cases.append(row)
    body = {
        "schema": "ode-edit-bgode-r2-stage-c-topology-manifest/v1",
        "selection_influence": "TOKENIZER_IDS_ONLY_NO_MODEL_OUTPUT_EFFICACY_LOCALITY",
        "sample": sample.receipt(),
        "sample_identity": sample.identity,
        "case_count": len(cases),
        "model_count": 2,
        "topology_count_per_model": 3,
        "termination_token_count": 0,
        "tokenizations": tokenizations,
        "cases": cases,
        "scientific_promotion": False,
    }
    body["identity"] = sha256_bytes(canonical_json(body).encode("utf-8"))
    TOPOLOGY_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    _write_once(TOPOLOGY_MANIFEST, (canonical_json(body) + "\n").encode("utf-8"))
    print(canonical_json({"path": str(TOPOLOGY_MANIFEST.resolve()), "identity": body["identity"], "case_count": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
