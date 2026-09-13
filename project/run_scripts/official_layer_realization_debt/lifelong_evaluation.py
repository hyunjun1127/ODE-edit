"""Evaluation-only current/retention/locality checkpoint panels."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.fixed_z_nonuniqueness.evaluation import sequence_metrics
from project.run_scripts.ode_bf.contracts import canonical_hash

from .contracts import DATASET_SHA256, ObservationBoundary


def load_evaluation_rows(
    dataset_path: Path, case_ids: set[int]
) -> dict[int, dict[str, Any]]:
    if hashlib.sha256(dataset_path.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ObservationBoundary("lifelong evaluator dataset identity differs")
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows: dict[int, dict[str, Any]] = {}
    for value in raw:
        case_id = int(value["case_id"])
        if case_id not in case_ids:
            continue
        rewrite = value["requested_rewrite"]
        rows[case_id] = {
            "case_id": case_id,
            "prompt": str(rewrite["prompt"]),
            "subject": str(rewrite["subject"]),
            "target_new": str(rewrite["target_new"]["str"]),
            "target_true": str(rewrite["target_true"]["str"]),
            "paraphrase_prompts": tuple(str(item) for item in value["paraphrase_prompts"]),
            "neighborhood_prompts": tuple(str(item) for item in value["neighborhood_prompts"]),
        }
    if set(rows) != case_ids:
        raise ObservationBoundary("lifelong evaluator row inventory differs")
    return rows


def _metrics(model: Any, tokenizer: Any, prompts: Sequence[str], target: str) -> dict[str, Any]:
    if not prompts:
        return {"count": 0, "nll_mean": None, "strict_count": 0, "margin_mean": None}
    value = sequence_metrics(model, tokenizer, list(prompts), target)
    nll = [float(item) for item in value["nll"]]
    margin = [float(item) for item in value["margin"]]
    strict = [bool(item) for item in value["strict"]]
    if not all(math.isfinite(item) for item in [*nll, *margin]):
        raise ObservationBoundary("lifelong evaluator metric is nonfinite")
    return {
        "count": len(prompts),
        "nll_mean": sum(nll) / len(nll),
        "strict_count": sum(strict),
        "margin_mean": sum(margin) / len(margin),
    }


def evaluate_panel(
    *,
    model: Any,
    tokenizer: Any,
    requests: Sequence[Mapping[str, Any]],
    evaluation_rows: Mapping[int, Mapping[str, Any]],
    include_rephrase: bool,
    include_locality: bool,
    panel_role: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for request in requests:
            case_id = int(request["case_id"])
            row = evaluation_rows[case_id]
            if (
                str(row["prompt"]) != str(request["prompt"])
                or str(row["subject"]) != str(request["subject"])
                or str(row["target_new"]) != str(request["target_new"])
                or str(row["target_true"]) != str(request["target_true"])
            ):
                raise ObservationBoundary("lifelong evaluator request binding differs")
            prompt = str(row["prompt"]).format(str(row["subject"]))
            rewrite_new = _metrics(model, tokenizer, [prompt], str(row["target_new"]))
            rewrite_true = _metrics(model, tokenizer, [prompt], str(row["target_true"]))
            rephrase = (
                _metrics(
                    model,
                    tokenizer,
                    list(row["paraphrase_prompts"]),
                    str(row["target_new"]),
                )
                if include_rephrase
                else {"status": "NOT_RECORDED_LOW_COST_BATCH"}
            )
            locality = (
                _metrics(
                    model,
                    tokenizer,
                    list(row["neighborhood_prompts"]),
                    str(row["target_true"]),
                )
                if include_locality
                else {"status": "NOT_RECORDED_LOW_COST_BATCH"}
            )
            record = {
                "request_sha256": str(request["request_sha256"]),
                "case_identity_sha256": hashlib.sha256(
                    f"case|{case_id}|{request['request_sha256']}".encode()
                ).hexdigest(),
                "rewrite_target_new": rewrite_new,
                "rewrite_target_true": rewrite_true,
                "rephrase_target_new": rephrase,
                "locality_target_true": locality,
                "raw_prompt_logit_generation_publish_count": 0,
            }
            record["identity_sha256"] = canonical_hash(record)
            records.append(record)
    return {
        "schema": "odeedit.s06.layer-realization-debt.functional-panel.v1",
        "panel_role": panel_role,
        "request_count": len(records),
        "include_rephrase": include_rephrase,
        "include_locality": include_locality,
        "controller_or_selection_influence_count": 0,
        "records": records,
        "identity_sha256": canonical_hash(records),
    }


def retention_cohorts(
    accepted: Sequence[Mapping[str, Any]], *, recent_end: int
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    history = tuple(accepted[:recent_end])
    if not history:
        return {"earliest": (), "recent": (), "hash_stratified": ()}
    earliest = history[:32]
    recent = history[-32:]
    ranked = sorted(
        history,
        key=lambda row: hashlib.sha256(
            f"retention|{row['request_sha256']}".encode()
        ).hexdigest(),
    )
    stratified = tuple(ranked[: min(64, len(ranked))])
    return {
        "earliest": tuple(earliest),
        "recent": tuple(recent),
        "hash_stratified": stratified,
    }


__all__ = ["evaluate_panel", "load_evaluation_rows", "retention_cohorts"]
