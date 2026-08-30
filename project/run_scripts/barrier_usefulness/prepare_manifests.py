"""Metadata-only deterministic fresh-case and sealed anchor-pool manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from project.run_scripts.fixed_z_nonuniqueness.contracts import MODEL_SPECS
from project.run_scripts.fixed_z_nonuniqueness.evaluation import target_ids

from .contracts import BarrierLock
from .hashing import canonical_hash, file_sha256, write_json_once


def _request(row: dict[str, Any]) -> dict[str, Any]:
    req = row["requested_rewrite"]
    return {
        "case_id": int(row["case_id"]),
        "subject": req["subject"],
        "relation_id": req["relation_id"],
        "prompt": req["prompt"],
        "target_new": req["target_new"]["str"],
        "target_true": req["target_true"]["str"],
    }


def _used_ids(prior: dict[str, Any]) -> set[int]:
    answer = {int(row["case_id"]) for row in prior["screen_cases"]}
    for rows in prior["roles"].values():
        answer.update(int(row["case_id"]) for row in rows)
    return answer


def _rank(namespace: str, row: dict[str, Any]) -> str:
    return canonical_hash([BarrierLock().seed_namespace, namespace, _request(row)])


def build(dataset: Path, prior_manifest: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = json.loads(dataset.read_text())
    prior = json.loads(prior_manifest.read_text())
    used = _used_ids(prior)
    tokenizers = {
        alias: AutoTokenizer.from_pretrained(spec.model_path, local_files_only=True, use_fast=True, trust_remote_code=False)
        for alias, spec in MODEL_SPECS.items()
    }
    eligible = [row for row in rows if int(row["case_id"]) not in used]
    eligible.sort(key=lambda row: _rank("fresh-edit", row))
    selected: list[dict[str, Any]] = []
    unique_subjects: set[str] = set()
    unique_relations: set[str] = set()
    unique_targets: set[str] = set()
    target_class_counts = {"single_both": 0, "multi_both": 0}
    for row in eligible:
        req = row["requested_rewrite"]
        counts = {alias: int(target_ids(tok, req["target_new"]["str"]).numel()) for alias, tok in tokenizers.items()}
        category = "single_both" if all(value == 1 for value in counts.values()) else "multi_both" if all(value > 1 for value in counts.values()) else None
        if category is None or target_class_counts[category] >= 4:
            continue
        if req["subject"] in unique_subjects or req["relation_id"] in unique_relations:
            continue
        target_identity = canonical_hash([req["target_new"]["str"], req["target_true"]["str"]])
        if target_identity in unique_targets:
            continue
        payload = _request(row)
        payload.update({
            "ordinal": len(selected),
            "target_class": category,
            "target_new_token_counts": counts,
            "row_sha256": canonical_hash(row),
            "request_sha256": canonical_hash(_request(row)),
            "selection_rank": _rank("fresh-edit", row),
        })
        selected.append(payload)
        target_class_counts[category] += 1
        unique_subjects.add(req["subject"])
        unique_relations.add(req["relation_id"])
        unique_targets.add(target_identity)
        if len(selected) == 8:
            break
    if target_class_counts != {"single_both": 4, "multi_both": 4}:
        raise RuntimeError(f"fresh 4/4 tokenizer-balanced selection absent: {target_class_counts}")

    selected_ids = {row["case_id"] for row in selected}
    pools = []
    for edit in selected:
        candidates = []
        for row in eligible:
            req = row["requested_rewrite"]
            if int(row["case_id"]) in selected_ids:
                continue
            if req["subject"] == edit["subject"] or req["relation_id"] == edit["relation_id"]:
                continue
            values = {req["subject"], req["target_new"]["str"], req["target_true"]["str"]}
            forbidden = {edit["subject"], edit["target_new"], edit["target_true"]}
            if values & forbidden:
                continue
            candidates.append(row)
        ctrl = sorted(candidates, key=lambda row: _rank(f"ctrl-pool|{edit['case_id']}", row))[:96]
        ctrl_ids = {int(row["case_id"]) for row in ctrl}
        gate = sorted(
            [row for row in candidates if int(row["case_id"]) not in ctrl_ids],
            key=lambda row: _rank(f"gate-pool|{edit['case_id']}", row),
        )[:96]
        if len(ctrl) != 96 or len(gate) != 96:
            raise RuntimeError("insufficient disjoint metadata anchor pools")
        pools.append({
            "edit_case_id": edit["case_id"],
            "ctrl_pool": [{**_request(row), "row_sha256": canonical_hash(row), "rank": _rank(f"ctrl-pool|{edit['case_id']}", row)} for row in ctrl],
            "gate_pool": [{**_request(row), "row_sha256": canonical_hash(row), "rank": _rank(f"gate-pool|{edit['case_id']}", row)} for row in gate],
            "ctrl_gate_overlap_count": 0,
        })
    fresh = {
        "schema": "odeedit.s06.barrier-usefulness.fresh-case-manifest.v1",
        "selection": "metadata-only sha256 rank; 4 single-token-both + 4 multi-token-both; unique subject/relation/target pair",
        "dataset": {"path": str(dataset), "sha256": file_sha256(dataset), "bytes": dataset.stat().st_size},
        "prior_manifest": {"path": str(prior_manifest), "sha256": file_sha256(prior_manifest)},
        "excluded_prior_case_count": len(used),
        "fresh_cases": selected,
        "replacement_count": 0,
        "outcome_influence_count": 0,
        "final_audit_open_count": 0,
    }
    fresh["root_digest"] = canonical_hash(fresh)
    anchor = {
        "schema": "odeedit.s06.barrier-usefulness.anchor-pool-manifest.v1",
        "rule": {
            "pool_size_per_split": 96,
            "final_size_per_split": 32,
            "admissibility": "W0 strict factual target_true continuation and every token margin > locked numerical floor",
            "final_selection": "first 32 admissible in preregistered rank order independently for model/edit/split",
            "candidate_outcome_access_count": 0,
            "replacement_after_candidate_access": 0,
        },
        "pools": pools,
        "sample_duplication_count": 0,
    }
    anchor["root_digest"] = canonical_hash(anchor)
    return fresh, anchor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--fresh-output", type=Path, required=True)
    parser.add_argument("--anchor-output", type=Path, required=True)
    args = parser.parse_args()
    fresh, anchor = build(args.dataset, args.prior_manifest)
    write_json_once(args.fresh_output, fresh)
    write_json_once(args.anchor_output, anchor)


if __name__ == "__main__":
    main()
