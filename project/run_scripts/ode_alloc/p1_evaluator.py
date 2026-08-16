"""Independent endpoint-frozen teacher-forced P1 evaluator."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .contracts import ODEAllocContractError, canonical_hash
from .p1_contracts import EndpointFreezeToken, INNER_MICROBATCH_SIZE
from .p1_scoring import TeacherForcedScorer, TeacherForcedSpec
from .selection import _iter_top_level_objects, project_request_identity


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: int
    request_hash: str
    prompt: str
    subject: str
    target_new: str
    target_old: str
    paraphrases: tuple[str, ...]
    neighborhoods: tuple[str, ...]


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ODEAllocContractError(f"endpoint {label} panel differs")
    return tuple(value)


def load_evaluation_cases_after_freeze(
    dataset_path: str | Path,
    *,
    freeze: EndpointFreezeToken,
    approved: Sequence[Mapping[str, Any]],
    inner_requests: Sequence[Mapping[str, Any]],
) -> tuple[EvaluationCase, ...]:
    freeze.validate()
    if len(approved) != 4 or len(inner_requests) != 4:
        raise ODEAllocContractError("endpoint evaluation request count differs")
    expected = {
        int(item["case_id"]): (str(item["request_hash"]), request)
        for item, request in zip(approved, inner_requests, strict=True)
    }
    selected: dict[int, bytes] = {}
    for blob in _iter_top_level_objects(Path(dataset_path).resolve(strict=True)):
        identity = project_request_identity(blob)
        if identity.case_id in expected:
            if identity.case_id in selected:
                raise ODEAllocContractError("endpoint evaluation case repeats")
            selected[identity.case_id] = blob
    if set(selected) != set(expected):
        raise ODEAllocContractError("endpoint evaluation case set differs")
    result: list[EvaluationCase] = []
    for case_id, (request_hash, inner) in expected.items():
        identity = project_request_identity(selected[case_id])
        if identity.request_hash != request_hash:
            raise ODEAllocContractError("endpoint evaluation request hash differs")
        try:
            row = json.loads(selected[case_id])
            rewrite = row["requested_rewrite"]
            prompt = rewrite["prompt"]
            subject = rewrite["subject"]
            target_new = rewrite["target_new"]["str"]
            target_old = rewrite["target_true"]["str"]
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ODEAllocContractError("endpoint evaluation row schema differs") from exc
        if (
            prompt != inner["prompt"]
            or subject != inner["subject"]
            or target_new != inner["target_new"]
            or target_old != inner["target_old"]
        ):
            raise ODEAllocContractError("endpoint evaluation and inner request differ")
        try:
            paraphrases = _strings(row["paraphrase_prompts"], "paraphrase")
            neighborhoods = _strings(row["neighborhood_prompts"], "neighborhood")
        except KeyError as exc:
            raise ODEAllocContractError("endpoint evaluation panels are absent") from exc
        result.append(
            EvaluationCase(
                case_id=case_id,
                request_hash=request_hash,
                prompt=prompt,
                subject=subject,
                target_new=target_new,
                target_old=target_old,
                paraphrases=paraphrases,
                neighborhoods=neighborhoods,
            )
        )
        row.clear()
    return tuple(result)


def _pair_record(margins: Sequence[float]) -> dict[str, Any]:
    if not margins or any(not math.isfinite(value) for value in margins):
        raise ODEAllocContractError("endpoint metric panel differs")
    return {
        "count": len(margins),
        "mean_margin": sum(margins) / len(margins),
        "minimum_margin": min(margins),
        "success_rate": sum(value >= 0.0 for value in margins) / len(margins),
    }


def evaluate_frozen_endpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    freeze: EndpointFreezeToken,
    cases: Sequence[EvaluationCase],
    anchor_requests: Sequence[Mapping[str, Any]],
    theta0_anchor_teacher: Sequence[float],
) -> dict[str, Any]:
    freeze.validate()
    if (
        len(cases) != 4
        or len(anchor_requests) != 16
        or len(theta0_anchor_teacher) != 16
    ):
        raise ODEAllocContractError("endpoint evaluator sealed panels differ")
    specs: list[TeacherForcedSpec] = []
    for case in cases:
        canonical = case.prompt.format(case.subject)
        specs.extend(
            (
                TeacherForcedSpec(f"case-{case.case_id}-eff-new", canonical, case.target_new),
                TeacherForcedSpec(f"case-{case.case_id}-eff-old", canonical, case.target_old),
            )
        )
        for index, prompt in enumerate(case.paraphrases):
            specs.extend(
                (
                    TeacherForcedSpec(f"case-{case.case_id}-gen-{index}-new", prompt, case.target_new),
                    TeacherForcedSpec(f"case-{case.case_id}-gen-{index}-old", prompt, case.target_old),
                )
            )
        for index, prompt in enumerate(case.neighborhoods):
            specs.extend(
                (
                    TeacherForcedSpec(f"case-{case.case_id}-loc-{index}-new", prompt, case.target_new),
                    TeacherForcedSpec(f"case-{case.case_id}-loc-{index}-old", prompt, case.target_old),
                )
            )
    for index, request in enumerate(anchor_requests):
        specs.append(
            TeacherForcedSpec(
                f"anchor-{index}-old",
                str(request["prompt"]).format(str(request["subject"])),
                str(request["target_old"]),
            )
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_event = end_event = None
    started = time.perf_counter()
    result = TeacherForcedScorer(
        model, tokenizer, microbatch_size=INNER_MICROBATCH_SIZE
    ).score(specs, gradient=False)
    if end_event is not None and start_event is not None:
        end_event.record()
        torch.cuda.synchronize()
        gpu_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
    else:
        gpu_seconds = 0.0
    wall_seconds = time.perf_counter() - started
    values = {key: float(value.detach().to(device="cpu")) for key, value in result.logp.items()}
    case_rows: list[dict[str, Any]] = []
    for position, case in enumerate(cases):
        efficacy = [
            values[f"case-{case.case_id}-eff-new"]
            - values[f"case-{case.case_id}-eff-old"]
        ]
        generalization = [
            values[f"case-{case.case_id}-gen-{index}-new"]
            - values[f"case-{case.case_id}-gen-{index}-old"]
            for index in range(len(case.paraphrases))
        ]
        locality = [
            values[f"case-{case.case_id}-loc-{index}-old"]
            - values[f"case-{case.case_id}-loc-{index}-new"]
            for index in range(len(case.neighborhoods))
        ]
        case_rows.append(
            {
                "case_id": case.case_id,
                "role": "current" if position == len(cases) - 1 else "previous",
                "efficacy": _pair_record(efficacy),
                "generalization": _pair_record(generalization),
                "locality": _pair_record(locality),
            }
        )
    anchor_damage = tuple(
        max(0.0, float(teacher) - values[f"anchor-{index}-old"])
        for index, teacher in enumerate(theta0_anchor_teacher)
    )
    payload = {
        "schema_version": "ode-alloc-s04-p1-endpoint-evaluation/v1",
        "freeze_token_id": freeze.token_id,
        "arm": freeze.arm.value,
        "case_metrics": case_rows,
        "anchor_preservation": {
            "count": len(anchor_damage),
            "raw_damage": anchor_damage,
            "mean_damage": sum(anchor_damage) / len(anchor_damage),
            "max_damage": max(anchor_damage),
        },
        "accounting": {
            "n_eval_specs": len(specs),
            "model_forward_calls": result.model_forward_calls,
            "processed_tokens": result.processed_tokens,
            "wall_seconds": wall_seconds,
            "gpu_seconds": gpu_seconds,
            "panel_id": result.panel_id,
        },
        "model_generate_calls": 0,
        "action_feedback_to_controller": False,
    }
    payload["evaluation_id"] = canonical_hash(payload)
    values.clear()
    return payload
