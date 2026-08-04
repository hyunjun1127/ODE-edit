"""Endpoint-only CounterFact evaluation for Session 03 CT-K4."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from project.run_scripts.ode_edit_motivation.contracts import EditRequest
from project.run_scripts.ode_edit_motivation.manifests import (
    COUNTERFACT_RELATIVE_PATH,
    _iter_top_level_json_objects,
)

from .contracts import MethodContractError, canonical_hash
from .events import ControllerRequest, InformationFirewall


def load_evaluation_payloads(
    easyedit_root: str | Path,
    requests: Sequence[ControllerRequest],
) -> dict[str, Mapping[str, Any]]:
    """Load private held-out fields for closed firewalls, never controllers."""

    expected = {request.case_id: request for request in requests}
    if not expected or len(expected) != len(requests):
        raise MethodContractError("evaluation request identities differ")
    source = Path(easyedit_root).resolve(strict=True) / COUNTERFACT_RELATIVE_PATH
    found: dict[str, Mapping[str, Any]] = {}
    for blob in _iter_top_level_json_objects(source):
        row = json.loads(blob)
        case_id = str(row.get("case_id"))
        if case_id not in expected:
            continue
        request = ControllerRequest.from_counterfact_row(row)
        if request != expected[case_id]:
            raise MethodContractError("evaluation row differs from controller projection")
        rewrite = row.get("requested_rewrite")
        paraphrases = row.get("paraphrase_prompts")
        neighborhoods = row.get("neighborhood_prompts")
        if (
            not isinstance(rewrite, Mapping)
            or not isinstance(paraphrases, list)
            or not paraphrases
            or not isinstance(neighborhoods, list)
            or not neighborhoods
            or any(not isinstance(value, str) or not value.strip() for value in paraphrases)
            or any(not isinstance(value, str) or not value.strip() for value in neighborhoods)
        ):
            raise MethodContractError("CounterFact evaluation payload differs")
        target_true = str(rewrite.get("target_true", {}).get("str", ""))
        if not target_true.strip():
            raise MethodContractError("CounterFact locality target is empty")
        private = {
            "paraphrase_prompts": tuple(paraphrases),
            "neighborhood_prompts": tuple(neighborhoods),
            "target_true": target_true,
        }
        found[case_id] = private
        if len(found) == len(expected):
            break
    if set(found) != set(expected):
        raise MethodContractError("CounterFact evaluation rows are incomplete")
    return found


def make_firewall(
    request: ControllerRequest,
    private_payload: Mapping[str, Any] | None,
) -> InformationFirewall:
    return InformationFirewall(request, private_payload)


def _mean(values: Sequence[float]) -> float:
    locked = tuple(float(value) for value in values)
    if not locked or any(not math.isfinite(value) for value in locked):
        raise MethodContractError("evaluation metric panel is invalid")
    return math.fsum(locked) / len(locked)


def _token_agreement(before: Sequence[Sequence[int]], after: Sequence[Sequence[int]]) -> float:
    if len(before) != len(after) or not before:
        raise MethodContractError("locality token panels differ")
    scores = []
    for left, right in zip(before, after, strict=True):
        if len(left) != len(right) or not left:
            raise MethodContractError("locality token sequences differ")
        scores.append(
            math.fsum(1.0 if a == b else 0.0 for a, b in zip(left, right, strict=True))
            / len(left)
        )
    return _mean(scores)


def evaluate_frozen_endpoint(
    *,
    runtime: Any,
    hparams: Any,
    request: ControllerRequest,
    firewall: InformationFirewall,
    baseline_checkpoint: Any,
    endpoint_checkpoint: Any,
    instrumentation: Any,
) -> dict[str, Any]:
    """Evaluate only after action freeze, restoring the endpoint afterward."""

    if not firewall.action_frozen:
        raise MethodContractError("endpoint evaluation precedes action freeze")
    private = firewall.open_evaluation()
    try:
        paraphrases = tuple(private["paraphrase_prompts"])
        neighborhoods = tuple(private["neighborhood_prompts"])
        target_true = str(private["target_true"])
    except (KeyError, TypeError) as exc:
        raise MethodContractError("opened evaluation payload differs") from exc
    # Import only after action freeze.  This is a read-only EasyEdit evaluator.
    from easyeditor.evaluate.evaluate_utils import test_prediction_acc

    def accuracy(prompts: Sequence[str], target: str, *, locality: bool) -> Any:
        instrumentation.increment("N_eval")
        return test_prediction_acc(
            runtime.model,
            runtime.tokenizer,
            hparams,
            list(prompts),
            [target for _ in prompts],
            0,
            locality=locality,
        )

    with instrumentation.component("evaluation"):
        endpoint_checkpoint.restore(runtime.model)
        efficacy = tuple(
            float(value)
            for value in accuracy((request.prompt,), request.target_new, locality=False)
        )
        generalization = tuple(
            float(value)
            for value in accuracy(paraphrases, request.target_new, locality=False)
        )
        endpoint_locality_tokens = accuracy(
            neighborhoods, target_true, locality=True
        )
        locality_target_accuracy = tuple(
            float(value)
            for value in accuracy(neighborhoods, target_true, locality=False)
        )
        baseline_checkpoint.restore(runtime.model)
        baseline_locality_tokens = accuracy(
            neighborhoods, target_true, locality=True
        )
        endpoint_checkpoint.restore(runtime.model)
    result = {
        "schema_version": "ode-edit-session03-ct-k4-eval/v1",
        "case_id": request.case_id,
        "payload_hash": canonical_hash(
            {
                "case_id": request.case_id,
                "paraphrase_count": len(paraphrases),
                "neighborhood_count": len(neighborhoods),
                "request_id": EditRequest.from_mapping(
                    {
                        "case_id": request.case_id,
                        "prompt": request.prompt,
                        "subject": request.subject,
                        "target_new": request.target_new,
                    }
                ).request_id,
            }
        ),
        "efficacy_token_accuracy": _mean(efficacy),
        "generalization_token_accuracy": _mean(generalization),
        "locality_pre_post_token_agreement": _token_agreement(
            baseline_locality_tokens, endpoint_locality_tokens
        ),
        "neighborhood_target_true_token_accuracy": _mean(
            locality_target_accuracy
        ),
        "paraphrase_count": len(paraphrases),
        "neighborhood_count": len(neighborhoods),
        "generation_executed": False,
        "action_frozen_before_open": True,
    }
    return result
