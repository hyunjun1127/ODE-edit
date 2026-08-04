"""Endpoint-only CounterFact evaluation for Session 03 CT-K4."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from project.run_scripts.ode_edit_motivation.contracts import EditRequest
from project.run_scripts.ode_edit_motivation.manifests import (
    COUNTERFACT_RELATIVE_PATH,
    _iter_top_level_json_objects,
)

from .contracts import MethodContractError, canonical_hash
from .events import (
    ControllerRequest,
    InformationFirewall,
    build_allowed_contexts,
    normalize_object_text,
)


EVALUATION_SCHEMA_VERSION = "ode-edit-session03-ct-k4-eval/v2"
EVALUATION_METRIC_POLICY = {
    "schema_version": "ode-edit-session03-ct-k4-eval-policy/v2",
    "efficacy_context": "controller-direct-subject-formatted",
    "literal_rewrite_template_allowed": False,
    "target_span": "normalized-target-exact-suffix-derived-start",
    "generalization_inputs": "counterfact-paraphrases-byte-preserved",
    "locality_inputs": "counterfact-neighborhoods-byte-preserved",
}
EVALUATION_METRIC_POLICY_ID = canonical_hash(EVALUATION_METRIC_POLICY)


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


def _normalized_target_ids(tokenizer: Any, target: str) -> tuple[int, ...]:
    values = tuple(
        int(value)
        for value in tokenizer.encode(
            normalize_object_text(target),
            add_special_tokens=False,
        )
    )
    bos = getattr(tokenizer, "bos_token_id", None)
    unk = getattr(tokenizer, "unk_token_id", None)
    if values and values[0] in {bos, unk}:
        values = values[1:]
    if not values:
        raise MethodContractError("evaluation normalized target tokenization is empty")
    return values


def _canonical_efficacy_prompt(
    tokenizer: Any,
    request: ControllerRequest,
) -> tuple[str, str]:
    """Bind endpoint efficacy to the controller's direct rendered context."""

    formatted = request.prompt.format(request.subject)
    controller_direct = build_allowed_contexts(request, (("{}",),))[0]
    if (
        formatted != controller_direct
        or formatted == request.prompt
        or "{}" in formatted
    ):
        raise MethodContractError("efficacy prompt is not the canonical rendered context")
    formatted_ids = tuple(
        int(value)
        for value in tokenizer.encode(formatted, add_special_tokens=True)
    )
    controller_ids = tuple(
        int(value)
        for value in tokenizer.encode(controller_direct, add_special_tokens=True)
    )
    literal_ids = tuple(
        int(value)
        for value in tokenizer.encode(request.prompt, add_special_tokens=True)
    )
    if (
        not formatted_ids
        or formatted_ids != controller_ids
        or literal_ids == formatted_ids
    ):
        raise MethodContractError("efficacy prompt token identity differs")
    return formatted, canonical_hash(list(formatted_ids))


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


def _model_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    raw = getattr(model, "device", torch.device("cpu"))
    return torch.device(raw)


def _move_batch_to_device(batch: Any, device: torch.device) -> Mapping[str, Any]:
    if hasattr(batch, "to"):
        moved = batch.to(device)
    elif isinstance(batch, Mapping):
        moved = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
    else:
        raise MethodContractError("evaluation tokenizer batch differs")
    if not isinstance(moved, Mapping):
        raise MethodContractError("evaluation tokenizer batch differs")
    return moved


def _config_snapshot(model: torch.nn.Module) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        return None
    if hasattr(config, "to_dict"):
        return copy.deepcopy(config.to_dict())
    if hasattr(config, "__dict__"):
        return copy.deepcopy(vars(config))
    return copy.deepcopy(config)


def _parameter_guards(model: torch.nn.Module) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            name,
            parameter.data_ptr(),
            parameter._version,
            parameter.requires_grad,
            parameter.grad,
        )
        for name, parameter in model.named_parameters()
    )


def _assert_evaluation_invariants(
    model: torch.nn.Module,
    *,
    parameter_guards: tuple[tuple[Any, ...], ...],
    config_snapshot: Any,
    cpu_rng: torch.Tensor,
    cuda_rng: tuple[torch.Tensor, ...],
) -> None:
    current = _parameter_guards(model)
    if len(current) != len(parameter_guards):
        raise MethodContractError("evaluation model parameter set changed")
    for before, after in zip(parameter_guards, current, strict=True):
        if before[:4] != after[:4] or before[4] is not after[4]:
            raise MethodContractError("evaluation model parameter state changed")
    if _config_snapshot(model) != config_snapshot:
        raise MethodContractError("evaluation model config changed")
    if not torch.equal(torch.get_rng_state(), cpu_rng):
        raise MethodContractError("evaluation CPU RNG changed")
    if cuda_rng:
        current_cuda_rng = tuple(torch.cuda.get_rng_state_all())
        if len(current_cuda_rng) != len(cuda_rng) or any(
            not torch.equal(before, after)
            for before, after in zip(cuda_rng, current_cuda_rng, strict=True)
        ):
            raise MethodContractError("evaluation CUDA RNG changed")


def teacher_forced_token_accuracy(
    model: torch.nn.Module,
    tokenizer: Any,
    hparams: Any,
    prompts: Sequence[str] | str,
    targets: Sequence[str] | str,
    *,
    locality: bool = False,
) -> list[float] | list[list[int]]:
    """EasyEdit-compatible next-token scoring without importing its evaluator."""

    prompt_rows = (prompts,) if isinstance(prompts, str) else tuple(prompts)
    target_rows = (targets,) if isinstance(targets, str) else tuple(targets)
    if (
        not prompt_rows
        or len(prompt_rows) != len(target_rows)
        or any(not isinstance(value, str) or not value for value in prompt_rows)
        or any(not isinstance(value, str) or not value.strip() for value in target_rows)
    ):
        raise MethodContractError("evaluation prompt/target panel differs")
    rendered_prompts: tuple[str, ...] = prompt_rows
    if not locality and bool(getattr(hparams, "use_chat_template", False)):
        rendered = tokenizer.apply_chat_template(
            [[{"role": "user", "content": prompt}] for prompt in prompt_rows],
            add_generation_prompt=True,
            tokenize=False,
        )
        if isinstance(rendered, str):
            rendered = [rendered]
        rendered_prompts = tuple(rendered)
        if len(rendered_prompts) != len(prompt_rows) or any(
            not isinstance(value, str) or not value for value in rendered_prompts
        ):
            raise MethodContractError("evaluation chat template output differs")

    combined = tuple(
        prompt + " " + target
        for prompt, target in zip(rendered_prompts, target_rows, strict=True)
    )
    normalized_target_ids = tuple(
        _normalized_target_ids(tokenizer, target) for target in target_rows
    )
    encoded_lengths = tuple(len(tokenizer.encode(value)) for value in combined)
    if not encoded_lengths or any(length <= 0 for length in encoded_lengths):
        raise MethodContractError("evaluation encoded sequence is empty")
    try:
        configured_max_length = int(hparams.max_length)
    except (AttributeError, TypeError, ValueError) as exc:
        raise MethodContractError("evaluation max length differs") from exc
    max_length = max(configured_max_length, max(encoded_lengths) + 1)
    if max_length <= 1:
        raise MethodContractError("evaluation max length differs")

    parameter_guards = _parameter_guards(model)
    config_snapshot = _config_snapshot(model)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    original_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        combined_tokens = tokenizer(
            list(combined),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if not isinstance(combined_tokens, Mapping):
            raise MethodContractError("evaluation combined tokenization differs")
        combined_input_ids = combined_tokens.get("input_ids")
        combined_attention_mask = combined_tokens.get("attention_mask")
        if (
            not isinstance(combined_input_ids, torch.Tensor)
            or combined_input_ids.ndim != 2
            or not isinstance(combined_attention_mask, torch.Tensor)
            or combined_attention_mask.shape != combined_input_ids.shape
        ):
            raise MethodContractError("evaluation combined tokenization differs")
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad_token_id, int):
            raise MethodContractError("evaluation pad token differs")
        width = int(combined_input_ids.shape[1])
        starts: list[int] = []
        for row_index, target_ids in enumerate(normalized_target_ids):
            mask = combined_attention_mask[row_index]
            nonpad = int(mask.sum().item())
            padding = width - nonpad
            if (
                nonpad <= len(target_ids)
                or padding < 0
                or bool(mask[:padding].any())
                or not bool(mask[padding:].all())
            ):
                raise MethodContractError("evaluation left-padding layout differs")
            end = padding + nonpad
            start = end - len(target_ids)
            suffix = tuple(
                int(value)
                for value in combined_input_ids[row_index, start:end]
            )
            if suffix != target_ids:
                raise MethodContractError(
                    "evaluation combined sequence target suffix differs"
                )
            starts.append(start)
        device_batch = _move_batch_to_device(combined_tokens, _model_device(model))
        input_ids = device_batch.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise MethodContractError("evaluation combined tokenization differs")
        if input_ids.shape[0] != len(prompt_rows):
            raise MethodContractError("evaluation batch size differs")
        with torch.no_grad():
            outputs = model(**device_batch)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
            if (
                not isinstance(logits, torch.Tensor)
                or logits.ndim != 3
                or logits.shape[:2] != input_ids.shape
            ):
                raise MethodContractError("evaluation model logits differ")
            predictions = torch.argmax(logits, dim=-1).detach().cpu()
            labels = input_ids.detach().cpu()
        token_predictions: list[list[int]] = []
        token_accuracies: list[float] = []
        sequence_length = int(labels.shape[1])
        for row_index, start in enumerate(starts):
            if start < 1 or start >= sequence_length:
                raise MethodContractError("evaluation target span is empty")
            predicted = predictions[row_index, start - 1 : -1]
            expected = labels[row_index, start:]
            if predicted.numel() == 0 or predicted.shape != expected.shape:
                raise MethodContractError("evaluation target alignment differs")
            token_predictions.append([int(value) for value in predicted.tolist()])
            token_accuracies.append(
                float((predicted == expected).to(torch.float64).mean().item())
            )
    finally:
        tokenizer.padding_side = original_padding_side
        _assert_evaluation_invariants(
            model,
            parameter_guards=parameter_guards,
            config_snapshot=config_snapshot,
            cpu_rng=cpu_rng,
            cuda_rng=cuda_rng,
        )
    return token_predictions if locality else token_accuracies


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
    efficacy_prompt, efficacy_context_token_hash = _canonical_efficacy_prompt(
        runtime.tokenizer, request
    )
    def accuracy(prompts: Sequence[str], target: str, *, locality: bool) -> Any:
        instrumentation.increment("N_eval")
        return teacher_forced_token_accuracy(
            runtime.model,
            runtime.tokenizer,
            hparams,
            tuple(prompts),
            tuple(target for _ in prompts),
            locality=locality,
        )

    with instrumentation.component("evaluation"):
        endpoint_checkpoint.restore(runtime.model)
        efficacy = tuple(
            float(value)
            for value in accuracy((efficacy_prompt,), request.target_new, locality=False)
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
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "case_id": request.case_id,
        "metric_policy_id": EVALUATION_METRIC_POLICY_ID,
        "metric_policy": dict(EVALUATION_METRIC_POLICY),
        "canonical_efficacy_context_token_hash": efficacy_context_token_hash,
        "canonical_efficacy_context_token_identity": True,
        "normalized_target_suffix_identity": True,
        "payload_hash": canonical_hash(
            {
                "case_id": request.case_id,
                "paraphrase_count": len(paraphrases),
                "neighborhood_count": len(neighborhoods),
                "metric_policy_id": EVALUATION_METRIC_POLICY_ID,
                "canonical_efficacy_context_token_hash": efficacy_context_token_hash,
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
