"""Actual causal-LM factual observations for the official baseline streams.

Raw rows are local-only: they contain token IDs, prompts and per-request IDs.
``summary`` alone is suitable for a scalar-only report. This module does not
generate text, fit targets, change native edit state, or infer missing scores.

CF preserves native ``prefix + ' ' + target`` and independent no-BOS target
tokenization. It checks their token boundary before running the model, instead
of silently shifting suffix positions. zsRE uses exact teacher-forced token
prefixes, not decode/re-tokenize or a model-name-specific BOS slicing hack.
This canonical token-prefix implementation needs the prescribed actual native
evaluator parity check; passing the CPU tests is not that check.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import time
from copy import deepcopy

import numpy as np
import torch

from .reduce import counterfact, zsre


SCHEMA = "official-factual-causal-v1"
W0_SCHEMA = "official-zsre-W0-token-predictions-v1"
TOKENIZATION = "NATIVE_CF_VERIFIED_BOUNDARY_ZSRE_EXACT_TOKEN_PREFIX_NO_TARGET_BOS"


class FactualError(ValueError):
    """A typed input/identity/numeric failure, never a low-quality gate."""


def _require(condition, code):
    if not condition:
        raise FactualError(code)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _ids(tokenizer, text, *, special=True):
    _require(isinstance(text, str) and bool(text), "FACTUAL_EMPTY_TEXT")
    value = tokenizer(text, add_special_tokens=special, truncation=False)["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    _require(isinstance(value, list) and bool(value)
             and all(type(token) is int and token >= 0 for token in value),
             "FACTUAL_TOKENIZER_IDS")
    return value


def _target(tokenizer, text):
    # A target's BOS is never an answer token. The prompt retains its native BOS.
    return _ids(tokenizer, " " + text, special=False)


def _occurrence(record):
    value = record.get("occurrence_index", record.get("ordered_occurrence"))
    _require(type(value) is int and value > 0, "FACTUAL_OCCURRENCE_REQUIRED")
    _require(type(record.get("case_id")) is int, "FACTUAL_CASE_ID_REQUIRED")
    return value


def _groups(record):
    rewrite = record["requested_rewrite"]
    template, subject = rewrite["prompt"], rewrite["subject"]
    _require(isinstance(template, str) and template.count("{}") == 1
             and isinstance(subject, str) and bool(subject), "FACTUAL_REWRITE_SCHEMA")
    paraphrases, neighbors = record["paraphrase_prompts"], record["neighborhood_prompts"]
    _require(isinstance(paraphrases, list) and bool(paraphrases)
             and isinstance(neighbors, list) and bool(neighbors), "FACTUAL_MISSING_PROMPTS")
    return {"rewrite": [template.format(subject)], "paraphrase": paraphrases,
            "neighborhood": neighbors}


def _limit(model, tokenizer):
    values = []
    for source, names in ((getattr(model, "config", None),
                           ("max_position_embeddings", "n_positions", "max_sequence_length")),
                          (tokenizer, ("model_max_length",))):
        for name in names:
            value = getattr(source, name, None)
            if type(value) is int and 0 < value < 10 ** 9:
                values.append(value)
    return min(values) if values else None


def _device(model, supplied):
    if supplied is not None:
        return torch.device(supplied)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _tensor_signature(model):
    return tuple((kind, name, id(value), value.data_ptr(), value._version,
                  tuple(value.shape), str(value.dtype), str(value.device), value.requires_grad)
                 for kind, iterator in (("parameter", model.named_parameters()),
                                        ("buffer", model.named_buffers()))
                 for name, value in iterator)


def _control_signature(model):
    hooks = tuple((name, tuple((attribute, tuple((key, id(hook)) for key, hook in
                    getattr(module, attribute, {}).items())) for attribute in
                    ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks")))
                  for name, module in model.named_modules())
    config = getattr(model, "config", None)
    value = config.to_dict() if hasattr(config, "to_dict") else vars(config) if config is not None else {}
    return hooks, _digest(value)


@contextmanager
def _observation(model):
    """No grad/eval, exact heterogeneous training flags and edit RNG restored."""
    before = _tensor_signature(model)
    control_before = _control_signature(model)
    training = [(module, module.training) for module in model.modules()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    # Forward must not consume edit RNG, including Python/NumPy custom hooks.
    import random
    py_rng, np_rng = random.getstate(), np.random.get_state()
    original_error = None
    try:
        model.eval()
        with torch.no_grad():
            yield
    except BaseException as exc:
        original_error = exc
        raise
    finally:
        for module, flag in training:
            module.training = flag
        random.setstate(py_rng)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        # Preserve the originating exception; a second guard failure cannot hide it.
        mutated = _tensor_signature(model) != before or _control_signature(model) != control_before
        if mutated:
            if original_error is None:
                raise FactualError("FACTUAL_MODEL_MUTATED")
            if hasattr(original_error, "add_note"):
                original_error.add_note("FACTUAL_MODEL_MUTATED observed while preserving original error")


def _cf_sequence(tokenizer, prompt, target):
    prefix, suffix = _ids(tokenizer, prompt), _target(tokenizer, target)
    combined = _ids(tokenizer, prompt + " " + target)
    _require(combined == prefix + suffix, "FACTUAL_CF_TARGET_BOUNDARY_MISMATCH")
    return combined, len(prefix), suffix


def _zsre_sequence(tokenizer, prompt, target):
    prefix, suffix = _ids(tokenizer, prompt), _target(tokenizer, target)
    return prefix + suffix, len(prefix), suffix


def _plan(records, dataset, tokenizer):
    _require(dataset in ("cf", "zsre"), "FACTUAL_UNKNOWN_DATASET")
    cases, queries, signatures, seen = [], [], [], set()
    for record in records:
        occurrence = _occurrence(record)
        _require(occurrence not in seen, "FACTUAL_DUPLICATE_OCCURRENCE")
        seen.add(occurrence)
        groups = _groups(record)
        rw = record["requested_rewrite"]
        new = rw["target_new"]["str"]
        _require(isinstance(new, str) and bool(new), "FACTUAL_EMPTY_TARGET")
        case = dict(case_id=record["case_id"], occurrence_index=occurrence)
        case_index = len(cases)
        case_queries = []
        for kind, prompts in groups.items():
            for prompt_index, prompt in enumerate(prompts):
                if dataset == "cf":
                    _require(isinstance(prompt, str), "FACTUAL_CF_PROMPT_TYPE")
                    true = rw["target_true"]["str"]
                    targets = [("new", new), ("true", true)]
                elif kind == "neighborhood":
                    _require(isinstance(prompt, dict) and isinstance(prompt.get("prompt"), str)
                             and isinstance(prompt.get("target"), str), "FACTUAL_ZSRE_LOC_SCHEMA")
                    targets = [("loc_ans", prompt["target"])]
                    prompt = prompt["prompt"]
                else:
                    _require(isinstance(prompt, str), "FACTUAL_ZSRE_PROMPT_TYPE")
                    targets = [("new", new)]
                for target_kind, target in targets:
                    make = _cf_sequence if dataset == "cf" else _zsre_sequence
                    ids, start, tokens = make(tokenizer, prompt, target)
                    # Keep the entire canonical candidate, as in native CF.
                    query = dict(case_index=case_index, occurrence_index=occurrence,
                                 kind=kind, prompt_index=prompt_index, prompt=prompt,
                                 target_kind=target_kind, target=target,
                                 input_token_ids=ids, target_start=start,
                                 target_token_ids=tokens)
                    queries.append(query)
                    case_queries.append({key: query[key] for key in (
                        "kind", "prompt_index", "prompt", "target_kind", "target",
                        "input_token_ids", "target_start", "target_token_ids")})
        signatures.append(dict(case_id=case["case_id"], occurrence_index=occurrence,
                               queries=case_queries))
        cases.append(case)
    _require(bool(cases), "FACTUAL_EMPTY_COHORT")
    return cases, queries, signatures


def _infer(model, tokenizer, queries, *, batch_size, device, progress=None):
    _require(type(batch_size) is int and batch_size > 0, "FACTUAL_BATCH_SIZE")
    device = _device(model, device)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    _require(type(pad) is int and pad >= 0, "FACTUAL_PADDING_TOKEN_REQUIRED")
    limit, results = _limit(model, tokenizer), []
    work = dict(forward_calls=0, candidate_sequences=len(queries),
                physical_input_tokens=0, target_tokens=0, padded_input_tokens=0)
    for begin in range(0, len(queries), batch_size):
        chunk = queries[begin:begin + batch_size]
        # Full candidates (including the final target token) preserve native
        # CF model inputs. Never let tokenizer truncation hide an oversized row.
        width = max(len(query["input_token_ids"]) for query in chunk)
        _require(limit is None or width <= limit, "FACTUAL_CONTEXT_LENGTH_EXCEEDED")
        ids = torch.full((len(chunk), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for index, query in enumerate(chunk):
            tokens = query["input_token_ids"]
            ids[index, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            mask[index, :len(tokens)] = 1
        # Explicit RIGHT padding is independent of tokenizer.padding_side. For
        # these native causal models, non-padded positions remain 0..length-1.
        with torch.autocast(device_type=device.type, enabled=False):
            output = model(input_ids=ids, attention_mask=mask, use_cache=False)
        logits = output.logits if hasattr(output, "logits") else output[0]
        _require(logits.ndim == 3 and logits.shape[:2] == ids.shape,
                 "FACTUAL_LOGITS_SHAPE")
        work["forward_calls"] += 1
        work["physical_input_tokens"] += sum(len(q["input_token_ids"]) for q in chunk)
        work["padded_input_tokens"] += ids.numel()
        for index, query in enumerate(chunk):
            start, targets = query["target_start"], query["target_token_ids"]
            positions = torch.arange(start - 1, start + len(targets) - 1, device=device)
            selected = logits[index, positions, :].float()
            _require(torch.isfinite(selected).all().item(), "FACTUAL_NONFINITE_LOGITS")
            gold = torch.tensor(targets, dtype=torch.long, device=device)
            _require(gold.max().item() < selected.shape[-1], "FACTUAL_TARGET_OUT_OF_VOCAB")
            nll = -selected.log_softmax(-1).gather(1, gold[:, None]).squeeze(1)
            _require(torch.isfinite(nll).all().item(), "FACTUAL_NONFINITE_NLL")
            predictions = selected.argmax(-1).detach().cpu().tolist()
            values = nll.detach().cpu().tolist()
            # Native CF accumulates/divides an FP32 array. Keep that operation
            # order (and record per-token values) rather than centered/logits sums.
            total = np.float32(0)
            for value in values:
                total = np.float32(total + np.float32(value))
            mean_nll = float(np.float32(total / np.float32(len(values))))
            correct = [prediction == target for prediction, target in zip(predictions, targets)]
            results.append(dict(query, predicted_token_ids=predictions,
                                token_correct=correct, token_count=len(targets),
                                token_correct_count=sum(correct), strict_correct=all(correct),
                                nll_by_token=values, mean_nll=mean_nll))
            work["target_tokens"] += len(targets)
        if progress is not None:
            progress(dict(completed_candidate_sequences=begin + len(chunk),
                          total_candidate_sequences=len(queries), **work))
    return results, work


def _accuracy(rows):
    if not rows:
        return None
    token_count = sum(row["token_count"] for row in rows)
    correct_count = sum(row["token_correct_count"] for row in rows)
    prompt_rates = [row["token_correct_count"] / row["token_count"] for row in rows]
    return dict(prompt_count=len(rows), token_count=token_count,
                token_correct_count=correct_count,
                token_acc_pct=100 * correct_count / token_count,
                prompt_acc_pct=100 * math.fsum(prompt_rates) / len(rows),
                strict_acc_pct=100 * sum(row["strict_correct"] for row in rows) / len(rows))


def _assemble_cf(cases, results):
    groups = {kind: [] for kind in ("rewrite", "paraphrase", "neighborhood")}
    by_prompt = {}
    for row in results:
        group = by_prompt.setdefault((row["case_index"], row["kind"]), {})
        group.setdefault(row["prompt_index"], {})[row["target_kind"]] = row
    for index, case in enumerate(cases):
        for kind in groups:
            pairs = [pair for _, pair in sorted(by_prompt[(index, kind)].items())]
            probs, correct, observations = [], [], []
            for pair in pairs:
                new, true = pair["new"], pair["true"]
                desired = true if kind == "neighborhood" else new
                probs.append(dict(target_new=new["mean_nll"], target_true=true["mean_nll"]))
                correct.append(desired["strict_correct"])
                observations.append(dict(prompt_index=desired["prompt_index"], prompt=desired["prompt"],
                    desired_target="true" if kind == "neighborhood" else "new",
                    target_new=new, target_true=true,
                    margin_true_minus_new=true["mean_nll"] - new["mean_nll"]))
                groups[kind].append(desired)
            case[kind + "_prompts_probs"] = probs
            case[kind + "_prompts_correct"] = correct
            case[kind + "_observations"] = observations
    summary = counterfact(cases)
    return summary, {kind: _accuracy(rows) for kind, rows in groups.items()}


def _reference_lookup(reference, signatures):
    _require(reference.get("schema") == W0_SCHEMA, "FACTUAL_W0_SCHEMA")
    _require(reference.get("identity_sha256") == _digest(reference["identity"]),
             "FACTUAL_W0_IDENTITY_HASH")
    _require(reference.get("payload_sha256") == _digest(
        {key: value for key, value in reference.items() if key != "payload_sha256"}),
        "FACTUAL_W0_PAYLOAD_HASH")
    identity = reference["identity"]
    _require(identity.get("tokenization") == TOKENIZATION, "FACTUAL_W0_TOKENIZATION")
    lookup = {row["occurrence_index"]: row for row in reference["cases"]}
    _require(len(lookup) == len(reference["cases"]), "FACTUAL_W0_DUPLICATE_OCCURRENCE")
    for signature in signatures:
        old = lookup.get(signature["occurrence_index"])
        _require(old is not None and old["case_id"] == signature["case_id"],
                 "FACTUAL_W0_COHORT_IDENTITY")
        expected = [query for query in signature["queries"] if query["kind"] == "neighborhood"]
        _require(old["queries"] == expected, "FACTUAL_W0_TOKEN_PREFIX_IDENTITY")
        _require(len(old["predictions"]) == len(expected)
                 and all(len(values) == len(query["target_token_ids"])
                         and all(type(value) is int and value >= 0 for value in values)
                         for values, query in zip(old["predictions"], expected)),
                 "FACTUAL_W0_PREDICTION_SHAPE")
    return lookup


def _assemble_zsre(cases, results, reference, signatures):
    old = _reference_lookup(reference, signatures) if reference is not None else None
    groups = {kind: [] for kind in ("rewrite", "paraphrase", "neighborhood")}
    by_group = {}
    for row in results:
        by_group.setdefault((row["case_index"], row["kind"]), []).append(row)
    for index, case in enumerate(cases):
        for kind in groups:
            rows = by_group[(index, kind)]
            case[kind + "_prompts_correct"] = [correct for row in rows for correct in row["token_correct"]]
            case[kind + "_observations"] = rows
            groups[kind].extend(rows)
            if kind == "neighborhood":
                if old is None:
                    case["neighborhood_W0_agreement"] = None
                else:
                    prior = old[case["occurrence_index"]]["predictions"]
                    case["neighborhood_W0_agreement"] = [actual == base
                        for row, expected in zip(rows, prior)
                        for actual, base in zip(row["predicted_token_ids"], expected)]
    # Request macro, not an unweighted average of all tokens across requests.
    summary = zsre(cases)
    if old is None:
        summary["W0_prediction_agreement_availability"] = "NOT_MEASURED_W0_REFERENCE_REQUIRED"
    return summary, {kind: _accuracy(rows) for kind, rows in groups.items()}


def _retained_plan(input_signature, dataset):
    """Validate independently retained token queries, without re-tokenization."""
    _require(dataset in ("cf", "zsre") and type(input_signature) is list
             and bool(input_signature), "FACTUAL_RETAINED_INPUT_SIGNATURE_REQUIRED")
    keys = {"kind", "prompt_index", "prompt", "target_kind", "target",
            "input_token_ids", "target_start", "target_token_ids"}
    cases, queries, seen = [], [], set()
    for case_index, signature in enumerate(input_signature):
        _require(type(signature) is dict and set(signature) == {"case_id", "occurrence_index", "queries"}
                 and type(signature["case_id"]) is int
                 and type(signature["occurrence_index"]) is int and signature["occurrence_index"] > 0
                 and signature["occurrence_index"] not in seen,
                 "FACTUAL_RETAINED_CASE_ORDER_OR_SCHEMA")
        seen.add(signature["occurrence_index"])
        original = signature["queries"]
        _require(type(original) is list and bool(original), "FACTUAL_RETAINED_QUERY_PLAN_REQUIRED")
        expected_order = []
        for kind in ("rewrite", "paraphrase", "neighborhood"):
            group = [q for q in original if type(q) is dict and q.get("kind") == kind]
            targets = ("new", "true") if dataset == "cf" else (
                "loc_ans" if kind == "neighborhood" else "new",)
            _require(bool(group) and len(group) % len(targets) == 0,
                     "FACTUAL_RETAINED_PROMPT_GROUP_REQUIRED")
            prompt_count = len(group) // len(targets)
            _require(kind != "rewrite" or prompt_count == 1, "FACTUAL_RETAINED_REWRITE_COUNT")
            for prompt_index in range(prompt_count):
                pair = group[prompt_index * len(targets):(prompt_index + 1) * len(targets)]
                for row, target_kind in zip(pair, targets):
                    _require(type(row) is dict and set(row) == keys and row["kind"] == kind
                             and type(row["prompt_index"]) is int and row["prompt_index"] == prompt_index
                             and row["target_kind"] == target_kind
                             and type(row["prompt"]) is str and bool(row["prompt"])
                             and type(row["target"]) is str and bool(row["target"]),
                             "FACTUAL_RETAINED_QUERY_SCHEMA_OR_ORDER")
                    tokens, gold, start = row["input_token_ids"], row["target_token_ids"], row["target_start"]
                    _require(type(tokens) is list and type(gold) is list and bool(gold)
                             and all(type(token) is int and token >= 0 for token in tokens + gold)
                             and type(start) is int and 1 <= start < len(tokens)
                             and tokens[start:] == gold, "FACTUAL_RETAINED_CAUSAL_TOKEN_PREFIX")
                    expected_order.append(row)
                    queries.append(dict(row, case_index=case_index,
                                        occurrence_index=signature["occurrence_index"]))
                _require(all(row["prompt"] == pair[0]["prompt"]
                             and row["target_start"] == pair[0]["target_start"]
                             and row["input_token_ids"][:row["target_start"]] ==
                                 pair[0]["input_token_ids"][:pair[0]["target_start"]]
                             for row in pair), "FACTUAL_RETAINED_CF_PAIR_PREFIX")
        _require(original == expected_order, "FACTUAL_RETAINED_QUERY_PLAN_ORDER")
        cases.append(dict(case_id=signature["case_id"], occurrence_index=signature["occurrence_index"]))
    return cases, queries


def retained_query_work(input_signature, dataset, *, batch_size=16, per_case=False):
    """Derive exact physical/logical counts from retained unpadded token plans.

    Canonical inference uses contiguous ``batch_size`` chunks; the original CF
    oracle uses one padded call per case (``per_case=True``). Neither is a new
    model observation or a certification that claimed forwards really occurred.
    """
    _require(type(batch_size) is int and batch_size > 0 and type(per_case) is bool,
             "FACTUAL_RETAINED_BATCH_LAYOUT")
    cases, queries = _retained_plan(input_signature, dataset)
    if per_case:
        chunks = [[] for _ in cases]
        for query in queries:
            chunks[query["case_index"]].append(query)
    else:
        chunks = [queries[start:start + batch_size] for start in range(0, len(queries), batch_size)]
    return dict(forward_calls=len(chunks), candidate_sequences=len(queries),
        physical_input_tokens=sum(len(q["input_token_ids"]) for q in queries),
        target_tokens=sum(len(q["target_token_ids"]) for q in queries),
        padded_input_tokens=sum(len(chunk) * max(len(q["input_token_ids"]) for q in chunk)
                                for chunk in chunks))


def validate_retained_observation(endpoint, dataset, *, input_signature,
                                  batch_size=16, w0_reference=None, query_plan=None):
    """CPU audit of completed canonical raw against separately retained inputs.

    Every token/NLL observation and measured-work field is mandatory. A newly
    signed payload cannot substitute a probability array or shared wrong work
    counter for retained query evidence. Caller supplies the independently locked
    signature, not a signature reconstructed only from the raw being checked.
    Actual execution/source/device qualification remains a separate caller gate;
    a CPU fixture/schema audit never becomes actual GPU evidence here.
    """
    cases, queries = _retained_plan(input_signature, dataset)
    if query_plan is not None:
        _require(query_plan == queries, "FACTUAL_RETAINED_FLAT_QUERY_PLAN_MISMATCH")
    _require(type(endpoint) is dict and all(endpoint.get(key) is True for key in
             ("raw_local_only", "model_no_mutation", "RNG_restored")), "FACTUAL_RETAINED_OBSERVATION_GUARDS")
    identity = endpoint.get("identity")
    _require(type(identity) is dict and identity.get("schema") == SCHEMA
             and identity.get("dataset") == dataset and identity.get("tokenization") == TOKENIZATION
             and identity.get("cohort_sha256") == _digest(input_signature)
             and identity.get("ordered_occurrences") == [row["occurrence_index"] for row in cases]
             and identity.get("padding") == "RIGHT_EXPLICIT_ATTENTION_MASK"
             and identity.get("use_cache") is False
             and endpoint.get("identity_sha256") == _digest(identity),
             "FACTUAL_RETAINED_ENDPOINT_INPUT_IDENTITY")
    actual_cases = endpoint.get("cases")
    _require(type(actual_cases) is list and len(actual_cases) == len(cases)
             and all(type(row) is dict and row.get("case_id") == expected["case_id"]
                     and row.get("occurrence_index") == expected["occurrence_index"]
                     for row, expected in zip(actual_cases, cases)), "FACTUAL_RETAINED_ORDERED_CASES")
    reference = w0_reference
    if reference is not None:
        _require(dataset == "zsre", "FACTUAL_W0_REFERENCE_ZSRE_ONLY")
        from .w0_reference import PortableZSREReference, validate_zsre_reference
        if isinstance(reference, PortableZSREReference):
            reference, binding = validate_zsre_reference(reference,
                consumer_external_identity=identity.get("external_identity", {}), signatures=input_signature)
            _require(endpoint.get("W0_reference_binding") == binding
                     and identity.get("W0_reference_binding_sha256") == binding["binding_sha256"],
                     "FACTUAL_RETAINED_PORTABLE_REFERENCE_BINDING")
        else:
            _require(reference["identity"].get("external_identity", {}) == identity.get("external_identity", {}),
                     "FACTUAL_W0_EXTERNAL_IDENTITY")
        _reference_lookup(reference, input_signature)
        is_self = (identity.get("W0_reference_sha256") is None and
            {key: value for key, value in endpoint.items() if key != "work"} ==
            {key: value for key, value in reference.get("evaluation", {}).items() if key != "work"})
        _require(identity.get("W0_reference_sha256") == (None if is_self else reference["identity_sha256"]),
                 "FACTUAL_RETAINED_REFERENCE_IDENTITY")
    else:
        _require(identity.get("W0_reference_sha256") is None, "FACTUAL_RETAINED_REFERENCE_REQUIRED")
    results, by_case = [], [[] for _ in cases]
    for query in queries:
        by_case[query["case_index"]].append(query)
    for case_index, (actual, signature) in enumerate(zip(actual_cases, input_signature)):
        planned = by_case[case_index]
        for kind in ("rewrite", "paraphrase", "neighborhood"):
            selected = [q for q in planned if q["kind"] == kind]
            observations = actual.get(kind + "_observations")
            width = 2 if dataset == "cf" else 1
            _require(type(observations) is list and len(observations) == len(selected) // width,
                     "FACTUAL_RETAINED_CANONICAL_OBSERVATIONS_REQUIRED")
            rows = []
            for observation in observations:
                _require(type(observation) is dict, "FACTUAL_RETAINED_OBSERVATION_SCHEMA")
                rows.extend((observation.get("target_new"), observation.get("target_true"))
                            if dataset == "cf" else (observation,))
            for row, expected in zip(rows, selected):
                _require(type(row) is dict and all(key in row for key in expected)
                         and _digest({key: row[key] for key in expected}) == _digest(expected),
                         "FACTUAL_RETAINED_RAW_QUERY_TOKEN_IDENTITY")
                gold = expected["target_token_ids"]
                predicted, nll = row.get("predicted_token_ids"), row.get("nll_by_token")
                _require(type(predicted) is list and len(predicted) == len(gold)
                         and all(type(token) is int and token >= 0 for token in predicted),
                         "FACTUAL_RETAINED_PREDICTION_TOKENS")
                correct = [a == b for a, b in zip(predicted, gold)]
                _require(type(row.get("token_correct")) is list
                         and all(type(bit) is bool for bit in row["token_correct"])
                         and row["token_correct"] == correct
                         and type(row.get("token_count")) is int and row["token_count"] == len(gold)
                         and type(row.get("token_correct_count")) is int and row["token_correct_count"] == sum(correct)
                         and type(row.get("strict_correct")) is bool and row["strict_correct"] == all(correct),
                         "FACTUAL_RETAINED_CORRECTNESS_NUMERATOR_DENOMINATOR")
                _require(type(nll) is list and len(nll) == len(gold)
                         and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in nll),
                         "FACTUAL_RETAINED_PER_TOKEN_NLL_REQUIRED")
                total = np.float32(0)
                for value in nll:
                    total = np.float32(total + np.float32(value))
                mean_nll = float(np.float32(total / np.float32(len(nll))))
                _require(math.isfinite(mean_nll) and type(row.get("mean_nll")) in (int, float)
                         and row["mean_nll"] == mean_nll, "FACTUAL_RETAINED_NATIVE_FP32_NLL_MEAN")
                results.append(deepcopy(row))
    rebuilt = deepcopy(cases)
    if dataset == "cf":
        summary, accuracy = _assemble_cf(rebuilt, results)
    else:
        summary, accuracy = _assemble_zsre(rebuilt, results, reference, input_signature)
    _require(_digest(actual_cases) == _digest(rebuilt), "FACTUAL_RETAINED_RAW_PROBABILITY_STRICT_OR_AGREEMENT")
    _require(_digest(endpoint.get("summary")) == _digest(summary)
             and _digest(endpoint.get("accuracy")) == _digest(accuracy),
             "FACTUAL_RETAINED_REQUEST_MACRO_ACCURACY_REDUCTION")
    expected_work = retained_query_work(input_signature, dataset, batch_size=batch_size)
    work = endpoint.get("work")
    _require(type(work) is dict and all(type(work.get(key)) is int and work[key] == value
             for key, value in expected_work.items()), "FACTUAL_RETAINED_QUERY_DERIVED_WORK_REQUIRED")
    _require(type(work.get("seconds")) in (int, float) and math.isfinite(work["seconds"])
             and work["seconds"] > 0, "FACTUAL_RETAINED_WORK_ELAPSED_FINITE")
    return dict(schema="official-retained-factual-CPU-audit-v1", dataset=dataset,
        requests=len(cases), **expected_work, endpoint_identity_sha256=endpoint["identity_sha256"],
        input_signature_sha256=_digest(input_signature), observed_work_validated=True,
        independent_raw_reduction=True, actual_model_forward_calls=0, GPU_parity_claim=False)


def evaluate(model, tokenizer, records, dataset, *, w0_reference=None,
             batch_size=16, device=None, progress=None, identity=None):
    """Score a stream/cohort without edits. Return local raw plus scalar summary.

    ``identity`` should include the runner's model/revision/tokenizer/runtime
    locks. Plain zsRE references must match the W0 external identity. An explicit
    reviewed PortableZSREReference preserves producer raw and separately binds
    the actual consumer identity before any forward; no implicit rebinding.
    For W0 use ``build_zsre_w0_reference`` and its already-observed ``evaluation``
    to avoid a redundant forward. A missing W0 reference is explicit/omitted.
    """
    records = list(records)
    cases, queries, signatures = _plan(records, dataset, tokenizer)
    external = json.loads(json.dumps(identity or {}, sort_keys=True, allow_nan=False))
    reference_binding = None
    if w0_reference is not None:
        _require(dataset == "zsre", "FACTUAL_W0_REFERENCE_ZSRE_ONLY")
        from .w0_reference import PortableZSREReference, validate_zsre_reference
        if isinstance(w0_reference, PortableZSREReference):
            w0_reference, reference_binding = validate_zsre_reference(w0_reference,
                consumer_external_identity=external, signatures=signatures)
        else:
            # Classic exact-identity reference semantics remain unchanged.
            _require(w0_reference["identity"].get("external_identity", {}) == external,
                     "FACTUAL_W0_EXTERNAL_IDENTITY")
        # Check before the expensive model measurement, not only after it.
        _reference_lookup(w0_reference, signatures)
    started = time.monotonic()
    with _observation(model):
        results, work = _infer(model, tokenizer, queries, batch_size=batch_size,
                               device=device, progress=progress)
    if dataset == "cf":
        summary, accuracy = _assemble_cf(cases, results)
    else:
        summary, accuracy = _assemble_zsre(cases, results, w0_reference, signatures)
    work["seconds"] = time.monotonic() - started
    endpoint_identity = dict(schema=SCHEMA, dataset=dataset, tokenization=TOKENIZATION,
        cohort_sha256=_digest(signatures), ordered_occurrences=[_occurrence(row) for row in records],
        external_identity=external, padding="RIGHT_EXPLICIT_ATTENTION_MASK", use_cache=False,
        W0_reference_sha256=w0_reference["identity_sha256"] if w0_reference is not None else None)
    if reference_binding is not None:
        endpoint_identity["W0_reference_binding_sha256"] = reference_binding["binding_sha256"]
    value = dict(summary=summary, accuracy=accuracy, cases=cases, identity=endpoint_identity,
                identity_sha256=_digest(endpoint_identity), work=work,
                raw_local_only=True, model_no_mutation=True, RNG_restored=True)
    if reference_binding is not None:
        value["W0_reference_binding"] = reference_binding
    return value


def build_zsre_w0_reference(model, tokenizer, records, *, identity=None,
                            batch_size=16, device=None, progress=None):
    """One actual W0 observation; retain all loc_ans-prefix predictions locally.

    The returned ``evaluation`` includes W0 efficacy/generalization/loc_ans and
    true W0 self-agreement, with the actual measurement work. No extra forward.
    Store the complete object in the runner's immutable W0 asset namespace.
    """
    records = list(records)
    endpoint = evaluate(model, tokenizer, records, "zsre", identity=identity,
                        batch_size=batch_size, device=device, progress=progress)
    _, _, signatures = _plan(records, "zsre", tokenizer)
    cases = []
    for row, signature in zip(endpoint["cases"], signatures):
        queries = [query for query in signature["queries"] if query["kind"] == "neighborhood"]
        predictions = [observation["predicted_token_ids"]
                       for observation in row["neighborhood_observations"]]
        cases.append(dict(case_id=row["case_id"], occurrence_index=row["occurrence_index"],
                          queries=queries, predictions=predictions))
        row["neighborhood_W0_agreement"] = [True for group in predictions for _ in group]
    endpoint["summary"] = zsre(endpoint["cases"])
    reference_identity = dict(schema=W0_SCHEMA, tokenization=TOKENIZATION,
        external_identity=endpoint["identity"]["external_identity"],
        ordered_occurrences=[row["occurrence_index"] for row in cases],
        cohort_sha256=_digest(signatures), state="W0_COLD_BASE_MODEL")
    value = dict(schema=W0_SCHEMA, identity=reference_identity,
                 identity_sha256=_digest(reference_identity), cases=cases,
                 evaluation=endpoint, raw_local_only=True)
    value["payload_sha256"] = _digest(value)
    return value


def evaluate_counterfact(model, tokenizer, records, **kwargs):
    return evaluate(model, tokenizer, records, "cf", **kwargs)


def evaluate_zsre(model, tokenizer, records, **kwargs):
    return evaluate(model, tokenizer, records, "zsre", **kwargs)
