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
    summary = dict(requests=len(cases))
    keys = (("rewrite_prompts_correct", "Efficacy"),
            ("paraphrase_prompts_correct", "Generalization"),
            ("neighborhood_prompts_correct", "Specificity_loc_ans"))
    for key, label in keys:
        rates = [math.fsum(float(x) for x in row[key]) / len(row[key]) for row in cases]
        summary[label] = 100 * math.fsum(rates) / len(rates)
    if old is not None:
        # Use the canonical reducer after exact W0 input-query validation.
        summary = zsre(cases)
    else:
        summary["Specificity_availability"] = "NOT_MEASURED_W0_REFERENCE_REQUIRED"
    return summary, {kind: _accuracy(rows) for kind, rows in groups.items()}


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
