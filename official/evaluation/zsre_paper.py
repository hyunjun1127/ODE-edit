"""Eval-only public zsRE decode/re-tokenize queries, explicit model branches.

Input is the immutable official stream (one full loc_ans per neighborhood).
This is NOT the old exact-token-prefix evaluator and never re-labels its raw.
CPU query parity does not establish numerical pretrained-model output parity.
"""
from copy import deepcopy
import time

import torch

from .factual import (_device, _digest, _ids, _limit, _observation, _occurrence,
                      _require)
from .reduce import zsre

SCHEMA = "official-zsre-public-query-eval-only-v1"
FAMILIES = {"llama3": "llama", "gptj": "gptj", "qwen25": "qwen2"}
GROUPS = ("rewrite", "paraphrase", "neighborhood")


def _native_tokens(tokenizer, text, *, llama):
    ids = _ids(tokenizer, text)
    if llama:
        # Upstream strips [:1] from answer and scores retokenized target[:,1].
        # Validate the native BOS contract rather than blindly slicing Qwen.
        bos = getattr(tokenizer, "bos_token_id", None)
        _require(type(bos) is int and len(ids) >= 2 and ids[0] == bos,
                 "ZSRE_PAPER_LLAMA_BOS_CONTRACT")
        return ids[1:]
    return ids


def build_queries(tokenizer, records, *, model_family):
    """Return local-only native strings/IDs/targets and immutable query hashes.

The neighborhood records must be the unexpanded official stream; callers must
bind its SHA, not pass a public loader's already expanded token rows. Prefixes
include the loader's default special tokens, including Llama BOS in loc_ans.
No truncation, target normalization, deduplication or W0 predictions are used.
"""
    _require(model_family in FAMILIES, "ZSRE_PAPER_MODEL_FAMILY")
    _require(getattr(tokenizer, "padding_side", None) == "right",
             "ZSRE_PAPER_RIGHT_PADDING_REQUIRED")
    _require(type(getattr(tokenizer, "pad_token_id", None)) is int,
             "ZSRE_PAPER_PAD_ID_REQUIRED")
    records = list(records)
    _require(bool(records), "ZSRE_PAPER_EMPTY_STREAM")
    cases, queries, seen = [], [], set()
    llama = model_family == "llama3"
    for rec in records:
        occurrence = _occurrence(rec)
        _require(occurrence not in seen, "ZSRE_PAPER_DUPLICATE_OCCURRENCE")
        seen.add(occurrence)
        rw = rec["requested_rewrite"]
        _require(rw["prompt"].count("{}") == 1, "ZSRE_PAPER_REWRITE_TEMPLATE")
        neighbors = rec["neighborhood_prompts"]
        _require(len(neighbors) == 1 and neighbors[0]["prompt"].endswith("?")
                 and "nq question: " in neighbors[0]["prompt"],
                 "ZSRE_PAPER_UNEXPANDED_NEIGHBOR_REQUIRED")
        _require(bool(rec["paraphrase_prompts"]), "ZSRE_PAPER_EMPTY_PARAPHRASE")
        case_index = len(cases)
        cases.append(dict(case_id=rec["case_id"], occurrence_index=occurrence))

        def append(group, prompt_index, token_index, prompt, target):
            inputs = _ids(tokenizer, prompt)
            scored = _native_tokens(tokenizer, target, llama=llama)[0]
            queries.append(dict(case_index=case_index, case_id=rec["case_id"],
                                occurrence_index=occurrence, group=group,
                                prompt_index=prompt_index, token_index=token_index,
                                prompt=prompt, target_text=target,
                                input_ids=inputs, target_token_id=scored))

        target = _native_tokens(tokenizer, " " + rw["target_new"]["str"], llama=llama)
        for group, prompts in (("rewrite", [rw["prompt"].format(rw["subject"])]),
                               ("paraphrase", rec["paraphrase_prompts"])):
            for p, prompt in enumerate(prompts):
                for i, token in enumerate(target):
                    text = prompt + (" " if llama and i > 0 else "") + tokenizer.decode(target[:i])
                    append(group, p, i, text, tokenizer.decode(token))
        # This is the public dataset loader, which deliberately does NOT remove
        # BOS here. Public evaluator then retokenizes each decoded target.
        neighbor = neighbors[0]
        loc = _ids(tokenizer, " " + neighbor["target"])
        for i, token in enumerate(loc):
            prompt = (neighbor["prompt"] + tokenizer.decode(loc[:i])).format(rw)
            append("neighborhood", 0, i, prompt, tokenizer.decode(token))
    counts = {g: sum(q["group"] == g for q in queries) for g in GROUPS}
    return dict(schema=SCHEMA, model_family=model_family, requests=len(cases),
                cases=cases, queries=queries, token_denominators=counts,
                stream_sha256=_digest(records), query_sha256=_digest(queries),
                raw_local_only=True)


def evaluate(model, tokenizer, records, *, model_family, batch_size=16,
             device=None, identity=None, progress=None):
    """Evaluate one restored state; caller enforces W20/2K checkpoint identity.

No W0, editing, generation or reference model forwards. Summary is request
macro in percent. Cases include actual predictions/targets; keep them local.
Immutable file SHA checks before/after restore/evaluation remain caller-owned.
"""
    _require(type(batch_size) is int and batch_size > 0, "ZSRE_PAPER_BATCH_SIZE")
    _require(model_family in FAMILIES and
             getattr(model.config, "model_type", None) == FAMILIES[model_family],
             "ZSRE_PAPER_MODEL_TYPE_MISMATCH")
    plan = build_queries(tokenizer, records, model_family=model_family)
    queries = plan["queries"]
    limit = _limit(model, tokenizer)
    _require(limit is None or all(len(q["input_ids"]) <= limit for q in queries),
             "ZSRE_PAPER_CONTEXT_OVERFLOW_NO_TRUNCATION")
    where = _device(model, device)
    cases = deepcopy(plan["cases"])
    for case in cases:
        for group in GROUPS:
            case[group + "_prompts_correct"] = []
            case[group + "_observations"] = []
    start, calls = time.monotonic(), 0
    with _observation(model):
        for offset in range(0, len(queries), batch_size):
            batch = queries[offset:offset + batch_size]
            width = max(len(q["input_ids"]) for q in batch)
            ids = torch.full((len(batch), width), tokenizer.pad_token_id,
                             dtype=torch.long, device=where)
            mask = torch.zeros_like(ids)
            for i, query in enumerate(batch):
                length = len(query["input_ids"])
                ids[i, :length] = torch.tensor(query["input_ids"], device=where)
                mask[i, :length] = 1
            # Native right-padding full-prefix next-token inference. No cache
            # across queries and no inherited edit use_cache/config mutation.
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            selected = logits[torch.arange(len(batch), device=where), mask.sum(1) - 1]
            _require(bool(torch.isfinite(selected).all()), "ZSRE_PAPER_NONFINITE_LOGITS")
            predictions = selected.argmax(-1).detach().cpu().tolist()
            calls += 1
            for query, prediction in zip(batch, predictions):
                group, case = query["group"], cases[query["case_index"]]
                correct = prediction == query["target_token_id"]
                case[group + "_prompts_correct"].append(correct)
                case[group + "_observations"].append(dict(
                    prompt_index=query["prompt_index"], token_index=query["token_index"],
                    predicted_token_id=prediction, target_token_id=query["target_token_id"],
                    correct=correct))
            if progress:
                progress(dict(completed_queries=min(offset + len(batch), len(queries)),
                              total_queries=len(queries), physical_forward_calls=calls,
                              elapsed_seconds=time.monotonic() - start))
            del logits, selected
    summary = zsre(cases)
    return dict(schema=SCHEMA, model_family=model_family, summary=summary, cases=cases,
                identity=deepcopy(identity), identity_sha256=_digest(identity),
                stream_sha256=plan["stream_sha256"], query_sha256=plan["query_sha256"],
                token_denominators=plan["token_denominators"],
                work=dict(physical_forward_calls=calls, queries=len(queries),
                          elapsed_seconds=time.monotonic() - start),
                raw_local_only=True, model_no_mutation=True, RNG_restored=True,
                numerical_public_output_parity="NOT_SEPARATELY_MEASURED")
