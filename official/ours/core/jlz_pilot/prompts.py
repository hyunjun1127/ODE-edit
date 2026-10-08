"""Native AlphaEdit text preparation and KL(current || entry) for JLZ.

Only the native prompt/target/lookup and smooth loss are represented here.
JLZ's block norm penalty and clamp belong to the proximal optimizer. Context
groups affect native key averages; rewrite losses average all contexts equally.
"""
from __future__ import annotations

import copy
import hashlib
import json

import torch
import torch.nn.functional as F


def subject_last(tokenizer, prompt: str, subject: str) -> int:
    """Match native repr_tools.get_words_idxs_in_templates(..., 'last')."""
    if prompt.count("{}") != 1:
        raise ValueError("Native subject lookup requires exactly one {} slot")
    prefix = prompt.split("{}", 1)[0]
    position = len(tokenizer.encode(prefix + subject)) - 1
    if position < 0:
        raise ValueError("Subject tokenization is empty")
    return position


def _to_device(tokens, device):
    return {name: value.to(device) for name, value in tokens.items()}


def prepare(tokenizer, requests, contexts, device):
    """Prepare native rewrite/KL rows and separate target-free key rows.

    Rows are request-major: all rewrite contexts, then one ``{} is a`` KL
    prompt. The tokenizer must use right padding. Requests are copied before
    native target leading-space normalization. ``contexts`` is a sequence of
    nonempty context groups, conventionally [["{}"], [five templates]].

    ``key_context_weights`` stores mean-of-group-means coefficients, whereas
    each rewrite row receives 1/n_rw in the smooth objective. ``targets`` has
    all rows (KL rows are ignored); ``rewrite_targets`` is native-shaped.
    """
    if tokenizer.padding_side != "right":
        raise ValueError("Native pilot requires right padding")
    if not requests or not contexts or any(not group for group in contexts):
        raise ValueError("Requests and every context group must be nonempty")
    if any(isinstance(group, str) for group in contexts):
        raise ValueError("Contexts must preserve native nested context groups")
    flat_contexts = [context for group in contexts for context in group]
    if any(context.count("{}") != 1 for context in flat_contexts):
        raise ValueError("Each context must have exactly one prompt slot")
    if flat_contexts[0] != "{}":
        raise ValueError("Native canonical anchor requires context zero '{}'")
    normalized = copy.deepcopy(list(requests))
    formatted, specs, row_request, row_kind, lookup = [], [], [], [], []
    key_formatted, key_lookup, key_request, key_context_weights = [], [], [], []
    context_weights = [1.0 / (len(contexts) * len(group))
                       for group in contexts for _ in group]
    n_rw = len(flat_contexts)
    for b, request in enumerate(normalized):
        target_text = request["target_new"]["str"]
        if not target_text:
            raise ValueError("Native target must be nonempty")
        if not target_text.startswith(" "):
            target_text = " " + target_text
            request["target_new"]["str"] = target_text
        target = tokenizer(target_text, return_tensors="pt")["input_ids"][0]
        if target.numel() and int(target[0]) in (tokenizer.bos_token_id, tokenizer.unk_token_id):
            target = target[1:]
        if not target.numel():
            raise ValueError("Native target has no tokens after BOS/UNK removal")
        base_prompts = [context.format(request["prompt"]) for context in flat_contexts]
        rewrite = [prompt + tokenizer.decode(target[:-1]) for prompt in base_prompts]
        prompts = rewrite + ["{} is a"]
        positions = [subject_last(tokenizer, prompt, request["subject"]) for prompt in prompts]
        offset = len(formatted)
        specs.append(dict(target=target.to(device), lookup=positions, n_rw=n_rw,
                          offset=offset, case_id=request.get("case_id")))
        formatted.extend(prompt.format(request["subject"]) for prompt in prompts)
        row_request.extend([b] * len(prompts))
        row_kind.extend(["rewrite"] * n_rw + ["kl"])
        lookup.extend(positions)
        key_formatted.extend(prompt.format(request["subject"]) for prompt in base_prompts)
        key_lookup.extend(subject_last(tokenizer, prompt, request["subject"]) for prompt in base_prompts)
        key_request.extend([b] * n_rw)
        key_context_weights.extend(context_weights)
    tokens = _to_device(tokenizer(formatted, return_tensors="pt", padding=True), device)
    key_tokens = _to_device(tokenizer(key_formatted, return_tensors="pt", padding=True), device)
    targets = torch.full_like(tokens["input_ids"], -100)
    rw_rows, kl_rows, kl_cols = [], [], []
    for spec in specs:
        for j in range(n_rw):
            row = spec["offset"] + j
            length = int(tokens["attention_mask"][row].sum())
            if length < spec["target"].numel():
                raise ValueError("Native target position underflow")
            targets[row, length-spec["target"].numel():length] = spec["target"]
            rw_rows.append(row)
        kl_rows.append(spec["offset"] + n_rw)
        kl_cols.append(spec["lookup"][-1])
    for row, position in enumerate(lookup):
        if position >= int(tokens["attention_mask"][row].sum()):
            raise ValueError("Native subject lookup extends past prompt")
    for row, position in enumerate(key_lookup):
        if position >= int(key_tokens["attention_mask"][row].sum()):
            raise ValueError("Native key subject lookup extends past prompt")
    identity = hashlib.sha256(json.dumps(dict(
        ids=tokens["input_ids"].cpu().tolist(),
        mask=tokens["attention_mask"].cpu().tolist(),
        lookup=lookup, targets=[s["target"].cpu().tolist() for s in specs],
        key_ids=key_tokens["input_ids"].cpu().tolist(),
        key_lookup=key_lookup, key_context_weights=key_context_weights,
    ), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    group_lens = [len(group) for group in contexts]
    group_bounds = [0]
    for length in group_lens:
        group_bounds.append(group_bounds[-1] + length)
    return dict(tokens=tokens, specs=specs, requests=normalized, targets=targets,
                rewrite_targets=targets[rw_rows], n_rw=n_rw, n_requests=len(specs),
                context_group_lens=group_lens,
                context_group_slices=list(zip(group_bounds[:-1], group_bounds[1:])),
                rw_rows=rw_rows, kl_rows=kl_rows, kl_cols=kl_cols,
                row_request=row_request, row_kind=row_kind, lookup=lookup,
                intervention_rows=list(range(len(formatted))), intervention_cols=lookup,
                canonical_rows=[s["offset"] for s in specs],
                key_tokens=key_tokens, key_lookup=key_lookup, key_request=key_request,
                key_context_weights=key_context_weights,
                canonical_key_rows=[b*n_rw for b in range(len(specs))],
                prompts=formatted, key_prompts=key_formatted, identity=identity)


def entry_teacher(logits, spec):
    """Frozen full-vocabulary log distributions at native KL subject positions."""
    return logits[spec["kl_rows"], spec["kl_cols"]].log_softmax(-1).detach().clone()


def native_loss(logits, spec, teacher_logprobs, active_mask, rows=None, kl_factor=.0625):
    """Return (smooth loss, per-request NLL, per-request unweighted native KL).

    ``rows`` maps local logit rows to the global prepared rows. Partial-row
    calls return partial per-request contributions with global normalization;
    summing loss/metrics/gradients across a disjoint row partition equals the
    full call (up to floating-point reduction order). The scalar sums active
    requests, matching the JLZ design; it does not average across requests.

    ``teacher_logprobs=None`` is an NLL-only observation mode: KL values are
    zero placeholders and the scalar contains NLL only. Joint optimization
    must always pass the frozen entry teacher; this mode does not measure KL.
    """
    if logits.ndim != 3:
        raise ValueError("Expected logits [local rows, padded sequence, vocabulary]")
    rows = list(range(len(spec["row_request"]))) if rows is None else list(rows)
    if len(rows) != logits.shape[0] or len(set(rows)) != len(rows):
        raise ValueError("Logit rows must have a unique global row mapping")
    if any(row < 0 or row >= len(spec["row_request"]) for row in rows):
        raise ValueError("Global row mapping out of range")
    n_requests = len(spec["specs"])
    active = torch.as_tensor(active_mask, device=logits.device, dtype=torch.bool)
    if tuple(active.shape) != (n_requests,):
        raise ValueError("Active mask must have one entry per request")
    if teacher_logprobs is not None and tuple(teacher_logprobs.shape) != (n_requests, logits.shape[-1]):
        raise ValueError("Teacher must have one full-vocabulary KL row per request")
    # A differentiable scalar zero avoids scanning/retaining the entire head.
    zero = logits.reshape(-1)[0] * 0.0
    nll_parts, kl_parts = [[] for _ in range(n_requests)], [[] for _ in range(n_requests)]
    for local, row in enumerate(rows):
        request = spec["row_request"][row]
        if spec["row_kind"][row] == "rewrite":
            target_row = spec["targets"][row].to(logits.device)
            selected = target_row != -100
            logp = logits[local, selected].log_softmax(-1)
            gathered = logp.gather(1, target_row[selected, None]).squeeze(1)
            value = -gathered.sum() / spec["specs"][request]["target"].numel() / spec["n_rw"]
            nll_parts[request].append(value)
        elif teacher_logprobs is not None:
            current = logits[local, spec["lookup"][row]].log_softmax(-1).unsqueeze(0)
            entry = teacher_logprobs[request:request+1].detach().to(logits.device)
            kl_parts[request].append(F.kl_div(entry, current, log_target=True, reduction="batchmean"))
    nll = torch.stack([torch.stack(parts).sum() if parts else zero for parts in nll_parts])
    kl = torch.stack([torch.stack(parts).sum() if parts else zero for parts in kl_parts])
    loss = (nll[active] + kl_factor * kl[active]).sum()
    if not torch.isfinite(loss) or not torch.isfinite(nll).all() or not torch.isfinite(kl).all():
        raise FloatingPointError("Nonfinite native pilot smooth loss")
    return loss, nll, kl
