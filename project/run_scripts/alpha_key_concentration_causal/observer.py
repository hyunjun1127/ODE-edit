"""Post-seal, observation-only AlphaEdit canonical and fixed-panel metrics.

The historical evaluator (helper 1075540) is imported with exact source hashes.
Its tokenization, manual LEFT tensor padding, default position IDs, FP32 NLL
reduction and microbatch shape are not reimplemented. A call-scoped proxy only
reads returned logits to retain additional full-vocabulary diagnostics. This is
different from the native writer's right-padding/add_bos_token=False route.

Callers own endpoint sealing and W/M/RNG restoration/nonmutation verification.
This module neither writes weights/history nor claims to establish that guard.
Returned ``raw_local_only`` contains prompts and must never be committed to Git.
``compact`` contains hashes/scalars/token IDs, not prompt/target strings.
No function performs generation, target fitting, basis fitting or selection.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Mapping, Sequence


NATIVE_HELPER_COMMIT = "1075540b45c29269e690ac63aae44758d8d63174"
NATIVE_SOURCE_SHA256 = {
    "evaluator.py": "4f5af6dbf8854c79aedc5e36134fddfcb2995b60f05d270604f8ea735672d93e",
    "contracts.py": "1c4aab5f974b4f02531693b733f546ad981b98b26d9260f900f6d074a061a88e",
}
_PACKAGE = "project.run_scripts.alphaedit_strength_neutral_barrier"


class ObserverBoundary(RuntimeError):
    """Nonfinite, identity, denominator or source mismatch: never a fallback."""


def digest(value: Any) -> str:
    # Exact historical canonical digest, including non-ASCII prompt escaping.
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bind_native_evaluator(source_root: str | Path | None = None) -> Any:
    """Bind byte-identical historical evaluator without eager writer imports.

    ``source_root`` is a repository/extracted source root with ``project/``.
    An already imported kernel from other bytes fails closed; no monkey patch.
    """
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[3]
    package_path = root / "project/run_scripts/alphaedit_strength_neutral_barrier"
    for name, expected in NATIVE_SOURCE_SHA256.items():
        path = package_path / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ObserverBoundary(f"NATIVE_EVALUATOR_SOURCE_MISMATCH: {path}")
    if _PACKAGE not in sys.modules:
        # Historical package __init__ imports an unrelated writer. Do not run it.
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(package_path)]
        package.__package__ = _PACKAGE
        sys.modules[_PACKAGE] = package
    for short in ("contracts", "evaluator"):
        module = importlib.import_module(f"{_PACKAGE}.{short}")
        path = Path(module.__file__)
        if hashlib.sha256(path.read_bytes()).hexdigest() != NATIVE_SOURCE_SHA256[f"{short}.py"]:
            raise ObserverBoundary(f"NATIVE_EVALUATOR_IMPORTED_BYTES_MISMATCH: {path}")
    return sys.modules[f"{_PACKAGE}.evaluator"]


def _check_call(model: Any, tok: Any, endpoint_id: str, postseal: bool,
                microbatch_size: int) -> None:
    import torch
    if not postseal or not endpoint_id:
        raise ObserverBoundary("OBSERVER_REQUIRES_POSTSEAL_ENDPOINT_ID")
    if microbatch_size <= 0 or getattr(tok, "padding_side", None) != "right":
        raise ObserverBoundary("NATIVE_EVALUATOR_TOKENIZER_OR_MICROBATCH")
    if getattr(model, "training", True):
        raise ObserverBoundary("OBSERVER_MODEL_NOT_EVAL")
    if torch.is_autocast_enabled():
        raise ObserverBoundary("OBSERVER_AUTOCAST_ENABLED")
    # Actual model FP32/eager/TF32 identity is sealed by the caller/loader.
    # This observer checks returned logits, not a redundant full model scan.


def _logit_diagnostics(logits: Any, target_ids: Sequence[int], *, topk: int = 0) -> dict:
    """Full-vocab argmax/ties/margins; no truncated-vocabulary probabilities."""
    import torch
    if logits.dtype != torch.float32 or logits.ndim != 2:
        raise ObserverBoundary("OBSERVER_LOGITS_NOT_FP32_TV")
    if logits.shape[0] != len(target_ids) or logits.shape[1] < 2:
        raise ObserverBoundary("OBSERVER_TOKEN_VOCAB_SHAPE")
    if not torch.isfinite(logits).all().item():
        raise ObserverBoundary("OBSERVER_NONFINITE_FULL_VOCAB_LOGITS")
    target = torch.tensor(target_ids, device=logits.device, dtype=torch.long)
    if target.numel() == 0 or int(target.min()) < 0 or int(target.max()) >= logits.shape[1]:
        raise ObserverBoundary("OBSERVER_TARGET_ID_OUT_OF_VOCAB")
    target_logits = logits.gather(1, target[:, None]).squeeze(1)
    top2 = torch.topk(logits, k=2, dim=-1)
    competitors = torch.where(top2.indices[:, 0] == target, top2.values[:, 1], top2.values[:, 0])
    margins = target_logits - competitors
    maxima = logits.max(dim=-1).values
    predictions = logits.argmax(dim=-1)  # lowest token ID for an exact tie
    result = {
        "vocab_size": int(logits.shape[1]),
        "true_token_ids": [int(x) for x in target_ids],
        "true_token_logits": target_logits.detach().cpu().tolist(),
        "logsumexp": torch.logsumexp(logits, dim=-1).detach().cpu().tolist(),
        "target_minus_best_other_margin": margins.detach().cpu().tolist(),
        "argmax_token_ids": predictions.detach().cpu().tolist(),
        "argmax_tie_counts": (logits == maxima[:, None]).sum(dim=-1).cpu().tolist(),
        "target_ties_best_other": (margins == 0).detach().cpu().tolist(),
        "all_unique_argmax_correct": bool((margins > 0).all().item()),
        "full_vocab_checked": True,
        "argmax_tie_policy": "lowest_token_id; pairwise NLL ties are failure",
    }
    if topk:
        count = min(int(topk), int(logits.shape[1]))
        # Stable sort supplies deterministic low-ID tie ordering including rank32.
        order = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :count]
        result["top_token_ids"] = order.detach().cpu().tolist()
        result["top_token_logits"] = logits.gather(1, order).detach().cpu().tolist()
        result["topk_requested"] = int(topk)
    return result


class _LogitObserver:
    """Read-only model proxy used by the original evaluator; no extra forward."""
    def __init__(self, model: Any, tok: Any, pairs: Sequence[Any], kernel: Any,
                 topk: int) -> None:
        self.model, self.tok, self.pairs, self.kernel, self.topk = model, tok, pairs, kernel, topk
        self.cursor = 0
        self.rows: list[dict] = []
        self.counter = {"forward_calls": 0, "extra_forward_calls": 0,
                        "input_tokens_including_padding": 0, "valid_input_tokens": 0,
                        "scored_target_tokens": 0}

    def __call__(self, **kwargs: Any) -> Any:
        import torch
        if kwargs.get("use_cache") is not False or "position_ids" in kwargs:
            raise ObserverBoundary("NATIVE_FORWARD_KWARGS_CHANGED")
        output = self.model(**kwargs)
        logits = output.logits
        if logits.dtype != torch.float32:
            raise ObserverBoundary("NATIVE_MODEL_LOGITS_NOT_FP32")
        batch, length, _ = logits.shape
        group = self.pairs[self.cursor:self.cursor + batch]
        if len(group) != batch:
            raise ObserverBoundary("OBSERVER_FORWARD_ROW_OVERFLOW")
        for row_index, pair in enumerate(group):
            prompt_ids, target_ids = self.kernel._encode_pair(self.tok, pair)
            start = length - (len(prompt_ids) + len(target_ids) - 1)
            target_start = start + len(prompt_ids) - 1
            expected = torch.tensor((prompt_ids + target_ids)[:-1],
                                    device=kwargs["input_ids"].device)
            if start < 0 or not torch.equal(kwargs["input_ids"][row_index, start:], expected):
                raise ObserverBoundary("NATIVE_ACTUAL_INPUT_ALIGNMENT")
            mask = kwargs["attention_mask"][row_index]
            if mask[:start].any().item() or not mask[start:].all().item():
                raise ObserverBoundary("NATIVE_ACTUAL_PAD_ALIGNMENT")
            selected_logits = logits[row_index, target_start:, :]
            metrics = _logit_diagnostics(selected_logits, target_ids, topk=self.topk)
            metrics.update(prompt_token_count=len(prompt_ids),
                           prompt_token_ids_sha256=digest(prompt_ids),
                           input_token_ids_sha256=digest((prompt_ids + target_ids)[:-1]),
                           scoring_positions_unpadded=list(range(len(prompt_ids) - 1,
                                                                len(prompt_ids) + len(target_ids) - 1)),
                           padding_tokens=start, actual_input_tokens=len(expected),
                           first_prompt_token_id=int(prompt_ids[0]),
                           prompt_starts_with_bos=bool(prompt_ids[0] == self.tok.bos_token_id))
            self.rows.append(metrics)
            self.counter["scored_target_tokens"] += len(target_ids)
        self.cursor += batch
        self.counter["forward_calls"] += 1
        self.counter["input_tokens_including_padding"] += int(kwargs["input_ids"].numel())
        self.counter["valid_input_tokens"] += int(kwargs["attention_mask"].sum().item())
        return output


def evaluate_pairs(model: Any, tok: Any, pairs: Sequence[Any], *, device: Any,
                   endpoint_id: str, postseal: bool, source_root: str | Path | None = None,
                   microbatch_size: int = 16, topk: int = 0) -> dict:
    """Original native scalar rows plus read-only diagnostic fields/counters."""
    _check_call(model, tok, endpoint_id, postseal, microbatch_size)
    kernel = bind_native_evaluator(source_root)
    observed = _LogitObserver(model, tok, pairs, kernel, topk)
    started = time.monotonic()
    rows = kernel.evaluate_pairs(observed, tok, pairs, device=device,
                                 microbatch_size=microbatch_size) if pairs else []
    if len(rows) != len(pairs) or observed.cursor != len(rows):
        raise ObserverBoundary("OBSERVER_MISSING_ROWS")
    for row, extra in zip(rows, observed.rows, strict=True):
        if not math.isfinite(row["nll"]):
            raise ObserverBoundary("OBSERVER_NONFINITE_NATIVE_NLL")
        if row["token_predictions"] != extra["argmax_token_ids"]:
            raise ObserverBoundary("OBSERVER_NATIVE_ARGMAX_MISMATCH")
        row["full_vocab"] = extra
        row["endpoint_id"] = endpoint_id
        row["identity"] = digest([row["case_id"], row["kind"], row["prompt_index"],
                                   row["prompt"], row["target"], row["target_token_ids"],
                                   extra["prompt_token_ids_sha256"]])
    if len({r["identity"] for r in rows}) != len(rows):
        raise ObserverBoundary("OBSERVER_DUPLICATE_ROWS")
    return {"rows": rows, "counter": observed.counter,
            "wall_seconds_inclusive": time.monotonic() - started,
            "endpoint_id": endpoint_id}


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict:
    numerator = sum(bool(r["success"]) for r in rows)
    new_tokens = sum(int(r["new_token_count"]) for r in rows)
    true_tokens = sum(int(r["true_token_count"]) for r in rows)
    return {"numerator": numerator, "denominator": len(rows),
            "rate": numerator / len(rows) if rows else None,
            "new_tf_strict": sum(bool(r["new_strict"]) for r in rows),
            "true_tf_strict": sum(bool(r["true_strict"]) for r in rows),
            "new_token_correct": sum(int(r["new_token_correct"]) for r in rows),
            "new_token_count": new_tokens,
            "true_token_correct": sum(int(r["true_token_correct"]) for r in rows),
            "true_token_count": true_tokens,
            "nll_ties": sum(r["new_nll"] == r["true_nll"] for r in rows),
            "rows": list(rows)}


def reduce_rpn(raw: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict:
    """Strict canonical NLL pairs; original success definition, separate TF IDs."""
    metrics = {}
    for category, tag, reverse in (("rewrite", "RS", False),
                                    ("rephrase", "PS", False), ("locality", "NS", True)):
        if category + "_target_new" not in raw:
            continue
        new_rows = raw[category + "_target_new"]
        true_rows = raw[category + "_target_true"]
        if len(new_rows) != len(true_rows):
            raise ObserverBoundary("OBSERVER_PAIR_DENOMINATOR")
        rows = []
        for a, b in zip(new_rows, true_rows, strict=True):
            if (a["case_id"], a["prompt_index"], a["prompt"]) != (b["case_id"], b["prompt_index"], b["prompt"]):
                raise ObserverBoundary("OBSERVER_PAIR_IDENTITY")
            if not math.isfinite(a["nll"]) or not math.isfinite(b["nll"]):
                raise ObserverBoundary("OBSERVER_NONFINITE_PAIR")
            if a.get("endpoint_id") != b.get("endpoint_id"):
                raise ObserverBoundary("OBSERVER_PAIR_ENDPOINT")
            if a.get("full_vocab", {}).get("prompt_token_ids_sha256") != b.get("full_vocab", {}).get("prompt_token_ids_sha256"):
                raise ObserverBoundary("OBSERVER_PAIR_PROMPT_TOKENS")
            row = {"case_id": a["case_id"], "prompt_index": a["prompt_index"],
                   "identity": digest([a["case_id"], a["prompt_index"], a["prompt"], a["target"], b["target"]]),
                   "prompt_sha256": digest(a["prompt"]),
                   "target_new_sha256": digest(a["target"]), "target_true_sha256": digest(b["target"]),
                   "new_nll": a["nll"], "true_nll": b["nll"],
                   "margin_true_minus_new": b["nll"] - a["nll"],
                   "desired_margin": a["nll"] - b["nll"] if reverse else b["nll"] - a["nll"],
                   "success": b["nll"] < a["nll"] if reverse else a["nll"] < b["nll"],
                   "new_strict": a["all_tokens_correct"], "true_strict": b["all_tokens_correct"],
                   "new_token_correct": sum(a["token_correct"]), "new_token_count": len(a["token_correct"]),
                   "true_token_correct": sum(b["token_correct"]), "true_token_count": len(b["token_correct"]),
                   "new_token_ids": a["target_token_ids"], "true_token_ids": b["target_token_ids"],
                   "new_token_predictions": a["token_predictions"], "true_token_predictions": b["token_predictions"],
                   "endpoint_id": a.get("endpoint_id")}
            for name, value in (("new", a), ("true", b)):
                row[f"{name}_full_vocab"] = value.get("full_vocab", {})
            rows.append(row)
        if len({r["identity"] for r in rows}) != len(rows):
            raise ObserverBoundary("OBSERVER_DUPLICATE_PAIRS")
        metrics[tag] = _metric_summary(rows)
    return metrics


def _counter_sum(results: Sequence[Mapping[str, Any]]) -> dict:
    return {key: sum(x["counter"][key] for x in results) for key in
            ("forward_calls", "extra_forward_calls", "input_tokens_including_padding",
             "valid_input_tokens", "scored_target_tokens")}


def evaluate_rpn(model: Any, tok: Any, records: Sequence[Mapping[str, Any]], *,
                 device: Any, endpoint_id: str, postseal: bool,
                 source_root: str | Path | None = None, include_neighborhood: bool = True,
                 microbatch_size: int = 16) -> dict:
    """R/P/N official observer; expected counts are derived without filtering."""
    kernel = bind_native_evaluator(source_root)
    if not records or len({int(r["case_id"]) for r in records}) != len(records):
        raise ObserverBoundary("OBSERVER_REQUESTS_EMPTY_OR_DUPLICATE")
    pairs = kernel.counterfact_pairs(records)
    if include_neighborhood:
        pairs["locality_target_new"] = [kernel.PromptTarget(
            int(record["case_id"]), "locality_target_new", i, prompt,
            record["requested_rewrite"]["target_new"]["str"])
            for record in records for i, prompt in enumerate(record["neighborhood_prompts"])]
    else:
        pairs = {k: v for k, v in pairs.items() if not k.startswith("locality_")}
    result = {kind: evaluate_pairs(model, tok, values, device=device,
                                   endpoint_id=endpoint_id, postseal=postseal,
                                   source_root=source_root, microbatch_size=microbatch_size)
              for kind, values in pairs.items()}
    raw = {k: v["rows"] for k, v in result.items()}
    metrics = reduce_rpn(raw)
    expected = {"RS": len(records), "PS": sum(len(r["paraphrase_prompts"]) for r in records)}
    if include_neighborhood:
        expected["NS"] = sum(len(r["neighborhood_prompts"]) for r in records)
    if {k: v["denominator"] for k, v in metrics.items()} != expected:
        raise ObserverBoundary("OBSERVER_CARDINALITY")
    rewrite = {r["case_id"]: r for r in metrics["RS"]["rows"]}
    rephrase = {r["case_id"]: [] for r in metrics["RS"]["rows"]}
    for row in metrics["PS"]["rows"]:
        rephrase[row["case_id"]].append(row)
    joint = [{"case_id": case, "rewrite_tf_strict": row["new_strict"],
              "paraphrase_denominator": len(rephrase[case]),
              "all_paraphrase_tf_strict": bool(rephrase[case]) and all(p["new_strict"] for p in rephrase[case]),
              "rewrite_and_all_paraphrase_tf_strict": bool(row["new_strict"]) and bool(rephrase[case]) and all(p["new_strict"] for p in rephrase[case]),
              "rewrite_and_all_paraphrase_nll_success": bool(row["success"]) and bool(rephrase[case]) and all(p["success"] for p in rephrase[case])}
             for case, row in rewrite.items()]
    return {"raw_local_only": raw, "compact": {
        "endpoint_id": endpoint_id, "requests": len(records),
        "request_order_sha256": digest([int(r["case_id"]) for r in records]),
        "metrics": metrics, "joint_rows": joint, "counter": _counter_sum(list(result.values())),
        "source": dict(NATIVE_SOURCE_SHA256), "native_helper_commit": NATIVE_HELPER_COMMIT,
        "postseal": True, "controller_influence": 0,
        "caller_state_restore_required": True, "state_nonmutation": "CALLER_VERIFICATION_REQUIRED",
        "tokenizer_route": "native evaluator: add_special_tokens=True prompt; target leading space/no specials; manual LEFT tensor pad; no explicit position_ids",
        "writer_tokenizer_route": "separate native writer: add_bos_token=False; right padding",
        "inclusive_kernel_wall_seconds": {k: v["wall_seconds_inclusive"] for k, v in result.items()}}}


def evaluate_neighborhood512(model: Any, tok: Any, panel: Mapping[str, Any], *,
                            device: Any, endpoint_id: str, postseal: bool,
                            source_root: str | Path | None = None,
                            microbatch_size: int = 16,
                            edit_target_token_ids: Sequence[int] | None = None) -> dict:
    """Fixed all512 observer with first-prediction top32 and complete TF rows."""
    kernel = bind_native_evaluator(source_root)
    records = panel["records"]
    if int(panel["n"]) != 512 or len(records) != 512:
        raise ObserverBoundary("N512_CARDINALITY")
    identities = [digest([r["prompt"], r["target_true"]]) for r in records]
    if len(set(identities)) != 512:
        raise ObserverBoundary("N512_DUPLICATE_PROMPT_TRUE")
    for record in records:
        original_identity = digest([record["case_id"], record["prompt_index"], record["prompt"],
                                    record["paired_edit_target"], record["target_true"]])
        if original_identity != record["base_identity"] or not math.isfinite(record["base_true_nll"]):
            raise ObserverBoundary("N512_W0_CANONICAL_IDENTITY")
    historical_target_tokens = set(map(int, edit_target_token_ids)) if edit_target_token_ids is not None else None
    result = {}
    for side, field in (("true", "target_true"), ("new", "paired_edit_target")):
        pairs = [kernel.PromptTarget(int(r["case_id"]), "locality_target_" + side,
                                    int(r["prompt_index"]), r["prompt"], r[field]) for r in records]
        result[side] = evaluate_pairs(model, tok, pairs, device=device, endpoint_id=endpoint_id,
                                      postseal=postseal, source_root=source_root,
                                      microbatch_size=microbatch_size, topk=32 if side == "true" else 0)
    # Selection panel explicitly consists of W0 one-token correct true labels.
    # A different evaluator tokenization must fail rather than silently redefine it.
    if any(len(r["target_token_ids"]) != 1 for r in result["true"]["rows"]):
        raise ObserverBoundary("N512_TRUE_NOT_ONE_TOKEN_IN_NATIVE_EVALUATOR")
    metrics = reduce_rpn({"locality_target_true": result["true"]["rows"],
                          "locality_target_new": result["new"]["rows"]})["NS"]
    rows = []
    for original, row, true in zip(records, metrics["rows"], result["true"]["rows"], strict=True):
        rows.append({**row, "base_identity": original["base_identity"],
                     "base_true_nll": float(original["base_true_nll"]),
                     "current_minus_w0_true_nll": row["true_nll"] - float(original["base_true_nll"]),
                     "w0_true_argmax_correct": True,
                     "true_argmax_correct": bool(true["all_tokens_correct"]),
                     "argmax_in_paired_edit_target_tokens": true["token_predictions"][0] in row["new_token_ids"],
                     "argmax_in_seen_edit_target_tokens": (true["token_predictions"][0] in historical_target_tokens
                                                            if historical_target_tokens is not None else "NOT_RECORDED"),
                     "top32": true["full_vocab"]})
    return {"raw_local_only": {side: value["rows"] for side, value in result.items()},
            "compact": {"endpoint_id": endpoint_id, "panel": "N512", "rows": rows,
                        "denominator": 512, "pairwise_NS_numerator": metrics["numerator"],
                        "true_argmax_correct": sum(r["true_argmax_correct"] for r in rows),
                        "true_unique_argmax_correct": sum(r["true_full_vocab"]["all_unique_argmax_correct"] for r in rows),
                        "true_nll_mean": sum(r["true_nll"] for r in rows) / 512,
                        "observer_only": True, "basis_or_strength_selection": False,
                        "seen_edit_target_token_set_sha256": (digest(sorted(historical_target_tokens))
                                                               if historical_target_tokens is not None else "NOT_RECORDED"),
                        "counter": _counter_sum(list(result.values())),
                        "state_nonmutation": "CALLER_VERIFICATION_REQUIRED"}}


def _fact(record: Mapping[str, Any]) -> tuple[str, str, str]:
    rewrite = record.get("requested_rewrite", record)
    relation = rewrite.get("relation_id", rewrite.get("relation"))
    target = rewrite.get("target_new", rewrite.get("target"))
    if isinstance(target, Mapping):
        target = target["str"]
    if relation is None or not isinstance(target, str) or "subject" not in rewrite:
        raise ObserverBoundary("H512_RAW_FACT_SCHEMA")
    return str(rewrite["subject"]), str(relation), target


def history_masks(panel: Mapping[str, Any], received_records: Sequence[Mapping[str, Any]]) -> dict:
    """All512 weight1 for statistics, target-valid mask from entry's past only.

    No normalization of raw(subject,relation), no success filtering. Earlier
    occurrences with the same latest target stay functionally valid. The caller
    supplies only events already received at this entry (not the future/current
    batch); the entry ledger hash makes that cutoff explicit.
    """
    if int(panel["n"]) != 512 or len(panel["records"]) != 512:
        raise ObserverBoundary("H512_CARDINALITY")
    latest: dict[tuple[str, str], tuple[str, int, int]] = {}
    arrived: dict[int, tuple[str, str, str]] = {}
    ledger = []
    for ordinal, record in enumerate(received_records):
        fact = _fact(record)
        case_id = int(record["case_id"])
        if case_id in arrived:
            raise ObserverBoundary("H512_DUPLICATE_RECEIVED_CASE")
        arrived[case_id] = fact
        latest[fact[:2]] = (fact[2], case_id, ordinal)
        ledger.append([ordinal, case_id, *fact])
    rows = []
    for record in panel["records"]:
        case_id = int(record["case_id"])
        fact = _fact(record)
        if arrived.get(case_id) != fact:
            raise ObserverBoundary("H512_BANK_NOT_IDENTICAL_RECEIVED_EVENT")
        target, latest_case, latest_ordinal = latest[fact[:2]]
        valid = fact[2] == target
        rows.append({"case_id": case_id, "fact_sha256": digest(list(fact[:2])),
                     "target_sha256": digest(fact[2]), "latest_target_sha256": digest(target),
                     "latest_event_case_id": latest_case, "latest_event_ordinal": latest_ordinal,
                     "functional_current_valid": valid, "functional_superseded": not valid,
                     "statistics_weight": 1, "write_batch": int(record["write_batch"]),
                     "timestamp_checkpoint": record["timestamp_checkpoint"]})
    if len({r["case_id"] for r in rows}) != 512:
        raise ObserverBoundary("H512_DUPLICATE_BANK_CASE")
    return {"rows": rows, "statistics_denominator": 512, "statistics_weight_sum": 512,
            "functional_current_valid": sum(r["functional_current_valid"] for r in rows),
            "functional_superseded": sum(r["functional_superseded"] for r in rows),
            "received_event_count": len(received_records), "entry_ledger_sha256": digest(ledger),
            "selection_uses_success": False, "statistics_include_superseded": True}


def evaluate_history512(model: Any, tok: Any, panel: Mapping[str, Any],
                        records_by_case: Mapping[int, Mapping[str, Any]],
                        received_records: Sequence[Mapping[str, Any]], *, device: Any,
                        endpoint_id: str, postseal: bool,
                        source_root: str | Path | None = None, microbatch_size: int = 16) -> dict:
    masks = history_masks(panel, received_records)
    records = [records_by_case[int(r["case_id"])] for r in panel["records"]]
    for record, expected in zip(records, panel["records"], strict=True):
        if _fact(record) != _fact(expected) or int(record["case_id"]) != int(expected["case_id"]):
            raise ObserverBoundary("H512_EVALUATION_RECORD_IDENTITY")
    result = evaluate_rpn(model, tok, records, device=device, endpoint_id=endpoint_id,
                          postseal=postseal, source_root=source_root,
                          include_neighborhood=False, microbatch_size=microbatch_size)
    flags = {r["case_id"]: r for r in masks["rows"]}
    subsets = {}
    for subset in ("ALL", "CURRENT_VALID", "SUPERSEDED"):
        subsets[subset] = {}
        for metric, value in result["compact"]["metrics"].items():
            rows = [r for r in value["rows"] if subset == "ALL" or
                    flags[r["case_id"]]["functional_current_valid"] == (subset == "CURRENT_VALID")]
            subsets[subset][metric] = _metric_summary(rows)
    result["compact"].update(panel="H512", masks=masks, functional_subsets=subsets,
                              functional_mask_fixed_at_entry=True,
                              statistics_bank="all512; weight1; no superseded removal")
    return result


def protection_transitions(before: Sequence[Mapping[str, Any]],
                           after: Sequence[Mapping[str, Any]]) -> dict:
    """Exact ordered IDs, entry success denominator, gross loss/recovery."""
    if len(before) != len(after):
        raise ObserverBoundary("PROTECTION_TRANSITION_DENOMINATOR")
    rows = []
    for a, b in zip(before, after, strict=True):
        if a["identity"] != b["identity"]:
            raise ObserverBoundary("PROTECTION_TRANSITION_IDENTITY")
        old, new = bool(a["true_argmax_correct"]), bool(b["true_argmax_correct"])
        rows.append({"identity": a["identity"], "entry_correct": old, "final_correct": new,
                     "lost": old and not new, "recovered": not old and new,
                     "true_nll_change": b["true_nll"] - a["true_nll"]})
    return {"denominator": len(rows), "entry_successes": sum(r["entry_correct"] for r in rows),
            "lost": sum(r["lost"] for r in rows), "recovered": sum(r["recovered"] for r in rows),
            "rows": rows}
