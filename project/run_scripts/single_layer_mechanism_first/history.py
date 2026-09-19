"""All-received semantic registry and entry-anchored streaming history oracle.

No Past64 sampling, success filtering, official P/N, recency weighting, or
native-memory mutation occurs here.  Prefix preparation is caller-owned and
request-streamed: at most the current request's canonical new/old paths are
resident.  Semantic overwrite does not remove old native Gram contributions.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import time

import torch
import torch.nn.functional as F

from .decision import (DecisionError, EndpointBinding, _json_sha, _require,
                       margins_from_logits, risk_from_margins, suffix_from_output)


DESIRED_NLL_TOLERANCE = 1e-4


def rewrite(record):
    return record.get("requested_rewrite", record)


def fact_identity(record):
    r = rewrite(record)
    _require(isinstance(r.get("subject"), str) and isinstance(r.get("relation_id"), str),
             "RAW_SUBJECT_RELATION_IDENTITY_REQUIRED")
    return r["subject"], r["relation_id"]


def valid_target(record):
    target = rewrite(record).get("target_new", {})
    return isinstance(target, dict) and isinstance(target.get("str"), str) and bool(target["str"])


def receive_all(ledger, records):
    """Record every received event including finite failures/no-correction."""
    out = deepcopy(ledger)
    seen = {str(row["case_id"]) for row in out}
    for record in records:
        _require("case_id" in record and str(record["case_id"]) not in seen, "DUPLICATE_OR_MISSING_RECEIVED_EVENT")
        fact_identity(record)
        event = deepcopy(record)
        event.update(ordinal=len(out), target_valid=valid_target(record))
        event["stable_event_id"] = _json_sha(dict(domain="SL-MECHANISM-FIRST-v1", ordinal=len(out),
            case_id=record["case_id"], subject=fact_identity(record)[0], relation=fact_identity(record)[1],
            target=rewrite(record).get("target_new")))
        seen.add(str(event["case_id"]))
        out.append(event)
    return out


def active_history(ledger, current):
    """Latest valid target per raw fact, excluding current valid overwrites.

    Returned order is received order, never a performance-dependent ranking.
    Invalid latest events remain in the ledger but cannot erase a prior valid
    target.  The caller must report their separate count.
    """
    latest = {}
    for event in ledger:
        if valid_target(event):
            latest[fact_identity(event)] = event
    overwritten = {fact_identity(r) for r in current if valid_target(r)}
    return deepcopy(sorted((r for key, r in latest.items() if key not in overwritten),
                           key=lambda r: r["ordinal"]))


def registry_status(ledger, current=()):
    active = {r["stable_event_id"] for r in active_history(ledger, current)}
    overwritten = {fact_identity(r) for r in current if valid_target(r)}
    return [dict(case_id=r["case_id"], stable_event_id=r["stable_event_id"], ordinal=r["ordinal"],
                 status="INVALID_TARGET" if not valid_target(r) else
                        "CURRENT_OVERWRITTEN" if fact_identity(r) in overwritten else
                        "ACTIVE" if r["stable_event_id"] in active else "SUPERSEDED") for r in ledger]


def safety_slack(anchor, current):
    """h_j and exact-ID guard, anchored at entry, not own-native."""
    _require(set(anchor) == set(current) and "new" in anchor, "HISTORY_BRANCH_INVENTORY")
    for branch in anchor:
        _require(all(anchor[branch][k] == current[branch][k] for k in ("positions", "labels", "input_identity")),
                 "HISTORY_TARGET_OR_PREFIX_CHANGED")
        _require(math.isfinite(current[branch]["nll"]) and
                 all(math.isfinite(x) for x in current[branch]["margins"]), "NONFINITE_HISTORY_SCORE")
    values = [("desired_nll", anchor["new"]["nll"] + DESIRED_NLL_TOLERANCE - current["new"]["nll"])]
    entry_preference = "old" in anchor and anchor["new"]["nll"] < anchor["old"]["nll"]
    if entry_preference:
        values.append(("preference", current["old"]["nll"] - current["new"]["nll"]))
    if anchor["new"]["strict"]:
        values.append(("strict", min(current["new"]["margins"])))
    # Branch order is fixed for equal slacks; token minimum uses first position.
    branch, slack = min(values, key=lambda v: v[1])
    reasons = []
    if current["new"]["nll"] > anchor["new"]["nll"] + DESIRED_NLL_TOLERANCE:
        reasons.append("ENTRY_DESIRED_NLL")
    if entry_preference and not current["new"]["nll"] < current["old"]["nll"]:
        reasons.append("ENTRY_PREFERENCE_ID_LOST")
    if anchor["new"]["strict"] and not current["new"]["strict"]:
        reasons.append("ENTRY_STRICT_ID_LOST")
    return dict(slack=slack, exposed_branch=branch, guard_pass=not reasons, reasons=reasons,
                entry_preference_success=entry_preference, entry_strict_success=anchor["new"]["strict"],
                component_slacks=dict(values), old_branch_available="old" in anchor)


@contextmanager
def _request_context(prepared):
    """Permit a caller-owned context manager to unload on exceptions."""
    if hasattr(prepared, "__enter__"):
        with prepared as value:
            yield value
    else:
        yield prepared


@dataclass
class HistoryObservation:
    endpoint_identity: str
    history_identity: str
    entry_anchor_identity: str | None
    rows: list
    phi_history: float
    guard_pass: bool
    gradient: torch.Tensor | None
    coverage: dict
    work: dict
    factors: object | None = None

    def anchor(self):
        """Owned scalar entry anchor; no hidden/model state or future inputs."""
        return dict(endpoint_identity=self.endpoint_identity, history_identity=self.history_identity,
                    rows=deepcopy(self.rows))

    def compact(self):
        return dict(endpoint_identity=self.endpoint_identity, history_identity=self.history_identity,
                    entry_anchor_identity=self.entry_anchor_identity, rows=deepcopy(self.rows),
                    phi_history=self.phi_history, guard_pass=self.guard_pass,
                    coverage=deepcopy(self.coverage), work=deepcopy(self.work),
                    gradient_norm=None if self.gradient is None else float(self.gradient.norm()))


class StreamingHistoryOracle:
    """All active canonical requests, at most two prefix graphs per request.

    ``prepare_request(record)`` returns (oracle, rows) or a context manager of
    that tuple.  Rows must be the provided canonical new/old teacher-forcing
    paths, never native augmentations/official paraphrases/neighborhoods.
    The factory may retain a read-only disk prefix cache, but must not keep all
    Past caches resident.  This class never calls it for more than one request
    concurrently.  Exact duplicate complete input paths are shared locally.
    """
    def __init__(self, records, prepare_request, *, max_active_requests=900):
        self.records = deepcopy(records)
        _require(len(records) <= max_active_requests and len({str(r["case_id"]) for r in records}) == len(records),
                 "HISTORY_MEMBERSHIP_OR_MAX900")
        _require(all(valid_target(r) for r in records), "HISTORY_VALID_TARGET_REQUIRED")
        self.history_identity = _json_sha([dict(case_id=r["case_id"], fact=fact_identity(r),
            target=rewrite(r)["target_new"], stable_event_id=r.get("stable_event_id")) for r in records])
        self.prepare_request = prepare_request
        self._gradient_centers = set()

    @staticmethod
    def _validate_paths(oracle, rows, case_id):
        _require(1 <= len(rows) <= 2 and {r["branch"] for r in rows} in ({"new"}, {"new", "old"}),
                 "HISTORY_ONE_CANONICAL_NEW_OPTIONAL_PROVIDED_OLD")
        _require(all(r["kind"] == "canonical" and r["case_id"] == case_id for r in rows),
                 "HISTORY_CANONICAL_ID_ONLY_NO_OBSERVER")
        _require({r["cache"] for r in rows} == set(range(len(oracle.caches))) and len(oracle.caches) <= 2,
                 "HISTORY_REQUEST_STREAMING_MAX_TWO_CACHES")
        _require(len({r["branch"] for r in rows}) == len(rows), "DUPLICATE_HISTORY_BRANCH")
        for r in rows:
            _require(bool(r["labels"]) and len(r["positions"]) == len(r["labels"]), "HISTORY_TARGET_LENGTH")

    @staticmethod
    def _paths(oracle, rows, endpoint, gradient):
        modules, keys, hidden, scores, tensors = {}, {}, {}, {}, {}
        for ci, cache in enumerate(oracle.caches):
            k = cache.keys.to(endpoint.device)
            output = F.linear(k, endpoint.device_weight).detach().requires_grad_(gradient)
            modules[ci], keys[ci] = output, k
            hidden[ci] = suffix_from_output(oracle, ci, output)
        for row in rows:
            ci = row["cache"]
            logits = oracle._head(hidden[ci], torch.tensor(row["positions"], dtype=torch.long))
            labels = torch.tensor(row["labels"], dtype=torch.long, device=endpoint.device)
            _require(logits.dtype == torch.float32 and bool(torch.isfinite(logits).all()), "NONFINITE_HISTORY_LOGITS")
            nll = -logits.log_softmax(-1)[torch.arange(len(labels), device=endpoint.device), labels].double().mean()
            margin = margins_from_logits(logits, labels)
            worst = min(range(len(margin["margins"])), key=lambda i: (margin["margins"][i], i))
            scalar_margin = logits[worst, labels[worst]] - logits[worst, margin["competitors"][worst]]
            branch = row["branch"]
            scores[branch] = dict(nll=float(nll.detach()), strict=all(margin["correct"]),
                margins=margin["margins"], predictions=margin["predictions"],
                positions=list(row["positions"]), labels=list(row["labels"]),
                input_identity=_json_sha(dict(ids=oracle.caches[ci].packed["input_ids"].tolist(),
                    mask=oracle.caches[ci].packed["attention_mask"].tolist(),
                    position_ids=oracle.caches[ci].packed["position_ids"].tolist())),
                worst_offset=worst, competitor=margin["competitors"][worst])
            tensors[branch] = dict(nll=nll, strict_margin=scalar_margin, cache=ci)
        return scores, tensors, modules, keys

    def evaluate(self, endpoint, *, entry_anchor=None, gradient=False, factor_sink=None):
        endpoint.guard()
        _require(not gradient or entry_anchor is not None, "HISTORY_DERIVATIVE_NEEDS_ENTRY_ANCHOR")
        _require(not gradient or factor_sink is not None or not self.records, "HISTORY_FACTORS_REQUIRED")
        if entry_anchor is not None:
            _require(entry_anchor["history_identity"] == self.history_identity and
                     [r["case_id"] for r in entry_anchor["rows"]] == [r["case_id"] for r in self.records],
                     "ENTRY_ANCHOR_FULL_ACTIVE_HISTORY_IDENTITY")
        key = (endpoint.identity, self.history_identity)
        _require(not gradient or key not in self._gradient_centers, "SECOND_HISTORY_CENTER_DERIVATIVE_FORBIDDEN")
        if gradient:
            self._gradient_centers.add(key)
        started = time.perf_counter()
        total = torch.zeros(endpoint.shape, dtype=torch.float64, device=endpoint.device) if gradient else None
        result = []
        work = dict(requests=0, unique_TF_paths=0, valid_input_tokens=0, scored_target_tokens=0,
                    suffix_forwards=0, full_vocab_head_rows=0, backward_requests=0,
                    backward_TF_paths=0, autograd_calls=0, factor_D2H_bytes=0,
                    peak_resident_requests=0, peak_resident_paths=0, dense_gradient_D2H_bytes=0,
                    official_observer_reads=0, native_history_appends=0)
        factor_index = 0
        for ordinal, record in enumerate(self.records):
            with _request_context(self.prepare_request(record)) as (oracle, rows):
                self._validate_paths(oracle, rows, record["case_id"])
                _require(tuple(oracle.shape) == endpoint.shape and oracle.device == endpoint.device,
                         "HISTORY_ENDPOINT_SHAPE_DEVICE")
                oracle._guard()
                with torch.set_grad_enabled(gradient):
                    scores, tensors, modules, keys = self._paths(oracle, rows, endpoint, gradient)
                    anchor_scores = scores if entry_anchor is None else entry_anchor["rows"][ordinal]["scores"]
                    slack = safety_slack(anchor_scores, scores)
                    if gradient:
                        branch = slack["exposed_branch"]
                        if branch == "desired_nll":
                            scalar = anchor_scores["new"]["nll"] + DESIRED_NLL_TOLERANCE - tensors["new"]["nll"]
                            needed = {tensors["new"]["cache"]}
                        elif branch == "preference":
                            scalar = tensors["old"]["nll"] - tensors["new"]["nll"]
                            needed = {tensors["new"]["cache"], tensors["old"]["cache"]}
                        else:
                            scalar = tensors["new"]["strict_margin"]
                            needed = {tensors["new"]["cache"]}
                        ordered = sorted(needed)
                        grads = torch.autograd.grad(scalar, tuple(modules[ci] for ci in ordered))
                        for ci, grad in zip(ordered, grads):
                            valid = oracle.caches[ci].packed["attention_mask"][0].bool().to(endpoint.device)
                            a, k = grad[0, valid].detach(), keys[ci][0, valid]
                            _require(a.dtype == torch.float32 and bool(torch.isfinite(a).all()), "NONFINITE_HISTORY_ACTIVATION_GRADIENT")
                            gi = a.T @ k
                            _require(bool(torch.isfinite(gi).all()), "NONFINITE_HISTORY_FACTOR_PRODUCT")
                            total.add_(gi.double(), alpha=-2 * max(0.0, -slack["slack"]) / len(self.records))
                            factor_sink.put(factor_index, a, dict(endpoint_identity=endpoint.identity,
                                history_identity=self.history_identity, request_ordinal=ordinal,
                                case_id=record["case_id"], cache_index=ci,
                                input_identity=_json_sha(oracle.caches[ci].input_identity),
                                exposed_branch=branch, entry_endpoint_identity=entry_anchor["endpoint_identity"]))
                            factor_index += 1
                            work["factor_D2H_bytes"] += a.numel() * a.element_size()
                            work["backward_TF_paths"] += 1
                        work["backward_requests"] += 1
                        work["autograd_calls"] += 1
                    result.append(dict(case_id=record["case_id"], ordinal=ordinal, scores=scores, **slack))
                work["requests"] += 1
                work["unique_TF_paths"] += len(oracle.caches)
                work["valid_input_tokens"] += sum(int(c.packed["attention_mask"].sum()) for c in oracle.caches)
                work["scored_target_tokens"] += sum(len(r["labels"]) for r in rows)
                work["full_vocab_head_rows"] += sum(len(r["labels"]) for r in rows)
                work["suffix_forwards"] += len(oracle.caches)
                work["peak_resident_requests"] = 1
                work["peak_resident_paths"] = max(work["peak_resident_paths"], len(oracle.caches))
                oracle._guard()
                # Explicitly drop graphs and prefix references before factory
                # context exit/next request. No all-Past resident list is made.
                del scores, tensors, modules, keys
            del oracle, rows
        endpoint.guard()
        if gradient:
            _require(bool(torch.isfinite(total).all()), "NONFINITE_HISTORY_ACCUMULATION")
            total = total.cpu()
            work["dense_gradient_D2H_bytes"] = total.numel() * total.element_size() if endpoint.device.type == "cuda" else 0
            if factor_sink is not None:
                factor_sink.seal(expected_count=factor_index)
        work["wall_seconds"] = time.perf_counter() - started
        return HistoryObservation(endpoint.identity, self.history_identity,
            None if entry_anchor is None else entry_anchor["endpoint_identity"], result,
            risk_from_margins([r["slack"] for r in result]) if result else 0.,
            all(r["guard_pass"] for r in result), total,
            dict(complete=True, active_requests=len(self.records), all_received_latest_valid=True,
                 sampling=False, entry_anchored=True, past_empty=not self.records), work,
            factor_sink if gradient else None)

    def jacobian(self, observation, directions):
        """Stream each request's paths/factors, no model evaluation/backward."""
        _require(observation.history_identity == self.history_identity and observation.coverage["complete"],
                 "HISTORY_JACOBIAN_IDENTITY")
        _require(1 <= len(directions) <= 5, "HISTORY_DIRECTION_COUNT")
        result = torch.zeros((len(self.records), len(directions)), dtype=torch.float64)
        if not self.records:
            return result
        _require(observation.factors is not None and observation.factors.sealed, "COMPLETE_HISTORY_FACTORS")
        by_request = {}
        for item in observation.factors.members:
            by_request.setdefault(item["metadata"]["request_ordinal"], []).append(item["index"])
        _require(set(by_request) == set(range(len(self.records))), "HISTORY_FACTOR_REQUEST_COVERAGE")
        for ordinal, record in enumerate(self.records):
            with _request_context(self.prepare_request(record)) as (oracle, rows):
                self._validate_paths(oracle, rows, record["case_id"])
                for index in by_request[ordinal]:
                    a, meta = observation.factors.get(index)
                    ci = meta["cache_index"]
                    _require(meta["endpoint_identity"] == observation.endpoint_identity and
                             meta["history_identity"] == self.history_identity and
                             meta["input_identity"] == _json_sha(oracle.caches[ci].input_identity),
                             "HISTORY_FACTOR_ENDPOINT_PREFIX_MISMATCH")
                    k = oracle.caches[ci].valid_keys().double()
                    for c, d in enumerate(directions):
                        result[ordinal, c] += (a.double() * (d.detach().cpu().double() @ k).T).sum()
        _require(bool(torch.isfinite(result).all()), "NONFINITE_HISTORY_JACOBIAN")
        return result
