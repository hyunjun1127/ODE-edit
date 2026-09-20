"""CPU-only semantic ledger and bounded replay policy.

This module owns metadata, never solver M, model weights, or tensor persistence.
The caller captures immutable final-at-write teachers and completes native history
finalization before ``commit``. A commit validates everything before one RAM state
swap. Restart from disk is deliberately unsupported.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np

ARMS = ("EN_ADAPT_H_RES", "EN_ADAPT_H_GSS_REC")


def _json_bytes(value: Any) -> bytes:
    # Reject tensors/arrays, unknown objects, and non-finite metadata; no default=str.
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def priority(identity: str, seed: int = 20260920) -> tuple[str, str]:
    """Canonical fixed hash priority, unrelated to age or model quality."""
    return hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest(), str(identity)


def history_weights(created_batches, current_batch: int, *, recency: bool = True,
                    cap: int = 512, batch_size: int = 100):
    """Per-version normalized history weights; same ages give exact uniform values."""
    created = np.asarray(created_batches, dtype=np.float64)
    if created.ndim != 1 or not np.isfinite(created).all():
        raise ValueError("finite one-dimensional creation batches required")
    if cap <= 0 or batch_size <= 0:
        raise ValueError("positive capacity and batch size required")
    ages = current_batch - 1 - created
    if np.any(ages < 0):
        raise ValueError("current/future version is not past history")
    if not len(ages):
        return np.empty(0, dtype=np.float64)
    if not recency or np.all(ages == ages[0]):
        return np.full(len(ages), 1 / len(ages), dtype=np.float64)
    weights = 1 + np.exp2(-ages / (cap / batch_size))
    return weights / weights.sum()


def gss_prune(features, identities, cap: int = 512, *, seed: int = 20260920,
              zero: float = 1e-12) -> dict:
    """Canonical signed-cosine backward pruning, with native Python receipts.

    The caller must provide fresh features at the same current native endpoint.
    This policy cannot establish feature provenance or neural sketch fidelity.
    """
    identities = list(identities)
    z = np.asarray(features, dtype=np.float64)
    if z.ndim != 2 or len(z) != len(identities) or len(set(identities)) != len(identities):
        raise ValueError("unique identity per feature row required")
    if cap <= 0 or not np.isfinite(z).all():
        raise ValueError("positive capacity and finite features required")
    n = len(z)
    if n <= cap:
        return dict(selected=list(range(n)), removed=[], trace=[], selection_exercised=False)
    norms = np.linalg.norm(z, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("nonfinite feature norms")
    active = [int(i) for i in np.flatnonzero(norms > zero)]
    zeros = [int(i) for i in np.flatnonzero(norms <= zero)]
    trace, removed = [], []
    if len(active) <= cap:
        extras = sorted(zeros, key=lambda j: priority(identities[j], seed))[:cap-len(active)]
        selected = sorted(active + extras)
        return dict(selected=selected, removed=[i for i in range(n) if i not in selected],
                    trace=trace, selection_exercised=True, zero_count=len(zeros))
    u = z[active] / norms[active, None]
    gram = u @ u.T
    local = list(range(len(active)))
    row_sum = gram.sum(axis=1)
    while len(local) > cap:
        best = max(row_sum[j] for j in local)
        tolerance = 64 * np.finfo(float).eps * max(1, abs(best))
        candidates = [j for j in local if best - row_sum[j] <= tolerance]
        chosen = min(candidates, key=lambda j: priority(identities[active[j]], seed))
        before = float(row_sum[local].sum())
        trace.append(dict(active=[active[j] for j in local], removed=active[chosen],
                          objective_before=before,
                          predicted_objective_after=before-2*float(row_sum[chosen])+1))
        local.remove(chosen)
        row_sum -= gram[:, chosen]
        removed.append(active[chosen])
    return dict(selected=sorted(active[j] for j in local), removed=removed + zeros,
                trace=trace, selection_exercised=True, zero_count=len(zeros))


def _record_identity(record: Mapping) -> tuple[tuple[str, str], dict]:
    request = record.get("requested_rewrite", record)
    fact = (request.get("subject"), request.get("relation_id"))
    if any(not isinstance(value, str) or not value for value in fact):
        raise ValueError("explicit subject/relation_id strings required; no inferred identity")
    target = request.get("target_new")
    if not isinstance(target, dict) or not isinstance(target.get("str"), str) or not target["str"]:
        raise ValueError("canonical target_new dictionary with nonempty str required")
    if "case_id" not in record:
        raise ValueError("case_id required for occurrence identity")
    # Both supplied text and source entity identity are bound. Prompt variations do
    # not refresh a teacher for an otherwise identical fact/target version.
    return fact, {"id": target.get("id"), "str": target["str"]}


class HistoryLedger:
    """Whole semantic ledger plus one independent arm's bounded replay state."""

    def __init__(self, arm: str, cap: int = 512, seed: int = 20260920,
                 pending_capacity: int = 100):
        if arm not in ARMS:
            raise ValueError(f"latest authority authorizes only {ARMS}")
        if cap <= 0 or pending_capacity <= 0:
            raise ValueError("positive capacities required")
        self.arm, self.cap, self.seed = arm, int(cap), int(seed)
        self.pending_capacity = int(pending_capacity)
        self.versions: dict[str, dict] = {}
        self.eventledger: list[dict] = []
        self.bank_ids: list[str] = []
        self.pending_ids: list[str] = []
        self.last_batch = 0
        self.commit_ids: list[str] = []
        self._active_by_fact: dict[tuple[str, str], str] = {}
        self._revision = 0

    def _state_identity(self) -> str:
        return _digest(dict(arm=self.arm, cap=self.cap, seed=self.seed,
                            revision=self._revision, last_batch=self.last_batch,
                            active=sorted((list(f), v) for f, v in self._active_by_fact.items()),
                            bank=self.bank_ids, pending=self.pending_ids,
                            occurrences=len(self.eventledger)))

    def active_ids(self) -> list[str]:
        """Latest valid versions in last-occurrence stream order."""
        return sorted(self._active_by_fact.values(),
                      key=lambda vid: (self.versions[vid]["latest_order"], vid))

    def latest_records(self) -> list[dict]:
        """One latest-occurrence record per active version; teachers keep originals."""
        return [copy.deepcopy(self.versions[vid]["latest_record"]) for vid in self.active_ids()]

    def active_case_ids(self) -> list[Any]:
        """All occurrences belonging to still-active versions, in original order."""
        return [row["case_id"] for row in self.eventledger
                if self.versions[row["version_id"]]["active"]]

    def occurrence_rows(self) -> list[dict]:
        """Preserve retired occurrences for all-seen/neighborhood denominators."""
        return [dict(row, active=self.versions[row["version_id"]]["active"],
                     superseded=not self.versions[row["version_id"]]["active"])
                for row in self.eventledger]

    def prepare(self, batch: int, records: Iterable[Mapping], *, terminal: bool = False) -> dict:
        """Plan replay and semantic commit without changing any ledger state."""
        if batch != self.last_batch + 1:
            raise ValueError("batch must be the next uncommitted batch")
        records = copy.deepcopy(list(records))
        if not records or len(records) > self.pending_capacity:
            raise ValueError("nonempty bounded current batch required")
        _json_bytes(records)
        identities = [_record_identity(record) for record in records]
        current_facts = set(fact for fact, _ in identities)
        eligible = {vid for fact, vid in self._active_by_fact.items() if fact not in current_facts}
        if self.arm == "EN_ADAPT_H_RES":
            pool = sorted(eligible, key=lambda vid: priority(vid, self.seed))[:self.cap]
        else:
            pool = list(dict.fromkeys(vid for vid in self.bank_ids + self.pending_ids if vid in eligible))
        if len(pool) > self.cap + self.pending_capacity:
            raise ValueError("selection pool exceeded bank plus pending capacity")
        if any(self.versions[vid]["teacher_binding"] is None for vid in pool):
            raise ValueError("eligible replay version lacks its original immutable teacher")

        active = dict(self._active_by_fact)
        new_rows: dict[str, dict] = {}
        events, retired = [], []
        for index, (record, (fact, target)) in enumerate(zip(records, identities)):
            previous = active.get(fact)
            previous_row = new_rows.get(previous, self.versions.get(previous))
            order = len(self.eventledger) + index
            if previous_row is not None and previous_row["target"] == target:
                vid, action = previous, "same_target_occurrence"
            else:
                if previous is not None:
                    retired.append(previous)
                vid = "fv-" + _digest(dict(fact=list(fact), target=target, batch=batch,
                                          order=order, case_id=record["case_id"]))
                action = "create_version" if previous is None else "overwrite_version"
                new_rows[vid] = dict(version_id=vid, fact=list(fact), fact_id=_digest(list(fact)),
                                     target=target, created_batch=batch, created_order=order,
                                     latest_order=order, record=record, latest_record=record,
                                     active=True, teacher_binding=None, occurrence_case_ids=[])
                active[fact] = vid
            events.append(dict(batch=batch, batch_index=index, order=order, case_id=record["case_id"],
                               fact=list(fact), version_id=vid, action=action,
                               previous_version_id=previous))
        final_current = [active[fact] for fact in current_facts]
        latest_order = {event["version_id"]: event["order"] for event in events}
        final_current.sort(key=lambda vid: (latest_order[vid], vid))
        new_final_ids = [vid for vid in final_current if vid in new_rows]
        payload = dict(batch=batch, terminal=bool(terminal), arm=self.arm,
                       state_identity=self._state_identity(), records_digest=_digest(records),
                       records=records, pool_ids=pool, current_facts=sorted(map(list, current_facts)),
                       event_plan=events, created_versions=list(new_rows.values()),
                       new_versions=[new_rows[vid] for vid in new_final_ids],
                       teacher_required_ids=[] if terminal else new_final_ids,
                       retired_ids=list(dict.fromkeys(retired)), final_current_ids=final_current,
                       active_after=sorted((list(fact), vid) for fact, vid in active.items()))
        return dict(payload, preparation_id=_digest(payload))

    @staticmethod
    def _verify_preparation(preparation: Mapping) -> dict:
        result = dict(preparation)
        supplied = result.pop("preparation_id", None)
        result.pop("selected_ids", None)
        if supplied is None or supplied != _digest(result):
            raise ValueError("preparation identity changed")
        return result

    def select(self, preparation: Mapping, selected_ids: Iterable[str]) -> dict:
        """Freeze a valid selection; no bank, version, or occurrence mutation."""
        payload = self._verify_preparation(preparation)
        if payload["state_identity"] != self._state_identity():
            raise ValueError("stale preparation")
        selected, pool = list(selected_ids), payload["pool_ids"]
        if len(selected) != len(set(selected)) or not set(selected).issubset(pool):
            raise ValueError("selection must be unique members of this exact pool")
        if len(selected) != min(self.cap, len(pool)):
            raise ValueError("selection must retain the full allowed number of pool members")
        if self.arm == "EN_ADAPT_H_RES" and set(selected) != set(pool):
            raise ValueError("RES uses the sealed full-active bottom-hash pool")
        return dict(preparation, selected_ids=[vid for vid in pool if vid in set(selected)])

    def commit(self, batch: int, records: Iterable[Mapping], selected_ids: Iterable[str],
               teacher_bindings: Mapping[str, Mapping], *, preparation: Mapping | None = None,
               terminal: bool = False) -> dict:
        """Atomically publish metadata after external teacher/native completion.

        Duplicate commits raise rather than append twice. Failed validation leaves
        the entire RAM ledger untouched. External native M rollback belongs to the
        caller; this pure policy cannot make a model mutation transactional.
        """
        records = list(records)
        if batch != self.last_batch + 1:
            raise ValueError("batch already committed or out of sequence")
        if preparation is None:
            preparation = self.prepare(batch, records, terminal=terminal)
        selected = self.select(preparation, selected_ids)
        if selected["batch"] != batch or selected["records_digest"] != _digest(records):
            raise ValueError("commit batch or request identity differs from preparation")
        if selected["terminal"] != bool(terminal):
            raise ValueError("terminal teacher policy differs from preparation")
        bindings = json.loads(_json_bytes(dict(teacher_bindings)))
        if set(bindings) != set(selected["teacher_required_ids"]):
            raise ValueError("supply exactly the new latest versions' required teacher bindings")
        if any(not isinstance(binding, dict) or not binding for binding in bindings.values()):
            raise ValueError("each teacher binding must be nonempty immutable metadata")

        versions = dict(self.versions)
        touched = set(selected["retired_ids"]) | {event["version_id"] for event in selected["event_plan"]}
        for vid in touched & self.versions.keys():
            versions[vid] = copy.deepcopy(self.versions[vid])
        for row in selected["created_versions"]:
            versions[row["version_id"]] = copy.deepcopy(row)
        active_after = {tuple(fact): vid for fact, vid in selected["active_after"]}
        final_ids = set(active_after.values())
        for vid in touched:
            versions[vid]["active"] = vid in final_ids
            if vid not in final_ids:
                versions[vid]["retired_batch"] = batch
        for row in selected["created_versions"]:
            vid = row["version_id"]
            if vid in bindings:
                versions[vid]["teacher_binding"] = bindings[vid]
            elif vid not in final_ids:
                versions[vid]["teacher_not_required"] = "superseded_same_batch_before_commit"
            elif terminal:
                versions[vid]["teacher_not_required"] = "terminal_no_future_replay"
        for event, record in zip(selected["event_plan"], records):
            row = versions[event["version_id"]]
            row["occurrence_case_ids"].append(record["case_id"])
            row["latest_record"] = copy.deepcopy(record)
            row["latest_order"] = event["order"]
        bank = list(selected["selected_ids"])
        pending = [] if terminal else list(selected["final_current_ids"])
        if any(vid not in final_ids for vid in bank + pending):
            raise ValueError("commit would retain a retired replay version")
        if set(bank) & set(pending) or len(bank) > self.cap or len(pending) > self.pending_capacity:
            raise ValueError("commit bank/pending capacity or disjointness violation")
        if not terminal and any(versions[vid]["teacher_binding"] is None for vid in bank + pending):
            raise ValueError("commit would publish a replay version without its original teacher")
        commit_id = _digest(dict(preparation_id=selected["preparation_id"], selected_ids=bank,
                                 teacher_bindings=bindings, terminal=terminal))
        receipt = dict(commit_id=commit_id, batch=batch, arm=self.arm,
                       preparation_id=selected["preparation_id"], occurrences_added=len(records),
                       created_versions=len(selected["created_versions"]),
                       retired_versions=len(selected["retired_ids"]), active_versions=len(final_ids),
                       bank_ids=bank, pending_ids=pending, teacher_created_ids=sorted(bindings),
                       terminal=bool(terminal), save_checkpoints=False)
        # Prepare all fallible work first; these final assignments publish one state.
        events = self.eventledger + copy.deepcopy(selected["event_plan"])
        commits = self.commit_ids + [commit_id]
        self.versions, self.eventledger = versions, events
        self._active_by_fact, self.bank_ids, self.pending_ids = active_after, bank, pending
        self.last_batch, self.commit_ids = batch, commits
        self._revision += 1
        return receipt

    def weights(self, selected_ids: Iterable[str], current_batch: int):
        ids = list(selected_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate selected version")
        return history_weights([self.versions[vid]["created_batch"] for vid in ids], current_batch,
                               recency=self.arm == "EN_ADAPT_H_GSS_REC", cap=self.cap,
                               batch_size=self.pending_capacity)

    def snapshot(self) -> dict:
        """Compact observation metadata, never a weight/M/resume artifact."""
        rows = []
        for row in sorted(self.versions.values(), key=lambda value: value["created_order"]):
            result = {key: copy.deepcopy(value) for key, value in row.items()
                      if key not in ("record", "latest_record")}
            result["representative_case_id"] = row["record"]["case_id"]
            result["latest_case_id"] = row["latest_record"]["case_id"]
            rows.append(result)
        return dict(arm=self.arm, last_batch=self.last_batch, versions=rows,
                    active_version_ids=self.active_ids(), bank_ids=list(self.bank_ids),
                    pending_ids=list(self.pending_ids), occurrences=self.occurrence_rows(),
                    commit_ids=list(self.commit_ids), save_checkpoints=False,
                    exact_crash_resume="NOT_AVAILABLE")
