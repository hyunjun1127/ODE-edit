"""EP accepted-label ledger. Failed requested intentions never supersede.

This module stores identities and scalar acceptance evidence only. It does not
write models/history or infer acceptance from a requested target. Runtime data
are private and must remain local; compact counts/hash receipts are raw-free.
"""
from copy import deepcopy
import hashlib
import json
import math


class LedgerError(RuntimeError):
    pass


def _case_id(record):
    value = record.get("case_id")
    if type(value) not in (int, str):
        raise LedgerError("CASE_ID_NOT_EXACT_JSON_INT_OR_STRING")
    return value


def request_identity(record):
    """Extract exact CounterFact bytes; no casing/whitespace normalization."""
    rewrite = record.get("requested_rewrite", record)
    subject, relation = rewrite.get("subject"), rewrite.get("relation_id")
    target = rewrite.get("target_new")
    if isinstance(target, dict):
        target = target.get("str")
    if not all(isinstance(x, str) for x in (subject, relation, target)):
        raise LedgerError("EXACT_SUBJECT_RELATION_TARGET_REQUIRED")
    return {"case_id": _case_id(record), "subject": subject,
            "relation_id": relation, "target": target}


def _key(event):
    return event["subject"], event["relation_id"]


def _json_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class AcceptedLedger:
    """Append-only requested events and separately accepted events.

    Active annotations refer to the latest *accepted target string* for each
    exact (subject, relation) key. Same-target reissues remain active, matching
    the campaign rule; failed requests do not alter that latest target.
    """

    schema = "EP_TW1_ACCEPTED_LABEL_LEDGER_V1"

    def __init__(self):
        self.requested_events = []
        self.accepted_events = []
        self.batch_receipts = []

    @property
    def next_ordinal(self):
        return len(self.requested_events)

    @property
    def processed_batches(self):
        return len(self.batch_receipts)

    @property
    def accepted_ids(self):
        return [x["case_id"] for x in self.accepted_events]

    @property
    def accepted_case_ids(self):
        return self.accepted_ids

    def commit_batch(self, records, strict_ids, per_request_nll, *, batch_index,
                     endpoint_identity, expected_count=100):
        """Validate whole input before one atomic in-memory append.

        ``per_request_nll`` is a mapping by exact case ID, from the *selected*
        actual endpoint. Strict membership is externally evaluated, not inferred
        from NLL magnitude. Caller invokes this after its one native finalizer.
        A repeated batch/case ID is an error rather than a second finalization.
        """
        if type(batch_index) is not int or batch_index != self.processed_batches + 1:
            raise LedgerError("BATCH_NOT_EXACT_NEXT_OR_DUPLICATE_FINALIZATION")
        if len(records) != expected_count or expected_count <= 0:
            raise LedgerError("ALL_REQUESTED_DENOMINATOR")
        if not isinstance(endpoint_identity, dict) or not endpoint_identity:
            raise LedgerError("SELECTED_ENDPOINT_IDENTITY_REQUIRED")
        # Also reject non-JSON/nonfinite identity before any mutation.
        _json_hash(endpoint_identity)
        identities = [request_identity(x) for x in records]
        ids = [x["case_id"] for x in identities]
        old_ids = {x["case_id"] for x in self.requested_events}
        if len(set(ids)) != len(ids) or old_ids.intersection(ids):
            raise LedgerError("DUPLICATE_CASE_ID_IN_CHAIN")
        successes = set(strict_ids)
        if not successes.issubset(ids):
            raise LedgerError("STRICT_ID_OUTSIDE_BATCH")
        if set(per_request_nll) != set(ids):
            raise LedgerError("NLL_IDENTITY_MISSING_OR_EXTRA")
        losses = {case: float(per_request_nll[case]) for case in ids}
        if not all(math.isfinite(x) for x in losses.values()):
            raise LedgerError("SELECTED_ENDPOINT_NLL_NONFINITE")
        requested, accepted = [], []
        offset = self.next_ordinal
        for index, identity in enumerate(identities):
            case = identity["case_id"]
            event = dict(identity, ordinal=offset + index, batch_index=batch_index,
                         selected_strict_success=case in successes,
                         selected_request_nll=losses[case])
            requested.append(event)
            if case in successes:
                accepted.append(dict(identity, ordinal=offset + index,
                                     acceptance_batch=batch_index,
                                     acceptance_loss=losses[case],
                                     endpoint_identity=deepcopy(endpoint_identity)))
        receipt = {
            "batch_index": batch_index, "ordinal_start": offset,
            "ordinal_stop": offset + expected_count, "requested_count": expected_count,
            "accepted_count": len(accepted), "unaccepted_count": expected_count - len(accepted),
            "selected_endpoint": deepcopy(endpoint_identity),
            "ordered_request_identity_sha256": _json_hash(identities),
            "accepted_event_sha256": _json_hash(accepted),
            "failed_requested_intent_changes_active_target": False,
        }
        self.requested_events.extend(requested)
        self.accepted_events.extend(accepted)
        self.batch_receipts.append(receipt)
        return deepcopy(receipt)

    def latest_accepted_targets(self):
        latest = {}
        for event in self.accepted_events:
            latest[_key(event)] = event["target"]
        return latest

    def annotated_accepted(self):
        latest = self.latest_accepted_targets()
        return [dict(deepcopy(x), status=("ACTIVE_TARGET" if latest[_key(x)] == x["target"]
                                        else "SUPERSEDED")) for x in self.accepted_events]

    def annotate_requested(self):
        """ALL requested denominator, with UNKNOWN where no accepted key exists."""
        latest = self.latest_accepted_targets()
        return [dict(deepcopy(x), status=("UNKNOWN" if _key(x) not in latest else
                     "ACTIVE_TARGET" if latest[_key(x)] == x["target"] else "SUPERSEDED"))
                for x in self.requested_events]

    def annotate_records(self, records):
        """Annotate arbitrary requested identities against accepted targets.

        A status is not a new acceptance decision; ``previously_accepted`` is
        separate so failed same-target intentions are never treated as success.
        """
        latest, accepted = self.latest_accepted_targets(), set(self.accepted_ids)
        result = []
        for record in records:
            event = request_identity(record)
            status = ("UNKNOWN" if _key(event) not in latest else
                      "ACTIVE_TARGET" if latest[_key(event)] == event["target"] else "SUPERSEDED")
            result.append(dict(event, status=status, previously_accepted=event["case_id"] in accepted))
        return result

    def to_dict(self):
        return deepcopy({"schema": self.schema, "requested_events": self.requested_events,
                         "accepted_events": self.accepted_events, "batch_receipts": self.batch_receipts,
                         "next_ordinal": self.next_ordinal, "processed_batches": self.processed_batches})

    def state_hash(self):
        return _json_hash(self.to_dict())

    snapshot = to_dict

    @classmethod
    def from_dict(cls, payload):
        """Rebuild by validated chronological event replay, without model work."""
        if payload.get("schema") != cls.schema:
            raise LedgerError("LEDGER_SCHEMA")
        ledger = cls()
        events = payload.get("requested_events", [])
        receipts = payload.get("batch_receipts", [])
        for receipt in receipts:
            start, stop = receipt["ordinal_start"], receipt["ordinal_stop"]
            if start != ledger.next_ordinal or stop <= start:
                raise LedgerError("LEDGER_CHRONOLOGY")
            batch = events[start:stop]
            if [x["ordinal"] for x in batch] != list(range(start, stop)):
                raise LedgerError("LEDGER_EVENT_ORDINALS")
            records = [{"case_id": x["case_id"], "subject": x["subject"],
                        "relation_id": x["relation_id"], "target_new": x["target"]} for x in batch]
            ledger.commit_batch(records, {x["case_id"] for x in batch if x["selected_strict_success"]},
                                {x["case_id"]: x["selected_request_nll"] for x in batch},
                                batch_index=receipt["batch_index"],
                                endpoint_identity=receipt["selected_endpoint"], expected_count=stop-start)
        if ledger.to_dict() != payload:
            raise LedgerError("LEDGER_RESTORED_CONTENT_MISMATCH")
        return ledger

    restore = from_dict
