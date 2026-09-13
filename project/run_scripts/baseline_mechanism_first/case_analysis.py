"""Exact recorded-row E1-A joins and descriptive strata (no model evaluation).

Raw source margin is true_nll-new_nll for RS, PS AND NS. Desired margin is
separate and sign-flipped only for NS. No aggregate-to-case reconstruction.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import unicodedata

from .evidence import canonical_sha


class CaseIntegrityError(ValueError):
    pass


MULTIPLICITY = {"RS": 1, "PS": 2, "NS": 10}


def normalize_row(row: dict, *, metric: str, metadata: dict) -> dict:
    """Copy an original row; explicit metadata must identify its execution scope."""
    required = ("family", "arm", "layer", "checkpoint_requests", "source_identity")
    if any(k not in metadata for k in required) or metric not in MULTIPLICITY:
        raise CaseIntegrityError("missing execution metadata or metric")
    if any(k not in row for k in ("case_id", "prompt_index", "identity", "new_nll", "true_nll", "margin", "success")):
        raise CaseIntegrityError("per-prompt original rows are required, not aggregate summaries")
    if not row["identity"]:
        raise CaseIntegrityError("missing prompt+target identity")
    if not isinstance(row["success"], bool):
        raise CaseIntegrityError("success must be a recorded boolean")
    new, true, raw = (float(row[k]) for k in ("new_nll", "true_nll", "margin"))
    if not all(math.isfinite(x) for x in (new, true, raw)):
        raise CaseIntegrityError("nonfinite original metric")
    if abs(raw - (true - new)) > 1e-12:
        raise CaseIntegrityError("raw margin differs from source true-new convention")
    desired = -raw if metric == "NS" else raw
    if row["success"] != (desired > 0):
        raise CaseIntegrityError("success disagrees with strict preference; ties fail")
    for side in ("new", "true"):
        count_key, correct_key, strict_key = f"{side}_token_count", f"{side}_token_correct", f"{side}_strict"
        if count_key in row and correct_key in row:
            if not 0 <= row[correct_key] <= row[count_key] or row[count_key] < 1:
                raise CaseIntegrityError("invalid token denominator")
            if strict_key in row and row[strict_key] != (row[correct_key] == row[count_key]):
                raise CaseIntegrityError("TF strict mismatch")
    result = dict(row)
    for key, value in metadata.items():
        if key in result and result[key] != value:
            raise CaseIntegrityError(f"metadata conflicts with original {key}")
        result[key] = value
    result.update(metric=metric, raw_margin=raw, desired_margin=desired,
                  desired_target="true" if metric == "NS" else "new", tie=raw == 0)
    return result


def rows_from_evaluation(raw: dict, *, case_ids: list, metadata: dict) -> list[dict]:
    if raw.get("requests") != len(case_ids) or len(set(case_ids)) != len(case_ids):
        raise CaseIntegrityError("request count/unique inventory mismatch")
    output = []
    for metric, mult in MULTIPLICITY.items():
        rows = raw["metrics"][metric]["rows"]
        expected = [(case, i) for case in case_ids for i in range(mult)]
        if [(r["case_id"], r["prompt_index"]) for r in rows] != expected:
            raise CaseIntegrityError("exact request/prompt order mismatch")
        if len({r["identity"] for r in rows}) != len(rows):
            raise CaseIntegrityError("duplicate prompt identity")
        output.extend(normalize_row(r, metric=metric, metadata=metadata) for r in rows)
    return output


def prompt_key(row: dict) -> tuple:
    return row["metric"], row["case_id"], row["prompt_index"]


def _index(rows):
    result = {}
    for row in rows:
        key = prompt_key(row)
        if key in result:
            raise CaseIntegrityError(f"duplicate case/prompt key {key}")
        result[key] = row
    return result


def attach_case_metadata(rows: list[dict], *, case_metadata: list[dict],
                         sample_identity: str) -> list[dict]:
    """Exact case metadata join, with no prompt/body copied into derived output.

    Metadata must already bind case_id, ordinal, relation and fact_identity to the
    sealed sample. Caller supplies recorded token counts; no tokenizer/model run.
    """
    if not sample_identity:
        raise CaseIntegrityError("missing sealed sample identity")
    fields = ("ordinal", "relation", "fact_identity")
    meta = {}
    for record in case_metadata:
        if record["case_id"] in meta or any(k not in record for k in fields):
            raise CaseIntegrityError("duplicate or incomplete case metadata")
        meta[record["case_id"]] = record
    result = []
    for row in rows:
        if row["case_id"] not in meta:
            raise CaseIntegrityError("case missing from sealed metadata")
        rec = meta[row["case_id"]]
        if any(k in row and row[k] != rec[k] for k in fields):
            raise CaseIntegrityError("case metadata identity conflict")
        if "checkpoint_requests" in row and rec["ordinal"] >= row["checkpoint_requests"]:
            raise CaseIntegrityError("future case at observed checkpoint")
        result.append(dict(row, **{k:rec[k] for k in fields}, sample_identity=sample_identity))
    return result


def exact_pair(atwrite: list[dict], checkpoint: list[dict]) -> tuple[list[dict], dict]:
    """One arm/family at-write prefix vs one recorded checkpoint; no row invention."""
    left, right = _index(atwrite), _index(checkpoint)
    pairs = []
    for key in left.keys() & right.keys():
        a, b = left[key], right[key]
        for name in ("identity", "family", "arm", "layer", "source_identity"):
            if a[name] != b[name]:
                raise CaseIntegrityError(f"pair binding differs: {name} at {key}")
        if a["checkpoint_requests"] > b["checkpoint_requests"]:
            raise CaseIntegrityError("checkpoint precedes at-write")
        pairs.append(dict(metric=key[0], case_id=key[1], prompt_index=key[2], identity=a["identity"],
                          family=a["family"], arm=a["arm"], layer=a["layer"],
                          write_checkpoint=a["checkpoint_requests"], checkpoint_requests=b["checkpoint_requests"],
                          age_requests=b["checkpoint_requests"]-a["checkpoint_requests"],
                          atwrite_success=a["success"], checkpoint_success=b["success"],
                          atwrite_raw_margin=a["raw_margin"], raw_margin=b["raw_margin"],
                          atwrite_desired_margin=a["desired_margin"], desired_margin=b["desired_margin"],
                          desired_margin_delta=b["desired_margin"]-a["desired_margin"],
                          new_nll_delta=b["new_nll"]-a["new_nll"], true_nll_delta=b["true_nll"]-a["true_nll"],
                          transition=("retained" if b["success"] else "lost") if a["success"] else
                                     ("recovery" if b["success"] else "initial_and_current_failure")))
    pairs.sort(key=lambda r: (r["metric"], str(r["case_id"]), r["prompt_index"]))
    return pairs, dict(atwrite_denominator=len(left), checkpoint_denominator=len(right), joined=len(pairs),
                       missing_atwrite=[list(k) for k in sorted(right.keys()-left.keys(), key=str)],
                       missing_checkpoint=[list(k) for k in sorted(left.keys()-right.keys(), key=str)],
                       imputation=0, population_replacement=False)


def fact_versions(records: list[dict], *, checkpoint_requests: int) -> list[dict]:
    """Metadata-only NFC/whitespace fact identity; prefix-active latest ordinal wins.

    Inputs explicitly provide ordinal/case_id/subject/relation/target_new_sha256.
    Same-target reissues and conflicting-target overwrites remain distinguishable.
    """
    norm = lambda s: " ".join(unicodedata.normalize("NFC", s).split())
    ordered = sorted(records, key=lambda r: r["ordinal"])
    if len({r["ordinal"] for r in ordered}) != len(ordered) or len({r["case_id"] for r in ordered}) != len(ordered):
        raise CaseIntegrityError("duplicate ordinal/case metadata")
    groups = defaultdict(list)
    for r in ordered:
        groups[(norm(r["subject"]), r["relation"])].append(r)
    result = []
    for fact, versions in groups.items():
        seen = [r for r in versions if r["ordinal"] < checkpoint_requests]
        for number, r in enumerate(seen, 1):
            later = [q for q in seen if q["ordinal"] > r["ordinal"]]
            result.append(dict(case_id=r["case_id"], ordinal=r["ordinal"],
                               fact_identity=canonical_sha(fact), version_in_seen_prefix=number,
                               active_at_checkpoint=not later,
                               superseded_by_case_id=later[0]["case_id"] if later else None,
                               later_different_target_in_seen_prefix=any(q["target_new_sha256"] != r["target_new_sha256"] for q in later),
                               checkpoint_requests=checkpoint_requests))
    return sorted(result, key=lambda r:r["ordinal"])


def first_observed_failure(observations: list[dict]) -> dict:
    """Interval-censored observations for ONE prompt; never infer hidden flips."""
    if not observations:
        return dict(status="NO_OBSERVATIONS", first_observed_failure_checkpoint=None)
    rows = sorted(observations, key=lambda r:r["checkpoint_requests"])
    if len({r["checkpoint_requests"] for r in rows}) != len(rows):
        raise CaseIntegrityError("duplicate checkpoint observation")
    for row in rows:
        if prompt_key(row) != prompt_key(rows[0]) or row["identity"] != rows[0]["identity"]:
            raise CaseIntegrityError("mixed prompt history")
        if any(row[k] != rows[0][k] for k in ("family", "arm", "layer", "source_identity")):
            raise CaseIntegrityError("mixed execution history")
    failure = next((i for i,r in enumerate(rows) if not r["success"]), None)
    return dict(status="NO_OBSERVED_FAILURE" if failure is None else "FIRST_OBSERVATION_FAILED" if failure == 0 else "INTERVAL_CENSORED_FAILURE",
                first_observed_failure_checkpoint=None if failure is None else rows[failure]["checkpoint_requests"],
                lower_last_observed_success=None if failure in (None,0) else rows[failure-1]["checkpoint_requests"],
                upper_first_observed_failure=None if failure is None else rows[failure]["checkpoint_requests"],
                observed_recovery_checkpoints=[] if failure is None else [r["checkpoint_requests"] for r in rows[failure+1:] if r["success"]],
                unobserved_transitions_imputed=0)


@dataclass(frozen=True)
class MatchingSpec:
    """Caller predeclares finite cutpoints; bins include -inf/+inf outside them."""
    margin_cutpoints: tuple[float, ...]
    age_cutpoints: tuple[int, ...]
    token_length_cutpoints: tuple[int, ...]
    declaration_identity: str

    def __post_init__(self):
        if not self.declaration_identity:
            raise CaseIntegrityError("matching specification must be predeclared")
        for points in (self.margin_cutpoints, self.age_cutpoints, self.token_length_cutpoints):
            if any(not math.isfinite(p) for p in points) or tuple(sorted(set(points))) != tuple(points):
                raise CaseIntegrityError("invalid predeclared cutpoints")


def common_support(groups: dict[str, list[dict]], *, spec: MatchingSpec) -> dict:
    """Return all-population, common-success and common-support masks separately.

    Group rows are same-prompt paired outputs plus explicit relation/new_token_count.
    Strata use only AT-WRITE desired margin, age and metadata, never final outcome.
    No matching weights, resampling or outcomes are constructed. Shared occupied
    strata need not contain the same cases; common-success is exact-prompt paired.
    """
    if not groups:
        raise CaseIntegrityError("no groups")
    indices = {name:_index(rows) for name,rows in groups.items()}
    common = set.intersection(*(set(rows) for rows in indices.values()))
    for key in common:
        if len({rows[key]["identity"] for rows in indices.values()}) != 1:
            raise CaseIntegrityError("cross-group prompt/target identity mismatch")
        if len({rows[key]["family"] for rows in indices.values()}) != 1:
            raise CaseIntegrityError("cross-family matching needs a separately declared comparison")
        for field in ("relation", "new_token_count", "age_requests", "checkpoint_requests"):
            if len({rows[key][field] for rows in indices.values()}) != 1:
                raise CaseIntegrityError(f"cross-group matching metadata differs: {field}")
    success = {key for key in common if all(rows[key]["atwrite_success"] for rows in indices.values())}
    strata = {}
    for name, rows in indices.items():
        buckets = defaultdict(list)
        for key, row in rows.items():
            values = row["atwrite_desired_margin"], row["age_requests"], row["new_token_count"]
            if not all(math.isfinite(v) for v in values):
                raise CaseIntegrityError("nonfinite matching variable")
            bucket = (row["relation"], bisect_right(spec.margin_cutpoints, values[0]),
                      bisect_right(spec.age_cutpoints, values[1]), bisect_right(spec.token_length_cutpoints, values[2]))
            buckets[bucket].append(key)
        strata[name] = buckets
    supported = set.intersection(*(set(x) for x in strata.values()))
    union = set.union(*(set(x) for x in strata.values()))
    results = {}
    for name, rows in indices.items():
        selected = {k for s in supported for k in strata[name][s]}
        n = len(rows)
        results[name] = dict(all_population_denominator=n,
                            common_prompt_denominator=len(common), common_atwrite_success_denominator=len(success),
                            common_support_denominator=len(selected), excluded_count=n-len(selected),
                            excluded_rate=(n-len(selected))/n if n else None,
                            common_atwrite_success_keys=sorted(success,key=str), common_support_keys=sorted(selected,key=str),
                            empty_in_this_group=[s for s in sorted(union,key=str) if s not in strata[name]])
    return dict(groups=results, declaration_identity=spec.declaration_identity,
                supported_strata=sorted(supported,key=str), all_strata=sorted(union,key=str),
                interpretation="DESCRIPTIVE_POST_TREATMENT_STRATIFICATION_NOT_CAUSAL",
                primary_population_unchanged=True, matching_uses_final_outcomes=False)
