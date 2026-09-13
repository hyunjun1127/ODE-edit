"""CPU-only, independent E01 completed-observation reducer.

No runtime/evaluator imports. Canonical identities are recomputed from the sealed
dataset, not row positions. Original raw margins remain true-minus-new; NS
desired margins are the opposite sign. All quantiles use linear interpolation.
Output is aggregate-only; this module never publishes prompts or per-case rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import unicodedata

DATASET_SHA = "3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1"
MULT = {"RS": 1, "PS": 2, "NS": 10}


class IntegrityError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise IntegrityError(message)


def member_read(path, inputs, expected=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "input not regular: " + str(path))
    before = path.stat()
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected:
        require(sha == expected, "input SHA mismatch: " + str(path))
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), "input mutated while read")
    inputs[str(path)] = dict(path=str(path), bytes=len(raw), sha256=sha,
                             mode=oct(after.st_mode & 0o777), stat_unchanged=True)
    return json.loads(raw)


def canonical_metadata(records):
    meta, identities = {}, {}
    for ordinal, rec in enumerate(records):
        cid = rec["case_id"]
        require(cid not in meta, "duplicate dataset case")
        r = rec["requested_rewrite"]
        subject = " ".join(unicodedata.normalize("NFC", r["subject"]).split())
        meta[cid] = dict(ordinal=ordinal, fact=(subject, r["relation_id"]),
                         target=r["target_new"]["str"])
        prompts = dict(RS=[r["prompt"].format(r["subject"])],
                       PS=rec["paraphrase_prompts"], NS=rec["neighborhood_prompts"])
        for metric, values in prompts.items():
            require(len(values) == MULT[metric], "dataset prompt multiplicity")
            for pi, text in enumerate(values):
                identities[(metric, cid, pi)] = digest(
                    [cid, pi, text, r["target_new"]["str"], r["target_true"]["str"]])
    return meta, identities


def versions(meta, n):
    groups = {}
    for cid, r in meta.items():
        if r["ordinal"] < n:
            groups.setdefault(r["fact"], []).append(cid)
    out = {}
    for ids in groups.values():
        ids.sort(key=lambda cid: meta[cid]["ordinal"])
        for j, cid in enumerate(ids):
            later = ids[j+1:]
            out[cid] = ("active" if not later else "superseded_conflicting_target"
                        if any(meta[q]["target"] != meta[cid]["target"] for q in later)
                        else "superseded_same_target")
    return out


def index_rows(rows, metric, identities=None):
    out = {}
    for r in rows:
        k = (r["case_id"], r["prompt_index"], r["identity"])
        require(k not in out, "duplicate prompt")
        if identities is not None:
            require(identities.get((metric, k[0], k[1])) == k[2], "prompt/target identity mismatch")
        require(all(math.isfinite(r[f]) for f in ("new_nll", "true_nll", "margin")), "nonfinite core")
        require(r["margin"] == r["true_nll"] - r["new_nll"], "raw margin convention mismatch")
        desired = -r["margin"] if metric == "NS" else r["margin"]
        require(type(r["success"]) is bool and r["success"] == (desired > 0), "strict tie/direction mismatch")
        for side in ("new", "true"):
            count, correct = r[side+"_token_count"], r[side+"_token_correct"]
            require(isinstance(count, int) and 0 <= correct <= count and count > 0, "token count invalid")
            require(r[side+"_strict"] == (count == correct), "strict/token inconsistency")
        out[k] = dict(r, desired_margin=desired)
    return out


def validate_raw(raw, case_ids, identities):
    require(raw["requests"] == len(case_ids), "request count mismatch")
    if "request_order" in raw:
        require(raw["request_order"] == digest(case_ids), "request order header mismatch")
    out = {}
    for metric, mult in MULT.items():
        block = raw["metrics"][metric]
        rows = block["rows"]
        require([(r["case_id"], r["prompt_index"]) for r in rows] ==
                [(cid, i) for cid in case_ids for i in range(mult)], "canonical order/multiplicity mismatch")
        out[metric] = index_rows(rows, metric, identities)
        require(block["denominator"] == len(rows) == len(case_ids)*mult, "raw denominator mismatch")
        require(block["numerator"] == sum(r["success"] for r in rows), "raw numerator mismatch")
        require(block["rate"] == block["numerator"]/block["denominator"], "raw rate mismatch")
    return out


def quantile(sorted_values, q):
    p = (len(sorted_values)-1)*q
    lo, hi = math.floor(p), math.ceil(p)
    return sorted_values[lo] + (p-lo)*(sorted_values[hi]-sorted_values[lo])


def stats(values):
    values = sorted(values)
    if not values:
        return {k: None for k in ("mean", "median", "p90", "p95", "p99", "min", "max")}
    return dict(mean=statistics.fmean(values), median=quantile(values, .5),
                p90=quantile(values, .9), p95=quantile(values, .95),
                p99=quantile(values, .99), min=values[0], max=values[-1])


def reduce_pair(a, b, *, endpoint, panel, metric):
    require(a.keys() == b.keys(), "pair support mismatch")
    labels = dict(endpoint=endpoint, panel=panel, metric=metric)
    keys = list(a)
    n = len(keys)
    orig = sum(a[k]["success"] for k in keys)
    replay = sum(b[k]["success"] for k in keys)
    loss = sum(a[k]["success"] and not b[k]["success"] for k in keys)
    gain = sum(not a[k]["success"] and b[k]["success"] for k in keys)
    both = sum(a[k]["success"] and b[k]["success"] for k in keys)
    failed = n-loss-gain-both
    require(replay-orig == gain-loss, "transition conservation")
    perf = dict(labels, requests=len({k[0] for k in keys}), denominator=n,
                baseline_numerator=orig, replay_numerator=replay,
                baseline_rate=orig/n if n else None, replay_rate=replay/n if n else None,
                delta_pp=100*(replay-orig)/n if n else None,
                baseline_ties=sum(a[k]["margin"] == 0 for k in keys),
                replay_ties=sum(b[k]["margin"] == 0 for k in keys),
                comparison="SAME_ENDPOINT_BASELINE_TO_REPLAY", status="OBSERVED" if n else "EMPTY_METADATA_SUBSET")
    trans = dict(labels, denominator=n, baseline_success_denominator=orig,
                 lost=loss, gained=gain, unchanged_success=both, unchanged_failure=failed,
                 loss_rate_among_baseline_success=loss/orig if orig else None,
                 paired_identity_root=digest(sorted(keys)))
    for side in ("new", "true"):
        for who, rows in (("baseline", a), ("replay", b)):
            correct = sum(rows[k][side+"_token_correct"] for k in keys)
            count = sum(rows[k][side+"_token_count"] for k in keys)
            strict = sum(rows[k][side+"_strict"] for k in keys)
            perf.update({who+"_"+side+"_strict_numerator": strict,
                         who+"_"+side+"_strict_denominator": n,
                         who+"_"+side+"_token_correct": correct,
                         who+"_"+side+"_token_denominator": count})
    dist = []
    for field in ("new_nll", "true_nll", "desired_margin"):
        x = [a[k][field] for k in keys]
        y = [b[k][field] for k in keys]
        for who, values in (("baseline", x), ("replay", y), ("paired_delta", [yy-xx for xx, yy in zip(x, y)])):
            dist.append(dict(labels, quantity=field, population=who, denominator=n, **stats(values)))
    return perf, trans, dist


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def run(root, dataset, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs, endpoint_rows, cohort_rows, transitions, distributions, checks = {}, [], [], [], [], []
    records = member_read(dataset, inputs, DATASET_SHA)
    require(len(records) == 10000, "fixed dataset size")
    meta, identities = canonical_metadata(records)
    for old, label in ((5000, "Middle-B060"), (9000, "Late-B100")):
        n = old+1000; base = root/f"n{old}"
        terminal = member_read(base/"output/terminal.json", inputs)
        lock = member_read(base/"input.lock.json", inputs)
        def sealed(m):
            value = member_read(m["path"], inputs, m["sha256"])
            require(inputs[m["path"]]["bytes"] == m["bytes"], "sealed bytes mismatch")
            return value
        original = sealed(terminal["baseline"]["seen-full"])
        replay = sealed(terminal["fullseen"])
        current = sealed(terminal["reuse_current"])
        original_current = sealed(terminal["baseline"]["current"])
        cases = [r["case_id"] for r in records[:n]]
        oi = validate_raw(original, cases, identities)
        ri = validate_raw(replay, cases, identities)
        ci = validate_raw(current, cases[-100:], identities)
        oci = validate_raw(original_current, cases[-100:], identities)
        require(replay["endpoint_weight_sha256"] == lock["expected_W"] and
                replay["endpoint_history_sha256"] == lock["expected_M"], "published endpoint W/M mismatch")
        require(terminal["evaluated_past_requests"] == n-100 and terminal["reused_current_requests"] == 100,
                "reuse cardinality mismatch")
        require(terminal["new_native_batches"] == terminal["new_z"] == terminal["new_history_append"] == 0,
                "observation run reports native action")
        merged, offset = {metric: {} for metric in MULT}, 0
        for receipt_member in terminal["shards"]:
            rec = sealed(receipt_member)
            require(rec["begin"] == offset and rec["end"] > offset, "shard gap/overlap")
            shard = sealed(rec["raw"])
            idx = validate_raw(shard, cases[offset:rec["end"]], identities)
            require(rec["requests"] == rec["end"]-offset and rec["prompt_pairs"] == 13*rec["requests"], "shard accounting")
            guard = rec["guard"]
            require(all(guard[k] is True for k in ("RNG_exact", "all_parameter_pointer_versions_exact", "weight_history_bytes_exact")), "shard guard false")
            require(guard["weight_sha256"] == lock["expected_W"] and guard["history_sha256"] == lock["expected_M"], "shard endpoint mismatch")
            require(guard["writer_calls"] == guard["z_calls"] == guard["history_append"] == 0, "shard action count")
            for metric in MULT:
                require(not (merged[metric].keys() & idx[metric].keys()), "duplicate shard prompts")
                merged[metric].update(idx[metric])
            offset = rec["end"]
        require(offset == n-100, "incomplete past shard range")
        for metric in MULT:
            require(not (merged[metric].keys() & ci[metric].keys()), "current duplicated in past shards")
            merged[metric].update(ci[metric])
            require(merged[metric] == ri[metric], "fullseen differs from exact shards+current")
            require(all(oi[metric][k] == v for k, v in oci[metric].items()), "original current reuse mismatch")
        versions_at_endpoint = versions(meta, n)
        panels = [("fullseen", set(cases)), ("Current100", set(cases[-100:])),
                  ("entry_old", set(cases[:old])), ("new_window1000", set(cases[old:]))]
        panels += [(f"cohort_B{batch:03d}", set(cases[(batch-1)*100:batch*100])) for batch in range(1, n//100+1)]
        panels += [(status, {cid for cid, v in versions_at_endpoint.items() if v == status})
                   for status in ("active", "superseded_conflicting_target", "superseded_same_target")]
        for panel, subset in panels:
            for metric in MULT:
                a = {k: r for k, r in oi[metric].items() if k[0] in subset}
                b = {k: r for k, r in ri[metric].items() if k[0] in subset}
                perf, trans, dist = reduce_pair(a, b, endpoint=label, panel=panel, metric=metric)
                (endpoint_rows if panel in ("fullseen", "Current100", "entry_old", "new_window1000") else cohort_rows).append(perf)
                transitions.append(trans); distributions.extend(dist)
        checks.append(dict(endpoint=label, requests=n, total_prompt_pairs=13*n,
            shard_count=len(terminal["shards"]), past_requests=offset, reused_current=100,
            identity_failures=0, nonfinite=0, duplicate_prompt_keys=0, shard_gap_overlap=0,
            final_equals_shards_plus_reused_current=True, original_current_equals_original_final_subset=True,
            raw_margin_and_strict_direction=True, original_header_absence_handled_by_dataset_identity=True,
            active_requests=sum(v == "active" for v in versions_at_endpoint.values()),
            superseded_conflicting_target_requests=sum(v == "superseded_conflicting_target" for v in versions_at_endpoint.values()),
            superseded_same_target_requests=sum(v == "superseded_same_target" for v in versions_at_endpoint.values()),
            endpoint_weight_sha256=lock["expected_W"], endpoint_history_sha256=lock["expected_M"]))
    for filename, rows in (("endpoint-performance.csv", endpoint_rows), ("cohort-performance.csv", cohort_rows),
                           ("paired-transitions.csv", transitions), ("distributions.csv", distributions)):
        write_csv(output/filename, rows)
    # A second byte read verifies immutability for all inputs consumed, including every shard.
    for p, identity in inputs.items():
        require(hashlib.sha256(Path(p).read_bytes()).hexdigest() == identity["sha256"], "input changed after reduction")
        identity["after_sha256"] = identity["sha256"]
    result = dict(schema="E01_INDEPENDENT_COMPLETED_REDUCTION_V1", checks=checks,
        input_members=sorted(inputs.values(), key=lambda r:r["path"]),
        input_member_root=digest(sorted(inputs.values(), key=lambda r:r["path"])),
        matching="metric + case_id + prompt_index + SHA256(canonical prompt/new/true target bytes)",
        quantiles="linear interpolation at (n-1)*q; all rows, no weighting or filtering",
        active_rule="NFC+whitespace subject and relation; latest ordinal in observed prefix; later different target classified separately",
        active_is_metadata_not_adjudicated_legitimate_overwrite=True,
        first_failure_time="NOT_OBSERVED_BETWEEN_CHECKPOINTS",
        atwrite_replay_fullprefix="NOT_RECORDED; this reducer only compares same-final-endpoint original and replay",
        input_hash_before_after_unchanged=True, imputation=0, new_evaluation=0,
        scientific_promotion=False, native_trajectory_equivalence=False)
    write_json(output/"reduction-checks.json", result)
    return result


def run_atwrite(root, dataset, output):
    """Bounded same-L4 original at-write joins and *recorded* replay at-write.

    Original at-write is a fixed reference for both final endpoints, never
    relabelled replay's own at-write. Replay first/last batch observations are
    partial and are not expanded to an unobserved native10-batch history.
    """
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    inputs, tables, coverage = {}, [], []
    records = member_read(dataset, inputs, DATASET_SHA)
    meta, identities = canonical_metadata(records)
    original_cache = {}
    for old, label in ((5000, "Middle-B060"), (9000, "Late-B100")):
        n = old+1000
        terminal = member_read(root/f"n{old}/output/terminal.json", inputs)
        cases = [r["case_id"] for r in records[:n]]
        originals = Path(terminal["baseline"]["current"]["path"]).parent.parent
        source_rows = {metric:{} for metric in MULT}
        available, missing = [], []
        for batch in range(1, n//100+1):
            p = originals/f"B{batch:03d}/current.json"
            if not p.is_file():
                missing.extend(cases[(batch-1)*100:batch*100]); continue
            if p not in original_cache:
                obj = member_read(p, inputs)
                original_cache[p] = validate_raw(obj, cases[(batch-1)*100:batch*100], identities)
            for metric in MULT:
                require(not (source_rows[metric].keys() & original_cache[p][metric].keys()), "atwrite duplicate")
                source_rows[metric].update(original_cache[p][metric])
            available.extend(cases[(batch-1)*100:batch*100])
        finals = {}
        for who, member in (("original_final", terminal["baseline"]["seen-full"]),
                            ("replay_final", terminal["fullseen"])):
            finals[who] = validate_raw(member_read(member["path"], inputs, member["sha256"]), cases, identities)
        # Native program saves first B100 and terminal Current100 only. The
        # latter is read from its original source, not manufactured from final.
        replay_sources = [("first_native_batch", Path(terminal["original_native_attempt"])/"output"/f"B{old//100+1:03d}"/"NATIVE-current.json", cases[old:old+100]),
                          ("terminal_current", Path(terminal["reuse_current"]["path"]), cases[-100:])]
        own = {metric:{} for metric in MULT}; own_available = []
        for _, p, ids in replay_sources:
            if p.is_file():
                obj = member_read(p, inputs)
                idx = validate_raw(obj, ids, identities)
                for metric in MULT:
                    require(not (own[metric].keys() & idx[metric].keys()), "own atwrite duplicate")
                    own[metric].update(idx[metric])
                own_available.extend(ids)
        statuses = versions(meta, n)
        panels = [("fullseen", set(cases)), ("entry_old", set(cases[:old])),
                  ("new_window1000", set(cases[old:])), ("Current100", set(cases[-100:]))]
        panels += [(s, {cid for cid, v in statuses.items() if v == s}) for s in
                   ("active", "superseded_conflicting_target", "superseded_same_target")]
        for reference_name, before, before_cases, final_names in (
                ("ORIGINAL_OWN_ATWRITE", source_rows, set(available), ("original_final", "replay_final")),
                ("REPLAY_OWN_RECORDED_ATWRITE", own, set(own_available), ("replay_final",))):
            for panel, subset in panels:
                selected = subset & before_cases
                for final_name in final_names:
                    for metric in MULT:
                        a = {k:r for k, r in before[metric].items() if k[0] in selected}
                        b = {k:r for k, r in finals[final_name][metric].items() if k[0] in selected}
                        p, t, _ = reduce_pair(a, b, endpoint=label, panel=panel, metric=metric)
                        tables.append(dict(endpoint=label, panel=panel, metric=metric,
                            atwrite_reference=reference_name, final_endpoint=final_name,
                            comparison=("SAME_ORIGINAL_TRAJECTORY_ATWRITE_TO_FINAL" if final_name == "original_final" else
                                        "FIXED_ORIGINAL_ATWRITE_REFERENCE_TO_REPLAY_FINAL_NOT_REPLAY_OWN_RETENTION" if reference_name == "ORIGINAL_OWN_ATWRITE" else
                                        "REPLAY_OWN_OBSERVED_ATWRITE_TO_REPLAY_FINAL_PARTIAL"),
                            planned_requests=len(subset), observed_requests=len(selected),
                            missing_atwrite_requests=len(subset-selected),
                            planned_prompt_denominator=len(subset)*MULT[metric], denominator=p["denominator"],
                            atwrite_numerator=p["baseline_numerator"], final_numerator=p["replay_numerator"],
                            atwrite_rate=p["baseline_rate"], final_rate=p["replay_rate"],
                            delta_pp=p["delta_pp"], lost=t["lost"], gained=t["gained"],
                            unchanged_success=t["unchanged_success"], unchanged_failure=t["unchanged_failure"],
                            atwrite_success_denominator=t["baseline_success_denominator"],
                            loss_rate_among_atwrite_success=t["loss_rate_among_baseline_success"],
                            status="OBSERVED_COMPLETE" if len(selected) == len(subset) and subset else
                                   "OBSERVED_PARTIAL" if selected else "NOT_MEASURED_OR_EMPTY_METADATA_SUBSET",
                            paired_identity_root=t["paired_identity_root"]))
        coverage.append(dict(endpoint=label, original_atwrite_requests=len(available),
            original_missing_atwrite_requests=len(missing), replay_own_atwrite_observed_requests=len(own_available),
            replay_own_atwrite_missing_prefix_requests=n-len(own_available),
            replay_own_atwrite_missing_new_window_requests=1000-len(own_available),
            replay_own_saved_batches=[old//100+1, n//100],
            intermediate_first_failure_time="NOT_RECORDED_NO_INTERPOLATION"))
    write_csv(output/"atwrite-transitions.csv", tables)
    for p, identity in inputs.items():
        require(hashlib.sha256(Path(p).read_bytes()).hexdigest() == identity["sha256"], "atwrite input changed")
        identity["after_sha256"] = identity["sha256"]
    result = dict(schema="E01_ATWRITE_DESCRIPTIVE_REFERENCE_JOIN_V1", coverage=coverage,
        input_members=sorted(inputs.values(), key=lambda r:r["path"]),
        missing_not_imputed=True, original_atwrite_is_not_replay_atwrite=True,
        supersession_is_metadata_not_causal_adjudication=True, new_evaluation=0, source_reexecution=0)
    write_json(output/"atwrite-checks.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atwrite-only", action="store_true")
    args = parser.parse_args()
    result = (run_atwrite if args.atwrite_only else run)(args.root, args.dataset, args.output)
    print(json.dumps(result["coverage" if args.atwrite_only else "checks"], indent=2))
