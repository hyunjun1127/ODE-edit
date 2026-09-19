"""CPU-only, fail-closed audit of the frozen L4 archive (no model imports).

All case/prompt rows, bootstrap work and panel identities are local artifacts.
Only aggregate CSVs are suitable for publication. No archived bytes are edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

ARM = "AlphaEdit_L4_ONLY"
METRICS = ("RS", "PS", "NS")
CP = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
WEIGHT = "model.layers.4.mlp.down_proj.weight"
ROW_SCALARS = ("case_id", "prompt_index", "identity", "new_nll", "true_nll", "margin", "success", "new_strict", "true_strict", "new_token_correct", "new_token_count", "true_token_correct", "true_token_count")


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    with open(path) as f:
        return json.load(f)


def save(path, x):
    with open(path, "x") as f:
        json.dump(x, f, sort_keys=True, indent=2, allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    def __init__(self, source_map):
        self.source_map = Path(source_map)
        obj = read(source_map)
        mapping = obj.get("source_to_local", obj.get("source_to_destination", obj.get("files", obj)))
        if isinstance(mapping, list):
            self.mapping = {r.get("source", r.get("source_path")): r.get("destination", r.get("local_path")) for r in mapping}
        else:
            self.mapping = mapping
        self.used = {}

    def resolve(self, source=None, suffix=None, fallback=None):
        if source in self.mapping:
            value = self.mapping[source]
        else:
            candidates = [(s, d) for s, d in self.mapping.items() if isinstance(s, str) and suffix and s.endswith(suffix)]
            if not candidates and fallback:
                source, value = fallback, fallback
            else:
                require(len(candidates) == 1, f"ambiguous/missing staged input: {source or suffix}, count={len(candidates)}")
                source, value = candidates[0]
        if isinstance(value, dict):
            value = value.get("path", value.get("destination", value.get("local_path")))
        path = Path(value)
        require(path.is_file(), f"not a file: {path}")
        if str(path) not in self.used:
            self.used[str(path)] = {"source": source, "path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}
        return path


def registry_from_sample(sample, records):
    """Validate exact bytes' semantic hashes; never select/filter a new stream."""
    rows = sample["records"]
    require(len({int(r["case_id"]) for r in rows}) == len(rows), "sample duplicate case")
    data = {int(r["case_id"]): r for r in records}
    registry, identities = [], {}
    for i, r in enumerate(rows):
        cid = int(r["case_id"])
        require(cid in data, f"missing dataset case {cid}")
        raw = data[cid]
        req = raw["requested_rewrite"]
        require(digest(raw) == r["raw_record_sha256"], f"raw dataset hash {cid}")
        require(digest(req) == r["request_sha256"], f"request hash {cid}")
        require(digest(req["target_new"]) == r["target_new_sha256"] and digest(req["target_true"]) == r["target_true_sha256"], f"target hashes {cid}")
        require(int(r["ordinal"]) == i, "ordinal order")
        batch = i // 100 + 1
        require(int(r["batch_index"]) == batch, "batch order")
        new, true = req["target_new"]["str"], req["target_true"]["str"]
        registry.append({**r, "arrival_batch": batch, "relation": req["relation_id"], "target_new_string_sha256": digest(new), "target_true_string_sha256": digest(true)})
        prompt_sets = {"RS": [req["prompt"].format(req["subject"])], "PS": raw["paraphrase_prompts"], "NS": raw["neighborhood_prompts"]}
        require(tuple(len(prompt_sets[m]) for m in METRICS) == (1, 2, 10), f"prompt inventory {cid}")
        for tag, prompts in prompt_sets.items():
            for pi, prompt in enumerate(prompts):
                ident = digest([cid, pi, prompt, new, true])
                identities[tag, ident] = {"case_id": cid, "prompt_index": pi, "relation": req["relation_id"], "arrival_batch": batch, "subject_relation_group": r["subject_relation_group"], "target_new_sha256": r["target_new_sha256"], "target_true_sha256": r["target_true_sha256"]}
    reg = pd.DataFrame(registry)
    reg["group_versions"] = reg.groupby("subject_relation_group")["case_id"].transform("size")
    reg["group_changed_target"] = reg.groupby("subject_relation_group")["target_new_sha256"].transform("nunique") > 1
    reg["batch_internal_conflict"] = reg.groupby(["subject_relation_group", "arrival_batch"])["target_new_sha256"].transform("nunique") > 1
    reg["version_ordinal"] = reg.groupby("subject_relation_group").cumcount() + 1
    reg["overwrite_status"] = np.where(reg.batch_internal_conflict, "BATCH_INTERNAL_CONFLICT", np.where(reg.group_changed_target, "CHANGED_TARGET_VERSION", "NO_TARGET_CONFLICT"))
    return reg, identities


def validate_rows(metric, tag, identities):
    rows = metric["rows"]
    require(len({r["identity"] for r in rows}) == len(rows), f"duplicate identity {tag}")
    result = []
    for r in rows:
        require((tag, r["identity"]) in identities, f"unbound row {tag}/{r['identity']}")
        meta = identities[tag, r["identity"]]
        require(int(r["case_id"]) == meta["case_id"] and int(r["prompt_index"]) == meta["prompt_index"], "row metadata mismatch")
        require(all(math.isfinite(float(r[k])) for k in ("new_nll", "true_nll", "margin")), "nonfinite evaluation")
        margin = r["true_nll"] - r["new_nll"]
        require(margin == r["margin"], "stored margin mismatch")
        safety = -margin if tag == "NS" else margin
        require(bool(r["success"]) == (safety > 0), "success/tie mismatch")
        for target in ("new", "true"):
            correct, count = int(r[target + "_token_correct"]), int(r[target + "_token_count"])
            require(count > 0 and 0 <= correct <= count, "token denominator mismatch")
            require(bool(r[target + "_strict"]) == (correct == count), "strict/token mismatch")
        target = "true" if tag == "NS" else "new"
        result.append({**r, **meta, "metric_tag": tag, "safety_margin": safety, "target_strict": bool(r[target + "_strict"]), "target_token_correct": int(r[target + "_token_correct"]), "target_token_count": int(r[target + "_token_count"])})
    require(metric["denominator"] == len(rows) and metric["numerator"] == sum(r["success"] for r in rows), "stored numerator/denominator mismatch")
    require(metric["rate"] == metric["numerator"] / metric["denominator"], "stored rate mismatch")
    require(metric["bit_order_sha256"] == digest([(r["identity"], r["success"]) for r in rows]), "bit order mismatch")
    return result


def same_observation(a, b):
    return all(a[k] == b[k] for k in ROW_SCALARS)


def read_archive(inputs, contract, reg, identities):
    source = contract["paths"]["server4_companion_root"]
    allrows, anchors, states, commits, duplicated = [], {}, {}, {}, 0
    context_notes = []
    ids = reg.case_id.tolist()
    cp_manifest = Path("/mnt/raid5/janghj/ODE-edit/local/blue-checkpoint-downstream/20260909-v1/imports/initial6-v1/source/checkpoint-manifest.json")
    cpmeta = {int(x["batch"]): x for x in read(cp_manifest)["checkpoints"] if x["arm"] == ARM}
    cp_hash = sha(cp_manifest)
    last_endpoint = None
    for batch in contract["inputs"]["current_batches"]:
        def load(name):
            return read(inputs.resolve(f"{source}/B{batch:03d}/{name}"))
        current, entry, commit = load("current.json"), load("entry.json"), load("commit.json")
        batch_ids = ids[(batch-1)*100:batch*100]
        require(current["requests"] == 100 and current["request_order"] == digest(batch_ids), f"current order B{batch}")
        require(entry["request_ids"] == batch_ids and entry["seen_before"] == (batch-1)*100, f"entry order B{batch}")
        require(entry["request_hashes"] == reg.loc[reg.arrival_batch == batch, "request_sha256"].tolist(), f"entry request hashes B{batch}")
        endpoint = commit["endpoint"]
        require({"weights": current["weight_state"], "cache": current["cache_sha256"]} == endpoint, f"current state B{batch}")
        require(current["before_after_exact"] is True and current["evaluator_controller_influence"] == 0, "evaluator mutation/influence")
        require(commit["batch"] == batch and commit["status"] == "BATCH_COMMITTED" and commit["nonfinite"] == 0, "commit status")
        if last_endpoint is not None:
            require(commit["entry"] == last_endpoint, f"commit chain B{batch}")
        last_endpoint = endpoint
        signature = entry["signature"]
        entry_content = {"weights":{k:v["sha256"] for k,v in signature["weights"].items()},"cache":signature["cache_sha256"]}
        require(entry_content == commit["entry"], f"entry signature B{batch}")
        contexts = load("contexts.json")
        require(digest(contexts) == commit["context_hash"], f"committed contexts B{batch}")
        if entry["context_hash"] != commit["context_hash"]:
            require(batch == 1 and entry["context_hash"] == digest(None), "unexpected entry context change")
            context_notes.append({"batch":batch,"entry_context_hash":entry["context_hash"],"entry_matches_json_null":True,"actual_context_hash":digest(contexts),"status":"RECORDED_FIRST_ENTRY_NULL_CONTEXT_NOT_EVALUATION_PARITY_FAILURE"})
        commits[batch] = commit
        states[batch] = endpoint
        now = {}
        for tag in METRICS:
            rows = validate_rows(current["metrics"][tag], tag, identities)
            expected_ids = [(cid, pi) for cid in batch_ids for pi in range({"RS":1,"PS":2,"NS":10}[tag])]
            require([(r["case_id"], r["prompt_index"]) for r in rows] == expected_ids, f"metric order B{batch}/{tag}")
            require({k:v for k,v in current["metrics"][tag].items() if k != "rows"} == commit["current"][tag], "commit metric summary")
            for r in rows:
                key = (tag, r["identity"])
                require(key not in anchors, "anchor duplicate")
                r = {**r, "arm": ARM, "checkpoint_batch": batch, "edit_age_batches": 0, "source_kind": "current", "weight_sha256": endpoint["weights"][WEIGHT], "cache_sha256": endpoint["cache"]}
                anchors[key] = r
                now[key] = r
        if batch in contract["inputs"]["seen_full_batches"]:
            seen = load("seen-full.json")
            require(seen["state"] == endpoint and seen["requests"] == batch*100, f"seen state B{batch}")
            require(seen["evaluation_type"] == "CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS", "wrong seen evaluation type")
            require(cpmeta[batch]["weights"][WEIGHT]["sha256"] == endpoint["weights"][WEIGHT] and cpmeta[batch]["method_state"]["sha256"] == endpoint["cache"], "checkpoint catalog state")
            seen_rows = []
            for tag in METRICS:
                rows = validate_rows(seen["metrics"][tag], tag, identities)
                expect = [(cid, pi) for cid in ids[:batch*100] for pi in range({"RS":1,"PS":2,"NS":10}[tag])]
                require([(r["case_id"], r["prompt_index"]) for r in rows] == expect, "seen ordered inventory")
                for r in rows:
                    key = (tag, r["identity"])
                    if key in now:
                        require(same_observation(now[key], r), "current/seen duplicate discrepancy")
                        duplicated += 1
                    seen_rows.append({**r, "arm": ARM, "checkpoint_batch": batch, "edit_age_batches": batch-r["arrival_batch"], "source_kind": "current+seen" if key in now else "seen", "weight_sha256": endpoint["weights"][WEIGHT], "cache_sha256": endpoint["cache"]})
            allrows.extend(seen_rows)
        else:
            allrows.extend(now.values())
    long = pd.DataFrame(allrows)
    require(not long.duplicated(["arm", "checkpoint_batch", "metric_tag", "identity"]).any(), "observation duplicate")
    require(len(anchors) == len(reg)*13, "at-write inventory")
    b1 = long[long.checkpoint_batch == 1]
    for tag, (n,d) in contract["functional"]["b1_archive_expected"].items():
        g = b1[b1.metric_tag == tag]
        require(len(g) == d and int(g.success.sum()) == n, f"B1 archive expected {tag}")
    return long, pd.DataFrame(anchors.values()), {"states": states, "checkpoint_catalog_path":str(cp_manifest), "checkpoint_catalog_sha256":cp_hash, "checkpoint_metadata_link": "catalog exact state/tensor SHA; physical tensor scan delegated to geometry stage", "current_seen_scalar_exact_duplicates":duplicated, "commits": commits, "context_notes":context_notes}


def conflict_masks(reg, start_batch, end_batch, variable_start=False):
    """Active version is diagnostic; ambiguous same-batch targets have no winner."""
    # Most groups are singletons. Avoid thousands of pandas slices per cohort.
    free = dict(zip(reg.case_id, reg.arrival_batch.le(end_batch)))
    active = free.copy()
    repeated = reg[reg.group_versions > 1]
    for _, group in repeated.groupby("subject_relation_group", sort=False):
        versions = list(group.itertuples())
        available = [r for r in versions if r.arrival_batch <= end_batch]
        latest = max((r.arrival_batch for r in available), default=-1)
        latest_rows = [r for r in available if r.arrival_batch == latest]
        unambiguous = len({r.target_new_sha256 for r in latest_rows}) == 1
        for r in versions:
            start = int(r.arrival_batch) if variable_start else int(start_batch)
            # Include the active target at interval entry as well as all versions
            # arriving inside the interval; never infer array order within a batch.
            at_start = [v for v in available if v.arrival_batch <= start]
            baseline_batch = max((v.arrival_batch for v in at_start), default=start)
            relevant = {v.target_new_sha256 for v in available if v.arrival_batch >= baseline_batch}
            free[r.case_id] = bool(len(relevant) == 1 and r.target_new_sha256 in relevant)
            active[r.case_id] = bool(unambiguous and r.arrival_batch == latest and not r.batch_internal_conflict)
    return free, active


def join_pair(left, right):
    keys = ["arm", "metric_tag", "identity", "case_id"]
    out = left.merge(right, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    require(len(out) == len(left), "matched cohort lost rows")
    require(out.target_token_count_a.equals(out.target_token_count_b), "target token lengths changed at same identity")
    return out


def summarize_pair(pair):
    a, b = pair.success_a.to_numpy(bool), pair.success_b.to_numpy(bool)
    n11, n10, n01, n00 = (int(np.count_nonzero(v)) for v in (a&b, a&~b, ~a&b, ~a&~b))
    sa, sb = pair.target_strict_a.to_numpy(bool), pair.target_strict_b.to_numpy(bool)
    return {"rows":len(pair), "requests":int(pair.case_id.nunique()), "n11":n11, "n10":n10, "n01":n01, "n00":n00, "start_success":n11+n10, "end_success":n11+n01, "loss_numerator":n10, "loss_denominator":n11+n10, "gain_numerator":n01, "gain_denominator":n01+n00, "net_success_delta":n01-n10, "margin_delta_mean":float((pair.safety_margin_b-pair.safety_margin_a).mean()) if len(pair) else None, "strict_start":int(sa.sum()), "strict_end":int(sb.sum()), "strict_lost":int((sa&~sb).sum()), "strict_gained":int((~sa&sb).sum()), "token_correct_start":int(pair.target_token_correct_a.sum()), "token_correct_end":int(pair.target_token_correct_b.sum()), "token_denominator_start":int(pair.target_token_count_a.sum()), "token_denominator_end":int(pair.target_token_count_b.sum())}


def bootstrap_pair(pair, cluster, replicates=2000, seed=20260920):
    """Exact cluster resampling. The same draws carry all scalar/strict rows.

    Membership-keyed RNG reuses draws for equal cohorts/metric times, so the
    resampled request carries all P/N rows and all observations together.
    """
    if pair.empty:
        return {"clusters":0,"status":"EMPTY"}
    labels = pair[cluster].astype(str)
    vals = pd.DataFrame({"cluster":labels, "n":1.0, "start":pair.success_a.astype(float), "end":pair.success_b.astype(float), "lost":(pair.success_a & ~pair.success_b).astype(float), "gained":(~pair.success_a & pair.success_b).astype(float), "margin":pair.safety_margin_b-pair.safety_margin_a, "strict_delta":pair.target_strict_b.astype(float)-pair.target_strict_a.astype(float)})
    grouped = vals.groupby("cluster", sort=True).sum()
    arrays = grouped.to_numpy(np.float64)
    n = len(arrays)
    stable_seed = int(digest([seed, cluster, grouped.index.tolist()])[:16], 16)
    rng = np.random.default_rng(stable_seed)
    outputs = []
    for offset in range(0, replicates, 32):
        picks = rng.integers(0, n, size=(min(32, replicates-offset), n))
        sums = arrays[picks].sum(axis=1)
        den, start, end, lost, gained, margin, strict = sums.T
        with np.errstate(divide="ignore", invalid="ignore"):
            outputs.append(np.stack([(end-start)/den, lost/start, gained/(den-start), margin/den, strict/den], axis=1))
    draws = np.concatenate(outputs)
    result = {"clusters":n, "replicates":replicates,"seed":seed,"resample_seed":str(stable_seed),"status":"CONDITIONAL_SINGLE_STREAM"}
    for i, name in enumerate(("net_rate", "loss_rate", "gain_rate", "margin_delta_mean", "strict_delta_rate")):
        finite = draws[np.isfinite(draws[:,i]),i]
        result[name+"_finite_replicates"] = len(finite)
        result[name+"_lo"] = float(np.quantile(finite, .025)) if len(finite) else None
        result[name+"_hi"] = float(np.quantile(finite, .975)) if len(finite) else None
    return result


def joint_rows(long):
    p = long[long.metric_tag.isin(("RS","PS"))]
    groups = p.groupby(["arm","checkpoint_batch","case_id"], sort=False)
    require(groups.size().eq(3).all(), "joint requires RS and two PS")
    rows = groups.agg(success=("success","all"),target_strict=("target_strict","all"),safety_margin=("safety_margin","min"),target_token_correct=("target_token_correct","sum"),target_token_count=("target_token_count","sum"),arrival_batch=("arrival_batch","first"),subject_relation_group=("subject_relation_group","first")).reset_index()
    rows["metric_tag"] = "RP_JOINT"
    rows["identity"] = rows.case_id.map(lambda cid:digest(["RP_JOINT",int(cid)]))
    rows["edit_age_batches"] = rows.checkpoint_batch-rows.arrival_batch
    return rows


def first_failures(long, anchors):
    observations = long.sort_values("checkpoint_batch").groupby(["metric_tag","identity"],sort=False)
    anchor_map = anchors.set_index(["metric_tag","identity"])
    rows = []
    for key, group in observations:
        a = anchor_map.loc[key]
        first_batch = int(a.arrival_batch)
        later = group[group.checkpoint_batch > first_batch]
        status, lower, upper, recovery = "AT_WRITE_FAILURE", None, first_batch, None
        if bool(a.success):
            failed = later[~later.success]
            if len(failed):
                upper = int(failed.iloc[0].checkpoint_batch)
                prior = group[group.checkpoint_batch < upper]
                lower = int(prior.checkpoint_batch.max())
                status = "INTERVAL_CENSORED_FIRST_OBSERVED_FAILURE"
            else:
                status, lower, upper = "RIGHT_CENSORED_NO_OBSERVED_FAILURE", int(group.checkpoint_batch.max()), None
        recovered = later[later.success]
        if not bool(a.success) and len(recovered):
            recovery = int(recovered.iloc[0].checkpoint_batch)
        rows.append({"case_id":int(a.case_id),"metric_tag":key[0],"identity":key[1],"arrival_batch":first_batch,"at_write_success":bool(a.success),"first_failure_status":status,"failure_after_batch":lower,"failure_observed_by_batch":upper,"first_recovery_observed_batch":recovery})
    return pd.DataFrame(rows)


def mechanism_panel(long, reg, seed=20260920):
    out, exclusions = [], []
    for a,b in ((1,10),(50,100)):
        left = long[(long.checkpoint_batch==a)&(long.metric_tag=="NS")&(long.arrival_batch==1)]
        right = long[(long.checkpoint_batch==b)&(long.metric_tag=="NS")&(long.arrival_batch==1)]
        require(len(left)==1000 and len(right)==1000,"mechanism first100 NS1000")
        pair = join_pair(left,right)
        free,_ = conflict_masks(reg,a,b)
        pair["conflict_free"] = pair.case_id.map(free)
        for row in pair[~pair.conflict_free].itertuples():
            exclusions.append({"interval_start":a,"interval_end":b,"identity":row.identity,"case_id":row.case_id,"reason":"INTERVAL_TARGET_CONFLICT"})
        pair = pair[pair.conflict_free].copy()
        pair["seed_hash"] = pair.identity.map(lambda ident:digest([seed,a,b,ident]))
        lost = pair[pair.success_a & ~pair.success_b].sort_values("seed_hash").head(16)
        retained = pair[pair.success_a & pair.success_b].copy()
        chosen = set()
        for rank,l in enumerate(lost.itertuples()):
            def emit(r, role, match=None):
                return {"interval_start":a,"interval_end":b,"selection_role":role,"selection_rank":rank,"case_id":r.case_id,"identity":r.identity,"prompt_index":int(r.prompt_index_a),"paired_lost_identity":l.identity,"arrival_batch":1,"metric_tag":"NS","entry_margin":float(r.safety_margin_a),"endpoint_margin":float(r.safety_margin_b),"entry_success":bool(r.success_a),"endpoint_success":bool(r.success_b),"new_target_tokens":int(r.new_token_count_a),"true_target_tokens":int(r.true_token_count_a),"relation":r.relation_a,"seed_hash":r.seed_hash,"target_length_mismatch":None if match is None else int(match[0]),"relation_mismatch":None if match is None else int(match[1]),"entry_margin_absolute_difference":None if match is None else float(match[2]),"population":"FIRST100_NS1000","selection_is_population_estimator":False}
            out.append(emit(l,"lost"))
            choices=[]
            for r in retained.itertuples():
                if r.identity in chosen:
                    continue
                key=(int((r.new_token_count_a,r.true_token_count_a)!=(l.new_token_count_a,l.true_token_count_a)),int(r.relation_a!=l.relation_a),abs(r.safety_margin_a-l.safety_margin_a),r.seed_hash)
                choices.append((key,r))
            if choices:
                key,r=min(choices,key=lambda z:z[0]); chosen.add(r.identity)
                out.append(emit(r,"matched_retained",key))
        exclusions.append({"interval_start":a,"interval_end":b,"identity":None,"case_id":None,"reason":"PANEL_COUNTS","candidate_lost":int((pair.success_a & ~pair.success_b).sum()),"candidate_retained":int((pair.success_a & pair.success_b).sum()),"selected_lost":len(lost),"selected_retained":len(chosen)})
    return pd.DataFrame(out),pd.DataFrame(exclusions)


def aggregate_analysis(long, anchors, reg, replicates=2000):
    combined = pd.concat([long,joint_rows(long)],ignore_index=True)
    anchor_combined = pd.concat([anchors,joint_rows(anchors)],ignore_index=True)
    transitions, cis, atwrite = [], [], []
    for (batch,tag),g in anchor_combined.groupby(["arrival_batch","metric_tag"],sort=True):
        atwrite.append({"arrival_batch":int(batch),"metric_tag":tag,"numerator":int(g.success.sum()),"denominator":len(g),"strict_numerator":int(g.target_strict.sum()),"token_correct":int(g.target_token_correct.sum()),"token_denominator":int(g.target_token_count.sum()),"safety_margin_mean":float(g.safety_margin.mean()),"safety_margin_q05":float(g.safety_margin.quantile(.05)),"anchor":"ACTUAL_ARRIVAL_CURRENT_JSON"})
    # Cohort draws use identical membership-keyed seeds across all observed times.
    for idx,b in enumerate(CP):
        end = combined[combined.checkpoint_batch==b]
        specs=[("at_write",0,anchor_combined[anchor_combined.arrival_batch<=b],True),("first100",1,combined[(combined.checkpoint_batch==1)&(combined.arrival_batch==1)],False)]
        if b>=10:
            specs.append(("first1000_from_B10",10,combined[(combined.checkpoint_batch==10)&(combined.arrival_batch<=10)],False))
        if idx:
            a=CP[idx-1];specs.append(("adjacent_checkpoint_matched",a,combined[combined.checkpoint_batch==a],False))
        for arrival in range(1,b+1):
            specs.append(("age",arrival,anchor_combined[anchor_combined.arrival_batch==arrival],True))
        for cohort,a,start,var_start in specs:
            pair=join_pair(start,end)
            free,active=conflict_masks(reg,a,b,var_start)
            for view,mask in (("original_all_case",np.ones(len(pair),bool)),("interval_target_conflict_free",pair.case_id.map(free).to_numpy(bool)),("separate_active_version",pair.case_id.map(active).to_numpy(bool))):
                viewed=pair[mask]
                for tag,g in viewed.groupby("metric_tag",sort=False):
                    meta={"cohort":cohort,"view":view,"start_batch":a,"end_batch":b,"age_batches":b-a if cohort=="age" else None,"metric_tag":tag,"anchor_is_arrival_current":var_start,"joint_margin_definition":"minimum RS/PS safety margin" if tag=="RP_JOINT" else "paired_target_mean_NLL"}
                    transitions.append({**meta,**summarize_pair(g)})
                    # All requested age/cohort comparisons get cluster intervals.
                    for cluster in ("case_id","subject_relation_group_a"):
                        cis.append({**meta,"cluster":cluster,**bootstrap_pair(g,cluster,replicates)})
        print(json.dumps({"stage":"archival_transitions","checkpoint":b,"summary_rows":len(transitions)}),flush=True)
    return pd.DataFrame(transitions),pd.DataFrame(cis),pd.DataFrame(atwrite)


def run(source_map, output, contract_path=None, bootstrap_replicates=2000):
    started=time.monotonic()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    contract_path=Path(contract_path or Path(__file__).resolve().parents[3]/"plans/global/2026-09-20-server2-checkpoint-mechanism-audit-contract-v1.json")
    contract=read(contract_path)
    require(bootstrap_replicates==contract["functional"]["bootstrap_replicates"],"bootstrap scientific contract")
    inputs=Inputs(source_map)
    try:
        sample=read(inputs.resolve(contract["paths"]["sample_source_server4"],suffix="/sample.lock.json"))
        require(sample["ordered_root"]==contract["identity"]["sample_root"],"sample root")
        dataset=read(inputs.resolve(suffix="/data/counterfact/counterfact.json",fallback=contract["paths"]["dataset"]))
        registry,identities=registry_from_sample(sample,dataset)
        require(len(registry)==10000,"sample10000")
        registry.to_csv(output/"request_registry.csv",index=False)
        counts={"requests":len(registry),"groups":int(registry.subject_relation_group.nunique()),"repeated_groups":int((registry.groupby("subject_relation_group").size()>1).sum()),"earlier_versions":len(registry)-int(registry.subject_relation_group.nunique()),"groups_with_changed_target":int(registry.groupby("subject_relation_group").target_new_sha256.nunique().gt(1).sum()),"within_batch_conflict_group_batches":int(registry.groupby(["subject_relation_group","arrival_batch"]).target_new_sha256.nunique().gt(1).sum())}
        require(counts==contract["functional"]["observed_group_counts"],"overwrite census differs")
        long,anchors,binding=read_archive(inputs,contract,registry,identities)
        long.to_parquet(output/"functional_long.parquet",index=False)
        anchors.to_parquet(output/"at_write_anchors.parquet",index=False)
        save(output/"state-bindings.json",binding)
        print(json.dumps({"stage":"ARCHIVE_IDENTITY_PASS","rows":len(long),"anchors":len(anchors),"duplicate_rows_verified":binding["current_seen_scalar_exact_duplicates"]}),flush=True)
        panel,excluded=mechanism_panel(long,registry)
        panel.to_csv(output/"mechanism_panel.csv",index=False)
        excluded.to_csv(output/"mechanism_panel_exclusions.csv",index=False)
        first_failures(long,anchors).to_parquet(output/"first_observed_failure.parquet",index=False)
        transitions,cis,atwrite=aggregate_analysis(long,anchors,registry,bootstrap_replicates)
        transitions.to_csv(output/"paired_transitions.csv",index=False)
        cis.to_csv(output/"paired_bootstrap.csv",index=False)
        atwrite.to_csv(output/"at_write_outcomes.csv",index=False)
        parents=long[(long.case_id.isin(panel.case_id.unique())) & long.metric_tag.isin(("RS","PS"))]
        parents.to_parquet(output/"mechanism_parent_rp.parquet",index=False)
        artifacts=[{"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(output.iterdir()) if p.is_file()]
        receipt={"status":"PASS","scope":"STORED_ARCHIVE_CPU_ONLY_NOT_MODEL_PARITY","arm":ARM,"observations":len(long),"at_write_anchors":len(anchors),"checkpoint_batches":list(CP),"overwrite_counts":counts,"panel_counts":panel.groupby(["interval_start","interval_end","selection_role"]).size().rename("rows").reset_index().to_dict("records"),"bootstrap_replicates":bootstrap_replicates,"bootstrap_seed":20260920,"W0_evaluation":"NOT_IN_ARCHIVE_STAGE","current_seen_duplicates":binding["current_seen_scalar_exact_duplicates"],"source_map":{"path":str(source_map),"sha256":sha(source_map)},"contract":{"path":str(contract_path),"sha256":sha(contract_path)},"inputs":list(inputs.used.values()),"outputs":artifacts,"wall_seconds":time.monotonic()-started,"new_model_forwards":0,"gpu":0,"checkpoint_saved":False}
        save(output/"archival-receipt.json",receipt)
        return receipt
    except Exception as exc:
        save(output/"archival-failure.json",{"status":"FAILED","error_type":type(exc).__name__,"error":str(exc),"wall_seconds":time.monotonic()-started,"inputs":list(inputs.used.values()),"gpu":0,"new_model_forwards":0})
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--source-map",required=True);p.add_argument("--output",required=True);p.add_argument("--contract")
    a=p.parse_args(); print(json.dumps(run(a.source_map,a.output,a.contract),sort_keys=True))


if __name__=="__main__":
    main()
