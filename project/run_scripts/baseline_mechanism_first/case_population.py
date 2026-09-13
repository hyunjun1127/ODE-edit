"""E1-A streamed, identity-exact descriptive analysis of sealed singleton rows.

No model/tokenizer import, evaluation, inference of unobserved outcomes, or
matching on final outcomes. Large paired prompt ledgers stay in local storage.
Bins are fixed before any outcome files are consumed and published as a lock.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import time

from .case_analysis import (CaseIntegrityError, MatchingSpec, common_support,
                            exact_pair, fact_versions, prompt_key,
                            rows_from_evaluation)
from .evidence import canonical_sha

DATASET_SHA = "3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1"
MATCHING = dict(margin_cutpoints=[-2, 0, .5, 1, 2, 4, 8],
                age_cutpoints=[1000, 3000, 6000, 9000],
                token_length_cutpoints=[2, 4, 8],
                primary_population="ALL_ORIGINAL_PROMPTS",
                common_success="EXACT_PROMPT_ATWRITE_SUCCESS_IN_ALL_FIVE_LAYERS",
                support="INTERSECTION_OF_OCCUPIED_RELATION_MARGIN_AGE_NEW_TOKEN_BINS",
                edges="bisect_right: cutpoint belongs to bin above it",
                selection_uses_final_outcomes=False,
                status="EXPLORATORY_DESCRIPTIVE_NOT_PREREGISTERED_EXPERIMENT_GATE")


def source_digest(value):
    """Original BLUE evaluator digest uses ensure_ascii=True, not canonical_sha."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as f:
        f.write(data)


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metadata(records):
    """Return hashed subject/fact identities; no raw prompt survives this join."""
    from unicodedata import normalize
    result, identities = {}, {}
    for ordinal, record in enumerate(records):
        case = int(record["case_id"])
        if case in result:
            raise CaseIntegrityError("duplicate canonical case")
        r = record["requested_rewrite"]
        subject = " ".join(normalize("NFC", r["subject"]).split())
        result[case] = dict(ordinal=ordinal, relation=r["relation_id"],
                            subject_sha256=canonical_sha(subject),
                            fact_identity=canonical_sha((subject, r["relation_id"])))
        groups = {"RS": [r["prompt"].format(r["subject"])],
                  "PS": record["paraphrase_prompts"], "NS": record["neighborhood_prompts"]}
        for metric, prompts in groups.items():
            if len(prompts) != {"RS": 1, "PS": 2, "NS": 10}[metric]:
                raise CaseIntegrityError("canonical prompt multiplicity mismatch")
            for i, prompt in enumerate(prompts):
                identities[(metric, case, i)] = source_digest(
                    [case, i, prompt, r["target_new"]["str"], r["target_true"]["str"]])
    return result, identities


def read_rows(member, case_ids, scope, identities, input_manifest):
    path = Path(member.get("local_reuse") or member["destination"])
    before = path.stat()
    data = path.read_bytes()
    if len(data) != member["bytes"] or hashlib.sha256(data).hexdigest() != member["sha256"]:
        raise CaseIntegrityError(f"sealed raw changed: {path}")
    raw = json.loads(data)
    if Path(member["source_path"]).name == "seen-full.json":
        # Original lifelong.evaluation.merge publishes the endpoint and merged
        # rows, not current.json's request_order/instrumentation fields.
        if (raw.get("evaluation_type") != "CHECKPOINT_FINAL_W_ON_ALL_SEEN_REQUESTS"
                or raw.get("current_rows_reused") is not True
                or not raw.get("state", {}).get("weights") or not raw["state"].get("cache")):
            raise CaseIntegrityError("original seen-full merge identity missing")
    else:
        if raw["request_order"] != source_digest(case_ids):
            raise CaseIntegrityError("source request order differs from fixed prefix")
        if raw.get("before_after_exact") is not True or raw.get("evaluator_controller_influence") != 0:
            raise CaseIntegrityError("original evaluator integrity marker missing")
    rows = rows_from_evaluation(raw, case_ids=case_ids, metadata=scope)
    for row in rows:
        if identities[prompt_key(row)] != row["identity"]:
            raise CaseIntegrityError("canonical prompt/target bytes digest mismatch")
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise CaseIntegrityError("input changed during read")
    input_manifest.append(dict(path=str(path), source_path=member["source_path"],
                               bytes=len(data), sha256=member["sha256"],
                               mode=oct(after.st_mode & 0o777), rows=len(rows),
                               before_after_stat_equal=True))
    return rows


def join_rows(atwrite, observed, meta, versions):
    paired, gate = exact_pair(atwrite, observed)
    if gate["missing_atwrite"] or gate["missing_checkpoint"]:
        raise CaseIntegrityError("incomplete prefix prompt join")
    aidx, bidx = ({prompt_key(r): r for r in rows} for rows in (atwrite, observed))
    for row in paired:
        key = prompt_key(row)
        a, b = aidx[key], bidx[key]
        row.update(meta[row["case_id"]])
        row.update({k: v for k, v in versions[row["case_id"]].items()
                    if k not in ("case_id", "ordinal", "fact_identity", "checkpoint_requests")})
        for name in ("new_nll", "true_nll", "new_strict", "true_strict", "new_token_count", "true_token_count"):
            row[name] = b[name]
            row["atwrite_" + name] = a[name]
        if (a["new_token_count"], a["true_token_count"]) != (b["new_token_count"], b["true_token_count"]):
            raise CaseIntegrityError("atwrite/final target token count differs")
    return paired


def summarize(rows, **labels):
    rows = list(rows)
    n = len(rows)
    count = Counter(r["transition"] for r in rows)
    before = sum(r["atwrite_success"] for r in rows)
    after = sum(r["checkpoint_success"] for r in rows)
    result = dict(labels, denominator=n, atwrite_numerator=before,
                  final_numerator=after, atwrite_rate=before/n if n else None,
                  final_rate=after/n if n else None,
                  retained=count["retained"], lost=count["lost"],
                  recovery=count["recovery"], initial_and_current_failure=count["initial_and_current_failure"],
                  entry_success_denominator=before, loss_rate_among_entry_success=count["lost"]/before if before else None)
    for name in ("atwrite_desired_margin", "desired_margin", "desired_margin_delta", "new_nll", "true_nll",
                 "new_nll_delta", "true_nll_delta"):
        values = sorted(float(r[name]) for r in rows)
        result[name + "_mean"] = statistics.fmean(values) if n else None
        result[name + "_median"] = statistics.median(values) if n else None
        result[name + "_p90"] = values[max(0, math.ceil(.9*n)-1)] if n else None
    return result


def interval_update(state, row):
    """Only observations at stored checkpoints, including the actual at-write row."""
    n, success = row["checkpoint_requests"], row["success"]
    if state is None:
        return dict(last_checkpoint=n, last_success=success, observations=1,
                    first_observed_failure=n if not success else None,
                    lower_success=None, first_observation_failed=not success,
                    observed_recoveries=0, observed_losses=0)
    if n < state["last_checkpoint"]:
        raise CaseIntegrityError("observation time goes backwards")
    if n == state["last_checkpoint"]:
        if success != state["last_success"]:
            raise CaseIntegrityError("same checkpoint has conflicting success")
        return state
    if not success and state["first_observed_failure"] is None:
        state["first_observed_failure"] = n
        state["lower_success"] = state["last_checkpoint"]
    state["observed_recoveries"] += int(success and not state["last_success"])
    state["observed_losses"] += int(not success and state["last_success"])
    state.update(last_checkpoint=n, last_success=success, observations=state["observations"]+1)
    return state


def interval_status(state):
    if state["first_observation_failed"]:
        return "FIRST_OBSERVATION_FAILED"
    return "NO_OBSERVED_FAILURE" if state["first_observed_failure"] is None else "INTERVAL_CENSORED_FAILURE"


def stratum(row, spec):
    return (row["relation"], bisect_right(spec.margin_cutpoints, row["atwrite_desired_margin"]),
            bisect_right(spec.age_cutpoints, row["age_requests"]),
            bisect_right(spec.token_length_cutpoints, row["new_token_count"]))


def compare_layers(groups, spec):
    """Same-family exact prompt comparisons; shared-stratum rates are unweighted."""
    matching = common_support(groups, spec=spec)
    summaries, bins, comparisons = [], [], []
    indices = {name: {prompt_key(r): r for r in rows} for name, rows in groups.items()}
    shared = set(map(tuple, matching["supported_strata"]))
    for name, rows in groups.items():
        stats = matching["groups"][name]
        support = set(map(tuple, stats["common_support_keys"]))
        common_success = set(map(tuple, stats["common_atwrite_success_keys"]))
        for subset, selected in (("all", rows), ("common_atwrite_success", [r for r in rows if prompt_key(r) in common_success]),
                                 ("common_support", [r for r in rows if prompt_key(r) in support]),
                                 ("active", [r for r in rows if r["active_at_checkpoint"]]),
                                 ("superseded", [r for r in rows if not r["active_at_checkpoint"]]),
                                 ("superseded_same_target_only", [r for r in rows if not r["active_at_checkpoint"] and not r["later_different_target_in_seen_prefix"]]),
                                 ("superseded_different_target", [r for r in rows if not r["active_at_checkpoint"] and r["later_different_target_in_seen_prefix"]])):
            summaries.append(summarize(selected, arm=name, subset=subset,
                                       original_denominator=len(rows),
                                       excluded_count=len(rows)-len(selected),
                                       excluded_rate=(len(rows)-len(selected))/len(rows)))
        buckets = defaultdict(list)
        for row in rows:
            buckets[stratum(row, spec)].append(row)
        for s in matching["all_strata"]:
            selected = buckets[tuple(s)]
            bins.append(summarize(selected, arm=name, relation=s[0], margin_bin=s[1], age_bin=s[2],
                                  new_token_length_bin=s[3], common_support=tuple(s) in shared,
                                  empty_bin=not selected))
    names = sorted(groups, key=lambda name: int(re.search(r"_L(\d+)_", name)[1]))
    for i, left in enumerate(names):
        for right in names[i+1:]:
            if "_L4_" not in left and "_L5_" not in left:
                continue
            keys = set(indices[left]) & set(indices[right])
            for subset in ("all_exact_paired", "common_atwrite_success"):
                use = keys if subset == "all_exact_paired" else {
                    k for k in keys if all(idx[k]["atwrite_success"] for idx in indices.values())}
                differences = [int(indices[right][k]["checkpoint_success"])-int(indices[left][k]["checkpoint_success"]) for k in use]
                n = len(differences)
                comparisons.append(dict(left=left, right=right, subset=subset, denominator=n,
                                        right_minus_left_success_rate=sum(differences)/n if n else None,
                                        right_better=sum(x>0 for x in differences), equal=sum(x==0 for x in differences),
                                        right_worse=sum(x<0 for x in differences)))
    return summaries, bins, comparisons, matching


def deterministic_gzip_jsonl(path, rows):
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            for row in rows:
                gz.write((json.dumps(row, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))+"\n").encode())


def analyze(plan_path, dataset_path, output, private):
    started = time.monotonic()
    output, private = Path(output), Path(private)
    output.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    # This lock is create-once, before any current/seen-full metric file is read.
    write_json(output/"matching.lock.json", MATCHING)
    spec = MatchingSpec(tuple(MATCHING["margin_cutpoints"]), tuple(MATCHING["age_cutpoints"]),
                        tuple(MATCHING["token_length_cutpoints"]), canonical_sha(MATCHING))
    plan = json.loads(Path(plan_path).read_text())
    cat_path = Path(plan["catalog"]["path"])
    if sha_file(cat_path) != plan["catalog"]["sha256"]:
        raise CaseIntegrityError("catalog mismatch")
    catalog = json.loads(cat_path.read_text())
    if sha_file(dataset_path) != DATASET_SHA:
        raise CaseIntegrityError("fixed10k SHA mismatch")
    records = json.loads(Path(dataset_path).read_text())
    if len(records) != 10000:
        raise CaseIntegrityError("fixed10k count mismatch")
    meta, identities = metadata(records)
    version_input = [dict(case_id=r["case_id"], ordinal=i, subject=r["requested_rewrite"]["subject"],
                          relation=r["requested_rewrite"]["relation_id"],
                          target_new_sha256=canonical_sha(r["requested_rewrite"]["target_new"]["str"])) for i,r in enumerate(records)]
    versions = {n: {r["case_id"]: r for r in fact_versions(version_input, checkpoint_requests=n)}
                for n in (100,500,*range(1000,10001,1000))}
    inputs, checkpoint_summary, interval_summary, ledgers, source_bindings = [], [], [], [], []
    family_final = defaultdict(dict)
    for arm in sorted(catalog["arms"], key=lambda a: a["config"]["arm"]):
        config = arm["config"]
        name = config["arm"]
        family = name.split("_")[0]
        layer = int(re.search(r"_L(\d+)_", name)[1])
        source_identity = canonical_sha({k:config[k] for k in ("runtime_sha256", "source_archive_sha256", "config_sha256", "model_revision")})
        source_bindings.append(dict(arm=name, family=family, layer=layer, source_identity=source_identity,
                                    **{k:config[k] for k in ("runtime_sha256", "source_head", "source_tree", "blue_head", "config_sha256", "model_revision")}))
        selected = [m for m in plan["members"] if m["kind"] == "E1_METRIC" and m["source_path"].startswith(config["raw_root"]+"/")]
        files = {(int(re.search(r"/B(\d+)/", m["source_path"])[1]), Path(m["source_path"]).name): m for m in selected}
        if len(selected) != 112 or set(b for b,f in files if f == "current.json") != set(range(1,101)):
            raise CaseIntegrityError("incomplete 100-current/12-seen inventory")
        atwrite, histories = [], {}
        # Interleave actual at-write records with stored seen checkpoints.
        for batch in range(1,101):
            n = batch*100
            scope = dict(family=family, arm=name, layer=layer, checkpoint_requests=n, source_identity=source_identity)
            current = read_rows(files[batch,"current.json"], [r["case_id"] for r in records[n-100:n]], scope, identities, inputs)
            atwrite.extend(current)
            for row in current:
                histories[prompt_key(row)] = interval_update(None, row)
            if (batch,"seen-full.json") not in files:
                continue
            seen = read_rows(files[batch,"seen-full.json"], [r["case_id"] for r in records[:n]], scope, identities, inputs)
            for row in seen:
                histories[prompt_key(row)] = interval_update(histories[prompt_key(row)], row)
            paired = join_rows(atwrite, seen, meta, versions[n])
            for metric in ("RS", "PS", "NS"):
                mr = [r for r in paired if r["metric"] == metric]
                for subset in ("all", "active", "superseded"):
                    use = mr if subset == "all" else [r for r in mr if r["active_at_checkpoint"] == (subset == "active")]
                    checkpoint_summary.append(summarize(use, family=family, arm=name, layer=layer, checkpoint_requests=n, metric=metric, subset=subset))
            if batch == 100:
                family_final[family][name] = paired
        for metric in ("RS", "PS", "NS"):
            counts = Counter(interval_status(v) for k,v in histories.items() if k[0] == metric)
            for state, count in sorted(counts.items()):
                interval_summary.append(dict(family=family, arm=name, metric=metric, status=state, count=count,
                                             denominator=sum(counts.values()), unobserved_transitions_imputed=0))
        for row in family_final[family][name]:
            row["observed_history"] = histories[prompt_key(row)]
            row["observed_failure_status"] = interval_status(histories[prompt_key(row)])
        ledger = private/(name+"-paired-final.jsonl.gz")
        deterministic_gzip_jsonl(ledger, family_final[family][name])
        ledgers.append(dict(arm=name, path=str(ledger.absolute()), bytes=ledger.stat().st_size,
                            sha256=sha_file(ledger), rows=len(family_final[family][name])))
        print(json.dumps(dict(arm=name, paired_final=len(family_final[family][name]), status="CPU_JOIN_VALID")), flush=True)
    matched, bins, comparisons, support_manifest = [], [], [], {}
    for family, groups in family_final.items():
        for metric in ("RS", "PS", "NS"):
            mgroups = {name:[r for r in rows if r["metric"] == metric] for name,rows in groups.items()}
            sm, bn, cp, support = compare_layers(mgroups, spec)
            matched.extend(dict(family=family, metric=metric, **r) for r in sm)
            bins.extend(dict(family=family, metric=metric, **r) for r in bn)
            comparisons.extend(dict(family=family, metric=metric, **r) for r in cp)
            support_manifest[family+"/"+metric] = dict(
                common_strata=len(support["supported_strata"]), union_strata=len(support["all_strata"]),
                groups={name:{k:v for k,v in s.items() if not k.endswith("_keys") and k != "empty_in_this_group"}
                        for name,s in support["groups"].items()})
    write_csv(output/"checkpoint-paired-retention.csv", checkpoint_summary)
    write_csv(output/"margin_age_matched_retention.csv", matched)
    write_csv(output/"matching-strata-with-empty-bins.csv", bins)
    write_csv(output/"layer-paired-differences.csv", comparisons)
    write_csv(output/"observed-failure-intervals.csv", interval_summary)
    write_json(output/"matching-support.json", support_manifest)
    write_json(output/"source-bindings.json", source_bindings)
    write_json(output/"paired_case_ledger_manifest.json", dict(
        dataset_sha256=DATASET_SHA, allowlist_sha256=sha_file(plan_path),
        canonical_identity_validation="case/prompt/targets SHA matched for every consumed row",
        private_ledgers=ledgers, total_final_prompt_pairs=sum(r["rows"] for r in ledgers),
        unique_cases=10000, arms=10, imputation=0, gpu_action=0,
        input_members=inputs, input_member_root=canonical_sha(inputs),
        matching_lock_sha256=sha_file(output/"matching.lock.json"),
        scientific_promotion=False, wall_seconds=time.monotonic()-started,
        status="E1_A_CASE_JOIN_COMPLETE_NOT_E0_E1_CAMPAIGN_COMPLETE"))
    report = ["# E1-A singleton per-case 연결: CPU 사실 보고", "",
              "기존 AlphaEdit/MEMIT 각 L4–L8의 100개 current 및 12개 seen-full 파일을 재사용했다. 새 편집·모델 평가·토크나이저 실행은 없다.", "",
              "## 정의·분모", "", "각 arm의 최종 RS 10,000, PS 20,000, NS 100,000 prompt-pair를 같은 at-write prompt/target SHA와 연결했다. 10개 arm은 동일 10,000 요청을 반복 관측한 것이며 독립 100,000 요청이 아니다. RS/PS는 true−new NLL, NS는 new−true NLL > 0을 성공으로 한다. Tie는 실패이고 원 raw margin(true−new)은 별도 보존했다.", "",
              "전체 원분모가 주표다. common-at-write-success는 동일 family 5개 layer 모두 at-write 성공한 정확히 같은 prompt 부분집합이다. Common-support는 관계 × at-write margin × age × new target token count의 공동 점유 bin에 해당하는 각 arm 행이며, 동일 사례 매칭/가중 표준화와 다르다. 빈 bin과 제외율을 CSV에 남겼다. True token count도 local 원장에 보존했다. Bin은 분석 코드에 고정하고 raw outcome 파일 읽기 전 lock을 만들었지만 과거 공개 결과가 있는 탐색 분석이며 사전 실험 gate가 아니다.", "",
              "## 최종 전체 원분모 연결", "", "| Family | Layer | Metric | At-write n/d | Final n/d | Lost / entry-success | Recovery |", "|---|---:|---|---:|---:|---:|---:|"]
    for r in matched:
        if r["subset"] == "all":
            layer = re.search(r"_L(\d+)_", r["arm"])[1]
            report.append(f"| {r['family']} | {layer} | {r['metric']} | {r['atwrite_numerator']}/{r['denominator']} | {r['final_numerator']}/{r['denominator']} | {r['lost']}/{r['entry_success_denominator']} | {r['recovery']} |")
    report += ["", "## 범위·관측 한계", "",
               "Active/superseded는 NFC·공백 정규화 subject + relation의 관측 prefix 내 latest ordinal로 구분한다. 뒤의 target이 같은 재발행과 다른 target overwrite 여부를 별도로 기록했다. Superseded target 실패는 자동으로 실제 active-history loss로 해석하지 않는다. 원분모를 제거하지 않았다.", "",
               "실패·회복은 실제 at-write와 B001/B005/B010/B020…B100 seen-full 관측만 사용한다. 저장 간격 안의 최초 실패 시점 또는 숨은 실패·회복은 추정하지 않았다. At-write margin은 layer 선택 이후 변수이므로 matched gap을 인과 기여율로 해석할 수 없다. 동일 batch의 서로 다른 layer는 서로 다른 trajectory/state이므로 본 표는 관찰적 산술 비교다.", "",
               "BLUE L4+L8/원본5-layer/W0는 본 selective raw 입력에 포함되지 않은 별도 reference다. 기존 global/blue-native-lifelong-independent-audit-2026-09-11-v1/independent-review-ko.md의 봉인 집계를 context로만 연결하며, 이 singleton per-case 표에 복제하거나 source/config 동등성을 주장하지 않는다. 새 GPU E0 복원 equivalence, E1-B target/write/derivative와 general 항목의 완료 여부는 별도 owner package에서 판정한다.", "",
               "## 산출물·재현", "", "`margin_age_matched_retention.csv`: 전체/공통성공/공통지지/active strata, NLL와 margin 요약. `matching-strata-with-empty-bins.csv`: 모든 점유 union bin과 layer별 빈 bin. `checkpoint-paired-retention.csv`: 12개 저장 시점별 같은 at-write row 연결. `layer-paired-differences.csv`: L4/L5 대비 더 깊은 layer의 exact-paired 산술 차이. `observed-failure-intervals.csv`: 관측된 실패 구간 상태. 대형 per-prompt 원장은 local-only이며 SHA/경로는 `paired_case_ledger_manifest.json`에 있다.", "",
               "실행 명령은 아래 CLI이며 출력/로컬 디렉터리는 create-once다. PNG나 새 GPU 평가를 이 부분 보고의 선행조건으로 추가하지 않았다.", "", "```bash", f"python3 -m project.run_scripts.baseline_mechanism_first.case_population --plan {plan_path} --dataset {dataset_path} --output <new-output-directory> --private <new-local-directory>", "```", "", "scientific_promotion=false. SH1은 사실·수치 연결을 제공하며 최종 원인 종합은 GH 소유다."]
    with (output/"factual-report-ko.md").open("x") as f:
        f.write("\n".join(report)+"\n")
    members = [dict(name=p.name, bytes=p.stat().st_size, sha256=sha_file(p)) for p in sorted(output.iterdir()) if p.is_file()]
    write_json(output/"analysis-manifest.json", dict(members=members, member_root=canonical_sha(members),
                                                   source_sha256=sha_file(__file__), scientific_promotion=False))
    return dict(status="CPU_ANALYSIS_COMPLETE", members=len(members), final_rows=sum(r["rows"] for r in ledgers),
                report_sha256=sha_file(output/"factual-report-ko.md"))


def main():
    parser = argparse.ArgumentParser()
    for name in ("plan", "dataset", "output", "private"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.plan, args.dataset, args.output, args.private), sort_keys=True))


if __name__ == "__main__":
    main()
