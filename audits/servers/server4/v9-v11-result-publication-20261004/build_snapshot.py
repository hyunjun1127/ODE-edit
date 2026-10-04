"""읽기 전용 CPU 게시 검산. JSON으로 소형 파일 내용을 stdout에만 반환한다.

python3 -B audits/servers/server4/v9-v11-result-publication-20261004/build_snapshot.py
GPU, scheduler, 네트워크, 파일 쓰기 또는 과학 실행을 호출하지 않는다.
원 raw는 읽고 검산하되 반환 파일에는 포함하지 않는다.
"""
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
V9 = Path("/data/janghj/ODE-edit/local/jlz-realization-v9/20261004-2k-v1/attempt-r2")
V11 = Path("/data/janghj/ODE-edit/local/jlz-native-increment-v11/20261004-v1/attempt-r2-cap3")
R9 = "experiment-reports/servers/server4/jlz-realization-v9-ridge-bs100x20-s4-20261004-v1/results-publication-20261004"
R11 = "experiment-reports/servers/server4/jlz-native-increment-v11-bs100x20-20261004-v1/intermediate-W5-20261004"
AUDIT = "audits/servers/server4/v9-v11-result-publication-20261004"
inputs = {}
files = {}


def read(path):
    path = Path(path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
    inputs[str(path)] = dict(path=str(path), bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest(),
                            mtime_ns=after.st_mtime_ns)
    return data


def obj(path):
    return json.loads(read(path))


def rows(path):
    return list(csv.DictReader(io.StringIO(read(path).decode())))


def emit_json(path, value):
    files[path] = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def emit_csv(path, values):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(values[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(values)
    files[path] = buf.getvalue()


def validate_observer(path, requests):
    summary = obj(path / "summary.json")
    assert summary["requests"] == requests
    assert summary["row_count"] == 13 * requests
    assert summary["no_mutation"] and not summary["optimizer_feedback"]
    chunks = sorted(path.glob("chunk-*.json"))
    assert chunks
    raw = []
    for chunk in chunks:
        block = obj(chunk)
        assert block["state"] == summary["state"]
        raw.extend(block["rows"])
    assert len(raw) == requests * 13
    assert len({(r["case_id"], r["kind"], r["prompt_index"]) for r in raw}) == len(raw)
    assert len({r["case_id"] for r in raw}) == requests
    result = {}
    for kind, multiple in [("R", 1), ("P", 2), ("N", 10)]:
        rr = [r for r in raw if r["kind"] == kind]
        assert len(rr) == multiple * requests
        assert all(math.isfinite(r[k]) for r in rr for k in ["new_nll", "true_nll"])
        num = sum(r["true_nll"] < r["new_nll"] if kind == "N"
                  else r["new_nll"] < r["true_nll"] for r in rr)
        target = "true" if kind == "N" else "new"
        strict = sum(r[target + "_strict"] for r in rr)
        correct = sum(r[target + "_token_correct"] for r in rr)
        count = sum(r[target + "_token_count"] for r in rr)
        expected = summary["summary"][kind]
        assert (num, len(rr), strict, correct, count) == (
            expected["numerator"], expected["denominator"], expected["strict_numerator"],
            expected["desired_token_correct"], expected["desired_token_count"])
        assert abs(expected["rate"] - num / len(rr)) < 1e-12
        result[kind] = dict(numerator=num, denominator=len(rr), rate=num / len(rr),
                            strict_numerator=strict, desired_token_correct=correct,
                            desired_token_count=count)
    return summary, result


v9summary = obj(V9 / "report/summary.json")
assert v9summary["A"]["commits"] == 20 and v9summary["A"]["complete2000"]
assert v9summary["B"]["commits"] == 18 and not v9summary["B"]["complete2000"]
v9obs, v9check = validate_observer(V9 / "main-A/observe-W20", 2000)
source9 = v9summary["A"]["terminal"]["source"]
metrics9 = rows(V9 / "report/metrics.csv")
comparison9 = rows(V9 / "report/comparison-W20.csv")
for row in comparison9:
    if row["method"] == "V9_A":
        check = v9check[row["kind"]]
        assert int(row["numerator"]) == check["numerator"]
        assert int(row["denominator"]) == check["denominator"]
assert not any(r["method"] == "V9_B" for r in comparison9)

metrics11, commits11, checks11 = [], [], {}
for arm in ["MAIN", "NOALLOC"]:
    for batch in range(1, 6):
        d = V11 / ("main-" + arm) / ("batch-%02d" % batch)
        commit = obj(d / "commit.json")
        assert commit["history_appends"] == 5 and commit["observer_no_mutation"]
        assert not commit["checkpoint_saved"]
        assert commit["fit"]["candidates"] == 25 and commit["fit"]["updates"] == 24
        for phase in ["pre", "post"]:
            summary = obj(d / phase / "summary.json")
            scopes = [("current", summary.get("current", summary["summary"]))]
            if batch == 5 and phase == "post":
                summary, check = validate_observer(d / phase, 500)
                assert commit["post"] == summary["summary"]
                checks11[arm] = check
                scopes.append(("all_seen", summary["summary"]))
            for scope, ss in scopes:
                for kind in ["R", "P", "N"]:
                    row = dict(arm=arm, batch=batch, phase=phase, scope=scope, kind=kind)
                    row.update(ss[kind])
                    assert abs(row["rate"] - row["numerator"] / row["denominator"]) < 1e-12
                    metrics11.append(row)
        commits11.append(dict(arm=arm, batch=batch, source=commit["source"],
                              candidates=commit["fit"]["candidates"],
                              updates=commit["fit"]["updates"], early_stop=commit["fit"]["early_stop"],
                              history_appends=commit["history_appends"], seconds=commit["seconds"],
                              observer_no_mutation=commit["observer_no_mutation"]))
emit_csv(R11 + "/metrics-through-W5.csv", metrics11)
emit_csv(R11 + "/batch-accounting-through-W5.csv", commits11)

cakebase = ROOT / "experiment-reports/servers/server4/cake-native-lifelong-b100x100-2026-09-15-v1/completed-review-v2"
baseline = rows(cakebase / "baseline-cumulative.csv")
cake = rows(cakebase / "seen-prefix.csv")
memit = rows(ROOT / "experiment-reports/servers/server3/memit-history-fixed10k-20260928-v1/completion-review-r1/all-seen-metrics.csv")


def compact(method, batch, kind, n, d, evidence):
    return dict(method=method, endpoint=batch, kind=kind, numerator=int(n),
                denominator=int(d), percent=100 * int(n) / int(d), evidence=evidence)


tables = {5: [], 20: []}
for r in comparison9:
    if r["method"] != "AlphaEdit-BLUE":
        tables[20].append(compact(r["method"], 20, r["kind"], r["numerator"],
                                 r["denominator"], "V9_COLLECTOR; historical baselines not matched runtime"))
for r in metrics11:
    if r["batch"] == 5 and r["phase"] == "post" and r["scope"] == "all_seen":
        tables[5].append(compact("V11_" + r["arm"], 5, r["kind"], r["numerator"],
                                r["denominator"], "CURRENT_W5_RAW_CPU_RECOUNT"))
for r in metrics9:
    if r["arm"] == "A" and r["batch"] == "5" and r["scope"] == "all_seen":
        tables[5].append(compact("V9_A", 5, r["kind"], r["numerator"],
                                r["denominator"], "PREVIOUS_V9_SAME_FIRST500"))
for b in [5, 20]:
    for r in baseline:
        if int(r["batch"]) == b and r["arm"] == "AlphaEdit_BLUE(L4+L8)":
            tables[b].append(compact("AlphaEdit-BLUE_L4+L8", b, r["metric"][0],
                                    r["numerator"], r["denominator"], "HISTORICAL_AGGREGATE; raw not reaudited"))
        if b == 5 and int(r["batch"]) == 5 and r["arm"] == "BASE_ALPHAEDIT_NATIVE":
            tables[b].append(compact("BASE_ALPHAEDIT", b, r["metric"][0],
                                    r["numerator"], r["denominator"], "HISTORICAL_AGGREGATE; raw not reaudited"))
    for r in cake:
        if int(r["batch"]) == b and r["population"] == "ACTUAL_FULL_SEEN":
            tables[b].append(compact("CAKE", b, r["metric"][0], r["numerator"],
                                    r["denominator"], "HISTORICAL_AGGREGATE; raw not reaudited"))
for r in memit:
    if int(r["batch"]) == 5:
        tables[5].append(compact("MEMIT-H", 5, r["metric"][0], r["numerator"],
                                r["denominator"], "S3_HISTORICAL_AGGREGATE; not new matched V11 baseline"))
for b, values in tables.items():
    assert len({(r["method"], r["kind"]) for r in values}) == len(values)
    for r in values:
        assert r["denominator"] == b * 100 * dict(R=1, P=2, N=10)[r["kind"]]
emit_csv(R9 + "/comparison-W20-with-historical.csv", tables[20])
emit_csv(R11 + "/comparison-W5.csv", tables[5])

for p in [V9 / "execution.lock.json", V9 / "config.json", V11 / "execution.lock.json", V11 / "config.json"]:
    read(p)
for name in ["artifact-index.json", "terminal-context-decomposition.json"]:
    read(V9 / "report" / name)
accounting = obj(V9 / "report/accounting.json")
account_lines = [line.split("|") for line in accounting["output"].strip().splitlines()]
assert [(r[0], r[1], r[2], r[3]) for r in account_lines] == [
    ("57899", "FAILED", "0:11", "36710"),
    ("57900", "CANCELLED by 1025", "0:0", "32330")]
emit_json(R9 + "/scheduler-outcomes.json", {
    "evidence": "기존 collector accounting 재게시; 새로운 scheduler 조회 없음",
    "accounting_status": accounting["status"],
    "jobs": [dict(job_id=57899, state="FAILED", exit_code="0:11", allocated_gpu_seconds=36710,
                  saved_scientific_endpoint="20 commits and W20 stored; runner COMPLETED"),
             dict(job_id=57900, state="CANCELLED by 1025", exit_code="0:0", allocated_gpu_seconds=32330,
                  saved_scientific_endpoint="18 commits; no W20")],
    "failure_root_cause": "NOT_IDENTIFIED_BY_THIS_PUBLICATION"
})
emit_json(AUDIT + "/checks.json", {
    "observed_utc": datetime.now(timezone.utc).isoformat(),
    "authority": "USER: v11의 중간 결과, v9의 2000edit 모두 main에 push해",
    "scope": "저장 결과 CPU 검산/게시만; job/source/runtime 변경 없음",
    "v9_A_W20_raw_recount": v9check, "v9_B": "18 commits; NOT_2000_COMPLETE",
    "v11_W5_raw_recount": checks11, "v11_snapshot_endpoint": 5,
    "v11_snapshot_status": "INTERMEDIATE_NOT_FINAL",
    "raw_rows_checked": 39000, "duplicate_missing_nonfinite_checks": "PASS",
    "v11_first5_commit_checks": "PASS: 25 candidates/24 updates/5H and observer receipts",
    "independent_reviewer": "NOT_USED; side conversation CPU audit only",
    "GPU_evaluation": 0, "job_mutations": 0, "monitoring_active": False,
    "broadcast": "NO_BROADCAST_NOT_REQUIRED; raw/tensor/fullstdout local KEEP",
    "historical_comparison": "같은 cohort의 기존 집계 재사용. runtime/층/평가기 차이가 있으며 새로운 parity 증명이 아님."
})
emit_json(AUDIT + "/input-manifest.json", list(inputs.values()))
print(json.dumps(files, ensure_ascii=False))
