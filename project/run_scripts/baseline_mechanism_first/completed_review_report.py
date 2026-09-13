"""Korean factual report from CPU-derived E01 tables; no execution imports."""
import argparse
import csv
import json
from pathlib import Path


def read(root,name):
    with (root/name).open(newline="") as f:return list(csv.DictReader(f))


def fmt(v):
    if v in ("",None):return "NOT_RECORDED"
    try:
        x=float(v)
        return f"{x:.7g}"
    except (ValueError,TypeError):return str(v).replace("|","\\|").replace("\n"," ")


def table(headers, rows):
    return ["| "+" | ".join(h.replace("|", "\\|") for h in headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |"]+[
        "| "+" | ".join(str(v).replace("|","\\|").replace("\n"," ") for v in row)+" |" for row in rows]


def run(root):
    root=Path(root)
    perf=read(root,"endpoint-performance.csv");trans=read(root,"paired-transitions.csv")
    dist=read(root,"distributions.csv");cohort=read(root,"cohort-performance.csv")
    weights=read(root,"weight-history-differences.csv");cost=read(root,"compute-summary.csv")
    checks=json.loads((root/"reduction-checks.json").read_text())
    def tr(r):return next(t for t in trans if all(t[k]==r[k] for k in ("endpoint","panel","metric")))
    lines=["# E01 Middle/Late actual full-seen — 원본 L4-only 대비 상세 사실 보고", "",
        "상태: **지정 Middle/Late 관측 보완 및 이번 CPU 리뷰 완료. 전체 E01 완료 아님.**", "",
        "instruction_id: `ODEEDIT-S06-E01-MIDDLE-LATE-FULLSEEN-DETAILED-REVIEW-SH1-V1`  ",
        "작성 역할: SH1 사실·수치·검증. 과학적 원인 종합·선택·허용 claim은 GH 소유. `scientific_promotion=false`.", "",
        "## 1. 비교 대상·지표·분모", "",
        "원본은 고정 CounterFact10k의 **AlphaEdit BLUE-style L4_ONLY(singleton L2=1)** 원 trajectory B060/B100이다. Replay는 각각 그 원본 B050/B090 W/M에서 10개 B100 native batch를 실행한 별도 endpoint다. 원본 5-layer AlphaEdit, JV, 다른 stream 점수는 이번 표에 섞지 않았다.", "",
        "`B060=W6000`, `B100=W10000`이다. Current100은 마지막 batch B060 또는 B100의 100 requests를 **그 terminal weight에서** 평가한 값이다. Fullseen은 terminal에서 관측한 전체 과거6000/10000 requests이며, 각 batch의 온라인 at-write 점수를 더한 값이 아니다. 두 endpoint는 독립적으로 복원한 서로 다른 suffix이고 Middle에서 Late로 state를 넘기지 않았다.", "",
        "RS/PS는 rewrite/rephrase의 `new_nll < true_nll`, NS는 neighborhood의 `true_nll < new_nll`; tie는 모두 실패다. 각 NLL은 해당 target의 모든 teacher-forced token 평균이며 낮을수록 해당 후보에 더 높은 likelihood다. Desired margin은 RS/PS에서 `true−new`, NS에서 `new−true`; success는 margin>0이다. Raw true−new margin은 원본 그대로 보존했다.", "",
        "Request당 RS1/PS2/NS10 prompt pairs. Fullseen Middle 분모6000/12000/60000, Late10000/20000/100000. Current100 분모100/200/1000. Fullseen과 그 부분집합 Current/old/new/cohort/active를 서로 더하지 않는다. NS100000은 독립 run100000개가 아니다.", "",
        "## 2. 첫 핵심 결과 — 원본과 replay", "",
        "Δpp=replay−원본. Lost/gained는 동일 case·prompt·target 쌍의 원본 성공→replay 실패 / 원본 실패→replay 성공이다. 총 성공 수가 같아도 같은 문항이 성공했다는 뜻은 아니다.", ""]
    selected=[r for r in perf if r["panel"] in ("fullseen","Current100")]
    lines+=table(["Endpoint","Panel","Metric","원본 n/d (%)","Replay n/d (%)","Δpp","Lost","Gained"],[
        [r["endpoint"],r["panel"],r["metric"],f'{r["baseline_numerator"]}/{r["denominator"]} ({100*float(r["baseline_rate"]):.4f})',
         f'{r["replay_numerator"]}/{r["denominator"]} ({100*float(r["replay_rate"]):.4f})',f'{float(r["delta_pp"]):+.6f}',tr(r)["lost"],tr(r)["gained"]] for r in selected])
    lines += ["", "Late fullseen RS는 양쪽9939/10000이지만 lost4/gained4다. Middle fullseen NS는 +12건, Late는 −33건이며, 각각 lost416/gained428 및 lost770/gained737을 포함한다. 이 산술은 원인 판정이나 weight parity를 뜻하지 않는다.", "",
        "![Fullseen rates](figures/fullseen-rates.png)","", "![Current versus fullseen](figures/current-versus-fullseen-delta.png)", "",
        "## 3. 기존 entry-old와 이번 신규1000, 전체 cohort", "",
        "`entry_old`는 원 checkpoint까지의5000/9000 requests, `new_window1000`은 이번 10-batch suffix의1000 requests다. 이 구분은 평가 시점이 아니라 request의 편집 ordinal이다. 두 집단 모두 B060/B100 terminal 관측이다.", ""]
    lines+=table(["Endpoint","Panel","Metric","원본 n/d","Replay n/d","Δpp","Lost/Gained"],[
        [r["endpoint"],r["panel"],r["metric"],r["baseline_numerator"]+"/"+r["denominator"],r["replay_numerator"]+"/"+r["denominator"],f'{float(r["delta_pp"]):+.6f}',tr(r)["lost"]+"/"+tr(r)["gained"]] for r in perf if r not in selected])
    lines += ["", "모든 B001…B060 및 B001…B100 cohort 각각의 세 지표·strict/token 보조값은 [cohort-performance.csv](cohort-performance.csv), 각 cohort lost/gained/unchanged는 [paired-transitions.csv](paired-transitions.csv)에 있다. 대표 시점만 고른 표가 아니며 빈 subset은 0점으로 대체하지 않는다.", "",
        "## 4. Active·superseded·overwrite metadata", "",
        "NFC 및 공백 정규화 subject+relation의 관측 prefix 내 최신 ordinal을 active로 둔다. 이전 version은 뒤에 다른 target이 있으면 `superseded_conflicting_target`, 같은 target만 있으면 `superseded_same_target`다. 이 metadata 분류는 실제 정당한 지식 갱신을 외부에서 판정한 결과가 아니다. 원래 전체 분모를 유지하고 보조 subset을 따로 보여준다.", ""]
    lines+=table(["Endpoint","Subset","Requests","RS 원본→Replay","PS 원본→Replay","NS 원본→Replay"],[
        [endpoint,panel,next(r["requests"] for r in cohort if r["endpoint"]==endpoint and r["panel"]==panel),
         *[next(f'{r["baseline_numerator"]}→{r["replay_numerator"]}/{r["denominator"]}' for r in cohort if r["endpoint"]==endpoint and r["panel"]==panel and r["metric"]==metric) for metric in ("RS","PS","NS")]]
        for endpoint in ("Middle-B060","Late-B100") for panel in ("active","superseded_conflicting_target","superseded_same_target")])
    lines += ["", "실패가 처음 발생한 정확한 batch는 두 endpoint 비교에서 관측할 수 없다. Checkpoint 사이 실패·회복을 보간하지 않았다. Superseded 문항의 실패를 전부 active-memory 손실로 부르지 않는다.", "",
        "## 5. NLL 및 desired-margin 분포", "",
        "아래는 fullseen의 원본/replay 분포다. 모든 request/prompt가 포함되며 분위수는 정렬된 prompt-pair 값의 `(n−1)q` 선형 분위수다. tail이 같은 문항인지 알려면 paired delta를 보아야 한다. Current100·old/new·모든 cohort·overwrite subset의 mean/median/p90/p95/p99/min/max와 각 paired delta는 [distributions.csv](distributions.csv)에 전부 있다.", ""]
    for quantity,title in [("new_nll","Target-new NLL"),("true_nll","Target-true NLL"),("desired_margin","Desired margin")]:
        lines += [f"### {title}", ""]
        ds=[r for r in dist if r["panel"]=="fullseen" and r["quantity"]==quantity and r["population"] in ("baseline","replay")]
        lines+=table(["Endpoint","Metric","Population","n","Mean","Median","p90","p95","p99"],[[r["endpoint"],r["metric"],r["population"],r["denominator"],*[fmt(r[s]) for s in ("mean","median","p90","p95","p99")]] for r in ds])
        lines += [""]
    lines += ["RS/PS/NS preference 변화와 target-new NLL 변화는 같은 정의가 아니다. NS의 true와 competing-new 양쪽을 별도로 제시했으며, preference가 높다는 이유로 두 후보 모두의 NLL 개선이라고 쓰지 않는다.", "",
        "## 6. Paired 전이·strict/token 보조·at-write", "",
        "![Paired transitions](figures/paired-transitions.png)", "",
        "주표의 loss 분모는 전체 prompt pairs, `loss_rate_among_baseline_success`의 분모는 해당 원본 성공 수다. `unchanged_success`와 `unchanged_failure`까지 합해 전체 분모가 되는 conservation identity를 검산했다. Baseline vs replay final 전이와 각 trajectory의 시간상 forgetting은 다른 비교다.", "",
        "Token/strict는 canonical NLL-pair 지표와 별개다. RS/PS의 target-new와 NS의 target-true strict/token을 아래에 보이며 양 target의 전체 count는 CSV에 보존했다. Token 수는 multi-token target을 포함하므로 prompt 분모와 같지 않을 수 있다.", ""]
    lines+=table(["Endpoint","Metric","원본 strict n/d","Replay strict n/d","원본 token correct/d","Replay token correct/d"],[
        [r["endpoint"],r["metric"],*[r[f'{who}_{"true" if r["metric"]=="NS" else "new"}_{a}']+"/"+r[f'{who}_{"true" if r["metric"]=="NS" else "new"}_{b}'] for a,b in (("strict_numerator","strict_denominator"),("token_correct","token_denominator")) for who in ("baseline","replay")]] for r in perf if r["panel"]=="fullseen"])
    at=root/"atwrite-transitions.csv"
    if at.exists():
        lines += ["", "기존 원본 at-write 관측을 사용한 별도 endpoint 전이는 [atwrite-transitions.csv](atwrite-transitions.csv)에 수록했다. 원본 at-write를 replay 자체의 과거 at-write였다고 바꾸어 부르지 않는다. 수신·identity·비교범위와 missing은 companion receipt에 기록한다."]
        atr=read(root,"atwrite-transitions.csv")
        lines += [""]+table(["Endpoint","Metric","At-write 기준","Final","n/d at-write→final","Lost/Gained"],[
            [r["endpoint"],r["metric"],r["atwrite_reference"],r["final_endpoint"],r["atwrite_numerator"]+"→"+r["final_numerator"]+"/"+r["denominator"],r["lost"]+"/"+r["gained"]]
            for r in atr if r["panel"]=="fullseen" and r["atwrite_reference"]=="ORIGINAL_OWN_ATWRITE"])
        lines += ["", "위 두 final 열에 공통으로 사용한 것은 원본 at-write reference다. Replay 자체 at-write는 B051/B060 및 B091/B100의 각200 requests만 저장되어 있다. 전체 prefix 대비5800/9800, 신규 window1000 대비800 requests의 replay-own at-write가 없어 그 부분 forgetting을 추정하지 않는다. 부분관측 전이는 atwrite CSV의 REPLAY_OWN_RECORDED_ATWRITE 행이며 `observed_requests/missing_atwrite_requests`를 그대로 표시한다."]
    else:
        lines += ["", "Replay의 전체 prefix per-batch at-write 평가는 NOT_RECORDED다. 이전 E1-A10-arm 원본 at-write/final 분석은 evidence manifest로 재사용하며, replay에 원본의 온라인 성공합계를 대입하지 않는다."]
    lines += ["", "## 7. W/M tensor 차이와 성능 fidelity의 구분", "",
        "CPU mmap 및 FP64 chunk reduction으로 실제 저장된 원본/재개 B060/B100을 비교했다. Shape는 W=[4096,14336], M=[1,14336,14336], 저장 dtype FP32다. 상대 Frobenius는 `||replay−reference||F / ||reference||F`; 작은 history 상대차이를 bitexact로 바꾸지 않았다. Changed fraction은 FP32 원소가 정확히 다른 비율이며 유의성 비율이 아니다.", ""]
    lines+=table(["Entry→endpoint","Tensor","||difference||F","||reference||F","Relative F","Max abs","Changed / total","상태"],[
        [r["entry_n"]+"→"+r["endpoint_n"],r["tensor"],fmt(r["difference_frobenius"]),fmt(r["reference_frobenius"]),fmt(r["relative_frobenius"]),fmt(r["max_abs"]),r["changed_elements"]+"/"+r["element_count"],r["status"]] for r in weights if r["entry_n"] in ("5000","9000")])
    lines += ["", "전체 shape/dtype/hash/changed fraction 및 기존 B001/B020의 재사용 수치는 [weight-history-differences.csv](weight-history-differences.csv). B001/B020는 이번에 다시 모델 실행하거나 tensor 전수 재검산한 값이 아니라 기존 receipt 재사용이다. Tensor hash 규약과 파일 SHA는 구별하며 실행 guard hash와 단순 raw-byte hash를 동일 형식으로 간주하지 않는다.", "",
        "B060 W relative F 약8.2136%, B100 약7.4336%와 위의 성능 차이를 나란히 관측했다. **성능 비슷함은 weight/trajectory parity PASS가 아니고, weight nonexact는 성능 붕괴를 뜻하지 않는다.** 원인은 `NONEXACT_CAUSE_UNRESOLVED`; hardware, target, solver 중 하나로 추정 귀속하지 않는다.", "",
        "## 8. E0의 비교 수준과 source repair", "",
        "No-op/저장 CP 복원, 실제 다음 native write, 10-batch continuation, 원본 endpoint tensor 일치, 같은 endpoint 성능 비교를 분리한다. 이번 관측은 저장된 replay endpoint를 평가한 것이며 원본 trajectory와 exact하다고 판정하지 않는다. 첫 B051/B091의 계측을 최종 B060/B100 계측으로 읽지 않는다.", "",
        "Native source `b51dcf5ab825608bee81dd13549318d8d267e835`(tree `5da478cacf175e0d387452e63aee6f4c4925e295`) → observation source `58f50a25809779918b22ad0aceded732c097eab4`(tree `1c2b49a75ee7e39e03d51346fb35ccb8b0a602cf`) diff는 신규5파일/289줄, 기존 파일 수정0이다. 원본 seen-full에 없는 request_order를 전체 canonical row identity 검증 후 결속하는 관측 schema 수리다. Native equation/target/history/precision/tolerance 변경은 없다.", "",
        "45914/45915는 10 native batch와 terminal Current100을 저장한 뒤 원본 seen-full header의 `KeyError: request_order`에서 실패했다. 실패 전체를 valid 실험으로 바꾸지 않고, 완전한 저장 endpoint와 Current100만 별도 observer가 재사용했다. 46439/46440은 native/z/history append를 다시 수행하지 않는다. 0 count는 source call graph와 receipt의 선언값으로 확인되며, 별도 하드웨어 action-counter를 사용한 것은 아니다.", "",
        "## 9. Guard·복원·정보 분리·전체 검산", "",
        "- 모든 shard 전후 selected L4 W/M 전체 byte SHA, 모든 parameter pointer/version, Python/NumPy/Torch/CUDA RNG를 비교했다.",
        "- 매 shard의 모든 nonselected parameter/buffer byte를 전수 재해시했다고 주장하지 않는다.",
        "- Outer transaction은 모델 전체 parameter/buffer CPU backup 32,121,053,440bytes를 포착해 끝에서 pointer/bytes 및 RNG 복원 PASS를 기록했다. changed_version_count1이며 version원복은 NOT_CLAIMED_NATIVE_COPY_INCREMENTS다.",
        "- Fullseen 성능은 actual endpoint W/M scope 안에서 저장되고, 그 후 새로 로드한 pinned W0로 복구했다. 반환 후 W0 성능을 endpoint로 저장한 것이 아니다.",
        "- `C0_restored=false`는 CP가 covariance 자산까지 복원했다는 주장이 아님을 뜻한다. Hook registry는 source상 복원하며 per-hook independent numeric equality는 별도 미기록이다.",
        "- Historical evaluator microbatch16·manual left-padding·position override 없음·전체 target-token NLL·no_grad 경로가 source에 결속된다. Tokenizer global padding은 right로 유지한다.",
        "- Original header 부재를 행번호만으로 보완하지 않았다. Fixed dataset에서 모든 case/prompt/target SHA를 재계산해 순서/분모/중복/strict-sign/finite를 확인했다.", ""]
    lines+=table(["Endpoint","Past","Shards","Reused current","Unique fullseen","Prompt pairs","누락/중복/nonfinite"],[[c["endpoint"],c["past_requests"],c["shard_count"],c["reused_current"],c["requests"],c["total_prompt_pairs"],"0/0/0"] for c in checks["checks"]])
    lines += ["", "[reduction-checks.json](reduction-checks.json)은 원본/current/shards/fullseen의 SHA before/after, dataset identity 및 수치검산을 결속한다. Source/restore 독립 감사는 [source-integrity-review.md](../../../../../audits/servers/server1/2026-09-13-e01-middle-late-review/source-integrity-review.md)와 JSON이다. 새 GPU/forward/evaluator/Slurm mutation은0이다.", "",
        "## 10. 실제 E1-B 계측과 빠진 연결", "",
        "[instrumentation-summary.csv](instrumentation-summary.csv)는 B051/B091의 실제 저장된 target/write/query/geometry/signed/General 계측을 추출한 표다. B060/B100 terminal에서 새 query/General/backward를 수행하지 않았다.", "",
        "각 B051/B091의 승인 current100+historical128은2964 prompt pairs다. 관측 receipt forward_pairs2965는 별도 대표 query1을 포함하므로 canonical 평가 분모에1을 더하지 않는다. Signed backward는 Current/Historical160 pairs와 별도 General16 sequences이며, 대표 FD는 별도1 backward/5 diagnostic forwards다. 전체176개 모두 FD로 검증했다고 부르지 않는다.", "",
        "General은 기존 Wikipedia128의 W0/entry/첫 native endpoint per-sequence NLL이다. Canonical CounterFact NS가 아니며 full vocabulary teacher 자산이나 최종 B060/B100 general 성능을 대신하지 않는다. Signed desired-margin derivative와 General NLL derivative는 부호·단위가 달라 별도 기록한다. `progress_slope=max(0,−derivative)`를 signed derivative로 바꾸어 보고하지 않았다.", "",
        "Target 최종 trainingNLL/실제 iteration/stop/clamp는 NOT_OBSERVED. Compute_z 호출수만 최적화 수렴을 뜻하지 않는다. Projected C0/history exact spectrum/rank/condition은 NOT_RECORDED이며 native-system symmetry error는 condition number가 아니다. Query prefix의 SUBJECT_UNRESOLVED를 임의 subject 위치로 채우지 않았다. L5–L8 미실행으로 layer별 output-response 비교도 이번 관측에서 없다.", "",
        "## 11. 비용·실패 lineage·중복계수 금지", "",
        "Allocation GPU-sec는 단일 GPU job의 실제 scheduler elapsed이고 batch/extern 중복합산0. Native total과 z/key/solve/history 세부시간, program elapsed, evaluation/guard wall은 서로 포함관계이므로 모두 더하지 않는다. Host wall을 FLOPs/CUDA kernel 시간으로 표시하지 않는다.", ""]
    allocations=[r for r in cost if r["component"]=="allocated_GPU_seconds"]
    lines+=table(["Job","Entry","GPU-sec","GPU-hour","상태"],[[r["job_id"],r["entry_n"],fmt(r["value"]),f'{float(r["value"])/3600:.6f}',r["status"] or "PRIOR_COMPLETED_RECEIPT_REUSED"] for r in allocations])
    lines += ["", "46439+46440 신규 관측 allocation은 **7,469 GPU-sec(2.074722 GPUh)**. 재사용 native 실패 attempts45914+45915의 **19,475 GPU-sec**를 포함한 핵심 repair lineage는 **26,944 GPU-sec(7.484444 GPUh)**이다. 과거 user-superseded45908/45913은 각각1187/0초로 별도 보존하며, 이를 포함하면28131초다. 이 합은 모든 과거 E01 초기 job 비용을 포함한 campaign-total이라고 주장하지 않는다.", ""]
    component_names={"program_elapsed","model_load","fullseen_past_eval_guard_shards","native_10_batches","target_seconds","key_seconds","solve_seconds","history_seconds","peak_allocated_bytes","peak_reserved_bytes"}
    lines+=table(["Job","Component","값","단위","계수 경계"],[[r["job_id"],r["component"],fmt(r["value"]),r["unit"],r["accounting"]] for r in cost if r["component"] in component_names])
    lines += ["", "Fullseen 과거 관측/guard wall Middle2467.132626초, Late4269.783501초에는 hash/RNG overhead가 포함된다. 최초128 gate의130.374/136.791초를 전체 completion 비용으로 쓰지 않았다. 각종 I/O·load·native breakdown의 실제 기록과 미분리 항목은 [compute-summary.csv](compute-summary.csv)에 남겼다. 독립적인 FLOP·연산자별 kernel time은 NOT_RECORDED다.", "",
        "## 12. E01 전체 coverage와 이번 task의 완료 경계", "",
        "[coverage.csv](coverage.csv)는5 layers×4 entries=20 canonical E1cell 전부의 상태다. L4 n0/1000/5000/9000의 첫 native B100이 존재해 **4/20**, L5–L8의16cells는 NOT_RUN이다. Cold1+warm3×10의 canonical 저장 native31batches는20cell 진단을31개 완료했다는 뜻이 아니다. Target·signed·General 항목은 cell별 partial이다.", "",
        "E1-A 원본 AlphaEdit/MEMIT singleton10arm CPU per-case/matching package와 기존 cold45719/warm45805 검산은 [evidence-reuse-manifest.json](evidence-reuse-manifest.json)으로 결속해 재사용했다. 이 리뷰에서 이전10arm 전체를 새로 실행/전수 동기감사하지 않았다. B020 fullseen replay 및 L5–L8 E0동등성은 미측정/미실행으로 남는다.", "",
        "이번 완료 범위는 **B060/B100 실제 fullseen 성능 보완 + 저장 결과 상세 CPU 리뷰**뿐이다. E0 전체 원 trajectory fidelity, E1-B20cell, E2 이후, AOS/A/B/다른 paused task는 완료·재개하지 않았다. 성능 불리한 row를 제거하거나 추가 GPU 확인으로 메우지 않았다.", "",
        "## 13. 산출물·source·재현", "",
        "Raw weights/checkpoints/prompts/keys/full logs는 기존 local 원본 그대로이며 Git에 포함하지 않았다. 원격 새 raw transfer0; `NO_BROADCAST_NOT_REQUIRED`. 기존 실패bytes와 sealed report도 덮어쓰지 않았다.", "",
        "- [endpoint-performance.csv](endpoint-performance.csv): Current/fullseen/entry-old/new1000와 secondary count.",
        "- [cohort-performance.csv](cohort-performance.csv), [paired-transitions.csv](paired-transitions.csv), [distributions.csv](distributions.csv): 전체 cohort·metadata별 수치.",
        "- [weight-history-differences.csv](weight-history-differences.csv), [instrumentation-summary.csv](instrumentation-summary.csv), [compute-summary.csv](compute-summary.csv), [coverage.csv](coverage.csv).",
        "- [input-manifest.json](input-manifest.json), [source-manifest.json](source-manifest.json), [analysis-manifest.json](analysis-manifest.json), [rooted-receipt.json](rooted-receipt.json).",
        "- [plot-reproduction.json](plot-reproduction.json): 실제 동일 CSV 재실행 PNG3개 byte-identical, plotting 코드/environment/명령/input/output SHA. Imagegen/manual image edit0.", "",
        "분석 source는 `source-manifest.json`의 HEAD/tree+members로, 실제 실행은58f50a와 native b51dcf5로 별도 식별한다. Main에 올린 것은 이번 CPU analysis/reducer/report scope이며 다른 미완료 runtime을 완료로 표시하지 않았다.", "",
        "```bash",
        "python3 -m project.run_scripts.baseline_mechanism_first.completed_review_reducer \\",
        "  --root /mnt/raid5/janghj/.codex/worktrees/odeeditsh1-e01-fullseen-schema-repair-r3/local/baseline-mechanism-first-e01/20260912-v1/fullseen-schema-repair-r3 \\",
        "  --dataset /mnt/raid5/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/counterfact.json --output <새_분석_폴더>",
        "/mnt/raid5/janghj/EasyEdit/.venv/bin/python -m project.run_scripts.baseline_mechanism_first.completed_review_package plots --root <새_분석_폴더>",
        "python3 -m project.run_scripts.baseline_mechanism_first.completed_review_report --root <완비된_새_분석_폴더>",
        "python3 -m project.run_scripts.baseline_mechanism_first.completed_review_package verify --root <봉인_보고서_폴더>",
        "```", "",
        "원인·방법 우열·다음 실행 선택은 이 수치와 미검증 범위를 받은 GH의 별도 종합 대상이다. 본 보고서는 관측되지 않은 원인을 확정하거나 새 실험을 자동 제안·실행하지 않는다.", ""]
    insert=lines.index("## 11. 비용·실패 lineage·중복계수 금지")
    notes=(root/"evidence-notes.md").read_text().split("## 실제 첫-batch signed / General 요약",1)[1]
    extra=["### 첫 batch signed·General 실제 수치", "",notes.strip(),"",
        "B051/B091 원본 target 대 replay recomputation의 exact 일치는 각각0/100이다. 100 targets 중 max-abs의 최대값은0.00370012969/0.09020320326이며 원인을 hardware로 귀속하지 않는다. 나머지18 continuation batch의 원본 target 대조는 NOT_RECORDED다. B060/B100 checkpoint의 batch/seen_ids/model revision/contexts/RNG/covariance metadata 비교는 exact지만 W/M tensor는 위 표처럼 nonexact다.", ""]
    lines[insert:insert]=extra
    with (root/"diagnostic-report-ko.md").open("x") as f:f.write("\n".join(lines))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);a=p.parse_args();run(a.root)
