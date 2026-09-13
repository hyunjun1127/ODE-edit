# Seq10 report·plot state/schema bounded red review

Instruction: ODEEDIT-S06-LOWCOST-SEQ10-COMPLETED-DETAILED-REPORT-SH4-V1.

검토 범위는 새 report/plot generator, 생성 한국어 보고서 및 이미 검산된 raw-free CSV/JSON이다. 새 raw full rehash, model/GPU/forward/Slurm, runtime 변경은 하지 않았다. 실행 correctness 전체를 다시 승인하는 검토가 아니다.

## 고정한 검토 입력

- `seq_review_report.py` SHA256 `492bffc3d987d4c109eafc47c1c5472bd462778bfcf085bfda69d9d380078cf0`
- `seq_review_plots.py` SHA256 `4f6cf106ac2a5c05ec15acb8102da30b04e09e67ed9c5bf7179e23e2cafce08f`
- `diagnostic-report-ko.md` 검토 시점 SHA256 `1b4abad8c4b5a802865f9a2240e12e90e31b787c2f26ac45a3677d96b02a0d07`
- 기존 state-verification SHA256 `5cc380ed7e15c8e6261d2e9078260138b81729812aabeb5a6f92f0f14a78bf31`

경로 prefix: `experiment-reports/servers/server4/low-cost-write-donor-seq10-2026-09-13-v1/completed-review-v1/`. 최종 package manifest가 이후 report 재생성의 identity를 별도로 기록한다.

## 결과

**이 bounded state/schema 검토에서 blocking mismatch 없음.** 다음 9개 묶음의 소형 CSV/JSON 산술·coverage 검사는 통과했다.

1. 60 batches / 54 links / 18 CP / 90 fit·solve / 9000 z / 80 history append가 state summary와 일치.
2. state-history 60 rows, checkpoint-inventory 18 rows, layer-action 80 rows.
3. cohort-final 1080 rows: 6 arms × 60 cohorts × 3 metrics.
4. plot receipt 7개와 실제 코드 생성 PNG 7개가 존재.
5. static-core comparison 36 rows의 before/after numerator가 모두 동일. 보고서도 이를 tensor/kernel parity로 확대하지 않음.
6. compute-summary의 allocated GPU seconds 합 33475와 본문이 일치.
7. general panel 60 rows 모두 Wiki128 sequences / 24999 predicted tokens / MMLU denominator32.
8. layer plot은 실제 incremental norm, 그 norm의 path 합, B51/55/60 실제 CP net을 서로 다른 축으로 표시. net 미측정 batch를 데이터로 생성하지 않음.
9. report의 fullseen6000, old5000, suffix1000, current100 모집단 및 reused row 비독립성 설명과 plot population selection이 일치.

## 과장 방지 확인

- CP가 full model이 아니라 selected W/M와 공통 prepared/base/P/stats/context closure를 필요로 한다는 설명을 유지한다.
- GPU continuation / incremental exact replay / model-level observer off-on parity는 NOT_TESTED다. 기록된 runtime guard와 새 CPU 검산을 새로운 GPU 검사로 바꾸지 않았다.
- 462 output files의 신규 file SHA/size/stable-stat 범위가 attempt 전체 source/stdout라고 오기되지 않았다.
- 기존 runtime SHA reference와 비교 가능한 CP/increment/evaluation/commit, 이전 reference가 없어 이번 현재 bytes를 봉인한 target captures를 구분한다.
- .75/.875의 native candidate 전체 weight 미저장과 source/materialization-bound 검산 한계를 명시한다. Alpha1 endpoint-copy SHA 및 B51 CP−prepared FP32 equality만 직접 확인된 algebra로 서술한다.
- 초기 actual gate의 과거 관측은 N4 B51→B52만이며 다른 arm은 이번 사후 검산이라는 구분이 유지된다.
- `Layer-wise Update Magnitude` 제목을 사용하고 equal/uniform allocation reference를 추가하지 않는다. Cost 도표의 일반 stacked columns는 weight allocation plot이 아니다.
- 경로별 net은 B51/55/60 세 점만 연결하며 caption에서 측정 시점을 명시한다. 연결선은 중간 checkpoint가 존재한다는 증거가 아니다.
- low-cost/우월성/원인 확정, 다른 entry 또는 full10k 안정성의 확정 주장은 하지 않는다.

## 경계

이 검토는 그림 픽셀을 수동 수정하지 않았고 PNG를 새 도구로 생성하지 않았다. deterministic PNG 재실행 검증과 최종 package/member rehash는 parent publication 단계의 별도 receipt 범위다. 범위 밖 branch inventory, 다른 task 또는 중지 감사는 검토하지 않았다.
