# GH → SH4: server4 job 전수 목록 및 상세 사실 리뷰

Instruction ID: ODEEDIT-GH-SH4-ALL-JOBS-DETAILED-REVIEW-20260924-R1.
ACK nonce: ODEEDIT-GH-SH4-ALL-JOBS-DETAILED-REVIEW-20260924-R1.
대상 SH4 01a04939-b5c7-7a03-ba2d-ef3343d62cfd / /data/janghj/ODE-edit.
사용자: “server4 job들 모두 자세히 리뷰시켜라”.

## 범위와 실행 경계

이번 명시 recall은 server4 ODE-edit 사용자 소유 job들의 상태·산출물 리뷰다.
최근 alpha-key causal gate/geometry/writers/reducer 및 그 실패·수리·취소·
대체 attempt를 빠짐없이 포함하고, 그 밖 server4의 현재 등록 project job 및
직전 완료리뷰 이후 미리뷰 실행도 제출기록으로 대조하여 포함한다.
다른 사용자의 job은 제외한다. 오래된 이미 검토된 task는 report/receipt를 index로
재사용하고 이전 전체 raw를 관성적으로 전수 재계산하지 않는다.
최근 alpha-key만 보고 “모든 server4 job 검토”라고 쓰지 않는다.

기존 submission manifest/run lock/로그·보고 목록을 먼저 읽어 exact job/array/
physical ID/owner/이름/source/dependency를 표로 봉인한다. 필요한 범위의 한정
squeue/sacct snapshot으로 현재 queue와 terminal accounting을 교차 확인한다.
정확한 조회범위/시각/확인불가 job을 남기고 과거 기록으로 현재 상태를 추정하지 않는다.

이번에는 CPU/source/저장 산출물 기반 독립 리뷰만 수행한다.
새 GPU/model load/forward/evaluator/추가학습·편집·Slurm 제출/재제출/
cancel/hold/release/dependency 변경0. 이전 자율 repair 권한은 이번 리뷰가
새로 발동시키지 않는다. 이미 존재하는 실행 프로그램은 변경하지 않는다.
Running/PENDING이면 NOT_TERMINAL snapshot과 완료된 불변 산출물만 구분해서
기록하고 종료를 기다리거나 live파일을 완성본으로 해시/분석하지 않는다.
별도 source/input/raw/12CP 삭제·이동·대용량 재전송·새checkpoint 저장0.

## 필수 리뷰

1. 전체 job inventory: task/family, job/attempt/source/tree/archive/lock,
   requested resource/실제 allocation/시작·끝/exit/signal/의존성, raw/report 경로.
   COMPLETED/FAILED/CANCELLED/PENDING/RUNNING/NOT_FOUND를 구분한다.
   Scheduler COMPLETED는 scientific-valid와 별개다.
2. 실패·수리 lineage: 첫 exception/source line, 근거 있는 원인, 변경된 실행코드,
   수리 실제 적용 여부, 원 실패 receipt, 재사용과 재실행 범위, 중복 실행 여부.
   후속 dependency가 실행됐는지/막혔는지, 일부 결과·누락 셀을 명시한다.
3. 실행 설계 적합성: 정본→frozen source 함수/줄→stored evidence를 대조한다.
   현재 main과 실제 runtime를 혼동하지 않는다. CPU toy PASS를 model PASS로
   확대하지 않으며 실제 gate와 archive parity의 범위/미측정을 그대로 적는다.
4. 완료된 raw metrics는 독립 reducer로 sample/case/prompt/target/order/
   denominator/finite/ties/endpoint identity를 검산한다. Canonical R/P/N
   preference뿐 아니라 저장된 TF rewrite/rephrase/neighborhood accuracy,
   token-micro/prompt-macro/strict, true/new NLL, paired lost/gained를 정의와 함께
   보고한다. 없는 지표를 새 GPU 평가로 채우지 말고 NOT_RECORDED로 둔다.
5. 각 family를 분리한 실제 수치표와 전체 요약을 만든다. 서로 다른 모델/entry/
   layer/precision/sample을 같다고 합치지 않는다. Same-entry shadow와 독립
   sequential baseline, 신규/재사용 결과를 구분한다. 불리한 수치도 누락하지 않는다.
6. 계산비용: parent allocation GPU-sec/시간, 실패·취소·재실행·재사용 별도,
   stage/target/prefix/solve/observer/I-O timers의 중첩과 미계측 명시.
   cap2 실제 concurrency는 interval로 확인하고 allocation≠utilization을 유지한다.
7. artifact integrity: 결과파일 inventory/size/SHA, 필요한 schema/state/restore/
   history/cursor 검산, overwritten/missing/partial 결과 구분. 기존 input fullSHA
   검증 재사용과 새로운 output fullSHA를 구별하며 공용모델/12CP 재해시는
   무결성 이상 근거 없이 반복하지 않는다.

## Alpha-key E0–E4 상세 요구

원본 BASE_ALPHAEDIT blue=False/L4–L8/L2=10, 원 P/context/precision/tokenizer,
12CP transfer identity, 실제 G0–G3와 W50→B51 native100/SHAM 증거를 확인한다.
최근 BOS 및 writers 수리는 원 진단/실행source에서 원인·수치 변경 여부를 대조한다.

- E1 28 state×cohort: raw/centered/unit-normalized PR, mean-energy, norm-outlier,
  raw/projected, individual-context/actual-mean 분리, calibration512/assessment3488.
- E2 hybrid subset/upper negative control 및 stage 경로: 실제 완료 대비·미실행 분리.
- E3 4 entry native100: z 공유 범위, stage K/R/Δ 및 마지막 L8 residual,
  branch restore, N512 전체/H512/current R/P/overwrite-mask 자료.
- E4-H NATIVE/SHAM/H5/H6/H56/MASS56: timestamp/current parity, coverage,
  temporary M_eff와 persistent native append 구분.
- E4-W component/full-hook parity/정의가능 rank1/norm control,
  E4-KR 동일 receiving-state 2×2의 수치와 interaction을 산술로 제시한다.
- N/P observer-only, basis/강도/arm 선택 오염 여부를 source와 ledger로 점검한다.
- 승인94개 family 및 E0를 COMPLETED/TECHNICAL_FAILED/NOT_APPLICABLE/
  NOT_RUN/NOT_TERMINAL로 대응시키고 SEQ/ORDER/FUTURE는
  FOLLOWUP_NOT_SUBMITTED로 별도 표시한다. 미실행은 negative result가 아니다.

“집중의 원인”, “효능 우월”, “method 승격” 등 과학적 해석·다음 실험 선택은
GH 별도 리뷰 소유다. SH는 실제 수치·분모·paired 차이·계약판정·source-backed
기술 RCA·증거한계만 보고한다.

## 산출물과 완료

전용 branch codex/server4-all-jobs-detailed-review-20260924-v1.
허용 CPU 분석 코드 project/run_scripts/server4_completed_jobs_review_20260924/.
Raw scratch local/server4-completed-jobs-review/20260924-v1/.
종합 report 및 CSV/그림/manifest:
experiment-reports/servers/server4/completed-jobs-review-2026-09-24-v1/.
각 기존 family report 아래 새 completed-review-20260924-v1/ supplement 허용.
Audits: audits/servers/server4/2026-09-24-completed-jobs-review/.
ACK/message: messages/acks/server4/2026-09-24-completed-jobs-review.md 및
messages/server-heads/server4/2026-09-24-completed-jobs-review.md.
소형 run/status: runs/odeedit_server4_jobs_review_20260924/ 및
tasks/status/server4-jobs-review-20260924/server4.json.

첫 ACK/M0에는 job inventory/실제 상태와 포함·제외 범위를 전달한다.
첫 독립 actual 수치표를 조기 전달하고, 승인된 CPU 상세보고까지 계속한다.
Gate/첫 표에서 멈추거나 단계별 재승인을 기다리지 않는다.
누락/실패가 있으면 그 상태까지 보고서를 완성하되 실험을 몰래 재시작하지 않는다.
자체 검산과 별도 red 사용 여부를 정직하게 구분한다.
Markdown 실제 표렌더/링크, 재현 가능한 코드그림, source/manifest/member SHA 및
raw-free 검산 뒤 own branch와 clean integration의 own-scope 소형 보고/source를
nonforce main push하도록 승인한다. 다른 변경/dirty 보존, conflict는 보고한다.
원본 runtime/science/raw/평가값 수정0.

최종 GH 인계에 main/source/report SHA, exact jobs, 핵심 수치, 완료/미완료·한계,
비용·보고경로를 적고 TASK_COMPLETE_STOP.
사용자 다음 recall 전 주기 monitoring/자동재개/새실험0, NO_BROADCAST_NOT_REQUIRED.
