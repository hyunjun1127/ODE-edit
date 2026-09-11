# GH → SH4: server4 실험 상세 리뷰·통합 보고서

- instruction_id: ODEEDIT-S06-SERVER4-COMPLETED-EXPERIMENTS-DETAILED-REVIEW-SH4-V1
- nonce: ODEEDIT-GH-SH4-COMPLETED-REVIEW-20260911-R1
- 사용자 요청: “server4에서 진행했었던 실험들 자세히 리뷰시키자.”
- from: GH / 01a04939-8873-7673-8dca-4c7fc5e31af0
- to: SH4 / 01a04939-b5c7-7a03-ba2d-ef3343d62cfd
- server / CWD / repository: server4 / /data/janghj/ODE-edit / hyunjun1127/ODE-edit
- 배정 시 remote main: e6cbf8791e06cd78ceaec891f694a2d495e12268
- 권한: CPU analysis-only 상세 리뷰·보고서·완료한 본인 scope main 통합. 새 model/GPU/evaluator/Slurm 제출 권한 없음.

## 1. 담당 범위와 우선순위

SH4가 server4 최근 실험 inventory를 먼저 만들고, 완료 상태·보고서 유무·미완료/실패·비교 가능성을 구분한다. 특히 아직 상세 결과 보고가 GH에 전달되지 않은 BLUE L5/L6/L7 lifelong 6개와 native baseline 2개를 우선 확인한다. 이미 완료 보고된 BLUE/L4/L8 6개는 봉인 보고서·검증을 재사용하여 새로운 arm과 통합 비교한다.

SH4에게 결과 수집·분모/state 검증·상세 해석·최종 한국어 보고서·본인 scope main 반영을 맡긴다. GH의 중복 raw/code/test 점검이나 단계별 재승인을 기다리지 않는다. 다른 agent/사용자 작업과 같은 저장소를 사용하므로 변경을 되돌리지 말고 새 worktree/branch와 output에서 작업한다.

기존 PROTOCOL의 SH factual-only 규칙에 대해 이번 사용자 “자세히 리뷰” 요청 한정으로 관측·근거 기반 가능한 설명·상충·한계 분석을 허용한다. 확인된 사실과 설명/미확정 가설은 분리한다. Scientific promotion 및 다음 실험 자동 실행은 금지한다.

## 2. 대상 inventory

### 핵심: 동일 fixed10k Llama lifelong B100×100

각 method(MEMIT / AlphaEdit)에 대해 native baseline, BLUE(L4+L8), BLUE_L4_ONLY, L5_ONLY, L6_ONLY, L7_ONLY, L8_ONLY를 구분한다. 잠재적 전체 비교는 **2 families ×7 arms =14 chains**이며 모두 완료되었다고 미리 가정하지 않는다. 누락/실패/진행 중 arm도 status/reason 행으로 남긴다.

- 기존 BLUE/L4/L8 6개 canonical:
  - 39307 MEMIT_BLUE (L4+L8)
  - 39283_1 AlphaEdit_BLUE (L4+L8)
  - 39283_2 MEMIT_BLUE_L4_ONLY
  - 39283_3 AlphaEdit_BLUE_L4_ONLY
  - 39283_4 MEMIT_BLUE_L8_ONLY
  - 39283_5 AlphaEdit_BLUE_L8_ONLY
- 추가 single-layer:
  - 40441 MEMIT_BLUE_L5_ONLY, 40442 AlphaEdit_BLUE_L5_ONLY
  - 40443 MEMIT_BLUE_L6_ONLY, 40444 AlphaEdit_BLUE_L6_ONLY
  - 40445 MEMIT_BLUE_L7_ONLY, 40446 AlphaEdit_BLUE_L7_ONLY
- native (blue=False, original five-layer) baseline:
  - 42657 BASE_ALPHAEDIT
  - 42658 BASE_MEMIT
- SH2 완료 PRE_EDIT42673은 봉인된 W0 publication만 재사용하며 새 server2 job 조회/평가를 하지 않는다.

Canonical roots:
- /data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-cell0-checkpoint-r3/
- /data/janghj/ODE-edit/local/blue-lifelong-b100x100/attempt-checkpoint-r2/
- /data/janghj/ODE-edit/local/blue-lifelong-b100x100-l567/attempt-v1/
- /data/janghj/ODE-edit/local/fixed10k-native-baselines/attempt-v1/

공유 fixed10k ordered root=5b01356928ad2e2e46effad5a2c5621c87746b610992b7b92bf3ace16b6f5729, dataset SHA=3d7f5e31db47b23f3a281bb6fb447c2fb1776e3d4721cdfb5fe5c6f998f10ae1. 정책/데이터는 /data/janghj/ODE-edit/local/datasets/counterfact-fixed-10k-v1/ 에서 read-only 검증한다. 새 표본 추출·순서 교체 금지.

기존 정본: experiment-reports/servers/server4/blue-l4-l8-lifelong-b100x100-review-2026-09-09-v3/factual-report-ko.md, SHA839a903839743ad6de7c8cb08d255e7ad2e2c43099f6b81b27a3c91483fb991f.
W0 RS791/10000, PS1997/20000, NS89212/100000은 이 publication에서 source/order/config identity와 함께 재사용한다. W0를 arm마다 독립 측정으로 복제 계수하지 않는다.

### 보조: server4 완료 과거 실험의 요약·연결

BLUE 1k original/L4/L8(38940/38997/38988), JVP-L8 takeover38433_4/5와 JVP/O 1k의 기존 sealed publication을 비교 배경으로 인덱스에 포함한다. 1k prefix와 final10k를 동일 규모의 결과로 합치지 않는다. SH1 ABC/SH2 EP 같은 다른 서버 진단은 필요시 완료 보고서 링크만 사용하고 재감사/원격 raw 탐색/모델 실행하지 않는다.

ORBODE37649 누적 보고서는 기존 봉인 publication에 대한 요약 링크와 제한만 허용한다. **STOPPED_USER_REVIEW correctness audit는 재개 금지**. 사용자 승인으로 raw가 삭제된 사실과 raw 재검산 불가능 경계를 유지한다. 이번 포괄 리뷰 요청을 중지 감사 재개·raw 복구·rerun 승인으로 해석하지 않는다. 다른 오래된 server4 task는 inventory의 기존 보고서 링크와 보고 상태로 정리하고, 최근 BLUE/native 핵심 비교를 지연하는 전 저장소 재감사로 확대하지 않는다.

## 3. 허용 관측과 무변경 경계

정확한 위 job IDs/children에 대해 **단발 bounded scheduler 확인**을 허용한다. Scheduler COMPLETED와 raw terminal-valid는 구분한다. RUNNING/PENDING이면 상태만 남기고 작성 중 raw/checkpoint를 읽지 않는다. 완료된 다른 arm의 분석은 계속하며 미래 완료까지 polling/sleep/heartbeat를 만들지 않는다.

완료·봉인 source/raw만 읽는다. 새 평가(model forward), editing, checkpoint GPU continuation/replay, compute_z, stats/P 생성, tuning/sweep, Slurm submit/cancel/restart/requeue/throttle 변경은 모두0. 자산 삭제/overwrite, 기존 reports 수정·덮어쓰기0. 서버 cap2 정책을 유지하되 이번 task는 GPU0이다. CPU checkpoint 검증은 필요한 selected weight/state를 weights_only로 순차 검사하고 host memory/I/O를 제한한다.

기존 검증된 6개를 무조건 전체 재해시하는 중복 작업 대신 기존 manifest/receipt를 결속한다. 신규 L567/native terminal 결과는 필요한 raw/state/CP 검증과 독립 metric reduction을 수행한다. 재사용 검증, 새 file rehash, CPU tensor 검증, 미실행 GPU replay를 별도로 표시한다. 실제 오류가 보이면 근거·영향 범위를 보고하되 런타임 수정/재실험을 자동으로 하지 않는다.

## 4. 반드시 분리하여 검토할 구현·비교 조건

- 실행 source/config/archive와 analysis/report commit을 분리한다.
- BLUE와 base native는 blue=True/False, native layers, target 계산 위치/횟수와 residual divisor가 다르다.
- Single-layer config는 해당 물리 layer 하나, layer-local target/readout, 본래 native equation을 확인한다.
- AlphaEdit full 5-stack P의 물리 layer4..8 → asset index0..4 → single-layer local0 mapping을 실제 tensor identity로 확인한다. L4 과거 observer metadata 오류는 이미 알려진 사실과 실제 P0 identity를 구분하여 유지한다.
- Native AlphaEdit L2=10 vs BLUE L2=1 등 실제 config 차이를 전면 공개한다. 동일 sample이라는 이유로 layer만 바꾼 통제 비교 또는 method-only 인과효과라고 주장하지 않는다.
- MEMIT covariance/precision 예외, Alpha history append와 selected weight/state 저장 범위를 확인한다. MEMIT에 가짜 history를 만들지 않는다.
- W/M의 batch 연결, append count, restore evidence, current/full-seen evaluator가 가리키는 actual state를 구분한다.
- 모델/tokenizer/revision/context/seed/precision/backend/evaluator/prompt inventory가 일치하는지 표로 정리한다. 다른 config/stream/서버 결과를 paired 비교에 무조건 넣지 않는다.

## 5. 상세 결과·분석 요구

먼저 사용 가능한 **final actual W100 full10000** 표를 전달한다. MEMIT과 AlphaEdit는 별도 표로 쓰고 BLUE를 'original'로만 표기하지 않는다. 각 family에 W0/native/BLUE/L4/L5/L6/L7/L8을 명시하며 missing은 NA와 이유를 남긴다.

- Canonical RS/PS/NS numerator/denominator(통상10000/20000/100000), tie failure. NLL(new/true/margin) mean/median/p90/p99/tail과 strict/token secondary 분리.
- 모든 저장 checkpoint [1,5,10,20,30,40,50,60,70,80,90,100]에서 actual Wk seen-prefix와 current B100, online-at-write aggregate를 별도 표시한다. 평가가 없는 시점은 보간하지 않는다.
- 각 cohort의 at-write 성립, final 유지, 신규 forgetting, 실패 후 recovery, 처음부터 실패, overwrite/collision annotation을 분석한다. 전체 canonical 분모 및 conditional 분모 모두 보존한다.
- Early/middle/recent cohort, 첫100/500/1000 edit와 후반 cohort를 비교한다. FinalW100 retention을 current-B100 pooling과 혼동하지 않는다.
- 동일 prompt identity가 존재하면 W0→endpoint 및 native/BLUE/단일-layer 간 NS loss/recovery를 paired로 분해한다. True NLL 약화와 competing-new NLL 강화는 별도다. Row index만으로 pairing하지 않는다.
- L4–L8 위치별 신규 editability/PS/locality/retention/비용의 차이와 BLUE(L4+L8)의 상대적 trade-off를 두 family 각각 해석한다. 여러 layer 사용 또는 locality 총점 상승 자체를 성공/인과로 간주하지 않는다.
- CP/로그에 이미 존재하는 실제 write norm, 누적 ΔW, history·covariance 관련 관측은 근거 범위에서 사용한다. 추가 geometry/model forward를 위해 대규모 재구성을 먼저 완성하려고 결과 표를 지연하지 않는다.
- 신규 finite poor outcome을 제외하거나 완성 prefix를 final10k로 표시하지 않는다. 실패/취소/이전 superseded run은 비용과 분모 영향을 분리한다.
- 실제 allocated GPUh/wall, write/compute_z/eval/snapshot breakdown, output 저장량을 비교한다. 없는 구간은 NOT_RECORDED. 다른 GPU·환경의 wall 차이를 통제된 speedup으로 부르지 않는다.
- 필요시 이미 확보된 prompt-level rows로 request-cluster paired bootstrap을 하되 checkpoint별 표본 증가, 반복 Fixed cohort, 다중 탐색을 독립 반복으로 오해하지 않는다. 효과크기·분모·trade-off를 우선한다.

신규 native baselines가 아직 완료되지 않았으면 'NOT_AVAILABLE'를 유지하되 현존 결과를 보고한다. Native가 완료·유효한 경우 이전 v3의 미측정 baseline 공란을 새 버전에서 채운다. 기존 v3 bytes는 변경하지 않는다.

## 6. 산출물·소유 경로와 전달

전용 branch 권장 codex/server4-experiments-detailed-review-20260911-v1.
새 worktree 권장 /data/janghj/ODE-edit/local/worktrees/server4-experiments-detailed-review-20260911-v1.
동일 경로가 있으면 고유 version을 사용한다.

허용 write:
- project/run_scripts/server4_experiments_review/ : CPU reducer/analysis/tests/plot source만
- local/server4-experiments-review/20260911-v1/ : local scratch·비공개 prompt-level 분석/verification
- experiment-reports/servers/server4/blue-native-lifelong-comprehensive-review-2026-09-11-v1/
- audits/servers/server4/2026-09-11-completed-experiments-review/
- messages/server-heads/server4/2026-09-11-completed-experiments-review.md
- tasks/status/server4-completed-experiments-review-20260911-v1/server4.json
- runs/server4-completed-experiments-review-20260911-v1/
- plans/updates/server4/2026-09-11-completed-experiments-review.md

보고서 diagnostic-report-ko.md에 전 실험 inventory·availability/compatibility·family별 final 및 checkpoint 표·forgetting/locality·수치/분포·비용·원인 후보와 미분리 요인·근거 경계·미완료 항목을 포함한다. 결과 CSV와 직접 코드 생성 PNG, source/manifest/receipt 및 재현명령을 제공한다. Imagegen/이미지 생성 도구 사용0. Raw weights/cache/prompt/log는 Git0이며 기존 raw 이동/삭제/새 대형 transfer도0; raw broadcast는 analysis-only/no-new-raw 및 local-only 보존 이유를 명시해 생략한다.

진행: FULL_READ/단발 status inventory → 첫 family별 final 표 → 상세 분석/report → 본인 완료 scope 최신 main non-force 통합/push → report 경로/SHA/HEAD/한계 compact handoff. 이전 사용자 main 통합 정책에 따라 source+raw-free report의 본인 scope push 승인, conflict 시 중단·보고. Main에는 다른 agent 미완성 code를 임의 병합하지 않는다.

SH4가 전 과정을 담당하고 GH의 중복 점검/재승인을 기다리지 않는다. 이미 가능한 분석은 계속한다. 새 실행이 필요한 측정 공백은 보고서에서 정확히 표시하며 자동 새 run으로 채우지 않는다. 완료 후 STOP, 타 task monitoring은 재개하지 않는다.
