# GH → SH4: delayed-write E3 완료 결과 상세 리뷰

Instruction / nonce: ODEEDIT-GH-SH4-DELAYED-E3-COMPLETED-REVIEW-20260924-R1
Parent: GH-SH4-NATIVE-DELAYED-WRITE-E3-20260924-V1.
사용자: “server4에게 끝난 실험 리뷰 시키고 보고 명령해”.
SH4 session01a04939-b5c7-7a03-ba2d-ef3343d62cfd,
server4 / /data/janghj/ODE-edit / hyunjun1127/ODE-edit.
GH session01a04939-8873-7673-8dca-4c7fc5e31af0.

## 1. 명시 recall 및 권한

사용자의 이번 호출로 해당 task 결과 회수·CPU 상세검산·최종보고만 재개한다.
직전 user-pause-r1의 initial 이후 monitoring 중단은 역사대로 보존한다.
첫 E1 90/236 관측을 이후 완료로 소급하지 않는다.
GH direct inspect에서 현재 SH4 active turn 없음, 직전 turn completed 확인.
다른 과학 task 재개·새 GPU·모델 load/forward·evaluator·native fitting/write·
Slurm submit/retry/release/cancel/hold/dependency 변경·checkpoint 저장0.
이전 실행 envelope의 기술수리/재제출 권한은 이번 read-only review가 재발동하지 않는다.
이번 task는 cap2 유지와 무관하게 신규 GPU0, 기존 작업/dirty/source/raw/CP 보존이다.

Exact 지정 job은 GPU52823 odeedit_delayed_E3_science_s4 (G00–G60)와
CPU52824 collector afterany:52823 (G70/실패 수집)다.
제출 원자료는 lock ff2c8fd6b685a933b980b2255504200b1744a8827dbae128ee1aa4da38bd3b71,
execution3ebe0b07078940c2d46f9ea2226ccc20c0446162,
submission c6569653df755513837cd0b80a57a1320c12e52c27a038bb2f31c1aa7376054b.
원 SOURCE와 분석source를 별도로 결속한다. 현재 main 구현을 runtime이라고 대체하지 말라.
local root /data/janghj/ODE-edit/local/native-delayed-write-e3/20260924-v1/.
기존 storage-block→공간확보/제출→G00/G10 PASS→사용자pause provenance를 모두 보존한다.

정확 submission/job mapping과 owner를 먼저 확인하고 52823/52824 한정 sacct/squeue
snapshot을 한 번 수집한다. accounting 지연 때만 근거 있는 제한 재조회.
실제 RUNNING/PENDING이면 NOT_TERMINAL과 완료된 불변 receipt만 보고하고 대기/반복polling0.
사용자 '끝났다'는 표현만으로 COMPLETED/전체 scientific-valid라고 확정하지 않는다.
collector exit0도 G00–G60 완료의 대체 증거가 아니다.
기술 실패·누락이면 RCA/차단 stage/이미완료범위까지 보고하되 새 실행은 하지 않는다.

## 2. 읽기·분석 범위

원 instruction, e3-dependency-plan, dispatch-input-manifest, review/minimal-protocol,
original protocol/assets/math/cells, execution lock/source, 실제 panel manifest,
submission/release/초기 gate/user-pause receipt를 읽고 서로의 SHA를 대조한다.
기존 exact FULL_READ는 같은 bytes일 때 재사용하고 새 override/실제 runtime는 직접 확인한다.
원243cells가 아닌 승인된 E0 endpoint→E1→E3 12조합만 완료분모로 삼는다.
BASE_ALPHAEDIT42657/BASE_MEMIT42658, W0+24저장endpoint, original blue=False lineage를
명확히 하고 최신 upstream/BLUE/L4-only/다른 alpha-key task와 섞지 않는다.

첫 중간보고는 exact accounting·stage coverage와 실제 독립 수치표를 조기에 보낸다.
첫 표가 전체 검산 PASS는 아니다. 최종 report/main 게시까지 이번 CPU scope를 계속한다.
중간표/결과가 음성이거나 gate 일부실패라고 분석 자체를 중단하지 않는다.
추가 GH raw/GPU 중복감사나 단계별 승인을 선행조건으로 요구하지 않는다.

## 3. 필수 상세 검산

1. 실행/DAG: G00/G10/G20/G21/G30/G31/G40/G50/G51/G60/G70 각각
   PLANNED/COMPLETED/TECHNICAL_FAIL/BLOCKED/NOT_RUN 및 receipt SHA/source/input/
   panel/attempt 연결을 독립 점검. 실제 registered afterany와 내부 matching atomic PASS,
   first failure propagation/collector 출력/유실·중복 stage 여부를 설명한다.
   G10 original/repeat 각family B1 true/new16rows max0 등 bounded scope를
   전체panel/allstate/model numerical parity로 확대하지 않는다.
2. E1 coverage: 공통 W0+family별12=25 logical endpoint와 N_diag1000,
   H B1 R100/P200, BaseEval256, GeneralEval128. completion rows와 paired prompts/
   unique cases/documents를 구분. supplementary active-target11은 원 H분모와 분리.
   Base strata86/85/85와 다른relation적합후보0, alias resolver NOT_AVAILABLE 등
   사전 패널한계 및 실제 사용량을 확인한다. General128 W0 reference,64-token window,
   full-vocab KL과 TF 처리/종료/padding 규칙을 실제 source+row에 결속.
3. 독립 raw reducer: case/prompt/target/token order/denominator/finite/ties와 state SHA를
   확인하고 true/new NLL·m=true−new·g=−m·success/lost-gained를 원 row에서 재집계.
   NLL부등식 preference와 저장된 TF rewrite/rephrase/neighborhood accuracy,
   token-micro/prompt-macro/strict를 구분해 보고. 미저장지표는 NOT_RECORDED이며
   새 forward 없이 계산가능한 것만 계산한다.
   Active/superseded/현재target/원B1target, W0 behavior보존/정답성능을 분리한다.
4. E1 module: L4 고정prefix key불변, L4–L8 key drift 및
   WtKt−WsKs=W0δK+HsδK+Fs,tKt의 source·실측 residual/relative error·signed norm/
   contraction. module mapping과 최종 NLL 인과기여율을 혼동하지 않는다.
5. E3 factorial: core4+추가horizon4+누적prefix4 실제 coverage.
   Alpha(1,50)/(1,90)/(1,10)/(1,100)/(10,50)/(50,90),
   MEMIT(1,10)/(1,20)/(1,5)/(1,100)/(5,10)/(10,20).
   A는 L8만/B는 L4–L7만, 누적A와 B1component 구분,
   θref=θt−A−B 및 나머지weight고정/00·10·01·11 binding.
   K11=K01,K10=K00, v11−v01−v10+v00=AδK를 실측 row에서 검산한다.
   module항등식 성립과 nonlinear f11−f01−f10+f00를 분리한다.
6. E3 patch: actual11에서 v11−λAδK, λ0/.5/1/−1,
   token별 RMS맞춘 rotation/sign-permutation3seed와 모든valid TFtoken 반영,
   true/new tokensequence별 독립정렬/패딩제외/zero-vector 처리/rollback 확인.
   λ0=actual11 재사용 exactidentity 근거, 추가72modified와48logicalfactorial을
   실제 forward수/job수/재사용수와 구분. source 함수/라인→artifact worked example 포함.
   한 문항 subject key/context평균으로 alltoken K를 대체한 적 없는지 확인한다.
7. 실제 family/horizon/panel별 표: factorial 네상태와 interaction,
   actual11대 λ.5/1/−1·각rotation의 paired true/newNLL/strict/success,
   회복·악화ID/absolute counts/분모·mean·분포/q95/q99/tail.
   전체고정panel 주분석, lost-only는 보조. sign별·family별 반대증거/해로운patch도 포함.
   dose와RMS대조가 동일단위인지, 효과없는조합도 누락없이 보고한다.
   case/subject/repeatedprompt cluster bootstrap/불확실성은 사전seed/계약 유지.
   고정trajectory 문항CI를 새order 재현성/여러 독립실험으로 확대하지 않는다.
8. 비용/보존: parentGPU allocation(52823)과 CPUcollector(52824)를 구분,
   신규E1/E3/최소기술/준비·원baseline 재사용/실패·재실행/중첩timer를 분리.
   endpoint/materialization/capture/patch/observer/I-O/peak가 미계측이면 명시.
   noCP/no새native/history 확인. 출력inventory/size/fullSHA·실제immutable inputs/
   재사용priorSHA+stat를 구분하고 무결성 이상 없이 공용model/24CP를 중복 fullrehash0.
   missing/partial/corrupt를 삭제·보정하지 않는다.

## 4. 판정·보고와 작업 경계

SH4는 사실수치/설계구현적합성/기술RCA/반대증거/한계를 상세히 제시한다.
가설입증/최초성/원인기여율/method채택 또는 E2/E4–E6 확대는 GH 별도 판단이다.
query-specific patch를 배포가능weight repair나 다른학습trajectory라고 쓰지 않는다.
지표를 보고 threshold/패널/row/시간쌍/rotationseed 변경0.
이전 PENDING/NOT_OBSERVED, storageblock, pause를 덮지 않고 새 완료리뷰로 연결한다.

전용 branch codex/server4-native-delayed-write-e3-completed-review-20260924-v1.
새 분석만 project/run_scripts/native_delayed_write_e3_completed_review/ 허용.
원 runtime project/run_scripts/native_delayed_write_e3/와 frozen archive는 변경0.
보고/CSV/코드PNG/manifest/재현명령:
experiment-reports/servers/server4/native-delayed-write-e3-20260924-v1/completed-review-r1/.
대표 report-ko.md, execution/source/analysis분리, stage-coverage.csv,
paired-summary.csv, artifact-index/package-manifest/rooted-receipt 등을 보존.
Audit: audits/servers/server4/native-delayed-write-e3-20260924-v1/completed-review-r1/.
Plan: plans/updates/server4/native-delayed-write-e3-20260924-v1/completed-review-r1/.
ACK: messages/acks/server4/2026-09-24-native-delayed-write-e3-completed-review.md.
Message: messages/server-heads/server4/2026-09-24-native-delayed-write-e3-completed-review.md.
Status: tasks/status/native-delayed-write-e3-completed-review-20260924-v1/server4.json.
Run: runs/odeedit_native_delayed_write_e3_s4_20260924/completed-review-r1/.
Local scratch: local/native-delayed-write-e3/20260924-v1/completed-review-r1/.
전용 ignored session-boundary.env만 실제childCWD로 허용, shared정책/환경/config변경0.

Bounded pre/post CPU source/분모/행/hash/링크/표 실제렌더/코드그림 재현 검산.
독립red사용 여부와 자기검산을 정직히 구분한다. Renderer미설치는 미검증으로 명시.
완료된 원 CPUcollector 출력은 재사용하되 별도독립 raw reducer/계약검사와 구분한다.
분석 오류 최소수리는 새 분석source만 변경, 원 실행오류/실패는 그대로 남긴다.
Raw/tensor/prompt/log Git0, source+raw-free보고 ownscope nonforce cleanintegration
main 게시 허용. 다른변경 보존, conflict면 보고. NO_BROADCAST_NOT_REQUIRED 근거기록.
Final: exactjobs/state/실제 coverage/핵심수치/실패·한계/비용/source/main/reportSHA와
경로를 GH direct로 보내고 TASK_COMPLETE_STOP. 이후 모니터링/자동 recall0.
