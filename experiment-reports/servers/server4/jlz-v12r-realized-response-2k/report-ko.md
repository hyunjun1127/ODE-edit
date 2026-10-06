# v12-R realized-response 2k 실행 인계

2026-10-06 USER recall에 따른 최신 MAIN 중간 결과는 [W15 / 1,500-edit 보고서](intermediate-main-W15/report-ko.md)에 게시했다. 원 NLL/TF 행 CPU 재집계와 저장 집계가 일치했고, 15 commit·14 상태 연결·75 history append를 검산했다. W15 RS/PS/NS는 99.333% / 90.033% / 69.607%다. B16에서 `No space left on device`로 실패했으며 W20 결과는 없다. [소형 집계 manifest](intermediate-main-W15/manifest.json)에 성능·retention·실현·비용 파일을 결속했다.

아래는 최초 등록 당시의 역사적 인계 기록이다. 이후 완료/실패 및 중간 관측의 현재 증거는 위 보고서를 따른다.

Instruction/nonce: `USER-GH-SH4-JLZ-V12R-20261006`.
Task: `jlz-v12r-realized-response-2k`, 담당 SH4.

현재 등록 초기 인계 보고다. Owner CPU39 PASS, 별도 independent source 검토는 `PASS_WITH_LIMITATIONS`/미해결 확정 blocker0, final independent operational20 PASS다. 실제 pretrained GPU qualification과 MAIN B1/W20 측정은 `NOT_OBSERVED`이며 CPU fixture 통과를 GPU PASS로 표시하지 않는다.

## 실제 등록·봉인

실행 source `635798ba276957312ec1aceda686ad563c906957`, source tree `41d95dee323a6b4ad705ccedddc740e27c0a1dd2`. Config SHA `718a6b1428000e26d726e7094b170d715b1c3596e8821e4a30b6ba751a0b8d1a`, lock SHA `e7ca36fd76f0c0cfa3824c72694545e3fd36ecd8f5a4fe011dfde63a87e6360e`. Source archive는 170파일/1,638,400bytes이며 SHA `d64a2f9bed000c6be194108a9fef01668c0b6ffa2fce3d30be256a558f3cc26e`다.

| 역할 | Job | dependency |
|---|---:|---|
| qualification | 59261 | 없음 |
| MAIN | 59262 | afterany59261 |
| BLIND | 59263 | afterany59262 |
| L4-ONLY | 59264 | afterany59262 |
| BASE-2X | 59265 | afterany59263 |
| NO-EXPAND | 59266 | afterany59263/59264/59265 |
| CPU collector | 59267 | afterany 정확6GPU parents |

모든 job의 owner/fullargv/script bytes/source/node/CPU/GPU/explicit memory/wall/exportNONE/Requeue0/dependency를 held 상태에서 검산한 후 전량 release했다. 2026-10-05T22:24:11Z 한정 scheduler snapshot에서는 qualification RUNNING(server4), 나머지6개는 Dependency PENDING이었다. 이후 scheduler/log/결과 polling은 중단했다. MAIN이나 대조군의 성적은 dependency 조건이 아니다. Runner 안의 exact qualification READY 및 공통 source의 확인된 기술 오류 검사를 유지한다.

등록 receipt: `runs/jlz-v12r-realized-response-2k/submission.json`. Local 원본 receipt/lock/source/launchers: `/data/janghj/ODE-edit/local/jlz-v12r-realized-response-2k/attempt/`. 봉인 CPU collector 결과 목적지는 같은 attempt의 `collector/`이며 아직 읽거나 완료를 주장하지 않았다. 실행 source와 뒤의 게시/분석 commit은 구분한다.

## 범위

MAIN, BLIND, L4-ONLY, BASE-2X, NO-EXPAND가 각각 독립 cold W0/H0에서 같은 ordered first2000을 BS100×20으로 실행한다. MAIN 우선, priority2 대조군, 마지막 NO-EXPAND의 자원 순서이며 성능 promotion gate는 없다. 실제 MAIN B1이 첫 B100 fit이며 별도 B1/BS2 refit은 없다.

원문 550줄과 method/implementation/experiment/validation/clarifications 전체를 읽고 manifest의 8개 member size/SHA를 검산했다. 원문 SHA는 `23fedf548b88637f1c0daea7c5e2809600e8821d6e60ce5961b29f015f683e92`, manifest SHA는 `1d947d08d7bd5e1680115ebfc5448ad606e3e42714b3ae7758c9321f8a1db10c`다. 원문 CPU reference source는 제공되지 않아 완료표를 `NOT_VERIFIED`로 보존한다.

CPU source fixture는 projection·absolute Adam·active-mask·25/24·transaction·observer·partial reducer·DAG를 검산한다. Tiny random CPU Llama forward/backward를 포함하며 실제 pretrained 모델 asset load/GPU 실행은 아니다. 독립 reviewer의 source/CPU 검토와 owner 테스트를 구분한다.

GPU qualification은 main 밖 4요청의 고정 후보 총 3개 상한이며 fit/update/permanent commit 0이다. 생산 gradient는 same-layer FP64 Lambda M.T로서 full-builder reverse는 qualification/진단 전용이다. Lower-layer full-gradient 차이와 subject/all-token loss 차이는 관측값이다.

## 평가·미관측

각 arm W0 first2000, 매 batch pre/post current100, W5/10/15/20 all-seen 및 fixedfirst500/paired/cohort를 관측한다. W20 전체 분모는 R2000/P4000/N20000이며 미측정 분모·성적을 0으로 대체하지 않는다. 예상 총 commit은 100, history append는 420이다(L4-ONLY 20, 나머지 각100).

W20/RS/PS/NS/TF/NLL/Q/실현/실측 시간: `NOT_OBSERVED`. 새 baseline fit 0, 조건 검산 없는 역사 자료는 `HISTORICAL_REFERENCE/NOT_AVAILABLE`이다. 원문 500초/4.7GPUh는 미측정 참고이며 이번 실행 ETA가 아니다.

## 보존·자원

전용 non-main branch는 `codex/server4-jlz-v12r-realized-response-2k`, source namespace는 `project/run_scripts/jlz_v12r/`다. 이전 Causal Allocation Editing USER_STOP과 다른 task/source/raw/dirty root는 유지한다. 기존 job 취소·변경 0이다.

기본 1GPU/8CPU/59392MiB, host hard ceiling60416MiB, exportNONE/Requeue0. Task 동시 GPU≤2, fresh server4 project cap≤3 및 더 엄격한 정책을 적용한다. Main 요청 wall≤48h, qualification≤4h, CPU collector≤4h/8CPU/24576MiB/GPU0. 요청 wall은 ETA가 아니다. 소형 output/atomic/scratch reserve9GiB를 별도로 봉인한다.

NoCP, exact resume `NOT_AVAILABLE`. W/H/R/optimizer/RNG와 복원 동등 bundle은 영속 저장하지 않는다. Source/config/hash/scalar/per-case metric raw만 local 보존하고 Git에는 source·compact receipt/report·manifest만 게시한다. `NO_BROADCAST_NOT_REQUIRED`: 동일 S4 자산 재사용과 Git 소형 인계로 별도 대형 전송이 필요 없다.

정식 held 검사·release 후 resource/dependency pending snapshot에서 agent monitoring을 중단한다. 봉인된 qualification/5-arm runner/CPU collector만 승인 범위를 자연 진행하며 새 heartbeat/automatic retry/B21은 없다. 상세 완료 검토는 사용자 recall 시 수행한다.
