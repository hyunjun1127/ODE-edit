# v12-R realized-response 2k 실행 인계

Instruction/nonce: `USER-GH-SH4-JLZ-V12R-20261006`.
Task: `jlz-v12r-realized-response-2k`, 담당 SH4.

현재 준비 보고다. 실제 job 등록/source lock 및 CPU receipt는 등록 인계에서 아래에 결속한다. 현재 pretrained GPU qualification과 MAIN B1/W20 측정은 `NOT_OBSERVED`이며 CPU fixture 통과를 GPU PASS로 표시하지 않는다.

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
