# JLZ v5 server4 두 arm 2000 edit 실행 설계

2026-10-02 KST. 사용자가 현재 JLZ 실험 취소와 v5 구현·실험을 server4에 명시적으로 지시했다. GH가 SH4에 배정한다. [Method](../method-ko.md), [contract](../contract.json), [구현 명세](../implementation-ko.md)가 방법의 정본이다. 이 문서는 실행 범위이며 v4 알고리즘을 계승하지 않는다.

## 1 취소와 실행 범위

SH4는 기존 task `jlz-native-joint-v4-bs100x20-20261002-v1`의 현재 job/배열/collector/retry 의존성을 조회한다. 마지막 알려진 A/B는57292/57293이며, 이 ID를 현재 상태 확인 없이 재사용하지 않는다. 아직 RUNNING/PENDING인 동일 task의 작업을 취소하고 종료 상태를 확인한다. 이미 끝난 job은 종료 상태만 기록한다. 원자료·완료 batch·평가·source·log를 삭제하지 않는다. 종료가 임박한 collector는 강제 2k-complete 보고를 하지 않도록 partial/cancelled로 기록한다.

다른 server의 실험이나 server4의 무관한 job은 취소 범위가 아니다. 기존 v4의 진행 중 모델에서 v5를 이어 쓰지 않는다. 새 A/B 각각 원본 W0, H0=0, empty memory로 시작한다. 기존 STOP/no-new-submission 정책에 대해서는 이번 v5 task만 현재 사용자 예외로 등록한다.

## 2 고정 실행 profile

- 신규 chain은 **v5 A η=0 / B η=1 두 개만**. Baseline 신규 main/pilot/fallback은 실행하지 않는다.
- 기존 fixed10k의 같은 첫2000 요청, 같은 순서·target·native context, BS100×20 sequential commit. Method interface는 arbitrary B/partial batch/model/benchmark를 유지한다.
- 이번 실행 모델/tokenizer/C0/attention/runtime는 검증된 server4 v4 profile을 fingerprint로 결속한다. 전체 eligible L4–L8, native6+KL1, FP32 model/FP64 geometry, no autocast/TF32를 유지한다.
- 모든 candidate는 고정 full-context P의 실제 all-token 모델. 동일 accepted W_eff commit, current-key 재-solve0. D norm은 nominal local payload의 native-form norm이다.
- Norm=.5, native clamp=.75, current KL=.0625, C0 coefficient15000. B의 λK=.0625, λE=1, memory128, reference cap16, native 문장만 재사용.
- 후보당 전체 logical batch 공동 gradient와 공통 scalar proximal step. 최대25 candidate/24 backward, 거절 포함, 마지막 accepted 후보 반환. Adam24 update로 바꾸거나24수락을 보장하려 예산을 늘리지 않는다.
- Native input/readout/gradient/commit의 기술 오류는 수정한다. 낮은 RS/PS/NS, 한 층 집중, 고정 예산 미수렴은 main 중단·층 제거·예산 변경 gate가 아니다.

## 3 작은 pilot과 main

1. CPU 수식 11개 항목군·입력 identity·가변 shape를 검증하고 실제 tokenizer parity를 확인한다. 현재 설계 검산을 GPU 구현 검증으로 대신하지 않는다.
2. Main 밖의 개발 요청4개를 사용해 BS2 두 sequential batch의 작은 A/B 정합 pilot을 수행한다. 기존 개발 slice `[2000:2004]`는 main 첫2k와 구분해 기록한다. 이 입력의 평가 품질로 η·보존 계수를 선택하지 않는다. 최대 후보25의 정규 규칙 안에서 B1의 A/B 목적 동등성, B2의 replay gradient, commit·H·memory·rollback을 확인한다.
3. 실제 B100에서 3 candidate 정도의 짧은 timing을 측정한다. 준비비·메모리·reference·materialization·실제 F/B를 분리한다. Timing/pilot의 W/H/memory는 main에 이어 쓰지 않는다.
4. 기술 정합과 자원 조건이 갖춰지면 추가 사용자 확인 없이 cold A/B 2k main을 제출한다. 성능 gate는 추가하지 않는다.

## 4 평가와 비용

매 commit의 current100 전량, W5/W10/W20의 all-seen500/1000/2000을 평가한다. 최종 R/P/N 분모는2000/4000/20000이다. 기존 strict-preference, teacher-forced strict, cohort at-write→endpoint retention, NS true/new NLL·lost/gained, active/superseded 정의를 유지한다. Milestone current는 동일 endpoint raw에서 재사용한다.

W0는 입력/model/evaluator/runtime identity가 일치하는 기존 검증 결과를 재사용한다. 새 W0 전체 평가를 기본으로 재시작하지 않는다. 재사용 근거가 불일치하면 원인과 필요한 범위를 GH에 보고한다. W0 평가 재사용과 main 모델을 원본 W0에서 시작하는 것은 별개다.

최대 A/B 합계40 batch×25=1000 candidate 평가이며, line search의 거절도 포함한다. 실제 accepted update 수, 전체 native/reference 처리 token·head·F/B·geometry·CPU transfer·memory를 기록한다. 25candidate가 v4와 같은 runtime이라는 뜻은 아니다. ETA는 pilot 실측과 부대 비용으로 계산한다.

## 5 자원과 인계

실행 owner는 SH4, repository `/data/janghj/ODE-edit`, session `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, repository ID `hyunjun1127/ODE-edit`다. GH는 최신 registry와 cap를 다시 확인한다. 기존 기준 project cap2/job1GPU/host60416MiB 안에서 A/B 두 lane을 배정한다. 이외 작업을 선점하거나 사용자 model 설정을 바꾸지 않는다.

허용 신규 namespace는 `project/run_scripts/jlz_writer_coupled/`, 이 v5 plan/TeX, server4 소유 task/status/report/audit/run 경로와 ignored `local/jlz-writer-coupled-v5/`다. 기존 source는 정독·재사용할 수 있지만 다른 진행 중 runner를 hotpatch하지 않는다. Source/문서/소형 manifest만 Git에 게시하고 raw/model/대형 tensor·복원 동등 checkpoint는 영속 저장하지 않는다. W/H/memory/RNG transaction은 RAM에서 처리한다.

SH4는 취소 receipt, v5 task 수락, 실제 구현 commit, pilot·timing, 제출 job IDs, 대표 main B1 commit→B2 entry 또는 정식 resource-pending을 구분해 보고한다. 초기 확인 뒤 등록된20batch runner/collector는 자연 진행한다. 장기 agent polling·새 heartbeat·자동 재시도 task를 이번 지시만으로 추가하지 않는다. 완료 산출물은 실제 per-case raw와 W5/W10/W20·retention·배분·비용 보고서다.

## 6 정본 입력 schedule

원 요청 순서의 기준은 `plans/global/2026-10-02-jlz-native-joint-v4/experiment-2k/case-schedule.csv`, SHA256 `dccb4da4896aa7c3f653f7192bd01ab5d8799a0b439101417d02ee287d3fd2e2`다. 이 파일과 기존 evaluation schedule은 데이터·평가 identity의 근거로 전달한다. v4의 Adam/subject injection/current-key writer/추가 V/no-replay 규칙은 v5에 적용하지 않는다.
