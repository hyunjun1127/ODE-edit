# JLZ v12 2000 edit 실험 설계

사용자는 v12를 server4에서 2000 edits까지 실행하도록 GH에게 지시했다. GH는 정본을 게시하고 SH4에 구현·qualification·작은 pilot·본실험을 배정한다. 이번 새 지시는 v12 task에 한정되며, 중단된 v11을 재개하는 지시가 아니다.

## 실행 범위

- 본실험은 V12_MAIN 한 경로, Meta-Llama-3-8B-Instruct와 고정 CounterFact first2000, BS100×20이다. 500 edits는 중간 평가다.
- Method 정본은 ../method-ko.md, ../contract.json, ../implementation-ko.md와 TeX다. method 작성 당시 METHOD ONLY였던 실행 권한은 이번 execution-command.json이 새로 부여한다. 원 method의 수식·계수는 변경하지 않는다.
- Pilot은 main과 겹치지 않는 고정 stream[2000:2004]의 4개 요청으로 BS2×2, 작은 native operator parity를 포함한다. Pilot 뒤 cold W0/H0로 새 본실험을 시작한다.
- 기술 검증과 pilot PASS이면 추가 사용자 확인 없이 본실험을 제출한다. ACC·PS·NS·KKT·활성 층 수는 pilot promotion gate가 아니다.
- 추가 arm, 강도 sweep, L4 고정 계획, 새 baseline 실행은 자동으로 늘리지 않는다. 기존 완료 baseline은 GH가 source/input/runtime/evaluator/분모 정합을 확인한 범위에서 비교한다.

## 보존할 method

전체 후보 층의 subject delta 공동 계획, 상대 변화 공유 예산0.75, EfficiencyAdam과 좌표변환 epsilon, 요청별 최대25평가·24갱신, 마지막 평가 후보 채택을 유지한다. KKT는 진단이고 terminal은 추가 backward 없이 null 사유를 남긴다. Write는 actual 하층 편집 후 상층 key/hidden을 재측정하며 절대 local target residual로 native ridge를 푼다. 최종 key history append는 모든 write 후 한 번이다.

## 평가와 산출물

매 batch 편집 전후 current R/P/N을 측정한다. W0 전체2000, W5/10/15/20의 누적 평가를 수행하고 같은 endpoint의 동일 문항은 중복 forward 없이 재사용한다. 최종 분모는 R2000, P4000, N20000이며 ACC(strict/token micro/prompt macro), RS/PS/NS, new/true NLL, paired transition과 cohort retention을 함께 보고한다. 중복·상충 요청을 삭제하지 않고 occurrence-level 기본값과 active/superseded 보조값을 분리한다.

계획 share, 실제 applied-weight 기반 실현 share, inherited mismatch, candidate별 task/norm/total gradient, gamma, projection threshold, 별도 KKT 진단과 종료 사유를 기록한다. 평가와 telemetry를 후보·계수 선택에 사용하지 않는다.

## 자원과 완료

SH4가 server4의 현재 자원과 Slurm 정책을 확인해 명시적 CPU/memory/walltime을 정한다. Task GPU cap1이며 더 엄격한 project cap을 따른다. 기존 다른 작업을 중단하지 않는다. noCP를 유지하고 persistent weight/optimizer/update-factor resume bundle은 저장하지 않는다. 메모리 rollback은 유지한다.

실제 제출 여부는 job ID로 확인한다. 20회 commit과 W20 평가, 산출물 검증 및 SH4 사실 보고가 완료되어야 본실험 완료다. GH·SH4의 담당 수락은 제출·완료와 별도로 기록한다. 예측 ETA는 pilot 실측 후 산출한다.
