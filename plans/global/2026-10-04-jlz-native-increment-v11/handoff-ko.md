# GH 전달: SH4에 JLZ v11 구현 및 2,000-edit 실험 배정

2026-10-04. Instruction/nonce: **ODEEDIT-USER-GH-SH4-JLZ-V11-INCREMENT-2K-20261004-R1**.

사용자 원문:

> method 설계, 실험 설계까지 진행해서 GH에게 전달하라. SH4에게 task를 맡기면 된다. 500 sample 말고 2000개로 진행하는 것으로 하자.

발신자는 사용자 요청을 수행하는 연구 세션 `01a0f6b9-74e5-7683-a798-029e477c29b1`이다. GH를 대신하는 세션이 아니며 GH에게 정본 통합과 SH4 배정을 요청한다. 이 신규 task의 구현, 작은 pilot, 기술 검증 후 본실험은 사용자 명시 승인 범위다. 이전 STOP은 이 신규 task에 한해 예외이며, 기존 task의 재개/취소 권한은 주지 않는다.

## 수신과 담당

- GH: `01a04939-8873-7673-8dca-4c7fc5e31af0`, lab120/server1, `/mnt/raid5/janghj/ODE-edit`, repository `hyunjun1127/ODE-edit`.
- SH4: `01a04939-b5c7-7a03-ba2d-ef3343d62cfd`, lab163/server4, `/data/janghj/ODE-edit`, 같은 repository.
- Task: `jlz-native-increment-v11-bs100x20-20261004-v1`.
- 새 전용 non-main worktree/branch와 `project/run_scripts/jlz_native_increment/`를 사용한다. GH/SH4 모델·reasoning 설정은 사용자 관리값을 그대로 따른다.

GH는 동일 nonce의 기존 접수/배정/제출 기록을 먼저 확인해 중복 실행을 막고, 이 전달의 수신 ACK와 SH4의 직접 수락 ACK를 회수한다. ACK·구현·pilot·본실험 제출·완료를 별도 상태로 보고한다. 접수했다고 실험 시작으로 표시하지 않는다.

## 정본 패키지

1. [Method 설계](method-ko.md), [method 계약](contract.json)
2. [구현·qualification 계약](implementation-ko.md)
3. [2k 실험 설계](experiment-2k/experiment-ko.md), [실험 JSON](experiment-2k/experiment.json)
4. [입력 identity](experiment-2k/input-reference.json), [고정 first2000 순서](experiment-2k/case-schedule-first2000.csv)
5. [Method TeX](../../../docs/methods/jlz-native-increment-v11.tex)
6. [CPU 검산](math/validate_contract.py), [수치 결과](math/results.json), [검토 기록](math/review.json)
7. `artifact-manifest.json`: 위 산출물의 상대 경로·bytes·SHA256.

Method와 실험문서 둘 다 SH4에 전달한다. 이전 v10 혹은 사용자 검토용 absolute-target v11 초안을 실행 명세로 대신 쓰지 않는다. 정본이 없는 참조는 production source snapshot/입력 identity에 한하며 실제 경로·hash는 SH4에서 재결속한다.

## 바뀐 method의 핵심

- 모든 L4–L8에서 direct local δ를 subject 위치에 공동 주입한다. Native 문장/NLL/KL, canonical anchor 기반 layer norm을 유지한다.
- Fit geometry는 batch entry W/H/key로 한 번 만든 full-batch G+E 비용 proxy이며 candidate 동안 고정한다. Candidate마다 actual builder 또는 T′ 실현-v 주입을 하지 않는다.
- Main은 A형 층별 root 합 비용 λalloc=.1, 대조군 NOALLOC은0이다. NOALLOC은 uniform이 아니다. 예전 A/B 결합형태 둘을 다시 실험하지 않는다.
- q Adam .1, layer별 native-form clamp .75, 최대25 후보/24 update. 전체 `J_mean<.05`이면 logical batch 전체 종료하며 allocation까지 포함한다. 요청별 freeze나 NLL-only 충분성 stop은 없다.
- 최종 D를 고정하고 순차 native ridge writer가 current actual key를 매 층 갱신한다. Residual은 D 자체이며 absolute `z−h_current`, L8 divisor, inverse-M, 하층 오차 top-up은 없다.
- Final native mean-key history를 CPU FP32로 정확히 한 번 append한다. Prefix/model/geometry의 기존 안전한 캐시는 의미 보존 qualification 뒤만 재사용한다.
- Reference/replay/pulse/새 부수효과 KL/warm-up/계수 sweep/강도 matching을 추가하지 않는다. Entry cost와 actual cost 차이는 기록하며 semantic quality gate로 삼지 않는다.

## 실제 실행 범위

**동일 2,000개 요청을 각 방법에서 독립 cold W0/H0로 수행**한다. 요청을 나눠 세 방법 합계2,000개로 만드는 것이 아니다.

1. CPU 및 실제 모델 native/operator qualification.
2. Main 밖4개 요청(case541/6693/16935/17306)으로 MAIN·NOALLOC 각각BS2×2, native MEMIT-H 작은 parity pilot.
3. MAIN: BS100×20회, 2,000 edits.
4. NOALLOC: 동일 요청·순서, 독립 BS100×20회.
5. MEMIT-H: 동일 server4 runtime/모델/입력/evaluator에서 독립 BS100×20회. 모든 필수 endpoint와 정확한 identity가 일치하는 기존2k 결과가 있다는 것을 GH가 검증한 경우에만 재사용할 수 있다. 과거500 결과나 다른 runtime 비교표로 대체하지 않는다.

기술 PASS 후 각 단계마다 새 사용자 확인을 기다리지 말고 승인된 다음 단계로 진행한다. No B21, no additional scientific arms. 본실험 결과로 계수·cap·budget·layer를 바꾸지 않는다.

각 batch의 현재100개를 편집 직전/직후 평가하고 W5/W10/W15/W20에서 all-seen을 평가한다. W20 R/P/N 분모는2000/4000/20000이다. ACC(strict와 token/prompt), 선호율, NLL, paired cohort retention과 중복·상충 claim version 분리를 모두 남긴다. 기존 결과의 좋고 나쁨은 fit·후보 선택에 사용하지 않는다.

## 권한·자원·출력 경계

- SH4의 허용 write scope: 새 namespace와 그 tests, SH4 소유 plans/status/audits/report/run receipt, ignored `local/jlz-native-increment-v11/**`. 기존 baseline/shared method 소스는 read-only reference다.
- GH는 정본 문서 통합, task/inbox 관리, reviewed source의 비강제 publication을 담당한다. 이번 사용자 권한으로 SH4에 구현·Slurm 제출·raw 수집·factual 보고까지 배정한다.
- Task concurrent GPU cap1, per-job GPU1. 현재 server4 project cap과 host memory ceiling 중 더 엄격한 제한을 적용한다. 기본 CPU8, 명시적 `--mem`, walltime은 SH4가 실제 자산/작은 pilot과 큐 정책으로 결속한다. 기존 cap을 증액하거나 다른 job을 취소하지 않는다. 부족하면 `RESOURCE_PENDING`으로 큐 대기한다.
- NoCP 정책: weight/update-factor/optimizer 재개 bundle을 disk에 저장하지 않는다. 프로세스 내 W/H/RNG rollback은 유지한다. 종료 후 exact resume은 제공하지 않으며 incomplete/retry provenance를 숨기지 않는다.
- Dataset/model/C0 대량 재생성·재다운로드를 기본 실행에 추가하지 않는다. 기존 승인 자산을 exact hash로 재사용한다. 필요한 source/docs/input metadata 전송은 이 task에 승인된다.
- Raw 결과는 ignored local에 남기고 승인된 ODE-edit peer 경로의 artifact broadcast helper를 사용한다. `--delete`, credential 전송, 불명확한 외부 경로 쓰기는 금지한다.
- SH4 보고는 사실·수치·분모·identity·job 상태만 작성한다. 과학적 해석은 GH가 맡는다. No new recurring monitor/다른 task polling.

## 현재 준비 상태

CPU 대수/gradient/종료/geometry 계약 검산 PASS. 실제 native Llama qualification과 GPU pilot은 아직 실행하지 않았으며 SH4 소유 작업이다. TeX는 정적 문법 검사를 통과했지만 작성 환경에 LaTeX 엔진이 없어 PDF는 생성하지 않았다. 이는 실험 구현을 막지 않는다.

GH의 최초 응답에는 nonce, 수신 method/experiment manifest 확인, SH4 전달 accepted turn/ACK 상태, 다음 단계만 담는다. 본 연결은 bounded handoff이며 장기 실험 완료를 기다리지는 않는다. 전달 연결이 종료돼도 task 취소나 실행 중단을 뜻하지 않는다.
